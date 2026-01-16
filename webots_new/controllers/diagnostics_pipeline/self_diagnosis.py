"""Spot 自己診断の集計（動く/動かないの確率）"""

from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Dict, List, Sequence

from . import config
from .models import LegState, TrialResult
from .utils import clamp


class SelfDiagnosisAggregator:
    """各脚の 6 試行の self_can_raw を集計して spot_can を作る"""

    def __init__(self) -> None:
        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}

    def record_trial(
        self,
        leg: LegState,
        trial_index: int,
        direction: str,
        start_time: float,
        end_time: float,
        theta_cmd: Sequence[float],
        theta_meas: Sequence[float],
        omega_meas: Sequence[float],
        tau_meas: Sequence[float],
        tau_nominal: float,
        safety_level: str,
    ) -> TrialResult:
        # pipeline 側から spot_can_raw を上書きされる前提なので、ここでは最低限で計算
        tau_limit = max(tau_nominal * config.TAU_LIMIT_RATIO, 1e-6)

        track = self._score_track(theta_cmd, theta_meas)
        vel = self._score_velocity(omega_meas)
        tau = self._score_tau(tau_meas, tau_limit)
        safe = self._score_safe(safety_level)

        w = config.SELF_WEIGHTS
        raw = clamp(w["track"] * track + w["vel"] * vel + w["tau"] * tau + w["safe"] * safe)

        scores = self._raw_scores.setdefault(leg.leg_id, [])
        if len(scores) >= config.TRIAL_COUNT:
            scores.pop(0)
        scores.append(raw)

        trial = TrialResult(
            leg_id=leg.leg_id,
            trial_index=trial_index,
            direction=direction,
            start_time=start_time,
            end_time=end_time,
            self_can_raw=raw,
        )
        leg.trials.append(trial)
        return trial

    def record_raw_trial(
        self,
        leg: LegState,
        trial_index: int,
        direction: str,
        start_time: float,
        end_time: float,
        self_can_raw: float,
    ) -> TrialResult:
        """Spot側で計算された self_can_raw をそのまま集計に使う。

        Drone側には theta_cmd/theta_meas の系列が届かないため、
        （Spotの比較で求めた）raw値を入力として扱う。
        """
        try:
            raw = clamp(float(self_can_raw))
        except Exception:
            raw = 0.0

        scores = self._raw_scores.setdefault(leg.leg_id, [])
        if len(scores) >= config.TRIAL_COUNT:
            scores.pop(0)
        scores.append(raw)

        trial = TrialResult(
            leg_id=leg.leg_id,
            trial_index=int(trial_index),
            direction=str(direction),
            start_time=float(start_time),
            end_time=float(end_time),
            self_can_raw=raw,
        )
        leg.trials.append(trial)
        return trial

    def finalize_leg(self, leg: LegState) -> None:
        scores = self._raw_scores.get(leg.leg_id, [])
        if len(scores) < config.TRIAL_COUNT:
            return

        avg = mean(scores[: config.TRIAL_COUNT])

        # 一貫性が低い場合は保守的に少し補正（旧版の考え方を踏襲）
        if len(scores) >= 2:
            try:
                consistency = max(0.0, 1.0 - (stdev(scores[: config.TRIAL_COUNT]) / 0.3))
                if consistency < 0.6:
                    if avg > config.SELF_CAN_THRESHOLD:
                        avg *= 0.95
                    else:
                        avg *= 1.05
            except Exception:
                pass

        # シグモイド変換
        x = config.CONFIDENCE_STEEPNESS * (avg - config.SELF_CAN_THRESHOLD)
        if x > 20:
            leg.spot_can = 1.0
        elif x < -20:
            leg.spot_can = 0.0
        else:
            leg.spot_can = 1.0 / (1.0 + math.exp(-x))

    def reset_all(self) -> None:
        for leg_id in self._raw_scores:
            self._raw_scores[leg_id].clear()

    # ---- スコア関数（簡単な形） ----

    def _score_track(self, theta_cmd: Sequence[float], theta_meas: Sequence[float]) -> float:
        if not theta_cmd or len(theta_cmd) != len(theta_meas):
            return 0.0
        errors = [abs(c - m) for c, m in zip(theta_cmd, theta_meas)]
        return clamp(1.0 - (mean(errors) / config.E_MAX_DEG))

    def _score_velocity(self, omega_meas: Sequence[float]) -> float:
        if not omega_meas:
            return 0.0
        peak = max(abs(v) for v in omega_meas)
        return clamp(peak / config.OMEGA_REF_DEG_PER_SEC)

    def _score_tau(self, tau_meas: Sequence[float], tau_limit: float) -> float:
        if tau_limit <= 0:
            return 0.0
        if not tau_meas:
            return 1.0
        mean_abs = sum(abs(x) for x in tau_meas) / len(tau_meas)
        return clamp(1.0 - (mean_abs / tau_limit))

    def _score_safe(self, safety_level: str) -> float:
        label = (safety_level or "").strip().lower()
        if label == "normal":
            return config.SAFE_SCORE_NORMAL
        if label == "warn":
            return config.SAFE_SCORE_WARN
        if label == "error":
            return config.SAFE_SCORE_ERROR
        try:
            return clamp(float(safety_level))
        except (TypeError, ValueError):
            return config.SAFE_SCORE_NORMAL
