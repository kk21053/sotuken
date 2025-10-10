"""Spot self-diagnosis aggregation."""

from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Sequence

from . import config
from .models import LegState, TrialResult
from .utils import clamp


class SelfDiagnosisAggregator:
    """Aggregate four self-diagnosis trials per leg."""

    def __init__(self) -> None:
        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}

    @staticmethod
    def _mean_absolute(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        total = sum(abs(x) for x in values)
        return total / len(values)

    def _score_track(self, theta_cmd: Sequence[float], theta_meas: Sequence[float]) -> float:
        if not theta_cmd or len(theta_cmd) != len(theta_meas):
            return 0.0
        errors = [abs(c - m) for c, m in zip(theta_cmd, theta_meas)]
        score = 1.0 - (mean(errors) / config.E_MAX_DEG)
        return clamp(score)

    def _score_velocity(self, omega_meas: Sequence[float]) -> float:
        score = self._mean_absolute(omega_meas) / config.OMEGA_REF_DEG_PER_SEC
        return clamp(score)

    def _score_tau(self, tau_meas: Sequence[float], tau_limit: float) -> float:
        if tau_limit <= 0:
            return 0.0
        score = 1.0 - (self._mean_absolute(tau_meas) / tau_limit)
        return clamp(score)

    def _score_safe(self, safety_level: str) -> float:
        label = (safety_level or "").strip().lower()
        if label == "normal":
            return config.SAFE_SCORE_NORMAL
        if label == "warn":
            return config.SAFE_SCORE_WARN
        if label == "error":
            return config.SAFE_SCORE_ERROR
        try:
            value = float(safety_level)
            return clamp(value)
        except (TypeError, ValueError):
            return config.SAFE_SCORE_NORMAL

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
        tau_limit = max(tau_nominal * config.TAU_LIMIT_RATIO, 1e-6)
        score_track = self._score_track(theta_cmd, theta_meas)
        score_vel = self._score_velocity(omega_meas)
        score_tau = self._score_tau(tau_meas, tau_limit)
        score_safe = self._score_safe(safety_level)

        weights = config.SELF_WEIGHTS
        raw = (
            weights["track"] * score_track
            + weights["vel"] * score_vel
            + weights["tau"] * score_tau
            + weights["safe"] * score_safe
        )
        raw = clamp(raw)
        trials = self._raw_scores.setdefault(leg.leg_id, [])
        if len(trials) >= config.TRIAL_COUNT:
            trials.pop(0)
        trials.append(raw)

        if 1 <= trial_index <= config.TRIAL_COUNT:
            expected_dir = config.TRIAL_PATTERN[trial_index - 1]
            if expected_dir != direction:
                print(
                    f"[self_diagnosis] warning: leg {leg.leg_id} trial {trial_index} expected"
                    f" direction '{expected_dir}' but got '{direction}'",
                )

        trial = TrialResult(
            leg_id=leg.leg_id,
            trial_index=trial_index,
            direction=direction,
            start_time=start_time,
            end_time=end_time,
            self_can_raw=raw,
        )
        leg.trials.append(trial)

        expected_duration = config.TRIAL_DURATION_S
        if expected_duration > 0 and abs(trial.duration - expected_duration) > 0.25:
            print(
                f"[self_diagnosis] warning: leg {leg.leg_id} trial {trial_index}"
                f" duration {trial.duration:.2f}s differs from expected {expected_duration:.2f}s",
            )
        return trial

    def finalize_leg(self, leg: LegState) -> None:
        scores = self._raw_scores.get(leg.leg_id, [])
        if len(scores) < config.TRIAL_COUNT:
            return
        leg.self_can = mean(scores[: config.TRIAL_COUNT])
        leg.self_moves = leg.self_can >= config.SELF_CAN_THRESHOLD
        leg.moves_final = leg.self_moves

    def reset_leg(self, leg_id: str) -> None:
        self._raw_scores[leg_id] = []

    def reset_all(self) -> None:
        for leg_id in self._raw_scores:
            self._raw_scores[leg_id].clear()

    def trial_count(self, leg_id: str) -> int:
        return len(self._raw_scores.get(leg_id, []))
