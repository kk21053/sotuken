"""Spot self-diagnosis aggregation.

Spotの自己診断は「各脚が動くか動かないか」の2値判定のみを行います。
拘束原因(BURIED, TRAPPED, TANGLED)の詳細分析はドローン観測に委ねます。
"""

from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List, Sequence

from . import config
from .models import LegState, TrialResult
from .utils import clamp


class SelfDiagnosisAggregator:
    """Aggregate four self-diagnosis trials per leg.
    
    各脚について4回の試行を実施し、以下の指標を統合して
    「動く(can_move=True)」または「動かない(can_move=False)」を判定します:
    
    - tracking: 目標角度に追従できているか
    - velocity: 十分な角速度が出ているか
    - torque: トルクが正常範囲内か
    - safety: 安全レベルが正常か
    
    拘束の原因(砂に埋まっている、ワイヤーに絡まっているなど)は
    ドローンの多次元観測によって判定されます。
    """

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
        if not omega_meas:
            return 0.0
        # Use peak velocity instead of mean - normal legs reach high speeds
        peak_velocity = max(abs(v) for v in omega_meas)
        score = peak_velocity / config.OMEGA_REF_DEG_PER_SEC
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
        
        # 詳細ログ出力（FR脚の原因調査用）
        print(f"[self_diagnosis] {leg.leg_id} Trial {trial_index}:")
        print(f"  tracking_score: {score_track:.4f} (weight={weights['track']})")
        if theta_cmd and theta_meas:
            final_cmd = theta_cmd[-1] if theta_cmd else 0.0
            final_meas = theta_meas[-1] if theta_meas else 0.0
            error = abs(final_cmd - final_meas)
            print(f"    theta_cmd: {final_cmd:.3f}°, theta_meas: {final_meas:.3f}°, error: {error:.3f}°")
        print(f"  velocity_score: {score_vel:.4f} (weight={weights['vel']})")
        if omega_meas:
            peak_vel = max(abs(v) for v in omega_meas)
            print(f"    peak_velocity: {peak_vel:.3f}°/s (OMEGA_REF: {config.OMEGA_REF_DEG_PER_SEC}°/s)")
        print(f"  torque_score: {score_tau:.4f} (weight={weights['tau']})")
        if tau_meas:
            mean_tau = self._mean_absolute(tau_meas)
            print(f"    mean_torque: {mean_tau:.3f}, tau_limit: {tau_limit:.3f}")
        print(f"  safety_score: {score_safe:.4f} (weight={weights['safe']})")
        print(f"  → self_can_raw: {raw:.4f}")
        print()
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
        """脚の最終判定: Sigmoid確信度変換で確率的に判定。
        
        self_can → sigmoid変換 → confidence (0.0-1.0)
        
        confidence >= 0.8 → NORMAL (確信して「動く」)
        confidence <= 0.2 → FAULT (確信して「動かない」)
        0.2 < confidence < 0.8 → UNCERTAIN (不明、ドローン判定に委ねる)
        
        拘束原因の特定はドローン観測に委ねます。
        """
        scores = self._raw_scores.get(leg.leg_id, [])
        if len(scores) < config.TRIAL_COUNT:
            return
        
        # 基本スコア（4回の試行の平均）
        avg_score = mean(scores[: config.TRIAL_COUNT])
        
        # 一貫性の評価（標準偏差を使用）
        if len(scores) >= 2:
            from statistics import stdev
            try:
                std_dev = stdev(scores[: config.TRIAL_COUNT])
                # 一貫性スコア: 0（低） ~ 1（高）
                # 標準偏差が0.1以下なら高い一貫性、0.3以上なら低い一貫性
                consistency = max(0.0, 1.0 - (std_dev / 0.3))
                
                # 一貫性が低い場合（< 0.6）は保守的に判定
                if consistency < 0.6:
                    # 閾値に近づける（誤判定リスクを減らす）
                    if avg_score > config.SELF_CAN_THRESHOLD:
                        # 「動く」側なら少し下げる
                        avg_score = avg_score * 0.95
                    else:
                        # 「動かない」側なら少し上げる
                        avg_score = avg_score * 1.05
                    
                    print(f"[self_diagnosis] {leg.leg_id}: Low consistency ({consistency:.2f}), "
                          f"adjusted score from {mean(scores):.4f} to {avg_score:.4f}")
            except:
                # 標準偏差が計算できない場合（全て同じ値など）は一貫性が高い
                pass
        
        # 仕様ステップ3: シグモイド変換でspot_can計算
        import math
        steepness = config.CONFIDENCE_STEEPNESS
        threshold = config.SELF_CAN_THRESHOLD
        x = steepness * (avg_score - threshold)
        
        if x > 20:
            leg.spot_can = 1.0
        elif x < -20:
            leg.spot_can = 0.0
        else:
            leg.spot_can = 1.0 / (1.0 + math.exp(-x))
        
        # ログ出力
        print(f"[self_diagnosis] {leg.leg_id}: raw_score={avg_score:.4f} → "
              f"spot_can={leg.spot_can:.2%}")
    
    def reset_leg(self, leg_id: str) -> None:
        self._raw_scores[leg_id] = []

    def reset_all(self) -> None:
        for leg_id in self._raw_scores:
            self._raw_scores[leg_id].clear()

    def trial_count(self, leg_id: str) -> int:
        return len(self._raw_scores.get(leg_id, []))
