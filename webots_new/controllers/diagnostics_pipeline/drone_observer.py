"""Drone 側の観測集計（RoboPose相当の関節角などから推定）

ポイント:
- ドローンは「脚が動かない理由（拘束原因）」を推定する
- RoboPose の代わりに、Spotから送られる関節角度と姿勢（roll/pitch）を使う
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Dict, List, Sequence, Tuple

from . import config
from .models import LegState, TrialResult
from .utils import clamp, normalize_distribution

Vector3 = Tuple[float, float, float]


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class DroneObservationAggregator:
    """各試行の観測から drone_can / p_drone を計算する"""

    def __init__(self) -> None:
        if not config.USE_ONLY_ROBOPOSE:
            raise RuntimeError("RoboPose only mode is required by specification")

        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}
        self._feature_history: Dict[str, List[Dict[str, float]]] = {leg: [] for leg in config.LEG_IDS}
        self._cause_accumulator: Dict[str, Dict[str, float]] = {
            leg: {label: 0.0 for label in config.CAUSE_LABELS} for leg in config.LEG_IDS
        }
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}
        self._cause_weight_sums: Dict[str, float] = {leg: 0.0 for leg in config.LEG_IDS}

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        # 1) 関節角度の変化量 → drone_can_raw
        reduced = self._reduce_joint_series(joint_angles, trial.trial_index)
        delta_theta = (max(reduced) - min(reduced)) if reduced else 0.0
        raw = clamp(delta_theta / config.DELTA_THETA_REF_DEG)

        self._raw_scores[leg.leg_id].append(raw)
        trial.drone_can_raw = raw

        # 2) 特徴量（末端の移動など）
        features = self._build_features(delta_theta, raw, end_positions)
        existing = dict(getattr(trial, "features", {}) or {})
        existing.update(features)
        trial.features = existing

        # 3) drone_can（平均→シグモイド）
        drone_raw_avg = mean(self._raw_scores[leg.leg_id][: config.TRIAL_COUNT]) if self._raw_scores[leg.leg_id] else 0.0
        leg.drone_can = self._sigmoid(drone_raw_avg)

        # 4) 拘束原因の確率分布（各trialの分布を単純平均）
        dist_now = self._estimate_cause_distribution(features)

        # NOTE: TANGLED は「ある程度動くが進まない」試行が本質なので、
        # ほとんど動いていない試行（rawが極小）に平均を支配されないよう、
        # delta_theta 由来の raw を重みとして使う（簡潔な重み付き平均）。
        weight = max(config.EPSILON, float(raw) + 0.05)
        acc = self._cause_accumulator[leg.leg_id]
        for k, v in dist_now.items():
            acc[k] = acc.get(k, 0.0) + weight * float(v)
        self._trial_counts[leg.leg_id] += 1
        self._cause_weight_sums[leg.leg_id] += weight

        denom = max(config.EPSILON, self._cause_weight_sums[leg.leg_id])
        averaged = {k: acc.get(k, 0.0) / denom for k in config.CAUSE_LABELS}
        leg.p_drone = normalize_distribution(averaged)

    # ---- 内部処理（小さな部品） ----

    def _reduce_joint_series(self, joint_angles: Sequence[Sequence[float]], trial_index: int) -> List[float]:
        motor_index = config.TRIAL_MOTOR_INDICES[trial_index - 1] if 1 <= trial_index <= len(config.TRIAL_MOTOR_INDICES) else 0
        out: List[float] = []
        for frame in joint_angles:
            if not frame or len(frame) <= motor_index:
                out.append(0.0)
            else:
                out.append(float(frame[motor_index]))
        return out

    def _build_features(
        self,
        delta_theta_deg: float,
        delta_theta_norm: float,
        end_positions: Sequence[Vector3],
    ) -> Dict[str, float]:
        end_disp = 0.0
        path_len = 0.0
        reversals = 0.0

        if end_positions:
            end_disp = _distance(end_positions[0], end_positions[-1])
            for i in range(1, len(end_positions)):
                path_len += _distance(end_positions[i - 1], end_positions[i])

            # x方向の速度符号反転（簡易）
            vx = [end_positions[i][0] - end_positions[i - 1][0] for i in range(1, len(end_positions))]
            for i in range(1, len(vx)):
                if vx[i - 1] == 0.0:
                    continue
                if vx[i - 1] * vx[i] < 0:
                    reversals += 1.0

        return {
            "delta_theta_deg": float(delta_theta_deg),
            "delta_theta_norm": float(delta_theta_norm),
            "end_disp": float(end_disp),
            "path_length": float(path_len),
            "path_straightness": float(path_len / (end_disp + config.EPSILON) if end_disp > 0 else 0.0),
            "reversals": float(reversals),
        }

    def _sigmoid(self, raw_avg: float) -> float:
        x = config.CONFIDENCE_STEEPNESS * (raw_avg - config.SELF_CAN_THRESHOLD)
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _robust_distribution(self, leg_id: str) -> Dict[str, float]:
        # 互換のため残す（現行は未使用）
        acc = self._cause_accumulator[leg_id]
        count = max(1, self._trial_counts[leg_id])
        return {k: acc.get(k, 0.0) / count for k in config.CAUSE_LABELS}

    def _estimate_cause_distribution(self, f: Dict[str, float]) -> Dict[str, float]:
        end_disp = float(f.get("end_disp", 0.0))
        path_length = float(f.get("path_length", 0.0))
        delta_theta_deg = float(f.get("delta_theta_deg", 0.0))
        reversals = float(f.get("reversals", 0.0))
        path_straightness = float(f.get("path_straightness", 0.0))

        # 閾値（簡潔版）
        NORMAL_DISP_MIN = 0.015
        BURIED_DISP_MAX = 0.006
        TRAPPED_DISP_MAX = 0.012
        BURIED_ANGLE_MAX = 0.75
        TRAPPED_ANGLE_MIN = 1.25
        BURIED_REVERSALS_MIN = 8.0

        # NOTE: MALFUNCTION は RoboPose だけでは外部拘束と区別が難しいため、
        # ここでは強く出さず、最終段(LLM/Spot情報)で確定させる。

        # NONE: 十分に動けている
        if end_disp >= NORMAL_DISP_MIN:
            return normalize_distribution({"NONE": 0.85, "BURIED": 0.03, "TRAPPED": 0.03, "TANGLED": 0.03, "MALFUNCTION": 0.06})

        # TANGLED の兆候: 末端は進まないが「動いてはいる」
        tangled_score = 0.0
        if path_straightness >= 2.5:
            tangled_score += 1.0
        if reversals >= 3.0:
            tangled_score += 1.0
        if path_length >= 0.020 and end_disp <= 0.010:
            tangled_score += 1.0

        # end_disp が小さくても角度は大きく動く場合（蔓絡み寄り）
        # TRAPPED(挟まり)の試行では delta_theta がここまで大きく出にくい。
        if end_disp < BURIED_DISP_MAX and delta_theta_deg >= 1.0 and reversals <= 3.0:
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.06, "TRAPPED": 0.10, "TANGLED": 0.76, "MALFUNCTION": 0.03})

        # 強い拘束（BURIED/TRAPPED）
        if end_disp < BURIED_DISP_MAX and delta_theta_deg < BURIED_ANGLE_MAX:
            if reversals >= BURIED_REVERSALS_MIN:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.85, "TRAPPED": 0.03, "TANGLED": 0.03, "MALFUNCTION": 0.04})
            # TRAPPEDより少し動ける(末端変位が僅かに大きい)場合はTANGLED寄り
            # ※ TRAPPEDセッションでは end_disp が概ね 0.0016 以下に収まるため、ここを分岐点にする。
            if end_disp >= 0.0018 and reversals <= 3.0:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.06, "TRAPPED": 0.10, "TANGLED": 0.76, "MALFUNCTION": 0.03})
            if tangled_score >= 2.0:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.06, "TRAPPED": 0.10, "TANGLED": 0.76, "MALFUNCTION": 0.03})
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.08, "TRAPPED": 0.80, "TANGLED": 0.04, "MALFUNCTION": 0.03})

        # TRAPPED: 角度は動くが末端が動かない
        if delta_theta_deg >= TRAPPED_ANGLE_MIN and (BURIED_DISP_MAX <= end_disp < TRAPPED_DISP_MAX):
            # TANGLED(蔓など): 角度は動くが、末端の進みが小さく、往復が少ない
            if reversals <= 3.0 and end_disp >= 0.0065:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.10, "TANGLED": 0.75, "MALFUNCTION": 0.05})
            if tangled_score >= 2.0:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.10, "TANGLED": 0.75, "MALFUNCTION": 0.05})
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.80, "TANGLED": 0.05, "MALFUNCTION": 0.05})

        # TANGLED: 末端は動くが拘束っぽい
        if end_disp >= BURIED_DISP_MAX and tangled_score >= 1.0:
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.10, "TANGLED": 0.75, "MALFUNCTION": 0.05})

        # 曖昧: 全部を少しずつ
        return normalize_distribution({"NONE": 0.12, "BURIED": 0.12, "TRAPPED": 0.32, "TANGLED": 0.32, "MALFUNCTION": 0.12})
