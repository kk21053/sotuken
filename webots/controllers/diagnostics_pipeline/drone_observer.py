"""Drone-side aggregation that relies on RoboPose outputs.

ドローン観測の役割:
- Spotの自己診断では判定できない「拘束の原因」を特定
- 5種類の状態を区別: NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION
- 多次元的な特徴量(変位、角度変化、経路、直進性など)を総合評価

Spotの自己診断は「動く/動かない」の2値判定のみ。
ドローンは「なぜ動かないのか」を診断します。
注: MALFUNCTIONはドローンからは観測不可（LLMが検出）
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from . import config
from .models import LegState, TrialResult
from .utils import clamp, normalize_distribution

Vector3 = Tuple[float, float, float]


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class DroneObservationAggregator:
    """Aggregate RoboPose measurements for each trial (仕様ステップ2,4,8)."""

    def __init__(self) -> None:
        if not config.USE_ONLY_ROBOPOSE:
            raise RuntimeError("RoboPose only mode is required by specification")
        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}
        self._fallen: bool = False
        self._fallen_probability: float = 0.0  # 仕様ステップ8: 転倒確率
        self._cause_accumulator: Dict[str, Dict[str, float]] = {
            leg: {label: 0.0 for label in config.CAUSE_LABELS}
            for leg in config.LEG_IDS
        }
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}

    @property
    def fallen(self) -> bool:
        return self._fallen

    @property
    def fallen_probability(self) -> float:
        return self._fallen_probability

    def _reduce_joint_series(self, joint_angles: Sequence[Sequence[float]], trial_index: int) -> List[float]:
        """
        関節角度系列から診断対象の関節を抽出
        
        Args:
            joint_angles: フレームごとの関節角度リスト [[j0, j1, j2], [j0, j1, j2], ...]
            trial_index: 試行番号 (1-based)
        
        Returns:
            診断対象関節の角度系列
        """
        # 試行ごとに使用する関節インデックスを決定
        if trial_index <= len(config.TRIAL_MOTOR_INDICES):
            motor_index = config.TRIAL_MOTOR_INDICES[trial_index - 1]
        else:
            motor_index = 0  # デフォルトはshoulder abduction
        
        reduced: List[float] = []
        for frame in joint_angles:
            if not frame or len(frame) <= motor_index:
                reduced.append(0.0)
            else:
                # 指定された関節インデックスを抽出
                reduced.append(frame[motor_index])
        return reduced

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        reduced_angles = self._reduce_joint_series(joint_angles, trial.trial_index)
        print(f"[drone_observer] {leg.leg_id} trial {trial.trial_index}: received {len(joint_angles)} frames, reduced to {len(reduced_angles)} angles, range={min(reduced_angles) if reduced_angles else 0:.2f}~{max(reduced_angles) if reduced_angles else 0:.2f}°")
        if not reduced_angles:
            delta_theta = 0.0
        else:
            delta_theta = max(reduced_angles) - min(reduced_angles)
        print(f"[drone_observer] {leg.leg_id} trial {trial.trial_index}: delta_theta={delta_theta:.2f}°, drone_can_raw={clamp(delta_theta / config.DELTA_THETA_REF_DEG):.3f}")
        raw = clamp(delta_theta / config.DELTA_THETA_REF_DEG)
        scores = self._raw_scores.setdefault(leg.leg_id, [])
        if len(scores) >= config.TRIAL_COUNT:
            scores.pop(0)
        scores.append(raw)
        trial.drone_can_raw = raw

        # 仕様ステップ8: 転倒判定（確率で格納）
        fallen = False
        max_roll = 0.0
        max_pitch = 0.0
        for roll, pitch, _ in base_orientations:
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            if abs(roll) > config.FALLEN_THRESHOLD_DEG or abs(pitch) > config.FALLEN_THRESHOLD_DEG:
                fallen = True
        if fallen:
            self._fallen = True
            # 転倒確率を計算（角度が大きいほど確率が高い）
            max_angle = max(max_roll, max_pitch)
            self._fallen_probability = min(1.0, max_angle / (config.FALLEN_THRESHOLD_DEG * 2))
        else:
            # 転倒していない場合の確率
            max_angle = max(max_roll, max_pitch)
            self._fallen_probability = max_angle / config.FALLEN_THRESHOLD_DEG

        end_disp = 0.0
        path_length = 0.0
        reversals = 0
        if end_positions:
            end_disp = _distance(end_positions[0], end_positions[-1])
            for idx in range(1, len(end_positions)):
                path_length += _distance(end_positions[idx - 1], end_positions[idx])
            velocities = []
            for idx in range(1, len(end_positions)):
                velocities.append(end_positions[idx][0] - end_positions[idx - 1][0])
            for idx in range(1, len(velocities)):
                if velocities[idx - 1] == 0.0:
                    continue
                if velocities[idx - 1] * velocities[idx] < 0:
                    reversals += 1
        base_height = mean([pos[2] for pos in base_positions]) if base_positions else 0.0

        features = {
            "trial_index": trial.trial_index,  # 試行番号を記録
            "delta_theta_deg": delta_theta,
            "delta_theta_norm": raw,
            "end_disp": end_disp,
            "path_length": path_length,
            "path_straightness": path_length / (end_disp + config.EPSILON) if end_disp > 0 else 0.0,
            "reversals": float(reversals),
            "base_height": base_height,
            "max_roll": max_roll,
            "max_pitch": max_pitch,
            "fallen": bool(fallen),
        }
        trial.ok = bool(end_positions)

        effective_scores = scores[: config.TRIAL_COUNT]
        drone_can_avg = mean(effective_scores) if effective_scores else 0.0
        leg.drone_can_raw = drone_can_avg
        
        # 仕様ステップ4: シグモイド変換でdrone_can計算
        import math
        steepness = config.CONFIDENCE_STEEPNESS
        threshold = config.SELF_CAN_THRESHOLD
        x = steepness * (drone_can_avg - threshold)
        
        if x > 20:
            leg.drone_can = 1.0
        elif x < -20:
            leg.drone_can = 0.0
        else:
            leg.drone_can = 1.0 / (1.0 + math.exp(-x))
        
        # Store trial if not already in leg.trials
        if trial not in leg.trials:
            leg.trials.append(trial)
        
        # Update trial drone_can_raw
        trial.drone_can_raw = raw

        current_distribution = self._estimate_cause_distribution(features)
        accumulator = self._cause_accumulator.setdefault(
            leg.leg_id, {label: 0.0 for label in config.CAUSE_LABELS}
        )
        for label, value in current_distribution.items():
            accumulator[label] = accumulator.get(label, 0.0) + value
        self._trial_counts[leg.leg_id] = self._trial_counts.get(leg.leg_id, 0) + 1

        count = max(1, self._trial_counts[leg.leg_id])
        averaged = {
            label: accumulator.get(label, 0.0) / count for label in config.CAUSE_LABELS
        }
        leg.p_drone = normalize_distribution(averaged)
    
    def _estimate_cause_distribution(self, features: Dict[str, float]) -> Dict[str, float]:
        """仕様ステップ4: 拘束状況の確率分布を推論
        
        RoboPoseの観測データから、5種類の拘束状況の確率を推論します:
        - NONE: 正常に動作
        - BURIED: 地面や砂に埋まっている
        - TRAPPED: 関節は動くが末端が固定されている
        - TANGLED: ツタなどに絡まって小さくしか動けない
        - MALFUNCTION: センサー故障（ドローンからは観測不可）
        """
        joint_score = clamp(features.get("delta_theta_norm", 0.0))
        end_disp = features.get("end_disp", 0.0)
        path_length = features.get("path_length", 0.0)
        reversals = features.get("reversals", 0.0)
        delta_theta_deg = features.get("delta_theta_deg", 0)
        
        # 基本的な閾値
        TRAPPED_ANGLE_THRESHOLD = 0.1   # 関節が動く最小角度
        BURIED_ANGLE_THRESHOLD = 0.05   # 完全に動かない閾値
        TRAPPED_DISPLACEMENT_THRESHOLD = 0.008  # 末端がほぼ動かない閾値
        BURIED_DISPLACEMENT_THRESHOLD = 0.001   # 完全に埋まっている閾値
        
        # TRAPPED: 関節は動くが足先がほぼ動かない
        if end_disp < TRAPPED_DISPLACEMENT_THRESHOLD and delta_theta_deg > TRAPPED_ANGLE_THRESHOLD:
            score_trapped = clamp(0.6 + (delta_theta_deg / 5.0))
        else:
            score_trapped = 0.0
        
        # TANGLED: 不規則な経路、反転が多い
        path_straightness = path_length / (end_disp + config.EPSILON) if end_disp > 0 else 0.0
        if path_straightness > 1.5 or reversals > 0:
            score_tangled = clamp(0.3 + reversals * 0.2)
        else:
            score_tangled = 0.0
        
        # BURIED: 関節も足先もほとんど動かない
        if end_disp < BURIED_DISPLACEMENT_THRESHOLD and delta_theta_deg < BURIED_ANGLE_THRESHOLD:
            score_buried = clamp(0.8)
        else:
            score_buried = 0.0
        
        # MALFUNCTION: センサー故障（ドローンからは観測不可、常に低スコア）
        score_malfunction = 0.05
        
        # NONE: 正常（他のスコアが低ければ高くなる）
        score_none = clamp(1.0 - max(score_trapped, score_tangled, score_buried))

        scores = {
            "NONE": score_none,
            "BURIED": score_buried,
            "TRAPPED": score_trapped,
            "TANGLED": score_tangled,
            "MALFUNCTION": score_malfunction,
        }
        return normalize_distribution(scores)

    @property
    def fallen(self) -> bool:
        return self._fallen

    def reset(self) -> None:
        for leg_id in self._raw_scores:
            self._raw_scores[leg_id].clear()
            self._cause_accumulator[leg_id] = {label: 0.0 for label in config.CAUSE_LABELS}
            self._trial_counts[leg_id] = 0
        self._fallen = False
        self._fallen_probability = 0.0
