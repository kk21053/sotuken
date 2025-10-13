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
        self._trial_judgments: Dict[str, List[Dict]] = {leg: [] for leg in config.LEG_IDS}

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

        # 仕様ステップ8: 転倒判定（確率で格納）- トライアルごとに記録し、脚ごとに累積
        fallen = False
        max_roll = 0.0
        max_pitch = 0.0
        for roll, pitch, _ in base_orientations:
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            if abs(roll) > config.FALLEN_THRESHOLD_DEG or abs(pitch) > config.FALLEN_THRESHOLD_DEG:
                fallen = True
        
        # セッション全体の転倒フラグを更新
        if fallen:
            self._fallen = True
            # 転倒確率を計算（角度が大きいほど確率が高い）
            max_angle = max(max_roll, max_pitch)
            self._fallen_probability = min(1.0, max_angle / (config.FALLEN_THRESHOLD_DEG * 2))
        else:
            # 転倒していない場合の確率
            max_angle = max(max_roll, max_pitch)
            self._fallen_probability = max_angle / config.FALLEN_THRESHOLD_DEG
        
        # 各脚ごとの転倒フラグを更新（この脚のトライアル中に転倒した場合）
        if fallen:
            leg.fallen = True
            max_angle = max(max_roll, max_pitch)
            leg.fallen_probability = max(leg.fallen_probability, min(1.0, max_angle / (config.FALLEN_THRESHOLD_DEG * 2)))

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
        
        # このトライアルで最も確率の高い原因を記録（最頻値計算用）
        max_cause = max(current_distribution.items(), key=lambda x: x[1])[0]
        max_confidence = max(current_distribution.values())
        self._trial_judgments[leg.leg_id].append({
            'cause': max_cause,
            'confidence': max_confidence,
            'distribution': current_distribution.copy()
        })
        
        # 累積平均も計算（後方互換性のため）
        accumulator = self._cause_accumulator[leg.leg_id]
        for label, value in current_distribution.items():
            accumulator[label] = accumulator.get(label, 0.0) + value
        self._trial_counts[leg.leg_id] += 1

        count = max(1, self._trial_counts[leg.leg_id])
        averaged = {
            label: accumulator.get(label, 0.0) / count for label in config.CAUSE_LABELS
        }
        
        # 転倒が検出されている場合、p_droneをFALLEN優先に上書き
        if leg.fallen:
            fallen_confidence = min(0.95, leg.fallen_probability)
            leg.p_drone = normalize_distribution({
                "NONE": 0.01,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.01,
                "FALLEN": fallen_confidence,
            })
        else:
            # 最頻値ベースの判定（6回のトライアルが完了している場合）
            if self._trial_counts[leg.leg_id] >= 6:
                # 各原因が何回判定されたかをカウント
                cause_votes = {}
                confidence_sum = {}
                for judgment in self._trial_judgments[leg.leg_id]:
                    cause = judgment['cause']
                    confidence = judgment['confidence']
                    cause_votes[cause] = cause_votes.get(cause, 0) + 1
                    confidence_sum[cause] = confidence_sum.get(cause, 0.0) + confidence
                
                # 最も多く判定された原因を見つける
                max_votes = max(cause_votes.values())
                # 同じ投票数の原因が複数ある場合、確信度の合計が高い方を選ぶ
                candidates = [c for c, v in cause_votes.items() if v == max_votes]
                if len(candidates) == 1:
                    dominant_cause = candidates[0]
                else:
                    dominant_cause = max(candidates, key=lambda c: confidence_sum[c])
                
                # 支配的な原因の確率を計算（投票率に基づく）
                vote_ratio = cause_votes[dominant_cause] / 6.0
                dominant_probability = 0.5 + (vote_ratio * 0.4)  # 0.5-0.9の範囲
                
                # 確率分布を構築
                remaining_prob = 1.0 - dominant_probability
                leg.p_drone = normalize_distribution({
                    dominant_cause: dominant_probability,
                    **{label: remaining_prob / (len(config.CAUSE_LABELS) - 1) 
                       for label in config.CAUSE_LABELS if label != dominant_cause}
                })
            else:
                # トライアルが完了していない場合は累積平均を使用
                leg.p_drone = normalize_distribution(averaged)
    
    def _estimate_cause_distribution(self, features: Dict[str, float]) -> Dict[str, float]:
        """仕様ステップ4: 拘束状況の確率分布を推論
        
        RoboPoseの観測データから、6種類の拘束状況の確率を推論します:
        - NONE: 正常に動作
        - BURIED: 地面や砂に埋まっている
        - TRAPPED: 関節は動くが末端が固定されている
        - TANGLED: ツタなどに絡まって小さくしか動けない
        - MALFUNCTION: センサー故障（ドローンからは観測不可）
        - FALLEN: ロボットが転倒している
        """
        joint_score = clamp(features.get("delta_theta_norm", 0.0))
        end_disp = features.get("end_disp", 0.0)
        path_length = features.get("path_length", 0.0)
        reversals = features.get("reversals", 0.0)
        delta_theta_deg = features.get("delta_theta_deg", 0)
        
        # 転倒検出（最優先）
        fallen = features.get("fallen", False)
        max_roll = features.get("max_roll", 0.0)
        max_pitch = features.get("max_pitch", 0.0)
        
        if fallen:
            # 転倒している場合、FALLENの確率を非常に高くする
            max_angle = max(max_roll, max_pitch)
            fallen_confidence = min(0.95, max_angle / (config.FALLEN_THRESHOLD_DEG * 2))
            return normalize_distribution({
                "NONE": 0.01,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.01,
                "FALLEN": fallen_confidence,
            })
        
        # 基本的な閾値
        TRAPPED_ANGLE_THRESHOLD = 2.0   # 関節が動く最小角度（度）
        BURIED_ANGLE_THRESHOLD = 1.0    # 完全に動かない閾値（度）
        TRAPPED_DISPLACEMENT_THRESHOLD = 0.010  # 末端がほぼ動かない閾値（10mm）- より厳格に
        BURIED_DISPLACEMENT_THRESHOLD = 0.005   # 完全に埋まっている閾値（5mm）
        NORMAL_MIN_DISPLACEMENT = 0.015         # 正常な動きの最小変位（15mm）
        NORMAL_MIN_ANGLE = 3.0                  # 正常な動きの最小角度（3度）
        
        # 正常な動きの指標
        is_normal_movement = (
            delta_theta_deg >= NORMAL_MIN_ANGLE and 
            end_disp >= NORMAL_MIN_DISPLACEMENT
        )
        
        # TRAPPED（挟まる）: 関節は動くが足先がほぼ動かない
        # - 関節の角度変化が大きい（> 2度）
        # - 足先の変位が非常に小さい（< 10mm）
        # - これが最も明確な物理的特徴
        # - 閾値を厳格にして誤検出を防ぐ
        if end_disp < TRAPPED_DISPLACEMENT_THRESHOLD and delta_theta_deg > TRAPPED_ANGLE_THRESHOLD:
            # 関節が動くほど、TRAPPEDの確率が高い
            score_trapped = clamp(0.7 + (delta_theta_deg / 10.0))
        else:
            score_trapped = 0.02  # 明確でない場合は非常に低いスコア
        
        # BURIED（埋まる）: 関節も足先もほとんど動かない
        # - 関節の角度変化が非常に小さい（< 1度）
        # - 足先の変位も非常に小さい（< 5mm）
        if end_disp < BURIED_DISPLACEMENT_THRESHOLD and delta_theta_deg < BURIED_ANGLE_THRESHOLD:
            score_buried = clamp(0.85)
        else:
            score_buried = 0.01  # 明確でない場合は非常に低いスコア
        
        # TANGLED（絡まる）: 異常な経路パターン
        # - TRAPPEDやBURIEDではない
        # - 非常に不規則な経路（極端なpath_straightness）
        # - 異常に多い反転回数
        # 重要: 正常な診断動作でも5-7回程度のreversalsは発生するため、閾値を高く設定
        path_straightness = path_length / (end_disp + config.EPSILON) if end_disp > 0 else 0.0
        if score_trapped < 0.1 and score_buried < 0.1:  # TRAPPEDやBURIEDでない場合のみ
            # 極端に不規則な経路のみをTANGLEDと判定
            if path_straightness > 30.0 or reversals > 10:
                # 異常度が高いほどスコアが高い
                straightness_score = min(0.5, (path_straightness - 30.0) / 100.0) if path_straightness > 30.0 else 0.0
                reversal_score = min(0.5, (reversals - 10) * 0.05) if reversals > 10 else 0.0
                score_tangled = clamp(straightness_score + reversal_score)
            else:
                score_tangled = 0.01  # 明確でない場合は非常に低いスコア
        else:
            score_tangled = 0.01
        
        # MALFUNCTION: センサー故障（ドローンからは観測不可、常に低スコア）
        score_malfunction = 0.05
        
        # FALLEN: 転倒していない場合は非常に低いスコア
        score_fallen = 0.01
        
        # NONE（正常）: 積極的な正常判定
        # - 関節が十分に動いている（>= 3度）
        # - 足先も十分に動いている（>= 15mm）
        # - 他の異常スコアが低い
        if is_normal_movement and score_trapped < 0.1 and score_buried < 0.1 and score_tangled < 0.1:
            # 正常な動きとして高スコアを付与
            score_none = clamp(0.9)
        elif score_trapped < 0.5 and score_buried < 0.5 and score_tangled < 0.5:
            # 異常の証拠が弱い場合は正常寄りに
            score_none = clamp(0.7 - max(score_trapped, score_tangled, score_buried) * 0.5)
        else:
            # 明確な異常がある場合のみ、NONEを低くする
            score_none = clamp(1.0 - max(score_trapped, score_tangled, score_buried))

        scores = {
            "NONE": score_none,
            "BURIED": score_buried,
            "TRAPPED": score_trapped,
            "TANGLED": score_tangled,
            "MALFUNCTION": score_malfunction,
            "FALLEN": score_fallen,
        }
        
        # デバッグ情報（trial_indexがある場合のみ表示）
        if "trial_index" in features:
            max_cause = max(scores.items(), key=lambda x: x[1])
            if max_cause[1] > 0.3:  # 確率が30%以上の場合のみ表示
                print(f"[drone_observer] Trial features: delta_theta={delta_theta_deg:.2f}°, "
                      f"end_disp={end_disp*1000:.1f}mm, path_len={path_length*1000:.1f}mm, "
                      f"straightness={path_straightness:.2f}, reversals={reversals}")
                print(f"[drone_observer] Scores: TRAPPED={score_trapped:.3f}, TANGLED={score_tangled:.3f}, "
                      f"BURIED={score_buried:.3f}, NONE={score_none:.3f} → {max_cause[0]}")
        
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
