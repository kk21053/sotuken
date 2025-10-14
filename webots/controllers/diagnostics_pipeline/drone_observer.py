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
        # ロバスト統計のための特徴量履歴（中央値フィルタ用）
        self._feature_history: Dict[str, List[Dict[str, float]]] = {leg: [] for leg in config.LEG_IDS}

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

    def _estimate_weight_on_leg(
        self, 
        leg_id: str, 
        base_positions: Sequence[Vector3],
        trial_index: int
    ) -> float:
        """診断動作中の脚への荷重を推定
        
        base_heightの変化から重心移動を検出します。
        診断動作中にbase_heightが低下している場合、その脚に重心が移動していると推定します。
        
        Args:
            leg_id: 診断中の脚ID（FL, FR, RL, RR）
            base_positions: 本体の位置履歴
            trial_index: 試行番号
            
        Returns:
            重心荷重推定値（0.0-1.0、1.0=完全に荷重、0.0=無荷重）
        """
        if not base_positions or len(base_positions) < 10:
            return 0.25  # デフォルト値（均等荷重）
        
        # base_heightの変化を計算（z座標）
        # 診断動作中にbase_heightが低下している場合、その脚に重心が移動している
        initial_height = sum([pos[2] for pos in base_positions[:5]]) / 5  # 最初の5フレームの平均
        final_height = sum([pos[2] for pos in base_positions[-5:]]) / 5   # 最後の5フレームの平均
        height_change = initial_height - final_height  # 正の値=低下（荷重増加）
        
        # 高さ変化から荷重を推定
        # 0.01m（10mm）の低下で荷重0.5増加と仮定
        weight_change = height_change * 50.0  # 50倍で0.01m→0.5の変化
        weight = 0.25 + weight_change
        
        # 0.0-1.0の範囲にクランプ
        weight = max(0.0, min(1.0, weight))
        
        return weight

    def _compute_joint_movements(self, joint_angles: Sequence[Sequence[float]]) -> Dict[str, float]:
        """各関節（shoulder, hip, knee）の角度変化量を計算
        
        Args:
            joint_angles: [[shoulder, hip, knee], ...] 形式の角度データ
            
        Returns:
            各関節の最大-最小角度変化量（度）
        """
        if not joint_angles or len(joint_angles) == 0:
            return {"shoulder": 0.0, "hip": 0.0, "knee": 0.0}
        
        # 各関節の角度を抽出
        shoulder_angles = [frame[0] for frame in joint_angles if len(frame) >= 3]
        hip_angles = [frame[1] for frame in joint_angles if len(frame) >= 3]
        knee_angles = [frame[2] for frame in joint_angles if len(frame) >= 3]
        
        # 各関節の変化量を計算（最大-最小）
        # 注: joint_anglesは既に度の単位で格納されている
        shoulder_delta = abs(max(shoulder_angles) - min(shoulder_angles)) if shoulder_angles else 0.0
        hip_delta = abs(max(hip_angles) - min(hip_angles)) if hip_angles else 0.0
        knee_delta = abs(max(knee_angles) - min(knee_angles)) if knee_angles else 0.0
        
        return {
            "shoulder": shoulder_delta,
            "hip": hip_delta,
            "knee": knee_delta
        }

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        # 各関節の動きを計算
        joint_movements = self._compute_joint_movements(joint_angles)
        
        reduced_angles = self._reduce_joint_series(joint_angles, trial.trial_index)
        print(f"[drone_observer] {leg.leg_id} trial {trial.trial_index}: received {len(joint_angles)} frames, reduced to {len(reduced_angles)} angles, range={min(reduced_angles) if reduced_angles else 0:.2f}~{max(reduced_angles) if reduced_angles else 0:.2f}°")
        print(f"[drone_observer] {leg.leg_id} trial {trial.trial_index}: joint movements - shoulder={joint_movements['shoulder']:.2f}°, hip={joint_movements['hip']:.2f}°, knee={joint_movements['knee']:.2f}°")
        if not reduced_angles:
            delta_theta = 0.0
        else:
            delta_theta = max(reduced_angles) - min(reduced_angles)
        print(f"[drone_observer] {leg.leg_id} trial {trial.trial_index}: delta_theta={delta_theta:.2f}°, drone_can_raw={clamp(delta_theta / config.DELTA_THETA_REF_DEG):.3f}")
        raw = clamp(delta_theta / config.DELTA_THETA_REF_DEG)
        scores = self._raw_scores.setdefault(leg.leg_id, [])
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

        # 重心荷重の推定（base_positionから推論）
        # 診断動作中の脚の荷重を推定：base_heightが低い＝その脚に重心が乗っている
        weight_on_leg = self._estimate_weight_on_leg(leg.leg_id, base_positions, trial.trial_index)
        
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
            # 各関節の動き（新規追加）
            "shoulder_delta": joint_movements["shoulder"],
            "hip_delta": joint_movements["hip"],
            "knee_delta": joint_movements["knee"],
            # 重心荷重（新規追加）
            "weight_on_leg": weight_on_leg,
        }
        trial.ok = bool(end_positions)
        
        # 特徴量を履歴に保存（中央値フィルタ用）
        self._feature_history[leg.leg_id].append(features)
        
        # 特徴量をログに記録（デバッグ用）
        try:
            from pathlib import Path
            import json
            log_dir = Path(__file__).parent.parent / "drone_circular_controller" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "robopose_features.jsonl"
            with open(log_file, 'a') as f:
                log_entry = {
                    "leg_name": leg.leg_id,
                    **features
                }
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            pass  # ロギング失敗は無視

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
        
        # 全試行完了後はロバスト統計（中央値フィルタ）を使用
        if count >= config.TRIAL_COUNT:
            print(f"[drone_observer] {leg.leg_id}: 全{count}試行完了、ロバスト統計を適用")
            averaged = self._compute_robust_cause_distribution(leg.leg_id)
        else:
            # 試行途中は累積平均を使用
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
            # 通常の累積平均またはロバスト統計を使用
            leg.p_drone = normalize_distribution(averaged)
    
    def _compute_robust_cause_distribution(self, leg_id: str) -> Dict[str, float]:
        """ロバスト統計（中央値ベース）で拘束原因を推定
        
        全試行の特徴量から中央値を計算し、中央値を代表値として使用します。
        これにより、物理シミュレーションの累積誤差や試行順序の影響による
        異常値（外れ値）の影響を最小化します。
        
        中央値は平均値と異なり、極端な値の影響を受けにくく、
        6試行中2試行が異常でも残り4試行の中央値が使われます。
        """
        from statistics import median
        
        history = self._feature_history[leg_id]
        
        if len(history) < 3:
            # 試行数が少ない場合は通常の累積平均を使用
            return {label: self._cause_accumulator[leg_id].get(label, 0.0) / max(1, len(history)) 
                    for label in config.CAUSE_LABELS}
        
        # 全特徴量の中央値を計算
        median_features = {
            'delta_theta_deg': median([f['delta_theta_deg'] for f in history]),
            'delta_theta_norm': median([f['delta_theta_norm'] for f in history]),
            'end_disp': median([f['end_disp'] for f in history]),
            'path_length': median([f['path_length'] for f in history]),
            'path_straightness': median([f['path_straightness'] for f in history]),
            'reversals': median([f['reversals'] for f in history]),
            'base_height': median([f['base_height'] for f in history]),
            'max_roll': median([f['max_roll'] for f in history]),
            'max_pitch': median([f['max_pitch'] for f in history]),
            'fallen': any(f['fallen'] for f in history),  # fallenは論理和
            # 各関節の中央値
            'shoulder_delta': median([f['shoulder_delta'] for f in history]),
            'hip_delta': median([f['hip_delta'] for f in history]),
            'knee_delta': median([f['knee_delta'] for f in history]),
            # 重心荷重の中央値
            'weight_on_leg': median([f['weight_on_leg'] for f in history]),
        }
        
        # 中央値特徴量を使って拘束原因を推定
        median_distribution = self._estimate_cause_distribution(median_features)
        
        print(f"[drone_observer] {leg_id}: ロバスト統計（中央値）適用")
        print(f"  中央値delta_θ: {median_features['delta_theta_deg']:.2f}°")
        print(f"  中央値end_disp: {median_features['end_disp']*1000:.2f}mm")
        print(f"  中央値関節: shoulder={median_features['shoulder_delta']:.2f}°, hip={median_features['hip_delta']:.2f}°, knee={median_features['knee_delta']:.2f}°")
        print(f"  重心荷重: weight_on_leg={median_features['weight_on_leg']:.2f} (0=無荷重, 1=完全荷重)")
        print(f"  推定分布: NONE={median_distribution.get('NONE', 0):.3f}, TRAPPED={median_distribution.get('TRAPPED', 0):.3f}, BURIED={median_distribution.get('BURIED', 0):.3f}")
        
        return median_distribution
    
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
        delta_theta_deg = features.get("delta_theta_deg", 0.0)  # デフォルトを0.0に統一
        
        # 各関節の動き
        shoulder_delta = features.get("shoulder_delta", 0.0)
        hip_delta = features.get("hip_delta", 0.0)
        knee_delta = features.get("knee_delta", 0.0)
        
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
        
        # ハイブリッド判定閾値（実機対応）
        # end_dispを主要判定基準、関節パターンを補助的に使用
        
        # BURIED判定の閾値（最優先）
        BURIED_END_DISP_MAX = 0.002  # 足先がほとんど動かない（2mm）
        BURIED_KNEE_MAX = 1.0        # kneeもほとんど動かない（度）
        
        # TRAPPED判定の閾値
        TRAPPED_END_DISP_MIN = 0.002  # BURIEDより大きい（2mm以上）
        TRAPPED_END_DISP_MAX = 0.009  # 足先は制限される（9mm未満）
        TRAPPED_HIP_MIN = 2.5         # hipは少しだけ動く（2.5度以上）
        
        # TANGLED判定の閾値
        TANGLED_HIP_MIN = 10.0        # hipは明確に動く（度）
        TANGLED_SHOULDER_MIN = 10.0   # shoulderは明確に動く（度）
        TANGLED_END_DISP_MAX = 0.020  # 足先は制限される（20mm）
        
        # NORMAL判定の閾値
        NORMAL_END_DISP_MIN = 0.0115  # 足先が十分動く（11.5mm以上）
        
        # ==============================================================
        # ハイブリッド判定（end_disp主導 + 関節パターン補助）
        # ==============================================================
        
        # BURIED（埋まる）: 足先がほとんど動かず、kneeもほとんど動かない（最優先）
        buried_cond_end_disp = end_disp < BURIED_END_DISP_MAX
        buried_cond_knee = knee_delta < BURIED_KNEE_MAX
        
        if buried_cond_end_disp and buried_cond_knee:
            score_buried = clamp(0.90)
            print(f"[BURIED判定] end_disp={end_disp*1000:.2f}mm < {BURIED_END_DISP_MAX*1000:.0f}mm ✓, knee={knee_delta:.2f}° < {BURIED_KNEE_MAX}° ✓ → score={score_buried:.3f}")
        else:
            score_buried = 0.0
            print(f"[BURIED判定] end_disp={end_disp*1000:.2f}mm<{BURIED_END_DISP_MAX*1000:.0f}mm={buried_cond_end_disp}, knee={knee_delta:.2f}°<{BURIED_KNEE_MAX}°={buried_cond_knee} → score=0")
        
        # 重心荷重の取得
        weight_on_leg = features.get("weight_on_leg", 0.25)
        
        # TRAPPED（挟まる）: 足先は制限されるが、hipは動く
        # BURIEDでない場合のみ判定
        if score_buried < 0.1:
            # 正常判定の優先: 足先が十分動く場合はTRAPPEDと判定しない
            if end_disp >= NORMAL_END_DISP_MIN:
                score_trapped = 0.0
                print(f"[TRAPPED判定] end_disp={end_disp*1000:.2f}mm >= {NORMAL_END_DISP_MIN*1000:.0f}mm → 正常動作、TRAPPED=0")
            else:
                trapped_cond_end_disp = TRAPPED_END_DISP_MIN <= end_disp < TRAPPED_END_DISP_MAX
                trapped_cond_hip = hip_delta >= TRAPPED_HIP_MIN
                
                # 重心補正: 重心が乗っている脚は動きにくいため、閾値を緩和
                # 条件: weight_on_leg > 0.35 かつ end_disp >= 9mm かつ hip_delta < 2.5° の場合、正常と判定
                # hip_delta < 2.5°の条件により、実際にTRAPPEDで動いている脚(hip >= 2°)を除外
                if weight_on_leg > 0.35 and end_disp >= 0.009 and hip_delta < 2.5:
                    score_trapped = 0.0
                    print(f"[TRAPPED判定] 重心荷重補正: weight={weight_on_leg:.2f}>0.35, end_disp={end_disp*1000:.2f}mm>=9mm, hip={hip_delta:.2f}°<2.5° → 正常動作、TRAPPED=0")
                # 条件を満たす場合にTRAPPEDと判定
                elif trapped_cond_end_disp and trapped_cond_hip:
                    # 足先の変位が小さいほどTRAPPEDの確率が高い
                    # ただし、重心荷重が高い場合はスコアを下げる
                    base_score = 0.80 + 0.15 * (1.0 - end_disp / TRAPPED_END_DISP_MAX)
                    weight_penalty = weight_on_leg * 0.3  # 重心が乗るほどスコアを下げる
                    score_trapped = clamp(base_score - weight_penalty)
                    print(f"[TRAPPED判定] {TRAPPED_END_DISP_MIN*1000:.0f}mm <= end_disp={end_disp*1000:.2f}mm < {TRAPPED_END_DISP_MAX*1000:.0f}mm ✓, hip={hip_delta:.2f}° >= {TRAPPED_HIP_MIN}° ✓, weight_on_leg={weight_on_leg:.2f} → score={score_trapped:.3f}")
                else:
                    score_trapped = 0.0
                    print(f"[TRAPPED判定] {TRAPPED_END_DISP_MIN*1000:.0f}mm<=end_disp={end_disp*1000:.2f}mm<{TRAPPED_END_DISP_MAX*1000:.0f}mm={trapped_cond_end_disp}, hip={hip_delta:.2f}°>={TRAPPED_HIP_MIN}°={trapped_cond_hip} → score=0")
        else:
            score_trapped = 0.0
        

        # TANGLED（絡まる）: hipとshoulderは明確に動くが、足先は制限される
        # BURIED、TRAPPEDでない場合のみ判定
        if score_buried < 0.1 and score_trapped < 0.1:
            tangled_cond_hip = hip_delta >= TANGLED_HIP_MIN
            tangled_cond_shoulder = shoulder_delta >= TANGLED_SHOULDER_MIN
            tangled_cond_end_disp = end_disp < TANGLED_END_DISP_MAX
            
            if tangled_cond_hip and tangled_cond_shoulder and tangled_cond_end_disp:
                # 関節の動きが大きいほどTANGLEDの確率が高い
                hip_score = min(0.3, (hip_delta - TANGLED_HIP_MIN) / 30.0)
                shoulder_score = min(0.3, (shoulder_delta - TANGLED_SHOULDER_MIN) / 30.0)
                score_tangled = clamp(0.6 + hip_score + shoulder_score)
                print(f"[TANGLED判定] hip={hip_delta:.2f}° >= {TANGLED_HIP_MIN}° ✓, shoulder={shoulder_delta:.2f}° >= {TANGLED_SHOULDER_MIN}° ✓, end_disp={end_disp*1000:.2f}mm < {TANGLED_END_DISP_MAX*1000:.0f}mm ✓ → score={score_tangled:.3f}")
            else:
                score_tangled = 0.0
                print(f"[TANGLED判定] hip={hip_delta:.2f}°>={TANGLED_HIP_MIN}°={tangled_cond_hip}, shoulder={shoulder_delta:.2f}°>={TANGLED_SHOULDER_MIN}°={tangled_cond_shoulder}, end_disp={end_disp*1000:.2f}mm<{TANGLED_END_DISP_MAX*1000:.0f}mm={tangled_cond_end_disp} → score=0")
        else:
            score_tangled = 0.0
        
        # MALFUNCTION: センサー故障（ドローンからは観測不可、常に低スコア）
        score_malfunction = 0.05
        
        # FALLEN: 転倒していない場合は非常に低いスコア
        score_fallen = 0.01
        
        # NONE（正常）: 他の異常が検出されない場合の残余確率
        # 異常スコアの最大値から算出（異常が明確な場合はNONEを下げる）
        max_abnormal_score = max(score_trapped, score_tangled, score_buried)
        if max_abnormal_score > 0.7:
            # 異常が明確な場合、NONEをさらに低く
            score_none = clamp((1.0 - max_abnormal_score) * 0.5)
        else:
            score_none = clamp(1.0 - max_abnormal_score)
        
        scores = {
            "NONE": score_none,
            "BURIED": score_buried,
            "TRAPPED": score_trapped,
            "TANGLED": score_tangled,
            "MALFUNCTION": score_malfunction,
            "FALLEN": score_fallen,
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
