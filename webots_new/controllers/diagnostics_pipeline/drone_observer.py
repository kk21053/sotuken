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
    """各試行の観測から drone_can / p_drone / 転倒 を計算する"""

    def __init__(self) -> None:
        if not config.USE_ONLY_ROBOPOSE:
            raise RuntimeError("RoboPose only mode is required by specification")

        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}
        self._feature_history: Dict[str, List[Dict[str, float]]] = {leg: [] for leg in config.LEG_IDS}
        self._cause_accumulator: Dict[str, Dict[str, float]] = {
            leg: {label: 0.0 for label in config.CAUSE_LABELS} for leg in config.LEG_IDS
        }
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}

        self._fallen = False
        self._fallen_probability = 0.0

    @property
    def fallen(self) -> bool:
        return self._fallen

    @property
    def fallen_probability(self) -> float:
        return self._fallen_probability

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

        # 2) 転倒判定（roll/pitch の最大値）
        fallen, roll_max, pitch_max = self._detect_fallen(base_orientations)
        if fallen:
            self._fallen = True

        max_angle = max(roll_max, pitch_max)
        if fallen:
            self._fallen_probability = min(1.0, max_angle / (config.FALLEN_THRESHOLD_DEG * 2))
            leg.fallen = True
            leg.fallen_probability = max(leg.fallen_probability, self._fallen_probability)
        else:
            self._fallen_probability = max_angle / max(config.FALLEN_THRESHOLD_DEG, config.EPSILON)

        # 3) 特徴量（末端の移動など）
        features = self._build_features(delta_theta, raw, end_positions, base_orientations, base_positions, fallen)
        self._feature_history[leg.leg_id].append(features)
        trial.features = dict(features)

        # 4) drone_can（平均→シグモイド）
        drone_raw_avg = mean(self._raw_scores[leg.leg_id][: config.TRIAL_COUNT]) if self._raw_scores[leg.leg_id] else 0.0
        leg.drone_can = self._sigmoid(drone_raw_avg)

        # 5) 拘束原因の確率分布
        dist_now = self._estimate_cause_distribution(features)
        acc = self._cause_accumulator[leg.leg_id]
        for k, v in dist_now.items():
            acc[k] = acc.get(k, 0.0) + v
        self._trial_counts[leg.leg_id] += 1

        count = max(1, self._trial_counts[leg.leg_id])
        if count >= config.TRIAL_COUNT:
            averaged = self._robust_distribution(leg.leg_id)
        else:
            averaged = {k: acc.get(k, 0.0) / count for k in config.CAUSE_LABELS}

        if leg.fallen:
            fallen_conf = min(0.95, max(leg.fallen_probability, 0.01))
            leg.p_drone = normalize_distribution(
                {
                    "NONE": 0.01,
                    "BURIED": 0.01,
                    "TRAPPED": 0.01,
                    "TANGLED": 0.01,
                    "MALFUNCTION": 0.01,
                    "FALLEN": fallen_conf,
                }
            )
        else:
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

    def _detect_fallen(self, base_orientations: Sequence[Tuple[float, float, float]]):
        fallen = False
        roll_max = 0.0
        pitch_max = 0.0
        for roll, pitch, _ in base_orientations:
            roll_max = max(roll_max, abs(roll))
            pitch_max = max(pitch_max, abs(pitch))
            if abs(roll) > config.FALLEN_THRESHOLD_DEG or abs(pitch) > config.FALLEN_THRESHOLD_DEG:
                fallen = True
        return fallen, roll_max, pitch_max

    def _build_features(
        self,
        delta_theta_deg: float,
        delta_theta_norm: float,
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
        fallen: bool,
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

        base_height = mean([p[2] for p in base_positions]) if base_positions else 0.0
        roll_max = max([abs(r) for r, _, _ in base_orientations], default=0.0)
        pitch_max = max([abs(p) for _, p, _ in base_orientations], default=0.0)

        return {
            "delta_theta_deg": float(delta_theta_deg),
            "delta_theta_norm": float(delta_theta_norm),
            "end_disp": float(end_disp),
            "path_length": float(path_len),
            "path_straightness": float(path_len / (end_disp + config.EPSILON) if end_disp > 0 else 0.0),
            "reversals": float(reversals),
            "base_height": float(base_height),
            "max_roll": float(roll_max),
            "max_pitch": float(pitch_max),
            "fallen": bool(fallen),
        }

    def _sigmoid(self, raw_avg: float) -> float:
        x = config.CONFIDENCE_STEEPNESS * (raw_avg - config.SELF_CAN_THRESHOLD)
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _robust_distribution(self, leg_id: str) -> Dict[str, float]:
        history = self._feature_history[leg_id]
        if len(history) < 3:
            count = max(1, len(history))
            acc = self._cause_accumulator[leg_id]
            return {k: acc.get(k, 0.0) / count for k in config.CAUSE_LABELS}

        def _p75(values: List[float]) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            # inclusive, simple 75% index
            idx = int(round(0.75 * (len(s) - 1)))
            return float(s[max(0, min(len(s) - 1, idx))])

        med = {
            "delta_theta_deg": median([f["delta_theta_deg"] for f in history]),
            "delta_theta_norm": median([f["delta_theta_norm"] for f in history]),
            "end_disp": median([f["end_disp"] for f in history]),
            "path_length": median([f["path_length"] for f in history]),
            "path_straightness": median([f["path_straightness"] for f in history]),
            "reversals": median([f["reversals"] for f in history]),
            # 追加の要約統計（曖昧領域の誤検知対策）
            "end_disp_p75": _p75([f["end_disp"] for f in history]),
            "end_disp_max": max([f["end_disp"] for f in history]),
            "delta_theta_deg_max": max([f["delta_theta_deg"] for f in history]),
            "reversals_max": max([f["reversals"] for f in history]),
            "path_straightness_max": max([f["path_straightness"] for f in history]),
            "base_height": median([f["base_height"] for f in history]),
            "max_roll": median([f["max_roll"] for f in history]),
            "max_pitch": median([f["max_pitch"] for f in history]),
            "fallen": any(f["fallen"] for f in history),
        }
        return self._estimate_cause_distribution(med)

    def _estimate_cause_distribution(self, f: Dict[str, float]) -> Dict[str, float]:
        # 転倒が最優先
        if f.get("fallen", False):
            max_angle = max(float(f.get("max_roll", 0.0)), float(f.get("max_pitch", 0.0)))
            fallen_conf = min(0.95, max_angle / (config.FALLEN_THRESHOLD_DEG * 2))
            return normalize_distribution(
                {
                    "NONE": 0.01,
                    "BURIED": 0.01,
                    "TRAPPED": 0.01,
                    "TANGLED": 0.01,
                    "MALFUNCTION": 0.01,
                    "FALLEN": fallen_conf,
                }
            )

        end_disp = float(f.get("end_disp", 0.0))
        delta_theta_deg = float(f.get("delta_theta_deg", 0.0))
        path_straightness = float(f.get("path_straightness", 0.0))
        reversals = float(f.get("reversals", 0.0))

        # robust 集計がある場合の補助統計（なければ現値を使う）
        end_disp_p75 = float(f.get("end_disp_p75", end_disp))
        reversals_max = float(f.get("reversals_max", reversals))
        path_straightness_max = float(f.get("path_straightness_max", path_straightness))

        # ---- 汎用的な閾値ベース（仕様の簡潔版） ----
        # 旧版ベースだが、BURIEDは「角度変化が極小」という条件が本質なので、角度閾値を厳しめにする。
        TRAPPED_ANGLE_MIN = 1.25
        BURIED_ANGLE_THRESHOLD = 0.55
        TRAPPED_DISPLACEMENT_MAX = 0.012
        BURIED_DISPLACEMENT_THRESHOLD = 0.005
        NORMAL_DISPLACEMENT_MIN = 0.015

        buried = (end_disp < BURIED_DISPLACEMENT_THRESHOLD) and (delta_theta_deg < BURIED_ANGLE_THRESHOLD)
        if buried:
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.85,
                    "TRAPPED": 0.03,
                    "TANGLED": 0.03,
                    "MALFUNCTION": 0.03,
                    "FALLEN": 0.01,
                }
            )

        # end_disp が十分大きいなら正常優先
        if end_disp >= NORMAL_DISPLACEMENT_MIN:
            return normalize_distribution(
                {
                    "NONE": 0.85,
                    "BURIED": 0.03,
                    "TRAPPED": 0.03,
                    "TANGLED": 0.03,
                    "MALFUNCTION": 0.05,
                    "FALLEN": 0.01,
                }
            )

        trapped = (delta_theta_deg >= TRAPPED_ANGLE_MIN) and (
            BURIED_DISPLACEMENT_THRESHOLD < end_disp < TRAPPED_DISPLACEMENT_MAX
        )
        if trapped:
            score = clamp(0.75 + 0.20 * (1.0 - end_disp / TRAPPED_DISPLACEMENT_MAX))
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.05,
                    "TRAPPED": score,
                    "TANGLED": 0.05,
                    "MALFUNCTION": 0.04,
                    "FALLEN": 0.01,
                }
            )

        # TRAPPED の「中間帯」: 角度変化は少しあるが、ほぼ移動しない。
        # - BURIED ほど角度が極小ではない
        # - TRAPPED_ANGLE_MIN に届かない場合でも TRAPPED 寄りにする
        trapped_mid = (end_disp < BURIED_DISPLACEMENT_THRESHOLD) and (
            BURIED_ANGLE_THRESHOLD <= delta_theta_deg < TRAPPED_ANGLE_MIN
        )
        if trapped_mid:
            return normalize_distribution(
                {
                    "NONE": 0.06,
                    "BURIED": 0.10,
                    "TRAPPED": 0.72,
                    "TANGLED": 0.06,
                    "MALFUNCTION": 0.05,
                    "FALLEN": 0.01,
                }
            )

        # それ以外（曖昧領域）は、特徴量から滑らかに重み付けして分布を作る。
        # - end_disp が大きいほど NONE
        # - 角度変化がそこそこあり end_disp が小さければ TRAPPED
        # - path_straightness / reversals が大きければ TANGLED

        # NONE は「十分動けている」時に強くなるが、曖昧帯では強くしすぎない
        none_w = clamp((end_disp - 0.012) / 0.008)

        # TRAPPED は「角度変化がそこそこあり、移動が小さい」時に強くなる
        trapped_w = clamp((delta_theta_deg - 0.9) / 1.3) * clamp((0.013 - end_disp) / 0.010)

        # TANGLED は「進行方向のぶれ / 往復 / ぎくしゃく」を拾う
        # robust 集計では max を使って、単発の強い兆候も反映する
        t_rev = clamp(reversals_max / 3.0)
        t_straight = clamp((path_straightness_max - 1.10) / 1.00)
        t_hi_theta = clamp((delta_theta_deg - 2.0) / 1.3) * clamp((0.016 - end_disp) / 0.010)
        tangled_w = clamp(0.65 * t_rev + 0.25 * t_straight + 0.10 * t_hi_theta)

        # MALFUNCTION は受け皿（ただし強すぎると何でもMALFUNCTIONになる）
        mal_w = 0.18
        bur_w = 0.08

        return normalize_distribution(
            {
                "NONE": 0.12 + 0.55 * none_w,
                "BURIED": bur_w,
                "TRAPPED": 0.10 + 0.75 * trapped_w,
                "TANGLED": 0.18 + 0.75 * tangled_w,
                "MALFUNCTION": mal_w,
                "FALLEN": 0.01,
            }
        )
