"""Drone 側の観測集計（RoboPose相当の関節角などから推定）

ポイント:
- ドローンは「脚が動かない理由（拘束原因）」を推定する
- RoboPose の代わりに、Spotから送られる関節角度と姿勢（roll/pitch）を使う
"""

from __future__ import annotations

import math
from statistics import median
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

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        # 1) 関節角度の変化量
        reduced = self._reduce_joint_series(joint_angles, trial.trial_index)
        delta_theta = (max(reduced) - min(reduced)) if reduced else 0.0

        # 2) end_position はコントローラ側で「胴体ローカルの変位（FK差分）」として生成される。
        #    そのため、base_positions による補正は不要で、むしろ座標系の不一致で end_disp を
        #    極端に増幅して誤判定を招くことがある。
        end_positions_rel = list(end_positions)

        # 3) 特徴量（末端の移動など）
        raw_angle = clamp(delta_theta / config.DELTA_THETA_REF_DEG)
        features = self._build_features(delta_theta, raw_angle, end_positions_rel)
        self._feature_history[leg.leg_id].append(features)
        trial.features = dict(features)

        # 4) drone_can_raw: 「関節が動いた」だけでは足先が動かないケースがある。
        #    末端移動量も加味して「動ける確率」にする。
        raw_disp = clamp(float(features.get("end_disp", 0.0)) / max(config.EPSILON, config.END_DISP_REF_M))
        raw = clamp(0.20 * raw_angle + 0.80 * raw_disp)

        self._raw_scores[leg.leg_id].append(raw)
        trial.drone_can_raw = raw

        # 5) drone_can（頑健な集計→シグモイド）
        # 混在ケースでは一部の試行だけ end_disp/delta_theta が小さくなりやすく、単純平均だと
        # 本来の傾向を打ち消してしまう。中央値を使って外れ値に強くする。
        samples = self._raw_scores[leg.leg_id][: config.TRIAL_COUNT]
        drone_raw_agg = median(samples) if samples else 0.0
        leg.drone_can = self._sigmoid(drone_raw_agg)

        # 6) 拘束原因の確率分布
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
            "reversals_p75": _p75([f["reversals"] for f in history]),
            "reversals_max": max([f["reversals"] for f in history]),
            "path_straightness_max": max([f["path_straightness"] for f in history]),
        }
        return self._estimate_cause_distribution(med)

    def _estimate_cause_distribution(self, f: Dict[str, float]) -> Dict[str, float]:
        end_disp = float(f.get("end_disp", 0.0))
        delta_theta_deg = float(f.get("delta_theta_deg", 0.0))
        path_straightness = float(f.get("path_straightness", 0.0))
        reversals = float(f.get("reversals", 0.0))

        # robust 集計がある場合は、単発でも大きく出た兆候を拾う
        delta_theta_deg_max = float(f.get("delta_theta_deg_max", delta_theta_deg))
        end_disp_max = float(f.get("end_disp_max", end_disp))
        delta_theta_peak = max(delta_theta_deg, delta_theta_deg_max)
        end_disp_peak = max(end_disp, end_disp_max)

        # robust 集計がある場合の補助統計（なければ現値を使う）
        end_disp_p75 = float(f.get("end_disp_p75", end_disp))
        reversals_max = float(f.get("reversals_max", reversals))
        reversals_p75 = float(f.get("reversals_p75", reversals))
        path_straightness_max = float(f.get("path_straightness_max", path_straightness))

        # ---- 汎用的な閾値ベース（仕様の簡潔版） ----
        # 旧版ベースだが、BURIEDは「角度変化が極小」という条件が本質なので、角度閾値を厳しめにする。
        TRAPPED_ANGLE_MIN = 1.25
        TANGLED_ANGLE_MIN = 1.00
        BURIED_ANGLE_THRESHOLD = 0.75
        TRAPPED_DISPLACEMENT_MAX = 0.012
        BURIED_DISPLACEMENT_THRESHOLD = 0.006
        NORMAL_DISPLACEMENT_MIN = 0.015

        # 「埋没」と「罠で固定」の見分け用（汎用的・観測ベース）
        # - BURIED: ほぼ移動できず、末端位置が細かく揺れて往復(反転)が増えやすい
        # - TRAPPED: ほぼ移動できないが、揺れ(反転)は少ないことが多い
        BURIED_REVERSALS_MIN = 8.0
        TANGLED_REVERSALS_MAX = 2.0

        # robust集計がある場合でも、ここでは median値(reversals)を基本に使う
        reversals_med = reversals

        # TANGLED の兆候（TRAPPED分岐でも使う）
        t_rev = clamp(reversals_max / 3.0)
        t_straight = clamp((path_straightness_max - 1.10) / 1.00)

        # end_disp が極小だと path_straightness は数値的に不安定になりやすい。
        # そのため「ほぼ移動していない」領域では、直進度(path_straightness)を絡まり根拠に使わない。
        if end_disp_peak < BURIED_DISPLACEMENT_THRESHOLD:
            t_straight = 0.0

        # さらに極端に大きい場合は、反転も含めて無効化して誤検知を減らす。
        if end_disp_peak < 0.004 and path_straightness_max >= 10.0:
            t_rev = 0.0
            t_straight = 0.0

        tangled_hint = clamp(0.60 * t_rev + 0.40 * t_straight)

        # TANGLED（明確な兆候）:
        # - 末端位置がそこそこ動く / 関節角がそこそこ動く
        # - ただし「反転が多い」場合はノイズ(微小揺れ)の可能性が高いので除外する
        tangled_obvious = (
            (end_disp_peak >= BURIED_DISPLACEMENT_THRESHOLD or delta_theta_peak >= TANGLED_ANGLE_MIN)
            and (reversals_med <= TANGLED_REVERSALS_MAX)
            and (end_disp_p75 < NORMAL_DISPLACEMENT_MIN)
        )
        if tangled_obvious:
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.03,
                    "TRAPPED": 0.06,
                    "TANGLED": 0.83,
                    "MALFUNCTION": 0.03,
                }
            )

        # BURIED の強い兆候（混在ケースの誤検知対策）:
        # - 末端はほとんど移動しないが、細かい往復(反転)が多い
        #   → TRAPPED/TANGLED の補助(tangled_hint)で TANGLED に引っ張られないように先に確定する。
        # 反転(reversals)は微小変位でスパイクしやすい。
        # p75だけだと単発のスパイクでBURIEDが誤発火するため、中央値も併せて要求する。
        buried_by_reversals = (
            (end_disp_p75 < TRAPPED_DISPLACEMENT_MAX)
            and (reversals_p75 >= BURIED_REVERSALS_MIN)
            and (reversals_med >= (BURIED_REVERSALS_MIN / 2.0))
        )
        if buried_by_reversals:
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.85,
                    "TRAPPED": 0.04,
                    "TANGLED": 0.03,
                    "MALFUNCTION": 0.03,
                }
            )

        # BURIED / TRAPPED の分岐:
        # - 角度変化が小さく、末端移動も小さい → 強い拘束
        #   ここで「反転(揺れ)」の大小で BURIED と TRAPPED を分ける。
        delta_theta_for_buried = delta_theta_deg if "delta_theta_deg_max" in f else delta_theta_peak
        stuck_hard = (delta_theta_for_buried < BURIED_ANGLE_THRESHOLD) and (end_disp_p75 < BURIED_DISPLACEMENT_THRESHOLD)
        if stuck_hard:
            if reversals_med >= BURIED_REVERSALS_MIN:
                return normalize_distribution(
                    {
                        "NONE": 0.05,
                        "BURIED": 0.85,
                        "TRAPPED": 0.03,
                        "TANGLED": 0.03,
                        "MALFUNCTION": 0.03,
                    }
                )

            # 反転が少ないのに強く拘束されている場合は TRAPPED 寄り
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.08,
                    "TRAPPED": 0.80,
                    "TANGLED": 0.04,
                    "MALFUNCTION": 0.03,
                }
            )

        # end_disp が十分大きいなら正常優先
        if end_disp_p75 >= NORMAL_DISPLACEMENT_MIN:
            return normalize_distribution(
                {
                    "NONE": 0.85,
                    "BURIED": 0.03,
                    "TRAPPED": 0.03,
                    "TANGLED": 0.03,
                    "MALFUNCTION": 0.05,
                }
            )

        trapped = (delta_theta_peak >= TRAPPED_ANGLE_MIN) and (
            BURIED_DISPLACEMENT_THRESHOLD < end_disp_p75 < TRAPPED_DISPLACEMENT_MAX
        )
        if trapped:
            base_trapped = clamp(0.75 + 0.20 * (1.0 - end_disp_p75 / TRAPPED_DISPLACEMENT_MAX))
            # TANGLED兆候が強いときはTRAPPEDを下げ、TANGLEDを上げる（離散分岐ではなく連続的に）
            trapped_score = clamp(base_trapped * (1.0 - 0.80 * tangled_hint))
            tangled_score = 0.05 + 0.70 * tangled_hint
            return normalize_distribution(
                {
                    "NONE": 0.05,
                    "BURIED": 0.05,
                    "TRAPPED": trapped_score,
                    "TANGLED": tangled_score,
                    "MALFUNCTION": 0.04,
                }
            )

        # TRAPPED の「中間帯」: 角度変化は少しあるが、ほぼ移動しない。
        # - BURIED ほど角度が極小ではない
        # - TRAPPED_ANGLE_MIN に届かない場合でも TRAPPED 寄りにする
        trapped_mid = (end_disp < BURIED_DISPLACEMENT_THRESHOLD) and (
            BURIED_ANGLE_THRESHOLD <= delta_theta_peak < TRAPPED_ANGLE_MIN
        )
        if trapped_mid:
            # TRAPPED_ANGLE_MIN に届かない帯域は、TRAPPEDとBURIEDの境界になりやすい。
            # BURIEDは「角度変化がより小さい」傾向があるので、角度が小さいほどBURIEDへ滑らかに寄せる。
            denom = max(config.EPSILON, (TRAPPED_ANGLE_MIN - BURIED_ANGLE_THRESHOLD))
            buried_hint_mid = clamp((TRAPPED_ANGLE_MIN - delta_theta_peak) / denom)
            return normalize_distribution(
                {
                    "NONE": 0.06,
                    # ただし、絡まりの根拠(tangled_hint)が強い場合はBURIEDを下げる
                    "BURIED": (0.10 + 0.90 * buried_hint_mid) * (1.0 - 0.80 * tangled_hint),
                    "TRAPPED": 0.72 * (1.0 - 0.80 * tangled_hint) * (1.0 - 0.75 * buried_hint_mid),
                    "TANGLED": 0.06 + 0.70 * tangled_hint,
                    "MALFUNCTION": 0.05,
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
        t_hi_theta = clamp((delta_theta_peak - 2.0) / 1.3) * clamp((0.016 - end_disp_p75) / 0.010)
        tangled_w = clamp(0.65 * t_rev + 0.25 * t_straight + 0.10 * t_hi_theta)
        # ほぼ移動していない場合は「絡まり」を過大評価しない（TRAPPED/BURIEDと混同しやすい）
        move_hint = clamp((end_disp_peak - 0.002) / 0.010)
        tangled_w = tangled_w * (0.20 + 0.80 * move_hint)

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
            }
        )
