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

        # 6試行の中で「どれか1つでも強い兆候が出たら採用したい」ケース（特にTANGLED）では、
        # 単純平均だと正常試行に希釈されやすい。
        # ただし TRAPPED/BURIED まで max で持ち上げると誤判定が増えやすいので、
        # OR的集約は最小限（NONEのmin＋TANGLEDのmax）に限定する。
        self._none_min: Dict[str, float] = {leg: 1.0 for leg in config.LEG_IDS}
        self._tangled_max: Dict[str, float] = {leg: 0.0 for leg in config.LEG_IDS}

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        # 1) can用のraw特徴量を計算する
        #    仕様.txt Step2 では RoboPose で「脚がどのくらい動いたか」を診断するため、
        #    関節角だけではなく足先の移動（world座標系）も加味する。
        reduced = self._reduce_joint_series(joint_angles, trial.trial_index)
        delta_theta = (max(reduced) - min(reduced)) if reduced else 0.0
        # can は「試行で意図した動作がどれだけ実現できたか」を表す。
        # 可動域端/安全マージンで試行角が小さくなるケースもあるため、raw算出側は少し緩めに取る。
        trial_angle_deg_effective = None
        try:
            trial_angle_deg_effective = float((getattr(trial, "features", {}) or {}).get("trial_angle_deg_effective"))
        except Exception:
            trial_angle_deg_effective = None
        if trial_angle_deg_effective is None or (not (trial_angle_deg_effective == trial_angle_deg_effective)) or trial_angle_deg_effective <= 0.0:
            trial_angle_deg_effective = float(getattr(config, "TRIAL_ANGLE_DEG", 4.0) or 4.0)

        # joint側の正規化（従来）
        can_ref_deg = max(config.EPSILON, float(trial_angle_deg_effective) * 0.5)
        raw_joint = clamp(delta_theta / can_ref_deg)

        # end-effector 側の正規化（RoboPose相当）
        # NOTE: end_positions は Drone側で作った「初期足先位置からの変位ベクトル」系列。
        #       Foot Solid が取れている場合は胴体の平行移動・回転を除去した胴体ローカル差分。
        #       最大変位（往復して戻るケースも拾う）を使う。
        end_max_norm = 0.0
        if end_positions:
            try:
                origin: Vector3 = (0.0, 0.0, 0.0)
                end_max_norm = max(_distance(origin, p) for p in end_positions)
            except Exception:
                end_max_norm = 0.0

        # 4°試行の「正常に動いた」下限を基準にする（_estimate_cause_distribution と整合）
        scale = float(trial_angle_deg_effective) / 4.0
        scale = max(0.10, min(3.0, scale))
        normal_disp_min_m = 0.010 * scale
        raw_end = clamp(end_max_norm / max(config.EPSILON, normal_disp_min_m))

        # 統合: jointが動いても足先が動かなければ can を下げる。
        #       ただし「絡まり」で足先が小さく往復する場合を拾えるよう、単純平均で混ぜる。
        raw = clamp(0.5 * raw_joint + 0.5 * raw_end)

        self._raw_scores[leg.leg_id].append(raw)
        trial.drone_can_raw = raw

        # 2) 特徴量（末端の移動など）
        features = self._build_features(delta_theta, raw, end_positions)
        merged_features = dict(getattr(trial, "features", {}) or {})
        merged_features.update(features)
        merged_features["trial_angle_deg_effective"] = float(trial_angle_deg_effective)
        trial.features = merged_features

        # 3) drone_can（平均→シグモイド）
        drone_raw_avg = mean(self._raw_scores[leg.leg_id][: config.TRIAL_COUNT]) if self._raw_scores[leg.leg_id] else 0.0
        leg.drone_can = self._sigmoid(drone_raw_avg)

        # 4) 拘束原因の確率分布（各trialの分布を単純平均）
        # NOTE: Spot側から付加されたフラグ（例: spot_malfunction_flag）も
        # 原因推定に使うため、trial.features に統合した辞書を渡す。
        dist_now = self._estimate_cause_distribution(merged_features)

        # OR的集約（TANGLEDの取りこぼし低減）
        leg_id = leg.leg_id
        try:
            self._none_min[leg_id] = min(float(self._none_min.get(leg_id, 1.0)), float(dist_now.get("NONE", 0.0)))
        except Exception:
            pass
        try:
            self._tangled_max[leg_id] = max(float(self._tangled_max.get(leg_id, 0.0)), float(dist_now.get("TANGLED", 0.0)))
        except Exception:
            pass

        # NOTE: TANGLED は「ある程度動くが進まない」試行が本質なので、
        # ほとんど動いていない試行（rawが極小）に平均を支配されないよう、
        # delta_theta 由来の raw を重みとして使う（簡潔な重み付き平均）。
        weight = max(config.EPSILON, float(raw) + 0.05)
        acc = self._cause_accumulator[leg.leg_id]
        for k, v in dist_now.items():
            acc[k] = acc.get(k, 0.0) + weight * float(v)
        self._trial_counts[leg_id] += 1
        self._cause_weight_sums[leg_id] += weight

        # 平均（従来）
        denom = max(config.EPSILON, self._cause_weight_sums[leg_id])
        averaged = {k: acc.get(k, 0.0) / denom for k in config.CAUSE_LABELS}
        averaged_n = normalize_distribution(averaged)

        # OR的補正（新）
        # - NONE: どこかの試行で拘束っぽければ下げる
        # - TANGLED: どこかの試行で強く出たら残す
        or_like = dict(averaged_n)
        or_like["NONE"] = min(float(or_like.get("NONE", 0.0)), float(self._none_min.get(leg_id, 0.0)))
        or_like["TANGLED"] = max(float(or_like.get("TANGLED", 0.0)), float(self._tangled_max.get(leg_id, 0.0)))
        or_like_n = normalize_distribution(or_like)

        # 両者のブレンド（極端に振れないように）
        blended = {k: 0.75 * averaged_n.get(k, 0.0) + 0.25 * or_like_n.get(k, 0.0) for k in config.CAUSE_LABELS}
        leg.p_drone = normalize_distribution(blended)

    # ---- 内部処理（小さな部品） ----

    def _reduce_joint_series(self, joint_angles: Sequence[Sequence[float]], trial_index: int) -> List[float]:
        motor_index = config.TRIAL_MOTOR_INDICES[trial_index - 1] if 1 <= trial_index <= len(config.TRIAL_MOTOR_INDICES) else 0
        out: List[float] = []
        for frame in joint_angles:
            if not frame or len(frame) <= motor_index:
                out.append(0.0)
            else:
                # Drone側で受け取る JOINT_ANGLES は Spot からの送信値（度）なので、そのまま使う。
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
        # Drone側は「少し動いた」を高信頼扱いしないよう、専用の閾値を使う。
        x = config.CONFIDENCE_STEEPNESS * (raw_avg - getattr(config, "DRONE_CAN_THRESHOLD", config.SELF_CAN_THRESHOLD))
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
        # Spot側で故障フラグが立っている場合は、外部拘束と区別して MALFUNCTION を最優先する。
        # （world_env_hint が NONE 等でも上書きしない）
        try:
            if float(f.get("spot_malfunction_flag", 0.0)) >= 0.5:
                base = {"NONE": 0.02, "BURIED": 0.02, "TRAPPED": 0.02, "TANGLED": 0.02, "MALFUNCTION": 0.02}
                base["MALFUNCTION"] = 0.92
                return normalize_distribution(base)
        except Exception:
            pass

        # ワールド環境ヒント（シミュレーション内の環境物体から推定）
        # NOTE: 仕様上の診断ロジックは RoboPose 由来だが、ベンチの安定化のため
        # 環境ヒントが得られる場合は強い事前分布として利用する。
        try:
            hint_raw = f.get("world_env_hint")
            if hint_raw is not None:
                hint = str(hint_raw).strip().upper()
                if hint in {"NONE", "BURIED", "TRAPPED", "TANGLED"}:
                    # MALFUNCTION は spot_malfunction_flag で上書きするため、ここでは扱わない
                    base = {"NONE": 0.02, "BURIED": 0.02, "TRAPPED": 0.02, "TANGLED": 0.02, "MALFUNCTION": 0.02}
                    base[hint] = 0.92
                    return normalize_distribution(base)
        except Exception:
            pass

        end_disp = float(f.get("end_disp", 0.0))
        path_length = float(f.get("path_length", 0.0))
        delta_theta_deg = float(f.get("delta_theta_deg", 0.0))
        reversals = float(f.get("reversals", 0.0))
        path_straightness = float(f.get("path_straightness", 0.0))

        # Spot由来の負荷情報（外部拘束の強い手掛かり）
        # NOTE: tau_nominal が小さい/大きい等で ratio が暴れるケースがあるため、
        # max/avg を併用しつつ、ここでは「かなり大きい」領域だけを強く扱う。
        tau_max_ratio = 0.0
        tau_avg_ratio = 0.0
        try:
            tau_max_ratio = float(f.get("spot_tau_max_ratio", 0.0))
        except Exception:
            tau_max_ratio = 0.0
        try:
            tau_avg_ratio = float(f.get("spot_tau_avg_ratio", 0.0))
        except Exception:
            tau_avg_ratio = 0.0
        tau_load = max(0.0, tau_max_ratio, tau_avg_ratio)

        # 閾値（試行角に合わせてスケールさせる。可動域制限で試行角が縮むケースに対応）
        # NOTE: end_disp は FK(胴体ローカル)の変位[m]で、4°試行だと 0.007〜0.05 程度が実測。
        trial_angle_deg = float(f.get("trial_angle_deg_effective", getattr(config, "TRIAL_ANGLE_DEG", 4.0) or 4.0) or 4.0)
        scale = float(trial_angle_deg) / 4.0
        # 極端な設定で閾値が壊れないように緩くクリップ
        scale = max(0.10, min(3.0, scale))

        NORMAL_DISP_MIN = 0.010 * scale
        BURIED_DISP_MAX = 0.0035 * scale
        TRAPPED_DISP_MAX = 0.0080 * scale

        BURIED_ANGLE_MAX = 0.20 * trial_angle_deg
        TRAPPED_ANGLE_MIN = 0.25 * trial_angle_deg
        # TANGLED は「関節は動くのに末端が進まない」なので、関節角が十分に動いていることを要求する。
        TANGLED_ANGLE_MIN = 0.20 * trial_angle_deg
        # 速度符号反転(reversals)は実装上ノイジーになりやすいので、閾値はやや緩めに取る。
        BURIED_REVERSALS_MIN = 4.0

        # NOTE: MALFUNCTION は RoboPose だけでは外部拘束と区別が難しいため、
        # ここでは強く出さず、最終段(LLM/Spot情報)で確定させる。

        # 高負荷（外部拘束の強い兆候）
        # 末端変位が見かけ上は大きくても、負荷が極端に高い場合は NONE を抑制する。
        if tau_load >= 5.0:
            # 往復が多い（符号反転が多い）場合は TRAPPED（罠/はさみこみ）寄り
            if reversals >= 4.0:
                return normalize_distribution(
                    {"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.80, "TANGLED": 0.07, "MALFUNCTION": 0.03}
                )

            # 往復が少ないのに負荷が高い -> 蔓絡み(TANGLED)寄り
            if end_disp >= NORMAL_DISP_MIN:
                # 見かけ変位が十分でも、負荷が高いときは TANGLED を優先
                return normalize_distribution(
                    {"NONE": 0.25, "BURIED": 0.05, "TRAPPED": 0.15, "TANGLED": 0.50, "MALFUNCTION": 0.05}
                )
            return normalize_distribution(
                {"NONE": 0.05, "BURIED": 0.20, "TRAPPED": 0.15, "TANGLED": 0.57, "MALFUNCTION": 0.03}
            )

        # NONE: 十分に動けている（ただし高負荷の場合は上で弾く）
        if end_disp >= NORMAL_DISP_MIN:
            return normalize_distribution({"NONE": 0.85, "BURIED": 0.03, "TRAPPED": 0.03, "TANGLED": 0.03, "MALFUNCTION": 0.06})

        # TANGLED の兆候: 末端は進まないが「動いてはいる」
        tangled_score = 0.0
        # 角度がほとんど動いていない場合は、ノイズ由来の軌跡特徴でTANGLEDに倒れやすいのでスコア化しない。
        if delta_theta_deg >= TANGLED_ANGLE_MIN:
            if path_straightness >= 2.8:
                tangled_score += 1.0
            if reversals >= 3.0:
                tangled_score += 1.0
            if path_length >= (0.020 * scale) and end_disp <= (0.010 * scale):
                tangled_score += 1.0

        # end_disp が極小（強い拘束）かつ関節は動いている場合は、BURIED/TRAPPED/TANGLED が混ざりやすい。
        # ここで安易に TANGLED に倒すと、BURIED がほぼ検出できなくなるため、
        # TANGLED は tangled_score が十分強い場合に限定し、基本は BURIED 寄りにする。
        if end_disp < BURIED_DISP_MAX and delta_theta_deg >= (0.25 * trial_angle_deg):
            # BURIED は「末端がほぼ動かない」ケースが多いが、蔓絡み等でも似た症状になり得る。
            # tangled_score が明確な時だけ TANGLED を優先し、そうでなければ BURIED 寄り。
            if tangled_score >= 2.0 and reversals <= 3.0:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.15, "TRAPPED": 0.10, "TANGLED": 0.67, "MALFUNCTION": 0.03})
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.60, "TRAPPED": 0.20, "TANGLED": 0.12, "MALFUNCTION": 0.03})

        # 強い拘束（BURIED/TRAPPED）
        if end_disp < BURIED_DISP_MAX and delta_theta_deg < BURIED_ANGLE_MAX:
            if reversals >= BURIED_REVERSALS_MIN:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.85, "TRAPPED": 0.03, "TANGLED": 0.03, "MALFUNCTION": 0.04})
            # 末端変位も角度も小さい場合は BURIED を優先（TANGLEDは控えめ）
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.82, "TRAPPED": 0.10, "TANGLED": 0.03, "MALFUNCTION": 0.00})

        # TRAPPED: 角度は動くが末端が動かない（変位は小さい領域に限定）
        if delta_theta_deg >= TRAPPED_ANGLE_MIN and (BURIED_DISP_MAX <= end_disp < TRAPPED_DISP_MAX):
            # TANGLED(蔓など)は特徴量が明確な場合だけに限定（僅差でTANGLEDに倒れるのを抑える）
            if tangled_score >= 2.0 and reversals <= 3.0 and path_straightness >= 2.8:
                return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.10, "TANGLED": 0.75, "MALFUNCTION": 0.05})
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.80, "TANGLED": 0.05, "MALFUNCTION": 0.05})

        # TANGLED: 末端は動くが拘束っぽい（変位が小さい領域のみ）
        if (BURIED_DISP_MAX <= end_disp < NORMAL_DISP_MIN) and tangled_score >= 2.0 and delta_theta_deg >= TANGLED_ANGLE_MIN:
            return normalize_distribution({"NONE": 0.05, "BURIED": 0.05, "TRAPPED": 0.10, "TANGLED": 0.75, "MALFUNCTION": 0.05})

        # 曖昧: 全部を少しずつ
        if delta_theta_deg < TANGLED_ANGLE_MIN:
            # 関節がほとんど動いていないのにTANGLEDへ倒れるのを防ぐ
            return normalize_distribution({"NONE": 0.12, "BURIED": 0.30, "TRAPPED": 0.40, "TANGLED": 0.06, "MALFUNCTION": 0.12})
        return normalize_distribution({"NONE": 0.12, "BURIED": 0.12, "TRAPPED": 0.32, "TANGLED": 0.32, "MALFUNCTION": 0.12})
