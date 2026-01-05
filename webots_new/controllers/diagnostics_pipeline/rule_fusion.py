"""仕様.txt準拠のルール判定 + VLM融合ユーティリティ

仕様.txt Step7 のルール(①〜④)をそのまま実装する。
- 入力: spot_can, drone_can, p_drone(拘束状況の確率分布)
- 出力: movement_result, cause_rule, p_rule(one-hot)

さらに、VLMの確率分布(vlm_probs)とルール(one-hot)を重み付け融合する。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


CAUSES_5 = ("NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION")


def _safe_float(x: object, default: float) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def normalize_probs(probs: Dict[str, float], labels: Iterable[str] = CAUSES_5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = 0.0
    for lab in labels:
        v = _safe_float(probs.get(lab, 0.0), 0.0)
        if v < 0.0:
            v = 0.0
        out[lab] = v
        s += v
    if s <= 0.0:
        n = len(tuple(labels))
        if n <= 0:
            return {}
        return {lab: 1.0 / n for lab in labels}
    return {lab: out[lab] / s for lab in out}


def argmax_label(probs: Dict[str, float], labels: Iterable[str] = CAUSES_5) -> str:
    best_lab = "NONE"
    best_v = -1.0
    for lab in labels:
        v = _safe_float(probs.get(lab, 0.0), 0.0)
        if v > best_v:
            best_v = v
            best_lab = lab
    return best_lab


def one_hot(label: str, labels: Iterable[str] = CAUSES_5) -> Dict[str, float]:
    return {lab: (1.0 if lab == label else 0.0) for lab in labels}


def rule_based_decision(
    spot_can: float,
    drone_can: float,
    prob_dist: Dict[str, float],
    lo: float = 0.3,
    hi: float = 0.7,
) -> Tuple[str, str, Dict[str, float]]:
    """仕様.txt Step7 の①〜④に従って (movement_result, cause_rule, p_rule) を返す。

    argmaxに使う確率分布は、呼び出し側が与える（通常はp_llm優先、無ければp_drone）。
    """

    sc = _safe_float(spot_can, 0.5)
    dc = _safe_float(drone_can, 0.5)

    # ①
    if sc >= hi and dc >= hi:
        movement = "動く"
        cause = "NONE"
        return movement, cause, one_hot(cause)

    # ②
    if sc <= lo and dc <= lo:
        movement = "動かない"
        cause = argmax_label(prob_dist)
        return movement, cause, one_hot(cause)

    # ③
    if (sc >= hi and dc <= lo) or (sc <= lo and dc >= hi):
        movement = "動かない"
        cause = "MALFUNCTION"
        return movement, cause, one_hot(cause)

    # ④
    movement = "一部動く"
    cause = argmax_label(prob_dist)
    return movement, cause, one_hot(cause)


def fuse_rule_and_vlm(
    cause_rule: str,
    vlm_probs: Dict[str, float],
    rule_weight: float = 0.2,
    vlm_weight: float = 0.8,
    labels: Iterable[str] = CAUSES_5,
) -> Tuple[Dict[str, float], str]:
    """fused_probs = 0.2*one_hot(rule) + 0.8*vlm_probs, argmaxを返す。"""

    rp = one_hot(cause_rule, labels)
    vp = normalize_probs(vlm_probs, labels)

    fused: Dict[str, float] = {}
    for lab in labels:
        fused[lab] = rule_weight * rp.get(lab, 0.0) + vlm_weight * vp.get(lab, 0.0)

    fused = normalize_probs(fused, labels)
    return fused, argmax_label(fused, labels)
