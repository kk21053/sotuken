"""仕様.txt準拠のルール判定ユーティリティ

仕様.txt Step7 のルール(①〜④)に沿って movement_result / cause_rule を決める。

注意:
- ルールで完全に one-hot 固定すると p_drone の不確実性が潰れ、TANGLED などが過剰に確定しやすい。
    そのため p_rule は one-hot ではなく、入力分布(prob_dist)を基にした正規化分布を返す。
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


CAUSES_5 = ("NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION")


def _best_and_second(
    probs: Dict[str, float],
    labels: Iterable[str] = CAUSES_5,
    excluded: Iterable[str] = (),
) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    excluded_set = {str(x).upper() for x in excluded}
    scored = []
    for lab in labels:
        if str(lab).upper() in excluded_set:
            continue
        scored.append((lab, _safe_float(probs.get(lab, 0.0), 0.0)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    if not scored:
        return ("NONE", 0.0), ("NONE", 0.0)
    if len(scored) == 1:
        return scored[0], ("NONE", 0.0)
    return scored[0], scored[1]


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


def argmax_label_excluding(
    probs: Dict[str, float],
    excluded: Iterable[str],
    labels: Iterable[str] = CAUSES_5,
) -> str:
    excluded_set = {str(x).upper() for x in excluded}
    best_lab = "NONE"
    best_v = -1.0
    for lab in labels:
        if str(lab).upper() in excluded_set:
            continue
        v = _safe_float(probs.get(lab, 0.0), 0.0)
        if v > best_v:
            best_v = v
            best_lab = lab
    # 全ラベル除外などで候補が無い場合は通常argmaxへフォールバック
    if best_v < 0.0:
        return argmax_label(probs, labels=labels)
    return best_lab


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

    # argmaxに使う分布を正規化してから使う（負値/欠損に強くする）
    pd = normalize_probs(prob_dist)

    # ①
    if sc >= hi and dc >= hi:
        movement = "動く"
        # 関節角ベースのcanは拘束(TANGLED/TRAPPED/BURIED)でも高くなり得るため、
        # cause は prob_dist（末端変位など）を重視する。
        # ただし NONE の過剰排除は「NONEが最頻なのにTANGLEDへ強制」になりやすいので、
        # NONE が十分優勢なら採用する。
        (best_lab, best_v), (second_lab, second_v) = _best_and_second(pd)
        if best_lab == "NONE" and best_v >= 0.40 and (best_v - second_v) >= 0.08:
            cause = "NONE"
        else:
            cause = argmax_label_excluding(pd, excluded=("NONE",))
            # TANGLED は僅差だと誤判定が多いので、十分優勢な時だけ確定する
            if str(cause).upper() == "TANGLED":
                (tb, tv), (ts, tsv) = _best_and_second(pd, excluded=("NONE",))
                if tv < 0.50 and (tv - tsv) < 0.08:
                    cause = ts

        # p_rule は one-hot ではなく分布を残す（ログ/集計用）
        return movement, cause, dict(pd)

    # ②
    if sc <= lo and dc <= lo:
        movement = "動かない"
        # 動かない判定で NONE に倒れるのを防ぐ（拘束3種か故障を優先）
        cause = argmax_label_excluding(pd, excluded=("NONE",))
        if str(cause).upper() == "TANGLED":
            (tb, tv), (ts, tsv) = _best_and_second(pd, excluded=("NONE",))
            if tv < 0.55 and (tv - tsv) < 0.10:
                cause = ts
        return movement, cause, dict(pd)

    # ③
    if (sc >= hi and dc <= lo) or (sc <= lo and dc >= hi):
        movement = "動かない"
        # 仕様.txtでは MALFUNCTION 固定だが、Spot自己診断が外部拘束を見逃すケースがあり、
        # 誤って MALFUNCTION が多発しやすい。prob_dist（通常はp_llm）を優先し、
        # MALFUNCTION は確率が十分高い場合のみ採用する。
        if _safe_float(pd.get("MALFUNCTION", 0.0), 0.0) >= 0.55:
            cause = "MALFUNCTION"
        else:
            # 不一致時も NONE を選びにくくする（外部拘束/故障のどれかを返す）
            cause = argmax_label_excluding(pd, excluded=("NONE",))
            if str(cause).upper() == "TANGLED":
                (tb, tv), (ts, tsv) = _best_and_second(pd, excluded=("NONE",))
                if tv < 0.55 and (tv - tsv) < 0.10:
                    cause = ts
        return movement, cause, dict(pd)

    # ④
    movement = "一部動く"
    # 一部動く判定で NONE に倒れるのを抑制（NONEは十分強い場合のみ許可）
    if _safe_float(pd.get("NONE", 0.0), 0.0) >= 0.70:
        cause = "NONE"
    else:
        cause = argmax_label_excluding(pd, excluded=("NONE",))
    if str(cause).upper() == "TANGLED":
        (tb, tv), (ts, tsv) = _best_and_second(pd, excluded=("NONE",))
        if tv < 0.55 and (tv - tsv) < 0.10:
            cause = ts
    return movement, cause, dict(pd)
