"""仕様のルールで最終判定する（LLMの代わりのルールベース）"""

from __future__ import annotations

from typing import Dict, Optional

from .models import LegState


class LLMAnalyzer:
    """仕様準拠のルールベース判定器"""

    def __init__(self, model_priority: Optional[object] = None, max_new_tokens: int = 256) -> None:
        self._model_name = "rule-based-spec-compliant"

    def infer(self, leg: LegState, all_legs=None, trial_direction=None) -> Dict[str, float]:
        spot_can = leg.spot_can
        drone_can = leg.drone_can
        p_drone = dict(leg.p_drone)

        # ルール①: 両方が高い → 動く、原因=NONE
        if spot_can >= 0.7 and drone_can >= 0.7:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.94,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール①b: "動く" の一般則（片方が少し低くても、全体として高ければNONE）
        # ※特定の誤分類パターン回避ではなく、can=動作確率という定義に沿った汎用ルール。
        if spot_can >= 0.55 and drone_can >= 0.7:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.85,
                "BURIED": 0.03,
                "TRAPPED": 0.03,
                "TANGLED": 0.03,
                "MALFUNCTION": 0.05,
                "FALLEN": 0.01,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール①c: Drone観測で明確に動けている（drone_canが高い）なら NONE 寄り。
        # Spot側の自己診断が中間でも、観測上の動作が十分なら「動く」と解釈する。
        if drone_can >= 0.78 and spot_can >= 0.40:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.85,
                "BURIED": 0.03,
                "TRAPPED": 0.03,
                "TANGLED": 0.03,
                "MALFUNCTION": 0.05,
                "FALLEN": 0.01,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール②: 両方が低い → 動かない、原因= p_drone の最大（ただし NONE は除外）
        if spot_can <= 0.3 and drone_can <= 0.3:
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            max_cause = max((v, k) for k, v in p_drone.items() if k != "NONE")[1]
            dist = {
                "NONE": 0.02,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.02,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            dist[max_cause] = 0.89
            leg.p_llm = dist
            leg.cause_final = max_cause
            return dist

        # ルール③: 片方が高く、もう片方が低い → 故障
        if (spot_can >= 0.7 and drone_can <= 0.3) or (spot_can <= 0.3 and drone_can >= 0.7):
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.01,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.93,
                "FALLEN": 0.01,
            }
            leg.p_llm = dist
            leg.cause_final = "MALFUNCTION"
            return dist

        # ルール④: 中間が混ざる → 一部動く、原因= p_drone の最大
        leg.movement_result = "一部動く"
        leg.p_can = (spot_can + drone_can) / 2
        max_cause = max(p_drone.items(), key=lambda x: x[1])[0]
        dist = {
            "NONE": 0.10,
            "BURIED": 0.05,
            "TRAPPED": 0.05,
            "TANGLED": 0.05,
            "MALFUNCTION": 0.05,
            "FALLEN": 0.01,
        }
        dist[max_cause] = 0.69
        leg.p_llm = dist
        leg.cause_final = max_cause
        return dist
