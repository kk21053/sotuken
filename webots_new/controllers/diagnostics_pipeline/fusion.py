"""確率分布から最も確率が高いラベルを返す"""

from typing import Dict


def select_cause(distribution: Dict[str, float]) -> str:
    best_label = "NONE"
    best_value = -1.0
    for label, value in distribution.items():
        if value > best_value:
            best_label = label
            best_value = value
    return best_label
