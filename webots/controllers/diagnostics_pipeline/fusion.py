"""Simple utility for selecting the most likely cause from distribution."""

from __future__ import annotations

from typing import Dict


def select_cause(distribution: Dict[str, float]) -> str:
    """
    確率分布から最も確率の高い原因を選択する（仕様のステップ7で使用）
    
    Args:
        distribution: 各原因の確率分布
    
    Returns:
        最も確率の高い原因のラベル
    """
    best_label = "NONE"
    best_value = -1.0
    for label, value in distribution.items():
        if value > best_value:
            best_label = label
            best_value = value
    return best_label
