"""Probability fusion utilities."""

from __future__ import annotations

from typing import Dict

from . import config
from .utils import normalize_distribution


def fuse_probabilities(p_drone: Dict[str, float], p_llm: Dict[str, float]) -> Dict[str, float]:
    fused = {}
    weights = config.FUSION_WEIGHTS
    for label in config.CAUSE_LABELS:
        fused[label] = (
            weights["drone"] * float(p_drone.get(label, 0.0))
            + weights["llm"] * float(p_llm.get(label, 0.0))
        )
    return normalize_distribution(fused)


def select_cause(distribution: Dict[str, float]) -> str:
    best_label = "NONE"
    best_value = -1.0
    for label, value in distribution.items():
        if value > best_value:
            best_label = label
            best_value = value
    return best_label
