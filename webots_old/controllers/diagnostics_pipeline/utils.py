"""Small helper utilities for the diagnostics pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable

from . import config


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class JsonlWriter:
    """Append JSON Lines entries to a file."""

    def __init__(self, directory: str, filename: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename

    def append(self, payload: Dict) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def normalize_distribution(distribution: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in distribution.values())
    if total <= config.EPSILON:
        return {k: 1.0 / len(distribution) for k in distribution}
    return {key: max(0.0, value) / total for key, value in distribution.items()}


def ensure_probability_keys(values: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    result = {}
    for key in keys:
        result[key] = float(values.get(key, 0.0))
    return normalize_distribution(result)
