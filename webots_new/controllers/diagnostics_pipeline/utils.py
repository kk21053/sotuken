"""小さなユーティリティ（JSONL 保存など）"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from . import config


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class JsonlWriter:
    """JSON Lines 形式で追記保存する"""

    def __init__(self, directory: str, filename: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / filename

    def append(self, payload: Dict) -> None:
        line = json.dumps(payload, ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in dist.values())
    if total <= config.EPSILON:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: max(0.0, v) / total for k, v in dist.items()}


def ensure_probability_keys(values: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    fixed = {k: float(values.get(k, 0.0)) for k in keys}
    return normalize_distribution(fixed)
