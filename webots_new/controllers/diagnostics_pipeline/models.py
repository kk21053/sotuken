"""診断で使うデータ構造（dataclass）"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import config


def _uniform_dist() -> Dict[str, float]:
    n = len(config.CAUSE_LABELS)
    return {c: 1.0 / n for c in config.CAUSE_LABELS}


@dataclass
class TrialResult:
    leg_id: str
    trial_index: int
    direction: str
    start_time: float
    end_time: float
    self_can_raw: Optional[float] = None
    drone_can_raw: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)
    ok: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class LegState:
    leg_id: str

    # Spot / Drone の動作確率（0..1）
    spot_can: float = 0.5
    drone_can: float = 0.5

    # Drone/LLM の確率分布
    p_drone: Dict[str, float] = field(default_factory=_uniform_dist)
    p_llm: Dict[str, float] = field(default_factory=_uniform_dist)

    # 最終結果
    movement_result: str = "一部動く"
    cause_final: str = "NONE"
    p_can: float = 0.0

    # 仕様.txt準拠のルールベース判定
    cause_rule: Optional[str] = None
    p_rule: Optional[Dict[str, float]] = None

    # シナリオの期待値（正解率表示に使う）
    expected_cause: str = "NONE"

    trials: List[TrialResult] = field(default_factory=list)

    def snapshot(self) -> "LegStatus":
        return LegStatus(
            leg_id=self.leg_id,
            spot_can=self.spot_can,
            drone_can=self.drone_can,
            p_drone=dict(self.p_drone),
            p_llm=dict(self.p_llm),
            movement_result=self.movement_result,
            cause_final=self.cause_final,
            p_can=self.p_can,
            cause_rule=self.cause_rule,
            p_rule=dict(self.p_rule) if self.p_rule else None,
            expected_cause=self.expected_cause,
        )


@dataclass
class SessionState:
    session_id: str
    image_path: Optional[str] = None
    legs: Dict[str, LegState] = field(default_factory=dict)

    def ensure_leg(self, leg_id: str) -> LegState:
        if leg_id not in self.legs:
            self.legs[leg_id] = LegState(leg_id=leg_id)
        return self.legs[leg_id]


@dataclass
class LegStatus:
    leg_id: str
    spot_can: float
    drone_can: float
    p_drone: Dict[str, float]
    p_llm: Dict[str, float]
    movement_result: str
    cause_final: str
    p_can: float
    cause_rule: Optional[str] = None
    p_rule: Optional[Dict[str, float]] = None
    expected_cause: str = "NONE"


@dataclass
class SessionRecord:
    session_id: str
    image_path: Optional[str]
    legs: Dict[str, LegStatus]

    def to_dict(self) -> Dict:
        return {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "image_path": self.image_path,
            "legs": {
                leg_id: {
                    "spot_can": leg.spot_can,
                    "drone_can": leg.drone_can,
                    "p_drone": leg.p_drone,
                    "p_llm": leg.p_llm,
                    "movement_result": leg.movement_result,
                    "cause_final": leg.cause_final,
                    "p_can": leg.p_can,
                    "cause_rule": leg.cause_rule,
                    "p_rule": leg.p_rule,
                    "expected_cause": leg.expected_cause,
                }
                for leg_id, leg in self.legs.items()
            },
        }
