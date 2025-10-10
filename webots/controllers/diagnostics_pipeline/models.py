"""Dataclasses used across the diagnostics pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from . import config


@dataclass
class TrialResult:
    """Store aggregated values for a single trial."""

    leg_id: str
    trial_index: int
    direction: str
    start_time: float
    end_time: float
    self_can_raw: Optional[float] = None
    drone_can_raw: Optional[float] = None
    features_drone: Dict[str, float] = field(default_factory=dict)
    ok: bool = True
    timestamp: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class LegState:
    """Keep the running state for each leg."""

    leg_id: str
    self_can: float = 0.0
    self_moves: bool = False
    drone_can: float = 0.0
    p_drone: Dict[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(config.CAUSE_LABELS) for c in config.CAUSE_LABELS}
    )
    p_llm: Dict[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(config.CAUSE_LABELS) for c in config.CAUSE_LABELS}
    )
    p_final: Dict[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(config.CAUSE_LABELS) for c in config.CAUSE_LABELS}
    )
    cause_final: str = "NONE"
    conf_final: float = 0.0
    moves_final: bool = False
    trials: List[TrialResult] = field(default_factory=list)

    def snapshot(self) -> "LegStatus":
        return LegStatus(
            leg_id=self.leg_id,
            self_can=self.self_can,
            self_moves=self.self_moves,
            drone_can=self.drone_can,
            moves_final=self.moves_final,
            p_drone=dict(self.p_drone),
            p_llm=dict(self.p_llm),
            p_final=dict(self.p_final),
            cause_final=self.cause_final,
            conf_final=self.conf_final,
        )


@dataclass
class SessionState:
    """Session level information shared across legs."""

    session_id: str
    fallen: bool = False
    legs: Dict[str, LegState] = field(default_factory=dict)

    def ensure_leg(self, leg_id: str) -> LegState:
        if leg_id not in self.legs:
            self.legs[leg_id] = LegState(leg_id=leg_id)
        return self.legs[leg_id]


@dataclass
class LegStatus:
    leg_id: str
    self_can: float
    self_moves: bool
    drone_can: float
    moves_final: bool
    p_drone: Dict[str, float]
    p_llm: Dict[str, float]
    p_final: Dict[str, float]
    cause_final: str
    conf_final: float


@dataclass
class SessionRecord:
    session_id: str
    fallen: bool
    legs: Dict[str, LegStatus]

    def to_dict(self) -> Dict:
        return {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "fallen": self.fallen,
            "legs": {
                leg_id: {
                    "self_can": leg.self_can,
                    "self_moves": leg.self_moves,
                    "drone_can": leg.drone_can,
                    "moves_final": leg.moves_final,
                    "p_drone": leg.p_drone,
                    "p_llm": leg.p_llm,
                    "p_final": leg.p_final,
                    "cause_final": leg.cause_final,
                    "conf_final": leg.conf_final,
                }
                for leg_id, leg in self.legs.items()
            },
        }
