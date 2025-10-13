"""Dataclasses used across the diagnostics pipeline (仕様準拠版)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from . import config


@dataclass
class TrialResult:
    """単一試行の結果を格納（仕様のステップ1-2）"""

    leg_id: str
    trial_index: int
    direction: str
    start_time: float
    end_time: float
    self_can_raw: Optional[float] = None      # 仕様ステップ3で使用
    drone_can_raw: Optional[float] = None     # 仕様ステップ4で使用
    ok: bool = True  # ドローン観測成功フラグ（ログ用）

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class LegState:
    """各脚の状態を保持（仕様の診断フローに対応）"""

    leg_id: str
    
    # 仕様ステップ3: spot_can（シグモイド変換後の動作確率）
    spot_can: float = 0.5
    
    # 仕様ステップ4: drone_can（シグモイド変換後の動作確率）と確率分布
    drone_can: float = 0.5
    p_drone: Dict[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(config.CAUSE_LABELS) for c in config.CAUSE_LABELS}
    )
    
    # 仕様ステップ7: LLMの判定結果
    p_llm: Dict[str, float] = field(
        default_factory=lambda: {c: 1.0 / len(config.CAUSE_LABELS) for c in config.CAUSE_LABELS}
    )
    movement_result: str = "一部動く"  # "動く" | "動かない" | "一部動く"
    cause_final: str = "NONE"
    p_can: float = 0.0  # 仕様ステップ7: 最終的な動作確率
    
    trials: List[TrialResult] = field(default_factory=list)

    def snapshot(self) -> "LegStatus":
        return LegStatus(
            leg_id=self.leg_id,
            spot_can=self.spot_can,
            drone_can=self.drone_can,
            p_drone=self.p_drone.copy(),
            p_llm=self.p_llm.copy(),
            movement_result=self.movement_result,
            cause_final=self.cause_final,
            p_can=self.p_can,
        )


@dataclass
class SessionState:
    """セッション全体の状態（仕様の診断フロー全体）"""

    session_id: str
    fallen: bool = False           # 仕様ステップ8: 転倒判定結果
    fallen_probability: float = 0.0  # 転倒確率
    legs: Dict[str, LegState] = field(default_factory=dict)

    def ensure_leg(self, leg_id: str) -> LegState:
        if leg_id not in self.legs:
            self.legs[leg_id] = LegState(leg_id=leg_id)
        return self.legs[leg_id]


@dataclass
class LegStatus:
    """結果表示用の脚の状態スナップショット"""
    leg_id: str
    spot_can: float
    drone_can: float
    p_drone: Dict[str, float]
    p_llm: Dict[str, float]
    movement_result: str  # "動く" | "動かない" | "一部動く"
    cause_final: str
    p_can: float


@dataclass
class SessionRecord:
    """ログ記録用のセッション記録（仕様ステップ9）"""
    session_id: str
    fallen: bool
    fallen_probability: float
    legs: Dict[str, LegStatus]

    def to_dict(self) -> Dict:
        return {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "fallen": self.fallen,
            "fallen_probability": self.fallen_probability,
            "legs": {
                leg_id: {
                    "spot_can": leg.spot_can,
                    "drone_can": leg.drone_can,
                    "p_drone": leg.p_drone,
                    "p_llm": leg.p_llm,
                    "movement_result": leg.movement_result,
                    "cause_final": leg.cause_final,
                    "p_can": leg.p_can,
                }
                for leg_id, leg in self.legs.items()
            },
        }
