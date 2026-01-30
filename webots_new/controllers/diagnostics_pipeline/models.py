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
    # Qwen が出力した確率分布（取得失敗時は一様フォールバック）
    p_qwen: Dict[str, float] = field(default_factory=_uniform_dist)
    p_llm: Dict[str, float] = field(default_factory=_uniform_dist)

    # LLM(Qwen) の実行状況（ログ/集計用）
    # - qwen_used: Qwenから有効な確率分布を取得できたか
    # - qwen_status: 取得できなかった場合の理由（disabled/no_path/import_error/load_error/infer_error/parse_failed など）
    qwen_used: bool = False
    qwen_status: str = "unknown"

    # 最終結果
    movement_result: str = "一部動く"
    cause_final: str = "NONE"
    p_can: float = 0.0

    # 仕様.txt準拠のルールベース判定
    cause_rule: Optional[str] = None
    p_rule: Optional[Dict[str, float]] = None

    # シナリオの期待値（正解率表示に使う）
    expected_cause: str = "NONE"

    # ---- 所要時間（段階別, 秒） ----
    # timing_total_s: 段階ごとの累積秒（trialごとに加算）
    # timing_calls: 段階ごとの計測回数
    # timing_last_s: 最後に計測した値（最新trialの内訳確認用）
    timing_total_s: Dict[str, float] = field(default_factory=dict)
    timing_calls: Dict[str, int] = field(default_factory=dict)
    timing_last_s: Dict[str, float] = field(default_factory=dict)

    trials: List[TrialResult] = field(default_factory=list)

    def add_timing(self, name: str, seconds: float) -> None:
        key = str(name)
        try:
            v = float(seconds)
        except Exception:
            return
        if v < 0:
            return
        self.timing_total_s[key] = float(self.timing_total_s.get(key, 0.0)) + v
        self.timing_calls[key] = int(self.timing_calls.get(key, 0)) + 1
        self.timing_last_s[key] = v

    def snapshot(self) -> "LegStatus":
        avg: Dict[str, float] = {}
        try:
            for k, total in (self.timing_total_s or {}).items():
                c = int((self.timing_calls or {}).get(k, 0))
                if c > 0:
                    avg[str(k)] = float(total) / float(c)
        except Exception:
            avg = {}
        return LegStatus(
            leg_id=self.leg_id,
            spot_can=self.spot_can,
            drone_can=self.drone_can,
            p_drone=dict(self.p_drone),
            p_qwen=dict(getattr(self, "p_qwen", _uniform_dist())),
            p_llm=dict(self.p_llm),
            qwen_used=bool(getattr(self, "qwen_used", False)),
            qwen_status=str(getattr(self, "qwen_status", "unknown")),
            movement_result=self.movement_result,
            cause_final=self.cause_final,
            p_can=self.p_can,
            cause_rule=self.cause_rule,
            p_rule=dict(self.p_rule) if self.p_rule else None,
            expected_cause=self.expected_cause,
            timing_total_s=dict(self.timing_total_s) if self.timing_total_s else {},
            timing_avg_s=avg,
            timing_last_s=dict(self.timing_last_s) if self.timing_last_s else {},
            timing_calls=dict(self.timing_calls) if self.timing_calls else {},
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
    p_qwen: Dict[str, float]
    p_llm: Dict[str, float]
    movement_result: str
    cause_final: str
    p_can: float

    # LLM(Qwen) の実行状況（ログ/集計用）
    qwen_used: bool = False
    qwen_status: str = "unknown"

    cause_rule: Optional[str] = None
    p_rule: Optional[Dict[str, float]] = None
    expected_cause: str = "NONE"

    # 段階別の所要時間（秒）
    timing_total_s: Optional[Dict[str, float]] = None
    timing_avg_s: Optional[Dict[str, float]] = None
    timing_last_s: Optional[Dict[str, float]] = None
    timing_calls: Optional[Dict[str, int]] = None


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
                    "p_qwen": getattr(leg, "p_qwen", _uniform_dist()),
                    "p_llm": leg.p_llm,
                    "qwen_used": bool(getattr(leg, "qwen_used", False)),
                    "qwen_status": str(getattr(leg, "qwen_status", "unknown")),
                    "movement_result": leg.movement_result,
                    "cause_final": leg.cause_final,
                    "p_can": leg.p_can,
                    "cause_rule": leg.cause_rule,
                    "p_rule": leg.p_rule,
                    "expected_cause": leg.expected_cause,
                    "timing_total_s": leg.timing_total_s,
                    "timing_avg_s": leg.timing_avg_s,
                    "timing_last_s": leg.timing_last_s,
                    "timing_calls": leg.timing_calls,
                }
                for leg_id, leg in self.legs.items()
            },
        }
