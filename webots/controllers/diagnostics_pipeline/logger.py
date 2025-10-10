"""JSONL logging for diagnostics results."""

from __future__ import annotations

import time
from typing import Dict

from . import config
from .models import LegState, SessionRecord, SessionState, TrialResult
from .utils import JsonlWriter


class DiagnosticsLogger:
    def __init__(self, directory: str = config.JSONL_LOG_DIR) -> None:
        self.event_writer = JsonlWriter(directory, config.JSONL_EVENT_FILENAME)
        self.session_writer = JsonlWriter(directory, config.JSONL_SESSION_FILENAME)

    def log_trial(
        self,
        session_id: str,
        leg: LegState,
        trial: TrialResult,
        fallen: bool,
    ) -> None:
        payload: Dict = {
            "timestamp": time.time(),
            "session_id": session_id,
            "leg_id": leg.leg_id,
            "trial_index": trial.trial_index,
            "dir": trial.direction,
            "start_time": trial.start_time,
            "end_time": trial.end_time,
            "duration": trial.duration,
            "self_can_raw_i": trial.self_can_raw,
            "drone_can_raw_i": trial.drone_can_raw,
            "self_can": leg.self_can,
            "self_moves": leg.self_moves,
            "drone_can": leg.drone_can,
            "p_drone": leg.p_drone,
            "p_llm": leg.p_llm,
            "p_final": leg.p_final,
            "cause_final": leg.cause_final,
            "conf_final": leg.conf_final,
            "fallen": fallen,
            "features_drone": trial.features_drone,
            "trial_ok": trial.ok,
        }
        self.event_writer.append(payload)

    def log_session(self, session: SessionState) -> None:
        record = SessionRecord(
            session_id=session.session_id,
            fallen=session.fallen,
            legs={leg_id: leg.snapshot() for leg_id, leg in session.legs.items()},
        )
        self.session_writer.append(record.to_dict())
