"""High level orchestration for the diagnostics run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .drone_observer import DroneObservationAggregator
from .fusion import fuse_probabilities, select_cause
from .llm_client import LLMAnalyzer
from .logger import DiagnosticsLogger
from .models import LegState, SessionState, TrialResult
from .self_diagnosis import SelfDiagnosisAggregator

Vector3 = Tuple[float, float, float]


@dataclass
class TrialBuffer:
    leg_id: str
    trial_index: int
    direction: str
    start_time: float
    end_time: float
    joint_angles: List[Sequence[float]] = field(default_factory=list)
    end_positions: List[Vector3] = field(default_factory=list)
    base_orientations: List[Tuple[float, float, float]] = field(default_factory=list)
    base_positions: List[Vector3] = field(default_factory=list)

    @staticmethod
    def _to_vec3(values: Sequence[float]) -> Vector3:
        try:
            x, y, z = (float(values[0]), float(values[1]), float(values[2]))
        except (TypeError, ValueError, IndexError):
            raise ValueError("Vector3 requires three float values")
        return (x, y, z)

    def add_drone_frame(
        self,
        joint_angles: Optional[Sequence[float]],
        end_position: Optional[Sequence[float]],
        base_orientation: Optional[Sequence[float]],
        base_position: Optional[Sequence[float]],
    ) -> None:
        if joint_angles is not None:
            self.joint_angles.append(list(joint_angles))
        if end_position is not None:
            try:
                self.end_positions.append(self._to_vec3(end_position))
            except ValueError as exc:
                print(f"[pipeline] warning: skipping invalid end_position: {exc}")
        if base_orientation is not None:
            try:
                self.base_orientations.append(self._to_vec3(base_orientation))
            except ValueError as exc:
                print(f"[pipeline] warning: skipping invalid base_orientation: {exc}")
        if base_position is not None:
            try:
                self.base_positions.append(self._to_vec3(base_position))
            except ValueError as exc:
                print(f"[pipeline] warning: skipping invalid base_position: {exc}")

    def update_end_time(self, end_time: Optional[float]) -> None:
        if end_time is None:
            return
        self.end_time = end_time


class DiagnosticsPipeline:
    def __init__(self, session_id: str) -> None:
        self.self_diag = SelfDiagnosisAggregator()
        self.drone = DroneObservationAggregator()
        self.llm = LLMAnalyzer()
        self.logger = DiagnosticsLogger()
        self.start_session(session_id)

    def start_session(self, session_id: str) -> None:
        self.self_diag.reset_all()
        self.drone.reset()
        self.session = SessionState(session_id=session_id)
        self._states: Dict[str, str] = {leg: "IDLE" for leg in config.LEG_IDS}
        self._active_trials: Dict[str, TrialBuffer] = {}
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}

    def start_trial(
        self,
        leg_id: str,
        trial_index: int,
        direction: str,
        start_time: float,
        duration: Optional[float] = None,
    ) -> None:
        duration = duration if duration and duration > 0 else config.TRIAL_DURATION_S
        end_time = start_time + duration
        if 1 <= trial_index <= config.TRIAL_COUNT:
            expected_dir = config.TRIAL_PATTERN[trial_index - 1]
        else:
            expected_dir = direction
        if direction != expected_dir:
            print(
                f"[pipeline] warning: leg {leg_id} trial {trial_index} expected direction '{expected_dir}'"
                f" but got '{direction}'",
            )
        count = self._trial_counts.get(leg_id, 0)
        if count >= config.TRIAL_COUNT:
            print(
                f"[pipeline] warning: leg {leg_id} exceeded configured trial count"
                f" ({config.TRIAL_COUNT}). Oldest result will be replaced.",
            )
        buffer = TrialBuffer(
            leg_id=leg_id,
            trial_index=trial_index,
            direction=direction,
            start_time=start_time,
            end_time=end_time,
        )
        self._active_trials[leg_id] = buffer
        self._states[leg_id] = "ARMED"

    def record_robo_pose_frame(
        self,
        leg_id: str,
        joint_angles: Optional[Sequence[float]],
        end_position: Optional[Sequence[float]],
        base_orientation: Optional[Sequence[float]],
        base_position: Optional[Sequence[float]],
    ) -> None:
        buffer = self._active_trials.get(leg_id)
        if buffer is None:
            return
        if self._states.get(leg_id) == "ARMED":
            self._states[leg_id] = "MEASURING"
        buffer.add_drone_frame(joint_angles, end_position, base_orientation, base_position)

    def complete_trial(
        self,
        leg_id: str,
        theta_cmd: Sequence[float],
        theta_meas: Sequence[float],
        omega_meas: Sequence[float],
        tau_meas: Sequence[float],
        tau_nominal: float,
        safety_level: str,
        end_time: Optional[float] = None,
    ) -> Optional[TrialResult]:
        buffer = self._active_trials.pop(leg_id, None)
        if buffer is None:
            print(f"[pipeline] warning: complete_trial called without active buffer for {leg_id}")
            return None
        buffer.update_end_time(end_time)
        self._states[leg_id] = "DONE"

        leg = self.session.ensure_leg(leg_id)
        trial = self.self_diag.record_trial(
            leg,
            buffer.trial_index,
            buffer.direction,
            buffer.start_time,
            buffer.end_time,
            theta_cmd,
            theta_meas,
            omega_meas,
            tau_meas,
            tau_nominal,
            safety_level,
        )
        self.self_diag.finalize_leg(leg)

        self.drone.process_trial(
            leg,
            trial,
            buffer.joint_angles,
            buffer.end_positions,
            buffer.base_orientations,
            buffer.base_positions,
        )
        self.session.fallen = self.session.fallen or self.drone.fallen

        p_llm = self.llm.infer(leg)
        leg.p_final = fuse_probabilities(leg.p_drone, p_llm)
        leg.cause_final = select_cause(leg.p_final)
        leg.conf_final = leg.p_final.get(leg.cause_final, 0.0)

        if not leg.self_moves and leg.drone_can >= 0.85:
            leg.moves_final = True
        elif leg.self_moves and leg.drone_can <= 0.30:
            leg.moves_final = False
        else:
            leg.moves_final = leg.self_moves

        self._trial_counts[leg_id] = self._trial_counts.get(leg_id, 0) + 1
        self.logger.log_trial(self.session.session_id, leg, trial, self.session.fallen)
        self._states[leg_id] = "IDLE"
        return trial

    def finalize(self) -> SessionState:
        self.session.fallen = self.session.fallen or self.drone.fallen
        self.logger.log_session(self.session)
        return self.session

    def reset(self) -> None:
        self.self_diag.reset_all()
        self.drone.reset()
        self.session.fallen = False
        self.session.legs.clear()
        self._states = {leg: "IDLE" for leg in config.LEG_IDS}
        self._active_trials.clear()
        self._trial_counts = {leg: 0 for leg in config.LEG_IDS}
