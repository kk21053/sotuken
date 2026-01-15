
"""High level orchestration for the diagnostics run (webots_new).

診断プロセスは以下の3つ“のみ”で構成する:
1) Spot自己診断（Spot側で計算した self_can_raw を集計 → spot_can）
2) Drone観測（RoboPose相当: 関節角/足先変位系列 → drone_can / p_drone）
3) LLM(Qwen)（未設定時はフォールバック）+ 仕様.txt Step7 ルールで最終確定

このファイルは Drone コントローラから import される。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .drone_observer import DroneObservationAggregator
from .llm_client import LLMAnalyzer
from .logger import DiagnosticsLogger
from .models import SessionState, TrialResult
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
        if values is None or len(values) < 3:
            raise ValueError("Vector3 requires three float values")
        return (float(values[0]), float(values[1]), float(values[2]))

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
            except Exception:
                pass
        if base_orientation is not None:
            try:
                self.base_orientations.append(self._to_vec3(base_orientation))
            except Exception:
                pass
        if base_position is not None:
            try:
                self.base_positions.append(self._to_vec3(base_position))
            except Exception:
                pass

    def update_end_time(self, end_time: Optional[float]) -> None:
        if end_time is None:
            return
        self.end_time = float(end_time)


class DiagnosticsPipeline:
    """診断の司令塔"""

    def __init__(self, session_id: str) -> None:
        self.self_diag = SelfDiagnosisAggregator()
        self.drone = DroneObservationAggregator()
        self.llm = LLMAnalyzer()
        self.logger = DiagnosticsLogger()

        self.session = SessionState(session_id=session_id)
        self._active_trials: Dict[Tuple[str, int], TrialBuffer] = {}

    def set_expected_causes(self, expected: Dict[str, str]) -> None:
        for leg_id in config.LEG_IDS:
            cause = (expected.get(leg_id) or "NONE").strip().upper()
            leg = self.session.ensure_leg(leg_id)
            leg.expected_cause = cause

    def start_trial(
        self,
        leg_id: str,
        trial_index: int,
        direction: str,
        start_time: float,
        duration: Optional[float] = None,
    ) -> None:
        duration_s = float(duration) if duration and duration > 0 else config.TRIAL_DURATION_S
        end_time = float(start_time) + duration_s

        self.session.ensure_leg(leg_id)
        self._active_trials[(leg_id, int(trial_index))] = TrialBuffer(
            leg_id=str(leg_id),
            trial_index=int(trial_index),
            direction=str(direction),
            start_time=float(start_time),
            end_time=float(end_time),
        )

    def record_robo_pose_frame(
        self,
        leg_id: str,
        trial_index: int,
        joint_angles: Optional[Sequence[float]],
        end_position: Optional[Sequence[float]],
        base_orientation: Optional[Sequence[float]] = None,
        base_position: Optional[Sequence[float]] = None,
    ) -> None:
        buf = self._active_trials.get((leg_id, int(trial_index)))
        if buf is None:
            return
        buf.add_drone_frame(joint_angles, end_position, base_orientation, base_position)

    def complete_trial(
        self,
        leg_id: str,
        trial_index: int,
        theta_cmd: Sequence[float],
        theta_meas: Sequence[float],
        omega_meas: Sequence[float],
        tau_meas: Sequence[float],
        tau_nominal: float,
        safety_level: str,
        end_time: Optional[float] = None,
        spot_can_raw: Optional[float] = None,
        spot_tau_avg: Optional[float] = None,
        spot_tau_max: Optional[float] = None,
        spot_malfunction_flag: Optional[int] = None,
    ) -> Optional[TrialResult]:
        buf = self._active_trials.pop((leg_id, int(trial_index)), None)
        if buf is None:
            return None
        buf.update_end_time(end_time)

        leg = self.session.ensure_leg(leg_id)

        # ---- 1) Spot自己診断（rawを受けて集計するだけ） ----
        raw_for_spot = 0.35 if spot_can_raw is None else float(spot_can_raw)
        trial = self.self_diag.record_raw_trial(
            leg=leg,
            trial_index=int(trial_index),
            direction=str(buf.direction),
            start_time=float(buf.start_time),
            end_time=float(buf.end_time),
            self_can_raw=float(raw_for_spot),
        )

        # 付加情報（最終判断の根拠として trial.features に最小限載せる）
        try:
            if tau_nominal:
                if spot_tau_avg is not None:
                    trial.features["spot_tau_avg_ratio"] = float(spot_tau_avg) / max(float(tau_nominal), config.EPSILON)
                if spot_tau_max is not None:
                    trial.features["spot_tau_max_ratio"] = float(spot_tau_max) / max(float(tau_nominal), config.EPSILON)
            if spot_malfunction_flag is not None:
                trial.features["spot_malfunction_flag"] = float(int(spot_malfunction_flag))
        except Exception:
            pass

        # ---- 2) Drone観測（RoboPose） ----
        try:
            self.drone.process_trial(
                leg,
                trial,
                buf.joint_angles,
                buf.end_positions,
                buf.base_orientations,
                buf.base_positions,
            )
        except Exception:
            pass

        # ---- 集計確定 ----
        try:
            self.self_diag.finalize_leg(leg)
        except Exception:
            pass

        # ---- 3) 最終診断（LLM/Qwen + 仕様ルール） ----
        try:
            self.llm.infer(leg)
        except Exception:
            pass

        # 表示用（view_result.py が使う）
        try:
            leg.p_can = (float(leg.spot_can) + float(leg.drone_can)) / 2.0
        except Exception:
            pass

        # ログは最小（finalized のみ）
        try:
            self.logger.log_trial(self.session.session_id, leg, trial, stage="finalized")
        except Exception:
            pass

        return trial

    def finalize(self) -> SessionState:
        self.logger.log_session(self.session)
        return self.session

    def reset(self) -> None:
        self.self_diag.reset_all()
        self.session = SessionState(session_id=self.session.session_id, image_path=self.session.image_path)
        self._active_trials.clear()
