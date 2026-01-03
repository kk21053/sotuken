"""診断パイプラインの司令塔

使い方（ドローン側から）:
- start_trial() で試行開始を通知
- record_robo_pose_frame() で観測フレームを追加
- complete_trial() で試行を確定
- finalize() でセッション結果を保存
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .drone_observer import DroneObservationAggregator
from .llm_client import LLMAnalyzer
from .vlm_client import VLMAnalyzer
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

    def add_frame(
        self,
        joint_angles: Optional[Sequence[float]],
        end_position: Optional[Sequence[float]],
        base_orientation: Optional[Sequence[float]],
        base_position: Optional[Sequence[float]],
    ) -> None:
        if joint_angles is not None:
            self.joint_angles.append(list(joint_angles))
        if end_position is not None and len(end_position) >= 3:
            self.end_positions.append((float(end_position[0]), float(end_position[1]), float(end_position[2])))
        if base_orientation is not None and len(base_orientation) >= 3:
            self.base_orientations.append((float(base_orientation[0]), float(base_orientation[1]), float(base_orientation[2])))
        if base_position is not None and len(base_position) >= 3:
            self.base_positions.append((float(base_position[0]), float(base_position[1]), float(base_position[2])))


class DiagnosticsPipeline:
    def __init__(self, session_id: str) -> None:
        self.self_diag = SelfDiagnosisAggregator()
        self.drone = DroneObservationAggregator()
        self.llm = LLMAnalyzer()
        self.vlm = VLMAnalyzer()
        self.logger = DiagnosticsLogger()
        self.start_session(session_id)

    def start_session(self, session_id: str) -> None:
        self.self_diag.reset_all()
        self.session = SessionState(session_id=session_id)
        self._active: Dict[Tuple[str, int], TrialBuffer] = {}
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}

    def set_expected_causes(self, expected: Dict[str, str]) -> None:
        for leg_id, cause in expected.items():
            leg = self.session.ensure_leg(leg_id)
            leg.expected_cause = cause

    def start_trial(self, leg_id: str, trial_index: int, direction: str, start_time: float, duration: Optional[float] = None) -> None:
        dur = duration if duration and duration > 0 else config.TRIAL_DURATION_S
        buf = TrialBuffer(
            leg_id=leg_id,
            trial_index=trial_index,
            direction=direction,
            start_time=start_time,
            end_time=start_time + dur,
        )
        self._active[(leg_id, trial_index)] = buf

    def record_robo_pose_frame(
        self,
        leg_id: str,
        trial_index: int,
        joint_angles: Optional[Sequence[float]],
        end_position: Optional[Sequence[float]],
        base_orientation: Optional[Sequence[float]],
        base_position: Optional[Sequence[float]],
    ) -> None:
        buf = self._active.get((leg_id, trial_index))
        if not buf:
            return
        buf.add_frame(joint_angles, end_position, base_orientation, base_position)

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
    ) -> Optional[TrialResult]:
        buf = self._active.pop((leg_id, trial_index), None)
        if not buf:
            return None
        if end_time is not None:
            buf.end_time = end_time

        leg = self.session.ensure_leg(leg_id)

        trial = self.self_diag.record_trial(
            leg,
            buf.trial_index,
            buf.direction,
            buf.start_time,
            buf.end_time,
            theta_cmd,
            theta_meas,
            omega_meas,
            tau_meas,
            tau_nominal,
            safety_level,
        )

        # Spotが計算した self_can_raw を優先して上書き
        if spot_can_raw is not None:
            trial.self_can_raw = spot_can_raw
            scores = self.self_diag._raw_scores.get(leg_id, [])
            if scores:
                scores[-1] = spot_can_raw

        self.self_diag.finalize_leg(leg)

        self.drone.process_trial(
            leg,
            trial,
            buf.joint_angles,
            buf.end_positions,
            buf.base_orientations,
            buf.base_positions,
        )

        self.session.fallen = self.session.fallen or self.drone.fallen
        self.session.fallen_probability = max(self.session.fallen_probability, self.drone.fallen_probability)

        # 仕様ステップ7: ルールベース判定
        self.llm.infer(leg)

        self._trial_counts[leg_id] = self._trial_counts.get(leg_id, 0) + 1
        self.logger.log_trial(self.session.session_id, leg, trial, self.session.fallen)
        return trial

    def finalize(self) -> SessionState:
        self.session.fallen = self.drone.fallen
        self.session.fallen_probability = self.drone.fallen_probability

        # まずは確実にセッションログを書き出す（Webots終了時の強制終了に備える）
        self.logger.log_session(self.session)

        # VLM は重い場合があるので、デフォルトでは Webots 外（後処理）で回す。
        # どうしても Webots 内で実行したい場合のみ VLM_RUN_IN_WEBOTS=1。
        import os

        if os.getenv("VLM_ENABLE", "0").strip() == "1" and os.getenv("VLM_RUN_IN_WEBOTS", "0").strip() == "1":
            try:
                self.vlm.infer_session(self.session)
                # 推論結果を追記（同じsession_idの新しい行が末尾に来る）
                self.logger.log_session(self.session)
            except Exception:
                pass
        return self.session
