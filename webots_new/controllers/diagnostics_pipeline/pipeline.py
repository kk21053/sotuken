"""High level orchestration for the diagnostics run (webots_new).

Spot（自己診断）と Drone（外部観測）を統合して、脚ごとの
movement_result / cause_final を確定し、JSONLに記録する。

※このファイルは Drone コントローラから import されるため、空だと起動直後に落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .drone_observer import DroneObservationAggregator
from .fusion import select_cause
from .llm_client import LLMAnalyzer
from .logger import DiagnosticsLogger
from .models import LegState, SessionState, TrialResult
from .self_diagnosis import SelfDiagnosisAggregator
from .utils import clamp

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
		self.start_session(session_id)

	def start_session(self, session_id: str) -> None:
		self.self_diag.reset_all()
		self.session = SessionState(session_id=session_id)
		self._active_trials: Dict[Tuple[str, int], TrialBuffer] = {}

	def set_expected_causes(self, expected_causes: Dict[str, str]) -> None:
		for leg_id, cause in expected_causes.items():
			leg = self.session.ensure_leg(leg_id)
			leg.expected_cause = (cause or "NONE").strip().upper()

	def start_trial(
		self,
		leg_id: str,
		trial_index: int,
		direction: str,
		start_time: float,
		duration: Optional[float] = None,
	) -> None:
		duration = duration if duration and duration > 0 else config.TRIAL_DURATION_S
		end_time = float(start_time) + float(duration)
		self.session.ensure_leg(leg_id)
		self._active_trials[(leg_id, int(trial_index))] = TrialBuffer(
			leg_id=leg_id,
			trial_index=int(trial_index),
			direction=str(direction),
			start_time=float(start_time),
			end_time=end_time,
		)

	def record_robo_pose_frame(
		self,
		leg_id: str,
		trial_index: int,
		joint_angles: Optional[Sequence[float]],
		end_position: Optional[Sequence[float]],
		base_orientation: Optional[Sequence[float]],
		base_position: Optional[Sequence[float]],
	) -> None:
		buf = self._active_trials.get((leg_id, int(trial_index)))
		if buf is None:
			return
		buf.add_drone_frame(joint_angles, end_position, base_orientation, base_position)

	def _record_spot_raw_score(self, leg: LegState, trial: TrialResult, spot_can_raw: float) -> None:
		raw = clamp(float(spot_can_raw))
		trial.self_can_raw = raw
		scores = self.self_diag._raw_scores.setdefault(leg.leg_id, [])  # noqa: SLF001
		if len(scores) >= config.TRIAL_COUNT:
			scores.pop(0)
		scores.append(raw)

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
			print(f"[pipeline_new] warning: complete_trial without buffer: {leg_id}#{trial_index}")
			return None
		buf.update_end_time(end_time)

		leg = self.session.ensure_leg(leg_id)

		# TrialResult は必ず作る（Spotのrawが来ない場合でも最低限記録したい）
		trial = TrialResult(
			leg_id=leg.leg_id,
			trial_index=int(trial_index),
			direction=str(buf.direction),
			start_time=float(buf.start_time),
			end_time=float(buf.end_time),
		)
		leg.trials.append(trial)

		# Spot自己診断（raw）
		if spot_can_raw is not None:
			try:
				self._record_spot_raw_score(leg, trial, float(spot_can_raw))
			except Exception:
				pass
		else:
			# フォールバック（基本はSpotがself_can_rawを送る設計）
			try:
				fallback = self.self_diag.record_trial(
					leg,
					int(trial_index),
					str(buf.direction),
					float(buf.start_time),
					float(buf.end_time),
					theta_cmd,
					theta_meas,
					omega_meas,
					tau_meas,
					float(tau_nominal),
					str(safety_level),
				)
				trial.self_can_raw = fallback.self_can_raw
			except Exception:
				pass

		# spot_can を確定
		try:
			self.self_diag.finalize_leg(leg)
		except Exception:
			pass

		# Drone観測を反映
		try:
			self.drone.process_trial(
				leg,
				trial,
				buf.joint_angles,
				buf.end_positions,
				buf.base_orientations,
				buf.base_positions,
			)
		except Exception as exc:
			print(f"[pipeline_new] warning: drone.process_trial failed: {exc}")

		# Spotのtau情報（MALFUNCTION切り分け用）
		try:
			if spot_malfunction_flag is not None:
				trial.features["spot_malfunction_flag"] = int(spot_malfunction_flag)
			if spot_tau_avg is not None:
				trial.features["spot_tau_avg"] = float(spot_tau_avg)
			if spot_tau_max is not None:
				trial.features["spot_tau_max"] = float(spot_tau_max)
			if tau_nominal is not None and float(tau_nominal) > 1e-6:
				if spot_tau_avg is not None:
					trial.features["spot_tau_avg_ratio"] = abs(float(spot_tau_avg)) / float(tau_nominal)
				if spot_tau_max is not None:
					trial.features["spot_tau_max_ratio"] = abs(float(spot_tau_max)) / float(tau_nominal)
		except Exception:
			pass

		# 最終判定（仕様ルール）
		try:
			# 1) ルールベースで分布を作る（この段階で特徴量・スコアが確定している）
			dist_rule = self.llm.infer_rule_based_only(leg)
			leg.p_llm = dict(dist_rule)
			leg.cause_final = select_cause(dist_rule)
			self.llm.apply_spec_rule_logging(leg, dist_rule)

			# 2) この段階でJSONL出力（特徴量・スコア確定ログ）
			try:
				self.logger.log_trial(
					self.session.session_id,
					leg,
					trial,
					stage="features_ready",
				)
			except Exception as exc:
				print(f"[pipeline_new] warning: pre log_trial failed: {exc}")

			# 3) Qwenに渡すJSONを作成
			qwen_payload = self.llm.build_qwen_payload(leg, fallback=dist_rule)

			# 4) Qwenで推論（条件を満たすときのみ。満たさない場合はルール分布を使う）
			dist_final = dict(dist_rule)
			if self.llm.should_use_qwen(leg, dist_rule):
				try:
					print(
						f"[qwen] start session={self.session.session_id} leg={leg.leg_id} trial={trial.trial_index}",
						flush=True,
					)
				except Exception:
					pass
				started = None
				try:
					import time

					started = time.time()
				except Exception:
					started = None
				updated = self.llm.infer_with_qwen_payload(qwen_payload)
				if updated is not None:
					leg.p_llm = dict(updated)
					dist_final = dict(updated)
				try:
					import time

					elapsed = (time.time() - started) if started is not None else None
					if elapsed is None:
						print(f"[qwen] done session={self.session.session_id} leg={leg.leg_id}", flush=True)
					else:
						print(
							f"[qwen] done session={self.session.session_id} leg={leg.leg_id} ({elapsed:.1f}s)",
							flush=True,
						)
				except Exception:
					pass

			# 5) 最終診断結果の確定
			leg.cause_final = select_cause(dist_final)
			self.llm.apply_spec_rule_logging(leg, dist_final)

			# 6) 最終確定後のJSONL出力
			try:
				self.logger.log_trial(
					self.session.session_id,
					leg,
					trial,
					stage="finalized",
				)
			except Exception as exc:
				print(f"[pipeline_new] warning: post log_trial failed: {exc}")
		except Exception as exc:
			print(f"[pipeline_new] warning: llm/qwen pipeline failed: {exc}")
			# 何か壊れても落とさない（droneの分布で最低限の確定）
			leg.cause_final = select_cause(leg.p_drone)
			try:
				self.logger.log_trial(
					self.session.session_id,
					leg,
					trial,
					stage="finalized",
				)
			except Exception:
				pass

		return trial

	def finalize(self) -> SessionState:
		self.logger.log_session(self.session)
		return self.session

	def reset(self) -> None:
		self.start_session(self.session.session_id)
