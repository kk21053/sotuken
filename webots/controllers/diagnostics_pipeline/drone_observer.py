"""Drone-side aggregation that relies on RoboPose outputs."""

from __future__ import annotations

import math
from statistics import mean
from typing import Dict, List, Sequence, Tuple

from . import config
from .models import LegState, TrialResult
from .utils import clamp, normalize_distribution, softmax

Vector3 = Tuple[float, float, float]


def _distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class DroneObservationAggregator:
    """Aggregate RoboPose measurements for each trial."""

    def __init__(self) -> None:
        if not config.USE_ONLY_ROBOPOSE:
            raise RuntimeError("RoboPose only mode is required by specification")
        self._raw_scores: Dict[str, List[float]] = {leg: [] for leg in config.LEG_IDS}
        self._fallen: bool = False
        self._cause_accumulator: Dict[str, Dict[str, float]] = {
            leg: {label: 0.0 for label in config.CAUSE_LABELS}
            for leg in config.LEG_IDS
        }
        self._trial_counts: Dict[str, int] = {leg: 0 for leg in config.LEG_IDS}

    @staticmethod
    def _reduce_joint_series(joint_angles: Sequence[Sequence[float]]) -> List[float]:
        reduced: List[float] = []
        for frame in joint_angles:
            if not frame:
                reduced.append(0.0)
            else:
                reduced.append(sum(frame) / len(frame))
        return reduced

    def process_trial(
        self,
        leg: LegState,
        trial: TrialResult,
        joint_angles: Sequence[Sequence[float]],
        end_positions: Sequence[Vector3],
        base_orientations: Sequence[Tuple[float, float, float]],
        base_positions: Sequence[Vector3],
    ) -> None:
        reduced_angles = self._reduce_joint_series(joint_angles)
        if not reduced_angles:
            delta_theta = 0.0
        else:
            delta_theta = max(reduced_angles) - min(reduced_angles)
        raw = clamp(delta_theta / config.DELTA_THETA_REF_DEG)
        scores = self._raw_scores.setdefault(leg.leg_id, [])
        if len(scores) >= config.TRIAL_COUNT:
            scores.pop(0)
        scores.append(raw)
        trial.drone_can_raw = raw

        fallen = False
        max_roll = 0.0
        max_pitch = 0.0
        for roll, pitch, _ in base_orientations:
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            if abs(roll) > config.FALLEN_THRESHOLD_DEG or abs(pitch) > config.FALLEN_THRESHOLD_DEG:
                fallen = True
        if fallen:
            self._fallen = True

        end_disp = 0.0
        path_length = 0.0
        reversals = 0
        if end_positions:
            end_disp = _distance(end_positions[0], end_positions[-1])
            for idx in range(1, len(end_positions)):
                path_length += _distance(end_positions[idx - 1], end_positions[idx])
            velocities = []
            for idx in range(1, len(end_positions)):
                velocities.append(end_positions[idx][0] - end_positions[idx - 1][0])
            for idx in range(1, len(velocities)):
                if velocities[idx - 1] == 0.0:
                    continue
                if velocities[idx - 1] * velocities[idx] < 0:
                    reversals += 1
        base_height = mean([pos[2] for pos in base_positions]) if base_positions else 0.0

        features = {
            "delta_theta_deg": delta_theta,
            "delta_theta_norm": raw,
            "end_disp": end_disp,
            "path_length": path_length,
            "path_straightness": path_length / (end_disp + config.EPSILON) if end_disp > 0 else 0.0,
            "reversals": float(reversals),
            "base_height": base_height,
            "max_roll": max_roll,
            "max_pitch": max_pitch,
            "fallen": bool(fallen),
        }
        trial.features_drone = features
        trial.ok = bool(end_positions)

        effective_scores = scores[: config.TRIAL_COUNT]
        leg.drone_can = mean(effective_scores) if effective_scores else 0.0
        
        # Store trial if not already in leg.trials
        if trial not in leg.trials:
            leg.trials.append(trial)
        
        # Update trial features
        trial.features_drone = features
        trial.drone_can_raw = raw

        current_distribution = self._estimate_cause_distribution(features)
        accumulator = self._cause_accumulator.setdefault(
            leg.leg_id, {label: 0.0 for label in config.CAUSE_LABELS}
        )
        for label, value in current_distribution.items():
            accumulator[label] = accumulator.get(label, 0.0) + value
        self._trial_counts[leg.leg_id] = self._trial_counts.get(leg.leg_id, 0) + 1

        count = max(1, self._trial_counts[leg.leg_id])
        averaged = {
            label: accumulator.get(label, 0.0) / count for label in config.CAUSE_LABELS
        }
        leg.p_drone = normalize_distribution(averaged)

    def _estimate_cause_distribution(self, features: Dict[str, float]) -> Dict[str, float]:
        joint_score = clamp(features.get("delta_theta_norm", 0.0))
        end_disp = features.get("end_disp", 0.0)
        path_length = features.get("path_length", 0.0)
        straightness = features.get("path_straightness", 0.0)
        base_height = features.get("base_height", 0.0)
        reversals = features.get("reversals", 0.0)

        disp_norm = clamp(end_disp / 0.12)
        path_factor = clamp(path_length / (end_disp + config.EPSILON)) if end_disp > 0 else straightness

        # TRAPPED: joint moves but foot doesn't (high joint_score, low disp_norm)
        score_trapped = clamp((joint_score - disp_norm) * 1.2)
        
        # ENTANGLED: irregular path with reversals
        score_entangled = clamp((path_factor - 1.2) + reversals * 0.1)
        
        # BURIED: very low end displacement (foot stuck in sand/debris)
        # Use both low base_height AND low end_disp as indicators
        base_factor = clamp(max(0.0, (0.25 - base_height) * 3.0))
        disp_factor = clamp(max(0.0, (0.01 - end_disp) * 100.0))  # High score when disp < 1cm
        score_buried = max(base_factor, disp_factor)
        
        # NONE: normal movement (high disp, low other scores)
        score_none = clamp(1.0 - max(score_trapped, score_entangled, score_buried) * 0.6)

        scores = {
            "NONE": score_none,
            "BURIED": score_buried,
            "TRAPPED": score_trapped,
            "ENTANGLED": score_entangled,
        }
        return softmax(scores)

    @property
    def fallen(self) -> bool:
        return self._fallen

    def reset(self) -> None:
        for leg_id in self._raw_scores:
            self._raw_scores[leg_id].clear()
            self._cause_accumulator[leg_id] = {label: 0.0 for label in config.CAUSE_LABELS}
            self._trial_counts[leg_id] = 0
        self._fallen = False
