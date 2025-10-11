"""Configuration constants for the Spot diagnostics pipeline."""

from typing import Tuple

LEG_IDS: Tuple[str, str, str, str] = ("FL", "FR", "RL", "RR")
CAUSE_LABELS: Tuple[str, str, str, str] = ("NONE", "BURIED", "TRAPPED", "ENTANGLED")

TRIAL_COUNT: int = 4
TRIAL_PATTERN: Tuple[str, str, str, str] = ("+", "+", "-", "-")
TRIAL_DURATION_S: float = 0.5  # Standard duration for stable operation
TRIAL_ANGLE_DEG: float = 7.0

SELF_WEIGHTS = {
    "track": 0.4,
    "vel": 0.2,
    "tau": 0.3,
    "safe": 0.1,
}

SELF_CAN_THRESHOLD: float = 0.70
E_MAX_DEG: float = 5.0
OMEGA_REF_DEG_PER_SEC: float = 50.0
TAU_LIMIT_RATIO: float = 0.8
SAFE_SCORE_NORMAL: float = 1.0
SAFE_SCORE_WARN: float = 0.5
SAFE_SCORE_ERROR: float = 0.0

DELTA_THETA_REF_DEG: float = TRIAL_ANGLE_DEG
FALLEN_THRESHOLD_DEG: float = 20.0

FUSION_WEIGHTS = {
    "drone": 0.4,
    "llm": 0.6,
}

ROBOPOSE_FPS_TRIGGER: float = 18.0
ROBOPOSE_FPS_IDLE: float = 8.0

USE_ONLY_ROBOPOSE: bool = True
URDF_PATH: str = "config/spot_description.urdf"
CAMERA_INTRINSICS_PATH: str = "config/camera_intrinsics.json"

JSONL_LOG_DIR: str = "logs"
JSONL_EVENT_FILENAME: str = "leg_diagnostics_events.jsonl"
JSONL_SESSION_FILENAME: str = "leg_diagnostics_sessions.jsonl"

SOFTMAX_TEMPERATURE: float = 1.0
EPSILON: float = 1e-6

LLM_PRIMARY: str = "Qwen/Qwen2.5-7B-Instruct"
LLM_FALLBACKS: Tuple[str, str] = (
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
)
DEFAULT_LLM_MODELS: Tuple[str, ...] = (LLM_PRIMARY,) + LLM_FALLBACKS
