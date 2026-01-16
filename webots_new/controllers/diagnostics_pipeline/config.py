"""診断パイプラインの設定（仕様に合わせた定数）"""

import os
from typing import Tuple

# 各脚のID
LEG_IDS: Tuple[str, ...] = ("FL", "FR", "RL", "RR")

# 拘束原因ラベル
CAUSE_LABELS: Tuple[str, ...] = ("NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION")

# 6回の試行（関節×方向）
TRIAL_COUNT: int = 6
TRIAL_PATTERN: Tuple[str, ...] = ("+", "-", "+", "-", "+", "-")
TRIAL_MOTOR_INDICES: Tuple[int, ...] = (2, 2, 1, 1, 0, 0)
TRIAL_DURATION_S: float = 0.4
TRIAL_ANGLE_DEG: float = 4.0

# 自己診断の重み
SELF_WEIGHTS = {"track": 0.4, "vel": 0.25, "tau": 0.25, "safe": 0.1}
SELF_CAN_THRESHOLD: float = 0.50

# 確信度のシグモイド
CONFIDENCE_STEEPNESS: float = 15.0

# ルールの閾値
CONFIDENCE_HIGH_THRESHOLD: float = 0.7
CONFIDENCE_LOW_THRESHOLD: float = 0.3

# 自己診断の評価パラメータ
E_MAX_DEG: float = 3.0
OMEGA_REF_DEG_PER_SEC: float = 27.0
TAU_LIMIT_RATIO: float = 0.8
SAFE_SCORE_NORMAL: float = 1.0
SAFE_SCORE_WARN: float = 0.5
SAFE_SCORE_ERROR: float = 0.0
DELTA_THETA_REF_DEG: float = TRIAL_ANGLE_DEG

# RoboPose のみ使用（仕様）
USE_ONLY_ROBOPOSE: bool = True

# ログ
JSONL_LOG_DIR: str = "logs"
JSONL_EVENT_FILENAME: str = "leg_diagnostics_events.jsonl"
JSONL_SESSION_FILENAME: str = "leg_diagnostics_sessions.jsonl"

# eventログ（詳細ログ）はデフォルト無効。
# 必要な場合のみ環境変数で有効化する（ファイル増殖を防ぐ）。
ENABLE_EVENT_LOG: bool = os.getenv("DIAG_ENABLE_EVENT_LOG", "0") in {"1", "true", "TRUE", "yes", "YES"}

EPSILON: float = 1e-6
