"""Configuration constants for the Spot diagnostics pipeline."""

from typing import Tuple

# 仕様: 各脚のID（FL、FR、RL、RR）
LEG_IDS: Tuple[str, ...] = ("FL", "FR", "RL", "RR")

# 仕様: 拘束原因のラベル（正常、故障、埋まる、挟まる、絡まる、転倒）
CAUSE_LABELS: Tuple[str, ...] = ("NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION", "FALLEN")

# 仕様ステップ1: 各脚に対して6回ずつ内部診断（各関節を両方向にテスト）
TRIAL_COUNT: int = 6
TRIAL_PATTERN: Tuple[str, ...] = ("+", "-", "+", "-", "+", "-")
TRIAL_MOTOR_INDICES: Tuple[int, ...] = (2, 2, 1, 1, 0, 0)  # knee+, knee-, hip+, hip-, shoulder+, shoulder-
TRIAL_DURATION_S: float = 0.4
TRIAL_ANGLE_DEG: float = 4.0

# 仕様ステップ1: 診断基準「追従性*0.4 + 速度*0.25 + トルク*0.25 + 安全性*0.1」
SELF_WEIGHTS = {
    "track": 0.4,   # 追従性スコア
    "vel": 0.25,    # 速度スコア
    "tau": 0.25,    # トルクスコア
    "safe": 0.1,    # 安全性スコア
}

SELF_CAN_THRESHOLD: float = 0.50

# 仕様ステップ3,4: シグモイド関数により値の差をわかりやすく
CONFIDENCE_STEEPNESS: float = 15.0

# 仕様ステップ7: ルールベースLLMの閾値
CONFIDENCE_HIGH_THRESHOLD: float = 0.7  # 0.7以上 → 動く
CONFIDENCE_LOW_THRESHOLD: float = 0.3   # 0.3以下 → 動かない

# 自己診断の評価パラメータ
E_MAX_DEG: float = 3.0
OMEGA_REF_DEG_PER_SEC: float = 27.0
TAU_LIMIT_RATIO: float = 0.8
SAFE_SCORE_NORMAL: float = 1.0
SAFE_SCORE_WARN: float = 0.5
SAFE_SCORE_ERROR: float = 0.0

DELTA_THETA_REF_DEG: float = TRIAL_ANGLE_DEG

# 仕様ステップ8: 転倒判定
FALLEN_THRESHOLD_DEG: float = 20.0

# 仕様ステップ2: RoboPoseを用いた姿勢監査
USE_ONLY_ROBOPOSE: bool = True

# 仕様ステップ9: 結果をログに記録
JSONL_LOG_DIR: str = "logs"
JSONL_EVENT_FILENAME: str = "leg_diagnostics_events.jsonl"
JSONL_SESSION_FILENAME: str = "leg_diagnostics_sessions.jsonl"

EPSILON: float = 1e-6

# Advanced LLM settings (Jetson Orin Nano Super)
USE_LLM_ADVANCED: bool = False  # LLM診断を有効化するか（デフォルト: 無効）
LLM_CONFIDENCE_THRESHOLD: float = 0.6  # この値以下の信頼度でLLMを起動
LLM_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q4_k_m.gguf"
MANUAL_PDF_PATH: str = "/home/kk21053/sotuken/Spot_IFU-v2.1.2-ja.pdf"
MANUAL_EMBEDDINGS_CACHE: str = "data/manual_embeddings"

