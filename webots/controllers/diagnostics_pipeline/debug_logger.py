#!/usr/bin/env python3
"""デバッグ用: featuresをファイルに記録する"""

import json
from pathlib import Path
from typing import Dict

DEBUG_LOG = Path("/tmp/fl_debug.jsonl")

def log_features(leg_id: str, trial_index: int, features: Dict, scores: Dict):
    """featuresとscoresをログに記録"""
    with DEBUG_LOG.open('a') as f:
        data = {
            "leg_id": leg_id,
            "trial_index": trial_index,
            "features": features,
            "scores": scores
        }
        f.write(json.dumps(data) + "\n")
