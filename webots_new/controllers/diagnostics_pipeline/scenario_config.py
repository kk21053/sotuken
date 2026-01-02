"""scenario_v2.ini 形式の読み込み（互換用）

webots_new では controller 側は scenario.ini を使うので、
将来の拡張や互換のために簡易版だけ置いておきます。
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Dict


def load_leg_causes(path: Path) -> Dict[str, str]:
    """[LEGS] セクション（FL/FR/RL/RR）を読み込んで返す"""
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if "LEGS" not in cfg:
        return {"FL": "NONE", "FR": "NONE", "RL": "NONE", "RR": "NONE"}

    legs = {}
    for leg_id in ["FL", "FR", "RL", "RR"]:
        legs[leg_id] = cfg["LEGS"].get(leg_id, "NONE").strip().upper()
    return legs
