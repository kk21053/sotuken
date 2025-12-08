"""
シナリオ設定読み込みモジュール (Version 2)

各脚ごとに独立した障害を設定できる新しいフォーマットをサポート
"""

import configparser
from pathlib import Path
from typing import Dict, Tuple, Optional

# 障害タイプの定義
CAUSE_NONE = "NONE"
CAUSE_BURIED = "BURIED"
CAUSE_TRAPPED = "TRAPPED"
CAUSE_TANGLED = "TANGLED"
CAUSE_MALFUNCTION = "MALFUNCTION"

VALID_CAUSES = {CAUSE_NONE, CAUSE_BURIED, CAUSE_TRAPPED, CAUSE_TANGLED, CAUSE_MALFUNCTION}

# 脚のID定義
LEG_FL = "FL"
LEG_FR = "FR"
LEG_BL = "BL"
LEG_BR = "BR"

ALL_LEGS = [LEG_FL, LEG_FR, LEG_BL, LEG_BR]

# 脚名の変換マップ
LEG_ID_TO_NAME = {
    LEG_FL: "front_left",
    LEG_FR: "front_right",
    LEG_BL: "back_left",
    LEG_BR: "back_right",
}

LEG_NAME_TO_ID = {v: k for k, v in LEG_ID_TO_NAME.items()}


class ScenarioConfig:
    """シナリオ設定を管理するクラス"""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        
        # 各脚の障害タイプ
        self.leg_causes: Dict[str, str] = {}
        
        # 脚の位置オフセット
        self.foot_offsets: Dict[str, Tuple[float, float]] = {}
        
        # 障害モジュールのパラメータ
        self.buried_params: Dict[str, float] = {}
        self.trapped_params: Dict[str, float] = {}
        self.tangled_params: Dict[str, float] = {}
        
        self._load_config()
    
    def _load_config(self):
        """設定ファイルを読み込む"""
        if not self.config_path.exists():
            print(f"[scenario] Warning: Config file not found: {self.config_path}")
            self._set_defaults()
            return
        
        try:
            self.config.read(self.config_path)
            self._parse_leg_causes()
            self._parse_foot_offsets()
            self._parse_module_params()
        except Exception as e:
            print(f"[scenario] Error loading config: {e}")
            self._set_defaults()
    
    def _set_defaults(self):
        """デフォルト設定を適用"""
        for leg_id in ALL_LEGS:
            self.leg_causes[leg_id] = CAUSE_NONE
        
        self.foot_offsets = {
            LEG_FL: (0.70, 0.1),
            LEG_FR: (0.36, -0.18),
            LEG_RL: (-0.30, 0.18),
            LEG_RR: (-0.30, -0.18),
        }
    
    def _parse_leg_causes(self):
        """各脚の障害タイプを読み込む"""
        if "LEGS" not in self.config:
            print("[scenario] Warning: [LEGS] section not found")
            for leg_id in ALL_LEGS:
                self.leg_causes[leg_id] = CAUSE_NONE
            return
        
        legs_section = self.config["LEGS"]
        for leg_id in ALL_LEGS:
            cause = legs_section.get(leg_id, CAUSE_NONE).strip().upper()
            if cause not in VALID_CAUSES:
                print(f"[scenario] Warning: Invalid cause '{cause}' for {leg_id}, using NONE")
                cause = CAUSE_NONE
            self.leg_causes[leg_id] = cause
    
    def _parse_foot_offsets(self):
        """脚の位置オフセットを読み込む"""
        if "FOOT_OFFSETS" not in self.config:
            print("[scenario] Warning: [FOOT_OFFSETS] section not found, using defaults")
            self._set_defaults()
            return
        
        offsets_section = self.config["FOOT_OFFSETS"]
        for leg_id in ALL_LEGS:
            offset_str = offsets_section.get(leg_id, "")
            if not offset_str:
                continue
            
            try:
                parts = [p.strip() for p in offset_str.split(',')]
                if len(parts) == 2:
                    x, y = float(parts[0]), float(parts[1])
                    self.foot_offsets[leg_id] = (x, y)
            except ValueError as e:
                print(f"[scenario] Warning: Invalid offset for {leg_id}: {e}")
    
    def _parse_module_params(self):
        """障害モジュールのパラメータを読み込む"""
        # BURIED パラメータ
        if "BURIED_PARAMS" in self.config:
            buried = self.config["BURIED_PARAMS"]
            self.buried_params = {
                "radius": float(buried.get("radius", 0.4)),
                "height": float(buried.get("height", 0.25)),
                "topLevel": float(buried.get("topLevel", 0.125)),
                "friction": float(buried.get("friction", 50.0)),
                "bounce": float(buried.get("bounce", 0.0)),
                "color": buried.get("color", "0.9, 0.7, 0.3"),
            }
        
        # TRAPPED パラメータ
        if "TRAPPED_PARAMS" in self.config:
            trapped = self.config["TRAPPED_PARAMS"]
            self.trapped_params = {
                "friction": float(trapped.get("friction", 3.0)),
                "bounce": float(trapped.get("bounce", 0.0)),
                "offsetX": float(trapped.get("offsetX", 0.0)),
                "offsetY": float(trapped.get("offsetY", 0.0)),
                "offsetZ": float(trapped.get("offsetZ", 0.0)),
            }
        
        # TANGLED パラメータ
        if "TANGLED_PARAMS" in self.config:
            tangled = self.config["TANGLED_PARAMS"]
            self.tangled_params = {
                "friction": float(tangled.get("friction", 2.0)),
                "bounce": float(tangled.get("bounce", 0.0)),
                "rotation": float(tangled.get("rotation", 0.0)),
                "offsetX": float(tangled.get("offsetX", 0.0)),
                "offsetY": float(tangled.get("offsetY", 0.0)),
                "offsetZ": float(tangled.get("offsetZ", 0.05)),
            }
    
    def get_leg_cause(self, leg_id: str) -> str:
        """指定された脚の障害タイプを取得"""
        return self.leg_causes.get(leg_id, CAUSE_NONE)
    
    def get_foot_offset(self, leg_id: str) -> Optional[Tuple[float, float]]:
        """指定された脚の位置オフセットを取得"""
        return self.foot_offsets.get(leg_id)
    
    def get_affected_legs(self, cause: str) -> list:
        """指定された障害タイプの脚のリストを取得"""
        return [leg_id for leg_id, leg_cause in self.leg_causes.items() if leg_cause == cause]
    
    def print_summary(self):
        """設定内容のサマリーを表示"""
        print("\n" + "=" * 70)
        print("シナリオ設定サマリー")
        print("=" * 70)
        print("\n【各脚の障害設定】")
        for leg_id in ALL_LEGS:
            cause = self.leg_causes.get(leg_id, CAUSE_NONE)
            offset = self.foot_offsets.get(leg_id, (0, 0))
            print(f"  {leg_id}: {cause:12s} (位置: X={offset[0]:+.2f}, Y={offset[1]:+.2f})")
        
        print("\n【障害タイプ別の脚数】")
        for cause in VALID_CAUSES:
            legs = self.get_affected_legs(cause)
            count = len(legs)
            if count > 0:
                print(f"  {cause:12s}: {count}脚 {legs}")
        
        print("=" * 70 + "\n")


# グローバル設定インスタンス（後方互換性のため）
_global_config: Optional[ScenarioConfig] = None


def load_scenario_config(config_path: Path) -> ScenarioConfig:
    """シナリオ設定を読み込む"""
    global _global_config
    _global_config = ScenarioConfig(config_path)
    return _global_config


def get_global_config() -> Optional[ScenarioConfig]:
    """グローバル設定インスタンスを取得"""
    return _global_config
