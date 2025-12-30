"""Rule-based analysis combining self-diagnosis and drone observation."""

from __future__ import annotations

from typing import Dict

from . import config
from .models import LegState


CAUSE_DEFINITIONS = {
    "NONE": "脚は正常に動作しており障害がない状態",
    "BURIED": "脚が地面や砂に埋まっており大きく持ち上げられない状態",
    "TRAPPED": "関節は動くのに末端が障害物等に固定され前進できない状態",
    "TANGLED": "ツタなどに絡まり小さい往復でしか動けない状態",
    "MALFUNCTION": "センサー故障または測定エラー（物理的に矛盾する状態）",
    "FALLEN": "ロボットが転倒しており正常な診断が困難な状態",
}


class RuleBasedAnalyzer:
    """
    Rule-based analyzer for leg constraint diagnosis.
    
    ルールベース診断分析器:
    - 仕様のルールに基づいて診断
    - spot_can と drone_can の閾値比較
    - LLMを使用せず、高速で安定した推論
    
    Processing time: < 1ms per leg
    Memory usage: minimal
    """

    def __init__(self) -> None:
        """Initialize the rule-based analyzer."""
        print("[analyzer] Using rule-based inference")

    def infer(self, leg: LegState, all_legs=None, trial_direction=None) -> Dict[str, float]:
        """
        Rule-based inference for leg constraint diagnosis.
        
        仕様のルール:
        ①両方が0.7以上 → 動く、拘束原因=正常
        ②両方が0.3以下 → 動かない、拘束原因=確率分布の最大値
        ③片方が0.7以上、もう片方が0.3以下 → 動かない、拘束原因=故障
        ④片方が中間値 → 一部動く、拘束原因=確率分布の最大値
        
        Args:
            leg: LegState object with spot_can, drone_can, and p_drone
            all_legs: Not used (kept for API compatibility)
            trial_direction: Not used (kept for API compatibility)
        
        Returns:
            Dictionary of probabilities for each cause label
        """
        return self._infer_with_rules(leg)
    
    def infer_with_confidence(self, leg: LegState, all_legs=None, trial_direction=None) -> tuple[Dict[str, float], float]:
        """
        Rule-based inference with confidence score.
        
        Args:
            leg: LegState object with spot_can, drone_can, and p_drone
            all_legs: Not used (kept for API compatibility)
            trial_direction: Not used (kept for API compatibility)
        
        Returns:
            Tuple of (probability distribution, confidence score)
            confidence score = max probability in distribution
        """
        distribution = self._infer_with_rules(leg)
        confidence = max(distribution.values())
        return distribution, confidence
    
    
    def _infer_with_rules(self, leg: LegState) -> Dict[str, float]:
        """ルールベース推論（フォールバック）"""
        # 仕様通り、spot_can と drone_can を使用
        spot_can = leg.spot_can
        drone_can = leg.drone_can
        p_drone = dict(leg.p_drone)  # ドローンの確率分布をコピー
        
                # ルール①: 両方が0.7以上 → 動く、拘束原因=正常
        if spot_can >= 0.7 and drone_can >= 0.7:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2  # 平均を最終確率とする
            distribution = {
                "NONE": 0.94,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            leg.p_llm = distribution
            leg.cause_final = "NONE"
            return distribution
        
        # ルール②: 両方が0.3以下 → 動かない、拘束原因=確率分布の最大値（NONE除外）
        elif spot_can <= 0.3 and drone_can <= 0.3:
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            
            # 確率分布から最大値を見つける（NONE以外から選択）
            # 「動かない」場合は正常(NONE)ではないため
            max_cause = max(
                (v, k) for k, v in p_drone.items() if k != "NONE"
            )[1]
            
            # 最大値の原因を強調した分布を作成
            distribution = {
                "NONE": 0.02,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.02,
                "MALFUNCTION": 0.02,
                "FALLEN": 0.01,
            }
            distribution[max_cause] = 0.89
            
            leg.p_llm = distribution
            leg.cause_final = max_cause
            return distribution
        
        # ルール③: 片方が0.7以上、もう片方が0.3以下 → 動かない、拘束原因=故障
        elif (spot_can >= 0.7 and drone_can <= 0.3) or (spot_can <= 0.3 and drone_can >= 0.7):
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            distribution = {
                "NONE": 0.01,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.93,  # 故障を強く示唆
                "FALLEN": 0.01,
            }
            leg.p_llm = distribution
            leg.cause_final = "MALFUNCTION"
            return distribution
        
        # ルール④: 片方が中間値 → 一部動く、拘束原因=確率分布の最大値
        else:
            leg.movement_result = "一部動く"
            leg.p_can = (spot_can + drone_can) / 2
            
            # 確率分布から最大値を見つける（仕様通り）
            max_cause = max(p_drone.items(), key=lambda x: x[1])[0]
            
            # 中間的な分布を作成（確信度は低め）
            distribution = {
                "NONE": 0.10,
                "BURIED": 0.05,
                "TRAPPED": 0.05,
                "TANGLED": 0.05,
                "MALFUNCTION": 0.05,
                "FALLEN": 0.01,
            }
            distribution[max_cause] = 0.69
            
            leg.p_llm = distribution
            leg.cause_final = max_cause
            return distribution
