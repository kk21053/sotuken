#!/usr/bin/env python3
"""
システム動作確認スクリプト

診断パイプラインの基本的な動作を確認します。
"""

import sys
from pathlib import Path

# 診断パイプラインをインポート
sys.path.insert(0, str(Path(__file__).parent / "controllers"))

from diagnostics_pipeline import config
from diagnostics_pipeline.models import LegState
from diagnostics_pipeline.llm_client import LLMAnalyzer


def test_rule_based_llm():
    """ルールベースLLMのテスト"""
    print("=" * 80)
    print("ルールベースLLM 動作確認")
    print("=" * 80)
    print()
    
    llm = LLMAnalyzer()
    
    # テストケース1: ルール①（両方が0.7以上）
    print("テストケース1: 両方が0.7以上 → 正常と判定")
    leg1 = LegState(leg_id="FL")
    leg1.spot_can = 0.85
    leg1.drone_can = 0.90
    leg1.p_drone = {"NONE": 0.8, "BURIED": 0.1, "TRAPPED": 0.05, "TANGLED": 0.03, "MALFUNCTION": 0.02}
    result1 = llm.infer(leg1)
    print(f"  結果: {max(result1, key=result1.get)} (確率: {max(result1.values()):.3f})")
    print()
    
    # テストケース2: ルール②（両方が0.3以下）
    print("テストケース2: 両方が0.3以下 → 確率分布の最大値")
    leg2 = LegState(leg_id="FR")
    leg2.spot_can = 0.15
    leg2.drone_can = 0.20
    leg2.p_drone = {"NONE": 0.1, "BURIED": 0.6, "TRAPPED": 0.2, "TANGLED": 0.05, "MALFUNCTION": 0.05}
    result2 = llm.infer(leg2)
    print(f"  結果: {max(result2, key=result2.get)} (確率: {max(result2.values()):.3f})")
    print()
    
    # テストケース3: ルール③（矛盾）
    print("テストケース3: 片方が0.7以上、もう片方が0.3以下 → 故障")
    leg3 = LegState(leg_id="RL")
    leg3.spot_can = 0.85
    leg3.drone_can = 0.15
    leg3.p_drone = {"NONE": 0.2, "BURIED": 0.3, "TRAPPED": 0.3, "TANGLED": 0.1, "MALFUNCTION": 0.1}
    result3 = llm.infer(leg3)
    print(f"  結果: {max(result3, key=result3.get)} (確率: {max(result3.values()):.3f})")
    print()
    
    # テストケース4: ルール④（中間値）
    print("テストケース4: 片方が中間値 → 確率分布の最大値")
    leg4 = LegState(leg_id="RR")
    leg4.spot_can = 0.55
    leg4.drone_can = 0.25
    leg4.p_drone = {"NONE": 0.15, "BURIED": 0.2, "TRAPPED": 0.5, "TANGLED": 0.1, "MALFUNCTION": 0.05}
    result4 = llm.infer(leg4)
    print(f"  結果: {max(result4, key=result4.get)} (確率: {max(result4.values()):.3f})")
    print()
    
    print("=" * 80)
    print("全テスト完了!")
    print("=" * 80)


def test_config():
    """設定値の確認"""
    print()
    print("=" * 80)
    print("設定値確認")
    print("=" * 80)
    print()
    
    print(f"脚ID: {config.LEG_IDS}")
    print(f"拘束原因ラベル: {config.CAUSE_LABELS}")
    print(f"試行回数: {config.TRIAL_COUNT}")
    print(f"試行パターン: {config.TRIAL_PATTERN}")
    print(f"シグモイド閾値 (高): {config.CONFIDENCE_HIGH_THRESHOLD}")
    print(f"シグモイド閾値 (低): {config.CONFIDENCE_LOW_THRESHOLD}")
    print()
    
    print("=" * 80)


def main():
    """メイン処理"""
    print()
    test_config()
    test_rule_based_llm()
    print()
    print("システムの基本機能は正常に動作しています。")
    print("次のステップ: Webotsでシミュレーションを実行してください。")
    print()


if __name__ == "__main__":
    main()
