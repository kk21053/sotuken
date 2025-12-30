#!/usr/bin/env python3
"""
ルールベース診断のテストスクリプト

LLMを使用せず、ルールベースのみで診断が正しく動作するか確認します。
"""

import sys
from pathlib import Path

# diagnostics_pipelineをパスに追加
CONTROLLERS_ROOT = Path(__file__).resolve().parent / "controllers"
sys.path.insert(0, str(CONTROLLERS_ROOT))

from diagnostics_pipeline.llm_client import RuleBasedAnalyzer
from diagnostics_pipeline.models import LegState

def test_rule_based():
    """ルールベース診断のテスト"""
    print("=" * 80)
    print("ルールベース診断テスト")
    print("=" * 80)
    
    # アナライザーを初期化
    analyzer = RuleBasedAnalyzer()
    
    # テストケース1: ルール1（両方高い - 正常動作）
    print("\n[テストケース1] ルール1: 両方≥0.7（正常動作）")
    leg1 = LegState(leg_id="FL")
    leg1.spot_can = 0.9
    leg1.drone_can = 0.85
    leg1.p_drone = {
        "NONE": 0.85,
        "BURIED": 0.05,
        "TRAPPED": 0.03,
        "TANGLED": 0.02,
        "MALFUNCTION": 0.03,
        "FALLEN": 0.02,
    }
    
    result1 = analyzer.infer(leg1)
    print(f"  spot_can={leg1.spot_can}, drone_can={leg1.drone_can}")
    print(f"  → 結果: {leg1.movement_result}, 原因={leg1.cause_final}")
    print(f"  → 確率分布: {result1}")
    assert leg1.movement_result == "動く", f"期待: 動く, 実際: {leg1.movement_result}"
    assert leg1.cause_final == "NONE", f"期待: NONE, 実際: {leg1.cause_final}"
    print("  ✓ 合格")
    
    # テストケース2: ルール2（両方低い - 埋まっている）
    print("\n[テストケース2] ルール2: 両方≤0.3（動かない）")
    leg2 = LegState(leg_id="FR")
    leg2.spot_can = 0.2
    leg2.drone_can = 0.15
    leg2.p_drone = {
        "NONE": 0.05,
        "BURIED": 0.80,
        "TRAPPED": 0.05,
        "TANGLED": 0.03,
        "MALFUNCTION": 0.05,
        "FALLEN": 0.02,
    }
    
    result2 = analyzer.infer(leg2)
    print(f"  spot_can={leg2.spot_can}, drone_can={leg2.drone_can}")
    print(f"  → 結果: {leg2.movement_result}, 原因={leg2.cause_final}")
    print(f"  → 確率分布: {result2}")
    assert leg2.movement_result == "動かない", f"期待: 動かない, 実際: {leg2.movement_result}"
    assert leg2.cause_final == "BURIED", f"期待: BURIED, 実際: {leg2.cause_final}"
    print("  ✓ 合格")
    
    # テストケース3: ルール3（矛盾 - センサー故障）
    print("\n[テストケース3] ルール3: センサー矛盾（故障）")
    leg3 = LegState(leg_id="RL")
    leg3.spot_can = 0.8
    leg3.drone_can = 0.2
    leg3.p_drone = {
        "NONE": 0.15,
        "BURIED": 0.20,
        "TRAPPED": 0.30,
        "TANGLED": 0.10,
        "MALFUNCTION": 0.20,
        "FALLEN": 0.05,
    }
    
    result3 = analyzer.infer(leg3)
    print(f"  spot_can={leg3.spot_can}, drone_can={leg3.drone_can}")
    print(f"  → 結果: {leg3.movement_result}, 原因={leg3.cause_final}")
    print(f"  → 確率分布: {result3}")
    assert leg3.movement_result == "動かない", f"期待: 動かない, 実際: {leg3.movement_result}"
    assert leg3.cause_final == "MALFUNCTION", f"期待: MALFUNCTION, 実際: {leg3.cause_final}"
    print("  ✓ 合格")
    
    # テストケース4: ルール4（中間値 - 一部動く）
    print("\n[テストケース4] ルール4: 中間値（一部動く）")
    leg4 = LegState(leg_id="RR")
    leg4.spot_can = 0.5
    leg4.drone_can = 0.45
    leg4.p_drone = {
        "NONE": 0.20,
        "BURIED": 0.15,
        "TRAPPED": 0.40,
        "TANGLED": 0.15,
        "MALFUNCTION": 0.05,
        "FALLEN": 0.05,
    }
    
    result4 = analyzer.infer(leg4)
    print(f"  spot_can={leg4.spot_can}, drone_can={leg4.drone_can}")
    print(f"  → 結果: {leg4.movement_result}, 原因={leg4.cause_final}")
    print(f"  → 確率分布: {result4}")
    assert leg4.movement_result == "一部動く", f"期待: 一部動く, 実際: {leg4.movement_result}"
    assert leg4.cause_final == "TRAPPED", f"期待: TRAPPED, 実際: {leg4.cause_final}"
    print("  ✓ 合格")
    
    print("\n" + "=" * 80)
    print("全テスト合格！ ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_rule_based()
