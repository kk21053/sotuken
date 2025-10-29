#!/usr/bin/env python3
"""
LLM診断のテストスクリプト

Ollamaが正しく動作し、診断が行えるかテストします。
"""

import sys
from pathlib import Path

# diagnostics_pipelineをパスに追加
CONTROLLERS_ROOT = Path(__file__).resolve().parent / "controllers"
sys.path.insert(0, str(CONTROLLERS_ROOT))

from diagnostics_pipeline.llm_client import LLMAnalyzer
from diagnostics_pipeline.models import LegState
from diagnostics_pipeline import config

def test_llm_basic():
    """基本的なLLM動作テスト"""
    print("=" * 80)
    print("LLM診断テスト")
    print("=" * 80)
    
    # LLMアナライザーを初期化
    llm = LLMAnalyzer(model_name=config.LLM_MODEL, use_llm=True)
    
    # テストケース1: 両方高い確率（正常動作）
    print("\n[テストケース1] 正常動作（spot_can=0.9, drone_can=0.85）")
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
    
    result1 = llm.infer(leg1)
    print(f"結果: {leg1.movement_result}, 原因={leg1.cause_final}")
    print(f"確率分布: {result1}")
    
    # テストケース2: 両方低い確率（埋まっている）
    print("\n[テストケース2] 埋まっている（spot_can=0.2, drone_can=0.15）")
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
    
    result2 = llm.infer(leg2)
    print(f"結果: {leg2.movement_result}, 原因={leg2.cause_final}")
    print(f"確率分布: {result2}")
    
    # テストケース3: 不一致（センサー故障の可能性）
    print("\n[テストケース3] センサー不一致（spot_can=0.8, drone_can=0.2）")
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
    
    result3 = llm.infer(leg3)
    print(f"結果: {leg3.movement_result}, 原因={leg3.cause_final}")
    print(f"確率分布: {result3}")
    
    # テストケース4: 中間値（一部動く）
    print("\n[テストケース4] 一部動く（spot_can=0.5, drone_can=0.45）")
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
    
    result4 = llm.infer(leg4)
    print(f"結果: {leg4.movement_result}, 原因={leg4.cause_final}")
    print(f"確率分布: {result4}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    test_llm_basic()
