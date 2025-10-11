#!/usr/bin/env python3
"""
診断システムの動作確認スクリプト

Webotsを使わずに診断パイプラインの基本動作を確認できます。
"""

import sys
from pathlib import Path

# Add diagnostics_pipeline to path
SCRIPT_DIR = Path(__file__).resolve().parent
CONTROLLERS_DIR = SCRIPT_DIR / "webots" / "controllers"
sys.path.insert(0, str(CONTROLLERS_DIR))

from diagnostics_pipeline import config
from diagnostics_pipeline.models import LegState, TrialResult
from diagnostics_pipeline.self_diagnosis import SelfDiagnosisAggregator
from diagnostics_pipeline.drone_observer import DroneObservationAggregator
from diagnostics_pipeline.llm_client import LLMAnalyzer
from diagnostics_pipeline.fusion import fuse_probabilities, select_cause


def test_self_diagnosis():
    """自己診断モジュールのテスト"""
    print("=" * 60)
    print("自己診断モジュールのテスト")
    print("=" * 60)
    
    aggregator = SelfDiagnosisAggregator()
    leg = LegState(leg_id="FL")
    
    # 4回の試行をシミュレート
    for i in range(1, 5):
        direction = config.TRIAL_PATTERN[i - 1]
        print(f"\n試行 {i}: 方向={direction}")
        
        # ダミーデータ
        theta_cmd = [0, 2, 4, 6, 7] if direction == "+" else [0, -2, -4, -6, -7]
        theta_meas = [x * 0.8 for x in theta_cmd]  # 80%追従
        omega_meas = [10.0] * 5
        tau_meas = [5.0] * 5
        
        trial = aggregator.record_trial(
            leg, i, direction, 0.0, 0.5,
            theta_cmd, theta_meas, omega_meas, tau_meas,
            tau_nominal=60.0, safety_level="normal"
        )
        
        print(f"  self_can_raw: {trial.self_can_raw:.3f}")
    
    aggregator.finalize_leg(leg)
    print(f"\n最終結果:")
    print(f"  self_can: {leg.self_can:.3f}")
    print(f"  self_moves: {leg.self_moves}")
    print(f"  moves_final: {leg.moves_final}")


def test_drone_observation():
    """ドローン観測モジュールのテスト"""
    print("\n" + "=" * 60)
    print("ドローン観測モジュールのテスト")
    print("=" * 60)
    
    aggregator = DroneObservationAggregator()
    leg = LegState(leg_id="FL")
    
    for i in range(1, 5):
        direction = config.TRIAL_PATTERN[i - 1]
        print(f"\n試行 {i}: 方向={direction}")
        
        trial = TrialResult(
            leg_id="FL", trial_index=i, direction=direction,
            start_time=0.0, end_time=0.5
        )
        
        # ダミーRoboPoseデータ
        joint_angles = [[0.1 * j, 0.2 * j, 0.3 * j] for j in range(10)]
        end_positions = [(0.01 * j, 0.02 * j, 0.3) for j in range(10)]
        base_orientations = [(0.0, 0.0, 0.0)] * 10
        base_positions = [(0.0, 0.0, 0.32)] * 10
        
        aggregator.process_trial(
            leg, trial, joint_angles, end_positions,
            base_orientations, base_positions
        )
        
        print(f"  drone_can_raw: {trial.drone_can_raw:.3f}")
    
    print(f"\n最終結果:")
    print(f"  drone_can: {leg.drone_can:.3f}")
    print(f"  p_drone: {leg.p_drone}")
    print(f"  fallen: {aggregator.fallen}")


def test_llm_analyzer():
    """LLM分析モジュールのテスト"""
    print("\n" + "=" * 60)
    print("LLM分析モジュールのテスト")
    print("=" * 60)
    
    analyzer = LLMAnalyzer()
    leg = LegState(leg_id="FL")
    leg.self_can = 0.45
    leg.self_moves = False
    leg.drone_can = 0.40
    
    # ダミー特徴
    trial = TrialResult(
        leg_id="FL", trial_index=1, direction="+",
        start_time=0.0, end_time=0.5
    )
    trial.features_drone = {
        "delta_theta_deg": 3.0,
        "end_disp": 0.02,
        "path_straightness": 1.5,
    }
    leg.trials.append(trial)
    
    print("\n入力:")
    print(f"  self_can: {leg.self_can:.3f}")
    print(f"  drone_can: {leg.drone_can:.3f}")
    
    p_llm = analyzer.infer(leg)
    
    print(f"\nLLM推論結果:")
    for label, prob in p_llm.items():
        print(f"  {label}: {prob:.3f}")


def test_fusion():
    """確率融合のテスト"""
    print("\n" + "=" * 60)
    print("確率融合のテスト")
    print("=" * 60)
    
    p_drone = {"NONE": 0.2, "BURIED": 0.5, "TRAPPED": 0.2, "ENTANGLED": 0.1}
    p_llm = {"NONE": 0.1, "BURIED": 0.6, "TRAPPED": 0.2, "ENTANGLED": 0.1}
    
    print("\nドローン観測:")
    for label, prob in p_drone.items():
        print(f"  {label}: {prob:.3f}")
    
    print("\nLLM推論:")
    for label, prob in p_llm.items():
        print(f"  {label}: {prob:.3f}")
    
    p_final = fuse_probabilities(p_drone, p_llm)
    cause = select_cause(p_final)
    
    print(f"\n融合結果 (drone={config.FUSION_WEIGHTS['drone']}, llm={config.FUSION_WEIGHTS['llm']}):")
    for label, prob in p_final.items():
        marker = " ←" if label == cause else ""
        print(f"  {label}: {prob:.3f}{marker}")
    
    print(f"\n最終判定: {cause} (信頼度={p_final[cause]:.3f})")


def main():
    """全テストを実行"""
    print("\n診断パイプライン 動作確認テスト\n")
    
    try:
        test_self_diagnosis()
        test_drone_observation()
        test_llm_analyzer()
        test_fusion()
        
        print("\n" + "=" * 60)
        print("✅ 全テスト完了")
        print("=" * 60)
        print("\n次のステップ:")
        print("1. Webotsでシミュレーションを実行")
        print("2. RUN_SIMULATION.md の手順に従ってください")
        
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
