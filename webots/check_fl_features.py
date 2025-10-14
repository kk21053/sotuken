#!/usr/bin/env python3
"""
FL脚の詳細な動作特徴を確認するスクリプト
"""

import json
from pathlib import Path

# RoboPoseのログから特徴量を取得
robopose_log = Path("controllers/drone_circular_controller/logs/robopose_features.jsonl")

if robopose_log.exists():
    with open(robopose_log, 'r') as f:
        lines = f.readlines()
        
    print("=== 最新のRoboPose特徴量 (FL脚) ===\n")
    
    # 最後の診断セッションのFL脚データを取得
    fl_trials = []
    for line in reversed(lines):
        try:
            data = json.loads(line)
            if data.get('leg_name') == 'FL':
                fl_trials.append(data)
                if len(fl_trials) >= 6:  # 6試行分
                    break
        except:
            continue
    
    fl_trials.reverse()
    
    for i, trial in enumerate(fl_trials, 1):
        print(f"試行 {i}:")
        print(f"  delta_theta: {trial.get('delta_theta_deg', 'N/A'):.2f}°")
        print(f"  end_displacement: {trial.get('end_displacement', 'N/A')*1000:.2f}mm")
        print(f"  path_straightness: {trial.get('path_straightness', 'N/A'):.2f}")
        print(f"  reversals: {trial.get('reversals', 'N/A')}")
        
        # TRAPPED判定条件チェック
        end_disp = trial.get('end_displacement', 999) * 1000  # mm
        delta_theta = trial.get('delta_theta_deg', 0)
        
        trapped_cond = end_disp < 10 and delta_theta > 2
        print(f"  → TRAPPED条件 (end<10mm & delta>2°): {trapped_cond}")
        print()
else:
    print(f"RoboPoseログが見つかりません: {robopose_log}")
    print("\n代わりに診断セッションログを確認します...")
    
    # 診断セッションログから情報を取得
    session_log = Path("controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl")
    if session_log.exists():
        with open(session_log, 'r') as f:
            lines = f.readlines()
            data = json.loads(lines[-1])
            
        fl = data['legs']['FL']
        print("\n=== FL脚の診断結果 ===")
        print(f"spot_can: {fl['spot_can']:.3f}")
        print(f"drone_can: {fl['drone_can']:.3f}")
        print(f"movement_result: {fl['movement_result']}")
        print(f"cause_final: {fl['cause_final']}")
        print(f"expected_cause: {fl['expected_cause']}")
        print(f"\np_drone分布:")
        for cause, prob in sorted(fl['p_drone'].items(), key=lambda x: -x[1]):
            print(f"  {cause}: {prob:.4f}")
    else:
        print(f"診断セッションログも見つかりません: {session_log}")
