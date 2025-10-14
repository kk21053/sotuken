#!/usr/bin/env python3
"""
BURIED検出の問題を調査するデバッグスクリプト
"""

import json
from pathlib import Path

def analyze_buried_scenario():
    """BURIEDシナリオの診断結果を分析"""
    
    print("=" * 80)
    print("BURIED検出問題の調査")
    print("=" * 80)
    
    # 診断結果の確認
    session_log = Path("controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl")
    if not session_log.exists():
        print(f"エラー: 診断ログが見つかりません")
        return
    
    with open(session_log, 'r') as f:
        lines = f.readlines()
        session = json.loads(lines[-1])
    
    fl = session['legs']['FL']
    
    print("\n[1] FL脚の診断結果")
    print(f"  spot_can: {fl['spot_can']:.3f}")
    print(f"  drone_can: {fl['drone_can']:.3f}")
    print(f"  movement_result: {fl['movement_result']}")
    print(f"  cause_final: {fl['cause_final']}")
    print(f"  expected_cause: {fl['expected_cause']}")
    print(f"  正解: {'✓' if fl['cause_final'] == fl['expected_cause'] else '✗'}")
    
    print(f"\n[2] p_drone分布")
    for cause, prob in sorted(fl['p_drone'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 50)
        print(f"  {cause:12s}: {prob:.4f} {bar}")
    
    print(f"\n[3] 適用されたルール")
    spot = fl['spot_can']
    drone = fl['drone_can']
    if spot >= 0.7 and drone >= 0.7:
        print(f"  → ルール① (両方>=0.7): spot={spot:.3f}, drone={drone:.3f}")
    elif spot <= 0.3 and drone <= 0.3:
        print(f"  → ルール② (両方<=0.3): spot={spot:.3f}, drone={drone:.3f}")
    elif (spot >= 0.7 and drone <= 0.3) or (spot <= 0.3 and drone >= 0.7):
        print(f"  → ルール③ (矛盾): spot={spot:.3f}, drone={drone:.3f}")
    else:
        print(f"  → ルール④ (中間値): spot={spot:.3f}, drone={drone:.3f}")
    
    # RoboPose特徴量の確認
    print(f"\n[4] RoboPose特徴量（6試行分）")
    robopose_log = Path("controllers/drone_circular_controller/logs/robopose_features.jsonl")
    if robopose_log.exists():
        with open(robopose_log, 'r') as f:
            lines = f.readlines()
        
        fl_trials = []
        for line in reversed(lines):
            try:
                data = json.loads(line)
                if data.get('leg_name') == 'FL':
                    fl_trials.append(data)
                    if len(fl_trials) >= 6:
                        break
            except:
                continue
        
        fl_trials.reverse()
        
        if fl_trials:
            print("\n  閾値: BURIED条件 = end_disp < 5mm AND delta_theta < 0.8°")
            print("  " + "-" * 76)
            print(f"  {'試行':^4} | {'delta_θ':>8} | {'end_disp':>10} | {'TRAPPED?':^10} | {'BURIED?':^10}")
            print("  " + "-" * 76)
            
            for i, trial in enumerate(fl_trials, 1):
                delta = trial.get('delta_theta_deg', 0)
                end_disp = trial.get('end_displacement', 0) * 1000
                
                is_trapped = end_disp < 15 and delta > 1.5
                is_buried = end_disp < 5 and delta < 0.8
                
                trapped_str = "✓" if is_trapped else "✗"
                buried_str = "✓" if is_buried else "✗"
                
                print(f"  {i:^4} | {delta:7.2f}° | {end_disp:9.2f}mm | {trapped_str:^10} | {buried_str:^10}")
            
            print("  " + "-" * 76)
            
            # 統計
            buried_count = sum(1 for t in fl_trials 
                             if t.get('end_displacement', 0) * 1000 < 5 
                             and t.get('delta_theta_deg', 0) < 0.8)
            trapped_count = sum(1 for t in fl_trials 
                              if t.get('end_displacement', 0) * 1000 < 15 
                              and t.get('delta_theta_deg', 0) > 1.5)
            
            print(f"\n  BURIED条件を満たした試行: {buried_count}/6")
            print(f"  TRAPPED条件を満たした試行: {trapped_count}/6")
            
            if buried_count == 0 and trapped_count == 0:
                print("\n  ⚠️  警告: BURIEDもTRAPPEDも検出されていません")
                print("  → FL脚が実際に動いている可能性があります")
                
                avg_delta = sum(t.get('delta_theta_deg', 0) for t in fl_trials) / len(fl_trials)
                avg_end_disp = sum(t.get('end_displacement', 0) for t in fl_trials) / len(fl_trials) * 1000
                
                print(f"\n  平均値:")
                print(f"    delta_theta: {avg_delta:.2f}° (BURIED閾値: <0.8°)")
                print(f"    end_disp: {avg_end_disp:.2f}mm (BURIED閾値: <5mm)")
                
                print(f"\n  根本原因: 物理環境がFL脚を拘束できていません")
                print(f"  → BURIEDシナリオでも脚が動いてしまっています")
    else:
        print(f"  ログファイルが見つかりません: {robopose_log}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_buried_scenario()
