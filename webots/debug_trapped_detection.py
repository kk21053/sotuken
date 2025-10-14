#!/usr/bin/env python3
"""
TRAPPED検出の問題を調査するデバッグスクリプト

TRAPPEDシナリオでFL脚の実際の動作を詳細に記録し、
drone_observerの検出ロジックが正しく機能しているか確認する
"""

import sys
import json
from pathlib import Path

# コントローラーのパスを追加
sys.path.insert(0, str(Path(__file__).parent / "controllers" / "diagnostics_pipeline"))

def analyze_trapped_scenario():
    """TRAPPEDシナリオの診断結果を分析"""
    
    print("=" * 80)
    print("TRAPPED検出問題の調査")
    print("=" * 80)
    
    # 1. 診断結果の確認
    session_log = Path("controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl")
    if not session_log.exists():
        print(f"エラー: 診断ログが見つかりません: {session_log}")
        print("シミュレーションを実行してから再度お試しください")
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
    
    print(f"\n[2] p_drone分布 (drone_observerの出力)")
    for cause, prob in sorted(fl['p_drone'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 50)
        print(f"  {cause:12s}: {prob:.4f} {bar}")
    
    print(f"\n[3] p_llm分布 (llm_clientの出力)")
    for cause, prob in sorted(fl['p_llm'].items(), key=lambda x: -x[1]):
        bar = '█' * int(prob * 50)
        print(f"  {cause:12s}: {prob:.4f} {bar}")
    
    # 2. RoboPose特徴量の確認
    print(f"\n[4] RoboPose特徴量の分析")
    print("  ※ RoboPoseログから6試行分の特徴量を確認")
    
    robopose_log = Path("controllers/drone_circular_controller/logs/robopose_features.jsonl")
    if robopose_log.exists():
        with open(robopose_log, 'r') as f:
            lines = f.readlines()
        
        # 最後の6試行分のFL脚データを取得
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
            print("\n  試行ごとの特徴量:")
            print("  " + "-" * 76)
            print(f"  {'試行':^4} | {'delta_θ':>8} | {'end_disp':>10} | {'path_len':>10} | {'reversals':>10} | {'TRAPPED?':^10}")
            print("  " + "-" * 76)
            
            for i, trial in enumerate(fl_trials, 1):
                delta_theta = trial.get('delta_theta_deg', 0)
                end_disp = trial.get('end_displacement', 0) * 1000  # mm
                path_len = trial.get('path_length', 0) * 1000  # mm
                reversals = trial.get('reversals', 0)
                
                # TRAPPED条件: end_disp < 10mm AND delta_theta > 2°
                is_trapped = end_disp < 10 and delta_theta > 2
                trapped_str = "✓ YES" if is_trapped else "✗ NO"
                
                print(f"  {i:^4} | {delta_theta:7.2f}° | {end_disp:9.2f}mm | {path_len:9.2f}mm | {reversals:10.1f} | {trapped_str:^10}")
            
            print("  " + "-" * 76)
            
            # TRAPPED検出の統計
            trapped_count = sum(1 for t in fl_trials 
                              if t.get('end_displacement', 0) * 1000 < 10 
                              and t.get('delta_theta_deg', 0) > 2)
            print(f"\n  TRAPPED条件を満たした試行: {trapped_count}/6")
            
            if trapped_count == 0:
                print("\n  ⚠️  警告: 6試行すべてでTRAPPED条件を満たしていません")
                print("  → 物理環境がFL脚を十分に拘束できていない可能性があります")
                print("  → scenario.iniの設定を確認してください")
            elif trapped_count < 4:
                print(f"\n  ⚠️  警告: TRAPPED条件を満たした試行が {trapped_count}/6 と少ない")
                print("  → 検出精度が不安定な可能性があります")
        else:
            print("  FL脚のデータが見つかりませんでした")
    else:
        print(f"  ログファイルが見つかりません: {robopose_log}")
        print("  → RoboPose特徴量が記録されていない可能性があります")
    
    # 3. 問題診断と推奨事項
    print(f"\n[5] 問題診断と推奨事項")
    print("  " + "=" * 76)
    
    # p_droneでNONEが支配的な場合
    if fl['p_drone']['NONE'] > 0.8:
        print("\n  問題: p_drone['NONE']が支配的 ({:.1%})".format(fl['p_drone']['NONE']))
        print("  → drone_observerがTRAPPEDを検出できていません")
        
        if robopose_log.exists() and fl_trials:
            avg_end_disp = sum(t.get('end_displacement', 0) for t in fl_trials) / len(fl_trials) * 1000
            avg_delta_theta = sum(t.get('delta_theta_deg', 0) for t in fl_trials) / len(fl_trials)
            
            print(f"\n  実測値の平均:")
            print(f"    - end_displacement: {avg_end_disp:.2f}mm (閾値: <10mm)")
            print(f"    - delta_theta: {avg_delta_theta:.2f}° (閾値: >2°)")
            
            if avg_end_disp >= 10:
                print("\n  根本原因: 足先変位が大きすぎる (>= 10mm)")
                print("  → 物理環境（trap）が脚を拘束できていません")
                print("  → 推奨: scenario.iniのtrap.frictionを増やす、またはspot_controllerのモーター出力を制限")
            elif avg_delta_theta <= 2:
                print("\n  根本原因: 関節角度変化が小さすぎる (<= 2°)")
                print("  → BURIEDと誤認される可能性があります")
                print("  → 推奨: TRAPPED閾値の調整を検討")
            else:
                print("\n  根本原因: 閾値の問題")
                print("  → TRAPPED検出閾値が厳しすぎる可能性があります")
                print(f"  → 推奨: TRAPPED_DISPLACEMENT_THRESHOLDを10mm→15mmに緩和")
    
    # ルール④でNONEが選ばれた場合
    if fl['movement_result'] == "一部動く" and fl['cause_final'] == 'NONE':
        print("\n  問題: ルール④でNONEが選択されました")
        print("  → p_droneの最大値がNONEだったため")
        print("  → drone_observerの検出精度向上が必要です")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_trapped_scenario()
