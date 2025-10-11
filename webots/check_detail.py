#!/usr/bin/env python3
"""
診断結果の詳細確認スクリプト
使い方: python3 check_detail.py
"""
import json
from pathlib import Path

def check_detailed_results():
    """詳細な診断データを表示"""
    
    events_file = Path("controllers/spot_self_diagnosis/logs/leg_diagnostics_events.jsonl")
    
    if not events_file.exists():
        print("❌ エラー: イベントログが見つかりません")
        print(f"📁 確認場所: {events_file.absolute()}")
        return
    
    print("=" * 80)
    print("🔍 詳細診断データ - モーターの実際の動き")
    print("=" * 80)
    print()
    
    # 最新16イベント(4脚 x 4試行)を読み込む
    events = []
    with open(events_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'leg_id' in data and 'trial_index' in data:
                events.append(data)
    
    if len(events) < 16:
        print(f"⚠️  警告: イベント数が少ないです ({len(events)}個)")
        print()
    
    # 最新16イベントを取得
    recent_events = events[-16:] if len(events) >= 16 else events
    
    # 脚ごとにグループ化
    leg_trials = {}
    for evt in recent_events:
        leg_id = evt['leg_id']
        if leg_id not in leg_trials:
            leg_trials[leg_id] = []
        leg_trials[leg_id].append(evt)
    
    # 使用しているモーターを確認
    if recent_events:
        first_evt = recent_events[0]
        # モーター名を推測
        features = first_evt.get('features_drone', {})
        delta_theta = features.get('delta_theta_deg', 0)
        
        if abs(delta_theta) < 0.01:
            motor_type = "Elbow (index 2) - 動作不可"
        elif abs(delta_theta) < 1.0:
            motor_type = "不明 - 要確認"
        else:
            motor_type = "Shoulder (index 0 or 1)"
        
        print(f"🔧 使用モーター: {motor_type}")
        print()
    
    # 各脚の詳細データを表示
    for leg_id in ["FL", "FR", "RL", "RR"]:
        if leg_id not in leg_trials:
            continue
        
        expected = "BURIED" if leg_id == "FL" else "NONE"
        print(f"【{leg_id}脚】 期待値: {expected}")
        print("-" * 80)
        
        displacements = []
        thetas = []
        
        for evt in leg_trials[leg_id]:
            trial = evt['trial_index']
            direction = evt.get('dir', '?')
            
            features = evt.get('features_drone', {})
            delta_theta = features.get('delta_theta_deg', 0)
            end_disp = features.get('end_disp', 0)
            
            displacements.append(end_disp)
            thetas.append(abs(delta_theta))
            
            print(f"  Trial {trial} ({direction:>2}): "
                  f"角度変化={delta_theta:+7.3f}°, "
                  f"足先変位={end_disp:.6f}m")
        
        # 統計を計算
        avg_disp = sum(displacements) / len(displacements) if displacements else 0
        avg_theta = sum(thetas) / len(thetas) if thetas else 0
        
        # 判定
        detected = "BURIED" if avg_disp < 0.01 else "NONE"
        match = "✓" if detected == expected else "✗"
        
        print(f"\n  平均角度変化: {avg_theta:.3f}°")
        print(f"  平均足先変位: {avg_disp:.6f}m (閾値: 0.010000m)")
        print(f"  検出結果: {detected} {match}")
        print()
    
    # 分析コメント
    print("=" * 80)
    print("📊 分析")
    print("=" * 80)
    print()
    
    # すべての脚の平均角度変化を計算
    all_thetas = []
    for trials in leg_trials.values():
        for evt in trials:
            features = evt.get('features_drone', {})
            delta_theta = features.get('delta_theta_deg', 0)
            all_thetas.append(abs(delta_theta))
    
    overall_avg_theta = sum(all_thetas) / len(all_thetas) if all_thetas else 0
    
    if overall_avg_theta < 0.01:
        print("❌ モーターがほとんど動いていません (平均 < 0.01°)")
        print("   → Elbow motor (index 2) は物理的制約のため使用不可")
        print("   → 別のモーターに変更してください")
    elif overall_avg_theta < 1.0:
        print("⚠️  モーターの動きが小さいです (平均 < 1°)")
        print("   → 検出精度が低い可能性があります")
        print("   → モーター選択や角度設定を見直してください")
    else:
        print("✓ モーターは正常に動作しています (平均 > 1°)")
        print(f"   平均角度変化: {overall_avg_theta:.3f}°")
    
    print()

if __name__ == "__main__":
    check_detailed_results()
