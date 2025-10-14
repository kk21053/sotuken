#!/usr/bin/env python3
"""最新のrobopose_features.jsonlからFL脚とFR脚のデータを抽出"""
import json
from collections import defaultdict
from statistics import median

print("=" * 80)
print("最新のRoboPose特徴量データ分析 (FL, FR脚)")
print("=" * 80)

# leg_nameごとに最新の6試行を収集
leg_data = defaultdict(list)

with open('controllers/drone_circular_controller/logs/robopose_features.jsonl', 'r') as f:
    lines = f.readlines()
    
    # 後方から読み取り（最新データ）
    # FL, FRそれぞれについて最新の6試行を取得
    for line in reversed(lines):
        data = json.loads(line)
        leg_name = data.get('leg_name')
        
        if leg_name not in ['FL', 'FR']:
            continue
        
        if len(leg_data[leg_name]) >= 6:
            continue
        
        leg_data[leg_name].append({
            'trial_index': data.get('trial_index'),
            'end_disp': data.get('end_disp'),
            'hip_delta': data.get('hip_delta'),
            'shoulder_delta': data.get('shoulder_delta'),
            'knee_delta': data.get('knee_delta'),
            'weight_on_leg': data.get('weight_on_leg'),
            'delta_theta_deg': data.get('delta_theta_deg'),
            'fallen': data.get('fallen'),
        })
        
        # 両方の脚で6試行ずつ集まったら終了
        if len(leg_data['FL']) >= 6 and len(leg_data['FR']) >= 6:
            break

# 各脚のデータを表示
for leg_name in ['FL', 'FR']:
    if leg_name not in leg_data or not leg_data[leg_name]:
        print(f"\n{leg_name}脚: データなし")
        continue
    
    trials = list(reversed(leg_data[leg_name]))  # 時系列順に戻す
    
    print(f"\n{leg_name}脚 (最新6試行):")
    print(f"  {'Trial':<10} {'end_disp(mm)':<15} {'hip_delta(°)':<15} {'weight':<10} {'fallen':<10}")
    print(f"  {'-'*70}")
    
    for trial in trials:
        end_disp_mm = trial['end_disp'] * 1000 if trial['end_disp'] is not None else 0
        print(f"  {trial['trial_index']:<10} {end_disp_mm:<15.2f} {trial['hip_delta']:<15.2f} {trial['weight_on_leg']:<10.3f} {trial['fallen']}")
    
    # 中央値を計算
    end_disps = [t['end_disp'] for t in trials if t['end_disp'] is not None]
    hip_deltas = [t['hip_delta'] for t in trials if t['hip_delta'] is not None]
    weights = [t['weight_on_leg'] for t in trials if t['weight_on_leg'] is not None]
    
    if end_disps and hip_deltas and weights:
        print(f"  {'median':<10} {median(end_disps)*1000:<15.2f} {median(hip_deltas):<15.2f} {median(weights):<10.3f}")
        
        # TRAPPED判定の閾値チェック
        print(f"\n  判定分析:")
        print(f"    中央値 end_disp = {median(end_disps)*1000:.2f}mm")
        print(f"    中央値 hip_delta = {median(hip_deltas):.2f}°")
        print(f"    中央値 weight = {median(weights):.3f}")
        
        # 現在の閾値
        TRAPPED_MIN = 2.0  # mm
        TRAPPED_MAX = 8.0  # mm (新しい値)
        HIP_MIN = 2.0  # °
        WEIGHT_THRESHOLD = 0.32
        END_DISP_WEIGHT = 9.0  # mm
        HIP_WEIGHT = 3.0  # °
        
        med_end_disp_mm = median(end_disps) * 1000
        med_hip = median(hip_deltas)
        med_weight = median(weights)
        
        print(f"\n    TRAPPED判定条件:")
        print(f"      {TRAPPED_MIN}mm <= end_disp < {TRAPPED_MAX}mm: {TRAPPED_MIN <= med_end_disp_mm < TRAPPED_MAX}")
        print(f"      hip_delta >= {HIP_MIN}°: {med_hip >= HIP_MIN}")
        
        print(f"\n    重心補正条件:")
        print(f"      weight > {WEIGHT_THRESHOLD}: {med_weight > WEIGHT_THRESHOLD}")
        print(f"      end_disp >= {END_DISP_WEIGHT}mm: {med_end_disp_mm >= END_DISP_WEIGHT}")
        print(f"      hip_delta < {HIP_WEIGHT}°: {med_hip < HIP_WEIGHT}")
        
        # 最終判定
        is_trapped = (TRAPPED_MIN <= med_end_disp_mm < TRAPPED_MAX) and (med_hip >= HIP_MIN)
        is_weight_compensated = (med_weight > WEIGHT_THRESHOLD) and (med_end_disp_mm >= END_DISP_WEIGHT) and (med_hip < HIP_WEIGHT)
        
        print(f"\n    最終判定:")
        print(f"      TRAPPED条件を満たす: {is_trapped}")
        print(f"      重心補正を満たす: {is_weight_compensated}")
        
        if is_weight_compensated:
            print(f"      → NONE (重心補正により正常判定)")
        elif is_trapped:
            print(f"      → TRAPPED")
        else:
            print(f"      → その他の条件で判定")

print("\n" + "=" * 80)
