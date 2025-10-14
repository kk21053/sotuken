#!/usr/bin/env python3
"""過去のセッションからFR脚とFL脚のデータを抽出"""
import json
from collections import defaultdict

# 最新の3セッションを確認
sessions_to_check = []

print("=" * 80)
print("過去のセッションデータ分析")
print("=" * 80)

# セッションログから基本情報を取得
with open('controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl', 'r') as f:
    lines = f.readlines()
    # 最新3セッションを取得
    for line in lines[-3:]:
        data = json.loads(line)
        session_id = data['session_id']
        sessions_to_check.append(session_id)
        print(f"\nセッション: {session_id}")
        
        # 各脚の判定結果を表示
        for leg_id in ['FL', 'FR', 'RL', 'RR']:
            leg_data = data['legs'].get(leg_id, {})
            cause = leg_data.get('cause_final', 'N/A')
            expected = leg_data.get('expected_cause', 'N/A')
            print(f"  {leg_id}: {cause} (期待: {expected})")

# RoboPose特徴量ログから詳細データを取得
print("\n" + "=" * 80)
print("RoboPose特徴量データ")
print("=" * 80)

feature_data = defaultdict(lambda: defaultdict(list))

with open('controllers/drone_circular_controller/logs/robopose_features.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        session_id = data.get('session_id')
        if session_id not in sessions_to_check:
            continue
        
        leg_id = data.get('leg_name')  # 'leg_id' ではなく 'leg_name'
        trial_id = data.get('trial_index')  # 'trial_id' ではなく 'trial_index'
        
        # 特徴量を保存
        feature_data[session_id][leg_id].append({
            'trial_id': trial_id,
            'end_disp': data.get('end_disp'),
            'hip_delta': data.get('hip_delta'),
            'shoulder_delta': data.get('shoulder_delta'),
            'knee_delta': data.get('knee_delta'),
            'weight_on_leg': data.get('weight_on_leg'),
            'delta_theta_deg': data.get('delta_theta_deg'),
        })

# 詳細表示
for session_id in sessions_to_check:
    print(f"\n{'='*80}")
    print(f"セッション: {session_id}")
    print(f"{'='*80}")
    
    for leg_id in ['FL', 'FR']:
        if leg_id not in feature_data[session_id]:
            continue
            
        trials = feature_data[session_id][leg_id]
        if not trials:
            continue
        
        print(f"\n{leg_id}脚:")
        print(f"  {'Trial':<10} {'end_disp(mm)':<15} {'hip_delta(°)':<15} {'weight':<10}")
        print(f"  {'-'*60}")
        
        for trial in trials:
            if trial['end_disp'] is not None:
                print(f"  {trial['trial_id']:<10} {trial['end_disp']*1000:<15.2f} {trial['hip_delta']:<15.2f} {trial['weight_on_leg']:<10.3f}")
        
        # 中央値を計算
        if len(trials) >= 3:
            end_disps = [t['end_disp'] for t in trials if t['end_disp'] is not None]
            hip_deltas = [t['hip_delta'] for t in trials if t['hip_delta'] is not None]
            weights = [t['weight_on_leg'] for t in trials if t['weight_on_leg'] is not None]
            
            if end_disps and hip_deltas and weights:
                from statistics import median
                print(f"  {'median':<10} {median(end_disps)*1000:<15.2f} {median(hip_deltas):<15.2f} {median(weights):<10.3f}")

print("\n" + "=" * 80)
