#!/usr/bin/env python3
"""パターン3のFL脚の特徴量を詳細分析"""
import json

session_id = "drone_diagnosis_20251014_153146"

# イベントログからFL脚のトライアルデータを取得
trials_data = []
with open('controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('session_id') == session_id and data.get('leg_id') == 'FL':
            if data.get('event_type') == 'trial_processed':
                trials_data.append(data)

print("=" * 80)
print(f"パターン3: FL脚のトライアルデータ (セッション: {session_id})")
print("=" * 80)

if trials_data:
    print(f"\n全{len(trials_data)}トライアルの特徴量:")
    print(f"{'Trial':<8} {'end_disp(mm)':<15} {'hip_delta(°)':<15} {'knee_delta(°)':<15} {'weight':<10}")
    print("-" * 80)
    
    for trial in sorted(trials_data, key=lambda x: x.get('trial_index', 0)):
        idx = trial.get('trial_index', 'N/A')
        features = trial.get('features', {})
        end_disp = features.get('end_disp', 0) * 1000  # mm
        hip = features.get('hip_delta', 0)
        knee = features.get('knee_delta', 0)
        weight = features.get('weight_on_leg', 0.25)
        print(f"{idx:<8} {end_disp:<15.2f} {hip:<15.2f} {knee:<15.2f} {weight:<10.3f}")

# 中央値特徴量を取得
with open('controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('session_id') == session_id and data.get('leg_id') == 'FL':
            if data.get('event_type') == 'robust_features_computed':
                print("\n" + "=" * 80)
                print("中央値特徴量:")
                print("=" * 80)
                mf = data.get('median_features', {})
                print(f"  end_disp: {mf.get('end_disp', 0)*1000:.2f} mm")
                print(f"  delta_theta: {mf.get('delta_theta_deg', 0):.2f}°")
                print(f"  shoulder_delta: {mf.get('shoulder_delta', 0):.2f}°")
                print(f"  hip_delta: {mf.get('hip_delta', 0):.2f}°")
                print(f"  knee_delta: {mf.get('knee_delta', 0):.2f}°")
                print(f"  weight_on_leg: {mf.get('weight_on_leg', 0.25):.3f}")
                
                if 'cause_distribution' in data:
                    print(f"\n原因分布:")
                    for cause, prob in data['cause_distribution'].items():
                        if prob > 0.01:
                            print(f"    {cause}: {prob:.3f}")
                break

print("\n" + "=" * 80)
