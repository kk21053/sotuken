#!/usr/bin/env python3
"""パターン3(FL=TRAPPED)の詳細分析"""
import json

# 最新のパターン3セッションを探す
with open('controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl', 'r') as f:
    sessions = [json.loads(line) for line in f]

# パターン3を探す
pattern3_session = None
for session in reversed(sessions):
    if session['legs']['FL'].get('expected_cause') == 'TRAPPED':
        pattern3_session = session
        break

if not pattern3_session:
    print("パターン3のセッションが見つかりません")
    exit(1)

print("=" * 80)
print(f"パターン3分析: {pattern3_session['session_id']}")
print("=" * 80)

for leg_id in ['FL', 'FR', 'RL', 'RR']:
    leg = pattern3_session['legs'][leg_id]
    print(f"\n{leg_id}脚:")
    print(f"  判定結果: {leg.get('cause_final', 'N/A')}")
    print(f"  期待値: {leg.get('expected_cause', 'N/A')}")
    print(f"  動作状態: {leg.get('movement_result', 'N/A')}")
    print(f"  Spot CAN: {leg.get('spot_can', 0):.3f}")
    print(f"  Drone CAN: {leg.get('drone_can', 0):.3f}")
    print(f"  転倒: {leg.get('fallen', False)}, 確率: {leg.get('fallen_probability', 0):.3f}")
    
    if 'p_drone' in leg:
        print(f"  Droneの確率分布:")
        for cause, prob in leg['p_drone'].items():
            if prob > 0.05:
                print(f"    {cause}: {prob:.3f}")

print("\n" + "=" * 80)

# RoboPose特徴量を探す
print("\n特徴量データを探しています...")
with open('controllers/drone_circular_controller/logs/robopose_features.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('session_id') == pattern3_session['session_id']:
            if data.get('leg_id') in ['FL', 'RL']:
                if 'median_features' in data or data.get('event_type') == 'robust_features_computed':
                    print(f"\n{data['leg_id']}脚の中央値特徴量:")
                    features = data.get('median_features', data.get('features', {}))
                    print(f"  end_disp: {features.get('end_disp', 0)*1000:.2f} mm")
                    print(f"  delta_theta: {features.get('delta_theta_deg', 0):.2f}°")
                    print(f"  shoulder_delta: {features.get('shoulder_delta', 0):.2f}°")
                    print(f"  hip_delta: {features.get('hip_delta', 0):.2f}°")
                    print(f"  knee_delta: {features.get('knee_delta', 0):.2f}°")
                    print(f"  weight_on_leg: {features.get('weight_on_leg', 0.25):.3f}")
