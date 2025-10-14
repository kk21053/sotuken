#!/usr/bin/env python3
"""FR脚のデータを確認するスクリプト"""
import json

# セッションIDを指定
session_ids = ["drone_diagnosis_20251014_145418", "drone_diagnosis_20251014_145449"]

print("=" * 80)
print("FR脚のデータ確認")
print("=" * 80)

# イベントログから中央値データを取得
with open('controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('session_id') in session_ids and data.get('leg_id') == 'FR':
                if 'median_features' in data:
                    print(f"\nセッション: {data['session_id']}")
                    print(f"イベント: {data.get('event_type')}")
                    mf = data['median_features']
                    print(f"  end_disp: {mf.get('end_disp', 'N/A'):.4f} m")
                    print(f"  delta_theta: {mf.get('delta_theta', 'N/A'):.2f}°")
                    print(f"  shoulder_delta: {mf.get('shoulder_delta', 'N/A'):.2f}°")
                    print(f"  hip_delta: {mf.get('hip_delta', 'N/A'):.2f}°")
                    print(f"  knee_delta: {mf.get('knee_delta', 'N/A'):.2f}°")
                    print(f"  weight_on_leg: {mf.get('weight_on_leg', 'N/A'):.3f}")
                    if 'cause_distribution' in data:
                        print(f"  原因分布: {data['cause_distribution']}")
                    if 'estimated_cause' in data:
                        print(f"  推定原因: {data['estimated_cause']}")
        except:
            pass

print("\n" + "=" * 80)
