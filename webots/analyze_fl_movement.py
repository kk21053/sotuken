#!/usr/bin/env python3
"""FL脚の動きを分析するスクリプト"""

import json
from pathlib import Path

LOG_DIR = Path(__file__).parent / "controllers" / "drone_circular_controller" / "logs"
EVENT_LOG = LOG_DIR / "leg_diagnostics_events.jsonl"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"

# 最新セッションIDを取得
with SESSION_LOG.open('r') as f:
    lines = f.readlines()
    if not lines:
        print("セッションログが空です")
        exit(1)
    last_session = json.loads(lines[-1])
    session_id = last_session['session_id']

print(f"セッションID: {session_id}")
print("="*80)

# FL脚の各試行データを取得
fl_trials = []
with EVENT_LOG.open('r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('session_id') == session_id and data.get('leg_id') == 'FL':
                fl_trials.append(data)
        except:
            continue

if not fl_trials:
    print("FL脚のデータが見つかりません")
    exit(1)

print(f"\nFL脚の試行データ ({len(fl_trials)}試行):")
print("-"*80)
for i, trial in enumerate(fl_trials, 1):
    delta_theta = trial.get('delta_theta_deg', 0)
    end_disp = trial.get('end_disp', 0) * 1000  # mmに変換
    path_length = trial.get('path_length', 0) * 1000
    reversals = trial.get('reversals', 0)
    
    print(f"Trial {i}:")
    print(f"  関節角度変化: {delta_theta:.2f}°")
    print(f"  末端変位: {end_disp:.1f}mm")
    print(f"  経路長: {path_length:.1f}mm")
    print(f"  反転回数: {reversals}")

# 最終判定結果
print("\n" + "="*80)
fl_data = last_session['legs']['FL']
print("最終診断結果:")
print(f"  spot_can: {fl_data['spot_can']:.3f}")
print(f"  drone_can: {fl_data['drone_can']:.3f}")
print(f"  p_drone: {fl_data['p_drone']}")
print(f"  cause_final: {fl_data['cause_final']}")
print(f"  expected_cause: {fl_data['expected_cause']}")
