#!/usr/bin/env python3
"""FR脚の動きを分析"""

import json
from pathlib import Path

LOG_DIR = Path(__file__).parent / "controllers" / "drone_circular_controller" / "logs"
EVENT_LOG = LOG_DIR / "leg_diagnostics_events.jsonl"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"

# 最新セッションIDを取得
with SESSION_LOG.open('r') as f:
    lines = f.readlines()
    last_session = json.loads(lines[-1])
    session_id = last_session['session_id']

print(f"セッションID: {session_id}")
print("="*80)

# FR脚の各試行データを取得
fr_trials = []
with EVENT_LOG.open('r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get('session_id') == session_id and data.get('leg_id') == 'FR':
                fr_trials.append(data)
        except:
            continue

if not fr_trials:
    print("FR脚のデータが見つかりません")
    exit(1)

print(f"\nFR脚の試行データ ({len(fr_trials)}試行):")
print("-"*80)
for i, trial in enumerate(fr_trials, 1):
    delta_theta = trial.get('delta_theta_deg', 0)
    end_disp = trial.get('end_disp', 0) * 1000  # mmに変換
    path_length = trial.get('path_length', 0) * 1000
    reversals = trial.get('reversals', 0)
    
    print(f"Trial {i}:")
    print(f"  関節角度変化: {delta_theta:.2f}°")
    print(f"  末端変位: {end_disp:.1f}mm")
    print(f"  経路長: {path_length:.1f}mm")
    print(f"  反転回数: {reversals}")
    
    # p_droneを表示
    p_drone = trial.get('p_drone', {})
    max_cause = max(p_drone.items(), key=lambda x: x[1])
    print(f"  → {max_cause[0]}: {max_cause[1]:.3f}")

# 最終判定結果
print("\n" + "="*80)
fr_data = last_session['legs']['FR']
print("最終診断結果:")
print(f"  spot_can: {fr_data['spot_can']:.3f}")
print(f"  drone_can: {fr_data['drone_can']:.3f}")
print(f"  cause_final: {fr_data['cause_final']}")
print(f"  expected_cause: {fr_data['expected_cause']}")
print(f"\n  p_drone:")
for cause, prob in sorted(fr_data['p_drone'].items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"    {cause:12s}: {prob:.3f}")
