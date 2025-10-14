#!/usr/bin/env python3
"""FL脚の各試行のp_drone分布を確認"""

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

# FL脚の各試行のp_droneを取得
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

print(f"\nFL脚の各試行のp_drone分布:")
print("-"*80)
for i, trial in enumerate(fl_trials, 1):
    p_drone = trial.get('p_drone', {})
    print(f"\nTrial {i}:")
    print(f"  delta_theta: {trial.get('delta_theta_deg', 0):.2f}°")
    print(f"  end_disp: {trial.get('end_disp', 0)*1000:.1f}mm")
    print(f"  p_drone:")
    for cause, prob in sorted(p_drone.items(), key=lambda x: x[1], reverse=True):
        if prob > 0.05:  # 5%以上のみ表示
            print(f"    {cause:12s}: {prob:.3f}")

# 平均を計算
print("\n" + "="*80)
print("6試行の平均:")
avg_p_drone = {}
for cause in ['NONE', 'BURIED', 'TRAPPED', 'TANGLED', 'MALFUNCTION', 'FALLEN']:
    avg_p_drone[cause] = sum(trial.get('p_drone', {}).get(cause, 0) for trial in fl_trials) / len(fl_trials)

for cause, prob in sorted(avg_p_drone.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cause:12s}: {prob:.3f}")

print("\n最終p_drone (セッションログから):")
final_p_drone = last_session['legs']['FL']['p_drone']
for cause, prob in sorted(final_p_drone.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cause:12s}: {prob:.3f}")
