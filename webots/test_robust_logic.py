#!/usr/bin/env python3
"""ロバスト統計ロジックが正しく呼ばれるかテスト"""

import json

# 最新セッションのRR脚データを読み込む
with open('./controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl') as f:
    lines = f.readlines()

rr_trials = []
for line in reversed(lines):
    data = json.loads(line)
    if data.get('leg_id') == 'RR':
        rr_trials.append(data)
        if len(rr_trials) >= 6:
            break

rr_trials.reverse()

print(f"RR脚の最新6試行:")
for t in rr_trials:
    print(f"  Trial {t['trial_index']}: drone_can={t['drone_can']:.3f}, cause={t['cause_final']}")

# 診断完了時にロバスト統計が呼ばれているか
print(f"\n試行数: {len(rr_trials)}")
print(f"条件 count >= 6: {len(rr_trials) >= 6}")
