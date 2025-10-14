#!/usr/bin/env python3
"""drone_can計算の詳細確認"""

import json
import math
from pathlib import Path

LOG_DIR = Path(__file__).parent / "controllers" / "drone_circular_controller" / "logs"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"

# 最新セッション取得
with SESSION_LOG.open('r') as f:
    last_session = json.loads(f.readlines()[-1])

fl_data = last_session['legs']['FL']
print(f"FL脚のdrone_can計算:")
print(f"  drone_can_raw: {fl_data.get('drone_can_raw', 'N/A')}")
print(f"  drone_can: {fl_data['drone_can']}")

# シグモイド計算を再現
steepness = 15.0
threshold = 0.50
drone_can_raw = fl_data.get('drone_can_raw', 0.0)
x = steepness * (drone_can_raw - threshold)

print(f"\nシグモイド計算:")
print(f"  steepness: {steepness}")
print(f"  threshold: {threshold}")
print(f"  x = {steepness} * ({drone_can_raw} - {threshold}) = {x}")

if x > 20:
    sigmoid = 1.0
elif x < -20:
    sigmoid = 0.0
else:
    sigmoid = 1.0 / (1.0 + math.exp(-x))

print(f"  sigmoid(x) = {sigmoid}")
print(f"\n予測drone_can: {sigmoid}")
print(f"実際のdrone_can: {fl_data['drone_can']}")

# 逆算: drone_can = 0.505の場合のdrone_can_rawを求める
target_drone_can = fl_data['drone_can']
# sigmoid(x) = 0.505 => x ≈ 0.02
# 0.02 = 15 * (raw - 0.5) => raw = 0.5 + 0.02/15 ≈ 0.501
if 0.01 < target_drone_can < 0.99:
    x_reverse = math.log(target_drone_can / (1 - target_drone_can))
    raw_reverse = threshold + x_reverse / steepness
    print(f"\n逆算:")
    print(f"  drone_can={target_drone_can}の場合、x={x_reverse:.4f}")
    print(f"  drone_can_raw={raw_reverse:.4f}のはず")
