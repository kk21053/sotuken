#!/usr/bin/env python3
"""scenario.iniの読み込みをテスト"""

from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "config" / "scenario.ini"

def load_config():
    config = {}
    sand = {}
    trap = {}
    vine = {}
    if not CONFIG_PATH.exists():
        return config, sand, trap, vine

    with CONFIG_PATH.open() as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):  # セクションヘッダーをスキップ
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if key.startswith('sand.'):
                sand[key.split('.', 1)[1]] = value
            elif key.startswith('trap.'):
                trap[key.split('.', 1)[1]] = value
            elif key.startswith('vine.'):
                vine[key.split('.', 1)[1]] = value
            else:
                config[key] = value
    return config, sand, trap, vine

config, sand, trap, vine = load_config()

print("=" * 70)
print("scenario.ini の読み込み結果")
print("=" * 70)
print(f"\nscenario = {config.get('scenario', 'NOT FOUND')}")
print(f"fl_environment = {config.get('fl_environment', 'NOT FOUND')}")
print(f"fr_environment = {config.get('fr_environment', 'NOT FOUND')}")
print(f"rl_environment = {config.get('rl_environment', 'NOT FOUND')}")
print(f"rr_environment = {config.get('rr_environment', 'NOT FOUND')}")

print("\n環境判定:")
fl_env = config.get("fl_environment", "NONE").upper()
fr_env = config.get("fr_environment", "NONE").upper()
rl_env = config.get("rl_environment", "NONE").upper()
rr_env = config.get("rr_environment", "NONE").upper()
all_envs = {fl_env, fr_env, rl_env, rr_env}

print(f"  all_envs = {all_envs}")
print(f"  'BURIED' in all_envs = {'BURIED' in all_envs}")
print(f"  scenario == 'sand_burial' = {config.get('scenario', 'none') == 'sand_burial'}")

print("\nBURIED環境が適用されるか？")
scenario = config.get("scenario", "none")
if "BURIED" in all_envs:
    print("  ✅ YES: BURIED in all_envs")
elif scenario == "sand_burial":
    print("  ✅ YES: scenario == 'sand_burial'")
else:
    print("  ❌ NO: BURIED環境は適用されない")

print("\nTRAPPED環境が適用されるか？")
if "TRAPPED" in all_envs:
    print("  ✅ YES: TRAPPED in all_envs")
elif scenario == "foot_trap":
    print("  ✅ YES: scenario == 'foot_trap'")
else:
    print("  ❌ NO: TRAPPED環境は適用されない")

print("=" * 70)
