#!/usr/bin/env python3
"""ランダム10パターン検証

目的:
- 各脚(FL/FR/RL/RR)にランダムな環境(NONE/BURIED/TRAPPED/TANGLED/MALFUNCTION)を割り当てて
    10パターン実行し、`cause_final` の正解率を確認する。

使い方:
  cd webots_new
    python3 run_benchmark_random_10.py

オプション:
  --seed 123   : 乱数シード固定
  --count 10   : 実行数（デフォルト10）

注意:
- sessions.jsonl には追記される（上書きしない）。
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


# 出力が詰まって止まって見えるのを避ける
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
WORLD = ROOT / "worlds" / "sotuken_world.wbt"
SET_ENV = ROOT / "set_environment.py"
SESSIONS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_sessions.jsonl"
LOG_DIR = ROOT / "benchmarks" / "logs"

VENV_BIN = ROOT / ".venv" / "bin"
VENV_PY = VENV_BIN / "python3"

LEG_IDS = ["FL", "FR", "RL", "RR"]
ENV_CHOICES = ["NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION"]


def _ensure_local_venv() -> None:
    """PEP668環境でも動くように、webots_new/.venv を優先して使う。"""
    vpy = ROOT / ".venv" / "bin" / "python"
    if not vpy.exists():
        return
    # すでにvenv内なら何もしない
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return
    os.execv(str(vpy), [str(vpy)] + sys.argv)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def read_last_session() -> dict | None:
    if not SESSIONS.exists():
        return None
    last = None
    for line in SESSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    if not last:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def score_session(session: dict) -> tuple[int, int]:
    if not session:
        return 0, 0
    legs = session.get("legs", {}) or {}
    ok = 0
    total = 0
    for leg_id in LEG_IDS:
        leg = legs.get(leg_id, {}) or {}
        exp = str(leg.get("expected_cause", "NONE"))
        got = str(leg.get("cause_final", ""))
        total += 1
        if got == exp:
            ok += 1
    return ok, total


def run_one(name: str, envs: list[str]) -> dict | None:
    print("\n" + "=" * 70)
    print(f"[rand10] {name}: FL={envs[0]} FR={envs[1]} RL={envs[2]} RR={envs[3]}")
    print("=" * 70)

    before_lines = count_lines(SESSIONS)

    env = os.environ.copy()
    if VENV_BIN.exists():
        env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"

    python = str(VENV_PY) if VENV_PY.exists() else sys.executable

    # 1) 環境設定
    cmd_env = [python, str(SET_ENV)] + envs
    subprocess.run(cmd_env, check=True, env=env)

    # 2) Webots実行
    cmd_webots = [
        "webots",
        "--batch",
        "--no-rendering",
        "--mode=fast",
        "--stdout",
        "--stderr",
        str(WORLD),
    ]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        # Webots出力をファイルとコンソールの両方へ流す（Qwen進捗を見えるように）
        p = subprocess.Popen(
            cmd_webots,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            f.write(line)
            sys.stdout.write(line)
        ret = p.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd_webots)

    elapsed = time.time() - started
    print(f"[rand10] webots finished: {elapsed:.1f}s log={log_path}")

    after_lines = count_lines(SESSIONS)
    if after_lines <= before_lines:
        print("[rand10] warn: sessions.jsonl が増えていません")
        return read_last_session()

    return read_last_session()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=10)
    _ensure_local_venv()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(int(args.seed))
        print(f"[rand10] seed={args.seed}")

    results = []
    ok_sum = 0
    total_sum = 0

    for i in range(int(args.count)):
        envs = [random.choice(ENV_CHOICES) for _ in range(4)]
        name = f"R{i+1:02d}_" + "_".join(envs)
        session = run_one(name, envs)
        ok, total = score_session(session or {})
        sid = (session or {}).get("session_id")
        results.append((name, ok, total, sid))
        ok_sum += ok
        total_sum += total
        print(f"[rand10] score cause_final {ok}/{total} session={sid}")

    print("\n" + "=" * 70)
    print("[rand10] summary")
    print("=" * 70)
    for name, ok, total, sid in results:
        print(f"  {name}: cause_final {ok}/{total} ({sid})")
    if total_sum > 0:
        print(f"\n[rand10] total: {ok_sum}/{total_sum} ({(100.0*ok_sum/total_sum):.1f}%)")
    else:
        print("\n[rand10] total: 0/0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
