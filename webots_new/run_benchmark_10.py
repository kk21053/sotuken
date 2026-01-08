#!/usr/bin/env python3
"""10パターン検証スクリプト（描画なしでWebotsを実行）

目的:
- 各脚(FL/FR/RL/RR)に異なる環境(NONE/BURIED/TRAPPED/TANGLED/MALFUNCTION)を設定して
  合計10パターンの検証を自動で回す。
- Webotsは --no-rendering で起動し、動画出力は行わずに実行のみ行う。
- 各パターンの結果(leg_diagnostics_sessions.jsonl)から正解率を集計する。

使い方:
  cd webots_new
  python3 run_benchmark_10.py

注意:
- Webotsが起動できる環境（DISPLAY等）が必要です。
- 1パターンごとにWebotsを起動して終了します。
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


# パイプ（| tail 等）で実行すると標準出力がバッファされて「止まって見える」ので、行バッファにする
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
WORLD = ROOT / "worlds" / "sotuken_world.wbt"
SET_ENV = ROOT / "set_environment"
SESSIONS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_sessions.jsonl"
BENCH_DIR = ROOT / "benchmarks"
LOG_DIR = BENCH_DIR / "logs"

VENV_BIN = ROOT / ".venv" / "bin"
VENV_PY = VENV_BIN / "python3"

LEG_IDS = ["FL", "FR", "RL", "RR"]


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_last_session() -> dict | None:
    if not SESSIONS.exists():
        return None
    try:
        last = None
        for line in SESSIONS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            last = line
        if not last:
            return None
        return json.loads(last)
    except Exception:
        return None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def run_one(pattern_name: str, envs: list[str]) -> dict | None:
    print("\n" + "=" * 70)
    print(f"[benchmark] {pattern_name}: FL={envs[0]} FR={envs[1]} RL={envs[2]} RR={envs[3]}")
    print("=" * 70)

    before_lines = count_lines(SESSIONS)

    env = os.environ.copy()
    if VENV_BIN.exists():
        env["PATH"] = f"{VENV_BIN}:{env.get('PATH', '')}"

    # 1) 環境を設定
    python = str(VENV_PY) if VENV_PY.exists() else sys.executable
    cmd_env = [python, str(SET_ENV)] + envs
    subprocess.run(cmd_env, check=True, env=env)

    # 2) Webotsを描画なしで実行
    cmd_webots = [
        "webots",
        "--batch",
        "--no-rendering",
        "--mode=fast",
        "--stdout",
        "--stderr",
        str(WORLD),
    ]

    started = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{pattern_name}.log"
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd_webots, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    print(f"[benchmark] webots finished: {elapsed:.1f}s log={log_path}")
    # 3) 追加された最新セッションを読む
    after_lines = count_lines(SESSIONS)
    if after_lines <= before_lines:
        print("[benchmark] warn: sessions.jsonl が増えていません")
        return read_last_session()

    return read_last_session()


def score_session(session: dict, expected: dict[str, str]) -> tuple[int, int]:
    if not session:
        return 0, len(LEG_IDS)

    legs = session.get("legs", {})
    ok = 0
    total = 0

    for leg_id in LEG_IDS:
        total += 1
        exp = expected.get(leg_id, "NONE")
        got = (legs.get(leg_id, {}) or {}).get("cause_final", "")
        if got == exp:
            ok += 1

    return ok, total


def main() -> None:
    patterns = [
        ("P01_all_NONE", ["NONE", "NONE", "NONE", "NONE"]),
        ("P02_FL_BURIED", ["BURIED", "NONE", "NONE", "NONE"]),
        ("P03_FR_TRAPPED", ["NONE", "TRAPPED", "NONE", "NONE"]),
        ("P04_RL_TANGLED", ["NONE", "NONE", "TANGLED", "NONE"]),
        ("P05_RR_BURIED", ["NONE", "NONE", "NONE", "BURIED"]),
        ("P06_mix1", ["TRAPPED", "BURIED", "NONE", "TANGLED"]),
        ("P07_mix2", ["BURIED", "TANGLED", "TRAPPED", "NONE"]),
        ("P08_TRAPPED_FL_RL", ["TRAPPED", "NONE", "TRAPPED", "NONE"]),
        ("P09_all_BURIED", ["BURIED", "BURIED", "BURIED", "BURIED"]),
        ("P10_mix3", ["TANGLED", "TRAPPED", "BURIED", "NONE"]),
    ]

    def _parse_only_args(argv: list[str]) -> list[str]:
        """`--only P06` / `--only=P06_mix1,P10_mix3` の両方を受ける。"""
        only: list[str] = []
        i = 0
        while i < len(argv):
            a = argv[i]
            if a == "--only" and i + 1 < len(argv):
                raw = (argv[i + 1] or "").strip()
                if raw:
                    only.extend([x.strip() for x in raw.split(",") if x.strip()])
                i += 2
                continue
            if a.startswith("--only="):
                raw = a.split("=", 1)[1].strip()
                if raw:
                    only.extend([x.strip() for x in raw.split(",") if x.strip()])
                i += 1
                continue
            i += 1
        return only

    def _match_only(pattern_name: str, selector: str) -> bool:
        s = (selector or "").strip()
        if not s:
            return False
        # 完全一致
        if pattern_name == s:
            return True
        # `P06` のようなプレフィックス指定（P06_mix1 を含む）
        if len(s) == 3 and s.startswith("P") and s[1:].isdigit():
            return pattern_name.startswith(s + "_")
        # `P06_mix1` の部分一致（誤爆しにくい最小限）
        return pattern_name.startswith(s)

    only_selectors = _parse_only_args(sys.argv[1:])
    if only_selectors:
        patterns = [(n, e) for (n, e) in patterns if any(_match_only(n, s) for s in only_selectors)]
        print(f"[benchmark] only={only_selectors}")

    total_ok = 0
    total = 0

    results = []

    for name, envs in patterns:
        expected = {"FL": envs[0], "FR": envs[1], "RL": envs[2], "RR": envs[3]}
        session = run_one(name, envs)
        ok, n = score_session(session or {}, expected)
        total_ok += ok
        total += n

        sid = (session or {}).get("session_id", "")
        results.append((name, sid, ok, n))
        print(f"[benchmark] score cause_final {ok}/{n} session={sid}")

    print("\n" + "=" * 70)
    print("[benchmark] summary")
    print("=" * 70)
    for name, sid, ok, n in results:
        print(f"  {name}: cause_final {ok}/{n} ({sid})")
    if total > 0:
        acc = 100.0 * total_ok / total
        print(f"\n[benchmark] total: {total_ok}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
