#!/usr/bin/env python3
"""全脚に MALFUNCTION を適用し、Spotが指示しても関節が動かないことを確認する。

ユーザー定義:
- MALFUNCTION = Spotが関節を動かそうと指示(例:+4deg)しても、センサ出力がほぼ変化しない状態

このスクリプトがやること:
1) set_environment.py で FL/FR/RL/RR 全てを MALFUNCTION に設定
2) Webotsを --batch --no-rendering で1回だけ実行
3) Spotログの "MALFUNCTION_CHECK" 行から、cmd_delta_deg と actual_delta_deg を集計
4) actual_delta_deg が小さい(しきい値以内)ことを確認

使い方:
  cd webots_new
  python3 verify_malfunction_all.py

終了コード:
  0: 合格
  1: 不合格（動いている / ログ不足）
"""

import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SET_ENV = ROOT / "set_environment.py"
WORLD = ROOT / "worlds" / "sotuken_world.wbt"
LOG_DIR = ROOT / "benchmarks" / "logs"


LINE_RE = re.compile(
    r"\[spot_new\] MALFUNCTION_CHECK leg=(FL|FR|RL|RR) trial=(\d+) joint=([^\s]+) "
    r"cmd_delta_deg=([-0-9.]+) actual_delta_deg=([-0-9.]+|NaN)"
)


def run_webots_and_capture(log_path: Path) -> None:
    # 1) 全脚MALFUNCTIONを適用
    python = sys.executable
    subprocess.run(
        [python, str(SET_ENV), "MALFUNCTION", "MALFUNCTION", "MALFUNCTION", "MALFUNCTION"],
        check=True,
    )

    # 2) Webotsを1回実行（標準出力/標準エラーをログへ）
    cmd = [
        "webots",
        "--batch",
        "--no-rendering",
        "--mode=fast",
        "--stdout",
        "--stderr",
        str(WORLD),
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def parse_checks(log_text: str) -> list[dict]:
    out = []
    for line in log_text.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        leg_id = m.group(1)
        trial = int(m.group(2))
        joint = m.group(3)
        cmd = float(m.group(4))
        actual_s = m.group(5)
        if actual_s == "NaN":
            actual = None
        else:
            actual = float(actual_s)
        out.append(
            {
                "leg": leg_id,
                "trial": trial,
                "joint": joint,
                "cmd_delta_deg": cmd,
                "actual_delta_deg": actual,
            }
        )
    return out


def main() -> int:
    # しきい値: 実測Δがこの値より大きければ「動いている」と判定
    # 完全に0にはならない可能性があるので少し余裕を持たせる。
    tol_deg = float(os.getenv("MALFUNCTION_TOL_DEG", "0.5"))

    log_path = LOG_DIR / "verify_malfunction_all.log"
    run_webots_and_capture(log_path)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    checks = parse_checks(text)

    # 期待: 4脚×6試行 = 24行
    expected_lines = 24
    if len(checks) < expected_lines:
        print(f"[verify] FAIL: MALFUNCTION_CHECK lines are missing ({len(checks)}/{expected_lines})")
        print(f"[verify] log: {log_path}")
        return 1

    failures = []
    for c in checks:
        actual = c["actual_delta_deg"]
        if actual is None:
            failures.append((c, "no sensor samples"))
            continue
        if abs(actual) > tol_deg:
            failures.append((c, f"moved (|actual|={abs(actual):.3f} > {tol_deg:.3f})"))

    if failures:
        print(f"[verify] FAIL: {len(failures)} trials exceeded tolerance (tol_deg={tol_deg:.3f})")
        for c, reason in failures[:10]:
            print(
                f"  leg={c['leg']} trial={c['trial']} joint={c['joint']} "
                f"cmd_delta_deg={c['cmd_delta_deg']:.3f} actual_delta_deg={c['actual_delta_deg']} -> {reason}"
            )
        print(f"[verify] log: {log_path}")
        return 1

    print(f"[verify] PASS: MALFUNCTION seems correct (tol_deg={tol_deg:.3f})")
    print(f"[verify] log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
