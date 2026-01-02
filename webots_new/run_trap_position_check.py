#!/usr/bin/env python3
"""TRAPPED(FOOT_TRAP)が4脚すべてに正しく効いているかの簡易検証。

やること:
- set_environment.py でワールド上の FOOT_TRAP_{FL,FR,RL,RR} を配置
- Webotsを --no-rendering で1パターンずつ実行
- 生成された最新セッションの events.jsonl から特徴量の中央値を集計
- "TRAPPEDが効いている" を、関節角が十分動く(=delta_thetaが大)のに末端が動かない(end_dispが小)で判定

使い方:
  cd webots_new
  python3 run_trap_position_check.py

注意:
- Webotsの実行環境が必要です。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
WORLD = ROOT / "worlds" / "sotuken_world.wbt"
SET_ENV = ROOT / "set_environment.py"
SESSIONS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_sessions.jsonl"
EVENTS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_events.jsonl"

LEG_IDS = ["FL", "FR", "RL", "RR"]

# drone_observer.py の判定と整合する目安（"設置が効いている" の物理的判定用）
TRAPPED_ANGLE_MIN_DEG = 1.25
TRAPPED_DISP_MAX_M = 0.012


def _read_last_session() -> dict | None:
    if not SESSIONS.exists():
        return None
    last = None
    for line in SESSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    return json.loads(last) if last else None


def _run_one(tag: str, envs: list[str]) -> dict | None:
    print("\n" + "=" * 70)
    print(f"[trap-check] {tag}: FL={envs[0]} FR={envs[1]} RL={envs[2]} RR={envs[3]}")
    print("=" * 70)

    subprocess.run([sys.executable, str(SET_ENV)] + envs, check=True)

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
    subprocess.run(cmd_webots, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[trap-check] webots finished: {time.time() - started:.1f}s")

    return _read_last_session()


def _median_feature(events: list[dict], key: str) -> float | None:
    vals: list[float] = []
    for e in events:
        f = (e.get("features") or {})
        v = f.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return float(median(vals))


def _events_for_session(session_id: str) -> dict[str, list[dict]]:
    by_leg: dict[str, list[dict]] = {leg: [] for leg in LEG_IDS}
    if not EVENTS.exists():
        return by_leg

    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        if e.get("session_id") != session_id:
            continue
        leg = e.get("leg_id")
        if leg in by_leg and e.get("trial_ok", True):
            by_leg[leg].append(e)

    return by_leg


def _print_leg_summary(session_id: str) -> None:
    by_leg = _events_for_session(session_id)
    for leg in LEG_IDS:
        es = by_leg.get(leg, [])
        end_disp = _median_feature(es, "end_disp")
        dtheta = _median_feature(es, "delta_theta_deg")
        print(f"  {leg}: trials={len(es)} end_disp_med={end_disp} delta_theta_med={dtheta}")


def _trap_effective(events: list[dict]) -> bool:
    end_disp = _median_feature(events, "end_disp")
    dtheta = _median_feature(events, "delta_theta_deg")
    if end_disp is None or dtheta is None:
        return False
    return (dtheta >= TRAPPED_ANGLE_MIN_DEG) and (end_disp <= TRAPPED_DISP_MAX_M)


def main() -> None:
    patterns: list[tuple[str, list[str], str | None]] = [
        ("ALL_TRAPPED", ["TRAPPED", "TRAPPED", "TRAPPED", "TRAPPED"], None),
        ("FL_TRAPPED_ONLY", ["TRAPPED", "NONE", "NONE", "NONE"], "FL"),
        ("FR_TRAPPED_ONLY", ["NONE", "TRAPPED", "NONE", "NONE"], "FR"),
        ("RL_TRAPPED_ONLY", ["NONE", "NONE", "TRAPPED", "NONE"], "RL"),
        ("RR_TRAPPED_ONLY", ["NONE", "NONE", "NONE", "TRAPPED"], "RR"),
    ]

    failed: list[str] = []

    for tag, envs, check_leg in patterns:
        session = _run_one(tag, envs)
        if not session:
            print("[trap-check] ERROR: session not found")
            sys.exit(2)

        sid = session.get("session_id", "")
        print(f"[trap-check] session_id={sid}")
        _print_leg_summary(sid)

        if check_leg:
            by_leg = _events_for_session(sid)
            ok = _trap_effective(by_leg.get(check_leg, []))
            if ok:
                print(f"[trap-check] OK: {check_leg} trap looks effective")
            else:
                print(f"[trap-check] FAIL: {check_leg} trap may be misplaced (angles move but foot still moves)")
                failed.append(check_leg)

    if failed:
        print("\n[trap-check] summary: possibly misplaced traps:", ", ".join(sorted(set(failed))))
        sys.exit(1)

    print("\n[trap-check] summary: traps look effective for all legs")


if __name__ == "__main__":
    main()
