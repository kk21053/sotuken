#!/usr/bin/env python3
"""TANGLED(FOOT_VINE)が4脚すべてに正しく効いているかの簡易検証。

手順:
- set_environment.py で FOOT_VINE_{FL,FR,RL,RR} を配置
- Webotsを --no-rendering で実行
- events.jsonl の特徴量中央値から、"TANGLEDが効いている" っぽさを判定

判定方針（ルールベース）:
- TANGLEDは "足がもつれる" なので、
  - reversals が増える（往復運動が多い）
  - path_straightness が増える（回り道/引っかかりで直線比が悪化）
  が出やすい。
- 逆に、reversals≈0 かつ path_straightness≈1 なら、配置が外れている可能性が高い。

使い方:
  cd webots_new
  python3 run_vine_position_check.py
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
LOG_DIR = ROOT / "benchmarks" / "logs"

LEG_IDS = ["FL", "FR", "RL", "RR"]

# 経験則のしきい値（まずは保守的に）
REVERSALS_MIN = 2.0
STRAIGHTNESS_MIN = 1.15

# VINE設置チェックは「TANGLEDの分類が出るか」より先に、まず
# 1) その脚の運動がベースラインより有意に阻害されているか
# 2) あるいは、もつれっぽい往復/回り道が出ているか
# を見る。

# 位置が当たっていれば end_disp が縮むことが多い（TRAPほど極端でなくてもよい）。
END_DISP_ABS_MAX = 0.012
END_DISP_REL_MAX = 0.70

# 形状がうまく絡んだ場合は、角度変化もほぼ出ず end_disp が極小になることがある。
# （このスクリプトは分類器ではなく「設置が当たっているか」チェックなので、ここは許容する）
END_DISP_TINY_ABS_MAX = 0.0035
END_DISP_TINY_REL_MAX = 0.25

# 角度変化が極小でも「絡み」が成立することがあるので、厳しすぎない下限にする。
DELTA_THETA_MIN = 0.50


def _read_last_session() -> dict | None:
    if not SESSIONS.exists():
        return None
    last = None
    for line in SESSIONS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            last = line
    return json.loads(last) if last else None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def _run_one(tag: str, envs: list[str]) -> dict | None:
    print("\n" + "=" * 70)
    print(f"[vine-check] {tag}: FL={envs[0]} FR={envs[1]} RL={envs[2]} RR={envs[3]}")
    print("=" * 70)

    before_lines = _count_lines(SESSIONS)

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"vine_{tag}.log"
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(cmd_webots, check=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"[vine-check] webots finished: {time.time() - started:.1f}s")
    print(f"[vine-check] log={log_path}")

    after_lines = _count_lines(SESSIONS)
    if after_lines <= before_lines:
        print("[vine-check] warn: sessions.jsonl が増えていません（実行失敗/出力失敗の可能性）")

    return _read_last_session()


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


def _print_leg_summary(session_id: str) -> None:
    by_leg = _events_for_session(session_id)
    for leg in LEG_IDS:
        es = by_leg.get(leg, [])
        end_disp = _median_feature(es, "end_disp")
        dtheta = _median_feature(es, "delta_theta_deg")
        straight = _median_feature(es, "path_straightness")
        rev = _median_feature(es, "reversals")
        print(f"  {leg}: trials={len(es)} end_disp_med={end_disp} dtheta_med={dtheta} straight_med={straight} rev_med={rev}")


def _vine_effective(events: list[dict]) -> bool:
    straight = _median_feature(events, "path_straightness")
    rev = _median_feature(events, "reversals")
    if straight is None or rev is None:
        return False

    # 強い往復がある or 回り道 + 往復がある
    if rev >= REVERSALS_MIN:
        return True
    if straight >= STRAIGHTNESS_MIN and rev >= 1.0:
        return True
    return False


def _vine_effective_vs_baseline(events: list[dict], baseline_end_disp: float | None) -> bool:
    end_disp = _median_feature(events, "end_disp")
    dtheta = _median_feature(events, "delta_theta_deg")
    straight = _median_feature(events, "path_straightness")
    rev = _median_feature(events, "reversals")

    if end_disp is None or dtheta is None:
        return False

    # もつれっぽい挙動（従来の判定）
    if straight is not None and rev is not None:
        if rev >= REVERSALS_MIN:
            return True
        if straight >= STRAIGHTNESS_MIN and rev >= 1.0:
            return True

    # まずは「当たっているか」の判定: ベースラインより end_disp が縮む
    if baseline_end_disp is not None and baseline_end_disp > 0 and end_disp <= END_DISP_TINY_ABS_MAX:
        if (end_disp / baseline_end_disp) <= END_DISP_TINY_REL_MAX:
            return True

    if dtheta >= DELTA_THETA_MIN:
        if end_disp <= END_DISP_ABS_MAX:
            return True
        if baseline_end_disp is not None and baseline_end_disp > 0:
            if (end_disp / baseline_end_disp) <= END_DISP_REL_MAX:
                return True

    return False


def main() -> None:
    patterns: list[tuple[str, list[str], str | None]] = [
        ("ALL_NONE", ["NONE", "NONE", "NONE", "NONE"], None),
        ("ALL_TANGLED", ["TANGLED", "TANGLED", "TANGLED", "TANGLED"], None),
        ("FL_TANGLED_ONLY", ["TANGLED", "NONE", "NONE", "NONE"], "FL"),
        ("FR_TANGLED_ONLY", ["NONE", "TANGLED", "NONE", "NONE"], "FR"),
        ("RL_TANGLED_ONLY", ["NONE", "NONE", "TANGLED", "NONE"], "RL"),
        ("RR_TANGLED_ONLY", ["NONE", "NONE", "NONE", "TANGLED"], "RR"),
    ]

    failed: list[str] = []

    baseline_end_disp: dict[str, float | None] = {leg: None for leg in LEG_IDS}

    for tag, envs, check_leg in patterns:
        session = _run_one(tag, envs)
        if not session:
            print("[vine-check] ERROR: session not found")
            sys.exit(2)

        sid = session.get("session_id", "")
        print(f"[vine-check] session_id={sid}")
        _print_leg_summary(sid)

        if tag == "ALL_NONE":
            by_leg = _events_for_session(sid)
            for leg in LEG_IDS:
                baseline_end_disp[leg] = _median_feature(by_leg.get(leg, []), "end_disp")
            continue

        if check_leg:
            by_leg = _events_for_session(sid)
            ok = _vine_effective_vs_baseline(by_leg.get(check_leg, []), baseline_end_disp.get(check_leg))
            if ok:
                print(f"[vine-check] OK: {check_leg} vine looks effective")
            else:
                print(f"[vine-check] FAIL: {check_leg} vine may be misplaced")
                failed.append(check_leg)

    if failed:
        print("\n[vine-check] summary: possibly misplaced vines:", ", ".join(sorted(set(failed))))
        sys.exit(1)

    print("\n[vine-check] summary: vines look effective for all legs")


if __name__ == "__main__":
    main()
