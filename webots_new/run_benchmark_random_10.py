#!/usr/bin/env python3
"""ランダム10パターン検証（VLMの精度確認用）

目的:
- 各脚(FL/FR/RL/RR)にランダムな環境(NONE/BURIED/TRAPPED/TANGLED/MALFUNCTION)を割り当てて
  10パターン実行し、VLMの `vlm_pred` 正解率を確認する。

使い方:
  cd webots_new
  VLM_ENABLE=1 python3 run_benchmark_random_10.py

オプション:
  --seed 123   : 乱数シード固定
  --count 10   : 実行数（デフォルト10）

注意:
- VLMは重いので、1プロセス内でモデルを使い回す。
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


def maybe_run_vlm_postprocess() -> None:
    """VLM_ENABLE=1 のとき、最新セッションに VLM結果/融合結果を追記する（モデル使い回し）。"""
    if os.getenv("VLM_ENABLE", "0").strip() != "1":
        return

    try:
        from controllers.diagnostics_pipeline.models import LegState, SessionState
        from controllers.diagnostics_pipeline.vlm_client import VLMAnalyzer
    except Exception:
        print("[rand10] VLM enabled but deps missing -> skip")
        return

    session_dict = read_last_session()
    if not session_dict:
        return

    if bool(session_dict.get("vlm_completed", False)):
        return

    sid = session_dict.get("session_id", "")
    print(f"[rand10] vlm_postprocess start session={sid}")

    session = SessionState(session_id=str(session_dict.get("session_id", "")))
    img = session_dict.get("image_path")
    if isinstance(img, str) and img:
        p = Path(img)
        if p.exists():
            session.image_path = str(p)
        else:
            p2 = ROOT / p
            if p2.exists():
                session.image_path = str(p2)
            else:
                p3 = ROOT / "controllers" / "drone_circular_controller" / "logs" / p
                session.image_path = str(p3)
    else:
        session.image_path = None

    session.fallen = bool(session_dict.get("fallen", False))
    session.fallen_probability = float(session_dict.get("fallen_probability", 0.0) or 0.0)

    legs = session_dict.get("legs", {}) or {}
    for leg_id in LEG_IDS:
        legd = (legs.get(leg_id, {}) or {})
        leg = LegState(leg_id=leg_id)
        leg.spot_can = float(legd.get("spot_can", 0.5) or 0.5)
        leg.drone_can = float(legd.get("drone_can", 0.5) or 0.5)
        try:
            leg.p_drone = dict(legd.get("p_drone", {}) or {})
        except Exception:
            pass
        try:
            leg.p_llm = dict(legd.get("p_llm", {}) or {})
        except Exception:
            pass
        leg.movement_result = str(legd.get("movement_result", "一部動く"))
        leg.cause_final = str(legd.get("cause_final", "NONE"))
        leg.p_can = float(legd.get("p_can", 0.0) or 0.0)
        leg.expected_cause = str(legd.get("expected_cause", "NONE"))
        leg.fallen = bool(legd.get("fallen", False))
        leg.fallen_probability = float(legd.get("fallen_probability", 0.0) or 0.0)
        session.legs[leg_id] = leg

    global _VLM
    if "_VLM" not in globals() or globals().get("_VLM") is None:
        globals()["_VLM"] = VLMAnalyzer()
    vlm = globals()["_VLM"]

    started = time.time()

    # 仕様.txtのルール判定（①〜④）
    try:
        from controllers.diagnostics_pipeline.rule_fusion import fuse_rule_and_vlm, rule_based_decision
    except Exception:
        print("[rand10] rule_fusion missing -> skip")
        return

    to_vlm: list[str] = []
    for leg_id in LEG_IDS:
        leg = session.legs.get(leg_id)
        if leg is None:
            continue
        prob_dist = getattr(leg, "p_llm", None) or getattr(leg, "p_drone", {}) or {}
        movement, cause_rule, p_rule = rule_based_decision(leg.spot_can, leg.drone_can, prob_dist)
        leg.movement_result = movement
        leg.cause_rule = cause_rule
        leg.p_rule = p_rule
        if cause_rule != "NONE":
            to_vlm.append(leg_id)

    # ルールがNONE以外の脚だけVLM
    if to_vlm:
        vlm.infer_session(session, leg_ids=to_vlm)

    # 融合してcause_finalを最終確定として更新
    for leg_id in LEG_IDS:
        leg = session.legs.get(leg_id)
        if leg is None:
            continue
        cause_rule = str(getattr(leg, "cause_rule", None) or "NONE")
        if cause_rule == "NONE":
            leg.cause_fused = "NONE"
            leg.fused_probs = {"NONE": 1.0, "BURIED": 0.0, "TRAPPED": 0.0, "TANGLED": 0.0, "MALFUNCTION": 0.0}
            leg.cause_final = "NONE"
            continue
        vlm_probs = getattr(leg, "vlm_probs", None) or {}
        if not isinstance(vlm_probs, dict) or not vlm_probs:
            leg.cause_fused = cause_rule
            leg.fused_probs = {"NONE": 0.0, "BURIED": 0.0, "TRAPPED": 0.0, "TANGLED": 0.0, "MALFUNCTION": 0.0}
            leg.fused_probs[cause_rule] = 1.0
            leg.cause_final = cause_rule
            continue
        fused_probs, cause_fused = fuse_rule_and_vlm(cause_rule, vlm_probs, rule_weight=0.2, vlm_weight=0.8)
        leg.fused_probs = fused_probs
        leg.cause_fused = cause_fused
        leg.cause_final = cause_fused

    updated = dict(session_dict)
    updated["vlm_completed"] = True
    updated_legs = dict(updated.get("legs", {}) or {})
    for leg_id in LEG_IDS:
        legd = dict(updated_legs.get(leg_id, {}) or {})
        leg = session.legs.get(leg_id)
        legd["vlm_pred"] = (leg.vlm_pred if leg else None)
        legd["vlm_probs"] = (leg.vlm_probs if leg else None)
        legd["cause_rule"] = (leg.cause_rule if leg else None)
        legd["p_rule"] = (leg.p_rule if leg else None)
        legd["cause_fused"] = (leg.cause_fused if leg else None)
        legd["fused_probs"] = (leg.fused_probs if leg else None)
        # 最終確定をcause_finalへ反映
        legd["cause_final"] = (leg.cause_final if leg else legd.get("cause_final"))
        legd["movement_result"] = (leg.movement_result if leg else legd.get("movement_result"))
        updated_legs[leg_id] = legd
    updated["legs"] = updated_legs

    append_jsonl(SESSIONS, updated)
    elapsed = time.time() - started
    print(f"[rand10] vlm_postprocess appended ({elapsed:.1f}s)")


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
        subprocess.run(cmd_webots, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    print(f"[rand10] webots finished: {elapsed:.1f}s log={log_path}")

    # 3) VLM後処理
    maybe_run_vlm_postprocess()

    after_lines = count_lines(SESSIONS)
    if after_lines <= before_lines:
        print("[rand10] warn: sessions.jsonl が増えていません")
        return read_last_session()

    return read_last_session()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--count", type=int, default=10)
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
