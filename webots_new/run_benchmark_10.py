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
SET_ENV = ROOT / "set_environment.py"
SESSIONS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_sessions.jsonl"
VLM_POST = ROOT / "run_vlm_for_last_session.py"
BENCH_DIR = ROOT / "benchmarks"
LOG_DIR = BENCH_DIR / "logs"

VENV_BIN = ROOT / ".venv" / "bin"
VENV_PY = VENV_BIN / "python3"

LEG_IDS = ["FL", "FR", "RL", "RR"]


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def maybe_run_vlm_postprocess() -> None:
    """VLM_ENABLE=1 のとき、最新セッションに VLM結果/融合結果を追記する（重い推論を1回だけロードして使い回す）。"""
    if os.getenv("VLM_ENABLE", "0").strip() != "1":
        return

    started = time.time()

    try:
        from controllers.diagnostics_pipeline.models import LegState, SessionState
        from controllers.diagnostics_pipeline.vlm_client import VLMAnalyzer
    except Exception:
        print("[benchmark] VLM enabled but deps missing -> skip")
        return

    session_dict = read_last_session()
    if not session_dict:
        return

    # すでにvlm_completedなら二重に走らせない
    if bool(session_dict.get("vlm_completed", False)):
        return

    sid = session_dict.get("session_id", "")
    print(f"[benchmark] vlm_postprocess start session={sid}")

    session = SessionState(session_id=str(session_dict.get("session_id", "")))
    img = session_dict.get("image_path")
    # 画像パスの解決（古い形式/相対パスを吸収）
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

    # モデルはグローバルに1回だけロードして使い回す
    global _VLM
    if "_VLM" not in globals() or globals().get("_VLM") is None:
        globals()["_VLM"] = VLMAnalyzer()
    vlm = globals()["_VLM"]

    # 仕様.txtのルール判定（①〜④）
    try:
        from controllers.diagnostics_pipeline.rule_fusion import fuse_rule_and_vlm, rule_based_decision
    except Exception:
        print("[benchmark] rule_fusion missing -> skip")
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

    if to_vlm:
        vlm.infer_session(session, leg_ids=to_vlm)

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
        legd["cause_final"] = (leg.cause_final if leg else legd.get("cause_final"))
        legd["movement_result"] = (leg.movement_result if leg else legd.get("movement_result"))
        updated_legs[leg_id] = legd
    updated["legs"] = updated_legs

    append_jsonl(SESSIONS, updated)
    elapsed = time.time() - started
    print(f"[benchmark] vlm_postprocess appended ({elapsed:.1f}s)")


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

    # 2.5) VLM を Webots外で実行（このPythonプロセスでモデルを使い回す）
    maybe_run_vlm_postprocess()
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


def score_session_vlm(session: dict, expected: dict[str, str]) -> tuple[int, int]:
    if not session:
        return 0, len(LEG_IDS)

    legs = session.get("legs", {})
    ok = 0
    total = 0

    for leg_id in LEG_IDS:
        total += 1
        exp = expected.get(leg_id, "NONE")
        got = (legs.get(leg_id, {}) or {}).get("vlm_pred", None)
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

    total_ok_vlm = 0
    total_vlm = 0

    results = []

    for name, envs in patterns:
        expected = {"FL": envs[0], "FR": envs[1], "RL": envs[2], "RR": envs[3]}
        session = run_one(name, envs)
        ok, n = score_session(session or {}, expected)
        ok_vlm, n_vlm = score_session_vlm(session or {}, expected)
        total_ok += ok
        total += n

        total_ok_vlm += ok_vlm
        total_vlm += n_vlm

        sid = (session or {}).get("session_id", "")
        results.append((name, sid, ok, n, ok_vlm, n_vlm))
        print(f"[benchmark] score cause_final {ok}/{n} | vlm_pred {ok_vlm}/{n_vlm} session={sid}")

    print("\n" + "=" * 70)
    print("[benchmark] summary")
    print("=" * 70)
    for name, sid, ok, n, ok_vlm, n_vlm in results:
        print(f"  {name}: cause_final {ok}/{n} | vlm_pred {ok_vlm}/{n_vlm} ({sid})")
    if total > 0:
        acc = 100.0 * total_ok / total
        print(f"\n[benchmark] total: {total_ok}/{total} ({acc:.1f}%)")
    if total_vlm > 0:
        acc_vlm = 100.0 * total_ok_vlm / total_vlm
        print(f"[benchmark] total(vlm): {total_ok_vlm}/{total_vlm} ({acc_vlm:.1f}%)")


if __name__ == "__main__":
    main()
