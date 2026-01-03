#!/usr/bin/env python3
"""セッションJSONLの最新行に対してVLM推論を行い、結果を追記する。

目的:
- Webotsコントローラ内でVLM推論を行うと、Webots終了時に強制終了されて
  ログが残らないことがある。
- そこで Webots終了後に1回だけVLM推論を走らせ、同じsession_idの新しい行として
  `vlm_pred` を埋めたセッションレコードを追記する。

使い方（例）:
  cd webots_new
  . .venv/bin/activate
  VLM_ENABLE=1 python3 run_vlm_for_last_session.py

注意:
- このスクリプトは `VLM_ENABLE=1` のときのみ動作。
- `controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl` の
  最新行が対象。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from controllers.diagnostics_pipeline.models import LegState, SessionState
from controllers.diagnostics_pipeline.vlm_client import VLMAnalyzer


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "controllers" / "drone_circular_controller" / "logs"
SESSION_LOG = LOG_DIR / "leg_diagnostics_sessions.jsonl"


def load_last_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except Exception:
        return None


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def to_session_state(d: dict) -> SessionState:
    sid = str(d.get("session_id", ""))
    session = SessionState(session_id=sid)
    session.image_path = d.get("image_path")
    session.fallen = bool(d.get("fallen", False))
    session.fallen_probability = float(d.get("fallen_probability", 0.0) or 0.0)

    legs = d.get("legs", {}) or {}
    for leg_id, legd in legs.items():
        if not isinstance(legd, dict):
            continue
        leg = LegState(leg_id=str(leg_id))
        leg.spot_can = float(legd.get("spot_can", 0.5) or 0.5)
        leg.drone_can = float(legd.get("drone_can", 0.5) or 0.5)
        leg.movement_result = str(legd.get("movement_result", "一部動く"))
        leg.cause_final = str(legd.get("cause_final", "NONE"))
        leg.p_can = float(legd.get("p_can", 0.0) or 0.0)
        leg.expected_cause = str(legd.get("expected_cause", "NONE"))
        leg.fallen = bool(legd.get("fallen", False))
        leg.fallen_probability = float(legd.get("fallen_probability", 0.0) or 0.0)
        session.legs[str(leg_id)] = leg

    return session


def merge_vlm_into_dict(original: dict, session: SessionState) -> dict:
    out = dict(original)
    out["vlm_completed"] = True
    out["image_path"] = session.image_path
    out_legs = dict(out.get("legs", {}) or {})

    for leg_id, leg in session.legs.items():
        legd = dict(out_legs.get(leg_id, {}) or {})
        legd["vlm_pred"] = leg.vlm_pred
        out_legs[leg_id] = legd

    out["legs"] = out_legs
    return out


def main() -> None:
    if os.getenv("VLM_ENABLE", "0").strip() != "1":
        print("[vlm_post] VLM_ENABLE!=1 -> skip")
        return

    d = load_last_json(SESSION_LOG)
    if not d:
        print(f"[vlm_post] no session found: {SESSION_LOG}")
        sys.exit(1)

    sid = d.get("session_id", "")
    img = d.get("image_path", None)
    print(f"[vlm_post] session={sid} image_path={img}")

    session = to_session_state(d)

    vlm = VLMAnalyzer()
    vlm.infer_session(session)

    updated = merge_vlm_into_dict(d, session)
    append_jsonl(SESSION_LOG, updated)
    print("[vlm_post] appended updated session with vlm_pred")


if __name__ == "__main__":
    main()
