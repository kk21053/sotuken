#!/usr/bin/env python3
"""最後のセッションに対してVLM推論を実行し、vlm_predを追記する。

使い方:
  cd webots_new
  VLM_ENABLE=1 python3 run_vlm_for_last_session.py
  VLM_ENABLE=1 python3 run_vlm_for_last_session.py --session-id drone_YYYYmmdd_HHMMSS

注意:
- Moondream2のモデルは初回にダウンロードが走ることがあります。
- 元の行を書き換えず、sessions.jsonlへ新しい行として追記します。
"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "controllers" / "drone_circular_controller" / "logs" / "leg_diagnostics_sessions.jsonl"


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


def read_session_by_id(session_id: str) -> dict | None:
	if not SESSIONS.exists():
		return None
	for line in SESSIONS.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			obj = json.loads(line)
		except Exception:
			continue
		if str(obj.get("session_id", "")) == str(session_id):
			return obj
	return None


def append_jsonl(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_image_path(image_path: str | None) -> str | None:
	if not image_path:
		return None
	p = Path(image_path)
	if p.exists():
		return str(p)
	p2 = ROOT / p
	if p2.exists():
		return str(p2)
	p3 = ROOT / "controllers" / "drone_circular_controller" / "logs" / p
	if p3.exists():
		return str(p3)
	return None


def main() -> int:
	parser = argparse.ArgumentParser()
	parser.add_argument("--session-id", default="")
	args = parser.parse_args()

	if args.session_id:
		session_dict = read_session_by_id(args.session_id)
	else:
		session_dict = read_last_session()

	if not session_dict:
		print("[vlm_run] no session found")
		return 1

	# すでにvlm_completedなら二重に走らせない
	if bool(session_dict.get("vlm_completed", False)):
		print("[vlm_run] already vlm_completed -> skip")
		return 0

	try:
		from controllers.diagnostics_pipeline.models import LegState, SessionState
		from controllers.diagnostics_pipeline.vlm_client import VLMAnalyzer
	except Exception as exc:
		print(f"[vlm_run] import failed: {exc}")
		return 1

	sid = str(session_dict.get("session_id", ""))
	session = SessionState(session_id=sid)
	session.image_path = resolve_image_path(session_dict.get("image_path"))
	session.fallen = bool(session_dict.get("fallen", False))
	session.fallen_probability = float(session_dict.get("fallen_probability", 0.0) or 0.0)

	legs = session_dict.get("legs", {}) or {}
	for leg_id in ["FL", "FR", "RL", "RR"]:
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

	if not session.image_path:
		print("[vlm_run] image_path not found -> skip")
		return 1

	vlm = VLMAnalyzer()
	if not vlm.cfg.enabled:
		print("[vlm_run] VLM_ENABLE!=1 -> skip")
		return 1

	print(f"[vlm_run] start session={sid} model={vlm.cfg.model_id}")

	# 仕様.txtのルール判定（①〜④）
	try:
		from controllers.diagnostics_pipeline.rule_fusion import fuse_rule_and_vlm, rule_based_decision
	except Exception as exc:
		print(f"[vlm_run] import rule_fusion failed: {exc}")
		return 1

	to_vlm: list[str] = []
	for leg_id in ["FL", "FR", "RL", "RR"]:
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

	# ルールがNONE以外の脚だけVLMを走らせる
	if to_vlm:
		vlm.infer_session(session, leg_ids=to_vlm)

	# 0.2*ルール(one-hot) + 0.8*VLM で融合し、cause_finalを最終確定として更新
	for leg_id in ["FL", "FR", "RL", "RR"]:
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
			# VLM失敗時はルールを優先
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
	for leg_id in ["FL", "FR", "RL", "RR"]:
		legd = dict(updated_legs.get(leg_id, {}) or {})
		leg = session.legs.get(leg_id)
		legd["vlm_pred"] = (leg.vlm_pred if leg else None)
		legd["vlm_probs"] = (leg.vlm_probs if leg else None)
		legd["cause_rule"] = (leg.cause_rule if leg else None)
		legd["p_rule"] = (leg.p_rule if leg else None)
		legd["cause_fused"] = (leg.cause_fused if leg else None)
		legd["fused_probs"] = (leg.fused_probs if leg else None)
		# 最終確定をcause_finalに反映（仕様要求）
		legd["cause_final"] = (leg.cause_final if leg else legd.get("cause_final"))
		legd["movement_result"] = (leg.movement_result if leg else legd.get("movement_result"))
		updated_legs[leg_id] = legd
	updated["legs"] = updated_legs

	append_jsonl(SESSIONS, updated)
	print("[vlm_run] appended vlm_pred to sessions.jsonl")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

