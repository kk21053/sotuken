#!/usr/bin/env python3
"""logs_to_text_input.py

目的:
- Webotsの診断ログ(JSONL)から、text_diagnose_qwen.py に渡す入力JSONを自動生成する。
- さらに、ログ中の3セッションを選んで簡易評価(期待ラベルとのtop1一致率)を出す。

前提:
- ログは次の2ファイル。
  - controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl
  - controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl

使い方(例):
- 1セッション分の入力JSONを標準出力へ:
  python logs_to_text_input.py export --session-id drone_20260105_020021

- 1セッション分の入力JSONをファイルへ:
  python logs_to_text_input.py export --session-id drone_20260105_020021 --out /tmp/input.json

- ログから3セッション選んで評価:
  HF_HUB_DISABLE_XET=1 QWEN_GGUF_REPO='bartowski/Qwen2.5-3B-Instruct-GGUF' \
    QWEN_GGUF_FILENAME='Qwen2.5-3B-Instruct-IQ4_XS.gguf' QWEN_THREADS=4 \
    python logs_to_text_input.py evaluate --count 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SESSIONS_JSONL = (
    "controllers/drone_circular_controller/logs/leg_diagnostics_sessions.jsonl"
)
DEFAULT_EVENTS_JSONL = (
    "controllers/drone_circular_controller/logs/leg_diagnostics_events.jsonl"
)

LEG_IDS = ["FL", "FR", "RL", "RR"]


REQUIRED_CAUSES = {"MALFUNCTION", "BURIED", "TRAPPED", "TANGLED", "FALLEN"}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("{"):
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except (TypeError, ValueError):
        return None


def _top1_label(prob_map: Dict[str, float]) -> Optional[str]:
    if not isinstance(prob_map, dict) or not prob_map:
        return None
    best_k = None
    best_v = None
    for k, v in prob_map.items():
        fv = _safe_float(v)
        if fv is None:
            continue
        if best_v is None or fv > best_v:
            best_k = str(k)
            best_v = fv
    return best_k


@dataclass
class Session:
    session_id: str
    timestamp: Optional[float]
    fallen: Optional[bool]
    fallen_probability: Optional[float]
    legs: Dict[str, Dict[str, Any]]


def _load_sessions(sessions_jsonl: str) -> List[Session]:
    rows = _read_jsonl(sessions_jsonl)
    out: List[Session] = []
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            continue
        legs = r.get("legs")
        if not isinstance(legs, dict):
            continue
        out.append(
            Session(
                session_id=str(sid),
                timestamp=_safe_float(r.get("timestamp")),
                fallen=bool(r.get("fallen")) if r.get("fallen") is not None else None,
                fallen_probability=_safe_float(r.get("fallen_probability")),
                legs=legs,
            )
        )
    return out


def _index_events_by_session(events_jsonl: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = _read_jsonl(events_jsonl)
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            continue
        by_session.setdefault(str(sid), []).append(r)
    return by_session


def _extract_labels_from_session(session: Session) -> List[str]:
    labels: List[str] = []
    for _, leg in session.legs.items():
        p = leg.get("p_drone")
        if isinstance(p, dict):
            for k in p.keys():
                if k not in labels:
                    labels.append(k)
    if "NONE" in labels:
        labels = ["NONE"] + [x for x in labels if x != "NONE"]
    return labels


def build_input_json(
    session: Session,
    events_for_session: List[Dict[str, Any]],
    *,
    spot_move_deg: float = 5.0,
    include_ground_truth: bool = False,
) -> Dict[str, Any]:
    def _r(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        try:
            return round(float(v), 4)
        except Exception:
            return v

    legs_out: Dict[str, Any] = {}

    by_leg: Dict[str, List[Dict[str, Any]]] = {lid: [] for lid in LEG_IDS}
    for ev in events_for_session:
        lid = ev.get("leg_id")
        if lid in by_leg:
            by_leg[lid].append(ev)

    for leg_id in LEG_IDS:
        leg_summary = session.legs.get(leg_id, {})
        evs = by_leg.get(leg_id, [])

        def _key(e: Dict[str, Any]) -> Tuple[int, float]:
            ti = _safe_int(e.get("trial_index"))
            ts = _safe_float(e.get("timestamp"))
            return (ti if ti is not None else 10**9, ts if ts is not None else 0.0)

        evs_sorted = sorted(evs, key=_key)

        dirs: List[str] = []
        durations: List[float] = []

        spot_self_can_raw_i: List[float] = []
        drone_can_raw_i: List[float] = []

        feat_delta_theta_deg: List[float] = []
        feat_end_disp: List[float] = []
        # ここは入力が巨大化しやすいので、6トライアル分の配列として必要最小限にする
        # - delta_theta_deg / end_disp: 足先挙動
        # - path_length / path_straightness / reversals: 動きの軌跡の特徴
        # - max_roll / max_pitch: 姿勢
        # - spot_tau_*_ratio / spot_malfunction_flag: Spot側の異常兆候
        # - fallen: 転倒フラグ
        feat_path_length: List[float] = []
        feat_reversals: List[float] = []
        feat_max_pitch: List[float] = []
        feat_spot_malfunction_flag: List[int] = []
        feat_spot_tau_avg_ratio: List[float] = []
        feat_spot_tau_max_ratio: List[float] = []
        feat_fallen: List[int] = []

        trial_p_can: List[float] = []

        for ev in evs_sorted:
            d = ev.get("dir")
            if d in ("+", "-"):
                dirs.append(d)

            du = _safe_float(ev.get("duration"))
            if du is not None:
                durations.append(round(du, 4))

            ss = _safe_float(ev.get("self_can_raw_i"))
            ds = _safe_float(ev.get("drone_can_raw_i"))
            if ss is not None:
                spot_self_can_raw_i.append(round(ss, 4))
            if ds is not None:
                drone_can_raw_i.append(round(ds, 4))

            feats = ev.get("features")
            if isinstance(feats, dict):
                v = _safe_float(feats.get("delta_theta_deg"))
                if v is not None:
                    feat_delta_theta_deg.append(round(v, 4))
                v = _safe_float(feats.get("end_disp"))
                if v is not None:
                    feat_end_disp.append(round(v, 4))

                v = _safe_float(feats.get("path_length"))
                if v is not None:
                    feat_path_length.append(round(v, 4))
                v = _safe_float(feats.get("reversals"))
                if v is not None:
                    feat_reversals.append(round(v, 4))
                v = _safe_float(feats.get("max_pitch"))
                if v is not None:
                    feat_max_pitch.append(round(v, 4))

                smf = feats.get("spot_malfunction_flag")
                if isinstance(smf, (int, float)):
                    feat_spot_malfunction_flag.append(int(smf))
                v = _safe_float(feats.get("spot_tau_avg_ratio"))
                if v is not None:
                    feat_spot_tau_avg_ratio.append(round(v, 4))
                v = _safe_float(feats.get("spot_tau_max_ratio"))
                if v is not None:
                    feat_spot_tau_max_ratio.append(round(v, 4))

                fv = feats.get("fallen")
                if isinstance(fv, bool):
                    feat_fallen.append(1 if fv else 0)

            v = _safe_float(ev.get("p_can"))
            if v is not None:
                trial_p_can.append(round(v, 4))

        legs_out[leg_id] = {
            "spot_can": _r(_safe_float(leg_summary.get("spot_can"))),
            "drone_can": _r(_safe_float(leg_summary.get("drone_can"))),
            "p_drone": leg_summary.get("p_drone"),
            "movement_result": leg_summary.get("movement_result"),
            "trials": {
                "dirs": dirs,
                "duration": durations,
                "spot_self_can_raw_i": spot_self_can_raw_i,
                "drone_can_raw_i": drone_can_raw_i,
                "features": {
                    "delta_theta_deg": feat_delta_theta_deg,
                    "end_disp": feat_end_disp,
                    "path_length": feat_path_length,
                    "reversals": feat_reversals,
                    "max_pitch": feat_max_pitch,
                    "spot_malfunction_flag": feat_spot_malfunction_flag,
                    "spot_tau_avg_ratio": feat_spot_tau_avg_ratio,
                    "spot_tau_max_ratio": feat_spot_tau_max_ratio,
                    "fallen": feat_fallen,
                },
                "trial_can": {
                    "p_can": trial_p_can,
                },
            },
        }

        # 期待ラベル(正解)は、診断入力として与えるとリーク＆混乱要因になるため通常は含めない。
        if include_ground_truth:
            legs_out[leg_id]["expected_cause"] = leg_summary.get("expected_cause")

    out: Dict[str, Any] = {
        "meta": {
            "session_id": session.session_id,
            "timestamp": session.timestamp,
            "image_path": None,
        },
        "spot": {
            "position": None,
            "move_deg": spot_move_deg,
            "move_pattern": ["+", "-", "+", "-", "+", "-"],
        },
        "drone": {
            "position": None,
        },
        "fallen": session.fallen,
        "fallen_probability": _r(session.fallen_probability),
        "labels": _extract_labels_from_session(session),
        "rule_based": {
            "spot_can_threshold": 0.7,
            "drone_can_threshold": 0.3,
            "note": "仕様Step7のルール①〜④をtext_diagnose_qwen側のプロンプトに明記している前提",
        },
        "legs": legs_out,
    }
    return out


def _write_json(data: Dict[str, Any], out_path: Optional[str]) -> None:
    s = json.dumps(data, ensure_ascii=False, indent=2)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(s)
            f.write("\n")
    else:
        sys.stdout.write(s)
        sys.stdout.write("\n")


def _choose_sessions_for_eval(sessions: List[Session], count: int) -> List[Session]:
    required = set(REQUIRED_CAUSES)

    def session_cover(sess: Session) -> set:
        cover: set = set()
        for leg_id in LEG_IDS:
            leg = sess.legs.get(leg_id, {})
            ec = leg.get("expected_cause")
            if ec:
                cover.add(str(ec))
        # セッション転倒情報もカバーに含める（expected_causeにFALLENが無いログもあるため）
        if sess.fallen is True:
            cover.add("FALLEN")
        return cover & required

    def signature(sess: Session) -> Tuple[str, ...]:
        s: List[str] = []
        for leg_id in LEG_IDS:
            leg = sess.legs.get(leg_id, {})
            ec = leg.get("expected_cause")
            if not ec:
                continue
            s.append(str(ec))
        return tuple(sorted(set(s)))

    # まず、requiredをカバーし得る候補に絞る（全探索の爆発を避ける）
    candidates: List[Session] = []
    for sess in sessions:
        cov = session_cover(sess)
        if cov:
            candidates.append(sess)

    # 候補が少なすぎる場合は全体を使う
    base = candidates if len(candidates) >= count else sessions

    # 上位だけ使う（決定的な順序）
    scored_for_pool: List[Tuple[int, int, Session]] = []
    for i, sess in enumerate(base):
        cov = session_cover(sess)
        sig = signature(sess)
        uniq_cnt = len(set(sig))
        non_none_cnt = sum(1 for x in sig if x != "NONE")
        score = len(cov) * 100 + non_none_cnt * 10 + uniq_cnt
        scored_for_pool.append((score, i, sess))
    scored_for_pool.sort(key=lambda x: (-x[0], x[1]))

    pool = [s for _, _, s in scored_for_pool[:120]]

    # count==3のときは、requiredを必ず含む3セッションを探索する
    if count == 3 and len(pool) >= 3:
        best = None
        best_key = None
        for i in range(len(pool)):
            a = pool[i]
            ca = session_cover(a)
            for j in range(i + 1, len(pool)):
                b = pool[j]
                cb = session_cover(b)
                for k in range(j + 1, len(pool)):
                    c = pool[k]
                    cc = session_cover(c)
                    union = ca | cb | cc
                    cover_all = union >= required

                    # スコア: requiredの完全カバーを最優先、その次に多様性
                    overlap = (len(ca & cb) + len(ca & cc) + len(cb & cc))
                    key = (
                        1 if cover_all else 0,
                        len(union),
                        -overlap,
                        a.session_id,
                        b.session_id,
                        c.session_id,
                    )
                    if best_key is None or key > best_key:
                        best_key = key
                        best = [a, b, c]

        if best is not None and (best_key[0] == 1):
            return best

        # 完全カバーが無理なら、最大カバーの3つを返す（固定はする）
        if best is not None:
            return best

    scored: List[Tuple[int, int, Session]] = []
    for i, sess in enumerate(sessions):
        sig = signature(sess)
        uniq_cnt = len(set(sig))
        non_none_cnt = sum(1 for x in sig if x != "NONE")
        score = non_none_cnt * 10 + uniq_cnt
        scored.append((score, i, sess))

    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[Session] = []
    used_sigs: set = set()

    for _, _, sess in scored:
        sig = signature(sess)
        if sig in used_sigs:
            continue
        chosen.append(sess)
        used_sigs.add(sig)
        if len(chosen) >= count:
            break

    if len(chosen) < count:
        for sess in reversed(sessions):
            if sess in chosen:
                continue
            chosen.append(sess)
            if len(chosen) >= count:
                break

    return chosen[:count]


def _evaluate_sessions(
    sessions: List[Session],
    events_by_session: Dict[str, List[Dict[str, Any]]],
    count: int,
    *,
    details: bool = False,
    fail_only: bool = False,
) -> int:
    try:
        from text_diagnose_qwen import diagnose_from_text  # type: ignore
    except Exception as e:
        try:
            from webots_new.text_diagnose_qwen import diagnose_from_text  # type: ignore
        except Exception as e2:
            print(
                f"ERROR: text_diagnose_qwen import failed: {e}; fallback failed: {e2}",
                file=sys.stderr,
            )
            return 2

    chosen = _choose_sessions_for_eval(sessions, count=count)

    # どのセッションが選ばれたか・requiredを満たしているかを表示
    if count == 3:
        covered: set = set()
        for sess in chosen:
            for leg_id in LEG_IDS:
                ec = sess.legs.get(leg_id, {}).get("expected_cause")
                if ec:
                    covered.add(str(ec))
            if sess.fallen is True:
                covered.add("FALLEN")
        missing = sorted(list(REQUIRED_CAUSES - covered))
        print(f"chosen_sessions={[s.session_id for s in chosen]}")
        print(f"required_covered={sorted(list(covered & REQUIRED_CAUSES))}")
        if missing:
            print(f"WARNING: required_missing={missing}")

    total = 0
    correct = 0

    for sess in chosen:
        evs = events_by_session.get(sess.session_id, [])
        inp = build_input_json(sess, evs, include_ground_truth=False)
        inp_text = json.dumps(inp, ensure_ascii=False)

        pred = diagnose_from_text(inp_text)

        sess_total = 0
        sess_correct = 0

        for leg_id in LEG_IDS:
            expected = sess.legs.get(leg_id, {}).get("expected_cause")
            if not expected:
                continue

            prob_map = pred.get(leg_id, {}) if isinstance(pred, dict) else {}
            top1 = _top1_label(prob_map) if isinstance(prob_map, dict) else None

            sess_total += 1
            if top1 == expected:
                sess_correct += 1

            if details and (not fail_only or top1 != expected):
                leg_inp = (inp.get("legs") or {}).get(leg_id, {}) if isinstance(inp, dict) else {}
                sc = (leg_inp.get("spot_can") if isinstance(leg_inp, dict) else None)
                dc = (leg_inp.get("drone_can") if isinstance(leg_inp, dict) else None)
                p_drone = (leg_inp.get("p_drone") if isinstance(leg_inp, dict) else None)
                feats = (((leg_inp.get("trials") or {}).get("features") or {}) if isinstance(leg_inp, dict) else {})

                def _fmt_top3(pm: Dict[str, float]) -> str:
                    items: List[Tuple[str, float]] = []
                    for k, v in pm.items():
                        fv = _safe_float(v)
                        if fv is None:
                            continue
                        items.append((str(k), float(fv)))
                    items.sort(key=lambda x: x[1], reverse=True)
                    return ", ".join([f"{k}:{v:.3f}" for k, v in items[:3]])

                top3_pred = _fmt_top3(prob_map) if isinstance(prob_map, dict) else ""
                top3_drone = _fmt_top3(p_drone) if isinstance(p_drone, dict) else ""

                tau_avg = feats.get("spot_tau_avg_ratio") if isinstance(feats, dict) else None
                tau_max = feats.get("spot_tau_max_ratio") if isinstance(feats, dict) else None
                smf = feats.get("spot_malfunction_flag") if isinstance(feats, dict) else None
                dtheta = feats.get("delta_theta_deg") if isinstance(feats, dict) else None
                end_disp = feats.get("end_disp") if isinstance(feats, dict) else None
                fallen_trials = feats.get("fallen") if isinstance(feats, dict) else None

                print(
                    "  leg={leg} expected={exp} pred={pred} sc={sc} dc={dc} "
                    "pred_top3=[{pt3}] drone_top3=[{dt3}] "
                    "tau_avg={ta} tau_max={tm} smf={smf} dtheta={dth} end_disp={ed} fallen_trials={ft}".format(
                        leg=leg_id,
                        exp=str(expected),
                        pred=str(top1),
                        sc=sc,
                        dc=dc,
                        pt3=top3_pred,
                        dt3=top3_drone,
                        ta=tau_avg,
                        tm=tau_max,
                        smf=smf,
                        dth=dtheta,
                        ed=end_disp,
                        ft=fallen_trials,
                    )
                )

        total += sess_total
        correct += sess_correct

        acc = (sess_correct / sess_total * 100.0) if sess_total else 0.0
        print(
            f"session={sess.session_id} top1={sess_correct}/{sess_total} (acc={acc:.1f}%)"
        )

    overall = (correct / total * 100.0) if total else 0.0
    print(f"overall top1={correct}/{total} (acc={overall:.1f}%)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sessions-jsonl",
        default=DEFAULT_SESSIONS_JSONL,
        help="sessionsのJSONLパス(相対/絶対)",
    )
    p.add_argument(
        "--events-jsonl",
        default=DEFAULT_EVENTS_JSONL,
        help="eventsのJSONLパス(相対/絶対)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--session-id", required=True)
    p_export.add_argument("--out", default=None)
    p_export.add_argument("--spot-move-deg", type=float, default=5.0)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--count", type=int, default=3)
    p_eval.add_argument(
        "--details",
        action="store_true",
        help="脚ごとの予測/正解/主要特徴を表示する",
    )
    p_eval.add_argument(
        "--fail-only",
        action="store_true",
        help="--details時に外した脚だけ表示する",
    )

    args = p.parse_args()

    sessions = _load_sessions(args.sessions_jsonl)
    if not sessions:
        print("ERROR: sessions is empty", file=sys.stderr)
        return 2

    events_by_session = _index_events_by_session(args.events_jsonl)

    if args.cmd == "export":
        sid = args.session_id
        sess = next((s for s in sessions if s.session_id == sid), None)
        if sess is None:
            print(f"ERROR: session_id not found: {sid}", file=sys.stderr)
            return 2

        evs = events_by_session.get(sess.session_id, [])
        data = build_input_json(sess, evs, spot_move_deg=float(args.spot_move_deg))
        _write_json(data, args.out)
        return 0

    if args.cmd == "evaluate":
        n = int(args.count)
        if n <= 0:
            print("ERROR: --count must be >= 1", file=sys.stderr)
            return 2
        return _evaluate_sessions(
            sessions,
            events_by_session,
            count=n,
            details=bool(args.details),
            fail_only=bool(args.fail_only),
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
