"""診断パイプラインの最終推定（最小構成）

最終診断は以下のみ:
- Spot自己診断(spot_can)
- Drone観測(p_drone/drone_can)
- LLM(Qwen)（未設定時はフォールバック）

Qwen(GGUF) を使う場合のみ環境変数を設定:
- QWEN_ENABLE=1
- QWEN_GGUF_PATH=/path/to/model.gguf
"""

from __future__ import annotations

import json
import os
import re
import time
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import config
from .models import LegState
from .rule_fusion import normalize_probs, one_hot, rule_based_decision


_LABELS = tuple(config.CAUSE_LABELS)

_LLM = None
_LLM_PATH: Optional[str] = None


def _resolve_model_path(raw_path: str) -> str:
    """モデルパスを正規化する。

    - `QWEN_GGUF_PATH` が相対パスでも、WebotsコントローラのCWDに依存せず解決できるようにする。
    - まずそのまま/展開後を試し、無ければ webots_new 配下・リポジトリルート基準で解決を試みる。
    """
    p = Path(os.path.expandvars(os.path.expanduser(str(raw_path).strip())))
    if p.is_absolute():
        return str(p)

    project_root = Path(__file__).resolve().parents[2]  # .../webots_new
    candidates = [
        (project_root / p).resolve(),
        (project_root.parent / p).resolve(),
    ]
    for c in candidates:
        try:
            if c.exists():
                return str(c)
        except Exception:
            continue
    return str(p)


def _default_qwen_system_prompt() -> str:
    # 仕様.txt Step7(①〜④)をQwenにも明示する。
    # 重要: 出力形式は既存実装に合わせて「確率分布JSONのみ」を厳守。
    return (
        "あなたは四足歩行ロボットSpotの脚診断エキスパートです。"
        "出力は必ずJSONのみ。\n"
        "\n"
        "次のルール（仕様.txt Step7 の①〜④）も踏まえて、拘束原因の確率分布を推論してください。\n"
        "- ① spot_can と drone_can が共に 0.7 以上なら『動く』。拘束原因は『正常（NONE）』。\n"
        "- ② spot_can と drone_can が共に 0.3 以下なら『動かない』。拘束原因は確率分布の最大。\n"
        "- ③ どちらか一方が 0.7 以上、もう一方が 0.3 以下なら『動かない』。拘束原因は『故障（MALFUNCTION）』。\n"
        "- ④ どちらか一方が 0.3 より高く 0.7 より低い中間の値なら『一部動く』。拘束原因は確率分布の最大。\n"
        "\n"
        "あなたの出力は『拘束原因ラベル→確率(0..1)』のJSONオブジェクトのみ。"
        "確率の合計は必ず 1.0 にしてください。"
    )


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _get_qwen() -> Optional[object]:
    global _LLM, _LLM_PATH

    if not bool(int(os.getenv("QWEN_ENABLE", "0") or "0")):
        return None

    path = _resolve_model_path(os.getenv("QWEN_GGUF_PATH", ""))
    if not path:
        return None

    try:
        if not Path(path).exists():
            return None
    except Exception:
        return None

    if _LLM is not None and _LLM_PATH == path:
        return _LLM

    try:
        from llama_cpp import Llama
    except Exception:
        return None

    try:
        _LLM = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=int(os.cpu_count() or 4),
            n_gpu_layers=0,
            verbose=False,
        )
        _LLM_PATH = path
        return _LLM
    except Exception:
        _LLM = None
        _LLM_PATH = None
        return None


def _get_qwen_with_status() -> Tuple[Optional[object], str]:
    """Qwen(=llama_cpp) の取得と、取得できない理由を返す。

    戻り値:
      (llm, status)
      status:
        - ok
        - disabled (QWEN_ENABLE!=1)
        - no_path (QWEN_GGUF_PATH 未設定)
        - import_error (llama_cpp import失敗)
        - load_error (モデルロード失敗)
    """
    global _LLM, _LLM_PATH

    if not bool(int(os.getenv("QWEN_ENABLE", "0") or "0")):
        return None, "disabled"

    path = _resolve_model_path(os.getenv("QWEN_GGUF_PATH", ""))
    if not path:
        return None, "no_path"

    try:
        if not Path(path).exists():
            return None, "no_path"
    except Exception:
        return None, "no_path"

    if _LLM is not None and _LLM_PATH == path:
        return _LLM, "ok"

    def _try_import_llama() -> bool:
        # 1) 通常の import
        try:
            import llama_cpp  # noqa: F401

            return True
        except Exception:
            pass

        # 2) プロジェクト内 venv (.venv) からの import を試みる
        #    Webots が system python を使っている場合でも動かせるようにする。
        try:
            project_root = Path(__file__).resolve().parents[2]  # .../webots_new
            major = sys.version_info.major
            minor = sys.version_info.minor
            sp = project_root / ".venv" / "lib" / f"python{major}.{minor}" / "site-packages"
            if sp.exists() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
            import llama_cpp  # noqa: F401

            return True
        except Exception:
            return False

    if not _try_import_llama():
        return None, "import_error"

    try:
        from llama_cpp import Llama
    except Exception:
        return None, "import_error"

    try:
        _LLM = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=int(os.cpu_count() or 4),
            n_gpu_layers=0,
            verbose=False,
        )
        _LLM_PATH = path
        return _LLM, "ok"
    except Exception:
        _LLM = None
        _LLM_PATH = None
        return None, "load_error"


def _build_feature_summary(leg: LegState) -> Dict[str, float]:
    """Qwenに渡す最小サマリ（無くても動くが、あると誤判定が減る）。"""

    def series(key: str):
        out = []
        for t in getattr(leg, "trials", []) or []:
            f = getattr(t, "features", None) or {}
            v = f.get(key)
            if v is None:
                continue
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    def med(xs):
        if not xs:
            return None
        a = sorted(xs)
        return float(a[len(a) // 2])

    summary: Dict[str, float] = {}

    flags = [int(x) for x in series("spot_malfunction_flag") if float(x) == float(x)]
    if flags:
        summary["spot_malfunction_flag_any"] = float(1.0 if any(int(x) == 1 for x in flags) else 0.0)

    tau = series("spot_tau_max_ratio")
    if tau:
        summary["spot_tau_max_ratio_median"] = float(med(tau) or 0.0)

    dtheta = series("delta_theta_norm")
    if dtheta:
        summary["delta_theta_norm_median"] = float(med(dtheta) or 0.0)

    end_disp = series("end_disp")
    if end_disp:
        summary["end_disp_median"] = float(med(end_disp) or 0.0)

    return summary


def _first_json_value(raw: str) -> Optional[object]:
    """テキストから最初にデコード可能なJSON値(dict/list/...)を抽出する。

    LLM出力は以下のように崩れることがあるため、単純な正規表現の貪欲マッチだと
    JSONの末尾以降まで巻き込んでパース失敗しやすい。
    - 前後に説明文が付く
    - 複数のJSONブロックを出す
    - ```json ... ``` のコードフェンスを混ぜる

    本関数は、文字列中の '{' / '[' から順に `json.JSONDecoder().raw_decode()` を試す。
    """
    if not raw:
        return None

    text = str(raw).strip()
    # 先頭/末尾のコードフェンスを軽く除去
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    dec = json.JSONDecoder()

    # ある程度で打ち切り（極端に長いログでも重くしない）
    limit = min(len(text), 20000)
    i = 0
    while i < limit:
        ch = text[i]
        if ch not in "[{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text[i:])
            # raw_decode の end はスライス内の相対位置
            if end > 0:
                return obj
        except Exception:
            pass
        i += 1

    return None


def _extract_probs(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None

    # 1) JSONオブジェクト抽出（最初にパース可能なものを拾う）
    obj = _first_json_value(text)
    if isinstance(obj, dict):
        probs = {k: _safe_float(v, 0.0) for k, v in obj.items() if str(k).upper() in _LABELS}
        if probs:
            # ラベル欠けは0で埋める
            for lab in _LABELS:
                probs.setdefault(lab, 0.0)
            return probs

    # 2) ラベル単体
    up = text.upper()
    for lab in _LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", up):
            return one_hot(lab)
    return None


def _extract_batch_probs(text: str) -> Optional[Dict[str, Dict[str, float]]]:
    """LLM出力から、脚ID→確率分布 を抽出する。

    期待形式例:
      {"FL": {"NONE": 0.1, ...}, "FR": {...}, "RL": {...}, "RR": {...}}
    もしくは
      {"legs": {"FL": {...}, ...}}
    """
    if not text:
        return None

    raw = str(text)

    obj = _first_json_value(raw)
    if obj is None:
        return None

    # 受理する形式をいくつか吸収する
    # 1) {"FL": {...}, ...}
    # 2) {"legs": {"FL": {...}}}
    # 3) [{"leg_id":"FL", "probs": {...}}, ...]
    # 4) {"results": [...]} / {"outputs": [...]} など
    if isinstance(obj, dict):
        if "legs" in obj and isinstance(obj.get("legs"), dict):
            obj = obj["legs"]
        elif "results" in obj and isinstance(obj.get("results"), list):
            obj = obj["results"]
        elif "outputs" in obj and isinstance(obj.get("outputs"), list):
            obj = obj["outputs"]

    out: Dict[str, Dict[str, float]] = {}

    def _dist_from_maybe_dict(d: dict) -> Optional[Dict[str, float]]:
        probs = {str(k).upper(): _safe_float(v, 0.0) for k, v in d.items() if str(k).upper() in _LABELS}
        if not probs:
            return None
        for lab in _LABELS:
            probs.setdefault(lab, 0.0)
        return normalize_probs(probs, labels=_LABELS)

    if isinstance(obj, dict):
        for leg_id, val in obj.items():
            leg_key = str(leg_id).strip().upper()
            if leg_key not in config.LEG_IDS:
                continue

            if isinstance(val, dict):
                # val が直接分布か、{"probs":{...}} の可能性
                if "probs" in val and isinstance(val.get("probs"), dict):
                    dist = _dist_from_maybe_dict(val["probs"])
                else:
                    dist = _dist_from_maybe_dict(val)
                if dist is not None:
                    out[leg_key] = dist
                    continue

            if isinstance(val, str):
                up = val.strip().upper()
                for lab in _LABELS:
                    if re.search(rf"\b{re.escape(lab)}\b", up):
                        out[leg_key] = one_hot(lab)
                        break

        return out or None

    if isinstance(obj, list):
        for item in obj:
            if not isinstance(item, dict):
                continue
            leg_key = str(item.get("leg_id") or item.get("leg") or "").strip().upper()
            if leg_key not in config.LEG_IDS:
                continue

            if "probs" in item and isinstance(item.get("probs"), dict):
                dist = _dist_from_maybe_dict(item["probs"])
            elif "p" in item and isinstance(item.get("p"), dict):
                dist = _dist_from_maybe_dict(item["p"])
            else:
                # item 自体が分布になっている可能性
                dist = _dist_from_maybe_dict(item)

            if dist is not None:
                out[leg_key] = dist

        return out or None

    return None


class LLMAnalyzer:
    """最終診断（Drone分布とQwen分布を重み付き平均で統合 + Step7 で確定）"""

    def __init__(self, max_new_tokens: int = 256) -> None:
        self._max_new_tokens = int(max_new_tokens)

    def _infer_with_qwen(self, leg: LegState, fallback: Dict[str, float]) -> Tuple[Optional[Dict[str, float]], str]:
        llm, status = _get_qwen_with_status()
        if llm is None:
            return None, status

        payload = {
            "leg_id": leg.leg_id,
            "spot_can": _safe_float(getattr(leg, "spot_can", 0.5), 0.5),
            "drone_can": _safe_float(getattr(leg, "drone_can", 0.5), 0.5),
            "p_drone": {k: _safe_float(v, 0.0) for k, v in dict(getattr(leg, "p_drone", {}) or {}).items()},
            "fallback": {k: _safe_float(v, 0.0) for k, v in dict(fallback).items()},
            "trial_feature_summary": _build_feature_summary(leg),
        }

        system = os.getenv("QWEN_SYSTEM", _default_qwen_system_prompt()).strip()
        user = (
            "ラベル集合: {labels}\n"
            "出力はJSONオブジェクトのみ。各ラベルは0..1で合計1.0。\n"
            "入力(JSON): {payload}"
        ).format(labels=list(_LABELS), payload=json.dumps(payload, ensure_ascii=False))

        try:
            # 1脚推論は出力が短いJSONだけで十分なので、既定は128トークンに固定。
            # 優先度: QWEN_MAX_TOKENS > 既定(128)
            max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "128"))
            if hasattr(llm, "create_chat_completion"):
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                text = out["choices"][0]["message"]["content"]
            else:
                prompt = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                out = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                text = out["choices"][0]["text"]
        except Exception:
            return None, "infer_error"

        probs = _extract_probs(str(text))
        if not probs:
            return None, "parse_failed"
        return normalize_probs(probs, labels=_LABELS), "ok"

    def _infer_batch_with_qwen(
        self,
        legs: list[LegState],
        fallback_by_leg: Dict[str, Dict[str, float]],
    ) -> Tuple[Optional[Dict[str, Dict[str, float]]], str]:
        """4脚まとめてQwen推論。

        戻り値:
          (dist_by_leg, status)
          status は _get_qwen_with_status と同じ分類（ok/disabled/no_path/...）。
          dist_by_leg は取得できた脚のみ入る（パース失敗した脚は入らない）。
        """
        llm, status = _get_qwen_with_status()
        if llm is None:
            return None, status

        payload_legs = []
        for leg in legs:
            leg_id = str(getattr(leg, "leg_id", "")).strip().upper()
            if leg_id not in config.LEG_IDS:
                continue
            payload_legs.append(
                {
                    "leg_id": leg_id,
                    "spot_can": _safe_float(getattr(leg, "spot_can", 0.5), 0.5),
                    "drone_can": _safe_float(getattr(leg, "drone_can", 0.5), 0.5),
                    "p_drone": {k: _safe_float(v, 0.0) for k, v in dict(getattr(leg, "p_drone", {}) or {}).items()},
                    "fallback": {k: _safe_float(v, 0.0) for k, v in dict(fallback_by_leg.get(leg_id) or {}).items()},
                    "trial_feature_summary": _build_feature_summary(leg),
                }
            )

        system = os.getenv("QWEN_SYSTEM", _default_qwen_system_prompt()).strip()
        user = (
            "ラベル集合: {labels}\n"
            "あなたは4脚まとめて推論する。出力はJSONオブジェクトのみ。\n"
            "形式: {{\"FL\":{{...}},\"FR\":{{...}},\"RL\":{{...}},\"RR\":{{...}}}}\n"
            "各脚について、各ラベルは0..1で合計1.0。\n"
            "重要: 出力は1行のJSONのみ。説明文・コードブロック・改行は禁止。\n"
            "重要: 小数は小数点以下3桁までに丸める（例: 0.85 や 0.03）。長い 0.030000000000... は禁止。\n"
            "入力(JSON): {payload}"
        ).format(labels=list(_LABELS), payload=json.dumps({"legs": payload_legs}, ensure_ascii=False))

        # バッチ出力は1脚より長くなりやすいので、既定は256に固定して安定性を優先する。
        # 優先度: QWEN_MAX_TOKENS_BATCH > QWEN_MAX_TOKENS > 既定(256)
        max_tokens = int(os.getenv("QWEN_MAX_TOKENS_BATCH", os.getenv("QWEN_MAX_TOKENS", "256")))

        def _run_once(token_limit: int) -> Optional[str]:
            try:
                if hasattr(llm, "create_chat_completion"):
                    out = llm.create_chat_completion(
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                        temperature=0.0,
                        max_tokens=int(token_limit),
                    )
                    return str(out["choices"][0]["message"]["content"])

                prompt = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                out = llm(
                    prompt,
                    max_tokens=int(token_limit),
                    temperature=0.0,
                )
                return str(out["choices"][0]["text"])
            except Exception:
                return None

        text = _run_once(max_tokens)
        if text is None:
            return None, "infer_error"

        dist_by_leg = _extract_batch_probs(str(text))
        if not dist_by_leg:
            # 出力が長くなり切り捨て（max_tokens不足）でJSONが途中終了することがある。
            # その場合はトークン上限を増やして1回だけ再試行する。
            retry_limit = int(os.getenv("QWEN_MAX_TOKENS_BATCH_RETRY", str(max(512, max_tokens * 2))))
            if retry_limit > max_tokens:
                text2 = _run_once(retry_limit)
                if text2 is not None:
                    dist_by_leg = _extract_batch_probs(str(text2))
                    text = text2

        if not dist_by_leg:
            # デバッグ用: 生出力をファイルへ保存（サイズが大きい可能性があるので明示的にONの時だけ）
            if os.getenv("DIAG_DUMP_QWEN_RAW", "0") in {"1", "true", "TRUE", "yes", "YES"}:
                try:
                    project_root = Path(__file__).resolve().parents[2]  # .../webots_new
                    log_dir = project_root / "controllers" / "drone_circular_controller" / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    path = log_dir / f"qwen_batch_raw_{ts}.txt"
                    path.write_text(str(text), encoding="utf-8")
                    # 端末にも保存先だけ出す
                    print(f"[llm_client] dumped Qwen raw output -> {path}")
                except Exception:
                    pass
            return None, "parse_failed"
        return dist_by_leg, "ok"

    def _weighted_blend(
        self,
        p_drone: Dict[str, float],
        p_qwen: Dict[str, float],
        w_drone: float,
        w_qwen: float,
    ) -> Dict[str, float]:
        wd = _safe_float(w_drone, 0.0)
        wq = _safe_float(w_qwen, 0.0)
        if wd < 0.0:
            wd = 0.0
        if wq < 0.0:
            wq = 0.0
        if wd + wq <= 0.0:
            wd, wq = 0.7, 0.3

        pd = normalize_probs(p_drone or {}, labels=_LABELS)
        pq = normalize_probs(p_qwen or {}, labels=_LABELS)

        out: Dict[str, float] = {}
        for lab in _LABELS:
            out[lab] = wd * _safe_float(pd.get(lab, 0.0), 0.0) + wq * _safe_float(pq.get(lab, 0.0), 0.0)
        return normalize_probs(out, labels=_LABELS)

    def infer(self, leg: LegState, all_legs=None, trial_direction=None) -> Dict[str, float]:
        # Drone分布（無ければ一様）
        p_drone = dict(getattr(leg, "p_drone", {}) or {})
        if not p_drone:
            p_drone = {lab: 1.0 / len(_LABELS) for lab in _LABELS}
        p_drone = normalize_probs(p_drone, labels=_LABELS)

        # Qwen分布（毎回寄与させる想定：取得できない/失敗時は一様を採用）
        p_qwen_default = {lab: 1.0 / len(_LABELS) for lab in _LABELS}
        p_qwen = dict(p_qwen_default)

        t0 = time.perf_counter()
        dist_qwen, qwen_status = self._infer_with_qwen(leg, fallback=p_drone)
        try:
            if hasattr(leg, "add_timing"):
                leg.add_timing("llm_qwen_infer", time.perf_counter() - t0)
        except Exception:
            pass

        # ログ用: Qwenの利用状況（後から集計できる）
        try:
            leg.qwen_used = bool(dist_qwen is not None)
            leg.qwen_status = str(qwen_status)
        except Exception:
            pass

        if dist_qwen is not None:
            p_qwen = normalize_probs(dist_qwen, labels=_LABELS)

        # ログ/集計用に「Qwenが出した分布（失敗時は一様フォールバック）」を残す
        try:
            leg.p_qwen = dict(p_qwen)
        except Exception:
            pass

        # 統合: 0.7*Drone + 0.3*Qwen
        dist = self._weighted_blend(
            p_drone=p_drone,
            p_qwen=p_qwen,
            w_drone=getattr(config, "DRONE_CAUSE_WEIGHT", 0.7),
            w_qwen=getattr(config, "QWEN_CAUSE_WEIGHT", 0.3),
        )
        leg.p_llm = dict(dist)

        t1 = time.perf_counter()
        movement, cause_rule, p_rule = rule_based_decision(
            getattr(leg, "spot_can", 0.5),
            getattr(leg, "drone_can", 0.5),
            dist,
        )
        try:
            if hasattr(leg, "add_timing"):
                leg.add_timing("rule_based_decision", time.perf_counter() - t1)
        except Exception:
            pass
        leg.cause_rule = cause_rule
        leg.p_rule = dict(p_rule) if p_rule else one_hot(cause_rule)
        leg.movement_result = movement
        leg.cause_final = cause_rule
        try:
            leg.p_can = (float(leg.spot_can) + float(leg.drone_can)) / 2.0
        except Exception:
            pass
        return dist

    def infer_all_legs(self, legs_by_id: Dict[str, LegState]) -> Dict[str, Dict[str, float]]:
        """4脚まとめて最終推定する（Qwen推論は1回）。

        - 各脚: p_drone と p_qwen を 0.7/0.3 で統合して p_llm
        - 仕様.txt Step7 ルールで cause_final 等を確定

        戻り値: leg_id -> p_llm
        """
        # 対象脚
        legs: list[LegState] = []
        for leg_id in config.LEG_IDS:
            leg = legs_by_id.get(leg_id)
            if leg is not None:
                legs.append(leg)

        # Drone分布（無ければ一様）を脚ごとに正規化
        p_drone_by_leg: Dict[str, Dict[str, float]] = {}
        uniform = {lab: 1.0 / len(_LABELS) for lab in _LABELS}
        for leg in legs:
            leg_id = str(getattr(leg, "leg_id", "")).strip().upper()
            pd = dict(getattr(leg, "p_drone", {}) or {})
            if not pd:
                pd = dict(uniform)
            p_drone_by_leg[leg_id] = normalize_probs(pd, labels=_LABELS)

        # Qwen分布（取得失敗時は一様フォールバック）
        p_qwen_default = dict(uniform)
        p_qwen_by_leg: Dict[str, Dict[str, float]] = {leg_id: dict(p_qwen_default) for leg_id in p_drone_by_leg.keys()}

        t0 = time.perf_counter()
        dist_by_leg, status = self._infer_batch_with_qwen(legs, fallback_by_leg=p_drone_by_leg)
        dt = time.perf_counter() - t0

        # timing: 1回分を脚数で割って加算（1脚だけが極端に大きく見えないように）
        per_leg_dt = dt / max(1, len(legs))
        for leg in legs:
            try:
                if hasattr(leg, "add_timing"):
                    leg.add_timing("llm_qwen_infer", per_leg_dt)
            except Exception:
                pass

        # status を脚へ反映
        if dist_by_leg is None:
            # バッチ推論が失敗した場合、1脚ずつフォールバックしてQwenの利用率を上げる。
            for leg in legs:
                leg_id = str(getattr(leg, "leg_id", "")).strip().upper()
                pd = p_drone_by_leg.get(leg_id, uniform)
                t1 = time.perf_counter()
                d1, st1 = self._infer_with_qwen(leg, fallback=pd)
                dt1 = time.perf_counter() - t1
                try:
                    if hasattr(leg, "add_timing"):
                        leg.add_timing("llm_qwen_infer_fallback", dt1)
                except Exception:
                    pass

                if d1 is not None:
                    p_qwen_by_leg[leg_id] = normalize_probs(d1, labels=_LABELS)
                    try:
                        leg.qwen_used = True
                        leg.qwen_status = str(st1)
                    except Exception:
                        pass
                else:
                    try:
                        leg.qwen_used = False
                        leg.qwen_status = str(st1)
                    except Exception:
                        pass
                try:
                    leg.p_qwen = dict(p_qwen_by_leg.get(leg_id, p_qwen_default))
                except Exception:
                    pass
        else:
            # 取得できた脚のみ dist を反映。できなかった脚は1脚推論で再試行。
            for leg in legs:
                leg_id = str(getattr(leg, "leg_id", "")).strip().upper()
                d = dist_by_leg.get(leg_id) if isinstance(dist_by_leg, dict) else None
                if d is not None:
                    p = normalize_probs(d, labels=_LABELS)
                    p_qwen_by_leg[leg_id] = p
                    try:
                        leg.qwen_used = True
                        leg.qwen_status = str(status)
                    except Exception:
                        pass
                else:
                    pd = p_drone_by_leg.get(leg_id, uniform)
                    t1 = time.perf_counter()
                    d1, st1 = self._infer_with_qwen(leg, fallback=pd)
                    dt1 = time.perf_counter() - t1
                    try:
                        if hasattr(leg, "add_timing"):
                            leg.add_timing("llm_qwen_infer_fallback", dt1)
                    except Exception:
                        pass

                    if d1 is not None:
                        p_qwen_by_leg[leg_id] = normalize_probs(d1, labels=_LABELS)
                        try:
                            leg.qwen_used = True
                            leg.qwen_status = str(st1)
                        except Exception:
                            pass
                    else:
                        try:
                            leg.qwen_used = False
                            leg.qwen_status = str(st1 or "parse_failed")
                        except Exception:
                            pass

                try:
                    leg.p_qwen = dict(p_qwen_by_leg.get(leg_id, p_qwen_default))
                except Exception:
                    pass

        # 統合 + ルール確定
        out: Dict[str, Dict[str, float]] = {}
        for leg in legs:
            leg_id = str(getattr(leg, "leg_id", "")).strip().upper()
            pd = p_drone_by_leg.get(leg_id, uniform)
            pq = p_qwen_by_leg.get(leg_id, uniform)

            dist = self._weighted_blend(
                p_drone=pd,
                p_qwen=pq,
                w_drone=getattr(config, "DRONE_CAUSE_WEIGHT", 0.7),
                w_qwen=getattr(config, "QWEN_CAUSE_WEIGHT", 0.3),
            )
            leg.p_llm = dict(dist)

            t1 = time.perf_counter()
            movement, cause_rule, p_rule = rule_based_decision(
                getattr(leg, "spot_can", 0.5),
                getattr(leg, "drone_can", 0.5),
                dist,
            )
            try:
                if hasattr(leg, "add_timing"):
                    leg.add_timing("rule_based_decision", time.perf_counter() - t1)
            except Exception:
                pass

            leg.cause_rule = cause_rule
            leg.p_rule = dict(p_rule) if p_rule else one_hot(cause_rule)
            leg.movement_result = movement
            leg.cause_final = cause_rule
            try:
                leg.p_can = (float(leg.spot_can) + float(leg.drone_can)) / 2.0
            except Exception:
                pass

            out[leg_id] = dict(dist)

        # finalize全体の所要時間も脚へ加算（概算）
        for leg in legs:
            try:
                if hasattr(leg, "add_timing"):
                    leg.add_timing("final_infer_total", per_leg_dt)
            except Exception:
                pass

        return out
