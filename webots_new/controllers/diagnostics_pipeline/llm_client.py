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

    path = os.getenv("QWEN_GGUF_PATH", "").strip()
    if not path:
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

    path = os.getenv("QWEN_GGUF_PATH", "").strip()
    if not path:
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


def _extract_probs(text: str) -> Optional[Dict[str, float]]:
    if not text:
        return None

    # 1) JSONオブジェクト抽出
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                probs = {k: _safe_float(v, 0.0) for k, v in obj.items() if str(k).upper() in _LABELS}
                if probs:
                    # ラベル欠けは0で埋める
                    for lab in _LABELS:
                        probs.setdefault(lab, 0.0)
                    return probs
        except Exception:
            pass

    # 2) ラベル単体
    up = text.upper()
    for lab in _LABELS:
        if re.search(rf"\b{re.escape(lab)}\b", up):
            return one_hot(lab)
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

        system = os.getenv(
            "QWEN_SYSTEM",
            "あなたは四足歩行ロボットSpotの脚診断エキスパートです。出力は必ずJSONのみ。",
        ).strip()
        user = (
            "ラベル集合: {labels}\n"
            "出力はJSONオブジェクトのみ。各ラベルは0..1で合計1.0。\n"
            "入力(JSON): {payload}"
        ).format(labels=list(_LABELS), payload=json.dumps(payload, ensure_ascii=False))

        try:
            if hasattr(llm, "create_chat_completion"):
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=int(os.getenv("QWEN_MAX_TOKENS", str(self._max_new_tokens))),
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
                    max_tokens=int(os.getenv("QWEN_MAX_TOKENS", str(self._max_new_tokens))),
                    temperature=0.0,
                )
                text = out["choices"][0]["text"]
        except Exception:
            return None, "infer_error"

        probs = _extract_probs(str(text))
        if not probs:
            return None, "parse_failed"
        return normalize_probs(probs, labels=_LABELS), "ok"

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
