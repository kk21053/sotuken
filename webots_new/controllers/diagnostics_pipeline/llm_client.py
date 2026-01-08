"""診断パイプラインの最終推定（ルール + オプションでQwen）

デフォルトは仕様.txt準拠のルールベース推論（高速・安定）。

環境変数で GGUF版Qwen（llama-cpp-python）を有効化できる:
- QWEN_ENABLE=1
- QWEN_GGUF_PATH=/path/to/model.gguf
    もしくは
    QWEN_GGUF_REPO=... と QWEN_GGUF_FILENAME=...

注: Webotsの試行は1脚あたりTRIAL_COUNT回あるため、Qwenは「試行が揃った後」かつ
「異常が疑われる」場合のみ呼ぶ（毎trialで重い推論をしない）。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

from . import config
from .models import LegState
from .rule_fusion import one_hot, rule_based_decision


_LABELS = tuple(config.CAUSE_LABELS)
_LABEL_SET = set(_LABELS)


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _normalize_probs(probs: Dict[str, float], labels=_LABELS) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = 0.0
    for lab in labels:
        v = _safe_float(probs.get(lab, 0.0), 0.0)
        if v < 0.0:
            v = 0.0
        out[lab] = v
        s += v
    if s <= 0.0:
        n = len(tuple(labels))
        if n <= 0:
            return {}
        return {lab: 1.0 / n for lab in labels}
    return {lab: out[lab] / s for lab in out}


def _median(values):
    """簡単な中央値（要素数が小さい前提）"""

    if not values:
        return None
    a = sorted(values)
    return a[len(a) // 2]


def _collect_feature_series(leg: LegState, key: str):
    """leg.trials から features[key] の系列を集める（存在するものだけ）"""

    out = []
    try:
        for t in getattr(leg, "trials", []) or []:
            if not getattr(t, "features", None):
                continue
            v = t.features.get(key)
            if v is None:
                continue
            out.append(float(v))
    except Exception:
        return []
    return out


def _build_feature_summary(leg: LegState) -> Dict[str, float]:
    """Qwenに渡すための特徴量サマリを作る。

    目的:
    - MALFUNCTION と BURIED/TRAPPED/TANGLED の切り分け根拠を増やす
    - 6 trial 分をそのまま渡すと冗長なので、代表値（中央値/最大）を渡す
    """

    keys = [
        # Spot 側
        "spot_tau_avg_ratio",
        "spot_tau_max_ratio",
        "spot_malfunction_flag",
        # Drone 側（観測特徴）
        "delta_theta_norm",
        "end_disp",
        "path_straightness",
        "reversals",
    ]

    summary: Dict[str, float] = {}
    for k in keys:
        series = _collect_feature_series(leg, k)
        if not series:
            continue

        m = _median(series)
        if m is not None:
            summary[f"{k}_median"] = float(m)

        summary[f"{k}_max"] = float(max(series))

        # フラグは any 的に使えるようにする（1が含まれたら1）
        if k == "spot_malfunction_flag":
            summary["spot_malfunction_flag_any"] = float(1.0 if any(int(x) == 1 for x in series) else 0.0)

    return summary


def _extract_probs(text: str) -> Optional[Dict[str, float]]:
    """LLM出力から確率分布(JSON)を抽出する。

    期待形式: {"NONE":0.1, "BURIED":0.2, ...}
    フォールバック: ラベル単体出力の場合はone-hotとして扱う。
    """

    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    upper = raw.upper()
    m = re.search(r"\b(NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION)\b", upper)
    if m and "{" not in raw:
        lab = m.group(1)
        return {k: (1.0 if k == lab else 0.0) for k in _LABELS}

    # まずは JSON オブジェクトとして素直に解釈する
    l = raw.find("{")
    r = raw.rfind("}")
    snippet = raw[l : r + 1] if (l >= 0 and r >= 0 and r > l) else raw

    probs: Dict[str, float] = {}
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(k, str):
                    continue
                kk = k.strip().upper()
                if kk not in _LABEL_SET:
                    continue
                try:
                    probs[kk] = float(v)
                except Exception:
                    continue
    except Exception:
        # JSONとして壊れていても、ラベル:数値 を汎用的に抽出する（出力健全性チェックの一環）
        # 例: NONE:0.1, "BURIED": 0.2, MALFUNCTION = 0.7 など
        pair_re = re.compile(
            r"(?i)(?:\b|\")(?P<label>NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION)(?:\b|\")\s*[:=]\s*(?P<val>-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
        )
        for m2 in pair_re.finditer(raw):
            lab = (m2.group("label") or "").strip().upper()
            if lab not in _LABEL_SET:
                continue
            probs[lab] = _safe_float(m2.group("val"), 0.0)

    if not probs:
        return None

    # 全ラベルが0（または負）だと正規化で一様分布になり、
    # NONEに誤寄りする原因になる。
    # このケースは「LLM出力が壊れている」として失敗扱いにする。
    try:
        if max(float(x) for x in probs.values()) <= 0.0:
            return None
    except Exception:
        return None
    return _normalize_probs(probs)


def _resolve_gguf_path() -> Path:
    explicit = os.getenv("QWEN_GGUF_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"QWEN_GGUF_PATH not found: {p}")
        return p

    repo = os.getenv("QWEN_GGUF_REPO", "").strip()
    filename = os.getenv("QWEN_GGUF_FILENAME", "").strip()
    if not repo or not filename:
        raise RuntimeError("GGUF model is not specified")

    # 一部環境でCAS/Xet経由DLが失敗するため、未指定なら無効化しておく
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename=filename)
    return Path(downloaded)


_LLAMA_SINGLETON = None
_LLAMA_MODEL_PATH = None
_LLAMA_LOG_CONFIGURED = False


def _configure_llama_logging() -> None:
    """llama.cpp のログを抑制する（Webotsコンソールを汚さないため）。

    - デフォルト: 抑制
    - 有効化したい場合: QWEN_LLAMA_LOG=1
    """

    global _LLAMA_LOG_CONFIGURED
    if _LLAMA_LOG_CONFIGURED:
        return
    _llama_log = os.getenv("QWEN_LLAMA_LOG", "0").strip()
    if _llama_log == "1":
        _LLAMA_LOG_CONFIGURED = True
        return
    try:
        from llama_cpp import llama_log_set

        def _silent_log(level, text, user_data):
            return

        llama_log_set(_silent_log, None)
    except Exception:
        pass
    _LLAMA_LOG_CONFIGURED = True


def _get_llama_singleton():
    global _LLAMA_SINGLETON, _LLAMA_MODEL_PATH

    model_path = _resolve_gguf_path()
    if _LLAMA_SINGLETON is not None and _LLAMA_MODEL_PATH == str(model_path):
        return _LLAMA_SINGLETON

    from llama_cpp import Llama
    _configure_llama_logging()

    # Webotsコンソール上でモデルロードが分かるようにする
    try:
        if bool(int(os.getenv("QWEN_PROGRESS", "1"))):
            print(f"[qwen] model loading: {model_path}", flush=True)
    except Exception:
        pass

    n_ctx = int(os.getenv("QWEN_CTX", "2048"))
    n_threads = int(os.getenv("QWEN_THREADS", str(os.cpu_count() or 4)))
    n_gpu_layers = int(os.getenv("QWEN_GPU_LAYERS", "0"))
    verbose = bool(int(os.getenv("QWEN_VERBOSE", "0")))

    _LLAMA_SINGLETON = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    _LLAMA_MODEL_PATH = str(model_path)

    try:
        if bool(int(os.getenv("QWEN_PROGRESS", "1"))):
            print("[qwen] model loaded", flush=True)
    except Exception:
        pass
    return _LLAMA_SINGLETON


class LLMAnalyzer:
    """仕様準拠のルールベース判定器"""

    def __init__(self, model_priority: Optional[object] = None, max_new_tokens: int = 256) -> None:
        self._model_name = "rule-based-spec-compliant"
        self._max_new_tokens = int(max_new_tokens)

        # Qwenは重い推論なので、明示ON(=1)が基本。
        # ただしモデル指定がある場合はON扱いにする（設定漏れ対策）。
        repo = os.getenv("QWEN_GGUF_REPO", "").strip()
        fn = os.getenv("QWEN_GGUF_FILENAME", "").strip()
        has_model = bool(os.getenv("QWEN_GGUF_PATH", "").strip()) or (bool(repo) and bool(fn))
        self._qwen_enabled = bool(int(os.getenv("QWEN_ENABLE", "0"))) or has_model

    # ------------------------------
    # 外部から呼べる「段階別API」
    # ------------------------------
    def infer_rule_based_only(self, leg: LegState) -> Dict[str, float]:
        """まず仕様ベース（＋汎用的な補助）で分布を作る。"""
        return self._infer_rule_based(leg)

    def build_qwen_payload(self, leg: LegState, fallback: Dict[str, float]) -> Dict:
        """Qwenに渡す最小JSONを作る（オンライン用）。"""

        # 入力は「前処理」の一部。
        # ここでQwenに必要な根拠（特徴量サマリ）を入れて、
        # 後処理の条件分岐を増やさずに精度を上げる。
        payload = {
            "leg_id": leg.leg_id,
            "spot_can": _safe_float(leg.spot_can, 0.5),
            "drone_can": _safe_float(leg.drone_can, 0.5),
            "p_drone": {k: _safe_float(v, 0.0) for k, v in dict(leg.p_drone).items()},
            "fallback_rule_probs": {k: _safe_float(v, 0.0) for k, v in dict(fallback).items()},
        }

        # 6 trial ぶんの特徴量をまとめて渡す（中央値/最大など）
        try:
            payload["trial_feature_summary"] = _build_feature_summary(leg)
        except Exception:
            payload["trial_feature_summary"] = {}

        return payload

    def infer_with_qwen_payload(self, payload: Dict) -> Optional[Dict[str, float]]:
        """Qwen(GGUF)で確率分布を推定する。失敗したらNone。"""

        progress = True
        try:
            progress = bool(int(os.getenv("QWEN_PROGRESS", "1")))
        except Exception:
            progress = True

        if progress:
            try:
                print(
                    "[qwen] infer start leg={leg} spot_can={sc} drone_can={dc}".format(
                        leg=str(payload.get("leg_id")),
                        sc=payload.get("spot_can"),
                        dc=payload.get("drone_can"),
                    ),
                    flush=True,
                )
            except Exception:
                pass

        started = time.time()

        try:
            llm = _get_llama_singleton()
        except Exception as exc:
            print(f"[llm] Qwen init failed: {exc}")
            return None

        system = os.getenv(
            "QWEN_SYSTEM",
            "あなたは四足歩行ロボットSpotの脚診断エキスパートです。出力は必ずJSONのみ。",
        ).strip()

        # 重要: 「後処理で分岐を増やす」のではなく、
        # Qwenが判断できる根拠（spot_malfunction_flag / tau比 / drone特徴）を入力として渡し、
        # 判断方針もプロンプト側で明確化する。
        user = (
            "次のラベル集合の確率分布を推定してください。\n"
            f"ラベル: {list(_LABELS)}\n"
            "出力は JSONオブジェクトのみ（説明文やコードブロック禁止）。\n"
            "各ラベルの値は0..1の実数、合計は1.0に正規化してください。\n\n"
            "判断の基本方針（重要）:\n"
            "- spot_malfunction_flag_any=1 は MALFUNCTION の強い根拠だが、p_droneや他特徴と矛盾する場合は総合判断する\n"
            "- spot_tau_*_ratio が高いことは『外力/拘束で抵抗が大きい』根拠であり、単独で MALFUNCTION の根拠にしない\n"
            "- spot_can が高い(>=0.7)のに drone_can が低い(<=0.3)場合でも、p_drone が BURIED/TRAPPED/TANGLED を強く示すならそれを優先\n"
            "- spot_can が低い(<=0.3)場合、BURIEDは原則低く（BURIEDはSpot自己診断が高く出やすい）\n"
            "- p_drone は『拘束原因（BURIED/TRAPPED/TANGLED）』の手がかりとして重視\n"
            "- fallback_rule_probs は参考。根拠（フラグやp_drone）と矛盾するなら引きずられない\n\n"
            f"入力データ(JSON): {json.dumps(payload, ensure_ascii=False)}"
        )

        temperature = float(os.getenv("QWEN_TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv("QWEN_MAX_TOKENS", str(self._max_new_tokens)))

        try:
            if hasattr(llm, "create_chat_completion"):
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = out["choices"][0]["message"]["content"]
            else:
                prompt = (
                    f"<|im_start|>system\n{system}<|im_end|>\n"
                    f"<|im_start|>user\n{user}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
                text = out["choices"][0]["text"]
        except Exception as exc:
            print(f"[llm] Qwen call failed: {exc}")
            return None

        probs = _extract_probs(text)
        if not probs:
            print("[llm] Qwen output parse failed; using fallback")
            return None

        if progress:
            try:
                elapsed = time.time() - started
                print(f"[qwen] infer done ({elapsed:.1f}s)", flush=True)
            except Exception:
                pass

        return _normalize_probs(probs)

    def should_use_qwen(self, leg: LegState, dist: Dict[str, float]) -> bool:
        """オンラインでQwen推論を回すべきか。"""

        if not self._qwen_enabled:
            return False

        # 仕様上: 試行が揃ってからで十分（毎trialで重い推論をしない）
        trials_ready = len(getattr(leg, "trials", []) or []) >= int(config.TRIAL_COUNT)
        if not trials_ready:
            return False

        # 正常(NONE)っぽいならQwenは回さない
        best_label, best_prob = max(dist.items(), key=lambda kv: kv[1])
        if best_label == "NONE":
            return False

        # ---- ここから「前処理」: Qwenを呼ぶ条件を絞る ----
        # 目的:
        # - p_drone が十分に強いのに、QwenがMALFUNCTIONに寄せてしまう誤りを防ぐ
        # - ただし spot 側が故障を示す (spot_malfunction_flag) 場合はQwenを使って補強する

        # Spotが故障(=MALFUNCTION)を強く示すなら、Qwenを使う
        try:
            summary = _build_feature_summary(leg)
            if float(summary.get("spot_malfunction_flag_any", 0.0)) >= 1.0:
                return True
        except Exception:
            pass

        # Drone側の拘束原因が十分に強いなら、Qwenは使わずにそのまま採用する
        try:
            p_drone = dict(getattr(leg, "p_drone", {}) or {})
            drone_best_label, drone_best_prob = max(p_drone.items(), key=lambda kv: _safe_float(kv[1], 0.0))
            if drone_best_label in {"BURIED", "TRAPPED", "TANGLED"} and drone_best_prob >= 0.70:
                return False
        except Exception:
            pass

        # ルール分布が十分に尖っている（確信が高い）なら、Qwenは使わない
        # ※ dist は現状 0.69 固定になりやすいので、少し小さめに設定
        if best_prob >= 0.68:
            return False

        return True

    def apply_spec_rule_logging(self, leg: LegState, dist: Dict[str, float]) -> None:
        """仕様.txt Step7のルール結果を leg.cause_rule / leg.p_rule に残す。"""

        try:
            movement, cause_rule, p_rule = rule_based_decision(leg.spot_can, leg.drone_can, dist)
            leg.cause_rule = cause_rule
            leg.p_rule = dict(p_rule) if p_rule else one_hot(cause_rule)
            if not getattr(leg, "movement_result", None):
                leg.movement_result = movement
        except Exception:
            pass

    def _infer_rule_based(self, leg: LegState) -> Dict[str, float]:
        spot_can = leg.spot_can
        drone_can = leg.drone_can
        p_drone = dict(leg.p_drone)

        # 直近の試行ログ（features）から、Spotのトルク比を拾う（あれば）
        # - BURIED/TRAPPED/TANGLED: 動かないがトルクは出やすい
        # - MALFUNCTION: 指示しても動かず、トルクも出にくい（またはほぼ0）
        spot_tau_max_ratio = None
        spot_malfunction_flag = None
        try:
            ratios = []
            flags = []
            for t in leg.trials:
                if not t.features:
                    continue
                v = t.features.get("spot_tau_max_ratio")
                if v is None:
                    pass
                else:
                    ratios.append(float(v))

                mf = t.features.get("spot_malfunction_flag")
                if mf is None:
                    continue
                flags.append(int(mf))
            if ratios:
                ratios.sort()
                spot_tau_max_ratio = ratios[len(ratios) // 2]  # median
            if flags:
                flags.sort()
                spot_malfunction_flag = flags[len(flags) // 2]  # median
        except Exception:
            spot_tau_max_ratio = None
            spot_malfunction_flag = None

        # ルール①: 両方が非常に高い → 動く、原因=NONE（閾値を引き上げて範囲を狭める）
        if spot_can >= 0.80 and drone_can >= 0.80:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.90,
                "BURIED": 0.01,
                "TRAPPED": 0.01,
                "TANGLED": 0.01,
                "MALFUNCTION": 0.02,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール①b: "動く" の一般則（片方が少し低くても、全体として高ければNONE）
        # ※特定の誤分類パターン回避ではなく、can=動作確率という定義に沿った汎用ルール。
        # 閾値を引き上げて NONE への寄せを抑制
        if spot_can >= 0.65 and drone_can >= 0.80:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.80,
                "BURIED": 0.03,
                "TRAPPED": 0.03,
                "TANGLED": 0.03,
                "MALFUNCTION": 0.05,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール①c: Drone観測で明確に動けている（drone_canが高い）なら NONE 寄り。
        # Spot側の自己診断が中間でも、観測上の動作が十分なら「動く」と解釈する。
        # 閾値を引き上げて NONE の適用範囲を縮小
        if drone_can >= 0.85 and spot_can >= 0.50:
            leg.movement_result = "動く"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.80,
                "BURIED": 0.03,
                "TRAPPED": 0.03,
                "TANGLED": 0.03,
                "MALFUNCTION": 0.05,
            }
            leg.p_llm = dist
            leg.cause_final = "NONE"
            return dist

        # ルール②: 両方が低い → 動かない、原因= p_drone の最大（ただし NONE は除外）
        if spot_can <= 0.3 and drone_can <= 0.3:
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2

            # Spot側が「故障モード」と明示している場合は、MALFUNCTIONを優先する
            if spot_malfunction_flag == 1:
                dist = {
                    "NONE": 0.02,
                    "BURIED": 0.02,
                    "TRAPPED": 0.02,
                    "TANGLED": 0.02,
                    "MALFUNCTION": 0.89,
                }
                leg.p_llm = dist
                leg.cause_final = "MALFUNCTION"
                return dist

            # トルクがほぼ出ていないなら、拘束よりも故障（指示が効かない）を優先
            # 閾値は保守的に小さめ（0.2）にする。
            if spot_tau_max_ratio is not None and spot_tau_max_ratio <= 0.2:
                dist = {
                    "NONE": 0.02,
                    "BURIED": 0.02,
                    "TRAPPED": 0.02,
                    "TANGLED": 0.02,
                    "MALFUNCTION": 0.89,
                }
                leg.p_llm = dist
                leg.cause_final = "MALFUNCTION"
                return dist

            max_cause = max((v, k) for k, v in p_drone.items() if k != "NONE")[1]

            # 注意:
            # 以前はトルク比で BURIED/TRAPPED を寄せる処理を入れていたが、
            # Spotの動作指令が環境で変わらない前提では誤作動しやすい。
            # ここでは Drone側の拘束原因推定(p_drone)をそのまま採用する。
            dist = {
                "NONE": 0.02,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.02,
                "MALFUNCTION": 0.02,
            }
            dist[max_cause] = 0.89
            leg.p_llm = dist
            leg.cause_final = max_cause
            return dist

        # ルール③: 片方が高く片方が低い → 動かない、原因=MALFUNCTION
        # 仕様.txt Step7 に準拠。
        # 注意: ここでの MALFUNCTION は「外部観測と自己診断が強く矛盾する」状況を指し、
        # 追加のフラグが無くても成立しうる。
        if (spot_can >= 0.7 and drone_can <= 0.3) or (spot_can <= 0.3 and drone_can >= 0.7):
            leg.movement_result = "動かない"
            leg.p_can = (spot_can + drone_can) / 2
            dist = {
                "NONE": 0.02,
                "BURIED": 0.02,
                "TRAPPED": 0.02,
                "TANGLED": 0.02,
                "MALFUNCTION": 0.89,
            }
            leg.p_llm = dist
            leg.cause_final = "MALFUNCTION"
            return dist

        # ルール④: 中間が混ざる → 一部動く、原因= p_drone の最大
        leg.movement_result = "一部動く"
        leg.p_can = (spot_can + drone_can) / 2
        max_cause = max(p_drone.items(), key=lambda x: x[1])[0]
        dist = {
            "NONE": 0.10,
            "BURIED": 0.05,
            "TRAPPED": 0.05,
            "TANGLED": 0.05,
            "MALFUNCTION": 0.05,
        }
        dist[max_cause] = 0.69
        leg.p_llm = dist
        leg.cause_final = max_cause
        return dist

    def infer(self, leg: LegState, all_legs=None, trial_direction=None) -> Dict[str, float]:
        # まず仕様ベースで分布を作る
        dist = self.infer_rule_based_only(leg)

        # 条件を満たす場合のみQwen
        if self.should_use_qwen(leg, dist):
            payload = self.build_qwen_payload(leg, fallback=dist)
            updated = self.infer_with_qwen_payload(payload)
            if updated is not None:
                leg.p_llm = updated
                leg.cause_final = max(updated.items(), key=lambda kv: kv[1])[0]
                dist = updated

        # 仕様ルール結果をログ用に残す
        self.apply_spec_rule_logging(leg, dist)
        return dist
