#!/usr/bin/env python3
"""テキスト入力から、各脚の拘束原因を確率分布で出力する（Qwen GGUF）

目的
- ユーザーが用意したテキスト（JSON想定）に以下の情報を入れて渡す
  - spot/drone の位置
  - spot の動かし方（+/-5度、6 trial など）
  - spot/drone の生の測定値（配列でもOK）
  - 故障原因の種類（ラベル集合）
  - 前段階のルールベース判断と、そのルール（仕様.txt Step7）
- 出力: 各脚ごとに、各原因ラベルの確率（合計1.0）

注意
- 本スクリプトは webots_new 直下で実行することを想定（controllers を import するため）
- LLMのsystemプロンプトは、リポジトリ直下の「初期プロンプト」を読み込み、その範囲内で追加指示する

使い方（例）
    cd webots_new
    HF_HUB_DISABLE_XET=1 QWEN_GGUF_REPO='bartowski/Qwen2.5-3B-Instruct-GGUF' \
        QWEN_GGUF_FILENAME='Qwen2.5-3B-Instruct-IQ4_XS.gguf' \
        ./.venv/bin/python text_diagnose_qwen.py --self-test

    cd webots_new
    cat input.json | HF_HUB_DISABLE_XET=1 QWEN_GGUF_REPO=... QWEN_GGUF_FILENAME=... \
        ./.venv/bin/python text_diagnose_qwen.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# ラベル集合（仕様.txtの原因に合わせる）
LABELS = ("NONE", "BURIED", "TRAPPED", "TANGLED", "MALFUNCTION", "FALLEN")
LABEL_SET = set(LABELS)


def _topk_probs(prob_map: Dict[str, Any], k: int = 3) -> list:
    items = []
    if isinstance(prob_map, dict):
        for lab in LABELS:
            items.append((lab, safe_float(prob_map.get(lab, 0.0), 0.0)))
    items.sort(key=lambda x: x[1], reverse=True)
    return [[lab, round(v, 4)] for lab, v in items[:k]]


def _others_sum(prob_map: Dict[str, Any], k: int = 3) -> float:
    """top-k以外の確率合計（p_droneの情報を落としすぎないため）。"""

    items = []
    if isinstance(prob_map, dict):
        for lab in LABELS:
            items.append((lab, safe_float(prob_map.get(lab, 0.0), 0.0)))
    items.sort(key=lambda x: x[1], reverse=True)
    rest = items[k:]
    return round(sum(v for _, v in rest), 4)


def _stats(nums: list) -> Dict[str, Any]:
    xs = [safe_float(x, 0.0) for x in nums if isinstance(x, (int, float))]
    if not xs:
        return {"count": 0}
    xs_sorted = sorted(xs)
    return {
        "count": len(xs_sorted),
        "min": round(xs_sorted[0], 4),
        "max": round(xs_sorted[-1], 4),
        "mean": round(sum(xs_sorted) / len(xs_sorted), 4),
    }


def _arr_or_stats(nums: Any, *, max_len: int = 8) -> Any:
    if not isinstance(nums, list):
        return None
    if len(nums) <= max_len and all(isinstance(x, (int, float)) for x in nums):
        return [round(float(x), 4) for x in nums]
    return _stats(nums)


def build_llm_focus_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """LLMに渡すための要点サマリ入力を作る（長文化を避ける）。"""

    legs = data.get("legs")
    out_legs: Dict[str, Any] = {}
    if isinstance(legs, dict):
        for leg_id in ("FL", "FR", "RL", "RR"):
            leg = legs.get(leg_id)
            if not isinstance(leg, dict):
                continue
            trials = leg.get("trials") if isinstance(leg.get("trials"), dict) else {}
            features = trials.get("features") if isinstance(trials.get("features"), dict) else {}
            trial_can = trials.get("trial_can") if isinstance(trials.get("trial_can"), dict) else {}

            out_legs[leg_id] = {
                "spot_can": round(safe_float(leg.get("spot_can"), 0.5), 4),
                "drone_can": round(safe_float(leg.get("drone_can"), 0.5), 4),
                "p_drone_top3": _topk_probs(
                    leg.get("p_drone") if isinstance(leg.get("p_drone"), dict) else {}, 3
                ),
                "p_drone_others_sum": _others_sum(
                    leg.get("p_drone") if isinstance(leg.get("p_drone"), dict) else {}, 3
                ),
                "trials": {
                    "dirs": trials.get("dirs"),
                    "duration": _arr_or_stats(trials.get("duration")),
                    "spot_self_can_raw_i": _arr_or_stats(trials.get("spot_self_can_raw_i")),
                    "drone_can_raw_i": _arr_or_stats(trials.get("drone_can_raw_i")),
                    "p_can": _arr_or_stats(trial_can.get("p_can")),
                    "features": {
                        "delta_theta_deg": _arr_or_stats(features.get("delta_theta_deg")),
                        "end_disp": _arr_or_stats(features.get("end_disp")),
                        "path_length": _arr_or_stats(features.get("path_length")),
                        "reversals": _arr_or_stats(features.get("reversals")),
                        "max_pitch": _arr_or_stats(features.get("max_pitch")),
                        "spot_malfunction_flag": _arr_or_stats(features.get("spot_malfunction_flag")),
                        "spot_tau_avg_ratio": _arr_or_stats(features.get("spot_tau_avg_ratio")),
                        "spot_tau_max_ratio": _arr_or_stats(features.get("spot_tau_max_ratio")),
                        "fallen": _arr_or_stats(features.get("fallen")),
                    },
                },
            }

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    return {
        "meta": {
            "session_id": meta.get("session_id"),
            "timestamp": meta.get("timestamp"),
        },
        "fallen": data.get("fallen"),
        "fallen_probability": data.get("fallen_probability"),
        "labels": list(LABELS),
        "legs": out_legs,
    }


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = 0.0
    for lab in LABELS:
        v = safe_float(probs.get(lab, 0.0), 0.0)
        if v < 0.0:
            v = 0.0
        out[lab] = v
        s += v
    if s <= 0.0:
        return {lab: 1.0 / len(LABELS) for lab in LABELS}
    return {lab: out[lab] / s for lab in out}


def probs_from_p_drone(leg: Dict[str, Any]) -> Dict[str, float]:
    """入力JSONの p_drone を LABELS に合わせて正規化する。"""

    p_drone = leg.get("p_drone")
    if not isinstance(p_drone, dict):
        return {lab: 1.0 / len(LABELS) for lab in LABELS}

    pd: Dict[str, float] = {}
    for lab in LABELS:
        pd[lab] = safe_float(p_drone.get(lab, 0.0), 0.0)
    return normalize_probs(pd)


def blend_probs(a: Dict[str, float], b: Dict[str, float], alpha: float) -> Dict[str, float]:
    """2つの分布を線形結合して正規化する（alphaはa側の重み）。"""

    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0
    out: Dict[str, float] = {}
    for lab in LABELS:
        out[lab] = alpha * safe_float(a.get(lab, 0.0), 0.0) + (1.0 - alpha) * safe_float(
            b.get(lab, 0.0), 0.0
        )
    return normalize_probs(out)


def extract_probs(text: str) -> Optional[Dict[str, float]]:
    """LLM出力から確率分布(JSON)を抽出する。"""

    if not text:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    # 初期プロンプト要求の冒頭文などが付く場合があるので、JSON部分を探す
    l = raw.find("{")
    r = raw.rfind("}")
    if l < 0 or r < 0 or r <= l:
        # ラベル単体だけの出力も許容（one-hot）
        m = re.search(r"\b(NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION|FALLEN)\b", raw.upper())
        if m:
            lab = m.group(1)
            return {k: (1.0 if k == lab else 0.0) for k in LABELS}
        return None

    snippet = raw[l : r + 1]
    try:
        obj = json.loads(snippet)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # 期待形式: {"FL": {"NONE":0.1,...}, "FR": {...}, ...}
    # もし1脚だけの分布だった場合は、"ALL"として扱う
    if all(isinstance(v, (int, float)) for v in obj.values()):
        # {"NONE":0.1,...} 形式
        return normalize_probs({str(k).upper(): float(v) for k, v in obj.items() if str(k).upper() in LABEL_SET})

    return None


def summarize_numbers(obj: Any) -> Any:
    """生データが巨大になりすぎないよう、配列を要約する。"""

    # 数値はそのままだと桁が長くなってトークンが増えるため、適度に丸める
    if isinstance(obj, (int, float)):
        try:
            return round(float(obj), 4)
        except Exception:
            return obj

    if isinstance(obj, list):
        nums = []
        for x in obj:
            if isinstance(x, (int, float)):
                nums.append(float(x))
        # ログ由来の配列(例: 6トライアル)は並び順自体が情報になり得るため、
        # 短い配列は保持し、長い配列のみ統計量に要約する。
        if len(nums) >= 30:
            nums_sorted = sorted(nums)
            return {
                "count": len(nums_sorted),
                "min": round(nums_sorted[0], 4),
                "max": round(nums_sorted[-1], 4),
                "median": round(nums_sorted[len(nums_sorted) // 2], 4),
                "mean": round(sum(nums_sorted) / len(nums_sorted), 4),
            }
        # 数値配列は丸めて保持
        if len(nums) == len(obj):
            return [round(x, 4) for x in nums]
        return [summarize_numbers(x) for x in obj]

    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            out[str(k)] = summarize_numbers(v)
        return out

    return obj


def load_initial_prompt() -> str:
    """リポジトリ直下の「初期プロンプト」を読み込む。"""

    # 1) 環境変数で明示
    p = os.getenv("INITIAL_PROMPT_PATH", "").strip()
    if p:
        path = Path(p)
        return path.read_text(encoding="utf-8")

    # 2) webots_new/text_diagnose_qwen.py から見て2階層上がリポジトリ直下
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "初期プロンプト"
    return path.read_text(encoding="utf-8")


def resolve_gguf_path() -> Path:
    """GGUFモデルの場所を決める（PATH優先、なければHFからDL）。"""

    explicit = os.getenv("QWEN_GGUF_PATH", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"QWEN_GGUF_PATH not found: {path}")
        return path

    repo = os.getenv("QWEN_GGUF_REPO", "").strip()
    filename = os.getenv("QWEN_GGUF_FILENAME", "").strip()
    if not repo or not filename:
        raise SystemExit(
            "GGUFモデル未指定です。\n"
            "- ローカル: QWEN_GGUF_PATH=/path/to/model.gguf\n"
            "- HF: QWEN_GGUF_REPO=... と QWEN_GGUF_FILENAME=...\n"
        )

    # 以前ダウンロード済みなら、そのローカルパスを優先（ネットワーク不安定対策）
    cached = os.getenv("QWEN_GGUF_CACHE_PATH", "").strip()
    if cached:
        p = Path(cached)
        if p.exists():
            return p

    # 一部環境でCAS/Xet経由が失敗するため、未指定なら無効化
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename=filename)
    os.environ["QWEN_GGUF_CACHE_PATH"] = downloaded
    return Path(downloaded)


_LLAMA = None
_LLAMA_PATH = None


def get_llama() -> Any:
    """llama-cpp-python の Llama をシングルトンで保持する。"""

    global _LLAMA, _LLAMA_PATH

    model_path = resolve_gguf_path()
    if _LLAMA is not None and _LLAMA_PATH == str(model_path):
        return _LLAMA

    from llama_cpp import Llama

    n_ctx = int(os.getenv("QWEN_CTX", "2048"))
    n_threads = int(os.getenv("QWEN_THREADS", str(os.cpu_count() or 4)))
    n_gpu_layers = int(os.getenv("QWEN_GPU_LAYERS", "0"))
    verbose = bool(int(os.getenv("QWEN_VERBOSE", "0")))

    _LLAMA = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    _LLAMA_PATH = str(model_path)
    return _LLAMA


def build_user_prompt(data: Dict[str, Any]) -> str:
    """仕様.txtの範囲で、LLMに渡すユーザープロンプトを作る。"""

    # 仕様.txt Step7 のルールをそのまま列挙
    rules = (
        "① spot_canとdrone_canが共に0.7以上なら動く。原因=NONE\n"
        "② spot_canとdrone_canが共に0.3以下なら動かない。原因=確率分布の最大\n"
        "③ どちらかが0.7以上で、もう一方が0.3以下なら動かない。原因=MALFUNCTION\n"
        "④ どちらか一方が0.3より高く0.7より低い中間なら一部動く。原因=確率分布の最大\n"
    )

    # 生データは巨大になりやすいので、脚ごとの要点に圧縮して渡す
    focused = build_llm_focus_input(data)

    return (
        "出力はJSONのみ（説明文、コードブロック禁止）。\n"
        "各脚(FL/FR/RL/RR)ごとに原因ラベルの確率分布を出力。\n"
        f"原因ラベル: {list(LABELS)}\n"
        "形式:\n"
        "{\n"
        "  \"FL\": {\"NONE\":0.1, ...},\n"
        "  \"FR\": {...},\n"
        "  \"RL\": {...},\n"
        "  \"RR\": {...}\n"
        "}\n"
        "各脚の確率は0..1、合計は1.0に正規化してください。\n\n"
        "ルール（仕様.txt Step7）:\n"
        f"{rules}\n"
        "入力データ（要点サマリJSON）:\n"
        f"{json.dumps(focused, ensure_ascii=False)}\n"
    )


def call_qwen(system_prompt: str, user_prompt: str) -> str:
    """Qwenを呼び、テキストを返す。"""

    llm = get_llama()

    temperature = float(os.getenv("QWEN_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("QWEN_MAX_TOKENS", "512"))

    # chat completion があれば使う
    if hasattr(llm, "create_chat_completion"):
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return out["choices"][0]["message"]["content"]

    # フォールバック（ChatML）
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = llm(prompt, max_tokens=max_tokens, temperature=temperature)
    return out["choices"][0]["text"]


def rule_based_distribution_for_leg(leg: Dict[str, Any]) -> Dict[str, float]:
    """仕様.txt Step7 のルールで、最低限の分布（one-hot）を作る。"""

    # controllers 側のルール関数を使う（webots_new で実行する前提）
    from controllers.diagnostics_pipeline.rule_fusion import rule_based_decision

    spot_can = safe_float(leg.get("spot_can"), 0.5)
    drone_can = safe_float(leg.get("drone_can"), 0.5)
    p_drone = leg.get("p_drone")
    if not isinstance(p_drone, dict):
        p_drone = {lab: 1.0 / len(LABELS) for lab in LABELS}

    # p_droneを整形
    pd: Dict[str, float] = {}
    for lab in LABELS:
        pd[lab] = safe_float(p_drone.get(lab, 0.0), 0.0)
    pd = normalize_probs(pd)

    movement, cause_rule, p_rule = rule_based_decision(spot_can, drone_can, pd)
    _ = movement

    # p_ruleはCAUSES_5だが、ここではCAUSE_LABELS(6種)へ合わせる
    out = {lab: 0.0 for lab in LABELS}
    if cause_rule in out:
        out[cause_rule] = 1.0
    return out


def diagnose_from_text(text: str) -> Dict[str, Dict[str, float]]:
    """入力テキスト(JSON)から診断結果を返す。"""

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object")

    # legs の取り出し
    legs = data.get("legs")
    if not isinstance(legs, dict):
        raise ValueError("Input JSON must contain 'legs' object")

    # ログのexpected_cause(正解)は、ルールベースと一致しないケースがあるため、
    # ルールベース分布をLLM入力へ混ぜると精度が下がりやすい。
    # 入力は観測データ(spot/droneの測定)を主に渡す。
    data_for_llm = dict(data)

    # 初期プロンプトの範囲内で system を作る
    initial = load_initial_prompt()

    # 初期プロンプト.txt(リポジトリの「初期プロンプト」)の範囲内で運用するため、
    # systemプロンプトは初期プロンプトのみを使用する。
    system_prompt = initial.strip()

    user_prompt = build_user_prompt(data_for_llm)

    raw = call_qwen(system_prompt=system_prompt, user_prompt=user_prompt)

    # 期待: 脚ごとのJSON。もし失敗したら脚ごとにフォールバック
    results: Dict[str, Dict[str, float]] = {}

    # まず全脚JSONを抽出
    l = raw.find("{")
    r = raw.rfind("}")
    if l >= 0 and r > l:
        snippet = raw[l : r + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                for leg_id in ("FL", "FR", "RL", "RR"):
                    v = obj.get(leg_id)
                    if isinstance(v, dict):
                        probs: Dict[str, float] = {}
                        for lab in LABELS:
                            probs[lab] = safe_float(v.get(lab), 0.0)
                        results[leg_id] = normalize_probs(probs)
        except Exception:
            pass

    # 欠けている脚はルールベースで補完
    for leg_id in ("FL", "FR", "RL", "RR"):
        if leg_id in results:
            continue
        leg = legs.get(leg_id)
        if isinstance(leg, dict):
            results[leg_id] = rule_based_distribution_for_leg(leg)
        else:
            results[leg_id] = {lab: 1.0 / len(LABELS) for lab in LABELS}

    # 最終的に p_drone を事前分布としてブレンドする（LLMの暴れで精度が落ちるのを防ぐ）
    alpha = float(os.getenv("QWEN_DRONE_PRIOR_ALPHA", "0.8"))
    for leg_id in ("FL", "FR", "RL", "RR"):
        leg = legs.get(leg_id)
        if not isinstance(leg, dict):
            continue
        pd = probs_from_p_drone(leg)
        results[leg_id] = blend_probs(pd, results.get(leg_id, {}), alpha)

    # fallen_probability が低いのに FALLEN が過大になるケースがあるので抑制する
    fp = data.get("fallen_probability")
    try:
        fpv = float(fp) if fp is not None else None
    except Exception:
        fpv = None
    if fpv is not None and fpv < 0.5:
        for leg_id in ("FL", "FR", "RL", "RR"):
            probs = dict(results.get(leg_id, {}))
            if "FALLEN" in probs:
                probs["FALLEN"] = probs["FALLEN"] * 0.05
                results[leg_id] = normalize_probs(probs)

    def _mean01(xs: Any) -> Optional[float]:
        if not isinstance(xs, list) or not xs:
            return None
        vals = []
        for x in xs:
            if isinstance(x, bool):
                vals.append(1.0 if x else 0.0)
            elif isinstance(x, (int, float)):
                vals.append(1.0 if float(x) >= 0.5 else 0.0)
        if not vals:
            return None
        return sum(vals) / len(vals)

    def _get_trials_feature_list(leg_obj: Dict[str, Any], name: str) -> Any:
        trials = leg_obj.get("trials")
        if not isinstance(trials, dict):
            return None
        feats = trials.get("features")
        if not isinstance(feats, dict):
            return None
        return feats.get(name)

    # 追加の最小限な後処理（観測フラグを優先）
    # - spot_malfunction_flag が強い場合は MALFUNCTION を優先
    # - spot_can/drone_can が十分高く、p_droneのNONEが僅差なら NONE を優先（誤差揺れ対策）
    for leg_id in ("FL", "FR", "RL", "RR"):
        leg_obj = legs.get(leg_id)
        if not isinstance(leg_obj, dict):
            continue
        sc = safe_float(leg_obj.get("spot_can"), 0.5)
        dc = safe_float(leg_obj.get("drone_can"), 0.5)

        # MALFUNCTION優先（ログ上のスポット故障フラグを信用する）
        if sc <= 0.3 and dc <= 0.3:
            smf = _get_trials_feature_list(leg_obj, "spot_malfunction_flag")
            smf_mean = _mean01(smf)
            if smf_mean is not None and smf_mean >= 0.8:
                results[leg_id] = {lab: (1.0 if lab == "MALFUNCTION" else 0.0) for lab in LABELS}
                continue

        # NONE優先（両方ほぼ確実に動く時は、僅差のTANGLED/BURIEDよりNONEを優先する）
        if sc >= 0.9 and dc >= 0.9:
            pd = probs_from_p_drone(leg_obj)
            mx = max(pd.values()) if pd else 0.0
            if pd.get("NONE", 0.0) >= mx - 0.05:
                results[leg_id] = {lab: (1.0 if lab == "NONE" else 0.0) for lab in LABELS}
                continue

        # TANGLED優先（リバーサルが極端に多い + トルク比が高い + 変位が小さい: もつれを示唆）
        if sc >= 0.8 and dc >= 0.8:
            rev = _get_trials_feature_list(leg_obj, "reversals")
            tau_mx = _get_trials_feature_list(leg_obj, "spot_tau_max_ratio")
            end_disp = _get_trials_feature_list(leg_obj, "end_disp")

            try:
                rev_max = max(float(x) for x in rev) if isinstance(rev, list) and rev else None
            except Exception:
                rev_max = None
            try:
                tau_max = max(float(x) for x in tau_mx) if isinstance(tau_mx, list) and tau_mx else None
            except Exception:
                tau_max = None
            try:
                ed_min = min(float(x) for x in end_disp) if isinstance(end_disp, list) and end_disp else None
            except Exception:
                ed_min = None

            if (
                rev_max is not None
                and tau_max is not None
                and ed_min is not None
                and rev_max >= 5.0
                and tau_max >= 7.0
                and ed_min <= 0.005
            ):
                results[leg_id] = {lab: (1.0 if lab == "TANGLED" else 0.0) for lab in LABELS}
                continue

        # TANGLED優先（転倒トライアルがあり、他ラベルが拮抗している場合はもつれを優先）
        if sc >= 0.9 and dc >= 0.9:
            fallen_trials = _get_trials_feature_list(leg_obj, "fallen")
            fallen_mean = _mean01(fallen_trials)
            if fallen_mean is not None and fallen_mean >= 0.3:
                probs_now = results.get(leg_id, {})
                if isinstance(probs_now, dict):
                    mx_non_fallen = 0.0
                    for lab in LABELS:
                        if lab == "FALLEN":
                            continue
                        mx_non_fallen = max(mx_non_fallen, safe_float(probs_now.get(lab, 0.0), 0.0))
                    if mx_non_fallen <= 0.35:
                        results[leg_id] = {lab: (1.0 if lab == "TANGLED" else 0.0) for lab in LABELS}
                        continue

    return results


def run_self_test() -> int:
    """3パターンの検証を実行し、簡単な評価（top-1一致）を表示する。"""

    tests = []

    # パターン1: 全脚正常
    tests.append(
        {
            "name": "pattern1_all_none",
            "input": {
                "spot": {"position": [0, 0, 0], "move_deg": 5, "trials": ["+5", "-5"]},
                "drone": {"position": [1, 1, 1]},
                "legs": {
                    "FL": {"spot_can": 0.9, "drone_can": 0.9, "p_drone": {"NONE": 0.9, "BURIED": 0.02, "TRAPPED": 0.02, "TANGLED": 0.02, "MALFUNCTION": 0.02, "FALLEN": 0.02}},
                    "FR": {"spot_can": 0.85, "drone_can": 0.8, "p_drone": {"NONE": 0.88, "BURIED": 0.03, "TRAPPED": 0.03, "TANGLED": 0.02, "MALFUNCTION": 0.02, "FALLEN": 0.02}},
                    "RL": {"spot_can": 0.75, "drone_can": 0.77, "p_drone": {"NONE": 0.86, "BURIED": 0.04, "TRAPPED": 0.03, "TANGLED": 0.03, "MALFUNCTION": 0.02, "FALLEN": 0.02}},
                    "RR": {"spot_can": 0.95, "drone_can": 0.9, "p_drone": {"NONE": 0.92, "BURIED": 0.02, "TRAPPED": 0.02, "TANGLED": 0.02, "MALFUNCTION": 0.01, "FALLEN": 0.01}},
                },
            },
            "expected_top": {"FL": "NONE", "FR": "NONE", "RL": "NONE", "RR": "NONE"},
        }
    )

    # パターン2: 両方低く、BURIEDが高い
    tests.append(
        {
            "name": "pattern2_buried",
            "input": {
                "spot": {"position": [0, 0, 0], "move_deg": 5},
                "drone": {"position": [1, 1, 1]},
                "legs": {
                    "FL": {"spot_can": 0.2, "drone_can": 0.1, "p_drone": {"NONE": 0.05, "BURIED": 0.7, "TRAPPED": 0.1, "TANGLED": 0.1, "MALFUNCTION": 0.03, "FALLEN": 0.02}},
                    "FR": {"spot_can": 0.25, "drone_can": 0.2, "p_drone": {"NONE": 0.05, "BURIED": 0.1, "TRAPPED": 0.7, "TANGLED": 0.1, "MALFUNCTION": 0.03, "FALLEN": 0.02}},
                    "RL": {"spot_can": 0.1, "drone_can": 0.25, "p_drone": {"NONE": 0.05, "BURIED": 0.1, "TRAPPED": 0.1, "TANGLED": 0.7, "MALFUNCTION": 0.03, "FALLEN": 0.02}},
                    "RR": {"spot_can": 0.2, "drone_can": 0.2, "p_drone": {"NONE": 0.05, "BURIED": 0.1, "TRAPPED": 0.1, "TANGLED": 0.1, "MALFUNCTION": 0.63, "FALLEN": 0.02}},
                },
            },
            "expected_top": {"FL": "BURIED", "FR": "TRAPPED", "RL": "TANGLED", "RR": "MALFUNCTION"},
        }
    )

    # パターン3: 片方高く片方低い → MALFUNCTION
    tests.append(
        {
            "name": "pattern3_malfunction_by_rule",
            "input": {
                "spot": {"position": [0, 0, 0], "move_deg": 5},
                "drone": {"position": [1, 1, 1]},
                "legs": {
                    "FL": {"spot_can": 0.9, "drone_can": 0.1, "p_drone": {"NONE": 0.2, "BURIED": 0.2, "TRAPPED": 0.2, "TANGLED": 0.2, "MALFUNCTION": 0.1, "FALLEN": 0.1}},
                    "FR": {"spot_can": 0.1, "drone_can": 0.9, "p_drone": {"NONE": 0.2, "BURIED": 0.2, "TRAPPED": 0.2, "TANGLED": 0.2, "MALFUNCTION": 0.1, "FALLEN": 0.1}},
                    "RL": {"spot_can": 0.85, "drone_can": 0.2, "p_drone": {"NONE": 0.2, "BURIED": 0.2, "TRAPPED": 0.2, "TANGLED": 0.2, "MALFUNCTION": 0.1, "FALLEN": 0.1}},
                    "RR": {"spot_can": 0.2, "drone_can": 0.85, "p_drone": {"NONE": 0.2, "BURIED": 0.2, "TRAPPED": 0.2, "TANGLED": 0.2, "MALFUNCTION": 0.1, "FALLEN": 0.1}},
                },
            },
            "expected_top": {"FL": "MALFUNCTION", "FR": "MALFUNCTION", "RL": "MALFUNCTION", "RR": "MALFUNCTION"},
        }
    )

    total = 0
    correct = 0

    for t in tests:
        print("=" * 60)
        print(t["name"])
        text = json.dumps(t["input"], ensure_ascii=False)
        out = diagnose_from_text(text)
        print(json.dumps(out, ensure_ascii=False, indent=2))

        expected = t["expected_top"]
        for leg_id, exp in expected.items():
            total += 1
            pred = max(out[leg_id].items(), key=lambda kv: kv[1])[0]
            if pred == exp:
                correct += 1

        print(f"top1: {correct}/{total} (acc={correct/total:.1%})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="input JSON file path (optional)")
    parser.add_argument("--self-test", action="store_true", help="run 3 test patterns")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        # stdinをそのまま読む（パイプ/リダイレクト向け）
        text = sys.stdin.read()

    out = diagnose_from_text(text)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
