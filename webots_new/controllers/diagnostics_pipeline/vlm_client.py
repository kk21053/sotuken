"""VLM による追加推定（確率分布を返す）

目的:
- 診断終了時に取得した画像1枚 + 数値情報から、拘束原因ラベルを推定
- `vlm_probs`（確率分布）と `vlm_pred`（argmaxラベル）をログに保存する

有効化:
- 環境変数 `VLM_ENABLE=1` のときのみ実行

実装方針:
- Moondream2 (transformers + trust_remote_code) を想定
- 依存が無い/モデル取得できない場合は安全にスキップ
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

from . import config
from .models import SessionState
from .rule_fusion import CAUSES_5, argmax_label, normalize_probs


_LABEL_SET = set(CAUSES_5)


def _extract_probs(text: str) -> Optional[Dict[str, float]]:
    """VLM出力から確率分布(JSON)を抽出する。

    期待形式: {"NONE":0.1, "BURIED":0.2, ...}
    フォールバック: ラベル単体出力の場合はone-hotとして扱う。
    """
    if not text:
        return None

    raw = str(text).strip()
    if not raw:
        return None

    # ラベル単体の出力を許容
    upper = raw.upper()
    m = re.search(r"\b(NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION)\b", upper)
    if m and "{" not in raw:
        lab = m.group(1)
        return {k: (1.0 if k == lab else 0.0) for k in CAUSES_5}

    # JSON部分だけ切り出し
    l = raw.find("{")
    r = raw.rfind("}")
    if l < 0 or r < 0 or r <= l:
        return None

    snippet = raw[l : r + 1]
    try:
        obj = json.loads(snippet)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    probs: Dict[str, float] = {}
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

    if not probs:
        return None

    return normalize_probs(probs, CAUSES_5)


@dataclass
class VLMConfig:
    enabled: bool
    model_id: str


class VLMAnalyzer:
    def __init__(self) -> None:
        self.cfg = VLMConfig(
            enabled=os.getenv("VLM_ENABLE", "0").strip() == "1",
            model_id=os.getenv("VLM_MODEL_ID", "vikhyatk/moondream2").strip(),
        )
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._tokenizer is not None:
            return True

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                model.to(device)
            except Exception:
                pass
            try:
                model.eval()
            except Exception:
                pass

            self._model = model
            self._tokenizer = tokenizer
            return True
        except Exception:
            return False

    def _load_image_rgb(self, image_path: str):
        """VLM用に画像を読み込み、脚周辺が入りやすいように軽く前処理する。"""
        try:
            from PIL import Image  # type: ignore

            img = Image.open(image_path).convert("RGB")

            # 正方形crop（縦長画像は下寄せにして足元を残す）
            w, h = img.size
            s = min(w, h)
            if w > h:
                left = max(0, (w - s) // 2)
                top = 0
            else:
                left = 0
                # 縦方向に余白がある場合、少し下側を優先（脚/地面が写りやすい）
                slack = h - s
                top = max(0, int(round(slack * 0.55)))
            img = img.crop((left, top, left + s, top + s))

            # 推論負荷を抑えつつ細部を残す
            try:
                img = img.resize((320, 320), resample=Image.BICUBIC)
            except Exception:
                img = img.resize((320, 320))

            return img
        except Exception:
            return None

    def infer_session(self, session: SessionState, leg_ids: Optional[Sequence[str]] = None) -> None:
        if not self.cfg.enabled:
            return
        if not session.image_path:
            return

        if not self._ensure_loaded():
            return

        img = self._load_image_rgb(session.image_path)
        if img is None:
            return

        try:
            enc = self._model.encode_image(img)
        except Exception:
            return

        target_leg_ids = list(leg_ids) if leg_ids is not None else list(session.legs.keys())

        # 画像1枚を全脚で共有しつつ、指定脚ごとに確率分布を出す
        for leg_id in target_leg_ids:
            leg = session.legs.get(leg_id)
            if leg is None:
                continue

            hints = (
                f"spot_can={getattr(leg, 'spot_can', 0.5):.2f} "
                f"drone_can={getattr(leg, 'drone_can', 0.5):.2f} "
                f"movement_result={getattr(leg, 'movement_result', '一部動く')}"
            )

            question = (
                "You are diagnosing the constraint cause of ONE specified leg of a quadruped robot in a single photo.\n"
                "The image contains ALL 4 legs. Judge ONLY the specified leg and ignore other legs.\n"
                f"Leg: {leg_id}\n"
                "Return a JSON object with probabilities for these labels (must sum to 1):\n"
                "NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION\n"
                "Definitions:\n"
                "- TRAPPED: the foot is clearly inside a trap device (metal jaws/plate).\n"
                "- TANGLED: a vine/rope is clearly wrapped around the leg or foot.\n"
                "- BURIED: a localized mound of sand covers the foot (plain sandy ground is still NONE).\n"
                "- NONE: normal/free leg and foot.\n"
                "- MALFUNCTION: the leg looks free but does not actuate; use hints if needed.\n"
                f"Hints: {hints}\n"
                "Output ONLY valid JSON like: {\"NONE\":0.1,\"BURIED\":0.2,\"TRAPPED\":0.2,\"TANGLED\":0.2,\"MALFUNCTION\":0.3}"
            )

            try:
                raw = self._model.answer_question(enc, question, self._tokenizer)
            except Exception:
                raw = ""

            probs = _extract_probs(str(raw) if raw is not None else "")
            if probs is None:
                # 最低限のフォールバック: ラベル抽出できる場合はone-hot、無理なら一様
                m = re.search(r"\b(NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION)\b", str(raw).upper())
                if m:
                    lab = m.group(1)
                    probs = {k: (1.0 if k == lab else 0.0) for k in CAUSES_5}
                else:
                    probs = {k: 1.0 / len(CAUSES_5) for k in CAUSES_5}

            probs = normalize_probs(probs, CAUSES_5)
            leg.vlm_probs = probs
            leg.vlm_pred = argmax_label(probs, CAUSES_5)
