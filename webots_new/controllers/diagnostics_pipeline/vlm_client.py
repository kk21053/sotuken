"""VLM による追加推定（cause_final には反映しない）

目的:
- 診断終了時に取得した画像1枚 + 数値情報から、拘束原因ラベルを推定
- まずは `vlm_pred` としてログに保存し、既存の `cause_final` は維持する

有効化:
- 環境変数 `VLM_ENABLE=1` のときのみ実行

実装方針:
- Moondream2 (transformers + trust_remote_code) を想定
- 依存が無い/モデル取得できない場合は安全にスキップ
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

from . import config
from .models import SessionState


_LABEL_SET = set(config.CAUSE_LABELS)


def _extract_label(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()

    # 先頭トークン or "LABEL: XXX" などを許容
    m = re.search(r"\b(NONE|BURIED|TRAPPED|TANGLED|MALFUNCTION|FALLEN)\b", t)
    if not m:
        return None
    label = m.group(1)
    return label if label in _LABEL_SET else None


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

    def _answer_question(self, image_path: str, question: str) -> Optional[str]:
        if not self._ensure_loaded():
            return None

        try:
            from PIL import Image  # type: ignore

            img = Image.open(image_path).convert("RGB")

            # Moondream2 の trust_remote_code API を想定
            enc = self._model.encode_image(img)
            ans = self._model.answer_question(enc, question, self._tokenizer)
            return str(ans)
        except Exception:
            return None

    def infer_session(self, session: SessionState) -> None:
        if not self.cfg.enabled:
            return
        if not session.image_path:
            return

        # 画像1枚を全脚で共有しつつ、脚ごとにラベルを出す（最小実装）
        for leg_id, leg in session.legs.items():
            question = (
                "You are classifying the constraint cause of Spot's leg.\n"
                f"Leg: {leg_id}\n"
                "Choose exactly one label from: NONE, BURIED, TRAPPED, TANGLED, MALFUNCTION, FALLEN.\n"
                f"fallen={session.fallen} fallen_probability={session.fallen_probability:.2f}\n"
                f"spot_can={leg.spot_can:.2f} drone_can={leg.drone_can:.2f}\n"
                f"movement_result={leg.movement_result}\n"
                "Return only the label."
            )

            raw = self._answer_question(session.image_path, question)
            leg.vlm_pred = _extract_label(raw or "")
