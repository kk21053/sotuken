"""Interface with lightweight instruction-tuned LLMs."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from . import config
from .models import LegState
from .utils import ensure_probability_keys

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover
    pipeline = None


CAUSE_DEFINITIONS = {
    "NONE": "脚は正常に動作しており障害がない状態",
    "BURIED": "脚が地面や砂に埋まっており大きく持ち上げられない状態",
    "TRAPPED": "関節は動くのに末端が障害物等に固定され前進できない状態",
    "ENTANGLED": "ツタなどに絡まり小さい往復でしか動けない状態",
}


class LLMAnalyzer:
    """Call an instruction-tuned model and parse JSON probabilities."""

    def __init__(
        self,
        model_priority: Optional[Iterable[str]] = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_priority = list(model_priority or config.DEFAULT_LLM_MODELS)
        self.max_new_tokens = max_new_tokens
        self._pipeline = None
        self._model_name: Optional[str] = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        if pipeline is None:
            raise RuntimeError("transformers is not installed")

        for name in self.model_priority:
            try:
                generator = pipeline(
                    "text-generation",
                    model=name,
                    tokenizer=name,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._pipeline = generator
                self._model_name = name
                print(f"[llm] model ready: {name}")
                return
            except Exception as exc:  # pragma: no cover
                print(f"[llm] failed to load {name}: {exc}")
        raise RuntimeError("no LLM model could be loaded")

    def _build_prompt(self, leg: LegState) -> str:
        lines: List[str] = []
        lines.append("あなたは四足ロボットの故障診断アシスタントです。")
        lines.append(
            "以下の情報から cause=NONE/BURIED/TRAPPED/ENTANGLED の確率を JSON で出力してください。"
        )
        lines.append(
            "必ず JSON のみを返し、説明文は書かないでください。確率の合計は必ず1.0にしてください。"
        )
        lines.append("")
        lines.append("### cause 定義")
        for key, message in CAUSE_DEFINITIONS.items():
            lines.append(f"- {key}: {message}")
        lines.append("")
        lines.append("### 自己診断まとめ")
        lines.append(f"self_can={leg.self_can:.3f}")
        lines.append(f"self_moves={str(leg.self_moves).lower()}")
        if leg.trials:
            last_trial = leg.trials[-1]
            lines.append(
                f"last_self_can_raw={last_trial.self_can_raw if last_trial.self_can_raw is not None else 'null'}"
            )
        lines.append("")
        lines.append("### ドローン観測まとめ")
        if leg.trials and leg.trials[-1].features_drone:
            for key, value in leg.trials[-1].features_drone.items():
                lines.append(f"{key}={value}")
        else:
            lines.append("no_drone_features=true")
        lines.append("")
        lines.append("### 出力形式")
        lines.append('{"prob":{"NONE":0.0,"BURIED":0.0,"TRAPPED":0.0,"ENTANGLED":0.0}}')
        return "\n".join(lines)

    def infer(self, leg: LegState) -> Dict[str, float]:
        try:
            payload = self._invoke_model(leg)
            distribution = self._parse_payload(payload)
        except Exception as exc:  # pragma: no cover
            print(f"[llm] inference failure: {exc}")
            uniform = 1.0 / len(config.CAUSE_LABELS)
            distribution = {label: uniform for label in config.CAUSE_LABELS}
        leg.p_llm = distribution
        return distribution

    def _invoke_model(self, leg: LegState) -> str:
        self._ensure_pipeline()
        prompt = self._build_prompt(leg)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if not outputs:
            raise RuntimeError("empty LLM output")
        text = outputs[0]["generated_text"].strip()
        return text

    def _parse_payload(self, payload: str) -> Dict[str, float]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            raise RuntimeError("LLM output is not valid JSON")
        if "prob" not in data:
            raise RuntimeError("missing 'prob' key")
        values = ensure_probability_keys(data["prob"], config.CAUSE_LABELS)
        return values
