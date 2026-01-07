from __future__ import annotations

import os
from pathlib import Path


def _chatml_prompt(system: str, user: str) -> str:
    # Minimal ChatML prompt. llama.cppのchat_format='chatml' を使うのが基本だが、
    # 万一 create_chat_completion が使えない場合のフォールバックとして用意。
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _resolve_model_path() -> Path:
    explicit = os.getenv("QWEN_GGUF_PATH", "").strip()
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"QWEN_GGUF_PATH not found: {p}")
        return p

    repo = os.getenv("QWEN_GGUF_REPO", "").strip()
    filename = os.getenv("QWEN_GGUF_FILENAME", "").strip()
    if not repo or not filename:
        raise SystemExit(
            "GGUFモデルが未指定です。\n"
            "- ローカルファイルを使う: QWEN_GGUF_PATH=/path/to/model.gguf\n"
            "- HuggingFaceから取得: QWEN_GGUF_REPO=... と QWEN_GGUF_FILENAME=... を指定\n"
        )

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(repo_id=repo, filename=filename)
    return Path(downloaded)


def main() -> None:
    model_path = _resolve_model_path()

    system = os.getenv("QWEN_SYSTEM", "You are a concise assistant.").strip()
    user = os.getenv("QWEN_PROMPT", "こんにちは。1文で自己紹介して。\n").strip()

    n_ctx = int(os.getenv("QWEN_CTX", "2048"))
    n_threads = int(os.getenv("QWEN_THREADS", str(os.cpu_count() or 4)))

    from llama_cpp import Llama

    llm = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=bool(int(os.getenv("QWEN_VERBOSE", "0"))),
    )

    # chat completion (推奨)
    if hasattr(llm, "create_chat_completion"):
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        print(out["choices"][0]["message"]["content"])
        return

    # fallback
    prompt = _chatml_prompt(system, user)
    out = llm(prompt, max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "128")), temperature=0.0)
    print(out["choices"][0]["text"])


if __name__ == "__main__":
    main()
