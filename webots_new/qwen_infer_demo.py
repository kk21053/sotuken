from __future__ import annotations

import os


def main() -> None:
    model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct").strip()
    prompt = os.getenv("QWEN_PROMPT", "脚の異常原因を推定してください。出力は短く。\n").strip()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    use_cuda = bool(torch.cuda.is_available())

    if not use_cuda and not bool(int(os.getenv("QWEN_ALLOW_CPU", "0"))):
        raise SystemExit(
            "CUDAが無い環境では、transformers経由のQwen実行は(4bitにならず)大容量DLで重くなりがちです。\n"
            "CPUで4bit相当(GGUF)で動かす: webots_new/qwen_gguf_demo.py を使ってください。\n"
            "どうしてもCPUでtransformers実行したい場合: QWEN_ALLOW_CPU=1 を指定してください。\n"
        )

    quantization_config = None
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "auto" if use_cuda else {"": "cpu"}

    if use_cuda:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception:
            quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids

    inputs = inputs.to(model.device)

    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=int(os.getenv("QWEN_MAX_NEW_TOKENS", "128")),
            do_sample=False,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
