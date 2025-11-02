import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .perf_utils import configure_torch, maybe_compile


@dataclass
class AdaptedArtifacts:
    model_dir: Optional[str]
    tokenizer_dir: Optional[str]
    report_path: Optional[str]


def find_latest_adapted_artifacts(search_dir: str) -> Optional[AdaptedArtifacts]:
    if not os.path.isdir(search_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(search_dir, "*")))
    candidates = [d for d in candidates if os.path.isdir(d)]
    if not candidates:
        return None
    latest = candidates[-1]
    model_dir = os.path.join(latest, "model")
    tokenizer_dir = os.path.join(latest, "tokenizer")
    report = os.path.join(latest, "report.md")
    return AdaptedArtifacts(
        model_dir=model_dir if os.path.isdir(model_dir) else None,
        tokenizer_dir=tokenizer_dir if os.path.isdir(tokenizer_dir) else None,
        report_path=report if os.path.isfile(report) else None,
    )


def load_hf_model_and_tokenizer(
    model_id_or_path: str,
    precision: str = "bf16",
    attn_impl: str = "flash_attention_2",
    compile_model: bool = True,
    grad_ckpt: bool = False,
    allow_tf32: bool = True,
    max_model_len: int = 8192,
    tokenizer_dir_override: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    configure_torch(allow_tf32=allow_tf32, matmul_precision="high")

    dtype = torch.bfloat16 if precision.lower() == "bf16" else torch.float16

    tokenizer_path = tokenizer_dir_override or model_id_or_path
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    cfg = AutoConfig.from_pretrained(model_id_or_path)
    try:
        cfg.attn_implementation = attn_impl
    except Exception:
        pass
    try:
        cfg.max_position_embeddings = max_model_len
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=False,
    )

    if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model = maybe_compile(model, enabled=compile_model)
    return model, tok


def load_vllm_engine(
    model: str,
    tokenizer: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 8192,
    enforce_eager: bool = False,
    trust_remote_code: bool = False,
):
    try:
        from vllm import LLM
        from vllm.sampling_params import SamplingParams
    except Exception as e:
        raise RuntimeError("vLLM is not installed or failed to import") from e

    engine = LLM(
        model=model,
        tokenizer=tokenizer or model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
    )
    return engine


def vllm_generate(engine, prompts, max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0, stop=None):
    from vllm.sampling_params import SamplingParams
    stop = stop or []
    params = SamplingParams(
        max_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        stop=stop,
    )
    outputs = engine.generate(prompts, params)
    # vLLM returns a list of RequestOutput, each with .outputs list
    texts = []
    for out in outputs:
        if out.outputs:
            texts.append(out.outputs[0].text)
        else:
            texts.append("")
    return texts


def hf_generate(model, tokenizer, prompts, max_new_tokens=64, temperature=0.0, top_p=1.0, top_k=0, stop=None, batch_size=4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = []
    stop = stop or []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tok = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            gen_ids = model.generate(
                **tok,
                max_new_tokens=int(max_new_tokens),
                do_sample=(temperature > 0.0),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
            )
            texts = tokenizer.batch_decode(gen_ids[:, tok.input_ids.shape[1]:], skip_special_tokens=True)
            # naive stop handling
            clipped = []
            for t in texts:
                for s in stop:
                    if s in t:
                        t = t.split(s)[0]
                clipped.append(t)
            results.extend(clipped)
    return results


