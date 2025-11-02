from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple
import math

import torch
import yaml

from .datasets_medical import (
    load_pubmedqa,
    load_medqa,
    load_mednli,
    load_ncbi_disease,
    load_bc5cdr,
    load_perplexity_corpus,
    format_pubmedqa_prompt,
    format_medqa_prompt,
    format_mednli_prompt,
    extract_ner_gold_spans,
    format_ner_prompt,
)
from .llama_utils import (
    find_latest_adapted_artifacts,
    hf_generate,
    load_hf_model_and_tokenizer,
    load_vllm_engine,
    vllm_generate,
)
from .metrics_medical import accuracy, extract_letter, ner_span_f1, perplexity


def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="MedTokAlign - Medical Evaluation Orchestrator")
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = cfg.get("output_dir", "medical_tokalign/runs/medical_eval")
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, ts)
    ensure_dir(run_dir)

    # Discover adapted artifacts
    adapted = None
    if cfg.get("alignment", {}).get("prefer_adapted_artifacts", True):
        search_dir = cfg.get("alignment", {}).get("search_dir", "medical_tokalign/runs/tokenizer_adapt")
        adapted = find_latest_adapted_artifacts(search_dir)

    model_id = cfg.get("model_id")
    backend = cfg.get("eval_backend", "vllm").lower()
    precision = cfg.get("precision", "bf16")
    attn_impl = cfg.get("attn_impl", "flash_attention_2")
    compile_flag = bool(cfg.get("compile", True))
    grad_ckpt = bool(cfg.get("grad_ckpt", False))
    allow_tf32 = bool(cfg.get("allow_tf32", True))

    # Load model engine
    engine = None
    hf_model = None
    hf_tok = None
    if backend == "vllm":
        vllm_cfg = cfg.get("vllm", {})
        try:
            if adapted and adapted.model_dir and adapted.tokenizer_dir:
                engine = load_vllm_engine(
                    model=adapted.model_dir,
                    tokenizer=adapted.tokenizer_dir,
                    tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
                    gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.92)),
                    max_model_len=int(vllm_cfg.get("max_model_len", 8192)),
                    enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
                    trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
                )
            else:
                engine = load_vllm_engine(
                    model=model_id,
                    tokenizer=None,
                    tensor_parallel_size=int(vllm_cfg.get("tensor_parallel_size", 1)),
                    gpu_memory_utilization=float(vllm_cfg.get("gpu_memory_utilization", 0.92)),
                    max_model_len=int(vllm_cfg.get("max_model_len", 8192)),
                    enforce_eager=bool(vllm_cfg.get("enforce_eager", False)),
                    trust_remote_code=bool(vllm_cfg.get("trust_remote_code", False)),
                )
        except Exception as e:
            print(f"[MedTokAlign] vLLM unavailable or failed to initialize ({e}). Falling back to HF backend.")
            backend = "hf"
    else:
        # HF path
        model_path = adapted.model_dir if (adapted and adapted.model_dir) else model_id
        tokenizer_override = adapted.tokenizer_dir if (adapted and adapted.tokenizer_dir) else None
        hf_model, hf_tok = load_hf_model_and_tokenizer(
            model_id_or_path=model_path,
            precision=precision,
            attn_impl=attn_impl,
            compile_model=compile_flag,
            grad_ckpt=grad_ckpt,
            allow_tf32=allow_tf32,
            max_model_len=int(cfg.get("hf", {}).get("max_model_len", 8192)),
            tokenizer_dir_override=tokenizer_override,
        )

    # If we attempted vLLM but fell back, ensure HF is loaded
    if backend != "vllm" and hf_model is None:
        model_path = adapted.model_dir if (adapted and adapted.model_dir) else model_id
        tokenizer_override = adapted.tokenizer_dir if (adapted and adapted.tokenizer_dir) else None
        hf_model, hf_tok = load_hf_model_and_tokenizer(
            model_id_or_path=model_path,
            precision=precision,
            attn_impl=attn_impl,
            compile_model=compile_flag,
            grad_ckpt=grad_ckpt,
            allow_tf32=allow_tf32,
            max_model_len=int(cfg.get("hf", {}).get("max_model_len", 8192)),
            tokenizer_dir_override=tokenizer_override,
        )

    # Paths
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medical")
    bench_dir = os.path.join(data_root, "benchmarks")
    proc_dir = os.path.join(data_root, "processed")

    # Evaluation
    metrics: Dict[str, float] = {}
    samples: List[str] = []
    per_dataset_outputs: Dict[str, List[Dict[str, str]]] = {}

    def do_generate(prompts: List[str]) -> List[str]:
        gen = cfg.get("gen", {})
        if backend == "vllm":
            return vllm_generate(
                engine,
                prompts,
                max_new_tokens=int(gen.get("max_new_tokens", 64)),
                temperature=float(gen.get("temperature", 0.0)),
                top_p=float(gen.get("top_p", 1.0)),
                top_k=int(gen.get("top_k", 0)),
                stop=gen.get("stop", []) or [],
            )
        else:
            return hf_generate(
                hf_model,
                hf_tok,
                prompts,
                max_new_tokens=int(gen.get("max_new_tokens", 64)),
                temperature=float(gen.get("temperature", 0.0)),
                top_p=float(gen.get("top_p", 1.0)),
                top_k=int(gen.get("top_k", 0)),
                stop=gen.get("stop", []) or [],
                batch_size=int(cfg.get("hf", {}).get("per_device_batch_size", 4)),
            )

    # PubMedQA
    if cfg.get("datasets", {}).get("pubmedqa", True):
        ds = load_pubmedqa(bench_dir)
        test = ds.get("test") or []
        prompts, labels = [], []
        for ex in test:
            p, y = format_pubmedqa_prompt(ex)
            prompts.append(p); labels.append(y)
        preds = do_generate(prompts)
        # map to yes/no/maybe
        preds_mapped = []
        for t in preds:
            low = t.strip().lower()
            mapped = "maybe"
            if low.startswith("yes"):
                mapped = "yes"
            elif low.startswith("no"):
                mapped = "no"
            elif low.startswith("maybe"):
                mapped = "maybe"
            preds_mapped.append(mapped)
        acc = accuracy(preds_mapped, labels)
        metrics["pubmedqa_acc"] = acc
        per_dataset_outputs["pubmedqa"] = [{"prompt": pr, "pred": pm, "label": y} for pr, pm, y in zip(prompts, preds_mapped, labels)]
        samples.append(f"PubMedQA: acc={acc:.3f}")

    # MedQA (USMLE)
    if cfg.get("datasets", {}).get("medqa_usmle", True):
        ds = load_medqa(bench_dir)
        test = ds.get("test") or []
        prompts, labels = [], []
        for ex in test:
            p, y = format_medqa_prompt(ex)
            prompts.append(p); labels.append(y)
        preds = do_generate(prompts)
        letters = [extract_letter(t) for t in preds]
        acc = accuracy(letters, labels)
        metrics["medqa_acc"] = acc
        per_dataset_outputs["medqa"] = [{"prompt": pr, "pred": le, "label": y} for pr, le, y in zip(prompts, letters, labels)]
        samples.append(f"MedQA: acc={acc:.3f}")

    # MedNLI
    if cfg.get("datasets", {}).get("mednli", True):
        ds = load_mednli(bench_dir)
        test = ds.get("test") or []
        prompts, labels = [], []
        for ex in test:
            p, y = format_mednli_prompt(ex)
            prompts.append(p); labels.append(y)
        preds = do_generate(prompts)
        preds_mapped = []
        for t in preds:
            low = t.strip().lower()
            if low.startswith("entailment"):
                preds_mapped.append("entailment")
            elif low.startswith("contradiction"):
                preds_mapped.append("contradiction")
            else:
                preds_mapped.append("neutral")
        acc = accuracy(preds_mapped, labels)
        metrics["mednli_acc"] = acc
        per_dataset_outputs["mednli"] = [{"prompt": pr, "pred": pm, "label": y} for pr, pm, y in zip(prompts, preds_mapped, labels)]
        samples.append(f"MedNLI: acc={acc:.3f}")

    # NER: NCBI-Disease
    if cfg.get("datasets", {}).get("ncbi_disease", True):
        ds = load_ncbi_disease(bench_dir)
        test = ds.get("test") or []
        prompts, gold_lists = [], []
        for ex in test:
            txt = ex.get("text") or ex.get("document") or ex.get("passages") or ""
            if isinstance(txt, list):
                txt = "\n".join(txt)
            prompts.append(format_ner_prompt(str(txt), entity_type="Disease"))
            gold_lists.append(extract_ner_gold_spans(ex))
        preds = do_generate(prompts)
        pred_lists = [list(filter(None, (ln.strip() for ln in p.splitlines()))) for p in preds]
        ner_scores = ner_span_f1(pred_lists, gold_lists)
        for k, v in ner_scores.items():
            metrics[f"ncbi_disease_{k}"] = float(v)
        per_dataset_outputs["ncbi_disease"] = [{"prompt": pr, "pred": pred_lists[i], "gold": gold_lists[i]} for i, pr in enumerate(prompts)]
        samples.append(f"NCBI-Disease: micro_f1={ner_scores['micro_f1']:.3f}")

    # NER: BC5CDR
    if cfg.get("datasets", {}).get("bc5cdr", True):
        ds = load_bc5cdr(bench_dir)
        test = ds.get("test") or []
        prompts, gold_lists = [], []
        for ex in test:
            txt = ex.get("text") or ex.get("document") or ex.get("passages") or ""
            if isinstance(txt, list):
                txt = "\n".join(txt)
            prompts.append(format_ner_prompt(str(txt), entity_type="Chemical or Disease"))
            gold_lists.append(extract_ner_gold_spans(ex))
        preds = do_generate(prompts)
        pred_lists = [list(filter(None, (ln.strip() for ln in p.splitlines()))) for p in preds]
        ner_scores = ner_span_f1(pred_lists, gold_lists)
        for k, v in ner_scores.items():
            metrics[f"bc5cdr_{k}"] = float(v)
        per_dataset_outputs["bc5cdr"] = [{"prompt": pr, "pred": pred_lists[i], "gold": gold_lists[i]} for i, pr in enumerate(prompts)]
        samples.append(f"BC5CDR: micro_f1={ner_scores['micro_f1']:.3f}")

    # Perplexity (HF path, teacher forcing)
    ppl_src = cfg.get("datasets", {}).get("perplexity_corpus", "pubmed_rct")
    texts = load_perplexity_corpus(proc_dir, source=ppl_src)
    if texts:
        # HF model is required for token-level loss
        if hf_model is None:
            # load minimally if engine only
            hf_model, hf_tok = load_hf_model_and_tokenizer(
                model_id_or_path=adapted.model_dir if (adapted and adapted.model_dir) else model_id,
                precision=precision,
                attn_impl=attn_impl,
                compile_model=False,
                grad_ckpt=False,
                allow_tf32=True,
                max_model_len=int(cfg.get("hf", {}).get("max_model_len", 8192)),
                tokenizer_dir_override=adapted.tokenizer_dir if (adapted and adapted.tokenizer_dir) else None,
            )
        device = next(hf_model.parameters()).device
        hf_model.eval()
        nlls = []
        with torch.no_grad():
            for doc in texts:
                enc = hf_tok(doc, return_tensors="pt").to(device)
                out = hf_model(**enc, labels=enc.input_ids)
                nll = float(out.loss.detach().cpu())
                if not math.isnan(nll):
                    nlls.append(nll)
        metrics["perplexity"] = float(perplexity(nlls))
        samples.append(f"PPL: {metrics['perplexity']:.2f}")

    # Save outputs
    save_json(metrics, os.path.join(run_dir, "metrics.json"))
    for name, rows in per_dataset_outputs.items():
        path = os.path.join(run_dir, f"{name}.json")
        save_json(rows, path)
    with open(os.path.join(run_dir, "samples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(samples))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


