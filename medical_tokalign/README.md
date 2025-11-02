Medical TokAlign for Llama 3 on RunPod (H100, Single GPU)

Overview
- A self-contained tokenizer-alignment extension for medical domains, optimized for one NVIDIA H100 PCIe 80GB on RunPod.
- Implements a TokAlign‑style approach: builds a term↔token-bundle alignment matrix from robust medical corpora and lexicons, selects high-benefit medical terms to add to the tokenizer, resizes model embeddings, and warm‑initializes new rows.
- Default target: meta-llama/Meta-Llama-3.1-8B. Backend: vLLM by default; HF fallback; optional TensorRT-LLM stub.

Key Features
- Full adaptation: add medical tokens + resize model embeddings; optional short CLM warmup for new rows only.
- H100-tuned performance: BF16, FlashAttention-2, torch.compile, TF32 allowed, paged attention (vLLM), CUDA memory expandable segments.
- Robust datasets via Hugging Face standard splits (no full-corpus eval): PubMedQA, MedQA-USMLE, MedNLI, NCBI-Disease, BC5CDR; separate perplexity corpus from PubMed RCT.
- One-command evaluation producing metrics.json, per-dataset CSVs, and samples.md.

Directory Layout
```
medical_tokalign/
  README.md
  requirements.txt
  configs/
    eval_medical.yaml
    runpod_env.example
  scripts/
    prepare_medical_data.sh
    run_vocab_adaptation.sh
    eval_medical.sh
  src/
    __init__.py
    datasets_medical.py
    metrics_medical.py
    llama_utils.py
    perf_utils.py
    medical_eval.py
    alignment_core.py
    alignment_runner.py
  notebooks/
    MedTokAlign_RunPod.ipynb
  runs/  # auto-created for artifacts
```

RunPod Quickstart
1) Launch container (recommended):
   - Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
   - Hardware: One H100 PCIe (80GB)

2) Environment (set once per session):
```
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/.cache/huggingface
# If the model is gated, set your token:
export HF_TOKEN=hf_...
```

3) Install
```
# Torch 2.8.0 and CUDA 12.8.1 come preinstalled in this image.
# Use the bootstrap script to install Python deps and build GloVe tools.
bash medical_tokalign/scripts/bootstrap_runpod.sh
```

4) Prepare data (downloads/caches standard HF splits locally)
```
bash medical_tokalign/scripts/prepare_medical_data.sh
```

5) Run tokenizer adaptation (TokAlign‑style, adds tokens + resizes embeddings)
```
bash medical_tokalign/scripts/run_vocab_adaptation.sh \
  --model_id meta-llama/Meta-Llama-3.1-8B \
  --top_k 8192
```
Artifacts are saved under `medical_tokalign/runs/tokenizer_adapt/<timestamp>/` and include:
- tokenizer/ (adapted tokenizer)
- model/ (adapted HF model dir)
- align_matrix.npz (sparse term↔bundle co-occurrence)
- indices.json (term and bundle indices)
- added_terms.txt, report.md

6) Evaluate (vLLM if available; auto‑fallback to HF)
```
bash medical_tokalign/scripts/eval_medical.sh --config medical_tokalign/configs/eval_medical.yaml
```
Outputs under `medical_tokalign/runs/medical_eval/<timestamp>/`:
- metrics.json (aggregate)
- per-dataset CSVs (predictions, references)
- samples.md (qualitative samples)

TokAlign‑style Adaptation Summary
- Build alignment matrix A between medical terms and observed token bundles from the base tokenizer on medical corpora.
- Score each term by expected fragmentation reduction × frequency × boundary consistency.
- Select top‑K terms not already atomic; add to tokenizer; resize model embeddings.
- Initialize new embeddings as weighted averages of the most frequent covering subpieces, add small noise; optionally warm up only new rows with a brief CLM step on medical text.

H100‑Optimized Defaults
- precision: bf16
- allow_tf32: true, matmul_precision: high
- attn_implementation: flash_attention_2 (falls back to SDPA if Flash‑Attn unavailable)
- torch.compile: true (inductor, autotune)
- Inference backend: vLLM with paged attention
- vLLM sizing (80GB): gpu_memory_utilization≈0.92, max_batch_tokens≈65536, max_model_len=8192

Datasets & Splits (standard HF sources)
- PubMedQA: `pubmed_qa` (config `pqa_labeled`) — accuracy (test/dev)
- MedQA-USMLE: `openlifescienceai/MedQA-USMLE-4-options` (fallback `bigbio/med_qa_usmle`) — accuracy (test)
- MedNLI: `bigbio/mednli` (task `pairs`) — accuracy (test)
- NCBI-Disease NER: `bigbio/ncbi_disease` (config `bigbio_ner`) — span‑level F1 micro/macro (test)
- BC5CDR NER: `bigbio/bc5cdr` (config `bigbio_ner`) — span‑level F1 (test)
- Perplexity corpus: PubMed RCT abstracts (prefer `bigbio/pubmed_rct`) — perplexity via chunked teacher‑forcing

Llama 3 Variants
- Default: `meta-llama/Meta-Llama-3.1-8B`
- You can switch models in `configs/eval_medical.yaml` via `model_id`.
  Ensure HF access is granted (set `HF_TOKEN` if gated).

Notes on FP8 (TransformerEngine)
- The stack defaults to BF16 for stability. FP8 (TransformerEngine) hooks can be enabled in advanced scenarios; requirements include matching CUDA/TE wheels.

Troubleshooting
- OOM: lower `max_batch_tokens` (vLLM) or `per_device_batch_size` (HF); reduce `max_model_len`.
- Gated models: export `HF_TOKEN` and ensure it has access to the Llama weights.
- SciSpacy model: if auto-download fails, set `SCISPACY_MODEL_URL` to a direct wheel URL and rerun the data script.
- vLLM wheels: on fresh Torch/CUDA combos vLLM may be unavailable. The pipeline will fall back to HF automatically. You can also set `eval_backend: hf` in `configs/eval_medical.yaml` to force HF.

Citation & Licenses
- Datasets and model licenses apply. Ensure you comply with source licenses (e.g., BigBio, PubMedQA, MedQA, MedNLI, NCBI, BC5CDR).


