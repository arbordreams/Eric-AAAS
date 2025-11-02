#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/medical"
RAW_DIR="$DATA_DIR/raw"
PROC_DIR="$DATA_DIR/processed"
BENCH_DIR="$DATA_DIR/benchmarks"
SPLITS_DIR="$DATA_DIR/splits"

mkdir -p "$RAW_DIR" "$PROC_DIR" "$BENCH_DIR" "$SPLITS_DIR"

echo "[MedTokAlign] Preparing datasets under $DATA_DIR"

# Optional SciSpacy model installation if URL provided
if [[ -n "${SCISPACY_MODEL_URL:-}" ]]; then
  echo "[MedTokAlign] Installing SciSpacy model from $SCISPACY_MODEL_URL"
  pip install --no-cache-dir "$SCISPACY_MODEL_URL" || true
fi

python - <<'PY'
import os, json
from datasets import load_dataset

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(root, "data", "medical")
proc_dir = os.path.join(data_dir, "processed")
bench_dir = os.path.join(data_dir, "benchmarks")
splits_dir = os.path.join(data_dir, "splits")
os.makedirs(proc_dir, exist_ok=True)
os.makedirs(bench_dir, exist_ok=True)
os.makedirs(splits_dir, exist_ok=True)

def save_jsonl(ds, path):
    with open(path, 'w', encoding='utf-8') as f:
        for ex in ds:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("[MedTokAlign] Downloading PubMedQA (pqa_labeled)...")
pubmedqa = load_dataset("pubmed_qa", "pqa_labeled")
for split in pubmedqa:
    save_jsonl(pubmedqa[split], os.path.join(bench_dir, f"pubmedqa_{split}.jsonl"))

print("[MedTokAlign] Downloading MedQA-USMLE 4-options...")
try:
    medqa = load_dataset("openlifescienceai/MedQA-USMLE-4-options")
except Exception:
    medqa = load_dataset("bigbio/med_qa_usmle")
for split in medqa:
    save_jsonl(medqa[split], os.path.join(bench_dir, f"medqa_{split}.jsonl"))

print("[MedTokAlign] Downloading MedNLI (bigbio, pairs)...")
mednli = load_dataset("bigbio/mednli", "pairs")
for split in mednli:
    save_jsonl(mednli[split], os.path.join(bench_dir, f"mednli_{split}.jsonl"))

print("[MedTokAlign] Downloading NCBI-Disease NER (bigbio_ner)...")
ncbi = load_dataset("bigbio/ncbi_disease", "bigbio_ner")
for split in ncbi:
    save_jsonl(ncbi[split], os.path.join(bench_dir, f"ncbi_disease_{split}.jsonl"))

print("[MedTokAlign] Downloading BC5CDR NER (bigbio_ner)...")
bc5 = load_dataset("bigbio/bc5cdr", "bigbio_ner")
for split in bc5:
    save_jsonl(bc5[split], os.path.join(bench_dir, f"bc5cdr_{split}.jsonl"))

print("[MedTokAlign] Downloading PubMed RCT abstracts for perplexity...")
try:
    rct = load_dataset("bigbio/pubmed_rct", split="test")
    save_jsonl(rct, os.path.join(proc_dir, "pubmed_rct_test.jsonl"))
except Exception:
    pass

print("[MedTokAlign] Done.")
PY

echo "[MedTokAlign] Data preparation completed."


