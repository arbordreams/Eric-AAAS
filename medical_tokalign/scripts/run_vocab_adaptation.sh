#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
DATA_DIR="$ROOT_DIR/data/medical"
BENCH_DIR="$DATA_DIR/benchmarks"
PROC_DIR="$DATA_DIR/processed"

MODEL_ID="meta-llama/Llama-3.2-8B"
TOP_K=8192
PIVOT=300
GLOVE_DIR="$ROOT_DIR/tools/GloVe"

usage() {
  echo "Usage: $0 [--model_id MODEL] [--top_k N] [--pivot N]" >&2
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --model_id) MODEL_ID="$2"; shift 2;;
    --top_k) TOP_K="$2"; shift 2;;
    --pivot) PIVOT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$ROOT_DIR/runs/tokenizer_adapt/$TS"
TOK_DIR="$RUN_DIR/tokenizer"
MODEL_OUT_DIR="$RUN_DIR/model"
ALIGN_JSON="$RUN_DIR/align_matrix.json"
GOLD_JSON="$RUN_DIR/gold.json"
CORPUS_SRC="$RUN_DIR/glove_source.txt"
CORPUS_TGT="$RUN_DIR/glove_target.txt"
VEC_SRC="$RUN_DIR/vec-source.txt"
VEC_TGT="$RUN_DIR/vec-target.txt"

mkdir -p "$RUN_DIR" "$TOK_DIR"

echo "[MedTokAlign] Building medical-extended tokenizer (top_k=$TOP_K) ..."
python - <<PY
import os, json, re
from collections import Counter
from transformers import AutoTokenizer

root = "$ROOT_DIR"
bench_dir = "$BENCH_DIR"
proc_dir = "$PROC_DIR"
tok_out = "$TOK_DIR"
model_id = "$MODEL_ID"
top_k = int("$TOP_K")

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

def load_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except FileNotFoundError:
        return

texts = []
for name in [
    'pubmedqa_train.jsonl', 'pubmedqa_validation.jsonl', 'pubmedqa_test.jsonl',
    'medqa_train.jsonl', 'medqa_validation.jsonl', 'medqa_test.jsonl',
    'mednli_train.jsonl', 'mednli_validation.jsonl', 'mednli_test.jsonl',
    'ncbi_disease_train.jsonl', 'ncbi_disease_validation.jsonl', 'ncbi_disease_test.jsonl',
    'bc5cdr_train.jsonl', 'bc5cdr_validation.jsonl', 'bc5cdr_test.jsonl',
]:
    for ex in load_jsonl(os.path.join(bench_dir, name)) or []:
        for k in ('text','document','passages','context','long_answer','premise','hypothesis','question'):
            v = ex.get(k)
            if isinstance(v, list):
                v = "\n".join(map(str, v))
            if isinstance(v, str):
                texts.append(v)

rct = os.path.join(proc_dir, 'pubmed_rct_test.jsonl')
for ex in load_jsonl(rct) or []:
    v = ex.get('text') or ex.get('abstract') or ex.get('sentence')
    if isinstance(v, list): v = "\n".join(map(str, v))
    if isinstance(v, str): texts.append(v)

freq = Counter()
token_re = re.compile(r"[A-Za-z][A-Za-z0-9_\-/]{3,}")
for t in texts:
    for w in token_re.findall(t):
        if len(w) <= 64:
            freq[w] += 1

existing = set(tok.get_vocab().keys())
new_terms = [w for w,_ in freq.most_common(top_k*3) if w not in existing]
new_terms = new_terms[:top_k]

added = tok.add_tokens(new_terms, special_tokens=False)
tok.save_pretrained(tok_out)
print(f"Added {added} tokens (requested {top_k}). Saved to {tok_out}")
PY

echo "[MedTokAlign] Building GloVe corpora ..."
python - <<PY
import os, json
from transformers import AutoTokenizer

bench_dir = "$BENCH_DIR"
proc_dir = "$PROC_DIR"
model_id = "$MODEL_ID"
tok_tgt_dir = "$TOK_DIR"
corpus_src = "$CORPUS_SRC"
corpus_tgt = "$CORPUS_TGT"

tok_src = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tok_tgt = AutoTokenizer.from_pretrained(tok_tgt_dir, use_fast=True)

def load_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except FileNotFoundError:
        return

def iter_texts():
    files = [
        'pubmedqa_train.jsonl','pubmedqa_validation.jsonl','pubmedqa_test.jsonl',
        'medqa_train.jsonl','medqa_validation.jsonl','medqa_test.jsonl',
        'mednli_train.jsonl','mednli_validation.jsonl','mednli_test.jsonl',
        'ncbi_disease_train.jsonl','ncbi_disease_validation.jsonl','ncbi_disease_test.jsonl',
        'bc5cdr_train.jsonl','bc5cdr_validation.jsonl','bc5cdr_test.jsonl',
    ]
    for name in files:
        for ex in load_jsonl(os.path.join(bench_dir, name)) or []:
            for k in ('text','document','passages','context','long_answer','premise','hypothesis','question'):
                v = ex.get(k)
                if isinstance(v, list):
                    v = "\n".join(map(str, v))
                if isinstance(v, str):
                    yield v
    for ex in load_jsonl(os.path.join(proc_dir, 'pubmed_rct_test.jsonl')) or []:
        v = ex.get('text') or ex.get('abstract') or ex.get('sentence')
        if isinstance(v, list): v = "\n".join(map(str, v))
        if isinstance(v, str): yield v

with open(corpus_src, 'w') as fs, open(corpus_tgt, 'w') as ft:
    for t in iter_texts():
        ids_s = tok_src(t, add_special_tokens=False, truncation=True, max_length=8192)['input_ids']
        ids_t = tok_tgt(t, add_special_tokens=False, truncation=True, max_length=8192)['input_ids']
        if len(ids_s) >= 15:
            fs.write(" ".join(map(str, ids_s)) + "\n")
        if len(ids_t) >= 15:
            ft.write(" ".join(map(str, ids_t)) + "\n")
print(f"Wrote corpora: {corpus_src}, {corpus_tgt}")
PY

echo "[MedTokAlign] Ensuring Stanford GloVe is available ..."
if [[ ! -d "$GLOVE_DIR" ]]; then
  mkdir -p "$(dirname "$GLOVE_DIR")"
  git clone https://github.com/stanfordnlp/GloVe.git "$GLOVE_DIR"
fi

echo "[MedTokAlign] Training GloVe vectors ..."
cd "$GLOVE_DIR"

NAME_SRC=$(basename "$VEC_SRC")
NAME_SRC="${NAME_SRC%.*}"
echo "Training source GloVe -> $NAME_SRC"
bash "$ROOT_DIR/scripts/train_glove.sh" "$CORPUS_SRC" "$NAME_SRC"
mv "${NAME_SRC}.txt" "$VEC_SRC"

NAME_TGT=$(basename "$VEC_TGT")
NAME_TGT="${NAME_TGT%.*}"
echo "Training target GloVe -> $NAME_TGT"
bash "$ROOT_DIR/scripts/train_glove.sh" "$CORPUS_TGT" "$NAME_TGT"
mv "${NAME_TGT}.txt" "$VEC_TGT"

cd "$ROOT_DIR"

echo "[MedTokAlign] Building gold overlap dict ..."
python "$SRC_DIR/count_dict.py" \
  -s "$MODEL_ID" \
  -t "$TOK_DIR" \
  -o "$GOLD_JSON"

echo "[MedTokAlign] Getting vocab sizes ..."
VOCAB_SRC=$(python "$SRC_DIR/count_vocab.py" -m "$MODEL_ID")
VOCAB_TGT=$(python - <<PY
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("$TOK_DIR")
print(len(tok))
PY
)

echo "[MedTokAlign] Computing 1-1 alignment (relative rep, pivots=$PIVOT) ..."
python "$SRC_DIR/cal_trans_matrix.py" \
  -s "$VEC_SRC" -s1 "$VOCAB_SRC" \
  -t "$VEC_TGT" -s2 "$VOCAB_TGT" \
  -r -n "$PIVOT" \
  -g "$GOLD_JSON" \
  -o "$ALIGN_JSON"

echo "[MedTokAlign] Applying alignment to model ..."
python "$SRC_DIR/convert.py" \
  -m "$ALIGN_JSON" \
  -s "$MODEL_ID" \
  -t "$TOK_DIR" \
  -o "$MODEL_OUT_DIR"

echo "[MedTokAlign] Done. Artifacts under: $RUN_DIR"


