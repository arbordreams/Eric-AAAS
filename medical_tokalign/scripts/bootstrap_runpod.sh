#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[MedTokAlign] Bootstrapping RunPod environment..."

# Recommended env for memory behavior
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if command -v apt-get >/dev/null 2>&1; then
  echo "[MedTokAlign] Installing OS build tools (requires root)..."
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git ca-certificates curl
fi

echo "[MedTokAlign] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[MedTokAlign] Installing Python dependencies (excluding torch which should come with the image)..."
# Ensure flash-attn is installed with a CUDA-compatible wheel if available
if ! python - <<'PY'
try:
    import flash_attn  # noqa: F401
    print("ok")
except Exception:
    pass
PY
then
  echo "[MedTokAlign] Installing flash-attn (no build isolation) ..."
  python -m pip install --no-build-isolation -U flash-attn || true
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ -n "${SCISPACY_MODEL_URL:-}" ]]; then
  echo "[MedTokAlign] Installing SciSpacy model from $SCISPACY_MODEL_URL"
  python -m pip install --no-cache-dir "$SCISPACY_MODEL_URL" || true
fi

TOOLS_DIR="$ROOT_DIR/tools"
GLOVE_DIR="$TOOLS_DIR/GloVe"
if [[ ! -d "$GLOVE_DIR" ]]; then
  echo "[MedTokAlign] Cloning Stanford GloVe..."
  mkdir -p "$TOOLS_DIR"
  git clone https://github.com/stanfordnlp/GloVe.git "$GLOVE_DIR"
fi

echo "[MedTokAlign] Building GloVe binaries..."
make -C "$GLOVE_DIR"

echo "[MedTokAlign] Bootstrap complete. Next steps:"
echo "  1) bash $ROOT_DIR/scripts/prepare_medical_data.sh"
echo "  2) bash $ROOT_DIR/scripts/run_vocab_adaptation.sh --model_id meta-llama/Meta-Llama-3.1-8B --top_k 8192"
echo "  3) bash $ROOT_DIR/scripts/eval_medical.sh --config $ROOT_DIR/configs/eval_medical.yaml"


