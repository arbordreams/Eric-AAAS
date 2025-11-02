#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="medical_tokalign/configs/eval_medical.yaml"

usage() {
  echo "Usage: $0 [--config PATH]" >&2
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --config) CONFIG_PATH="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "[MedTokAlign] Starting medical evaluation with config: $CONFIG_PATH"

# Run as a module to ensure package-relative imports work
python -m medical_tokalign.src.medical_eval --config "$CONFIG_PATH"


