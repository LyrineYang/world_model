#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="world-model-filter"
WITH_TRANSNET=false
WITH_OCR=false
PYTHON_VERSION="3.10"

usage() {
  cat <<'USAGE'
Usage: bash scripts/env/create_filter_env.sh [options]

Options:
  --name <env>          Conda environment name (default: world-model-filter)
  --python <version>    Python version (default: 3.10)
  --with-transnet       Install optional DALI/TransNet dependencies
  --with-ocr            Install optional OCR dependency
  -h, --help            Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --with-transnet)
      WITH_TRANSNET=true
      shift
      ;;
    --with-ocr)
      WITH_OCR=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH." >&2
  exit 1
fi

echo "[env] Ensuring conda environment ${ENV_NAME} (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip ffmpeg

echo "[env] Installing torch/torchvision cu121"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

echo "[env] Installing base filter lock"
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/env/pip/filter.lock.txt"

if [[ "${WITH_TRANSNET}" == "true" ]]; then
  echo "[env] Installing optional transnet dependencies"
  conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/env/pip/filter.optional_transnet.lock.txt"
fi

if [[ "${WITH_OCR}" == "true" ]]; then
  echo "[env] Installing optional OCR dependencies"
  conda run -n "${ENV_NAME}" python -m pip install rapidocr-onnxruntime==1.3.24
fi

echo "[env] Done: ${ENV_NAME}"
