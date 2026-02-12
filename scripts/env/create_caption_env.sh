#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="world-model-caption"
PYTHON_VERSION="3.10"

usage() {
  cat <<'USAGE'
Usage: bash scripts/env/create_caption_env.sh [options]

Options:
  --name <env>          Conda environment name (default: world-model-caption)
  --python <version>    Python version (default: 3.10)
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

echo "[env] Installing caption lock"
conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/env/pip/caption_qwen3.lock.txt"

echo "[env] Done: ${ENV_NAME}"
