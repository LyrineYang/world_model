#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME=""
INSTALL_TRANSNET=false
INSTALL_OCR=false
INSTALL_AES=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/env/install_optional.sh --name <env> [--transnet] [--ocr] [--aes]

Options:
  --name <env>   Target conda environment name (required)
  --transnet     Install DALI/TransNet optional dependencies
  --ocr          Install OCR optional dependencies
  --aes          Install LAION-AES optional dependency (open_clip_torch)
  -h, --help     Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      ENV_NAME="$2"
      shift 2
      ;;
    --transnet)
      INSTALL_TRANSNET=true
      shift
      ;;
    --ocr)
      INSTALL_OCR=true
      shift
      ;;
    --aes)
      INSTALL_AES=true
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

if [[ -z "${ENV_NAME}" ]]; then
  echo "--name is required" >&2
  usage
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH." >&2
  exit 1
fi

if [[ "${INSTALL_TRANSNET}" == "true" ]]; then
  echo "[optional] Installing transnet into ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m pip install -r "${ROOT_DIR}/env/pip/filter.optional_transnet.lock.txt"
fi

if [[ "${INSTALL_OCR}" == "true" ]]; then
  echo "[optional] Installing OCR into ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m pip install rapidocr-onnxruntime==1.3.24
fi

if [[ "${INSTALL_AES}" == "true" ]]; then
  echo "[optional] Installing AES into ${ENV_NAME}"
  conda run -n "${ENV_NAME}" python -m pip install open_clip_torch==2.24.0
fi

echo "[optional] Done"
