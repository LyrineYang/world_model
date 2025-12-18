#!/usr/bin/env bash
set -euo pipefail

# Download recommended UniMatch optical flow weight for motion filtering.
# Default target: unimatch/pretrained/gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/unimatch/pretrained"
WEIGHT_FILE="gmflow-scale1-mixdata-train320x576-4c3a6e9a.pth"
URL="https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/${WEIGHT_FILE}"

mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

if [ -f "${WEIGHT_FILE}" ]; then
  echo "Weight already exists: ${TARGET_DIR}/${WEIGHT_FILE}"
  exit 0
fi

echo "Downloading UniMatch weight..."
curl -L -o "${WEIGHT_FILE}" "${URL}"
echo "Done: ${TARGET_DIR}/${WEIGHT_FILE}"
