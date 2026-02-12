#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_PATH="${ROOT_DIR}/third_party/manifest.lock.yaml"
FLOATING=false
INCLUDE_DISABLED=false

usage() {
  cat <<'USAGE'
Usage: bash scripts/bootstrap_third_party.sh [options]

Options:
  --manifest <path>      Path to third_party manifest lock file
  --floating             Track remote branches (do not checkout pinned commits)
  --include-disabled     Also process repos with enabled=false
  --install-editable     Deprecated and ignored (kept for compatibility)
  -h, --help             Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST_PATH="$2"
      shift 2
      ;;
    --floating)
      FLOATING=true
      shift
      ;;
    --include-disabled)
      INCLUDE_DISABLED=true
      shift
      ;;
    --install-editable)
      echo "[third_party] --install-editable is deprecated and ignored." >&2
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

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

read_manifest_tsv() {
  local manifest="$1"
  python3 - "$manifest" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
text = manifest_path.read_text(encoding="utf-8")

repos = []

# Try PyYAML first.
try:
    import yaml  # type: ignore

    payload = yaml.safe_load(text) or {}
    loaded = payload.get("repos", [])
    if isinstance(loaded, list):
        repos = [r for r in loaded if isinstance(r, dict)]
except Exception:
    repos = []

# Fallback parser for simple YAML structure.
if not repos:
    cur = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("repos:"):
            continue
        if line.startswith("- "):
            if cur:
                repos.append(cur)
            cur = {}
            line = line[2:]
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if value.lower() in {"true", "false"}:
            cur[key] = value.lower() == "true"
        else:
            cur[key] = value
    if cur:
        repos.append(cur)

for repo in repos:
    name = str(repo.get("name", "")).strip()
    url = str(repo.get("url", "")).strip()
    commit = str(repo.get("commit", "")).strip()
    path = str(repo.get("path", "")).strip()
    enabled = bool(repo.get("enabled", True))
    if not (name and url and path):
        continue
    print("\t".join([name, url, commit, path, "true" if enabled else "false"]))
PY
}

clone_if_needed() {
  local url="$1"
  local dst="$2"

  if [[ -d "${dst}/.git" ]]; then
    return
  fi
  if [[ -e "${dst}" ]]; then
    echo "[third_party] Path exists and is not a git repo: ${dst}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${dst}")"
  echo "[third_party] Cloning $(basename "${dst}") -> ${dst}"
  git clone "${url}" "${dst}"
}

checkout_pinned() {
  local dst="$1"
  local commit="$2"

  if [[ -z "${commit}" ]]; then
    echo "[third_party] Missing commit for ${dst} in pinned mode" >&2
    exit 1
  fi

  git -C "${dst}" fetch --tags --prune origin
  git -C "${dst}" checkout --detach "${commit}"
}

checkout_floating() {
  local dst="$1"

  git -C "${dst}" fetch --tags --prune origin
  local default_branch
  default_branch="$(git -C "${dst}" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')"
  if [[ -z "${default_branch}" ]]; then
    default_branch="main"
  fi

  if git -C "${dst}" show-ref --verify --quiet "refs/heads/${default_branch}"; then
    git -C "${dst}" checkout "${default_branch}"
  else
    git -C "${dst}" checkout -B "${default_branch}" "origin/${default_branch}"
  fi
  git -C "${dst}" pull --ff-only origin "${default_branch}" || true
}

while IFS=$'\t' read -r name url commit rel_path enabled; do
  if [[ -z "${name}" ]]; then
    continue
  fi
  if [[ "${enabled}" != "true" && "${INCLUDE_DISABLED}" != "true" ]]; then
    echo "[third_party] Skip disabled repo ${name}"
    continue
  fi

  dst="${ROOT_DIR}/${rel_path}"
  clone_if_needed "${url}" "${dst}"

  current_url="$(git -C "${dst}" remote get-url origin 2>/dev/null || true)"
  if [[ -n "${current_url}" && "${current_url}" != "${url}" ]]; then
    echo "[third_party] Updating origin URL for ${name}: ${current_url} -> ${url}"
    git -C "${dst}" remote set-url origin "${url}"
  fi

  if [[ "${FLOATING}" == "true" ]]; then
    echo "[third_party] Floating sync ${name}"
    checkout_floating "${dst}"
  else
    echo "[third_party] Pin ${name} at ${commit}"
    checkout_pinned "${dst}" "${commit}"
  fi

done < <(read_manifest_tsv "${MANIFEST_PATH}")

echo "[third_party] Done."
