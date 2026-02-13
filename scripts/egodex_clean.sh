#!/usr/bin/env bash
set -euo pipefail

# Standardized EgoDex cleaning entrypoint for server usage.
# Fixed repo path and fixed conda env name by design.

REPO_DIR="/scratch/ayuille1/jchen293/wcloong/world_model"
ENV_NAME="world-model-filter"
CONFIG_PATH="${EGODEX_CONFIG_PATH:-configs/config_filter_egodex300g_dual_gpu.yaml}"
DATA_ZIP_ROOT="/scratch/ayuille1/jchen293/wcloong/egoworld/egoworld/data"
DEFAULT_EGODEX_ROOT="${DATA_ZIP_ROOT}/extracted"
DEFAULT_OUTPUT_DIR="${REPO_DIR}/workdir_egodex300g/output/offline_egodex300g"
DEFAULT_WORKDIR="${REPO_DIR}/workdir_egodex300g"
EGODEX_FLUSH_EVERY="${EGODEX_FLUSH_EVERY:-2}"
EGODEX_RUNLOG_EVERY="${EGODEX_RUNLOG_EVERY:-5}"
EGODEX_WRITE_BUFFER_RECORDS="${EGODEX_WRITE_BUFFER_RECORDS:-128}"
EGODEX_RESUME="${EGODEX_RESUME:-1}"
EGODEX_CUDA_VISIBLE_DEVICES="${EGODEX_CUDA_VISIBLE_DEVICES:-0,1}"

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/egodex_clean.sh setup
  bash scripts/egodex_clean.sh doctor
  bash scripts/egodex_clean.sh prepare-data [zip_root] [extract_dir]
  bash scripts/egodex_clean.sh run [egodex_root] [output_dir]
  bash scripts/egodex_clean.sh all [egodex_root] [output_dir]
  bash scripts/egodex_clean.sh clean-output
  bash scripts/egodex_clean.sh clean-workdir
  bash scripts/egodex_clean.sh clean-env
  bash scripts/egodex_clean.sh clean-all

Examples:
  bash scripts/egodex_clean.sh prepare-data
  bash scripts/egodex_clean.sh all
  bash scripts/egodex_clean.sh run /scratch/ayuille1/jchen293/wcloong/egoworld/egoworld/data/extracted /scratch/output/egodex_clean
  EGODEX_CONFIG_PATH=configs/config_filter_egodex300g_dual_gpu.yaml bash scripts/egodex_clean.sh run
  EGODEX_CUDA_VISIBLE_DEVICES=0,1 bash scripts/egodex_clean.sh run
  EGODEX_FLUSH_EVERY=1 EGODEX_RUNLOG_EVERY=2 EGODEX_WRITE_BUFFER_RECORDS=64 bash scripts/egodex_clean.sh run
USAGE
}

ensure_repo() {
  if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[error] repo not found: ${REPO_DIR}" >&2
    exit 1
  fi
  cd "${REPO_DIR}"
}

activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found in PATH" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
}

safe_rm_dir() {
  local target="$1"
  if [[ -z "${target}" || "${target}" == "/" ]]; then
    echo "[error] unsafe delete target: ${target}" >&2
    exit 1
  fi
  case "${target}" in
    "${REPO_DIR}"/*) ;;
    *)
      echo "[error] refuse to delete outside repo: ${target}" >&2
      exit 1
      ;;
  esac
  rm -rf "${target}"
}

cmd_setup() {
  ensure_repo
  bash scripts/env/create_filter_env.sh --name "${ENV_NAME}"
  activate_env
  bash scripts/bootstrap_third_party.sh
  bash scripts/download_unimatch_weights.sh
  python scripts/doctor.py filter
}

cmd_doctor() {
  ensure_repo
  activate_env
  python scripts/doctor.py filter
}

cmd_prepare_data() {
  local zip_root="${1:-${DATA_ZIP_ROOT}}"
  local extract_dir="${2:-${DEFAULT_EGODEX_ROOT}}"
  if [[ ! -d "${zip_root}" ]]; then
    echo "[error] zip root not found: ${zip_root}" >&2
    exit 1
  fi
  if ! command -v unzip >/dev/null 2>&1; then
    echo "[error] unzip not found in PATH" >&2
    exit 1
  fi
  mkdir -p "${extract_dir}"

  shopt -s nullglob
  local zips=("${zip_root}"/*.zip)
  shopt -u nullglob
  if [[ ${#zips[@]} -eq 0 ]]; then
    echo "[error] no zip files found under ${zip_root}" >&2
    exit 1
  fi

  echo "[prepare] unzip ${#zips[@]} archives -> ${extract_dir}"
  local z
  for z in "${zips[@]}"; do
    unzip -q -n "${z}" -d "${extract_dir}"
  done

  local count
  count="$(find "${extract_dir}" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l | tr -d ' ')"
  echo "[prepare] video files found: ${count}"
  if [[ "${count}" == "0" ]]; then
    echo "[warn] no video files detected after extraction: ${extract_dir}" >&2
  fi
}

cmd_run() {
  ensure_repo
  activate_env
  local egodex_root="${1:-${DEFAULT_EGODEX_ROOT}}"
  local output_dir="${2:-${DEFAULT_OUTPUT_DIR}}"
  local resume_flag="--resume"
  case "$(echo "${EGODEX_RESUME}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off)
      resume_flag="--no-resume"
      ;;
  esac
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${EGODEX_CUDA_VISIBLE_DEVICES}"
  fi
  if [[ ! -d "${egodex_root}" ]]; then
    echo "[error] egodex_root not found: ${egodex_root}" >&2
    echo "[hint] run: bash scripts/egodex_clean.sh prepare-data" >&2
    exit 1
  fi
  local count
  count="$(find "${egodex_root}" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.mkv" -o -iname "*.webm" \) | wc -l | tr -d ' ')"
  if [[ "${count}" == "0" ]]; then
    echo "[error] no video files found under: ${egodex_root}" >&2
    echo "[hint] if this path contains zip files, run: bash scripts/egodex_clean.sh prepare-data" >&2
    exit 1
  fi

  echo "[run] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  python scripts/run_workflow.py offline-filter \
    --config "${CONFIG_PATH}" \
    --input-dir "${egodex_root}" \
    --recursive \
    --copy-mode link \
    --output-dir "${output_dir}" \
    "${resume_flag}" \
    --flush-every "${EGODEX_FLUSH_EVERY}" \
    --runlog-every "${EGODEX_RUNLOG_EVERY}" \
    --write-buffer-records "${EGODEX_WRITE_BUFFER_RECORDS}"
}

cmd_all() {
  local egodex_root="${1:-${DEFAULT_EGODEX_ROOT}}"
  if [[ "${egodex_root}" == "${DEFAULT_EGODEX_ROOT}" && ! -d "${DEFAULT_EGODEX_ROOT}" ]]; then
    cmd_prepare_data "${DATA_ZIP_ROOT}" "${DEFAULT_EGODEX_ROOT}"
  fi
  cmd_setup
  cmd_run "${egodex_root}" "${2:-${DEFAULT_OUTPUT_DIR}}"
}

cmd_clean_output() {
  ensure_repo
  safe_rm_dir "${DEFAULT_OUTPUT_DIR}"
  echo "[ok] cleaned output: ${DEFAULT_OUTPUT_DIR}"
}

cmd_clean_workdir() {
  ensure_repo
  safe_rm_dir "${DEFAULT_WORKDIR}"
  echo "[ok] cleaned workdir: ${DEFAULT_WORKDIR}"
}

cmd_clean_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "[error] conda not found in PATH" >&2
    exit 1
  fi
  conda env remove -n "${ENV_NAME}" -y || true
  echo "[ok] cleaned conda env: ${ENV_NAME}"
}

cmd_clean_all() {
  cmd_clean_output
  cmd_clean_workdir
  cmd_clean_env
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    setup)
      cmd_setup
      ;;
    doctor)
      cmd_doctor
      ;;
    prepare-data)
      shift || true
      cmd_prepare_data "${1:-${DATA_ZIP_ROOT}}" "${2:-${DEFAULT_EGODEX_ROOT}}"
      ;;
    run)
      shift || true
      cmd_run "${1:-${DEFAULT_EGODEX_ROOT}}" "${2:-${DEFAULT_OUTPUT_DIR}}"
      ;;
    all)
      shift || true
      cmd_all "${1:-${DEFAULT_EGODEX_ROOT}}" "${2:-${DEFAULT_OUTPUT_DIR}}"
      ;;
    clean-output)
      cmd_clean_output
      ;;
    clean-workdir)
      cmd_clean_workdir
      ;;
    clean-env)
      cmd_clean_env
      ;;
    clean-all)
      cmd_clean_all
      ;;
    -h|--help|help|"")
      usage
      ;;
    *)
      echo "[error] unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "${@}"
