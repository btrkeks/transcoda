#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-data/datasets/train_rewrite}"
TARGET_SAMPLES="${TARGET_SAMPLES:-1000}"
WORKERS="${WORKERS:-4}"
ARTIFACTS_OUT_DIR="${ARTIFACTS_OUT_DIR:-data/datasets/_runs}"
RESUME_MODE="${RESUME_MODE:-auto}"
BASE_SEED="${BASE_SEED:-0}"
FAILURE_POLICY="${FAILURE_POLICY:-balanced}"
QUARANTINE_IN="${QUARANTINE_IN:-}"
QUARANTINE_OUT="${QUARANTINE_OUT:-}"
QUIET="${QUIET:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-}"

if [[ -n "${SOURCE_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  source_dirs=(${SOURCE_DIRS})
else
  source_dirs=(
    data/interim/train/pdmx/3_normalized
    data/interim/train/grandstaff/3_normalized
    data/interim/train/openscore-lieder/3_normalized
    data/interim/train/openscore-stringquartets/3_normalized
  )
fi

source "${ROOT_DIR}/.venv/bin/activate"

cmd=(
  python -m scripts.dataset_generation.dataset_generation.main
  "${source_dirs[@]}"
  --output_dir "${OUTPUT_DIR}"
  --target_samples "${TARGET_SAMPLES}"
  --num_workers "${WORKERS}"
  --artifacts_out_dir "${ARTIFACTS_OUT_DIR}"
  --resume_mode "${RESUME_MODE}"
  --base_seed "${BASE_SEED}"
  --failure_policy "${FAILURE_POLICY}"
  --quiet "${QUIET}"
)

if [[ -n "${MAX_ATTEMPTS}" ]]; then
  cmd+=(--max_attempts "${MAX_ATTEMPTS}")
fi

if [[ -n "${QUARANTINE_IN}" ]]; then
  cmd+=(--quarantine_in "${QUARANTINE_IN}")
fi

if [[ -n "${QUARANTINE_OUT}" ]]; then
  cmd+=(--quarantine_out "${QUARANTINE_OUT}")
fi

echo "Running rewrite train generation with:"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "  WORKERS=${WORKERS}"
echo "  RESUME_MODE=${RESUME_MODE}"
echo "  FAILURE_POLICY=${FAILURE_POLICY}"
echo "  BALANCING=mandatory (bundled spec)"
echo "  SOURCE_DIRS=${source_dirs[*]}"

"${cmd[@]}"
