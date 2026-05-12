#!/bin/bash

set -euo pipefail

WORKERS="${WORKERS:-4}"
TARGET_SAMPLES="${TARGET_SAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-data/datasets/target_system_train}"
ARTIFACTS_OUT_DIR="${ARTIFACTS_OUT_DIR:-data/datasets/_runs}"
RESUME_MODE="${RESUME_MODE:-auto}"
BASE_SEED="${BASE_SEED:-0}"
FAILURE_POLICY="${FAILURE_POLICY:-balanced}"
QUIET="${QUIET:-false}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-}"
QUARANTINE_IN="${QUARANTINE_IN:-}"
QUARANTINE_OUT="${QUARANTINE_OUT:-}"

if [[ -n "${SOURCE_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  source_dirs=(${SOURCE_DIRS})
else
  source_dirs=(
    data/interim/train/pdmx/3_normalized
    data/interim/train/grandstaff/3_normalized
    data/interim/train/musetrainer/3_normalized
    data/interim/train/openscore-lieder/3_normalized
    data/interim/train/openscore-stringquartets/3_normalized
  )
fi

unsupported_env_vars=(
  DATASET_PRESET
  DATA_SPEC_PATH
  NUM_SAMPLES
  TARGET_ACCEPTED_SAMPLES
  MAX_SCHEDULED_TASKS
  VARIANTS_PER_FILE
  ADAPTIVE_VARIANTS_ENABLED
  OVERFLOW_TRUNCATION_MAX_TRIALS
  COURTESY_NATURALS_PROBABILITY
  DISABLE_OFFLINE_IMAGE_AUGMENTATIONS
  TARGET_MIN_SYSTEMS
  TARGET_MAX_SYSTEMS
  RENDER_LAYOUT_PROFILE
  PREFILTER_MIN_NON_EMPTY_LINES
  PREFILTER_MAX_NON_EMPTY_LINES
  PREFILTER_MIN_MEASURE_COUNT
  PREFILTER_MAX_MEASURE_COUNT
  RENDER_PEDALS_ENABLED
  RENDER_INSTRUMENT_PIANO_ENABLED
  RENDER_SFORZANDO_ENABLED
  RENDER_ACCENT_ENABLED
  RENDER_TEMPO_ENABLED
  RENDER_HAIRPINS_ENABLED
  RENDER_DYNAMIC_MARKS_ENABLED
)

for var_name in "${unsupported_env_vars[@]}"; do
  if [[ -n "${!var_name:-}" ]]; then
    echo "Unsupported legacy setting for rewrite pipeline: ${var_name}" >&2
    exit 1
  fi
done

source .venv/bin/activate

cmd=(
  python -m scripts.dataset_generation.dataset_generation.main
  "${source_dirs[@]}"
  --num_workers "${WORKERS}"
  --target_samples "${TARGET_SAMPLES}"
  --output_dir "${OUTPUT_DIR}"
  --artifacts_out_dir "${ARTIFACTS_OUT_DIR}"
  --resume_mode "${RESUME_MODE}"
  --base_seed "${BASE_SEED}"
  --failure_policy "${FAILURE_POLICY}"
  --quiet "${QUIET}"
)

append_optional_arg() {
  local flag="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    cmd+=("${flag}" "${value}")
  fi
}

append_optional_arg --max_attempts "${MAX_ATTEMPTS}"
append_optional_arg --quarantine_in "${QUARANTINE_IN}"
append_optional_arg --quarantine_out "${QUARANTINE_OUT}"

echo "Running target-system dataset generation with:"
echo "  WORKERS=${WORKERS}"
echo "  TARGET_SAMPLES=${TARGET_SAMPLES}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  SOURCE_DIRS=${source_dirs[*]}"

"${cmd[@]}"
