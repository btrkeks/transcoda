#!/bin/bash

set -euo pipefail

DATASET_PRESET="${DATASET_PRESET:-target_system_polish_clean}"
DATA_SPEC_PATH="${DATA_SPEC_PATH:-config/data_spec.json}"
WORKERS="${WORKERS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
TARGET_ACCEPTED_SAMPLES="${TARGET_ACCEPTED_SAMPLES:-}"
MAX_SCHEDULED_TASKS="${MAX_SCHEDULED_TASKS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ARTIFACTS_OUT_DIR="${ARTIFACTS_OUT_DIR:-data/datasets/_runs}"

source .venv/bin/activate

cmd=(
  python -m scripts.dataset_generation.dataset_generation.main
  --dataset_preset "${DATASET_PRESET}"
  --num_workers "${WORKERS}"
  --data_spec_path "${DATA_SPEC_PATH}"
  --artifacts_out_dir "${ARTIFACTS_OUT_DIR}"
)

append_optional_arg() {
  local flag="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    cmd+=("${flag}" "${value}")
  fi
}

append_optional_arg --output_dir "${OUTPUT_DIR}"
append_optional_arg --num_samples "${NUM_SAMPLES}"
append_optional_arg --target_accepted_samples "${TARGET_ACCEPTED_SAMPLES}"
append_optional_arg --max_scheduled_tasks "${MAX_SCHEDULED_TASKS}"
append_optional_arg --variants_per_file "${VARIANTS_PER_FILE:-}"
append_optional_arg --adaptive_variants_enabled "${ADAPTIVE_VARIANTS_ENABLED:-}"
append_optional_arg --failure_policy "${FAILURE_POLICY:-}"
append_optional_arg --overflow_truncation_max_trials "${OVERFLOW_TRUNCATION_MAX_TRIALS:-}"
append_optional_arg --courtesy_naturals_probability "${COURTESY_NATURALS_PROBABILITY:-}"
append_optional_arg --disable_offline_image_augmentations "${DISABLE_OFFLINE_IMAGE_AUGMENTATIONS:-}"
append_optional_arg --target_min_systems "${TARGET_MIN_SYSTEMS:-}"
append_optional_arg --target_max_systems "${TARGET_MAX_SYSTEMS:-}"
append_optional_arg --render_layout_profile "${RENDER_LAYOUT_PROFILE:-}"
append_optional_arg --prefilter_min_non_empty_lines "${PREFILTER_MIN_NON_EMPTY_LINES:-}"
append_optional_arg --prefilter_max_non_empty_lines "${PREFILTER_MAX_NON_EMPTY_LINES:-}"
append_optional_arg --prefilter_min_measure_count "${PREFILTER_MIN_MEASURE_COUNT:-}"
append_optional_arg --prefilter_max_measure_count "${PREFILTER_MAX_MEASURE_COUNT:-}"
append_optional_arg --render_pedals_enabled "${RENDER_PEDALS_ENABLED:-}"
append_optional_arg --render_instrument_piano_enabled "${RENDER_INSTRUMENT_PIANO_ENABLED:-}"
append_optional_arg --render_sforzando_enabled "${RENDER_SFORZANDO_ENABLED:-}"
append_optional_arg --render_accent_enabled "${RENDER_ACCENT_ENABLED:-}"
append_optional_arg --render_tempo_enabled "${RENDER_TEMPO_ENABLED:-}"
append_optional_arg --render_hairpins_enabled "${RENDER_HAIRPINS_ENABLED:-}"
append_optional_arg --render_dynamic_marks_enabled "${RENDER_DYNAMIC_MARKS_ENABLED:-}"

echo "Running target-system dataset generation with:"
echo "  DATASET_PRESET=${DATASET_PRESET}"
echo "  WORKERS=${WORKERS}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "  OUTPUT_DIR=${OUTPUT_DIR}"
fi
if [[ -n "${NUM_SAMPLES}" ]]; then
  echo "  NUM_SAMPLES=${NUM_SAMPLES}"
fi
if [[ -n "${TARGET_ACCEPTED_SAMPLES}" ]]; then
  echo "  TARGET_ACCEPTED_SAMPLES=${TARGET_ACCEPTED_SAMPLES}"
fi

"${cmd[@]}"
