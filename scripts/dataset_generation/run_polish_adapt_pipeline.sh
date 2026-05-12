#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

DATA_SPEC_PATH="${DATA_SPEC_PATH:-config/data_spec.json}"
WORKERS="${WORKERS:-4}"
POLISH_MANUAL_FIXES_REFRESH="${POLISH_MANUAL_FIXES_REFRESH:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
SYNTH_DATASET_PATH="${SYNTH_DATASET_PATH:-data/datasets/train_full}"
MIXED_TARGET_SYNTH_COUNT="${MIXED_TARGET_SYNTH_COUNT:-1000}"
MIXED_TARGET_REAL_COUNT="${MIXED_TARGET_REAL_COUNT:-996}"
MIXED_SEED="${MIXED_SEED:-42}"
MIXED_OUTPUT_DIR="${MIXED_OUTPUT_DIR:-data/datasets/train_polish_adapt_mixed}"
REPORT_PATH="${REPORT_PATH:-reports/dataset_generation/polish_adapt/latest.json}"

set -euo pipefail

filter_and_normalize_polish() {
  local base="data/interim/val/polish-scores"

  python -m scripts.dataset_generation.filtering.main \
    "${base}/1_kern_conversions/" \
    "${base}/2_filtered/" \
    --workers "${WORKERS}"

  python -m scripts.dataset_generation.normalization.main \
    --input_dir "${base}/2_filtered" \
    --output_dir "${base}/3_normalized" \
    --workers "${WORKERS}"
}

prepare_polish_manual_fixes() {
  local refresh_flag="False"
  if [ "${POLISH_MANUAL_FIXES_REFRESH}" = "1" ]; then
    refresh_flag="True"
  fi

  python -m scripts.dataset_generation.manual_fixes.main prepare \
    --base_dir data/interim/val/polish-scores/3_normalized \
    --working_dir data/interim/val/polish-scores/4_manual_fixes \
    --overrides_dir curation/polish-scores/overrides \
    --manifest_out reports/dataset_generation/manual_fixes/polish-scores/latest.json \
    --refresh "${refresh_flag}" \
    --quiet
}

assemble_split() {
  local split="$1"
  local output_dir="$2"
  local cmd=(
    python -m scripts.dataset_generation.assemble_polish_scores_dataset
    --normalized_dir data/interim/val/polish-scores/4_manual_fixes
    --output_dir "${output_dir}"
    --include_splits "${split}"
    --data_spec_path "${DATA_SPEC_PATH}"
    --quiet
  )
  if [ -n "${HF_CACHE_DIR}" ]; then
    cmd+=(--hf_cache_dir "${HF_CACHE_DIR}")
  fi
  "${cmd[@]}"
}

source .venv/bin/activate

python "${SCRIPT_DIR}/extract_raw_data/extract_polish_scores_transcriptions.py" --target_split val
"${SCRIPT_DIR}/kern_conversions/convert_ekern_to_kern.sh" \
  data/interim/val/polish-scores/0_raw_ekern \
  data/interim/val/polish-scores/1_kern_conversions
filter_and_normalize_polish
prepare_polish_manual_fixes

assemble_split train data/datasets/train_polish_adapt
assemble_split val data/datasets/validation/polish_dev
assemble_split test data/datasets/validation/polish_test

if [ -d "${SYNTH_DATASET_PATH}" ]; then
  python -m scripts.dataset_generation.assemble_mixed_train_dataset \
    --synth_dataset_path "${SYNTH_DATASET_PATH}" \
    --real_dataset_path data/datasets/train_polish_adapt \
    --output_dir "${MIXED_OUTPUT_DIR}" \
    --target_synth_count "${MIXED_TARGET_SYNTH_COUNT}" \
    --target_real_count "${MIXED_TARGET_REAL_COUNT}" \
    --seed "${MIXED_SEED}" \
    --metadata_out "${REPORT_PATH}"
else
  echo "Skipping mixed dataset assembly; synth dataset not found at ${SYNTH_DATASET_PATH}" >&2
fi
