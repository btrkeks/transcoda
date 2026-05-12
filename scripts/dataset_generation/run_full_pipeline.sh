#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

DATA_SPEC_PATH="config/data_spec.json"
WORKERS="${WORKERS:-4}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-data/datasets/train_medium}"
VALIDATION_SYNTH_TARGET_SAMPLES="${VALIDATION_SYNTH_TARGET_SAMPLES:-1000}"
TRAIN_DATASETS_CSV="pdmx,musetrainer,grandstaff,openscore-lieder,openscore-stringquartets"
PIPELINE_TARGET="${PIPELINE_TARGET:-train_full}"
POLISH_MANUAL_FIXES_REFRESH="${POLISH_MANUAL_FIXES_REFRESH:-0}"
BUILD_POLISH_RAW_AUDIT="${BUILD_POLISH_RAW_AUDIT:-0}"
REPORTS_ROOT="reports/dataset_generation/pipeline_drops"
REPORT_RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")"
REPORT_RUN_DIR="${REPORTS_ROOT}/${REPORT_RUN_ID}"
REPORT_STAGE_STATS_DIR="${REPORT_RUN_DIR}/stage_stats"
TRAIN_REPORT_PATH="${REPORT_RUN_DIR}/report.json"
LATEST_REPORT_PATH="${REPORTS_ROOT}/latest.json"

set -e

if [ ! -f "${DATA_SPEC_PATH}" ]; then
  echo "Missing data spec: ${DATA_SPEC_PATH}" >&2
  exit 1
fi

mkdir -p "${REPORT_STAGE_STATS_DIR}"

filter_and_normalize() {
  local split="$1"
  local dataset="$2"
  local run_filter="${3:-true}"
  local base="data/interim/${split}/${dataset}"
  local filter_stats_path="${REPORT_STAGE_STATS_DIR}/${split}_${dataset}_filter.json"
  local normalize_stats_path="${REPORT_STAGE_STATS_DIR}/${split}_${dataset}_normalize.json"

  echo "=== Filter & Normalize: ${dataset} ==="

  ## Filter
  if [ "${run_filter}" = "true" ]; then
    python -m scripts.dataset_generation.filtering.main \
      "${base}/1_kern_conversions/" \
      "${base}/2_filtered/" \
      --workers "${WORKERS}" \
      --stats_json "${filter_stats_path}"
  fi

  ## Normalize
  python -m scripts.dataset_generation.normalization.main \
    --input_dir "${base}/2_filtered" \
    --output_dir "${base}/3_normalized" \
    --workers "${WORKERS}" \
    --stats_json "${normalize_stats_path}"
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

generate_train_dataset() {
  ## PDMX
  # "$SCRIPT_DIR/extract_raw_data/extract_mxl.sh" data/raw/pdmx/train data/interim/train/pdmx/0_raw_xml
  # "$SCRIPT_DIR/kern_conversions/convert_xml_to_kern.sh" data/interim/train/pdmx/0_raw_xml data/interim/train/pdmx/1_kern_conversions
  filter_and_normalize train pdmx

  ## Musetrainer
  # "$SCRIPT_DIR/extract_raw_data/extract_mxl.sh" data/raw/musetrainer/ data/interim/train/musetrainer/0_raw_xml
  # "$SCRIPT_DIR/kern_conversions/convert_xml_to_kern.sh" data/interim/train/musetrainer/0_raw_xml data/interim/train/musetrainer/1_kern_conversions
  # filter_and_normalize train musetrainer

  ## Grandstaff
  # python "$SCRIPT_DIR/extract_raw_data/extract_grandstaff_transcriptions.py" --split train
  # "$SCRIPT_DIR/kern_conversions/convert_ekern_to_kern.sh" data/interim/train/grandstaff/0_raw_ekern data/interim/train/grandstaff/1_kern_conversions
  # filter_and_normalize train grandstaff

  ## OpenScore Lieder
  # python "$SCRIPT_DIR/extract_raw_data/extract_openscore_xml.py" \
  #   --dataset lieder \
  #   --output_dir data/interim/train/openscore-lieder/0_raw_xml
  # "$SCRIPT_DIR/kern_conversions/convert_xml_to_kern.sh" \
  #   data/interim/train/openscore-lieder/0_raw_xml \
  #   data/interim/train/openscore-lieder/1_kern_conversions
  # filter_and_normalize train openscore-lieder true

  ## OpenScore StringQuartets
  # python "$SCRIPT_DIR/extract_raw_data/extract_openscore_xml.py" \
  #   --dataset stringquartets \
  #   --musescore_cmd "${MUSESCORE_CMD:-mscore}" \
  #   --output_dir data/interim/train/openscore-stringquartets/0_raw_xml
  # "$SCRIPT_DIR/kern_conversions/convert_xml_to_kern.sh" \
  #   data/interim/train/openscore-stringquartets/0_raw_xml \
  #   data/interim/train/openscore-stringquartets/1_kern_conversions
  # filter_and_normalize train openscore-stringquartets true

  ## 3 - Validation

  ## Grammar Validation
  # python scripts/dataset_validation/validate_grammar.py --directory data/interim/train/pdmx/3_normalized
  # python scripts/dataset_validation/validate_grammar.py --directory data/interim/train/musetrainer/3_normalized
  # python scripts/dataset_validation/validate_grammar.py --directory data/interim/train/grandstaff/3_normalized
  # python scripts/dataset_validation/validate_grammar.py --directory data/interim/train/openscore-lieder/3_normalized
  # python scripts/dataset_validation/validate_grammar.py --directory data/interim/train/openscore-stringquartets/3_normalized

  ## 4 - Tokenizer generation
  # trash ./vocab/bpe3k-splitspaces-tokenizer.json
  # trash ./vocab/bpe3k-splitspaces
  # python scripts/tokenizer/build_bpe_tokenizer.py \
  #   data/interim/train/pdmx/3_normalized \
  #   data/interim/train/grandstaff/3_normalized \
  #   data/interim/train/musetrainer/3_normalized \
  #   data/interim/train/openscore-lieder/3_normalized \
  #   data/interim/train/openscore-stringquartets/3_normalized \
  #   --split_spaces \
  #   --vocab_name bpe3k-splitspaces \
  #   --vocab_size 3000 \
  #   --out_dir vocab

  ## 5 - Dataset generation
  # python -m scripts.dataset_generation.dataset_generation.main \
  #   data/interim/train/pdmx/3_normalized \
  #   data/interim/train/grandstaff/3_normalized \
  #   data/interim/train/openscore-lieder/3_normalized \
  #   data/interim/train/openscore-stringquartets/3_normalized \
  #   --target_samples 1000 \
  #   --num_workers "${WORKERS}" \
  #   --output_dir "${TRAIN_OUTPUT_DIR}" \
  #   --failure_policy balanced \
  #   --quiet false

  ## 6 - Filter outliers
}

regenerate_openscore_normalized_kern() {
  echo "=== Regenerate OpenScore normalized kern data ==="

  filter_and_normalize train openscore-lieder true

  filter_and_normalize train openscore-stringquartets true
}

write_train_drop_report() {
  python -m scripts.dataset_generation.pipeline_drop_report \
    --pipeline train \
    --split train \
    --datasets "${TRAIN_DATASETS_CSV}" \
    --interim_root data/interim \
    --stage_stats_dir "${REPORT_STAGE_STATS_DIR}" \
    --data_spec_path "${DATA_SPEC_PATH}" \
    --workers "${WORKERS}" \
    --run_id "${REPORT_RUN_ID}" \
    --train_output_dir "${TRAIN_OUTPUT_DIR}" \
    --report_path "${TRAIN_REPORT_PATH}" \
    --latest_path "${LATEST_REPORT_PATH}"

  echo "Drop report: ${TRAIN_REPORT_PATH}"
}

generate_validation_datasets() {
  ## Prepare Data
  ### PDMX
  "$SCRIPT_DIR/extract_raw_data/extract_mxl.sh" data/raw/pdmx/val data/interim/val/pdmx/0_raw_xml
  "$SCRIPT_DIR/kern_conversions/convert_xml_to_kern.sh" data/interim/val/pdmx/0_raw_xml data/interim/val/pdmx/1_kern_conversions
  filter_and_normalize val pdmx

  ### Grandstaff
  python "$SCRIPT_DIR/extract_raw_data/extract_grandstaff_transcriptions.py" --split val
  "$SCRIPT_DIR/kern_conversions/convert_ekern_to_kern.sh" data/interim/val/grandstaff/0_raw_ekern data/interim/val/grandstaff/1_kern_conversions
  filter_and_normalize val grandstaff

  ### Polish
  python "$SCRIPT_DIR/extract_raw_data/extract_polish_scores_transcriptions.py" --target_split val
  "$SCRIPT_DIR/kern_conversions/convert_ekern_to_kern.sh" data/interim/val/polish-scores/0_raw_ekern data/interim/val/polish-scores/1_kern_conversions
  filter_and_normalize val polish-scores
  prepare_polish_manual_fixes

  ## Grammar Validation
  python scripts/dataset_validation/validate_grammar.py --directory data/interim/val/pdmx/3_normalized
  python scripts/dataset_validation/validate_grammar.py --directory data/interim/val/grandstaff/3_normalized
  python scripts/dataset_validation/validate_grammar.py --directory data/interim/val/polish-scores/4_manual_fixes

  ## synth
  python -m scripts.dataset_generation.dataset_generation.main \
    data/interim/val/grandstaff/3_normalized \
    data/interim/val/pdmx/3_normalized \
    data/interim/val/polish-scores/4_manual_fixes \
    --target_samples "${VALIDATION_SYNTH_TARGET_SAMPLES}" \
    --num_workers 4 \
    --output_dir data/datasets/validation/synth \
    --quiet false

  ## polish
  python -m scripts.dataset_generation.assemble_polish_validation \
    --normalized_dir data/interim/val/polish-scores/4_manual_fixes \
    --output_dir data/datasets/validation/polish \
    --data_spec_path "${DATA_SPEC_PATH}" \
    --quiet

  if [ "${BUILD_POLISH_RAW_AUDIT}" = "1" ]; then
    python -m scripts.dataset_generation.assemble_polish_validation \
      --normalized_dir data/interim/val/polish-scores/3_normalized \
      --output_dir data/datasets/validation/polish_raw \
      --data_spec_path "${DATA_SPEC_PATH}" \
      --quiet
  fi
}

generate_olimpic_dataset() {
  ## OlimpIC real-scan pipeline (XML -> kern -> filter -> normalize -> assemble).
  ## Update manifest path if you keep it elsewhere.
  "$SCRIPT_DIR/run_olimpic_pipeline.sh" data/raw/olimpic-1.0-scanned/split_manifest.jsonl
}

source .venv/bin/activate

case "${PIPELINE_TARGET}" in
train_full)
  generate_train_dataset
  write_train_drop_report
  ;;
openscore_normalized)
  regenerate_openscore_normalized_kern
  ;;
olimpic)
  generate_olimpic_dataset
  ;;
validation)
  generate_validation_datasets
  ;;
*)
  echo "Unknown PIPELINE_TARGET: ${PIPELINE_TARGET}" >&2
  echo "Expected one of: train_full, openscore_normalized, olimpic, validation" >&2
  exit 1
  ;;
esac

notify-send "Dataset generation completed!"
