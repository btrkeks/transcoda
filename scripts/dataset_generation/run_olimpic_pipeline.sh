#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

DATA_SPEC_PATH="config/data_spec.json"
if [[ ! -f "${DATA_SPEC_PATH}" ]]; then
  echo "Missing data spec: ${DATA_SPEC_PATH}" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage: scripts/dataset_generation/run_olimpic_pipeline.sh <split_manifest> [raw_root]

Arguments:
  split_manifest  Path to OlimpIC split manifest (.jsonl/.json/.csv)
  raw_root        Base directory for source files (default: data/raw/olimpic-1.0-scanned)

Environment variables:
  WORKERS         Parallel workers for filtering/normalization (default: 4)
EOF
  exit 1
fi

SPLIT_MANIFEST="$1"
RAW_ROOT="${2:-data/raw/olimpic-1.0-scanned}"
WORKERS="${WORKERS:-4}"

if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  echo "Split manifest not found: ${SPLIT_MANIFEST}" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Missing .venv. Create environment first." >&2
  exit 1
fi

source .venv/bin/activate

python -m scripts.dataset_generation.extract_raw_data.extract_olimpic_transcriptions \
  --split_manifest "${SPLIT_MANIFEST}" \
  --raw_root "${RAW_ROOT}" \
  --output_dir data/interim

for split in train val test; do
  base="data/interim/${split}/olimpic"
  if [[ ! -d "${base}/0_raw_xml" ]]; then
    continue
  fi

  echo "=== OlimpIC ${split}: convert/filter/normalize ==="
  "${SCRIPT_DIR}/kern_conversions/convert_xml_to_kern.sh" \
    "${base}/0_raw_xml" \
    "${base}/1_kern_conversions"

  python -m scripts.dataset_generation.filtering.main \
    "${base}/1_kern_conversions" \
    "${base}/2_filtered" \
    --workers "${WORKERS}"

  python -m scripts.dataset_generation.normalization.main \
    --input_dir "${base}/2_filtered" \
    --output_dir "${base}/3_normalized" \
    --workers "${WORKERS}"

  python scripts/dataset_validation/validate_grammar.py \
    --directory "${base}/3_normalized"

done

if [[ -d data/interim/train/olimpic/3_normalized && -f data/interim/train/olimpic/metadata/manifest.jsonl ]]; then
  echo "=== Assemble train_olimpic_scanned ==="
  python -m scripts.dataset_generation.assemble_olimpic_dataset \
    --normalized_dir data/interim/train/olimpic/3_normalized \
    --manifest_path data/interim/train/olimpic/metadata/manifest.jsonl \
    --output_dir data/datasets/train_olimpic_scanned \
    --data_spec_path "${DATA_SPEC_PATH}"
fi

if [[ -d data/interim/val/olimpic/3_normalized && -f data/interim/val/olimpic/metadata/manifest.jsonl ]]; then
  echo "=== Assemble validation/olimpic ==="
  python -m scripts.dataset_generation.assemble_olimpic_dataset \
    --normalized_dir data/interim/val/olimpic/3_normalized \
    --manifest_path data/interim/val/olimpic/metadata/manifest.jsonl \
    --output_dir data/datasets/validation/olimpic \
    --data_spec_path "${DATA_SPEC_PATH}"
fi

if [[ -d data/datasets/train_olimpic_scanned ]]; then
  python scripts/dataset_validation/validate_image_sizes.py data/datasets/train_olimpic_scanned
fi

if [[ -d data/datasets/validation/olimpic ]]; then
  python scripts/dataset_validation/validate_image_sizes.py data/datasets/validation/olimpic
fi

echo "OlimpIC pipeline completed."
