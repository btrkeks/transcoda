#!/bin/bash
# Convert MusicXML (.xml) files to Humdrum **kern format
# Uses GNU parallel for speed and musicxml2hum from humlib

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <xml_dir> <output_dir>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XML_DIR="$1"
OUT_DIR="$2"
LOG_FILE="${SCRIPT_DIR}/conversion_errors.log"
MUSICXML2HUM_ERRORS="${SCRIPT_DIR}/musicxml2hum_errors.count"

# Check dependencies
for cmd in musicxml2hum parallel; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd not found in PATH"
    exit 1
  fi
done

# Create output directory
mkdir -p "${OUT_DIR}"

# Count files
TOTAL=$(find "${XML_DIR}" -name "*.xml" -type f | wc -l)

echo "Found ${TOTAL} .xml files"
echo "Output: ${OUT_DIR}"
echo ""

# Clear error log and counter
>"${LOG_FILE}"
>"${MUSICXML2HUM_ERRORS}"

# Export function and variables for parallel
convert_one() {
  local xml_file="$1"

  local base_name="${xml_file##*/}"
  base_name="${base_name%.xml}"
  local out_file="${OUT_DIR}/${base_name}.krn"

  if ! musicxml2hum "${xml_file}" >"${out_file}" 2>/dev/null; then
    echo "${xml_file}: conversion failed" >>"${LOG_FILE}"
    echo "x" >>"${MUSICXML2HUM_ERRORS}"
    rm -f "${out_file}"
    return 1
  fi

  # Remove empty outputs (conversion silently failed)
  [[ -s "${out_file}" ]] || {
    rm -f "${out_file}"
    return 1
  }
}
export -f convert_one
export OUT_DIR LOG_FILE MUSICXML2HUM_ERRORS

# Process all files (pipe to parallel to avoid argument list limit)
find "${XML_DIR}" -name "*.xml" -type f |
  parallel --halt never --jobs "$(nproc)" convert_one || true

# Summary
echo ""
echo "Done!"
echo "Errors logged to: ${LOG_FILE}"
echo "Total errors: $(wc -l <"${LOG_FILE}")"
echo "musicxml2hum errors: $(wc -l <"${MUSICXML2HUM_ERRORS}")"
echo "Converted: $(find "${OUT_DIR}" -name "*.krn" | wc -l) files"
