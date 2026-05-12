#!/bin/bash
# Convert ekern (.ekern) files to Humdrum **kern format
# Removes @ and · delimiters and renames **ekern spines to **kern
# Uses GNU parallel for speed

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <ekern_dir> <output_dir>"
  exit 1
fi

EKERN_DIR="$1"
OUT_DIR="$2"

# Check dependencies
if ! command -v parallel &>/dev/null; then
  echo "Error: parallel (GNU parallel) not found in PATH"
  exit 1
fi

# Create output directory
mkdir -p "${OUT_DIR}"

# Count files
TOTAL=$(find "${EKERN_DIR}" -name "*.ekern" -type f | wc -l)

echo "Found ${TOTAL} .ekern files"
echo "Output: ${OUT_DIR}"
echo ""

# Export function and variables for parallel
convert_one() {
  local ekern_file="$1"

  local base_name="${ekern_file##*/}"
  base_name="${base_name%.ekern}"
  local out_file="${OUT_DIR}/${base_name}.krn"

  # Replace **ekern (including versioned variants like **ekern_1.0) with **kern
  # Remove @ and · (middle dot) delimiters
  sed -e 's/\*\*ekern[^\t]*/\*\*kern/g' \
      -e 's/@//g' \
      -e 's/·//g' \
      "${ekern_file}" > "${out_file}"
}
export -f convert_one
export OUT_DIR

# Process all files (pipe to parallel to avoid argument list limit)
find "${EKERN_DIR}" -name "*.ekern" -type f |
  parallel --halt never --jobs "$(nproc)" convert_one || true

# Summary
echo ""
echo "Done!"
echo "Converted: $(find "${OUT_DIR}" -name "*.krn" -type f | wc -l) files"
