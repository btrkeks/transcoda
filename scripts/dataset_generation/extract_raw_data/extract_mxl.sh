#!/bin/bash
# Extract MusicXML (.xml) files from compressed MXL archives
# MXL files are zip archives containing a .xml file

set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <mxl_dir> <output_dir>"
  exit 1
fi

MXL_DIR="$1"
OUT_DIR="$2"

mkdir -p "${OUT_DIR}"

TOTAL=$(find "${MXL_DIR}" -name "*.mxl" -type f | wc -l)
echo "Found ${TOTAL} .mxl files in ${MXL_DIR}"
echo "Output: ${OUT_DIR}"
echo ""

ERRORS=0
SKIPPED=0
EXTRACTED=0

for mxl in "${MXL_DIR}"/*.mxl; do
  [ -f "${mxl}" ] || continue

  base_name="${mxl##*/}"
  base_name="${base_name%.mxl}"
  xml_file="${OUT_DIR}/${base_name}.xml"

  if [ -f "${xml_file}" ]; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if ! unzip -p "${mxl}" "*.xml" > "${xml_file}" 2>/dev/null; then
    echo "Failed to extract: ${mxl}"
    rm -f "${xml_file}"
    ERRORS=$((ERRORS + 1))
    continue
  fi

  if [ ! -s "${xml_file}" ]; then
    rm -f "${xml_file}"
    ERRORS=$((ERRORS + 1))
    continue
  fi

  EXTRACTED=$((EXTRACTED + 1))
done

echo ""
echo "Done!"
echo "Extracted: ${EXTRACTED}"
echo "Skipped (already exist): ${SKIPPED}"
echo "Errors: ${ERRORS}"
