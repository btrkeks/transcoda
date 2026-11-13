#!/usr/bin/env bash
# Rebuild the README figures from the TikZ sources in this directory.
# Needs pdflatex (TeX Live with pgfplots, FiraSans, sourcecodepro), pdftoppm and ImageMagick.
# Usage: docs/figures/src/build.sh [name ...]
set -euo pipefail
cd "$(dirname "$0")"
out=..
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
names=("$@")
[[ ${#names[@]} -gt 0 ]] || names=(teaser architecture synthetic-data qualitative)
for name in "${names[@]}"; do
    TEXINPUTS=".:" pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$build" "$name.tex" > "$build/$name.log" || { tail -30 "$build/$name.log"; exit 1; }
    # 1700 px wide: about 2x the README content width.
    pdftoppm -png -scale-to-x 1700 -scale-to-y -1 -singlefile "$build/$name.pdf" "$build/$name"
    # White background for dark mode; a 64-color palette keeps files small.
    magick "$build/$name.png" -background white -alpha remove -alpha off +dither -colors 64 -strip -define png:compression-level=9 "$out/$name.png"
    echo "$out/$name.png: $(magick identify -format '%wx%h' "$out/$name.png")"
done
