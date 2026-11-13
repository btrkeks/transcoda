#!/usr/bin/env bash
set -euo pipefail
# Called explicitly by the reader. No network requests occur at import or install.
root=external/verosim
if [[ -e "$root" ]]; then
  echo "Refusing to overwrite $root" >&2
  exit 1
fi
git clone --no-checkout https://github.com/btrkeks/verosim.git "$root"
git -C "$root" checkout --detach d63cb62b61379a83c619b5001e0425d3411fd84c
git clone --no-checkout https://github.com/btrkeks/verovio.git "$root/verovio"
git -C "$root/verovio" checkout --detach 74a74adbff229ec875289faba3e262a34b0836b9
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "$root/build" -j "${WORKERS:-2}"
