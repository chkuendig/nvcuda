#!/usr/bin/env bash
# Deck prototype only; does not install a tool or change a Steam prefix.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
devroot=${NVCUDA_DEVROOT:-$HOME/Desktop/zluda/devroot}
out=${NVCUDA_WOW64_OUT:-$repo/build/wow64-prototype}
src=$repo/dlls/nvcuda/wow64
mingw=$devroot/usr/bin/i686-w64-mingw32-gcc
mkdir -p "$out/i386-windows" "$out/x86_64-unix"
python3 "$repo/tests/generate-wow64-unsupported.py" "$repo" "$out"
"$mingw" -O2 -Wall -Wextra -Werror -shared "$src/pe.c" "$out/unsupported.c" "$out/exports.def" \
    -Wl,--kill-at -o "$out/i386-windows/nvcuda.dll"
"$devroot/usr/bin/winebuild" --builtin "$out/i386-windows/nvcuda.dll"
cc -O2 -Wall -Wextra -Werror -fPIC -shared "$src/unix.c" -ldl -pthread \
    -o "$out/x86_64-unix/nvcuda.so"
"$mingw" -O2 -Wall -Wextra -Werror "$repo/tests/wow64_compute.c" \
    -o "$out/wow64-compute.exe"
"$mingw" -O2 -Wall -Wextra -Werror "$repo/tests/nvapi_compute_probe.c" \
    -o "$out/nvapi-compute-probe.exe"
file "$out/i386-windows/nvcuda.dll" "$out/x86_64-unix/nvcuda.so" "$out/wow64-compute.exe"
sha256sum "$out/i386-windows/nvcuda.dll" "$out/x86_64-unix/nvcuda.so" "$out/wow64-compute.exe" "$out/nvapi-compute-probe.exe" > "$out/SHA256SUMS"
