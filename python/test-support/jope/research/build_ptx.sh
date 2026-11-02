#!/usr/bin/env bash
# Regenerate the committed PTX for battle-gpu's ridge-screen kernels.
#
# The COMMITTED .ptx is the source of truth: it is embedded into the crate via
# include_str! and loaded through cuModuleLoadData at runtime, so build hosts
# and CI never need nvcc — only hosts that CHANGE the kernels rerun this.
#
# -fmad=false is load-bearing: it forbids FMA contraction so every f64 op in
# the kernels rounds exactly like the scalar CPU reference (bit-exact scores).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=vendor/battle-gpu/kernels/ridge_screen.cu
OUT=vendor/battle-gpu/kernels/ridge_screen.ptx
nvcc -arch=sm_86 -ptx -fmad=false -o "$OUT" "$SRC"
echo "wrote $OUT"
echo "sha256: $(sha256sum "$OUT" | cut -d' ' -f1)"
echo "(update PTX_SHA256 in vendor/battle-gpu/src/lib.rs if it changed)"
