#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

bash ops/ci/path-audit.sh
cargo fmt --all -- --check
cargo test --locked --workspace --jobs "${CARGO_BUILD_JOBS:-4}"
cargo run --locked --bin jope | grep -E '^jope 0\.'

printf '%s\n' 'required ok: jope-dime'
