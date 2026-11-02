#!/usr/bin/env bash
# CPU-only plan emission. --execute is refused unconditionally by runner.py.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 OMP_MAX_ACTIVE_LEVELS=1
export OMP_NESTED=FALSE PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec nice -n 19 python3 -B "$script_dir/runner.py" "$@"
