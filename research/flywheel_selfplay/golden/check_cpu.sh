#!/bin/sh
# Cargo external-subcommand entry for remote_test.sh auto; no Rust compilation or shared-target writes.
set -eu
source_sha=${1:?explicit source SHA required}
case "$source_sha" in *[!0-9a-f]*|'') exit 2 ;; esac
[ "${#source_sha}" = 40 ] || exit 2
cd "$(dirname "$0")/../../.."
run_root="$HOME/jope_plan/evidence/RES-07/gb1915/cpu-$source_sha"
mkdir -p "$run_root/scratch"
export TMPDIR="$run_root/scratch" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CUDA_VISIBLE_DEVICES=''
export JOPE_RES07_BUNDLE="${JOPE_RES07_BUNDLE:-$HOME/jope-data/res-07/gb1915/model-prior-bundle}"
export JOPE_RES07_BUNDLE_DIGEST='sha256:94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c'
printf 'RES07_CPU_SOURCE=%s\n' "$source_sha"
python3 -m unittest discover -s research/flywheel_selfplay/golden -p test_generate.py -v
for seed in 17 2026; do
    PYTHONHASHSEED=$seed python3 research/flywheel_selfplay/golden/generate.py \
        --bundle "$JOPE_RES07_BUNDLE" --expected-bundle-digest "$JOPE_RES07_BUNDLE_DIGEST" \
        > "$run_root/validation-$seed.json"
done
cmp "$run_root/validation-17.json" "$run_root/validation-2026.json"
cat "$run_root/validation-17.json"
printf '\nRES07_CPU_REPORTS_BYTE_IDENTICAL=1; fixture_files_written=0\n'
