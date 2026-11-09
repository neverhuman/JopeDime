#!/usr/bin/env bash
# H414 two-fold DL-Val91 evaluator build: H257's eval_frontier_dl_val91.rs + kimik3p1_max (standalone rustc 1.97.1;
# no cargo, no external crates). Same flags as the H414 frontier AUDIT build. The stored H257 binary's exact flags
# are not recorded (its h256 source copy no longer exists), so equivalence is shown empirically instead: on the 16
# TRAIN two-fold envelopes the four H257 entrants must give records identical to the stored binary except wall_ms.
# Usage: build.sh <output-binary-path>
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:?output binary path}
export CARGO_INCREMENTAL=0
unset CARGO_TARGET_DIR || true
TC=+1.97.1
FLAGS=(--edition 2021 -C opt-level=3 -C codegen-units=1 -C debuginfo=0 -C panic=unwind --cap-lints allow)
mkdir -p lib
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name opus5p5_extra      --out-dir lib src/opus5p5_extra/Opus5p5_Extra.rs
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name gemini4_argon_high --out-dir lib src/gemini4_argon_high/Gemini4_argon_high.rs
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name gpt6p1_sol_max     --out-dir lib src/gpt6p1_sol_max/GPT6p1_Sol_Max.rs
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name gpt6_astra_max     --out-dir lib src/gpt6_astra_max/GPT6_Astra_Max.rs
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name kimik3p1_max       --out-dir lib src/kimik3p1_max/kimik3p1_max.rs
rustc $TC --edition 2021 -C opt-level=3 -C codegen-units=1 -C debuginfo=0 -C panic=unwind -C lto=fat \
  --crate-type bin --crate-name eval_frontier_dl_val91_h414 -L dependency=lib \
  --extern opus5p5_extra=lib/libopus5p5_extra.rlib \
  --extern gemini4_argon_high=lib/libgemini4_argon_high.rlib \
  --extern gpt6p1_sol_max=lib/libgpt6p1_sol_max.rlib \
  --extern gpt6_astra_max=lib/libgpt6_astra_max.rlib \
  --extern kimik3p1_max=lib/libkimik3p1_max.rlib \
  -o "$OUT" harness/eval_frontier_dl_val91_h414.rs
sha256sum "$OUT"
