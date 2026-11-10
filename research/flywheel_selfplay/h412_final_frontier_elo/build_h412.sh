#!/usr/bin/env bash
# H412 frontier AUDIT evaluator build (H335 build.sh + grok4p7_xhigh) (standalone rustc; no cargo, no external crates).
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
rustc $TC "${FLAGS[@]}" --crate-type rlib --crate-name grok4p7_xhigh      --out-dir lib src/grok4p7_xhigh/grok4p7_xhigh.rs
rustc $TC --edition 2021 -C opt-level=3 -C codegen-units=1 -C debuginfo=0 -C panic=unwind -C lto=fat \
  --crate-type bin --crate-name frontier_audit_h412 -L dependency=lib \
  --extern opus5p5_extra=lib/libopus5p5_extra.rlib \
  --extern gemini4_argon_high=lib/libgemini4_argon_high.rlib \
  --extern gpt6p1_sol_max=lib/libgpt6p1_sol_max.rlib \
  --extern gpt6_astra_max=lib/libgpt6_astra_max.rlib \
  --extern grok4p7_xhigh=lib/libgrok4p7_xhigh.rlib \
  -o "$OUT" harness/frontier_audit_h412.rs
sha256sum "$OUT"
