#!/usr/bin/env bash
# docs/paper/eurekazero/scripts/synthesize.sh — regenerate the SYNTHETIC north-star data.
#
#   bash scripts/synthesize.sh            rewrite data/synthetic/*.csv and generated/numbers.tex
#   bash scripts/synthesize.sh OUTDIR     write the same files into OUTDIR (used by check.sh)
#
# Everything this writes is illustrative and labelled SYNTHETIC; none of it is a measured result.
# The generator (synthesize.awk) is deterministic for its fixed seed on any POSIX awk.
set -euo pipefail
paper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [ $# -gt 1 ]; then echo "usage: $0 [OUTDIR]" >&2; exit 2; fi

if [ $# -eq 1 ]; then
  out=$1
  mkdir -p -- "$out"
  awk -v out="$out" -f "$paper/scripts/synthesize.awk"
  exit 0
fi

repo=$(git -C "$paper" rev-parse --show-toplevel)
stage="$repo/target/eurekazero-paper/synthetic-stage"
rm -rf -- "$stage"
mkdir -p -- "$stage"
awk -v out="$stage" -f "$paper/scripts/synthesize.awk"
mv -f -- "$stage/numbers.tex" "$paper/generated/numbers.tex"
rm -f -- "$paper"/data/synthetic/*.csv
mv -f -- "$stage"/*.csv "$paper/data/synthetic/"
rmdir -- "$stage"
echo "synthetic north-star data regenerated in data/synthetic/ and generated/numbers.tex"
