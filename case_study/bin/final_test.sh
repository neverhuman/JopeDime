#!/usr/bin/env bash
# final_test.sh <case_dir> [--main STAMP] [--extended STAMP]
#
# The pre-declared single look at the sealed test split, run only after every selection is
# frozen. For each case it collects the champions of
#   the main run's arm A and arm C (each RNG seed), and
#   the extended run's final round for arm A (each RNG seed), when one exists,
# and scores each distinct genome ONCE with
#   jope pipeline score --split test --fitness @objective.json --registry <case-study registry>
# The score refits on every train.csv row, measures the same resource readings on the test rows
# and reports the business objective, test RMSE and R². The registry refuses a second look at a
# (dataset, genome), so a genome that several runs selected is scored once and reported for each.
# Nothing here feeds back into selection. Output: <case out>/test-look/<stamp>/looks.jsonl.
set -euo pipefail
# Run dirs recorded inside the xbabe4 container (/w/home/…) are mapped onto this host's evidence tree.
export JOPE_EV=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
die() { echo "final_test: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "usage: $0 <case_dir> [--main STAMP] [--extended STAMP]"
case_dir=$(cd "$1" && pwd); shift; main=""; extended=""
while [ $# -gt 0 ]; do
  case $1 in --main) main=$2; shift 2;; --extended) extended=$2; shift 2;; *) die "unknown argument $1";; esac
done
here=$(cd "$(dirname "$0")" && pwd); root=${JOPE_CASE_ROOT:-$(git -C "$here" rev-parse --show-toplevel)}
jope=${JOPE_BIN:-$root/target/release/jope}; [ -x "$jope" ] || die "no jope binary at $jope"
name=$(basename "$case_dir"); base=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}/$name
registry=${JOPE_CASE_TEST_REGISTRY:-$HOME/jope-data/case_study/test-registry}
[ -n "$main" ] || main=$(basename "$( { ls -d "$base"/2*Z 2>/dev/null || true; } | { grep -v smoke || true; } | while read -r r; do [ -f "$r/SUPERSEDED.txt" ] || ! grep -q '"arm":"A"' "$r/runs.jsonl" 2>/dev/null || echo "$r"; done | tail -1)")
out=$base/test-look/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$out" "$registry"
{ jq -r 'select(.status == "verified" and (.arm == "A" or .arm == "C")) | "main \(.arm) \(.rng_seed) \(.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))"' "$base/$main/runs.jsonl"
  if [ -n "$extended" ]; then
    for s in "$base/extended-$extended"/A/s*; do
      last=$(jq -r 'select(.status == "verified") | (.dir | sub("^.*/jope-data/case_study/"; $ENV.JOPE_EV + "/"))' "$s/rounds.jsonl" | tail -1)
      [ -n "$last" ] && echo "extended A ${s##*/s} $last"
    done
  fi
} > "$out/candidates.txt"
declare -A looked
while read -r source arm seed dir; do
  genome=$(jq -r .search.selected_genome "$dir/receipt.json")
  sha=$(printf '%s' "$genome" | sha256sum | cut -c1-64)
  if [ -z "${looked[$sha]:-}" ]; then
    set +e
    "$jope" pipeline score --receipt "$dir/receipt.json" --split test --fitness "@$case_dir/objective.json" \
      --registry "$registry" --json > "$out/$sha.json" 2> "$out/$sha.err"
    rc=$?; set -e
    looked[$sha]=$rc
  fi
  jq -nc --arg source "$source" --arg arm "$arm" --argjson seed "$seed" --arg dir "$dir" --arg sha "$sha" \
    --argjson rc "${looked[$sha]}" --slurpfile s <( [ -s "$out/$sha.json" ] && cat "$out/$sha.json" || echo null ) \
    '{source:$source, arm:$arm, rng_seed:$seed, run:$dir, program_sha256:$sha, exit:$rc, score:$s[0]}' >> "$out/looks.jsonl"
done < "$out/candidates.txt"
echo "final_test: $(wc -l < "$out/looks.jsonl") looks recorded in $out/looks.jsonl"
