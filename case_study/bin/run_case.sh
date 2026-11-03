#!/usr/bin/env bash
# run_case.sh <case_dir> [--arms A,B,C,D] [--seeds 7,11,13] [--smoke] [--out DIR] [--stamp S] [--no-summary]
#
# Runs one case study through the product program search (`jope pipeline run`) and writes every
# run, its log, its evidence verification and a per-generation summary OUTSIDE the repository:
#   ${JOPE_CASE_OUT:-$HOME/jope-data/case_study}/<case name>/<UTC stamp>/
#
# Arms (case.json "arms"): each pairs a generation-0 source with a ranking objective.
#   A  DL seeds   + resource-aware objective      B  random gen 0 + resource-aware objective
#   C  DL seeds   + accuracy-only baseline         D  random gen 0 + accuracy-only baseline
# At one --rng-seed the random draws are identical across arms, so A vs B isolates the seeds and
# A vs C isolates the objective. All arms share one evaluation budget (compute-matched).
#
# Environment:
#   JOPE_BIN (default <repo>/target/release/jope)   JOPE_CASE_CPUS (taskset list, optional)
#   JOPE_CASE_RAYON (rayon threads per fit, default 4)   AWS_PROFILE (for a first fetch)
# Measured resources need one candidate at a time, so every run uses --threads 1. Never reads
# test.csv: scoring the sealed test split is a separate, once-per-genome `jope pipeline score`.
set -euo pipefail
die() { echo "run_case: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "usage: $0 <case_dir> [--arms A,B,C,D] [--seeds 7,11,13] [--smoke] [--out DIR]"
case_dir=$(cd "$1" && pwd); shift
arms=""; seeds=""; smoke=0; stamp=""; summary=1; out_root=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}
while [ $# -gt 0 ]; do
  case $1 in
    --arms) arms=$2; shift 2;;
    --seeds) seeds=$2; shift 2;;
    --smoke) smoke=1; shift;;
    --out) out_root=$2; shift 2;;
    --stamp) stamp=$2; shift 2;;
    --no-summary) summary=0; shift;;
    *) die "unknown argument $1";;
  esac
done
here=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
jope=${JOPE_BIN:-$root/target/release/jope}
[ -x "$jope" ] || die "no jope binary at $jope (cargo build --locked --release -p jope-cli --bin jope)"
spec=$case_dir/case.json
[ -f "$spec" ] || die "$spec missing"
jq -e '.schema == "jope.case-study/v1"' "$spec" >/dev/null || die "$spec is not jope.case-study/v1"
name=$(jq -r .name "$spec"); id=$(jq -r .dataset.id "$spec")
[ "$name" = "$(basename "$case_dir")" ] || die "case.json name $name != directory $(basename "$case_dir")"
case "$out_root" in /tmp/*|"$root"/*) die "--out must be outside /tmp and the repository";; esac

"$here/check_sealed.sh" "$id" || die "$id is sealed or held out; refusing"

cache=${JOPE_CACHE:-$HOME/.cache/jope}/datasets/by-id/$id
if [ ! -f "$cache/manifest.json" ]; then
  echo "run_case: fetching $id into the dataset cache"
  AWS_PROFILE=${AWS_PROFILE:-veox-jepson-prod} "$jope" datasets fetch "$id" >/dev/null
fi
[ -f "$cache/train.csv" ] || die "$id has no cached train.csv"

[ -n "$arms" ] || arms=$(jq -r '.arms | keys | join(",")' "$spec")
[ -n "$seeds" ] || seeds=$(jq -r '.rng_seeds | map(tostring) | join(",")' "$spec")
evaluations=$(jq -r .budget.evaluations "$spec"); seconds=$(jq -r .budget.seconds "$spec")
if [ $smoke = 1 ]; then evaluations=$(jq -r '.budget.smoke_evaluations // 20' "$spec"); seconds=600; seeds=${seeds%%,*}; fi
seed_file=$case_dir/$(jq -r .seeds.file "$spec"); seed_limit=$(jq -r .seeds.limit "$spec")

[ -n "$stamp" ] || stamp=$(date -u +%Y%m%dT%H%M%SZ); out=$out_root/$name/$stamp$( [ $smoke = 1 ] && echo -smoke)
mkdir -p "$out"
taskset_cmd=(); [ -n "${JOPE_CASE_CPUS:-}" ] && taskset_cmd=(taskset -c "$JOPE_CASE_CPUS")
sha() { sha256sum "$1" | cut -c1-64; }
[ -f "$out/run.json" ] || jq -n --arg name "$name" --arg id "$id" --arg stamp "$stamp" --arg host "$(hostname)" \
  --arg commit "$(git -C "$root" rev-parse HEAD)" --arg dirty "$(git -C "$root" status --porcelain -- case_study crates | wc -l)" \
  --arg binary "$(sha "$jope")" --arg case_sha "$(sha "$spec")" --arg arms "$arms" --arg seeds "$seeds" \
  --argjson evaluations "$evaluations" --argjson seconds "$seconds" --argjson smoke $smoke \
  --arg load "$(cut -d' ' -f1-3 /proc/loadavg)" --arg cpus "${JOPE_CASE_CPUS:-all}" --arg rayon "${JOPE_CASE_RAYON:-4}" \
  '{schema:"jope.case-study-run/v1", name:$name, dataset_id:$id, started_utc:$stamp, host:$host,
    source_commit:$commit, dirty_source_files:($dirty|tonumber), jope_sha256:$binary, case_json_sha256:$case_sha,
    arms:($arms|split(",")), rng_seeds:($seeds|split(",")|map(tonumber)),
    budget:{evaluations:$evaluations, seconds:$seconds}, smoke:($smoke==1),
    limits:{threads:1, nice:19, cpus:$cpus, rayon_threads:($rayon|tonumber), numeric_threads:1}, load_at_start:$load}' \
  > "$out/run.json"

for arm in ${arms//,/ }; do
  fitness_key=$(jq -r --arg a "$arm" '.arms[$a].fitness // empty' "$spec")
  [ -n "$fitness_key" ] || die "case has no arm $arm"
  fitness_file=$case_dir/$(jq -r --arg k "$fitness_key" '.[$k]' "$spec")
  use_seeds=$(jq -r --arg a "$arm" '.arms[$a].seeds' "$spec")
  for seed in ${seeds//,/ }; do
    dir=$out/$arm/s$seed; mkdir -p "$out/$arm"
    args=(pipeline run --dataset "$id" --cuts 1 --threads 1 --policy skip --emit rust
          --budget "evaluations=$evaluations,seconds=$seconds" --rng-seed "$seed"
          --fitness "@$fitness_file" --output "$dir")
    status=run
    if [ "$use_seeds" = true ]; then
      if [ ! -s "$seed_file" ]; then
        jq -nc --arg arm "$arm" --argjson seed "$seed" '{arm:$arm, rng_seed:$seed, status:"pending: no DL seeds file"}' | flock "$out/runs.jsonl.lock" tee -a "$out/runs.jsonl" >/dev/null
        echo "run_case: $arm s$seed pending (no $seed_file)"; continue
      fi
      args+=(--seed-genomes "$seed_file" --seed-limit "$seed_limit")
    fi
    t0=$(date -u +%FT%TZ); echo "run_case: $arm s$seed -> $dir"
    set +e
    env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS="${JOPE_CASE_RAYON:-4}" \
      nice -n 19 "${taskset_cmd[@]}" "$jope" "${args[@]}" > "$dir.log" 2>&1
    rc=$?
    vrc=-1
    if [ $rc = 0 ]; then "$jope" pipeline verify-receipt --receipt "$dir/receipt.json" --evidence > "$dir.verify.log" 2>&1; vrc=$?; fi
    set -e
    jq -nc --arg arm "$arm" --argjson seed "$seed" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" \
      --argjson rc $rc --argjson vrc $vrc --arg dir "$dir" --arg fitness "$(basename "$fitness_file")" \
      --arg seeds "$( [ "$use_seeds" = true ] && basename "$seed_file" || echo none)" \
      '{arm:$arm, rng_seed:$seed, fitness:$fitness, seeds:$seeds, started:$t0, ended:$t1, exit:$rc, verify_exit:$vrc, dir:$dir,
        status:(if $rc != 0 then "failed" elif $vrc != 0 then "unverified" else "verified" end)}' | flock "$out/runs.jsonl.lock" tee -a "$out/runs.jsonl" >/dev/null
  done
done
if [ $summary = 1 ]; then "$here/summarize.sh" "$out" | tee "$out/SUMMARY.md"; fi
echo "run_case: results in $out"
