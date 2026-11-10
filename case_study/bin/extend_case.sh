#!/usr/bin/env bash
# extend_case.sh <case_dir> [--arm A] [--seed 7] [--rounds 3] [--evaluations 1000] [--seconds 600] [--cuts 1] [--threads 1]
#                [--stamp S]
#
# Longer evolution than one search allows. One `jope pipeline run` is capped at 1000 evaluations
# and 600 s, so this chains ROUNDS:
#   round 1 = the arm's own generation 0 (arm A: the DL seeds) under the case objective;
#   round r > 1 = a fresh search whose generation 0 is the bootstrap, the 11 best distinct
#     FEASIBLE programs found in all earlier rounds (re-measured, never trusted) and new random
#     draws from RNG stream seed*100 + r.
# Every round is its own verified receipt (`verify-receipt --evidence`). The lineage is
# rounds.jsonl. generations.csv numbers generations across rounds (global generation index), with
# the best-so-far objective, so the curve shows how far longer evolution pushes the custom
# fitness. Outputs go to ${JOPE_CASE_OUT:-~/jope-data/case_study}/<case>/extended-<stamp>/<arm>/s<seed>/.
# Never reads test.csv rows (the search only verifies its hash).
set -euo pipefail
die() { echo "extend_case: $*" >&2; exit 2; }
[ $# -ge 1 ] || die "usage: $0 <case_dir> [--arm A] [--seed 7] [--rounds 3] [--evaluations 1000] [--seconds 600] [--cuts 1] [--threads 1] [--stamp S]"
case_dir=$(cd "$1" && pwd); shift
arm=A; seed=7; rounds=3; evaluations=1000; seconds=600; stamp=""; cuts=1; threads=1
while [ $# -gt 0 ]; do
  case $1 in
    --arm) arm=$2; shift 2;; --seed) seed=$2; shift 2;; --rounds) rounds=$2; shift 2;;
    --evaluations) evaluations=$2; shift 2;; --seconds) seconds=$2; shift 2;; --stamp) stamp=$2; shift 2;;
    --cuts) cuts=$2; shift 2;; --threads) threads=$2; shift 2;;
    *) die "unknown argument $1";;
  esac
done
{ [ "$seconds" -ge 1 ] && [ "$seconds" -le 600 ]; } || die "--seconds must be 1..=600"
{ [ "$evaluations" -ge 24 ] && [ "$evaluations" -le 1000 ]; } || die "--evaluations must be 24..=1000"
# --cuts K > 1 ranks every program by its mean over K folds of train.csv (the pipeline's cross-cut
# selection); test.csv is never read. Measured-resource objectives still need --threads 1.
{ [ "$cuts" -ge 1 ] && [ "$cuts" -le 5 ]; } || die "--cuts must be 1..=5"
{ [ "$threads" -ge 1 ] && [ "$threads" -le 16 ]; } || die "--threads must be 1..=16"
here=$(cd "$(dirname "$0")" && pwd); root=${JOPE_CASE_ROOT:-$(git -C "$here" rev-parse --show-toplevel)}
jope=${JOPE_BIN:-$root/target/release/jope}; [ -x "$jope" ] || die "no jope binary at $jope"
spec=$case_dir/case.json; name=$(jq -r .name "$spec"); id=$(jq -r .dataset.id "$spec")
"$here/check_sealed.sh" "$id" >/dev/null || die "$id is sealed or held out; refusing"
fitness_key=$(jq -r --arg a "$arm" '.arms[$a].fitness // empty' "$spec"); [ -n "$fitness_key" ] || die "no arm $arm"
fitness_file=$case_dir/$(jq -r --arg k "$fitness_key" '.[$k]' "$spec")
use_seeds=$(jq -r --arg a "$arm" '.arms[$a].seeds' "$spec")
dl_seeds=$case_dir/$(jq -r --arg a "$arm" '.arms[$a].seeds_file // .seeds.file' "$spec")
for f in manifest.json meta.json train.csv test.csv; do
  [ -f "${JOPE_CACHE:-$HOME/.cache/jope}/datasets/by-id/$id/$f" ] || die "$id is not fully cached (run run_case.sh once)"
done
[ -n "$stamp" ] || stamp=$(date -u +%Y%m%dT%H%M%SZ)
out=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}/$name/extended-$stamp/$arm/s$seed
case "$out" in /tmp/*|"$root"/*) die "output must be outside /tmp and the repository";; esac
mkdir -p "$out"; : > "$out/rounds.jsonl"
taskset_cmd=(); [ -n "${JOPE_CASE_CPUS:-}" ] && taskset_cmd=(taskset -c "$JOPE_CASE_CPUS")
jq -n --arg name "$name" --arg id "$id" --arg arm "$arm" --argjson seed "$seed" --argjson rounds "$rounds" \
  --argjson evaluations "$evaluations" --argjson seconds "$seconds" --arg commit "${JOPE_CASE_COMMIT:-$(git -C "$root" rev-parse HEAD)}" \
  --arg jope "$(sha256sum "$jope" | cut -c1-64)" --arg fitness "$(sha256sum "$fitness_file" | cut -c1-64)" \
  --argjson cuts "$cuts" --argjson threads "$threads" --arg host "$(hostname)" --arg load "$(cut -d' ' -f1-3 /proc/loadavg)" \
  '{schema:"jope.case-study-extended/v1", name:$name, dataset_id:$id, arm:$arm, rng_seed:$seed, rounds:$rounds,
    per_round_budget:{evaluations:$evaluations, seconds:$seconds}, source_commit:$commit, jope_sha256:$jope,
    fitness_sha256:$fitness, host:$host, load_at_start:$load, carried_seeds_per_round:11, cuts:$cuts, threads:$threads}' > "$out/run.json"

cv_programs() {  # <round dir>: every program measured on all $cuts cuts, with its mean objective and birth generation
  jq -s --argjson k "$cuts" '[to_entries[] | .key as $c | .value[] | select(.status == "measured" and .objective_loss != null)
      | {g: .genome, c: $c, l: .objective_loss, gen: .generation, o: .origin}]
    | group_by(.g) | map({genome: .[0].g, born: (map(select(.o != "remeasure") | .gen) | min), cov: (map(.c) | unique | length),
        objective_loss: (group_by(.c) | map(map(.l) | add / length) | add / length), status: "measured"})
    | map(select(.cov == $k and .born != null))' "$1"/search/cut-*/observations.json
}
carried() {  # the 11 best distinct feasible programs over every finished round, as a seeds file
  local file=$1; shift
  { if [ "$cuts" -gt 1 ]; then for d in "$@"; do cv_programs "$d"; done | jq -s add; else jq -s add "$@"; fi; } \
  | jq 'map(select(.status == "measured" and .objective_loss != null))
         | sort_by([.objective_loss, .genome]) | unique_by(.genome) | sort_by([.objective_loss, .genome])
         | .[0:11] | map(.genome)' \
  | jq -r '.[]' | while IFS= read -r g; do
      jq -nc --arg g "$g" --arg h "$(printf '%s' "$g" | sha256sum | cut -c1-64)" '{genome:$g, program_sha256:$h}'
    done | jq -s --arg id "$id" '{schema:"jope.pipeline-seeds/v1", dataset_hash:$id, genomes:.}' > "$file"
}

previous=()
for r in $(seq 1 "$rounds"); do
  rd=$out/round-$(printf %02d "$r"); rng=$((seed * 100 + r))
  args=(pipeline run --dataset "$id" --cuts "$cuts" --threads "$threads" --policy skip --emit rust
        --budget "evaluations=$evaluations,seconds=$seconds" --rng-seed "$rng"
        --fitness "@$fitness_file" --output "$rd")
  seeds_used=none
  if [ "$r" -eq 1 ]; then
    if [ "$use_seeds" = true ]; then args+=(--seed-genomes "$dl_seeds" --seed-limit 11); seeds_used=$(basename "$dl_seeds"); fi
  else
    carried "$out/round-$(printf %02d "$r").seeds.json" "${previous[@]}"
    args+=(--seed-genomes "$out/round-$(printf %02d "$r").seeds.json" --seed-limit 11)
    seeds_used=round-$(printf %02d "$r").seeds.json
  fi
  t0=$(date -u +%FT%TZ); set +e
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS="${JOPE_CASE_RAYON:-4}" \
    nice -n 19 "${taskset_cmd[@]}" "$jope" "${args[@]}" > "$rd.log" 2>&1
  rc=$?; vrc=-1
  if [ $rc = 0 ]; then "$jope" pipeline verify-receipt --receipt "$rd/receipt.json" --evidence > "$rd.verify.log" 2>&1; vrc=$?; fi
  set -e
  obs=$rd/search/cut-0/observations.json
  summary='{}'
  if [ $rc = 0 ]; then
    summary=$(jq -c --slurpfile s "$rd/search/cut-0/search.json" '{
        evaluations: (map(select(.origin != "remeasure")) | length),
        generations: (map(.generation) | max + 1),
        refused: (map(select((.failure // "") | contains("resource_infeasible"))) | length),
        best_objective: (map(select(.status == "measured") | .objective_loss) | min),
        stopped_by: ($s[0].stopped_by // null)}' "$obs")
    if [ "$cuts" -gt 1 ]; then
      summary=$(jq -c --argjson s "$summary" --argjson k "$cuts" '{cuts: $k, best_objective: .search.best.validation_objective, best_rmse: .search.best.validation_rmse,
          evaluations: ([.search.cuts[].completed_evaluations] | add), generations: ([.search.cuts[].generations] | max)} as $cv | $s + $cv' "$rd/receipt.json")
      previous+=("$rd")
    else previous+=("$obs"); fi
  fi
  jq -nc --argjson r "$r" --argjson rng "$rng" --arg t0 "$t0" --arg t1 "$(date -u +%FT%TZ)" --argjson rc $rc \
    --argjson vrc $vrc --arg seeds "$seeds_used" --argjson s "$summary" --arg dir "$rd" \
    '{round:$r, rng_seed:$rng, seeds:$seeds, started:$t0, ended:$t1, exit:$rc, verify_exit:$vrc, dir:$dir,
      status:(if $rc != 0 then "failed" elif $vrc != 0 then "unverified" else "verified" end)} + $s' >> "$out/rounds.jsonl"
  [ $rc = 0 ] && [ $vrc = 0 ] || { echo "extend_case: round $r failed (see $rd.log)"; break; }
done

# Best-so-far objective over a global generation index across rounds.
echo "round,generation,global_generation,best_so_far_objective,round_best_objective" > "$out/generations.csv"
offset=0; best=""
while read -r line; do
  [ "$(jq -r .status <<<"$line")" = verified ] || continue
  rd=$(jq -r .dir <<<"$line"); r=$(jq -r .round <<<"$line")
  if [ "$cuts" -gt 1 ]; then  # generation g: the best 5-fold mean among programs born in generation g
    cv_programs "$rd" | jq -r --argjson r "$r" --argjson off "$offset" \
      --argjson last "$(jq -s '[.[][] | .generation] | max' "$rd"/search/cut-*/observations.json)" '
      . as $p | range(0; $last + 1) as $g | [$r, $g, ($off + $g), ($p | map(select(.born == $g) | .objective_loss) | min)] | @csv'
  else
  jq -r --argjson r "$r" --argjson off "$offset" '
    map(select(.origin != "remeasure")) as $o | ($o | map(.generation) | max) as $last
    | range(0; $last + 1) as $g
    | ($o | map(select(.generation == $g and .status == "measured") | .objective_loss) | min) as $gb
    | [$r, $g, ($off + $g), $gb] | @csv' "$rd/search/cut-0/observations.json"
  fi \
  | gawk -F, -v best="$best" '{ if ($4 != "" && (best == "" || $4 + 0 < best + 0)) best = $4; print $1 "," $2 "," $3 "," best "," $4 }' \
    >> "$out/generations.csv"
  best=$(tail -1 "$out/generations.csv" | cut -d, -f4)
  offset=$(( offset + $(jq -s '[.[][] | select(.origin != "remeasure") | .generation] | max + 1' "$rd"/search/cut-*/observations.json) ))
done < "$out/rounds.jsonl"
jq -s -c '{rounds: length, verified: map(select(.status == "verified")) | length,
  generations_total: (map(.generations // 0) | add), evaluations_total: (map(.evaluations // 0) | add),
  best_objective_by_round: map(.best_objective)}' "$out/rounds.jsonl" | tee "$out/extended-summary.json"
echo "extend_case: results in $out"
