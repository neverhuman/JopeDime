#!/usr/bin/env bash
# calibrate.sh <case_dir> : turn case.json's objective_template into objective.json, from a
# calibration run declared in case.json ("calibration": arm D, its own rng_seed, smoke budget).
#
# The rules are fixed before any main run and recorded in case.json "calibration":
#   {"$median": "<resource>"}   -> the median reading of that resource over the calibration
#                                  run's measured candidates, to 2 significant figures (a limit
#                                  at the median binds for about half of them);
#   {"$target_sigma": true}     -> the standard deviation of the train.csv target column;
#   {"$price": {"accuracy": A, "resource": R, "share": s}}
#                               -> s x median accuracy loss / median R reading, so a candidate at
#                                  the median reading pays s of the median error.
# The calibration rng_seed is never one of the main rng_seeds. Writes objective.json and the
# calibration record; re-running overwrites both from a new calibration run.
set -euo pipefail
case_dir=$(cd "${1:?usage: calibrate.sh <case_dir>}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
spec=$case_dir/case.json
id=$(jq -r .dataset.id "$spec"); seed=$(jq -r .calibration.rng_seed "$spec")
jq -e --argjson s "$seed" '.rng_seeds | index($s) | not' "$spec" >/dev/null \
  || { echo "calibrate: calibration rng_seed $seed is also a main seed" >&2; exit 2; }
out=$("$here/run_case.sh" "$case_dir" --arms D --seeds "$seed" --smoke | sed -n 's/^run_case: results in //p')
[ -d "$out" ] || { echo "calibrate: no calibration output" >&2; exit 2; }
obs=$out/D/s$seed/search/cut-0/observations.json
[ -f "$obs" ] || { echo "calibrate: calibration run failed; see $out/D/s$seed.log" >&2; exit 2; }
train=${JOPE_CACHE:-$HOME/.cache/jope}/datasets/by-id/$id/train.csv
sigma=$(gawk -F, '{ y = $NF; n++; s += y; ss += y * y } END { m = s / n; printf "%.6g", sqrt(ss / n - m * m) }' "$train")
medians=$(jq '[.[] | select(.status == "measured" and .origin != "remeasure")] as $m
  | ($m | map(.resources // {} | keys[]) | unique) as $keys
  | ($keys | map(. as $k | {key: $k, value: ($m | map(.resources[$k]) | sort | .[(length / 2 | floor)])}) | from_entries)
    + {mae: ($m | map(.objective_loss) | sort | .[(length / 2 | floor)]), candidates: ($m | length)}' "$obs")
jq --argjson med "$medians" --argjson sigma "$sigma" '
  def sig2: if . == 0 then 0 else (. as $x | ($x | fabs | log10 | floor) as $e | (($x / pow(10; $e - 1)) | round) * pow(10; $e - 1)) end;
  .objective_template | walk(
    if type == "object" and has("$median") then ($med[.["$median"]] // error("no reading for " + .["$median"])) | sig2
    elif type == "object" and has("$target_sigma") then $sigma | sig2
    elif type == "object" and has("$price") then (.["$price"] as $p | ($p.share * $med[$p.accuracy] / $med[$p.resource]) | sig2)
    else . end)' "$spec" > "$case_dir/objective.json"
jq --argjson med "$medians" --argjson sigma "$sigma" --arg out "$out" --arg sha "$(sha256sum "$obs" | cut -c1-64)" \
  '.calibration += {status: "calibrated", run: $out, observations_sha256: $sha, target_sigma: $sigma, medians: $med,
    rules: "limits = median calibration reading (2 s.f.); composite resource scale = median reading, accuracy scale = train target sigma; prices = share x median error / median reading"}' \
  "$spec" > "$spec.tmp" && mv "$spec.tmp" "$spec"
echo "calibrate: wrote $case_dir/objective.json"; jq -c . "$case_dir/objective.json"
