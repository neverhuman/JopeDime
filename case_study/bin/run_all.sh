#!/usr/bin/env bash
# run_all.sh <calibrate|main|extended> [--jobs N] [--rounds R] [--cases "01 02 ..."] [--arms B,D] [--seeds 7,11,13]
#
# calibrate: run bin/calibrate.sh for every selected case not yet calibrated (N in parallel).
# main:      run every (case, arm, seed) as its own job, N in parallel, all jobs of one case into
#            one shared output directory (one UTC stamp), then write each case's SUMMARY.md.
# Each job is one serial program search (--threads 1) whose fits use JOPE_CASE_RAYON threads
# (default 4), so N jobs use about 4N cores; keep 4N plus the host's load under 85% of its cores.
set -euo pipefail
phase=${1:?usage: run_all.sh <calibrate|main|extended> [--jobs N] [--cases "01 02"] [--arms B,D] [--seeds 7,11,13]}; shift
jobs=8; cases=""; arms=""; seeds=""; rounds=3
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) jobs=$2; shift 2;; --cases) cases=$2; shift 2;;
    --arms) arms=$2; shift 2;; --seeds) seeds=$2; shift 2;;
    --rounds) rounds=$2; shift 2;;
    *) echo "run_all: unknown argument $1" >&2; exit 2;;
  esac
done
here=$(cd "$(dirname "$0")" && pwd); top=$(dirname "$here")
dirs=()
for d in "$top"/[0-9][0-9]_*/; do
  n=$(basename "$d"); [ -z "$cases" ] || [[ " $cases " == *" ${n%%_*} "* ]] || continue; dirs+=("${d%/}")
done
cores=$(nproc); load=$(cut -d' ' -f2 /proc/loadavg)   # 5-minute average: CI shards burst the 1-minute one
gawk -v l="$load" -v j="$jobs" -v c="$cores" -v r="${JOPE_CASE_RAYON:-4}" 'BEGIN { if (l + j * r > 0.85 * c) { printf "run_all: load %.1f + %d jobs x %d threads exceeds 85%% of %d cores\n", l, j, r, c > "/dev/stderr"; exit 1 } }'
case $phase in
  calibrate)
    for d in "${dirs[@]}"; do
      jq -e '.calibration.status == "calibrated"' "$d/case.json" >/dev/null && continue
      echo "$d"
    done | xargs -r -P "$jobs" -I{} sh -c '"$1/calibrate.sh" "$2" > "$3/$(basename "$2").calibrate.log" 2>&1 && echo "calibrated $(basename "$2")" || echo "FAILED $(basename "$2") (see $3/$(basename "$2").calibrate.log)"' _ "$here" {} "${JOPE_CASE_OUT:-$HOME/jope-data/case_study}"
    for d in "${dirs[@]}"; do "$here/render_readme.sh" "$d" >/dev/null; done ;;
  main)
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    for d in "${dirs[@]}"; do
      [ -f "$d/objective.json" ] || { echo "run_all: $(basename "$d") is not calibrated; skipping" >&2; continue; }
      a=${arms:-$(jq -r '.arms | keys | join(",")' "$d/case.json")}
      s=${seeds:-$(jq -r '.rng_seeds | map(tostring) | join(",")' "$d/case.json")}
      for arm in ${a//,/ }; do for seed in ${s//,/ }; do echo "$d $arm $seed"; done; done
    done | xargs -r -P "$jobs" -L1 sh -c '"$1/run_case.sh" "$3" --arms "$4" --seeds "$5" --stamp "$2" --no-summary > /dev/null 2>&1; echo "done $(basename "$3") $4 s$5 rc=$?"' _ "$here" "$stamp" || true
    for d in "${dirs[@]}"; do
      out=${JOPE_CASE_OUT:-$HOME/jope-data/case_study}/$(basename "$d")/$stamp
      [ -f "$out/runs.jsonl" ] && "$here/summarize.sh" "$out" > "$out/SUMMARY.md" && echo "summary $out/SUMMARY.md"
    done ;;
  extended)
    # Chained-round evolution (extend_case.sh) for every selected case x seed, one shared stamp.
    stamp=$(date -u +%Y%m%dT%H%M%SZ); echo "run_all: extended stamp $stamp"
    for d in "${dirs[@]}"; do
      [ -f "$d/objective.json" ] || { echo "run_all: $(basename "$d") is not calibrated; skipping" >&2; continue; }
      s=${seeds:-$(jq -r '.rng_seeds | map(tostring) | join(",")' "$d/case.json")}
      for seed in ${s//,/ }; do echo "$d $seed"; done
    done | xargs -r -P "$jobs" -L1 sh -c '"$1/extend_case.sh" "$5" --arm "$3" --seed "$6" --rounds "$2" --stamp "$4" > /dev/null 2>&1; echo "done $(basename "$5") s$6 rc=$?"' _ "$here" "$rounds" "${arms:-A}" "$stamp" ;;
  *) echo "run_all: phase must be calibrate or main" >&2; exit 2;;
esac
