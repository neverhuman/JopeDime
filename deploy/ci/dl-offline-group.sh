#!/usr/bin/env bash
# deploy/ci/dl-offline-group.sh — the strict DL offline lane as its pinned module shards plus --combine under ONE common
# deadline: the budget the single-process lane had (fast-lane.sh passes 900). Every step runs under `timeout` with only the
# time left until that deadline, so shards that each fit on their own still fail together when their sum (combine included)
# does not; a step that would start after the deadline fails without starting. Exit 124 on either deadline failure, otherwise
# the first failing step's status. Run from python/ (the uv project). Usage: dl-offline-group.sh SECONDS OUTPUT_DIR
# DL_OFFLINE_RUNNER replaces the runner command prefix for deploy/ci/test-dl-offline-group.py only; fast-lane.sh unsets it.
set -euo pipefail
budget=$1 out=$2
[[ $budget =~ ^[1-9][0-9]*$ ]] || { echo "dl-offline-group: SECONDS must be a positive integer, got '$budget'" >&2; exit 2; }
read -r -a runner <<< "${DL_OFFLINE_RUNNER:-uv run --no-sync python ../deploy/dl-trainonly/run_lane_offline.py}"
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
deadline_ms=$(( $(now_ms) + budget * 1000 ))
step() {
    local left_ms rc=0
    left_ms=$(( deadline_ms - $(now_ms) ))
    if (( left_ms <= 0 )); then
        echo "dl-offline-group: the common ${budget}s deadline passed before: $*" >&2
        return 124
    fi
    timeout "$(( left_ms / 1000 )).$(printf '%03d' $(( left_ms % 1000 )))s" "${runner[@]}" "$@" || rc=$?
    if (( rc == 124 )); then
        echo "dl-offline-group: the common ${budget}s deadline expired during: $*" >&2
    fi
    return "$rc"
}
for shard in 1/3 2/3 3/3; do
    step --shard "$shard" --output "$out/shard-${shard%/*}"
done
step --combine --output "$out"
echo "dl-offline-group: 3 shards + combine finished within the common ${budget}s deadline ($(( $(now_ms) - deadline_ms + budget * 1000 )) ms)"
