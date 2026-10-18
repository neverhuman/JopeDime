#!/usr/bin/env bash
# Run the strict DL offline module shards concurrently, then verify their receipts with --combine,
# all under ONE common deadline. Serial execution exceeded 900 s on CI before shard 2 completed.
# Every shard and the combine step get only the time left until that deadline.
# Usage: dl-offline-group.sh SECONDS OUTPUT_DIR (from python/, the uv project).
set -euo pipefail
budget=$1 out=$2
[[ $budget =~ ^[1-9][0-9]*$ ]] || { echo "dl-offline-group: SECONDS must be a positive integer, got '$budget'" >&2; exit 2; }
read -r -a runner <<< "${DL_OFFLINE_RUNNER:-uv run --no-sync python ../deploy/dl-trainonly/run_lane_offline.py}"
now_ms() { echo $(( $(date +%s%N) / 1000000 )); }
deadline_ms=$(( $(now_ms) + budget * 1000 ))

run_shard() {
    local index=$1 shard=$2 left_ms rc=0
    left_ms=$(( deadline_ms - $(now_ms) ))
    if (( left_ms <= 0 )); then
        echo "dl-offline-group: the common ${budget}s deadline passed before shard $shard" >&2
        return 124
    fi
    timeout "$(( left_ms / 1000 )).$(printf '%03d' $(( left_ms % 1000 )))s" \
        "${runner[@]}" --shard "$shard" --output "$out/shard-$index" \
        >"$out/shard-$index.run.log" 2>&1 || rc=$?
    if (( rc == 124 )); then
        echo "dl-offline-group: the common ${budget}s deadline expired during shard $shard" >&2
        cat "$out/shard-$index.run.log" >&2
    elif (( rc != 0 )); then
        echo "dl-offline-group: shard $shard failed with status $rc" >&2
        cat "$out/shard-$index.run.log" >&2
    fi
    return "$rc"
}

run_combine() {
    local left_ms rc=0
    left_ms=$(( deadline_ms - $(now_ms) ))
    if (( left_ms <= 0 )); then
        echo "dl-offline-group: the common ${budget}s deadline passed before --combine" >&2
        return 124
    fi
    timeout "$(( left_ms / 1000 )).$(printf '%03d' $(( left_ms % 1000 )))s" \
        "${runner[@]}" --combine --output "$out" || rc=$?
    if (( rc == 124 )); then
        echo "dl-offline-group: the common ${budget}s deadline expired during --combine" >&2
    fi
    return "$rc"
}

mkdir -p "$out"
pids=()
for index in 1 2 3; do
    run_shard "$index" "$index/3" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=$?
done
(( status == 0 )) || exit "$status"
run_combine
echo "dl-offline-group: 3 shards + combine finished within the common ${budget}s deadline ($(( $(now_ms) - deadline_ms + budget * 1000 )) ms)"
