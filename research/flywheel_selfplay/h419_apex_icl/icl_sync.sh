#!/bin/bash
# H419 addendum 3: coordinator of the parameter-averaging group (runs on xbabe2 as a systemd user unit).
# Rounds are wall-clock: round k = floor(unix time / S). In round k every worker writes out_<name>.pt and
# out_<name>.json {"round": k}; this script collects the weights that arrive within OUT_WAIT seconds, averages them
# (icl_avg.py) and returns in.pt / in.json {"round": k, "n", "dist"} to every worker that took part. A worker that gets
# no mean within its own wait continues on its own weights. Exchange files are fixed names and are replaced each
# round; checkpoints are untouched.
#   icl_sync.sh CODE_DIR GROUP_DIR S OUT_WAIT name:host:syncdir ...      (host "local" = this machine)
set -uo pipefail
CODE=${1:?}; G=${2:?}; S=${3:?}; OUT_WAIT=${4:?}; shift 4
mkdir -p "$G"; LOG=$G/rounds.jsonl
sh() { if [ "$1" = local ]; then bash -c "$2"; else ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "$2"; fi; }
last=$(( $(date +%s) / S ))
echo "$(date -u +%FT%TZ) coordinator start S=$S workers: $*" >> "$G/coordinator.log"
while :; do
  k=$(( $(date +%s) / S ))
  if [ "$k" -le "$last" ]; then sleep 1; continue; fi
  last=$k; t0=$(date +%s); deadline=$(( k * S + OUT_WAIT )); present=()
  while :; do
    present=()
    for w in "$@"; do
      IFS=: read -r name host dir <<< "$w"
      r=$(sh "$host" "jq -r .round $dir/out_$name.json 2>/dev/null" 2>/dev/null || true)
      [ "$r" = "$k" ] && present+=("$w")
    done
    [ "${#present[@]}" = "$#" ] && break
    [ "$(date +%s)" -ge "$deadline" ] && break
    sleep 2
  done
  if [ "${#present[@]}" -lt 2 ]; then
    echo "{\"round\": $k, \"n\": ${#present[@]}, \"skipped\": true, \"time\": \"$(date -u +%FT%TZ)\"}" >> "$LOG"; continue
  fi
  files=()
  for w in "${present[@]}"; do
    IFS=: read -r name host dir <<< "$w"
    if [ "$host" = local ]; then cp "$dir/out_$name.pt" "$G/out_$name.pt" & else rsync -a "$host:$dir/out_$name.pt" "$G/out_$name.pt" & fi
    files+=("$G/out_$name.pt")
  done
  wait; t1=$(date +%s)
  res=$(PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 python3 -B "$CODE/icl_avg.py" --out "$G/in.pt.tmp" "${files[@]}" 2>>"$G/coordinator.log") || { echo "{\"round\": $k, \"error\": \"average failed\"}" >> "$LOG"; continue; }
  mv "$G/in.pt.tmp" "$G/in.pt"; t2=$(date +%s)
  meta=$(echo "$res" | jq -c --argjson k "$k" '. + {round: $k}')
  for w in "${present[@]}"; do
    IFS=: read -r name host dir <<< "$w"
    ( if [ "$host" = local ]; then cp "$G/in.pt" "$dir/in.pt.tmp" && mv "$dir/in.pt.tmp" "$dir/in.pt" && echo "$meta" > "$dir/in.json.tmp" && mv "$dir/in.json.tmp" "$dir/in.json"
      else rsync -a "$G/in.pt" "$host:$dir/in.pt.tmp" && echo "$meta" | ssh -o BatchMode=yes "$host" "mv $dir/in.pt.tmp $dir/in.pt && cat > $dir/in.json.tmp && mv $dir/in.json.tmp $dir/in.json"; fi ) &
  done
  wait; t3=$(date +%s)
  names=$(printf '%s\n' "${present[@]}" | cut -d: -f1 | paste -sd, -)
  echo "$meta" | jq -c --arg names "$names" --arg time "$(date -u +%FT%TZ)" --argjson a $((t1 - t0)) --argjson b $((t2 - t1)) --argjson c $((t3 - t2)) '. + {workers: $names, time: $time, collect_pull_s: $a, average_s: $b, push_s: $c}' >> "$LOG"
done
