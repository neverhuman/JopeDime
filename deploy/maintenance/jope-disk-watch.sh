#!/usr/bin/env bash
# Hourly, read-only disk pressure receipt. Keep receipts on the archive volume.
set -euo pipefail
config=${1:?usage: jope-disk-watch.sh CONFIG CUSTODY}
custody=${2:?usage: jope-disk-watch.sh CONFIG CUSTODY}
mkdir -p -- "$custody"
exec 9>"$custody/.lock"
flock -n 9 || exit 0
scratch=$(mktemp -d -p "$custody" .disk-watch.XXXXXXXX)
trap 'rm -r -- "$scratch"' EXIT
now=$(date -u +%s)
stamp=$(date -u +%Y%m%dT%H%M%SZ)
free=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
used=$(df -B1 --output=used / | tail -1 | tr -d ' ')
: >"$scratch/rows.jsonl"
while IFS= read -r item; do
  path=$(jq -r .path <<<"$item")
  group=$(jq -r .group <<<"$item")
  privileged=$(jq -r '.privileged // false' <<<"$item")
  if [[ "$privileged" == true ]]; then
    if measurement=$(sudo -n /usr/bin/du -sx -B1 -- "$path" 2>"$scratch/du.stderr"); then
      status=0
    else
      status=$?
    fi
  else
    if measurement=$(/usr/bin/du -sx -B1 -- "$path" 2>"$scratch/du.stderr"); then
      status=0
    else
      status=$?
    fi
  fi
  if [[ -n "$measurement" && "$status" == 0 ]]; then
    bytes=${measurement%%$'\t'*}
    jq -cn --arg path "$path" --arg group "$group" --argjson bytes "$bytes" \
      '{path:$path,group:$group,bytes:$bytes}' >>"$scratch/rows.jsonl"
  elif [[ -n "$measurement" && "$status" == 1 && -s "$scratch/du.stderr" ]] &&
       ! grep -qv 'No such file or directory$' "$scratch/du.stderr"; then
    bytes=${measurement%%$'\t'*}
    jq -cn --arg path "$path" --arg group "$group" --argjson bytes "$bytes" \
      '{path:$path,group:$group,bytes:$bytes,unstable_paths:true}' >>"$scratch/rows.jsonl"
  else
    error=$(head -c 500 "$scratch/du.stderr")
    jq -cn --arg path "$path" --arg group "$group" --arg error "$error" \
      '{path:$path,group:$group,error:$error}' >>"$scratch/rows.jsonl"
  fi
done < <(jq -c '.measurement_roots[]' "$config")
jq -s . "$scratch/rows.jsonl" >"$scratch/rows.json"
for unit in jope-build-sweep.service scq-runs-gc.service; do
  result=$(systemctl --user show "$unit" -p Result --value 2>&1) || result="query failed: $result"
  jq -cn --arg unit "$unit" --arg result "$result" '{unit:$unit,result:$result}'
done | jq -s . >"$scratch/services.json"
previous='null'
if [[ -s "$custody/latest.json" ]]; then
  previous=$(jq -c '{at,root_free_bytes}' "$custody/latest.json")
fi
jq -n --argjson at "$now" --argjson free "$free" --argjson used "$used" \
  --argjson previous "$previous" \
  --slurpfile rows "$scratch/rows.json" --slurpfile services "$scratch/services.json" '
  {schema:"jope.disk-watch/v1",at:$at,root_free_bytes:$free,root_used_bytes:$used,
   measurements:$rows[0],services:$services[0],
   growth_bytes_per_day:(if $previous != null and $at > $previous.at
     then (($previous.root_free_bytes - $free) * 86400 / ($at - $previous.at) | floor)
     else null end)} |
  .alerts = ([if .root_free_bytes < 1000000000000 then "root_free_below_1tb" else empty end,
              if .growth_bytes_per_day != null and .growth_bytes_per_day > 100000000000
                then "root_growth_above_100gb_per_day" else empty end,
              if any(.measurements[]; has("error")) then "measurement_failed" else empty end,
              if any(.services[]; .result != "success") then "cleanup_service_failed" else empty end])
  ' >"$scratch/report.json"
install -m 0600 "$scratch/report.json" "$custody/$stamp.json"
install -m 0600 "$scratch/report.json" "$custody/latest.json"
if jq -e '.alerts | length > 0' "$scratch/report.json" >/dev/null; then
  jq -c '{at,alerts,root_free_bytes,growth_bytes_per_day}' "$scratch/report.json" | systemd-cat -t jope-disk-watch -p warning
fi
cat "$scratch/report.json"
