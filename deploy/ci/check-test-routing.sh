#!/usr/bin/env bash
# Bind feature/custody exclusions to the actual nextest collection. A listed
# contract must remain collected and executable in its owning lane.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?cpu, neural, or host}
collection=${2:?nextest collection JSON}
case $mode in
  cpu) list=deploy/ci/neural-cpu-only-tests.txt; ignored=false ;;
  neural) list=deploy/ci/neural-cpu-only-tests.txt; ignored=true ;;
  host) list=deploy/ci/real-corpus-tests.txt; ignored=true ;;
  *) echo "unknown routing mode: $mode" >&2; exit 2 ;;
esac
rows=$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$list" | jq -Rsc 'split("\n") | map(select(length > 0) | [scan("[^[:space:]]+")])')
jq -e --argjson rows "$rows" --argjson ignored "$ignored" --arg mode "$mode" '
  .["rust-suites"] as $suites |
  ($rows | length) > 0 and
  ($rows | length) == ($rows | unique | length) and
  all($rows[]; length == 2) and
  all($rows[]; . as $row |
    $suites[$row[0]].testcases[$row[1]] as $test |
    $test != null and $test.ignored == $ignored and
    ($mode != "cpu" or $test["filter-match"].status == "matches"))
' "$collection" >/dev/null || {
  echo "$mode test routing changed: missing, duplicated, or incorrectly ignored contract" >&2
  exit 1
}
printf '%s routing verified: %s contracts\n' "$mode" "$(jq length <<< "$rows")"
