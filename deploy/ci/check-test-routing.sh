#!/usr/bin/env bash
# Bind feature/custody exclusions to the actual nextest collection, in both directions.
#
# Forward: a listed contract must remain collected with its lane's ignore status (cpu: not
# ignored and selected; neural and host: ignored).
# Reverse: a test whose ignore reason routes it to a list must be on that list, so a new
# neural-only or corpus-only test cannot be excluded silently; and a listed row whose test no
# longer carries a routing reason is stale. A collection records whether a test is ignored, not
# why, so the reasons are read from the source attributes under crates/:
#   neural-cpu-only-tests.txt (cpu, neural):
#     #[cfg_attr(feature = "neural", ignore = "cpu-fit-service: ...")]
#   real-corpus-tests.txt (host):
#     #[ignore = "real-corpus: ..."] and #[ignore = "deferred-classification: ..."]
# Any other reason (perf probe:, capture:, fixture runner:, slow:, needs ...) routes nowhere and
# is not checked here. A declaring function matches the collected testcases of its own crate
# (the suite's cwd ends in the crate directory) named <function> or ending in ::<function>.
# Limitation: the reverse match keys on the crate and the bare function name, not the module
# path or the test binary, so test functions of one crate that share a name are
# indistinguishable. If only one of them carries a routing reason, every namesake in that crate
# counts as routed: an unlisted namesake without the reason is reported "unlisted" (a false
# failure; rename one of the functions), and a listed row whose own routing reason was removed
# is not reported "stale" while a namesake keeps the reason. The forward check catches that row
# only if it is then no longer ignored (neural mode for the neural list, host mode for the corpus
# list); the cpu mode never does.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?cpu, neural, or host}
collection=${2:?nextest collection JSON}
case $mode in
  cpu) list=deploy/ci/neural-cpu-only-tests.txt; ignored=false; reasons='cpu-fit-service' ;;
  neural) list=deploy/ci/neural-cpu-only-tests.txt; ignored=true; reasons='cpu-fit-service' ;;
  host) list=deploy/ci/real-corpus-tests.txt; ignored=true; reasons='real-corpus|deferred-classification' ;;
  *) echo "unknown routing mode: $mode" >&2; exit 2 ;;
esac
# The collection may be a pipe (real-corpus-lane.sh passes a process substitution): read it once.
json=$(cat "$collection")
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
' <<< "$json" >/dev/null || {
  echo "$mode test routing changed: missing, duplicated, or incorrectly ignored contract" >&2
  exit 1
}

# Reverse. "<crate dir>\t<function>" for every function whose ignore attribute carries a routing
# reason of this list: the attribute line (single-line or inside a multi-line cfg_attr) sets
# the pending reason, the next `fn` line names the test.
declared=$(
  { grep -rlE --include='*.rs' "ignore[[:space:]]*=[[:space:]]*\"($reasons):" crates 2>/dev/null || true; } |
    LC_ALL=C sort |
    while IFS= read -r file; do
      awk -v reasons="$reasons" '
        FNR == 1 {
          dir = FILENAME
          if (match(dir, /\/(src|tests|benches|examples)\//)) dir = substr(dir, 1, RSTART - 1)
          attr = "^[[:space:]]*(#\\[ignore|#\\[cfg_attr\\(.*,[[:space:]]*ignore|ignore)[[:space:]]*=[[:space:]]*\"(" reasons "):"
          pending = 0
        }
        $0 ~ attr { pending = 1; next }
        pending && match($0, /^[[:space:]]*(pub(\([^)]*\))?[[:space:]]+)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
          name = substr($0, RSTART, RLENGTH)
          sub(/.*fn[[:space:]]+/, "", name)
          print dir "\t" name
          pending = 0
        }
      ' "$file"
    done
)
problems=$(jq -r --argjson rows "$rows" --arg declared "$declared" --arg list "$list" --arg reasons "$reasons" '
  ($declared | split("\n") | map(select(length > 0) | split("\t") | {dir: .[0], name: .[1]})) as $decl |
  [ .["rust-suites"] | to_entries[] | .key as $bin | (.value.cwd // "") as $cwd |
    [$decl[] | . as $d | select($cwd | endswith("/" + $d.dir)) | $d.name] as $names |
    select($names | length > 0) |
    .value.testcases | keys[] | . as $t |
    select(any($names[]; . as $n | $t == $n or ($t | endswith("::" + $n)))) | [$bin, $t] ] as $routed |
  ($routed | map({key: (.[0] + " " + .[1]), value: true}) | from_entries) as $is_routed |
  ($rows | map({key: (.[0] + " " + .[1]), value: true}) | from_entries) as $is_listed |
  ($routed[] | select($is_listed[.[0] + " " + .[1]] | not) |
    "unlisted: \(.[0]) \(.[1]) is ignored for a \($reasons) reason in its source but is not in \($list)"),
  ($rows[] | select(length == 2) | select($is_routed[.[0] + " " + .[1]] | not) |
    "stale: \($list) lists \(.[0]) \(.[1]) but its source carries no \($reasons) ignore reason")
' <<< "$json")
if [ -n "$problems" ]; then
  while IFS= read -r problem; do echo "$mode test routing: $problem" >&2; done <<< "$problems"
  echo "$mode test routing: every test ignored for a $reasons reason must be listed, and every listed row must carry one" >&2
  exit 1
fi
printf '%s routing verified: %s contracts; %s declared in source, none unlisted\n' \
  "$mode" "$(jq length <<< "$rows")" "$(grep -c . <<< "$declared" || true)"
