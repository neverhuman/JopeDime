#!/usr/bin/env bash
# deploy/appliance/test-clean-host.sh — run a packaged `jopedime` as the only file in an empty
# directory with an empty environment (SPECIFICATION.md §14.2), or the Linux REL-04 pre-stage.
#
#   test-clean-host.sh PACKAGE WORKDIR [--require-isolation]
#   test-clean-host.sh JOPEDIME WORKDIR --prestage [PRESTAGE_OPTIONS...]
#
# Every command class runs: version, doctor, methods, fit (bounded selection), predict,
# fit-predict, verify-self, licenses, and the documented exit statuses. When `unshare` can drop
# the network, fitting runs without one; when `strace` exists, the run is traced and must start
# no other program and open no network socket. --require-isolation makes both mandatory (for
# trusted qualification hosts); hosted runners may lack them, and the receipt says which ran.
set -euo pipefail

package=${1:?usage: $0 PACKAGE WORKDIR [--require-isolation]}
work=${2:?usage: $0 PACKAGE WORKDIR [--require-isolation]}
require=${3:-}
if [[ $require == --prestage ]]; then
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  shift 3
  exec bash "$script_dir/test-clean-host-prestage.sh" "$package" "$work" "$@"
fi
case $work in /tmp | /tmp/*) echo "$0: never under /tmp (AGENTS.md)" >&2; exit 2 ;; esac
[ ! -e "$work" ] || { echo "$0: $work exists" >&2; exit 2; }
mkdir -p "$work"
work=$(cd "$work" && pwd)
cp -- "$package" "$work/jopedime"
chmod 0755 "$work/jopedime"

# Fixture tables, generated here so the directory holds nothing but the executable and data.
awk 'BEGIN {
  print "x1,x2,color,y"
  for (i = 0; i < 160; i++) {
    x1 = sin(i * 0.37) * 3; x2 = (i % 13) / 4; c = (i % 3 == 0) ? "red" : ((i % 3 == 1) ? "blue" : "NA")
    y = 2 * x1 - x2 * x2 + (c == "red" ? 1.5 : 0) + (i % 7) * 0.01
    printf "%.5f,%s,%s,%.5f\n", x1, (i % 11 == 0 ? "" : sprintf("%.3f", x2)), c, y
  }
}' > "$work/train.csv"
printf 'color,x2,x1\nred,1.0,0.5\nblue,,-1.5\npurple,2.0,\n' > "$work/rows.csv"

run() { (cd "$work" && env -i ./jopedime "$@"); }
expect() {
  local want=$1; shift
  local got=0
  run "$@" > /dev/null 2>&1 || got=$?
  [ "$got" = "$want" ] || { echo "jopedime $* exited $got, expected $want" >&2; exit 1; }
}

# `env -i` clears PATH, so the harness tools are named by absolute path.
isolation=none
fit_prefix=()
unshare_bin=$(command -v unshare || true)
if [ -n "$unshare_bin" ] && "$unshare_bin" -rn true 2> /dev/null; then
  isolation=no-network-namespace
  fit_prefix=("$unshare_bin" -rn)
fi
trace=none
strace_bin=$(command -v strace || true)
[ -z "$strace_bin" ] || trace=strace
if [ "$require" = --require-isolation ] && { [ "$isolation" = none ] || [ "$trace" = none ]; }; then
  echo "$0: --require-isolation, but isolation=$isolation trace=$trace" >&2; exit 1
fi

echo "== info commands"
run --version
run --version --json > "$work/version.json"
run doctor --json > "$work/doctor.json"
jq -e '.ok == true and .package.status == "authenticated"' "$work/doctor.json" > /dev/null
run methods --json > "$work/methods.json"
jq -e '[.methods[] | select(.kind == "native" and .executable)] | length == 26' "$work/methods.json" > /dev/null

echo "== fit, predict, fit-predict (isolation: $isolation, trace: $trace)"
fit=(fit --train train.csv --target y --output model.jope --candidates ridge,histgbm,extra_trees,pls --receipt fit.json)
if [ "$trace" = strace ]; then
  (cd "$work" && env -i "${fit_prefix[@]}" "$strace_bin" -f -qq -e trace=execve,socket,connect -o strace.log ./jopedime "${fit[@]}")
  execs=$(grep -c 'execve(' "$work/strace.log" || true)
  [ "$execs" = 1 ] || { echo "the fit started another program:" >&2; grep 'execve(' "$work/strace.log" >&2; exit 1; }
  if grep -E 'socket\(AF_INET|socket\(AF_INET6|connect\(' "$work/strace.log"; then
    echo "the fit opened a network socket" >&2; exit 1
  fi
else
  (cd "$work" && env -i "${fit_prefix[@]}" ./jopedime "${fit[@]}")
fi
jq -e '.outcome == "succeeded" and .device.selected == "cpu"' "$work/fit.json" > /dev/null
method=$(jq -r .method.selected "$work/fit.json")
run predict --model model.jope --input rows.csv --output predictions.csv
run fit-predict --train train.csv --target y --input rows.csv --output again.csv --method "$method"
cmp "$work/predictions.csv" "$work/again.csv"
[ "$(wc -l < "$work/predictions.csv")" = 4 ]

echo "== self-verification and notices"
run verify-self --json > "$work/verify-self.json"
jq -e '.ok == true' "$work/verify-self.json" > /dev/null
run licenses > "$work/licenses.txt"
grep -q Apache "$work/licenses.txt"

echo "== exit statuses"
expect 2 fit --train missing.csv --target y --output m.jope
expect 4 fit --train train.csv --target y --output m.jope --device cuda:0
expect 5 fit --train train.csv --target y --output m.jope --time-limit-seconds 0
expect 7 fit --train train.csv --target y --output no/such/dir/m.jope --method ridge

jq -n --arg isolation "$isolation" --arg trace "$trace" --arg method "$method" \
  --arg package_sha256 "$(sha256sum < "$work/jopedime" | cut -d' ' -f1)" \
  '{schema: "jope.appliance-clean-host/v1", ok: true, package_sha256: $package_sha256,
    isolation: $isolation, trace: $trace, selected_method: $method}' | tee "$work/clean-host.json"
