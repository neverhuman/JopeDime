#!/usr/bin/env bash
# A relative or empty HOME must not name the fitness dataset directory.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
scratch=$root/target/fitness-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch"
fetch=$here/fetch.sh
bench=$here/benchmark.sh
root_data=/jope-data
existed=0
[[ -e $root_data ]] && existed=1
mtime=
[[ $existed == 1 ]] && mtime=$(stat -c %Y -- "$root_data")

unchanged() {
  [[ ! -e $scratch/relative ]]
  if [[ $existed == 0 ]]; then
    [[ ! -e $root_data ]]
  else
    [[ $(stat -c %Y -- "$root_data") == "$mtime" ]]
  fi
}

run() {
  local status=0
  (cd -- "$scratch" && env -u JOPE_FITNESS_DATA "$@") \
    >"$scratch/out" 2>"$scratch/err" || status=$?
  printf '%s\n' "$status"
}

status=$(run HOME=relative/home bash "$fetch")
[[ $status == 2 ]]
grep -q 'HOME must be an absolute path' "$scratch/err"
unchanged
printf 'PASS fetch-relative\n'

status=$(run HOME= bash "$fetch")
[[ $status == 2 ]]
grep -q 'HOME must be an absolute path' "$scratch/err"
unchanged
printf 'PASS fetch-empty\n'

status=$(run HOME=relative/home bash "$bench")
[[ $status == 2 ]]
grep -q 'HOME must be an absolute path' "$scratch/err"
[[ ! -e $root/target/fitness-real-world ]]
unchanged
printf 'PASS benchmark-relative\n'

status=$(run HOME= bash "$bench")
[[ $status == 2 ]]
grep -q 'HOME must be an absolute path' "$scratch/err"
unchanged
printf 'PASS benchmark-empty\n'

# The override stays verbatim. Inside the checkout, fetch.sh refuses before curl.
status=$(run HOME=relative/home JOPE_FITNESS_DATA=relative/override bash "$fetch")
[[ $status == 2 ]]
grep -q 'refusing to write datasets inside the checkout' "$scratch/err"
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ -d $scratch/relative/override ]]
[[ ! -e $scratch/relative/home ]]
printf 'PASS fetch-override-stays-verbatim\n'

status=$(run HOME=relative/home bash "$fetch" relative/arg)
[[ $status == 2 ]]
grep -q 'refusing to write datasets inside the checkout' "$scratch/err"
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ -d $scratch/relative/arg ]]
[[ ! -e $scratch/relative/home ]]
printf 'PASS fetch-arg-stays-verbatim\n'

status=$(run HOME=relative/home JOPE_FITNESS_DATA=relative/data JOPEDIME=/usr/bin/true bash "$bench")
[[ $status == 0 ]]
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ ! -e $scratch/relative/home ]]
[[ ! -e $scratch/relative/data ]]
printf 'PASS benchmark-override-stays-verbatim\n'
