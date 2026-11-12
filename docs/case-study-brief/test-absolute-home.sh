#!/usr/bin/env bash
# A relative or empty HOME must not name the case-study brief evidence roots.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
scratch=$root/target/brief-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch"
data=$here/build-data.sh
winners=$here/build-winners.sh
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
  (cd -- "$scratch" && env -u JOPE_CASE_OUT -u JOPE_CASE_BASELINES "$@") \
    >"$scratch/out" 2>"$scratch/err" || status=$?
  printf '%s\n' "$status"
}

for script in "$data" "$winners"; do
  name=$(basename "$script")
  status=$(run HOME=relative/home bash "$script")
  [[ $status == 2 ]]
  grep -q 'HOME must be an absolute path' "$scratch/err"
  unchanged
  status=$(run HOME= bash "$script")
  [[ $status == 2 ]]
  grep -q 'HOME must be an absolute path' "$scratch/err"
  unchanged
  printf 'PASS %s\n' "$name"
done

# Both overrides stay verbatim, so a relative HOME never reaches the helper.
# build-winners.sh then stops on the missing evidence, before it writes winners-data.
status=$(run HOME=relative/home JOPE_CASE_OUT=relative/out JOPE_CASE_BASELINES=relative/base bash "$winners")
[[ $status == 2 ]]
grep -q 'no cross-validated' "$scratch/err"
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ ! -e $scratch/relative/home ]]
printf 'PASS winners-overrides-stay-verbatim\n'
