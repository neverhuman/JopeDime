#!/usr/bin/env bash
# A relative or empty HOME must not name a case-study cache or output directory.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
helper=$here/absolute-home.sh
scratch=$root/target/case-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch/case"
root_cache=/.cache/jope
root_out=/jope-data
cache_existed=0
out_existed=0
[[ -e $root_cache ]] && cache_existed=1
[[ -e $root_out ]] && out_existed=1
cache_mtime=
out_mtime=
[[ $cache_existed == 1 ]] && cache_mtime=$(stat -c %Y -- "$root_cache")
[[ $out_existed == 1 ]] && out_mtime=$(stat -c %Y -- "$root_out")

unchanged() {
  [[ ! -e $scratch/relative ]]
  [[ ! -e $scratch/jope-data ]]
  [[ ! -e $scratch/.cache ]]
  if [[ $cache_existed == 0 ]]; then
    [[ ! -e $root_cache ]]
  else
    [[ $(stat -c %Y -- "$root_cache") == "$cache_mtime" ]]
  fi
  if [[ $out_existed == 0 ]]; then
    [[ ! -e $root_out ]]
  else
    [[ $(stat -c %Y -- "$root_out") == "$out_mtime" ]]
  fi
}

run() {
  local status=0
  (cd -- "$scratch" && env -u JOPE_CACHE -u JOPE_CASE_OUT -u JOPE_CASE_CATALOG \
    -u JOPE_CASE_BASELINES -u JOPE_CASE_TEST_REGISTRY "$@") \
    >"$scratch/out" 2>"$scratch/err" || status=$?
  printf '%s\n' "$status"
}

status=$(run HOME=/users/alice "$helper")
[[ $status == 0 ]]
[[ $(<"$scratch/out") == /users/alice ]]
printf 'PASS absolute-home\n'

status=$(run HOME=relative/home "$helper")
[[ $status == 2 ]]
grep -q 'case_study: HOME must be an absolute path' "$scratch/err"
status=$(run HOME= "$helper")
[[ $status == 2 ]]
grep -q 'HOME must be an absolute path' "$scratch/err"
status=$(run env -u HOME "$helper")
[[ $status == 2 ]]
unchanged
printf 'PASS helper-refuses-non-absolute-home\n'

refuse() {
  local label=$1
  shift
  local status=0
  status=$(run "$@")
  [[ $status == 2 ]]
  grep -q 'case_study: HOME must be an absolute path' "$scratch/err"
  unchanged
  printf 'PASS %s\n' "$label"
}

refuse collect-relative HOME=relative/home "$here/collect.sh"
refuse collect-empty HOME= "$here/collect.sh"
refuse run-case-relative HOME=relative/home "$here/run_case.sh" "$scratch/case"
refuse check-sealed-relative HOME=relative/home "$here/check_sealed.sh"
refuse final-test-relative HOME=relative/home "$here/final_test.sh"
refuse compare-relative HOME=relative/home "$here/compare.sh"
refuse report-relative HOME=relative/home "$here/report.sh"

got=$(JOPE_CASE_OUT=relative/out HOME=relative/home bash -c \
  'printf %s "${JOPE_CASE_OUT:-$("'"$helper"'")/jope-data/case_study}"')
[[ $got == relative/out ]]
unchanged
printf 'PASS override-stays-verbatim\n'

while IFS= read -r file; do
  while IFS= read -r line; do
    [[ $line =~ ^[[:space:]]*# ]] && continue
    code=${line%%#*}
    if [[ $code == *'$HOME/.cache'* || $code == *'$HOME/jope-data'* ]]; then
      printf 'unguarded HOME join in %s: %s\n' "$file" "$line" >&2
      exit 1
    fi
  done <"$file"
done < <(find "$here" -maxdepth 1 -name '*.sh' ! -name 'test-absolute-home.sh' ! -name 'absolute-home.sh' -print)
printf 'PASS no-unguarded-home-joins\n'
