#!/usr/bin/env bash
# A relative HOME must not create the jope-test-tmp cache. Empty HOME still fails ${HOME:?}.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
bump=$here/test-bump.sh
release_test=$root/deploy/release/test-release.sh
release=$root/deploy/release/release.sh
scratch=$root/target/release-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch"
root_cache=/.cache/jope-test-tmp
evidence=/home/ubuntu/jope-evidence
root_cache_existed=0
evidence_existed=0
[[ -e $root_cache ]] && root_cache_existed=1
[[ -e $evidence ]] && evidence_existed=1
evidence_mtime=
[[ $evidence_existed == 1 ]] && evidence_mtime=$(stat -c %Y -- "$evidence")

# The same pin_tmpdir constraints as test-release.sh: main() starts with pin_tmpdir, and only
# builtins and keywords run before export TMPDIR.
static_pin_check() {
  local first pre line seg kind
  local -a segs words
  first=$(awk '/^main\(\) \{$/ { m = 1; next } m && NF && $1 !~ /^#/ { print; exit }' "$release")
  [[ $first =~ ^[[:space:]]*pin_tmpdir[[:space:]]*$ ]] || { printf 'main() starts with: %s\n' "$first"; return 1; }
  pre=$(awk '/^pin_tmpdir\(\) \{$/ { p = 1; next }
    p && /^[[:space:]]*export TMPDIR=/ { found = 1; exit }
    p && /^\}$/ { exit }
    p { print }
    END { if (!found) print "NO-EXPORT" }' "$release")
  [[ $pre != *NO-EXPORT* ]] || { printf 'pin_tmpdir does not export TMPDIR\n'; return 1; }
  while IFS= read -r line; do
    [[ ! $line =~ ^[[:space:]]*(#|$) ]] || continue
    if [[ $line == *'$('* || $line == *'`'* || $line == *'<('* || $line == *'>('* || $line == *'|'* || $line == *'<<'* ]]; then
      printf 'pin_tmpdir runs a command before it exports TMPDIR: %s\n' "$line"
      return 1
    fi
    IFS=';' read -ra segs <<<"$line"
    for seg in "${segs[@]}"; do
      read -ra words <<<"$seg"
      while (( ${#words[@]} )) && [[ ${words[0]} =~ ^(if|then|else|elif|fi|while|until|do|done|!)$ ]]; do
        words=("${words[@]:1}")
      done
      (( ${#words[@]} )) || continue
      [[ ! ${words[0]} =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || continue
      kind=$(type -t -- "${words[0]}" || true)
      [[ $kind == builtin || $kind == keyword ]] \
        || { printf 'pin_tmpdir runs %s (%s) before it exports TMPDIR\n' "${words[0]}" "${kind:-not found}"; return 1; }
    done
  done <<<"$pre"
}

static_pin_check
printf 'PASS static-pin-tmpdir\n'

unchanged_caches() {
  [[ ! -e $scratch/relative ]]
  if [[ $root_cache_existed == 0 ]]; then
    [[ ! -e $root_cache ]]
  fi
  if [[ $evidence_existed == 0 ]]; then
    [[ ! -e $evidence ]]
  else
    [[ $(stat -c %Y -- "$evidence") == "$evidence_mtime" ]]
  fi
}

run_refused() {
  local label=$1 expect=$2
  shift 2
  local status=0
  (cd -- "$scratch" && env -u SHELLOPTS -u BASH_ENV "$@") >"$scratch/out" 2>"$scratch/err" || status=$?
  [[ $status == "$expect" ]]
  unchanged_caches
  printf 'PASS %s\n' "$label"
}

run_refused relative-bump 2 HOME=relative/home bash "$bump"
grep -q 'test-bump.sh: HOME must be an absolute path' "$scratch/err"
run_refused empty-bump 1 HOME= bash "$bump"
grep -q 'HOME must be set' "$scratch/err"

run_refused relative-release-test 2 HOME=relative/home bash "$release_test"
grep -q 'test-release.sh: HOME must be an absolute path' "$scratch/err"
run_refused empty-release-test 1 HOME= bash "$release_test"
grep -q 'HOME must be set' "$scratch/err"

# JOPE_RELEASE_TEST=1 keeps a missed guard on the scratch cache, not the evidence root.
run_refused relative-release 1 HOME=relative/home JOPE_RELEASE_TEST=1 bash -p "$release"
grep -q 'release.sh: REFUSED: HOME must be an absolute path' "$scratch/err"
run_refused empty-release 1 HOME= JOPE_RELEASE_TEST=1 bash -p "$release"
grep -q 'HOME must be set' "$scratch/err"

abs=$scratch/abs-home
mkdir -p -- "$abs"
env -u SHELLOPTS -u BASH_ENV HOME="$abs" JOPE_RELEASE_TEST=1 bash -p -c \
  'set -euo pipefail; source "$1"; pin_tmpdir; printf "%s\n" "$TMPDIR"' \
  release-unit "$release" >"$scratch/pin.out"
[[ $(<"$scratch/pin.out") == "$abs/.cache/jope-test-tmp/release-tmp" ]]
[[ -d $abs/.cache/jope-test-tmp/release-tmp ]]
[[ ! -e $scratch/relative ]]
printf 'PASS absolute-home-pins-test-tmp\n'
