#!/usr/bin/env bash
# A relative or empty HOME must not create the clean-host fixture cache.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
script=$here/test-clean-host-prestage-controls.sh
wt=$(cd -- "$here/../.." && pwd)
scratch=$wt/target/fixture-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch"
root_cache=/.cache/jopedime-clean-host-controls
root_cache_existed=0
[[ -e $root_cache ]] && root_cache_existed=1

refuse() {
  local label=$1
  shift
  local status=0
  (cd -- "$scratch" && env "$@" bash "$script") >"$scratch/out" 2>"$scratch/err" || status=$?
  [[ $status == 2 ]]
  grep -q 'HOME must be an absolute path' "$scratch/err"
  [[ ! -e $scratch/relative ]]
  [[ ! -e $scratch/relative-xdg ]]
  printf 'PASS %s\n' "$label"
}

refuse relative-home HOME=relative/home XDG_CACHE_HOME=
refuse empty-home HOME= XDG_CACHE_HOME=
refuse relative-xdg-without-absolute-home HOME=relative/home XDG_CACHE_HOME=relative-xdg
if [[ $root_cache_existed == 0 ]]; then
  [[ ! -e $root_cache ]]
fi

home=$scratch/abs-home
mkdir -p -- "$home"
(cd -- "$scratch" && HOME=$home XDG_CACHE_HOME=relative-xdg bash "$script")
[[ ! -e $scratch/relative-xdg ]]
[[ -d $home/.cache/jopedime-clean-host-controls ]]
printf 'PASS absolute-home-ignores-relative-xdg\n'
