#!/usr/bin/env bash
# A relative HOME or XDG_CACHE_HOME must not name the paper font cache.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)
scratch=$root/target/font-home-proof
rm -rf -- "$scratch"
mkdir -p -- "$scratch"
ez=$here/eurekazero/scripts/build.sh
iz=$here/innovationzero/scripts/build.sh
root_cache=/.cache/innovationzero-fonts
existed=0
[[ -e $root_cache ]] && existed=1
mtime=
[[ $existed == 1 ]] && mtime=$(stat -c %Y -- "$root_cache")

unchanged() {
  [[ ! -e $scratch/relative ]]
  [[ ! -e $scratch/.cache ]]
  if [[ $existed == 0 ]]; then
    [[ ! -e $root_cache ]]
  else
    [[ $(stat -c %Y -- "$root_cache") == "$mtime" ]]
  fi
}

run() {
  local status=0
  (cd -- "$scratch" && env -u EZ_FONT_CACHE -u IZ_FONT_CACHE -u XDG_CACHE_HOME "$@") \
    >"$scratch/out" 2>"$scratch/err" || status=$?
  printf '%s\n' "$status"
}

for script in "$ez" "$iz"; do
  name=$(basename "$(dirname "$(dirname "$script")")")
  status=$(run HOME=relative/home bash "$script")
  [[ $status == 2 ]]
  grep -q 'HOME must be an absolute path' "$scratch/err"
  status=$(run HOME= bash "$script")
  [[ $status == 2 ]]
  grep -q 'HOME must be an absolute path' "$scratch/err"
  status=$(run HOME=relative/home bash "$script" --clean)
  [[ $status == 0 ]]
  unchanged
  printf 'PASS %s\n' "$name"
done

# A relative XDG_CACHE_HOME is ignored. Hide the TeX tools so the script stops
# before mkdir and before any font download.
abs=$scratch/abs-home
mkdir -p -- "$abs"
status=$(run HOME="$abs" XDG_CACHE_HOME=relative/xdg PATH=/usr/bin:/bin bash "$ez")
[[ $status == 2 ]]
grep -q 'xelatex is required' "$scratch/err"
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ ! -e $scratch/relative ]]
[[ ! -e $abs/.cache ]]
printf 'PASS relative-xdg-uses-absolute-home\n'

# The font-cache override stays verbatim and does not consult HOME.
status=$(run HOME=relative/home EZ_FONT_CACHE=relative/font PATH=/usr/bin:/bin bash "$ez")
[[ $status == 2 ]]
grep -q 'xelatex is required' "$scratch/err"
grep -q 'HOME must be an absolute path' "$scratch/err" && exit 1
[[ ! -e $scratch/relative ]]
printf 'PASS override-stays-verbatim\n'
