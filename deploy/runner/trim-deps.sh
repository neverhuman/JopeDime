#!/usr/bin/env bash
# deploy/runner/trim-deps.sh <target-root> <keep> [--unused-days=<days>] [--apply] — bounded trim
# of one lane's persistent Cargo target volume (run by jope_runner_trim_lane in common.sh, inside a
# network-less container that mounts only that volume). Candidates are the files directly in
# <target-root>/<lane>/<profile>/deps (debug, release, dist-ci or any other profile; never tmp,
# build or incremental) whose names are <stem>-<16 lowercase hex>[.<ext>]: Cargo's per-build
# outputs. Within each (directory, stem, extension) the newest <keep> by mtime stay, so every
# feature/profile variant still has its recent builds. With --unused-days, an executable (a name
# without extension: test, bench and binary outputs, ~75 MB each) is also selected when its last
# use, the later of its mtime (last link) and atime (last execution, at relatime granularity), is
# older than that many days and it has a single link (Cargo hard-links current binaries into the
# profile directory). A current binary removed this way is only rebuilt. Nothing else is a
# candidate: other names, other directories (build/, incremental/, tmp/), deeper paths and any
# directory whose name contains "evidence" (such as ci-evidence-retained/) are never touched.
# Prints "path<TAB>bytes<TAB>mtime" for every selected file; with --apply it deletes exactly
# those, and only after the whole listing and selection succeeded.
set -euo pipefail
usage='usage: trim-deps.sh <target-root> <keep> [--unused-days=<days>] [--apply]'
root=${1:?$usage}
keep=${2:?$usage}
shift 2
mode='' unused_days=0
for arg in "$@"; do
  case "$arg" in
    --apply) mode=--apply ;;
    --unused-days=*) unused_days=${arg#--unused-days=}
      [[ $unused_days =~ ^[1-9][0-9]*$ ]] || { echo "trim-deps: unused days must be a positive integer" >&2; exit 2; } ;;
    *) echo "trim-deps: unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[[ $keep =~ ^[1-9][0-9]*$ ]] || { echo "trim-deps: keep must be a positive integer" >&2; exit 2; }
[[ -d $root && ! -L $root ]] || { echo "trim-deps: not a directory: $root" >&2; exit 2; }
root=${root%/}
# 0 disables the unused rule: no mtime or atime is older than the epoch.
cutoff=0
(( unused_days == 0 )) || cutoff=$(( $(date +%s) - unused_days * 86400 ))

# mawk has no interval expressions, so the 16-hex-digit hash is spelled out.
hash=''
for _ in {1..16}; do hash+='[0-9a-f]'; done

# A find error (an unreadable directory, a vanished volume) fails the pipeline under pipefail,
# so the assignment fails and nothing is deleted.
selected=$(
  find "$root" -mindepth 4 -maxdepth 4 -type f -path "$root/*/*/deps/*" -printf '%T@\t%A@\t%n\t%s\t%p\n' |
  awk -F'\t' -v root="$root/" -v hash="$hash" '
    NF == 5 && index($5, root) == 1 {
      rest = substr($5, length(root) + 1)
      n = split(rest, part, "/")
      if (n != 4 || part[3] != "deps") next
      if (part[1] !~ /^[A-Za-z0-9_.+-]+$/ || part[2] !~ /^[A-Za-z0-9_.+-]+$/ || part[4] !~ /^[A-Za-z0-9_.+-]+$/) next
      if (part[1] ~ /evidence/ || part[2] ~ /evidence/) next
      if (part[2] == "tmp" || part[2] == "build" || part[2] == "incremental") next
      name = part[4]
      if (!match(name, "-" hash "(\\.[A-Za-z0-9_]+)?$")) next
      stem = substr(name, 1, RSTART - 1)
      ext = substr(name, RSTART + 17)
      if (stem == "") next
      printf "%s/%s|%s|%s\t%s\t%s\t%s\t%s\t%s\t%s\n", part[1], part[2], stem, ext, $1, $2, $3, $4, $5, (ext == "")
    }' |
  LC_ALL=C sort -t "$(printf '\t')" -k1,1 -k2,2gr |
  awk -F'\t' -v keep="$keep" -v cutoff="$cutoff" '{
    used = ($3 + 0 > $2 + 0) ? $3 + 0 : $2 + 0
    if (++count[$1] > keep || ($7 == 1 && $4 == 1 && used < cutoff)) printf "%s\t%s\t%s\n", $6, $5, $2
  }'
)
[[ -n $selected ]] || exit 0
printf '%s\n' "$selected"
[[ $mode == --apply ]] || exit 0
while IFS=$'\t' read -r path _ _; do
  rm -f -- "$path"
done <<< "$selected"
