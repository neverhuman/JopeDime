#!/usr/bin/env bash
# deploy/runner/trim-deps.sh <target-root> <keep> [--apply] — bounded trim of one lane's persistent
# Cargo target volume (run by jope_runner_trim_lane in common.sh, inside a network-less container
# that mounts only that volume). Candidates are the files directly in <target-root>/*/debug/deps
# whose names are <stem>-<16 lowercase hex>[.<ext>]: Cargo's per-build outputs. Within each
# (directory, stem, extension) the newest <keep> by mtime stay, so every feature/profile variant
# still has its recent builds. Nothing else is a candidate: other names, other directories
# (build/, incremental/, tmp/), deeper paths and any top-level directory whose name contains
# "evidence" (such as ci-evidence-retained/) are never touched. Prints "path<TAB>bytes<TAB>mtime" for every selected file; with --apply it deletes
# exactly those, and only after the whole listing and selection succeeded.
set -euo pipefail
root=${1:?usage: trim-deps.sh <target-root> <keep> [--apply]}
keep=${2:?usage: trim-deps.sh <target-root> <keep> [--apply]}
mode=${3:-}
[[ $keep =~ ^[1-9][0-9]*$ ]] || { echo "trim-deps: keep must be a positive integer" >&2; exit 2; }
[[ -z $mode || $mode == --apply ]] || { echo "trim-deps: unknown argument: $mode" >&2; exit 2; }
[[ -d $root && ! -L $root ]] || { echo "trim-deps: not a directory: $root" >&2; exit 2; }
root=${root%/}

# mawk has no interval expressions, so the 16-hex-digit hash is spelled out.
hash=''
for _ in {1..16}; do hash+='[0-9a-f]'; done

# A find error (an unreadable directory, a vanished volume) fails the pipeline under pipefail,
# so the assignment fails and nothing is deleted.
selected=$(
  find "$root" -mindepth 4 -maxdepth 4 -type f -path "$root/*/debug/deps/*" -printf '%T@\t%s\t%p\n' |
  awk -F'\t' -v root="$root/" -v hash="$hash" '
    NF == 3 && index($3, root) == 1 {
      rest = substr($3, length(root) + 1)
      n = split(rest, part, "/")
      if (n != 4 || part[2] != "debug" || part[3] != "deps") next
      if (part[1] !~ /^[A-Za-z0-9_.+-]+$/ || part[4] !~ /^[A-Za-z0-9_.+-]+$/) next
      if (part[1] ~ /evidence/) next
      name = part[4]
      if (!match(name, "-" hash "(\\.[A-Za-z0-9_]+)?$")) next
      stem = substr(name, 1, RSTART - 1)
      ext = substr(name, RSTART + 17)
      if (stem == "") next
      printf "%s|%s|%s\t%s\t%s\t%s\n", part[1], stem, ext, $1, $2, $3
    }' |
  LC_ALL=C sort -t "$(printf '\t')" -k1,1 -k2,2gr |
  awk -F'\t' -v keep="$keep" '{ if (++count[$1] > keep) printf "%s\t%s\t%s\n", $4, $3, $2 }'
)
[[ -n $selected ]] || exit 0
printf '%s\n' "$selected"
[[ $mode == --apply ]] || exit 0
while IFS=$'\t' read -r path _ _; do
  rm -f -- "$path"
done <<< "$selected"
