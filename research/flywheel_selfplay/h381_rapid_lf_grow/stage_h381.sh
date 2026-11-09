#!/bin/bash
# Stage an H381 GP code root on xbabe2 and copy it to xbabe0 (create-once): the frozen H335 files (each byte-identical
# to origin/v1/next) + gp_h381.py + run_gp_h381_xbabe0.sh, read-only, with CODE.sha256 next to code/.
#   stage_h381.sh TAG   ->  /home/ubuntu/h381-stage/gpc-TAG on xbabe2 and xbabe0
set -euo pipefail
SRC=$(dirname "$(readlink -f "$0")")
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
GITDIR=research/flywheel_selfplay/h335_aggressive_locked_elo
DST=/home/ubuntu/h381-stage/gpc-$TAG
SSH=(ssh -o BatchMode=yes xbabe0)
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
mkdir -p "$DST/code"
(cd "$H335" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##') > "$DST/h335.files"
while read -r f; do
  cmp -s "$H335/$f" <(git -C "$REPO" show "origin/v1/next:$GITDIR/$f") \
    || { echo "H335 file $f differs from origin/v1/next" >&2; exit 4; }
done < "$DST/h335.files"
rsync -a --chmod=Fa-w --files-from="$DST/h335.files" "$H335/" "$DST/code/"
for f in gp_h381.py run_gp_h381_xbabe0.sh; do install -m 0555 "$SRC/$f" "$DST/code/$f"; done
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/h335.files"
"${SSH[@]}" "mkdir -p /home/ubuntu/h381-stage"
rsync -a "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0); CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
