#!/bin/bash
# Stage the H357 GP code root (H357 PREREG §3): the frozen H335 files + gp_c_h357.py + run_gp_c_xbabe0.sh, read-only,
# create-once.
#   stage_gpc_h357.sh TAG  ->  /home/ubuntu/h357-stage/gpc-TAG/code  (+ CODE.sha256 next to code/)
# Every H335 file must be byte-identical both to the H335 staged copy that ran Phase T (/home/ubuntu/h335-stage/code)
# and to research/flywheel_selfplay/h335_aggressive_locked_elo/ on origin/v1/next.
# Use: xbabe2  H335_CODE=<stage>/code sandbox_h335.sh ...;  xbabe0  H335_STAGE=<stage> sandbox_h335_docker.sh ...
set -euo pipefail
SRC=$(dirname "$(readlink -f "$0")")
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
GITDIR=research/flywheel_selfplay/h335_aggressive_locked_elo
DST=/home/ubuntu/h357-stage/gpc-$TAG
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
mkdir -p "$DST/code"
(cd "$H335" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##') > "$DST/h335.files"
while read -r f; do
  cmp -s "$H335/$f" <(git -C "$REPO" show "origin/v1/next:$GITDIR/$f") \
    || { echo "H335 file $f differs from origin/v1/next" >&2; exit 4; }
done < "$DST/h335.files"
rsync -a --chmod=Fa-w --files-from="$DST/h335.files" "$H335/" "$DST/code/"
for f in gp_c_h357.py run_gp_c_xbabe0.sh run_auditx_xbabe0.sh run_score_xbabe0.sh; do install -m 0555 "$SRC/$f" "$DST/code/$f"; done
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/h335.files"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST/code; CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
