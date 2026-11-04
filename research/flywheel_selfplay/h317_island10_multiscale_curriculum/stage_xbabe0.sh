#!/bin/bash
# H317 GP: stage the TRAIN-only inputs + code + JopeDimeMicro subset from xbabe2 to xbabe0, with a sha manifest.
# Run on xbabe2:  ./stage_xbabe0.sh [manifest_tag]
#   1. stage_check_h317.py inside sandbox_h317.sh (bwrap, TRAIN-only view) builds the manifest and asserts that no staged
#      path / text file / npz cid list names a DL-Val context, dataset or family id (redacted selected.json dev rows),
#      that tables/pools are exactly the 1,219 TRAIN cids, worker sha e3be9386..., views-search = one SEARCH view set per
#      pilot context. Nothing is copied if it fails.
#   2. rsync --files-from each group (exact files only, no directory walk) to xbabe0, read-only modes.
#   3. sha256sum -c of every group on xbabe0; manifest copied to both hosts.
# xbabe0 layout (sandbox_h317_docker.sh mounts these; DL-Val inputs for Stage V live in a SEPARATE tree,
# $STAGE_V/dlval-view, created only after the pinned LOCK unlock and never mounted in Gate H0 runs):
#   $STAGE/code               -> ro at the H317 source path
#   $STAGE/res01-train-view   -> ro at /home/ubuntu/jope-data/res-01/input
#   $STAGE/micro              -> ro at /home/ubuntu/JopeDimeMicro
#   $DATA/neural/views_c32_train1219.npz, $DATA/views-search/, $DATA/neural/runs/H0.1-A/   -> ro over the rw $DATA
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
DATA=/home/ubuntu/jope-data/h317-cl
STAGE=/home/ubuntu/h317-stage
HOST=${H317_STAGE_HOST:-xbabe0}
TAG=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
MAN=$DATA/gp/staging/manifest_$TAG
mkdir -p "$DATA/gp/staging"
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
cd "$SRC"
nice -n 19 ./sandbox_h317.sh python3 stage_check_h317.py "$MAN"
ssh -o BatchMode=yes "$HOST" "mkdir -p $STAGE/code $STAGE/res01-train-view $STAGE/micro $DATA/neural $DATA/views-search $DATA/gp/staging"
sync_group() {  # group source_root target_root
  rsync -a --chmod=Fa-w --files-from="$MAN.$1.files" "$2/" "$HOST:$3/"
  scp -q "$MAN.$1.sha256" "$HOST:$DATA/gp/staging/"
  ssh -o BatchMode=yes "$HOST" "cd $3 && sha256sum --quiet -c $DATA/gp/staging/$(basename "$MAN").$1.sha256" \
    && echo "group $1: sha256 OK on $HOST"
}
sync_group code "$SRC" "$STAGE/code"
sync_group train_view "$DATA/res01-train-view" "$STAGE/res01-train-view"
sync_group micro /home/ubuntu/JopeDimeMicro "$STAGE/micro"
sync_group views "$DATA" "$DATA"
sync_group screener "$DATA" "$DATA"
scp -q "$MAN.json" "$HOST:$DATA/gp/staging/"
echo "manifest: $MAN.json (copied to $HOST:$DATA/gp/staging/)"
