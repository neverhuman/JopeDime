#!/bin/bash
# H335: stage code, superset worker, screener-model links and P201b SEARCH views from xbabe2 to xbabe0 (sha-checked).
# Run on xbabe2. Never while an h335-* unit is active on xbabe0 (shards import the staged code).
#   code    -> xbabe0:/home/ubuntu/h335-stage/code (read-only modes), sha256 of every file checked on arrival
#   bin     -> xbabe0:$DATA/bin/bootstrap.s1main.superset-2a848c8f8085
#   neural  -> xbabe0:$DATA/neural = hard links of xbabe0's staged H317 screener inputs (views_c32_train1219.npz,
#              runs/H0.1-A/**), each file's sha256 checked against xbabe2's h335-cl/neural manifest
#   views   -> xbabe0:$DATA/views-search/<cid>.npz for the P201b cids (TRAIN contexts only; list checked)
# The frozen H317 code and TRAIN view already staged on xbabe0 (/home/ubuntu/h317-stage) are verified, not re-copied.
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
DATA=/home/ubuntu/jope-data/h335-cl
H317DATA=/home/ubuntu/jope-data/h317-cl
STAGE=/home/ubuntu/h335-stage
HOST=${H335_STAGE_HOST:-xbabe0}
TAG=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
MAN=$DATA/staging/manifest_$TAG
mkdir -p "$DATA/staging"
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
if ssh -o BatchMode=yes "$HOST" "systemctl --user list-units --state=active --no-legend 'h335-*' | grep -q ."; then
  echo "refused: an h335-* unit is active on $HOST" >&2; exit 4
fi
# 1. frozen H317 code on xbabe0 == the pinned modules
grep -oE '"[a-z0-9_]+\.py": "[0-9a-f]{64}"' "$SRC/h335_core.py" | sed -E 's/"([^"]+)": "([0-9a-f]+)"/\2  \1/' \
  > "$MAN.h317pins.sha256"
[ "$(wc -l < "$MAN.h317pins.sha256")" = 18 ] || { echo "pin list != 18" >&2; exit 5; }
ssh -o BatchMode=yes "$HOST" "mkdir -p $DATA/staging"; scp -q "$MAN.h317pins.sha256" "$HOST:$DATA/staging/h335_pins_$TAG.sha256"
ssh -o BatchMode=yes "$HOST" "cd /home/ubuntu/h317-stage/code && sha256sum --quiet -c $DATA/staging/h335_pins_$TAG.sha256" \
  || { echo "STAGING CHECK FAILED" >&2; exit 6; }; echo "frozen H317 code on $HOST: pins OK"
# 2. code
(cd "$SRC" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.rs' \) \
   ! -name '.*' | sort | sed 's#^\./##' > "$MAN.code.files" && xargs -a "$MAN.code.files" sha256sum > "$MAN.code.sha256")
ssh -o BatchMode=yes "$HOST" "mkdir -p $STAGE/code $DATA/bin $DATA/neural/runs $DATA/views-search $DATA/staging"
rsync -a --chmod=Fa-w --files-from="$MAN.code.files" "$SRC/" "$HOST:$STAGE/code/"
scp -q "$MAN.code.sha256" "$HOST:$DATA/staging/"
ssh -o BatchMode=yes "$HOST" "cd $STAGE/code && sha256sum --quiet -c $DATA/staging/$(basename "$MAN").code.sha256" \
  || { echo "STAGING CHECK FAILED" >&2; exit 6; }; echo "code: sha256 OK on $HOST"
# 3. superset worker
rsync -a --chmod=Fa-w "$DATA/bin/bootstrap.s1main.superset-2a848c8f8085" "$HOST:$DATA/bin/"
ssh -o BatchMode=yes "$HOST" "sha256sum $DATA/bin/bootstrap.s1main.superset-2a848c8f8085" | grep -q '^2a848c8f8085a392' \
  || { echo "STAGING CHECK FAILED" >&2; exit 6; }; echo "superset worker: sha256 OK on $HOST"
# 4. screener inputs (copies of xbabe2's h335-cl/neural; --ignore-existing never writes through files already present,
#    e.g. hard links to xbabe0's staged H317 copies), every file sha256-checked against xbabe2's manifest
scp -q "$DATA/neural_manifest.sha256" "$HOST:$DATA/staging/neural_manifest.sha256"
rsync -a --ignore-existing --chmod=Fa-w "$DATA/neural/" "$HOST:$DATA/neural/"
ssh -o BatchMode=yes "$HOST" "cd $DATA/neural && sha256sum --quiet -c $DATA/staging/neural_manifest.sha256" \
  || { echo "STAGING CHECK FAILED" >&2; exit 6; }; echo "neural: sha256 OK on $HOST"
# 5. P201b SEARCH views (TRAIN P201b cids only)
(cd "$SRC" && ./sandbox_h335.sh env H335_GROUP=BASE python3 -c \
  'import h335_core as K; print("\n".join(K.panel_p201b(K.C.train_rows())))') > "$MAN.p201b.cids"
[ "$(wc -l < "$MAN.p201b.cids")" = 201 ] || { echo "P201b list != 201" >&2; exit 5; }
sed 's/$/.npz/' "$MAN.p201b.cids" > "$MAN.views.files"
(cd "$DATA/views-search" && xargs -a "$MAN.views.files" sha256sum > "$MAN.views.sha256")
rsync -a --chmod=Fa-w --files-from="$MAN.views.files" "$DATA/views-search/" "$HOST:$DATA/views-search/"
scp -q "$MAN.views.sha256" "$HOST:$DATA/staging/"
ssh -o BatchMode=yes "$HOST" "cd $DATA/views-search && sha256sum --quiet -c $DATA/staging/$(basename "$MAN").views.sha256 \
  && [ \$(ls $DATA/views-search | grep -c npz) = 201 ]" || { echo "STAGING CHECK FAILED" >&2; exit 6; }; echo "views: 201 P201b SEARCH view sets, sha256 OK on $HOST"
echo "manifest prefix: $MAN"
