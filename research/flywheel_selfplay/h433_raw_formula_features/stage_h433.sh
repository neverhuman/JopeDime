#!/bin/bash
# Stage an H433 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2: the published H430 stage
# ~/apex-stage/h433-base = ~/apex-stage/h430/pub-r3 (frozen H335 ... H430 files, verified against its CODE.sha256) plus the H433 files at
# their published bytes on origin/v1/next. PREREG.md is staged as PREREG_H433.md.
#   stage_h433.sh TAG
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h430/pub-r3
G=research/flywheel_selfplay/h433_raw_formula_features
FILES=(formula_core.py cols_h433.py stack_h433.py census_h433.py test_formula_core.py
       stage_h433.sh driver_h433.sh PREREG.md)
DST=/home/ubuntu/apex-stage/h433/pub-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"; chmod u+w "$DST/code"
sha256sum "$BASE/CODE.sha256" | cut -d' ' -f1 > "$DST/base_CODE.sha256"
git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
for f in "${FILES[@]}"; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H433.md
  [ ! -e "$DST/code/$dst" ] || { echo "$dst already exists in the base stage" >&2; exit 5; }
  git -C "$REPO" show "origin/v1/next:$G/$f" > "$DST/code/$dst" || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo pub > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
