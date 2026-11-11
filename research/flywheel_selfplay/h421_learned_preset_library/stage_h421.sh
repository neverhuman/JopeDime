#!/bin/bash
# Stage an H421 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2: the published H420 stage
# ~/apex-stage/h420/pub-p4 (frozen H335 ... H420 files, verified against its CODE.sha256) plus the H421 files.
# PREREG.md is staged as PREREG_H421.md, the addenda as PREREG_H421_ADDENDUM_<n>.md.
#   stage_h421.sh pub TAG   H421 files = their published bytes on origin/v1/next (the fill, the selection, DEV deploys)
#   stage_h421.sh dev TAG   H421 files from this directory (smoke and timing sample only; never scored)
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h420/pub-p4
G421=research/flywheel_selfplay/h421_learned_preset_library
FILES=(harvest_h421.py fill_h421.py select_h421.py train_h421.py h421_shards.sh stage_h421.sh)
DOCS=(PREREG.md PREREG_ADDENDUM_1.md)
DST=/home/ubuntu/apex-stage/h421/$MODE-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"; chmod u+w "$DST/code"   # the base code dir is read-only
sha256sum "$BASE/CODE.sha256" | cut -d' ' -f1 > "$DST/base_CODE.sha256"
if [ "$MODE" = pub ]; then
  git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
    fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
fi
for f in "${FILES[@]}" "${DOCS[@]}"; do
  dst=$f
  case $f in PREREG.md) dst=PREREG_H421.md ;; PREREG_ADDENDUM_*) dst=PREREG_H421_${f#PREREG_} ;; esac
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G421/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  elif [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/code/$dst"
  else
    continue
  fi
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
