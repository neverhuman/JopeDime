#!/bin/bash
# Stage an H435 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2: the published H434 stage
# ~/apex-stage/h434/pub-r1 (frozen H335 ... H434 files, verified against its CODE.sha256) plus the H435 files.
# PREREG.md is staged as PREREG_H435.md.
#   stage_h435.sh pub TAG   H435 files = their published bytes on origin/v1/next (the scored steps)
#   stage_h435.sh dev TAG   H435 files from this directory (code path on random numbers only; never run on stored columns)
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h434/pub-r1
G432=research/flywheel_selfplay/h435_weight_screen_prospective
FILES=(screen_h435.py gates_h435.sh stage_h435.sh)
DOCS=(PREREG.md)
DST=/home/ubuntu/apex-stage/h435/$MODE-$TAG
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
  case $f in PREREG.md) dst=PREREG_H435.md ;; esac
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G432/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  elif [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/code/$dst"
  else
    continue
  fi
  chmod 0555 "$DST/code/$dst"
done
# the stack step's helpers must be the ones H424 ran
cmp -s "$DST/code/stack_h424.py" /home/ubuntu/apex-stage/h424/pub-r2/code/stack_h424.py || { echo "stack_h424.py differs from the H424 stage" >&2; exit 7; }
cmp -s "$DST/code/cache_h422.py" /home/ubuntu/apex-stage/h424/pub-r2/code/cache_h422.py || { echo "cache_h422.py differs from the H424 stage" >&2; exit 7; }
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
