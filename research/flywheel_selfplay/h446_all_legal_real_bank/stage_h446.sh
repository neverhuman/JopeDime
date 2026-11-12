#!/bin/bash
# Stage an H446 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2:
#   /home/ubuntu/apex-stage/h440/pub-r2 (672c99372a969dda) + H425 screen_h425/CONFIRM_META + H419 icl_inject + published H446 files.
#   stage_h446.sh TAG
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h440/pub-r2
SHA_BASE=672c99372a969dda
G=research/flywheel_selfplay/h446_all_legal_real_bank
FILES=(icl_bank_all_legal.py stage_h446.sh driver_h446.sh gates_h446.sh PREREG.md)
DST=/home/ubuntu/apex-stage/h446/pub-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(sha256sum "$BASE/CODE.sha256" | cut -c1-16)" = "$SHA_BASE" ] || { echo "$BASE CODE.sha256 != $SHA_BASE" >&2; exit 4; }

mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"
chmod u+w "$DST/code"
sha256sum "$BASE/CODE.sha256" | cut -c1-16 > "$DST/base_CODE.sha256"

git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next

git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/screen_h425.py" > "$DST/code/screen_h425.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/CONFIRM_META.json" > "$DST/code/CONFIRM_META.json"
chmod 0555 "$DST/code/screen_h425.py"
chmod 0444 "$DST/code/CONFIRM_META.json"

for f in "${FILES[@]}"; do
  dst=$f
  [ "$f" = PREREG.md ] && dst=PREREG_H446.md
  [ ! -e "$DST/code/$dst" ] || { echo "$dst already exists in base stage" >&2; exit 5; }
  git -C "$REPO" show "origin/v1/next:$G/$f" > "$DST/code/$dst" || { echo "$f is not on origin/v1/next" >&2; exit 5; }
  chmod 0555 "$DST/code/$dst"
done

git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo pub > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
