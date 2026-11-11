#!/bin/bash
# Stage a H440 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2:
#   /home/ubuntu/apex-stage/h424/pub-r1 + published H440 files on origin/v1/next.
#   stage_h440.sh TAG
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h424/pub-r1
G=research/flywheel_selfplay/h440_guard_and_duan_matern_cols
FILES=(guard_matern_core.py cols_matern.py stack_matern.py test_guard_matern.py smoke_matern.py
       PROBED_P201B.txt stage_h440.sh driver_h440.sh gates_h440.sh PREREG.md)
DST=/home/ubuntu/apex-stage/h440/pub-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"
chmod u+w "$DST/code"
sha256sum "$BASE/CODE.sha256" | cut -d' ' -f1 > "$DST/base_CODE.sha256"
git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
for f in "${FILES[@]}"; do
  dst=$f
  [ "$f" = PREREG.md ] && dst=PREREG_H440.md
  [ ! -e "$DST/code/$dst" ] || { echo "$dst already exists in base stage" >&2; exit 5; }
  git -C "$REPO" show "origin/v1/next:$G/$f" > "$DST/code/$dst" || { echo "$f is not on origin/v1/next" >&2; exit 5; }
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo pub > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
