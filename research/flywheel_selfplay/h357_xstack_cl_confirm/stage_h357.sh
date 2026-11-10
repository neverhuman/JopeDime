#!/bin/bash
# Stage the H357 xbabe2 code root, read-only and create-once, outside ~/jope_plan/evidence/ (the H317 guard):
#   stage_h357.sh TAG  ->  /home/ubuntu/h357-stage/code-TAG
# = the H354 files h354_core.py, phase_d_h354.py, lib_oof_h354.py, sandbox_h354.sh, test_h354.py exactly as in the H354
#   closeout commit f7530131f0 on origin/v1/next, plus this directory's H357 files; every file listed in CODE.sha256.
set -euo pipefail
SRC=$(dirname "$(readlink -f "$0")")
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
H354_COMMIT=f7530131f07e71bf13348d4c5727fa8d6537d817
H354_DIR=research/flywheel_selfplay/h354_xstack_cross_paradigm
DST=/home/ubuntu/h357-stage/code-$TAG
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
git -C "$REPO" merge-base --is-ancestor "$H354_COMMIT" origin/v1/next || { echo "H354 commit not on origin/v1/next" >&2; exit 4; }
mkdir -p "$DST"
for f in h354_core.py phase_d_h354.py lib_oof_h354.py sandbox_h354.sh test_h354.py; do
  git -C "$REPO" show "$H354_COMMIT:$H354_DIR/$f" > "$DST/$f"
done
chmod 0555 "$DST/sandbox_h354.sh" "$DST/lib_oof_h354.py" "$DST/phase_d_h354.py"
chmod 0444 "$DST/h354_core.py" "$DST/test_h354.py"
(cd "$SRC" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' \) ! -name '.*' | sort \
  | sed 's#^\./##') > "$DST/.h357.files"
rsync -a --chmod=Fa-w --files-from="$DST/.h357.files" "$SRC/" "$DST/"
(cd "$DST" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##' > .code.files)
(cd "$DST" && xargs -a .code.files sha256sum > CODE.sha256 && chmod a-w CODE.sha256 .code.files .h357.files)
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST; CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
