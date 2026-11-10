#!/bin/bash
# Stage an H420 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2 and copy it to the GP hosts:
#   the published H417 stage ~/apex-stage/h417/pub-p1 (frozen H335 / H381 / H410 / H415 / H416 / H417 files, verified
#   against its CODE.sha256) plus the H420 files. PREREG.md is staged as PREREG_H420.md.
#   stage_h420.sh pub TAG [host...]   H420 files = their published bytes on origin/v1/next (every scored run)
#   stage_h420.sh dev TAG [host...]   H420 files from this directory (probe and smoke only; never scored)
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}; shift 2
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h417/pub-p1
G420=research/flywheel_selfplay/h420_lang1_grammar_exposure
FILES=(lang_h420.py gp_h420.py train_h420.py readout_h420.py run_h420.sh run_h420_local.sh h420_chain.sh stage_h420.sh PREREG_ADDENDUM_1.md)
DST=/home/ubuntu/apex-stage/h420/$MODE-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"
sha256sum "$BASE/CODE.sha256" | cut -d' ' -f1 > "$DST/base_CODE.sha256"
if [ "$MODE" = pub ]; then
  git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
    fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
fi
for f in "${FILES[@]}" PREREG.md; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H420.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G420/$f" > "$DST/code/$dst" \
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
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode"
chmod -R a-w "$DST/code"
for h in "$@"; do
  ssh -o BatchMode=yes "$h" "[ ! -e $DST ] && mkdir -p $(dirname "$DST")"
  rsync -a "$DST" "$h:$(dirname "$DST")/"
  ssh -o BatchMode=yes "$h" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"
done
echo "staged $DST ($MODE) CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -d' ' -f1) $(wc -l < "$DST/CODE.sha256") files hosts: xbabe2 $*"
