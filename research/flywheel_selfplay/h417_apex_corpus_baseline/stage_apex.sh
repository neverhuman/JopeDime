#!/bin/bash
# Stage an H417 code root on xbabe2 and copy it to xbabe0 (create-once, read-only, CODE.sha256 next to code/):
#   * the published H416 code root ~/h416-stage/pub-pub3/code, verified against its own CODE.sha256 (the frozen H335
#     files and gp_h381 / deploy_h410 / *_h415 / *_h416 / meta_h416 / DESIGN.json at their origin/v1/next bytes);
#   * the H417 files. PREREG.md is staged as PREREG_H417.md.
#   stage_apex.sh pub TAG   H417 files = their published bytes on origin/v1/next (every scored run)
#   stage_apex.sh dev TAG   H417 files from this directory (smoke and identity check only; never scored)
#   -> /home/ubuntu/apex-stage/h417/<mode>-TAG on xbabe2 and xbabe0
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/h416-stage/pub-pub3
G417=research/flywheel_selfplay/h417_apex_corpus_baseline
FILES=(apex_core.py apex_core_meta.py gp_apex.py train_apex.py run_apex_xbabe0.sh run_apex.sh stage_apex.sh cohorts_h417.sh g0_apex.py)
DST=/home/ubuntu/apex-stage/h417/$MODE-$TAG
SSH=(ssh -o BatchMode=yes xbabe0)
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
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
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H417.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G417/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  elif [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/code/$dst"
  else
    [ "$f" = PREREG.md ] || { echo "missing $SRC/$f" >&2; exit 5; }
    continue
  fi
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode"
"${SSH[@]}" "mkdir -p /home/ubuntu/apex-stage/h417"
rsync -a "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0) mode $MODE; CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
