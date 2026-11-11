#!/bin/bash
# Stage an H428 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2: the published H422 stage
# ~/apex-stage/h422/pub-r1 (frozen H335 ... H422 files, verified against its CODE.sha256) plus the H428 files.
# PREREG.md is staged as PREREG_H428.md, the addenda as PREREG_H428_ADDENDUM_<n>.md.
#   stage_h428.sh pub TAG   H428 files = their published bytes on origin/v1/next (the scored cache, cross-fit, fit and apply steps)
#   stage_h428.sh dev TAG   H428 files from this directory (development on design contexts only; never scored on DEV)
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h427/pub-r1
G428=research/flywheel_selfplay/h428_wide_table_bagging
FILES=(bag_h428.py readout_h428.py gates_h428.sh stage_h428.sh)
H419=(icl_model.py icl_resid.py icl_diag.py icl_inject.py)   # the in-context model code, at its published bytes
G419=research/flywheel_selfplay/h419_apex_icl
DOCS=(PREREG.md)
DST=/home/ubuntu/apex-stage/h428/$MODE-$TAG
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
  case $f in PREREG.md) dst=PREREG_H428.md ;; PREREG_ADDENDUM_*) dst=PREREG_H428_${f#PREREG_} ;; esac
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G428/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  elif [ -f "$SRC/$f" ]; then
    cp "$SRC/$f" "$DST/code/$dst"
  else
    continue
  fi
  chmod 0555 "$DST/code/$dst"
done
for f in "${H419[@]}"; do
  git -C "$REPO" show "origin/v1/next:$G419/$f" > "$DST/code/$f" || { echo "$f of H419 is not on origin/v1/next" >&2; exit 5; }
  cmp -s "$DST/code/$f" "/home/ubuntu/apex-stage/h419/r9/code/$f" || { echo "$f differs from the training stage r9" >&2; exit 6; }
  chmod 0555 "$DST/code/$f"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
