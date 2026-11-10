#!/bin/bash
# Stage an H418 code root on xbabe2 and copy it to xbabe0 (create-once), read-only, with CODE.sha256 next to code/:
#   * the frozen H335 files (each byte-identical to origin/v1/next);
#   * at their published origin/v1/next bytes: gp_h381.py, deploy_h410.py, the H415 files gp_h415.py, deploy_h415.py,
#     train_h415.py, and the H416 files gp_h416.py, deploy_h416.py, meta_h416.py, train_h416.py, DESIGN.json;
#   * the H418 files. The H418 PREREG is staged as PREREG_H418.md (code/PREREG.md is H335's).
#   stage_h418.sh pub TAG   H418 files = their published bytes on origin/v1/next (every step of the score)
#   stage_h418.sh dev TAG   H418 files from this directory's working copy (import self-test only; never scored)
#   -> /home/ubuntu/h418-stage/<mode>-TAG on xbabe2 and xbabe0
# No board file and no dev91 byte is staged: the board files are LOCKed inputs read by analyze_h418.py after the
# unlock, from run_h418.sh's export.
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
case $MODE in pub|dev) ;; *) echo "mode is pub or dev" >&2; exit 2 ;; esac
SRCDIR=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
G335=research/flywheel_selfplay/h335_aggressive_locked_elo
G381=research/flywheel_selfplay/h381_rapid_lf_grow
G410=research/flywheel_selfplay/h410_deploy_population
G415=research/flywheel_selfplay/h415_committee_kfold_bag
G416=research/flywheel_selfplay/h416_stability_stack
G418=research/flywheel_selfplay/h418_sxb_dev91
H418_FILES=(d91_h418.py sandbox_h418_d91.sh analyze_h418.py selftest_h418.py run_h418.sh stage_h418.sh PREREG.md)
DST=/home/ubuntu/h418-stage/$MODE-$TAG
SSH=(ssh -o BatchMode=yes xbabe0)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
[ -e "$DST" ] && { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
mkdir -p "$DST/code"
(cd "$H335" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##') > "$DST/h335.files"
while read -r f; do
  cmp -s "$H335/$f" <(git -C "$REPO" show "origin/v1/next:$G335/$f") \
    || { echo "H335 file $f differs from origin/v1/next" >&2; exit 4; }
done < "$DST/h335.files"
rsync -a --chmod=Fa-w --files-from="$DST/h335.files" "$H335/" "$DST/code/"
git -C "$REPO" show "origin/v1/next:$G381/gp_h381.py" > "$DST/code/gp_h381.py"
git -C "$REPO" show "origin/v1/next:$G410/deploy_h410.py" > "$DST/code/deploy_h410.py"
for f in gp_h415.py deploy_h415.py train_h415.py; do git -C "$REPO" show "origin/v1/next:$G415/$f" > "$DST/code/$f"; done
for f in gp_h416.py deploy_h416.py meta_h416.py train_h416.py DESIGN.json; do
  git -C "$REPO" show "origin/v1/next:$G416/$f" > "$DST/code/$f"
done
chmod 0444 "$DST"/code/gp_h381.py "$DST"/code/deploy_h410.py "$DST"/code/*_h415.py "$DST"/code/*_h416.py \
  "$DST"/code/DESIGN.json
for f in "${H418_FILES[@]}"; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H418.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G418/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  else
    cp "$SRCDIR/$f" "$DST/code/$dst"
  fi
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/h335.files" "$DST/source_commit" "$DST/mode" "$DST/code"
"${SSH[@]}" "mkdir -p /home/ubuntu/h418-stage"
rsync -a "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256 && [ ! -w $DST/code/d91_h418.py ]"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0) mode $MODE from" \
  "$(cut -c1-10 "$DST/source_commit"); CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
