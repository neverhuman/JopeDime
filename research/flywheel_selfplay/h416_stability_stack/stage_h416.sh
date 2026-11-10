#!/bin/bash
# Stage an H416 code root on xbabe2 and copy it to xbabe0 (create-once), read-only, with CODE.sha256 next to code/:
#   * the frozen H335 files (each byte-identical to origin/v1/next);
#   * gp_h381.py, deploy_h410.py and the H415 files gp_h415.py, deploy_h415.py, train_h415.py at their published bytes;
#   * the P201b board files at their origin/v1/next bytes: BOARD.json, ladder.json and every scores/<id>.jsonl, as
#     board_p201b_BOARD.json, board_p201b_ladder.json, board_p201b__<id>.jsonl;
#   * the named H416 files. The H416 PREREG is staged as PREREG_H416.md (code/PREREG.md is H335's).
#   stage_h416.sh pub TAG FILE...   H416 files = their published bytes on origin/v1/next (every scored run)
#   stage_h416.sh dev TAG FILE...   H416 files from the working copy (smoke only; never scored)
#   -> /home/ubuntu/h416-stage/<mode>-TAG on xbabe2 and xbabe0
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}; shift 2
[ $# -ge 1 ] || { echo "name the H416 files" >&2; exit 2; }
SRC=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
G335=research/flywheel_selfplay/h335_aggressive_locked_elo
G381=research/flywheel_selfplay/h381_rapid_lf_grow
G410=research/flywheel_selfplay/h410_deploy_population
G415=research/flywheel_selfplay/h415_committee_kfold_bag
G416=research/flywheel_selfplay/h416_stability_stack
BRD=research/elo/reference-board/boards/p201b-train-audit-v1
DST=/home/ubuntu/h416-stage/$MODE-$TAG
SSH=(ssh -o BatchMode=yes xbabe0)
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
for f in gp_h415.py deploy_h415.py train_h415.py; do
  git -C "$REPO" show "origin/v1/next:$G415/$f" > "$DST/code/$f"
done
git -C "$REPO" show "origin/v1/next:$BRD/BOARD.json" > "$DST/code/board_p201b_BOARD.json"
git -C "$REPO" show "origin/v1/next:$BRD/ladder.json" > "$DST/code/board_p201b_ladder.json"
git -C "$REPO" ls-tree --name-only "origin/v1/next:$BRD/scores/" | while read -r s; do
  git -C "$REPO" show "origin/v1/next:$BRD/scores/$s" > "$DST/code/board_p201b__$s"
done
chmod 0444 "$DST"/code/gp_h381.py "$DST"/code/deploy_h410.py "$DST"/code/*_h415.py "$DST"/code/board_p201b_*
for f in "$@" PREREG.md; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H416.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G416/$f" > "$DST/code/$dst" \
      || { echo "$f is not published on origin/v1/next" >&2; exit 5; }
  else
    cp "$SRC/$f" "$DST/code/$dst"
  fi
  chmod 0555 "$DST/code/$dst"
done
git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo "$MODE" > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/h335.files" "$DST/source_commit" "$DST/mode"
"${SSH[@]}" "mkdir -p /home/ubuntu/h416-stage"
rsync -a "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0) mode $MODE from $(cut -c1-10 "$DST/source_commit"); CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16)"
