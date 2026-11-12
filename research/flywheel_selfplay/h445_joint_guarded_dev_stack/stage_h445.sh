#!/bin/bash
# Stage an H445 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2:
#   /home/ubuntu/apex-stage/h440/pub-r2 (672c99372a969dda) + H426 restate + H430/H433/H442 modules + published H445 files.
#   stage_h445.sh TAG
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h440/pub-r2
SHA_BASE=672c99372a969dda
G=research/flywheel_selfplay/h445_joint_guarded_dev_stack
FILES=(cols_cw5_dev.py stack_joint_dev.py stage_h445.sh driver_h445.sh gates_h445.sh PREREG.md)
DST=/home/ubuntu/apex-stage/h445/pub-$TAG
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
[ "$(cat "$BASE/mode")" = pub ] || { echo "$BASE is not a pub stage" >&2; exit 4; }
(cd "$BASE/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(sha256sum "$BASE/CODE.sha256" | cut -c1-16)" = "$SHA_BASE" ] || { echo "$BASE CODE.sha256 != $SHA_BASE" >&2; exit 4; }

mkdir -p "$DST/code"
cp -a "$BASE/code/." "$DST/code/"
chmod u+w "$DST/code"
sha256sum "$BASE/CODE.sha256" | cut -d' ' -f1 > "$DST/base_CODE.sha256"

git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next

git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h426_dev_cluster_restatement/restate_h426.py" > "$DST/code/restate_h426.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h430_exact_key_encoders/keys_core.py" > "$DST/code/keys_core.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h433_raw_formula_features/formula_core.py" > "$DST/code/formula_core.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h442_cw5_hybrid_icl_design/struct_cols.py" > "$DST/code/struct_cols.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h442_cw5_hybrid_icl_design/eval_cw5_fitsel.py" > "$DST/code/eval_cw5_fitsel.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h442_cw5_hybrid_icl_design/FROZEN.json" > "$DST/code/FROZEN_CW5.json"
chmod 0555 "$DST/code/restate_h426.py" "$DST/code/keys_core.py" "$DST/code/formula_core.py" \
           "$DST/code/struct_cols.py" "$DST/code/eval_cw5_fitsel.py"
chmod 0444 "$DST/code/FROZEN_CW5.json"

for f in "${FILES[@]}"; do
  dst=$f
  [ "$f" = PREREG.md ] && dst=PREREG_H445.md
  [ ! -e "$DST/code/$dst" ] || { echo "$dst already exists in base stage" >&2; exit 5; }
  git -C "$REPO" show "origin/v1/next:$G/$f" > "$DST/code/$dst" || { echo "$f is not on origin/v1/next" >&2; exit 5; }
  chmod 0555 "$DST/code/$dst"
done

git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo pub > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
