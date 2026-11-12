#!/bin/bash
# Stage an H447 code root (create-once, read-only, CODE.sha256 next to code/) on xbabe2:
#   /home/ubuntu/apex-stage/h440/pub-r2 (672c99372a969dda) + H419/H430/H431/H433/H442/H446 modules + published H447 files.
#   stage_h447.sh TAG
set -euo pipefail
TAG=${1:?TAG}
REPO=/home/ubuntu/JopeDime
BASE=/home/ubuntu/apex-stage/h440/pub-r2
SHA_BASE=672c99372a969dda
G=research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8
FILES=(warm_continue_icl.py cols_r4_e8.py eval_h447.py stage_h447.sh driver_h447.sh gates_h447.sh PREREG.md)
DST=/home/ubuntu/apex-stage/h447/pub-$TAG
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

git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h419_apex_icl/icl_prior.py" > "$DST/code/icl_prior.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h431_cw3ab_icl_battery/icl_prior_v3s.py" > "$DST/code/icl_prior_v3s.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h430_exact_key_encoders/keys_core.py" > "$DST/code/keys_core.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h433_raw_formula_features/formula_core.py" > "$DST/code/formula_core.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h442_cw5_hybrid_icl_design/struct_cols.py" > "$DST/code/struct_cols.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h442_cw5_hybrid_icl_design/eval_cw5_fitsel.py" > "$DST/code/eval_cw5_fitsel.py"
git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/h446_all_legal_real_bank/icl_bank_all_legal.py" > "$DST/code/icl_bank_all_legal.py"
chmod 0555 "$DST/code/icl_prior.py" "$DST/code/icl_prior_v3s.py" "$DST/code/keys_core.py" \
           "$DST/code/formula_core.py" "$DST/code/struct_cols.py" "$DST/code/eval_cw5_fitsel.py" \
           "$DST/code/icl_bank_all_legal.py"

for f in "${FILES[@]}"; do
  dst=$f
  [ "$f" = PREREG.md ] && dst=PREREG_H447.md
  [ ! -e "$DST/code/$dst" ] || { echo "$dst already exists in base stage" >&2; exit 5; }
  git -C "$REPO" show "origin/v1/next:$G/$f" > "$DST/code/$dst" || { echo "$f is not on origin/v1/next" >&2; exit 5; }
  chmod 0555 "$DST/code/$dst"
done

git -C "$REPO" rev-parse origin/v1/next > "$DST/source_commit"
echo pub > "$DST/mode"
(cd "$DST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$DST/CODE.sha256"
chmod a-w "$DST/CODE.sha256" "$DST/base_CODE.sha256" "$DST/source_commit" "$DST/mode" "$DST/code"
echo "staged $DST ($(wc -l < "$DST/CODE.sha256") files, CODE.sha256 $(sha256sum "$DST/CODE.sha256" | cut -c1-16))"
