#!/bin/bash
# Stage an H449 code root on xbabe2 and copy it to xbabe0 (create-once, read-only, CODE.sha256 next to code/).
#   stage_h449.sh pub TAG   H449 files from published origin/v1/next bytes
#   stage_h449.sh dev TAG   H449 files from working copy (import-check only; never scored)
#   -> /home/ubuntu/h449-stage/<mode>-TAG on xbabe2 and xbabe0
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
case $MODE in pub|dev) ;; *) echo "mode is pub or dev" >&2; exit 2 ;; esac
SRCDIR=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
G=research/flywheel_selfplay
G447=$G/h447_legal_icl_104m_24m_r4e8
G449=$G/h449_unfair_all_d91_diagnostic
H449_FILES=(d91_h449.py sandbox_h449_unfair_d91.sh gates_h449_prelock.sh build_non_d91_complete_bank.py
            train_unfair_icl_h449.py analyze_h449_unfair.py token_compute_ledger.py
            selftest_h449.py run_h449.sh stage_h449.sh PREREG.md)
DST=/home/ubuntu/h449-stage/$MODE-$TAG
SSH=(ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes xbabe0)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
git -C "$REPO" -c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential" \
  fetch -q https://github.com/neverhuman/JopeDime.git +refs/heads/v1/next:refs/remotes/origin/v1/next
mkdir -p "$DST/code"
(cd "$H335" && find . -maxdepth 1 -type f ! -name '.*' | sort | sed 's#^\./##') > "$DST/h335.files"
while read -r f; do
  cmp -s "$H335/$f" <(git -C "$REPO" show "origin/v1/next:$G/h335_aggressive_locked_elo/$f") \
    || { echo "H335 file $f differs from origin/v1/next" >&2; exit 4; }
done < "$DST/h335.files"
rsync -a --chmod=Fa-w --files-from="$DST/h335.files" "$H335/" "$DST/code/"

git -C "$REPO" show "origin/v1/next:$G/h381_rapid_lf_grow/gp_h381.py" > "$DST/code/gp_h381.py"
git -C "$REPO" show "origin/v1/next:$G/h410_deploy_population/deploy_h410.py" > "$DST/code/deploy_h410.py"
for f in gp_h415.py deploy_h415.py train_h415.py; do
  git -C "$REPO" show "origin/v1/next:$G/h415_committee_kfold_bag/$f" > "$DST/code/$f"
done
for f in gp_h416.py deploy_h416.py meta_h416.py train_h416.py DESIGN.json; do
  git -C "$REPO" show "origin/v1/next:$G/h416_stability_stack/$f" > "$DST/code/$f"
done
git -C "$REPO" show "origin/v1/next:$G/h418_sxb_dev91/a2_cells.json" > "$DST/code/a2_cells.json"
for f in icl_model.py icl_prior.py icl_inject.py icl_diag.py icl_resid.py; do
  git -C "$REPO" show "origin/v1/next:$G/h419_apex_icl/$f" > "$DST/code/$f"
done
git -C "$REPO" show "origin/v1/next:$G/h431_cw3ab_icl_battery/icl_prior_v3s.py" > "$DST/code/icl_prior_v3s.py"
git -C "$REPO" show "origin/v1/next:$G/h422_learned_stacker/cache_h422.py" > "$DST/code/cache_h422.py"
git -C "$REPO" show "origin/v1/next:$G/h430_exact_key_encoders/keys_core.py" > "$DST/code/keys_core.py"
git -C "$REPO" show "origin/v1/next:$G/h433_raw_formula_features/formula_core.py" > "$DST/code/formula_core.py"
git -C "$REPO" show "origin/v1/next:$G/h440_guard_and_duan_matern_cols/guard_matern_core.py" > "$DST/code/guard_matern_core.py"
for f in struct_cols.py eval_cw5_fitsel.py; do
  git -C "$REPO" show "origin/v1/next:$G/h442_cw5_hybrid_icl_design/$f" > "$DST/code/$f"
done
git -C "$REPO" show "origin/v1/next:$G447/cols_r4_e8.py" > "$DST/code/cols_r4_e8.py"

if [ "$MODE" = pub ]; then
  git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json" > "$DST/code/FROZEN_FAIR_CHAMPION.json" \
    || { echo "FROZEN_FAIR_CHAMPION.json is not published on origin/v1/next" >&2; exit 5; }
else
  if git -C "$REPO" cat-file -e "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json" 2>/dev/null; then
    git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json" > "$DST/code/FROZEN_FAIR_CHAMPION.json"
  elif [ -f /home/ubuntu/jope_plan/evidence/H447/results/FROZEN_FAIR_CHAMPION.json ]; then
    cp /home/ubuntu/jope_plan/evidence/H447/results/FROZEN_FAIR_CHAMPION.json "$DST/code/FROZEN_FAIR_CHAMPION.json"
  else
    printf '{"schema":"jope.h447.frozen_fair_champion/v1","selected_arm":"STACK_R4_E8_BASE","blocks":{"icl":{"enabled":true,"col_name":"r4_e8","training_bank":"fit_icl_bank.npz","training_bank_sha256":"%s"}}}\n' \
      f6e283cebf9c7e40cd62b7cedf8f1283d010f7dc5feff0ab5045d0b2fd8e8ce7 > "$DST/code/FROZEN_FAIR_CHAMPION.json"
  fi
fi

chmod 0444 "$DST"/code/gp_h381.py "$DST"/code/deploy_h410.py "$DST"/code/*_h415.py "$DST"/code/*_h416.py \
  "$DST"/code/DESIGN.json "$DST"/code/a2_cells.json "$DST"/code/icl_*.py "$DST"/code/cache_h422.py \
  "$DST"/code/keys_core.py "$DST"/code/formula_core.py \
  "$DST"/code/guard_matern_core.py "$DST"/code/struct_cols.py "$DST"/code/eval_cw5_fitsel.py \
  "$DST"/code/cols_r4_e8.py "$DST"/code/FROZEN_FAIR_CHAMPION.json

for f in "${H449_FILES[@]}"; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H449.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G449/$f" > "$DST/code/$dst" \
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
"${SSH[@]}" "mkdir -p /home/ubuntu/h449-stage"
rsync -a -e "ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes" "$DST/" "xbabe0:$DST/"
"${SSH[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256 && [ ! -w $DST/code/d91_h449.py ]"
echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0) mode $MODE"
