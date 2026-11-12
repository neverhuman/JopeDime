#!/bin/bash
# Stage an H450 code root on xbabe2 and copy it to xbabe0 & xbabe3 (create-once, read-only, CODE.sha256 next to code/).
#   stage_h450.sh pub TAG   H450 files from published origin/v1/next bytes
#   stage_h450.sh dev TAG   H450 files from working copy (import-check only; never scored)
#   -> /home/ubuntu/h450-stage/<mode>-TAG on xbabe2, xbabe0, and xbabe3
set -euo pipefail
MODE=${1:?pub|dev}; TAG=${2:?TAG}
case $MODE in pub|dev) ;; *) echo "mode is pub or dev" >&2; exit 2 ;; esac
SRCDIR=$(dirname "$(readlink -f "$0")")
REPO=/home/ubuntu/JopeDime
H335=/home/ubuntu/h335-stage/code
G=research/flywheel_selfplay
G447=$G/h447_legal_icl_104m_24m_r4e8
G450=$G/h450_aggressive_random_all_elo1
H450_FILES=(PREREG.md gates_h450_quarantine.sh icl_model_h450.py relational_struct_h450.py robust_resid_h450.py
            build_combined_f64_bank_h450.py d91_h450.py sandbox_h450_quarantine.sh train_aggressive_random_h450.py
            cols_h450_gpu.py cols_tab_h450.py stack_h450.py analyze_h450.py token_compute_ledger_h450.py
            selftest_h450.py gpu_window_x2_x3.sh stage_h450.sh run_h450.sh)
DST=/home/ubuntu/h450-stage/$MODE-$TAG
SSH_X0=(ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes xbabe0)
SSH_X3=(ssh -o BatchMode=yes xbabe3)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
[ ! -e "$DST" ] || { echo "$DST exists (create-once)" >&2; exit 3; }
"${SSH_X0[@]}" "[ ! -e $DST ]" || { echo "xbabe0:$DST exists (create-once)" >&2; exit 3; }
"${SSH_X3[@]}" "[ ! -e $DST ]" || { echo "xbabe3:$DST exists (create-once)" >&2; exit 3; }
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
git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json" > "$DST/code/FROZEN_FAIR_CHAMPION.json"

chmod 0444 "$DST"/code/gp_h381.py "$DST"/code/deploy_h410.py "$DST"/code/*_h415.py "$DST"/code/*_h416.py \
  "$DST"/code/DESIGN.json "$DST"/code/a2_cells.json "$DST"/code/icl_*.py "$DST"/code/cache_h422.py \
  "$DST"/code/keys_core.py "$DST"/code/formula_core.py \
  "$DST"/code/guard_matern_core.py "$DST"/code/struct_cols.py "$DST"/code/eval_cw5_fitsel.py \
  "$DST"/code/cols_r4_e8.py "$DST"/code/FROZEN_FAIR_CHAMPION.json

for f in "${H450_FILES[@]}"; do
  dst=$f; [ "$f" = PREREG.md ] && dst=PREREG_H450.md
  if [ "$MODE" = pub ]; then
    git -C "$REPO" show "origin/v1/next:$G450/$f" > "$DST/code/$dst" \
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

"${SSH_X0[@]}" "mkdir -p /home/ubuntu/h450-stage"
rsync -a -e "ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes" "$DST/" "xbabe0:$DST/"
"${SSH_X0[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256 && [ ! -w $DST/code/d91_h450.py ]"

"${SSH_X3[@]}" "mkdir -p /home/ubuntu/h450-stage"
rsync -a -e "ssh -o BatchMode=yes" "$DST/" "xbabe3:$DST/"
"${SSH_X3[@]}" "cd $DST/code && sha256sum --quiet -c ../CODE.sha256"

echo "staged $(wc -l < "$DST/CODE.sha256") files to $DST (xbabe2 + xbabe0 + xbabe3) mode $MODE"
