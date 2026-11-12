#!/bin/bash
# H450 Quarantined Random-Split All-Data Maximum-Elo Orchestration (`run_h450.sh`).
# STRICTLY QUARANTINED (`canonical_promotion_forbidden = true`):
#   - Enforced by `gates_h450_quarantine.sh` + separate owner receipt `H450_UNFAIR_LOCK_sha256.txt`
#   - Zero writes to `research/elo/reference-board/` or canonical checkpoint directories
#   - Zero opens of `APEX-CONFIRM`, `DL-Holdout`, or `sealed`
#   - Bounded GPU window strictly on `xbabe2` + `xbabe3` (`xbabe1` untouched), verifying `loaded=true` sync on release.
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H450_STAGE:?H450_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h450_aggressive_random_all_elo1
G447=research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8
BRD=research/elo/reference-board/boards/dev91-audit-v1
EV=/home/ubuntu/jope_plan/evidence/H450
EV448=/home/ubuntu/jope_plan/evidence/H448
P418=/home/ubuntu/jope_plan/evidence/H418/pulled
P448=/home/ubuntu/jope_plan/evidence/H448/pulled
DATA=/home/ubuntu/jope-data/h450-quarantine
DATA_X3=/home/ubuntu/jope-data/h450-quarantine-x3
NON_D91_ROOT=/home/ubuntu/jope-data/h449-non-d91
SV=/home/ubuntu/h335-stage-v
PREP=x0_prep_20261105T005754Z
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H450_UNFAIR_LOCK_sha256.txt
H450_FILES=(gates_h450_quarantine.sh icl_model_h450.py relational_struct_h450.py robust_resid_h450.py
            build_combined_f64_bank_h450.py d91_h450.py sandbox_h450_quarantine.sh train_aggressive_random_h450.py
            cols_h450_gpu.py cols_tab_h450.py stack_h450.py analyze_h450.py token_compute_ledger_h450.py
            selftest_h450.py gpu_window_x2_x3.sh stage_h450.sh run_h450.sh)
GH=/home/ubuntu/.local/bin/gh-role
CRED=(-c credential.helper= -c "credential.helper=!$GH writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git
SSH_X0=(ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes xbabe0)
SSH_X3=(ssh -o BatchMode=yes xbabe3)
rs0() { rsync -a -e "ssh -i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes" "$@"; }
rs3() { rsync -a -e "ssh -o BatchMode=yes" "$@"; }
die() { echo "run_h450 refused: $*" >&2; exit 3; }

[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "staged code differs from CODE.sha256"
MODE=$(cat "$STAGE/mode")
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$EV/logs" "$EV/diag_results" "$EV/inputs" "$EV/pulled"

case $STEP in
  selftest|import-check) ;;
  *)
    [ "$MODE" = pub ] || die "$STAGE is not a pub stage"
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next
    bash "$STAGE/code/gates_h450_quarantine.sh"
    cmp -s "$STAGE/code/PREREG_H450.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
      || die "staged PREREG differs from origin/v1/next"
    cmp -s "$STAGE/code/FROZEN_FAIR_CHAMPION.json" <(git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json") \
      || die "staged FROZEN_FAIR_CHAMPION.json differs from origin/v1/next"
    for f in "${H450_FILES[@]}"; do
      cmp -s "$STAGE/code/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") || die "$f differs from origin/v1/next"
    done ;;
esac

sandbox() {
  local m=$1 c=$2 mem=$3; shift 3
  "${SSH_X0[@]}" "H450_STAGE=$STAGE H450_CPUSET=$c H450_MEMORY=$mem bash $STAGE/code/sandbox_h450_quarantine.sh $m $*"
}
prereg_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$WDIR/PREREG.md"; }
h448_close_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$BRD/scores/h448.APEX_FAIR_CHAMPION.jsonl"; }

case $STEP in
  selftest)
    nice -n 19 python3 -B "$STAGE/code/selftest_h450.py" | tee "$EV/logs/selftest.txt" ;;

  import-check)
    sandbox prelock 8-9 4g python3 -u d91_h450.py --help | head -n 6 ;;

  inputs)
    PC=$(prereg_commit); HC=$(h448_close_commit)
    [ ${#PC} = 40 ] && [ ${#HC} = 40 ] || die "missing PREREG or H448 closeout commit on origin/v1/next"
    NON_CK_L=$NON_D91_ROOT/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt
    NON_CK_M=$NON_D91_ROOT/ckpts_v2/all_non_d91_M/ckpt_unfair_final.pt
    [ -s "$NON_CK_L" ] && [ -s "$NON_CK_M" ] || die "missing 0%-d91 GPU checkpoints under $NON_D91_ROOT"

    B=$EV/board_post_h448/dev91-audit-v1
    [ ! -e "$B" ] || die "$B exists (create-once)"
    [ ! -e "$DATA/inputs/non_d91_plus_cousins_f64.npz" ] || die "$DATA/inputs/non_d91_plus_cousins_f64.npz exists (create-once)"
    "${SSH_X0[@]}" "[ ! -e $DATA/h335-prep ] && [ -z \"\$(ls -A $DATA/inputs 2>/dev/null || true)\" ]" || die "xbabe0 inputs exist (create-once)"

    mkdir -p "$DATA/inputs" "$DATA/logs"
    chmod 0700 "$DATA"
    echo "$(date -u +%FT%TZ) Building combined float64 CSPRNG-shuffled non_d91 + d91_cousin_s3 bank (3,434 tables)..."
    nice -n 19 python3 -B "$STAGE/code/build_combined_f64_bank_h450.py" \
      --out "$DATA/inputs/non_d91_plus_cousins_f64.npz" --procs 32 | tee "$EV/logs/build_non_d91_f64.json"
    cp "$DATA/inputs/non_d91_plus_cousins_f64.npz.json" "$EV/inputs/non_d91_plus_cousins_f64.json"

    cp "$NON_CK_L" "$DATA/inputs/ckpt_non_d91_L.pt"
    cp "$NON_CK_M" "$DATA/inputs/ckpt_non_d91_M.pt"
    cp "$NON_D91_ROOT/ckpts_v2/all_non_d91_L/unfair_manifest.json" "$EV/inputs/non_d91_manifest_L.json"
    cp "$NON_D91_ROOT/ckpts_v2/all_non_d91_M/unfair_manifest.json" "$EV/inputs/non_d91_manifest_M.json"
    cp "$P418/D1.sha256" "$EV/inputs/d1_pulled.sha256"
    cp "$P418/D2.sha256" "$EV/inputs/d2_pulled.sha256"

    mkdir -p "$B/scores"
    git -C "$REPO" show "$HC:$BRD/BOARD.json" > "$B/BOARD.json"
    git -C "$REPO" show "$HC:$BRD/ladder.json" > "$B/ladder.json"
    git -C "$REPO" ls-tree --name-only "$HC:$BRD/scores/" | while read -r s; do
      git -C "$REPO" show "$HC:$BRD/scores/$s" > "$B/scores/$s"
    done
    [ -s "$B/scores/h448.APEX_FAIR_CHAMPION.jsonl" ] || die "post-H448 board lacks h448.APEX_FAIR_CHAMPION.jsonl"
    (cd "$B" && find . -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$EV/inputs/board_post_h448.sha256"
    chmod -R a-w "$B" "$EV/inputs/board_post_h448.sha256"

    "${SSH_X0[@]}" "set -euo pipefail; mkdir -p $DATA/h335-prep $DATA/inputs && chmod 0700 $DATA
      for k in tables root; do cp $SV/$PREP.\$k.sha256 $DATA/h335-prep/; chmod a-w $DATA/h335-prep/$PREP.\$k.sha256; done"

    rs0 "$DATA/inputs/non_d91_plus_cousins_f64.npz" "$DATA/inputs/ckpt_non_d91_L.pt" "$DATA/inputs/ckpt_non_d91_M.pt" \
        "$EV/inputs/d1_pulled.sha256" "$EV/inputs/d2_pulled.sha256" "$EV/inputs/board_post_h448.sha256" "xbabe0:$DATA/inputs/"
    "${SSH_X0[@]}" "set -euo pipefail; cd $DATA/inputs
      ls -1 non_d91_plus_cousins_f64.npz ckpt_non_d91_L.pt ckpt_non_d91_M.pt d1_pulled.sha256 d2_pulled.sha256 | sort | xargs sha256sum > h450_inputs.sha256
      chmod 0444 non_d91_plus_cousins_f64.npz ckpt_non_d91_L.pt ckpt_non_d91_M.pt d1_pulled.sha256 d2_pulled.sha256 board_post_h448.sha256 h450_inputs.sha256"
    rs0 "xbabe0:$DATA/inputs/h450_inputs.sha256" "$EV/inputs/h450_inputs.sha256"

    jq -n --arg p "$PC" --arg s "$(cat "$STAGE/source_commit")" --arg h "$HC" \
      '{prereg_commit: $p, stage_commit: $s, h448_closeout_commit: $h}' > "$EV/inputs/spec.json"
    rs0 "$EV/inputs/spec.json" "xbabe0:$DATA/inputs/spec.json"
    "${SSH_X0[@]}" "chmod a-w $DATA/inputs/*"
    chmod 0444 "$DATA/inputs/non_d91_plus_cousins_f64.npz" "$DATA/inputs/ckpt_non_d91_L.pt" "$DATA/inputs/ckpt_non_d91_M.pt"
    echo "H450 inputs OK" | tee -a "$EV/logs/inputs.log" ;;

  prelock)
    [ ! -e "$RECEIPT" ] || die "$RECEIPT already exists"
    sandbox prelock 8-15 8g python3 -u d91_h450.py lock --spec "$DATA/inputs/spec.json" | tee "$EV/logs/prelock.log"
    S=$("${SSH_X0[@]}" "sha256sum $DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    mkdir -p "$EV/stage_v"
    rs0 "xbabe0:$DATA/stage_v/LOCK.json" "$EV/stage_v/LOCK.json"
    chmod 0444 "$EV/stage_v/LOCK.json"
    echo "H450_UNFAIR_PRELOCK_READY sha256=$S (awaiting separate owner receipt $RECEIPT)" | tee -a "$EV/logs/prelock.log" ;;

  unlock-and-banks)
    [ -s "$RECEIPT" ] || die "missing separate owner receipt $RECEIPT"
    S_LOCK=$(sha256sum "$EV/stage_v/LOCK.json" | cut -d' ' -f1)
    [ "$(awk '{print $1}' "$RECEIPT")" = "$S_LOCK" ] || die "receipt != LOCK.json"
    if ! "${SSH_X0[@]}" "[ -s $RECEIPT ]"; then
      R_CONTENT=$(cat "$RECEIPT")
      "${SSH_X0[@]}" "mkdir -p $(dirname "$RECEIPT") && set -o noclobber && printf '%s\n' '$R_CONTENT' > $RECEIPT && chmod 0444 $RECEIPT"
    fi
    sandbox unlock 8-15 8g python3 -u d91_h450.py unlock | tee -a "$EV/logs/unlock.log"
    sandbox run 8-15 24g python3 -u d91_h450.py build_banks | tee "$EV/logs/build_banks.log"
    rs0 "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/banks/banks_manifest_f64.json" "xbabe0:$DATA/stage_v/STAGE_V_STARTED" \
      "$EV/stage_v/"
    chmod a-w "$EV"/stage_v/*

    mkdir -p "$DATA/banks"
    rs0 "xbabe0:$DATA/banks/everything_combined_f64_h450_bank.npz" "$DATA/banks/everything_combined_f64_h450_bank.npz"
    rs0 "xbabe0:$DATA/bank_d91_search_f64.npz" "$DATA/bank_d91_search_f64.npz"
    EXP_ALL=$(jq -r .everything_combined_f64.sha256 "$EV/stage_v/banks_manifest_f64.json")
    EXP_EVL=$(jq -r .eval_bank_d91_search_f64.sha256 "$EV/stage_v/banks_manifest_f64.json")
    [ "$(sha256sum "$DATA/banks/everything_combined_f64_h450_bank.npz" | awk '{print $1}')" = "$EXP_ALL" ] || die "everything_combined_f64 sha mismatch"
    [ "$(sha256sum "$DATA/bank_d91_search_f64.npz" | awk '{print $1}')" = "$EXP_EVL" ] || die "bank_d91_search_f64 sha mismatch"
    chmod 0444 "$DATA/banks/everything_combined_f64_h450_bank.npz" "$DATA/bank_d91_search_f64.npz"
    echo "H450_STAGE_V_BANKS_READY all_sha=$EXP_ALL eval_sha=$EXP_EVL" ;;

  cols-tab-cpu)
    PROCS=${1:-32}
    echo "$(date -u +%FT%TZ) Running cols_tab_h450.py (--procs $PROCS) on bank_d91_search_f64.npz (0 audit targets)..."
    nice -n 19 python3 -B "$STAGE/code/cols_tab_h450.py" \
      --bank "$DATA/bank_d91_search_f64.npz" \
      --out "$DATA/cols_tab_h450_D91" \
      --procs "$PROCS" | tee "$EV/logs/cols_tab_h450.log"
    cp "$DATA/cols_tab_h450_D91/summary.json" "$EV/pulled/cols_tab_h450_summary.json" ;;

  gpu-train-and-cols)
    HOURS=${1:-0.48}
    ALL_BANK=$DATA/banks/everything_combined_f64_h450_bank.npz
    EVAL_BANK=$DATA/bank_d91_search_f64.npz
    CK_NON_L=$DATA/inputs/ckpt_non_d91_L.pt
    CK_NON_M=$DATA/inputs/ckpt_non_d91_M.pt
    [ -s "$ALL_BANK" ] && [ -s "$EVAL_BANK" ] || die "missing Stage-V banks under $DATA"

    echo "$(date -u +%FT%TZ) Pre-staging quarantined inputs to xbabe3:$DATA_X3 BEFORE GPU pause (0 wasted GPU time)..."
    "${SSH_X3[@]}" "rm -rf $DATA_X3 && mkdir -p $DATA_X3/banks $DATA_X3/inputs $DATA_X3/logs $DATA_X3/ckpts && chmod 0700 $DATA_X3"
    rs3 "$ALL_BANK" "xbabe3:$DATA_X3/banks/everything_combined_f64_h450_bank.npz"
    rs3 "$EVAL_BANK" "xbabe3:$DATA_X3/bank_d91_search_f64.npz"
    rs3 "$CK_NON_L" "$CK_NON_M" "xbabe3:$DATA_X3/inputs/"

    cleanup_gpu() {
      local rc=$?
      echo "$(date -u +%FT%TZ) Cleaning up xbabe3 quarantined scratch and releasing xbabe2+xbabe3 GPUs (rc=$rc)..."
      "${SSH_X3[@]}" "rm -rf $DATA_X3" || true
      bash "$STAGE/code/gpu_window_x2_x3.sh" release || true
      exit $rc
    }
    trap cleanup_gpu EXIT

    bash "$STAGE/code/gpu_window_x2_x3.sh" acquire

    echo "$(date -u +%FT%TZ) [GPU 1/4] 2-step smoke check on xbabe2 RTX 3090..."
    rm -rf "$DATA/smoke_check"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B "$STAGE/code/train_aggressive_random_h450.py" \
      --size M --init "$CK_NON_M" --bank "$ALL_BANK" --out "$DATA/smoke_check" \
      --max-steps 2 --workers 2 --tokens 96000 --row-tokens 3072 --bmax 6 --grad-accum 1 --seed 450999 > "$EV/logs/gpu_smoke.log" 2>&1
    rm -rf "$DATA/smoke_check"
    echo "$(date -u +%FT%TZ) Smoke check passed!"

    echo "$(date -u +%FT%TZ) [GPU 2/4] Launching 4-model multi-seed online random-split fine-tuning (--hours $HOURS) across xbabe2 & xbabe3..."
    mkdir -p "$DATA/ckpts" "$DATA/logs"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B "$STAGE/code/train_aggressive_random_h450.py" \
      --size L --init "$CK_NON_L" --bank "$ALL_BANK" --out "$DATA/ckpts/L_seedA" \
      --hours "$HOURS" --lr 3.0e-5 --d91-mix 0.30 --synth-prob 0.08 \
      --tokens 128000 --row-tokens 4096 --bmax 8 --grad-accum 3 --workers 8 --seed 450101 \
      > "$DATA/logs/train_L_seedA.log" 2>&1 &
    PID_LA=$!

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B "$STAGE/code/train_aggressive_random_h450.py" \
      --size M --init "$CK_NON_M" --bank "$ALL_BANK" --out "$DATA/ckpts/M_seedA" \
      --hours "$HOURS" --lr 4.0e-5 --d91-mix 0.30 --synth-prob 0.08 \
      --tokens 144000 --row-tokens 4096 --bmax 8 --grad-accum 3 --workers 6 --seed 450303 \
      > "$DATA/logs/train_M_seedA.log" 2>&1 &
    PID_MA=$!

    "${SSH_X3[@]}" "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      python3 -B $STAGE/code/train_aggressive_random_h450.py \
        --size L --init $DATA_X3/inputs/ckpt_non_d91_L.pt --bank $DATA_X3/banks/everything_combined_f64_h450_bank.npz \
        --out $DATA_X3/ckpts/L_seedB --hours $HOURS --lr 3.0e-5 --d91-mix 0.50 --synth-prob 0.08 \
        --tokens 128000 --row-tokens 4096 --bmax 8 --grad-accum 3 --workers 8 --seed 450202 \
        > $DATA_X3/logs/train_L_seedB.log 2>&1 &
      P1=\$!
      python3 -B $STAGE/code/train_aggressive_random_h450.py \
        --size M --init $DATA_X3/inputs/ckpt_non_d91_M.pt --bank $DATA_X3/banks/everything_combined_f64_h450_bank.npz \
        --out $DATA_X3/ckpts/M_seedB --hours $HOURS --lr 4.0e-5 --d91-mix 0.50 --synth-prob 0.08 \
        --tokens 144000 --row-tokens 4096 --bmax 8 --grad-accum 3 --workers 6 --seed 450404 \
        > $DATA_X3/logs/train_M_seedB.log 2>&1 &
      P2=\$!
      rc=0; wait \$P1 || rc=\$?; wait \$P2 || rc=\$?; exit \$rc" &
    PID_X3_TR=$!

    rc=0
    wait "$PID_LA" || rc=$?
    wait "$PID_MA" || rc=$?
    wait "$PID_X3_TR" || rc=$?
    [ "$rc" = 0 ] || die "GPU training failed (rc=$rc)"

    echo "$(date -u +%FT%TZ) Pulling SeedB SWA checkpoints + manifests from xbabe3 & syncing SeedA to xbabe3 for sharded eval..."
    rs3 "xbabe3:$DATA_X3/ckpts/L_seedB" "xbabe3:$DATA_X3/ckpts/M_seedB" "$DATA/ckpts/"
    rs3 "xbabe3:$DATA_X3/logs/" "$DATA/logs/x3_"
    rs3 "$DATA/ckpts/L_seedA" "$DATA/ckpts/M_seedA" "xbabe3:$DATA_X3/ckpts/"

    for k in L_seedA M_seedA L_seedB M_seedB; do
      cp "$DATA/ckpts/$k/manifest.json" "$EV/pulled/manifest_${k}.json"
    done

    echo "$(date -u +%FT%TZ) [GPU 3/4] Running sharded cols_h450_gpu.py for 4-model H450 SWA soup across xbabe2 & xbabe3..."
    mkdir -p "$DATA/cols_gpu_h450_D91"
    "${SSH_X3[@]}" "mkdir -p $DATA_X3/cols_gpu_h450_D91 $DATA_X3/cols_gpu_zero_d91_D91"

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B "$STAGE/code/cols_h450_gpu.py" \
      --bank "$EVAL_BANK" \
      --ckpt-la "$DATA/ckpts/L_seedA/ckpt_swa_soup.pt" \
      --ckpt-ma "$DATA/ckpts/M_seedA/ckpt_swa_soup.pt" \
      --ckpt-lb "$DATA/ckpts/L_seedB/ckpt_swa_soup.pt" \
      --ckpt-mb "$DATA/ckpts/M_seedB/ckpt_swa_soup.pt" \
      --out "$DATA/cols_gpu_h450_D91" --shard-idx 0 --num-shards 2 > "$DATA/logs/cols_gpu_s0.log" 2>&1 &
    P_S0=$!

    "${SSH_X3[@]}" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B $STAGE/code/cols_h450_gpu.py \
      --bank $DATA_X3/bank_d91_search_f64.npz \
      --ckpt-la $DATA_X3/ckpts/L_seedA/ckpt_swa_soup.pt \
      --ckpt-ma $DATA_X3/ckpts/M_seedA/ckpt_swa_soup.pt \
      --ckpt-lb $DATA_X3/ckpts/L_seedB/ckpt_swa_soup.pt \
      --ckpt-mb $DATA_X3/ckpts/M_seedB/ckpt_swa_soup.pt \
      --out $DATA_X3/cols_gpu_h450_D91 --shard-idx 1 --num-shards 2 > $DATA_X3/logs/cols_gpu_s1.log 2>&1" &
    P_S1=$!

    wait "$P_S0" || rc=$?
    wait "$P_S1" || rc=$?
    [ "$rc" = 0 ] || die "H450 GPU column generation failed (rc=$rc)"
    rs3 "xbabe3:$DATA_X3/cols_gpu_h450_D91/" "$DATA/cols_gpu_h450_D91/"

    echo "$(date -u +%FT%TZ) [GPU 4/4] Running sharded cols_h450_gpu.py for 0%-dev91 upgraded control (all_non_d91_L + M)..."
    mkdir -p "$DATA/cols_gpu_zero_d91_D91"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B "$STAGE/code/cols_h450_gpu.py" \
      --bank "$EVAL_BANK" \
      --ckpt-la "$CK_NON_L" --ckpt-ma "$CK_NON_M" \
      --out "$DATA/cols_gpu_zero_d91_D91" --shard-idx 0 --num-shards 2 > "$DATA/logs/cols_zero_s0.log" 2>&1 &
    PZ0=$!

    "${SSH_X3[@]}" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -B $STAGE/code/cols_h450_gpu.py \
      --bank $DATA_X3/bank_d91_search_f64.npz \
      --ckpt-la $DATA_X3/inputs/ckpt_non_d91_L.pt --ckpt-ma $DATA_X3/inputs/ckpt_non_d91_M.pt \
      --out $DATA_X3/cols_gpu_zero_d91_D91 --shard-idx 1 --num-shards 2 > $DATA_X3/logs/cols_zero_s1.log 2>&1" &
    PZ1=$!

    wait "$PZ0" || rc=$?
    wait "$PZ1" || rc=$?
    [ "$rc" = 0 ] || die "0%-d91 GPU column generation failed (rc=$rc)"
    rs3 "xbabe3:$DATA_X3/cols_gpu_zero_d91_D91/" "$DATA/cols_gpu_zero_d91_D91/"

    N_H450=$(find "$DATA/cols_gpu_h450_D91" -maxdepth 1 -name '*.npz' | wc -l)
    N_ZERO=$(find "$DATA/cols_gpu_zero_d91_D91" -maxdepth 1 -name '*.npz' | wc -l)
    [ "$N_H450" = 91 ] && [ "$N_ZERO" = 91 ] || die "incomplete GPU cols: h450=$N_H450 zero=$N_ZERO"

    trap - EXIT
    "${SSH_X3[@]}" "rm -rf $DATA_X3"
    bash "$STAGE/code/gpu_window_x2_x3.sh" release
    echo "$(date -u +%FT%TZ) H450 GPU TRAIN + COLS + @cl_apex RESUME VERIFIED!" ;;

  analyze-and-ledger)
    echo "$(date -u +%FT%TZ) Running analyze_h450.py (quarantined readout + 0%-d91 control + in-corpus arms)..."
    nice -n 19 python3 -B "$STAGE/code/analyze_h450.py" \
      --bank "$DATA/bank_d91_search_f64.npz" \
      --gpu-cols "$DATA/cols_gpu_h450_D91" \
      --tab-cols "$DATA/cols_tab_h450_D91" \
      --zero-d91-gpu-cols "$DATA/cols_gpu_zero_d91_D91" \
      --h448-pulled "$P448" \
      --h418-pulled "$P418" \
      --board "$EV/board_post_h448/dev91-audit-v1" \
      --lock "$EV/stage_v/LOCK.json" \
      --receipt "$RECEIPT" \
      --diag-out "$EV/diag_results/H450_DEV91.json" \
      --diag-txt "$EV/diag_results/H450_DEV91.txt" \
      --diag-board "$EV/diag_results/quarantined_board_dev91_h450" \
      --procs 16

    echo "$(date -u +%FT%TZ) Building Unified Token, FLOP & Compute Cost Ledger..."
    python3 -B "$STAGE/code/token_compute_ledger_h450.py" \
      --h448-readout "$EV448/results/H448_DEV91.json" \
      --h450-readout "$EV/diag_results/H450_DEV91.json" \
      --h450-banks-manifest "$EV/stage_v/banks_manifest_f64.json" \
      --non-d91-manifest-l "$EV/inputs/non_d91_manifest_L.json" \
      --non-d91-manifest-m "$EV/inputs/non_d91_manifest_M.json" \
      --h450-manifest-la "$EV/pulled/manifest_L_seedA.json" \
      --h450-manifest-ma "$EV/pulled/manifest_M_seedA.json" \
      --h450-manifest-lb "$EV/pulled/manifest_L_seedB.json" \
      --h450-manifest-mb "$EV/pulled/manifest_M_seedB.json" \
      --out-json "$EV/diag_results/TOKEN_AND_COMPUTE_COST_LEDGER.json" \
      --out-md "$EV/diag_results/TOKEN_AND_COMPUTE_COST_LEDGER.md" ;;

  *)
    die "unknown step $STEP" ;;
esac
