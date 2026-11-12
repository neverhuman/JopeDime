#!/bin/bash
# H449 quarantined NON-DEV91-COMPLETE vs ENTIRE (+100% dev91) orchestration (`run_h449.sh`).
# STRICTLY GATED by `gates_h449_prelock.sh`: refuses prelock/unlock unless H448's official FAIR dev91 score and S3
# archive are published & GET-verified on origin/v1/next, and requires separate receipt H449_UNFAIR_LOCK_sha256.txt.
# Writes ZERO files to canonical checkpoints or canonical reference-board.
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H449_STAGE:?H449_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h449_unfair_all_d91_diagnostic
G447=research/flywheel_selfplay/h447_legal_icl_104m_24m_r4e8
BRD=research/elo/reference-board/boards/dev91-audit-v1
EV=/home/ubuntu/jope_plan/evidence/H449
EV447=/home/ubuntu/jope_plan/evidence/H447
EV448=/home/ubuntu/jope_plan/evidence/H448
P418=/home/ubuntu/jope_plan/evidence/H418/pulled
P448=/home/ubuntu/jope_plan/evidence/H448/pulled
DATA=/home/ubuntu/jope-data/h449-unfair-d91
NON_D91_ROOT=/home/ubuntu/jope-data/h449-non-d91
SV=/home/ubuntu/h335-stage-v
PREP=x0_prep_20261105T005754Z
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H449_UNFAIR_LOCK_sha256.txt
H449_FILES=(d91_h449.py sandbox_h449_unfair_d91.sh gates_h449_prelock.sh build_non_d91_complete_bank.py
            train_unfair_icl_h449.py analyze_h449_unfair.py token_compute_ledger.py
            selftest_h449.py run_h449.sh stage_h449.sh)
GH=/home/ubuntu/.local/bin/gh-role
CRED=(-c credential.helper= -c "credential.helper=!$GH writer auth git-credential")
URL=https://github.com/neverhuman/JopeDime.git
SSH_OPTS=(-i /home/ubuntu/.ssh/xbabe2_ed25519 -o BatchMode=yes)
x0() { ssh "${SSH_OPTS[@]}" xbabe0 "$@"; }
rs0() { rsync -a -e "ssh ${SSH_OPTS[*]}" "$@"; }
die() { echo "run_h449 refused: $*" >&2; exit 3; }

[ "$(hostname)" = xbabe2 ] || die "run on xbabe2"
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "staged code differs from CODE.sha256"
MODE=$(cat "$STAGE/mode")
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$EV/logs" "$EV/diag_results" "$EV/inputs"

case $STEP in
  selftest|import-check) ;;
  *)
    [ "$MODE" = pub ] || die "$STAGE is not a pub stage"
    git -C "$REPO" "${CRED[@]}" fetch -q "$URL" +refs/heads/v1/next:refs/remotes/origin/v1/next
    bash "$STAGE/code/gates_h449_prelock.sh"
    cmp -s "$STAGE/code/PREREG_H449.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
      || die "staged PREREG differs from origin/v1/next"
    cmp -s "$STAGE/code/FROZEN_FAIR_CHAMPION.json" <(git -C "$REPO" show "origin/v1/next:$G447/FROZEN_FAIR_CHAMPION.json") \
      || die "staged FROZEN_FAIR_CHAMPION.json differs from origin/v1/next"
    for f in "${H449_FILES[@]}"; do
      cmp -s "$STAGE/code/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") || die "$f differs from origin/v1/next"
    done ;;
esac

sandbox() {
  local m=$1 c=$2 mem=$3; shift 3
  x0 "H449_STAGE=$STAGE H449_CPUSET=$c H449_MEMORY=$mem bash $STAGE/code/sandbox_h449_unfair_d91.sh $m $*"
}
prereg_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$WDIR/PREREG.md"; }
h448_close_commit() { git -C "$REPO" log -1 --format=%H origin/v1/next -- "$BRD/scores/h448.APEX_FAIR_CHAMPION.jsonl"; }

case $STEP in
  selftest)
    DIR=${1:?scratch dir containing /h449 or /H449/}
    [ ! -e "$DIR" ] || die "$DIR exists"
    nice -n 19 python3 -B "$STAGE/code/selftest_h449.py" --dir "$DIR" | tee "$EV/logs/selftest.txt" ;;

  import-check)
    sandbox prelock 16-17 4g python3 -u d91_h449.py --help | head -n 5 ;;

  inputs)
    PC=$(prereg_commit); HC=$(h448_close_commit)
    [ ${#PC} = 40 ] && [ ${#HC} = 40 ] || die "missing PREREG or H448 closeout commit on origin/v1/next"
    NON_BANK=$NON_D91_ROOT/banks_v2/all_non_d91_exhaustive_bank.npz
    COUSIN_BANK=$NON_D91_ROOT/banks_v2/d91_cousin_s3_bank.npz
    NON_CK_L=$NON_D91_ROOT/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt
    NON_CK_M=$NON_D91_ROOT/ckpts_v2/all_non_d91_M/ckpt_unfair_final.pt
    [ -s "$NON_BANK" ] && [ -s "$COUSIN_BANK" ] && [ -s "$NON_CK_L" ] && [ -s "$NON_CK_M" ] \
      || die "missing GPU-trained exhaustive NON-DEV91 bank/cousins or checkpoints under $NON_D91_ROOT"

    B=$EV/board_post_h448/dev91-audit-v1
    [ ! -e "$B" ] || die "$B exists (create-once)"
    x0 "[ ! -e $DATA/h335-prep ] && [ -z \"\$(ls -A $DATA/inputs 2>/dev/null || true)\" ]" || die "xbabe0 inputs exist (create-once)"

    x0 "set -euo pipefail; mkdir -p $DATA/h335-prep $DATA/inputs
      for k in tables root; do cp $SV/$PREP.\$k.sha256 $DATA/h335-prep/; chmod a-w $DATA/h335-prep/$PREP.\$k.sha256; done"

    cp "$P418/D1.sha256" "$EV/inputs/d1_pulled.sha256"
    cp "$P418/D2.sha256" "$EV/inputs/d2_pulled.sha256"
    cp "$NON_D91_ROOT/ckpts_v2/all_non_d91_L/unfair_manifest.json" "$EV/inputs/non_d91_manifest_L.json"
    cp "$NON_D91_ROOT/ckpts_v2/all_non_d91_M/unfair_manifest.json" "$EV/inputs/non_d91_manifest_M.json"
    rs0 "$EV/inputs/d1_pulled.sha256" "$EV/inputs/d2_pulled.sha256" "xbabe0:$DATA/inputs/"

    mkdir -p "$B/scores"
    git -C "$REPO" show "$HC:$BRD/BOARD.json" > "$B/BOARD.json"
    git -C "$REPO" show "$HC:$BRD/ladder.json" > "$B/ladder.json"
    git -C "$REPO" ls-tree --name-only "$HC:$BRD/scores/" | while read -r s; do
      git -C "$REPO" show "$HC:$BRD/scores/$s" > "$B/scores/$s"
    done
    [ -s "$B/scores/h448.APEX_FAIR_CHAMPION.jsonl" ] || die "post-H448 board lacks h448.APEX_FAIR_CHAMPION.jsonl"
    (cd "$B" && find . -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$EV/inputs/board_post_h448.sha256"
    chmod -R a-w "$B" "$EV/inputs/board_post_h448.sha256"
    rs0 "$EV/inputs/board_post_h448.sha256" "xbabe0:$DATA/inputs/board_post_h448.sha256"

    rs0 "$NON_BANK" "xbabe0:$DATA/inputs/all_non_d91_complete_bank.npz"
    rs0 "$COUSIN_BANK" "xbabe0:$DATA/inputs/d91_cousin_s3_bank.npz"
    rs0 "$NON_CK_L" "xbabe0:$DATA/inputs/ckpt_non_d91_L.pt"
    rs0 "$NON_CK_M" "xbabe0:$DATA/inputs/ckpt_non_d91_M.pt"
    rs0 /home/ubuntu/jope-data/h447/inputs/ICL_L_g1_w2_ckpt_h030.pt \
        /home/ubuntu/jope-data/h447/inputs/ICL_M_r3_ckpt_h040.pt "xbabe0:$DATA/inputs/"
    x0 "set -euo pipefail; cd $DATA/inputs
      ls -1 all_non_d91_complete_bank.npz d91_cousin_s3_bank.npz ckpt_non_d91_L.pt ckpt_non_d91_M.pt ICL_*.pt | sort | xargs sha256sum > unfair_inputs.sha256
      chmod 0444 all_non_d91_complete_bank.npz d91_cousin_s3_bank.npz ckpt_non_d91_L.pt ckpt_non_d91_M.pt ICL_*.pt unfair_inputs.sha256"
    rs0 "xbabe0:$DATA/inputs/unfair_inputs.sha256" "$EV/inputs/unfair_inputs.sha256"

    jq -n --arg p "$PC" --arg s "$(cat "$STAGE/source_commit")" --arg h "$HC" \
      '{prereg_commit: $p, stage_commit: $s, h448_closeout_commit: $h}' > "$EV/inputs/spec.json"
    rs0 "$EV/inputs/spec.json" "xbabe0:$DATA/inputs/spec.json"
    x0 "chmod a-w $DATA/inputs/*"
    echo "H449 inputs OK" | tee -a "$EV/logs/inputs.log" ;;

  prelock)
    [ ! -e "$RECEIPT" ] || die "$RECEIPT already exists"
    sandbox prelock 16-23 8g python3 -u d91_h449.py lock --spec "$DATA/inputs/spec.json" | tee "$EV/logs/prelock.log"
    S=$(x0 "sha256sum $DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    mkdir -p "$EV/stage_v"
    rs0 "xbabe0:$DATA/stage_v/LOCK.json" "$EV/stage_v/LOCK.json"
    chmod 0444 "$EV/stage_v/LOCK.json"
    echo "H449_UNFAIR_PRELOCK_READY sha256=$S (awaiting separate owner receipt $RECEIPT)" | tee -a "$EV/logs/prelock.log" ;;

  unlock)
    [ -s "$RECEIPT" ] || die "missing separate owner receipt $RECEIPT"
    S_LOCK=$(sha256sum "$EV/stage_v/LOCK.json" | cut -d' ' -f1)
    [ "$(awk '{print $1}' "$RECEIPT")" = "$S_LOCK" ] || die "receipt != LOCK.json"
    if ! x0 "[ -s $RECEIPT ]"; then
      R_CONTENT=$(cat "$RECEIPT")
      x0 "mkdir -p $(dirname "$RECEIPT") && set -o noclobber && printf '%s\n' '$R_CONTENT' > $RECEIPT && chmod 0444 $RECEIPT"
    fi
    sandbox unlock 16-23 8g python3 -u d91_h449.py unlock | tee -a "$EV/logs/unlock.log"
    sandbox run 16-23 24g python3 -u d91_h449.py build_banks | tee "$EV/logs/build_banks.log"
    rs0 "xbabe0:$DATA/cids91.json" "xbabe0:$DATA/banks/banks_manifest.json" "xbabe0:$DATA/stage_v/STAGE_V_STARTED" \
      "$EV/stage_v/"
    chmod a-w "$EV"/stage_v/* ;;

  train-and-cols-x0)
    HOURS=${1:-0.28}
    X0_CPUS="16-39,41-63,80-127"
    LINE="$(date -u +%FT%TZ) agy_apex_w10: H449 quarantined diagnostic taking xbabe0 cpus $X0_CPUS (docker --network none, nice 19) for ~0.45h: ALL-NON-DEV91 R4-e8 inference + ENTIRE (+100% d91) 104M+24M fine-tuning + R4-e8 inference. Will post release when done."
    printf '%s\n' "$LINE" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
    SHA_ALL=$(jq -r .everything_all_d91.sha256 "$EV/stage_v/banks_manifest.json")
    SHA_NON_L=$(jq -r .ckpt_sha256 "$EV/inputs/non_d91_manifest_L.json")
    SHA_NON_M=$(jq -r .ckpt_sha256 "$EV/inputs/non_d91_manifest_M.json")

    sandbox run "$X0_CPUS" 110g bash -c "'
      set -euo pipefail
      mkdir -p $DATA/logs $DATA/cols_fair_all_non_d91_D91 $DATA/cols_unfair_all_D91
      ICL_CPUS=( \$(seq 16 39) \$(seq 41 63) \$(seq 80 123) )

      echo \"\$(date -u +%FT%TZ) [1/3] Running cols_r4_e8.py (48 LPT workers, 110g safe) for GPU-trained ALL-NON-DEV91 (0% d91)\"
      rc=0
      for w in \$(seq 0 47); do
        c=\${ICL_CPUS[\$w]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_r4_e8.py run \
          --bank $DATA/bank_d91_search.npz \
          --ckpt-l $DATA/inputs/ckpt_non_d91_L.pt --sha-l $SHA_NON_L --size-l L \
          --ckpt-m $DATA/inputs/ckpt_non_d91_M.pt --sha-m $SHA_NON_M --size-m M \
          --out $DATA/cols_fair_all_non_d91_D91 --device cpu --threads 1 \
          --queue --worker-id \"\$w\" --ctx-s 3600 > $DATA/logs/icl_non_d91_q\$w.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      [ \"\$rc\" = 0 ]

      echo \"\$(date -u +%FT%TZ) [2/3] Fine-tuning 104M (16-39,41-63) & 24M (80-127) on everything_all_d91_bank.npz (init=ckpt_non_d91)\"
      OMP_NUM_THREADS=47 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        taskset -c 16-39,41-63 nice -n 19 python3 -B train_unfair_icl_h449.py train \
        --size L --init $DATA/inputs/ckpt_non_d91_L.pt --init-sha256 $SHA_NON_L \
        --real-bank $DATA/banks/everything_all_d91_bank.npz --bank-sha256 $SHA_ALL \
        --out $DATA/ckpts/all_d91_L --device cpu --threads 47 --precache-workers 24 \
        --d91-folds 6 --precache-other 256 --struct-frac 0.65 --ema-decay 0.985 --bmax 4 --struct-bmax 2 --tokens 120000 \
        --lr 2.5e-5 --hours $HOURS --seed 44911 > $DATA/logs/train_L.log 2>&1 &
      PID_L=\$!

      OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        taskset -c 80-127 nice -n 19 python3 -B train_unfair_icl_h449.py train \
        --size M --init $DATA/inputs/ckpt_non_d91_M.pt --init-sha256 $SHA_NON_M \
        --real-bank $DATA/banks/everything_all_d91_bank.npz --bank-sha256 $SHA_ALL \
        --out $DATA/ckpts/all_d91_M --device cpu --threads 48 --precache-workers 24 \
        --d91-folds 6 --precache-other 256 --struct-frac 0.65 --ema-decay 0.985 --bmax 4 --struct-bmax 2 --tokens 120000 \
        --lr 3.0e-5 --hours $HOURS --seed 44912 > $DATA/logs/train_M.log 2>&1 &
      PID_M=\$!

      wait \$PID_L || rc=\$?
      wait \$PID_M || rc=\$?
      [ \"\$rc\" = 0 ]

      NEW_SHA_L=\$(sha256sum $DATA/ckpts/all_d91_L/ckpt_unfair_final.pt | cut -d\" \" -f1)
      NEW_SHA_M=\$(sha256sum $DATA/ckpts/all_d91_M/ckpt_unfair_final.pt | cut -d\" \" -f1)

      echo \"\$(date -u +%FT%TZ) [3/3] Running cols_r4_e8.py (48 LPT workers, 110g safe) for ENTIRE (+100% d91)\"
      for w in \$(seq 0 47); do
        c=\${ICL_CPUS[\$w]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_r4_e8.py run \
          --bank $DATA/bank_d91_search.npz \
          --ckpt-l $DATA/ckpts/all_d91_L/ckpt_unfair_final.pt --sha-l \"\$NEW_SHA_L\" --size-l L \
          --ckpt-m $DATA/ckpts/all_d91_M/ckpt_unfair_final.pt --sha-m \"\$NEW_SHA_M\" --size-m M \
          --out $DATA/cols_unfair_all_D91 --device cpu --threads 1 \
          --queue --worker-id \"\$w\" --ctx-s 3600 > $DATA/logs/icl_unfair_q\$w.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      [ \"\$rc\" = 0 ]
      echo \"\$(date -u +%FT%TZ) H449 NON-DEV91 COLS + ENTIRE TRAIN + ENTIRE COLS DONE\"
    '" | tee -a "$EV/logs/train_and_cols.log"

    REL="$(date -u +%FT%TZ) agy_apex_w10: H449 xbabe0 run complete; releasing xbabe0 cpus $X0_CPUS."
    printf '%s\n' "$REL" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null) ;;

  pull-and-analyze)
    P=$EV/pulled
    [ ! -e "$P" ] || die "$P exists (create-once)"
    mkdir -p "$P"
    rs0 "xbabe0:$DATA/bank_d91_search.npz" \
        "xbabe0:$DATA/cols_fair_all_non_d91_D91" \
        "xbabe0:$DATA/cols_unfair_all_D91" "$P/"
    rs0 "xbabe0:$DATA/ckpts/all_d91_L/unfair_manifest.json" "$P/unfair_manifest_L.json"
    rs0 "xbabe0:$DATA/ckpts/all_d91_M/unfair_manifest.json" "$P/unfair_manifest_M.json"
    rs0 "xbabe0:$DATA/ckpts/all_d91_L/log.jsonl" "$P/log_L.jsonl"
    rs0 "xbabe0:$DATA/ckpts/all_d91_M/log.jsonl" "$P/log_M.jsonl"
    chmod -R a-w "$P"
    nice -n 19 python3 -B "$STAGE/code/analyze_h449_unfair.py" \
      --pulled "$P" --h448-pulled "$P448" --h418-pulled "$P418" \
      --board "$EV/board_post_h448/dev91-audit-v1" \
      --lock "$EV/stage_v/LOCK.json" --receipt "$RECEIPT" \
      --procs 16 \
      --diag-out "$EV/diag_results/H449_UNFAIR_DEV91.json" \
      --diag-board "$EV/diag_board/dev91-audit-v1" \
      | tee "$EV/diag_results/H449_UNFAIR_DEV91.txt"
    chmod a-w "$EV/diag_results/H449_UNFAIR_DEV91.json" "$EV/diag_results/H449_UNFAIR_DEV91.txt"

    ST447=$(cat "$EV447/logs/pub_stage" 2>/dev/null || echo "")
    R447="/home/ubuntu/jope-data/h335-cl/h447/$(basename "${ST447:-pub-r3}")"
    WC_ARGS=()
    if [ -f "$R447/wc_L/wc_manifest.json" ] && [ -f "$R447/wc_M/wc_manifest.json" ]; then
      WC_ARGS=(--h447-wc-l "$R447/wc_L/wc_manifest.json" --h447-wc-m "$R447/wc_M/wc_manifest.json")
    fi
    nice -n 19 python3 -B "$STAGE/code/token_compute_ledger.py" \
      --h447-frozen "$STAGE/code/FROZEN_FAIR_CHAMPION.json" \
      "${WC_ARGS[@]}" \
      --h448-readout "$EV448/results/H448_DEV91.json" \
      --h449-readout "$EV/diag_results/H449_UNFAIR_DEV91.json" \
      --h449-banks-manifest "$EV/stage_v/banks_manifest.json" \
      --non-d91-manifest-l "$EV/inputs/non_d91_manifest_L.json" \
      --non-d91-manifest-m "$EV/inputs/non_d91_manifest_M.json" \
      --all-d91-manifest-l "$P/unfair_manifest_L.json" \
      --all-d91-manifest-m "$P/unfair_manifest_M.json" \
      --bank-d91-search "$P/bank_d91_search.npz" \
      --out-json "$EV/diag_results/TOKEN_AND_COMPUTE_COST_LEDGER.json" \
      --out-md "$EV/diag_results/TOKEN_AND_COMPUTE_COST_LEDGER.md" ;;

  *) die "unknown step $STEP" ;;
esac
