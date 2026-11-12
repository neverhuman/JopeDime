#!/bin/bash
# H450 pub-r4 (cl_fwriter): maximum-entropy random-split fine-tune of the H449 best checkpoints on every non-sealed
# regression table (dev91 included), with per-epoch dev91 in-corpus Elo tracking.
#
#   run_r4_h450.sh stage COMMIT TAG                      ~/h450-stage/TAG = pub-r3 code + this folder at COMMIT
#   run_r4_h450.sh smoke-cpu                             2000-item CPU sampler smoke (cpus 64-71) -> smoke_cl/
#   run_r4_h450.sh train HOST SIZE RUN HOURS LR SEED [EXTRA_ARGS]   HOST xbabe1|xbabe2|xbabe3; MemoryMax $H450_MEMMAX (20G)
#                                                        one training process as a user unit on HOST (xbabe2|xbabe3)
#   run_r4_h450.sh cols-epoch EPOCH RUN_L RUN_M [RUN_LB RUN_MB]
#                                                        dev91 GPU columns (fit X_search only) from epoch checkpoints
#   run_r4_h450.sh analyze-epoch TAG COLS_DIR [ZERO_DIR] in-corpus dev91 Elo (y_audit read; H450 lock + receipt)
# Quarantine: every output under ~/jope-data/h450-quarantine/r4 or ~/jope_plan/evidence/H450/diag_results/r4_*;
# canonical_promotion_forbidden; APEX-CONFIRM and DL-Holdout are not in the bank (trainer gate refuses them).
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
REPO=/home/ubuntu/JopeDime
GROUP=research/flywheel_selfplay/h450_aggressive_random_all_elo1
ST=$(dirname "$(dirname "$(readlink -f "$0")")")   # the stage this script runs from (stage COMMIT TAG creates a new one)
BASE_STAGE=/home/ubuntu/h450-stage/pub-r3
DATA=/home/ubuntu/jope-data/h450-quarantine
R4=$DATA/r4
EV=/home/ubuntu/jope_plan/evidence/H450
BANK=$DATA/banks/everything_combined_f64_h450_bank.npz
BANK_SHA=9dfc60ed29a72266bf69228addcfce5e946064905edde79115914dee6eb0c507
EVAL_BANK=$DATA/bank_d91_search_f64.npz
CK_L=/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt
CK_L_SHA=b854b518b07b2c243bc62069b36e9cbd05b0a5cd77ca5faba60b5347623a3a29
CK_M=/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_M/ckpt_unfair_final.pt
CK_M_SHA=5aadefdfec053ac7a955381194bfb233d0ce8341f413b4d0d89699140715d4a0
X3=xbabe3
X3_ROOT=/home/ubuntu/jope-data/h450-r4-x3
log() { echo "$(date -u +%FT%TZ) $*"; }
die() { echo "$(date -u +%FT%TZ) ERROR: $*" >&2; exit 1; }
ssh3() { ssh -o BatchMode=yes "$X3" "$@"; }
rs3() { rsync -a -e "ssh -o BatchMode=yes" "$@"; }

check_stage() {
  [ "$(cat "$ST/mode")" = pub ] || die "$ST is not a pub stage"
  (cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256) || die "stage CODE.sha256 does not verify"
}

CMD=${1:?stage|smoke-cpu|train|cols-epoch|analyze-epoch}
case $CMD in
stage)
  C=${2:?COMMIT}; TAG=${3:?TAG e.g. pub-r5}
  ST=/home/ubuntu/h450-stage/$TAG
  [ ! -e "$ST" ] || die "$ST exists (create-once)"
  git -C "$REPO" merge-base --is-ancestor "$C" origin/v1/next || die "$C is not on origin/v1/next"
  (cd "$BASE_STAGE/code" && sha256sum --quiet -c ../CODE.sha256) || die "pub-r3 stage does not verify"
  mkdir -p "$ST/code"
  cp -a "$BASE_STAGE/code/." "$ST/code/"
  chmod -R u+w "$ST/code"
  for f in $(git -C "$REPO" ls-tree --name-only "$C" "$GROUP/"); do
    b=$(basename "$f"); dst=$b; [ "$b" = PREREG.md ] && dst=PREREG_H450.md
    git -C "$REPO" show "$C:$f" > "$ST/code/$dst"
    [ -x "$ST/code/$dst" ] || case $b in *.sh) chmod 0755 "$ST/code/$dst" ;; esac
  done
  rm -rf "$ST/code/__pycache__"
  echo "$C" > "$ST/source_commit"; echo pub > "$ST/mode"
  (cd "$ST/code" && find . -maxdepth 1 -type f | sort | sed 's#^\./##' | xargs sha256sum) > "$ST/CODE.sha256"
  chmod -R a-w "$ST"
  log "staged $ST from $C: $(wc -l < "$ST/CODE.sha256") files, CODE.sha256 $(sha256sum "$ST/CODE.sha256" | cut -c1-16)"
  ;;
smoke-cpu)
  check_stage
  mkdir -p "$R4/smoke"
  out=$R4/smoke/smoke_2000_$(date -u +%Y%m%dT%H%MZ).json
  (cd "$ST/code" && CUDA_VISIBLE_DEVICES="" taskset -c 64-71 nice -n 19 python3 -B smoke_sampler_h450.py \
     --bank "$BANK" --bank-sha256 "$BANK_SHA" --items 2000 --workers 7 --out "$out")
  log "smoke written: $out"
  ;;
train)
  HOST=${2:?xbabe2|xbabe3}; SIZE=${3:?L|M}; RUN=${4:?RUN}; HOURS=${5:?HOURS}; LR=${6:?LR}; SEED=${7:?SEED}; RES=${8:-}
  check_stage
  case $SIZE in L) CK=$CK_L; CKS=$CK_L_SHA ;; M) CK=$CK_M; CKS=$CK_M_SHA ;; *) die "SIZE L|M" ;; esac
  [[ $RUN =~ ^[A-Za-z0-9_]+$ ]] || die "bad RUN"
  ARGS="--size $SIZE --init INIT --init-sha256 $CKS --bank BANK --bank-sha256 $BANK_SHA --out OUT --hours $HOURS --lr $LR --seed $SEED --epoch-hours 1.0 --workers 8 $RES"
  UNIT=h450-r4-$RUN
  if [ "$HOST" = xbabe2 ]; then
    OUT=$R4/runs/$RUN; mkdir -p "$OUT"
    A=${ARGS//INIT/$CK}; A=${A//BANK/$BANK}; A=${A//OUT/$OUT}
    systemd-run --user --unit="$UNIT" -p MemoryMax="${H450_MEMMAX:-20G}" -p WorkingDirectory="$ST/code" -p StandardOutput=append:"$OUT/stdout.log" \
      -p StandardError=append:"$OUT/stdout.log" --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --setenv=PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B "$ST/code/train_aggressive_random_h450.py" $A
  else
    case $HOST in   # pub-r9: any GPU host; each keeps its own root
      xbabe3) X3=xbabe3; X3_ROOT=/home/ubuntu/jope-data/h450-r4-x3 ;;
      xbabe1) X3=xbabe1; X3_ROOT=/home/ubuntu/jope-data/h450-r4-x1 ;;
      *) die "HOST xbabe1|xbabe2|xbabe3" ;;
    esac
    ssh3 "mkdir -p $X3_ROOT/banks $X3_ROOT/inputs $X3_ROOT/runs $X3_ROOT/stage"
    rs3 "$ST/" "$X3:$X3_ROOT/stage/$(basename "$ST")/"
    rs3 "$BANK" "$X3:$X3_ROOT/banks/"
    rs3 "$CK" "$X3:$X3_ROOT/inputs/init_$SIZE.pt"
    ssh3 "cd $X3_ROOT/stage/$(basename "$ST")/code && sha256sum --quiet -c ../CODE.sha256 && echo '$BANK_SHA  $X3_ROOT/banks/$(basename "$BANK")' | sha256sum --quiet -c - && echo '$CKS  $X3_ROOT/inputs/init_$SIZE.pt' | sha256sum --quiet -c -" || die "$X3 copies do not verify"
    OUT=$X3_ROOT/runs/$RUN
    A=${ARGS//INIT/$X3_ROOT/inputs/init_$SIZE.pt}; A=${A//BANK/$X3_ROOT/banks/$(basename "$BANK")}; A=${A//OUT/$OUT}
    ssh3 "mkdir -p $OUT && systemd-run --user --unit=$UNIT -p MemoryMax=${H450_MEMMAX:-20G} -p WorkingDirectory=$X3_ROOT/stage/$(basename "$ST")/code -p StandardOutput=append:$OUT/stdout.log -p StandardError=append:$OUT/stdout.log --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --setenv=PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B $X3_ROOT/stage/$(basename "$ST")/code/train_aggressive_random_h450.py $A"
  fi
  log "launched $UNIT on $HOST ($SIZE, $HOURS h, lr $LR, seed $SEED) -> $OUT"
  ;;
cols-epoch)
  EP=${2:?EPOCH}; RL=${3:?RUN_L}; RM=${4:?RUN_M}; RLB=${5:-}; RMB=${6:-}
  check_stage
  e=$(printf '%02d' "$EP")
  ck() {  # RUN -> local path of its epoch checkpoint (copied from xbabe3 when the run lives there)
    local r=$1 f="$R4/runs/$1/ckpt_epoch_$e.pt"
    if [ ! -f "$f" ]; then
      mkdir -p "$R4/runs/$r"
      rs3 "$X3:$X3_ROOT/runs/$r/ckpt_epoch_$e.pt" "$X3:$X3_ROOT/runs/$r/epoch_$e.json" "$R4/runs/$r/" >&2
    fi
    [ -f "$f" ] || die "missing $f"
    echo "$f"
  }
  CA=(--ckpt-la "$(ck "$RL")" --ckpt-ma "$(ck "$RM")")
  [ -z "$RLB" ] || CA+=(--ckpt-lb "$(ck "$RLB")" --ckpt-mb "$(ck "$RMB")")
  TAG=ep${e}_${RL}_${RM}${RLB:+_$RLB}${RMB:+_$RMB}
  OUTC=$R4/cols/$TAG
  [ ! -e "$OUTC" ] || die "$OUTC exists (create-once)"
  mkdir -p "$OUTC"
  (cd "$ST/code" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 python3 -B cols_h450_gpu.py --bank "$EVAL_BANK" \
     "${CA[@]}" --out "$OUTC" --device cuda) > "$R4/cols/$TAG.log" 2>&1
  n=$(find "$OUTC" -maxdepth 1 -name '[0-9a-f]*.npz' ! -name '.*' | wc -l)
  log "cols $TAG: $n/91 contexts"
  [ "$n" = 91 ] || die "incomplete cols $TAG"
  ;;
analyze-epoch)
  TAG=${2:?TAG}; COLS=${3:?COLS_DIR}; ZERO=${4:-}   # ZERO: the epoch-0 (init, 0 % dev91) column dir, as the control arm
  check_stage
  D=$EV/diag_results/r4_$TAG
  [ ! -e "$D" ] || die "$D exists (create-once)"
  [ -z "$ZERO" ] || [ -d "$ZERO" ] || die "missing $ZERO"
  mkdir -p "$D"
  (cd "$ST/code" && CUDA_VISIBLE_DEVICES="" taskset -c 64-79 nice -n 19 python3 -B analyze_h450.py --bank "$EVAL_BANK" \
     ${ZERO:+--zero-d91-gpu-cols "$ZERO"} \
     --gpu-cols "$COLS" --tab-cols "$DATA/cols_tab_h450_D91" --h448-pulled /home/ubuntu/jope_plan/evidence/H448/pulled \
     --h418-pulled /home/ubuntu/jope_plan/evidence/H418/pulled --board "$EV/board_post_h448/dev91-audit-v1" \
     --lock "$EV/stage_v/LOCK.json" --receipt /home/ubuntu/jope_plan/ns/receipts/H450_UNFAIR_LOCK_sha256.txt \
     --diag-out "$D/H450_DEV91.json" --diag-txt "$D/H450_DEV91.txt" --diag-board "$D/board" --procs 16) > "$D/analyze.log" 2>&1
  log "analyze $TAG done: $D/H450_DEV91.json"
  ;;
*) die "unknown command $CMD" ;;
esac
