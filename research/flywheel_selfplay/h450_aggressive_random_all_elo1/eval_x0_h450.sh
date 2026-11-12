#!/bin/bash
# H450 pub-r6 (cl_fwriter): in-corpus dev91 Elo of one checkpoint set, on the original dev91 split.
#
# The dev91 column step (cols_h450_gpu.py --device cpu) runs on xbabe0's CPUs. xbabe0's GPU is in error, and its CPUs
# are otherwise idle, so every checkpoint is scored on the same host and backend and the training GPUs are left alone.
# cols_h450_gpu.py fits on X_search only and refuses a bank that contains audit targets. analyze_h450.py then reads
# y_audit on xbabe2, through run_r4_h450.sh analyze-epoch, with the epoch-0 columns as the zero-dev91 control arm.
#
#   eval_x0_h450.sh baseline                                epoch 0 = the H449 init L and M checkpoints (pair A = pair B)
#   eval_x0_h450.sh epoch EPOCH RUN_L RUN_M [RUN_LB RUN_MB] ckpt_epoch_NN.pt of those runs (local or on xbabe3)
#
# Environment: H450_EVAL_SHARDS (default 26) shards x H450_EVAL_THREADS (default 4) threads. 104 of xbabe0's 128 cpus
# stay under the owner's 85 % CPU cap.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
ST=$(dirname "$(dirname "$(readlink -f "$0")")")
RUN=$ST/code/run_r4_h450.sh
DATA=/home/ubuntu/jope-data/h450-quarantine
R4=$DATA/r4
EVAL_BANK=$DATA/bank_d91_search_f64.npz
EVAL_SHA=0bce958572faee0ddae494df6d75c837e1c14936e34e565f042ef13aa727d955
CK_L=/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_L/ckpt_unfair_final.pt
CK_M=/home/ubuntu/jope-data/h449-non-d91/ckpts_v2/all_non_d91_M/ckpt_unfair_final.pt
X0=xbabe0
X0R=$DATA/r4-eval
X3=xbabe3
X3_ROOT=/home/ubuntu/jope-data/h450-r4-x3
SHARDS=${H450_EVAL_SHARDS:-26}
THREADS=${H450_EVAL_THREADS:-4}
log() { echo "$(date -u +%FT%TZ) $*"; }
die() { echo "$(date -u +%FT%TZ) ERROR: $*" >&2; exit 1; }
ssh0() { ssh -o BatchMode=yes "$X0" "$@"; }
rs() { rsync -a -e "ssh -o BatchMode=yes" "$@"; }
[ "$(cat "$ST/mode")" = pub ] || die "$ST is not a pub stage"
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256) || die "stage CODE.sha256 does not verify"
[ "$(sha256sum "$EVAL_BANK" | cut -c1-64)" = "$EVAL_SHA" ] || die "local eval bank sha differs"

ROLES=(); CKS=()
case ${1:?baseline|epoch} in
baseline)
  TAG=ep00_init
  ROLES=(la ma); CKS=("$CK_L" "$CK_M")
  ;;
epoch)
  e=$(printf '%02d' "${2:?EPOCH}"); RL=${3:?RUN_L}; RM=${4:?RUN_M}; RLB=${5:-}; RMB=${6:-}
  ck() {  # RUN -> local path of its epoch checkpoint (copied from xbabe3 when the run lives there)
    local r=$1 f="$R4/runs/$1/ckpt_epoch_$e.pt"
    if [ ! -f "$f" ]; then
      mkdir -p "$R4/runs/$r"
      rs "$X3:$X3_ROOT/runs/$r/ckpt_epoch_$e.pt" "$X3:$X3_ROOT/runs/$r/epoch_$e.json" "$R4/runs/$r/" >&2
    fi
    [ -f "$f" ] || die "missing $f"
    echo "$f"
  }
  ROLES=(la ma); CKS=("$(ck "$RL")" "$(ck "$RM")")
  if [ -n "$RLB" ]; then ROLES+=(lb mb); CKS+=("$(ck "$RLB")" "$(ck "$RMB")"); fi
  TAG=ep${e}_${RL}_${RM}${RLB:+_$RLB}${RMB:+_$RMB}
  ;;
*) die "unknown command $1" ;;
esac

LOC=$R4/cols/$TAG
[ ! -e "$LOC" ] || die "$LOC exists (create-once)"
SN=$(basename "$ST")
ssh0 "mkdir -p $X0R/stage $X0R/ckpts/$TAG $X0R/cols $X0R/logs"
ssh0 "[ ! -e $X0R/cols/$TAG ]" || die "xbabe0 $X0R/cols/$TAG exists (create-once)"
rs "$ST/" "$X0:$X0R/stage/$SN/"
ssh0 "cd $X0R/stage/$SN/code && sha256sum --quiet -c ../CODE.sha256 && echo '$EVAL_SHA  $EVAL_BANK' | sha256sum --quiet -c -" \
  || die "xbabe0 stage or eval bank does not verify"
CA=()
: > "$R4/cols/$TAG.ckpts.sha256.part"
for i in "${!ROLES[@]}"; do
  r=${ROLES[$i]}; f=${CKS[$i]}
  s=$(sha256sum "$f" | cut -c1-64)
  rs "$f" "$X0:$X0R/ckpts/$TAG/$r.pt"
  ssh0 "echo '$s  $X0R/ckpts/$TAG/$r.pt' | sha256sum --quiet -c -" || die "xbabe0 copy of $f does not verify"
  echo "$s  $r  $f" >> "$R4/cols/$TAG.ckpts.sha256.part"
  CA+=("--ckpt-$r" "$X0R/ckpts/$TAG/$r.pt")
done
mv "$R4/cols/$TAG.ckpts.sha256.part" "$R4/cols/$TAG.ckpts.sha256"
UNIT=h450-eval-$TAG
seq 0 $((SHARDS - 1)) | ssh0 "cat > $X0R/logs/$TAG.shards"
t0=$(date +%s)
log "cols $TAG on xbabe0: $SHARDS shards x $THREADS threads (${ROLES[*]})"
# xargs runs the shards; systemd-run gets no shell, so nothing in ExecStart is expanded twice.
ssh0 "systemd-run --user --unit=$UNIT -p WorkingDirectory=$X0R/stage/$SN/code -p StandardOutput=append:$X0R/logs/$TAG.log \
  -p StandardError=append:$X0R/logs/$TAG.log --setenv=OMP_NUM_THREADS=$THREADS --setenv=MKL_NUM_THREADS=$THREADS \
  --setenv=CUDA_VISIBLE_DEVICES= --setenv=PYTHONDONTWRITEBYTECODE=1 /usr/bin/xargs -a $X0R/logs/$TAG.shards -P $SHARDS -I@ \
  /usr/bin/nice -n 10 /usr/bin/python3 -B $X0R/stage/$SN/code/cols_h450_gpu.py --bank $EVAL_BANK ${CA[*]} \
  --out $X0R/cols/$TAG --device cpu --num-shards $SHARDS --shard-idx @" >/dev/null
while ssh0 "systemctl --user is-active -q $UNIT"; do sleep 20; done
res=$(ssh0 "find $X0R/cols/$TAG -maxdepth 1 -name 'shard_*_of_${SHARDS}.json' | wc -l")
n=$(ssh0 "find $X0R/cols/$TAG -maxdepth 1 -name '[0-9a-f]*.npz' | wc -l")
log "cols $TAG finished on xbabe0 in $(( $(date +%s) - t0 )) s: $n/91 contexts, $res/$SHARDS shard summaries"
[ "$n" = 91 ] && [ "$res" = "$SHARDS" ] || die "incomplete cols $TAG (see xbabe0:$X0R/logs/$TAG.log)"
mkdir -p "$LOC"
rs "$X0:$X0R/cols/$TAG/" "$LOC/"
[ "$(find "$LOC" -maxdepth 1 -name '[0-9a-f]*.npz' | wc -l)" = 91 ] || die "local copy of $TAG incomplete"
chmod -R a-w "$LOC"
ZERO=$R4/cols/ep00_init
if [ "$TAG" = ep00_init ]; then "$RUN" analyze-epoch "$TAG" "$LOC"; else "$RUN" analyze-epoch "$TAG" "$LOC" "$ZERO"; fi
D=/home/ubuntu/jope_plan/evidence/H450/diag_results/r4_$TAG
jq -c '{icl_only: .in_corpus_icl_boost_only_91.elo.elo, champion_stack: .in_corpus_random_split_champion_91.elo.elo,
        legacy_stack: .in_corpus_legacy_guard_stack_91.elo.elo,
        champion_vs_kumo_L: .in_corpus_random_split_champion_91.gap_elo["vs_kumo_tabular_large_e8.v1"],
        icl_vs_kumo_L: .in_corpus_icl_boost_only_91.gap_elo["vs_kumo_tabular_large_e8.v1"]}' "$D/H450_DEV91.json" \
  | tee "$D/SUMMARY.json"
