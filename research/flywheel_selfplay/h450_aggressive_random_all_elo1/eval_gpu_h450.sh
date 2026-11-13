#!/bin/bash
# H450 pub-r11 (cl_fwriter): in-corpus dev91 Elo of the 104M fine-tune, on the original dev91 split, on the xbabe2 + xbabe3 GPUs.
#
# Replaces eval_x0_h450.sh. On xbabe0's CPUs, cols_h450_gpu.py finished 0 of 91 contexts in 17 min: each context runs
# about 1,000 model forwards (10 folds x views x modes x 2 pairs).
#
# The owner asked for the 104M model, so the 24M (M) companion of every pair stays at its H449 init checkpoint.
# Every Elo change is then due to the 104M fine-tune alone:
#   pair A = (L_A at epoch N, M_init)   pair B = (L_B at epoch N, M_init)   epoch 0: L_init in both pairs
# cols_h450_gpu.py fits on X_search only and refuses a bank that contains audit targets. analyze_h450.py then reads
# y_audit through run_r4_h450.sh analyze-epoch, with the epoch-0 columns as the zero-dev91 control arm.
#
#   eval_gpu_h450.sh baseline
#   eval_gpu_h450.sh epoch EPOCH RUN_LA RUN_LB      runs on xbabe2 (local) or xbabe3 (copied first)
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
X3=xbabe3
X3_ROOT=/home/ubuntu/jope-data/h450-r4-x3
log() { echo "$(date -u +%FT%TZ) $*"; }
die() { echo "$(date -u +%FT%TZ) ERROR: $*" >&2; exit 1; }
[ "$(cat "$ST/mode")" = pub ] || die "$ST is not a pub stage"
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256) || die "stage CODE.sha256 does not verify"
[ "$(sha256sum "$EVAL_BANK" | cut -c1-64)" = "$EVAL_SHA" ] || die "eval bank sha differs"

case ${1:?baseline|epoch} in
baseline)
  TAG=g00_init; LA=$CK_L; LB=$CK_L
  ;;
epoch)
  e=$(printf '%02d' "${2:?EPOCH}"); RA=${3:?RUN_LA}; RB=${4:?RUN_LB}
  ck() {
    local r=$1 f="$R4/runs/$1/ckpt_epoch_$e.pt"
    if [ ! -f "$f" ]; then
      mkdir -p "$R4/runs/$r"
      rsync -a -e "ssh -o BatchMode=yes" "$X3:$X3_ROOT/runs/$r/ckpt_epoch_$e.pt" "$X3:$X3_ROOT/runs/$r/epoch_$e.json" "$R4/runs/$r/" >&2
    fi
    [ -f "$f" ] || die "missing $f"
    echo "$f"
  }
  LA=$(ck "$RA"); LB=$(ck "$RB")
  TAG=g${e}_${RA}_${RB}
  ;;
*) die "unknown command $1" ;;
esac

mkdir -p "$R4/cols"
OUTC=$R4/cols/$TAG
[ ! -e "$OUTC" ] || die "$OUTC exists (create-once)"
sha256sum "$LA" "$LB" "$CK_M" > "$R4/cols/$TAG.ckpts.sha256"
t0=$(date +%s)
# Contexts are computed on the xbabe2 and xbabe3 GPUs (see the pub-r11 note below).
# Both hosts run torch 2.7.0 and numpy 1.26.4.
SN=$(basename "$ST")
XE=$X3_ROOT/eval
ssh3() { ssh -o BatchMode=yes "$X3" "$@"; }
rs3() { rsync -a -e "ssh -o BatchMode=yes" "$@"; }
ssh3 "mkdir -p $XE/ckpts/$TAG $XE/cols $XE/logs && [ ! -e $XE/cols/$TAG ]" || die "xbabe3 $XE/cols/$TAG exists (create-once)"
rs3 "$ST/" "$X3:$X3_ROOT/stage/$SN/"
rs3 "$EVAL_BANK" "$X3:$XE/"
RX=()
for p in "la:$LA" "lb:$LB" "m:$CK_M"; do
  r=${p%%:*}; f=${p#*:}; s=$(sha256sum "$f" | cut -c1-64)
  rs3 "$f" "$X3:$XE/ckpts/$TAG/$r.pt"
  RX+=("$s  $XE/ckpts/$TAG/$r.pt")
done
printf '%s\n' "${RX[@]}" "$EVAL_SHA  $XE/$(basename "$EVAL_BANK")" | ssh3 "cd $X3_ROOT/stage/$SN/code && sha256sum --quiet -c ../CODE.sha256 && sha256sum --quiet -c -" \
  || die "xbabe3 stage, eval bank or checkpoint copies do not verify"
# pub-r11: 91 one-context tasks (--num-shards 91 --shard-idx I) handed to whichever 3090 is free. This is safe because
# both 3090s give byte-identical columns: baseline context 016648dc, recomputed on xbabe3, cmp-matched xbabe2's copy.
Q=$R4/cols/$TAG.queue
seq 0 90 > "$Q"
pop() { flock "$Q.lock" bash -c 'i=$(head -1 "$1"); [ -n "$i" ] && sed -i 1d "$1"; echo "$i"' _ "$Q"; }
worker() {  # local | xbabe3
  local w=$1 i
  while i=$(pop); [ -n "$i" ]; do
    if [ "$w" = local ]; then
      (cd "$ST/code" && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nice -n 10 python3 -B cols_h450_gpu.py --bank "$EVAL_BANK" \
         --ckpt-la "$LA" --ckpt-ma "$CK_M" --ckpt-lb "$LB" --ckpt-mb "$CK_M" --out "$OUTC" --device cuda \
         --num-shards 91 --shard-idx "$i") >> "$R4/cols/$TAG.local.log" 2>&1 || { echo "local task $i failed" >&2; return 1; }
    else
      ssh3 "cd $X3_ROOT/stage/$SN/code && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 nice -n 10 python3 -B cols_h450_gpu.py \
         --bank $XE/$(basename "$EVAL_BANK") --ckpt-la $XE/ckpts/$TAG/la.pt --ckpt-ma $XE/ckpts/$TAG/m.pt --ckpt-lb $XE/ckpts/$TAG/lb.pt \
         --ckpt-mb $XE/ckpts/$TAG/m.pt --out $XE/cols/$TAG --device cuda --num-shards 91 --shard-idx $i" >> "$R4/cols/$TAG.x3.log" 2>&1 \
        || { echo "xbabe3 task $i failed" >&2; return 1; }
      rs3 --ignore-existing --include='[0-9a-f]*.npz' --exclude='*' "$X3:$XE/cols/$TAG/" "$OUTC/"
      echo "$i" >> "$R4/cols/$TAG.x3.tasks"
    fi
  done
}
mkdir -p "$OUTC"
log "cols $TAG: 91 one-context tasks on $(hostname) + $X3 (dynamic); la=$LA lb=$LB m=init"
worker local & P0=$!
worker xbabe3 & P1=$!
wait "$P0" || die "local worker failed"
wait "$P1" || die "xbabe3 worker failed"
n1=$(wc -l < "$R4/cols/$TAG.x3.tasks" 2>/dev/null || echo 0)
log "cols $TAG: xbabe3 computed $n1 of 91 contexts"
n=$(find "$OUTC" -maxdepth 1 -name '[0-9a-f]*.npz' | wc -l)
log "cols $TAG: $n/91 contexts in $(( $(date +%s) - t0 )) s"
[ "$n" = 91 ] || die "incomplete cols $TAG"
chmod -R a-w "$OUTC"
if [ "$TAG" = g00_init ]; then "$RUN" analyze-epoch "$TAG" "$OUTC"; else "$RUN" analyze-epoch "$TAG" "$OUTC" "$R4/cols/g00_init"; fi
D=/home/ubuntu/jope_plan/evidence/H450/diag_results/r4_$TAG
jq -c --arg tag "$TAG" '{tag: $tag, icl_only: .in_corpus_icl_boost_only_91.elo.elo, champion_stack: .in_corpus_random_split_champion_91.elo.elo,
        legacy_stack: .in_corpus_legacy_guard_stack_91.elo.elo,
        champion_vs_kumo_L: .in_corpus_random_split_champion_91.gap_elo["vs_kumo_tabular_large_e8.v1"],
        icl_vs_kumo_L: .in_corpus_icl_boost_only_91.gap_elo["vs_kumo_tabular_large_e8.v1"]}' "$D/H450_DEV91.json" \
  | tee "$D/SUMMARY.json"
log "eval $TAG done in $(( $(date +%s) - t0 )) s"
