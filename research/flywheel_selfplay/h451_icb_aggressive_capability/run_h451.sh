#!/bin/bash
# H451: the clean ICB_v1 capability battery on the H450 aggressive fine-tune (init vs final per size).
# Scores with the frozen H431 stage (icb_run.py battery, EMA weights, CPU only); writes nothing outside ROOT.
#
#   run_h451.sh det PREREG_COMMIT CPU
#       determinism gate: re-scores shard 0/34 of the stored M_h024 battery (10 tasks, 1 thread) and compares
#       every task file with the stored one, wall_s removed. Exit 1 if any differs.
#   run_h451.sh battery PREREG_COMMIT LABEL SIZE CKPT CKPT_SHA256 CPULIST
#       LABEL init_M | init_L | final_<name>. M: all 340 tasks, one single-thread shard per cpu.
#       L: the 68 rep-0 tasks (--rep0-only), one 4-thread shard per 4 cpus, as H431 scored L_h018.
# Every output directory is create-once and read-only when done.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
REPO=/home/ubuntu/JopeDime
GROUP=research/flywheel_selfplay/h451_icb_aggressive_capability
ST=/home/ubuntu/apex-stage/h431/pub-r3; SHA_ST=f9df082d1800cad9
STORED=/home/ubuntu/jope-data/h335-cl/h431/pub-r2
ICB=$STORED/ICB_v1.npz; SHA_ICB=492bb143672958247b65fef72adafa99b327766f293c7cf7b5cebd5f562a58a5
CK_M24=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h024.pt; SHA_M24=7fbca0e702bee00b4df0a58fac4062f550ceb71e9319fbbed5356a9b326ef176
ROOT=/home/ubuntu/jope-data/cw-battery/h451
SELF=$(readlink -f "$0"); HERE=$(dirname "$SELF")
log() { echo "$(date -u +%FT%TZ) $*"; }

preflight() {  # PREREG_COMMIT
  local pc=$1 f
  git -C "$REPO" merge-base --is-ancestor "$pc" origin/v1/next || { echo "$pc is not on origin/v1/next" >&2; exit 2; }
  for f in PREREG.md run_h451.sh summarize_h451.sh; do
    cmp -s "$HERE/$f" <(git -C "$REPO" show "$pc:$GROUP/$f") || { echo "$f differs from $pc:$GROUP/$f" >&2; exit 2; }
  done
  [ "$(cat "$ST/mode")" = pub ] || { echo "$ST is not a pub stage" >&2; exit 2; }
  (cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
  [ "$(sha256sum "$ST/CODE.sha256" | cut -c1-16)" = "$SHA_ST" ] || { echo "H431 stage CODE.sha256 is not $SHA_ST" >&2; exit 2; }
  [ "$(sha256sum "$ICB" | cut -d' ' -f1)" = "$SHA_ICB" ] || { echo "ICB_v1 sha mismatch" >&2; exit 2; }
}

cpus() {  # CPULIST a-b -> space-separated list
  seq -s ' ' "${1%-*}" "${1#*-}"
}

strip_wall() { sed -E 's/, "wall_s": [-0-9.eE+]+\}$/}/' "$1"; }

CMD=${1:?det|battery}
case $CMD in
det)
  PC=${2:?PREREG_COMMIT}; CPU=${3:?CPU}
  preflight "$PC"
  OUT=$ROOT/det_M_h024_s0of34
  [ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
  mkdir -p "$OUT"
  log "det start: M_h024 shard 0/34 on cpu $CPU"
  (cd "$ST/code" && taskset -c "$CPU" nice -n 19 python3 -B icb_run.py battery --icb "$ICB" --icb-sha256 "$SHA_ICB" \
     --ckpt "$CK_M24" --ckpt-sha256 "$SHA_M24" --size M --out "$OUT" --shard 0/34 --threads 1) > "$OUT.log" 2>&1
  same=0; diff=()
  for f in "$OUT"/task_*.json; do
    b=$(basename "$f")
    if cmp -s <(strip_wall "$f") <(strip_wall "$STORED/bat_M_h024/$b"); then same=$((same + 1)); else diff+=("$b"); fi
  done
  n=$(find "$OUT" -maxdepth 1 -name 'task_*.json' | wc -l)
  jq -n --argjson n "$n" --argjson same "$same" --arg pc "$PC" --arg diff "${diff[*]:-}" \
    '{schema: "jope.h451.det/v1", prereg_commit: $pc, tasks: $n, identical: $same,
      differ: ($diff | split(" ") | map(select(length > 0))), pass: ($n == 10 and $same == 10)}' > "$OUT.json"
  chmod -R a-w "$OUT" "$OUT.json" "$OUT.log"
  log "det done: $(cat "$OUT.json" | jq -c .)"
  [ "$(jq -r .pass "$OUT.json")" = true ]
  ;;
battery)
  PC=${2:?PREREG_COMMIT}; LABEL=${3:?LABEL}; SIZE=${4:?M|L}; CK=${5:?CKPT}; CKSHA=${6:?CKPT_SHA256}; CPUS=${7:?CPULIST}
  [[ $LABEL =~ ^(init_M|init_L|final_[A-Za-z0-9_]+)$ ]] || { echo "LABEL must be init_M, init_L or final_<name>" >&2; exit 2; }
  [ -f "$ROOT/det_M_h024_s0of34.json" ] && [ "$(jq -r .pass "$ROOT/det_M_h024_s0of34.json")" = true ] \
    || { echo "determinism gate has not passed" >&2; exit 2; }
  preflight "$PC"
  [ "$(sha256sum "$CK" | cut -d' ' -f1)" = "$CKSHA" ] || { echo "checkpoint sha mismatch" >&2; exit 2; }
  OUT=$ROOT/bat_$LABEL
  [ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
  read -r -a cpu <<< "$(cpus "$CPUS")"
  case $SIZE in
    M) TH=1; NS=${#cpu[@]}; EXTRA=() ;;
    L) TH=4; NS=$(( ${#cpu[@]} / 4 )); EXTRA=(--rep0-only)
       [ $(( ${#cpu[@]} % 4 )) = 0 ] || { echo "L needs a multiple of 4 cpus" >&2; exit 2; } ;;
    *) echo "SIZE must be M or L" >&2; exit 2 ;;
  esac
  [ "$NS" -ge 1 ] && [ "${#cpu[@]}" -le 16 ] || { echo "CPULIST must name 1-16 cpus" >&2; exit 2; }
  mkdir -p "$OUT" "$ROOT/logs"
  { echo "prereg_commit $PC"; echo "label $LABEL"; echo "size $SIZE"; echo "ckpt $CK $CKSHA"
    echo "stage $ST $SHA_ST"; echo "icb $ICB $SHA_ICB"; echo "cpus $CPUS shards $NS threads $TH"
    echo "start $(date -u +%FT%TZ)"; } > "$ROOT/RUN_INFO_$LABEL.txt"
  log "battery $LABEL start ($SIZE, $NS shards x $TH threads on $CPUS)"
  for ((k = 0; k < NS; k++)); do
    set_=$(echo "${cpu[@]:$((k * TH)):$TH}" | tr ' ' ',')
    (cd "$ST/code" && OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH taskset -c "$set_" nice -n 19 python3 -B icb_run.py battery \
       --icb "$ICB" --icb-sha256 "$SHA_ICB" --ckpt "$CK" --ckpt-sha256 "$CKSHA" --size "$SIZE" --out "$OUT" \
       --shard "$k/$NS" --threads "$TH" "${EXTRA[@]}") > "$ROOT/logs/${LABEL}_$k.log" 2>&1 &
  done
  rc=0
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  n=$(find "$OUT" -maxdepth 1 -name 'task_*.json' | wc -l)
  nf=$(find "$OUT" -maxdepth 1 -name 'task_*.json' -print0 | xargs -0 cat | jq -s '[.[] | select(.ok | not)] | length')
  (cd "$OUT" && find . -maxdepth 1 -type f -name '*.json' | sort | xargs sha256sum) > "$ROOT/MANIFEST_$LABEL.sha256"
  echo "end $(date -u +%FT%TZ) worst_exit $rc tasks $n not_ok $nf manifest $(sha256sum "$ROOT/MANIFEST_$LABEL.sha256" | cut -c1-16)" \
    >> "$ROOT/RUN_INFO_$LABEL.txt"
  chmod -R a-w "$OUT" "$ROOT/RUN_INFO_$LABEL.txt" "$ROOT/MANIFEST_$LABEL.sha256"
  log "battery $LABEL done: worst exit $rc, $n tasks, $nf not ok"
  [ "$rc" = 0 ]
  ;;
*) echo "unknown command $CMD" >&2; exit 2 ;;
esac
