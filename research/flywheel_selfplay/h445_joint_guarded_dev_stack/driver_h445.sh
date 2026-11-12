#!/bin/bash
# H445 two-hands scored CPU driver on APEX-DEV (`cpus 72-79` @ nice 19 during cl_fwriter's `64-71` K3+F4 run,
# with optional `cols-helper` on `64-71` once cl_fwriter finishes so combined threads never exceed 16; 0 GPU):
#   driver_h445.sh cols                                           -> selftest + cols_matern_DEV + LPT queue cols_cw5_DEV on cpus 72-79 (8 procs)
#   driver_h445.sh cols-helper                                    -> joins LPT queue cols_cw5_DEV on cpus 64-71 (8 procs) AFTER cl_fwriter exits
#   driver_h445.sh apply CL_COLS_DIR PREREG_COMMIT REVIEW_RECEIPT -> runs full pre-apply gate check (M1) BEFORE stack_joint_dev.py apply
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CMD=${1:?cols|cols-helper|apply}
EV=/home/ubuntu/jope_plan/evidence/H445
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h445/$(basename "$ST")}
P=${WAVE_PROCS:-8}
CPUSET=${WAVE_CPUSET:-72-79}
[ "$P" -le 16 ] || { echo "P=$P exceeds 16-thread total wave ceiling" >&2; exit 2; }

REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h445_joint_guarded_dev_stack
SHA_BASE=672c99372a969dda
BANK_DEV=/home/ubuntu/jope-data/h335-cl/h424/pub/bank_DEV.npz
SHA_DEV=dccf9beb3312f42e5a28c32aba33389c13f9c582add85084b0fc7d236f613c9b
SCR_DEV=/home/ubuntu/jope_plan/evidence/H425/results/SCREEN_DEV.json
SHA_SCR=fa8edd8a08a9aef30af98b8240f3e3b1f40d11e9681b157d7190cb00ff58891e
EXCL_DEV=/home/ubuntu/jope_plan/evidence/H425/results/EXCLUSION_DEV.json
RUN_DEV=/home/ubuntu/jope_plan/evidence/H420/pulled/H420_DEV/H420_DEV
H417_BASE=/home/ubuntu/jope_plan/evidence/H417/pulled/APEX_BASE/APEX_BASE
H424_COLS=/home/ubuntu/jope-data/h424/pub/cols_DEV
H424_REF=/home/ubuntu/jope_plan/evidence/H424/results/H424_DEV.apply.json
SHA_H424_REF=ce0eb88749ebfb5c92aa6c10cd621563b1ab4a041756cc6e882b282be824faca
SHA_FRZ=c55ca3389ebf24e2ce4362db2ce9aa50a38f3215fb2cc4662486b6a75bd0f027
CKPT_M20=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt
SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7

log() { echo "$(date -u +%FT%TZ) $*"; }
(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
[ "$(head -c16 "$ST/base_CODE.sha256")" = "$SHA_BASE" ]
[ "$(jq -r .audit_targets_written "$BANK_DEV.json")" = false ]
[ "$(sha256sum "$BANK_DEV" | cut -d' ' -f1)" = "$SHA_DEV" ]
mkdir -p "$R" "$EV/logs" "$EV/results"
cd "$ST/code"

read -r -a CPUS <<< "$(seq -s ' ' "${CPUSET%-*}" "${CPUSET#*-}")"
[ "${#CPUS[@]}" -ge "$P" ] || { echo "CPUSET=$CPUSET has fewer than $P cores" >&2; exit 2; }

count_npz() { find "$1" -maxdepth 1 -type f -regextype posix-extended -regex '.*/[0-9a-f]{64}\.npz' 2>/dev/null | wc -l; }

case "$CMD" in
cols)
  [ "$P" -le 8 ] || { echo "primary cols uses <= 8 cores ($CPUSET) while cl_fwriter uses 64-71" >&2; exit 2; }
  T0=$(date +%s)
  echo "$T0" > "$EV/logs/t_cols_start"
  echo "$P" > "$EV/logs/max_procs"
  log "cols_start P=$P CPUSET=$CPUSET stage=$ST CODE.sha256=$(sha256sum "$ST/CODE.sha256" | cut -c1-16)"
  if [ ! -e "$R/selftest.json" ]; then
    taskset -c "${CPUS[0]}" nice -n 19 python3 -B stack_joint_dev.py selftest > "$R/selftest.raw.txt"
    head -n 1 "$R/selftest.raw.txt" > "$R/selftest.json"
    tail -n 1 "$R/selftest.raw.txt"
  fi
  rc=0
  for k in $(seq 0 $((P - 1))); do
    taskset -c "${CPUS[$k]}" nice -n 19 python3 -B cols_matern.py \
      --bank "$BANK_DEV" --out "$R/cols_matern_DEV" --shard "$k/$P" --ctx-s 900 > "$EV/logs/matern_$k.log" 2>&1 &
  done
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  log "cols_matern_DEV ended (worst_rc=$rc): $(count_npz "$R/cols_matern_DEV") context files"
  [ "$rc" -eq 0 ] || exit "$rc"

  if [ "$(jq -r .promote_to_joint_dev "$ST/code/FROZEN_CW5.json")" = true ]; then
    log "promote_to_joint_dev=true: draining cols_cw5_DEV via LPT queue ($P workers on $CPUSET, --ctx-s 7200)"
    rc=0
    for k in $(seq 0 $((P - 1))); do
      taskset -c "${CPUS[$k]}" nice -n 19 python3 -B cols_cw5_dev.py \
        --bank "$BANK_DEV" --ckpt "$CKPT_M20" --ckpt-sha256 "$SHA_M20" --size M \
        --frozen "$ST/code/FROZEN_CW5.json" --out "$R/cols_cw5_DEV" --threads 1 \
        --queue --worker-id "$k" --ctx-s 7200 > "$EV/logs/cw5_q$k.log" 2>&1 &
    done
    for j in $(jobs -p); do wait "$j" || rc=$?; done
    log "cols_cw5_DEV primary workers ended (worst_rc=$rc): $(count_npz "$R/cols_cw5_DEV") context files"
    [ "$rc" -eq 0 ] || exit "$rc"
  fi
  echo "$(date +%s)" > "$EV/logs/t_cols_end"
  log "cols phase complete"
  ;;

cols-helper)
  H_CPUSET=${WAVE_HELPER_CPUSET:-64-71}
  H_P=${WAVE_HELPER_PROCS:-8}
  [ "$H_P" -le 8 ] || { echo "helper procs $H_P > 8" >&2; exit 2; }
  if pgrep -f "cols_h430\.py|cols_h433\.py|cols_keys\.py|cols_formula\.py|dev_cols_k3_f4\.sh" >/dev/null 2>&1; then
    echo "REFUSING cols-helper: cl_fwriter K3/F4 processes still active on 64-71" >&2
    exit 3
  fi
  read -r -a HCPUS <<< "$(seq -s ' ' "${H_CPUSET%-*}" "${H_CPUSET#*-}")"
  echo "16" > "$EV/logs/max_procs"
  log "cols-helper joining LPT queue on $H_CPUSET ($H_P workers, --ctx-s 7200; cl_fwriter 64-71 verified idle)"
  rc=0
  for k in $(seq 0 $((H_P - 1))); do
    wid=$((8 + k))
    taskset -c "${HCPUS[$k]}" nice -n 19 python3 -B cols_cw5_dev.py \
      --bank "$BANK_DEV" --ckpt "$CKPT_M20" --ckpt-sha256 "$SHA_M20" --size M \
      --frozen "$ST/code/FROZEN_CW5.json" --out "$R/cols_cw5_DEV" --threads 1 \
      --queue --worker-id "$wid" --ctx-s 7200 > "$EV/logs/cw5_q$wid.log" 2>&1 &
  done
  for j in $(jobs -p); do wait "$j" || rc=$?; done
  echo "$(date +%s)" > "$EV/logs/t_helper_end"
  log "cols-helper ended (worst_rc=$rc): $(count_npz "$R/cols_cw5_DEV") context files"
  [ "$rc" -eq 0 ] || exit "$rc"
  ;;

apply)
  CL_OUT=${2:?path to cl_fwriter DEV cols output directory containing MANIFEST.sha256}
  PC=${3:?commit on origin/v1/next that published PREREG and code}
  REV=${4:?path to cl_fwriter review receipt}
  PRE_OUT=$EV/results/H445.pre_apply.json
  [ ! -e "$R/APPLY_DEV.json" ] || { echo "$R/APPLY_DEV.json exists (create-once)" >&2; exit 3; }
  [ ! -e "$EV/results/READOUT_DEV.json" ] || { echo "$EV/results/READOUT_DEV.json exists" >&2; exit 3; }
  [ ! -e "$PRE_OUT" ] || { echo "$PRE_OUT exists (create-once)" >&2; exit 3; }

  # M1 Pre-apply gate verification BEFORE computing any DEV AUDIT utilities:
  git -C "$REPO" merge-base --is-ancestor "$PC" origin/v1/next
  cmp -s "$ST/code/PREREG_H445.md" <(git -C "$REPO" show "$PC:$WDIR/PREREG.md")
  pc_short=${PC:0:10}
  [ -f "$REV" ] && grep -Eq "$PC|$pc_short" "$REV" && grep -qi 'NO BLOCKING FINDING' "$REV"
  ptime=$(git -C "$REPO" log -1 --format=%ct "$PC")
  rev_time=$(stat -c %Y "$REV")
  first_our=$(find "$R" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
  first_cl=$(find "$CL_OUT" -type f -printf '%T@\n' | cut -d. -f1 | sort -n | awk 'NR==1')
  first=$(( first_our < first_cl ? first_our : first_cl ))
  [ "$first" -gt "$ptime" ] && [ "$first" -ge "$rev_time" ]

  [ "$(sha256sum "$SCR_DEV" | cut -d' ' -f1)" = "$SHA_SCR" ]
  cmp -s "$EXCL_DEV" <(git -C "$REPO" show origin/v1/next:research/flywheel_selfplay/h425_contamination_screen/EXCLUSION_DEV.json)
  (cd /home/ubuntu/jope_plan/evidence/H420/pulled/H420_DEV && sha256sum --quiet -c ../H420_DEV.sha256)
  [ "$(sha256sum "$H424_REF" | cut -d' ' -f1)" = "$SHA_H424_REF" ]
  [ "$(sha256sum "$ST/code/FROZEN_CW5.json" | cut -d' ' -f1)" = "$SHA_FRZ" ]
  (cd "$CL_OUT" && sha256sum --quiet -c MANIFEST.sha256)
  grep -Eq "^prereg_commit ($PC|$pc_short)" "$CL_OUT/RUN_INFO.txt"
  grep -q 'c84af3af9fae4f32' "$CL_OUT/RUN_INFO.txt"
  grep -q '436737c85ad4d398' "$CL_OUT/RUN_INFO.txt"
  [ "$(jq -r .all_passed "$R/selftest.json")" = true ]
  [ "$(find /home/ubuntu/jope-data/h335-cl/apex/tables -maxdepth 1 -type f | wc -l)" = 661 ]

  T0=$(cat "$EV/logs/t_cols_start")
  T_COLS_END=$(cat "$EV/logs/t_cols_end")
  if [ -f "$EV/logs/t_helper_end" ]; then
    T_H_END=$(cat "$EV/logs/t_helper_end")
    [ "$T_H_END" -gt "$T_COLS_END" ] && T_COLS_END=$T_H_END
  fi
  COLS_WALL=$((T_COLS_END - T0))
  [ "$COLS_WALL" -le 35400 ] || { echo "REFUSING apply: cols_wall_s=$COLS_WALL exceeds pre-apply budget" >&2; exit 4; }

  CW5_ARGS=()
  if [ "$(jq -r .promote_to_joint_dev "$ST/code/FROZEN_CW5.json")" = true ]; then
    CW5_ARGS+=(--cw5-cols "$R/cols_cw5_DEV")
  fi
  taskset -c "${CPUS[0]}" nice -n 19 python3 -B stack_joint_dev.py precheck \
    --run-dir "$RUN_DEV" --tag TBOTH --matern-cols "$R/cols_matern_DEV" \
    --h430-cols "$CL_OUT/cols_K3_DEV" --h433-cols "$CL_OUT/cols_F4_DEV" \
    --h424-cols "$H424_COLS" --frozen-cw5 "$ST/code/FROZEN_CW5.json" "${CW5_ARGS[@]}" \
    --min-ok 109 > "$PRE_OUT.part"
  mv -n "$PRE_OUT.part" "$PRE_OUT"
  chmod 0444 "$PRE_OUT"
  log "pre-apply gate check PASSED ($(cat "$PRE_OUT"))"

  MAXP=$(cat "$EV/logs/max_procs" 2>/dev/null || echo "$P")
  T_AP0=$(date +%s)
  taskset -c "$CPUSET" nice -n 19 python3 -B stack_joint_dev.py apply \
    --run-dir "$RUN_DEV" --h417-dir "$H417_BASE" --tag TBOTH --panel DEV \
    --bank "$BANK_DEV" --matern-cols "$R/cols_matern_DEV" \
    --h430-cols "$CL_OUT/cols_K3_DEV" --h433-cols "$CL_OUT/cols_F4_DEV" \
    --h424-cols "$H424_COLS" --frozen-cw5 "$ST/code/FROZEN_CW5.json" "${CW5_ARGS[@]}" \
    --h424-apply-ref "$H424_REF" --out "$R/APPLY_DEV.json" --procs "$P"
  T_END=$(date +%s)
  APPLY_WALL=$((T_END - T_AP0))
  COMP_WALL=$((COLS_WALL + APPLY_WALL))
  echo "$CL_OUT" > "$EV/logs/cl_cols_root"
  jq -n --argjson run_start "$T0" --argjson cols_end "$T_COLS_END" --argjson apply_start "$T_AP0" --argjson run_end "$T_END" \
    --argjson cols_wall_s "$COLS_WALL" --argjson apply_wall_s "$APPLY_WALL" --argjson compute_wall_s "$COMP_WALL" \
    --argjson procs "$MAXP" --arg cpuset "64-79" --arg cl_out "$CL_OUT" \
    '{run_start:$run_start, cols_end:$cols_end, apply_start:$apply_start, run_end:$run_end, cols_wall_s:$cols_wall_s, apply_wall_s:$apply_wall_s, compute_wall_s:$compute_wall_s, end_to_end_wall_s:($run_end-$run_start), procs:$procs, cpuset:$cpuset, cl_cols_root:$cl_out}' > "$EV/logs/run_times.json"
  log "apply phase done (compute_wall_s=${COMP_WALL}s, apply_wall_s=${APPLY_WALL}s; ready for gates_h445.sh then single-shot readout)"
  ;;

*)
  echo "unknown subcommand $CMD" >&2
  exit 2
  ;;
esac
