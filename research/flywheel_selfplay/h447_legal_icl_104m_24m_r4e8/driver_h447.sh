#!/bin/bash
# H447 driver on FIT-SEL (106) and P201b (201) -> stops BEFORE readout so gates_h447.sh runs first!
#   driver_h447.sh selftest              -> runs all 3 H447 selftests on xbabe2
#   driver_h447.sh cols-base-x0          -> syncs stage + banks + base ckpts to idle xbabe0 (96 nice-19 LPT workers on 16-63,80-127)
#                                           and pulls sha-verified cols_r4e8_base_{FITSEL,P201B} back to $R
#   driver_h447.sh wc-gpu HOST_L GPU_L HOST_M GPU_M -> optional 1.25h warm-continuation on all_legal_bank.npz + GPU WC cols
#   driver_h447.sh apply                 -> runs eval_h447.py apply on FIT-SEL and P201b (stops before readout)
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

CMD=${1:?selftest|cols-base-x0|wc-gpu|apply}
EV=/home/ubuntu/jope_plan/evidence/H447
ST=$(cat "$EV/logs/pub_stage")
R=${WAVE_OUT:-/home/ubuntu/jope-data/h335-cl/h447/$(basename "$ST")}
REPO=/home/ubuntu/JopeDime
SHA_BASE=672c99372a969dda

BF=/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz
SHA_BF=f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069
BP=/home/ubuntu/jope-data/h430/bank_P201B.npz
SHA_BP=da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af
RF=/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL
RP=/home/ubuntu/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM

M7_F=/home/ubuntu/jope-data/h335-cl/h440/pub-r1/cols_FITSEL
M7_P=/home/ubuntu/jope-data/h335-cl/h440/pub-r1/cols_P201B
K3_F=/home/ubuntu/jope-data/h430/pub/cols_FITSEL
K3_P=/home/ubuntu/jope-data/h430/pub/cols_P201B
F4_F=/home/ubuntu/jope-data/h433/pub/cols_FITSEL
F4_P=/home/ubuntu/jope-data/h433/pub/cols_P201B

CK_L=/home/ubuntu/jope-data/h447/inputs/ICL_L_g1_w2_ckpt_h030.pt
SHA_L=637bc123db18ca027fb46f5113a8f778f3aafb703fecb8c0edc91da7faeebffc
CK_M=/home/ubuntu/jope-data/h447/inputs/ICL_M_r3_ckpt_h040.pt
SHA_M=eb3a0903a7da136e2cdedd1fa3f1863fd351121d49f50a4b49e5b0e6d57e9ae4
CK_M20=/home/ubuntu/jope-data/h424/ICL_M_r3_ckpt_h020.pt
SHA_M20=e3c8fd9972ce0c1684630ff06b159e950f6f498c4461d0454e0c6a6f44d38dd7

BANK_ALL=/home/ubuntu/jope-data/h335-cl/h446/pub-r6/banks/all_legal_bank.npz
SHA_BANK_ALL=7d4164f9a9b5e618a2c47b8437c7e06ef87b1de11f3fd36ede2a48d7bfc2dc04
BANK_FIT_ICL=/home/ubuntu/jope-data/apex-icl/bank/fit_icl_bank.npz
SHA_FIT_ICL=f6e283cebf9c7e40cd62b7cedf8f1283d010f7dc5feff0ab5045d0b2fd8e8ce7

log() { echo "$(date -u +%FT%TZ) $*"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
x0() { ssh -o BatchMode=yes xbabe0 "$@"; }

(cd "$ST/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$ST/mode")" = pub ]
[ "$(head -c16 "$ST/base_CODE.sha256")" = "$SHA_BASE" ]
[ "$(sha "$BF")" = "$SHA_BF" ]
[ "$(sha "$BP")" = "$SHA_BP" ]
[ "$(sha "$CK_L")" = "$SHA_L" ]
[ "$(sha "$CK_M")" = "$SHA_M" ]
[ "$(sha "$CK_M20")" = "$SHA_M20" ]
[ "$(sha "$BANK_ALL")" = "$SHA_BANK_ALL" ]
[ "$(sha "$BANK_FIT_ICL")" = "$SHA_FIT_ICL" ]
mkdir -p "$R" "$EV/logs" "$EV/results"
cd "$ST/code"

case "$CMD" in
selftest)
  [ ! -e "$R/selftests.json" ] || { log "selftests.json already exists"; exit 0; }
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B warm_continue_icl.py selftest > "$R/selftest_wc.txt"
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B cols_r4_e8.py selftest > "$R/selftest_cols.txt"
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B eval_h447.py selftest > "$R/selftest_eval.txt"
  jq -n \
    --slurpfile wc <(grep '"all_passed"' "$R/selftest_wc.txt") \
    --slurpfile co <(grep '"all_passed"' "$R/selftest_cols.txt") \
    --slurpfile ev <(grep '"all_passed"' "$R/selftest_eval.txt") \
    '{wc: $wc[0], cols: $co[0], eval: $ev[0], all_passed: ($wc[0].all_passed and $co[0].all_passed and $ev[0].all_passed)}' \
    > "$R/selftests.json"
  cat "$R/selftests.json"
  ;;

cols-base-x0)
  X0_ROOT=/home/ubuntu/jope-data/h447/$(basename "$ST")
  X0_CPUS="16-63,80-127"
  LINE="$(date -u +%FT%TZ) agy_apex_w10: H447 taking idle xbabe0 cpus $X0_CPUS (96 single-thread nice-19 LPT workers) for ~0.6h: R4-e8 (104M+24M) + CW-5 R4 M20 5-fold columns on FIT-SEL (106) and P201b (201). Will post release when done."
  printf '%s\n' "$LINE" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
  echo "$LINE" >> "$EV/logs/announce_x0.txt"

  T0=$(date +%s)
  echo "$T0" > "$EV/logs/t_cols_start"
  log "syncing stage + banks + base ckpts to xbabe0:$X0_ROOT"
  x0 "mkdir -p $X0_ROOT/inputs $X0_ROOT/stage $X0_ROOT/logs"
  rsync -a "$ST/" "xbabe0:$X0_ROOT/stage/"
  rsync -a "$BF" "$BP" "$CK_L" "$CK_M" "$CK_M20" "xbabe0:$X0_ROOT/inputs/"
  x0 "set -euo pipefail
    (cd $X0_ROOT/stage/code && sha256sum --quiet -c ../CODE.sha256)
    [ \"\$(sha256sum $X0_ROOT/inputs/bank_FITSEL.npz | cut -d' ' -f1)\" = '$SHA_BF' ]
    [ \"\$(sha256sum $X0_ROOT/inputs/bank_P201B.npz | cut -d' ' -f1)\" = '$SHA_BP' ]
    [ \"\$(sha256sum $X0_ROOT/inputs/ICL_L_g1_w2_ckpt_h030.pt | cut -d' ' -f1)\" = '$SHA_L' ]
    [ \"\$(sha256sum $X0_ROOT/inputs/ICL_M_r3_ckpt_h040.pt | cut -d' ' -f1)\" = '$SHA_M' ]
    [ \"\$(sha256sum $X0_ROOT/inputs/ICL_M_r3_ckpt_h020.pt | cut -d' ' -f1)\" = '$SHA_M20' ]"

  for PANEL_SPEC in "FITSEL:bank_FITSEL.npz" "P201B:bank_P201B.npz"; do
    PN=${PANEL_SPEC%%:*}
    BN=${PANEL_SPEC#*:}
    log "draining cols_r4e8_base_$PN on xbabe0 (96 LPT workers on cpus $X0_CPUS)"
    x0 "set -euo pipefail
      export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
      cd $X0_ROOT/stage/code
      CPUS=( \$(seq 16 63) \$(seq 80 127) )
      rc=0
      for w in \$(seq 0 95); do
        c=\${CPUS[\$w]}
        taskset -c \"\$c\" nice -n 19 python3 -B cols_r4_e8.py run \
          --bank $X0_ROOT/inputs/$BN \
          --ckpt-l $X0_ROOT/inputs/ICL_L_g1_w2_ckpt_h030.pt --sha-l '$SHA_L' --size-l L \
          --ckpt-m $X0_ROOT/inputs/ICL_M_r3_ckpt_h040.pt --sha-m '$SHA_M' --size-m M \
          --ckpt-m20 $X0_ROOT/inputs/ICL_M_r3_ckpt_h020.pt --sha-m20 '$SHA_M20' \
          --out $X0_ROOT/cols_r4e8_base_$PN --device cpu --threads 1 \
          --queue --worker-id \"\$w\" --ctx-s 7200 > $X0_ROOT/logs/${PN}_q\$w.log 2>&1 &
      done
      for j in \$(jobs -p); do wait \"\$j\" || rc=\$?; done
      exit \"\$rc\""
    mkdir -p "$R/cols_r4e8_base_$PN"
    rsync -a --exclude='.*.claim' "xbabe0:$X0_ROOT/cols_r4e8_base_$PN/" "$R/cols_r4e8_base_$PN/"
    (cd "$R/cols_r4e8_base_$PN" && find . -maxdepth 1 -type f ! -name '.*' | sort | xargs sha256sum) > "$R/cols_r4e8_base_${PN}.sha256"
    x0 "cd $X0_ROOT/cols_r4e8_base_$PN && sha256sum --quiet -c" < "$R/cols_r4e8_base_${PN}.sha256"
    log "pulled & sha-verified cols_r4e8_base_$PN ($(find "$R/cols_r4e8_base_$PN" -maxdepth 1 -name '*.npz' | wc -l) npz files)"
  done
  echo "$(date +%s)" > "$EV/logs/t_cols_end"
  LINE_REL="$(date -u +%FT%TZ) agy_apex_w10: H447 finished FIT-SEL + P201b R4-e8 column pass on xbabe0; releasing xbabe0 cpus $X0_CPUS."
  printf '%s\n' "$LINE_REL" | (cd "$REPO" && flock AGENT_CHAT.md.lock tee -a AGENT_CHAT.md >/dev/null)
  ;;

apply)
  [ ! -e "$R/APPLY_FITSEL.json" ] || { echo "$R/APPLY_FITSEL.json exists (create-once)" >&2; exit 3; }
  [ ! -e "$R/APPLY_P201B.json" ] || { echo "$R/APPLY_P201B.json exists (create-once)" >&2; exit 3; }
  [ ! -e "$EV/results/READOUT_H447.json" ] || { echo "READOUT_H447.json exists" >&2; exit 3; }
  [ ! -e "$EV/results/FROZEN_FAIR_CHAMPION.json" ] || { echo "FROZEN_FAIR_CHAMPION.json exists" >&2; exit 3; }

  WC_F_ARGS=()
  WC_P_ARGS=()
  WC_CKPT_L=""
  WC_SHA_L=""
  WC_CKPT_M=""
  WC_SHA_M=""
  if [ -d "$R/cols_r4e8_wc_FITSEL" ] && [ -d "$R/cols_r4e8_wc_P201B" ]; then
    WC_F_ARGS+=(--r4e8-wc-cols "$R/cols_r4e8_wc_FITSEL")
    WC_P_ARGS+=(--r4e8-wc-cols "$R/cols_r4e8_wc_P201B")
    WC_CKPT_L="$R/wc_L/ckpt_wc_final.pt"
    WC_SHA_L=$(sha "$WC_CKPT_L")
    WC_CKPT_M="$R/wc_M/ckpt_wc_final.pt"
    WC_SHA_M=$(sha "$WC_CKPT_M")
  fi

  T_AP0=$(date +%s)
  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B eval_h447.py apply \
    --run-dir "$RF" --tag TBOTH --panel FIT-SEL --bank "$BF" \
    --matern-cols "$M7_F" --h430-cols "$K3_F" --h433-cols "$F4_F" \
    --r4e8-base-cols "$R/cols_r4e8_base_FITSEL" "${WC_F_ARGS[@]}" \
    --out "$R/APPLY_FITSEL.json" --procs 8

  CUDA_VISIBLE_DEVICES="" nice -n 19 python3 -B eval_h447.py apply \
    --run-dir "$RP" --tag P201B --panel P201B --bank "$BP" \
    --matern-cols "$M7_P" --h430-cols "$K3_P" --h433-cols "$F4_P" \
    --r4e8-base-cols "$R/cols_r4e8_base_P201B" "${WC_P_ARGS[@]}" \
    --out "$R/APPLY_P201B.json" --procs 8
  T_AP1=$(date +%s)

  jq -n \
    --arg bcl "$CK_L" --arg bsl "$SHA_L" \
    --arg bcm "$CK_M" --arg bsm "$SHA_M" \
    --arg m20c "$CK_M20" --arg m20s "$SHA_M20" \
    --arg wcl "$WC_CKPT_L" --arg wsl "$WC_SHA_L" \
    --arg wcm "$WC_CKPT_M" --arg wsm "$WC_SHA_M" \
    --arg sball "$SHA_BANK_ALL" --arg sbfit "$SHA_FIT_ICL" \
    '{base_ckpt_l:$bcl, base_sha_l:$bsl, base_ckpt_m:$bcm, base_sha_m:$bsm,
      m20_ckpt:$m20c, m20_sha:$m20s, wc_ckpt_l:$wcl, wc_sha_l:$wsl, wc_ckpt_m:$wcm, wc_sha_m:$wsm,
      all_legal_bank_sha256:$sball, fit_icl_bank_sha256:$sbfit}' > "$EV/logs/pins.json"

  T0=$(cat "$EV/logs/t_cols_start")
  TC=$(cat "$EV/logs/t_cols_end")
  jq -n --argjson t0 "$T0" --argjson tc "$TC" --argjson tap0 "$T_AP0" --argjson tap1 "$T_AP1" \
    '{cols_wall_s:($tc-$t0), apply_wall_s:($tap1-$tap0), total_wall_s:($tap1-$t0)}' > "$EV/logs/run_times.json"
  log "apply complete (ready for gates_h447.sh then eval_h447.py readout)"
  ;;

*)
  echo "unknown subcommand $CMD" >&2
  exit 2
  ;;
esac
