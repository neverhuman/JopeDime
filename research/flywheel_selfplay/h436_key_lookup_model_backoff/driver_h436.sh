#!/bin/bash
# H436 scored run from the published stage (PREREG stages A and B): FIT-SEL design check, then P201b. CPU only,
# 8 single-thread processes at nice 19. Stops before P201b if the design check shows harm or more than 5% of FIT-SEL
# contexts are not ok. Every output file is create-once; a restart continues.
set -euo pipefail
export PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EV=/home/ubuntu/jope_plan/evidence/H436; ST=$(cat $EV/logs/pub_stage); R=/home/ubuntu/jope-data/h436/pub; P=8
BF=/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz
BP=/home/ubuntu/jope-data/h430/bank_P201B.npz
RF=/home/ubuntu/jope_plan/evidence/H420/pulled/H420_FIT-SEL/H420_FIT-SEL
RP=/home/ubuntu/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM
K430=/home/ubuntu/jope-data/h430/pub        # H430's column files (read-only): the common base and the KD arm
log() { echo "$(date -u +%FT%TZ) $*"; }
(cd $ST/code && sha256sum --quiet -c ../CODE.sha256); [ "$(cat $ST/mode)" = pub ]
[ "$(sha256sum $BP | cut -d' ' -f1)" = "$(jq -r .sha256 $BP.json)" ]
mkdir -p $R; cd $ST/code
log "run_start; stage $ST CODE.sha256 $(sha256sum $ST/CODE.sha256 | cut -c1-16)"
nice -n 19 python3 -B test_hybrid_core.py | tail -1
cols() {   # bank, out dir, log label
  local rc=0
  for k in $(seq 0 $((P - 1))); do
    nice -n 19 python3 -B cols_h436.py --bank $1 --out $2 --shard $k/$P > $EV/logs/cols_$3_$k.log 2>&1 &
  done
  for j in $(jobs -p); do wait $j || rc=$?; done
  log "$3 columns ended (worst exit $rc): $(ls $2 | grep -c '^[0-9a-f]\{64\}\.npz$') context files"
  return $rc
}
notok() { ls $1 | grep -E '^[0-9a-f]{64}\.json$' | sed "s#^#$1/#" | xargs cat | jq -s '[.[] | select(.ok | not)] | length'; }
stack() {  # label, run dir, tag, panel, bank, cols dir
  [ -e $R/APPLY_$1.json ] || nice -n 19 python3 -B stack_h436.py apply --run-dir $2 --tag $3 --panel $4 --bank $5 \
    --cols $6 --out $R/APPLY_$1.json --procs $P --base-cols $K430/cols_$1 --base-name key_gbm,factor_glm
  [ -e $R/READOUT_$1.json ] || python3 -B stack_h436.py readout --apply $R/APPLY_$1.json --out $R/READOUT_$1.json \
    | tee $R/READOUT_$1.txt
}
# ---- stage A: FIT-SEL design check ----
cols $BF $R/cols_FITSEL FITSEL
stack FITSEL $RF TBOTH FIT-SEL $BF $R/cols_FITSEL
nf=$(notok $R/cols_FITSEL)
kb=$(jq -r '.pairs.KH_vs_B | (.clipped_mean < 0 and .mirror_signflip_p < 0.025)' $R/READOUT_FITSEL.json)
if [ "$kb" = true ]; then log "HYBRID_HARM_DESIGN: stop before P201b"; exit 0; fi
if [ $((nf * 20)) -gt 106 ]; then log "FIT-SEL incomplete ($nf contexts not ok): stop before P201b"; exit 0; fi
log "stage A done ($nf contexts not ok); no harm; continuing to P201b"
# ---- stage B: P201b ----
cols $BP $R/cols_P201B P201B
stack P201B $RP P201B P201B $BP $R/cols_P201B
[ -e $R/READOUT_P201B_excl10.json ] || python3 -B stack_h436.py readout --apply $R/APPLY_P201B.json \
  --out $R/READOUT_P201B_excl10.json --exclude PROBED_P201B.txt | tee $R/READOUT_P201B_excl10.txt
log "stage B done ($(notok $R/cols_P201B) contexts not ok)"
