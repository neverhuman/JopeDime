#!/bin/bash
# H415 orchestration, run on xbabe2. Every step refuses unless the staged PREREG (code/PREREG_H415.md) is byte-identical
# to the H415 PREREG on origin/v1/next, and the stage is a `pub` stage (code = published bytes).
#   H415_STAGE=/home/ubuntu/h415-stage/pub-TAG run_h415.sh STEP [args]
#   encode            P201e SEARCH views on xbabe2 (H335 bwrap sandbox, H335_VIEWS_RW=1; DOPE sha checked)
#   push              P201e panel file + the 185 views + the encode receipt -> xbabe0 (same paths; sha-compared)
#   idcheck           xbabe0: ID2 replay (H364 memo copy, seed 364101, H0.1-A) must equal the stored X0, 0 worker runs
#   gp ARM CPUS SHARDS WORKERS MEM
#                     xbabe0: systemd-run --user h415-gp-ARM (docker sandbox) for ARM in GS GE GL
#   ltrain            xbabe1: systemd-run --user h415-ltrain (H0.2-L recipe, folds 1-4, GPU under qualification.lock)
#   lassemble         H415-L = H410 C1 fold-0 pair + the H415 folds 1-4 -> xbabe0 $DATA/neural/runs/H415-L (read-only)
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H415_STAGE:?H415_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h415_committee_kfold_bag
CIDS_SHA=4ba54ff659340b786504be21aef57d112beb6e06e98e66516433d4d613d32ea0
N=185
DEADLINE=2026-11-09T06:00:00Z
DATA=/home/ubuntu/jope-data/h335-cl
EV=/home/ubuntu/jope_plan/evidence/H415
SSH0=(ssh -o BatchMode=yes xbabe0)
SSH1=(ssh -o BatchMode=yes xbabe1)
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
declare -A NEURAL=([GS]=H0.1-A [GE]=H381-E2-S3fz [GL]=H415-L)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
[ "$(cat "$STAGE/mode")" = pub ] || { echo "$STAGE is not a pub stage" >&2; exit 3; }
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
  +refs/heads/v1/next:refs/remotes/origin/v1/next
cmp -s "$STAGE/code/PREREG_H415.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
  || { echo "staged PREREG is not the H415 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; }
mkdir -p "$EV/logs"
case $STEP in
  encode)
    cd /home/ubuntu
    [ "$(jq -r .cids_sha256 "$DATA/h415/P201e.json")" = "$CIDS_SHA" ] || { echo "P201e != PREREG panel" >&2; exit 5; }
    H335_VIEWS_RW=1 H335_CODE=$STAGE/code nice -n 19 "$STAGE/code/sandbox_h335.sh" \
      python3 -B gp_h415.py encode --threads 8 | tee -a "$EV/logs/encode.log" ;;
  push)
    jq -r '.cids[]' "$DATA/h415/P201e.json" | sed 's/$/.npz/' > "$EV/logs/p201e_views.txt"
    [ "$(wc -l < "$EV/logs/p201e_views.txt")" = $N ] || exit 5
    (cd "$DATA/views-search" && ls encode_receipt_h415_P201e_* >> "$EV/logs/p201e_views.txt")
    (cd "$DATA/views-search" && xargs sha256sum < "$EV/logs/p201e_views.txt") > "$EV/logs/p201e_views.sha256"
    "${SSH0[@]}" "mkdir -p $DATA/h415"
    rsync -a "$DATA/h415/P201e.json" "xbabe0:$DATA/h415/P201e.json"
    rsync -a --files-from="$EV/logs/p201e_views.txt" "$DATA/views-search/" "xbabe0:$DATA/views-search/"
    "${SSH0[@]}" "cd $DATA/views-search && sha256sum --quiet -c" < "$EV/logs/p201e_views.sha256"
    "${SSH0[@]}" "chmod a-w $DATA/h415/P201e.json; jq -r .cids_sha256 $DATA/h415/P201e.json" | grep -qx "$CIDS_SHA"
    echo "pushed $(wc -l < "$EV/logs/p201e_views.sha256") files to xbabe0, sha OK" ;;
  idcheck)
    R=$DATA/runs/H415_ID2
    ID2=$(git -C "$REPO" show origin/v1/next:research/flywheel_selfplay/h381_rapid_lf_grow/panels.json | jq -r '.S60.cids[0:2][]')
    [ "$(echo "$ID2" | wc -l)" = 2 ] || exit 5
    "${SSH0[@]}" "[ ! -e $R ] && mkdir -p $R/gproot_AGG/memo"
    for c in $ID2; do
      "${SSH0[@]}" "cp $DATA/runs/H364_C_xbabe0/gproot_AGG/memo/$c.split.jsonl $R/gproot_AGG/memo/"
    done
    "${SSH0[@]}" "systemd-run --user --unit h415-idcheck --wait --collect -p MemoryMax=24G -p MemorySwapMax=0 \
      bash $STAGE/code/run_h415_xbabe0.sh $STAGE 16-31 20g gp_h415.py launch --run-dir $R \
      --neural-run-dir $DATA/neural/runs/H0.1-A --cpus 16-31 --shards 1 --workers-per-shard 16 --panel ID2" \
      | tee -a "$EV/logs/idcheck.log"
    "${SSH0[@]}" "systemd-run --user --unit h415-idcheck2 --wait --collect -p MemoryMax=10G -p MemorySwapMax=0 \
      bash $STAGE/code/run_h415_xbabe0.sh $STAGE 16-31 8g gp_h415.py idcheck --run-dir $R; \
      cat $R/idcheck.json" | tee -a "$EV/logs/idcheck.log" ;;
  gp)
    ARM=${1:?ARM}; CPUS=${2:?CPUS}; SHARDS=${3:?SHARDS}; WORKERS=${4:?WORKERS}; MEM=${5:?MEM}
    ND=${NEURAL[$ARM]:?unknown arm}
    R=$DATA/runs/H415_$ARM
    "${SSH0[@]}" "ls $DATA/neural/runs/$ND/seed*/fold4/model.pt" >/dev/null \
      || { echo "screener $ND has no fold-4 models on xbabe0" >&2; exit 6; }
    "${SSH0[@]}" "[ ! -e $R ] && mkdir -p $R && systemd-run --user --unit h415-gp-$ARM --collect -p MemoryMax=${MEM^^} \
      -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_h415_xbabe0.sh $STAGE $CPUS ${MEM} gp_h415.py launch \
      --run-dir $R --neural-run-dir $DATA/neural/runs/$ND --cpus $CPUS --shards $SHARDS --workers-per-shard $WORKERS \
      --panel P201e" | tee -a "$EV/logs/gp_$ARM.log" ;;
  ltrain)
    scp -q "$STAGE/code/train_l_h415.sh" xbabe1:/home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh
    "${SSH1[@]}" "cmp -s /home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh /dev/stdin" < "$STAGE/code/train_l_h415.sh"
    "${SSH1[@]}" "systemd-run --user --unit h415-ltrain --collect -p CPUQuota=400% -p MemoryMax=28G -p MemorySwapMax=0 \
      -p Nice=10 bash /home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh" | tee -a "$EV/logs/ltrain.log" ;;
  lassemble)
    L1=/home/ubuntu/jope-data/h317-cl/neural/runs-h415/H0.2-L
    X2=/home/ubuntu/jope-data/h415/H415-L
    [ ! -e "$X2" ] || { echo "$X2 exists (create-once)" >&2; exit 3; }
    "${SSH1[@]}" "ls $L1/seed317101/fold4/model.pt $L1/seed317102/fold4/model.pt" >/dev/null
    mkdir -p "$X2"
    for s in 317101 317102; do
      for f in 1 2 3 4; do mkdir -p "$X2/seed$s/fold$f"; rsync -a "xbabe1:$L1/seed$s/fold$f/" "$X2/seed$s/fold$f/"; done
      mkdir -p "$X2/seed$s/fold0"; rsync -a "xbabe0:$DATA/neural/runs/H410-C1-L/seed$s/fold0/" "$X2/seed$s/fold0/"
    done
    (cd "$X2" && find . -type f | sort | xargs sha256sum) > "$EV/logs/H415-L.sha256"
    "${SSH1[@]}" "cd $L1 && sha256sum --quiet -c" < <(grep -v '/fold0/' "$EV/logs/H415-L.sha256")
    "${SSH0[@]}" "cd $DATA/neural/runs/H410-C1-L && sha256sum --quiet -c" < <(grep '/fold0/' "$EV/logs/H415-L.sha256")
    "${SSH0[@]}" "[ ! -e $DATA/neural/runs/H415-L ]"
    rsync -a "$X2/" "xbabe0:$DATA/neural/runs/H415-L/"
    "${SSH0[@]}" "cd $DATA/neural/runs/H415-L && sha256sum --quiet -c && chmod -R a-w ." < "$EV/logs/H415-L.sha256"
    chmod -R a-w "$X2"
    echo "H415-L assembled: $(grep -c model.pt "$EV/logs/H415-L.sha256") model.pt files, sha OK on xbabe0" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
