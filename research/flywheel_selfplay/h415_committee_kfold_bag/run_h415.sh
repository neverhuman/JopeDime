#!/bin/bash
# H415 orchestration, run on xbabe2. Every step refuses unless the staged PREREG (code/PREREG_H415.md) is byte-identical
# to the H415 PREREG on origin/v1/next, and the stage is a `pub` stage (code = published bytes).
#   H415_STAGE=/home/ubuntu/h415-stage/pub-TAG run_h415.sh STEP [args]
#   encode            P201e SEARCH views on xbabe2 (H335 bwrap sandbox, H335_VIEWS_RW=1; DOPE sha checked)
#   push              P201e panel file + the 185 views + the encode receipt -> xbabe0 (same paths; sha-compared)
#   idcheck           xbabe0: ID2 replay (H364 memo copy, seed 364101, H0.1-A) must equal the stored X0, 0 worker runs
#   gp ARM CPUS SHARDS WORKERS MEM
#                     xbabe0: systemd-run --user h415-gp-ARM (docker sandbox) for ARM in GS GE GL
#   ltrain            xbabe1 + xbabe3: systemd-run --user h415-ltrain-HOST (H0.2-L recipe; folds 1-2 / 3-4 per FOLD_HOST)
#   lassemble         H415-L = H410 C1 fold-0 pair + the H415 folds 1-4 -> xbabe0 $DATA/neural/runs/H415-L (read-only)
#   deploy-prep RUN ARMS
#                     xbabe0: create-once $DATA/runs/RUN with gproot_AGG/memo/<cid>.split.jsonl = the concatenation of
#                     the arms' memos (ARMS e.g. GS,GE,GL; arm run dirs H415_<ARM>); every arm must hold one stage
#                     record per context of its config; sha256 manifest of sources and outputs -> logs/
#   deploy RUN TAG ARMS PANEL CPUS JOBS MEM
#                     xbabe0: systemd-run --user h415-deploy-TAG: train_h415.py deploy (docker sandbox)
#   pull-analyze RUN TAG ARMS PANEL
#                     copy the deploy + arm stage records to $EV/pulled/RUN (sha-compared), run train_h415.py analyze
#                     from the stage on xbabe2 -> $EV/results/RUN_TAG.json
#   smoke-gp ARM CPUS WORKERS MEM
#                     DEV2E GP arm (never scored) into $DATA/runs/H415_SMOKE_<ARM>; deploy-prep / deploy /
#                     pull-analyze then take RUN=H415_SMOKE_DEPLOY with ARMS SMOKE_GS,SMOKE_GE and PANEL DEV2E.
#                     A dev stage is accepted only for DEV2E smoke steps and only with H415_DEV_OK=1.
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
MODE=$(cat "$STAGE/mode")
SMOKE=0
case "$STEP ${1:-}" in "smoke-gp "*|"deploy-prep H415_SMOKE_"*|"deploy H415_SMOKE_"*|"pull-analyze H415_SMOKE_"*) SMOKE=1 ;; esac
if [ "$MODE" != pub ]; then
  [ "$SMOKE" = 1 ] && [ "${H415_DEV_OK:-0}" = 1 ] || { echo "$STAGE is not a pub stage" >&2; exit 3; }
fi
DEVENV=(); [ "$MODE" = pub ] || DEVENV=(-E H415_DEV_OK=1)
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
    for h in xbabe1 xbabe3; do
      scp -q "$STAGE/code/train_l_h415.sh" "$h:/home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh"
      ssh -o BatchMode=yes "$h" "cmp -s /home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh /dev/stdin" \
        < "$STAGE/code/train_l_h415.sh"
      ssh -o BatchMode=yes "$h" "systemd-run --user --unit h415-ltrain-$h --collect -p CPUQuota=400% -p MemoryMax=28G \
        -p MemorySwapMax=0 -p Nice=10 bash /home/ubuntu/jope-data/h317-cl/neural/train_l_h415.sh" \
        | tee -a "$EV/logs/ltrain.log"
    done ;;
  lassemble)
    L1=/home/ubuntu/jope-data/h317-cl/neural/runs-h415/H0.2-L
    X2=/home/ubuntu/jope-data/h415/H415-L
    declare -A HOST_OF=([1]=xbabe1 [2]=xbabe1 [3]=xbabe3 [4]=xbabe3)
    [ ! -e "$X2" ] || { echo "$X2 exists (create-once)" >&2; exit 3; }
    mkdir -p "$X2"
    for s in 317101 317102; do
      for f in 1 2 3 4; do
        mkdir -p "$X2/seed$s/fold$f"; rsync -a "${HOST_OF[$f]}:$L1/seed$s/fold$f/" "$X2/seed$s/fold$f/"
      done
      mkdir -p "$X2/seed$s/fold0"; rsync -a "xbabe0:$DATA/neural/runs/H410-C1-L/seed$s/fold0/" "$X2/seed$s/fold0/"
    done
    (cd "$X2" && find . -type f | sort | xargs sha256sum) > "$EV/logs/H415-L.sha256"
    [ "$(grep -c '/model.pt$' "$EV/logs/H415-L.sha256")" = 10 ] || { echo "H415-L needs 10 model.pt files" >&2; exit 6; }
    "${SSH1[@]}" "cd $L1 && sha256sum --quiet -c" < <(grep -E '/fold[12]/' "$EV/logs/H415-L.sha256")
    ssh -o BatchMode=yes xbabe3 "cd $L1 && sha256sum --quiet -c" < <(grep -E '/fold[34]/' "$EV/logs/H415-L.sha256")
    "${SSH0[@]}" "cd $DATA/neural/runs/H410-C1-L && sha256sum --quiet -c" < <(grep '/fold0/' "$EV/logs/H415-L.sha256")
    "${SSH0[@]}" "[ ! -e $DATA/neural/runs/H415-L ]"
    rsync -a "$X2/" "xbabe0:$DATA/neural/runs/H415-L/"
    "${SSH0[@]}" "cd $DATA/neural/runs/H415-L && sha256sum --quiet -c && chmod -R a-w ." < "$EV/logs/H415-L.sha256"
    chmod -R a-w "$X2"
    echo "H415-L assembled: $(grep -c model.pt "$EV/logs/H415-L.sha256") model.pt files, sha OK on xbabe0" ;;
  smoke-gp)
    ARM=${1:?ARM}; CPUS=${2:?CPUS}; WORKERS=${3:?WORKERS}; MEM=${4:?MEM}
    ND=${NEURAL[$ARM]:?unknown arm}
    R=$DATA/runs/H415_SMOKE_$ARM
    "${SSH0[@]}" "[ ! -e $R ] && mkdir -p $R && systemd-run --user --unit h415-smoke-gp-$ARM --collect \
      ${DEVENV[*]} -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_h415_xbabe0.sh $STAGE \
      $CPUS ${MEM} gp_h415.py launch --run-dir $R --neural-run-dir $DATA/neural/runs/$ND --cpus $CPUS --shards 1 \
      --workers-per-shard $WORKERS --panel DEV2E" | tee -a "$EV/logs/smoke_gp_$ARM.log" ;;
  deploy-prep)
    RUN=${1:?RUN}; ARMS=${2:?ARMS}
    R=$DATA/runs/$RUN
    IFS=, read -ra A <<< "$ARMS"
    MAN=$EV/logs/memo_$RUN.sha256
    [ ! -e "$MAN" ] || { echo "$MAN exists (create-once)" >&2; exit 3; }
    "${SSH0[@]}" "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }
      C0=\$(jq -c .cids $DATA/runs/H415_${A[0]}/config.json)
      for a in ${A[*]}; do
        D=$DATA/runs/H415_\$a
        [ \"\$(jq -c .cids \$D/config.json)\" = \"\$C0\" ] || { echo \"arm \$a: cids differ\" >&2; exit 4; }
        for c in \$(jq -r '.[]' <<< \"\$C0\"); do
          [ \$(grep -c '\"type\": \"result\"' \$D/stages/E415_AGG/\${c}__415101.jsonl) = 1 ] \
            || { echo \"arm \$a \$c: no single stage record\" >&2; exit 4; }
          [ -s \$D/gproot_AGG/memo/\$c.split.jsonl ] || { echo \"arm \$a \$c: no memo\" >&2; exit 4; }
        done
      done
      mkdir -p $R/gproot_AGG/memo
      for c in \$(jq -r '.[]' <<< \"\$C0\"); do
        src=; for a in ${A[*]}; do src=\"\$src $DATA/runs/H415_\$a/gproot_AGG/memo/\$c.split.jsonl\"; done
        cat \$src > $R/gproot_AGG/memo/\$c.split.jsonl
        sha256sum \$src $R/gproot_AGG/memo/\$c.split.jsonl
      done" > "$MAN"
    chmod a-w "$MAN"
    echo "deploy-prep $RUN from $ARMS: $(grep -c "$RUN/gproot_AGG" "$MAN") concatenated memos; manifest $MAN" ;;
  deploy)
    RUN=${1:?RUN}; TAG=${2:?TAG}; ARMS=${3:?ARMS}; PANEL=${4:?PANEL}; CPUS=${5:?CPUS}; JOBS=${6:?JOBS}; MEM=${7:?MEM}
    R=$DATA/runs/$RUN
    SRC=${ARMS//SMOKE_/}
    "${SSH0[@]}" "[ -d $R/gproot_AGG/memo ] && [ ! -e $R/config_$TAG.json ] && systemd-run --user --unit h415-deploy-$TAG \
      --collect ${DEVENV[*]} -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_h415_xbabe0.sh \
      $STAGE $CPUS ${MEM} train_h415.py deploy --panel $PANEL --run-dir $R --tag $TAG --sources $SRC --cores $CPUS \
      --jobs $JOBS" | tee -a "$EV/logs/deploy_$TAG.log" ;;
  pull-analyze)
    RUN=${1:?RUN}; TAG=${2:?TAG}; ARMS=${3:?ARMS}; PANEL=${4:?PANEL}
    P=$EV/pulled/$RUN
    [ ! -e "$P" ] || { echo "$P exists (create-once)" >&2; exit 3; }
    "${SSH0[@]}" "ls $DATA/runs/$RUN/status_${TAG}_*.json" >/dev/null || { echo "deploy $TAG has no status file" >&2; exit 6; }
    IFS=, read -ra A <<< "$ARMS"
    mkdir -p "$P/$RUN" "$EV/results"
    rsync -a "xbabe0:$DATA/runs/$RUN/config_$TAG.json" "xbabe0:$DATA/runs/$RUN/stages" "$P/$RUN/"
    rsync -a "xbabe0:$DATA/runs/$RUN/" --include='status_*.json' --exclude='*' "$P/$RUN/"
    GPA=()
    for a in "${A[@]}"; do
      mkdir -p "$P/H415_$a"
      rsync -a "xbabe0:$DATA/runs/H415_$a/config.json" "xbabe0:$DATA/runs/H415_$a/stages" "$P/H415_$a/"
      GPA+=(--gp "${a#SMOKE_}=$P/H415_$a")
    done
    (cd "$P" && find . -type f | sort | xargs sha256sum) > "$P.sha256"
    "${SSH0[@]}" "cd $DATA/runs && sha256sum --quiet -c" < "$P.sha256"
    chmod -R a-w "$P" "$P.sha256"
    PF=(); [ "$PANEL" = P201e ] && PF=(--panel-file "$DATA/h415/P201e.json")
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/train_h415.py" analyze --run-dir "$P/$RUN" \
      --tag "$TAG" --panel "$PANEL" "${GPA[@]}" "${PF[@]}" --out "$EV/results/${RUN}_$TAG.json" \
      | tee "$EV/results/${RUN}_$TAG.txt"
    chmod a-w "$EV/results/${RUN}_$TAG.json" "$EV/results/${RUN}_$TAG.txt" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
