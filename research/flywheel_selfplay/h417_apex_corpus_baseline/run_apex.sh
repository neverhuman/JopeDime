#!/bin/bash
# H417 orchestration, run on xbabe2. Scored steps refuse unless the stage is a `pub` stage whose PREREG_H417.md is
# byte-identical to the H417 PREREG on origin/v1/next and it is before the compute deadline. Smoke and identity steps
# (run names APEX_SMOKE_* / APEX_ID2B*, panels SMK2 / ID2B) also accept a dev stage with APEX_DEV_OK=1.
#   APEX_STAGE=/home/ubuntu/apex-stage/h417/<mode>-TAG run_apex.sh STEP [args]
#   tables PANEL      the panel's train-only tables -> $DATA/apex/tables on xbabe2 and xbabe0 (sha256 = manifest), 0444;
#                     the manifest itself to $DATA/apex/ on both hosts
#   encode PANEL      SEARCH view sets on xbabe2 (H335 bwrap sandbox, H335_VIEWS_RW=1; DOPE sha checked), pushed to
#                     xbabe0 (sha-compared)
#   gp RUN ARM PANEL CPUS SHARDS WORKERS MEM
#                     xbabe0: systemd-run --user apex-gp-RUN-ARM (docker sandbox) -> $DATA/runs/RUN_ARM
#   deploy-prep RUN PANEL
#                     xbabe0: create-once $DATA/runs/RUN/gproot_AGG/memo/<cid>.split.jsonl = the concatenation of the
#                     RUN_GS, RUN_GE and RUN_GL memos (H416 order); panel ID2B: a copy of the stored H416_CONFIRM memo
#   deploy RUN TAG PANEL CPUS JOBS MEM
#                     xbabe0: systemd-run --user apex-deploy-TAG: train_apex.py deploy (docker sandbox)
#   pull RUN TAG      copy the deploy (config, stages, npz, status) to $EV/pulled/RUN, sha-compared, read-only
#   idcheck RUN TAG   train_apex.py idcheck: the pulled ID2B deploy vs the stored H416 confirm records
#   g0 RUN CPUS JOBS MEM | g0-pull RUN | g0-readout RUN
#                     the screener gate G0 (PREREG 6.2, part B): g0_apex.py on xbabe0, pulled, read out on xbabe2
#   readout RUN TAG | headroom RUN TAG
#                     train_apex.py from the stage on the pulled run -> $EV/results/RUN_TAG.{readout,headroom}.json
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${APEX_STAGE:?APEX_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h417_apex_corpus_baseline
DEADLINE=2026-11-12T06:00:00Z
SEED=417101
DATA=/home/ubuntu/jope-data/h335-cl
EV=/home/ubuntu/jope_plan/evidence/H417
MAN=$DATA/apex/APEX_COHORTS.json
MAN_SHA=11a57d80a67fcefbbdc83986062e50608eed78026cec7508ecbc5040999bc4eb
H416P=/home/ubuntu/jope_plan/evidence/H416/pulled/H416_CONFIRM/H416_CONFIRM
SSH0=(ssh -o BatchMode=yes xbabe0)
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
declare -A NEURAL=([GS]=H0.1-A [GE]=H381-E2-S3fz [GL]=H415-L)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
MODE=$(cat "$STAGE/mode")
SMOKE=0
case "$STEP ${1:-} ${2:-}" in
  "tables "*|"encode "*) SMOKE=1 ;;   # data preparation: no outcome is produced; outputs are sha-pinned to the manifest
  *" APEX_SMOKE_"*|*" APEX_ID2B"*) SMOKE=1 ;;
esac
if [ "$MODE" != pub ]; then
  [ "$SMOKE" = 1 ] && [ "${APEX_DEV_OK:-0}" = 1 ] || { echo "$STAGE is not a pub stage" >&2; exit 3; }
fi
DEVENV=(); [ "$MODE" = pub ] || DEVENV=(-E APEX_DEV_OK=1)
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
if [ "$MODE" = pub ]; then
  git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
    +refs/heads/v1/next:refs/remotes/origin/v1/next
  cmp -s "$STAGE/code/PREREG_H417.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
    || { echo "staged PREREG is not the H417 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
fi
[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; }
[ "$(sha256sum "$MAN" | cut -d' ' -f1)" = "$MAN_SHA" ] || { echo "manifest pin" >&2; exit 4; }
mkdir -p "$EV/logs" "$EV/results" "$EV/pulled"
X0=("${SSH0[@]}")
cids_of() {  # panel -> context ids (ID2B: the first 2 P201b contexts)
  if [ "$1" = ID2B ]; then jq -r '.cids | sort | .[0:2][]' "$DATA/h416/P201b.json"
  else H317_DATA=$DATA python3 -B "$STAGE/code/apex_core_meta.py" panel "$1"; fi
}
smoke_name() { case $1 in APEX_SMOKE_*|APEX_ID2B*) return 0 ;; *) return 1 ;; esac; }
panel_ok() {  # RUN PANEL: smoke panels go to smoke run names and nothing else does
  case $2 in SMK2|ID2B) smoke_name "$1" || { echo "$2 runs go to APEX_SMOKE_* / APEX_ID2B* run dirs" >&2; exit 2; } ;;
    *) ! smoke_name "$1" || { echo "$2 is not a smoke panel" >&2; exit 2; } ;; esac
}
case $STEP in
  tables)
    PANEL=${1:?PANEL}
    cids_of "$PANEL" > "$EV/logs/cids_$PANEL.txt"
    mkdir -p "$DATA/apex/tables"
    jq -r '.contexts[] | [.context_sha256, .source_wave, .train_only_jtt1_sha256] | @tsv' "$MAN" \
      | gawk -F'\t' 'NR==FNR {want[$1]=1; next} ($1 in want) {print}' "$EV/logs/cids_$PANEL.txt" - > "$EV/logs/tables_$PANEL.tsv"
    [ "$(wc -l < "$EV/logs/tables_$PANEL.tsv")" = "$(wc -l < "$EV/logs/cids_$PANEL.txt")" ] || { echo "manifest join" >&2; exit 5; }
    : > "$EV/logs/tables_$PANEL.sha256"
    while IFS=$'\t' read -r c w s; do
      dst=$DATA/apex/tables/$c.jtt1
      [ -f "$dst" ] || { cp "/home/ubuntu/jope-data/$w/fresh/tables/$c.jtt1" "$dst"; chmod 0444 "$dst"; }
      echo "$s  $c.jtt1" >> "$EV/logs/tables_$PANEL.sha256"
    done < "$EV/logs/tables_$PANEL.tsv"
    (cd "$DATA/apex/tables" && sha256sum --quiet -c "$EV/logs/tables_$PANEL.sha256")
    "${X0[@]}" "mkdir -p $DATA/apex/tables"
    rsync -a "$MAN" "xbabe0:$MAN"
    cut -d' ' -f3 "$EV/logs/tables_$PANEL.sha256" > "$EV/logs/tables_$PANEL.files"
    rsync -a --files-from="$EV/logs/tables_$PANEL.files" "$DATA/apex/tables/" "xbabe0:$DATA/apex/tables/"
    "${X0[@]}" "cd $DATA/apex/tables && sha256sum --quiet -c && [ \$(sha256sum $MAN | cut -d' ' -f1) = $MAN_SHA ]" \
      < "$EV/logs/tables_$PANEL.sha256"
    echo "tables $PANEL: $(wc -l < "$EV/logs/tables_$PANEL.sha256") on xbabe2 + xbabe0, sha OK" | tee -a "$EV/logs/tables.log" ;;
  encode)
    PANEL=${1:?PANEL}
    cids_of "$PANEL" > "$EV/logs/cids_$PANEL.txt"
    H335_VIEWS_RW=1 H335_CODE=$STAGE/code nice -n 19 "$STAGE/code/sandbox_h335.sh" \
      python3 -B gp_apex.py encode --panel "$PANEL" --threads 8 | tee -a "$EV/logs/encode_$PANEL.log"
    sed 's/$/.npz/' "$EV/logs/cids_$PANEL.txt" > "$EV/logs/views_$PANEL.txt"
    (cd "$DATA/views-search" && ls encode_receipt_h417_"${PANEL}"_* >> "$EV/logs/views_$PANEL.txt")
    (cd "$DATA/views-search" && xargs sha256sum < "$EV/logs/views_$PANEL.txt") > "$EV/logs/views_$PANEL.sha256"
    rsync -a --files-from="$EV/logs/views_$PANEL.txt" "$DATA/views-search/" "xbabe0:$DATA/views-search/"
    "${X0[@]}" "cd $DATA/views-search && sha256sum --quiet -c" < "$EV/logs/views_$PANEL.sha256"
    echo "encode $PANEL: pushed $(wc -l < "$EV/logs/views_$PANEL.sha256") files to xbabe0, sha OK" | tee -a "$EV/logs/encode_$PANEL.log" ;;
  gp)
    RUN=${1:?RUN}; ARM=${2:?ARM}; PANEL=${3:?PANEL}; CPUS=${4:?CPUS}; SHARDS=${5:?SHARDS}; WORKERS=${6:?WORKERS}; MEM=${7:?MEM}
    panel_ok "$RUN" "$PANEL"
    case $PANEL in ID2B) echo "ID2B has no GP arm (stored memo)" >&2; exit 2 ;; esac
    ND=${NEURAL[$ARM]:?unknown arm}
    R=$DATA/runs/${RUN}_$ARM
    "${X0[@]}" "ls $DATA/neural/runs/$ND/seed*/fold0/model.pt" >/dev/null \
      || { echo "screener $ND has no fold-0 models on xbabe0" >&2; exit 6; }
    "${X0[@]}" "[ ! -e $R ] && mkdir -p $R && systemd-run --user --unit apex-gp-$RUN-$ARM --collect ${DEVENV[*]} \
      -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_apex_xbabe0.sh $STAGE $CPUS $MEM \
      gp_apex.py launch --run-dir $R --arm $ARM --neural-run-dir $DATA/neural/runs/$ND --cpus $CPUS --shards $SHARDS \
      --workers-per-shard $WORKERS --panel $PANEL" | tee -a "$EV/logs/gp_${RUN}_$ARM.log" ;;
  deploy-prep)
    RUN=${1:?RUN}; PANEL=${2:?PANEL}
    panel_ok "$RUN" "$PANEL"
    R=$DATA/runs/$RUN
    MANI=$EV/logs/memo_$RUN.sha256
    [ ! -e "$MANI" ] || { echo "$MANI exists (create-once)" >&2; exit 3; }
    cids_of "$PANEL" > "$EV/logs/cids_$RUN.txt"
    if [ "$PANEL" = ID2B ]; then
      "${X0[@]}" "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }
        mkdir -p $R/gproot_AGG/memo
        for c in \$(cat); do
          cp $DATA/runs/H416_CONFIRM/gproot_AGG/memo/\$c.split.jsonl $R/gproot_AGG/memo/
          sha256sum $DATA/runs/H416_CONFIRM/gproot_AGG/memo/\$c.split.jsonl $R/gproot_AGG/memo/\$c.split.jsonl
        done" < "$EV/logs/cids_$RUN.txt" > "$MANI"
    else
      "${X0[@]}" "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }
        for c in \$(cat); do
          src=
          for a in GS GE GL; do
            D=$DATA/runs/${RUN}_\$a
            jq -e --arg c \$c '.cids | index(\$c)' \$D/config.json >/dev/null || { echo \"arm \$a: no \$c\" >&2; exit 4; }
            [ \$(grep -c '\"type\": \"result\"' \$D/stages/E417_AGG/\${c}__$SEED.jsonl) = 1 ] \
              || { echo \"arm \$a \$c: no single stage record\" >&2; exit 4; }
            [ -s \$D/gproot_AGG/memo/\$c.split.jsonl ] || { echo \"arm \$a \$c: no memo\" >&2; exit 4; }
            src=\"\$src \$D/gproot_AGG/memo/\$c.split.jsonl\"
          done
          mkdir -p $R/gproot_AGG/memo
          cat \$src > $R/gproot_AGG/memo/\$c.split.jsonl
          sha256sum \$src $R/gproot_AGG/memo/\$c.split.jsonl
        done" < "$EV/logs/cids_$RUN.txt" > "$MANI"
    fi
    chmod a-w "$MANI" "$EV/logs/cids_$RUN.txt"
    echo "deploy-prep $RUN ($PANEL): $(grep -c "$RUN/gproot_AGG" "$MANI") memos; manifest $MANI" ;;
  deploy)
    RUN=${1:?RUN}; TAG=${2:?TAG}; PANEL=${3:?PANEL}; CPUS=${4:?CPUS}; JOBS=${5:?JOBS}; MEM=${6:?MEM}
    panel_ok "$RUN" "$PANEL"
    R=$DATA/runs/$RUN
    "${X0[@]}" "[ -d $R/gproot_AGG/memo ] && [ ! -e $R/config_$TAG.json ] && systemd-run --user --unit apex-deploy-$TAG \
      --collect ${DEVENV[*]} -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_apex_xbabe0.sh \
      $STAGE $CPUS $MEM train_apex.py deploy --panel $PANEL --run-dir $R --tag $TAG --cores $CPUS --jobs $JOBS" \
      | tee -a "$EV/logs/deploy_$TAG.log" ;;
  pull)
    RUN=${1:?RUN}; TAG=${2:?TAG}
    P=$EV/pulled/$RUN
    [ ! -e "$P" ] || { echo "$P exists (create-once)" >&2; exit 3; }
    "${X0[@]}" "ls $DATA/runs/$RUN/status_${TAG}_*.json" >/dev/null || { echo "deploy $TAG has no status file" >&2; exit 6; }
    mkdir -p "$P/$RUN"
    rsync -a "xbabe0:$DATA/runs/$RUN/config_$TAG.json" "xbabe0:$DATA/runs/$RUN/stages" \
      "xbabe0:$DATA/runs/$RUN/h416_${TAG}_npz" "$P/$RUN/"
    rsync -a "xbabe0:$DATA/runs/$RUN/" --include='status_*.json' --exclude='*' "$P/$RUN/"
    (cd "$P" && find . -type f | sort | xargs sha256sum) > "$P.sha256"
    "${X0[@]}" "cd $DATA/runs && sha256sum --quiet -c" < "$P.sha256"
    chmod -R a-w "$P" "$P.sha256"
    echo "pulled $RUN: $(wc -l < "$P.sha256") files, sha OK" ;;
  idcheck)
    RUN=${1:?RUN}; TAG=${2:?TAG}
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/train_apex.py" idcheck --run-dir "$EV/pulled/$RUN/$RUN" \
      --tag "$TAG" --stored-run "$H416P" | tee "$EV/results/${RUN}_$TAG.idcheck.json" ;;
  readout|headroom)
    RUN=${1:?RUN}; TAG=${2:?TAG}
    OUT=$EV/results/${RUN}_$TAG.$STEP.json
    [ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
    H317_DATA=$DATA PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/train_apex.py" "$STEP" \
      --run-dir "$EV/pulled/$RUN/$RUN" --tag "$TAG" --out "$OUT" | tee "$EV/results/${RUN}_$TAG.$STEP.txt"
    chmod a-w "$OUT" "$EV/results/${RUN}_$TAG.$STEP.txt" ;;
  g0)
    RUN=${1:?RUN}; CPUS=${2:?CPUS}; JOBS=${3:?JOBS}; MEM=${4:?MEM}
    R=$DATA/runs/$RUN
    LIM=(); ! smoke_name "$RUN" || LIM=(--limit 1)
    cmp -s "$STAGE/code/g0_apex.py" <(git -C "$REPO" show "origin/v1/next:$WDIR/g0_apex.py") || [ "$MODE" != pub ] \
      || { echo "staged g0_apex.py is not on origin/v1/next (PREREG part B)" >&2; exit 3; }
    "${X0[@]}" "[ ! -e $R/config_G0.json ] && mkdir -p $R && systemd-run --user --unit apex-g0-$RUN --collect \
      ${DEVENV[*]} -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_apex_xbabe0.sh $STAGE \
      $CPUS $MEM g0_apex.py run --run-dir $R --cores $CPUS --jobs $JOBS ${LIM[*]}" | tee -a "$EV/logs/g0_$RUN.log" ;;
  g0-pull)
    RUN=${1:?RUN}
    P=$EV/pulled/$RUN
    [ ! -e "$P" ] || { echo "$P exists (create-once)" >&2; exit 3; }
    "${X0[@]}" "ls $DATA/runs/$RUN/status_G0_*.json" >/dev/null || { echo "G0 has no status file" >&2; exit 6; }
    mkdir -p "$P/$RUN"
    rsync -a "xbabe0:$DATA/runs/$RUN/config_G0.json" "xbabe0:$DATA/runs/$RUN/stages" "$P/$RUN/"
    rsync -a "xbabe0:$DATA/runs/$RUN/" --include='status_*.json' --exclude='*' "$P/$RUN/"
    (cd "$P" && find . -type f | sort | xargs sha256sum) > "$P.sha256"
    "${X0[@]}" "cd $DATA/runs && sha256sum --quiet -c" < "$P.sha256"
    chmod -R a-w "$P" "$P.sha256"
    echo "pulled $RUN: $(wc -l < "$P.sha256") files, sha OK" ;;
  g0-readout)
    RUN=${1:?RUN}
    OUT=$EV/results/$RUN.g0.json
    [ ! -e "$OUT" ] || { echo "$OUT exists (create-once)" >&2; exit 3; }
    H317_DATA=$DATA PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/g0_apex.py" readout \
      --run-dir "$EV/pulled/$RUN/$RUN" --out "$OUT" | tee "$EV/results/$RUN.g0.txt"
    chmod a-w "$OUT" "$EV/results/$RUN.g0.txt" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
