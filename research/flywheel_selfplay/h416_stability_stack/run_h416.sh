#!/bin/bash
# H416 orchestration, run on xbabe2. Every step refuses unless the staged PREREG (code/PREREG_H416.md) is byte-identical
# to the H416 PREREG on origin/v1/next, the stage is a `pub` stage (code = published bytes), and it is before the
# compute deadline. Smoke steps (run dirs H416_SMOKE_*, panels SMK2 / ID2B) also accept a dev stage with H416_DEV_OK=1.
#   H416_STAGE=/home/ubuntu/h416-stage/pub-TAG run_h416.sh STEP [args]
#   panel             xbabe0 docker: gp_h416.py panel -> $DATA/h416/P201b.json (create-once; = the P201b board contexts);
#                     copied to xbabe2 (same path), sha-compared, read-only on both hosts
#   tprep             xbabe0: create-once $DATA/runs/H416_GS = per P201b context the stored Phase T memo and T_AGG record
#                     (T_xbabe0 on xbabe0: 150 contexts; T_xbabe2 on xbabe2: 51, rsynced), sha256-verified, read-only
#   idcheck           xbabe0: ID2B replay (memo copy of 2 contexts from H416_GS, seed 335101, H0.1-A) must equal the
#                     stored Phase T records and the board's h335.H335_AGG scores, with 0 worker runs
#   gp ARM CPUS SHARDS WORKERS MEM
#                     xbabe0: systemd-run --user h416-gp-ARM (docker sandbox), ARM in GE GL, panel P201b
#   smoke-gp ARM CPUS WORKERS MEM
#                     the same on ID2B into $DATA/runs/H416_SMOKE_<ARM> (never scored)
#   deploy-prep RUN PANEL
#                     xbabe0: create-once $DATA/runs/RUN/gproot_AGG/memo/<cid>.split.jsonl:
#                       P201b / ID2B  the concatenation of the GS (H416_GS), GE and GL (H416_[SMOKE_]GE/GL) memos;
#                                     every arm holds one stage record per context
#                       P201e / SMK2  a copy of the stored H415 deploy memo (H415_DEPLOY; P201e = H415's panel)
#                     sha256 manifest of sources and outputs -> $EV/logs/memo_RUN.sha256
#   deploy RUN TAG MODE PANEL CPUS JOBS MEM [DESIGN]
#                     xbabe0: systemd-run --user h416-deploy-TAG: train_h416.py deploy (docker sandbox). Scored confirm
#                     runs (P201b) take code/DESIGN.json, which must equal the H416 DESIGN.json on origin/v1/next.
#   pull-select RUN TAG
#                     copy the dev deploy to $EV/pulled/RUN (sha-compared) and run train_h416.py select from the stage
#                     on xbabe2 -> $EV/results/RUN_TAG.design.json + RUN_TAG.p201e_refs.json
#   pull-analyze RUN TAG
#                     copy the confirm deploy, the GE / GL arm stages and the H416_GS Phase T records to $EV/pulled/RUN
#                     (sha-compared) and run train_h416.py analyze from the stage -> $EV/results/RUN_TAG.json
set -euo pipefail
STEP=${1:?step}; shift
STAGE=${H416_STAGE:?H416_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h416_stability_stack
DEADLINE=2026-11-10T08:00:00Z
SEED=335101
DATA=/home/ubuntu/jope-data/h335-cl
EV=/home/ubuntu/jope_plan/evidence/H416
EV15=/home/ubuntu/jope_plan/evidence/H415
H415RUN=$DATA/runs/H415_DEPLOY
P201B=$DATA/h416/P201b.json
P201E=$DATA/h415/P201e.json
SSH0=(ssh -o BatchMode=yes xbabe0)
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
declare -A NEURAL=([GS]=H0.1-A [GE]=H381-E2-S3fz [GL]=H415-L)
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
MODE=$(cat "$STAGE/mode")
SMOKE=0
case "$STEP ${1:-}" in
  "smoke-gp "*|"deploy-prep H416_SMOKE_"*|"deploy H416_SMOKE_"*|"pull-select H416_SMOKE_"*|"pull-analyze H416_SMOKE_"*)
    SMOKE=1 ;;
  # data preparation (no scoring; outputs create-once and re-verified by the published code at every use) and the
  # identity check (a dev stage writes H416_SMOKE_ID2B; the published stage repeats it into H416_ID2B)
  "panel "*|"tprep "*|"idcheck "*) SMOKE=1 ;;
esac
if [ "$MODE" != pub ]; then
  [ "$SMOKE" = 1 ] && [ "${H416_DEV_OK:-0}" = 1 ] || { echo "$STAGE is not a pub stage" >&2; exit 3; }
fi
DEVENV=(); [ "$MODE" = pub ] || DEVENV=(-E H416_DEV_OK=1)
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
  +refs/heads/v1/next:refs/remotes/origin/v1/next
if [ "$MODE" = pub ]; then
  cmp -s "$STAGE/code/PREREG_H416.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
    || { echo "staged PREREG is not the H416 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
fi
[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; }
mkdir -p "$EV/logs" "$EV/results"
X0=("${SSH0[@]}")
sd() {  # sd UNIT CPUS MEM ENTRY ARGS...: one docker-sandbox systemd-run unit on xbabe0, waited for
  local unit=$1 cpus=$2 mem=$3; shift 3
  "${X0[@]}" "systemd-run --user --unit $unit --wait --collect ${DEVENV[*]} -p MemoryMax=${mem^^} -p MemorySwapMax=0 \
    -p Nice=19 bash $STAGE/code/run_h416_xbabe0.sh $STAGE $cpus $mem $*"
}
smoke_name() { case $1 in H416_SMOKE_*) return 0 ;; *) return 1 ;; esac; }
case $STEP in
  panel)
    sd h416-panel 16-31 16g gp_h416.py panel | tee -a "$EV/logs/panel.log"
    [ ! -e "$P201B" ] || { echo "$P201B exists on xbabe2 (create-once)" >&2; exit 3; }
    mkdir -p "$DATA/h416"
    rsync -a "xbabe0:$P201B" "$P201B"
    "${X0[@]}" "chmod a-w $P201B; sha256sum $P201B" | tee -a "$EV/logs/panel.log"
    chmod a-w "$P201B"
    sha256sum "$P201B" | tee -a "$EV/logs/panel.log"
    [ "$(jq -r '.n, .families' "$P201B" | tr '\n' ' ')" = "201 201 " ] || { echo "P201b is not 201/201" >&2; exit 5; } ;;
  tprep)
    R=$DATA/runs/H416_GS
    MAN=$EV/logs/H416_GS.sha256
    [ ! -e "$MAN" ] || { echo "$MAN exists (create-once)" >&2; exit 3; }
    "${X0[@]}" "[ ! -e $R ]" || { echo "xbabe0:$R exists (create-once)" >&2; exit 3; }
    jq -r '.cids[]' "$P201B" > "$EV/logs/p201b_cids.txt"
    : > "$EV/logs/tprep_xbabe2.txt"; : > "$EV/logs/tprep_xbabe0.txt"
    while read -r c; do
      if [ -f "$DATA/runs/T_xbabe2/stages/T_AGG/${c}__$SEED.jsonl" ]; then echo "$c" >> "$EV/logs/tprep_xbabe2.txt"
      else echo "$c" >> "$EV/logs/tprep_xbabe0.txt"; fi
    done < "$EV/logs/p201b_cids.txt"
    echo "tprep: $(wc -l < "$EV/logs/tprep_xbabe2.txt") contexts from T_xbabe2 (xbabe2), \
$(wc -l < "$EV/logs/tprep_xbabe0.txt") from T_xbabe0 (xbabe0)" | tee -a "$EV/logs/tprep.log"
    [ "$(wc -l < "$EV/logs/tprep_xbabe2.txt")" = 51 ] && [ "$(wc -l < "$EV/logs/tprep_xbabe0.txt")" = 150 ] \
      || { echo "expected 51 + 150 Phase T contexts" >&2; exit 5; }
    files() { sed "s#.*#gproot_AGG/memo/&.split.jsonl\nstages/T_AGG/&__$SEED.jsonl#" "$1"; }
    files "$EV/logs/tprep_xbabe2.txt" > "$EV/logs/tprep_xbabe2.files"
    files "$EV/logs/tprep_xbabe0.txt" > "$EV/logs/tprep_xbabe0.files"
    (cd "$DATA/runs/T_xbabe2" && xargs sha256sum < "$EV/logs/tprep_xbabe2.files") > "$EV/logs/tprep_xbabe2.sha256"
    "${X0[@]}" "cd $DATA/runs/T_xbabe0 && xargs sha256sum" < "$EV/logs/tprep_xbabe0.files" > "$EV/logs/tprep_xbabe0.sha256"
    [ "$(wc -l < "$EV/logs/tprep_xbabe2.sha256")" = 102 ] && [ "$(wc -l < "$EV/logs/tprep_xbabe0.sha256")" = 300 ] \
      || { echo "missing Phase T files" >&2; exit 5; }
    "${X0[@]}" "mkdir -p $R/gproot_AGG/memo $R/stages/T_AGG && cd $DATA/runs/T_xbabe0 && \
      xargs cp --parents -t $R" < "$EV/logs/tprep_xbabe0.files"
    rsync -a --files-from="$EV/logs/tprep_xbabe2.files" "$DATA/runs/T_xbabe2/" "xbabe0:$R/"
    sort -k2 "$EV/logs/tprep_xbabe2.sha256" "$EV/logs/tprep_xbabe0.sha256" > "$MAN"
    "${X0[@]}" "cd $R && sha256sum --quiet -c && [ \$(find . -type f | wc -l) = 402 ] && chmod -R a-w ." < "$MAN"
    chmod a-w "$MAN" "$EV"/logs/tprep_*
    echo "H416_GS = 201 Phase T memos + records on xbabe0, sha OK ($(sha256sum "$MAN" | cut -c1-16))" \
      | tee -a "$EV/logs/tprep.log" ;;
  idcheck)
    R=$DATA/runs/H416_ID2B; [ "$MODE" = pub ] || R=$DATA/runs/H416_SMOKE_ID2B
    ID2B=$(jq -r '.cids | sort | .[0:2][]' "$P201B")
    "${X0[@]}" "[ ! -e $R ] && mkdir -p $R/gproot_AGG/memo"
    for c in $ID2B; do "${X0[@]}" "cp $DATA/runs/H416_GS/gproot_AGG/memo/$c.split.jsonl $R/gproot_AGG/memo/"; done
    sd h416-idcheck 16-31 20g gp_h416.py launch --run-dir "$R" --neural-run-dir "$DATA/neural/runs/H0.1-A" \
      --cpus 16-31 --shards 1 --workers-per-shard 16 --panel ID2B | tee -a "$EV/logs/idcheck.log"
    sd h416-idcheck2 16-31 8g gp_h416.py idcheck --run-dir "$R" | tee -a "$EV/logs/idcheck.log"
    "${X0[@]}" "cat $R/idcheck.json" | tee -a "$EV/logs/idcheck.log" ;;
  gp|smoke-gp)
    ARM=${1:?ARM}; CPUS=${2:?CPUS}
    if [ "$STEP" = gp ]; then SHARDS=${3:?SHARDS}; WORKERS=${4:?WORKERS}; MEM=${5:?MEM}; PANEL=P201b; R=$DATA/runs/H416_$ARM
    else SHARDS=1; WORKERS=${3:?WORKERS}; MEM=${4:?MEM}; PANEL=ID2B; R=$DATA/runs/H416_SMOKE_$ARM; fi
    case $ARM in GE|GL) ;; *) echo "GP arms are GE and GL (GS = the stored Phase T run)" >&2; exit 2 ;; esac
    ND=${NEURAL[$ARM]}
    "${X0[@]}" "ls $DATA/neural/runs/$ND/seed*/fold4/model.pt" >/dev/null \
      || { echo "screener $ND has no fold-4 models on xbabe0" >&2; exit 6; }
    "${X0[@]}" "[ ! -e $R ] && mkdir -p $R && systemd-run --user --unit h416-$STEP-$ARM --collect ${DEVENV[*]} \
      -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_h416_xbabe0.sh $STAGE $CPUS $MEM \
      gp_h416.py launch --run-dir $R --neural-run-dir $DATA/neural/runs/$ND --cpus $CPUS --shards $SHARDS \
      --workers-per-shard $WORKERS --panel $PANEL" | tee -a "$EV/logs/${STEP}_$ARM.log" ;;
  deploy-prep)
    RUN=${1:?RUN}; PANEL=${2:?PANEL}
    R=$DATA/runs/$RUN
    MAN=$EV/logs/memo_$RUN.sha256
    [ ! -e "$MAN" ] || { echo "$MAN exists (create-once)" >&2; exit 3; }
    case $PANEL in
      P201b|P201e) ! smoke_name "$RUN" || { echo "$PANEL is not a smoke panel" >&2; exit 2; } ;;
      ID2B|SMK2) smoke_name "$RUN" || { echo "$PANEL runs go to H416_SMOKE_* run dirs" >&2; exit 2; } ;;
      *) echo "unknown panel $PANEL" >&2; exit 2 ;;
    esac
    case $PANEL in
      P201b) CIDS=$(jq -r '.cids[]' "$P201B"); PFX=H416_ ;;
      ID2B) CIDS=$(jq -r '.cids | sort | .[0:2][]' "$P201B"); PFX=H416_SMOKE_ ;;
      P201e) CIDS=$(jq -r '.cids[]' "$P201E") ;;
      SMK2) CIDS=$(jq -r '.cids[0:2][]' "$P201E") ;;
    esac
    echo "$CIDS" > "$EV/logs/cids_$RUN.txt"
    if [ "$PANEL" = P201b ] || [ "$PANEL" = ID2B ]; then
      "${X0[@]}" "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }
        for c in \$(cat); do
          src=$DATA/runs/H416_GS/gproot_AGG/memo/\$c.split.jsonl
          [ \$(grep -c '\"type\": \"result\"' $DATA/runs/H416_GS/stages/T_AGG/\${c}__$SEED.jsonl) = 1 ] \
            || { echo \"GS \$c: no single Phase T record\" >&2; exit 4; }
          for a in GE GL; do
            D=$DATA/runs/$PFX\$a
            jq -e --arg c \$c '.cids | index(\$c)' \$D/config.json >/dev/null || { echo \"arm \$a: no \$c\" >&2; exit 4; }
            [ \$(grep -c '\"type\": \"result\"' \$D/stages/E416_AGG/\${c}__$SEED.jsonl) = 1 ] \
              || { echo \"arm \$a \$c: no single stage record\" >&2; exit 4; }
            [ -s \$D/gproot_AGG/memo/\$c.split.jsonl ] || { echo \"arm \$a \$c: no memo\" >&2; exit 4; }
            src=\"\$src \$D/gproot_AGG/memo/\$c.split.jsonl\"
          done
          mkdir -p $R/gproot_AGG/memo
          cat \$src > $R/gproot_AGG/memo/\$c.split.jsonl
          sha256sum \$src $R/gproot_AGG/memo/\$c.split.jsonl
        done" < "$EV/logs/cids_$RUN.txt" > "$MAN"
    else
      [ "$(jq -c .cids "$P201E")" = "$("${X0[@]}" "jq -c .cids $H415RUN/config_P201E.json")" ] \
        || { echo "H415_DEPLOY is not the P201e deploy" >&2; exit 4; }
      "${X0[@]}" "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }
        mkdir -p $R/gproot_AGG/memo
        for c in \$(cat); do
          cp $H415RUN/gproot_AGG/memo/\$c.split.jsonl $R/gproot_AGG/memo/
          sha256sum $H415RUN/gproot_AGG/memo/\$c.split.jsonl $R/gproot_AGG/memo/\$c.split.jsonl
        done" < "$EV/logs/cids_$RUN.txt" > "$MAN"
    fi
    chmod a-w "$MAN" "$EV/logs/cids_$RUN.txt"
    echo "deploy-prep $RUN ($PANEL): $(grep -c "$RUN/gproot_AGG" "$MAN") memos; manifest $MAN" ;;
  deploy)
    RUN=${1:?RUN}; TAG=${2:?TAG}; DMODE=${3:?MODE}; PANEL=${4:?PANEL}; CPUS=${5:?CPUS}; JOBS=${6:?JOBS}; MEM=${7:?MEM}
    DESIGN=${8:-}
    R=$DATA/runs/$RUN
    case $PANEL in
      P201b|P201e) ! smoke_name "$RUN" || { echo "$PANEL is not a smoke panel" >&2; exit 2; } ;;
      ID2B|SMK2) smoke_name "$RUN" || { echo "$PANEL runs go to H416_SMOKE_* run dirs" >&2; exit 2; } ;;
      *) echo "unknown panel $PANEL" >&2; exit 2 ;;
    esac
    DA=()
    if [ "$DMODE" = confirm ]; then
      if [ "$PANEL" = P201b ]; then
        [ -z "$DESIGN" ] || { echo "the scored confirm run takes the staged code/DESIGN.json" >&2; exit 2; }
        cmp -s "$STAGE/code/DESIGN.json" <(git -C "$REPO" show "origin/v1/next:$WDIR/DESIGN.json") \
          || { echo "staged DESIGN.json is not the H416 DESIGN.json on origin/v1/next (PREREG part B)" >&2; exit 3; }
        DA=(--design DESIGN.json)
      else
        DA=(--design "${DESIGN:?smoke confirm needs a design file under $DATA}")
      fi
    fi
    "${X0[@]}" "[ -d $R/gproot_AGG/memo ] && [ ! -e $R/config_$TAG.json ] && systemd-run --user --unit h416-deploy-$TAG \
      --collect ${DEVENV[*]} -p MemoryMax=${MEM^^} -p MemorySwapMax=0 -p Nice=19 bash $STAGE/code/run_h416_xbabe0.sh \
      $STAGE $CPUS $MEM train_h416.py deploy --mode $DMODE --panel $PANEL --run-dir $R --tag $TAG --cores $CPUS \
      --jobs $JOBS ${DA[*]}" | tee -a "$EV/logs/deploy_$TAG.log" ;;
  pull-select|pull-analyze)
    RUN=${1:?RUN}; TAG=${2:?TAG}
    P=$EV/pulled/$RUN
    [ ! -e "$P" ] || { echo "$P exists (create-once)" >&2; exit 3; }
    "${X0[@]}" "ls $DATA/runs/$RUN/status_${TAG}_*.json" >/dev/null || { echo "deploy $TAG has no status file" >&2; exit 6; }
    mkdir -p "$P/$RUN"
    rsync -a "xbabe0:$DATA/runs/$RUN/config_$TAG.json" "xbabe0:$DATA/runs/$RUN/stages" \
      "xbabe0:$DATA/runs/$RUN/h416_${TAG}_npz" "$P/$RUN/"
    rsync -a "xbabe0:$DATA/runs/$RUN/" --include='status_*.json' --exclude='*' "$P/$RUN/"
    PANEL=$(jq -r .panel "$P/$RUN/config_$TAG.json")
    OUT=$EV/results/${RUN}_$TAG
    if [ "$STEP" = pull-select ]; then
      case $PANEL in P201e|SMK2) ;; *) echo "pull-select reads a dev run (P201e / SMK2)" >&2; exit 2 ;; esac
      (cd "$P" && find . -type f | sort | xargs sha256sum) > "$P.sha256"
      "${X0[@]}" "cd $DATA/runs && sha256sum --quiet -c" < "$P.sha256"
      chmod -R a-w "$P" "$P.sha256"
      PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/train_h416.py" select --run-dir "$P/$RUN" \
        --tag "$TAG" --h415-dir "$EV15/pulled/H415_DEPLOY/H415_DEPLOY/stages/H415_P201E" --panel-file "$P201E" \
        --out "$OUT.design.json" --refs-out "$OUT.p201e_refs.json" | tee "$OUT.select.txt"
      chmod a-w "$OUT.design.json" "$OUT.p201e_refs.json" "$OUT.select.txt"
    else
      case $PANEL in
        P201b) PFX=H416_; DSG=$STAGE/code/DESIGN.json; REFS=$EV/results/H416_DEV_P201E.p201e_refs.json ;;
        ID2B) PFX=H416_SMOKE_; DSG=${3:?smoke design}; REFS=${4:?smoke p201e refs} ;;
        *) echo "pull-analyze reads a confirm run (P201b / ID2B)" >&2; exit 2 ;; esac
      [ -f "$REFS" ] || { echo "no P201e refs file $REFS (pull-select first)" >&2; exit 6; }
      [ "$PANEL" = P201b ] && { cmp -s "$DSG" <(git -C "$REPO" show "origin/v1/next:$WDIR/DESIGN.json") \
        || { echo "staged DESIGN.json is not on origin/v1/next" >&2; exit 3; }; }
      GPA=()
      for a in GE GL; do
        mkdir -p "$P/$PFX$a"
        rsync -a "xbabe0:$DATA/runs/$PFX$a/config.json" "xbabe0:$DATA/runs/$PFX$a/stages" "$P/$PFX$a/"
        GPA+=(--gp "$a=$P/$PFX$a")
      done
      mkdir -p "$P/H416_GS/stages"
      rsync -a "xbabe0:$DATA/runs/H416_GS/stages/T_AGG" "$P/H416_GS/stages/"
      (cd "$P" && find . -type f | sort | xargs sha256sum) > "$P.sha256"
      "${X0[@]}" "cd $DATA/runs && sha256sum --quiet -c" < "$P.sha256"
      chmod -R a-w "$P" "$P.sha256"
      rc=0
      PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 -B "$STAGE/code/train_h416.py" analyze --run-dir "$P/$RUN" \
        --tag "$TAG" --gs-t "$P/H416_GS/stages/T_AGG" "${GPA[@]}" --board-dir "$STAGE/code" --panel-file "$P201B" \
        --design "$DSG" --p201e "$REFS" --out "$OUT.json" > "$OUT.txt" 2>&1 || rc=$?
      chmod a-w "$OUT.txt"; [ -e "$OUT.json" ] && chmod a-w "$OUT.json"
      # the ID2B smoke reads 2 confirmation contexts before the design is locked: only its harness checks are shown
      if [ "$PANEL" != ID2B ]; then cat "$OUT.txt"; elif [ "$rc" = 0 ]; then grep '^checks:' "$OUT.txt"
      else tail -n 25 "$OUT.txt"; fi
      echo "analyze exit $rc"; exit "$rc"
    fi ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
