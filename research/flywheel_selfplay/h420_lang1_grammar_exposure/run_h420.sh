#!/bin/bash
# H420 orchestration (run on xbabe2). GP hosts: xbabe2 (bubblewrap sandbox, local) and xbabe3 (docker sandbox, paused
# by the CI-yield watcher while a GitHub runner container is busy). H417's arms and memos stay on xbabe0.
#   run_h420.sh gx     HOST PANEL CPUS SHARDS WORKERS MEM [RUNTAG]   GX arm -> $DATA/runs/H420_GX_<PANEL>
#   run_h420.sh memo   HOST PANEL [BASERUN] [RUNTAG]                 GS+GE+GL (xbabe0, H417) + GX (HOST) -> one memo
#   run_h420.sh chain  HOST PANEL CPUS JOBS MEM [RUNTAG]             replay check + deploys B, TGX, TLIB, TBOTH
#   run_h420.sh status HOST PANEL [RUNTAG]
# RUNTAG (default empty) names smoke runs: H420_SMOKE_<x>... with APEX_DEV_OK=1 and a dev stage.
set -euo pipefail
STEP=${1:?step}; HOST=${2:?host}; PANEL=${3:?panel}; shift 3
STAGE=${H420_STAGE:?H420_STAGE}
REPO=/home/ubuntu/JopeDime
WDIR=research/flywheel_selfplay/h420_lang1_grammar_exposure
DEADLINE=2026-11-13T06:00:00Z
SEED=417101
DATA=/home/ubuntu/jope-data/h335-cl
EV=/home/ubuntu/jope_plan/evidence/H420
CRED=(-c credential.helper= -c "credential.helper=!/home/ubuntu/.local/bin/gh-role writer auth git-credential")
[ "$(hostname)" = xbabe2 ] || { echo "run on xbabe2" >&2; exit 2; }
export XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus
MODE=$(cat "$STAGE/mode")
DEVENV=(); [ "$MODE" = pub ] || DEVENV=(-E APEX_DEV_OK=1)
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
if [ "$MODE" = pub ]; then
  git -C "$REPO" "${CRED[@]}" fetch -q https://github.com/neverhuman/JopeDime.git \
    +refs/heads/v1/next:refs/remotes/origin/v1/next
  cmp -s "$STAGE/code/PREREG_H420.md" <(git -C "$REPO" show "origin/v1/next:$WDIR/PREREG.md") \
    || { echo "staged PREREG is not the H420 PREREG on origin/v1/next: nothing runs before publication" >&2; exit 3; }
fi
[ "$(date -u +%s)" -lt "$(date -u -d "$DEADLINE" +%s)" ] || { echo "past the compute deadline $DEADLINE" >&2; exit 8; }
mkdir -p "$EV/logs" "$EV/results" "$EV/pulled"
case $HOST in xbabe2) KIND=bwrap ;; xbabe1|xbabe3) KIND=docker ;; *) echo "host $HOST is not an H420 GP host" >&2; exit 2 ;; esac
on() { if [ "$HOST" = xbabe2 ]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); $1"; fi; }
unit() {  # UNIT MEM <command...>
  local U=$1 M=$2; shift 2
  on "systemd-run --user --unit $U --collect ${DEVENV[*]} -p MemoryMax=${M^^} -p MemorySwapMax=0 -p Nice=19 $*"
}
launcher() {  # CPUS MEM -> launcher prefix for this host
  if [ "$KIND" = bwrap ]; then echo "bash $STAGE/code/run_h420_local.sh $STAGE $1"
  else echo "bash $STAGE/code/run_apex_xbabe0.sh $STAGE $1 $2"; fi
}
smoke_ok() {  # RUNTAG PANEL: smoke runs are named and use the smoke panel; scored runs use a pub stage
  if [ -n "$1" ]; then
    [[ "$1" == SMOKE_* ]] && [ "$2" = SMK2 ] || { echo "run tags are SMOKE_<x> on panel SMK2" >&2; exit 3; }
  else
    [ "$MODE" = pub ] && [ "$2" != SMK2 ] || { echo "scored runs need a pub stage and a scored panel" >&2; exit 3; }
  fi
}
name() { echo "H420${1:+_$1}"; }   # run-name prefix
case $STEP in
  gx)
    CPUS=${1:?CPUS}; SHARDS=${2:?SHARDS}; WORKERS=${3:?WORKERS}; MEM=${4:?MEM}; RT=${5:-}
    smoke_ok "$RT" "$PANEL"
    R=$DATA/runs/$(name "$RT")_GX_$PANEL
    on "[ ! -e $R ] && mkdir -p $R" || { echo "$HOST:$R exists (create-once)" >&2; exit 3; }
    unit "h420-gx-${RT:+$RT-}$PANEL" "$MEM" "$(launcher "$CPUS" "$MEM") gp_h420.py launch --run-dir $R --panel $PANEL \
      --cpus $CPUS --shards $SHARDS --workers-per-shard $WORKERS" | tee -a "$EV/logs/gx_${RT:+${RT}_}${PANEL}_$HOST.log" ;;
  memo)
    BASE=${1:-APEX_BASE}; RT=${2:-}
    smoke_ok "$RT" "$PANEL"
    R=$DATA/runs/$(name "$RT")_$PANEL; GXR=$DATA/runs/$(name "$RT")_GX_$PANEL
    L=$EV/logs/memo_$(name "$RT")_${PANEL}_$HOST
    [ ! -e "$L.sha256" ] || { echo "$L.sha256 exists (create-once)" >&2; exit 3; }
    PYTHONDONTWRITEBYTECODE=1 python3 -B "$STAGE/code/apex_core_meta.py" panel "$PANEL" > "$L.cids"
    # H417 arm memos: checked and listed on xbabe0, then copied to the GP host
    ssh -o BatchMode=yes xbabe0 "set -euo pipefail; for c in \$(cat); do for a in GS GE GL; do
        D=$DATA/runs/${BASE}_\$a
        [ \$(grep -c '\"type\": \"result\"' \$D/stages/E417_AGG/\${c}__$SEED.jsonl) = 1 ] || { echo \"arm \$a \$c: no single stage record\" >&2; exit 4; }
        [ -s \$D/gproot_AGG/memo/\$c.split.jsonl ] || { echo \"arm \$a \$c: no memo\" >&2; exit 4; }
        (cd $DATA/runs && sha256sum ${BASE}_\$a/gproot_AGG/memo/\$c.split.jsonl)
      done; done" < "$L.cids" > "$L.src.sha256"
    cut -c67- "$L.src.sha256" > "$L.src.files"
    mkdir -p "$DATA/h420/src"
    rsync -a --files-from="$L.src.files" "xbabe0:$DATA/runs/" "$DATA/h420/src/"
    [ "$HOST" = xbabe2 ] || { ssh -o BatchMode=yes "$HOST" "mkdir -p $DATA/h420/src"; rsync -a --files-from="$L.src.files" "$DATA/h420/src/" "$HOST:$DATA/h420/src/"; }
    on "cd $DATA/h420/src && sha256sum --quiet -c" < "$L.src.sha256"
    on "set -euo pipefail; [ ! -e $R ] || { echo '$R exists (create-once)' >&2; exit 3; }; mkdir -p $R/gproot_AGG/memo
      for c in \$(cat); do
        [ \$(grep -c '\"type\": \"result\"' $GXR/stages/E420_AGG/\${c}__$SEED.jsonl) = 1 ] || { echo \"GX \$c: no single stage record\" >&2; exit 4; }
        src=\"$DATA/h420/src/${BASE}_GS/gproot_AGG/memo/\$c.split.jsonl $DATA/h420/src/${BASE}_GE/gproot_AGG/memo/\$c.split.jsonl $DATA/h420/src/${BASE}_GL/gproot_AGG/memo/\$c.split.jsonl $GXR/gproot_AGG/memo/\$c.split.jsonl\"
        cat \$src > $R/gproot_AGG/memo/\$c.split.jsonl
        sha256sum \$src $R/gproot_AGG/memo/\$c.split.jsonl
      done" < "$L.cids" > "$L.sha256"
    chmod a-w "$L.sha256" "$L.cids" "$L.src.sha256"
    echo "memo $(name "$RT") $PANEL on $HOST: $(grep -c "$R/gproot_AGG" "$L.sha256") concatenated memos; manifest $L.sha256" ;;
  chain)
    CPUS=${1:?CPUS}; JOBS=${2:?JOBS}; MEM=${3:?MEM}; RT=${4:-}
    smoke_ok "$RT" "$PANEL"
    R=$DATA/runs/$(name "$RT")_$PANEL
    on "[ -d $R/gproot_AGG/memo ] && [ ! -e $R/config_B.json ]" || { echo "$HOST:$R is not ready or already deployed" >&2; exit 3; }
    unit "h420-chain-${RT:+$RT-}$PANEL" "$MEM" "${H420_VARIANTS:+-E H420_VARIANTS='$H420_VARIANTS'} bash $STAGE/code/h420_chain.sh $STAGE $KIND $CPUS $MEM $PANEL $R $JOBS" \
      | tee -a "$EV/logs/chain_${RT:+${RT}_}${PANEL}_$HOST.log" ;;
  status)
    RT=${1:-}
    on "for u in h420-gx-${RT:+$RT-}$PANEL h420-chain-${RT:+$RT-}$PANEL; do echo \"\$u \$(systemctl --user is-active \$u 2>/dev/null)\"; done
      G=$DATA/runs/$(name "$RT")_GX_$PANEL; R=$DATA/runs/$(name "$RT")_$PANEL
      echo \"GX records \$(ls \$G/stages/E420_AGG 2>/dev/null | wc -l) shard status files \$(ls \$G/shards/status_* 2>/dev/null | wc -l)\"
      for v in B TGX TLIB TBOTH; do echo \"\$v records \$(ls \$R/stages/H416_\$v 2>/dev/null | wc -l) status \$(ls \$R/status_\${v}_*.json 2>/dev/null | wc -l)\"; done" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
