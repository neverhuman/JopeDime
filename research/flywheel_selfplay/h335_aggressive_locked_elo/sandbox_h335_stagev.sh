#!/bin/bash
# H335 Stage V bwrap sandbox for xbabe2 (= the frozen sandbox_h317_stagev.sh with H335 paths; stage_v_mounts_h335.md).
# sandbox_h335.sh is unchanged and remains the Phase T / write-lock sandbox. No network in any mode.
# Usage: sandbox_h335_stagev.sh [--print] unlock|v <command...>
#   unlock  the sandbox_h335.sh layout (TRAIN-only view at the canonical RES01; $DATA rw with neural/ bin/ (superset
#           worker, frontier binary) views-search/ re-bound ro, redteam/ hidden; H317 + H335 source ro (H335_CODE = a
#           frozen copy bound at the H335 source path, as in sandbox_h335.sh); JopeDimeMicro ro with
#           assets/benchmark_data hidden) PLUS the create-once H335 LOCK receipt ro. No DL-Val byte and no Stage V root
#           is visible. Refused unless the receipt names sha256($DATA/stage_v/LOCK.json) and no STAGE_V_STARTED exists.
#   v       prep / stats, only after the unlock: the V view $SV/res01-v-view (TRAIN + the 91 DL-Val tables and pools, the
#           ORIGINAL selected.json, cpu_exact_fused.py) ro at RES01; $SV rw with res01-v-view/ and inputs/ re-bound ro;
#           $DATA/stage_v ro (LOCK, marker); the receipt ro; H335_STAGEV_ROOT=$SV. Refused unless STAGE_V_STARTED names
#           the receipt's LOCK sha and the V view holds 1,219 + 91 tables.
#   --print print the bwrap argv (one item per line) instead of running it; skips the precondition checks (tests).
# Both modes require H335_CODE = the frozen read-only code copy (e.g. /home/ubuntu/h335-stage/code); it is bound at the H335
# source path, where the LOCK hashes the runtime code (unlock and every resume_stage_v re-verify it).
# The guard refusal log stays the production default ($DATA/guard_refusals.log) unless H317_GUARD_LOG is set.
set -euo pipefail
SRC317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
DATA=${H335_DATA:-/home/ubuntu/jope-data/h335-cl}
SV=${H335_STAGEV_ROOT:-/home/ubuntu/jope-data/h335-stagev}
VIEW=/home/ubuntu/jope-data/h335-view/res01-train-view  # = sandbox_h335.sh (H317 TRAIN view links + cpu_exact_fused.py)
MICRO=/home/ubuntu/JopeDimeMicro
RES01=/home/ubuntu/jope-data/res-01/input
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H335_LOCK_sha256.txt
N_TARGET_TABLES=1310  # 1,219 TRAIN + 91 DL-Val
PRINT=0
if [ "${1:-}" = "--print" ]; then PRINT=1; shift; fi
MODE=${1:-}
[ $# -ge 2 ] || { echo "usage: $0 [--print] unlock|v <command...>" >&2; exit 2; }
shift
refuse() { echo "sandbox_h335_stagev.sh $MODE refused: $*" >&2; exit 3; }
frozen_code() {
  [ -n "${H335_CODE:-}" ] && [ -s "$H335_CODE/stage_v_h335.py" ] || refuse "H335_CODE (the frozen code copy) is required"
  [ ! -w "$H335_CODE/stage_v_h335.py" ] || refuse "H335_CODE/stage_v_h335.py is writable: not the frozen read-only copy"
}
receipt_sha() { [ -s "$RECEIPT" ] || refuse "no LOCK receipt $RECEIPT"; cut -d' ' -f1 "$RECEIPT"; }
lock_sha() { sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1; }

BASE=(--ro-bind / /
  --tmpfs /home/ubuntu --tmpfs /tmp --tmpfs /mnt --tmpfs /media --tmpfs /srv
  --tmpfs /app --tmpfs /review --tmpfs /Users --tmpfs /opt/work
  --dev /dev --proc /proc
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local
  --ro-bind "$SRC317" "$SRC317"
  --ro-bind "${H335_CODE:-$SRC}" "$SRC"
  --bind "$DATA" "$DATA" --tmpfs "$DATA/redteam"
  --ro-bind "$DATA/neural" "$DATA/neural" --ro-bind "$DATA/bin" "$DATA/bin"
  --ro-bind "$DATA/views-search" "$DATA/views-search"
  --ro-bind "$MICRO" "$MICRO" --tmpfs "$MICRO/assets/benchmark_data")
ENVS=(--setenv HOME /home/ubuntu --setenv H317_RES01 "$RES01" --setenv H317_DATA "$DATA"
  --setenv PYTHONDONTWRITEBYTECODE 1)
[ -n "${H317_GUARD_LOG:-}" ] && ENVS+=(--setenv H317_GUARD_LOG "$H317_GUARD_LOG")

case "$MODE" in
  unlock)
    if [ $PRINT = 0 ]; then
      frozen_code
      r=$(receipt_sha)
      [ "$(lock_sha)" = "$r" ] || refuse "LOCK.json does not match the receipt"
      [ ! -e "$DATA/stage_v/STAGE_V_STARTED" ] || refuse "already unlocked (STAGE_V_STARTED exists; one-shot)"
      [ "$(ls "$VIEW/tables" | wc -l)" = 1219 ] || refuse "TRAIN view tables != 1219"
    fi
    ARGS=("${BASE[@]}"
      --ro-bind "$VIEW" "$RES01"
      --ro-bind "$RECEIPT" "$RECEIPT"
      "${ENVS[@]}" --setenv H317_SANDBOX stagev-unlock)
    ;;
  v)
    if [ $PRINT = 0 ]; then
      frozen_code
      r=$(receipt_sha)
      [ "$(lock_sha)" = "$r" ] || refuse "LOCK.json does not match the receipt"
      [ -s "$DATA/stage_v/STAGE_V_STARTED" ] || refuse "not unlocked (no STAGE_V_STARTED)"
      [ "$(cut -d' ' -f2 "$DATA/stage_v/STAGE_V_STARTED")" = "$r" ] || refuse "STAGE_V_STARTED does not name this LOCK"
      [ -f "$SV/res01-v-view/selected.json" ] || refuse "no V view at $SV/res01-v-view"
      [ "$(ls "$SV/res01-v-view/tables" | wc -l)" = "$N_TARGET_TABLES" ] || refuse "V view tables != $N_TARGET_TABLES"
      [ -d "$SV/inputs" ] || refuse "no $SV/inputs"
      mkdir -p "$SV/run"
    fi
    ARGS=("${BASE[@]}"
      --ro-bind "$DATA/stage_v" "$DATA/stage_v"
      --bind "$SV" "$SV"
      --ro-bind "$SV/res01-v-view" "$SV/res01-v-view"
      --ro-bind "$SV/inputs" "$SV/inputs"
      --ro-bind "$SV/res01-v-view" "$RES01"
      --ro-bind "$RECEIPT" "$RECEIPT"
      "${ENVS[@]}" --setenv H335_STAGEV_ROOT "$SV" --setenv H317_SANDBOX stagev)
    ;;
  *)
    echo "usage: $0 [--print] unlock|v <command...>" >&2
    exit 2
    ;;
esac
ARGS+=(--chdir "$SRC" --unshare-net --die-with-parent)
if [ $PRINT = 1 ]; then
  printf '%s\n' bwrap "${ARGS[@]}" "$@"
  exit 0
fi
exec bwrap "${ARGS[@]}" "$@"
