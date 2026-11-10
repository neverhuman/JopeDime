#!/bin/bash
# H317 Stage V bwrap sandbox for xbabe2 (PREREG §2.6; stage_v_mounts.md). sandbox_h317.sh is unchanged and remains the
# Gate H0 sandbox; this variant adds only what Stage V needs. No network in any mode.
# Usage: sandbox_h317_stagev.sh [--print] unlock|v <command...>
#   unlock  exactly the sandbox_h317.sh layout (TRAIN-only view at the canonical RES01, $H317_DATA rw, JopeDimeMicro ro
#           with assets/benchmark_data hidden, the H317 source ro) PLUS the create-once LOCK receipt bound read-only at
#           its canonical path (common_h317.LOCK_RECEIPT is fixed). No DL-Val byte and no Stage V root are visible.
#           Refused unless the receipt exists and names sha256($H317_DATA/stage_v/LOCK.json) and no STAGE_V_STARTED
#           marker exists yet (the python side also refuses a second unlock).
#   v       prep / stats, only after the unlock: the res-01 V view $SV/res01-v-view (TRAIN + the 91 DL-Val tables and
#           pools, the ORIGINAL selected.json, cpu_exact_fused.py) read-only at the canonical RES01 path; $SV rw with
#           res01-v-view/ and inputs/ re-bound read-only; $H317_DATA rw with $H317_DATA/stage_v re-bound read-only (LOCK,
#           marker); the receipt read-only; H317_STAGEV_ROOT=$SV. Refused unless STAGE_V_STARTED exists and names the
#           receipt's LOCK sha, and the V view holds 1,219 + 91 tables.
#   --print print the bwrap argv (one item per line) instead of running it; skips the precondition checks (review/tests).
# The guard refusal log stays the production default ($H317_DATA/guard_refusals.log) unless H317_GUARD_LOG is set.
set -euo pipefail
SRC=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
DATA=${H317_DATA:-/home/ubuntu/jope-data/h317-cl}
SV=${H317_STAGEV_ROOT:-/home/ubuntu/jope-data/h317-stagev}
VIEW=/home/ubuntu/jope-data/h317-cl/res01-train-view
MICRO=/home/ubuntu/JopeDimeMicro
RES01=/home/ubuntu/jope-data/res-01/input
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H317_LOCK_sha256.txt
N_TARGET_TABLES=1310  # 1,219 TRAIN + 91 DL-Val
PRINT=0
if [ "${1:-}" = "--print" ]; then PRINT=1; shift; fi
MODE=${1:-}
[ $# -ge 2 ] || { echo "usage: $0 [--print] unlock|v <command...>" >&2; exit 2; }
shift
refuse() { echo "sandbox_h317_stagev.sh $MODE refused: $*" >&2; exit 3; }
receipt_sha() { [ -s "$RECEIPT" ] || refuse "no LOCK receipt $RECEIPT"; cut -d' ' -f1 "$RECEIPT"; }
lock_sha() { sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1; }

BASE=(--ro-bind / /
  --tmpfs /home/ubuntu --tmpfs /tmp --tmpfs /mnt --tmpfs /media --tmpfs /srv
  --tmpfs /app --tmpfs /review --tmpfs /Users --tmpfs /opt/work
  --dev /dev --proc /proc
  --ro-bind /home/ubuntu/.local /home/ubuntu/.local
  --ro-bind "$SRC" "$SRC"
  --bind "$DATA" "$DATA" --tmpfs "$DATA/redteam"
  --ro-bind "$MICRO" "$MICRO" --tmpfs "$MICRO/assets/benchmark_data")
ENVS=(--setenv HOME /home/ubuntu --setenv H317_RES01 "$RES01" --setenv H317_DATA "$DATA"
  --setenv PYTHONDONTWRITEBYTECODE 1)
[ -n "${H317_GUARD_LOG:-}" ] && ENVS+=(--setenv H317_GUARD_LOG "$H317_GUARD_LOG")

case "$MODE" in
  unlock)
    if [ $PRINT = 0 ]; then
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
      "${ENVS[@]}" --setenv H317_STAGEV_ROOT "$SV" --setenv H317_SANDBOX stagev)
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
