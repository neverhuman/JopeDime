#!/bin/bash
# H420 launcher on xbabe2 (systemd-run target): the H335 bubblewrap sandbox with a staged H420 code root.
#   run_h420_local.sh STAGE CPUSET ENTRY.py <args...>
set -euo pipefail
STAGE=${1:?STAGE}; CPUSET=${2:?CPUSET}; ENTRY=${3:?ENTRY}; shift 3
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$STAGE/mode")" = pub ] || [ "${APEX_DEV_OK:-0}" = 1 ] || { echo "scored runs use a pub stage" >&2; exit 3; }
echo "start $(date -u +%FT%TZ) stage $STAGE mode $(cat "$STAGE/mode") CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) cpus $CPUSET"
exec env H335_GROUP=AGG H335_CODE="$STAGE/code" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c "$CPUSET" "$STAGE/code/sandbox_h335.sh" \
  nice -n 19 python3 -u "$ENTRY" "$@"
