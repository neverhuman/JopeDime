#!/bin/bash
# H417 launcher on xbabe0 (systemd-run target): the H335 docker sandbox with a staged APEX code root.
#   run_apex_xbabe0.sh STAGE CPUSET MEM ENTRY.py <args...>
set -euo pipefail
STAGE=${1:?STAGE}; CPUSET=${2:?CPUSET}; MEM=${3:?MEM}; ENTRY=${4:?ENTRY}; shift 4
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
[ "$(cat "$STAGE/mode")" = pub ] || [ "${APEX_DEV_OK:-0}" = 1 ] || { echo "scored runs use a pub stage" >&2; exit 3; }
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=$CPUSET H335_MEMORY=$MEM
echo "start $(date -u +%FT%TZ) stage $STAGE mode $(cat "$STAGE/mode") CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) cpus $CPUSET"
exec "$STAGE/code/sandbox_h335_docker.sh" nice -n 19 python3 -u "$ENTRY" "$@"
