#!/bin/bash
# H410 TRAIN launcher on xbabe0: the H335 docker sandbox with a staged H410 code root (systemd-run target).
#   run_h410_xbabe0.sh STAGE CPUSET MEM <train_h410.py args...>
set -euo pipefail
STAGE=${1:?STAGE}; CPUSET=${2:?CPUSET}; MEM=${3:?MEM}; shift 3
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=$CPUSET H335_MEMORY=$MEM
echo "start $(date -u +%FT%TZ) stage $STAGE CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) cpus $CPUSET"
exec "$STAGE/code/sandbox_h335_docker.sh" python3 -u train_h410.py "$@"
