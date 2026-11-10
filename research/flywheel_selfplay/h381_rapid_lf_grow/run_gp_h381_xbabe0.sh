#!/bin/bash
# H381 GP launcher on xbabe0 (PREREG §2): the H335 docker sandbox with the staged H381 code root.
#   run_gp_h381_xbabe0.sh STAGE RUN CPUSET MEM <gp_h381.py launch args...>
# STAGE holds code/ (frozen H335 files + gp_h381.py) and CODE.sha256; RUN is the run dir name under h335-cl/h381/.
# Started as a transient user service: systemd-run --user --unit h381-<name> -p MemoryMax=.. <this script> ...
set -euo pipefail
STAGE=${1:?STAGE}; RUN=${2:?RUN}; CPUSET=${3:?CPUSET}; MEM=${4:?MEM}; shift 4
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=$CPUSET H335_MEMORY=$MEM
D=/home/ubuntu/jope-data/h335-cl/h381/$RUN
mkdir -p "$D"
echo "start $(date -u +%FT%TZ) stage $STAGE CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1) run $D cpus $CPUSET"
exec "$STAGE/code/sandbox_h335_docker.sh" python3 -u gp_h381.py launch --run-dir "$D" --cpus "$CPUSET" "$@"
