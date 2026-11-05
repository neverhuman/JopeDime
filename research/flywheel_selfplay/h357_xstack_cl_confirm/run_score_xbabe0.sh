#!/bin/bash
# H357 Amendment 2 on xbabe0: phase_c_h357.py arms + decide on the host and in the docker sandbox that computed the
# stored GP stacks, so the X0 harness check is bit-exact (the frozen Frank-Wolfe stacker is not bit-reproducible
# across CPUs on near-degenerate stacks). Launched by `run_c_h357.sh score0` as
#   systemd-run --user --unit h357-C-score -p MemoryMax=40G -p MemorySwapMax=0 <gpc>/code/run_score_xbabe0.sh
set -euo pipefail
STAGE=$(dirname "$(dirname "$(readlink -f "$0")")")
(cd "$STAGE/code" && sha256sum --quiet -c ../CODE.sha256)
[ -s "$STAGE/PREREG_ON_V1NEXT" ] || { echo "no PREREG_ON_V1NEXT marker" >&2; exit 3; }
DATA=/home/ubuntu/jope-data/h335-cl
SC=$DATA/h357/score
SRC=/home/ubuntu/jope_plan/evidence/H335/source/research/flywheel_selfplay/h335_aggressive_locked_elo
(cd "$SC/code" && sha256sum --quiet -c ../code.sha256)
(cd "$SC/lib" && sha256sum --quiet -c ../lib.sha256)
export H335_GROUP=AGG H335_STAGE=$STAGE H335_CPUSET=16-31,80-95 H335_MEMORY=32g
echo "start $(date -u +%FT%TZ) stage $STAGE CODE.sha256 $(sha256sum "$STAGE/CODE.sha256" | cut -d' ' -f1)"
exec "$STAGE/code/sandbox_h335_docker.sh" env H357_DATA="$SC" H354_H335_STATS="$SRC/h335_stats.py" \
  bash -c "python3 -B $SC/code/phase_c_h357.py arms --jobs 16 && python3 -B $SC/code/phase_c_h357.py decide"
