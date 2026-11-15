#!/usr/bin/env bash
# Prepares another host to run H458 data workers (`train_h458.py --remote-workers host:n`): copies the staged runtime
# copy, the three dependency stages and the family pool to the same absolute paths, then verifies every byte
# there (CODE.sha256, dependency pins, pool data sha256). Run on the training host. Copies no checkpoint, no
# evaluation bank and no label file other than the pool itself, which stays inside the H458 quarantine path.
# usage: sync_remote_h458.sh <host> [<stage dir>]
set -euo pipefail
HOST="$1"
STAGE="${2:-$(cd "$(dirname "$0")" && pwd)}"
POOL=/home/ubuntu/jope-data/h458-quarantine/family_pool_v1
DEPS="/home/ubuntu/h450-stage/pub-r6 /home/ubuntu/h450-stage/fleet-r10 /home/ubuntu/h448-stage/pub-r1"
[ -f "$STAGE/CODE.sha256" ] || { echo "$STAGE is not a staged runtime copy"; exit 2; }
for D in "$STAGE" $DEPS "$POOL"; do
  ssh -o BatchMode=yes "$HOST" "mkdir -p '$D'"
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$D/" "$HOST:$D/"
done
for D in $DEPS; do  # each dependency stage lists its files relative to its code folder
  ssh -o BatchMode=yes "$HOST" "cd '$D/code' && sha256sum --quiet -c ../CODE.sha256"
done
ssh -o BatchMode=yes "$HOST" "cd '$STAGE' && sha256sum --quiet -c CODE.sha256 && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 JOPE_REF_CODE= /usr/bin/python3 -B -c '
import paths_h458 as P
P.setup_paths()
import train_h458  # noqa: F401  (imports every dependency the workers use)
from family_pool_h458 import FamilyPool
pool = FamilyPool(P.POOL_DIR, verify_data=True)
deps = P.verify_deps(strict=True)
print(\"remote ok:\", len(deps), \"dependency files pinned, pool\", pool.sha256[:16], \"families\", len(pool.families))
'"
echo "synced $STAGE to $HOST"
