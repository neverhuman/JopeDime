#!/bin/bash
# H415 PREREG §4 on xbabe1 (RTX 4090): the H317 H0.2-L recipe (= H410 C1), folds 1-4, keep models, TRAIN true targets.
# Fold 0 is the stored H410 C1 pair. Create-once output root; GPU taken under flock -n qualification.lock.
#   train_l_h415.sh   (run as a systemd-run --user unit on xbabe1)
set -euo pipefail
[ "$(hostname)" = xbabe1 ] || { echo "run on xbabe1" >&2; exit 2; }
CODE=/home/ubuntu/jope-data/h317-cl/code/h317_island10_multiscale_curriculum
NEU=/home/ubuntu/jope-data/h317-cl/neural
OUT=$NEU/runs-h415
LAUNCH=$NEU/launch/h415-L
[ "$(sha256sum "$CODE/rapid_hypothesis_lab.py" | cut -d' ' -f1)" = \
  31161becd922ecd25784f1f08b16b445c81f8f0a1d3940f06f41e2fd6a805e25 ] || { echo "rapid_hypothesis_lab.py pin" >&2; exit 4; }
[ ! -e "$OUT/H0.2-L" ] || { echo "$OUT/H0.2-L exists (create-once)" >&2; exit 3; }
mkdir -p "$LAUNCH" "$OUT"
printf '%s\n' '[{"arm_id": "H0.2-L", "scale": "L", "recipe": "opt1", "views": "A4", "subset": "A525", "mixup": false}]' \
  > "$LAUNCH/arms_h415_L.json"
cmp -s "$LAUNCH/arms_h415_L.json" "$NEU/launch/20261106T154906Z-h410/arms_h410_L.json" \
  || { echo "arms differ from H410 C1" >&2; exit 5; }
cd "$CODE"
export PYTHONDONTWRITEBYTECODE=1
echo "start $(date -u +%FT%TZ) H415 L folds 1-4"
exec flock -n /home/ubuntu/jope_plan/qualification.lock /usr/bin/python3 -u rapid_hypothesis_lab.py oof \
  --arms-json "$LAUNCH/arms_h415_L.json" --budget-json '{"steps": 3000}' --folds 1,2,3,4 --keep-models \
  --out-root "$OUT" --threads 2 --confirm-true-targets
