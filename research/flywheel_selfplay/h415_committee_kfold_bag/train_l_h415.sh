#!/bin/bash
# H415 PREREG §4: the H317 H0.2-L recipe (= H410 C1), new folds 1-4, keep models, TRAIN true targets. Fold 0 is the
# stored H410 C1 pair. The H317 FOLD_HOST pinning is kept: folds 1-2 on xbabe1 (RTX 4090), folds 3-4 on xbabe3
# (RTX 3090). The trainer reads the TRAIN-only table view h317-cl/res01-train (1,219 TRAIN tables + pools; its
# selected.json is the pinned 981960ad...), because the old res-01/input path no longer exists on these hosts.
# Create-once output root per host; the GPU is taken under flock -n qualification.lock.
#   train_l_h415.sh   (run as a systemd-run --user unit on xbabe1 or xbabe3)
set -euo pipefail
case $(hostname) in xbabe1) FOLDS=1,2 ;; xbabe3) FOLDS=3,4 ;; *) echo "run on xbabe1 or xbabe3" >&2; exit 2 ;; esac
CODE=/home/ubuntu/jope-data/h317-cl/code/h317_island10_multiscale_curriculum
NEU=/home/ubuntu/jope-data/h317-cl/neural
OUT=$NEU/runs-h415
LAUNCH=$NEU/launch/h415-L
VIEW=/home/ubuntu/jope-data/h317-cl/res01-train
[ "$(sha256sum "$CODE/rapid_hypothesis_lab.py" | cut -d' ' -f1)" = \
  31161becd922ecd25784f1f08b16b445c81f8f0a1d3940f06f41e2fd6a805e25 ] || { echo "rapid_hypothesis_lab.py pin" >&2; exit 4; }
[ "$(sha256sum "$VIEW/selected.json" | cut -d' ' -f1)" = \
  981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1 ] || { echo "TRAIN view selected.json pin" >&2; exit 4; }
[ "$(ls "$VIEW/tables" | wc -l)" = 1219 ] || { echo "TRAIN view tables != 1219" >&2; exit 4; }
[ ! -e "$OUT/H0.2-L" ] || { echo "$OUT/H0.2-L exists (create-once)" >&2; exit 3; }
mkdir -p "$LAUNCH" "$OUT"
printf '%s\n' '[{"arm_id": "H0.2-L", "scale": "L", "recipe": "opt1", "views": "A4", "subset": "A525", "mixup": false}]' \
  > "$LAUNCH/arms_h415_L.json"   # byte-identical to H410's launch/20261106T154906Z-h410/arms_h410_L.json
cd "$CODE"
export PYTHONDONTWRITEBYTECODE=1 H317_RES01=$VIEW
echo "start $(date -u +%FT%TZ) H415 L folds $FOLDS on $(hostname)"
exec flock -n /home/ubuntu/jope_plan/qualification.lock /usr/bin/python3 -u rapid_hypothesis_lab.py oof \
  --arms-json "$LAUNCH/arms_h415_L.json" --budget-json '{"steps": 3000}' --folds "$FOLDS" --keep-models \
  --out-root "$OUT" --threads 2 --confirm-true-targets
