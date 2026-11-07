#!/bin/bash
# H412 steps (xbabe2). Usage: run_h412.sh STEP   with H412_CODE=~/h412-stage/code-TAG for every step after `stage`.
#   stage TAG  frozen read-only copy of this dir's H412 files, taken from origin/v1/next (outside the evidence tree:
#              the H317 guard denies that tree before the unlock)
#   lock       LOCK.json (code: staged files, harness binary + rlibs + the 5 entrant sources, h335_core + H317 pins;
#              inputs: BOARD.json) + create-once receipt
#   unlock     one-shot Stage V unlock (refuses unless the staged files equal origin/v1/next)
#   score | rows
set -euo pipefail
STEP=${1:?step}
REPO=/home/ubuntu/JopeDime
DATA=${H412_DATA:-/home/ubuntu/jope-data/h412/d91}
WDIR=research/flywheel_selfplay/h412_final_frontier_elo
FILES=(PREREG.md d91_h412.py frontier_audit_h412.py frontier_audit_h412.rs build_h412.sh run_h412.sh)
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H412_LOCK_sha256.txt
BOARD_ROOT=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1
BIN=/home/ubuntu/jope-data/h412/bin/frontier_audit_h412
BT=/home/ubuntu/jope-data/h412/build/frontier
H335C=/home/ubuntu/h335-stage/code
H317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$STEP" = stage ]; then
  TAG=${2:?tag}; OUT=/home/ubuntu/h412-stage/code-$TAG
  [ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
  mkdir -p "$OUT"
  for f in "${FILES[@]}"; do git -C "$REPO" show "origin/v1/next:$WDIR/$f" > "$OUT/$f"; done
  chmod 0555 "$OUT"/*.sh; chmod 0444 "$OUT"/*.py "$OUT"/*.md "$OUT"/*.rs
  (cd "$OUT" && sha256sum -- "${FILES[@]}" > CODE.sha256 && chmod 0444 CODE.sha256)
  chmod 0555 "$OUT"; echo "staged $OUT $(sha256sum "$OUT/CODE.sha256" | cut -d' ' -f1)"; exit 0
fi
CODE=${H412_CODE:?H412_CODE}
cd "$CODE" && sha256sum --quiet -c CODE.sha256
for f in "${FILES[@]}"; do
  cmp -s "$CODE/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") \
    || { echo "$f is not equal to origin/v1/next: refusing (publish the PREREG first)" >&2; exit 3; }
done
cmp -s "$CODE/frontier_audit_h412.rs" "$BT/harness/frontier_audit_h412.rs" || { echo "harness source != build tree" >&2; exit 3; }
cmp -s "$CODE/frontier_audit_h412.py" "$BT/harness/frontier_audit_h412.py" || { echo "wrapper != build tree" >&2; exit 3; }
PY=(nice -n 19 python3 -B "$CODE/d91_h412.py")
case $STEP in
  lock)
    [ -e "$DATA/stage_v/LOCK.json" ] && { echo "LOCK.json exists" >&2; exit 2; }
    mkdir -p "$DATA/stage_v"
    { for f in "${FILES[@]}"; do sha256sum "$CODE/$f"; done
      sha256sum "$BIN" "$BT"/lib/*.rlib "$BT"/src/*/*.rs "$BT"/src/*/*/*.rs 2>/dev/null || true
      sha256sum "$H335C/h335_core.py" "$H317/common_h317.py"; } \
      | jq -R -n '[inputs | split("  ") | {(.[1]): .[0]}] | add' > "$DATA/stage_v/code.json"
    # dev91-path inputs are verified after the unlock (score / rows), never by unlock_stage_v (the guard refuses them)
    sha256sum "$BOARD_ROOT/BOARD.json" | jq -R -n '[inputs | split("  ") | {(.[1]): .[0]}] | add' \
      > "$DATA/stage_v/inputs.json"
    jq -n --slurpfile c "$DATA/stage_v/code.json" --slurpfile i "$DATA/stage_v/inputs.json" \
      --arg utc "$(date -u +%FT%TZ)" --arg bin "$BIN" --arg bs "$(sha256sum "$BIN" | cut -d' ' -f1)" \
      '{wave:"H412", entrant:"grok4p7_xhigh", model_id:"h412.grok4p7_xhigh",
        entrant_source_sha256:"8e648cef93186aca9d10868ed1ed2a62407950169082aa1f4fab87af379664f8",
        entrant_commit:"1177c4f9c1", board:"dev91-audit-v1",
        board_sha256:"40cb4828cf5c9a0d2d9cc0f32b7900819c992ec9a58c5d18f83ac84b29fa4069", seed:42,
        cpu_limit_s:300, wall_limit_s:900, binary:$bin, binary_sha256:$bs, utc:$utc, code_sha256:$c[0],
        inputs_sha256:$i[0]}' > "$DATA/stage_v/LOCK.json"
    chmod 0444 "$DATA/stage_v/LOCK.json"
    S=$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    ( set -o noclobber; echo "$S  $DATA/stage_v/LOCK.json" > "$RECEIPT" ) && chmod 0444 "$RECEIPT"
    echo "LOCK $S" ;;
  unlock|score|rows) "${PY[@]}" "$STEP" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
