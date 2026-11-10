#!/bin/bash
# H369 steps (xbabe2). Usage: run_h369.sh STEP   with H369_CODE=~/h369-stage/code-TAG for every step after `stage`.
#   stage TAG  frozen read-only copy: H354/H364 files from origin/v1/next + this dir's H369 files
#   lock       LOCK.json (code / H317 pins / GP pull list / BOARD.json sha256) + create-once receipt
#   unlock     one-shot Stage V unlock (refuses unless PREREG.md + d91_h369.py + run_h369.sh equal origin/v1/next)
#   prepare | align | lib | score | rows
set -euo pipefail
STEP=${1:?step}
REPO=/home/ubuntu/JopeDime
DATA=${H369_DATA:-/home/ubuntu/jope-data/h369-d91}
WDIR=research/flywheel_selfplay/h369_dev91_stack_entrants
SRC=/home/ubuntu/jope_plan/evidence/H369/source/$WDIR
H317=/home/ubuntu/jope_plan/evidence/H317/source/research/flywheel_selfplay/h317_island10_multiscale_curriculum
RECEIPT=/home/ubuntu/jope_plan/ns/receipts/H369_LOCK2_sha256.txt  # attempt 1: H369_LOCK_sha256.txt (PREREG section 6)
BOARD_ROOT=/home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$STEP" = stage ]; then
  TAG=${2:?tag}; OUT=/home/ubuntu/h369-stage/code-$TAG
  [ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 2; }
  mkdir -p "$OUT"
  for f in h354_xstack_cross_paradigm/h354_core.py h354_xstack_cross_paradigm/lib_oof_h354.py \
           h354_xstack_cross_paradigm/sandbox_h354.sh h364_xstack_two_level/h364_core.py \
           h364_xstack_two_level/DESIGN.json; do
    git -C "$REPO" show "origin/v1/next:research/flywheel_selfplay/$f" > "$OUT/$(basename "$f")"
  done
  cp "$SRC"/d91_h369.py "$SRC"/run_h369.sh "$SRC"/PREREG.md "$OUT/"
  chmod 0555 "$OUT"/*.sh; chmod 0444 "$OUT"/*.py "$OUT"/*.md "$OUT"/*.json
  (cd "$OUT" && sha256sum -- *.py *.sh *.md *.json > CODE.sha256 && chmod 0444 CODE.sha256)
  chmod 0555 "$OUT"; echo "staged $OUT $(sha256sum "$OUT/CODE.sha256" | cut -d' ' -f1)"; exit 0
fi
CODE=${H369_CODE:?H369_CODE}
cd "$CODE" && sha256sum --quiet -c CODE.sha256
case $STEP in unlock|prepare|align|lib|score|rows)
  for f in PREREG.md d91_h369.py run_h369.sh; do
    cmp -s "$CODE/$f" <(git -C "$REPO" show "origin/v1/next:$WDIR/$f") \
      || { echo "$f is not equal to origin/v1/next: refusing (publish the PREREG first)" >&2; exit 3; }
  done ;;
esac
PY=(taskset -c 38-53,102-117 nice -n 19 python3 -B "$CODE/d91_h369.py")
case $STEP in
  lock)
    [ -e "$DATA/stage_v/LOCK.json" ] && { echo "LOCK.json exists" >&2; exit 2; }
    mkdir -p "$DATA/stage_v"
    { for f in "$CODE"/*.py "$CODE"/*.sh "$CODE"/*.md "$CODE"/*.json; do sha256sum "$f"; done
      for f in common_h317.py gp_engine_h317.py gp_grammar_h317.py stacking_h317.py; do sha256sum "$H317/$f"; done
      sha256sum "$DATA/logs/xbabe0.sha256"; } \
      | jq -R -n '[inputs | split("  ") | {(.[1]): .[0]}] | add' > "$DATA/stage_v/code.json"
    # dev91-path inputs are verified after the unlock (prepare), never by unlock_stage_v (the guard refuses them)
    sha256sum "$BOARD_ROOT/BOARD.json" | jq -R -n '[inputs | split("  ") | {(.[1]): .[0]}] | add' \
      > "$DATA/stage_v/inputs.json"
    jq -n --slurpfile c "$DATA/stage_v/code.json" --slurpfile i "$DATA/stage_v/inputs.json" \
      --arg utc "$(date -u +%FT%TZ)" --arg design "$(jq -r .chosen_key "$CODE/DESIGN.json")" \
      '{wave:"H369", entrant:"h364.XSTACK_2L", design_key:$design, design_commit:"f052a59f35",
        board:"dev91-audit-v1", seeds:[335101,335102], stage:"V_AGG", utc:$utc, code_sha256:$c[0],
        inputs_sha256:$i[0], lock_attempt:2}' \
      > "$DATA/stage_v/LOCK.json"
    chmod 0444 "$DATA/stage_v/LOCK.json"
    S=$(sha256sum "$DATA/stage_v/LOCK.json" | cut -d' ' -f1)
    ( set -o noclobber; echo "$S  $DATA/stage_v/LOCK.json" > "$RECEIPT" ) && chmod 0444 "$RECEIPT"
    echo "LOCK $S" ;;
  unlock|prepare|align) "${PY[@]}" "$STEP" ;;
  lib)
    cat > "$DATA/run_lib.sh" <<EOS
#!/bin/bash
set -euo pipefail
cd "$CODE" && sha256sum --quiet -c CODE.sha256
echo "CODE.sha256 \$(sha256sum CODE.sha256 | cut -d' ' -f1) start \$(date -u +%FT%TZ)"
taskset -c 38-53,102-117 "$CODE/sandbox_h354.sh" "$DATA/d" python3 -B "$CODE/lib_oof_h354.py" \\
  --inputs "$DATA/d/inputs" --out "$DATA/d/lib" --jobs 10 --audit stored \\
  --learners catboost.v1,lightgbm.v1,xgboost.v1,random_forest.v1
echo "done \$(date -u +%FT%TZ)"
EOS
    chmod +x "$DATA/run_lib.sh"
    systemd-run --user --unit=h369-lib --collect -p MemoryMax=48G -p MemorySwapMax=0 -p Nice=19 -p CPUQuota=3200% \
      --setenv=PYTHONDONTWRITEBYTECODE=1 -p StandardOutput=append:"$DATA/logs/lib.log" \
      -p StandardError=append:"$DATA/logs/lib.err" "$DATA/run_lib.sh" ;;
  score) "${PY[@]}" score --jobs 12 ;;
  rows) "${PY[@]}" rows --out "$DATA/h364.XSTACK_2L.jsonl" ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
