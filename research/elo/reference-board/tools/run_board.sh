#!/bin/bash
# Score reference models on one board, each (model, context) exactly once. Resumable: a (model, cid) that already has
# a record in scores/<model>.jsonl is never run again (a failure is final too; re-scoring needs a NEW registry id).
# Usage: run_board.sh BOARD_ROOT JOBS MODEL_ID [MODEL_ID...]
#   e.g. systemd-run --user --unit refboard-dev91 -p CPUQuota=3200% -p MemoryMax=96G -p MemorySwapMax=0 \
#          <this> /home/ubuntu/jope-data/h335-stagev/refboard/dev91-audit-v1 4 catboost.v1 tabpfn_v3p5.v1
# Cores: nice 19, taskset $REFBOARD_CPUS (default 38-53,102-117). Per-job wall limit: models.json timeout_s; a job that
# hits it gets status "timeout", u = 0.
set -euo pipefail
BR=$(readlink -f "$1"); JOBS=$2; shift 2
TOOLS=$(dirname "$(readlink -f "$0")")
REG=$TOOLS/../models.json
CPUS=${REFBOARD_CPUS:-38-53,102-117}
# one driver per board: a second driver on the same board exits (no duplicate (model, context) records)
exec 9>>"$BR/.run_board.lock"
flock -n 9 || { echo "$(date -u +%FT%TZ) another run_board.sh holds $(basename "$BR"); nothing to do"; exit 0; }
for m in "$@"; do
  jq -e --arg m "$m" '.models[$m]' "$REG" >/dev/null || { echo "unknown model id $m" >&2; exit 2; }
  py=$(jq -r --arg m "$m" '.models[$m].python' "$REG")
  "$TOOLS/sandbox_refboard.sh" "$BR" "$py" "$TOOLS/score_reference.py" check "$m" >/dev/null \
    || { echo "preflight (versions + weights sha) failed for $m" >&2; exit 3; }
done
worklist=$(mktemp -p "$BR" .todo.XXXXXX)
for m in "$@"; do
  touch "$BR/scores/$m.jsonl"
  jq -r '.contexts[].cid' "$BR/BOARD.json" | grep -vxFf <(jq -r .cid "$BR/scores/$m.jsonl") | sed "s/^/$m /" || true
done > "$worklist"
echo "$(date -u +%FT%TZ) board=$(basename "$BR") todo=$(wc -l < "$worklist") jobs=$JOBS models=$*"
export BR TOOLS REG CPUS
xargs -a "$worklist" -P "$JOBS" -L 1 bash -c '
  m=$0; c=$1
  py=$(jq -r --arg m "$m" ".models[\$m].python" "$REG"); to=$(jq -r --arg m "$m" ".models[\$m].timeout_s" "$REG")
  timeout -k 15 "$to" nice -n 19 taskset -c "$CPUS" "$TOOLS/sandbox_refboard.sh" "$BR" "$py" \
    "$TOOLS/score_reference.py" one "$BR" "$m" "$c" >/dev/null 2>>"$BR/scores/$m.stderr.log"; rc=$?
  if ! jq -e --arg c "$c" "select(.cid==\$c)" "$BR/scores/$m.jsonl" >/dev/null 2>&1; then
    jq -cn --arg c "$c" --arg m "$m" --argjson rc "$rc" --arg b "$(jq -r .board "$BR/BOARD.json")" \
      "{board:\$b,cid:\$c,model_id:\$m,status:(if \$rc==124 or \$rc==137 then \"timeout\" else \"failed\" end),utility:0.0,error:(\"exit \"+(\$rc|tostring)),harness_version:\"refboard-score-v1\"}" \
      >> "$BR/scores/$m.jsonl"
  fi'
mv "$worklist" "$BR/.todo.done.$(date -u +%Y%m%dT%H%M%SZ)"
for m in "$@"; do
  echo "$m: $(jq -r .status "$BR/scores/$m.jsonl" | sort | uniq -c | tr '\n' ' ')"
done
