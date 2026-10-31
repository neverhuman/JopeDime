#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
plan_json="$(python3 "$script_dir/runner.py" --dry-run --json)"

python3 - "$plan_json" <<'PY'
import json
import sys

plan = json.loads(sys.argv[1])
assert plan["schema"] == "jope.rd-breakthrough.h69-w7-plan/v1"
assert plan["wave"] == "W7"
assert plan["claim"] == "H69"
assert len(plan["arms"]) == 3
assert plan["seeds"] == [701, 702, 703]
assert plan["family_folds"] == 5
assert plan["acquisition_rounds"] == 20
assert plan["terminal_updates"] == 40
assert plan["training_context_rounds_per_arm_seed"] == 100
assert plan["planned_fits"] == (
    len(plan["arms"])
    * len(plan["seeds"])
    * plan["fits_per_context_round"]
    * plan["training_context_rounds_per_arm_seed"]
)
assert plan["planned_fits"] == 3600
assert plan["planned_fits"] <= plan["fit_ceiling"] == 4800
assert plan["execution_admitted"] is False
assert plan["data_access"] == "none"
assert plan["model_fits_executed"] == 0
assert plan["numerical_results_claimed"] is False
PY

if python3 "$script_dir/runner.py" --json >/dev/null 2>&1; then
    echo "runner accepted invocation without required --dry-run" >&2
    exit 1
fi

echo "H69 runner contract: PASS (offline plan only; no study execution)"
