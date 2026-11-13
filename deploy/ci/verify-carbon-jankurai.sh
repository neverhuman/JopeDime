#!/usr/bin/env bash
# Fail-closed Jankurai advisory gate on Carbon source, deployment scripts, and matrix workflow.
set -euo pipefail
cd "$(dirname "$0")/../.."

AUDITOR="${JANKURAI_BIN:-target/jankurai/jankurai-1.7.6-x86_64-unknown-linux-gnu/jankurai}"
if [[ ! -x "$AUDITOR" ]]; then
  bash deploy/ci/install-jankurai.sh
  AUDITOR="target/jankurai/jankurai-1.7.6-x86_64-unknown-linux-gnu/jankurai"
fi
[[ "$("$AUDITOR" --version)" == "jankurai 1.7.6" ]] || {
  echo "expected pinned jankurai 1.7.6, got: $("$AUDITOR" --version)" >&2
  exit 2
}

python3 - "$AUDITOR" <<'PY'
import concurrent.futures
import json
import pathlib
import subprocess
import sys

auditor = sys.argv[1]
paths = sorted(
    str(p)
    for base in ("crates/jope-carbon", "deploy/carbon")
    for p in pathlib.Path(base).rglob("*")
    if p.is_file()
)
for extra in (
    "crates/jope-api/src/carbon.rs",
    "crates/jope-api/tests/carbon_contract.rs",
    ".github/workflows/carbon-cli-matrix.yml",
):
    if pathlib.Path(extra).is_file():
        paths.append(extra)


def check_file(rel: str):
    proc = subprocess.run(
        [auditor, "audit-file", "--path", rel, "--mode", "advisory", ".", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        return (rel, f"rc={proc.returncode}: {proc.stderr[-300:]}")
    try:
        decision = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return (rel, f"invalid JSON: {exc}")
    if decision.get("verdict") != "pass" or decision.get("exit_code") != 0:
        return (rel, f"verdict={decision.get('verdict')} summary={decision.get('summary')}")
    blocking = decision.get("blocking", {})
    advisory = decision.get("advisory", {})
    issues = []
    for key in ("new_hard_findings", "worsened_findings", "always_block_findings"):
        issues.extend(blocking.get(key) or [])
    issues.extend(advisory.get("new_soft_findings") or [])
    if issues:
        summary = "; ".join(
            f"{f.get('severity')}/{f.get('rule_id') or f.get('check_id')}@L{f.get('line')}"
            for f in issues
        )
        return (rel, summary)
    return (rel, None)


failures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for rel, err in pool.map(check_file, paths):
        if err is not None:
            failures.append((rel, err))

if failures:
    for rel, err in failures:
        print(f"carbon-jankurai FAIL: {rel}: {err}", file=sys.stderr)
    raise SystemExit(1)

print(f"carbon-jankurai PASS: {len(paths)} files verified with 0 new hard/soft findings")
PY
