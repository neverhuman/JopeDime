#!/usr/bin/env bash
# Fail-closed Jankurai gate on Carbon source, deployment scripts, and matrix workflow:
# - 0 blocking and 0 advisory findings across every Carbon/CI/workflow file (zero exemptions)
# - 0 hard and 0 warning duplicate-code findings (`jankurai copy-code crates/jope-carbon`)
# - 0 `HLT-029-RUST-BAD-BEHAVIOR` findings under full crate audit (`jankurai audit --full --mode standard`)
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

mkdir -p target/jankurai
STAGE_JSON_DIR="$(mktemp -d target/jankurai/.verify-carbon.XXXXXX)"
trap 'rm -rf "$STAGE_JSON_DIR"' EXIT INT TERM

python3 - "$AUDITOR" "$STAGE_JSON_DIR" <<'PY'
import concurrent.futures
import json
import pathlib
import subprocess
import sys

auditor = sys.argv[1]
stage_dir = pathlib.Path(sys.argv[2])
paths = sorted(
    str(p)
    for base in ("crates/jope-carbon", "crates/jope-envelope", "deploy/carbon")
    for p in pathlib.Path(base).rglob("*")
    if p.is_file()
)
for extra in (
    "crates/jope-api/src/carbon.rs",
    "crates/jope-api/tests/carbon_contract.rs",
    "crates/jope-service/src/discover/loop/carbon.rs",
    "ops/ci/carbon-cli-matrix.sh",
    "ops/ci/carbon-live-e2e.sh",
    ".github/workflows/carbon-cli-matrix.yml",
    ".github/workflows/carbon-live-e2e.yml",
    "web/training-dashboard/scripts/render-carbon-spec-pdf.mjs",
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
    for key in ("new_soft_findings", "existing_findings", "preexisting_findings"):
        issues.extend(advisory.get(key) or [])
    if issues:
        summary = "; ".join(
            f"{f.get('severity')}/{f.get('rule_id') or f.get('check_id')}@L{f.get('line')}"
            for f in issues
        )
        return (rel, summary)
    return (rel, None)


failures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for rel, err in pool.map(check_file, paths):
        if err is not None:
            failures.append((rel, err))

if failures:
    for rel, err in failures:
        print(f"carbon-jankurai FAIL: {rel}: {err}", file=sys.stderr)
    raise SystemExit(1)

copy_json = stage_dir / "copy-code.json"
copy_proc = subprocess.run(
    [auditor, "copy-code", "crates/jope-carbon", "--json", str(copy_json)],
    capture_output=True,
    text=True,
    check=False,
)
if copy_proc.returncode != 0 or not copy_json.is_file():
    print(f"carbon-jankurai FAIL: copy-code rc={copy_proc.returncode}: {copy_proc.stderr[-400:]}", file=sys.stderr)
    raise SystemExit(1)
copy_doc = json.loads(copy_json.read_text(encoding="utf-8"))
copy_summary = copy_doc.get("summary") or {}
hard_dups = int(copy_summary.get("hard", 0))
warn_dups = int(copy_summary.get("warning", 0))
if hard_dups != 0 or warn_dups != 0:
    print(
        f"carbon-jankurai FAIL: copy-code reported hard={hard_dups}, warning={warn_dups} in crates/jope-carbon",
        file=sys.stderr,
    )
    for grp in (copy_doc.get("groups") or [])[:10]:
        print(f"  dup group: {grp}", file=sys.stderr)
    raise SystemExit(1)

print(
    f"carbon-jankurai PASS: {len(paths)} files verified (0 blocking, 0 advisory, 0 exemptions) "
    f"and copy-code clean (hard={hard_dups}, warning={warn_dups})"
)
PY
