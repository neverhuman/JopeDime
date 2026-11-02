#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
evidence="$GITHUB_WORKSPACE/target/ci-evidence/$GITHUB_RUN_ID/$GITHUB_RUN_ATTEMPT/$GITHUB_JOB"
mkdir -p "$evidence"
case "${1:-}" in
  build)
    git rev-parse HEAD > "$evidence/source-sha.txt"
    nice -n 10 cargo build --locked -p jope-appliance --no-default-features --features client 2>&1 | tee "$evidence/lite-build.log"
    nice -n 10 cargo tree --locked -p jope-appliance --no-default-features --features client -e normal --prefix none --format '{p}' > "$evidence/lite-cargo-tree.txt"
    python3 - "$evidence/lite-cargo-tree.txt" <<'PY'
import pathlib
import sys
packages = {line.split()[0] for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.split()}
banned = {"jope-service", "candle-core", "tokio", "dope-core", "onig", "onig_sys", "esaxx-rs", "tch", "torch-sys", "rusqlite", "libsqlite3-sys", "pyo3", "openssl-sys", "aws-lc-sys", "syntect"}
found = sorted(packages & banned)
if found:
    raise SystemExit(f"lite normal graph contains forbidden packages: {found}")
if "jope-client" not in packages:
    raise SystemExit("lite normal graph does not contain the HTTP client")
print("lite graph: HTTP client present; engine and forbidden packages absent")
PY
    ;;
  contracts)
    # The engine feature is confined to the test process; the shipped lite executable stays client-only.
    nice -n 10 cargo nextest run -P pr --locked -p jope-client --features engine --test contract --build-jobs 8 --test-threads 1 --no-tests=fail 2>&1 | tee "$evidence/transport-contract.log"
    python3 - "$evidence" <<'PY'
import pathlib
import re
import shutil
import sys
import xml.etree.ElementTree as ET
evidence = pathlib.Path(sys.argv[1])
source = pathlib.Path("target/nextest/pr/junit.xml")
log = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", (evidence / "transport-contract.log").read_text())
runs = re.findall(r"Nextest run ID ([0-9a-f-]{36})", log)
summary = re.findall(r"Summary[^\n]*?(\d+) tests? run: (\d+) passed, (\d+) skipped", log)
report = ET.parse(source).getroot()
if len(runs) != 1 or report.get("uuid") != runs[0]:
    raise SystemExit("JUnit report does not match this nextest run")
cases = list(report.iter("testcase"))
if not cases or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped")):
    raise SystemExit("contract suite requires nonzero executed cases with no failures or skips")
if len(summary) != 1 or tuple(map(int, summary[0])) != (len(cases), len(cases), 0):
    raise SystemExit("contract summary must match the executed JUnit cases with zero skips")
shutil.copyfile(source, evidence / "contract-junit.xml")
print(f"transport contract: {len(cases)} executed cases")
PY
    ;;
  *) echo 'usage: lite-client.sh build|contracts' >&2; exit 2 ;;
esac
