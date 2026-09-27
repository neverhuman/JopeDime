#!/usr/bin/env bash
# One independently bounded family; receipt.py records the original failing exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
EVIDENCE=${JOPE_LANE_EVIDENCE:?fresh receipt directory required}
lane=${1:?family required}
# Repeat the host reserve check between families; running out of disk is a failure.
available=$(df -B1 --output=avail "$EVIDENCE" | tail -1)
(( available >= 50000000000 )) || { echo 'unavailable: less than 50 GB disk reserve'; exit 78; }
case $lane in
  full-build-reuse)
    cargo metadata --locked --no-deps --format-version 1 > "$EVIDENCE/metadata.json"
    cargo build --locked -p jope-paths --bin jope-units-render --message-format=json-render-diagnostics > "$EVIDENCE/artifacts.jsonl"
    cargo run -q --locked -p jope-ci -- cargo-artifact select --metadata "$EVIDENCE/metadata.json" --messages "$EVIDENCE/artifacts.jsonl" --package jope-paths --bin jope-units-render > "$EVIDENCE/executable.txt"
    python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['JOPE_LANE_EVIDENCE'])
path = Path((root / 'executable.txt').read_text().strip())
messages = [json.loads(line) for line in (root / 'artifacts.jsonl').read_text().splitlines() if line.startswith('{')]
selected = [m for m in messages if m.get('executable') == str(path)]
assert len(selected) == 1
record = {'executable': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'fresh': selected[0]['fresh'], 'target_directory': json.loads((root / 'metadata.json').read_text())['target_directory']}
(root / 'reuse.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
PY
    ;;
  full-doctests)
    cargo test --workspace --doc --no-fail-fast --locked
    ;;
  full-python)
    (cd python && uv sync --locked --extra dev --extra cpu)
    # The maintained families are explicit in coverage.json; research scripts are separate.
    mapfile -t families < <(python3 - <<'PY'
import json
mapping = json.load(open('deploy/ci/coverage.json'))
for name, item in sorted(mapping['items'].items()):
    if name.startswith('python:python/tests/') and 'full-python' in item['lanes']:
        print(name.removeprefix('python:python/'))
PY
    )
    ((${#families[@]} > 0)) || { echo 'no maintained Python families selected'; exit 1; }
    printf '%s\n' "${families[@]}" > "$EVIDENCE/families.txt"
    (cd python && bash ../deploy/ci/pytest-evidence.sh uv run --no-sync pytest -q -m 'not xbabe2' -p no:cacheprovider "${families[@]}")
    python3 deploy/ci/training-proof.py python --junit "$EVIDENCE/junit.xml"
    ;;
  full-inputs|full-e2e|full-emitters)
    case $lane in
      full-inputs) fast=inputs ;;
      full-e2e) fast=e2e-smoke ;;
      full-emitters) fast=emitters ;;
    esac
    bash deploy/ci/fast-lane.sh "$fast"
    ;;
  full-default|full-scratch|full-policy|full-s3-offline|full-neural)
    case $lane in
      full-default) selection=(--workspace --features "jope-cli/asm,jope-cli/qualified-training-context") ;;
      full-scratch) selection=(-p feat-math -p jope-parallel -p jope-dime-regressor --features "feat-math/gpr-scratch-reuse,jope-parallel/gpr-scratch-reuse,jope-dime-regressor/gpr-scratch-reuse") ;;
      full-policy) selection=(-p jope-policy --features "testing,context-producer") ;;
      full-s3-offline) selection=(-p jope-arena --features s3) ;;
      full-neural)
        # Discovery receipt (never overwritten) for the unavailable or discovered SDK state.
        export JOPE_NEURAL_ENV_RECEIPT="$EVIDENCE/neural-discovery.json"
        # SDK/runtime availability is measured independently; no GPU device/credentials.
        # shellcheck disable=SC1091
        source deploy/ci/neural-env.sh || { echo 'unavailable: neural SDK environment'; exit 78; }
        selection=(-p jope-parallel --features neural)
        ;;
    esac
    cargo run -q --locked -p jope-ci -- cargo-artifact nextest-config --output "$EVIDENCE/nextest.toml"
    # Cargo's effective target still owns the nextest store; each report is run-specific.
    python3 - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['JOPE_LANE_EVIDENCE'])
path = root / 'nextest.toml'
text = path.read_text()
needle = 'path = "junit-main.xml"'
assert text.count(needle) == 1
path.write_text(text.replace(needle, 'path = ' + json.dumps(str(root / 'junit.xml'))))
PY
    cargo nextest list --config-file "$EVIDENCE/nextest.toml" -P main --locked "${selection[@]}" --message-format json > "$EVIDENCE/collection.json"
    case $lane in
      full-default|full-scratch)
        bash deploy/ci/check-test-routing.sh cpu "$EVIDENCE/collection.json"
        bash deploy/ci/check-test-routing.sh host "$EVIDENCE/collection.json"
        ;;
      full-neural) bash deploy/ci/check-test-routing.sh neural "$EVIDENCE/collection.json" ;;
    esac
    touch "$EVIDENCE/nextest-started"
    cargo nextest run --config-file "$EVIDENCE/nextest.toml" -P main --locked --test-threads "${JOPE_CPU_TEST_THREADS:-16}" "${selection[@]}"
    case $lane in
      full-default) python3 deploy/ci/training-proof.py rust --collection "$EVIDENCE/collection.json" --junit "$EVIDENCE/junit.xml" ;;
      full-policy) python3 deploy/ci/training-proof.py policy --collection "$EVIDENCE/collection.json" --junit "$EVIDENCE/junit.xml" ;;
    esac
    ;;
  *) echo "unknown full CPU family: $lane" >&2; exit 2 ;;
esac
