#!/usr/bin/env bash
# One independently bounded family; receipt.py records the original failing exit.
set -euo pipefail
cd "$(dirname "$0")/../.."
EVIDENCE=${JOPE_LANE_EVIDENCE:?fresh receipt directory required}
lane=${1:?family required}
export JOPE_FULL_FAMILY=$lane
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
  full-inputs)
    # Build the two input-check harnesses in one Cargo invocation: their links
    # can overlap instead of consuming the 300s family budget in serial builds.
    # The fast lane still executes both original filters and both fixture checks.
    cargo test --locked -p jope-dime-regressor --lib --bin jope-dime-regressor --no-run
    bash deploy/ci/fast-lane.sh inputs
    ;;
  full-e2e)
    bash deploy/ci/fast-lane.sh e2e-smoke
    ;;
  full-emitters)
    JOPE_FULL_EMITTERS_PARALLEL=1 bash deploy/ci/fast-lane.sh emitters
    ;;
  full-default|full-scratch|full-policy|full-s3-offline|full-neural)
    case $lane in
      full-default) selection=(--workspace --features "jope-cli/asm,jope-cli/qualified-training-context") ;;
      full-scratch) selection=(-p feat-math -p jope-parallel -p jope-dime-regressor --features "feat-math/gpr-scratch-reuse,jope-parallel/gpr-scratch-reuse,jope-dime-regressor/gpr-scratch-reuse") ;;
      full-policy) selection=(-p jope-policy --features "testing,context-producer") ;;
      full-s3-offline) selection=(-p jope-arena --features s3) ;;
      full-neural)
        # Discover the SDK before either build so its CXX setting is stable:
        # changing it between collections rebuilds feat-math and its dependants.
        # SDK/runtime availability is measured independently; no GPU device/credentials.
        export JOPE_NEURAL_ENV_RECEIPT="$EVIDENCE/neural-discovery.json"
        # shellcheck disable=SC1091
        source deploy/ci/neural-env.sh || { echo 'unavailable: neural SDK environment'; exit 78; }
        # A lifecycle-only child exercises the real authority in a same-source
        # default-feature image. Prebuild it outside the unchanged start deadline.
        cargo nextest list --locked -p jope-parallel --lib --no-default-features --message-format json > "$EVIDENCE/synthetic-executor-collection.json"
        export JOPE_SYNTHETIC_EXECUTABLE
        JOPE_SYNTHETIC_EXECUTABLE="$(python3 - <<'PYBUILD'
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['JOPE_LANE_EVIDENCE'])
collection = json.loads((root / 'synthetic-executor-collection.json').read_text())
suite = collection['rust-suites']['jope-parallel']
assert suite['kind'] == 'lib' and suite['package-name'] == 'jope-parallel'
assert 'eval_registry::authority::tests::synthetic_executor' in suite['testcases']
path = Path(suite['binary-path'])
assert path.is_file()
record = {'executable': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'features': [], 'feature_selection': 'no-default-features',
          'source_sha': os.environ.get('JOPE_SOURCE_SHA'),
          'scope': 'same-source synthetic child; collection only, no test execution'}
(root / 'synthetic-executor.json').write_text(json.dumps(record, indent=2) + '\n')
print(path)
PYBUILD
)"
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
text = text.replace(needle, 'path = ' + json.dumps(str(root / 'junit.xml')))
# Start the long golden tail first, retaining every test and its original deadline.
# Only this run's full-main profile changes; the checked-in PR/default profiles do not.
text += '''
[[profile.main.overrides]]
filter = "package(feat-math) & binary(=golden) & (test(=fast_profile_portfolio_head) | test(=golden_ridge_gcv_screen_wide) | test(=golden_portfolio_core))"
priority = 100
'''
if os.environ.get('JOPE_FULL_FAMILY') == 'full-neural':
    # Authority checks hash all mapped neural libraries at each start. Limit
    # competing full-runtime reads, retaining every verification and test.
    # Run the three-control receipt first, alone, rather than as a contended tail.
    text += '''
[test-groups.full-neural-runtime-proof]
max-threads = 4

[[profile.main.overrides]]
filter = "package(jope-parallel) & test(=teaching_store::read_once_tests::a_search_receipt_is_validated_from_the_bytes_its_ledger_commits)"
priority = 100
threads-required = "num-test-threads"
test-group = "full-neural-runtime-proof"

[[profile.main.overrides]]
filter = "package(jope-parallel) & (test(teaching_store::read_once_tests::) | test(teaching_store::admission_tests::) | test(eval_registry::authority::) | test(eval_registry::witness::))"
priority = 90
test-group = "full-neural-runtime-proof"
'''
path.write_text(text)
PY
    cargo nextest list --config-file "$EVIDENCE/nextest.toml" -P main --locked "${selection[@]}" --message-format json > "$EVIDENCE/collection.json"
    case $lane in
      full-default|full-scratch)
        bash deploy/ci/check-test-routing.sh cpu "$EVIDENCE/collection.json"
        bash deploy/ci/check-test-routing.sh host "$EVIDENCE/collection.json"
        ;;
      full-neural) bash deploy/ci/check-test-routing.sh neural "$EVIDENCE/collection.json" ;;
    esac
    if [ "$lane" = full-default ]; then
      # bounded_smartcluster runs this workspace's own scqd and never builds it inside a test
      # (W7-D5). The workspace test build above produces it (jain-smartcluster-daemon's
      # integration tests make Cargo build its binary); refuse to run the tests without it.
      scqd="$(cargo run -q --locked -p jope-ci -- cargo-artifact target-dir)/debug/scqd"
      [ -x "$scqd" ] || { echo "scqd was not built by the workspace test build: $scqd"; exit 1; }
      sha256sum "$scqd" > "$EVIDENCE/scqd.sha256"
    fi
    touch "$EVIDENCE/nextest-started"
    cargo nextest run --config-file "$EVIDENCE/nextest.toml" -P main --locked --test-threads "${JOPE_CPU_TEST_THREADS:-16}" "${selection[@]}"
    case $lane in
      full-default) python3 deploy/ci/training-proof.py rust --collection "$EVIDENCE/collection.json" --junit "$EVIDENCE/junit.xml" ;;
      full-policy) python3 deploy/ci/training-proof.py policy --collection "$EVIDENCE/collection.json" --junit "$EVIDENCE/junit.xml" ;;
    esac
    ;;
  *) echo "unknown full CPU family: $lane" >&2; exit 2 ;;
esac
