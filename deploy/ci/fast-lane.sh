#!/usr/bin/env bash
# One receipt wraps each lane; a command failure stops that lane and remains failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
EVIDENCE=${JOPE_LANE_EVIDENCE:?receipt wrapper must allocate fresh evidence}
case "${1:?lane required}" in
  changes)
    python3 deploy/ci/ci-plan.py --output "$EVIDENCE/plan.json"
    ;;
  hygiene)
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # Install linters
set -euo pipefail
# The runner image ships shellcheck, actionlint and gitleaks pinned by sha256. It has
# no sudo and a read-only root filesystem, so assert they are present rather than
# installing them; a missing tool is an image defect and should fail loudly here.
for t in shellcheck actionlint gitleaks; do
  command -v "$t" >/dev/null || { echo "runner image is missing $t" >&2; exit 1; }
  printf '%-12s %s\n' "$t" "$(command -v "$t")"
done
    # Root layout allowlist: research and run output belong under research/, never the repo root
root_allow=' .config .githooks .github .gitignore .gitleaks.toml .gitleaksignore .jope-home AGENTS.md AGENT_CHAT.md CHANGELOG.md CLAUDE.md CODE_OF_CONDUCT.md CONTRIBUTING.md Cargo.lock Cargo.toml LICENSE MASTER_CLEAN_UP.md MASTER_PLAN.md NOTICE PYTHON_EXCEPTIONS.md PY_TO_RUST_PLAN.md README.md SECURITY.md assets benchmarks catalogs cliff.toml crates datasets deny.toml deploy docs examples justfile models ops python research rust-toolchain.toml schemas scripts test-support tools vendor web xtools '
root_bad=0
while IFS= read -r e; do
  case "$root_allow" in *" $e "*) ;; *) echo "root layout: '$e' is not allowed at the repo root (H-wave output goes in research/flywheel_selfplay/hNN/)" >&2; root_bad=1 ;; esac
done < <(git ls-tree --name-only HEAD)
[ "$root_bad" = 0 ]
    # Cargo artifact protocol tests
cargo test -q --locked -p jope-ci
    # Coverage mapping and collection failure tests
cargo test -q --locked -p jope-records --lib -- coverage:: scope_check:: parity:: readme::
python3 deploy/ci/test-security-lane.py
python3 deploy/ci/test-gitleaks-config.py
python3 deploy/ci/test-rendered-ux-receipt.py
python3 deploy/ci/test-browser-deps.py
python3 -B deploy/ci/test-vendor-security.py
python3 deploy/ci/test-receipt.py
python3 deploy/ci/test-results.py
python3 deploy/ci/test-aggregate.py
python3 deploy/ci/test-status.py
python3 deploy/ci/test-runner.py
python3 deploy/ci/test-provenance.py
python3 deploy/ci/test-full-main.py
    # Split jobs cover the families and rust-test slices of the single jobs they replaced (and,
    # on a pull request, nothing the base declared is missing)
python3 deploy/ci/check_test_partition.py static --base "${BASE_SHA:-}" --output "$EVIDENCE/test-partition-static.json"
python3 deploy/ci/test-training-proof.py
python3 deploy/ci/test-install-private.py
python3 deploy/ci/test-qualify-demo.py
python3 deploy/ci/test-prepare-asm-runtime.py
bash deploy/ci/test-neural-env.sh
bash deploy/ci/test-test-routing.sh
bash deploy/ci/test-real-corpus-lane.sh
bash deploy/release/test-release.sh
cargo run -q --locked -p jope-records -- verify-findings
cargo run -q --locked -p jope-records -- verify-dispositions
cargo run -q --locked -p jope-records -- verify-test-inventory
cargo run -q --locked -p jope-records -- verify-claims
cargo run -q --locked -p jope-records -- claim-inventory
cargo run -q --locked -p jope-records -- conformance
cargo run -q --locked -p jope-records -- task-status
cargo run -q --locked -p jope-records -- progress-view
cargo run -q --locked -p jope-records -- elo-history --require-all
    # Hygiene
bash deploy/ci/check.sh
    # Lambda workspaces (deploy/lambda: standalone Cargo workspaces, not part of the jope build)
bash deploy/ci/lambda.sh
python3 deploy/lambda/training/teacher-v2/test_materialize_g0.py
python3 deploy/lambda/training/teacher-v2/qualify_s1.py --out "$EVIDENCE/s1-mutants"
    # `just bump` keeps every workspace consistent (after lambda.sh, whose cargo runs fill the cache
    # that the bump's --offline cargo needs)
bash deploy/ci/test-bump.sh
    ;;
  rust-check)
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # cargo check --workspace --all-targets --locked
    # Search commentary contracts: exact diffs, producer parity and real streaming.
bash ops/ci/narration.sh
cargo check --workspace --all-targets --locked
    # Telemetry's optional native Python ABI and engine feature gates are part of the crate contract.
cargo check -p jope-telemetry --features python --all-targets --locked
cargo check -p jope-parallel --features telemetry --locked
    # Rendered systemd units resolve under the rendered roots
cargo test -p jope-dime-regressor --locked --test systemd_unit_paths
    # Workspace MSRV (decision 8: at most 1.97.1)
set -euo pipefail
cargo metadata --no-deps --locked --format-version 1 > "$EVIDENCE/metadata.json"
python3 - <<'PY'
import json, os
packages = json.load(open(os.environ['JOPE_LANE_EVIDENCE'] + '/metadata.json'))['packages']
versions = sorted({p['rust_version'] for p in packages if p.get('rust_version')})
print('rust-version values in the workspace:', versions)
assert versions and versions[-1].split('.')[:2] <= ['1', '97'], versions
PY
    ;;
  emitters)
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # Compilers on this runner
set -euo pipefail
for tool in gcc clang g++ clang++ node python3 rustc; do printf '%s: ' "$tool"; "$tool" --version | head -1; done
    # Emitted-code matrix
EMITTERS_OUT="$EVIDENCE/emitters" bash deploy/ci/emitters.sh
    # Strict portable parity — cxx helper unit tests
cargo test -p feat-math --locked --lib -- cxx::
    # Strict portable parity — portable_all_helper_backends_match_bits
cargo test -p feat-math --locked --test portable_parity
    # Fitted C++ compile scratch outside /tmp, as lambda.sh keeps it: the codec compile and the tests' tempdirs follow TMPDIR
case ${TMPDIR:-/tmp} in
  /tmp | /tmp/*) export TMPDIR=${CARGO_TARGET_DIR:-$PWD/target}/fitted-cpp-tmp; mkdir -p -- "$TMPDIR" ;;
esac
    # Fitted C++17 under AddressSanitizer and UndefinedBehaviorSanitizer (kernel2 X3): every advanced_v3_fitted_cpp test
if [ "${JOPE_FULL_EMITTERS_PARALLEL:-0}" = 1 ]; then
    # Build once, then overlap the two independent matrices. Each test owns its
    # tempdir; four workers per matrix bound the concurrent compiler footprint.
    cargo test -p feat-math --locked --test advanced_v3_fitted_cpp --no-run
    JOPE_FITTED_CPP_SANITIZER=address,undefined cargo test -p feat-math --locked --test advanced_v3_fitted_cpp -- --test-threads=4 > "$EVIDENCE/fitted-sanitizer.log" 2>&1 &
    sanitizer_pid=$!
    JOPE_CPP_STD=20 cargo test -p feat-math --locked --test advanced_v3_fitted_cpp -- --test-threads=4 > "$EVIDENCE/fitted-cxx20.log" 2>&1 &
    cxx20_pid=$!
    sanitizer_status=0; wait "$sanitizer_pid" || sanitizer_status=$?
    cxx20_status=0; wait "$cxx20_pid" || cxx20_status=$?
    cat "$EVIDENCE/fitted-sanitizer.log" "$EVIDENCE/fitted-cxx20.log"
    # Both matrices finish and publish their complete logs, including failures.
    (( sanitizer_status == 0 )) || exit "$sanitizer_status"
    (( cxx20_status == 0 )) || exit "$cxx20_status"
else
    JOPE_FITTED_CPP_SANITIZER=address,undefined cargo test -p feat-math --locked --test advanced_v3_fitted_cpp
    # Fitted C++ at -std=c++20 with g++ (fitted model compilation requires g++ 13.3, so clang++ is out of scope)
    JOPE_CPP_STD=20 cargo test -p feat-math --locked --test advanced_v3_fitted_cpp
fi
    ;;
  inputs)
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # Verify the committed fixtures offline
cargo run --locked --bin jope -- verify-inputs --offline --json --root datasets/synthetic
    # Generator reproduces the committed fixtures
cargo run --locked --bin jope -- generate-fixtures --check --json --output datasets/synthetic
if [ "${JOPE_FULL_FAMILY:-}" = full-inputs ]; then
    # Preserve the original CLI-before-tests feature graph order. Build both
    # test harnesses together so their links can overlap in the full family.
    cargo test --locked -p jope-dime-regressor --lib --bin jope-dime-regressor --no-run
fi
    # Check both migrated renderer trees and the Python policy copy against Rust.
    # Keep the narrow library harness separate from deferred campaign execution.
python3 deploy/ci/receipt.py --lane inputs --backend rust --output "$EVIDENCE/campaign-spec-policy" -- cargo test --locked -p jope-dime-regressor --lib spec_drift::tests -- --test-threads=1
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["counts"]["executed"] > 0, "campaign policy collected no tests"' "$EVIDENCE/campaign-spec-policy/receipt.json"
    # Data-free campaign ranking parity against the maintained arena reference.
python3 deploy/ci/receipt.py --lane inputs --backend rust --output "$EVIDENCE/campaign-elo" -- cargo test --locked -p jope-dime-regressor --bin jope-dime-regressor commands::train_only_redteam::tests::bradley_terry -- --test-threads=1
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["counts"]["executed"] > 0, "campaign Elo collected no tests"' "$EVIDENCE/campaign-elo/receipt.json"
    ;;
  python)
    # uv from the image
uv --version | grep -qF 'uv 0.11.28' || { uv --version >&2; echo 'expected the image-pinned uv 0.11.28' >&2; exit 1; }
    # Sync the locked environment (CPU torch)
(
cd python
uv sync --locked --extra dev --extra cpu --extra benchmark --extra foundry --extra dl-custody
)
    # Ruff lint and format
(
cd python
set -euo pipefail
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
)
    # Pytest (xbabe2-only tests deselected)
(
cd python
bash ../deploy/ci/pytest-evidence.sh uv run --no-sync pytest -q tests ../benchmarks/test_prepare.py ../crates/jope-bench/adapters/test_panel.py -m "not xbabe2" -p no:cacheprovider
)
python3 deploy/ci/training-proof.py python --junit "$EVIDENCE/junit.xml"
    # The exact DL source handoff has its own offline unittest receipt; pytest totals stay separate.
    # It runs as the pinned module shards in sequence plus --combine, all under ONE common 900 s deadline
    # (the single lane's budget): each step gets only the time left, and --combine fails unless the shards
    # together are exactly the full strict lane.
(
cd python
unset DL_OFFLINE_RUNNER
bash ../deploy/ci/dl-offline-group.sh 900 "$EVIDENCE/dl-lane-offline"
timeout 300s uv run --no-sync python ../deploy/dl-flywheel/run_offline.py --output "$EVIDENCE/flywheel-offline"
timeout 600s uv run --no-sync python ../deploy/dl-flywheel/run_v2_offline.py --output "$EVIDENCE/flywheel-v2-offline"
)
    # RLFW fixtures include torch IPO/DPO checks. Run them in the locked CPU
    # torch environment synced above; system python3 does not provide torch.
( cd python; uv run --no-sync python ../deploy/dl-flywheel/pipelines/rlfw/run_rlfw_offline.py --receipt "$EVIDENCE/rlfw-offline.json" )
    # Lifecycle proposals and the executable cache. No bucket calls. DuckDB is optional.
python3 -m unittest discover -s deploy/dl-flywheel/pipelines/rlfw_s3gp/tests -q
    # J9 q-head adapter fixtures (stub policy, stdlib only); the frozen-pool core is Rust (crates/jope-learning/src/pools).
python3 -m unittest discover -s deploy/dl-flywheel/pipelines/rlfw/pools/tests -t deploy/dl-flywheel/pipelines -q
    # C4/S7/S8 finite-action and actual CPU float64 autograd fixtures.
(
cd python
uv run --no-sync python ../deploy/dl-flywheel/pipelines/rlfw/onpolicy/run_onpolicy_offline.py --receipt "$EVIDENCE/onpolicy-offline.json"
)
( cd python; uv run --no-sync python ../deploy/dl-flywheel/pipelines/rlfw/run_kv_offline.py --receipt "$EVIDENCE/kv-trie-offline.json" )
    # R25-kNN-CE neighbour-table and harness-gate CPU fixtures (torch CPU; no data, checkpoint or GPU).
( cd python; uv run --no-sync python ../deploy/dl-flywheel/pipelines/rlfw/r25_knnce/run_r25_offline.py --receipt "$EVIDENCE/r25-knnce-offline.json" )
    # J22 pure CPU analysis fixtures; no training or evaluator execution.
python3 deploy/dl-flywheel/pipelines/rlfw/multiseed/run_multiseed_offline.py --receipt "$EVIDENCE/multiseed-offline.json"
    # Analytics uses the locked numpy/pyarrow environment; all inputs are fixtures.
uv run --project python --no-sync python deploy/dl-flywheel/pipelines/rlfw/analytics/run_offline.py --receipt "$EVIDENCE/rlfw-analytics-offline.json"
    # New ingest data contracts use copied fixtures; no S3 credentials or traffic.
uv run --project python --no-sync python deploy/dl-flywheel/ingest_joined_roster/run_offline.py --receipt "$EVIDENCE/ingest-offline.json"
    # Phase 7 flywheel v3 offline CPU synthetic/stub unit checks (9 tests; no GPU, S3, or campaign compute).
CUDA_VISIBLE_DEVICES="" AWS_EC2_METADATA_DISABLED=true \
  uv run --project python --no-sync python -B \
  research/flywheel_selfplay/test_verify_flywheel_wider_train_extension_v3.py
    ;;
  rust-test)
    status=0
    trap 'status=$?; trap - EXIT; bash deploy/ci/retain-nextest.sh "$EVIDENCE" || status=1; exit "$status"' EXIT
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # cargo-nextest from the image
cargo nextest --version | grep -qF 'cargo-nextest 0.9.133' || { cargo nextest --version >&2; echo 'expected the image-pinned cargo-nextest 0.9.133' >&2; exit 1; }
    # Product-crate tests (blocking), one slice of them per ci.yml matrix leg
set -euo pipefail
mkdir -p "$EVIDENCE"
partition=${JOPE_RUST_TEST_PARTITION:?ci.yml sets the slice:M/N partition of each rust-test leg}
if ! [[ $partition =~ ^slice:([1-9][0-9]*)/([1-9][0-9]*)$ ]] || (( BASH_REMATCH[1] > BASH_REMATCH[2] )); then
  echo "invalid rust-test partition: $partition" >&2; exit 1
fi
# Capture first so a failing scope script aborts here: inside `read -ra pkgs <<< "$(...)"`
# the command substitution's exit status is discarded, pkgs would be empty, and cargo
# would silently run UNSCOPED — straight back into the 72 GB link this lane exists to avoid.
scope="$(bash deploy/ci/rust-test-scope.sh)"
read -ra pkgs <<< "$scope"
[ "${#pkgs[@]}" -gt 0 ] || { echo "rust-test-scope.sh produced an empty package set" >&2; exit 1; }
# Identity of what actually ran, so the uploaded report can be tied to a revision and a
# package set rather than to "the lane was green once".
# Captured as statements, not inside `echo "$( )"`: a command substitution in an echo
# argument cannot trip `set -e`, so a failing rustup would be recorded as an empty value.
toolchain="$(rustup show active-toolchain)"
runner="$(uname -srm)"
{
  echo "revision=${GITHUB_SHA}"
  echo "ref=${GITHUB_REF}"
  echo "run=${GITHUB_RUN_ID}/${GITHUB_RUN_ATTEMPT}"
  echo "runner=${runner}"
  echo "toolchain=${toolchain}"
  echo "packages=${scope}"
  echo "partition=${partition}"
} | tee "$EVIDENCE"/identity.txt
df -B1 --output=source,size,used,avail,pcent / | tee "$EVIDENCE"/disk-before.txt
# Marks that nextest was reached, so the retention step can tell "the report is missing"
# from "the run never got that far" instead of accepting both.
target=$(cargo run -q --locked -p jope-ci -- cargo-artifact target-dir)
junit="$target/nextest/pr/junit.xml"
# Preserve a previous report before execution, so a failed build cannot
# upload it as this run's test result from the persistent target.
if [ -f "$junit" ]; then mv "$junit" "$EVIDENCE"/prior-junit.xml; fi
cargo run -q --locked -p jope-ci -- cargo-artifact nextest-config --output "$EVIDENCE"/nextest.toml
selection=(--config-file "$EVIDENCE"/nextest.toml -P pr --locked "${pkgs[@]}" --features "jope-cli/asm,jope-cli/qualified-training-context")
# The unpartitioned listing is the whole pre-split rust-test selection. The aggregate job's
# check_test_partition.py proves the slices select and execute it exactly once between them.
cargo nextest list "${selection[@]}" --message-format json > "$EVIDENCE/collection-unsharded.json"
cargo nextest list "${selection[@]}" --partition "$partition" --message-format json > "$EVIDENCE/collection.json"
touch "$EVIDENCE"/nextest-started
# Bound simultaneous compiler fixtures while retaining their original deadlines.
cargo nextest run "${selection[@]}" --partition "$partition" --test-threads 4
# Required training proofs outside this slice are proven in the other one (aggregate job).
python3 deploy/ci/training-proof.py rust --partitioned --collection "$EVIDENCE/collection.json" --junit "$junit"
    ;;
  e2e-smoke)
    # Toolchain (from rust-toolchain.toml)
rustup show active-toolchain
    # Compilers this lane needs are in the image
set -euo pipefail
# Asserted, not printed. e2e-smoke.sh compiles the emitted C++ and runs the emitted
# JavaScript and Python, so a missing toolchain here is an image defect, not a code
# failure, and should say so rather than surfacing as a confusing script error.
for tool in gcc clang g++ clang++ node python3 rustc cargo; do
  command -v "$tool" >/dev/null || { echo "runner image is missing $tool" >&2; exit 1; }
  printf '%-10s %s\n' "$tool" "$("$tool" --version | head -1)"
done
    # End-to-end pipeline smoke
E2E_OUT="$EVIDENCE/pipeline" bash deploy/ci/e2e-smoke.sh
    # The jopedime appliance (deploy/appliance): build with the PR profile, pack and sign the fixture
    # profile with the development key, then run the packed file alone in an empty directory.
    # The packages stay in the lane target; only their receipts are evidence.
appliance_target=$(cargo run -q --locked -p jope-ci -- cargo-artifact target-dir)
appliance_work="$appliance_target/appliance-lane/${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-$$"
bash deploy/appliance/build.sh --profile fixture --cargo-profile dist-ci --out "$appliance_work/build" \
  --source-sha "${JOPE_SOURCE_SHA:-$(git rev-parse HEAD)}"
bash deploy/appliance/test-clean-host.sh "$appliance_work/build/jopedime" "$appliance_work/clean-host"
mkdir -p "$EVIDENCE/appliance"
cp -- "$appliance_work"/build/{build,pack-receipt,verify,size,sbom-models}.json \
  "$appliance_work"/clean-host/{clean-host,fit}.json "$EVIDENCE/appliance/"
rm -r -- "$appliance_work"
    ;;
  *) echo "unknown CI lane: $1" >&2; exit 2 ;;
esac
