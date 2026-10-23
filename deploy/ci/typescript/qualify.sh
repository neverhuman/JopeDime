#!/usr/bin/env bash
set -euo pipefail
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only JOPE_TYPESCRIPT_PARITY=1
export JOPE_TSC="$PWD/deploy/ci/typescript/node_modules/.bin/tsc"
mkdir -p target/typescript-evidence target/typescript-tmp
export TMPDIR="$PWD/target/typescript-tmp"
source_sha=$(git rev-parse HEAD)
test "$source_sha" = "${JOPE_SOURCE_SHA:-$source_sha}"
printf 'source=%s\nrun=%s\nattempt=%s\n' "$source_sha" "${GITHUB_RUN_ID:-local}" "${GITHUB_RUN_ATTEMPT:-1}" > target/typescript-evidence/identity.txt
rustc --version > target/typescript-evidence/toolchains.txt
node --version >> target/typescript-evidence/toolchains.txt
"$JOPE_TSC" --version >> target/typescript-evidence/toolchains.txt
node deploy/ci/typescript/prepare-contract.mjs
JOPE_TS_CONTRACT_SOURCE="$PWD/target/typescript-contract/model.js" node --test --experimental-test-coverage \
  --test-coverage-include='**/typescript-contract/model.js' \
  --test-coverage-lines=100 --test-coverage-branches=100 --test-coverage-functions=100 \
  deploy/ci/typescript/contract.test.cjs 2>&1 | tee target/typescript-evidence/contracts.log
cargo test --locked -p feat-math --no-default-features --lib typescript::tests:: -- --test-threads=2 --nocapture 2>&1 | tee target/typescript-evidence/census.log
cargo test --locked -p feat-math --no-default-features --test golden -- --test-threads=2 --nocapture 2>&1 | tee target/typescript-evidence/recipes.log
cargo test --locked -p feat-math --no-default-features --lib advanced_v3::emit:: -- --test-threads=2 --nocapture 2>&1 | tee target/typescript-evidence/fitted.log
cargo test --locked -p feat-math --no-default-features --test artifact_target_binding 2>&1 | tee target/typescript-evidence/bindings.log
cargo test --locked -p jope-pipeline --test artifact_target_binding 2>&1 | tee target/typescript-evidence/pipeline.log
cargo test --locked -p jope-cli --no-default-features --lib kernel::artifact_v3_tests:: 2>&1 | tee target/typescript-evidence/cli.log
node deploy/ci/typescript/verify.mjs
node --test deploy/ci/typescript/evidence.test.mjs 2>&1 | tee target/typescript-evidence/receipt-controls.log
