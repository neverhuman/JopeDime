#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=2
export JOPE_GO_PARITY=1 JOPE_PARITY_STRICT=1 CGO_ENABLED=0 GOMAXPROCS=2
export TMPDIR="$PWD/target/go-tmp"
mkdir -p "$TMPDIR" target/go-evidence
out=target/go-evidence
{ git rev-parse HEAD; git diff --stat; rustc -Vv; go version; date -u +%FT%TZ; sha256sum crates/feat-math/src/go/model.go crates/feat-math/src/go/{buffers,recipe,bound}.rs.txt crates/feat-math/src/go/go.{mod,sum}.txt; } > "$out/identity.log"
cargo test --locked -p feat-math --lib go::tests:: -- --nocapture --test-threads=1 2>&1 | tee "$out/census.log"
bash deploy/ci/go/contracts.sh 2>&1 | tee "$out/contracts.log"
cargo test --locked -p feat-math --features go-parity-tests --test go_golden -- --nocapture --test-threads=2 2>&1 | tee "$out/recipes.log"
cargo test --locked -p feat-math --lib advanced_v3::emit -- --nocapture --test-threads=1 2>&1 | tee "$out/fitted.log"
cargo test --locked -p feat-math --test artifact_target_binding -- --nocapture 2>&1 | tee "$out/bindings.log"
cargo test --locked -p jope-pipeline --test artifact_target_binding -- --nocapture 2>&1 | tee "$out/pipeline.log"
cargo test --locked -p jope-cli --lib kernel::artifact_v3_tests -- --nocapture 2>&1 | tee "$out/cli.log"
python3 deploy/ci/go/verify.py
