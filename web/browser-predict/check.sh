#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
target=${CARGO_TARGET_DIR:-$PWD/target}
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
node web/browser-predict/security-check.mjs
bash web/browser-predict/security.sh
mkdir -p "$target/browser-predict"
# Each native execution must generate every fixture afresh; persistent runner
# caches must not supply a missing or zero-test native comparison.
fixtures=$(mktemp -d "$target/browser-predict/native-fixtures.XXXXXX")
JOPE_BROWSER_FIXTURES="$fixtures" \
  cargo test --locked --manifest-path web/browser-predict/Cargo.toml --target-dir "$target"
bash web/browser-predict/build.sh
node web/browser-predict/check.mjs "$target/wasm32-unknown-unknown/release/jope_browser_predict.wasm" "$fixtures" | tee "$target/browser-predict/parity.json"
