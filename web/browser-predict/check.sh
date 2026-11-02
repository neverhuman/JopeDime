#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
target=${CARGO_TARGET_DIR:-$PWD/target}
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
JOPE_BROWSER_FIXTURES="$target/browser-predict/native-fixtures" \
  cargo test --locked --manifest-path web/browser-predict/Cargo.toml --target-dir "$target"
bash web/browser-predict/build.sh
node web/browser-predict/check.mjs "$target/wasm32-unknown-unknown/release/jope_browser_predict.wasm" "$target/browser-predict/native-fixtures"
