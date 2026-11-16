#!/usr/bin/env bash
# Local/CI parity entrypoint for Fresh-Docker Carbon installer & dev+prod 4-tier E2E qualification.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT_DIR="${1:-target/carbon-live-e2e/local}"
SOURCE_SHA="${JOPE_SOURCE_SHA:-$(git rev-parse HEAD)}"
USE_LIVE_PRO_FLAG="${USE_LIVE_PRO:-false}"

mkdir -p "$OUT_DIR"

if [[ "${CARBON_INSTALL_PINNED_RUSTUP:-0}" == "1" ]]; then
  export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rust/rustup}"
  sudo install -d -m 0755 -o "$(id -un)" -g "$(id -gn)" "$RUSTUP_HOME"
  rustup toolchain install 1.97.1 --profile minimal
  rustup default 1.97.1
fi

test "$(git rev-parse HEAD)" = "$SOURCE_SHA"

export CARGO_INCREMENTAL="${CARGO_INCREMENTAL:-0}"
export CARGO_PROFILE_DEV_DEBUG="${CARGO_PROFILE_DEV_DEBUG:-line-tables-only}"
export CARGO_PROFILE_DIST_CI_LTO="${CARGO_PROFILE_DIST_CI_LTO:-off}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

CARBON_EXE="$(
  cargo build --locked --profile dist-ci -p jope-carbon --bin carbon --message-format=json \
    | python3 -c 'import json,sys; print([m["executable"] for l in sys.stdin if l.strip() for m in [json.loads(l)] if m.get("reason")=="compiler-artifact" and m.get("target",{}).get("name")=="carbon" and m.get("executable")][-1])'
)"

extra_args=()
if [[ "$USE_LIVE_PRO_FLAG" == "true" ]]; then
  extra_args+=(--require-live-pro)
fi

"$CARBON_EXE" ci-live-4tier \
  --source-sha "$SOURCE_SHA" \
  --out-dir "$OUT_DIR" \
  "${extra_args[@]}"
