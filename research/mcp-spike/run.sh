#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_TARGET_DIR="$PWD/target" TMPDIR="$PWD/target/tmp"
mkdir -p "$TMPDIR" research/mcp-spike/evidence/local
python3 - <<'PY'
import os
cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
assert os.getloadavg()[0] <= 0.85 * cpus, "host load exceeds CPU budget"
PY
nice -n 10 cargo fmt --manifest-path research/mcp-spike/Cargo.toml --all -- --check
nice -n 10 cargo clippy --manifest-path research/mcp-spike/Cargo.toml --workspace --all-targets --locked -- -D warnings
nice -n 10 cargo build --manifest-path research/mcp-spike/Cargo.toml --workspace --locked
nice -n 10 python3 research/mcp-spike/conformance.py --bin-dir target/debug \
  --output research/mcp-spike/evidence/local/replay.json
