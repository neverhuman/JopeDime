#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
out_dir="${CARGO_TARGET_DIR:-target}/api-schema-check"
mkdir -p "$out_dir"

nice -n 10 cargo run --quiet --locked -p jope-api --bin jope-api-openapi -- "$out_dir/openapi.generated.json"
diff -u schemas/jope.api.v1.openapi.json "$out_dir/openapi.generated.json"

JOPE_SCHEMA_SNAPSHOT_OUTPUT="$out_dir/schemas.generated.json" \
  nice -n 10 cargo run --quiet --locked -p jope-api --example schema_snapshots
diff -u crates/jope-api/tests/snapshots/schemas.json "$out_dir/schemas.generated.json"
