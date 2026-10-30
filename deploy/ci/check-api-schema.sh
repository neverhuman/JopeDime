#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

export CARGO_BUILD_JOBS=8
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
generated="target/api-schema.generated.json"
mkdir -p target

nice -n 10 cargo run --quiet --locked -p jope-api --bin jope-api-openapi -- "$generated"
diff -u schemas/jope.api.v1.openapi.json "$generated"
