#!/usr/bin/env bash
# Shared environment for the declared fast CI lanes and their local replay.
set -euo pipefail
jope_ci_environment() {
  export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
  # CARGO_TARGET_DIR stays intact: CI deliberately assigns a persistent target.
}
