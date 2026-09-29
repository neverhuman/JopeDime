#!/usr/bin/env bash
set -euo pipefail
export ZIG_GLOBAL_CACHE_DIR=/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/target/zig-cache
export ZIG_LOCAL_CACHE_DIR="$ZIG_GLOBAL_CACHE_DIR/local"
args=()
for arg in "$@"; do
  case "$arg" in
    --target=aarch64-unknown-linux-musl) ;;
    *) args+=("$arg") ;;
  esac
done
exec /home/ubuntu/scratch/jopedime-lambda-cost-20260928/tools/ziglang/zig cc -target aarch64-linux-musl "${args[@]}"
