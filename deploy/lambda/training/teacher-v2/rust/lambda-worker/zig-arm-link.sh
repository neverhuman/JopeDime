#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export ZIG_GLOBAL_CACHE_DIR="$ROOT/target/zig-cache"
export ZIG_LOCAL_CACHE_DIR="$ZIG_GLOBAL_CACHE_DIR/local"
args=()
for arg in "$@"; do
  case "$arg" in
    --target=aarch64-unknown-linux-musl) ;;
    */self-contained/crt1.o|*/self-contained/crti.o|*/self-contained/crtbegin.o|*/self-contained/crtend.o|*/self-contained/crtn.o) ;;
    *) args+=("$arg") ;;
  esac
done
exec /home/ubuntu/scratch/jopedime-lambda-cost-20260928/tools/ziglang/zig cc -target aarch64-linux-musl "${args[@]}"
