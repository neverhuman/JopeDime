#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
out=${C17_OUT:-target/c17-example/$(date -u +%Y%m%dT%H%M%S.%NZ)-$$}
[ ! -e "$out" ] || { echo "refusing to overwrite $out" >&2; exit 1; }
cargo run --locked -q -p jope-cli --bin jope -- emit \
  --genome '(pipe X (standardize) (histgbm rounds=20))' --lang c17 --output "$out" --json
cp examples/c17/main.c "$out/c17/main.c"
(cd "$out/c17"
 rustc --edition=2021 -O -C panic=unwind --crate-type=staticlib model_ffi.rs -o libmodel.a
 "${CC:-gcc}" -std=c17 -O2 -Wall -Wextra -Werror -pedantic-errors -fno-fast-math -ffp-contract=off \
   main.c libmodel.a -ldl -lpthread -lm -o caller
 ./caller)
