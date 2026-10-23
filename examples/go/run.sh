#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CGO_ENABLED=0
out="$PWD/target/go-example"
mkdir -p "$out"
cargo run --locked -p jope-cli --bin jope -- emit --genome '(pipe X (ridge))' --lang go --output "$out/model"
(
 cd "$out/model"
 rustc --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib --crate-name model_kernel -C opt-level=3 -C panic=abort model.go.kernel.rs -o model.wasm
 go build -mod=readonly ./...
)
cp examples/go/main.go "$out/main.go"
cat > "$out/go.mod" <<'MOD'
module jope.go.example

go 1.25.0

require jope.generated/model v0.0.0

replace jope.generated/model => ./model
MOD
cp "$out/model/go.sum" "$out/go.sum"
cd "$out"
go run -mod=mod .
