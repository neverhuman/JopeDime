#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  preflight)
    rustup target add wasm32-unknown-unknown
    cargo fmt --all --check
    git diff --check
    cargo check --locked -p feat-math -p jope-cli -p jope-pipeline --no-default-features --all-targets
    ;;
  sanitizer)
    cargo test --locked -p feat-math --lib runtime_c_abi_is_sanitizer_clean_and_fail_closed -- --nocapture
    cargo test --locked -p feat-math --lib cpp17_numeric_abi_is_bit_exact_and_has_no_data_access -- --nocapture
    ;;
  qualify) bash deploy/ci/go/qualify.sh ;;
  race) (cd target/go-contract && go test -mod=readonly -race -count=1 -timeout=180s ./...) ;;
  fuzz) (cd target/go-contract && go test -mod=readonly -run '^$' -fuzz '^FuzzLifecycle$' -fuzztime=20s -parallel=2) ;;
  examples)
    bash examples/go/run.sh
    cd target/go-example
    mkdir -p ../go-executables
    for target in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64; do
      GOOS=${target%/*} GOARCH=${target#*/} CGO_ENABLED=0 go build -mod=readonly -o "../go-executables/${target%/*}-${target#*/}" .
    done
    ;;
  platform)
    rustup target add wasm32-unknown-unknown
    cargo test --locked -p feat-math --no-default-features --lib go::tests:: -- --nocapture --test-threads=1
    bash deploy/ci/go/contracts.sh
    ;;
  *) echo 'usage: go.sh preflight|sanitizer|qualify|race|fuzz|examples|platform' >&2; exit 2 ;;
esac
