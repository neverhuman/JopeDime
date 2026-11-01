#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
case "${1:-}" in
  linux-prestage)
    out=target/rel04/linux
    mkdir -p "$out/models"
    source_sha=$(git rev-parse HEAD)
    code=$(JOPE_BUILD_SOURCE_SHA="$source_sha" nice -n 10 cargo run -q --locked -p jope-ci -- cargo-artifact build --package jope-appliance --bin jopedime --profile dist-ci --no-default-features --features embedded,dev-keys)
    nice -n 10 cargo run -q --locked -p jope-dist -- fixture --models "$out/models"
    nice -n 10 cargo run -q --locked -p jope-dist -- notices --out "$out/notices.json" --features embedded,dev-keys
    nice -n 10 cargo run -q --locked -p jope-dist -- pack --profile deploy/appliance/profiles/fixture.toml --code "$code" --models "$out/models" --notices "$out/notices.json" --source-sha "$source_sha" --target x86_64-unknown-linux-gnu --receipt "$out/pack-receipt.json" --out "$out/jopedime"
    nice -n 10 cargo run -q --locked -p jope-dist -- verify --file "$out/jopedime" --development-key --no-release-key > "$out/verify.json"
    ;;
  linux-exercise)
    sudo env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory \
      "GIT_CONFIG_VALUE_0=$GITHUB_WORKSPACE" \
      bash deploy/appliance/test-clean-host.sh \
      target/rel04/linux/jopedime target/rel04/linux-clean-host --require-isolation
    ;;
  macos) exec bash deploy/appliance/test-clean-host-macos.sh --build-fixture ;;
  *) echo 'usage: clean-host.sh linux-prestage|linux-exercise|macos' >&2; exit 2 ;;
esac
