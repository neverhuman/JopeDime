#!/usr/bin/env bash
# The admitted runner image pins Rust 1.97.1 and rustup. Copy its read-only state;
# add Wasm through that trusted rustup into our own writable task state only.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
state="$repo/target/tool-cache/typescript-rustup"
test "${RUSTUP_HOME:?pinned runner Rustup required}" = /opt/rust/rustup
mkdir -p "$repo/target/tmp" "$state"
export TMPDIR="$repo/target/tmp"
cp -a /opt/rust/rustup/. "$state/"
export RUSTUP_HOME="$state"
rustup target add --toolchain 1.97.1 wasm32-unknown-unknown
test "$(rustc --version | cut -d' ' -f2)" = 1.97.1
rustup target list --installed | grep -Fx wasm32-unknown-unknown
# Later Actions steps retain the image's immutable compiler executable and the
# shared registry cache; only its Rustup data location is task-local.
printf 'RUSTUP_HOME=%s\n' "$RUSTUP_HOME" >> "${GITHUB_ENV:?Actions environment required}"
