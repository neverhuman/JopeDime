#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MATRIX_LABEL="${1:?matrix label required}"
MATRIX_TARGET="${2:?matrix target required}"
MATRIX_ASSET="${3:?matrix asset required}"

mkdir -p target/carbon-matrix

if [[ "$MATRIX_LABEL" == "linux-x86_64" ]]; then
  rustup target add x86_64-unknown-linux-musl
  cargo build --locked --profile carbon-cli --target x86_64-unknown-linux-musl \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cp target/x86_64-unknown-linux-musl/carbon-cli/carbon "target/carbon-matrix/$MATRIX_ASSET"
  "target/carbon-matrix/$MATRIX_ASSET" --version
elif [[ "$MATRIX_LABEL" == "darwin-universal" ]]; then
  rustup target add aarch64-apple-darwin x86_64-apple-darwin
  cargo build --locked --profile carbon-cli --target aarch64-apple-darwin \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cargo build --locked --profile carbon-cli --target x86_64-apple-darwin \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cp target/aarch64-apple-darwin/carbon-cli/carbon target/carbon-matrix/carbon-darwin-arm64
  cp target/x86_64-apple-darwin/carbon-cli/carbon target/carbon-matrix/carbon-darwin-x86_64
  lipo -create \
    target/carbon-matrix/carbon-darwin-arm64 \
    target/carbon-matrix/carbon-darwin-x86_64 \
    -output target/carbon-matrix/carbon-darwin-universal
  codesign --force -s - target/carbon-matrix/carbon-darwin-arm64
  codesign --force -s - target/carbon-matrix/carbon-darwin-x86_64
  codesign --force -s - target/carbon-matrix/carbon-darwin-universal
  target/carbon-matrix/carbon-darwin-universal --version
else
  rustup target add "$MATRIX_TARGET"
  cargo build --locked --profile carbon-cli --target "$MATRIX_TARGET" \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cp "target/$MATRIX_TARGET/carbon-cli/carbon.exe" "target/carbon-matrix/$MATRIX_ASSET"
  if command -v powershell.exe >/dev/null 2>&1 || command -v pwsh >/dev/null 2>&1; then
    PS_BIN="$(command -v pwsh || command -v powershell.exe)"
    BASE_ZIP="${MATRIX_ASSET%.exe}.zip"
    STAGE_EXE="target/carbon-matrix/carbon.exe"
    cp "target/$MATRIX_TARGET/carbon-cli/carbon.exe" "$STAGE_EXE"
    "$PS_BIN" -NoProfile -Command "Compress-Archive -LiteralPath '$STAGE_EXE' -DestinationPath 'target/carbon-matrix/$BASE_ZIP' -Force"
    rm -f "$STAGE_EXE"
  fi
  if [[ "$MATRIX_LABEL" == "windows-x86_64" ]]; then
    "target/carbon-matrix/$MATRIX_ASSET" --version
  fi
fi
