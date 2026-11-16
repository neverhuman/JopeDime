#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MATRIX_LABEL="${1:?matrix label required}"
MATRIX_TARGET="${2:?matrix target required}"
MATRIX_ASSET="${3:?matrix asset required}"

mkdir -p target/carbon-matrix

emit_posix_tarballs() {
  local asset_name="$1"
  local bin_src="$2"
  local stage_dir="target/carbon-matrix/.stage-${asset_name}"
  rm -rf "$stage_dir"
  mkdir -p "$stage_dir"
  cp "$bin_src" "$stage_dir/carbon"
  chmod 0755 "$stage_dir/carbon"
  local tar_flags=()
  if [[ "$(uname -s)" == "Darwin" ]]; then
    tar_flags=(--no-xattrs --no-mac-metadata)
  elif tar --version 2>/dev/null | grep -q 'GNU tar'; then
    tar_flags=(--format=ustar --owner=0 --group=0 --numeric-owner --mtime='@0')
  fi
  COPYFILE_DISABLE=1 tar "${tar_flags[@]}" -C "$stage_dir" -czf "target/carbon-matrix/${asset_name}.tar.gz" carbon
  if command -v xz >/dev/null 2>&1; then
    COPYFILE_DISABLE=1 tar "${tar_flags[@]}" -C "$stage_dir" -cf - carbon | xz -c > "target/carbon-matrix/${asset_name}.tar.xz"
  fi
  if command -v zstd >/dev/null 2>&1; then
    COPYFILE_DISABLE=1 tar "${tar_flags[@]}" -C "$stage_dir" -cf - carbon | zstd -q -19 -o "target/carbon-matrix/${asset_name}.tar.zst"
  fi
  rm -rf "$stage_dir"
}

if [[ "$MATRIX_LABEL" == "linux-x86_64" ]]; then
  rustup target add x86_64-unknown-linux-musl
  cargo build --locked --profile carbon-cli --target x86_64-unknown-linux-musl \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cp target/x86_64-unknown-linux-musl/carbon-cli/carbon "target/carbon-matrix/$MATRIX_ASSET"
  emit_posix_tarballs "$MATRIX_ASSET" "target/carbon-matrix/$MATRIX_ASSET"
  "target/carbon-matrix/$MATRIX_ASSET" --version
elif [[ "$MATRIX_LABEL" == "linux-aarch64" ]]; then
  rustup target add aarch64-unknown-linux-musl
  cargo build --locked --profile carbon-cli --target aarch64-unknown-linux-musl \
    -p jope-carbon --no-default-features --features cli --bin carbon
  cp target/aarch64-unknown-linux-musl/carbon-cli/carbon "target/carbon-matrix/$MATRIX_ASSET"
  emit_posix_tarballs "$MATRIX_ASSET" "target/carbon-matrix/$MATRIX_ASSET"
  if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
    "target/carbon-matrix/$MATRIX_ASSET" --version
  fi
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
  for darwin_slice in carbon-darwin-arm64 carbon-darwin-x86_64 carbon-darwin-universal; do
    emit_posix_tarballs "$darwin_slice" "target/carbon-matrix/$darwin_slice"
  done
  target/carbon-matrix/carbon-darwin-universal --version
elif [[ "$MATRIX_LABEL" == "windows-x86_64" || "$MATRIX_LABEL" == "windows-arm64" || "$MATRIX_LABEL" == "windows-aarch64" ]]; then
  rustup target add "$MATRIX_TARGET"
  if [[ "$MATRIX_TARGET" == *-pc-windows-msvc ]]; then
    if ! command -v nasm >/dev/null 2>&1; then
      export AWS_LC_SYS_NO_ASM=1
    fi
    export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="${CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS:+$CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS }-C target-feature=+crt-static"
    export CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS="${CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS:+$CARGO_TARGET_AARCH64_PC_WINDOWS_MSVC_RUSTFLAGS }-C target-feature=+crt-static"
  fi
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
else
  echo "unsupported matrix label: $MATRIX_LABEL" >&2
  exit 2
fi

(
  cd target/carbon-matrix
  rm -f "SHA256SUMS-${MATRIX_LABEL}.txt"
  if command -v sha256sum >/dev/null 2>&1; then
    find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -exec basename {} \; | LC_ALL=C sort | xargs sha256sum > "SHA256SUMS-${MATRIX_LABEL}.txt"
  else
    find . -maxdepth 1 -type f ! -name 'SHA256SUMS*' -exec basename {} \; | LC_ALL=C sort | xargs shasum -a 256 > "SHA256SUMS-${MATRIX_LABEL}.txt"
  fi
)
