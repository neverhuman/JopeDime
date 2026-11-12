#!/bin/sh
# Carbon Zero-Knowledge Serverless CLI Installer (POSIX sh: Ubuntu, Debian, Alpine/musl).
#
# One-line install:
#   curl -fsSL https://github.com/neverhuman/JopeDime/releases/download/carbon-cli-v0.4.3-live.1/install.sh | sh
#
# With options:
#   curl -fsSL .../install.sh | sh -s -- --bin-dir /usr/local/bin --skip-ping

set -eu

DEFAULT_TAG="carbon-cli-v0.4.3-live.1"
SIGNING_PUB_FINGERPRINT="de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e"

TAG="$DEFAULT_TAG"
BIN_DIR=""
CONFIG_DIR=""
BASE_URL=""
SOURCE_DIR=""
TEST_PUBKEY_FILE=""
SKIP_PING=0

say() {
  printf '[carbon-install] %s\n' "$*" >&2
}

die() {
  printf '[carbon-install] ERROR: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version|--tag)
      [ "$#" -ge 2 ] || die "--version requires an argument"
      TAG="$2"
      shift 2
      ;;
    --bin-dir)
      [ "$#" -ge 2 ] || die "--bin-dir requires an absolute directory path"
      BIN_DIR="$2"
      shift 2
      ;;
    --config-dir)
      [ "$#" -ge 2 ] || die "--config-dir requires an absolute directory path"
      CONFIG_DIR="$2"
      shift 2
      ;;
    --base-url)
      [ "$#" -ge 2 ] || die "--base-url requires a URL"
      BASE_URL="$2"
      shift 2
      ;;
    --source-dir)
      [ "$#" -ge 2 ] || die "--source-dir requires a directory path"
      SOURCE_DIR="$2"
      shift 2
      ;;
    --allow-test-pubkey)
      [ "$#" -ge 2 ] || die "--allow-test-pubkey requires a PEM file path"
      TEST_PUBKEY_FILE="$2"
      shift 2
      ;;
    --skip-ping)
      SKIP_PING=1
      shift 1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: install.sh [OPTIONS]

Options:
  --version <TAG>              Release tag (default: carbon-cli-v0.4.3-live.1)
  --bin-dir <ABS_DIR>          Binary destination (default: $HOME/.local/bin)
  --config-dir <DIR>           Config destination (default: $HOME/.config/jopedime)
  --base-url <URL>             Override HTTPS release download prefix
  --source-dir <DIR>           Install from a local verified release directory (offline)
  --allow-test-pubkey <FILE>   Override ECDSA-P256 public key for CI/E2E test fixtures only
  --skip-ping                  Skip post-install live `/v1/carbon/ping` probe
EOF
      exit 0
      ;;
    *)
      die "unknown option '$1' (pass --help for usage)"
      ;;
  esac
done

if [ -z "$BIN_DIR" ] || [ -z "$CONFIG_DIR" ]; then
  case "${HOME:-}" in
    /*) ;;
    *)
      die "HOME must be set to an absolute path when --bin-dir or --config-dir is omitted"
      ;;
  esac
  [ -n "$BIN_DIR" ] || BIN_DIR="$HOME/.local/bin"
  [ -n "$CONFIG_DIR" ] || CONFIG_DIR="$HOME/.config/jopedime"
fi

case "$BIN_DIR" in
  /*) ;;
  *) die "--bin-dir must be an absolute path (got '$BIN_DIR')" ;;
esac
case "$CONFIG_DIR" in
  /*) ;;
  *) die "--config-dir must be an absolute path (got '$CONFIG_DIR')" ;;
esac

OS_NAME="$(uname -s)"
[ "$OS_NAME" = "Linux" ] || die "unsupported operating system '$OS_NAME' (Linux static musl binaries only)"

ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
  x86_64|amd64)
    BIN_ASSET="carbon-linux-x86_64-musl"
    ;;
  aarch64|arm64)
    BIN_ASSET="carbon-linux-aarch64-musl"
    ;;
  *)
    die "unsupported CPU architecture '$ARCH_RAW' (expected x86_64 or aarch64)"
    ;;
esac

for req_cmd in openssl sha256sum base64 awk grep; do
  command -v "$req_cmd" >/dev/null 2>&1 || die "required command '$req_cmd' is not installed"
done
if [ -z "$SOURCE_DIR" ]; then
  command -v curl >/dev/null 2>&1 || die "required command 'curl' is not installed"
fi

if [ -z "$BASE_URL" ]; then
  BASE_URL="https://github.com/neverhuman/JopeDime/releases/download/$TAG"
fi

mkdir -p "$BIN_DIR" "$CONFIG_DIR"
chmod 0700 "$CONFIG_DIR" 2>/dev/null || true

STAGE_DIR="$CONFIG_DIR/.install-stage.$$"
rm -rf "$STAGE_DIR"
mkdir -m 0700 "$STAGE_DIR"
trap 'rm -rf "$STAGE_DIR"' EXIT INT TERM

if [ -n "$TEST_PUBKEY_FILE" ]; then
  [ -f "$TEST_PUBKEY_FILE" ] || die "missing test public key file '$TEST_PUBKEY_FILE'"
  say "WARNING: using test public key override ($TEST_PUBKEY_FILE) — NOT for production installs"
  cp "$TEST_PUBKEY_FILE" "$STAGE_DIR/signing.pub"
else
  cat > "$STAGE_DIR/signing.pub" <<'EOF'
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMx44DTDE4ObFoh7otiKzAJA9fd8U
eEunkkDgvRCL74VyxIW9Q0wt49FMtTqv7aH7sj//B2uqMwPdm/MlBu45Nw==
-----END PUBLIC KEY-----
EOF

  GOT_PUB_SHA="$(sha256sum "$STAGE_DIR/signing.pub" | awk '{print $1}')"
  [ "$GOT_PUB_SHA" = "$SIGNING_PUB_FINGERPRINT" ] || die "embedded release public key fingerprint mismatch ($GOT_PUB_SHA)"
fi

fetch_asset() {
  asset_name="$1"
  dest_path="$2"
  if [ -n "$SOURCE_DIR" ]; then
    [ -f "$SOURCE_DIR/$asset_name" ] || die "missing $SOURCE_DIR/$asset_name"
    cp "$SOURCE_DIR/$asset_name" "$dest_path"
  else
    curl -fsSL "${BASE_URL%/}/$asset_name" -o "$dest_path" || die "download failed: ${BASE_URL%/}/$asset_name"
  fi
}

say "Fetching Carbon release '$TAG' ($BIN_ASSET)..."
fetch_asset "SHA256SUMS.txt" "$STAGE_DIR/SHA256SUMS.txt"
fetch_asset "SHA256SUMS.txt.sig" "$STAGE_DIR/SHA256SUMS.txt.sig"
fetch_asset "carbon-clouds.json" "$STAGE_DIR/carbon-clouds.json"
fetch_asset "$BIN_ASSET" "$STAGE_DIR/$BIN_ASSET"

say "Verifying ECDSA-P256 release signature..."
base64 -d < "$STAGE_DIR/SHA256SUMS.txt.sig" > "$STAGE_DIR/SHA256SUMS.txt.sig.der" \
  || die "malformed base64 signature SHA256SUMS.txt.sig"
openssl dgst -sha256 -verify "$STAGE_DIR/signing.pub" \
  -signature "$STAGE_DIR/SHA256SUMS.txt.sig.der" "$STAGE_DIR/SHA256SUMS.txt" >/dev/null 2>&1 \
  || die "ECDSA-P256 signature verification FAILED for SHA256SUMS.txt"

say "Verifying SHA-256 artifact digests..."
for required_asset in "$BIN_ASSET" "carbon-clouds.json"; do
  expected_line="$(grep "  ${required_asset}\$" "$STAGE_DIR/SHA256SUMS.txt" || true)"
  [ -n "$expected_line" ] || die "$required_asset missing from signed SHA256SUMS.txt"
  expected_hex="$(printf '%s\n' "$expected_line" | awk '{print $1}')"
  actual_hex="$(sha256sum "$STAGE_DIR/$required_asset" | awk '{print $1}')"
  [ "$expected_hex" = "$actual_hex" ] \
    || die "SHA-256 mismatch for $required_asset (expected $expected_hex, got $actual_hex)"
done

chmod 0755 "$STAGE_DIR/$BIN_ASSET"
chmod 0644 "$STAGE_DIR/carbon-clouds.json"
cp "$STAGE_DIR/$BIN_ASSET" "$BIN_DIR/.carbon.new.$$"
chmod 0755 "$BIN_DIR/.carbon.new.$$"
mv -f "$BIN_DIR/.carbon.new.$$" "$BIN_DIR/carbon"

cp "$STAGE_DIR/carbon-clouds.json" "$CONFIG_DIR/.carbon-clouds.new.$$"
chmod 0644 "$CONFIG_DIR/.carbon-clouds.new.$$"
mv -f "$CONFIG_DIR/.carbon-clouds.new.$$" "$CONFIG_DIR/carbon-clouds.json"

say "Installed verified static binary -> $BIN_DIR/carbon"
say "Installed verified cloud pins    -> $CONFIG_DIR/carbon-clouds.json"

CARBON_CONFIG_DIR="$CONFIG_DIR" "$BIN_DIR/carbon" version >&2

if [ "$SKIP_PING" -eq 0 ]; then
  say "Probing live Carbon prod cloud..."
  CARBON_CONFIG_DIR="$CONFIG_DIR" "$BIN_DIR/carbon" ping --cloud prod >&2
fi

case ":${PATH:-}:" in
  *":$BIN_DIR:"*) ;;
  *)
    say "NOTE: Add $BIN_DIR to your PATH: export PATH=\"$BIN_DIR:\$PATH\""
    ;;
esac

say "Ready! Quickstart:"
say "  Register Free (30d): $BIN_DIR/carbon register --free --cloud prod"
say "  Run Free:            $BIN_DIR/carbon run --free --cloud prod --effort medium --synthetic-rows 500 --features 6 --shred"
say "  Run Pro (with tok):  $BIN_DIR/carbon run --pro --cloud prod --token <ctk1...> --effort high --csv data.csv --target y --shred"
