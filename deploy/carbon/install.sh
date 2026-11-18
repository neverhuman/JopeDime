#!/bin/sh
# Carbon Zero-Knowledge Serverless CLI Installer (POSIX sh: Linux glibc/musl + macOS arm64/x86_64).
#
# One-line install (official HTTPS endpoint with GitHub Releases failover):
#   curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh
#   curl -fsSL https://github.com/neverhuman/JopeDime/releases/latest/download/install.sh | sh
#
# Windows PowerShell:
#   irm https://downloads.neverhuman.org/carbon/install.ps1 | iex
#
# With options:
#   curl -fsSL https://downloads.neverhuman.org/carbon/install.sh | sh -s -- --bin-dir "$HOME/.local/bin" --skip-ping

set -eu

DEFAULT_TAG="carbon-cli-v1.3.0"
SIGNING_PUB_FINGERPRINT="de84945ec5326877d202d0738e655492a8b2c3b794873a2eb248939c4d326e4e"

TAG="$DEFAULT_TAG"
BIN_DIR=""
CONFIG_DIR=""
BASE_URL=""
EXPLICIT_BASE_URL=0
SOURCE_DIR=""
TEST_PUBKEY_FILE=""
SKIP_PING=0
MODIFY_PATH=0
UNINSTALL=0

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
      EXPLICIT_BASE_URL=1
      shift 2
      ;;
    --source-dir)
      [ "$#" -ge 2 ] || die "--source-dir requires a directory path"
      SOURCE_DIR="$2"
      SKIP_PING=1
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
    --modify-path)
      MODIFY_PATH=1
      shift 1
      ;;
    --no-modify-path)
      MODIFY_PATH=0
      shift 1
      ;;
    --uninstall)
      UNINSTALL=1
      shift 1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: install.sh [OPTIONS]

Options:
  --version <TAG>              Release tag (default: carbon-cli-v1.3.0)
  --bin-dir <ABS_DIR>          Binary destination (default: $HOME/.local/bin)
  --config-dir <ABS_DIR>       Config destination (default: $HOME/.config/jopedime)
  --base-url <URL>             Override HTTPS release download prefix
  --source-dir <DIR>           Install from a local verified release directory (offline)
  --allow-test-pubkey <FILE>   Override ECDSA-P256 public key for CI/E2E test fixtures only
  --skip-ping                  Skip post-install live `/v1/carbon/ping` probe
  --modify-path                Append --bin-dir export to shell profile if missing from PATH
  --no-modify-path             Do not edit shell profile files (default)
  --uninstall                  Remove installed Carbon binary and release manifests
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

if [ "$UNINSTALL" -eq 1 ]; then
  if [ -f "$BIN_DIR/carbon" ]; then
    rm -f "$BIN_DIR/carbon"
    say "Removed $BIN_DIR/carbon"
  fi
  for stale_bin in "$BIN_DIR"/.carbon.new.* "$BIN_DIR"/carbon.upgrade.* "$BIN_DIR"/carbon.exe.old.*; do
    if [ -f "$stale_bin" ]; then
      rm -f "$stale_bin"
      say "Removed staging artifact $stale_bin"
    fi
  done
  for cfg_asset in "carbon-clouds.json" "SHA256SUMS.txt" "SHA256SUMS.txt.sig"; do
    if [ -f "$CONFIG_DIR/$cfg_asset" ]; then
      rm -f "$CONFIG_DIR/$cfg_asset"
      say "Removed $CONFIG_DIR/$cfg_asset"
    fi
  done
  say "Uninstall complete (saved zero-knowledge run keys under $CONFIG_DIR/runs were preserved)."
  exit 0
fi

check_writable_target_dir() {
  target_dir="$1"
  flag_name="$2"
  probe="$target_dir"
  while [ ! -e "$probe" ]; do
    parent="$(dirname "$probe")"
    [ "$parent" != "$probe" ] || break
    probe="$parent"
  done
  if [ ! -w "$probe" ]; then
    die "$flag_name '$target_dir' is not writable by current user (closest existing parent '$probe' lacks write permission; pass --bin-dir \"\$HOME/.local/bin\" or run with appropriate privileges)"
  fi
}

check_writable_target_dir "$BIN_DIR" "--bin-dir"
check_writable_target_dir "$CONFIG_DIR" "--config-dir"

OS_NAME="$(uname -s)"
ARCH_RAW="$(uname -m)"

CANDIDATE_ASSETS=""
case "$OS_NAME" in
  Linux)
    case "$ARCH_RAW" in
      x86_64|amd64)
        CANDIDATE_ASSETS="carbon-linux-x86_64-musl"
        ;;
      aarch64|arm64|armv8*)
        USERLAND_BITS="$(getconf LONG_BIT 2>/dev/null || printf '64')"
        if [ "$ARCH_RAW" = "armv8l" ] || [ "$USERLAND_BITS" = "32" ]; then
          CANDIDATE_ASSETS="carbon-linux-armv7-musleabihf carbon-linux-aarch64-musl"
        else
          CANDIDATE_ASSETS="carbon-linux-aarch64-musl carbon-linux-armv7-musleabihf"
        fi
        ;;
      armv7l|armv7|armhf)
        CANDIDATE_ASSETS="carbon-linux-armv7-musleabihf"
        ;;
      *)
        die "unsupported Linux CPU architecture '$ARCH_RAW' (expected x86_64, aarch64, armv8l, or armv7l)"
        ;;
    esac
    ;;
  Darwin)
    # Detect Apple Silicon even when invoked inside an x86_64 Rosetta 2 shell or minimal PATH.
    IS_APPLE_ARM64=0
    if [ "$ARCH_RAW" = "arm64" ] || [ "$ARCH_RAW" = "aarch64" ]; then
      IS_APPLE_ARM64=1
    elif [ -x /usr/sbin/sysctl ]; then
      if [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || printf '0')" = "1" ]; then
        IS_APPLE_ARM64=1
      fi
    elif command -v sysctl >/dev/null 2>&1; then
      if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || printf '0')" = "1" ]; then
        IS_APPLE_ARM64=1
      fi
    fi
    if [ "$IS_APPLE_ARM64" -eq 1 ]; then
      CANDIDATE_ASSETS="carbon-darwin-arm64 carbon-darwin-universal"
    else
      case "$ARCH_RAW" in
        x86_64|amd64)
          CANDIDATE_ASSETS="carbon-darwin-x86_64 carbon-darwin-universal"
          ;;
        *)
          die "unsupported macOS CPU architecture '$ARCH_RAW' (expected arm64 or x86_64)"
          ;;
      esac
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    die "Windows detected ('$OS_NAME'); install with PowerShell: irm https://downloads.neverhuman.org/carbon/install.ps1 | iex"
    ;;
  *)
    die "unsupported operating system '$OS_NAME' (supported: Linux, Darwin/macOS, and Windows via install.ps1)"
    ;;
esac

for req_cmd in awk grep mktemp; do
  if ! command -v "$req_cmd" >/dev/null 2>&1; then
    die "required command '$req_cmd' is not installed"
  fi
done

HAVE_OPENSSL=0
if command -v openssl >/dev/null 2>&1; then
  HAVE_OPENSSL=1
fi
if [ "$HAVE_OPENSSL" -eq 0 ] && ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
  die "none of 'sha256sum', 'shasum', or 'openssl' is installed (required for SHA-256 verification)"
fi

DOWNLOADER=""
if [ -z "$SOURCE_DIR" ]; then
  if command -v curl >/dev/null 2>&1; then
    DOWNLOADER="curl"
  elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
  else
    die "neither 'curl' nor 'wget' is installed (required for HTTPS download)"
  fi
fi

TELEMETRY_OPTOUT=0
if [ "${DO_NOT_TRACK:-0}" = "1" ] || [ "${CARBON_NO_TELEMETRY:-0}" = "1" ]; then
  TELEMETRY_OPTOUT=1
fi
if [ "$TELEMETRY_OPTOUT" -eq 1 ]; then
  INSTALLER_UA="jope-carbon-installer/1"
else
  INSTALLER_UA="jope-carbon-installer/1 ($TAG; $OS_NAME-$ARCH_RAW)"
fi

compute_sha256() {
  file_path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file_path" | awk '{print tolower($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file_path" | awk '{print tolower($1)}'
  else
    openssl dgst -sha256 -r "$file_path" | awk '{print tolower($1)}'
  fi
}

decode_base64_file() {
  in_file="$1"
  out_file="$2"
  tr -d ' \t\r\n' < "$in_file" > "${in_file}.clean"
  if openssl base64 -d -A -in "${in_file}.clean" -out "$out_file" 2>/dev/null; then
    rm -f "${in_file}.clean"
    return 0
  fi
  if command -v base64 >/dev/null 2>&1; then
    if base64 -d < "${in_file}.clean" > "$out_file" 2>/dev/null || base64 -D < "${in_file}.clean" > "$out_file" 2>/dev/null; then
      rm -f "${in_file}.clean"
      return 0
    fi
  fi
  rm -f "${in_file}.clean"
  return 1
}

if [ -z "$BASE_URL" ]; then
  if [ "$TELEMETRY_OPTOUT" -eq 1 ]; then
    BASE_URL="https://github.com/neverhuman/JopeDime/releases/download/$TAG"
    MIRROR_URL="https://downloads.neverhuman.org/carbon/releases/$TAG"
  else
    BASE_URL="https://downloads.neverhuman.org/carbon/releases/$TAG"
    MIRROR_URL="https://github.com/neverhuman/JopeDime/releases/download/$TAG"
  fi
else
  case "$BASE_URL" in
    https://*|http://127.0.0.1:*|http://127.0.0.1/*|http://localhost:*|http://localhost/*|http://\[::1\]:*|http://\[::1\]/*) ;;
    *) die "insecure or unsupported --base-url '$BASE_URL' (HTTPS required for non-loopback hosts)" ;;
  esac
  MIRROR_URL="https://github.com/neverhuman/JopeDime/releases/download/$TAG"
fi

STAGE_PARENT="${CARBON_INSTALL_STAGE_ROOT:-}"
if [ -z "$STAGE_PARENT" ]; then
  if [ -n "${HOME:-}" ] && [ -d "$HOME" ] && [ -w "$HOME" ]; then
    STAGE_PARENT="$HOME/.cache"
    mkdir -p "$STAGE_PARENT" 2>/dev/null || STAGE_PARENT="$(dirname "$CONFIG_DIR")"
  else
    STAGE_PARENT="$(dirname "$CONFIG_DIR")"
  fi
fi
mkdir -p "$STAGE_PARENT"

umask 077
STAGE_DIR="$(mktemp -d "$STAGE_PARENT/.carbon-install-stage.XXXXXX")" \
  || die "failed to create secure staging directory under '$STAGE_PARENT'"
chmod 0700 "$STAGE_DIR"
trap 'rm -rf "$STAGE_DIR"' EXIT INT TERM

if [ -n "$TEST_PUBKEY_FILE" ]; then
  [ -f "$TEST_PUBKEY_FILE" ] || die "missing test public key file '$TEST_PUBKEY_FILE'"
  say "WARNING: using test public key override ($TEST_PUBKEY_FILE) — NOT for production installs"
  tr -d '\r' < "$TEST_PUBKEY_FILE" > "$STAGE_DIR/signing.pub"
else
  cat > "$STAGE_DIR/signing.pub" <<'EOF'
-----BEGIN PUBLIC KEY-----
MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEMx44DTDE4ObFoh7otiKzAJA9fd8U
eEunkkDgvRCL74VyxIW9Q0wt49FMtTqv7aH7sj//B2uqMwPdm/MlBu45Nw==
-----END PUBLIC KEY-----
EOF

  GOT_PUB_SHA="$(compute_sha256 "$STAGE_DIR/signing.pub")"
  [ "$GOT_PUB_SHA" = "$SIGNING_PUB_FINGERPRINT" ] || die "embedded release public key fingerprint mismatch ($GOT_PUB_SHA)"
fi

CUSTOM_CA_BUNDLE="${CARBON_CA_BUNDLE:-${SSL_CERT_FILE:-}}"
if [ -n "$CUSTOM_CA_BUNDLE" ] && [ ! -f "$CUSTOM_CA_BUNDLE" ]; then
  die "configured TLS CA bundle '$CUSTOM_CA_BUNDLE' does not exist"
fi

http_download_one() {
  src_url="$1"
  dst_file="$2"
  case "$src_url" in
    https://*)
      if [ "$DOWNLOADER" = "curl" ]; then
        if [ -n "$CUSTOM_CA_BUNDLE" ]; then
          curl -fsSL --proto '=https' --tlsv1.2 --cacert "$CUSTOM_CA_BUNDLE" --retry 5 --retry-connrefused --connect-timeout 10 -A "$INSTALLER_UA" "$src_url" -o "$dst_file"
        else
          curl -fsSL --proto '=https' --tlsv1.2 --retry 5 --retry-connrefused --connect-timeout 10 -A "$INSTALLER_UA" "$src_url" -o "$dst_file"
        fi
      else
        if [ -n "$CUSTOM_CA_BUNDLE" ]; then
          wget -q --https-only --secure-protocol=TLSv1_2 --ca-certificate="$CUSTOM_CA_BUNDLE" -T 15 -t 5 -U "$INSTALLER_UA" -O "$dst_file" "$src_url"
        else
          wget -q --https-only --secure-protocol=TLSv1_2 -T 15 -t 5 -U "$INSTALLER_UA" -O "$dst_file" "$src_url"
        fi
      fi
      ;;
    http://127.0.0.1:*|http://127.0.0.1/*|http://localhost:*|http://localhost/*|http://\[::1\]:*|http://\[::1\]/*)
      if [ "$DOWNLOADER" = "curl" ]; then
        curl -fsSL --retry 3 --connect-timeout 10 -A "$INSTALLER_UA" "$src_url" -o "$dst_file"
      else
        wget -q -T 15 -t 3 -U "$INSTALLER_UA" -O "$dst_file" "$src_url"
      fi
      ;;
    *)
      die "refusing non-HTTPS download URL '$src_url'"
      ;;
  esac
}

fetch_asset() {
  asset_name="$1"
  dest_path="$2"
  if [ -n "$SOURCE_DIR" ]; then
    [ -f "$SOURCE_DIR/$asset_name" ] || die "missing $SOURCE_DIR/$asset_name"
    cp "$SOURCE_DIR/$asset_name" "$dest_path"
  elif [ "$EXPLICIT_BASE_URL" -eq 1 ]; then
    http_download_one "${BASE_URL%/}/$asset_name" "$dest_path" \
      || die "download failed: ${BASE_URL%/}/$asset_name"
  else
    if ! http_download_one "${BASE_URL%/}/$asset_name" "$dest_path" 2>/dev/null; then
      say "Primary download unavailable for $asset_name; failing over to $MIRROR_URL..."
      http_download_one "${MIRROR_URL%/}/$asset_name" "$dest_path" \
        || die "download failed from both ${BASE_URL%/}/$asset_name and ${MIRROR_URL%/}/$asset_name"
    fi
  fi
}

manifest_has_asset() {
  needle="$1"
  awk -v n="$needle" '$2 == n || $2 == ("*" n) { found=1; exit } END { exit(found ? 0 : 1) }' "$STAGE_DIR/SHA256SUMS.txt"
}

manifest_digest_for() {
  needle="$1"
  awk -v n="$needle" '$2 == n || $2 == ("*" n) { print tolower($1); exit }' "$STAGE_DIR/SHA256SUMS.txt"
}

asset_available_in_source() {
  needle="$1"
  if [ -n "$SOURCE_DIR" ]; then
    [ -f "$SOURCE_DIR/$needle" ]
  else
    return 0
  fi
}

say "Fetching signed release manifest for '$TAG'..."
fetch_asset "SHA256SUMS.txt" "$STAGE_DIR/SHA256SUMS.txt"
fetch_asset "SHA256SUMS.txt.sig" "$STAGE_DIR/SHA256SUMS.txt.sig"

if [ "$HAVE_OPENSSL" -eq 1 ]; then
  say "Verifying ECDSA-P256 release signature via openssl..."
  decode_base64_file "$STAGE_DIR/SHA256SUMS.txt.sig" "$STAGE_DIR/SHA256SUMS.txt.sig.der" \
    || die "malformed base64 signature SHA256SUMS.txt.sig"
  openssl dgst -sha256 -verify "$STAGE_DIR/signing.pub" \
    -signature "$STAGE_DIR/SHA256SUMS.txt.sig.der" "$STAGE_DIR/SHA256SUMS.txt" >/dev/null 2>&1 \
    || die "ECDSA-P256 signature verification FAILED for SHA256SUMS.txt"
else
  say "NOTE: 'openssl' CLI not present; staging payload in 0700 sandbox for native aws-lc-rs ECDSA-P256 verification before install..."
fi

SELECTED_BASE=""
DOWNLOAD_ASSET=""
ARCHIVE_KIND="raw"

HAVE_TAR=0
command -v tar >/dev/null 2>&1 && HAVE_TAR=1
HAVE_ZSTD=0
command -v zstd >/dev/null 2>&1 && HAVE_ZSTD=1
HAVE_XZ=0
command -v xz >/dev/null 2>&1 && HAVE_XZ=1
HAVE_GZIP=0
command -v gzip >/dev/null 2>&1 && HAVE_GZIP=1

for base_candidate in $CANDIDATE_ASSETS; do
  if [ "$HAVE_TAR" -eq 1 ] && [ "$HAVE_ZSTD" -eq 1 ] \
    && manifest_has_asset "${base_candidate}.tar.zst" \
    && asset_available_in_source "${base_candidate}.tar.zst"; then
    SELECTED_BASE="$base_candidate"
    DOWNLOAD_ASSET="${base_candidate}.tar.zst"
    ARCHIVE_KIND="tar.zst"
    break
  fi
  if [ "$HAVE_TAR" -eq 1 ] && [ "$HAVE_XZ" -eq 1 ] \
    && manifest_has_asset "${base_candidate}.tar.xz" \
    && asset_available_in_source "${base_candidate}.tar.xz"; then
    SELECTED_BASE="$base_candidate"
    DOWNLOAD_ASSET="${base_candidate}.tar.xz"
    ARCHIVE_KIND="tar.xz"
    break
  fi
  if [ "$HAVE_TAR" -eq 1 ] && [ "$HAVE_GZIP" -eq 1 ] \
    && manifest_has_asset "${base_candidate}.tar.gz" \
    && asset_available_in_source "${base_candidate}.tar.gz"; then
    SELECTED_BASE="$base_candidate"
    DOWNLOAD_ASSET="${base_candidate}.tar.gz"
    ARCHIVE_KIND="tar.gz"
    break
  fi
  if manifest_has_asset "$base_candidate" && asset_available_in_source "$base_candidate"; then
    SELECTED_BASE="$base_candidate"
    DOWNLOAD_ASSET="$base_candidate"
    ARCHIVE_KIND="raw"
    break
  fi
done

[ -n "$DOWNLOAD_ASSET" ] || die "no signed Carbon asset for ($OS_NAME / $ARCH_RAW; tried: $CANDIDATE_ASSETS) found in SHA256SUMS.txt"

say "Fetching verified payload '$DOWNLOAD_ASSET' + 'carbon-clouds.json'..."
fetch_asset "carbon-clouds.json" "$STAGE_DIR/carbon-clouds.json"
fetch_asset "$DOWNLOAD_ASSET" "$STAGE_DIR/$DOWNLOAD_ASSET"

say "Verifying SHA-256 artifact digests..."
for required_asset in "$DOWNLOAD_ASSET" "carbon-clouds.json"; do
  expected_hex="$(manifest_digest_for "$required_asset")"
  [ -n "$expected_hex" ] || die "$required_asset missing from signed SHA256SUMS.txt"
  actual_hex="$(compute_sha256 "$STAGE_DIR/$required_asset")"
  [ "$expected_hex" = "$actual_hex" ] \
    || die "SHA-256 mismatch for $required_asset (expected $expected_hex, got $actual_hex)"
done

# Two-step decompression so POSIX `/bin/sh` never masks decompressor exit codes across a pipe.
mkdir -p "$STAGE_DIR/extracted"
case "$ARCHIVE_KIND" in
  tar.zst)
    zstd -dc "$STAGE_DIR/$DOWNLOAD_ASSET" > "$STAGE_DIR/payload.tar" \
      || die "zstd decompression failed for $DOWNLOAD_ASSET"
    tar -xf "$STAGE_DIR/payload.tar" -C "$STAGE_DIR/extracted" \
      || die "tar extraction failed for $DOWNLOAD_ASSET"
    rm -f "$STAGE_DIR/payload.tar"
    ;;
  tar.xz)
    xz -dc "$STAGE_DIR/$DOWNLOAD_ASSET" > "$STAGE_DIR/payload.tar" \
      || die "xz decompression failed for $DOWNLOAD_ASSET"
    tar -xf "$STAGE_DIR/payload.tar" -C "$STAGE_DIR/extracted" \
      || die "tar extraction failed for $DOWNLOAD_ASSET"
    rm -f "$STAGE_DIR/payload.tar"
    ;;
  tar.gz)
    gzip -dc "$STAGE_DIR/$DOWNLOAD_ASSET" > "$STAGE_DIR/payload.tar" \
      || die "gzip decompression failed for $DOWNLOAD_ASSET"
    tar -xf "$STAGE_DIR/payload.tar" -C "$STAGE_DIR/extracted" \
      || die "tar extraction failed for $DOWNLOAD_ASSET"
    rm -f "$STAGE_DIR/payload.tar"
    ;;
  raw)
    cp "$STAGE_DIR/$DOWNLOAD_ASSET" "$STAGE_DIR/extracted/carbon"
    ;;
esac

if [ -f "$STAGE_DIR/extracted/carbon" ]; then
  EXTRACTED_BIN="$STAGE_DIR/extracted/carbon"
elif [ -f "$STAGE_DIR/extracted/$SELECTED_BASE" ]; then
  EXTRACTED_BIN="$STAGE_DIR/extracted/$SELECTED_BASE"
else
  die "archive $DOWNLOAD_ASSET did not contain expected 'carbon' executable"
fi

if [ "$ARCHIVE_KIND" != "raw" ] && manifest_has_asset "$SELECTED_BASE"; then
  expected_bin_hex="$(manifest_digest_for "$SELECTED_BASE")"
  actual_bin_hex="$(compute_sha256 "$EXTRACTED_BIN")"
  [ "$expected_bin_hex" = "$actual_bin_hex" ] \
    || die "extracted binary SHA-256 mismatch against $SELECTED_BASE (expected $expected_bin_hex, got $actual_bin_hex)"
fi

if [ "$OS_NAME" = "Darwin" ] && command -v xattr >/dev/null 2>&1; then
  xattr -d com.apple.quarantine "$EXTRACTED_BIN" 2>/dev/null || true
fi

chmod 0755 "$EXTRACTED_BIN"
if ! CARBON_CONFIG_DIR="$STAGE_DIR" "$EXTRACTED_BIN" version >/dev/null 2>&1; then
  die "staged executable '$DOWNLOAD_ASSET' failed smoke execution on ($OS_NAME / $ARCH_RAW)"
fi

if [ "$HAVE_OPENSSL" -eq 0 ]; then
  if [ -n "$TEST_PUBKEY_FILE" ]; then
    CARBON_CONFIG_DIR="$STAGE_DIR" "$EXTRACTED_BIN" self-verify \
      --manifest "$STAGE_DIR/SHA256SUMS.txt" \
      --signature "$STAGE_DIR/SHA256SUMS.txt.sig" \
      --clouds-manifest "$STAGE_DIR/carbon-clouds.json" \
      --allow-test-pubkey "$TEST_PUBKEY_FILE" \
      --require-manifest >/dev/null \
      || die "ECDSA-P256 signature verification FAILED for SHA256SUMS.txt"
  else
    CARBON_CONFIG_DIR="$STAGE_DIR" "$EXTRACTED_BIN" self-verify \
      --manifest "$STAGE_DIR/SHA256SUMS.txt" \
      --signature "$STAGE_DIR/SHA256SUMS.txt.sig" \
      --clouds-manifest "$STAGE_DIR/carbon-clouds.json" \
      --require-manifest >/dev/null \
      || die "ECDSA-P256 signature verification FAILED for SHA256SUMS.txt"
  fi
fi

mkdir -p "$BIN_DIR" "$CONFIG_DIR"
chmod 0700 "$CONFIG_DIR" 2>/dev/null || true

BIN_TMP="$(mktemp "$BIN_DIR/.carbon.new.XXXXXX")" || die "failed to stage executable in '$BIN_DIR'"
cp "$EXTRACTED_BIN" "$BIN_TMP"
chmod 0755 "$BIN_TMP"
mv -f "$BIN_TMP" "$BIN_DIR/carbon"

for cfg_asset in "carbon-clouds.json" "SHA256SUMS.txt" "SHA256SUMS.txt.sig"; do
  CFG_TMP="$(mktemp "$CONFIG_DIR/.${cfg_asset}.new.XXXXXX")" || die "failed to stage '$cfg_asset' in '$CONFIG_DIR'"
  cp "$STAGE_DIR/$cfg_asset" "$CFG_TMP"
  chmod 0644 "$CFG_TMP"
  mv -f "$CFG_TMP" "$CONFIG_DIR/$cfg_asset"
done

say "Installed verified binary     -> $BIN_DIR/carbon ($DOWNLOAD_ASSET)"
say "Installed verified cloud pins -> $CONFIG_DIR/carbon-clouds.json"

CARBON_CONFIG_DIR="$CONFIG_DIR" "$BIN_DIR/carbon" version >&2
if [ -z "$TEST_PUBKEY_FILE" ]; then
  CARBON_CONFIG_DIR="$CONFIG_DIR" "$BIN_DIR/carbon" self-verify >/dev/null \
    || die "post-install 'carbon self-verify' failed"
fi

if [ "$SKIP_PING" -eq 0 ]; then
  say "Probing live Carbon prod cloud..."
  if ! CARBON_CONFIG_DIR="$CONFIG_DIR" "$BIN_DIR/carbon" ping --cloud prod >&2; then
    say "WARNING: post-install 'carbon ping --cloud prod' did not succeed (binary and cryptographic pins are verified; outbound HTTPS may be offline or restricted)."
  fi
fi

if [ "$OS_NAME" = "Linux" ] && { [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; }; then
  if command -v carbon.exe >/dev/null 2>&1; then
    say "NOTE: Windows 'carbon.exe' is also visible on WSL PATH; invoke Linux 'carbon' (not 'carbon.exe') inside WSL."
  fi
fi

case ":${PATH:-}:" in
  *":$BIN_DIR:"*) ;;
  *)
    SHELL_BASE="$(basename "${SHELL:-sh}")"
    if [ "$MODIFY_PATH" -eq 1 ] && [ -n "${HOME:-}" ]; then
      case "$SHELL_BASE" in
        bash)
          if [ "$OS_NAME" = "Darwin" ] && [ ! -f "$HOME/.bashrc" ]; then
            RC_FILE="$HOME/.bash_profile"
          else
            RC_FILE="$HOME/.bashrc"
          fi
          RC_LINE="export PATH=\"$BIN_DIR:\$PATH\""
          ;;
        zsh)
          RC_FILE="$HOME/.zshrc"
          RC_LINE="export PATH=\"$BIN_DIR:\$PATH\""
          ;;
        fish)
          RC_FILE="$HOME/.config/fish/config.fish"
          RC_LINE="fish_add_path \"$BIN_DIR\""
          mkdir -p "$(dirname "$RC_FILE")"
          ;;
        *)
          RC_FILE="$HOME/.profile"
          RC_LINE="export PATH=\"$BIN_DIR:\$PATH\""
          ;;
      esac
      if [ ! -f "$RC_FILE" ] || ! grep -F "$BIN_DIR" "$RC_FILE" >/dev/null 2>&1; then
        printf '\n# Added by Carbon CLI installer\n%s\n' "$RC_LINE" >> "$RC_FILE"
        say "Added $BIN_DIR to $RC_FILE (open a new shell or source it to activate)"
      fi
    else
      case "$SHELL_BASE" in
        bash) say "NOTE: Add $BIN_DIR to PATH: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc && source ~/.bashrc" ;;
        zsh)  say "NOTE: Add $BIN_DIR to PATH: echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.zshrc && source ~/.zshrc" ;;
        fish) say "NOTE: Add $BIN_DIR to PATH: fish_add_path \"$BIN_DIR\"" ;;
        *)    say "NOTE: Add $BIN_DIR to PATH: export PATH=\"$BIN_DIR:\$PATH\"" ;;
      esac
    fi
    ;;
esac

say "Ready! Quickstart:"
say "  1. 1-command demo:   $BIN_DIR/carbon quickstart"
say "  2. Run on your CSV:  $BIN_DIR/carbon data.csv"
say "  3. Verify integrity: $BIN_DIR/carbon self-verify && $BIN_DIR/carbon doctor"
say "  4. Register Free:    $BIN_DIR/carbon register --free"
