#!/usr/bin/env bash
# End-to-end deterministic Carbon release packager, staged dev -> prod deployer,
# ECDSA-P256 release signer, and GitHub Release publisher.
#
# Usage:
#   AWS_PROFILE=veox-jepson-prod bash deploy/carbon/release-and-promote.sh \
#     --tag carbon-cli-v1.1.0 \
#     --r882-dir target/lambda-pkgs/layer-root/jope-models/r882 \
#     --out-dir target/carbon-release-v1.1.0

set -euo pipefail

TAG="carbon-cli-v1.1.0"
R882_DIR="target/lambda-pkgs/layer-root/jope-models/r882"
OUT_DIR=""
PROD_REGIONS="wave1"
SIGNING_KEY="/etc/jope-release/signing.key"
COSIGN_BIN="/home/ubuntu/.local/bin/cosign"
COSIGN_SHA256="caaad125acef1cb81d58dcdc454a1e429d09a750d1e9e2b3ed1aed8964454708"
EXTRA_ASSETS_DIR=""
SKIP_DEPLOY=0
SKIP_GH_RELEASE=0

die() {
  printf '[carbon-release] ERROR: %s\n' "$*" >&2
  exit 1
}

say() {
  printf '[carbon-release] %s\n' "$*" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --r882-dir) R882_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --prod-regions) PROD_REGIONS="$2"; shift 2 ;;
    --signing-key) SIGNING_KEY="$2"; shift 2 ;;
    --extra-assets-dir) EXTRA_ASSETS_DIR="$2"; shift 2 ;;
    --skip-deploy) SKIP_DEPLOY=1; shift 1 ;;
    --skip-gh-release) SKIP_GH_RELEASE=1; shift 1 ;;
    *) die "unknown argument '$1'" ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree must be clean before cutting an immutable Carbon release"
fi

SOURCE_SHA="$(git rev-parse HEAD)"
[[ ${#SOURCE_SHA} -eq 40 ]] || die "failed to resolve 40-char HEAD SHA"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$REPO_ROOT/target/carbon-release-$TAG-${SOURCE_SHA:0:12}"
fi
mkdir -p "$OUT_DIR"

export JOPE_BUILD_SOURCE_SHA="$SOURCE_SHA"
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"

say "Step 1/5: Packaging ARM64 Lambda cells (carbon-edge + carbon-worker + r882-layer) + Linux musl thin CLIs at $SOURCE_SHA..."
cargo run --release --locked -p jope-carbon --bin carbon -- package \
  --r882-dir "$R882_DIR" \
  --out-dir "$OUT_DIR" \
  --jobs "$CARGO_BUILD_JOBS" \
  --include-cli-binaries

if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
  say "Step 2/5: Deploying and red-teaming dev cell (us-east-2)..."
  cargo run --release --locked -p jope-carbon --bin carbon -- deploy \
    --manifest "$OUT_DIR/release-manifest.json" \
    --env dev \
    --regions us-east-2 \
    --bootstrap-secrets \
    --json-out "$OUT_DIR/deploy-dev-receipt.json"

  say "Step 3/5: Promoting and red-teaming prod cells ($PROD_REGIONS)..."
  cargo run --release --locked -p jope-carbon --bin carbon -- deploy \
    --manifest "$OUT_DIR/release-manifest.json" \
    --env prod \
    --regions "$PROD_REGIONS" \
    --bootstrap-secrets \
    --json-out "$OUT_DIR/deploy-prod-receipt.json"

  say "Step 4/5: Generating multi-region pinned carbon-clouds.json..."
  cargo run --release --locked -p jope-carbon --bin carbon -- clouds-manifest \
    --dev-regions us-east-2 \
    --prod-regions "$PROD_REGIONS" \
    --out "$OUT_DIR/carbon-clouds.json"
fi

cp deploy/carbon/install.sh "$OUT_DIR/install.sh"
cp deploy/carbon/install.ps1 "$OUT_DIR/install.ps1"

if [[ -n "$EXTRA_ASSETS_DIR" ]]; then
  [[ -d "$EXTRA_ASSETS_DIR" ]] || die "missing --extra-assets-dir '$EXTRA_ASSETS_DIR'"
  (
    shopt -s nullglob
    for extra in "$EXTRA_ASSETS_DIR"/carbon-linux-* "$EXTRA_ASSETS_DIR"/carbon-darwin-* "$EXTRA_ASSETS_DIR"/carbon-windows-*; do
      cp -f "$extra" "$OUT_DIR/$(basename "$extra")"
    done
  )
fi

if [[ -f "$OUT_DIR/carbon-clouds.json" ]]; then
  (
    cd "$OUT_DIR"
    rm -f SHA256SUMS.txt SHA256SUMS.txt.sig
    shopt -s nullglob
    assets=(
      carbon-linux-*
      carbon-darwin-*
      carbon-windows-*
      carbon-clouds.json
      install.sh
      install.ps1
    )
    [[ ${#assets[@]} -gt 0 ]] || exit 1
    LC_ALL=C sha256sum "${assets[@]}" | LC_ALL=C sort -k2 > SHA256SUMS.txt
  )

  say "Step 5/5: Signing SHA256SUMS.txt with $SIGNING_KEY and verifying against docs/release/signing.pub..."
  SIGNING_DIR="$(dirname "$SIGNING_KEY")"
  if sudo -n test -f "$SIGNING_DIR/signing.password" 2>/dev/null; then
    [[ -x "$COSIGN_BIN" ]] || die "missing cosign binary at '$COSIGN_BIN'"
    IFS= read -r -d '' SIGN_SCRIPT <<'ROOT' || true
set -eu
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
work=$(mktemp -d /root/jope-carbon-sign.XXXXXX)
trap 'rm -rf -- "$work"' EXIT
TMPDIR=$work/tmp
export TMPDIR
mkdir -m 0700 -- "$TMPDIR"
cp -- "$3" "$work/cosign"
chmod 0500 -- "$work/cosign"
got=$(sha256sum < "$work/cosign")
got=${got%% *}
if [ "$got" != "$4" ]; then
  echo "cosign at $3 has sha256 $got, not pinned $4; refusing to sign" >&2
  exit 3
fi
cd "$5"
COSIGN_PASSWORD="$(cat signing.password)" "$work/cosign" sign-blob --yes --key "$6" \
  --tlog-upload=false --output-signature "$1" "$2"
chown "$7:$8" "$1"
chmod 0644 "$1"
ROOT
    sudo -n /bin/sh -c "$SIGN_SCRIPT" sign \
      "$OUT_DIR/SHA256SUMS.txt.sig" \
      "$OUT_DIR/SHA256SUMS.txt" \
      "$COSIGN_BIN" \
      "$COSIGN_SHA256" \
      "$SIGNING_DIR" \
      "$SIGNING_KEY" \
      "$(id -u)" \
      "$(id -g)"
    printf '\n' >> "$OUT_DIR/SHA256SUMS.txt.sig"
    TUF_ROOT="$OUT_DIR/.tuf-verify" "$COSIGN_BIN" verify-blob \
      --key docs/release/signing.pub \
      --insecure-ignore-tlog=true \
      --signature "$OUT_DIR/SHA256SUMS.txt.sig" \
      "$OUT_DIR/SHA256SUMS.txt" >/dev/null
    rm -rf "$OUT_DIR/.tuf-verify"
  else
    openssl dgst -sha256 -sign "$SIGNING_KEY" "$OUT_DIR/SHA256SUMS.txt" \
      | openssl base64 -A > "$OUT_DIR/SHA256SUMS.txt.sig"
    printf '\n' >> "$OUT_DIR/SHA256SUMS.txt.sig"
  fi

  tr -d ' \t\r\n' < "$OUT_DIR/SHA256SUMS.txt.sig" | openssl base64 -d -A -out "$OUT_DIR/SHA256SUMS.txt.sig.der"
  openssl dgst -sha256 -verify docs/release/signing.pub \
    -signature "$OUT_DIR/SHA256SUMS.txt.sig.der" "$OUT_DIR/SHA256SUMS.txt" >/dev/null
  rm -f "$OUT_DIR/SHA256SUMS.txt.sig.der"

  if [[ "$SKIP_GH_RELEASE" -eq 0 ]]; then
    GH_ROLE_CMD="${GH_ROLE_BIN:-$(command -v gh-role || printf '%s/.local/bin/gh-role' "$HOME")}"
    say "Publishing immutable GitHub Release '$TAG'..."
    (
      shopt -s nullglob
      rel_assets=(
        "$OUT_DIR"/SHA256SUMS.txt
        "$OUT_DIR"/SHA256SUMS.txt.sig
        "$OUT_DIR"/carbon-clouds.json
        "$OUT_DIR"/install.sh
        "$OUT_DIR"/install.ps1
        "$OUT_DIR"/carbon-linux-*
        "$OUT_DIR"/carbon-darwin-*
        "$OUT_DIR"/carbon-windows-*
      )
      "$GH_ROLE_CMD" writer release create "$TAG" \
        --repo neverhuman/JopeDime \
        --target "$SOURCE_SHA" \
        --latest \
        --title "Carbon Zero-Knowledge Serverless CLI ($TAG)" \
        --notes "Source SHA: \`$SOURCE_SHA\`. Verify with \`carbon self-verify\` or \`openssl dgst -sha256 -verify docs/release/signing.pub\`." \
        "${rel_assets[@]}"
    )
  fi
fi

say "Completed Carbon release pipeline -> $OUT_DIR"
