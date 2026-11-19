#!/usr/bin/env bash
# End-to-end deterministic Carbon release packager, staged dev -> prod deployer,
# ECDSA-P256 release signer, and GitHub Release publisher.
#
# Usage:
#   AWS_PROFILE=veox-jepson-prod bash deploy/carbon/release-and-promote.sh \
#     --tag carbon-cli-v1.3.3 \
#     --r882-dir target/lambda-pkgs/layer-root/jope-models/r882 \
#     --out-dir target/carbon-release-v1.3.3

set -euo pipefail

TAG="carbon-cli-v1.3.3"
R882_DIR="target/lambda-pkgs/layer-root/jope-models/r882"
OUT_DIR=""
PROD_REGIONS="wave1"
SIGNING_KEY="/etc/jope-release/signing.key"
COSIGN_BIN="${COSIGN_BIN:-$(command -v cosign || printf '%s/.local/bin/cosign' "${HOME:-}")}"
COSIGN_SHA256="caaad125acef1cb81d58dcdc454a1e429d09a750d1e9e2b3ed1aed8964454708"
EXTRA_ASSETS_DIR=""
STAGE_RELEASE_TAG=""
DEV_APP_PUBKEYS_B64=""
PROD_APP_PUBKEYS_B64=""
SKIP_DEPLOY=0
SKIP_GH_RELEASE=0
ALLOW_PARTIAL_MATRIX=0

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
    --stage-release-tag) STAGE_RELEASE_TAG="$2"; shift 2 ;;
    --dev-app-pubkeys-b64) DEV_APP_PUBKEYS_B64="$2"; shift 2 ;;
    --prod-app-pubkeys-b64) PROD_APP_PUBKEYS_B64="$2"; shift 2 ;;
    --skip-deploy) SKIP_DEPLOY=1; shift 1 ;;
    --skip-gh-release) SKIP_GH_RELEASE=1; shift 1 ;;
    --allow-partial-matrix) ALLOW_PARTIAL_MATRIX=1; shift 1 ;;
    *) die "unknown argument '$1'" ;;
  esac
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [[ -z "$DEV_APP_PUBKEYS_B64" && -s "$REPO_ROOT/deploy/carbon/pubkeys/veox-app-dev.pub" ]]; then
  DEV_APP_PUBKEYS_B64="$(tr -d '[:space:]' < "$REPO_ROOT/deploy/carbon/pubkeys/veox-app-dev.pub")"
fi
if [[ -z "$PROD_APP_PUBKEYS_B64" && -s "$REPO_ROOT/deploy/carbon/pubkeys/veox-app-prod.pub" ]]; then
  PROD_APP_PUBKEYS_B64="$(tr -d '[:space:]' < "$REPO_ROOT/deploy/carbon/pubkeys/veox-app-prod.pub")"
fi

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
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

say "Step 1/5: Packaging ARM64 Lambda cells (carbon-edge + carbon-worker + r882-layer) + Linux musl thin CLIs at $SOURCE_SHA..."
cargo run --release --locked -p jope-carbon --bin carbon -- package \
  --r882-dir "$R882_DIR" \
  --out-dir "$OUT_DIR" \
  --jobs "$CARGO_BUILD_JOBS" \
  --include-cli-binaries

if [[ "$SKIP_DEPLOY" -eq 0 ]]; then
  say "Step 1b/5: Running local Lambda Custom Runtime + SigV4 S3/SecretsManager preflight..."
  cargo build --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --no-default-features --features edge --bin carbon-edge
  cargo build --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --no-default-features --features eval --bin carbon-worker
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- emu --self-test --real-workers >/dev/null

  DEV_PUBKEY_ARGS=()
  if [[ -n "$DEV_APP_PUBKEYS_B64" ]]; then
    DEV_PUBKEY_ARGS=(--app-auth-pubkeys-b64 "$DEV_APP_PUBKEYS_B64")
  fi
  PROD_PUBKEY_ARGS=()
  if [[ -n "$PROD_APP_PUBKEYS_B64" ]]; then
    PROD_PUBKEY_ARGS=(--app-auth-pubkeys-b64 "$PROD_APP_PUBKEYS_B64")
  fi

  say "Step 2/5: Deploying and red-teaming dev cell (us-east-2)..."
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- deploy \
    --manifest "$OUT_DIR/release-manifest.json" \
    --env dev \
    --regions us-east-2 \
    --bootstrap-secrets \
    "${DEV_PUBKEY_ARGS[@]}" \
    --json-out "$OUT_DIR/deploy-dev-receipt.json"

  say "Step 2b/5: Running mandatory live dev E2E qualification gate BEFORE prod promotion..."
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- clouds-manifest \
    --dev-regions us-east-2 \
    --prod-regions "$PROD_REGIONS" \
    --out "$OUT_DIR/carbon-clouds-dev-stage.json"
  python3 "$REPO_ROOT/deploy/carbon/verify-live-cell.py" \
    --env dev \
    --regions us-east-2 \
    --carbon-bin "$OUT_DIR/carbon-linux-x86_64-musl" \
    --clouds-manifest "$OUT_DIR/carbon-clouds-dev-stage.json" \
    --json-out "$OUT_DIR/e2e-dev-receipt.json" >/dev/null
  rm -f "$OUT_DIR/carbon-clouds-dev-stage.json"

  say "Step 3/5: Promoting and red-teaming prod cells ($PROD_REGIONS)..."
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- deploy \
    --manifest "$OUT_DIR/release-manifest.json" \
    --env prod \
    --regions "$PROD_REGIONS" \
    --bootstrap-secrets \
    "${PROD_PUBKEY_ARGS[@]}" \
    --json-out "$OUT_DIR/deploy-prod-receipt.json"

  say "Step 4/5: Generating multi-region pinned carbon-clouds.json and verifying zero drift..."
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- clouds-manifest \
    --dev-regions us-east-2 \
    --prod-regions "$PROD_REGIONS" \
    --out "$OUT_DIR/carbon-clouds.json"
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- deploy \
    --env dev \
    --regions us-east-2 \
    --check-drift \
    --clouds-manifest "$OUT_DIR/carbon-clouds.json" \
    --json-out "$OUT_DIR/drift-dev-receipt.json" >/dev/null
  cargo run --release --locked -j "$CARGO_BUILD_JOBS" -p jope-carbon --bin carbon -- deploy \
    --env prod \
    --regions "$PROD_REGIONS" \
    --check-drift \
    --clouds-manifest "$OUT_DIR/carbon-clouds.json" \
    --json-out "$OUT_DIR/drift-prod-receipt.json" >/dev/null

  say "Step 4b/5: Running post-promotion live prod E2E qualification gate ($PROD_REGIONS)..."
  python3 "$REPO_ROOT/deploy/carbon/verify-live-cell.py" \
    --env prod \
    --regions "$PROD_REGIONS" \
    --carbon-bin "$OUT_DIR/carbon-linux-x86_64-musl" \
    --clouds-manifest "$OUT_DIR/carbon-clouds.json" \
    --json-out "$OUT_DIR/e2e-prod-receipt.json" >/dev/null
fi

cp deploy/carbon/install.sh "$OUT_DIR/install.sh"
cp deploy/carbon/install.ps1 "$OUT_DIR/install.ps1"
python3 - "$OUT_DIR/install.sh" "$OUT_DIR/install.ps1" "$TAG" <<'PY'
import re, sys
sh_path, ps1_path, tag = sys.argv[1], sys.argv[2], sys.argv[3]
with open(sh_path, "r", encoding="utf-8") as fh:
    sh = fh.read()
sh = re.sub(r'^DEFAULT_TAG="[^"]+"', f'DEFAULT_TAG="{tag}"', sh, count=1, flags=re.M)
sh = re.sub(r'\(default: carbon-cli-v[^)]+\)', f'(default: {tag})', sh, count=1)
with open(sh_path, "w", encoding="utf-8") as fh:
    fh.write(sh)
with open(ps1_path, "r", encoding="utf-8") as fh:
    ps1 = fh.read()
ps1 = re.sub(r'(\[string\]\$Version\s*=\s*)"[^"]+"', rf'\1"{tag}"', ps1, count=1)
with open(ps1_path, "w", encoding="utf-8") as fh:
    fh.write(ps1)
PY

if [[ "$ALLOW_PARTIAL_MATRIX" -ne 0 && "$TAG" == carbon-cli-v* && "$SKIP_GH_RELEASE" -eq 0 ]]; then
  die "--allow-partial-matrix is forbidden when publishing a production '$TAG' GitHub Release"
fi

if [[ -z "$STAGE_RELEASE_TAG" && -z "$EXTRA_ASSETS_DIR" && "$SKIP_GH_RELEASE" -eq 0 ]]; then
  STAGE_RELEASE_TAG="carbon-matrix-stage-${SOURCE_SHA:0:12}"
fi

GH_ROLE_CMD="${GH_ROLE_BIN:-$(command -v gh-role || printf '%s/.local/bin/gh-role' "$HOME")}"
if [[ -n "$STAGE_RELEASE_TAG" ]]; then
  STAGE_DL_DIR="$OUT_DIR/.hosted-matrix-stage"
  rm -rf "$STAGE_DL_DIR"
  mkdir -p "$STAGE_DL_DIR"
  say "Fetching source-bound Darwin & Windows CLI assets from draft release '$STAGE_RELEASE_TAG'..."
  "$GH_ROLE_CMD" writer release download "$STAGE_RELEASE_TAG" \
    --repo neverhuman/JopeDime \
    --dir "$STAGE_DL_DIR" \
    --clobber
  (
    cd "$STAGE_DL_DIR"
    for sum_file in SHA256SUMS-darwin-universal.txt SHA256SUMS-windows-x86_64.txt SHA256SUMS-windows-arm64.txt; do
      [[ -s "$sum_file" ]] || die "missing $sum_file in staged release '$STAGE_RELEASE_TAG'"
      sha256sum -c "$sum_file" >/dev/null || die "checksum verification failed for $sum_file"
    done
    for raw_bin in \
      carbon-darwin-arm64 \
      carbon-darwin-x86_64 \
      carbon-darwin-universal \
      carbon-windows-x86_64-msvc.exe \
      carbon-windows-aarch64-msvc.exe
    do
      [[ -s "$raw_bin" ]] || die "missing staged binary '$raw_bin'"
      grep -a -q -F "$SOURCE_SHA" "$raw_bin" || die "staged binary '$raw_bin' does not embed expected commit SHA $SOURCE_SHA"
      grep -a -q -F "$TAG" "$raw_bin" || die "staged binary '$raw_bin' does not embed expected release tag $TAG"
    done
    shopt -s nullglob
    for asset in carbon-darwin-* carbon-windows-*; do
      cp -f "$asset" "$OUT_DIR/$asset"
    done
  )
  rm -rf "$STAGE_DL_DIR"
fi

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
    if [[ "$ALLOW_PARTIAL_MATRIX" -eq 0 ]]; then
      compgen -G "$OUT_DIR/carbon-linux-x86_64-musl*" >/dev/null || die "missing Linux x86_64 CLI asset in $OUT_DIR"
      compgen -G "$OUT_DIR/carbon-linux-aarch64-musl*" >/dev/null || die "missing Linux aarch64 CLI asset in $OUT_DIR"
      compgen -G "$OUT_DIR/carbon-darwin-*" >/dev/null || die "missing macOS Darwin CLI asset in $OUT_DIR (pass --extra-assets-dir or --allow-partial-matrix)"
      compgen -G "$OUT_DIR/carbon-windows-x86_64-msvc*" >/dev/null || die "missing Windows x86_64 CLI asset in $OUT_DIR (pass --extra-assets-dir or --allow-partial-matrix)"
      compgen -G "$OUT_DIR/carbon-windows-aarch64-msvc*" >/dev/null || die "missing Windows ARM64 CLI asset in $OUT_DIR (pass --extra-assets-dir or --allow-partial-matrix)"
    fi
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
      if [[ -n "$STAGE_RELEASE_TAG" ]]; then
        say "Cleaning up temporary draft staging release '$STAGE_RELEASE_TAG'..."
        "$GH_ROLE_CMD" writer release delete "$STAGE_RELEASE_TAG" --repo neverhuman/JopeDime --yes --cleanup-tag || true
      fi
    )
  fi
fi

say "Completed Carbon release pipeline -> $OUT_DIR"
