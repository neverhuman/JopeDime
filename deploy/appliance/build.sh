#!/usr/bin/env bash
# deploy/appliance/build.sh — build, pack and check one `jopedime` package.
#
#   build.sh --profile NAME --out DIR [--cargo-profile dist|dist-ci] [--models DIR] [--source-sha SHA] [--rehearsal]
#
# A development-scope profile (fixture) is packed and signed with the development key into
# DIR/jopedime. Every other scope stops at DIR/stage: the release host signs
# DIR/stage/signing-message.bin with its root-held cosign key and runs `jope-dist seal`
# (deploy/appliance/README.md). Thin by design: every deterministic step is `jope-dist`.
#
# Outputs under DIR: code (the executable Cargo built), notices.json, the package or stage,
# pack-receipt.json, verify.json, size.json, sbom-models.json, build.json.
set -euo pipefail
cd "$(dirname "$0")/../.."

profile='' out='' cargo_profile=dist models='' rehearsal=false source_sha=${JOPE_SOURCE_SHA:-}
while [ $# -gt 0 ]; do
  case $1 in
    --profile) profile=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --cargo-profile) cargo_profile=$2; shift 2 ;;
    --models) models=$2; shift 2 ;;
    --source-sha) source_sha=$2; shift 2 ;;
    --rehearsal) rehearsal=true; shift ;;
    *) echo "usage: $0 --profile NAME --out DIR [--cargo-profile P] [--models DIR] [--source-sha SHA] [--rehearsal]" >&2; exit 2 ;;
  esac
done
[ -n "$profile" ] && [ -n "$out" ] || { echo "$0: --profile and --out are required" >&2; exit 2; }
profile_file=deploy/appliance/profiles/$profile.toml
[ -f "$profile_file" ] || { echo "$0: no profile $profile_file" >&2; exit 2; }
case $out in /tmp | /tmp/*) echo "$0: never build under /tmp (AGENTS.md)" >&2; exit 2 ;; esac
[ ! -e "$out" ] || { echo "$0: $out exists; give a new directory" >&2; exit 2; }
[ -n "$source_sha" ] || source_sha=$(git rev-parse HEAD)
[[ $source_sha =~ ^[0-9a-f]{40}$ ]] || { echo "$0: source sha must be 40 lower-case hex: $source_sha" >&2; exit 2; }
scope=$(sed -n 's/^scope = "\(.*\)"$/\1/p' "$profile_file")
if [ "$rehearsal" = true ] && { [ "$profile" != full ] || [ "$scope" != internal-restricted ]; }; then
  echo "$0: --rehearsal requires the internal-restricted full profile" >&2; exit 2
fi
features=embedded
[ "$scope" = development ] && features=embedded,dev-keys
# Both R882 runtime editions need the ordinary defaults despite --no-default-features below.
case $profile in full | fused-cpu) features=$features,engine,mcp,client ;; esac
case $profile in dope-gpu | fused-cuda) features=$features,dope-cuda ;; esac
mkdir -p "$out"

dist() { cargo run -q --locked -p jope-dist -- "$@"; }

# Offline notices metadata enumerates locked workspace packages beyond the built target.
# Fetch their sources explicitly while network access is allowed; keep audit/notices offline.
echo "== fetch locked workspace metadata dependencies"
cargo fetch --locked

echo "== lock and dependency fence"
dist lock --check
dist audit
if [ "$profile" = dope-gpu ] || [ "$profile" = fused-cuda ]; then
  dist audit --policy deploy/appliance/dependencies-dope-gpu.toml
fi

echo "== build jopedime ($cargo_profile, features $features, source $source_sha)"
code=$(JOPE_BUILD_SOURCE_SHA=$source_sha cargo run -q --locked -p jope-ci -- cargo-artifact build \
  --package jope-appliance --bin jopedime --profile "$cargo_profile" --no-default-features --features "$features")
cp -- "$code" "$out/code"

if [ "$profile" = full ] || [ "$profile" = fused-cpu ]; then
  echo "== verify $profile v1 compiled features and source"
  "$out/code" --version --json > "$out/version.json"
  jq -e --arg source "$source_sha" '
    .source_sha40 == $source and
    .features.embedded == true and
    .features.engine == true and
    .features.mcp == true and
    .features.client == true and
    .features.dev_keys == false and
    .trusts_development_key == false
  ' "$out/version.json" > /dev/null || {
    echo "$0: $profile requires engine,mcp,client,embedded, release trust and the declared source" >&2
    exit 1
  }
fi

echo "== dynamic dependencies: the C runtime only, no CUDA or other shared libraries"
needed=$(readelf -d "$out/code" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | sort)
printf '%s\n' "$needed"
while IFS= read -r lib; do
  case $lib in
    libc.so.6 | libm.so.6 | libgcc_s.so.1 | ld-linux-x86-64.so.2 | libpthread.so.0 | libdl.so.2 | librt.so.1) ;;
    *) echo "$0: unexpected NEEDED entry $lib" >&2; exit 1 ;;
  esac
done <<< "$needed"

echo "== notices"
dist notices --out "$out/notices.json" --features "$features"

if [ -z "$models" ]; then
  models=$out/models
  if [ "$profile" = dope-gpu ]; then
    mkdir -p "$models/bundles/dope-mlc-v3"
    cp deploy/appliance/assets/dope-mlc-v3/model.safetensors deploy/appliance/assets/dope-mlc-v3/LICENSE "$models/bundles/dope-mlc-v3/"
  elif [ "$scope" = development ]; then
    dist fixture --models "$models"
  else
    mkdir -p "$models"
  fi
fi

common=(--profile "$profile_file" --code "$out/code" --models "$models" --notices "$out/notices.json"
  --source-sha "$source_sha" --receipt "$out/pack-receipt.json")
if [ "$rehearsal" = true ]; then
  echo "== full-asset rehearsal only (not signable or release-qualified)"
  dist pack "${common[@]}" --rehearsal "$out/rehearsal"
  package=''
elif [ "$scope" = development ]; then
  echo "== pack and sign (development key)"
  dist pack "${common[@]}" --out "$out/jopedime"
  keys=(--development-key --no-release-key)
  dist verify --file "$out/jopedime" "${keys[@]}" > "$out/verify.json"
  dist size-report --file "$out/jopedime" "${keys[@]}" --budget deploy/appliance/size-budgets.toml > "$out/size.json"
  dist sbom-models --file "$out/jopedime" "${keys[@]}" --out "$out/sbom-models.json"
  package=$out/jopedime
else
  echo "== stage for release signing"
  dist pack "${common[@]}" --stage "$out/stage"
  package=''
  echo "sign $out/stage/signing-message.bin with the release key, then:"
  echo "  jope-dist seal --code $out/code --stage $out/stage --signature SIG --out $out/jopedime"
fi

jq -n --arg profile "$profile" --arg scope "$scope" --arg cargo_profile "$cargo_profile" \
  --arg features "$features" --arg source_sha "$source_sha" --arg package "$package" --argjson rehearsal "$rehearsal" \
  --arg code_sha256 "$(sha256sum < "$out/code" | cut -d' ' -f1)" \
  '{schema: "jope.appliance-build/v1", profile: $profile, scope: $scope, cargo_profile: $cargo_profile,
    features: $features, source_sha40: $source_sha, base_executable_sha256: $code_sha256,
    rehearsal: $rehearsal, package: (if $package == "" then null else $package end)}' > "$out/build.json"
cat "$out/build.json"
