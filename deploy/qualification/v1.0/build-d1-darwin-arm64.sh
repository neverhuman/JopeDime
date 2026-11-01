#!/usr/bin/env bash
# Build D1's native Darwin arm64 base executable from an exact source revision.
set -euo pipefail

usage() {
  echo "usage: $0 <40-hex-source-sha> <new-absolute-evidence-directory>" >&2
  exit 2
}
[[ $# -eq 2 ]] || usage
source_sha=$1
evidence=$2
[[ $source_sha =~ ^[0-9a-f]{40}$ ]] || usage
[[ $evidence = /* ]] || usage
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || {
  echo "D1 Darwin build requires a native Darwin arm64 host" >&2
  exit 2
}
for tool in cargo nice git lipo jq shasum; do
  command -v "$tool" >/dev/null || { echo "D1 Darwin build requires $tool" >&2; exit 2; }
done

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
case $evidence in /tmp | /tmp/*) echo "D1 evidence cannot be under /tmp" >&2; exit 2 ;; esac
evidence_parent=$(dirname -- "$evidence")
[[ -d $evidence_parent ]] || { echo "D1 evidence parent must already exist" >&2; exit 2; }
evidence="$(cd -- "$evidence_parent" && pwd -P)/$(basename -- "$evidence")"
[[ ! -e $evidence && ! -L $evidence ]] || { echo "D1 evidence path already exists" >&2; exit 2; }
case $evidence/ in "$repo"/*) echo "D1 evidence must be outside the source checkout" >&2; exit 2 ;; esac

head=$(git -C "$repo" rev-parse HEAD)
[[ $head == "$source_sha" ]] || { echo "D1 source SHA differs from checkout HEAD" >&2; exit 2; }
[[ -z $(git -C "$repo" status --porcelain --untracked-files=all) ]] || {
  echo "D1 build requires a clean source checkout" >&2; exit 2;
}

mkdir -p "$evidence"
artifact="$evidence/jopedime-darwin-arm64-code"

unset CARGO_TARGET_DIR
export CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
features=embedded,engine,mcp,client
echo "== native Darwin arm64 full-profile D1 build at $source_sha"
code=$(JOPE_BUILD_SOURCE_SHA="$source_sha" nice -n 19 cargo run -q --locked -p jope-ci -- cargo-artifact build \
  --package jope-appliance --bin jopedime --profile dist \
  --no-default-features --features "$features")
cp -- "$code" "$artifact"
chmod 0755 "$artifact"

[[ $(lipo -archs "$artifact") == arm64 ]] || {
  echo "D1 artifact is not native arm64: $(lipo -archs "$artifact")" >&2; exit 1;
}
"$artifact" --version --json > "$evidence/version.json"
jq -e --arg source "$source_sha" '
  .source_sha40 == $source and .target == "aarch64-apple-darwin" and
  .features.embedded == true and .features.engine == true and
  .features.mcp == true and .features.client == true and
  .features.dev_keys == false and .features.cuda == false and
  .features.dope_cuda == false and .trusts_development_key == false
' "$evidence/version.json" >/dev/null || {
  echo "D1 artifact source, target, or release feature identity is invalid" >&2; exit 1;
}

echo "== Darwin fused-cpu dynamic-library allowlist"
bash "$repo/deploy/ci/link-audit.sh" "$artifact" fused-cpu \
  > "$evidence/link-audit.txt" 2>&1

jq -n --arg source "$source_sha" --arg artifact "$(shasum -a 256 "$artifact" | awk '{print $1}')" \
  --arg path "$artifact" \
  '{schema:"jope.rel07.d1-darwin-build/v1",source_sha:$source,
    target_triple:"aarch64-apple-darwin",profile:"full",
    features:["embedded","engine","mcp","client"],
    base_executable_path:$path,base_executable_sha256:$artifact,
    checks:{native_arm64:true,embedded_source_pin:true,full_features:true,
    macos_os_libraries_only:true}}' > "$evidence/darwin-build.json"
echo "D1 Darwin base executable ready for RC2 packaging; sha256=$(shasum -a 256 "$artifact" | awk '{print $1}')"
