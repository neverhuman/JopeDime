#!/usr/bin/env bash
# Native macOS REL-04 clean-host driver. Runtime is network-denied by sandbox-exec.
# Pass an already packaged jopedime, or --build-fixture to build, target-check, pack and verify
# the owner-supported native arm64 fixture package in the lane's existing Cargo target.
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../.." && pwd)
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || { echo "macOS clean-host requires Darwin arm64" >&2; exit 2; }
for tool in sandbox-exec jq shasum otool nc lsof curl; do command -v "$tool" >/dev/null || { echo "required tool missing: $tool" >&2; exit 2; }; done
source_sha=$(git -C "$repo" rev-parse HEAD)
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
base="$repo/.agent/cleanhost/$run_id"
work="$base/work"
home="$base/home"
tmp="$base/tmp"
tool_path="$base/no-toolchain"
profile="$base/no-network.sb"
port=$(( 49000 + ($$ % 1000) ))
network_probe=not-run
started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p "$home" "$tmp" "$tool_path"
chmod 700 "$home" "$tmp"

binary_arg=${1:?usage: test-clean-host-macos.sh JOPEDIME_BINARY|--build-fixture}
binary=$binary_arg
[[ $binary_arg != --build-fixture ]] || binary="$repo/target/rel04-macos/$run_id/jopedime"
cleanup() {
  if [[ -n ${listener_pid:-} ]]; then
    kill "$listener_pid" >/dev/null 2>&1 || true
    wait "$listener_pid" >/dev/null 2>&1 || true
    listener_pid=
  fi
}
finish() {
  status=$?
  trap - EXIT
  cleanup
  package_sha256=
  if [[ -f $binary ]]; then package_sha256=$(shasum -a 256 "$binary" | awk '{print $1}'); fi
  jq -n --arg source_sha "$source_sha" --arg run_id "$run_id" \
    --arg started_at "$started_at" --arg package "$binary" \
    --arg package_sha256 "$package_sha256" --arg work "$work" \
    --arg home "$home" --arg tmp "$tmp" --arg network_probe "$network_probe" \
    --argjson exit_status "$status" \
    '{schema:"jope.rel04-macos-clean-host-driver/v1",source_sha:$source_sha,run_id:$run_id,
      started_at:$started_at,finished_at:(now|todate),package:$package,package_sha256:$package_sha256,
      work:$work,home:$home,tmp:$tmp,network_probe:$network_probe,
      status:(if $exit_status == 0 then "passed" else "failed" end),exit_status:$exit_status,
      harness_receipt:($work + "/clean-host.json")}' \
    > "$base/driver.json"
  cat "$base/driver.json"
  exit "$status"
}
trap finish EXIT

if [[ $binary_arg == --build-fixture ]]; then
  build_root="$repo/target/rel04-macos/$run_id"
  mkdir -p "$build_root/models"
  cargo build --locked -p jope-appliance --bin jopedime --no-default-features --features embedded,dev-keys
  cargo_target=$(cargo run -q --locked -p jope-ci -- cargo-artifact target-dir)
  code="$cargo_target/debug/jopedime"
  [[ -x $code ]] || { echo "native Cargo artifact is not executable: $code" >&2; exit 1; }
  cargo run -q --locked -p jope-dist -- fixture --models "$build_root/models"
  cargo run -q --locked -p jope-dist -- notices --out "$build_root/notices.json" \
    --target aarch64-apple-darwin --features embedded,dev-keys
  cargo run -q --locked -p jope-dist -- pack \
    --profile "$repo/deploy/appliance/profiles/fixture.toml" --code "$code" \
    --models "$build_root/models" --notices "$build_root/notices.json" \
    --source-sha "$source_sha" --target aarch64-apple-darwin \
    --receipt "$build_root/pack-receipt.json" --out "$build_root/jopedime"
  cargo run -q --locked -p jope-dist -- verify --file "$build_root/jopedime" \
    --development-key --no-release-key > "$build_root/verify.json"
else
  [[ -x $binary ]] || { echo "binary is not executable: $binary" >&2; exit 2; }
  binary=$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")
fi

cat > "$profile" <<'PROFILE'
(version 1)
(allow default)
(deny network*)
PROFILE

# Require the profile to deny a real local listener, so a closed port cannot look like proof.
if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "loopback probe port is already in use: $port" >&2; exit 1
fi
nc -l 127.0.0.1 "$port" >/dev/null 2>&1 &
listener_pid=$!
ready=0
for _ in {1..50}; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | grep -qx "$listener_pid"; then ready=1; break; fi
  sleep 0.1
done
[[ $ready == 1 ]] || { echo "could not start loopback denial probe listener" >&2; exit 1; }
probe_output=$(sandbox-exec -f "$profile" /usr/bin/curl -v --connect-timeout 1 --max-time 2 -sS -o /dev/null "http://127.0.0.1:$port/" 2>&1) && probe_status=0 || probe_status=$?
[[ $probe_status -ne 0 && $probe_output == *"Operation not permitted"* ]] || {
  echo "sandbox profile did not deny loopback: status=$probe_status $probe_output" >&2; exit 1;
}
cleanup
network_probe=passed

printf 'source_sha=%s\nrun_id=%s\n' "$source_sha" "$run_id"
printf 'network_probe=passed (loopback denied by sandbox-exec)\n'
JOPE_CLEAN_HOST_SANDBOX_PROFILE="$profile" \
JOPE_CLEAN_HOST_HOME="$home" \
JOPE_CLEAN_HOST_TMPDIR="$tmp" \
JOPE_CLEAN_HOST_PATH="$tool_path" \
  "$script_dir/test-clean-host.sh" "$binary" "$work" --require-isolation
