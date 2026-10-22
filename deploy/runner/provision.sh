#!/usr/bin/env bash
# deploy/runner/provision.sh — one-time host provisioning of a self-hosted runner (run as root by
# the owner; docs/runbooks/runner.md). It creates the named volumes, provisions the neural SDK
# volume from deploy/receipts/libtorch.json (sha256-verified, exactly as docs/runbooks/neural-sdk.md),
# builds the runner image from deploy/docker/runner.Dockerfile, installs the systemd units and
# creates the PAT file location. It registers NOTHING with GitHub: registration happens per job,
# from the JIT config that run.sh requests with the PAT the owner places in the 0600 file.
#
# Usage: deploy/runner/provision.sh [--skip-sdk] [--skip-image]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
# shellcheck source=deploy/runner/common.sh
# shellcheck disable=SC1091  # sourced relative to this file; run shellcheck -x to follow
. "$here/common.sh"

skip_sdk=0; skip_image=0
for arg in "$@"; do
  case $arg in
    --skip-sdk) skip_sdk=1 ;;
    --skip-image) skip_image=1 ;;
    *) echo "unknown argument $arg" >&2; exit 2 ;;
  esac
done
[[ $(id -u) -eq 0 ]] || { echo "provision.sh must run as root (volumes, /etc/jope-runner, systemd)" >&2; exit 2; }
for tool in docker jq curl unzip sha256sum systemctl; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 2; }
done
lanes_text=$(jope_runner_lanes "$JOPE_RUNNER_HOST") || exit 2
read -ra lanes <<< "$lanes_text"
JOPE_VOLUMES=(jope-sdk)
for lane in "${lanes[@]}"; do
  jope_runner_role "$JOPE_RUNNER_HOST" "$lane"
  for kind in work cargo target uv; do JOPE_VOLUMES+=("jope-$JOPE_RUNNER_DOMAIN-$lane-$kind"); done
done

# 1. volumes
for v in "${JOPE_VOLUMES[@]}"; do
  docker volume inspect "$v" >/dev/null 2>&1 || docker volume create "$v" >/dev/null
  echo "volume $v: $(docker volume inspect --format '{{.Mountpoint}}' "$v")"
done
# 2. image (local build; never pushed) — built before the SDK step, which runs inside it
if [[ $skip_image -eq 0 ]]; then
  sha=$(git -C "$repo" rev-parse --short=12 HEAD)
  DOCKER_BUILDKIT=1 docker build -q -f "$repo/deploy/docker/runner.Dockerfile" -t "jope-runner:$sha" -t "$JOPE_RUNNER_IMAGE" "$repo"
  echo "images: jope-runner:$sha ($(docker image inspect --format '{{.Id}}' "jope-runner:$sha"))"
fi
docker image inspect "$JOPE_RUNNER_IMAGE" >/dev/null 2>&1 || { echo "image $JOPE_RUNNER_IMAGE is absent (drop --skip-image)" >&2; exit 1; }

# Initialize volume root ownership only; retain all previous volumes and their evidence.
for v in "${JOPE_VOLUMES[@]}"; do
  docker run --rm --user 0 --network none --cap-drop ALL --cap-add CHOWN \
    -v "$v:/v" --entrypoint chown "$JOPE_RUNNER_IMAGE" "$JOPE_RUNNER_UID:$JOPE_RUNNER_UID" /v
done

# 3. neural SDK volume (read-only for the runner; provisioned once from the receipt)
if [[ $skip_sdk -eq 0 ]]; then
  receipt=$repo/deploy/receipts/libtorch.json
  expected=$(jq -r .libtorch.version "$receipt")
  sdk_mount=$(docker volume inspect --format '{{.Mountpoint}}' jope-sdk)
  if [[ -f $sdk_mount/libtorch/build-version && $(< "$sdk_mount/libtorch/build-version") == "$expected" ]]; then
    echo "sdk volume already holds LibTorch $expected"
  else
    echo "provisioning the SDK volume from $receipt (LibTorch $expected + $(jq '.companions | length' "$receipt") companion wheels)"
    docker run --rm --user "$JOPE_RUNNER_UID" --network host --cap-drop ALL --security-opt no-new-privileges \
      --entrypoint bash -v jope-sdk:/opt/jope-sdk -v "$receipt:/receipt.json:ro" -w /opt/jope-sdk \
      "$JOPE_RUNNER_IMAGE" -c '
        set -euo pipefail
        R=/receipt.json; SDK=/opt/jope-sdk
        mkdir -p "$SDK/dl" && cd "$SDK/dl"
        curl -fsSL --retry 3 -C - -o libtorch.zip "$(jq -r .libtorch.url "$R")"
        echo "$(jq -r .libtorch.sha256 "$R")  libtorch.zip" | sha256sum -c -
        unzip -q -o libtorch.zip -d "$SDK"
        jq -r ".companions[] | \"\(.url) \(.sha256) \(.filename)\"" "$R" | while read -r url sha file; do
          curl -fsSL --retry 5 --retry-all-errors -C - -o "$file" "$url"
          echo "$sha  $file" | sha256sum -c -
          unzip -q -o "$file" "nvidia/*/lib/*" "nvidia/*/include/*" -d "$SDK"
        done
        rm -rf "$SDK/dl"
        cat "$SDK/libtorch/build-version"' 2>&1 | tail -3
  fi
fi

# 4. host files: PAT location (empty, 0600), state dir, scripts, units
install -d -m 0700 -o root -g root /etc/jope-runner
[[ -f $JOPE_RUNNER_PAT_FILE ]] || { install -m 0600 -o root -g root /dev/null "$JOPE_RUNNER_PAT_FILE"; echo "created empty $JOPE_RUNNER_PAT_FILE — paste the fine-grained PAT (Administration: write on $JOPE_RUNNER_REPO only) into it"; }
install -d -m 0750 -o root -g root "$JOPE_RUNNER_STATE"
install -d -m 0755 /opt/jope-runner
# tmpfiles creates the fixed lock after reboot without replacing an existing
# locked inode. Never install /dev/null over a live advisory-lock file.
install -d -m 0755 /etc/tmpfiles.d
install -m 0644 "$here"/jope-exclusive-cpu.conf /etc/tmpfiles.d/
systemd-tmpfiles --create /etc/tmpfiles.d/jope-exclusive-cpu.conf
install -m 0755 "$here"/common.sh "$here"/run.sh "$here"/disk-guard.sh "$here"/prune.sh "$here"/trim-deps.sh "$here"/job-started.sh "$here"/job-policy.py /opt/jope-runner/
install -m 0644 "$here"/jope-runner@.service "$here"/jope-runner@.timer "$here"/jope-runner-prune.service "$here"/jope-runner-prune.timer "$here"/jope-docker-proxy.service /etc/systemd/system/
# The drop-in was committed but never installed here, so a freshly provisioned host ran with the
# upstream RuntimeMaxSec=1800 -- the mid-job SIGTERM the drop-in exists to prevent. Installed as a
# directory so any future .conf in it is picked up without editing this line again.
install -d -m 0755 "/etc/systemd/system/jope-runner@.service.d"
install -m 0644 "$here"/jope-runner@.service.d/*.conf "/etc/systemd/system/jope-runner@.service.d/"
systemctl daemon-reload

# 5. provenance: record what is actually deployed, so a lane can prove it rather than assume it.
# run.sh injects these into every job; .github/workflows/ci.yml compares the script digest with
# the checkout's, which fails the lane when the host is running something other than the commit
# under test. Inspection cannot establish that; only a comparison the lane performs can.
# Sourced from the freshly installed copy so the digest describes what is now deployed, not what
# this script was invoked from.
# shellcheck disable=SC1091
. /opt/jope-runner/common.sh
scripts_sha=$(jope_runner_deployed_digest)
image_id=$(docker image inspect --format '{{.Id}}' "$JOPE_RUNNER_IMAGE" 2>/dev/null || echo unknown)
jq -n --arg image "$JOPE_RUNNER_IMAGE" --arg image_id "$image_id" --arg scripts_sha "$scripts_sha" \
      --arg provisioned_from "$here" --arg host "$JOPE_RUNNER_HOST" \
  '{host:$host, image:$image, image_id:$image_id, scripts_sha256:$scripts_sha, provisioned_from:$provisioned_from}' \
  > "$JOPE_RUNNER_STATE/image.json"
chmod 0640 "$JOPE_RUNNER_STATE/image.json"
echo "recorded $JOPE_RUNNER_STATE/image.json: image_id=$image_id scripts_sha256=$scripts_sha"

echo "approved lanes on $JOPE_RUNNER_HOST: ${lanes[*]}; installed units: jope-runner@<lane>.timer, jope-runner-prune.timer, jope-docker-proxy.service (none enabled — see docs/runbooks/runner.md)"
echo "full-lane labels: jope-full-cpu,jope-ci (owner permanent; lockstep provision all hosts after merge)"
