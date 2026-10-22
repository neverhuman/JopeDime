#!/usr/bin/env bash
# deploy/runner/run.sh <ci|ci2|full> — start ONE ephemeral runner job container (called by
# jope-runner@<lane>.service, as root, on the timer's cadence).
#
#   1. at most daily, trim older duplicate builds from this lane's target volume (common.sh
#      jope_runner_trim_lane), then the disk guard (refuse under 50 GB free);
#   2. read the fine-grained PAT from the root-0600 file (never exported, never mounted) and ask
#      GitHub for a single-use JIT config:
#        POST /repos/<owner>/<repo>/actions/runners/generate-jitconfig
#        {"name": "<host>-<lane>-<stamp>", "runner_group_id": 1, "labels": [...],
#         "work_folder": "/home/runner/work/<lane>"}
#      The response's encoded_jit_config is passed to the container on stdin only;
#   3. docker run the image with the plan's hardening flags and the named volumes.
# The container exits when its one job ends; the runner deregisters itself (JIT = ephemeral).
# If GitHub has no queued job the listener idles; workflows own accepted-job timeouts.
set -euo pipefail
# shellcheck source=deploy/runner/common.sh
# shellcheck disable=SC1091  # sourced relative to this file; run shellcheck -x to follow
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

lane=${1:?usage: run.sh <ci|ci2|full>}
jope_runner_role "$JOPE_RUNNER_HOST" "$lane"
labels=$(jope_runner_labels "$lane")
# Refuse unrecorded systemd overrides, including the historical per-instance budget.
dropins=$(systemctl show "jope-runner@$lane.service" -p DropInPaths --value)
[[ $dropins == /etc/systemd/system/jope-runner@.service.d/runtime-max.conf ]] || {
  echo "unrecorded runner drop-ins: $dropins" >&2; exit 2;
}
[[ $(systemctl show "jope-runner@$lane.service" -p FragmentPath --value) == /etc/systemd/system/jope-runner@.service ]] || exit 2
# A root-owned per-lane lock prevents duplicate service/manual listeners sharing storage.
exec 9>"$JOPE_RUNNER_STATE/$lane.lock"
flock -n 9 || { echo "lane already owned: $lane" >&2; exit 2; }
jope_runner_check_existing "$lane"
jope_runner_exclusive_cpu "$lane"
volume_prefix="jope-$JOPE_RUNNER_DOMAIN-$lane"
policy="$JOPE_RUNNER_STATE/$lane-policy.json"
jq -n --arg domain "$JOPE_RUNNER_DOMAIN" --arg lane "$lane" --arg host "$JOPE_RUNNER_HOST" \
  --arg memory "$JOPE_RUNNER_MEMORY" --arg prefix "$volume_prefix" \
  '{domain:$domain,lane:$lane,host:$host,memory:$memory,cpus:16,volume_prefix:$prefix}' > "$policy"
chmod 0444 "$policy"
[[ $(id -u) -eq 0 ]] || { echo "run.sh must run as root (reads $JOPE_RUNNER_PAT_FILE)" >&2; exit 2; }
# The lane lock is held and no container of this lane runs: the one safe point to trim its cache.
jope_runner_trim_lane "$lane" || true
"$(dirname "${BASH_SOURCE[0]}")/disk-guard.sh"

[[ -f $JOPE_RUNNER_PAT_FILE ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE is missing (docs/runbooks/runner.md)" >&2; exit 2; }
perms=$(stat -c '%a %U' "$JOPE_RUNNER_PAT_FILE")
[[ $perms == "600 root" ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE must be root:root 0600, is $perms" >&2; exit 2; }
[[ -s $JOPE_RUNNER_PAT_FILE ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE is empty — the owner has not registered yet" >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
name="$JOPE_RUNNER_HOST-$lane-$stamp"
# work_folder is absolute and per-lane. Absolute because entrypoint.sh copies the runner to
# /home/runner/runner, so a relative "work" resolves inside the small /home/runner tmpfs instead
# of the jope-work volume. Per-lane because the four lanes run concurrently against one volume:
# a shared root means actions/checkout in one lane deletes the target/ another lane is building.
# $lane must be passed to jq as an argument -- jq does not expand shell variables, so writing
# "$lane" inside the program yields a literal directory named '$lane' shared by every lane.
body=$(jq -nc --arg name "$name" --arg labels "$labels" --arg lane "$lane" \
  '{name:$name, runner_group_id:1, labels:(["self-hosted","linux","x64"] + ($labels|split(","))), work_folder:"/home/runner/work/\($lane)"}')
# The token goes to curl through a config file on a private fd, never on the command line.
response=$(curl -fsS --retry 3 -X POST \
  -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
  -K <(printf 'header = "Authorization: Bearer %s"\n' "$(< "$JOPE_RUNNER_PAT_FILE")") \
  "https://api.github.com/repos/$JOPE_RUNNER_REPO/actions/runners/generate-jitconfig" \
  --data "$body") || { echo "run.sh: generate-jitconfig failed for $name" >&2; exit 1; }
jit=$(jq -r '.encoded_jit_config // empty' <<<"$response")
[[ -n $jit ]] || { echo "run.sh: no encoded_jit_config in the response: $(jq -c 'del(.encoded_jit_config)' <<<"$response")" >&2; exit 1; }
echo "run.sh: runner $name id=$(jq -r '.runner.id' <<<"$response") labels=$labels"

# Provenance injected into the job, so a lane can prove what it is running instead of assuming.
# The script digest is recomputed from the deployed files on every launch -- reading it back from
# provision.sh's receipt would go stale precisely when someone edits /opt/jope-runner in place,
# which is the case this exists to catch. The receipt is used only for the image id provisioning
# recorded, which the lane compares against the image actually being started: they disagree when
# the image was rebuilt or retagged without re-running provision.sh.
prov=$JOPE_RUNNER_STATE/image.json
scripts_sha=$(jope_runner_deployed_digest) || {
  echo "run.sh: a file in JOPE_RUNNER_DEPLOYED is missing ($scripts_sha) -- re-run provision.sh" >&2
  exit 2
}
image_id=$(docker image inspect --format '{{.Id}}' "$JOPE_RUNNER_IMAGE" 2>/dev/null || echo unknown)
recorded_image_id=unknown
[[ -r $prov ]] && recorded_image_id=$(jq -r '.image_id // "unknown"' "$prov")

docker_args=(
  run --rm --init --name "$name" --label jope.runner=1 --label "jope.lane=$lane"
  --user "$JOPE_RUNNER_UID"
  --cap-drop ALL --security-opt no-new-privileges --pids-limit "$JOPE_RUNNER_PIDS"
  --memory "$JOPE_RUNNER_MEMORY" --memory-swap "$JOPE_RUNNER_MEMORY" --cpus "$JOPE_RUNNER_CPUS"
  --read-only --tmpfs "/tmp:size=$JOPE_RUNNER_TMPFS_SIZE,exec" --tmpfs "/home/runner:size=4g,exec,uid=$JOPE_RUNNER_UID,gid=$JOPE_RUNNER_UID"
  -v "$volume_prefix-work:/home/runner/work"
  -v "$volume_prefix-cargo:/var/cache/jope/cargo" -v "$volume_prefix-target:/var/cache/jope/target" -v "$volume_prefix-uv:/var/cache/jope/uv"
  -v jope-sdk:/opt/jope-sdk:ro
  -v /opt/jope-runner/job-started.sh:/opt/jope-runner/job-started.sh:ro
  -v /opt/jope-runner/job-policy.py:/opt/jope-runner/job-policy.py:ro
  -v "$policy:/opt/jope-runner/policy.json:ro"
  -e ACTIONS_RUNNER_HOOK_JOB_STARTED=/opt/jope-runner/job-started.sh
  -e "JOPE_RUNNER_LANE=$lane"
  -e "JOPE_RUNNER_HOST=$JOPE_RUNNER_HOST" -e "JOPE_RUNNER_SCRIPTS_SHA256=$scripts_sha"
  -e "JOPE_RUNNER_IMAGE_ID=$image_id" -e "JOPE_RUNNER_IMAGE_ID_RECORDED=$recorded_image_id"
  -i
)
docker_args+=("$JOPE_RUNNER_IMAGE")
# Record swap=0 in systemd once the scope exists (common.sh). Beside, not before, the docker
# CLI so its signal forwarding is unchanged; lock fds 8/9 are closed so it never holds a lane.
jope_runner_pin_swap "$name" 8<&- 9>&- &
printf '%s' "$jit" | exec docker "${docker_args[@]}"
