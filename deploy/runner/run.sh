#!/usr/bin/env bash
# deploy/runner/run.sh <gpu|kernel> — start ONE ephemeral runner job container (called by
# jope-runner@<lane>.service, as root, on the timer's cadence).
#
#   1. disk guard (refuse under 50 GB free);
#   2. read the fine-grained PAT from the root-0600 file (never exported, never mounted) and ask
#      GitHub for a single-use JIT config:
#        POST /repos/<owner>/<repo>/actions/runners/generate-jitconfig
#        {"name": "xbabe2-<lane>-<stamp>", "runner_group_id": 1, "labels": [...], "work_folder": "work"}
#      The response's encoded_jit_config is passed to the container on stdin only;
#   3. docker run the image with the plan's hardening flags and the named volumes.
# The container exits when its one job ends; the runner deregisters itself (JIT = ephemeral).
# If GitHub has no queued job the listener idles until the service's RuntimeMaxSec stops it.
set -euo pipefail
# shellcheck source=deploy/runner/common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

lane=${1:?usage: run.sh <gpu|kernel>}
labels=$(jope_runner_labels "$lane")
[[ $(id -u) -eq 0 ]] || { echo "run.sh must run as root (reads $JOPE_RUNNER_PAT_FILE)" >&2; exit 2; }
"$(dirname "${BASH_SOURCE[0]}")/disk-guard.sh"

[[ -f $JOPE_RUNNER_PAT_FILE ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE is missing (docs/runbooks/runner.md)" >&2; exit 2; }
perms=$(stat -c '%a %U' "$JOPE_RUNNER_PAT_FILE")
[[ $perms == "600 root" ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE must be root:root 0600, is $perms" >&2; exit 2; }
[[ -s $JOPE_RUNNER_PAT_FILE ]] || { echo "run.sh: $JOPE_RUNNER_PAT_FILE is empty — the owner has not registered yet" >&2; exit 2; }

stamp=$(date -u +%Y%m%dT%H%M%SZ)
name="xbabe2-$lane-$stamp"
body=$(jq -nc --arg name "$name" --arg labels "$labels" \
  '{name:$name, runner_group_id:1, labels:(["self-hosted","linux","x64"] + ($labels|split(","))), work_folder:"work"}')
# The token goes to curl through a config file on a private fd, never on the command line.
response=$(curl -fsS --retry 3 -X POST \
  -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
  -K <(printf 'header = "Authorization: Bearer %s"\n' "$(< "$JOPE_RUNNER_PAT_FILE")") \
  "https://api.github.com/repos/$JOPE_RUNNER_REPO/actions/runners/generate-jitconfig" \
  --data "$body") || { echo "run.sh: generate-jitconfig failed for $name" >&2; exit 1; }
jit=$(jq -r '.encoded_jit_config // empty' <<<"$response")
[[ -n $jit ]] || { echo "run.sh: no encoded_jit_config in the response: $(jq -c 'del(.encoded_jit_config)' <<<"$response")" >&2; exit 1; }
echo "run.sh: runner $name id=$(jq -r '.runner.id' <<<"$response") labels=$labels"

docker_args=(
  run --rm --init --name "$name" --label jope.runner=1 --label "jope.lane=$lane"
  --user "$JOPE_RUNNER_UID"
  --cap-drop ALL --security-opt no-new-privileges --pids-limit "$JOPE_RUNNER_PIDS"
  --memory "$JOPE_RUNNER_MEMORY" --cpus "$JOPE_RUNNER_CPUS"
  --read-only --tmpfs "/tmp:size=$JOPE_RUNNER_TMPFS_SIZE,exec" --tmpfs "/home/runner:size=4g,uid=$JOPE_RUNNER_UID,gid=$JOPE_RUNNER_UID"
  -v jope-work:/home/runner/work
  -v jope-cargo:/var/cache/jope/cargo -v jope-target:/var/cache/jope/target -v jope-uv:/var/cache/jope/uv
  -v jope-sdk:/opt/jope-sdk:ro -v jope-models:/var/lib/jope/models
  -e "JOPE_RUNNER_LANE=$lane" -e "JOPE_GPU_DEVICE=$JOPE_RUNNER_GPU"
  -i
)
case $lane in
  gpu)    docker_args+=(--gpus "$JOPE_RUNNER_GPU") ;;
  kernel) # no GPU; docker only through the socket proxy (containers + images, nothing else)
          [[ -S $JOPE_RUNNER_DOCKER_PROXY_SOCK ]] || { echo "run.sh: docker socket proxy $JOPE_RUNNER_DOCKER_PROXY_SOCK is not up (jope-docker-proxy.service)" >&2; exit 1; }
          docker_args+=(-v "$JOPE_RUNNER_DOCKER_PROXY_SOCK:/var/run/docker.sock" -e DOCKER_HOST=unix:///var/run/docker.sock) ;;
esac
docker_args+=("$JOPE_RUNNER_IMAGE")
printf '%s' "$jit" | exec docker "${docker_args[@]}"
