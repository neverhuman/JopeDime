#!/usr/bin/env bash
# deploy/runner/common.sh — shared settings of the self-hosted runner scripts, on whichever
# xbabe host the lanes are installed on (JOPE_RUNNER_HOST, default `hostname -s`). Sourced,
# never executed. Nothing here contains a credential; the PAT lives only in
# /etc/jope-runner/github-pat (root:root 0600, read by run.sh at job start, never mounted).
# shellcheck shell=bash
# shellcheck disable=SC2034 # Shared settings are consumed by the sourcing scripts.

JOPE_RUNNER_IMAGE=${JOPE_RUNNER_IMAGE:-jope-runner:current}          # deploy/docker/runner.Dockerfile, tagged by provision.sh
JOPE_RUNNER_REPO=${JOPE_RUNNER_REPO:-veox-ai/JopeDime}
JOPE_RUNNER_PAT_FILE=${JOPE_RUNNER_PAT_FILE:-/etc/jope-runner/github-pat}
JOPE_RUNNER_STATE=${JOPE_RUNNER_STATE:-/var/lib/jope-runner}          # receipts, last-prune stamp (root-owned)
JOPE_RUNNER_MIN_FREE_GB=50                # disk guard: refuse to start under this
JOPE_RUNNER_GPU=${JOPE_RUNNER_GPU:-device=0}                          # --gpus value; device 0 of whichever host runs the gpu lane
# shellcheck disable=SC2034  # read by run.sh, prune.sh and provision.sh after sourcing
JOPE_RUNNER_UID=1001                                                  # the image's non-root user
JOPE_RUNNER_HOST=$(hostname -s)       # runner names and labels follow the host
# Approved CPU deployment, deliberately not overridable by service Environment entries.
JOPE_RUNNER_CPUS=16
JOPE_RUNNER_PIDS=4096
JOPE_RUNNER_TMPFS_SIZE=4g
JOPE_RUNNER_DOCKER_PROXY_SOCK=${JOPE_RUNNER_DOCKER_PROXY_SOCK:-/run/jope-docker-proxy/docker.sock}

# Every file provision.sh installs onto the host, in a fixed order. This list is the single
# source of truth for the provenance digest: run.sh hashes these, and .github/workflows/ci.yml
# parses this very array out of the checkout and hashes the matching deploy/runner/<name> files,
# so the workflow cannot drift from what is actually deployed. entrypoint.sh is absent on purpose
# -- it lives inside the image, and the lane binds it separately from inside the container.
# shellcheck disable=SC2034  # read by run.sh and provision.sh after sourcing
JOPE_RUNNER_DEPLOYED=(
  /opt/jope-runner/common.sh
  /opt/jope-runner/run.sh
  /opt/jope-runner/disk-guard.sh
  /opt/jope-runner/prune.sh
  /opt/jope-runner/job-started.sh
  /opt/jope-runner/job-policy.py
  /etc/systemd/system/jope-runner@.service
  /etc/systemd/system/jope-runner@.service.d/runtime-max.conf
  /etc/systemd/system/jope-runner@.timer
  /etc/systemd/system/jope-runner-prune.service
  /etc/systemd/system/jope-runner-prune.timer
  /etc/systemd/system/jope-docker-proxy.service
  /etc/tmpfiles.d/jope-exclusive-cpu.conf
)

# The digest of what is deployed right now. Always recomputed, never cached: a receipt written at
# provision time is exactly wrong for the case this exists to catch, an in-place edit afterwards.
jope_runner_deployed_digest() {
  local f
  for f in "${JOPE_RUNNER_DEPLOYED[@]}"; do
    [[ -r $f ]] || { echo "missing-$f"; return 1; }
  done
  cat "${JOPE_RUNNER_DEPLOYED[@]}" | sha256sum | cut -d' ' -f1
}

# Fixed host allocation enforces exclusivity among this repository's runner jobs.
# Historical GPU/kernel/ci3/ci4 lanes remain unavailable in this CPU foundation.
jope_runner_role() { # <host> <lane>; callers use actual hostname, never a job variable
  case "$1/$2" in
    xbabe1/ci|xbabe1/ci2) JOPE_RUNNER_DOMAIN="pr"; JOPE_RUNNER_MEMORY=16g ;;
    xbabe2/full) JOPE_RUNNER_DOMAIN=main; JOPE_RUNNER_MEMORY=48g ;;
    *) echo "unapproved runner host/lane: $1/$2" >&2; return 2 ;;
  esac
}

jope_runner_labels() {
  jope_runner_role "$JOPE_RUNNER_HOST" "$1" || return
  case $1 in
    ci|ci2) echo "$JOPE_RUNNER_HOST-$1,jope-ci" ;;
    full) echo "$JOPE_RUNNER_HOST-full,jope-full-cpu" ;;
  esac
}

# A dead docker CLI can release flock while its container survives. Refuse a second
# writer and any legacy role still running on a host assigned to this CPU foundation.
jope_runner_check_existing() { # requested lane; docker failure is never an empty fleet
  local existing id running
  existing=$(docker ps --filter label=jope.runner=1 --format '{{.ID}} {{.Label "jope.lane"}}') || return
  while read -r id running; do
    [[ -n $id ]] || continue
    case "$JOPE_RUNNER_HOST/$running" in
      xbabe1/ci|xbabe1/ci2|xbabe2/full) ;;
      *) echo "unapproved existing JopeDime container: $id lane=$running" >&2; return 2 ;;
    esac
    [[ $running != "$1" ]] || { echo "existing container owns lane $1: $id" >&2; return 2; }
  done <<< "$existing"
}

# The campaign and full runner share this inode. The orphan cgroup check closes
# the interval between a killed controller releasing flock and systemd draining
# its descendants. A lock alone does not prove that the CPU host is idle.
jope_runner_check_asm_groups() {
  local root=${1:?cgroup root required} group field value found
  for group in "$root"/jope-asm-*.service; do
    [[ -e $group ]] || continue
    [[ -r $group/cgroup.events ]] || { echo "ASM cgroup state unavailable: $group" >&2; return 2; }
    found=0
    while read -r field value; do
      [[ $field == populated ]] || continue
      found=1
      [[ $value == 0 ]] || { echo "ASM work still owns CPU host: $group" >&2; return 2; }
    done < "$group/cgroup.events"
    [[ $found == 1 ]] || { echo "ASM populated counter missing: $group" >&2; return 2; }
  done
}

jope_runner_exclusive_cpu() {
  [[ $JOPE_RUNNER_HOST == xbabe2 && $1 == full ]] || return 0
  [[ -f /run/lock/jope-exclusive-cpu.lock ]] || {
    echo "shared CPU lock not provisioned" >&2; return 2;
  }
  exec 8<>/run/lock/jope-exclusive-cpu.lock
  flock -n 8 || { echo "ASM/exclusive CPU allocation active" >&2; return 2; }
  jope_runner_check_asm_groups /sys/fs/cgroup/system.slice
}
