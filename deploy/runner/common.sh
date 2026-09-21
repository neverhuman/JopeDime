#!/usr/bin/env bash
# deploy/runner/common.sh — shared settings of the self-hosted runner scripts, on whichever
# xbabe host the lanes are installed on (JOPE_RUNNER_HOST, default `hostname -s`). Sourced,
# never executed. Nothing here contains a credential; the PAT lives only in
# /etc/jope-runner/github-pat (root:root 0600, read by run.sh at job start, never mounted).
# shellcheck shell=bash

JOPE_RUNNER_IMAGE=${JOPE_RUNNER_IMAGE:-jope-runner:current}          # deploy/docker/runner.Dockerfile, tagged by provision.sh
JOPE_RUNNER_REPO=${JOPE_RUNNER_REPO:-veox-ai/JopeDime}
JOPE_RUNNER_PAT_FILE=${JOPE_RUNNER_PAT_FILE:-/etc/jope-runner/github-pat}
JOPE_RUNNER_STATE=${JOPE_RUNNER_STATE:-/var/lib/jope-runner}          # receipts, last-prune stamp (root-owned)
JOPE_RUNNER_MIN_FREE_GB=${JOPE_RUNNER_MIN_FREE_GB:-50}                # disk guard: refuse to start under this
JOPE_RUNNER_GPU=${JOPE_RUNNER_GPU:-device=0}                          # --gpus value; device 0 of whichever host runs the gpu lane
# shellcheck disable=SC2034  # read by run.sh, prune.sh and provision.sh after sourcing
JOPE_RUNNER_UID=1001                                                  # the image's non-root user
JOPE_RUNNER_HOST=${JOPE_RUNNER_HOST:-$(hostname -s)}       # runner names and labels follow the host
JOPE_RUNNER_MEMORY=${JOPE_RUNNER_MEMORY:-48g}              # fits xbabe1's 62 GB with headroom; raise per host
JOPE_RUNNER_CPUS=${JOPE_RUNNER_CPUS:-64}
JOPE_RUNNER_PIDS=${JOPE_RUNNER_PIDS:-4096}
JOPE_RUNNER_TMPFS_SIZE=${JOPE_RUNNER_TMPFS_SIZE:-32g}
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
  /etc/systemd/system/jope-runner@.service
  /etc/systemd/system/jope-runner@.service.d/runtime-max.conf
  /etc/systemd/system/jope-runner@.timer
  /etc/systemd/system/jope-runner-prune.service
  /etc/systemd/system/jope-runner-prune.timer
  /etc/systemd/system/jope-docker-proxy.service
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

# Named volumes (created by provision.sh; owned by uid 1001 inside).
# shellcheck disable=SC2034  # read by provision.sh
JOPE_VOLUMES=(jope-work jope-cargo jope-target jope-uv jope-sdk jope-models)

# Runner labels per lane, prefixed with the host. `<host>-gpu`: the GPU, no docker socket.
# `<host>-kernel`: no GPU, the
# docker socket *proxy* (containers/images only) for deploy/ci/kernel-v1-product-build.sh.
jope_runner_labels() { # <lane>
  case $1 in
    gpu) echo "${JOPE_RUNNER_HOST:-xbabe2}-gpu" ;;
    kernel) echo "${JOPE_RUNNER_HOST:-xbabe2}-kernel" ;;
    # The CPU lane: rust-test, e2e-smoke and bench need no GPU and no docker socket. It carries a
    # host-neutral label as well so a workflow can target "the CI runner" without naming a box.
    # ci, ci2..ci4: four instances of the same CPU lane so one workflow can run its jobs
    # concurrently. They share the host-neutral jope-ci label; the per-host label keeps
    # them identifiable in the runner list.
    ci|ci2|ci3|ci4) echo "${JOPE_RUNNER_HOST:-xbabe2}-ci,jope-ci" ;;
    *) echo "jope-runner: unknown lane '$1' (gpu|kernel|ci)" >&2; return 2 ;;
  esac
}
