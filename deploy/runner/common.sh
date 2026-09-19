#!/usr/bin/env bash
# deploy/runner/common.sh — shared settings of the xbabe2 self-hosted runner scripts. Sourced,
# never executed. Nothing here contains a credential; the PAT lives only in
# /etc/jope-runner/github-pat (root:root 0600, read by run.sh at job start, never mounted).
# shellcheck shell=bash

JOPE_RUNNER_IMAGE=${JOPE_RUNNER_IMAGE:-jope-runner:current}          # deploy/docker/runner.Dockerfile, tagged by provision.sh
JOPE_RUNNER_REPO=${JOPE_RUNNER_REPO:-veox-ai/JopeDime}
JOPE_RUNNER_PAT_FILE=${JOPE_RUNNER_PAT_FILE:-/etc/jope-runner/github-pat}
JOPE_RUNNER_STATE=${JOPE_RUNNER_STATE:-/var/lib/jope-runner}          # receipts, last-prune stamp (root-owned)
JOPE_RUNNER_MIN_FREE_GB=${JOPE_RUNNER_MIN_FREE_GB:-50}                # disk guard: refuse to start under this
JOPE_RUNNER_GPU=${JOPE_RUNNER_GPU:-device=0}                          # --gpus value (the RTX 3090, GPU-44c63d19…)
# shellcheck disable=SC2034  # read by run.sh, prune.sh and provision.sh after sourcing
JOPE_RUNNER_UID=1001                                                  # the image's non-root user
JOPE_RUNNER_MEMORY=${JOPE_RUNNER_MEMORY:-96g}
JOPE_RUNNER_CPUS=${JOPE_RUNNER_CPUS:-64}
JOPE_RUNNER_PIDS=${JOPE_RUNNER_PIDS:-4096}
JOPE_RUNNER_TMPFS_SIZE=${JOPE_RUNNER_TMPFS_SIZE:-32g}
JOPE_RUNNER_DOCKER_PROXY_SOCK=${JOPE_RUNNER_DOCKER_PROXY_SOCK:-/run/jope-docker-proxy/docker.sock}

# Named volumes (created by provision.sh; owned by uid 1001 inside).
# shellcheck disable=SC2034  # read by provision.sh
JOPE_VOLUMES=(jope-work jope-cargo jope-target jope-uv jope-sdk jope-models)

# Runner labels per lane. `xbabe2-gpu`: the GPU, no docker socket. `xbabe2-kernel`: no GPU, the
# docker socket *proxy* (containers/images only) for deploy/ci/kernel-v1-product-build.sh.
jope_runner_labels() { # <lane>
  case $1 in
    gpu) echo "xbabe2-gpu" ;;
    kernel) echo "xbabe2-kernel" ;;
    *) echo "jope-runner: unknown lane '$1' (gpu|kernel)" >&2; return 2 ;;
  esac
}
