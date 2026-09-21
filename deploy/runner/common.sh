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
JOPE_RUNNER_HOST=${JOPE_RUNNER_HOST:-$(hostname -s)}       # runner names and labels follow the host
JOPE_RUNNER_MEMORY=${JOPE_RUNNER_MEMORY:-48g}              # xbabe1 has 62 GB; leave the host headroom
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
