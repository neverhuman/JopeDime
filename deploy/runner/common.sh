#!/usr/bin/env bash
# deploy/runner/common.sh — shared settings of the self-hosted runner scripts, on whichever
# xbabe host the lanes are installed on (JOPE_RUNNER_HOST, default `hostname -s`). Sourced,
# never executed. Nothing here contains a credential; the PAT lives only in
# /etc/jope-runner/github-pat (root:root 0600, read by run.sh at job start, never mounted).
# shellcheck shell=bash
# shellcheck disable=SC2034 # Shared settings are consumed by the sourcing scripts.

JOPE_RUNNER_IMAGE=${JOPE_RUNNER_IMAGE:-jope-runner:current}          # deploy/docker/runner.Dockerfile, tagged by provision.sh
JOPE_RUNNER_REPO=${JOPE_RUNNER_REPO:-neverhuman/JopeDime}
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
# Build-cache trim (jope_runner_trim_lane): newest builds kept per crate and kind, at most daily.
JOPE_RUNNER_TRIM_KEEP=3
JOPE_RUNNER_TRIM_INTERVAL_S=86400
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
  /opt/jope-runner/trim-deps.sh
  /opt/jope-runner/job-started.sh
  /opt/jope-runner/job-policy.py
  /opt/jope-runner/mac-controller.sh
  /opt/jope-runner/mac-guest.sh
  /opt/jope-runner/mac-capacity.json
  /etc/systemd/system/jope-mac-runner@.service
  /etc/systemd/system/jope-mac-runner@.timer
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

# runc writes memory.swap.max=0 for --memory-swap == --memory, but registers the transient
# docker-<id>.scope with MemorySwapMax=infinity (it omits the property when the v2 swap value
# is 0). Any later systemd re-apply of that scope (daemon-reload, enable/disable, provision.sh)
# then flips the live listener to swap=max and verify-container.py rejects its next job. Record
# the cap in systemd itself so a re-apply keeps swap=0. Fails closed: a listener whose cap cannot
# be recorded is killed before a job relies on it.
jope_runner_pin_swap() { # <container name>
  local name=${1:?container name required} id='' i
  for ((i = 0; i < 120; i++)); do
    id=$(docker inspect --format '{{if .State.Running}}{{.Id}}{{end}}' "$name" 2>/dev/null) || id=''
    [[ -n $id ]] && break
    sleep 0.5
  done
  [[ -n $id ]] || { echo "swap pin: container $name is not running" >&2; return 2; }
  if systemctl set-property --runtime "docker-$id.scope" MemorySwapMax=0 &&
     [[ $(systemctl show "docker-$id.scope" -p MemorySwapMax --value) == 0 ]]; then
    echo "swap pin: docker-$id.scope MemorySwapMax=0"
    return 0
  fi
  echo "swap pin: could not record MemorySwapMax=0 for $name; killing it" >&2
  docker kill "$name" >/dev/null 2>&1 || true
  return 2
}

# Human allocation, 2026-10-23: retain the seven reviewed heavy slots and add
# four PR-only slots in the dedicated xbabe4 Linux Docker VM plus a small guard
# slot on xbabe1/2/3. Guard slots run metadata-only main-guard, never Rust/full CI.
# Existing full slots retain the owner-authorized jope-ci label. Each lane has
# separate volumes; only full takes the host-local ASM allocation lock.
jope_runner_role() { # <host> <lane>; callers use actual hostname, never a job variable
  case "$1/$2" in
    xbabe1/ci|xbabe1/ci2|xbabe2/ci|xbabe3/ci|xbabe0/ci|xbabe4/ci|xbabe4/ci2|xbabe4/ci3|xbabe4/ci4) JOPE_RUNNER_DOMAIN="pr"; JOPE_RUNNER_MEMORY=16g; JOPE_RUNNER_CPUS=16 ;;
    xbabe2/full|xbabe3/full) JOPE_RUNNER_DOMAIN=main; JOPE_RUNNER_MEMORY=48g; JOPE_RUNNER_CPUS=16 ;;
    xbabe1/guard|xbabe2/guard|xbabe3/guard) JOPE_RUNNER_DOMAIN=guard; JOPE_RUNNER_MEMORY=2g; JOPE_RUNNER_CPUS=2 ;;
    *) echo "unapproved runner host/lane: $1/$2" >&2; return 2 ;;
  esac
}

# A reviewed central controller may supply only a one-use JIT config to the
# dedicated Mac Linux guest. No long-lived GitHub credential crosses this boundary.
# Match the image entrypoint's 262144-byte bound and never print the private input.
jope_runner_external_jit() { # <actual host> <lane> <public runner name>, encoded config on stdin
  local host=${1:?host required} lane=${2:?lane required} name=${3:?runner name required} payload
  export -n payload JOPE_EXTERNAL_JIT
  JOPE_EXTERNAL_JIT=''
  case "$host/$lane" in
    xbabe4/ci|xbabe4/ci2|xbabe4/ci3|xbabe4/ci4) ;;
    *) echo "external JIT is not approved for this host/lane" >&2; return 2 ;;
  esac
  [[ $name == "$host-$lane-"* && $name =~ ^xbabe4-ci(2|3|4)?-[0-9]{8}T[0-9]{6}Z$ ]] || {
    echo "external JIT runner name does not match its lane" >&2; return 2;
  }
  # The sentinel preserves trailing newlines so they cannot disappear before validation.
  payload=$(head -c 262145 && printf '.') || return 2
  payload=${payload%.}
  [[ ${#payload} -ge 32 && ${#payload} -le 262144 && $payload =~ ^[A-Za-z0-9+/=]+$ ]] || {
    echo "external JIT stdin is malformed or exceeds the bounded input" >&2; return 2;
  }
  JOPE_EXTERNAL_JIT=$payload
}

# The approved lanes of a host, in provisioning order; every one is also a jope_runner_role case.
jope_runner_lanes() { # <host>
  case $1 in
    xbabe1) echo "ci ci2 guard" ;;
    xbabe2|xbabe3) echo "full ci guard" ;;
    xbabe0) echo "ci" ;;
    xbabe4) echo "ci ci2 ci3 ci4" ;;
    *) echo "unapproved CPU runner host: $1" >&2; return 2 ;;
  esac
}

jope_runner_labels() {
  jope_runner_role "$JOPE_RUNNER_HOST" "$1" || return
  case $1 in
    ci|ci2|ci3|ci4) echo "$JOPE_RUNNER_HOST-$1,jope-ci" ;;
    full) echo "$JOPE_RUNNER_HOST-full,jope-full-cpu,jope-ci" ;;
    guard) echo "$JOPE_RUNNER_HOST-guard,jope-main-guard" ;;
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
      xbabe1/ci|xbabe1/ci2|xbabe1/guard|xbabe2/full|xbabe2/ci|xbabe2/guard|xbabe3/full|xbabe3/ci|xbabe3/guard|xbabe0/ci|xbabe4/ci|xbabe4/ci2|xbabe4/ci3|xbabe4/ci4) ;;
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
  # Each full lane takes its own host's lock inode; a host's ASM campaigns use the same one.
  case $JOPE_RUNNER_HOST/$1 in
    xbabe2/full|xbabe3/full) ;;
    *) return 0 ;;
  esac
  [[ -f /run/lock/jope-exclusive-cpu.lock ]] || {
    echo "shared CPU lock not provisioned" >&2; return 2;
  }
  # flock on this local inode needs no file writes. The runner's strict service
  # mount namespace deliberately keeps /run/lock read-only.
  exec 8</run/lock/jope-exclusive-cpu.lock
  flock -n 8 || { echo "ASM/exclusive CPU allocation active" >&2; return 2; }
  jope_runner_check_asm_groups /sys/fs/cgroup/system.slice
}

# A lane's persistent Cargo target grows with every build: xbabe1's two pr targets reached 355 GB
# each, and the disk guard then refused every listener (2026-10-22). run.sh calls this while it
# holds the lane lock and before the lane's container starts, so no build of this lane can be
# running. At most once per JOPE_RUNNER_TRIM_INTERVAL_S it removes older duplicate builds from
# <target volume>/*/debug/deps with trim-deps.sh (the newest JOPE_RUNNER_TRIM_KEEP per crate and
# kind stay), inside a network-less container that mounts only this lane's target volume. Every
# removed path is recorded in a manifest under the state directory. A failure never blocks the
# listener, and nothing is removed that was not listed first.
jope_runner_trim_lane() { # <lane>; needs JOPE_RUNNER_DOMAIN from jope_runner_role
  local lane=${1:?lane required} dir stamp_file now last manifest volume summary
  volume="jope-$JOPE_RUNNER_DOMAIN-$lane-target"
  dir="$JOPE_RUNNER_STATE/trim"
  stamp_file="$dir/$lane.last"
  now=$(date +%s)
  if [[ -f $stamp_file ]]; then
    last=$(stat -c %Y "$stamp_file") || last=0
    (( now - last >= JOPE_RUNNER_TRIM_INTERVAL_S )) || return 0
  fi
  docker volume inspect "$volume" >/dev/null 2>&1 || { echo "trim $lane: no volume $volume, skipped"; return 0; }
  mkdir -p "$dir" || return 0
  manifest="$dir/$(date -u +%Y%m%dT%H%M%SZ)-$lane.tsv"
  if docker run --rm --network none --user "$JOPE_RUNNER_UID" --cap-drop ALL \
       --security-opt no-new-privileges --read-only --tmpfs /tmp:size=64m \
       --pids-limit 256 --memory 1g --cpus 2 \
       -v "$volume:/v" -v /opt/jope-runner/trim-deps.sh:/opt/jope-runner/trim-deps.sh:ro \
       --entrypoint bash "$JOPE_RUNNER_IMAGE" \
       /opt/jope-runner/trim-deps.sh /v "$JOPE_RUNNER_TRIM_KEEP" --apply > "$manifest"; then
    touch "$stamp_file"
    summary=$(awk -F'\t' '{n++; s+=$2} END {printf "%d files, %d bytes", n, s}' "$manifest")
    echo "trim $lane: removed $summary from $volume; manifest $manifest sha256 $(sha256sum < "$manifest" | cut -d' ' -f1)"
  else
    echo "trim $lane: failed; anything removed is listed in $manifest; the listener continues" >&2
  fi
  return 0
}
