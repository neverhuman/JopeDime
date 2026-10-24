#!/usr/bin/env bash
# Read-only preflight in the dedicated xbabe4 Linux VM. No registration or container start.
set -euo pipefail
export LC_ALL=C

jope_mac_guest_container_ok() { # public safe Docker fields on stdin, reviewed policy JSON
  local policy=${1:?policy required}
  jq -e --argjson p "$policy" '
      (.lane | test("^ci([2-9]|1[0-2])?$")) and
      .memory == 17179869184 and .swap == .memory and .cpus == 16000000000 and
      .image == $p.image_id and .readonly == true and .pids == 4096 and
      (.caps | index("ALL") != null) and (.security | index("no-new-privileges") != null) and
      (.lane as $lane | .mounts | all(
        if .Type == "volume" then
          (.Name == $p.git_emulation.volume and .Destination == "/opt/jope-git-emulation" and .RW == false) or
          (.Name == "jope-sdk" and .Destination == "/opt/jope-sdk" and .RW == false) or
          (.Name == ("jope-pr-"+$lane+"-work") and .Destination == "/home/runner/work") or
          (.Name == ("jope-pr-"+$lane+"-cargo") and .Destination == "/var/cache/jope/cargo") or
          (.Name == ("jope-pr-"+$lane+"-target") and .Destination == "/var/cache/jope/target") or
          (.Name == ("jope-pr-"+$lane+"-uv") and .Destination == "/var/cache/jope/uv")
        elif .Type == "bind" then .RW == false and (
          (.Source == "/opt/jope-runner/job-started.sh" and .Destination == .Source) or
          (.Source == "/opt/jope-runner/job-policy.py" and .Destination == .Source) or
          (.Source == "/opt/jope-runner/mac-git-entrypoint.sh" and .Destination == .Source) or
          (.Source == "/opt/jope-runner/mac-capacity.json" and .Destination == .Source) or
          (.Source == ("/var/lib/jope-runner/"+$lane+"-policy.json") and .Destination == "/opt/jope-runner/policy.json"))
        else false end)) and (.mounts | length == 11) and
      (.mounts | map(.Destination) | unique | length == 11)
  ' >/dev/null
}

jope_mac_guest() {
  local lane=$1 policy info image recorded active='[]' cid safe running cg quota period sdk_root sdk
  local digest total available root_free docker_free lock_idle=true git_tool
  [[ $(id -u) == 0 && $(hostname -s) == xbabe4 ]] || return 2
  case $lane in ci|ci2|ci3|ci4|ci5|ci6|ci7|ci8|ci9|ci10|ci11|ci12) ;; *) return 2 ;; esac
  policy=$(cat /opt/jope-runner/mac-capacity.json) || return 2
  # Keep the runner, Rust and Node on the working Rosetta engine. Only Git uses
  # the separate pinned public emulator; refuse drift before any new JIT mint.
  [[ $(head -1 /proc/sys/fs/binfmt_misc/rosetta) == enabled ]] || return 2
  if [[ -e /proc/sys/fs/binfmt_misc/qemu-x86_64 ]]; then
    [[ $(head -1 /proc/sys/fs/binfmt_misc/qemu-x86_64) == disabled ]] || return 2
  fi
  # shellcheck source=deploy/runner/mac-git-entrypoint.sh
  # shellcheck disable=SC1091
  . /opt/jope-runner/mac-git-entrypoint.sh
  git_tool=$(jope_mac_git_tool inspect-volume /opt/jope-runner/mac-capacity.json '') || return 2
  info=$(docker info --format '{"root":{{json .DockerRootDir}},"version":{{json .CgroupVersion}},"driver":{{json .CgroupDriver}}}') || return 2
  image=$(docker image inspect --format '{"id":{{json .Id}},"arch":{{json .Architecture}}}' jope-runner:current) || return 2
  recorded=$(cat /var/lib/jope-runner/image.json) || return 2
  # Safe fields only. Config.Env and Args can carry private input and are never inspected.
  local ids
  ids=$(docker ps --filter label=jope.runner=1 --format '{{.ID}}') || return 2
  while IFS= read -r cid; do
    [[ -n $cid ]] || continue
    safe=$(docker inspect --format \
      '{"id":{{json .Id}},"lane":{{json (index .Config.Labels "jope.lane")}},"memory":{{json .HostConfig.Memory}},"swap":{{json .HostConfig.MemorySwap}},"cpus":{{json .HostConfig.NanoCpus}},"image":{{json .Image}},"readonly":{{json .HostConfig.ReadonlyRootfs}},"caps":{{json .HostConfig.CapDrop}},"security":{{json .HostConfig.SecurityOpt}},"pids":{{json .HostConfig.PidsLimit}},"mounts":{{json .Mounts}}}' "$cid") || return 2
    printf '%s' "$safe" | jope_mac_guest_container_ok "$policy" || return 2
    running=$(printf '%s' "$safe" | jq -er .lane) || return 2
    active=$(printf '%s' "$active" | jq -c --arg lane "$running" '. + [$lane]') || return 2
    cid=$(printf '%s' "$safe" | jq -er .id) || return 2
    [[ $cid =~ ^[0-9a-f]{64}$ ]] || return 2
    cg=/sys/fs/cgroup/system.slice/docker-$cid.scope
    [[ $(cat "$cg/memory.max") == 17179869184 && $(cat "$cg/memory.swap.max") == 0 && $(cat "$cg/pids.max") == 4096 ]] || return 2
    read -r quota period < "$cg/cpu.max" || return 2
    [[ $quota =~ ^[0-9]+$ && $period =~ ^[0-9]+$ && $period -gt 0 ]] && (( quota == 16*period )) || return 2
  done <<< "$ids"
  [[ ! -L /var/lib/jope-runner/$lane.lock ]] || return 2
  if [[ -e /var/lib/jope-runner/$lane.lock ]]; then
    exec 9<"/var/lib/jope-runner/$lane.lock" || return 2
    flock -n 9 || lock_idle=false
  fi
  sdk_root=$(docker volume inspect --format '{{.Mountpoint}}' jope-sdk) || return 2
  [[ ! -L $sdk_root/libtorch && ! -L $sdk_root/libtorch/build-version ]] || return 2
  sdk=$(head -c 65 "$sdk_root/libtorch/build-version") || return 2
  [[ $sdk =~ ^[0-9A-Za-z.+_-]{1,64}$ ]] || return 2
  digest=$(bash -ec 'source /opt/jope-runner/common.sh; jope_runner_deployed_digest') || return 2
  total=$(awk '/^MemTotal:/{printf "%.6f",$2/1048576}' /proc/meminfo) || return 2
  available=$(awk '/^MemAvailable:/{printf "%.6f",$2/1048576}' /proc/meminfo) || return 2
  root_free=$(df -B1 --output=avail / | tail -1 | tr -d ' ') || return 2
  docker_free=$(df -B1 --output=avail "$(printf '%s' "$info" | jq -er .root)" | tail -1 | tr -d ' ') || return 2
  jq -nc --arg host xbabe4 --arg arch "$(uname -m)" --argjson cpus "$(getconf _NPROCESSORS_ONLN)" \
    --argjson total "$total" --argjson available "$available" --argjson root "$root_free" \
    --argjson docker "$docker_free" --argjson info "$info" --argjson image "$image" \
    --argjson recorded "$recorded" --arg digest "$digest" --arg sdk "$sdk" --argjson idle "$lock_idle" \
    --argjson active "$active" --argjson git_tool "$git_tool" --arg lane "$lane" \
    '{host:$host,architecture:$arch,cpus:$cpus,total_gib:$total,available_gib:$available,
      root_free_gib:($root/1073741824),docker_free_gib:($docker/1073741824),
      cgroup_version:($info.version|tostring),cgroup_driver:$info.driver,
      image_id:$image.id,image_architecture:$image.arch,recorded_image_id:$recorded.image_id,
      digest:$digest,recorded_digest:$recorded.scripts_sha256,sdk_version:$sdk,
      lane_idle:($idle and ($active|index($lane)==null)),active_lanes:$active,reserved_gib:(($active|length)*16),
      rosetta_enabled:true,kernel_qemu_enabled:false,git_emulation:$git_tool}'
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  if [[ $# != 1 ]] || ! jope_mac_guest "$1"; then
    echo 'Mac guest preflight refused; no registration or job was started.' >&2
    exit 1
  fi
fi
