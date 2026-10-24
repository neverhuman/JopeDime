#!/usr/bin/env bash
# One-use JIT controller on Ubuntu xbabe2. Private response/config bytes stay in pipes/memory.
set +x
set +a
set -uo pipefail
export LC_ALL=C
JOPE_MAC_GH=/home/ubuntu/.local/bin/gh-role

jope_mac_digest() {
  bash -ec 'source /opt/jope-runner/common.sh; jope_runner_deployed_digest'
}

jope_mac_api() { # path [public JSON request]; no private response diagnostics
  if (( $# == 1 )); then
    timeout 60s "$JOPE_MAC_GH" writer api "$1" 2>/dev/null
  else
    printf '%s' "$2" | timeout 60s "$JOPE_MAC_GH" writer api "$1" --method POST --input - 2>/dev/null
  fi
}

jope_mac_ssh() { # host | preflight lane | run lane name (private stdin)
  local mode=$1 remote='' host_code
  local -a args ssh_args
  shift
  ssh_args=(/usr/bin/ssh -T -oBatchMode=yes -oStrictHostKeyChecking=yes
    -oUpdateHostKeys=no -oForwardAgent=no -oServerAliveInterval=30 -oServerAliveCountMax=3)
  case $mode in
    host)
      # shellcheck disable=SC2016 # This program expands on the remote Mac, not here.
      host_code='set -eu; h=$(/bin/hostname); f=$(/bin/df -Pk /Users/agent | /usr/bin/awk "NR==2{print \$4}"); case $f in ""|*[!0-9]*) exit 2;; esac; printf "{\"host\":\"%s\",\"free_bytes\":%s}\n" "$h" "$((f*1024))"'
      args=(/bin/sh -c "$host_code") ;;
    preflight) args=(/opt/homebrew/bin/colima ssh --profile jope-ci -- sudo -n /opt/jope-runner/mac-guest.sh "$1") ;;
    run) args=(/opt/homebrew/bin/colima ssh --profile jope-ci -- sudo -n /opt/jope-runner/run.sh "$1" --jit-stdin "$2") ;;
    *) return 2 ;;
  esac
  printf -v remote '%q ' "${args[@]}"
  "${ssh_args[@]}" xbabe2 "$remote"
}

jope_mac_write_state() { # public JSON only, atomically durable; never a JIT value
  local file=$1 body=$2
  [[ ! -L $file && ! -e $file.new && ! -L $file.new ]] || return 2
  (umask 077; set -C; printf '%s\n' "$body" > "$file.new") || return 2
  sync "$file.new" || return 2
  mv -T -- "$file.new" "$file" || return 2
  sync "$(dirname "$file")" || return 2
}

jope_mac_finish() { # dynamic locals state/current/history/name/start are public, from launch
  local phase=$1 finished elapsed receipt
  finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) || return 2
  elapsed=$((SECONDS-start))
  state=$(printf '%s' "$state" | jq -c --arg phase "$phase" --arg ended "$finished" --argjson wall "$elapsed" \
    '. + {phase:$phase,ended_utc:$ended,wall_seconds:$wall}') || return 2
  jope_mac_write_state "$current" "$state" || return 2
  receipt=$history/$name.json
  (umask 077; set -C; printf '%s\n' "$state" > "$receipt") || return 2
  sync "$receipt" "$history" || return 2
}

jope_mac_validate_guest() { # public facts, policy, lane, expected installed digest
  printf '%s' "$1" | jq -e --argjson p "$2" --arg lane "$3" --arg digest "$4" '
    ([.total_gib,.available_gib,.root_free_gib,.docker_free_gib] | all(type == "number")) and
    .host == "xbabe4" and .architecture == "aarch64" and .cpus == $p.guest_cpus and
    .total_gib >= $p.guest_min_total_gib and .available_gib >= $p.guest_min_available_gib and
    .root_free_gib >= $p.guest_min_root_free_gib and .docker_free_gib >= $p.guest_min_docker_free_gib and
    .cgroup_version == "2" and .cgroup_driver == "systemd" and .image_id == $p.image_id and
    .recorded_image_id == $p.image_id and .image_architecture == "amd64" and
    .digest == $digest and .recorded_digest == $digest and .sdk_version == $p.sdk_version and
    .rosetta_enabled == true and .kernel_qemu_enabled == false and
    .git_emulation == ($p.git_emulation | {
      binary_sha256:.binary_sha256,binary_bytes:.binary_bytes,architecture:"aarch64",
      uid:0,gid:0,mode:"0555",volume:.volume,version:.version}) and
    .lane_idle == true and (.active_lanes | type == "array") and
    (.active_lanes | index($lane) == null) and
    (.active_lanes | all(test("^ci([2-9]|1[0-2])?$"))) and
    (.active_lanes | length == (unique | length)) and
    .reserved_gib == ((.active_lanes | length) * 16) and (.reserved_gib + 16 <= 192)
  ' >/dev/null
}

jope_mac_launch() { # lane, provisioned state dir, public policy JSON, optional test timestamp
  local lane=$1 state_dir=$2 cfg=$3 stamp=${4:-} digest mac guest user name current history state start request response='' encoded='' id labels
  export -n response encoded
  case $lane in ci|ci2|ci3|ci4|ci5|ci6|ci7|ci8|ci9|ci10|ci11|ci12) ;; *) return 2 ;; esac
  printf '%s' "$cfg" | jq -e '
    .schema == "jope.mac-capacity/v1" and .ssh_alias == "xbabe2" and .profile == "jope-ci" and
    .runner_label == "jope-mac-ci" and
    .git_emulation.schema == "jope.mac-git/v1" and
    (.git_emulation.binary_sha256 | type == "string" and test("^[0-9a-f]{64}$")) and
    (.git_emulation.binary_bytes | type == "number" and . >= 20 and . <= 67108864) and
    .git_emulation.volume == ("jope-git-emulation-" + .git_emulation.binary_sha256[0:12]) and
    .colima == "/opt/homebrew/bin/colima" and .mac_hostname == "Mac-Studio.local" and
    .guest_hostname == "xbabe4" and (.image_id | test("^sha256:[0-9a-f]{64}$"))
  ' >/dev/null || return 2
  [[ -d $state_dir && ! -L $state_dir && ! -L $state_dir/$lane.lock ]] || return 2
  umask 077
  exec 9>>"$state_dir/$lane.lock" || return 2
  flock -n 9 || return 2
  current=$state_dir/$lane.json
  [[ ! -L $current && ! -e $current.new && ! -L $current.new ]] || return 2
  if [[ -e $current ]]; then
    jq -e '.phase == "complete"' "$current" >/dev/null || return 2
  fi
  digest=$(jope_mac_digest) || return 2
  [[ $digest =~ ^[0-9a-f]{64}$ ]] || return 2
  mac=$(jope_mac_ssh host) || return 2
  printf '%s' "$mac" | jq -e --argjson p "$cfg" '.host == $p.mac_hostname and (.free_bytes|type == "number") and .free_bytes >= ($p.mac_min_free_gib*1073741824)' >/dev/null || return 2
  guest=$(jope_mac_ssh preflight "$lane") || return 2
  jope_mac_validate_guest "$guest" "$cfg" "$lane" "$digest" || return 2
  user=$(jope_mac_api user) || return 2
  printf '%s' "$user" | jq -e '.login == "jepsontaylor"' >/dev/null || return 2
  [[ -n $stamp ]] || stamp=$(date -u +%Y%m%dT%H%M%SZ) || return 2
  [[ $stamp =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || return 2
  name=xbabe4-$lane-$stamp
  history=$state_dir/history
  [[ ! -L $history ]] || return 2
  mkdir -p -- "$history" || return 2
  [[ ! -e $history/$name.json && ! -L $history/$name.json ]] || return 2
  start=$SECONDS
  state=$(jq -nc --arg name "$name" --arg lane "$lane" --arg digest "$digest" \
    --arg utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{phase:"mint_requested",name:$name,lane:$lane,digest:$digest,started_utc:$utc}') || return 2
  # Persist BEFORE the non-idempotent API call. An interrupted phase blocks the next launch.
  jope_mac_write_state "$current" "$state" || return 2
  labels=$(jq -nc --arg lane "$lane" '["self-hosted","linux","x64","jope-mac-ci","xbabe4-"+$lane]') || return 2
  request=$(jq -nc --arg name "$name" --arg lane "$lane" --argjson labels "$labels" \
    '{name:$name,runner_group_id:1,labels:$labels,work_folder:("/home/runner/work/"+$lane)}') || return 2
  if ! response=$(jope_mac_api repos/neverhuman/JopeDime/actions/runners/generate-jitconfig "$request"); then
    jope_mac_finish unknown || return 2; return 1
  fi
  # Private JSON goes through pipes, never a here-string/temp file or an argv value.
  if (( ${#response} > 2097152 )) || ! printf '%s' "$response" | jq -e --arg name "$name" --argjson labels "$labels" '
    .runner.name == $name and (.runner.id | type == "number" and . > 0 and floor == .) and
    ([.runner.labels[].name | ascii_downcase] | sort) == ($labels | sort)
  ' >/dev/null 2>&1; then jope_mac_finish unknown || return 2; return 1; fi
  id=$(printf '%s' "$response" | jq -er '.runner.id' 2>/dev/null) || { jope_mac_finish unknown; return 1; }
  encoded=$(printf '%s' "$response" | jq -jer '.encoded_jit_config | select(type == "string")' 2>/dev/null && printf '.') || { jope_mac_finish unknown; return 1; }
  encoded=${encoded%.}
  if (( ${#encoded} < 32 || ${#encoded} > 262144 )) || [[ ! $encoded =~ ^[A-Za-z0-9+/=]+$ ]]; then
    jope_mac_finish unknown || return 2; return 1
  fi
  state=$(printf '%s' "$state" | jq -c --argjson id "$id" '. + {phase:"minted",runner_id:$id}') || return 2
  jope_mac_write_state "$current" "$state" || return 2
  # Only stdin crosses SSH. Normal runner output stays with Actions, not this controller.
  if ! printf '%s' "$encoded" | jope_mac_ssh run "$lane" "$name" >/dev/null 2>&1; then
    jope_mac_finish unknown || return 2; return 1
  fi
  unset encoded response
  jope_mac_finish complete || return 2
  printf '%s\n' "$state"
}

jope_mac_main() {
  [[ $# == 1 && $(hostname -s) == xbabe2 ]] || return 2
  local cfg
  cfg=$(cat /opt/jope-runner/mac-capacity.json) || return 2
  jope_mac_launch "$1" /var/lib/jope-mac-controller "$cfg"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  if ! jope_mac_main "$@"; then
    echo 'Mac runner refused; inspect public controller state and remote job before retry.' >&2
    exit 1
  fi
fi
