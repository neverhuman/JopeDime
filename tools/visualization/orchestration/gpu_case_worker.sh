#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' 'Usage: gpu_case_worker.sh MODE CASE_ID SCENE_SHA RECEIPT_SHA JOB'
  exit 0
fi

[[ "${JAIN_ENABLE_REMOTE_RENDER:-}" == 1 ]] || { printf '%s\n' 'Remote rendering is disabled' >&2; exit 2; }

base=/home/ubuntu/jope-data/jain-render-20261103
runtime=/home/ubuntu/jope-data/jain-render-20261103/runtime/blender-5.2.3-linux-x64/blender
worker=/home/ubuntu/jope-data/jain-render-20261103/variable-attribution-v1/code/tools/visualization/blender/render_animation.py
mode="${1:?preview or loops}"
case_id="${2:?case ID}"
scene_sha="${3:?scene SHA-256}"
receipt_sha="${4:?receipt SHA-256}"
job="${5:?unique job directory name}"

[[ "$mode" == preview || "$mode" == loops ]]
[[ "$case_id" =~ ^case-[a-j]$ ]]
[[ "$scene_sha" =~ ^[0-9a-f]{64}$ && "$receipt_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$job" =~ ^[a-z0-9-]+$ ]]
root="$base/dendrite-final-v1/revisions/${scene_sha:0:12}-${receipt_sha:0:12}"
scene="$root/scene/scene.blend"
receipt="$root/scene/scene.receipt.json"
job_dir="$root/jobs/$job"
finish() {
  result=$?
  printf '%s\n' "$result" > "$job_dir/status.tmp"
  mv -- "$job_dir/status.tmp" "$job_dir/status"
  exit "$result"
}
trap finish EXIT

exec 9>"$base/dendrite-gpu-render.lock"
flock -n 9

printf '%s  %s\n' "$scene_sha" "$scene" "$receipt_sha" "$receipt" \
  ca3c3f8e8866c01ffc54048fc1cee2e4321e213fef555b873b76579572f4dbf5 "$worker" \
  b8b07a190fb25db76aae6de08dbc6c2b3d64e22e82bed6ba90de9545859f3431 "$runtime" | sha256sum -c

free_mib="$(timeout 15s nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
compute_pids="$(timeout 15s nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
printf 'host=%s started_utc=%s free_mib=%s existing_compute_pids=%s\n' \
  "$(hostname)" "$(date -u +%FT%TZ)" "$free_mib" "$compute_pids"
[[ "$free_mib" =~ ^[0-9]+$ ]]
((free_mib >= 6144))
[[ -z "$compute_pids" ]]

output="$root/$mode/$case_id"
test ! -e "$output"
range=()
if [[ "$mode" == preview ]]; then
  case "$case_id" in
    case-a) frame=89 ;;
    case-g) frame=810 ;;
    *) exit 2 ;;
  esac
  range=(--frame-start "$frame" --frame-end "$frame")
fi

command=(timeout --kill-after=30s 7m "$runtime" --background --factory-startup --disable-autoexec --threads 8
  --python-exit-code 1 --python "$worker" -- --scene "$scene" --receipt "$receipt"
  --case-id "$case_id" --output-dir "$output" --device OPTIX --samples 8 "${range[@]}")
printf 'argv:'; printf ' %q' "${command[@]}"; printf '\n'
"${command[@]}"
python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"]=="PASS"' "$output/render.receipt.json"
printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)"
