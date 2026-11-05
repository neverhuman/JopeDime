#!/bin/sh
set -eu

case "${1:-}" in
  -h|--help) printf '%s\n' 'Usage: JAIN_ARTIFACT_ROOT=/absolute/h256 sh compose_dendrite_film.sh [SCENE_RECEIPT [LOOPS [CUDA_ICON]]]'; exit 0 ;;
esac
orchestration=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
worktree=$(python3 "$orchestration/common.py" repo)
dendrite_root=$(python3 "$orchestration/common.py" dendrite)
variable_root=$(dirname -- "$dendrite_root")
capture="$variable_root/capture"
scene_receipt=${1:-$dendrite_root/scene/scene.receipt.json}
loops=${2:-$dendrite_root/loops}
cuda_icon=${3:-$(python3 "$orchestration/common.py" cuda)}
output=$(python3 "$orchestration/common.py" path "$dendrite_root/final")
binding_output=$(python3 "$orchestration/common.py" path "$dendrite_root/composition-binding.receipt.json")
qa_root=$(python3 "$orchestration/common.py" path "$dendrite_root/qa-binding")

test ! -e "$output"
test -s "$cuda_icon"
rtk proxy python3 "$orchestration/verify_bindings.py" "$scene_receipt" "$loops" \
    > "$binding_output"

rtk proxy python3 "$worktree/tools/visualization/intelligence_wave.py" \
    --scene-receipt "$scene_receipt" --source-catalog "$capture/multilingual-source/catalog.json" \
    --loops-root "$loops" --table-panels "$capture/table-panels" \
    --neural-influence "$capture/neural-influence.json" --cuda-icon "$cuda_icon" --output-dir "$output"

film="$output/JAIN-intelligence-wave-42s.mp4"
rtk proxy ffprobe -v error -count_frames -select_streams v:0 -show_streams -show_format -show_chapters \
    -of json "$film" > "$output/ffprobe.json"
rtk proxy ffmpeg -nostdin -v error -xerror -i "$film" -map 0:v:0 -f null - 2> "$output/decode.log"
rtk proxy ffmpeg -nostdin -v error -xerror -i "$film" -vf 'select=eq(mod(n\,126)\,17),scale=480:270,tile=5x2' \
    -frames:v 1 "$output/ten-dataset-review.png"

# Bind the existing verifier's relative paths to the immutable capture assets
# and the Dendrite scene/output, so the original pixel and source QA runs intact.
rtk proxy python3 - "$qa_root" "$capture" "$scene_receipt" "$output" <<'PY'
from pathlib import Path
import sys

qa_root, capture, scene_receipt, output = (Path(arg).resolve() for arg in sys.argv[1:])
qa_capture = qa_root / "capture"
(qa_capture / "scene").mkdir(parents=True)
(qa_capture / "scene/scene.receipt.json").symlink_to(scene_receipt)
(qa_capture / "final").symlink_to(output, target_is_directory=True)
for name in ("multilingual-source", "table-panels", "neural-influence.json"):
    target = capture / name
    (qa_capture / name).symlink_to(target, target_is_directory=target.is_dir())
PY
rtk proxy python3 - "$orchestration/verify_neural_cut.py" "$worktree" "$qa_root" "$cuda_icon" <<'PY'
from functools import partial
from pathlib import Path
import runpy
import sys

verifier, worktree, qa_root, cuda_icon = (Path(arg).resolve() for arg in sys.argv[1:])
sys.path.insert(0, str(verifier.parent))
sys.path.insert(0, str(worktree / "tools/visualization"))
import intelligence_wave

intelligence_wave.load_icon = partial(intelligence_wave.load_icon, cuda_icon=cuda_icon)
sys.argv = [str(verifier), str(qa_root)]
runpy.run_path(str(verifier), run_name="__main__")
PY

rtk proxy python3 - "$dendrite_root/composition-binding.receipt.json" "$output" "$cuda_icon" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

bindings, output, cuda_icon = (Path(arg).resolve() for arg in sys.argv[1:])
encode_path = output / "encode.receipt.json"
review_path = output / "review.receipt.json"
bound, encode, review = (json.loads(path.read_bytes()) for path in (bindings, encode_path, review_path))
assert bound["status"] == encode["status"] == review["status"] == "PASS"
assert bound["scene_receipt_sha256"] == encode["scene_receipt_sha256"]
assert bound["spatial_layout"] == "dendrite"
for loop, case in zip(bound["cases"], encode["cases"], strict=True):
    assert loop["case_id"] == case["case_id"]
    assert loop["render_receipt_sha256"] == case["loop_receipt_sha256"]
assert encode["movie_sha256"] == review["movie_sha256"]
cuda_icon_sha = hashlib.sha256(cuda_icon.read_bytes()).hexdigest()
cuda_icon_hashes = {case["language_icon_hashes"]["cuda"] for case in encode["cases"]
                   if "cuda" in case["language_icon_hashes"]}
assert cuda_icon_hashes == {cuda_icon_sha}
receipt = {
    "schema": "jain.dendrite-delivery/v1", "status": "PASS", "spatial_layout": "dendrite",
    "movie": encode["movie"], "movie_sha256": encode["movie_sha256"],
    "encode_sha256": hashlib.sha256(encode_path.read_bytes()).hexdigest(),
    "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
    "binding_sha256": hashlib.sha256(bindings.read_bytes()).hexdigest(),
    "scene_sha256": bound["scene_sha256"], "scene_receipt_sha256": bound["scene_receipt_sha256"],
    "capture_trace_sha256": bound["capture_trace_sha256"], "capture_metrics_sha256": bound["capture_metrics_sha256"],
    "cuda_icon_override": str(cuda_icon), "cuda_icon_override_sha256": cuda_icon_sha,
    "decoded_frames": review["decoded_frames"], "duration_seconds": review["duration_seconds"],
    "width": review["width"], "height": review["height"], "fps": review["fps"],
    "chapters": review["chapters"], "language_count": review["language_count"],
    "verified_cell_colors": review["verified_cell_colors"], "full_source_scroll": review["full_source_scroll"],
}
with (output / "dendrite-delivery.receipt.json").open("x") as file:
    json.dump(receipt, file, indent=2)
    file.write("\n")
print(json.dumps(receipt, indent=2))
PY
