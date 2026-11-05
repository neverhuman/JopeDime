#!/usr/bin/env python3
"""Verify Dendrite geometry is the sole inference-facing change before composition."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


from common import REPO, artifact_family, validate_configuration

ROOT = artifact_family("variable-attribution-v1/dendrite-final-v1")
CAPTURE = ROOT.parent / "capture"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("loops", type=Path)
    args = parser.parse_args()
    validate_configuration()
    worktree, scene_path, loops = REPO, args.scene.resolve(), args.loops.resolve()
    sys.path.insert(0, str(worktree / "tools/visualization"))
    from animate_source import case_ids, verify_loop

    baseline = read(CAPTURE / "scene/scene.receipt.json")
    scene = read(scene_path)
    scene_sha = digest(scene_path)
    blend = scene_path.with_name(scene_path.name.removesuffix(".receipt.json") + ".blend")
    blend_sha = digest(blend)
    for key in ("trace_sha256", "influence_sha256", "round_trip_metrics_sha256",
                "round_trip_metrics", "dataset_catalog_sha256", "case_frames", "tables",
                "whole_source_pages", "source_viewport", "fps", "loop_frames"):
        require(scene[key] == baseline[key], f"Dendrite changed capture or film binding: {key}")
    require(scene["animation_loop"] is True, "Expected a reusable animation-loop scene")
    network, original_network = scene["network"], baseline["network"]
    require(network["spatial_layout"] == "dendrite", "Scene is not Dendrite")
    for key in ("stage_planes", "operations", "functional_projection_proofs", "displayed_nodes",
                "displayed_strands", "available_nonzero_scalar_connections"):
        require(network[key] == original_network[key], f"Dendrite changed captured network field: {key}")
    for name, expected in scene["source_hashes"].items():
        require(digest(worktree / "tools/visualization/blender" / name) == expected,
                f"Scene and checked-out renderer source differ: {name}")
    cases = []
    for cid, (first, last) in zip(case_ids(scene), scene["case_frames"], strict=True):
        receipt, receipt_path = verify_loop(loops, {"case_id": cid, "loop_frame_start": first,
                                                   "loop_frame_end": last}, scene_sha)
        require(receipt["scene_sha256"] == blend_sha, f"{cid} was rendered from a different .blend")
        require(len(receipt["frames"]) == 120, f"{cid} does not have 120 frames")
        cases.append({"case_id": cid, "render_receipt_sha256": digest(receipt_path),
                      "render_receipt_path": str(receipt_path), "frames": len(receipt["frames"])})
    require(len(cases) == 10, "Expected ten distinct dataset render loops")
    print(json.dumps({"schema": "jain.dendrite-composition-binding/v1", "status": "PASS",
                      "scene_receipt_sha256": scene_sha, "scene_sha256": blend_sha,
                      "capture_trace_sha256": scene["trace_sha256"],
                      "capture_metrics_sha256": scene["round_trip_metrics_sha256"],
                      "spatial_layout": network["spatial_layout"], "frames": 1200, "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
