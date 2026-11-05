#!/usr/bin/env python3
"""Map the approved 45-second film to exactly the native plates it uses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


from common import artifact_family, validate_configuration, external_path

ROOT = artifact_family("jain-final-4k-v1")
INTRO = ROOT.parent / "jain-intro-v2"
DEMO = ROOT.parent / "variable-attribution-v1/dendrite-final-v1"
CASE_IDS = [f"case-{letter}" for letter in "abcdefghij"]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    validate_configuration()
    files = {
        "approved_film_receipt": INTRO / "final/intro-film.receipt.json",
        "intro_scene": INTRO / "scene/intro.blend",
        "intro_scene_receipt": INTRO / "scene/intro.receipt.json",
        "demo_encode_receipt": DEMO / "final/encode.receipt.json",
        "demo_scene": DEMO / "scene/scene.blend",
        "demo_scene_receipt": DEMO / "scene/scene.receipt.json",
    }
    inputs = {key: {"path": str(path), "sha256": digest(path)} for key, path in files.items()}
    approved, intro, encode, scene = (read_json(files[key]) for key in (
        "approved_film_receipt", "intro_scene_receipt", "demo_encode_receipt", "demo_scene_receipt",
    ))
    require(all(document["status"] == "PASS" for document in (approved, intro, encode)), "input receipt failed")
    require((intro["frame_start"], intro["frame_end"], intro["fps"]) == (1, 330, 30), "unexpected intro span")
    require((encode["frames"], encode["fps"]) == (1260, 30), "unexpected original demo span")
    require(approved["intro_scene_receipt_sha256"] == inputs["intro_scene_receipt"]["sha256"]
            and approved["demo_receipt_sha256"] == inputs["demo_encode_receipt"]["sha256"]
            and intro["source_scene_sha256"] == inputs["demo_scene"]["sha256"]
            and intro["source_receipt_sha256"] == encode["scene_receipt_sha256"]
            == inputs["demo_scene_receipt"]["sha256"], "approved film and scenes do not share provenance")
    require([case["case_id"] for case in encode["cases"]] == CASE_IDS, "case order differs")
    selected = approved["retime"]["selected_source_indices"]
    require(len(selected) == 1020 and selected[0] == 0 and selected[-1] == 1259
            and selected == sorted(set(selected)), "approved retime is not strictly increasing")

    mapping = []
    cases = []
    for index, (case, span) in enumerate(zip(encode["cases"], scene["case_frames"], strict=True)):
        cid = case["case_id"]
        first, last = span
        loop_path = DEMO / "loops" / cid / "render.receipt.json"
        loop_digest = digest(loop_path)
        loop = read_json(loop_path)
        require(loop_digest == case["loop_receipt_sha256"] and loop["status"] == "PASS"
                and loop["scene_sha256"] == inputs["demo_scene"]["sha256"]
                and loop["scene_receipt_sha256"] == inputs["demo_scene_receipt"]["sha256"]
                and [frame["frame"] for frame in loop["frames"]] == list(range(first, last + 1)),
                f"original {cid} camera plate receipt differs")
        require((case["start"], case["end"]) == (index * 126, (index + 1) * 126)
                and len(case["plate_indices"]) == 126, f"unexpected {cid} source span")
        indices = [value for value in selected if case["start"] <= value < case["end"]]
        raw_frames = []
        for source_index in indices:
            source_local = source_index - case["start"]
            loop_index = case["plate_indices"][source_local]
            require(0 <= loop_index < len(loop["frames"]), f"invalid camera plate index for {cid}")
            raw = loop["frames"][loop_index]
            raw_frames.append(raw["frame"])
            mapping.append({"master_frame0": 330 + len(mapping), "source_frame0": source_index,
                            "case_id": cid, "source_case_frame0": source_local,
                            "loop_frame0": loop_index, "scene_frame": raw["frame"],
                            "original_hd_plate_sha256": raw["sha256"]})
        unique = sorted(set(raw_frames))
        require(len(indices) == 102 and len(unique) == 96, f"unexpected {cid} sparse plate count")
        cases.append({"case_id": cid, "frames": unique, "case_frame_start": first, "case_frame_end": last,
                      "original_loop_receipt_sha256": loop_digest, "mapped_frames": len(indices)})
    require([record["source_frame0"] for record in mapping] == selected, "assembled source map differs")
    payload = {"schema": "jain.final-4k-render-plan/v1", "status": "PASS", "inputs": inputs,
               "width": 3840, "height": 2160, "fps": 30, "intro_frames": list(range(1, 331)),
               "demo_cases": cases, "demo_mapping": mapping, "demo_unique_frames": 960,
               "native_frames_total": 1290, "standalone_frames": 360, "master_frames": 1350}
    output = external_path(ROOT / "render-plan.receipt.json")
    with external_path(output).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(output, digest(output), flush=True)


if __name__ == "__main__":
    main()
