#!/usr/bin/env python3
"""Render one saved case loop with explicitly selected Cycles GPU devices.

Run with Blender --background --disable-autoexec --python this_file -- ... .
This worker performs no inference, training, installation or host changes.
CPU devices are disabled; absence of the requested GPU backend is fatal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def select_gpu(bpy, backend: str, identifier: str | None) -> list[dict]:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = backend
    prefs.get_devices()
    devices = list(prefs.devices)
    matches = [device for device in devices if device.type == backend
               and (identifier is None or identifier in (device.id, device.name))]
    if not matches:
        raise RuntimeError(f"requested Cycles {backend} GPU unavailable; CPU fallback refused")
    chosen = matches[0]
    for device in devices:
        device.use = device == chosen
    if any(device.use and device.type == "CPU" for device in devices):
        raise RuntimeError("CPU device remained enabled")
    return [{"id": chosen.id, "name": chosen.name, "type": chosen.type}]


def render(args) -> dict:
    import bpy
    source_scene, source_receipt = args.scene.resolve(), args.receipt.resolve()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("render output must be fresh and outside the checkout")
    receipt_bytes = source_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    ids = receipt.get("case_ids") or [case["case_id"] for case in receipt.get("tables", [])]
    if not ids and isinstance(receipt.get("round_trip_metrics"), dict):
        ids = list(receipt["round_trip_metrics"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.case_id) or args.case_id not in ids:
        raise ValueError("requested case is absent from saved scene receipt")
    first, last = receipt["case_frames"][ids.index(args.case_id)]
    begin = args.frame_start if args.frame_start is not None else first
    end = args.frame_end if args.frame_end is not None else last
    if not first <= begin <= end <= last:
        raise ValueError("render range escapes requested case span")
    scene_sha = digest(source_scene)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(source_scene), load_ui=False, use_scripts=False)
    embedded = bpy.data.texts.get("JAIN_CINEMATIC_RECEIPT.json")
    if embedded is None or json.loads(embedded.as_string()) != receipt:
        raise ValueError("embedded scene receipt does not equal supplied receipt")
    scene = bpy.context.scene
    devices = select_gpu(bpy, args.device, args.device_id)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    # OptiX denoising stays on the selected NVIDIA GPU, unlike CPU OIDN.
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX"
    scene.cycles.seed = 20261103
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    if args.samples is not None:
        scene.cycles.samples = args.samples
    scene.render.resolution_percentage = 100
    if (scene.render.resolution_x, scene.render.resolution_y) != (1920, 1080):
        raise ValueError("saved animation scene must already be1920x1080")
    if scene.render.fps / scene.render.fps_base != 30:
        raise ValueError("saved animation scene must already be30fps")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.use_file_extension = True
    scene.render.use_overwrite = False
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic_ns()
    result = {"schema": "jain.gpu-scene-loop/v1", "status": "RUNNING", "case_id": args.case_id,
              "scene_path": str(source_scene), "scene_sha256": scene_sha,
              "scene_receipt_path": str(source_receipt), "scene_receipt_sha256": receipt_sha,
              "worker_sha256": digest(Path(__file__)), "backend": args.device, "devices": devices,
              "gpu_only": True, "cpu_fallback": False, "denoising": "OPTIX",
              "persistent_data": True, "cycles_seed": 20261103, "animated_seed": False,
              "width": 1920, "height": 1080, "fps": 30, "samples": scene.cycles.samples,
              "case_frame_start": first, "case_frame_end": last, "frame_start": begin, "frame_end": end,
              "complete_case": begin == first and end == last,
              "blender_version": bpy.app.version_string, "frames": [], "argv": sys.argv}
    journal = output / "frames.jsonl"
    try:
        with journal.open("x", encoding="utf-8") as stream:
            for frame in range(begin, end + 1):
                name = f"frame-{frame:06d}.png"
                destination = output / name
                scene.frame_set(frame)
                scene.render.filepath = str(destination)
                frame_started = time.monotonic_ns()
                bpy.ops.render.render(write_still=True)
                if not destination.is_file():
                    raise RuntimeError("Cycles returned without the requested PNG")
                entry = {"frame": frame, "file": name, "sha256": digest(destination),
                         "elapsed_ns": time.monotonic_ns() - frame_started}
                result["frames"].append(entry)
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                print(json.dumps({"case_id": args.case_id, **entry}), flush=True)
        if digest(source_receipt) != receipt_sha or digest(source_scene) != scene_sha:
            raise RuntimeError("saved scene or receipt changed during rendering")
        result["status"] = "PASS"
    except BaseException as error:
        result["status"], result["error"] = "FAIL", f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_ns"] = time.monotonic_ns() - started
        write_json(output / "render.receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("CUDA", "OPTIX"), default="OPTIX")
    parser.add_argument("--device-id")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--samples", type=int)
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.samples is not None and args.samples < 1:
        parser.error("positive sample count required")
    result = render(args)
    print(json.dumps({"status": result["status"], "frames": len(result["frames"]),
                      "receipt": str(args.output_dir.resolve() / "render.receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
