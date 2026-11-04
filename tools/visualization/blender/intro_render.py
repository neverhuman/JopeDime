#!/usr/bin/env python3
"""Render verified JAIN intro frames on a single Cycles GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_animation import (QUALITY_PROFILES, configure_picture, digest, select_gpu, validate_render_profile,
                              verify_png, write_json)  # noqa: E402


def frame_range(value: str) -> tuple[int, int]:
    start, separator, end = value.partition(":")
    try:
        return int(start), int(end if separator else start)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use FRAME or FIRST:LAST") from error


def render(args) -> dict:
    import bpy

    source_scene, source_receipt = args.scene.resolve(), args.receipt.resolve()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("render output must be fresh and outside the checkout")
    validate_render_profile(args.scale, args.samples, args.quality)
    receipt_bytes = source_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt.get("schema") != "jain.intro-scene/v1" or receipt.get("status") != "PASS":
        raise ValueError("expected a passing JAIN intro scene receipt")
    timeline = receipt.get("frame_start"), receipt.get("frame_end")
    if timeline != (1, 330):
        raise ValueError("intro receipt must cover frames 1 through 330")
    scene_sha = digest(source_scene)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    worker_path = Path(__file__)
    shared_worker_path = worker_path.with_name("render_animation.py")
    worker_sha, shared_worker_sha = digest(worker_path), digest(shared_worker_path)
    bpy.ops.wm.open_mainfile(filepath=str(source_scene), load_ui=False, use_scripts=False)
    embedded = bpy.data.texts.get("JAIN_INTRO_RECEIPT.json")
    if embedded is None or json.loads(embedded.as_string()) != receipt:
        raise ValueError("embedded intro receipt differs from the supplied receipt")
    scene = bpy.context.scene
    if (scene.frame_start, scene.frame_end) != timeline:
        raise ValueError("saved intro timeline differs from the receipt")
    settings = scene.render
    if ((settings.resolution_x, settings.resolution_y) != (1920, 1080)
            or settings.fps / settings.fps_base != 30):
        raise ValueError("intro scene must be 1920×1080 at 30 fps")
    if any(not scene.frame_start <= first <= last <= scene.frame_end for first, last in args.frame):
        raise ValueError("requested frame range escapes the saved intro timeline")
    frames = sorted({frame for first, last in args.frame for frame in range(first, last + 1)})
    if len(frames) != sum(last - first + 1 for first, last in args.frame):
        raise ValueError("requested frame ranges overlap")

    devices = select_gpu(bpy, args.device, args.device_id)
    denoiser = "OPTIX" if args.device == "OPTIX" else "OPENIMAGEDENOISE"
    settings.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = denoiser
    scene.cycles.denoising_use_gpu = True
    scene.cycles.seed = 20261103
    scene.cycles.use_animated_seed = False
    scene.cycles.samples = args.samples
    settings.use_persistent_data = True
    profile = configure_picture(scene, args.scale, args.quality)
    width, height = settings.resolution_x * args.scale // 100, settings.resolution_y * args.scale // 100
    output.mkdir(parents=True, exist_ok=False)

    result = {
        "schema": "jain.intro-gpu-frames/v1", "status": "RUNNING",
        "scene_sha256": scene_sha, "scene_receipt_sha256": receipt_sha,
        "worker_sha256": worker_sha, "shared_worker_sha256": shared_worker_sha,
        "backend": args.device, "devices": devices, "gpu_only": True, "cpu_fallback": False,
        "cycles_seed": 20261103, "animated_seed": False, "samples": args.samples,
        "denoising": denoiser, "denoising_use_gpu": True, "persistent_data": True,
        "width": width, "height": height, "fps": 30, **profile,
        "blender_version": bpy.app.version_string, "requested_frames": frames,
        "frames": [], "argv": sys.argv,
    }
    started = time.monotonic_ns()
    try:
        with (output / "frames.jsonl").open("x", encoding="utf-8") as journal:
            for frame in frames:
                name = f"frame-{frame:06d}.png"
                destination = output / name
                scene.frame_set(frame)
                settings.filepath = str(destination)
                frame_started = time.monotonic_ns()
                bpy.ops.render.render(write_still=True)
                if args.quality == "uhd-master":
                    verify_png(destination, width, height, profile["color_depth"])
                entry = {"frame": frame, "file": name, "sha256": digest(destination),
                         "elapsed_ns": time.monotonic_ns() - frame_started}
                result["frames"].append(entry)
                journal.write(json.dumps(entry) + "\n")
                journal.flush()
                print(json.dumps(entry), flush=True)
        if digest(source_scene) != scene_sha or digest(source_receipt) != receipt_sha:
            raise RuntimeError("saved intro scene or receipt changed during rendering")
        if digest(worker_path) != worker_sha or digest(shared_worker_path) != shared_worker_sha:
            raise RuntimeError("intro or shared render worker changed during rendering")
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame", action="append", type=frame_range, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--scale", type=int, choices=(100, 200), default=100)
    parser.add_argument("--quality", choices=QUALITY_PROFILES, default="standard")
    parser.add_argument("--device", choices=("OPTIX", "METAL"), default="OPTIX")
    parser.add_argument("--device-id")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if not 1 <= args.samples <= 4096:
        parser.error("--samples must be in [1, 4096]")
    render(args)


if __name__ == "__main__":
    main()
