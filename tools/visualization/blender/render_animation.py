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
import math
from pathlib import Path
import re
import struct
import sys
import time


QUALITY_PROFILES = ("standard", "uhd-master")
UHD_ADAPTIVE_THRESHOLD = .003
UHD_MIN_SAMPLES = 32


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


def validate_render_profile(scale: int, samples: int | None, quality: str) -> None:
    if quality not in QUALITY_PROFILES or scale not in (100, 200):
        raise ValueError("Unknown render quality or scale")
    if quality == "uhd-master" and (scale != 200 or samples is None or samples < UHD_MIN_SAMPLES):
        raise ValueError("uhd-master requires --scale 200 and at least 32 samples")


def configure_picture(scene, scale: int, quality: str) -> dict:
    """Configure the native raster without changing the saved scene's view transform."""
    settings = scene.render
    settings.resolution_percentage = scale
    settings.image_settings.file_format = "PNG"
    settings.image_settings.color_mode = "RGB"
    settings.image_settings.color_depth = "16" if quality == "uhd-master" else "8"
    settings.image_settings.compression = 35 if quality == "uhd-master" else 15
    settings.use_file_extension = True
    settings.use_overwrite = False
    profile = {
        "quality_profile": quality, "native_render": True, "resolution_percentage": scale,
        "file_format": settings.image_settings.file_format, "color_mode": settings.image_settings.color_mode,
        "color_depth": int(settings.image_settings.color_depth),
        "png_compression": settings.image_settings.compression,
    }
    if quality == "uhd-master":
        cycles = scene.cycles
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = UHD_ADAPTIVE_THRESHOLD
        cycles.adaptive_min_samples = UHD_MIN_SAMPLES
        cycles.denoising_use_gpu = True
        cycles.denoising_quality = "HIGH"
        cycles.denoising_prefilter = "ACCURATE"
        cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        if not math.isclose(cycles.adaptive_threshold, UHD_ADAPTIVE_THRESHOLD, abs_tol=1e-9):
            raise RuntimeError("Cycles did not accept the UHD adaptive threshold")
        profile.update(adaptive_sampling=cycles.use_adaptive_sampling,
                       adaptive_threshold=UHD_ADAPTIVE_THRESHOLD, adaptive_min_samples=cycles.adaptive_min_samples,
                       denoising_use_gpu=cycles.denoising_use_gpu, denoising_quality=cycles.denoising_quality,
                       denoising_prefilter=cycles.denoising_prefilter,
                       denoising_input_passes=cycles.denoising_input_passes)
    return profile


def verify_png(path: Path, width: int, height: int, depth: int) -> None:
    """Reject unexpected size, bit depth, or channel layout before hashing a plate."""
    with path.open("rb") as stream:
        header = stream.read(29)
    if len(header) != 29 or header[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise ValueError(f"{path.name}: invalid PNG IHDR")
    actual = struct.unpack(">IIBB", header[16:26])
    if actual != (width, height, depth, 2):
        raise ValueError(f"{path.name}: expected {width}x{height} RGB{depth} PNG, found IHDR {actual}")


def requested_case_frames(first: int, last: int, start: int | None, end: int | None,
                          sparse: list[int] | None) -> list[int]:
    if sparse is not None:
        if start is not None or end is not None:
            raise ValueError("--frame is mutually exclusive with --frame-start and --frame-end")
        if not sparse or sparse != sorted(set(sparse)) or not first <= sparse[0] <= sparse[-1] <= last:
            raise ValueError("--frame values must increase strictly within the requested case span")
        return sparse
    begin = first if start is None else start
    finish = last if end is None else end
    if not first <= begin <= finish <= last:
        raise ValueError("render range escapes requested case span")
    return list(range(begin, finish + 1))


def render(args) -> dict:
    import bpy
    source_scene, source_receipt = args.scene.resolve(), args.receipt.resolve()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("render output must be fresh and outside the checkout")
    quality, scale = args.quality, args.scale
    samples = 256 if quality == "uhd-master" and args.samples is None else args.samples
    validate_render_profile(scale, samples, quality)
    receipt_bytes = source_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    ids = receipt.get("case_ids") or [case["case_id"] for case in receipt.get("tables", [])]
    if not ids and isinstance(receipt.get("round_trip_metrics"), dict):
        ids = list(receipt["round_trip_metrics"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.case_id) or args.case_id not in ids:
        raise ValueError("requested case is absent from saved scene receipt")
    first, last = receipt["case_frames"][ids.index(args.case_id)]
    frames = requested_case_frames(first, last, args.frame_start, args.frame_end, args.frame)
    scene_sha = digest(source_scene)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    worker_path = Path(__file__)
    worker_sha = digest(worker_path)
    bpy.ops.wm.open_mainfile(filepath=str(source_scene), load_ui=False, use_scripts=False)
    embedded = bpy.data.texts.get("JAIN_CINEMATIC_RECEIPT.json")
    if embedded is None or json.loads(embedded.as_string()) != receipt:
        raise ValueError("embedded scene receipt does not equal supplied receipt")
    scene = bpy.context.scene
    devices = select_gpu(bpy, args.device, args.device_id)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    denoiser = "OPENIMAGEDENOISE" if args.device == "METAL" else "OPTIX"
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = denoiser
    if args.device == "METAL":
        scene.cycles.denoising_use_gpu = True
    scene.cycles.seed = 20261103
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
    if samples is not None:
        scene.cycles.samples = samples
    if (scene.render.resolution_x, scene.render.resolution_y) != (1920, 1080):
        raise ValueError("saved animation scene must already be1920x1080")
    if scene.render.fps / scene.render.fps_base != 30:
        raise ValueError("saved animation scene must already be30fps")
    profile = configure_picture(scene, scale, quality)
    width, height = 1920 * scale // 100, 1080 * scale // 100
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic_ns()
    result = {"schema": "jain.gpu-scene-loop/v1", "status": "RUNNING", "case_id": args.case_id,
              "scene_path": str(source_scene), "scene_sha256": scene_sha,
              "scene_receipt_path": str(source_receipt), "scene_receipt_sha256": receipt_sha,
              "worker_sha256": worker_sha, "backend": args.device, "devices": devices,
              "gpu_only": True, "cpu_fallback": False, "denoising": denoiser,
              "persistent_data": True, "cycles_seed": 20261103, "animated_seed": False,
              "width": width, "height": height, "fps": 30, "samples": scene.cycles.samples,
              "case_frame_start": first, "case_frame_end": last, "frame_start": frames[0], "frame_end": frames[-1],
              "requested_frames": frames, "complete_case": len(frames) == last - first + 1,
              **profile,
              "blender_version": bpy.app.version_string, "frames": [], "argv": sys.argv}
    journal = output / "frames.jsonl"
    try:
        with journal.open("x", encoding="utf-8") as stream:
            for frame in frames:
                name = f"frame-{frame:06d}.png"
                destination = output / name
                scene.frame_set(frame)
                scene.render.filepath = str(destination)
                frame_started = time.monotonic_ns()
                bpy.ops.render.render(write_still=True)
                if not destination.is_file():
                    raise RuntimeError("Cycles returned without the requested PNG")
                if quality == "uhd-master":
                    verify_png(destination, width, height, profile["color_depth"])
                entry = {"frame": frame, "file": name, "sha256": digest(destination),
                         "elapsed_ns": time.monotonic_ns() - frame_started}
                result["frames"].append(entry)
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                print(json.dumps({"case_id": args.case_id, **entry}), flush=True)
        if digest(source_receipt) != receipt_sha or digest(source_scene) != scene_sha:
            raise RuntimeError("saved scene or receipt changed during rendering")
        if digest(worker_path) != worker_sha:
            raise RuntimeError("render worker changed during rendering")
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
    parser.add_argument("--device", choices=("CUDA", "OPTIX", "METAL"), default="OPTIX")
    parser.add_argument("--device-id")
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--frame", type=int, action="append", help="Select an absolute case frame; repeat in order")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--scale", type=int, choices=(100, 200), default=100)
    parser.add_argument("--quality", choices=QUALITY_PROFILES, default="standard")
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(arguments)
    if args.samples is not None and args.samples < 1:
        parser.error("positive sample count required")
    result = render(args)
    print(json.dumps({"status": result["status"], "frames": len(result["frames"]),
                      "receipt": str(args.output_dir.resolve() / "render.receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
