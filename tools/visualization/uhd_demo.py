#!/usr/bin/env python3
"""Compose the approved sparse Dendrite timeline over native RGB16 plates."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import closing
from itertools import groupby
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from animate_source import case_ids, digest, ffmpeg_path, inside, load_json, viewport, write_json
from cinematic_pages import PALETTE
from intelligence_wave import (CASE_FRAMES, FPS, SCROLL_FIRST, TABLE_SWITCH_FRAMES, code_strip, hud,
                               load_icon, shuffled_tables)
from intelligence_wave_effect import NETWORK_ROI, _smoothstep
from intro_film import ffmpeg_process, plan_retime, require
from table_panels import BRAND_HEIGHT, WIDTH as TABLE_WIDTH
from uhd_pixels import UHD_SIZE, blend_rgba8, png_frames, rgb8_to_rgb16


SCALE = 2
WIDTH, HEIGHT = UHD_SIZE
UHD_ROI = tuple(value * SCALE for value in NETWORK_ROI)
QUALITY = {
    "quality_profile": "uhd-master", "native_render": True, "resolution_percentage": 200,
    "width": WIDTH, "height": HEIGHT, "fps": FPS, "samples": 256, "file_format": "PNG",
    "color_mode": "RGB", "color_depth": 16, "png_compression": 35, "adaptive_sampling": True,
    "adaptive_min_samples": 32, "adaptive_threshold": .003, "denoising_use_gpu": True,
    "denoising_quality": "HIGH", "denoising_prefilter": "ACCURATE",
    "denoising_input_passes": "RGB_ALBEDO_NORMAL", "gpu_only": True, "cpu_fallback": False,
}


def wave_rgb16(frame, progress, energy):
    """Apply the same display-referred exposure wave at the native plate depth."""
    if energy == 0. or progress in (0., 1.):
        return
    left, top, right, bottom = UHD_ROI
    width, height = right - left, bottom - top
    x = np.arange(width, dtype=np.float32) + .5
    y = np.arange(height, dtype=np.float32) + .5
    depth = x / width
    feather_x = _smoothstep(np.minimum(x, width - x) / (24. * SCALE))
    feather_y = _smoothstep(np.minimum(y, height - y) / (36. * SCALE))
    center = -.08 + 1.16 * progress
    front = np.exp(-.5 * ((depth - center) / .075) ** 2)
    tail = .26 * np.exp(-.5 * ((depth - center + .10) / .12) ** 2)
    fade = _smoothstep(progress / .12) * _smoothstep((1. - progress) / .12)
    envelope = energy * fade * feather_y[:, None] * (front + tail) * feather_x
    region = frame[top:bottom, left:right]
    pixels = region.astype(np.float32) / 65535.
    peak = pixels.max(axis=2)
    strength = envelope * _smoothstep((peak - 5. / 255.) / (28. / 255.))
    cyan = np.array([.12, .92, 1.], dtype=np.float32)
    violet = np.array([.82, .27, 1.], dtype=np.float32)
    tint = cyan + _smoothstep(depth)[:, None] * (violet - cyan)
    lift = strength[:, :, None] * (.85 * pixels + .28 * peak[:, :, None] * tint)
    region[:] = np.rint((pixels + (1. - pixels) * np.minimum(lift, 1.)) * 65535.).astype("<u2")


def verified_inputs(args):
    scene, reference, plan, catalog = (load_json(path) for path in
                                      (args.scene_receipt, args.reference_receipt, args.render_plan, args.source_catalog))
    scene_sha, reference_sha, plan_sha, catalog_sha = (digest(path) for path in
                                                      (args.scene_receipt, args.reference_receipt, args.render_plan,
                                                       args.source_catalog))
    ids = case_ids(scene)
    require(len(ids) == 10 and scene["network"]["spatial_layout"] == "dendrite", "Expected ten Dendrite cases")
    require(reference["schema"] == "jain.intelligence-wave-encode/v2" and reference["status"] == "PASS"
            and (reference["width"], reference["height"], reference["fps"], reference["frames"]) == (1920, 1080, FPS, 1260)
            and reference["scene_receipt_sha256"] == scene_sha, "Reference demo does not bind this Dendrite scene")
    require(catalog["schema"] == "jain.multilingual-wave-source/v1" and catalog["status"] == "PASS"
            and reference["recipe_catalog_sha256"] == catalog_sha and catalog["trace_sha256"] == scene["trace_sha256"],
            "Source catalog differs from the approved fitted export")
    require(plan["schema"] == "jain.final-4k-render-plan/v1" and plan["status"] == "PASS"
            and (plan["width"], plan["height"], plan["fps"], plan["demo_unique_frames"]) == (WIDTH, HEIGHT, FPS, 960),
            "Wrong native UHD render plan")
    inputs = plan["inputs"]
    require(inputs["demo_scene_receipt"]["sha256"] == scene_sha
            and inputs["demo_encode_receipt"]["sha256"] == reference_sha,
            "Native render plan refers to different demo receipts")
    approved_path = Path(inputs["approved_film_receipt"]["path"])
    approved_sha = digest(approved_path)
    approved = load_json(approved_path)
    require(approved_sha == inputs["approved_film_receipt"]["sha256"] and approved["status"] == "PASS"
            and approved["demo_receipt_sha256"] == reference_sha, "Approved film receipt differs")
    selected = plan_retime(reference)["selected_source_indices"]
    require(selected == approved["retime"]["selected_source_indices"]
            == [entry["source_frame0"] for entry in plan["demo_mapping"]], "Sparse plan changes the approved retime")
    require(all([case["case_id"] for case in records] == ids for records in
                (reference["cases"], catalog["cases"], plan["demo_cases"])), "Case order differs across sources")
    require(list(viewport(scene)) == reference["source_viewport"], "Approved source viewport differs from the scene")
    binding = {"scene_receipt_sha256": scene_sha, "reference_receipt_sha256": reference_sha,
               "render_plan_sha256": plan_sha, "recipe_catalog_sha256": catalog_sha,
               "approved_film_receipt_sha256": approved_sha,
               "scene_sha256": inputs["demo_scene"]["sha256"]}
    return scene, reference, plan, catalog, selected, binding


def verify_cases(root, scene, reference, plan, binding):
    """Check sparse receipt inventory against the original case time and plate index."""
    groups = {cid: list(rows) for cid, rows in groupby(plan["demo_mapping"], key=lambda row: row["case_id"])}
    require(list(groups) == case_ids(scene), "Sparse mapping case order differs")
    jobs = []
    worker_sha = digest(Path(__file__).with_name("blender") / "render_animation.py")
    for index, (original, expected, span) in enumerate(zip(reference["cases"], plan["demo_cases"],
                                                          scene["case_frames"], strict=True)):
        cid, first, last = original["case_id"], *span
        records = groups[cid]
        require((original["start"], original["end"]) == (index * CASE_FRAMES, (index + 1) * CASE_FRAMES)
                and len(original["plate_indices"]) == CASE_FRAMES and (first, last) ==
                (expected["case_frame_start"], expected["case_frame_end"])
                and len(records) == expected["mapped_frames"] == 102
                and original["loop_receipt_sha256"] == expected["original_loop_receipt_sha256"],
                f"{cid}: original case or loop binding differs")
        for output, row in enumerate(records, start=330 + index * 102):
            local = row["source_frame0"] - original["start"]
            require(0 <= local < CASE_FRAMES, f"{cid}: mapped source frame escapes its case")
            slot = original["plate_indices"][local]
            require((row["master_frame0"], row["source_case_frame0"], row["loop_frame0"], row["scene_frame"])
                    == (output, local, slot, first + slot) and first <= row["scene_frame"] <= last,
                    f"{cid}: mapped plate disagrees with the approved playback")
        unique = sorted({row["scene_frame"] for row in records})
        require(unique == expected["frames"] and len(unique) == 96, f"{cid}: sparse plate inventory differs")
        directory = (root / cid).resolve()
        require(directory.is_relative_to(root), f"{cid}: plate directory escapes its root")
        receipt_path = inside(directory, "render.receipt.json")
        receipt = load_json(receipt_path)
        required = {**QUALITY, "schema": "jain.final-4k-case-plates/v1", "status": "PASS", "case_id": cid,
                    "scene_sha256": binding["scene_sha256"], "scene_receipt_sha256": binding["scene_receipt_sha256"],
                    "render_plan_sha256": binding["render_plan_sha256"], "worker_sha256": worker_sha,
                    "requested_frames": unique, "complete_case": False}
        require(all(receipt.get(key) == value for key, value in required.items()), f"{cid}: wrong native plate receipt")
        require([row["frame"] for row in receipt["frames"]] == unique, f"{cid}: native receipt frame order differs")
        paths = []
        for row in receipt["frames"]:
            require(row["file"] == f"frame-{row['frame']:06d}.png", f"{cid}: invalid plate filename")
            paths.append(inside(directory, row["file"]))
        jobs.append((records, receipt_path, receipt, paths))
    return jobs


def case_source(case, recipe, metric, output, args, rect):
    """Re-rasterize the same files and validate source and cue identity at 2×."""
    require(recipe["selected_program_sha256"] == metric["selected_program_sha256"]
            and recipe["model_id"] == metric["model_id"] == case["model_id"]
            and recipe["original_fitted_source_sha256"] == metric["rust_sha256"],
            f"{case['case_id']}: fitted export and recipe differ")
    files = recipe["files"]
    require(bool(files), f"{case['case_id']}: no verified source files")
    primary = files[0]["language"]
    icons = {language: load_icon(language, cuda_icon=args.cuda_icon, scale=SCALE)
             for language in sorted(({item["language"] for item in files} - {"rust"}) | {primary})}
    require(primary == case["primary_language"]
            and {language: value[1] for language, value in icons.items()} == case["language_icon_hashes"],
            f"{case['case_id']}: approved language assets differ")
    strip, rows, line_height, _, generated_cues, bindings = code_strip(
        files, args.source_catalog.parent, output, args.code_font, *rect[2:], scale=SCALE,
    )
    require(len(rows) == case["visual_rows"] and line_height == SCALE * case["line_height_px"],
            f"{case['case_id']}: native source wrapping differs")
    for index, (old, new) in enumerate(zip(case["files"], bindings, strict=True)):
        new["index"] = index
        require(all(new[key] == old[key] for key in old if key != "manifest_sha256"),
                f"{case['case_id']}: source file or logical rows differ")
    cues = case["cues"]
    require(all(all(new[key] == old[key] for key in old if key != "case_frame")
                for old, new in zip(cues, generated_cues, strict=True)), f"{case['case_id']}: GP code cue differs")
    return strip, rows, line_height, cues, bindings, icons


def composite_frame(raw, local, offset, code, table, metric, rect, fonts, overlay):
    """Composite native-sized text and tables without quantizing Blender RGB16 pixels."""
    strip, rows, line_height, cues, bindings, icons = code
    x, y, width, height = rect
    frame = raw.copy()
    gap = min((abs(cue["case_frame"] - local) for cue in cues), default=math.inf)
    energy = .76 + .44 * math.exp(-.5 * (gap / 3) ** 2)
    wave_rgb16(frame, (local % 34) / 33, energy)
    frame[BRAND_HEIGHT*SCALE:, :TABLE_WIDTH*SCALE] = table
    source = strip.crop((0, offset, width, offset + height))
    draw = ImageDraw.Draw(source)
    visible = height
    cue = None
    if local < SCROLL_FIRST:
        visible = SCALE * max(line_height // SCALE, round((height // SCALE) * (local + 1) / SCROLL_FIRST))
        draw.rectangle((0, visible, width, height), fill=PALETTE["background"])
    else:
        cue = min((item for item in cues if abs(item["case_frame"] - local) <= 4
                   and item["rows"][0] * line_height < offset + height
                   and (item["rows"][1] + 1) * line_height > offset),
                  key=lambda item: abs(item["case_frame"] - local), default=None)
        if cue:
            first, last = cue["rows"]
            top, bottom = first * line_height - offset, (last + 1) * line_height - offset - 1
            if top < height and bottom >= 0:
                draw.rectangle((4*SCALE, max(0, top), 7*SCALE-1, min(height-1, bottom)), fill="#B476F8")
    frame[y:y+height, x:x+width] = rgb8_to_rgb16(source)
    first_row = offset // line_height
    last_row = min(len(rows)-1, (offset + visible) // line_height - 1)
    file_index = rows[first_row][0]
    while rows[last_row][0] != file_index:
        last_row -= 1
    file = bindings[file_index]
    language = file["language"] if file["language"] in icons else bindings[0]["language"]
    # The centered benchmark extends below the editor's top line; the two
    # headers share this slab but only its nontransparent pixels replace RGB16.
    slabs = ((0, max(y, 187*SCALE)), (890*SCALE, HEIGHT))
    for top, bottom in slabs:
        overlay.paste((0, 0, 0, 0), (0, top, WIDTH, bottom))
    hud(overlay, *fonts, metric, file, rows[last_row][1], cue, rect, icons[language][0], language, scale=SCALE)
    for top, bottom in slabs:
        slab = overlay.crop((0, top, WIDTH, bottom))
        box = slab.getbbox()
        if box:
            left, y0, right, y1 = box
            blend_rgba8(frame[top+y0:top+y1, left:right], slab.crop(box))
    return frame


def encode_command(ffmpeg, movie):
    return [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-n", "-f", "rawvideo", "-pixel_format", "rgb48le",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "ffv1",
            "-level", "3", "-slicecrc", "1", "-slices", "16", "-threads", "8", "-pix_fmt", "gbrp16le", str(movie)]


def verify_video(ffprobe, movie, count):
    probe = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_read_frames", "-of", "json", str(movie),
    ]))
    stream, = probe["streams"]
    require((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"],
             stream["avg_frame_rate"], int(stream["nb_read_frames"])) == (WIDTH, HEIGHT, "ffv1", "gbrp16le", "30/1", count),
            "Native lossless demo stream differs from its encoding contract")
    return stream


def render(args):
    output = args.output_dir.resolve()
    require(not output.exists() and not output.is_relative_to(Path(__file__).resolve().parents[2]),
            "Use a fresh output directory outside the checkout")
    names = ("scene_receipt", "reference_receipt", "source_catalog", "render_plan", "neural_influence",
             "cuda_icon", "code_font", "ui_font")
    inputs = {name: digest(getattr(args, name)) for name in names}
    source_files = ("uhd_demo.py", "uhd_pixels.py", "intelligence_wave.py", "intelligence_wave_effect.py",
                    "table_panels.py", "scaled_drawing.py", "code_pages.py", "intro_film.py")
    generator = {name: digest(Path(__file__).with_name(name)) for name in source_files}
    scene, reference, plan, catalog, selected, binding = verified_inputs(args)
    jobs = verify_cases(args.plates.resolve(), scene, reference, plan, binding)
    tables, order_bindings, table_binding = shuffled_tables(args.table_panels, scene, args.neural_influence, scale=SCALE)
    require(table_binding == reference["table_overlay"], "Verified table cells or attribution changed from the approved demo")
    rect = tuple(SCALE * value for value in viewport(scene))
    fonts = [ImageFont.truetype(str(args.ui_font), size*SCALE) for size in (16, 48, 24)]
    overlay = Image.new("RGBA", UHD_SIZE)
    ffmpeg = ffmpeg_path(args.ffmpeg)
    output.mkdir(parents=True)
    movie = output / "JAIN-Dendrite-UHD-lossless.mkv"
    command = encode_command(ffmpeg, movie)
    cases, plate_receipts = [], []
    count = 0
    with ffmpeg_process(command, output / "ffmpeg.stderr.log", writing=True) as stream:
        for case, recipe, (records, receipt_path, native, paths) in zip(reference["cases"], catalog["cases"], jobs, strict=True):
            cid = case["case_id"]
            metric = scene["round_trip_metrics"][cid]
            require(case["table_row_orders"] == order_bindings[cid], f"{cid}: displayed row shuffles differ")
            code = case_source(case, recipe, metric, output / "code-pages" / cid, args, rect)
            offsets = [value * SCALE for value in case["scroll_offsets_px"]]
            receipt_sha = digest(receipt_path)
            cases.append({**case, "files": code[4], "line_height_px": code[2], "scroll_offsets_px": offsets,
                          "uhd_loop_receipt_sha256": receipt_sha})
            plate_receipts.append({"case_id": cid, "path": str(receipt_path), "sha256": receipt_sha,
                                   "frames": native["frames"]})
            grouped = groupby(records, key=lambda row: row["scene_frame"])
            hashes = [row["sha256"] for row in native["frames"]]
            prior_slot = None
            with closing(png_frames(paths, ffmpeg=ffmpeg, expected_sha256=hashes)) as decoded:
                for (frame, group), native_frame, raw in zip(grouped, native["frames"], decoded, strict=True):
                    require(frame == native_frame["frame"], f"{cid}: decoded native plate order differs")
                    for record in group:
                        local = record["source_case_frame0"]
                        slot = bisect_right(TABLE_SWITCH_FRAMES, local) - 1
                        if slot != prior_slot:
                            panel = tables[cid][slot].crop((0, BRAND_HEIGHT*SCALE, TABLE_WIDTH*SCALE, HEIGHT))
                            table = rgb8_to_rgb16(panel.convert("RGB"))
                            prior_slot = slot
                        frame_rgb = composite_frame(raw, local, offsets[local], code, table, metric, rect, fonts, overlay)
                        stream.write(memoryview(frame_rgb).cast("B"))
                        count += 1
            print(f"{cid}: {len(records)} composited UHD frames from {len(paths)} native plates", flush=True)
    require(count == len(selected) == 1020, "Wrong sparse demo output count")
    require(all(digest(getattr(args, name)) == sha for name, sha in inputs.items())
            and all(digest(Path(__file__).with_name(name)) == sha for name, sha in generator.items())
            and all(digest(Path(item["path"])) == item["sha256"] for item in plate_receipts),
            "UHD composition inputs changed during encoding")
    stream = verify_video(args.ffprobe, movie, count)
    receipt = {"schema": "jain.uhd-demo/v1", "status": "PASS", "movie": str(movie), "movie_sha256": digest(movie),
               "width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": reference["frames"], "encoded_frames": count,
               "logical_duration_seconds": reference["frames"] / FPS, "duration_seconds": count / FPS,
               "source_frame_indices": selected, "source_viewport": rect, "cases": cases, "native_render": True,
               "wave_roi": UHD_ROI, "table_overlay": table_binding, "plate_receipts": plate_receipts,
               "native_plate_frames": plan["demo_unique_frames"], "inputs_sha256": inputs, "generator_sha256": generator,
               "codec_probe": stream, "encoder_command": command, **binding}
    plan_retime(receipt)
    write_json(output / "encode.receipt.json", receipt)
    print(movie, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene-receipt", "reference-receipt", "render-plan", "source-catalog", "table-panels",
                 "neural-influence", "cuda-icon", "plates", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--code-font", type=Path, default=Path("/System/Library/Fonts/Menlo.ttc"))
    parser.add_argument("--ui-font", type=Path, default=Path("/System/Library/Fonts/Avenir Next.ttc"))
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    args.source_catalog = args.source_catalog.resolve()
    render(args)


if __name__ == "__main__":
    main()
