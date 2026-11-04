#!/usr/bin/env python3
"""Render the 42-second JAIN cut from verified Blender plates and complete Rust recipes."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from animate_source import (
    case_ids, digest, ffmpeg_path, inside, load_json, verify_loop, viewport,
    write_chapters, write_json,
)
from cinematic_pages import PALETTE, presentation
from code_pages import make_pages, reconstruct_source
from intelligence_wave_effect import NETWORK_ROI, apply_wave
from table_panels import load_panels

WIDTH, HEIGHT, FPS = 1920, 1080, 30
CASE_FRAMES = 126
SCROLL_FIRST, SCROLL_LAST = 7, 119


def scroll_offsets(rows: int, line_height: int, view_height: int) -> list[int]:
    """Sample a linear top-to-bottom scroll and reject any uncaptured source rows."""
    distance = max(0, rows * line_height - view_height)
    offsets = [
        round(distance * min(1., max(0., (frame - SCROLL_FIRST) / (SCROLL_LAST - SCROLL_FIRST))))
        for frame in range(CASE_FRAMES)
    ]
    if line_height > view_height or any(b - a > view_height - line_height
                                        for a, b in zip(offsets, offsets[1:])):
        raise ValueError("scroll sampling could skip a complete visual row")
    return offsets


def code_strip(source: Path, expected_sha: str, output: Path, font: Path,
               width: int, height: int, operations: list[dict]) -> tuple:
    with presentation():
        manifest = make_pages(source, output, font, font_size=15, columns=54, rows=56)
    raw = reconstruct_source(manifest)
    if manifest["source_sha256"] != expected_sha or raw != source.read_bytes():
        raise ValueError("complete Rust strip differs from the source catalog")
    rows = [row for page in manifest["pages"] for row in page["rows"]]
    line_height = manifest["line_height"]
    top = manifest["code_origin"][1]
    strip = Image.new("RGB", (width, max(height, len(rows) * line_height)), PALETTE["background"])
    y = 0
    for page in manifest["pages"]:
        path = inside(output.resolve(), page["file"])
        if digest(path) != page["sha256"]:
            raise ValueError("code raster differs from its manifest")
        page_height = len(page["rows"]) * line_height
        with Image.open(path) as raster:
            region = raster.convert("RGB").crop((1, top, raster.width - 1, top + page_height))
            strip.paste(region.resize((width, page_height), Image.Resampling.LANCZOS), (0, y))
        y += page_height

    offsets = scroll_offsets(len(rows), line_height, height)
    ends = [row["source_byte_end"] for row in rows]
    cues = []
    for operation in operations:
        for phase in ("fit", "predict"):
            start, end = operation[f"{phase}_span"]
            if not 0 <= start < end <= len(raw):
                raise ValueError("GP operation span escapes the exact Rust source")
            first, last = bisect_right(ends, start), bisect_left(ends, end)
            center = (first + last + 1) * line_height / 2
            cue_frame = min(range(SCROLL_FIRST, SCROLL_LAST + 1),
                            key=lambda frame: abs(center - offsets[frame] - height * .56))
            cues.append({"operation": operation["label"], "ordinal": operation["ordinal"],
                         "phase": phase, "bytes": [start, end], "rows": [first, last],
                         "case_frame": cue_frame})
    if not cues:
        raise ValueError("GP operation cues are required for each Rust recipe")
    return strip, rows, line_height, offsets, cues, manifest


def hud(frame, title_font, value_font, rate_font, metric, case_number, source_line, total_lines,
        cue, rect):
    draw = ImageDraw.Draw(frame)
    # These opaque regions cover the archived benchmark labels and source title only.
    draw.rectangle((674, 23, 1247, 186), fill="black")
    draw.text((960, 54), "FITTED EXPORT BENCHMARK", anchor="mm", font=title_font, fill="#CDDCEB")
    draw.text((960, 99), f"{metric['total_ms']:.1f} ms", anchor="mm", font=value_font, fill="#EDF4FF")
    draw.text((960, 140), f"{metric['equivalent_rust_tokens_per_s']:,.0f} Rust tokens/sec",
              anchor="mm", font=rate_font, fill="#D6E2F2")
    x, y, width, _ = rect
    draw.rectangle((x - 6, 23, x + width + 6, 91), fill="black")
    draw.text((x + width / 2, 56), "FULL RUST RECIPE", anchor="mm", font=title_font, fill="#E7DEF6")
    draw.rectangle((x, y - 20, x + width - 1, y - 1), fill="#1B2535")
    left = f"{case_number:02d}/10   model.rs"
    if cue:
        left = f"{cue['phase'].upper()}  ·  {cue['operation']}"
    draw.text((x + 10, y - 10), left, anchor="lm", font=title_font,
              fill="#DCB2FE" if cue else "#AFC0D3")
    draw.text((x + width - 10, y - 10), f"{source_line:,}/{total_lines:,}", anchor="rm",
              font=title_font, fill="#AFC0D3")


def render(args):
    scene_path, source_catalog = args.scene_receipt.resolve(), args.source_catalog.resolve()
    scene, catalog = load_json(scene_path), load_json(source_catalog)
    scene_hash, source_catalog_hash = digest(scene_path), digest(source_catalog)
    ids = case_ids(scene)
    if len(ids) != 10 or catalog.get("schema") != "jain.short-recipe-source/v1" or catalog.get("status") != "PASS":
        raise ValueError("ten-case scene and passing Rust recipe source catalog required")
    if [case["case_id"] for case in catalog["cases"]] != ids:
        raise ValueError("Blender and source case order differs")
    rect = viewport(scene)
    x, y, width, height = rect
    tables, table_binding = load_panels(args.table_panels, scene)
    loops = [
        verify_loop(args.loops_root, {"case_id": cid, "loop_frame_start": first,
                                    "loop_frame_end": last}, scene_hash)
        for cid, (first, last) in zip(ids, scene["case_frames"], strict=True)
    ]
    if len({receipt["scene_sha256"] for receipt, _ in loops}) != 1:
        raise ValueError("Blender case loops refer to different saved scenes")
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("use a new output directory outside the checkout")
    output.mkdir(parents=True)
    table_records = {table["case_id"]: table for table in scene["tables"]}
    cases = [{"case_id": cid, "dataset_name": table_records[cid]["dataset_name"],
              "total_rows": table_records[cid]["total_rows"], "start": i * CASE_FRAMES,
              "end": (i + 1) * CASE_FRAMES} for i, cid in enumerate(ids)]
    chapters = write_chapters({"cases": cases}, output / "chapters.ffmetadata")
    movie = output / "JAIN-intelligence-wave-42s.mp4"
    command = [
        ffmpeg_path(args.ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
        "-i", "pipe:0", "-f", "ffmetadata", "-i", chapters["file"],
        "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie),
    ]
    fonts = [ImageFont.truetype(str(args.ui_font), size) for size in (16, 48, 24)]
    previews = []
    with (output / "ffmpeg.stderr.log").open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for index, (case, source_case, (loop, loop_path)) in enumerate(
                    zip(cases, catalog["cases"], loops, strict=True), start=1):
                cid = case["case_id"]
                metric = scene["round_trip_metrics"][cid]
                if source_case["selected_program_sha256"] != metric["selected_program_sha256"]:
                    raise ValueError(f"{cid}: Rust recipe does not match the Blender selection")
                if (source_case["model_id"] != metric["model_id"]
                        or source_case["original_fitted_source_sha256"] != metric["rust_sha256"]):
                    raise ValueError(f"{cid}: saved object differs from the captured selection")
                source = inside(source_catalog.parent, source_case["source_path"])
                pages = output / "code-pages" / cid
                strip, rows, line_height, offsets, cues, manifest = code_strip(
                    source, source_case["source_sha256"], pages, args.code_font, width, height,
                    source_case["operations"],
                )
                plate_indices = [
                    round(frame * (len(loop["frames"]) - 1) / (CASE_FRAMES - 1))
                    for frame in range(CASE_FRAMES)
                ]
                case.update({"source_sha256": manifest["source_sha256"],
                             "source_bytes": manifest["source_bytes"],
                             "source_lines": manifest["source_lines"], "visual_rows": len(rows),
                             "code_manifest_sha256": digest(pages / "manifest.json"),
                             "model_id": source_case["model_id"], "cues": cues,
                             "line_height_px": line_height, "plate_indices": plate_indices,
                             "scroll_offsets_px": offsets, "loop_receipt_sha256": digest(loop_path)})
                peak = min(cues, key=lambda cue: abs(cue["case_frame"] - CASE_FRAMES / 2))["case_frame"]
                samples = {0: "start", peak: "gp-cue", CASE_FRAMES - 1: "end"} if index in (1, 7, 10) else {}
                for frame_index, (offset, loop_index) in enumerate(zip(offsets, plate_indices, strict=True)):
                    with Image.open(inside(loop_path.parent, loop["frames"][loop_index]["file"])) as raster:
                        if raster.size != (WIDTH, HEIGHT):
                            raise ValueError("Blender frame is not 1080p")
                        frame = raster.convert("RGB")
                    focus = min(cues, key=lambda cue: abs(cue["case_frame"] - frame_index))
                    gap = abs(focus["case_frame"] - frame_index)
                    visible_cues = [
                        cue for cue in cues
                        if abs(cue["case_frame"] - frame_index) <= 4
                        and cue["rows"][0] * line_height < offset + height
                        and (cue["rows"][1] + 1) * line_height > offset
                    ]
                    cue = min(visible_cues, key=lambda item: abs(item["case_frame"] - frame_index),
                              default=None) if frame_index >= SCROLL_FIRST else None
                    # The envelope follows rendered pixels. GP span timing controls only gain.
                    energy = .76 + .44 * math.exp(-.5 * (gap / 3) ** 2)
                    frame = apply_wave(frame, (frame_index % 34) / 33, energy=energy)
                    table = tables[cid]
                    frame.paste(table, (0, 0), table)
                    code = strip.crop((0, offset, width, offset + height))
                    code_draw = ImageDraw.Draw(code)
                    if frame_index < SCROLL_FIRST:
                        visible = max(line_height, round(height * (frame_index + 1) / SCROLL_FIRST))
                        code_draw.rectangle((0, visible, width, height), fill=PALETTE["background"])
                    elif cue:
                        first, last = cue["rows"]
                        top = first * line_height - offset
                        bottom = (last + 1) * line_height - offset - 1
                        if top < height and bottom >= 0:
                            code_draw.rectangle((4, max(0, top), 6, min(height - 1, bottom)), fill="#B476F8")
                    frame.paste(code, (x, y))
                    line = rows[min(len(rows) - 1, (offset + height - 1) // line_height)]["source_line"]
                    hud(frame, *fonts, metric, index, line, manifest["source_lines"], cue, rect)
                    if frame_index in samples:
                        still = output / f"{cid}-{samples[frame_index]}.png"
                        frame.save(still)
                        previews.append({"path": still.name, "sha256": digest(still),
                                         "frame": case["start"] + frame_index})
                    process.stdin.write(frame.tobytes())
                print(f"{cid}: {len(rows)} wrapped rows, {len(cues)} GP call-site cues", flush=True)
            process.stdin.close()
            if process.wait():
                raise RuntimeError("ffmpeg failed; inspect ffmpeg.stderr.log")
        except BaseException:
            process.kill()
            process.wait()
            raise
    if digest(scene_path) != scene_hash or digest(source_catalog) != source_catalog_hash:
        raise ValueError("source or scene catalog changed during composition")
    receipt = {
        "schema": "jain.intelligence-wave-encode/v1", "status": "PASS", "movie": str(movie),
        "movie_sha256": digest(movie), "frames": len(cases) * CASE_FRAMES, "fps": FPS,
        "width": WIDTH, "height": HEIGHT, "duration_seconds": len(cases) * CASE_FRAMES / FPS,
        "cases": cases, "chapters": chapters, "previews": previews, "command": command,
        "scene_receipt_sha256": scene_hash, "recipe_catalog_sha256": source_catalog_hash,
        "generator_sha256": digest(Path(__file__)),
        "wave_effect_sha256": digest(Path(__file__).with_name("intelligence_wave_effect.py")),
        "table_overlay": table_binding,
        "source_viewport": rect, "wave_roi": NETWORK_ROI,
        "metric_scope": "archived data-to-complete-fitted-Rust benchmark, separate from visible recipe",
        "wave_scope": "editorial exposure sweep of existing Blender pixels with source-span gain cues",
    }
    write_json(output / "encode.receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("scene-receipt", "source-catalog", "loops-root", "table-panels", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--code-font", type=Path, default=Path("/System/Library/Fonts/Menlo.ttc"))
    parser.add_argument("--ui-font", type=Path, default=Path("/System/Library/Fonts/Avenir Next.ttc"))
    parser.add_argument("--ffmpeg")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
