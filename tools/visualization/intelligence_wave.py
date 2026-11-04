#!/usr/bin/env python3
"""Render the 42-second JAIN cut from verified Blender plates and complete source files."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

from PIL import Image, ImageDraw, ImageFont

from animate_source import (
    case_ids, digest, ffmpeg_path, inside, load_json, verify_loop, viewport,
    write_chapters, write_json,
)
from cinematic_pages import PALETTE, presentation
from code_pages import make_pages, reconstruct_source
from intelligence_wave_effect import NETWORK_ROI, apply_wave
from table_panels import (
    BODY_FONTS, HEADING_FONTS, ROWS, draw_panel, font_path, importance_percentiles, load_panels,
)

WIDTH, HEIGHT, FPS = 1920, 1080, 30
CASE_FRAMES = 126
SCROLL_FIRST, SCROLL_LAST = 7, 119
# Five times the original three table holds. Round each boundary to the
# nearest video frame so 8- and 9-frame holds average 0.28 seconds.
TABLE_HOLDS = 3 * 5
TABLE_SWITCH_FRAMES = tuple(round(hold * CASE_FRAMES / TABLE_HOLDS) for hold in range(TABLE_HOLDS))
LANGUAGES = {
    "rust": "Rust", "python": "Python", "cpp": "C++", "javascript": "JavaScript",
    "typescript": "TypeScript", "go": "Go", "cuda": "CUDA C++", "c": "C",
}
ICON_ROOT = (Path(__file__).resolve().parents[2]
             / "docs/plans/blender-fused-first-render/language-icons")


def display_rate(rate: float) -> str:
    thousands = round(rate / 1_000)
    if thousands >= 1_000:
        value = f"{rate / 1_000_000:.1f}".removesuffix(".0") + "M"
    elif rate >= 1_000:
        value = f"{thousands}k"
    else:
        value = f"{rate:.0f}"
    return f"{value} tokens/sec"


def table_orders(case_id, row_ids):
    """Keep the first view in capture order, then pick reproducible shuffles."""
    if len(row_ids) != ROWS or len(set(row_ids)) != ROWS:
        raise ValueError("shuffle requires 17 distinct verified rows")
    rng = random.Random(f"jain.wave.table-shuffle/v1:{case_id}")
    first = list(range(len(row_ids)))
    orders = [first]
    while len(orders) < len(TABLE_SWITCH_FRAMES):
        order = rng.sample(first, len(first))
        if order not in orders:
            orders.append(order)
    return orders


def shuffled_tables(root, scene):
    _, binding = load_panels(root, scene)
    catalog_bytes = (root / "catalog.receipt.json").read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != binding["catalog_sha256"]:
        raise ValueError("table catalog changed after panel verification")
    catalog = json.loads(catalog_bytes)
    influence_bytes = Path(catalog["influence"]["path"]).read_bytes()
    if hashlib.sha256(influence_bytes).hexdigest() != binding["influence_sha256"]:
        raise ValueError("sensitivity records changed after panel verification")
    records = json.loads(influence_bytes)["cases"]
    body = font_path(None, BODY_FONTS)
    heading = font_path(None, HEADING_FONTS)
    if digest(body) != catalog["body_font_sha256"] or digest(heading) != catalog["heading_font_sha256"]:
        raise ValueError("shuffled panels require the original table fonts")
    variants, order_bindings = {}, {}
    for case in binding["cases"]:
        cid = case["case_id"]
        record = records[cid]
        rows, scores, ids = record["raw_table"], record["scores"], record["row_ids"]
        percentiles = importance_percentiles(scores)
        header = [*record["feature_names"], record["target_name"]]
        variants[cid] = []
        order_bindings[cid] = []
        for start, order in zip(TABLE_SWITCH_FRAMES, table_orders(cid, ids), strict=True):
            row_ids = [ids[i] for i in order]
            order_bindings[cid].append({"case_frame": start, "row_ids": row_ids})
            panel, _ = draw_panel(case["dataset_name"], case["total_rows"], header, row_ids,
                                  [rows[i] for i in order], [scores[i] for i in order], body, heading,
                                  percentiles=[percentiles[i] for i in order])
            variants[cid].append(panel)
    return variants, order_bindings, binding


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


def code_strip(files, source_root, output, font, width, height):
    """Join verified, separately highlighted files in one clipped vertical scroll."""
    rows, cues, bindings, rasters = [], [], [], []
    heading = ImageFont.truetype(str(font), 13)
    for index, item in enumerate(files):
        language = item["language"]
        if language not in LANGUAGES:
            raise ValueError(f"unknown displayed language {language}")
        source = inside(source_root, item["source_path"])
        pages = output / str(index)
        with presentation(language):
            manifest = make_pages(source, pages, font, font_size=15, columns=54, rows=56,
                                  language=language)
        raw = reconstruct_source(manifest)
        if manifest["source_sha256"] != item["source_sha256"] or raw != source.read_bytes():
            raise ValueError("complete code strip differs from the source catalog")
        line_height = manifest["line_height"]
        top = manifest["code_origin"][1]
        if index:
            divider = Image.new("RGB", (width, 2 * line_height), "#1B2535")
            ImageDraw.Draw(divider).text((14, line_height),
                f"{source.name}  ·  {LANGUAGES[language]}", font=heading,
                fill="#B4C8DE", anchor="lm")
            rasters.append(divider)
            rows.extend([(index, 1)] * 2)
        first_row = len(rows)
        file_rows = [row for page in manifest["pages"] for row in page["rows"]]
        rows.extend((index, row["source_line"]) for row in file_rows)
        bindings.append({"filename": source.name, "language": language,
                         "source_path": item["source_path"], "source_sha256": manifest["source_sha256"],
                         "source_bytes": len(raw), "source_lines": manifest["source_lines"],
                         "visual_start": first_row, "visual_rows": len(file_rows),
                         "manifest_sha256": digest(pages / "manifest.json")})
        for page in manifest["pages"]:
            path = inside(pages.resolve(), page["file"])
            if digest(path) != page["sha256"]:
                raise ValueError("code raster differs from its manifest")
            page_height = len(page["rows"]) * line_height
            with Image.open(path) as raster:
                region = raster.convert("RGB").crop((1, top, raster.width - 1, top + page_height))
                rasters.append(region.resize((width, page_height), Image.Resampling.LANCZOS))
        ends = [row["source_byte_end"] for row in file_rows]
        for operation in item["operations"]:
            spans = operation["spans"]
            phases = ("recipe",) if len(spans) == 1 else ("fit", "predict")
            for phase, (start, end) in zip(phases, spans, strict=True):
                if not 0 <= start < end <= len(raw):
                    raise ValueError("GP operation span escapes the exact source file")
                first = first_row + bisect_right(ends, start)
                last = first_row + bisect_left(ends, end)
                cues.append({"operation": operation["label"], "ordinal": operation["ordinal"],
                             "phase": phase, "bytes": [start, end], "rows": [first, last],
                             "file_index": index})
    strip = Image.new("RGB", (width, max(height, len(rows) * line_height)), PALETTE["background"])
    y = 0
    for raster in rasters:
        strip.paste(raster, (0, y))
        y += raster.height
    offsets = scroll_offsets(len(rows), line_height, height)
    for cue in cues:
        first, last = cue["rows"]
        center = (first + last + 1) * line_height / 2
        cue["case_frame"] = min(range(SCROLL_FIRST, SCROLL_LAST + 1),
                                key=lambda frame: abs(center - offsets[frame] - height * .56))
    return strip, rows, line_height, offsets, cues, bindings


def hud(frame, title_font, value_font, rate_font, metric, source_file, source_line,
        cue, rect, icon, primary_language, binding_label):
    draw = ImageDraw.Draw(frame)
    # These opaque regions cover the archived benchmark labels and source title only.
    draw.rectangle((674, 23, 1247, 186), fill="black")
    draw.text((960, 54), "FITTED EXPORT BENCHMARK", anchor="mm", font=title_font, fill="#CDDCEB")
    draw.text((960, 99), f"{metric['total_ms']:.1f} ms", anchor="mm", font=value_font, fill="#EDF4FF")
    draw.text((960, 140), display_rate(metric["equivalent_rust_tokens_per_s"]),
              anchor="mm", font=rate_font, fill="#D6E2F2")
    x, y, width, _ = rect
    draw.rectangle((x - 6, 23, x + width + 6, 91), fill="black")
    language = LANGUAGES[source_file["language"]]
    draw.text((x + width / 2, 56), f"{language.upper()} SOURCE",
              anchor="mm", font=title_font, fill="#E7DEF6")
    draw.rectangle((x, y - 20, x + width - 1, y - 1), fill="#1B2535")
    draw.text((x + 10, y - 10), source_file["filename"], anchor="lm", font=title_font,
              fill="#AFC0D3")
    if cue and cue["file_index"] == source_file["index"]:
        draw.text((x + width / 2, y - 10), f"{cue['phase'].upper()} · {cue['operation']}",
                  anchor="mm", font=title_font, fill="#DCB2FE")
    draw.text((x + width - 10, y - 10), f"Ln {source_line:,}", anchor="rm",
              font=title_font, fill="#AFC0D3")
    frame.paste(icon, (960 - icon.width // 2, 895), icon)
    draw.text((960, 996), LANGUAGES[primary_language], anchor="mm", font=rate_font, fill="#EDF4FF")
    if binding_label:
        draw.text((960, 1021), binding_label, anchor="mm", font=title_font, fill="#AFC0D3")


def load_icon(language):
    path = ICON_ROOT / f"{language}.png"
    with Image.open(path) as raster:
        icon = raster.convert("RGBA")
    icon.thumbnail((86, 86), Image.Resampling.LANCZOS)
    return icon, digest(path)


def render(args):
    scene_path, source_catalog = args.scene_receipt.resolve(), args.source_catalog.resolve()
    scene, catalog = load_json(scene_path), load_json(source_catalog)
    scene_hash, source_catalog_hash = digest(scene_path), digest(source_catalog)
    ids = case_ids(scene)
    if (len(ids) != 10 or catalog.get("schema") != "jain.multilingual-wave-source/v1"
            or catalog.get("status") != "PASS"):
        raise ValueError("ten-case scene and passing multilingual source catalog required")
    if [case["case_id"] for case in catalog["cases"]] != ids:
        raise ValueError("Blender and source case order differs")
    rect = viewport(scene)
    x, y, width, height = rect
    tables, table_order_bindings, table_binding = shuffled_tables(args.table_panels, scene)
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
                    raise ValueError(f"{cid}: code recipe does not match the Blender selection")
                if (source_case["model_id"] != metric["model_id"]
                        or source_case["original_fitted_source_sha256"] != metric["rust_sha256"]):
                    raise ValueError(f"{cid}: saved object differs from the captured selection")
                files = source_case["files"]
                if not files:
                    raise ValueError(f"{cid}: missing source files")
                primary_language = files[0]["language"]
                icon, icon_sha = load_icon(primary_language)
                binding_label = ("Rust / Wasm binding" if primary_language in {"go", "typescript"}
                                 else "Rust core" if primary_language in {"c", "cuda"} else "")
                pages = output / "code-pages" / cid
                strip, rows, line_height, offsets, cues, file_bindings = code_strip(
                    files, source_catalog.parent, pages, args.code_font, width, height,
                )
                for file_index, file_binding in enumerate(file_bindings):
                    file_binding["index"] = file_index
                plate_indices = [
                    round(frame * (len(loop["frames"]) - 1) / (CASE_FRAMES - 1))
                    for frame in range(CASE_FRAMES)
                ]
                case.update({"files": file_bindings, "visual_rows": len(rows),
                             "primary_language": primary_language, "language_icon_sha256": icon_sha,
                             "model_id": source_case["model_id"], "cues": cues,
                             "line_height_px": line_height, "plate_indices": plate_indices,
                             "table_row_orders": table_order_bindings[cid],
                             "scroll_offsets_px": offsets, "loop_receipt_sha256": digest(loop_path)})
                peak = min((cue["case_frame"] for cue in cues),
                           key=lambda frame: abs(frame - CASE_FRAMES / 2), default=CASE_FRAMES // 2)
                samples = {SCROLL_FIRST: "language"}
                if index in (1, 7, 10):
                    samples.update({0: "start", peak: "gp-cue", CASE_FRAMES - 1: "end"})
                if index == 1:
                    samples.update(dict(zip(TABLE_SWITCH_FRAMES[1:3], ("shuffle-1", "shuffle-2"), strict=True)))
                for frame_index, (offset, loop_index) in enumerate(zip(offsets, plate_indices, strict=True)):
                    with Image.open(inside(loop_path.parent, loop["frames"][loop_index]["file"])) as raster:
                        if raster.size != (WIDTH, HEIGHT):
                            raise ValueError("Blender frame is not 1080p")
                        frame = raster.convert("RGB")
                    gap = min((abs(cue["case_frame"] - frame_index) for cue in cues), default=math.inf)
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
                    table = tables[cid][bisect_right(TABLE_SWITCH_FRAMES, frame_index) - 1]
                    frame.paste(table, (0, 0), table)
                    code = strip.crop((0, offset, width, offset + height))
                    code_draw = ImageDraw.Draw(code)
                    visible = height
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
                    first_row = offset // line_height
                    last_row = min(len(rows) - 1, (offset + visible) // line_height - 1)
                    file_index = rows[first_row][0]
                    while rows[last_row][0] != file_index:
                        last_row -= 1
                    line = rows[last_row][1]
                    hud(frame, *fonts, metric, file_bindings[file_index], line, cue, rect,
                        icon, primary_language, binding_label)
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
        "schema": "jain.intelligence-wave-encode/v2", "status": "PASS", "movie": str(movie),
        "movie_sha256": digest(movie), "frames": len(cases) * CASE_FRAMES, "fps": FPS,
        "width": WIDTH, "height": HEIGHT, "duration_seconds": len(cases) * CASE_FRAMES / FPS,
        "cases": cases, "chapters": chapters, "previews": previews, "command": command,
        "scene_receipt_sha256": scene_hash, "recipe_catalog_sha256": source_catalog_hash,
        "generator_sha256": digest(Path(__file__)),
        "wave_effect_sha256": digest(Path(__file__).with_name("intelligence_wave_effect.py")),
        "table_overlay": table_binding,
        "table_shuffle_scope": "Seeded display permutations of the same 17 verified FIT rows every 0.28 seconds, rounded to video frames",
        "table_percentile_scope": "Per-case percentile across 68 measured feature sensitivities, averaging ties and anchoring zeros to low; target excluded",
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
