#!/usr/bin/env python3
"""Compose exact Rust page typing over short, pre-rendered neural loops.

Typing, page dwell and loop repetition are presentation timing. Every original
source byte is checked through the existing page reconstructor; this process
does not synthesize, shorten or reformat the emitted Rust. Generated movies and
receipts must stay outside the checkout.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from code_pages import _cells, reconstruct_source

SCHEMA = "jain.source-animation-timeline/v1"
CASE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def inside(root: Path, reference: str) -> Path:
    relative = Path(reference)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"absent or escaping artifact: {reference}")
    return path


def page_kind(page: dict) -> str:
    """Pacing follows actual source text; it never changes source coverage."""
    text = "".join(row["text"] for row in page["rows"])
    if re.search(r"\bfn\s+fit\s*\(", text) or "Fit every stage" in text:
        return "training"
    if re.search(r"\bfn\s+(?:predict|predict_numeric)\s*\(", text):
        return "prediction"
    visible = re.sub(r"\s", "", text)
    numeric = sum(char.isdigit() or char in ".+-eE," for char in visible)
    return "dense_state" if len(visible) > 300 and numeric / len(visible) > .55 else "source"


def case_ids(scene: dict) -> list[str]:
    ids = scene.get("case_ids")
    if ids is None:
        ids = [case["case_id"] for case in scene.get("tables", [])]
    if not ids and isinstance(scene.get("round_trip_metrics"), dict):
        ids = list(scene["round_trip_metrics"])
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 16
            or len(set(ids)) != len(ids) or any(not CASE_ID.fullmatch(cid) for cid in ids)):
        raise ValueError("scene must identify 1..16 distinct source cases")
    return ids


def viewport(scene: dict) -> tuple[int, int, int, int]:
    value = scene.get("source_viewport")
    if isinstance(value, dict):
        value = [value.get("x", value.get("left")), value.get("y", value.get("top")),
                 value.get("width"), value.get("height")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("scene.source_viewport must provide x,y,width,height in output pixels")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("finite pixel viewport required")
    x, y, width, height = [round(v) for v in value]
    if x < 0 or y < 0 or min(width, height) <= 0 or x + width > 1920 or y + height > 1080:
        raise ValueError("source viewport escapes 1920x1080 output")
    return x, y, width, height


def make_timeline(scene_path: Path, pages_root: Path, *, fps: int = 30,
                  intro: float = 3., end: float = 2., normal: float = .7,
                  dense: float = .3, semantic: float = 6., first_typing: float = .6,
                  scroll: float = 1.2) -> dict:
    scene_path, pages_root = scene_path.resolve(), pages_root.resolve()
    scene = load_json(scene_path)
    ids = case_ids(scene)
    spans = scene["case_frames"]
    if len(spans) != len(ids) or any(len(span) != 2 or span[1] < span[0] for span in spans):
        raise ValueError("scene case frame spans differ from case inventory")
    if fps != 30 or min(intro, end, first_typing, scroll) <= 0 or min(normal, dense, semantic) < .3:
        raise ValueError("30 fps, positive timings and full-page holds of at least0.3s required")
    catalog_path = pages_root / "catalog.receipt.json"
    catalog = load_json(catalog_path)
    if catalog.get("status") != "PASS":
        raise ValueError("passing complete-source page catalog required")
    catalog_cases = {case["case_id"]: case for case in catalog["cases"]}
    bindings = {case["case_id"]: case for case in scene["whole_source_pages"]}
    if set(ids) != set(catalog_cases) or set(ids) != set(bindings):
        raise ValueError("scene/page catalog case identity differs")
    cases, current = [], 0
    for cid, span in zip(ids, spans):
        manifest_path = inside(pages_root, f"{cid}/manifest.json")
        manifest = load_json(manifest_path)
        raw = reconstruct_source(manifest)
        manifest_sha = digest(manifest_path)
        if (manifest_sha != catalog_cases[cid]["manifest_sha256"]
                or manifest_sha != bindings[cid]["manifest_sha256"]
                or manifest["source_sha256"] != catalog_cases[cid]["source_sha256"]
                or manifest["source_sha256"] != bindings[cid]["rust_sha256"]
                or manifest["page_count"] != bindings[cid]["page_count"]):
            raise ValueError(f"{cid}: complete-source binding differs")
        begin = current
        segments = [{"kind": "intro", "start": current, "end": current + round(intro * fps)}]
        current = segments[-1]["end"]
        for page in manifest["pages"]:
            page_path = inside(manifest_path.parent, page["file"])
            if digest(page_path) != page["sha256"]:
                raise ValueError(f"{cid}: page raster bytes changed")
            kind = page_kind(page)
            hold = round((semantic if kind in {"training", "prediction"}
                          else dense if kind == "dense_state" else normal) * fps)
            typing = max(1, round(first_typing * fps)) if page["page"] == 1 else 0
            if page["page"] > 1:
                frames = max(2, round(scroll * fps))
                segments.append({"kind": "scroll", "page": page["page"],
                                 "from_page": page["page"] - 1,
                                 "start": current, "end": current + frames,
                                 "scroll_offset_start_pages": page["page"] - 2,
                                 "scroll_offset_end_pages": page["page"] - 1,
                                 "interpolation": "monotone smoothstep; no page reset",
                                 "source_byte_start": manifest["pages"][page["page"] - 2]["source_byte_start"],
                                 "source_byte_end": page["source_byte_end"]})
                current += frames
            frames = typing + hold
            segments.append({"kind": "page", "pacing": kind, "page": page["page"],
                             "start": current, "end": current + frames,
                             "typing_frames": typing, "first_full_frame": current + typing,
                             "full_hold_frames": hold, "scroll_offset_pages": page["page"] - 1,
                             "page_sha256": page["sha256"],
                             "source_byte_start": page["source_byte_start"],
                             "source_byte_end": page["source_byte_end"]})
            current += frames
        loop_frames = span[1] - span[0] + 1
        end_frames = round(end * fps)
        padding = -(current - begin + end_frames) % loop_frames
        segments.append({"kind": "end", "start": current, "end": current + end_frames + padding,
                         "page": manifest["page_count"], "loop_phase_padding_frames": padding})
        current = segments[-1]["end"]
        cases.append({"case_id": cid, "start": begin, "end": current,
                      "loop_frame_start": span[0], "loop_frame_end": span[1],
                      "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                      "source_sha256": manifest["source_sha256"], "source_bytes": len(raw),
                      "first_source_byte": manifest["pages"][0]["source_byte_start"],
                      "last_source_byte": manifest["pages"][-1]["source_byte_end"],
                      "page_traversal": [page["page"] for page in manifest["pages"]],
                      "source_scroll_monotone": True,
                      "page_count": manifest["page_count"], "loop_phase_padding_frames": padding,
                      "loop_cycles": (current - begin) // loop_frames, "segments": segments})
    return {"schema": SCHEMA, "scene_receipt_path": str(scene_path),
            "scene_receipt_sha256": digest(scene_path), "code_catalog_sha256": digest(catalog_path),
            "fps": fps, "width": 1920, "height": 1080, "source_viewport": viewport(scene),
            "frames": current, "duration_seconds": current / fps,
            "page_count": sum(c["page_count"] for c in cases), "cases": cases,
            "source_coverage": "byte0 through exact file end, ordered continuous vertical scroll; every page fully visible at least0.3s",
            "timing_scope": "editorial initial typing, continuous source scrolling, semantic holds and repeating recorded neural loops",
            "pacing_seconds": {"intro": intro, "end": end, "source": normal, "dense_state": dense,
                               "training_prediction_full_hold": semantic,
                               "first_page_typing": first_typing, "scroll_transition": scroll}}


def font_cell_width(manifest: dict, override: Path | None = None) -> float:
    from PIL import ImageFont
    path = Path(override or manifest["font_path"])
    if not path.is_file() or digest(path) != manifest["font_sha256"]:
        raise ValueError("exact page font unavailable or changed; provide the original font with --font")
    font = ImageFont.truetype(str(path), manifest["font_size"])
    cell = font.getlength("M")
    if abs(cell - font.getlength("i")) > .05:
        raise ValueError("page font is not monospace")
    return cell


def reveal_page(image, manifest: dict, page: dict, progress: float, *, cursor: bool = True,
                cell_width: float | None = None):
    """Mask only unrevealed raster pixels; completed frames equal the PNG exactly."""
    from PIL import Image, ImageDraw
    if progress >= 1:
        return image.copy()
    rows = page["rows"]
    characters = sum(len(row["text"]) for row in rows)
    remaining = int(max(0., progress) * characters)
    left, top = manifest["code_origin"]
    line_height = manifest["line_height"]
    cell = cell_width if cell_width is not None else font_cell_width(manifest)
    result = Image.new("RGB", image.size, manifest["palette"]["background"])
    result.paste(image.crop((0, 0, image.width, top)), (0, 0))
    draw = ImageDraw.Draw(result)
    for index, row in enumerate(rows):
        y = top + index * line_height
        count = len(row["text"])
        if remaining >= count:
            result.paste(image.crop((0, y, image.width, y + line_height)), (0, y))
            remaining -= count
            continue
        visible = row["text"][:remaining].rstrip("\r\n")
        logical = row["source_column_start"]
        cells = 0
        for char in visible:
            advance = _cells(char, logical)
            cells += advance
            logical += advance
        edge = min(image.width - 1, round(left + cells * cell))
        if remaining:
            result.paste(image.crop((0, y, edge, min(image.height, y + line_height))), (0, y))
        if cursor:
            draw.rectangle((edge, y + 2, edge + 2, min(image.height - 1, y + line_height - 3)), fill="#76C9E8")
        break
    return result


def compose_frame(base, source_image, manifest: dict, page: dict,
                  rect: tuple[int, int, int, int], progress: float = 1.,
                  cell_width: float | None = None):
    """Same page placement/reveal for movie frames and early review posters."""
    from PIL import Image
    if base.size != (1920, 1080):
        raise ValueError("base frame must be1920x1080")
    x, y, width, height = rect
    source = source_image if progress >= 1 else reveal_page(
        source_image, manifest, page, progress, cell_width=cell_width)
    result = base.copy()
    result.paste(source.resize((width, height), Image.Resampling.LANCZOS), (x, y))
    return result


def scroll_pages(previous, following, progress: float, *, cursor: bool = True):
    """Move two consecutive exact page rasters through one clipped viewport.

    Both arguments are already scaled to viewport size. Smoothstep is monotone
    and has zero velocity at the two full-page reading positions. The cursor is
    presentation chrome; it never substitutes or generates a source character.
    """
    from PIL import Image, ImageDraw
    if previous.size != following.size:
        raise ValueError("consecutive scroll rasters must have identical dimensions")
    progress = max(0., min(1., progress))
    if progress == 0:
        return previous.copy()
    if progress == 1:
        return following.copy()
    width, height = previous.size
    eased = progress * progress * (3 - 2 * progress)
    offset = round(eased * height)
    result = Image.new("RGB", (width, height))
    result.paste(previous.crop((0, offset, width, height)), (0, 0))
    result.paste(following.crop((0, 0, width, offset)), (0, height - offset))
    if cursor:
        ImageDraw.Draw(result).rectangle((max(1, round(width * .055)), height - 13,
                                         max(2, round(width * .055)) + 1, height - 4), fill="#76C9E8")
    return result


def overlay_source_page(loop_png: Path, manifest_path: Path, page_index: int,
                        source_viewport, output_png: Path) -> dict:
    """Early full-page poster from any actual loop PNG; page_index is one-based.

    This binds the supplied image bytes, without claiming a full loop or GPU
    execution receipt. It is also suitable for a short renderer pilot.
    """
    from PIL import Image
    loop_png, manifest_path, output_png = map(lambda value: Path(value).resolve(),
                                             (loop_png, manifest_path, output_png))
    sidecar = output_png.with_suffix(".receipt.json")
    if (output_png.exists() or sidecar.exists()
            or output_png.is_relative_to(Path(__file__).resolve().parents[2])):
        raise ValueError("poster must be new and outside the checkout")
    manifest = load_json(manifest_path)
    raw = reconstruct_source(manifest)
    if not 1 <= page_index <= manifest["page_count"]:
        raise ValueError("one-based page index escapes exact source")
    page = manifest["pages"][page_index - 1]
    path = inside(manifest_path.parent, page["file"])
    if digest(path) != page["sha256"]:
        raise ValueError("poster page bytes differ from manifest")
    rect = viewport({"source_viewport": source_viewport})
    with Image.open(loop_png) as base, Image.open(path) as source:
        if source.size != (manifest["width"], manifest["height"]):
            raise ValueError("poster page dimensions differ from manifest")
        result = compose_frame(base.convert("RGB"), source.convert("RGB"), manifest, page, rect)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    with output_png.open("xb") as stream:
        result.save(stream, format="PNG")
    receipt = {"schema": "jain.source-animation-poster/v1", "status": "PASS",
               "base_png_sha256": digest(loop_png), "manifest_sha256": digest(manifest_path),
               "source_sha256": manifest["source_sha256"], "source_bytes_reconstructed": len(raw),
               "page": page_index, "page_count": manifest["page_count"], "page_sha256": page["sha256"],
               "source_byte_start": page["source_byte_start"], "source_byte_end": page["source_byte_end"],
               "source_viewport": rect, "output_png_sha256": digest(output_png),
               "scope": "one fully visible exact source page over supplied frame; full GPU-loop custody not asserted"}
    write_json(sidecar, receipt)
    return receipt


def ffmpeg_path(explicit: str | None) -> str:
    if explicit:
        path = shutil.which(explicit) or (str(Path(explicit).resolve()) if Path(explicit).is_file() else None)
    else:
        path = shutil.which("ffmpeg")
        if not path:
            try:
                import imageio_ffmpeg
                path = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                path = None
    if not path:
        raise FileNotFoundError("ffmpeg unavailable; provide --ffmpeg or an existing imageio_ffmpeg runtime")
    return path


def verify_loop(root: Path, case: dict, scene_sha: str) -> tuple[dict, Path]:
    directory = root / case["case_id"]
    receipt_path = directory / "render.receipt.json"
    receipt = load_json(receipt_path)
    if (receipt.get("schema") != "jain.gpu-scene-loop/v1" or receipt.get("status") != "PASS"
            or receipt.get("case_id") != case["case_id"] or receipt.get("complete_case") is not True
            or receipt.get("scene_receipt_sha256") != scene_sha or receipt.get("gpu_only") is not True
            or receipt.get("width") != 1920 or receipt.get("height") != 1080 or receipt.get("fps") != 30):
        raise ValueError(f"{case['case_id']}: passing matching GPU-only 1080p loop required")
    frames = receipt["frames"]
    expected = list(range(case["loop_frame_start"], case["loop_frame_end"] + 1))
    if [frame["frame"] for frame in frames] != expected:
        raise ValueError("loop frames are absent or reordered")
    for frame in frames:
        if digest(inside(directory, frame["file"])) != frame["sha256"]:
            raise ValueError("loop PNG differs from render receipt")
    return receipt, receipt_path


def encode(timeline: dict, loops_root: Path, output_dir: Path, *, ffmpeg: str | None = None,
           crf: int = 18, preset: str = "medium", cache_frames: int = 120,
           font: Path | None = None) -> dict:
    from PIL import Image
    executable = ffmpeg_path(ffmpeg)
    loops = []
    for case in timeline["cases"]:
        receipt, path = verify_loop(loops_root, case, timeline["scene_receipt_sha256"])
        loops.append((receipt, path))
    if len({receipt["scene_sha256"] for receipt, _ in loops}) != 1:
        raise ValueError("case loops were rendered from different saved scenes")
    movie = output_dir / "jain-1080p30.mp4"
    command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", "30",
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(movie)]
    log_path = output_dir / "ffmpeg.stderr.log"
    rendered, loop_bindings = 0, []
    x, y, width, height = timeline["source_viewport"]
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for case, (loop, loop_path) in zip(timeline["cases"], loops):
                manifest = load_json(Path(case["manifest_path"]))
                cell = font_cell_width(manifest, font)
                if digest(Path(case["manifest_path"])) != case["manifest_sha256"]:
                    raise ValueError("page manifest changed after timeline creation")
                cache = OrderedDict()
                page_cache = {}
                loop_bindings.append({"case_id": case["case_id"], "receipt_sha256": digest(loop_path),
                                      "scene_sha256": loop["scene_sha256"], "frames": len(loop["frames"])})
                for segment in case["segments"]:
                    page_index = segment.get("page")
                    page = manifest["pages"][page_index - 1] if page_index else None
                    source_image = None
                    if page:
                        if page_index not in page_cache:
                            path = inside(Path(case["manifest_path"]).parent, page["file"])
                            if digest(path) != page["sha256"]:
                                raise ValueError("source raster changed during composition")
                            with Image.open(path) as handle:
                                page_cache[page_index] = handle.convert("RGB")
                        source_image = page_cache[page_index]
                    complete = source_image.resize((width, height), Image.Resampling.LANCZOS) if source_image else None
                    previous = None
                    if segment["kind"] == "scroll":
                        prior_index = segment["from_page"]
                        if prior_index not in page_cache:
                            raise ValueError("scroll must follow its already displayed previous page")
                        previous = page_cache[prior_index].resize((width, height), Image.Resampling.LANCZOS)
                    for frame_number in range(segment["start"], segment["end"]):
                        relative = (frame_number - case["start"]) % len(loop["frames"])
                        if relative not in cache:
                            path = inside(loop_path.parent, loop["frames"][relative]["file"])
                            with Image.open(path) as handle:
                                if handle.size != (1920, 1080):
                                    raise ValueError("render PNG dimensions differ from1080p receipt")
                                cache[relative] = handle.convert("RGB")
                            while len(cache) > cache_frames:
                                cache.popitem(last=False)
                        cache.move_to_end(relative)
                        frame = cache[relative]
                        if source_image:
                            elapsed = frame_number - segment["start"]
                            if segment["kind"] == "scroll":
                                progress = elapsed / (segment["end"] - segment["start"] - 1)
                                frame = frame.copy()
                                frame.paste(scroll_pages(previous, complete, progress), (x, y))
                            elif segment.get("typing_frames", 0) and elapsed < segment["typing_frames"]:
                                frame = compose_frame(frame, source_image, manifest, page,
                                                      (x, y, width, height),
                                                      elapsed / segment["typing_frames"], cell)
                            else:
                                frame = frame.copy()
                                frame.paste(complete, (x, y))
                        process.stdin.write(frame.tobytes())
                        rendered += 1
                    print(json.dumps({"case_id": case["case_id"], "segment": segment["kind"],
                                      "page": page_index, "frames_written": rendered}), flush=True)
                del cache, page_cache
            process.stdin.close()
            code = process.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited {code}; see {log_path}")
        except BaseException:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    if rendered != timeline["frames"] or not movie.is_file() or movie.stat().st_size == 0:
        raise ValueError("encoded frame inventory differs from complete-source timeline")
    return {"schema": "jain.source-animation-encode/v1", "status": "PASS",
            "generator_sha256": digest(Path(__file__)), "timeline_sha256": digest(output_dir / "timeline.json"),
            "movie": str(movie), "movie_sha256": digest(movie), "frames": rendered,
            "fps": 30, "width": 1920, "height": 1080, "duration_seconds": rendered / 30,
            "source_bytes": sum(case["source_bytes"] for case in timeline["cases"]),
            "page_count": timeline["page_count"], "loops": loop_bindings,
            "typing_scope": "initial exact raster typing then monotone continuous full-file scroll; editorial cursor/timing, no generated text",
            "command": command}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-receipt", type=Path, required=True)
    parser.add_argument("--code-pages", type=Path, required=True)
    parser.add_argument("--loops-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--intro-seconds", type=float, default=3.)
    parser.add_argument("--end-seconds", type=float, default=2.)
    parser.add_argument("--page-seconds", type=float, default=.7, help="fully visible ordinary-page hold")
    parser.add_argument("--dense-seconds", type=float, default=.3, help="fully visible dense-state-page hold")
    parser.add_argument("--semantic-seconds", type=float, default=6., help="full training/prediction page hold after typing")
    parser.add_argument("--first-typing-seconds", type=float, default=.6)
    parser.add_argument("--scroll-seconds", type=float, default=1.2)
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--font", type=Path, help="relocated exact original page font; SHA must match manifest")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"))
    parser.add_argument("--cache-frames", type=int, default=120)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or output.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("use a fresh output directory outside the checkout")
    if not 1 <= args.cache_frames <= 240 or not 0 <= args.crf <= 40 or args.encode and not args.loops_root:
        parser.error("cache1..240, CRF0..40 and --loops-root with --encode required")
    timeline = make_timeline(args.scene_receipt, args.code_pages, fps=args.fps,
                             intro=args.intro_seconds, end=args.end_seconds, normal=args.page_seconds,
                             dense=args.dense_seconds, semantic=args.semantic_seconds,
                             first_typing=args.first_typing_seconds, scroll=args.scroll_seconds)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "timeline.json", timeline)
    print(json.dumps({"timeline": str(output / "timeline.json"), "cases": len(timeline["cases"]),
                      "pages": timeline["page_count"], "seconds": timeline["duration_seconds"]}), flush=True)
    if args.encode:
        receipt = encode(timeline, args.loops_root.resolve(), output, ffmpeg=args.ffmpeg,
                         crf=args.crf, preset=args.preset, cache_frames=args.cache_frames, font=args.font)
        write_json(output / "encode.receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "movie": receipt["movie"]}), flush=True)


if __name__ == "__main__":
    main()
