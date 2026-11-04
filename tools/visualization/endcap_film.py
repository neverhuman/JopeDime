#!/usr/bin/env python3
"""Assemble the approved JAIN opening, ten Dendrite cases, and a VEOX endcap."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import closing
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import wave

import numpy as np
from PIL import Image, ImageFont

import intro_film as intro
from intelligence_wave import TABLE_SWITCH_FRAMES


CASE_FRAMES = 84
ENDCAP_START = intro.PLATE_COUNT + 10 * CASE_FRAMES
ENDCAP_FRAMES = intro.MASTER_COUNT - ENDCAP_START
CROSSFADE_FRAMES = 14
BRAND_BOTTOM = 700
SAMPLES_PER_FRAME = intro.AUDIO_RATE // intro.FPS
VISUALIZATION_ROOT = Path(__file__).resolve().parent


def scroll_coverage(case, offsets, height):
    rows = set()
    line = case["line_height_px"]
    scroll = case["scroll_offsets_px"]
    for frame in offsets:
        visible = height if frame >= intro.SCROLL_FIRST else max(line, round(height * (frame + 1) / intro.SCROLL_FIRST))
        rows.update(range((scroll[frame] + line - 1) // line,
                          min(case["visual_rows"], (scroll[frame] + visible) // line)))
    return rows


def displayed_cues(case, height):
    """Mirror the original editor's nearest visible cue choice per source frame."""
    line = case["line_height_px"]
    result = {}
    for frame, scroll in enumerate(case["scroll_offsets_px"]):
        if frame < intro.SCROLL_FIRST:
            continue
        cue = min((i for i, item in enumerate(case["cues"]) if abs(item["case_frame"] - frame) <= 4
                   and item["rows"][0] * line < scroll + height and (item["rows"][1] + 1) * line > scroll),
                  key=lambda i: abs(case["cues"][i]["case_frame"] - frame), default=None)
        if cue is not None:
            result[frame] = cue
    return result


def distribute(first, last, steps):
    return [first + (i * (last - first) + steps // 2) // steps for i in range(1, steps + 1)]


def cue_preserving_offsets(case, previous, height, pinned):
    """Find the closest monotonic 84-frame schedule through the cue anchors."""
    nominal = pinned + distribute(pinned[-1], previous[-1], CASE_FRAMES - len(pinned))
    cue_at = displayed_cues(case, height)
    baseline = {cue_at[frame] for frame in previous if frame in cue_at}
    mandatory = {previous[-1]}
    pinned_cues = {cue_at[frame] for frame in pinned if frame in cue_at}
    for cue in sorted(baseline - pinned_cues):
        candidates = [frame for frame in nominal if cue_at.get(frame) == cue]
        if not candidates:
            candidates = [frame for frame in previous if cue_at.get(frame) == cue]
        anchor = min(candidates, key=lambda frame: abs(frame - case["cues"][cue]["case_frame"]))
        intro.require(anchor > pinned[-1], f"{case['case_id']}: cue conflicts with the approved opening")
        mandatory.add(anchor)

    states = {pinned[-1]: (0, tuple(pinned))}
    for target in nominal[len(pinned):]:
        following = {}
        for prior, (cost, path) in states.items():
            for frame in (prior + 1, prior + 2):
                if frame > previous[-1] or (frame == prior + 2 and prior + 1 in mandatory):
                    continue
                option = cost + abs(frame - target), (*path, frame)
                if frame not in following or option < following[frame]:
                    following[frame] = option
        states = following
    intro.require(previous[-1] in states, f"{case['case_id']}: no retime preserves the cue anchors")
    offsets = list(states[previous[-1]][1])
    emitted = {cue_at[frame] for frame in offsets if frame in cue_at}
    intro.require(baseline <= emitted, f"{case['case_id']}: approved source cue was dropped")
    return offsets, sorted(baseline), sorted(emitted)


def plan_timeline(source, approved):
    """Prove the exact 45-second schedule against the approved 1080p reference."""
    previous = intro.plan_retime(source)
    intro.require(previous == approved["retime"], "Approved cut and current source retime disagree")
    baseline = previous["selected_source_indices"]
    height = source["source_viewport"][3]
    chapters = [previous["chapters"][0]]
    selected, coverage = [], []
    for index, case in enumerate(source["cases"]):
        start, end = case["start"], case["end"]
        old = [frame - start for frame in baseline[index * intro.CASE_COUNT:(index + 1) * intro.CASE_COUNT]]
        pin = old[:intro.INTRO_COUNT - intro.PLATE_COUNT + 1] if index == 0 else old[:1]
        offsets, cues_before, cues_after = cue_preserving_offsets(case, old, height, pin)
        intro.require(len(offsets) == CASE_FRAMES and offsets[0] == 0 and offsets[-1] == end - start - 1
                      and all(1 <= b - a <= 2 for a, b in zip(offsets, offsets[1:])),
                      f"{case['case_id']}: invalid source schedule")
        visible = scroll_coverage(case, offsets, height)
        intro.require(visible == set(range(case["visual_rows"])), f"{case['case_id']}: lost fully visible source rows")
        files = {item["filename"]: item["source_sha256"] for item in case["files"]}
        intro.require(len(files) == len(case["files"]) and all(item["visual_rows"] > 0 and
                      set(range(item["visual_start"], item["visual_start"] + item["visual_rows"])) <= visible
                      for item in case["files"]), f"{case['case_id']}: incomplete file coverage")
        slots = sorted({bisect_right(TABLE_SWITCH_FRAMES, frame) - 1 for frame in offsets})
        intro.require(len(case["table_row_orders"]) == len(TABLE_SWITCH_FRAMES)
                      and slots == list(range(len(TABLE_SWITCH_FRAMES))), f"{case['case_id']}: lost a table shuffle")
        scroll = case["scroll_offsets_px"]
        jump = max(scroll[b] - scroll[a] for a, b in zip(offsets, offsets[1:]))
        intro.require(jump <= previous["code_coverage"][index]["largest_scroll_jump_px"],
                      f"{case['case_id']}: scroll step exceeds the approved cut")
        coverage.append({"case_id": case["case_id"], "visual_rows_fully_visible": len(visible),
                         "largest_scroll_jump_px": jump, "source_file_hashes": files,
                         "approved_displayed_cue_indices": cues_before, "displayed_cue_indices": cues_after,
                         "table_shuffle_slots": slots})
        out = intro.PLATE_COUNT + index * CASE_FRAMES
        chapters.append({"title": case["dataset_name"], "case_id": case["case_id"],
                         "start_frame": out, "end_frame": out + CASE_FRAMES})
        selected.extend(start + frame for frame in offsets)
    protected = intro.INTRO_COUNT - intro.PLATE_COUNT + 1
    intro.require(selected[:protected] == baseline[:protected] and len(selected) == ENDCAP_START - intro.PLATE_COUNT,
                  "The opening handoff or demo count changed")
    chapters.append({"title": "VEOX", "start_frame": ENDCAP_START, "end_frame": intro.MASTER_COUNT})
    return {"selected_source_indices": selected, "chapters": chapters, "code_coverage": coverage,
            "visual_rows_fully_visible": sum(item["visual_rows_fully_visible"] for item in coverage),
            "source_file_count": sum(len(item["source_file_hashes"]) for item in coverage),
            "approved_displayed_cue_count": sum(len(item["approved_displayed_cue_indices"]) for item in coverage),
            "table_shuffle_count": sum(len(item["table_shuffle_slots"]) for item in coverage),
            "first_moving_demo_frame": intro.PLATE_COUNT, "endcap_start_frame": ENDCAP_START,
            "unchanged_opening_source_frame_count": protected}


def read_endcap(root, receipt_path, scene_path, scene_file):
    receipt = json.loads(receipt_path.read_bytes())
    scene = json.loads(scene_path.read_bytes())
    scene_sha = intro.digest(scene_path)
    intro.require(scene["schema"] == "jain.veox-endcap-scene/v1" and scene["status"] == "PASS"
                  and (scene["width"], scene["height"], scene["fps"], scene["frame_start"], scene["frame_end"])
                  == (intro.WIDTH, intro.HEIGHT, intro.FPS, 1, ENDCAP_FRAMES)
                  and scene["copy"] == ["Turn what if into what’s next."]
                  and scene["logo_sha256"] == intro.digest(VISUALIZATION_ROOT / "assets/veox-logo.svg")
                  and scene["worker_sha256"] == intro.digest(VISUALIZATION_ROOT / "blender/veox_endcap.py"),
                  "Endcap scene does not match the approved copy, logo, builder, or timeline")
    intro.require(receipt["schema"] == "jain.veox-endcap-plates/v1" and receipt["status"] == "PASS"
                  and receipt["scene_receipt_sha256"] == scene_sha
                  and receipt["scene_sha256"] == intro.digest(scene_file)
                  and receipt["worker_sha256"] == scene["worker_sha256"]
                  and receipt["shared_worker_sha256"] == intro.digest(VISUALIZATION_ROOT / "blender/render_animation.py")
                  and (receipt["width"], receipt["height"], receipt["fps"]) == (intro.WIDTH, intro.HEIGHT, intro.FPS)
                  and receipt["gpu_only"] is True and receipt["samples"] == 96,
                  "Endcap receipt does not bind the supplied 1080p30 scene")
    frames = {}
    root = root.resolve()
    for expected, record in enumerate(receipt["frames"], start=1):
        filename = f"frame-{expected:06d}.png"
        path = (root / filename).resolve(strict=True)
        intro.require(record["frame"] == expected and record["file"] == filename and path.is_relative_to(root),
                      f"Invalid endcap plate path or order at frame {expected}")
        frames[expected] = path, record["sha256"]
    intro.require(len(frames) == ENDCAP_FRAMES, "Expected exactly 180 native endcap plates")
    return frames


def make_audio(original, master, tail):
    cut = ENDCAP_START * SAMPLES_PER_FRAME
    length = ENDCAP_FRAMES * SAMPLES_PER_FRAME
    with wave.open(str(original), "rb") as audio:
        intro.require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getnframes())
                      == (2, 2, intro.AUDIO_RATE, cut + length), "Approved intro audio profile differs")
        prefix = audio.readframes(cut)
        intro.require(not any(audio.readframes(length)), "Approved audio overlaps the new endcap")

    time = np.arange(length) / intro.AUDIO_RATE
    mix = np.zeros((length, 2), dtype=np.float64)
    for at, pitch, gain, pan in ((.42, 110., .105, -.20), (1.18, 164.814, .071, .20), (2.30, 220., .077, 0.),
                                 (2.85, 261.626, .055, .19), (3.20, 329.628, .043, -.19)):
        phase = np.maximum(time - at, 0.)
        envelope = np.sin(.5 * np.pi * np.minimum(phase / .36, 1.)) ** 2 * np.exp(-phase / 1.65)
        tone = sum(strength * np.sin(2 * np.pi * pitch * harmonic * phase)
                   for harmonic, strength in ((1., 1.), (2., .20), (3., .08)))
        mix += (gain * envelope * tone)[:, None] * np.sqrt(np.array((1. - pan, 1. + pan)) / 2.)
    mix *= np.cos(.5 * np.pi * np.clip((time - 4.7) / 1.25, 0., 1.))[:, None] ** 2
    peak = float(np.max(np.abs(mix)))
    intro.require(.02 < peak < .40, "Endcap audio level outside its expected range")
    pcm = np.rint(mix * 32767).astype("<i2").tobytes()
    intro.require(not any(pcm[:SAMPLES_PER_FRAME * 4]) and not any(pcm[-SAMPLES_PER_FRAME * 4:]),
                  "Endcap audio does not begin and end cleanly")
    for path, earlier in ((master, prefix), (tail, b"")):
        with wave.open(str(path), "wb") as audio:
            audio.setparams((2, 2, intro.AUDIO_RATE, (len(earlier) + len(pcm)) // 4, "NONE", "not compressed"))
            audio.writeframes(earlier)
            audio.writeframes(pcm)
    return {"rate_hz": intro.AUDIO_RATE, "channels": 2, "prefix_samples_per_channel": cut,
            "endcap_samples_per_channel": length, "master_samples_per_channel": cut + length,
            "original_prefix_pcm_sha256": hashlib.sha256(prefix).hexdigest(),
            "endcap_pcm_sha256": hashlib.sha256(pcm).hexdigest(), "endcap_peak_dbfs": 20 * math.log10(peak),
            "master_wav_sha256": intro.digest(master), "endcap_wav_sha256": intro.digest(tail)}


def encoder(ffmpeg, movie, audio, frames, metadata):
    return [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{intro.WIDTH}x{intro.HEIGHT}", "-framerate", str(intro.FPS), "-i", "pipe:0",
            "-i", str(audio), "-f", "ffmetadata", "-i", str(metadata), "-map_chapters", "2", "-map", "0:v:0",
            "-map", "1:a:0", "-t", str(frames / intro.FPS), "-vf", "scale=out_color_matrix=bt709", "-c:v", "libx264",
            "-crf", "16", "-preset", "slower", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart", str(movie)]


def endcap_bounds(image):
    """Measure the mark and copy above the reflective floor."""
    pixels = np.asarray(image).max(axis=2)
    active = pixels[:BRAND_BOTTOM] >= 90
    ys, xs = np.nonzero(active)
    if not len(xs):
        return {"peak_value": int(pixels.max()), "active_pixels": 0, "bounds_px": None}
    return {"peak_value": int(pixels.max()), "active_pixels": len(xs),
            "bounds_px": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}


def master_tail_frame(image, last_demo, frame):
    if frame >= CROSSFADE_FRAMES:
        return image
    return Image.blend(last_demo, image, intro.smooth(frame / CROSSFADE_FRAMES))


def compose(args, source, approved, timeline, opening, tail):
    root = args.output_dir.resolve()
    intro.require(not root.exists() and not root.is_relative_to(Path(__file__).resolve().parents[2]),
                  "Use a fresh output directory outside the checkout")
    root.mkdir(parents=True)
    fonts = {key: ImageFont.truetype(str(args.font), size) for key, size in (("headline", 72), ("eyebrow", 22))}
    master, endcap = root / "JAIN-The-Next-Move-VEOX-45s.mp4", root / "VEOX-Endcap-6s.mp4"
    master_wav, endcap_wav = root / "master-audio.wav", root / "endcap-audio.wav"
    audio = make_audio(args.approved_audio, master_wav, endcap_wav)
    master_metadata, endcap_metadata = root / "master.ffmetadata", root / "endcap.ffmetadata"
    tail_chapters = [{"title": "VEOX", "start_frame": 0, "end_frame": ENDCAP_FRAMES}]
    intro.chapter_metadata(master_metadata, timeline["chapters"])
    intro.chapter_metadata(endcap_metadata, tail_chapters)
    master_cmd = encoder(args.ffmpeg, master, master_wav, intro.MASTER_COUNT, master_metadata)
    tail_cmd = encoder(args.ffmpeg, endcap, endcap_wav, ENDCAP_FRAMES, endcap_metadata)
    anchors = {item["master_frame"]: item for item in approved["raster_handoff"]}
    measurements = []
    with intro.ffmpeg_process(master_cmd, root / "master-encode.log", writing=True) as long:
        with closing(intro.decode_source(args.ffmpeg, args.demo, root / "source-decode.log")) as frames:
            first = next(frames)
            reference = Image.frombytes("RGB", (intro.WIDTH, intro.HEIGHT), first)
            destination, boxes = intro.editor_destination(reference, source["source_viewport"])
            for frame in range(intro.PLATE_COUNT):
                plate = intro.image_for_plate(opening, frame + 1)
                image = intro.title_card(intro.prelap(plate, reference, destination, boxes, frame), frame, fonts)
                rgb = image.tobytes()
                if frame == intro.FULL_SOURCE_LOCK:
                    verify_anchor(frame, 0, rgb, anchors[frame])
                long.write(rgb)

            selections = iter(timeline["selected_source_indices"])
            wanted = next(selections)
            decoded, used = 0, 0
            for index, rgb in enumerate(chain((first,), frames)):
                decoded += 1
                if index != wanted:
                    continue
                master_frame = intro.PLATE_COUNT + used
                if master_frame in anchors:
                    verify_anchor(master_frame, index, rgb, anchors[master_frame])
                if master_frame == ENDCAP_START - 1:
                    verify_anchor(master_frame, index, rgb, anchors[intro.MASTER_COUNT - 1])
                    last_demo = Image.frombytes("RGB", (intro.WIDTH, intro.HEIGHT), rgb)
                long.write(rgb)
                used += 1
                wanted = next(selections, None)
            intro.require(decoded == source["frames"] and used == ENDCAP_START - intro.PLATE_COUNT and wanted is None,
                          "Demo decode or frame selection differs from the verified timeline")

        with intro.ffmpeg_process(tail_cmd, root / "endcap-encode.log", writing=True) as short:
            for frame in range(1, ENDCAP_FRAMES + 1):
                image = intro.image_for_plate(tail, frame)
                if frame in (1, CROSSFADE_FRAMES, 28, 55, 95, 121, 151, 161, 170, ENDCAP_FRAMES):
                    measurements.append({"frame": frame, **endcap_bounds(image)})
                long.write(master_tail_frame(image, last_demo, frame).tobytes())
                short.write(image.tobytes())

    intro.require(measurements[0]["peak_value"] <= 6 and measurements[-1]["peak_value"] <= 6,
                  "Standalone endcap does not open and finish on black")
    # The introductory light sweep may reach the edges. Check the settled
    # brand hold separately, when the mark and copy are visible.
    key_bounds = [item["bounds_px"] for item in measurements if 95 <= item["frame"] <= 170
                  and item["active_pixels"] > 1000]
    intro.require(key_bounds and all(0 < x0 < x1 < intro.WIDTH and 0 < y0 < y1 < BRAND_BOTTOM
                                   for x0, y0, x1, y1 in key_bounds), "Endcap silhouette absent or touches the frame edge")

    videos = {"master": intro.verify_movie(args.ffmpeg, args.ffprobe, master, intro.MASTER_COUNT,
                                           timeline["chapters"], root / "master-full-decode.log"),
              "endcap": intro.verify_movie(args.ffmpeg, args.ffprobe, endcap, ENDCAP_FRAMES,
                                           tail_chapters, root / "endcap-full-decode.log")}
    return videos, audio, measurements, [master_cmd, tail_cmd]


def verify_anchor(master_frame, source_frame, raster, approved):
    sha = hashlib.sha256(raster).hexdigest()
    intro.require(source_frame == approved["source_frame"] and
                  sha == approved["input_rgb_sha256"] == approved["assembled_rgb_sha256"],
                  f"Approved raster identity differs at master frame {master_frame}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("demo", "demo-receipt", "approved-film-receipt", "approved-audio", "intro-plates",
                 "intro-plates-receipt", "intro-scene-receipt", "endcap-plates", "endcap-plates-receipt",
                 "endcap-scene", "endcap-scene-receipt", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=intro.FONT_PATH)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ("demo", "demo_receipt", "approved_film_receipt", "approved_audio",
              "intro_plates_receipt", "intro_scene_receipt", "endcap_plates_receipt", "endcap_scene",
              "endcap_scene_receipt", "font")}
    inputs.update(compositor=Path(__file__), intro_compositor=Path(intro.__file__),
                  demo_compositor=VISUALIZATION_ROOT / "intelligence_wave.py",
                  endcap_builder=VISUALIZATION_ROOT / "blender/veox_endcap.py",
                  blender_render_helper=VISUALIZATION_ROOT / "blender/render_animation.py",
                  endcap_logo=VISUALIZATION_ROOT / "assets/veox-logo.svg")
    hashes = {name: intro.digest(path) for name, path in inputs.items()}
    approved = json.loads(args.approved_film_receipt.read_bytes())
    intro.require(approved["schema"] == "jain.intro-film/v1" and approved["status"] == "PASS"
                  and all(hashes[name] == approved[f"{name}_sha256"] for name in
                          ("demo", "demo_receipt", "intro_plates_receipt", "intro_scene_receipt", "font"))
                  and hashes["approved_audio"] == approved["audio"]["sha256"], "Approved inputs differ")
    source = intro.read_demo(args.demo, args.demo_receipt)
    timeline = plan_timeline(source, approved)
    opening, _ = intro.read_plates(args.intro_plates, args.intro_plates_receipt, args.intro_scene_receipt, source, full=True)
    tail = read_endcap(args.endcap_plates, args.endcap_plates_receipt, args.endcap_scene_receipt, args.endcap_scene)
    videos, audio, bounds, commands = compose(args, source, approved, timeline, opening, tail)
    intro.require(all(intro.digest(path) == hashes[name] for name, path in inputs.items()), "Assembly inputs changed")
    receipt = {"schema": "jain.veox-endcap-film/v1", "status": "PASS", "outputs": videos, "audio": audio,
               "width": intro.WIDTH, "height": intro.HEIGHT, "fps": intro.FPS, "inputs_sha256": hashes,
               "timeline": timeline, "endcap_plate_bounds": bounds, "encoder_commands": commands,
               "endcap_brand_bounds_roi": [0, 0, intro.WIDTH, BRAND_BOTTOM],
               "endcap_brand_bounds_note": "Measure the settled mark and copy above the physical floor reflection.",
               "opening_raster_anchors_verified": [329, 330, 359, 360], "last_demo_source_frame_verified": 1259,
               "master_endcap_crossfade_frames": CROSSFADE_FRAMES}
    intro.write_json(args.output_dir / "endcap-film.receipt.json", receipt)
    print(videos["master"]["path"], flush=True)


if __name__ == "__main__":
    main()
