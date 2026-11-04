#!/usr/bin/env python3
"""Assemble the JAIN intro and verified Dendrite demo into a 45-second cut."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 1920, 1080, 30
PLATE_COUNT, INTRO_COUNT, MASTER_COUNT = 330, 360, 1350
SOURCE_CASE_COUNT, CASE_COUNT = 126, 102
AUDIO_RATE = 48_000
SCROLL_FIRST = 7
FULL_CODE_REVEAL = 5
FONT_PATH = Path("/System/Library/Fonts/Avenir Next.ttc")
TITLE_CUES = (
    (12, 93, "What if innovation\nwas a game?", ""),
    (99, 159, "Every move.\nAn algorithm.", ""),
    (165, 228, "Test. Learn.\nPlay again.", "ALGORITHMIC SELF-PLAY · TRAINING"),
    (240, 315, "JAIN", "Just An Inference Network"),
)


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_demo(path, receipt_path):
    receipt = json.loads(receipt_path.read_bytes())
    require(receipt["schema"] == "jain.intelligence-wave-encode/v2" and receipt["status"] == "PASS"
            and receipt["movie_sha256"] == digest(path), "Demo movie identity differs")
    require((receipt["width"], receipt["height"], receipt["fps"], receipt["frames"]) == (WIDTH, HEIGHT, FPS, 1260),
            "Expected the verified 42-second 1080p30 Dendrite source")
    return receipt


def plan_retime(receipt):
    """Sample each chapter independently and prove complete visual-row coverage."""
    chapters = [{"title": "The Next Move", "start_frame": 0, "end_frame": PLATE_COUNT}]
    selected, coverage = [], []
    require(len(receipt["cases"]) == 10, "Expected all ten source datasets")
    view_x, _, _, view_height = receipt["source_viewport"]
    require(view_x >= 1310, "Source viewport would be obscured by the opening reveal")
    denominator = CASE_COUNT - 1
    offsets = [(index * (SOURCE_CASE_COUNT - 1) + denominator // 2) // denominator for index in range(CASE_COUNT)]
    for index, case in enumerate(receipt["cases"]):
        start, end = index * SOURCE_CASE_COUNT, (index + 1) * SOURCE_CASE_COUNT
        require((case["start"], case["end"]) == (start, end), f"{case['case_id']}: unexpected source chapter range")
        source_indices = [start + offset for offset in offsets]
        selected.extend(source_indices)
        out = PLATE_COUNT + index * CASE_COUNT
        chapters.append({"title": case["dataset_name"], "case_id": case["case_id"],
                         "start_frame": out, "end_frame": out + CASE_COUNT})
        scroll = case["scroll_offsets_px"]
        row_count, line_height = case["visual_rows"], case["line_height_px"]
        require(len(scroll) == SOURCE_CASE_COUNT and line_height > 0 and row_count > 0,
                f"{case['case_id']}: invalid scroll metadata")
        visited = set()
        for selected_index, source_local in enumerate(offsets):
            # During the match cut, count only frames where the entire editor
            # is opaque, including its first fully visible typed page.
            if index == 0 and selected_index < FULL_CODE_REVEAL:
                continue
            offset = scroll[source_local]
            visible = (view_height if source_local >= SCROLL_FIRST
                       else max(line_height, round(view_height * (source_local + 1) / SCROLL_FIRST)))
            visited.update(range((offset + line_height - 1) // line_height,
                                 min(row_count, (offset + visible) // line_height)))
        require(visited == set(range(row_count)), f"{case['case_id']}: retime dropped complete visual source rows")
        for file in case["files"]:
            first, count = file["visual_start"], file["visual_rows"]
            require(count > 0 and all(row in visited for row in range(first, first + count)),
                    f"{case['case_id']}: incomplete {file['filename']} coverage")
        maximum_jump = max(scroll[b] - scroll[a] for a, b in zip(offsets, offsets[1:]))
        coverage.append({"case_id": case["case_id"], "source_first": start, "source_last": end - 1,
                         "visual_rows_fully_visible": len(visited), "largest_scroll_jump_px": maximum_jump,
                         "source_file_hashes": {file["filename"]: file["source_sha256"] for file in case["files"]}})
    require(len(selected) == MASTER_COUNT - PLATE_COUNT and selected[0] == 0 and selected[-1] == 1259
            and all(a < b for a, b in zip(selected, selected[1:])), "Source retime is not strictly increasing")
    return {"selected_source_indices": selected, "chapters": chapters, "code_coverage": coverage,
            "first_fully_opaque_editor_frame": PLATE_COUNT + FULL_CODE_REVEAL,
            "last_intro_source_frame": selected[INTRO_COUNT - PLATE_COUNT - 1],
            "first_source_frame_after_intro": selected[INTRO_COUNT - PLATE_COUNT]}


def read_plates(root, receipt_path, scene_receipt_path, demo, *, full):
    scene_bytes = scene_receipt_path.read_bytes()
    scene = json.loads(scene_bytes)
    require(scene["schema"] == "jain.intro-scene/v1" and scene["status"] == "PASS"
            and (scene["frame_start"], scene["frame_end"]) == (1, PLATE_COUNT)
            and (scene["width"], scene["height"], scene["fps"]) == (WIDTH, HEIGHT, FPS),
            "Intro scene receipt is not an approved 330-frame 1080p30 scene")
    require(scene["source_receipt_sha256"] == demo["scene_receipt_sha256"],
            "Intro scene and demo movie have different Dendrite source receipts")
    receipt_bytes = receipt_path.read_bytes()
    data = json.loads(receipt_bytes)
    require(data["schema"] in {"jain.intro-plates/v1", "jain.intro-gpu-frames/v1"} and data["status"] == "PASS"
            and (data["width"], data["height"], data["fps"]) == (WIDTH, HEIGHT, FPS),
            "Intro plate render did not pass 1080p30 checks")
    scene_sha = hashlib.sha256(scene_bytes).hexdigest()
    require(data["scene_receipt_sha256"] == scene_sha, "Intro plates are not bound to the supplied intro scene")
    frames = {}
    for entry in data["frames"]:
        number = entry["frame"]
        require(number not in frames and 1 <= number <= PLATE_COUNT
                and entry["file"] == f"frame-{number:06d}.png", "Plate names or frame indices differ")
        frames[number] = (root / entry["file"], entry["sha256"])
    if data["schema"] == "jain.intro-gpu-frames/v1":
        require(data["gpu_only"] is True and data["cpu_fallback"] is False and data["requested_frames"] == list(frames),
                "Direct render receipt differs from its GPU frame inventory")
    if full:
        require(set(frames) == set(range(1, PLATE_COUNT + 1)), "Expected all 330 intro render plates")
    return frames, {"intro_plates_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                    "intro_scene_receipt_sha256": scene_sha}


def image_for_plate(frames, number):
    path, expected = frames[number]
    require(digest(path) == expected, f"Intro plate {number} differs from its render receipt")
    with Image.open(path) as png:
        require(png.size == (WIDTH, HEIGHT), f"Wrong raster size for intro plate {number}")
        return png.convert("RGB")


def smooth(value):
    value = max(0., min(1., value))
    return value * value * (3 - 2 * value)


def title_card(plate, frame, fonts):
    cue = next((cue for cue in TITLE_CUES if cue[0] <= frame < cue[1]), None)
    if cue is None:
        return plate
    start, end, headline, detail = cue
    amount = min(smooth((frame - start + 1) / 5), smooth((end - frame) / 6))
    layer = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(layer)
    x, y = 118, 134
    tint = (112, 238, 250, round(245 * amount))
    white = (231, 243, 255, round(255 * amount))
    draw.rounded_rectangle((x - 28, y + 9, x - 24, y + (104 if headline == "JAIN" else 80)),
                           radius=2, fill=tint)
    if headline == "JAIN":
        draw.text((x, y), headline, font=fonts["brand"], fill=white, anchor="lt")
        draw.text((x + 3, y + 99), detail, font=fonts["brand_detail"], fill=tint, anchor="lt")
    else:
        if detail:
            draw.text((x, y), detail, font=fonts["eyebrow"], fill=tint, anchor="lt")
            y += 39
        draw.multiline_text((x, y), headline, font=fonts["headline"], fill=white, spacing=5)
    return Image.alpha_composite(plate.convert("RGBA"), layer).convert("RGB")


def reveal(plate, source, index):
    if index >= INTRO_COUNT - PLATE_COUNT - 1:
        return source
    x = np.arange(WIDTH, dtype=float)
    right = np.clip((x - 1250) / 60, 0., 1.)
    left = np.clip((700 - x) / 80, 0., 1.)
    center = smooth((index + .35) / 29.)
    editor = smooth((index + .35) / FULL_CODE_REVEAL)
    table = smooth((index + .35) / 17.)
    line = np.rint(255 * (center + right * (editor - center) + left * (table - center))).astype(np.uint8)
    mask = Image.fromarray(line[None, :]).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)
    return Image.composite(source, plate, mask)


def synthesize_audio(path):
    """Procedural stereo cue. All oscillators, percussion, and air are original."""
    total = MASTER_COUNT * AUDIO_RATE // FPS
    bed_length = round(14.6 * AUDIO_RATE)
    mix = np.zeros((bed_length, 2), dtype=np.float64)
    time = np.arange(bed_length) / AUDIO_RATE
    fade_in = np.sin(np.pi / 2 * np.minimum(time / 1.35, 1.)) ** 2
    fade_out = np.cos(np.pi / 2 * np.clip((time - 10.7) / 3.9, 0., 1.)) ** 2
    envelope = fade_in * fade_out * .046
    for channel, cents in enumerate((-.9, .9)):
        hz = np.array([55., 82.4069, 110., 164.8138]) * 2 ** (cents / 1200.)
        tones = sum(gain * np.sin(2 * np.pi * frequency * time + phase)
                    for frequency, gain, phase in zip(hz, (.72, .4, .25, .11), (0., .8, 1.2, 2.5), strict=True))
        mix[:, channel] += envelope * tones
    rng = np.random.default_rng(261104)
    # The stone impact is a dry, three-resonance wooden click. Smaller
    # repetitions trace candidate, test, and result, with a soft closing tone.
    for at, pitch, gain, pan, decay in ((.9, 202., .21, -.10, .036), (3.62, 335., .072, -.08, .06),
                                       (6.08, 392., .062, .18, .07), (7.18, 293., .058, -.23, .08),
                                       (8.35, 220., .082, .04, .24), (11.08, 330., .064, .10, .42)):
        length = round(min(2.2, 9 * decay) * AUDIO_RATE)
        t = np.arange(length) / AUDIO_RATE
        raw_noise = rng.standard_normal(length + 1)
        transient = np.diff(raw_noise) * np.exp(-t / .0028) * .19
        body = (.62 * np.sin(2 * np.pi * pitch * t) + .22 * np.sin(2 * np.pi * pitch * 2.73 * t)
                + .09 * np.sin(2 * np.pi * pitch * 5.71 * t)) * np.exp(-t / decay)
        click = gain * np.minimum(t / .0015, 1.) * (body + transient)
        start = round(at * AUDIO_RATE)
        balance = np.array((math.sqrt((1 - pan) / 2), math.sqrt((1 + pan) / 2)))
        mix[start:start + length] += click[:, None] * balance
    peak = float(np.max(np.abs(mix)))
    require(0.03 < peak < .45, "Unexpected procedural audio level")
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, 2, AUDIO_RATE, total, "NONE", "not compressed"))
        wav.writeframes(np.rint(mix * 32767).astype("<i2").tobytes())
        wav.writeframes(bytes((total - bed_length) * 4))
    return {"rate_hz": AUDIO_RATE, "channels": 2, "samples_per_channel": total,
            "last_cue_seconds": bed_length / AUDIO_RATE, "peak_dbfs": 20 * math.log10(peak), "sha256": digest(path)}


def chapter_metadata(path, chapters):
    lines = [";FFMETADATA1"]
    for chapter in chapters:
        title = chapter["title"]
        for char in "\\=;#":
            title = title.replace(char, "\\" + char)
        require("\n" not in title, "Chapter title must be a single line")
        lines.extend(("[CHAPTER]", f"TIMEBASE=1/{FPS}", f"START={chapter['start_frame']}",
                      f"END={chapter['end_frame']}", f"title={title}"))
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


@contextmanager
def ffmpeg_process(command, log, *, writing):
    with log.open("xb") as errors:
        with subprocess.Popen(command, stdin=subprocess.PIPE if writing else subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL if writing else subprocess.PIPE, stderr=errors) as child:
            pipe = child.stdin if writing else child.stdout
            try:
                yield pipe
                pipe.close()
                if child.wait() != 0:
                    raise RuntimeError(f"FFmpeg failed, see {log}")
            except BaseException:
                if child.poll() is None:
                    child.kill()
                raise


def encoder_command(ffmpeg, movie, audio, frames, metadata=None):
    command = [ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0", "-i", str(audio)]
    if metadata:
        command.extend(("-f", "ffmetadata", "-i", str(metadata), "-map_chapters", "2"))
    if frames == INTRO_COUNT:
        command.extend(("-af", "afade=t=out:st=11.25:d=0.75"))
    return command + ["-map", "0:v:0", "-map", "1:a:0", "-t", str(frames / FPS), "-c:v", "libx264",
                      "-crf", "17", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                      "-movflags", "+faststart", str(movie)]


def decode_source(ffmpeg, movie, log):
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(movie), "-map", "0:v:0", "-an",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "pipe:1"]
    with ffmpeg_process(command, log, writing=False) as pipe:
        size = WIDTH * HEIGHT * 3
        while data := pipe.read(size):
            require(len(data) == size, "Demo decode ended partway through a frame")
            yield data


def verify_movie(ffmpeg, ffprobe, movie, expected_frames, chapters, log):
    data = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_chapters",
                                              "-of", "json", str(movie)]))
    video, = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audio, = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    require((video["codec_name"], video["width"], video["height"], video["avg_frame_rate"],
             video["pix_fmt"], int(video["nb_frames"])) == ("h264", WIDTH, HEIGHT, "30/1", "yuv420p", expected_frames),
            f"Wrong encoded video profile or frame count for {movie.name}")
    duration = expected_frames / FPS
    require(audio["codec_name"] == "aac" and int(audio["sample_rate"]) == AUDIO_RATE and audio["channels"] == 2
            and abs(float(audio["duration"]) - duration) <= .025, f"Wrong audio profile for {movie.name}")
    require(abs(float(video["duration"]) - duration) < .001, f"Wrong duration for {movie.name}")
    for actual, expected in zip(data["chapters"], chapters, strict=True):
        require(actual["tags"]["title"] == expected["title"] and
                abs(float(actual["start_time"]) - expected["start_frame"] / FPS) <= .001 and
                abs(float(actual["end_time"]) - expected["end_frame"] / FPS) <= .001, "Chapter shifted in muxing")
    with log.open("xb") as errors:
        progress = subprocess.check_output([ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(movie),
                                            "-map", "0:v:0", "-map", "0:a:0", "-progress", "pipe:1",
                                            "-f", "null", "-"], stderr=errors, text=True)
    frames = [int(line.split("=", 1)[1]) for line in progress.splitlines() if line.startswith("frame=")]
    require(frames and frames[-1] == expected_frames and "progress=end" in progress, "Full video decode did not finish")
    return {"path": str(movie), "sha256": digest(movie), "decoded_frames": frames[-1],
            "duration_seconds": duration, "audio_duration_seconds": float(audio["duration"]), "chapters": chapters}


def compose(args, source, plan, plates, fonts, custody):
    root = args.output_dir
    audio = root / "original-intro-audio.wav"
    audio_receipt = synthesize_audio(audio)
    metadata = root / "chapters.ffmetadata"
    chapter_metadata(metadata, plan["chapters"])
    standalone = root / "JAIN-The-Next-Move-Intro-12s.mp4"
    master = root / "JAIN-The-Next-Move-45s.mp4"
    stand_command = encoder_command(args.ffmpeg, standalone, audio, INTRO_COUNT)
    master_command = encoder_command(args.ffmpeg, master, audio, MASTER_COUNT, metadata)
    anchors = []
    with (ffmpeg_process(master_command, root / "master-encode.log", writing=True) as long,
          ffmpeg_process(stand_command, root / "intro-encode.log", writing=True) as short):
        for index in range(PLATE_COUNT):
            image = title_card(image_for_plate(plates, index + 1), index, fonts)
            raster = image.tobytes()
            long.write(raster)
            short.write(raster)
        hold = image_for_plate(plates, PLATE_COUNT)
        selected = iter(plan["selected_source_indices"])
        wanted = next(selected)
        count, decoded = 0, 0
        for index, raster in enumerate(decode_source(args.ffmpeg, args.demo, root / "source-decode.log")):
            decoded += 1
            if index != wanted:
                continue
            master_frame = PLATE_COUNT + count
            if master_frame < INTRO_COUNT:
                source_image = Image.frombytes("RGB", (WIDTH, HEIGHT), raster)
                final_raster = reveal(hold, source_image, count).tobytes()
                short.write(final_raster)
            else:
                final_raster = raster
            if master_frame in (PLATE_COUNT, INTRO_COUNT - 1, INTRO_COUNT, MASTER_COUNT - 1):
                anchors.append({"master_frame": master_frame, "source_frame": index,
                                "input_rgb_sha256": hashlib.sha256(raster).hexdigest(),
                                "assembled_rgb_sha256": hashlib.sha256(final_raster).hexdigest()})
            long.write(final_raster)
            count += 1
            wanted = next(selected, None)
        require(decoded == source["frames"] and count == MASTER_COUNT - PLATE_COUNT and wanted is None,
                "Unexpected demo decode or selected frame count")
    require(digest(args.demo) == source["movie_sha256"], "Demo movie changed during assembly")
    require(digest(args.plates_receipt) == custody["intro_plates_receipt_sha256"]
            and digest(args.intro_scene_receipt) == custody["intro_scene_receipt_sha256"],
            "Intro scene or plate receipt changed during assembly")
    for anchor in anchors[1:]:
        require(anchor["input_rgb_sha256"] == anchor["assembled_rgb_sha256"], "End of reveal altered source raster")
    videos = {
        "intro": verify_movie(args.ffmpeg, args.ffprobe, standalone, INTRO_COUNT, [], root / "intro-full-decode.log"),
        "master": verify_movie(args.ffmpeg, args.ffprobe, master, MASTER_COUNT, plan["chapters"], root / "master-full-decode.log"),
    }
    write_json(root / "intro-film.receipt.json", {
        "schema": "jain.intro-film/v1", "status": "PASS", "outputs": videos, "audio": audio_receipt,
        "demo_sha256": source["movie_sha256"], "demo_receipt_sha256": digest(args.demo_receipt),
        **custody, "compositor_sha256": digest(Path(__file__)),
        "font_sha256": digest(args.font), "chapters_metadata_sha256": digest(metadata), "title_cues": TITLE_CUES,
        "retime": plan, "raster_handoff": anchors, "encoder_commands": [stand_command, master_command],
    })
    print(master)


def previews(args, plan, plates, fonts):
    for frame in args.preview:
        require(0 <= frame < INTRO_COUNT, "Preview only accepts a master frame inside the standalone intro")
        if frame < PLATE_COUNT:
            image = title_card(image_for_plate(plates, frame + 1), frame, fonts)
        else:
            source_frame = plan["selected_source_indices"][frame - PLATE_COUNT]
            data = subprocess.check_output([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(args.demo),
                                            "-vf", f"select=eq(n\\,{source_frame})", "-frames:v", "1",
                                            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            image = reveal(image_for_plate(plates, PLATE_COUNT), Image.frombytes("RGB", (WIDTH, HEIGHT), data),
                           frame - PLATE_COUNT)
        destination = args.output_dir / f"preview-{frame:06d}.png"
        with destination.open("xb") as stream:
            image.save(stream, format="PNG")
        print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--demo-receipt", type=Path, required=True)
    parser.add_argument("--plates", type=Path)
    parser.add_argument("--plates-receipt", type=Path)
    parser.add_argument("--intro-scene-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font", type=Path, default=FONT_PATH)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preview", type=int, action="append")
    args = parser.parse_args()
    source = read_demo(args.demo, args.demo_receipt)
    plan = plan_retime(source)
    if args.preflight:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    require(args.plates and args.plates_receipt and args.intro_scene_receipt and args.output_dir,
            "Render assembly requires --plates, --plates-receipt, --intro-scene-receipt, and --output-dir")
    args.output_dir = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[2]
    require(not args.output_dir.is_relative_to(checkout), "Rendered outputs must stay outside the checkout")
    plates, custody = read_plates(args.plates, args.plates_receipt, args.intro_scene_receipt, source, full=not args.preview)
    fonts = {key: ImageFont.truetype(str(args.font), size) for key, size in
             (("brand", 104), ("brand_detail", 31), ("headline", 62), ("eyebrow", 24))}
    args.output_dir.mkdir(parents=True, exist_ok=bool(args.preview))
    if args.preview:
        previews(args, plan, plates, fonts)
    else:
        compose(args, source, plan, plates, fonts, custody)


if __name__ == "__main__":
    main()
