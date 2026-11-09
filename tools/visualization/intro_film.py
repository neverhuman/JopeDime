#!/usr/bin/env python3
"""Assemble the JAIN intro and verified Dendrite demo into a 45-second cut."""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import hashlib
from itertools import chain
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
PRELAP_START = 234
PANEL_READY = 248
CODE_REVEAL = 306
CODE_READY = 315
BRAND_REVEAL = 316
FULL_SOURCE_LOCK = PLATE_COUNT - 1
FONT_PATH = Path("/System/Library/Fonts/Avenir Next.ttc")
TITLE_CUES = (
    (0, 98, "Innovation is a search.", ""),
    (98, 158, "Every move.\nAn algorithm.", ""),
    (162, PRELAP_START, "Test. Learn.\nPlay again.", "ALGORITHMIC SELF-PLAY · TRAINING"),
    (PRELAP_START, BRAND_REVEAL, "One inference.\nOne algorithm.", ""),
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


def read_demo(path, receipt_path, size=(WIDTH, HEIGHT)):
    receipt = json.loads(receipt_path.read_bytes())
    schema = "jain.intelligence-wave-encode/v2" if size == (WIDTH, HEIGHT) else "jain.uhd-demo/v1"
    require(receipt["schema"] == schema and receipt["status"] == "PASS"
            and receipt["movie_sha256"] == digest(path), "Demo movie identity differs")
    require((receipt["width"], receipt["height"], receipt["fps"], receipt["frames"]) == (*size, FPS, 1260),
            "Expected the verified Dendrite source at the requested native resolution")
    return receipt


def plan_retime(receipt):
    """Sample each chapter independently and prove complete visual-row coverage."""
    chapters = [{"title": "The Next Move", "start_frame": 0, "end_frame": PLATE_COUNT}]
    selected, coverage = [], []
    require(len(receipt["cases"]) == 10, "Expected all ten source datasets")
    view_x, view_y, view_width, view_height = receipt["source_viewport"]
    width, height = receipt["width"], receipt["height"]
    scale = width // WIDTH
    require(1310 * scale <= view_x < view_x + view_width <= width
            and 20 * scale <= view_y < view_y + view_height <= height,
            "Source viewport would be obscured by the opening reveal")
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
        for source_local in offsets:
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
    if scale == 2:
        require(receipt["encoded_frames"] == len(selected) and receipt["source_frame_indices"] == selected,
                "Sparse UHD source does not match the verified retime mapping")
    return {"selected_source_indices": selected, "chapters": chapters, "code_coverage": coverage,
            "first_fully_opaque_editor_frame": CODE_READY,
            "first_moving_demo_frame": PLATE_COUNT,
            "last_intro_source_frame": selected[INTRO_COUNT - PLATE_COUNT - 1],
            "first_source_frame_after_intro": selected[INTRO_COUNT - PLATE_COUNT]}


def read_plates(root, receipt_path, scene_receipt_path, demo, *, full, size=(WIDTH, HEIGHT)):
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
            and (data["width"], data["height"], data["fps"]) == (*size, FPS),
            "Intro plate render does not match the requested native resolution and 30 fps")
    if size != (WIDTH, HEIGHT):
        require(data["quality_profile"] == "uhd-master" and data["color_depth"] == 16,
                "UHD assembly requires native RGB16 master plates")
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


def title_layer(frame, fonts, size=(WIDTH, HEIGHT)):
    cue = next((cue for cue in TITLE_CUES if cue[0] <= frame < cue[1]), None)
    if cue is None:
        return None
    start, end, headline, detail = cue
    amount = min(smooth((frame - start + 1) / 5), smooth((end - frame) / 6))
    layer = Image.new("RGBA", size)
    draw = ImageDraw.Draw(layer)
    scale = size[0] // WIDTH
    x = size[0] // 2
    tint = (112, 238, 250, round(235 * amount))
    white = (231, 243, 255, round(255 * amount))
    lines = headline.splitlines()
    baseline = (225 if len(lines) == 1 else 184) * scale
    for i, line in enumerate(lines):
        draw.text((x, baseline + i * 86 * scale), line, font=fonts["headline"], fill=white, anchor="ms")
    if detail:
        draw.text((x, 326 * scale), detail, font=fonts["eyebrow"], fill=tint, anchor="ms")
    else:
        y = baseline + ((len(lines) - 1) * 86 + 28) * scale
        draw.rounded_rectangle((x - 24 * scale, y, x + 24 * scale, y + 3 * scale), radius=2 * scale, fill=tint)
    return layer


def title_card(plate, frame, fonts):
    layer = title_layer(frame, fonts)
    if layer is None:
        return plate
    return Image.alpha_composite(plate.convert("RGBA"), layer).convert("RGB")


def blend_region(plate, reference, box, amount):
    if amount > 0.:
        region = reference.crop(box)
        if amount < 1.:
            region = Image.blend(plate.crop(box), region, amount)
        plate.paste(region, box[:2])


def editor_boxes(viewport, scale):
    x, y, width, height = viewport
    return ((x, y, x + width, y + height), (max(x, x + width - 96 * scale), y - 20 * scale, x + width, y))


def editor_destination(source, viewport):
    """Use actual editor chrome with its original blank viewport color."""
    x, y, width, height = viewport
    destination = source.copy()
    content, current_line = editor_boxes(viewport, 1)
    destination.paste(source.getpixel((x + width // 2, y + height - 2)), content)
    destination.paste(source.getpixel((x + width // 2, y - 10)), current_line)
    return destination, (content, current_line)


def prelap_regions(frame, content_boxes, scale):
    def fade(start, end):
        return smooth((frame - start + 1) / (end - start + 1))

    panels = fade(PRELAP_START, PANEL_READY)
    yield 0, (0, 125 * scale, 611 * scale, HEIGHT * scale), panels
    yield 1, (1310 * scale, 0, WIDTH * scale, HEIGHT * scale), panels
    for box in content_boxes:
        yield 0, box, fade(CODE_REVEAL, CODE_READY)
    center = fade(BRAND_REVEAL, FULL_SOURCE_LOCK)
    yield 0, (0, 0, 611 * scale, 125 * scale), center
    yield 0, (611 * scale, 0, 1310 * scale, HEIGHT * scale), center


def prelap(plate, source, destination, content_boxes, frame):
    if frame < PRELAP_START:
        return plate
    if frame >= FULL_SOURCE_LOCK:
        return source
    image = plate.copy()
    for reference, box, amount in prelap_regions(frame, content_boxes, 1):
        blend_region(image, (source, destination)[reference], box, amount)
    return image


def synthesize_audio(path, sample_width=2):
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
    if sample_width == 3:
        samples = np.rint(mix * 8388607).astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    else:
        samples = np.rint(mix * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2, sample_width, AUDIO_RATE, total, "NONE", "not compressed"))
        wav.writeframes(samples)
        wav.writeframes(bytes((total - bed_length) * 2 * sample_width))
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


def decode_source(ffmpeg, movie, log, size=(WIDTH, HEIGHT), *, uhd=False):
    pixel_format = "rgb48le" if uhd else "rgb24"
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-i", str(movie), "-map", "0:v:0", "-an",
               "-f", "rawvideo", "-pix_fmt", pixel_format, "-fps_mode", "passthrough", "pipe:1"]
    with ffmpeg_process(command, log, writing=False) as pipe:
        frame_bytes = size[0] * size[1] * (6 if uhd else 3)
        while data := pipe.read(frame_bytes):
            require(len(data) == frame_bytes, "Demo decode ended partway through a frame")
            yield data


def verify_movie(ffmpeg, ffprobe, movie, expected_frames, chapters, log, *,
                 size=(WIDTH, HEIGHT), profile=("h264", "yuv420p", "aac")):
    data = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_chapters",
                                              "-of", "json", str(movie)]))
    video, = [stream for stream in data["streams"] if stream["codec_type"] == "video"]
    audio, = [stream for stream in data["streams"] if stream["codec_type"] == "audio"]
    codec, pixel_format, audio_codec = profile
    require((video["codec_name"], video["width"], video["height"], video["avg_frame_rate"],
             video["pix_fmt"], int(video["nb_frames"])) == (codec, *size, "30/1", pixel_format, expected_frames),
            f"Wrong encoded video profile or frame count for {movie.name}")
    if codec == "prores":
        require(video["profile"] == "XQ", f"Wrong ProRes profile for {movie.name}")
    if size != (WIDTH, HEIGHT):
        require(all(video.get(key) == "bt709" for key in ("color_space", "color_transfer", "color_primaries")),
                f"Wrong Rec. 709 metadata for {movie.name}")
    duration = expected_frames / FPS
    require(audio["codec_name"] == audio_codec and int(audio["sample_rate"]) == AUDIO_RATE and audio["channels"] == 2
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
          ffmpeg_process(stand_command, root / "intro-encode.log", writing=True) as short,
          closing(decode_source(args.ffmpeg, args.demo, root / "source-decode.log")) as demo_frames):
        first = next(demo_frames)
        reference = Image.frombytes("RGB", (WIDTH, HEIGHT), first)
        destination, content_boxes = editor_destination(reference, source["source_viewport"])
        for index in range(PLATE_COUNT):
            plate = image_for_plate(plates, index + 1)
            image = title_card(prelap(plate, reference, destination, content_boxes, index), index, fonts)
            raster = image.tobytes()
            if index == FULL_SOURCE_LOCK:
                anchors.append({"master_frame": index, "source_frame": 0,
                                "input_rgb_sha256": hashlib.sha256(first).hexdigest(),
                                "assembled_rgb_sha256": hashlib.sha256(raster).hexdigest()})
            long.write(raster)
            short.write(raster)
        selected = iter(plan["selected_source_indices"])
        wanted = next(selected)
        count, decoded = 0, 0
        for index, raster in enumerate(chain((first,), demo_frames)):
            decoded += 1
            if index != wanted:
                continue
            master_frame = PLATE_COUNT + count
            if master_frame < INTRO_COUNT:
                short.write(raster)
            if master_frame in (PLATE_COUNT, INTRO_COUNT - 1, INTRO_COUNT, MASTER_COUNT - 1):
                sha = hashlib.sha256(raster).hexdigest()
                anchors.append({"master_frame": master_frame, "source_frame": index,
                                "input_rgb_sha256": sha, "assembled_rgb_sha256": sha})
            long.write(raster)
            count += 1
            wanted = next(selected, None)
        require(decoded == source["frames"] and count == MASTER_COUNT - PLATE_COUNT and wanted is None,
                "Unexpected demo decode or selected frame count")
    require(digest(args.demo) == source["movie_sha256"], "Demo movie changed during assembly")
    require(digest(args.plates_receipt) == custody["intro_plates_receipt_sha256"]
            and digest(args.intro_scene_receipt) == custody["intro_scene_receipt_sha256"],
            "Intro scene or plate receipt changed during assembly")
    for anchor in anchors:
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
        "source_prelap": {"panels_start_frame": PRELAP_START, "panels_ready_frame": PANEL_READY,
                          "editor_content_start_frame": CODE_REVEAL, "editor_content_ready_frame": CODE_READY,
                          "brand_and_benchmark_start_frame": BRAND_REVEAL, "full_source_lock_frame": FULL_SOURCE_LOCK,
                          "prelap_source_frame": 0, "first_moving_demo_frame": PLATE_COUNT},
        "retime": plan, "raster_handoff": anchors, "encoder_commands": [stand_command, master_command],
    })
    print(master)


def source_image(args, frame):
    data = subprocess.check_output([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(args.demo),
                                    "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1",
                                    "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return Image.frombytes("RGB", (WIDTH, HEIGHT), data)


def uhd_reference(raw, viewport):
    from uhd_pixels import UHD_SIZE

    source = np.frombuffer(raw, dtype="<u2").reshape(UHD_SIZE[1], UHD_SIZE[0], 3)
    destination = source.copy()
    x, y, width, height = viewport
    boxes = editor_boxes(viewport, 2)
    for (left, top, right, bottom), sample in zip(boxes, ((y + height - 4, x + width // 2), (y - 20, x + width // 2)), strict=True):
        destination[top:bottom, left:right] = source[sample]
    return source, destination, boxes


def uhd_intro_raster(plate, source, destination, boxes, frame, fonts):
    from uhd_pixels import UHD_SIZE, blend_rgb16, blend_rgba8

    if frame >= FULL_SOURCE_LOCK:
        return source
    image = plate.copy()
    if frame >= PRELAP_START:
        for reference, (left, top, right, bottom), amount in prelap_regions(frame, boxes, 2):
            region = np.s_[top:bottom, left:right]
            blend_rgb16(image[region], (source, destination)[reference][region], amount)
    layer = title_layer(frame, fonts, UHD_SIZE)
    if layer is not None:
        left, top, right, bottom = bounds = layer.getbbox()
        blend_rgba8(image[top:bottom, left:right], layer.crop(bounds))
    return image


def uhd_source_profile(ffprobe, movie):
    from uhd_pixels import UHD_SIZE

    data = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
                                              "-of", "json", str(movie)]))
    stream, = data["streams"]
    require((stream["codec_name"], stream["width"], stream["height"], stream["avg_frame_rate"])
            == ("ffv1", *UHD_SIZE, "30/1") and stream["pix_fmt"] in {"gbrp16le", "rgb48le"},
            "UHD intro requires a native 30 fps lossless RGB16 FFV1 source")


def run_logged(command, path):
    with path.open("xb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def prores_packets(ffprobe, movie):
    data = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
                                              "-show_data_hash", "sha256", "-show_entries", "packet=data_hash",
                                              "-of", "json", str(movie)]))
    return [packet["data_hash"] for packet in data["packets"]]


def compose_uhd(args, source, plan, plates, fonts, custody):
    from intro_hq import SHARE_PROFILE, UHD_PROFILE, master_command, share_command, trim_command
    from uhd_pixels import UHD_SIZE, png_frames

    uhd_source_profile(args.ffprobe, args.demo)
    root = args.output_dir
    audio = root / "original-intro-audio.wav"
    audio_receipt = synthesize_audio(audio, sample_width=3)
    metadata = root / "chapters.ffmetadata"
    chapter_metadata(metadata, plan["chapters"])
    master = root / "JAIN-The-Next-Move-45s-4K.mov"
    standalone = root / "JAIN-The-Next-Move-Intro-12s-4K.mov"
    master_share = master.with_suffix(".mp4")
    intro_share = standalone.with_suffix(".mp4")
    commands = [master_command(args.ffmpeg, master, audio, metadata, MASTER_COUNT / FPS, args.prores_hardware),
                trim_command(args.ffmpeg, master, standalone, INTRO_COUNT / FPS),
                share_command(args.ffmpeg, master, master_share), share_command(args.ffmpeg, standalone, intro_share)]
    anchors = []
    paths, hashes = zip(*(plates[number] for number in range(1, PLATE_COUNT + 1)), strict=True)
    plate_frames = png_frames(paths, ffmpeg=args.ffmpeg, expected_sha256=hashes)
    demo_frames = decode_source(args.ffmpeg, args.demo, root / "source-decode.log", UHD_SIZE, uhd=True)
    with (ffmpeg_process(commands[0], root / "master-encode.log", writing=True) as sink,
          closing(plate_frames), closing(demo_frames)):
        first = next(demo_frames)
        reference, destination, boxes = uhd_reference(first, source["source_viewport"])
        for index, plate in zip(range(PLATE_COUNT), plate_frames, strict=True):
            raster = memoryview(uhd_intro_raster(plate, reference, destination, boxes, index, fonts)).cast("B")
            if index == FULL_SOURCE_LOCK:
                anchors.append({"master_frame": index, "source_frame": 0,
                                "input_rgb_sha256": hashlib.sha256(first).hexdigest(),
                                "assembled_rgb_sha256": hashlib.sha256(raster).hexdigest()})
            sink.write(raster)
        for index, (source_index, raster) in enumerate(zip(plan["selected_source_indices"], chain((first,), demo_frames), strict=True), PLATE_COUNT):
            if index in (PLATE_COUNT, INTRO_COUNT - 1, INTRO_COUNT, MASTER_COUNT - 1):
                sha = hashlib.sha256(raster).hexdigest()
                anchors.append({"master_frame": index, "source_frame": source_index,
                                "input_rgb_sha256": sha, "assembled_rgb_sha256": sha})
            sink.write(raster)
    require(digest(args.demo) == source["movie_sha256"], "UHD demo movie changed during assembly")
    require(digest(args.plates_receipt) == custody["intro_plates_receipt_sha256"]
            and digest(args.intro_scene_receipt) == custody["intro_scene_receipt_sha256"],
            "Intro scene or plate receipt changed during UHD assembly")
    require(all(anchor["input_rgb_sha256"] == anchor["assembled_rgb_sha256"] for anchor in anchors),
            "UHD handoff changed source RGB16 raster")
    for command, name in zip(commands[1:], ("intro-trim", "master-share", "intro-share"), strict=True):
        run_logged(command, root / f"{name}-encode.log")
    master_packets = prores_packets(args.ffprobe, master)
    intro_packets = prores_packets(args.ffprobe, standalone)
    require(len(master_packets) == MASTER_COUNT and intro_packets == master_packets[:INTRO_COUNT],
            "Standalone ProRes opener does not preserve exact master video packets")
    videos = {}
    for name, movie, frames, chapters, profile in (
        ("master", master, MASTER_COUNT, plan["chapters"], UHD_PROFILE),
        ("intro", standalone, INTRO_COUNT, [], UHD_PROFILE),
        ("master_share", master_share, MASTER_COUNT, plan["chapters"], SHARE_PROFILE),
        ("intro_share", intro_share, INTRO_COUNT, [], SHARE_PROFILE),
    ):
        videos[name] = verify_movie(args.ffmpeg, args.ffprobe, movie, frames, chapters,
                                    root / f"{name}-full-decode.log", size=UHD_SIZE, profile=profile)
    write_json(root / "intro-film.receipt.json", {
        "schema": "jain.intro-film-uhd/v1", "status": "PASS", "outputs": videos, "audio": audio_receipt,
        "width": UHD_SIZE[0], "height": UHD_SIZE[1], "fps": FPS, "raster_pixel_format": "rgb48le",
        "prores_hardware": args.prores_hardware, "standalone_matches_master_video_packets": True,
        "demo_sha256": source["movie_sha256"], "demo_receipt_sha256": digest(args.demo_receipt),
        **custody, "compositor_sha256": digest(Path(__file__)), "font_sha256": digest(args.font),
        "chapters_metadata_sha256": digest(metadata), "title_cues": TITLE_CUES,
        "source_prelap": {"panels_start_frame": PRELAP_START, "panels_ready_frame": PANEL_READY,
                          "editor_content_start_frame": CODE_REVEAL, "editor_content_ready_frame": CODE_READY,
                          "brand_and_benchmark_start_frame": BRAND_REVEAL, "full_source_lock_frame": FULL_SOURCE_LOCK,
                          "prelap_source_frame": 0, "first_moving_demo_frame": PLATE_COUNT},
        "retime": plan, "raster_handoff": anchors, "encoder_commands": commands,
    })
    print(master)


def preview_uhd(args, source, plates, fonts):
    from uhd_pixels import png_frames

    def raw_frame(index):
        return subprocess.check_output([args.ffmpeg, "-nostdin", "-v", "error", "-i", str(args.demo),
                                        "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
                                        "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"])

    reference, destination, boxes = uhd_reference(raw_frame(0), source["source_viewport"])
    for index in args.preview:
        require(0 <= index < INTRO_COUNT, "Preview only accepts a master frame inside the standalone intro")
        if index < PLATE_COUNT:
            path, expected = plates[index + 1]
            plate, = png_frames((path,), ffmpeg=args.ffmpeg, expected_sha256=(expected,))
            raster = memoryview(uhd_intro_raster(plate, reference, destination, boxes, index, fonts)).cast("B")
        else:
            raster = raw_frame(index - PLATE_COUNT)
        output = args.output_dir / f"preview-{index:06d}.png"
        subprocess.run([args.ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb48le",
                        "-video_size", "3840x2160", "-i", "pipe:0", "-frames:v", "1", "-pix_fmt", "rgb48be", str(output)],
                       input=raster, check=True)
        print(output)


def previews(args, source, plan, plates, fonts):
    needs_prelap = any(PRELAP_START <= frame < PLATE_COUNT for frame in args.preview)
    if needs_prelap:
        reference = source_image(args, 0)
        destination, content_boxes = editor_destination(reference, source["source_viewport"])
    for frame in args.preview:
        require(0 <= frame < INTRO_COUNT, "Preview only accepts a master frame inside the standalone intro")
        if frame < PLATE_COUNT:
            image = image_for_plate(plates, frame + 1)
            if frame >= PRELAP_START:
                image = prelap(image, reference, destination, content_boxes, frame)
            image = title_card(image, frame, fonts)
        else:
            source_frame = plan["selected_source_indices"][frame - PLATE_COUNT]
            image = source_image(args, source_frame)
        output = args.output_dir / f"preview-{frame:06d}.png"
        with output.open("xb") as stream:
            image.save(stream, format="PNG")
        print(output)


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
    parser.add_argument("--resolution", choices=("1080p", "2160p"), default="1080p")
    parser.add_argument("--prores-hardware", action="store_true", help="Use Apple VideoToolbox for the 2160p XQ master")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preview", type=int, action="append")
    args = parser.parse_args()
    uhd = args.resolution == "2160p"
    scale = 2 if uhd else 1
    size = WIDTH * scale, HEIGHT * scale
    require(not args.prores_hardware or uhd, "Hardware ProRes is only available for UHD delivery")
    source = read_demo(args.demo, args.demo_receipt, size)
    plan = plan_retime(source)
    if args.preflight:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return
    require(args.plates and args.plates_receipt and args.intro_scene_receipt and args.output_dir,
            "Render assembly requires --plates, --plates-receipt, --intro-scene-receipt, and --output-dir")
    args.output_dir = args.output_dir.resolve()
    checkout = Path(__file__).resolve().parents[2]
    require(not args.output_dir.is_relative_to(checkout), "Rendered outputs must stay outside the checkout")
    plates, custody = read_plates(args.plates, args.plates_receipt, args.intro_scene_receipt, source,
                                  full=not args.preview, size=size)
    fonts = {key: ImageFont.truetype(str(args.font), base * scale) for key, base in (("headline", 72), ("eyebrow", 22))}
    args.output_dir.mkdir(parents=True, exist_ok=bool(args.preview))
    if args.preview and uhd:
        preview_uhd(args, source, plates, fonts)
    elif args.preview:
        previews(args, source, plan, plates, fonts)
    elif uhd:
        compose_uhd(args, source, plan, plates, fonts, custody)
    else:
        compose(args, source, plan, plates, fonts, custody)


if __name__ == "__main__":
    main()
