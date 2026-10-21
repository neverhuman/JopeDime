#!/usr/bin/env python3
"""Render an explicitly synthetic terminal concept; never executes the shown commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

BASE_SHA = "45490f2ce33075a355e08844351db197001beb90"
SIZE = (1200, 780)
COLORS = {
    "bg": "#0b1018",
    "panel": "#101925",
    "rule": "#29394b",
    "text": "#e0e8f1",
    "muted": "#92a4b8",
    "cyan": "#58d3e5",
    "green": "#7ce2a2",
    "amber": "#efc36a",
    "red": "#f0818e",
}
SOURCES = {
    "Rust": [
        "// illustrative recipe fragment",
        "fn predict(x: &[f64]) -> f64 {",
        "    let signal = 0.72 * x[0] + 0.18 * x[1];",
        "    signal.tanh() + 0.04",
        "}",
    ],
    "Python": [
        "# illustrative recipe fragment",
        "import math",
        "def predict(x):",
        "    signal = 0.72 * x[0] + 0.18 * x[1]",
        "    return math.tanh(signal) + 0.04",
    ],
    "C++17": [
        "// illustrative recipe fragment",
        "#include <cmath>",
        "double predict(const double* x) {",
        "    return std::tanh(0.72*x[0] + 0.18*x[1]) + 0.04;",
        "}",
    ],
    "JavaScript": [
        "// illustrative recipe fragment",
        "export function predict(x) {",
        "    const signal = 0.72 * x[0] + 0.18 * x[1];",
        "    return Math.tanh(signal) + 0.04;",
        "}",
    ],
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(font: ImageFont.FreeTypeFont, bold: ImageFont.FreeTypeFont):
    """Yield frames with explicit durations; the two scenes depict distinct runs."""
    frames = []
    durations = []

    def canvas(scene: str):
        im = Image.new("RGB", SIZE, COLORS["bg"])
        draw = ImageDraw.Draw(im)

        def text(x, y, value, color="text", strong=False):
            draw.text((x, y), value, font=bold if strong else font, fill=COLORS[color])

        draw.rounded_rectangle((20, 16, 1180, 762), radius=12, outline=COLORS["rule"])
        draw.rectangle((21, 50, 1179, 87), fill=COLORS["panel"])
        text(40, 24, "JOPEDIME  /  TERMINAL CONCEPT", "cyan", True)
        text(40, 57, "DESIGN PREVIEW  |  SYNTHETIC DATA  |  PROPOSED FLAGS", "amber")
        text(40, 102, scene, "text", True)
        draw.line((40, 139, 1160, 139), fill=COLORS["rule"])
        return im, draw, text

    def append(im, ms=650):
        # Fixed palette across all frames; quantize without random dithering.
        palette = Image.new("P", (1, 1))
        rgb = [tuple(bytes.fromhex(value[1:])) for value in COLORS.values()]
        # Add fixed intermediate shades to retain readable antialiased text.
        for fg in list(rgb):
            for weight in (0.25, 0.5, 0.75):
                rgb.append(tuple(round(a * weight + b * (1 - weight)) for a, b in zip(fg, rgb[0])))
        flat = [v for color in rgb for v in color]
        palette.putpalette((flat + [0] * 768)[:768])
        frames.append(im.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(ms)

    for step in range(14):
        im, draw, text = canvas("01  FIT CSV  /  native appliance")
        text(40, 158, "$ jopedime fit-predict --train train.csv --target y")
        text(64, 189, "--input test.csv --output predictions.csv --ui progress", "muted")
        text(40, 245, "INPUT    12,000 rows  /  24 features  /  regression", "muted")
        text(40, 278, "BUDGET   CPU  /  threads 8 requested, 8 effective", "muted")
        if step == 0:
            phase, detail, count, pct = "VERIFY / READ", "Checking package and CSV schema", None, 0
        elif step < 10:
            count = step
            phase, detail, pct = "SELECT", "Training CV  /  RMSE minimize  /  best 0.482", count / 9
        elif step == 10:
            phase, detail, count, pct = "REFIT", "Selection complete; fitting all training rows", None, None
        elif step == 11:
            phase, detail, count, pct = "PREDICT", "Writing 3,000 prediction rows", None, 0.7
        elif step == 12:
            phase, detail, count, pct = "COMMIT OUTPUTS", "Finalizing predictions.csv and run.json", None, None
        else:
            phase, detail, count, pct = "SUCCEEDED", "Predictions and receipt committed", None, 1
        text(40, 344, phase, "green" if step == 13 else "cyan", True)
        text(40, 384, detail)
        draw.rounded_rectangle((40, 438, 1160, 458), radius=6, fill=COLORS["rule"])
        if pct is not None and pct > 0:
            draw.rounded_rectangle((40, 438, 40 + int(1120 * pct), 458), radius=6,
                                   fill=COLORS["green" if step == 13 else "cyan"])
        else:
            text(40, 472, "Completion estimate unavailable in this phase", "muted")
        if count is not None:
            text(40, 480, f"FOLDS  {count}/9 complete  |  failed 0  |  elapsed {count * 3:02d}s")
            eta = f"{(9-count)*2}-{(9-count)*4}s" if 3 <= count < 9 else "estimating" if count < 3 else "complete"
            text(40, 517, f"Selection ETA  {eta}   /   refit has a separate estimate", "muted")
        elif step == 13:
            text(40, 495, "OUTPUT   predictions.csv   |   RECEIPT   run.json", "green")
            text(40, 532, "Exit 0  /  source artifacts: not provided by this method", "muted")
        text(40, 610, "One compact live view. No per-fold log flood.", "muted")
        text(40, 645, "JSON stdout stays clean; human progress uses stderr.", "muted")
        draw.line((40, 704, 1160, 704), fill=COLORS["rule"])
        text(40, 724, "Concept only. Current runnable commands: docs/terminal/quickstart.md", "amber")
        append(im, 1700 if step in (0, 13) else 500)

    for step in range(12):
        language = list(SOURCES)[(step // 2) % 4]
        occurrence = 42 if step < 8 else 43
        status = "PENDING" if step % 2 == 0 else "PASSED"
        im, draw, text = canvas("02  SOURCE WATCH  /  separate GP search run")
        text(40, 156, "$ jope source watch --ui tui --show-code latest", "text")
        text(40, 193, "WS 127.0.0.1:9876  /  connected  /  replay cursor retained", "muted")
        text(40, 246, f"occurrence demo-{occurrence:04d}   |   sequence {100+step:03d}   |   FOLLOW", "cyan")
        text(40, 282, "1 Rust    2 Python    3 C++17    4 JavaScript", "muted")
        draw.rounded_rectangle((40, 323, 1160, 571), radius=7,
                               fill=COLORS["panel"], outline=COLORS["rule"])
        text(60, 338, language, "cyan", True)
        text(310, 338, "CHECK " + status, "amber" if status == "PENDING" else "green")
        for i, line in enumerate(SOURCES[language]):
            text(60, 385 + i * 32, f"{i+1:2d}", "muted")
            text(112, 385 + i * 32, line, "muted" if i == 0 else "text")
        text(40, 596, "Source first, compiler check next. Snippets/status are synthetic.", "muted")
        text(40, 632, "Every occurrence stays in the source log; only this pane is replaced.", "muted")
        text(40, 668, "[space] pause display   [f] follow   [1-4] language   [q] detach", "cyan")
        draw.line((40, 704, 1160, 704), fill=COLORS["rule"])
        text(40, 724, "Concept only. Existing transport: jope.gp-source/v1 WebSocket", "amber")
        append(im, 1600 if step in (1, 7, 11) else 700)
    return frames, durations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--output", type=Path, default=here / "terminal-preview.gif")
    parser.add_argument("--still", type=Path)
    parser.add_argument("--font", type=Path,
                        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
    parser.add_argument("--bold-font", type=Path,
                        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"))
    args = parser.parse_args()
    font = ImageFont.truetype(str(args.font), 22)
    bold = ImageFont.truetype(str(args.bold_font), 22)
    frames, durations = render(font, bold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.output, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    if args.output.stat().st_size > 1024 * 1024:
        raise RuntimeError("preview exceeds the 1 MiB repository asset budget")
    if args.still:
        frames[19].convert("RGB").save(args.still)
    manifest = {
        "schema": "jope.terminal-preview/v1",
        "status": "design preview; synthetic values; proposed UI is not implemented",
        "source_base_sha40": BASE_SHA,
        "scenes": ["native CSV fit", "separate GP source viewer"],
        "dimensions": list(SIZE),
        "frames": len(frames),
        "durations_ms": durations,
        "loop": 0,
        "pillow_version": pillow_version,
        "font_sha256": digest(args.font),
        "bold_font_sha256": digest(args.bold_font),
        "generator_sha256": digest(Path(__file__)),
        "gif_sha256": digest(args.output),
        "gif_bytes": args.output.stat().st_size,
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(frames), "bytes": manifest["gif_bytes"],
                      "sha256": manifest["gif_sha256"]}))


if __name__ == "__main__":
    main()
