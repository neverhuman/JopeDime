#!/usr/bin/env python3
"""Build a raised 17-FIT-row table HUD from verified CSV cells and GP sensitivity.

These transparent source-native overlays replace only the left table area of
existing render plates. They do not change neural activations, model weights,
source code, benchmark values or the floor. Every feature-cell score is the
verified fitted-GP FIT-median replacement sensitivity, never neural saliency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from input_influence import load_influence, read_json, sha

SCHEMA = "jain.table-panels/v1"
WIDTH, HEIGHT, BRAND_HEIGHT, ROWS = 611, 1080, 125, 17
BODY_FONTS = ("/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
HEADING_FONTS = ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def require(value, message):
    if not value:
        raise ValueError("table panels: " + message)


def font_path(explicit, candidates):
    path = Path(explicit) if explicit else next((Path(p) for p in candidates if Path(p).is_file()), None)
    require(path is not None and path.is_file(), "font unavailable; provide --font/--heading-font")
    return path.resolve()


def check_image(image):
    require(image.mode == "RGBA" and image.size == (WIDTH, HEIGHT), "RGBA611x1080 overlay required")
    alpha = image.getchannel("A")
    require(alpha.crop((0, 0, WIDTH, BRAND_HEIGHT)).getextrema() == (0, 0), "brand region must be transparent")
    require(alpha.crop((0, BRAND_HEIGHT, WIDTH, HEIGHT)).getextrema() == (255, 255), "old table region must be fully erased")


def draw_panel(name, total_rows, header, row_ids, raw_table, scores, body_path, heading_path):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, BRAND_HEIGHT, WIDTH - 1, HEIGHT - 1), fill=(0, 0, 0, 255))
    body = ImageFont.truetype(str(body_path), 17)
    heading = ImageFont.truetype(str(heading_path), 18)
    name_size = 18
    while heading.getlength(name) > 560 and name_size > 12:
        name_size -= 1
        heading = ImageFont.truetype(str(heading_path), name_size)
    require(heading.getlength(name) <= 560, "dataset name does not fit without truncation")
    ink, muted = (224, 233, 243, 255), (127, 153, 175, 255)
    draw.text((38, 145), name, font=heading, fill=ink, anchor="lt")
    draw.text((38, 177), f"{total_rows:,} rows", font=ImageFont.truetype(str(heading_path), 17), fill=ink, anchor="lt")
    draw.text((38, 207), "TABULAR INPUT", font=ImageFont.truetype(str(heading_path), 18), fill=ink, anchor="lt")
    draw.rounded_rectangle((33, 232, 610, 1039), radius=14, fill=(8, 16, 25, 255), outline=(50, 140, 177, 255), width=1)
    widths = [50.5, 101., 101., 101., 101., 101.]
    for row in range(ROWS + 1):
        y = 244 + row * 43.5
        values = ["row", *header] if row == 0 else [str(row_ids[row - 1]), *[
            f"{float(value):.2f}".replace("-", "−") for value in raw_table[row - 1]]]
        x = 45.
        for col, (width, value) in enumerate(zip(widths, values)):
            target = col == 5
            score = float(scores[row - 1][col - 1]) if row and 0 < col < 5 else 0.
            if row == 0:
                color = (141, 28, 167, 255) if target else (29, 48, 63, 255)
            elif target:
                color = (34, 18, 46, 255)
            elif col == 0:
                color = (12, 23, 34, 255)
            else:
                color = (5, round(13 + 27 * score), round(23 + 34 * score), 255)
            draw.rectangle((round(x), round(y), round(x + width - 1), round(y + 42.5)), fill=color)
            draw.text((x + width / 2, y + 21), value, font=body, fill=ink, anchor="mm")
            if row and 0 < col < 5 and score > 0:
                draw.line((round(x + 7), round(y + 39), round(x + 7 + (width - 14) * score), round(y + 39)),
                          fill=(20, 222, 245, 255), width=1)
            x += width
    draw.text((38, 1054), "RELATIVE INPUT SENSITIVITY", font=ImageFont.truetype(str(heading_path), 11), fill=muted, anchor="lt")
    check_image(image)
    return image, name_size


def make_panels(trace_path, influence_path, dataset_catalog_path, output_dir, *, font=None, heading_font=None):
    trace_path, influence_path, dataset_catalog_path, output_dir = [Path(p).resolve() for p in
        (trace_path, influence_path, dataset_catalog_path, output_dir)]
    require(not output_dir.exists() and not output_dir.is_relative_to(Path(__file__).resolve().parents[2]),
            "use a fresh external output directory")
    require(trace_path.stat().st_size <= 512 * 1024 * 1024, "trace exceeds bounded input")
    before = {"trace": sha(trace_path), "influence": sha(influence_path), "dataset_catalog": sha(dataset_catalog_path)}
    data = read_json(trace_path)
    influence = load_influence(influence_path, trace_path, data)
    metadata = read_json(dataset_catalog_path)
    require(metadata.get("schema") == "jain.animation-dataset-catalog/v1", "unsupported dataset metadata")
    cases = data["cases"]
    names = {case["case_id"]: case for case in metadata["cases"]}
    require(len(names) == len(metadata["cases"]) == len(cases) and set(names) == {case["id"] for case in cases},
            "dataset metadata case inventory differs")
    body_path, heading_path = font_path(font, BODY_FONTS), font_path(heading_font, HEADING_FONTS)
    prepared = []
    for case in cases:
        cid, item = case["id"], names[case["id"]]
        require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cid), "unsafe case ID")
        require(item["dataset_sha256"] == case["dataset_sha256"] and item["rows"] == case["rows"]
                and item["features"] == case["features"] == 4, "dataset metadata identity differs")
        name = item["display_name"]
        require(isinstance(name, str) and 1 <= len(name) <= 80 and not any(ord(c) < 32 for c in name), "invalid dataset name")
        record = influence["cases"][cid]
        require(len(record["raw_table"]) == len(record["scores"]) == len(record["row_ids"]) == ROWS
                and record["row_ids"] == case["fit_row_ids"][:ROWS], "require all17actualFITrows and all68scores")
        csv_ref = Path(record["bindings"]["csv_path"])
        csv_path = (trace_path.parent / csv_ref).resolve()
        require(not csv_ref.is_absolute() and csv_path.is_relative_to(trace_path.parent)
                and sha(csv_path) == case["dataset_sha256"], "CSV artifact identity differs")
        with csv_path.open(newline="") as stream:
            reader = csv.reader(stream)
            header, raw_rows = next(reader), list(reader)
        values = [[float(cell) for cell in raw_rows[row_id]] for row_id in record["row_ids"]]
        require(header == case["input_header"] and len(header) == 5 and len(raw_rows) == case["rows"]
                and values == record["raw_table"] and all(len(row) == 4 for row in record["scores"]),
                "visible values do not map to actual CSV rows")
        require(record["target_name"] == header[-1] and record["target_perturbed"] is False,
                "target mapping differs")
        prepared.append((case, item, record, header, values))
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for case, item, record, header, values in prepared:
        image, name_size = draw_panel(item["display_name"], case["rows"], header, record["row_ids"],
                                     values, record["scores"], body_path, heading_path)
        filename = case["id"] + ".png"
        with (output_dir / filename).open("xb") as stream:
            image.save(stream, format="PNG")
        result = {"case_id": case["id"], "file": filename, "sha256": sha(output_dir / filename),
                  "dataset_name": item["display_name"], "total_rows": case["rows"],
                  "dataset_sha256": case["dataset_sha256"], "displayed_rows": ROWS,
                  "row_ids": record["row_ids"], "feature_score_shape": [ROWS, 4],
                  "raw_table_sha256": record["hashes"]["raw_table"], "scores_sha256": record["hashes"]["scores"],
                  "target_name": header[-1], "target_display_index": 5,
                  "model_id": record["bindings"]["model_id"], "name_font_size": name_size,
                  "all_displayed_input_cells_have_measured_scores": True}
        results.append(result)
        print(json.dumps(result), flush=True)
    for label, path in (("trace", trace_path), ("influence", influence_path), ("dataset_catalog", dataset_catalog_path)):
        require(sha(path) == before[label], label + " changed during panel generation")
    catalog = {"schema": SCHEMA, "status": "PASS", "generator_sha256": sha(__file__),
               "trace": {"path": str(trace_path), "sha256": before["trace"]},
               "influence": {"path": str(influence_path), "sha256": before["influence"]},
               "dataset_catalog": {"path": str(dataset_catalog_path), "sha256": before["dataset_catalog"]},
               "width": WIDTH, "height": HEIGHT, "brand_transparent_rows": BRAND_HEIGHT,
               "displayed_rows": ROWS, "body_font_size": 17, "body_font_sha256": sha(body_path),
               "heading_font_sha256": sha(heading_path), "cases": results,
               "method": influence["method"], "value_format": "two decimal places; original row IDs; target last",
               "scope": "static final table HUD override; original NN/floor/metrics/source remain unchanged"}
    with (output_dir / "catalog.receipt.json").open("x") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    return catalog


def load_panels(root, scene):
    """Verify the final HUD override against frozen scene/dataset identities."""
    from PIL import Image
    root = Path(root).resolve()
    catalog_path = root / "catalog.receipt.json"
    catalog = read_json(catalog_path)
    require(catalog.get("schema") == SCHEMA and catalog.get("status") == "PASS"
            and catalog.get("width") == WIDTH and catalog.get("height") == HEIGHT
            and catalog.get("displayed_rows") == ROWS, "incomplete overlay catalog")
    require(catalog["trace"]["sha256"] == scene["trace_sha256"], "overlay belongs to another neural trace")
    require(catalog["dataset_catalog"]["sha256"] == scene["dataset_catalog_sha256"], "overlay dataset names differ")
    scene_ids = scene.get("case_ids") or [case["case_id"] for case in scene["tables"]]
    cases = {case["case_id"]: case for case in catalog["cases"]}
    require(len(cases) == len(catalog["cases"]) == len(scene_ids) and set(cases) == set(scene_ids), "overlay case inventory differs")
    influence_path, metadata_path = Path(catalog["influence"]["path"]), Path(catalog["dataset_catalog"]["path"])
    require(sha(influence_path) == catalog["influence"]["sha256"] and sha(metadata_path) == catalog["dataset_catalog"]["sha256"],
            "overlay verification inputs changed")
    influence, metadata = read_json(influence_path), read_json(metadata_path)
    require(influence["trace"]["sha256"] == scene["trace_sha256"], "overlay sensitivity trace differs")
    names = {case["case_id"]: case for case in metadata["cases"]}
    images = {}
    for cid in scene_ids:
        case, record, item = cases[cid], influence["cases"][cid], names[cid]
        require(case["dataset_sha256"] == record["bindings"]["dataset_sha256"] == item["dataset_sha256"]
                and case["dataset_name"] == item["display_name"] and case["total_rows"] == item["rows"]
                and case["displayed_rows"] == ROWS and case["row_ids"] == record["row_ids"]
                and case["scores_sha256"] == record["hashes"]["scores"]
                and case["raw_table_sha256"] == record["hashes"]["raw_table"], "overlay row/score/dataset binding differs")
        ref = Path(case["file"])
        path = (root / ref).resolve()
        require(not ref.is_absolute() and path.is_relative_to(root) and sha(path) == case["sha256"], "overlay PNG identity differs")
        with Image.open(path) as handle:
            check_image(handle)
            images[cid] = handle.copy()
    summary = {"catalog_sha256": sha(catalog_path), "trace_sha256": catalog["trace"]["sha256"],
               "influence_sha256": catalog["influence"]["sha256"], "dataset_catalog_sha256": catalog["dataset_catalog"]["sha256"],
               "final_displayed_rows": ROWS, "original_scene_tables": scene["tables"],
               "scope": catalog["scope"], "cases": catalog["cases"]}
    return images, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace", "influence", "dataset-catalog", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--heading-font", type=Path)
    args = parser.parse_args()
    result = make_panels(args.trace, args.influence, args.dataset_catalog, args.output_dir,
                         font=args.font, heading_font=args.heading_font)
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "receipt": str(args.output_dir.resolve() / "catalog.receipt.json")}))


if __name__ == "__main__":
    main()
