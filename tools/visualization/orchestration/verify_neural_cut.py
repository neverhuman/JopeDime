"""Check captured Dendrite source, sensitivity colors and decoded film metadata."""
import argparse
from common import REPO, external_path, validate_configuration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="External variable-attribution or QA-binding root")
    args = parser.parse_args()
    validate_configuration()
    import hashlib
    import json
    from pathlib import Path
    import sys
    import zipfile

    import numpy as np
    from PIL import Image

    worktree, root = REPO, external_path(args.root)
    sys.path.insert(0, str(worktree / "tools/visualization"))
    from code_pages import reconstruct_source
    from intelligence_wave import LANGUAGES, load_icon, shuffled_tables
    from table_panels import BODY_FONTS, HEADING_FONTS, draw_panel, font_path


    def load(path):
        return json.loads(path.read_bytes())


    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()


    capture = root / "capture"
    out = capture / "final"
    sources = capture / "multilingual-source"
    catalog_path = sources / "catalog.json"
    source_catalog = load(catalog_path)
    encode = load(out / "encode.receipt.json")
    probe = load(out / "ffprobe.json")
    scene = load(capture / "scene/scene.receipt.json")
    stream, = probe["streams"]
    assert (stream["width"], stream["height"], stream["nb_read_frames"], stream["avg_frame_rate"]) == (
        1920, 1080, "1260", "30/1")
    assert stream["r_frame_rate"] == "30/1" and stream["codec_name"] == "h264"
    assert float(probe["format"]["duration"]) == 42 and not (out / "decode.log").read_bytes()
    assert encode["movie_sha256"] == sha(Path(encode["movie"]))
    assert encode["recipe_catalog_sha256"] == sha(catalog_path)
    assert source_catalog["trace_sha256"] == scene["trace_sha256"]
    assert encode["generator_sha256"] == sha(worktree / "tools/visualization/intelligence_wave.py")
    assert len(probe["chapters"]) == len(encode["cases"]) == len(source_catalog["cases"]) == 10
    assert {file["language"] for case in encode["cases"] for file in case["files"]} == set(LANGUAGES)
    assert [case["total_rows"] for case in encode["cases"]] == [64, 72, 80, 88, 104, 112, 120, 128, 144, 160]
    panel_root = capture / "table-panels"
    table_catalog = load(panel_root / "catalog.receipt.json")
    table_cases = {case["case_id"]: case for case in table_catalog["cases"]}
    neural_path = capture / "neural-influence.json"
    neural = load(neural_path)
    assert neural["trace_sha256"] == scene["trace_sha256"]
    assert neural["weights_before"] == neural["weights_after"]
    assert neural["interventions_per_case"] == 85 and neural["case_count"] == 10
    assert encode["table_overlay"]["neural_cell_influence_sha256"] == sha(neural_path)
    influences = neural["cases"]
    rendered_panels, panel_orders, _ = shuffled_tables(panel_root, scene, neural_path)
    ranked_by_case = {}
    shares_by_case = {}
    for cid, record in influences.items():
        scores = record["scores"]
        flat = [score for row in scores for score in row]
        assert len(flat) == 85
        ranked_by_case[cid] = [
            [
                (sum(other < score for other in flat) + (sum(other == score for other in flat) - 1) / 2) / 84
                if score else 0.
                for score in row
            ]
            for row in scores
        ]
        means = [sum(row[column] for row in scores) / len(scores) for column in range(4)]
        shares_by_case[cid] = [mean / sum(means) if sum(means) else 0 for mean in means]


    def expected_fill(rank, *, target=False):
        low, middle, high = (((34, 18, 46), (68, 35, 94), (101, 53, 141)) if target
                             else ((69, 23, 42), (91, 61, 29), (13, 84, 60)))
        if rank <= .5:
            return (*(round(a + (b - a) * (2 * rank)) for a, b in zip(low, middle)), 255)
        return (*(round(a + (b - a) * (2 * rank - 1)) for a, b in zip(middle, high)), 255)


    width, height = encode["source_viewport"][2:]
    checked = []
    archive_path = sources / "JAIN-multilingual-recipe-sources.zip"
    with zipfile.ZipFile(archive_path) as archive:
        for index, (case, recipe, chapter) in enumerate(zip(
                encode["cases"], source_catalog["cases"], probe["chapters"], strict=True)):
            cid = case["case_id"]
            assert cid == recipe["case_id"] and case["model_id"] == recipe["model_id"]
            assert case["start"] == 126 * index and case["end"] == 126 * (index + 1)
            assert abs(float(chapter["start_time"]) - index * 4.2) < 1e-9
            assert abs(float(chapter["end_time"]) - (index + 1) * 4.2) < 1e-9
            assert chapter["tags"]["title"] == case["dataset_name"]
            assert set(case["plate_indices"]) == set(range(120))
            offsets, line_height = case["scroll_offsets_px"], case["line_height_px"]
            assert offsets == sorted(offsets) and offsets[0] == 0
            assert len(offsets) == 126 and offsets[-1] == max(0, case["visual_rows"] * line_height - height)
            for row in range(case["visual_rows"]):
                assert any(top <= row * line_height and top + height >= (row + 1) * line_height
                           for top in offsets[7:])
            for file_index, (file, entry) in enumerate(zip(case["files"], recipe["files"], strict=True)):
                manifest_path = out / "code-pages" / cid / str(file_index) / "manifest.json"
                manifest = load(manifest_path)
                raw = (sources / entry["source_path"]).read_bytes()
                assert reconstruct_source(manifest) == raw == archive.read(entry["source_path"])
                assert file["source_sha256"] == entry["source_sha256"] == hashlib.sha256(raw).hexdigest()
                assert file["manifest_sha256"] == sha(manifest_path)
                assert manifest.get("language", "rust") == file["language"] == entry["language"]
                assert "Pygments" in manifest["lexer"]
                assert all(sha(manifest_path.parent / page["file"]) == page["sha256"] for page in manifest["pages"])
            assert len(case["cues"]) == sum(len(op["spans"]) for file in recipe["files"] for op in file["operations"])
            for cue in case["cues"]:
                ops = recipe["files"][cue["file_index"]]["operations"]
                op = next(op for op in ops if op["ordinal"] == cue["ordinal"])
                assert cue["bytes"] == op["spans"][int(cue["phase"] == "predict")]
                assert cue["operation"] == op["label"]
            original = table_cases[cid]["row_ids"]
            orders = case["table_row_orders"]
            assert [order["case_frame"] for order in orders] == [
                (42 * index + 2) // 5 for index in range(15)
            ]
            assert orders[0]["row_ids"] == original
            assert len({tuple(order["row_ids"]) for order in orders}) == 15
            assert all(sorted(order["row_ids"]) == sorted(original) for order in orders)
            assert orders == panel_orders[cid]
            for panel, order in zip(rendered_panels[cid], orders, strict=True):
                for slot, row_id in enumerate(order["row_ids"], start=1):
                    y = round(244 + slot * 43.5) + 3
                    ranks = ranked_by_case[cid][original.index(row_id)]
                    for x, rank in zip((101, 202, 303, 404), ranks[:4], strict=True):
                        assert panel.getpixel((x, y)) == expected_fill(rank)
                    assert panel.getpixel((505, y)) == expected_fill(ranks[4], target=True)
                shares = shares_by_case[cid]
                for x, share in zip((101, 202, 303, 404), shares, strict=True):
                    assert panel.getpixel((x, 248)) == expected_fill(share / max(shares) if max(shares) else 0)
                assert panel.getpixel((505, 248)) == (141, 28, 167, 255)
                assert panel.crop((0, 1040, 611, 1080)).convert("RGB").getbbox() is None
            assert case["language_icon_sha256"] == load_icon(case["primary_language"])[1]
            assert all(p["status"] == "PASS" for p in recipe["parity"])
            checked.append({"case_id": cid, "language": case["primary_language"],
                            "files": [{key: file[key] for key in ("filename", "language", "source_sha256")}
                                      for file in case["files"]],
                            "row_orders": orders, "target_execution": recipe["target_execution"]})

    body, heading = font_path(None, BODY_FONTS), font_path(None, HEADING_FONTS)
    for preview in encode["previews"]:
        path = out / preview["path"]
        assert sha(path) == preview["sha256"]
        frame = preview["frame"]
        case = encode["cases"][frame // 126]
        record = influences[case["case_id"]]
        order = max((item for item in case["table_row_orders"] if item["case_frame"] <= frame % 126),
                    key=lambda item: item["case_frame"])["row_ids"]
        indexes = [record["row_ids"].index(row_id) for row_id in order]
        panel, _ = draw_panel(case["dataset_name"], case["total_rows"],
                              [*record["feature_names"], record["target_name"]], order,
                              [record["raw_table"][i] for i in indexes],
                              [record["scores"][i] for i in indexes], body, heading,
                              percentiles=[ranked_by_case[case["case_id"]][i] for i in indexes],
                              column_shares=shares_by_case[case["case_id"]])
        active_row = case["scroll_offsets_px"][frame % 126] // case["line_height_px"]
        file = max((file for file in case["files"] if max(0, file["visual_start"] - 2) <= active_row),
                   key=lambda file: file["visual_start"])
        icon_language = file["language"] if file["language"] in case["language_icon_hashes"] else case["primary_language"]
        icon, _ = load_icon(icon_language)
        rgb_icon = np.asarray(icon)
        solid = rgb_icon[:, :, 3] == 255
        assert solid.any()
        with Image.open(path) as still:
            assert np.array_equal(np.asarray(still.crop((0, 125, 611, 1080))),
                                  np.asarray(panel.convert("RGB").crop((0, 125, 611, 1080))))
            x = 960 - icon.width // 2
            icon_pixels = np.asarray(still.crop((x, 895, x + icon.width, 895 + icon.height)))
            assert np.array_equal(icon_pixels[solid], rgb_icon[:, :, :3][solid])

    receipt = {
        "schema": "jain.intelligence-wave-neural-review/v1", "status": "PASS",
        "movie_sha256": encode["movie_sha256"], "encode_receipt_sha256": sha(out / "encode.receipt.json"),
        "ffprobe_sha256": sha(out / "ffprobe.json"), "decoded_frames": 1260, "fps": 30,
        "duration_seconds": 42, "width": 1920, "height": 1080, "chapters": 10,
        "language_count": len(LANGUAGES), "source_catalog_sha256": sha(catalog_path),
        "source_archive_sha256": sha(archive_path), "preview_count": len(encode["previews"]),
        "full_source_scroll": True, "verified_row_and_icon_pixels_in_previews": True,
        "table_hold_seconds_mean": .28, "orders_per_case": 15, "interior_shuffles_per_case": 14,
        "verified_cell_colors": 10 * 15 * 85,
        "verified_feature_headers": 10 * 15 * 4,
        "attribution_sha256": sha(neural_path),
        "cases": checked,
    }
    with (external_path(out / "review.receipt.json")).open("x") as file:
        json.dump(receipt, file, indent=2)
        file.write("\n")
    print({key: value for key, value in receipt.items() if key != "cases"})


if __name__ == "__main__":
    main()
