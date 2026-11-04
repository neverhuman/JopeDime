#!/usr/bin/env python3
"""Measure local fitted-GP sensitivity of the five visible table rows.

Each feature cell is replaced, alone, by that feature's median over the original
FIT rows. Scores are absolute prediction differences, normalized by the largest
difference in that case. Targets are never inputs or perturbed. This is local
fitted-GP sensitivity, not neural saliency or a causal feature attribution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import struct
import types

SCHEMA = "jain.input-cell-influence/v1"
METHOD = "fitted-gp-fit-median-replacement/v1"
ABS_TOL, REL_TOL = 1e-12, 1e-10


def require(condition, message):
    if not condition:
        raise ValueError("input sensitivity: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def cells_hash(values, dtype="d"):
    flat = []
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        else:
            number = float(value)
            require(math.isfinite(number), "nonfinite numeric cell")
            flat.append(number)
    visit(values)
    return hashlib.sha256(b"".join(struct.pack("<" + dtype, value) for value in flat)).hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def safe_file(root, relative):
    name = Path(relative)
    require(not name.is_absolute(), "artifact path must be relative")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "artifact missing or escapes proof root")
    return path


def matrix(values, rows, columns, label):
    require(isinstance(values, list) and len(values) == rows
            and all(isinstance(row, list) and len(row) == columns for row in values), label + " shape differs")
    require(all(type(v) in (int, float) and math.isfinite(v) for row in values for v in row), label + " has invalid cells")


def parity(actual, expected, label):
    require(len(actual) == len(expected) and all(math.isfinite(float(v)) for v in actual + expected), label + " has invalid predictions")
    errors = [abs(float(a) - float(b)) for a, b in zip(actual, expected)]
    require(all(error <= ABS_TOL + REL_TOL * abs(float(expected[i])) for i, error in enumerate(errors)), label + " prediction parity failed")
    return max(errors, default=0.)


def context(case, root):
    """Read only hash-bound source, model, CSV and original row partition."""
    source = case["source"]
    require(isinstance(source, dict), "case has no fitted-GP source proof")
    proof_path = safe_file(root, source["proof_path"])
    require(sha(proof_path) == source["proof_sha256"], "GP proof identity differs")
    proof = read_json(proof_path)
    require(proof["state"] == "pass" and proof["source_unchanged"] is True, "GP source proof is incomplete")
    directory = proof_path.parent
    python = directory / "predictor.py"
    rust = safe_file(root, source["rust_path"])
    require(sha(python) == source["python_sha256"] == proof["source_sha256"]["python"]
            and sha(rust) == source["rust_sha256"] == proof["source_sha256"]["rust"], "original source changed")
    csv_path, split_path = directory / "input.csv", directory / "split.json"
    require(sha(csv_path) == case["dataset_sha256"] == proof["input_csv"]["sha256"], "CSV identity differs")
    require(sha(split_path) == proof["split"]["sha256"], "split artifact changed")
    split = read_json(split_path)
    fit, hold = case["fit_row_ids"], case["holdout_row_ids"]
    require(split["fit"] == proof["split"]["fit"] == fit
            and split["holdout"] == proof["split"]["holdout"] == hold, "FIT/HOLD row identities differ")
    with csv_path.open(newline="") as stream:
        reader = csv.reader(stream)
        header, rows = next(reader), list(reader)
    require(case["features"] == 4 and header == case["input_header"] and len(header) == 5
            and len(set(header)) == 5 and len(rows) == case["rows"], "bounded numeric table contract differs")
    require(all(len(row) == 5 and all(math.isfinite(float(v)) for v in row) for row in rows), "table is not complete finite numeric data")
    require(fit and hold and len(fit) == len(set(fit)) and len(hold) == len(set(hold))
            and not set(fit) & set(hold) and set(fit) | set(hold) == set(range(len(rows))), "invalid row partition")
    visible_ids = case["table_preview_row_ids"]
    raw_table = case["table_preview"]
    matrix(raw_table, 5, 5, "visible table")
    require(len(visible_ids) == 5 and len(set(visible_ids)) == 5 and set(visible_ids) <= set(fit), "preview is not five actual FIT rows")
    actual_table = [[float(v) for v in rows[i]] for i in visible_ids]
    require(cells_hash(actual_table, "f") == cells_hash(raw_table, "f"), "preview does not map to exact CSV rows at native FP32 precision")
    model_id = source["model_id"]
    require(model_id == proof["model"]["model_id"], "fitted model identity differs")
    model_path = directory / "models" / model_id
    manifest_path = model_path / "manifest.json"
    require("sha256-" + sha(manifest_path) == model_id, "fitted model manifest changed")
    manifest = read_json(manifest_path)
    for part in manifest["files"]:
        path = safe_file(model_path, part["path"])
        require(sha(path) == part["sha256"] and path.stat().st_size == part["bytes"], "fitted model payload changed")
    require(read_json(model_path / "pipeline.json")["dsl"] == source["selected_program"]
            == case["selection"]["selected_program"], "model is not the actual selected GP")
    module = types.ModuleType("_jain_verified_gp_" + case["id"].replace("-", "_"))
    module.__file__ = str(python)
    exec(compile(python.read_bytes(), str(python), "exec"), module.__dict__)
    features, target = header[:4], header[4]
    require(set(module.MODEL_COLUMNS) == set(features) and len(module.MODEL_COLUMNS) == 4
            and target not in module.MODEL_COLUMNS, "predictor feature/target mapping differs")
    model_columns = list(module.MODEL_COLUMNS)
    column_indices = [header.index(name) for name in model_columns]
    model_rows = [[row[j] for j in column_indices] for row in rows]
    medians = [float(statistics.median(float(rows[i][j]) for i in fit)) for j in range(4)]
    require(all(math.isfinite(v) for v in medians), "FIT median overflow")
    native_path = directory / "native-predictions.json"
    require(sha(native_path) == proof["native_predictions_sha256"], "native prediction reference changed")
    native = read_json(native_path)
    py_output = directory / "proof/python-predictions.stdout"
    py_receipt = read_json(directory / "proof/python-predictions.receipt.json")
    require(py_receipt["exit_code"] == 0 and sha(py_output) == py_receipt["stdout_sha256"], "existing Python prediction proof changed")
    existing = read_json(py_output)
    require(len(native) == len(existing) == len(rows), "baseline reference row count differs")
    bindings = {"dataset_sha256": case["dataset_sha256"], "csv_path": str(csv_path.relative_to(root)),
        "csv_sha256": sha(csv_path), "split_path": str(split_path.relative_to(root)), "split_sha256": sha(split_path),
        "fit_row_ids_sha256": identity_hash(fit), "holdout_row_ids_sha256": identity_hash(hold),
        "proof_path": source["proof_path"], "proof_sha256": source["proof_sha256"],
        "python_path": str(python.relative_to(root)), "python_sha256": source["python_sha256"],
        "rust_sha256": source["rust_sha256"], "model_id": model_id, "model_manifest_sha256": sha(manifest_path),
        "native_predictions_sha256": sha(native_path), "existing_python_predictions_sha256": sha(py_output),
        "model_columns": model_columns}
    return {"module": module, "rows": model_rows, "native": native, "existing": existing,
            "visible_ids": visible_ids, "raw_table": raw_table, "features": features, "target": target,
            "medians": medians, "bindings": bindings}


def perturbations(ctx):
    rows = []
    for row_id in ctx["visible_ids"]:
        for feature, median in zip(ctx["features"], ctx["medians"]):
            changed = list(ctx["rows"][row_id])
            changed[ctx["bindings"]["model_columns"].index(feature)] = median
            rows.append(changed)
    return rows


def score_arrays(baseline, perturbed):
    matrix(perturbed, 5, 4, "perturbed predictions")
    require(len(baseline) == 5 and all(math.isfinite(v) for v in baseline), "visible baseline differs")
    deltas = [[abs(perturbed[i][j] - baseline[i]) for j in range(4)] for i in range(5)]
    require(all(math.isfinite(v) for row in deltas for v in row), "prediction difference overflow")
    maximum = max(v for row in deltas for v in row)
    scores = [[v / maximum if maximum > 0. else 0. for v in row] for row in deltas]
    return deltas, maximum, scores


def build_case(case, root):
    ctx = context(case, root)
    baseline_all = list(ctx["module"].predict_rows(ctx["rows"]))
    native_error = parity(baseline_all, ctx["native"], "all-row native baseline")
    python_error = parity(baseline_all, ctx["existing"], "existing all-row Python baseline")
    baseline = [baseline_all[i] for i in ctx["visible_ids"]]
    predictions = list(ctx["module"].predict_rows(perturbations(ctx)))
    require(len(predictions) == 20, "perturbed prediction row count differs")
    perturbed = [predictions[i * 4:(i + 1) * 4] for i in range(5)]
    deltas, maximum, scores = score_arrays(baseline, perturbed)
    record = {"case_id": case["id"], "row_ids": ctx["visible_ids"], "feature_names": ctx["features"],
        "feature_column_indices": [0, 1, 2, 3], "target_name": ctx["target"], "target_column_index": 4,
        "raw_table": ctx["raw_table"], "fit_medians": ctx["medians"], "baseline_predictions": baseline,
        "perturbed_predictions": perturbed, "absolute_deltas": deltas, "maximum_absolute_delta": maximum,
        "scores": scores, "bindings": ctx["bindings"], "target_perturbed": False,
        "baseline_parity": {"rows": len(baseline_all), "max_abs_error_native": native_error,
            "max_abs_error_existing_python": python_error, "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL}}
    record["hashes"] = {key: cells_hash(record[key], "f" if key == "raw_table" else "d")
        for key in ("raw_table", "fit_medians", "baseline_predictions", "perturbed_predictions", "absolute_deltas", "scores")}
    return record


def load_influence(path, trace_path, data=None):
    """Strict scene loader; recheck bindings and the 25 relevant GP predictions per case."""
    path, trace_path = Path(path), Path(trace_path)
    require(path.stat().st_size <= 4 * 1024 * 1024, "influence artifact exceeds bound")
    influence = read_json(path)
    require(influence["schema"] == SCHEMA and influence["method"]["id"] == METHOD
            and influence["method"]["normalization"] == "per-case maximum absolute delta; all-zero case stays zero"
            and influence["method"]["neural_saliency"] is False, "sensitivity method/scope differs")
    require(influence["trace"]["sha256"] == sha(trace_path), "influence belongs to another trace")
    data = read_json(trace_path) if data is None else data
    cases = {case["id"]: case for case in data["cases"]}
    require(set(influence["cases"]) == set(cases), "influence case inventory differs")
    for cid, case in cases.items():
        record, ctx = influence["cases"][cid], context(case, trace_path.parent)
        require(record["case_id"] == cid and record["bindings"] == ctx["bindings"], "source/model/FIT/CSV binding differs")
        require(record["row_ids"] == ctx["visible_ids"] and record["raw_table"] == ctx["raw_table"]
                and record["feature_names"] == ctx["features"] and record["feature_column_indices"] == [0, 1, 2, 3]
                and record["target_name"] == ctx["target"] and record["target_column_index"] == 4
                and record["target_perturbed"] is False and record["fit_medians"] == ctx["medians"], "cell/reference mapping differs")
        predicted = list(ctx["module"].predict_rows([ctx["rows"][i] for i in ctx["visible_ids"]] + perturbations(ctx)))
        require(len(predicted) == 25, "verification prediction count differs")
        parity(record["baseline_predictions"], predicted[:5], "visible baseline")
        parity(predicted[:5], [ctx["native"][i] for i in ctx["visible_ids"]], "visible native reference")
        matrix(record["perturbed_predictions"], 5, 4, "perturbed predictions")
        parity([v for row in record["perturbed_predictions"] for v in row], predicted[5:], "actual perturbation")
        deltas, maximum, scores = score_arrays(record["baseline_predictions"], record["perturbed_predictions"])
        require(record["absolute_deltas"] == deltas and record["maximum_absolute_delta"] == maximum
                and record["scores"] == scores, "sensitivity/normalization arithmetic differs")
        for key, digest in record["hashes"].items():
            require(key in {"raw_table", "fit_medians", "baseline_predictions", "perturbed_predictions", "absolute_deltas", "scores"}
                    and cells_hash(record[key], "f" if key == "raw_table" else "d") == digest, "numeric cell hash differs")
        require(set(record["hashes"]) == {"raw_table", "fit_medians", "baseline_predictions", "perturbed_predictions", "absolute_deltas", "scores"},
                "numeric hash inventory incomplete")
        matrix(record["scores"], 5, 4, "normalized sensitivity")
        require(all(0. <= v <= 1. for row in record["scores"] for v in row), "sensitivity outside 0..1")
    return influence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.output_dir.is_absolute() and not args.output_dir.exists(), "output must be a new absolute external directory")
    require(args.trace.stat().st_size <= 512 * 1024 * 1024, "trace exceeds bounded input")
    data = read_json(args.trace)
    require(data["schema"] == "jain.h256-stage2-capture/v1" and 1 <= len(data["cases"]) <= 16, "unsupported case capture")
    records = {case["id"]: build_case(case, args.trace.parent) for case in data["cases"]}
    influence = {"schema": SCHEMA, "trace": {"path": str(args.trace.resolve()), "sha256": sha(args.trace)},
        "method": {"id": METHOD, "label": "relative input sensitivity",
            "formula": "abs(f(x) - f(x with only feature j replaced by its original FIT-only median))",
            "normalization": "per-case maximum absolute delta; all-zero case stays zero",
            "reference_scope": "raw feature medians over original FIT row IDs only", "target_perturbed": False,
            "neural_saliency": False, "prediction_units": "original target units",
            "scope": "local fitted-GP median-replacement sensitivity; no neural or causal attribution"},
        "builder_source_sha256": sha(__file__), "cases": records}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir / "influence.json"
    with path.open("x") as stream:
        json.dump(influence, stream, indent=2, allow_nan=False)
        stream.write("\n")
    load_influence(path, args.trace, data)
    receipt = {"schema": "jain.input-cell-influence-proof/v1", "status": "PASS", "influence_sha256": sha(path),
        "trace_sha256": influence["trace"]["sha256"], "builder_source_sha256": sha(__file__),
        "cases": [{"id": cid, "scores": 20, "baseline_rows": record["baseline_parity"]["rows"],
            "max_baseline_native_error": record["baseline_parity"]["max_abs_error_native"],
            "maximum_absolute_delta": record["maximum_absolute_delta"], "score_sha256": record["hashes"]["scores"]}
            for cid, record in records.items()], "method": influence["method"],
        "strict_loader": "PASS including fresh 25-prediction cell verification per case"}
    with (args.output_dir / "influence.receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "path": str(path.resolve()), "sha256": sha(path), "cases": len(records)}))


if __name__ == "__main__":
    main()
