#!/usr/bin/env python3
"""Emit and qualify unfitted Rust recipes for the ten captured GP selections."""

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess


SCHEMA = "jain.short-recipe-source/v1"
CASE_IDS = tuple(f"case-{name}" for name in "abcdefghij")
# The selected DSL operation and its names in the portable Rust emitter.
CALLS = {
    "cheb": "chebyshev_basis",
    "gpr": "gaussian_process_composite_kernel_regression",
    "histgbm": "gradient_boosted_trees",
    "histgbm2": "leafwise_boosted_trees",
    "poly_interact": "pairwise_products",
    "quantile_bins": "quantile_bin_indexer",
    "rank": "mid_rank_transform",
    "ridge": "ridge_regression",
    "robust_scale": "robust_scaler",
    "sigmoid": "logistic_squash",
    "standardize": "standard_scaler",
    "winsor_clip": "winsor_limiter",
}
STATELESS = {"cheb", "sigmoid"}
EMITTER = """use feat_math::{GenomeIr, transpile::{self, EmitOptions}};
use std::io::{self, Read, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    let genome = GenomeIr::parse(&text)?;
    let emitted = transpile::emit(&genome, &EmitOptions::default())?;
    io::stdout().write_all(emitted.rust.as_bytes())?;
    Ok(())
}
"""


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_bytes(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, f"{path}: bytes differ from the pinned digest")
    return raw


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def command(args, raw=None):
    return subprocess.run(
        [str(arg) for arg in args], input=raw, check=True, capture_output=True, timeout=180,
    ).stdout


def source_spans(raw, recipe):
    """Map admitted, ordered operations to the actual emitted call sites."""
    grammar = rb"\(pipe X((?: \([a-z0-9_]+(?: [^()]+)?\))+)\)"
    matched = re.fullmatch(grammar, recipe)
    require(matched is not None, "expected a single flat selected pipeline")
    ops = [word.decode() for word in re.findall(rb"\(([a-z0-9_]+)", matched[1])]
    require(all(op in CALLS for op in ops), "unmapped recipe operation")
    lines = raw.splitlines(keepends=True)
    offsets = [0, *itertools.accumulate(map(len, lines))]
    model = lines.index(b"impl Model {\n")
    fit = next(i for i in range(model + 1, len(lines)) if lines[i].startswith(b"    fn fit("))
    fit_end = lines.index(b"    }\n", fit + 1)
    predict = next(i for i in range(fit_end + 1, len(lines)) if lines[i].startswith(b"    fn predict("))
    predict_end = lines.index(b"    }\n", predict + 1)
    fit_calls = [i for i in range(fit + 1, fit_end) if b"_apply(" in lines[i]]
    predict_calls = [i for i in range(predict + 1, predict_end) if b"_apply(" in lines[i]]
    require(len(fit_calls) == len(predict_calls) == len(ops) - 1, "emitted stage count differs")
    operations = []
    for ordinal, op in enumerate(ops, 1):
        family = CALLS[op].encode()
        if ordinal == len(ops):
            first = last = fit_end - 2
            other = predict_end - 1
            require(
                lines[first].lstrip().startswith(b"let head = " + family + b"_fit(")
                and lines[other].lstrip().startswith(family + b"_predict("),
                "emitted head calls differ",
            )
        else:
            first = last = fit_calls[ordinal - 1]
            other = predict_calls[ordinal - 1]
            stage = f"stage{ordinal}".encode()
            for index in (last, other):
                require(
                    lines[index].lstrip().startswith(b"let x = " + family + b"_apply("),
                    f"emitted stage {ordinal} call differs",
                )
            if op not in STATELESS:
                first -= 1
                require(
                    lines[first].lstrip().startswith(b"let " + stage + b" = " + family + b"_fit(")
                    and b"&" + stage in lines[last] and b"&self." + stage in lines[other],
                    f"emitted stage {ordinal} state differs",
                )
        fit_span = [offsets[first], offsets[last + 1]]
        predict_span = [offsets[other], offsets[other + 1]]
        operations.append({
            "ordinal": ordinal, "label": op, "spans": [fit_span, predict_span],
            "fit_span": fit_span, "predict_span": predict_span,
            "fit_lines": [first + 1, last + 1], "predict_lines": [other + 1, other + 1],
        })
    return operations, {
        "span": [offsets[fit], offsets[predict_end + 1]],
        "lines": [fit + 1, predict_end + 1],
    }


def csv_bytes(rows):
    return "".join(",".join(repr(value) for value in row) + "\n" for row in rows).encode()


def verify_predictions(directory, binary, source_dir, proof, recipe):
    model_id = proof["model"]["model_id"]
    model = source_dir / "models" / model_id
    manifest = json.loads(checked_bytes(model / "manifest.json", model_id.removeprefix("sha256-")))
    parts = {part["path"]: part["sha256"] for part in manifest["files"]}
    for name, key in (("input.json", "input_sha256"), ("pipeline.json", "pipeline_sha256")):
        require(parts[name] == proof["model"][key], f"{name}: model binding differs")
    contract = json.loads(checked_bytes(model / "input.json", parts["input.json"]))
    state = json.loads(checked_bytes(model / "pipeline.json", parts["pipeline.json"]))
    require(state["dsl"].encode() == recipe, "recipe differs from the fitted model")
    require(contract["prediction_version"] == 2, "expected version-two f64 input contract")
    raw = checked_bytes(source_dir / "input.csv", proof["input_csv"]["sha256"])
    records = list(csv.DictReader(raw.decode().splitlines()))
    split = json.loads(checked_bytes(source_dir / "split.json", proof["split"]["sha256"]))
    fit, holdout = split["fit"], split["holdout"]
    require(fit == proof["split"]["fit"] and holdout == proof["split"]["holdout"]
            and sorted(fit + holdout) == list(range(len(records))), "partition identity differs")
    columns = [contract["columns"][index] for index in contract["selected"]]
    target = contract["target"]

    def encode(record, column):
        value = float(record[column["name"]])
        require(math.isfinite(value), "nonfinite feature")
        if column["constant"]:
            return 0.0
        return min(1.0, max(0.0, (value - column["min"]) / (column["max"] - column["min"])))

    rows = [[encode(record, column) for column in columns] for record in records]
    labels = [(float(record[target["name"]]) - target["min"]) / target["scale"] for record in records]
    train = csv_bytes(rows[index] + [labels[index]] for index in fit)
    probe = csv_bytes(rows)
    train_path, probe_path = directory / "fit.csv", directory / "probe.csv"
    predictions_path = directory / "predictions-normalized.txt"
    write_new(train_path, train)
    write_new(probe_path, probe)
    argv = [binary, train_path, probe_path, predictions_path]
    command(argv)
    predictions = predictions_path.read_bytes()
    actual = [target["min"] if target["scale"] <= 1e-12 else target["min"] + float(value) * target["scale"]
              for value in predictions.splitlines()]
    expected = json.loads(checked_bytes(
        source_dir / "native-predictions.json", proof["native_predictions_sha256"],
    ))
    require(len(actual) == len(expected) == len(records), "prediction row count differs")
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    require(
        all(math.isfinite(a) and error <= 1e-12 + 1e-10 * abs(b)
            for a, b, error in zip(actual, expected, errors)),
        f"{source_dir.name}: recipe prediction parity failed (max absolute error {max(errors)})",
    )
    return {
        "status": "PASS", "command": list(map(str, argv)), "fit_rows": len(fit),
        "holdout_rows": len(holdout), "prediction_rows": len(actual),
        "max_abs_error": max(errors), "input_csv_sha256": sha(raw),
        "input_contract_sha256": parts["input.json"], "split_sha256": proof["split"]["sha256"],
        "encoded_fit_sha256": sha(train), "encoded_probe_sha256": sha(probe),
        "native_predictions_sha256": proof["native_predictions_sha256"],
        "recipe_predictions_sha256": sha(predictions), "unchanged_model_id": model_id,
    }


def prepare(gp_source, output, rustc, rlib):
    gp_source, output, rustc, rlib = (p.resolve(strict=True) if p != output else p.resolve()
                                     for p in (gp_source, output, rustc, rlib))
    controller_bytes = (gp_source / "controller.receipt.json").read_bytes()
    controller = json.loads(controller_bytes)
    require(controller["status"] == "PASS", "GP source controller must have passed")
    require(tuple(case["case_id"] for case in controller["cases"]) == CASE_IDS,
            "GP source controller must contain the ordered ten cases")
    output.mkdir(parents=True, exist_ok=True)
    build = output / "build"
    build.mkdir()
    emitter_source, emitter = build / "emit_recipe.rs", build / "emit_recipe"
    write_new(emitter_source, EMITTER.encode())
    emitter_args = [
        rustc, "--edition", "2021", "-O", "-C", "lto=fat", "-C", "codegen-units=1",
        "--extern", f"feat_math={rlib}", "-L", f"dependency={rlib.parent}",
        emitter_source, "-o", emitter,
    ]
    command(emitter_args)
    cases, binaries = [], {}
    for case in controller["cases"]:
        case_id = case["case_id"]
        source_dir = gp_source / case_id
        selected = case["selection"]
        proof = json.loads(checked_bytes(source_dir / "receipt.json", selected["proof_sha256"]))
        require(proof["state"] == "pass", f"{case_id}: original native source proof failed")
        recipe = checked_bytes(source_dir.with_suffix(".selected.dsl"), selected["selected_program_sha256"])
        require(recipe.decode() == selected["selected_program"] == proof["selected_dsl"]["canonical"],
                f"{case_id}: selected recipe differs")
        require(sha(recipe) == proof["selected_dsl"]["canonical_sha256"], "native recipe digest differs")
        checked_bytes(source_dir / "predictor.rs", selected["rust_sha256"])
        raw = command([emitter], recipe)
        require(not any(marker in raw for marker in
                        (b"fn fitted_model(", b"INPUT_SCALERS", b"TARGET_SCALE", b"SELF_CHECK")),
                f"{case_id}: emitted source unexpectedly contains fitted-state adapter code")
        operations, excerpt = source_spans(raw, recipe)
        source_sha = sha(raw)
        directory = output / case_id
        directory.mkdir()
        source_path = directory / "model.rs"
        write_new(source_path, raw)
        if source_sha not in binaries:
            binary = build / source_sha
            argv = [rustc, "--edition", "2021", "-O", source_path, "-o", binary]
            command(argv)
            binaries[source_sha] = (binary, argv, sha(binary.read_bytes()))
        binary, argv, binary_sha = binaries[source_sha]
        prediction = verify_predictions(directory, binary, source_dir, proof, recipe)
        require(prediction["unchanged_model_id"] == selected["model_id"], "selection model differs")
        cases.append({
            "case_id": case_id, "source_path": str(source_path.relative_to(output)),
            "source_sha256": source_sha, "source_bytes": len(raw), "source_lines": len(raw.splitlines()),
            "selected_program_sha256": sha(recipe), "selected_program": recipe.decode(),
            "selected_index": selected["selected_index"], "model_id": selected["model_id"],
            "operations": operations, "excerpt": excerpt,
            "original_fitted_source_sha256": selected["rust_sha256"],
            "original_gp_proof_sha256": selected["proof_sha256"],
            "compile": {"status": "PASS", "command": list(map(str, argv)), "binary_sha256": binary_sha},
            "prediction_proof": prediction,
        })
        print(case_id, len(raw), len(operations), prediction["max_abs_error"], flush=True)
    result = {
        "schema": SCHEMA, "status": "PASS", "cases": cases,
        "gp_controller_sha256": sha(controller_bytes), "trace_sha256": controller["render_trace_sha256"],
        "span_contract": "Zero-based half-open UTF-8 byte offsets; one-based inclusive source lines.",
        "scope": "Complete unfitted Rust recipes, with actual selected operation call sites. "
                 "Call-site highlights describe code correspondence. Prediction proofs use the "
                 "original v2 normalization and native FIT split. Saved objects remain unchanged.",
        "emitter": {
            "entry_point": "feat_math::transpile::emit default options, Rust output",
            "source_sha256": sha(EMITTER.encode()), "binary_sha256": sha(emitter.read_bytes()),
            "feat_math_rlib_sha256": sha(rlib.read_bytes()), "command": list(map(str, emitter_args)),
            "rustc_version": command([rustc, "-vV"]).decode(),
        },
    }
    write_new(output / "catalog.json", (json.dumps(result, indent=2) + "\n").encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("gp-source", "output", "rustc", "feat-math-rlib"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.gp_source, args.output, args.rustc, args.feat_math_rlib)


if __name__ == "__main__":
    main()
