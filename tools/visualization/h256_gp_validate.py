"""Bind the complete Rust file to the actual G-selected native GP refit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from h256_validate import safe_file, sha
from validate_trace import require


def prediction_vector(path):
    text = path.read_text()
    if path.suffix == ".json" or text.lstrip().startswith("["):
        values = json.loads(text)
    else:
        values = [float(line) for line in text.splitlines() if line.strip()]
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 1 and bool(np.isfinite(array).all()), "invalid prediction vector")
    return array


def validate_gp_sources(data, trace_path, gp_root):
    trace_path, root = Path(trace_path), Path(gp_root)
    binding = data.get("gp_binding")
    require(isinstance(binding, dict), "missing raw neural capture binding")
    original_path = Path(binding["raw_capture_path"])
    require(sha(original_path) == binding["raw_capture_sha256"], "original neural archive changed")
    original = json.loads(original_path.read_text())
    require(set(data) == set(original) | {"gp_binding"}, "render archive introduces unknown neural fields")
    for key in set(original) - {"cases", "coverage"}:
        require(data[key] == original[key], "render archive changed neural capture")
    expected_coverage = {**original["coverage"], "source_or_gp_causality": True}
    require(data["coverage"] == expected_coverage and len(data["cases"]) == len(original["cases"]),
            "render archive changed neural coverage/cases")
    verdicts = {}
    for case, raw_case in zip(data["cases"], original["cases"]):
        require({k: v for k, v in case.items() if k != "source"}
                == {k: v for k, v in raw_case.items() if k != "source"}, "render archive changed case observations")
        source = case["source"]
        require(isinstance(source, dict) and source["case_id"] == case["id"]
                and source["neural_capture_sha256"] == binding["raw_capture_sha256"], "source belongs to another neural capture")
        name = source["cost_output_name"]
        require(name == "final_k0_cost", "source selection must use complete observed G cost")
        costs = np.asarray(case["outputs"][name], dtype=np.float64)
        require(len(costs) == len(case["programs"]) and bool(np.isfinite(costs).all()), "invalid selection costs")
        index = min(range(4, len(costs)), key=lambda i: (float(costs[i]), i))
        dsl = case["programs"][index]
        dsl_sha = hashlib.sha256(dsl.encode()).hexdigest()
        require(source["selected_index"] == case["selection"]["selected_slate_index"] == index
                and source["selected_program"] == case["selection"]["selected_program"] == dsl
                and source["selected_program_sha256"] == dsl_sha, "source does not implement actual G selection")
        proof_path = safe_file(root, source["proof_path"])
        require(sha(proof_path) == source["proof_sha256"], "native source proof changed")
        proof = json.loads(proof_path.read_text())
        require(proof["schema"] == "jope.visual-selected-dsl-fit/v1" and proof["state"] == "pass"
                and proof.get("native_build_profile") == "release" and proof.get("native_debug_assertions") is False
                and proof.get("emitted_rust_training_profile") == "-O"
                and proof["selected_dsl"]["canonical_sha256"] == dsl_sha
                and proof["selected_dsl"]["sha256"] == dsl_sha, "native refit is another selected program")
        require(proof["input_csv"]["sha256"] == case["dataset_sha256"]
                and proof["split"]["fit"] == case["fit_row_ids"]
                and proof["split"]["holdout"] == case["holdout_row_ids"], "source was fitted on another dataset/partition")
        require(proof["prediction_rows"] == case["rows"] and proof["fit_rows"] == len(case["fit_row_ids"])
                and proof["heldout_rows"] == len(case["holdout_row_ids"]), "source proof row scope differs")
        rust_path = safe_file(root, source["rust_path"])
        directory = proof_path.parent
        python_path = directory / "predictor.py"
        require(sha(rust_path) == source["rust_sha256"] == proof["source_sha256"]["rust"]
                and sha(python_path) == source["python_sha256"] == proof["source_sha256"]["python"],
                "working source bytes differ")
        require(proof["source_unchanged"] is True and proof["self_check_rows_per_language"] == 64
                and proof["rust_self_check"] == proof["python_self_check"] == "pass"
                and proof["parity_abs_tolerance"] == 1e-12 and proof["parity_rel_tolerance"] == 1e-10,
                "source execution/selfcheck contract incomplete")
        model_path = Path(proof["model"]["path"])
        require(model_path.resolve().is_relative_to(directory.resolve()), "refitted model escapes proof directory")
        manifest_path = model_path / "manifest.json"
        require("sha256-" + sha(manifest_path) == source["model_id"] == proof["model"]["model_id"],
                "native model content address differs")
        manifest = json.loads(manifest_path.read_text())
        for part in manifest["files"]:
            file = safe_file(model_path, part["path"])
            require(sha(file) == part["sha256"] and file.stat().st_size == part["bytes"], "refitted GP model payload changed")
        state = json.loads((model_path / "pipeline.json").read_text())
        require(state["dsl"] == dsl, "fitted native GP tree differs from G selection")
        native_file = directory / "native-predictions.json"
        require(sha(native_file) == proof["native_predictions_sha256"], "native GP predictions changed")
        native = prediction_vector(native_file)
        require(len(native) == case["rows"], "native GP prediction count differs")
        # Command stdout filenames are explicit in the helper's per-command receipts.
        execution = directory / "proof"
        vector_names = ("rust-predictions", "python-predictions", "rust-training-predictions")
        observed = []
        for stem in vector_names:
            receipt_path = execution / (stem + ".receipt.json")
            command = json.loads(receipt_path.read_text())
            require(command["exit_code"] == 0, "source execution command failed")
            output = execution / (stem + ".stdout")
            require(sha(output) == command["stdout_sha256"], "source execution stdout changed")
            values = prediction_vector(output)
            require(values.shape == native.shape and bool((np.abs(values-native) <= 1e-12+1e-10*np.abs(native)).all()),
                    "working source or emitted training no longer matches native GP")
            observed.append(float(np.max(np.abs(values-native))))
        for stem in ("rust-selfcheck", "python-selfcheck", "rust-compile", "rust-training-compile"):
            command = json.loads((execution / (stem + ".receipt.json")).read_text())
            require(command["exit_code"] == 0
                    and sha(execution / (stem + ".stdout")) == command["stdout_sha256"]
                    and sha(execution / (stem + ".stderr")) == command["stderr_sha256"],
                    "compiler/selfcheck execution receipt differs")
        training = proof["training_parity"]
        derivative = execution / "train-driver.rs"
        require(sha(derivative) == training["derivative_source_sha256"]
                and derivative.read_bytes().startswith(rust_path.read_bytes())
                and training["unchanged_artifact_prefix_bytes"] == rust_path.stat().st_size,
                "emitted training proof changed the original Rust file")
        rate = None
        if "emit_metric_path" in source or "emit_metric_sha256" in source:
            metric_path = safe_file(root, source["emit_metric_path"])
            require(sha(metric_path) == source["emit_metric_sha256"], "source emission metric changed")
            metric = json.loads(metric_path.read_text())
            require(metric["schema"] == "jope.visual-emit-batch-metric/v1" and metric["state"] == "measured"
                    and metric["debug_assertions"] is False and metric["lexer"] == "jope.source-lex/v1"
                    and metric["model_id"] == source["model_id"]
                    and metric["source_sha256"] == source["rust_sha256"]
                    and metric["source_bytes"] == rust_path.stat().st_size
                    and metric["source_bytes_equal_verified_artifact"] is True
                    and metric["repeated_artifact"] is True and metric["unique_source_artifacts"] == 1
                    and metric["neural_decoding_tokens_per_second"] is None,
                    "source emission rate is unqualified or belongs to another model")
            elapsed = int(metric["batch_elapsed_ns"])
            total = int(metric["total_emitted_tokens"])
            require(elapsed >= max(100_000_000, int(metric["minimum_reporting_interval_ns"]),
                                   1000 * int(metric["clock_resolution_observed_ns"]),
                                   100 * int(metric["clock_pair_p99_ns"]))
                    and total == metric["rust_tokens_per_artifact"] * metric["repetitions"]
                    and metric["repetitions"] > 0, "source rate interval/token arithmetic differs")
            rate = float(metric["value"])
            require(np.isfinite(rate) and rate > 0 and abs(rate - total / elapsed * 1e9) <= 1e-10 * rate,
                    "source rate value differs from measured counters")
        verdicts[case["id"]] = {"status": "PASS", "causal_link_verified": True,
            "trace_sha256": sha(trace_path), "rust_sha256": source["rust_sha256"],
            "proof_sha256": source["proof_sha256"], "selected_program_sha256": dsl_sha,
            "rust_path": str(rust_path), "prediction_rows": len(native),
            "max_abs_error": max(observed), "selected_index": index,
            "rate_rust_lexical_tokens_per_s": rate,
            "scope": "actual pre-reveal G selection -> supplied-DSL native fit -> entire working Rust; no registered Engine champion claim"}
    return verdicts
