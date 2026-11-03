#!/usr/bin/env python3
"""Validate a private native fused trace before Blender consumes it.

This is a diagnostic contract, not a public-package disclosure filter. Raw traces
contain internal model values and must stay outside the tracked source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

SCHEMA = "jope.visual-trace-diagnostic/v1"
RUN_SCHEMA = "jain.visual-run-trace/v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array(values, shape, name: str) -> np.ndarray:
    require(isinstance(shape, list) and bool(shape), f"{name}: missing shape")
    require(all(type(n) is int and 0 < n <= 1_000_000 for n in shape), f"{name}: invalid shape")
    count = int(np.prod(shape, dtype=np.int64))
    require(count <= 16_000_000, f"{name}: allocation cap")
    value = np.asarray(values, dtype=np.float64)
    require(value.size == count, f"{name}: shape/value mismatch")
    require(bool(np.isfinite(value).all()), f"{name}: nonfinite value")
    return value.reshape(shape)


def affine_arrays(layer: dict):
    name = layer["name"]
    x = array(layer["input"], layer["input_shape"], f"{name}.input")
    w = array(layer["weights"], layer["weight_shape"], f"{name}.weight")
    y = array(layer["output"], layer["output_shape"], f"{name}.output")
    require(w.ndim == 2, f"{name}: expected [out,in] weights")
    require(x.shape[-1] == w.shape[1], f"{name}: input width")
    require(y.shape == (*x.shape[:-1], w.shape[0]), f"{name}: output shape")
    b = np.asarray(layer["bias"], dtype=np.float64)
    require(b.shape == (w.shape[0],) and bool(np.isfinite(b).all()), f"{name}: bias shape/values")
    return x.reshape(-1, w.shape[1]), w, b, y.reshape(-1, w.shape[0])


def gp_sexpression(text: str) -> list:
    """Display the saved model's literal syntax tree; implement no GP semantics."""
    require(0 < len(text) <= 16384, "GP preview exceeds display budget")
    tokens = re.findall(r'\(|\)|[^\s()]+', text)
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            require(len(stack) < 64, "GP preview nesting cap")
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            require(bool(stack) and bool(stack[-1]), "malformed GP preview")
            stack.pop()
        else:
            require(bool(stack), "GP atom outside root")
            stack[-1].append(token)
    require(not stack and len(roots) == 1 and isinstance(roots[0][0], str), "malformed GP root")
    return roots[0]


def validate(path: Path) -> tuple[dict, dict]:
    require(path.stat().st_size <= 256 * 1024 * 1024, "trace exceeds 256 MiB cap")
    data = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    require(data.get("schema") in {SCHEMA, RUN_SCHEMA}, "unsupported trace schema")
    actual_run = data["schema"] == RUN_SCHEMA
    if actual_run:
        require(data.get("source_identity_verified") is True and not data.get("runtime_service_source_dirty"),
                "actual-run source identity unverified")
        require(data.get("source_commit") == data.get("runtime_checkout_head") == data.get("build_base_commit")
                and len(data["source_commit"]) == 40, "actual-run compile/runtime source differs")
        require(data.get("diagnostic_slate_substitution") is False, "actual-run capture substitutes a diagnostic slate")
    require(bool(data.get("bundle_digest")), "missing authenticated bundle identity")
    cases = data.get("cases", [])
    require(1 <= len(cases) <= 16, "expected 1..16 captured cases")
    baseline = {}
    inventory = None
    ids = set()
    summaries = []
    for case in cases:
        cid = case["id"]
        require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cid)), "unsafe case identity")
        require(cid not in ids, "duplicate case identity")
        ids.add(cid)
        if actual_run:
            require(case.get("diagnostic") is False, "actual-run case is diagnostic")
            require(case["source"]["run_id"] == case["run_id"], "source comes from a different run")
            proof = case.get("validation", {})
            require(proof.get("actual_run_binding") is True and proof.get("observed_affine_closure_passed") is True
                    and proof.get("source_execution_parity") is True and proof.get("source_self_checks_passed") is True
                    and proof.get("dropped_records") == 0, "incomplete actual-run controls")
            journal_path = bound_path(path.parent, case["native_journal_path"])
            require(digest(journal_path) == case["native_journal_sha256"], "native journal identity differs")
            journal = json.loads(journal_path.read_text())
            require(journal["schema"] == "jope.actual-run-visual-journal/v1" and journal["run_id"] == case["run_id"]
                    and journal["dropped_records"] == 0, "native journal run binding failed")
            calls = {v["call_id"]: v for v in journal["calls"]}
            require(len(calls) == len(journal["calls"]) and len(calls) >= 2, "incomplete native call inventory")
            display = [calls[i] for i in case["display_call_ids"]]
            require(len(display) == 2 and display[0]["identity"]["phase"] == "fit_conditioning"
                    and display[1]["identity"]["phase"] == "ranker_admission", "wrong native calls displayed")
            require(all(v["identity"]["run_id"] == case["run_id"] for v in calls.values()), "journal contains another run")
            conditioning_identity = display[0]["identity"]
            journal_closure = 0.0
            for call in calls.values():
                identity = call["identity"]
                require(identity.get("dataset_id") == case.get("dataset_id")
                        and identity.get("fit_matrix_sha256") == conditioning_identity.get("fit_matrix_sha256")
                        and str(identity.get("bundle_digest", "")).removeprefix("sha256:") == data["bundle_digest"].removeprefix("sha256:"),
                        "journal dataset/FIT/bundle identity differs")
                for op in call["trace"]["affine"]:
                    xx, ww, bb, yy = affine_arrays(op)
                    error = float(np.max(np.abs(xx @ ww.T + bb - yy) / (np.abs(xx) @ np.abs(ww).T + np.abs(bb) + 1e-9)))
                    require(error <= 1e-5, "later native journal affine closure failed")
                    journal_closure = max(journal_closure, error)
                for tensor in call["trace"]["tensors"]:
                    array(tensor["values"], tensor["shape"], tensor["name"])
            for family in ("affine", "tensors"):
                expected = [(v["call_id"], item) for v in display for item in v["trace"][family]]
                observed = case["trace"][family]
                require(len(observed) == len(expected), "display drops native observations")
                for item, (call_id, native) in zip(observed, expected):
                    require(item["call_id"] == call_id and set(item)-{"call_id"} == set(native),
                            "display native-call identity differs")
                    for key, value in native.items():
                        if key in {"input", "weights", "bias", "output", "values"}:
                            # serde's direct f32 writer and Value conversion can
                            # print different decimal lengths for identical f32
                            # bits. Bind the native datatype, including ±0.
                            a = np.ascontiguousarray(item[key], dtype="<f4").view("<u4")
                            b = np.ascontiguousarray(value, dtype="<f4").view("<u4")
                            require(np.array_equal(a,b), "display differs from actual native FP32 bits")
                        else:
                            require(item[key] == value, "display native metadata differs")
            require(case["outputs"]["dope629"] == display[0]["outputs"]["dope629"]
                    and case["outputs"]["tokens"] == display[0]["outputs"]["tokens"]
                    and case["outputs"]["costs"] == display[1]["outputs"]["total_cost"], "actual native outputs differ")
            for file_key, hash_key in (("proof_path", "proof_sha256"), ("training_proof_path", "training_proof_sha256")):
                require(digest(bound_path(path.parent,case["source"][file_key])) == case["source"][hash_key],
                        "same-run execution proof identity differs")
        else:
            require(case.get("diagnostic") is True, "first-render contract is diagnostic")
            require(case.get("source") is None, "source proof requires a later schema")
        require(case.get("tokens_per_second") is None, "no unqualified token rate")
        require(len(case.get("dataset_sha256", "")) == 64, "missing dataset SHA-256")
        if not actual_run:
            proof = case.get("validation", {})
            require(proof.get("trace_on_off_bitwise_equal") is True, "native trace changes outputs")
            require(proof.get("closure_negative_control_rejected") is True, "missing negative control")
        layers = case["trace"]["affine"]
        require(1 <= len(layers) <= 512, "unexpected affine count")
        case_inventory = [(v["network"], v["name"], v["sequence"],
                           v.get("weight_layout", "out_in"), v["input_shape"][-1],
                           tuple(v["weight_shape"]), v["output_shape"][-1]) for v in layers]
        if inventory is None:
            inventory = case_inventory
        else:
            require(case_inventory == inventory, "affine inventory/order differs between cases")
        names, origins, maximum = set(), set(), 0.0
        for layer in layers:
            key = (layer["network"], layer["name"], layer["sequence"])
            require(key not in names, "duplicate affine identity")
            names.add(key)
            origins.add(layer["network"].lower())
            x, w, b, y = affine_arrays(layer)
            prediction = x @ w.T + b
            bound = np.abs(x) @ np.abs(w).T + np.abs(b) + 1e-9
            error = float(np.max(np.abs(prediction - y) / bound))
            require(error <= 1e-5, f"{key}: arithmetic closure failed ({error})")
            maximum = max(maximum, error)
            frozen = hashlib.sha256(np.asarray(w, dtype="<f4").tobytes() + np.asarray(b, dtype="<f4").tobytes()).hexdigest()
            if key in baseline:
                require(baseline[key] == frozen, f"{key}: weights changed between cases")
            else:
                baseline[key] = frozen
        require(any("dope" in n for n in origins) and any("jope" in n for n in origins), "both fused constituents must be captured")
        for tensor in case["trace"].get("tensors", []):
            array(tensor["values"], tensor["shape"], tensor["name"])
        bridge = [v for v in case["trace"].get("tensors", [])
                  if v["network"] == "bridge" and v["name"] == "stage1b_tokens"]
        require(len(bridge) == 1 and bridge[0]["shape"] == [1, 10, 32], "missing unique native bridge readout")
        output_tokens = array(case["outputs"]["tokens"], [1, 10, 32], "outputs.tokens")
        native_tokens = array(bridge[0]["values"], [1, 10, 32], "native.stage1b_tokens")
        require(np.array_equal(output_tokens, native_tokens), "bridge output differs from native readout")
        summaries.append({"id": cid, "affine_calls": len(layers), "max_normalized_closure_error": maximum})
        if actual_run:
            summaries[-1].update(journal_affine_calls=sum(len(v["trace"]["affine"]) for v in calls.values()),
                                 journal_max_normalized_closure_error=journal_closure)
    report = {"schema": "jope.visual-trace-validation/v1", "trace_sha256": digest(path),
              "status": "PASS", "cases": summaries, "scope": "private fused diagnostic; source and token rate unavailable"}
    if actual_run:
        report["scope"] = "same actual Engine journals and proof-file identities; source semantics checked by engine_evidence render gate"
    return data, report


def bound_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute() and ".." not in candidate.parts, "proof path escapes capture root")
    path = (root / candidate).resolve()
    require(path.is_relative_to(root.resolve()), "proof symlink escapes capture root")
    return path


def verify_actual_execution(root: Path, case: dict, summary: dict) -> dict:
    """Independently bind artifacts, full prediction vectors and refit proof."""
    source = case["source"]
    runproof = json.loads(bound_path(root,source["proof_path"]).read_text())
    require(runproof["schema"] == "jope.single-run-visual-proof/v1", "unsupported same-run execution proof")
    require(runproof["run_id"] == case["run_id"] and runproof["dataset_id"] == case["dataset_id"]
            and runproof["dataset_sha256"] == case["dataset_sha256"] and runproof["model_id"] == source["model_id"]
            and runproof["journal_sha256"] == case["native_journal_sha256"]
            and runproof["dropped_records"] == 0, "same-run execution identity differs")
    require(runproof["parity_rows"] == summary["engine_parity_rows"] == case["input_rows"]
            and runproof["parity_rows"] >= 64 and runproof["fit_rows"] == case["rows"]
            and runproof["heldout_rows"] == len(case["holdout_row_ids"]), "same-run parity row scope differs")
    artifacts = root/case["id"]/"export-rust-python/src"
    rust, python = artifacts/"rust/predictor.rs", artifacts/"python/predictor.py"
    require(digest(rust) == source["rust_sha256"] == runproof["rust_sha256"]
            and digest(python) == source["python_sha256"] == runproof["python_sha256"], "executed Rust/Python source changed")
    proof_dir = root/case["id"]/"proof"
    training = json.loads(bound_path(root,source["training_proof_path"]).read_text())
    derivative = proof_dir/"train-driver.rs"
    original = rust.read_bytes()
    require(training["schema"] == "jope.emitted-training-proof/v1"
            and training["artifact_sha256"] == digest(rust)
            and training["unchanged_artifact_prefix_bytes"] == len(original)
            and digest(derivative) == training["derivative_source_sha256"]
            and derivative.read_bytes()[:len(original)] == original,
            "emitted training proof changes the original source")
    require(training["fit_rows"] == case["rows"] and training["prediction_rows"] == case["input_rows"]
            and training["heldout_rows"] == len(case["holdout_row_ids"])
            and training["parity_abs_tolerance"] == 1e-12 and training["parity_rel_tolerance"] == 1e-10,
            "emitted training scope/tolerance differs")
    native = np.asarray(json.loads((proof_dir/"native-predictions.json").read_text()),dtype=np.float64)
    predicted = {}
    for name in ("rust-predictions", "python-predictions", "rust-training-predictions"):
        receipt = json.loads((proof_dir/f"{name}.receipt.json").read_text())
        require(receipt["exit_code"] == 0, "predictor process failed")
        raw = (proof_dir/f"{name}.stdout").read_text()
        value = json.loads(raw) if name.startswith("python") else [float(v) for v in raw.splitlines()]
        vector = np.asarray(value,dtype=np.float64)
        require(vector.shape == native.shape == (case["input_rows"],) and np.isfinite(vector).all()
                and np.isfinite(native).all(), "incomplete or nonfinite actual predictions")
        require(np.all(np.abs(vector-native) <= 1e-12+1e-10*np.abs(native)), "actual native/source prediction parity failed")
        predicted[name] = vector
    for name in ("rust-selfcheck-compile", "rust-selfcheck", "python-selfcheck", "rust-compile", "rust-training-compile"):
        require(json.loads((proof_dir/f"{name}.receipt.json").read_text())["exit_code"] == 0, "source compile/self-check failed")
    errors = {"rust_vs_engine": float(np.max(np.abs(predicted["rust-predictions"]-native))),
              "python_vs_engine":float(np.max(np.abs(predicted["python-predictions"]-native))),
              "rust_vs_python":float(np.max(np.abs(predicted["rust-predictions"]-predicted["python-predictions"])))}
    require(errors == summary["max_abs_errors"] == runproof["max_abs_errors"], "source parity summary differs from execution")
    train_error = float(np.max(np.abs(predicted["rust-training-predictions"]-native)))
    require(train_error == training["max_abs_error_vs_native_gp"] and training == runproof["training_parity"],
            "training parity summary differs from execution")
    return runproof


def engine_evidence(root: Path, trace: dict, metric_root: Path | None = None) -> list[dict]:
    """Bind separate signed-engine source proofs to matching input/FIT contexts.

    These remain separate diagnostic and search captures: matching contexts do
    not establish the provenance of the champion or equality of candidate slates.
    """
    report = json.loads((root / "export-proof-summary.json").read_text())
    by_id = {"case-" + item["case"]: item for item in report["cases"]}
    model_index = root / "model-paths.json"
    model_rows = json.loads(model_index.read_text()) if model_index.exists() else []
    models = {v["case_id"]: v for v in model_rows}
    require(len(models) == len(model_rows) and len(by_id) == len(report["cases"]), "duplicate source/model case inventory")
    if trace["schema"] == RUN_SCHEMA:
        require(set(models) == set(by_id) == {c["id"] for c in trace["cases"]}, "source/model case inventory differs")
    result = []
    for case in trace["cases"]:
        proof = by_id[case["id"]]
        require(proof.get("schema") == "jope-viz.export-proof/v1", "unqualified source proof schema")
        require(proof["input_sha256"] == case["dataset_sha256"], "source/trace dataset identity differs")
        actual_run = trace["schema"] == RUN_SCHEMA
        if actual_run:
            require(proof["run_id"] == case["run_id"] and proof["source_sha256"]["rust"] == case["source"]["rust_sha256"],
                    "source proof comes from a different native run")
            runproof = verify_actual_execution(root,case,proof)
        require(proof["fit_rows"] == case["rows"], "source/trace FIT row count differs")
        split = json.loads((root / case["id"] / "proof/split.json").read_text())
        require(split["fit"] == case["fit_row_ids"] and split["holdout"] == case["holdout_row_ids"],
                "source/trace FIT or holdout identities differ")
        if actual_run:
            require(len(set(split["fit"])) == len(split["fit"]) and len(set(split["holdout"])) == len(split["holdout"])
                    and not set(split["fit"]) & set(split["holdout"])
                    and sorted(split["fit"]+split["holdout"]) == list(range(case["input_rows"])), "FIT/HOLDOUT split is incomplete or overlaps")
        source_path = root / case["id"] / "export-rust-python/src/rust/predictor.rs"
        require(digest(source_path) == proof["source_sha256"]["rust"], "executed Rust source hash mismatch")
        require(proof["rust_self_check"] == "pass" and proof["python_self_check"] == "pass", "source self-check incomplete")
        require(proof["source_unchanged"] is True, "source proof mutated bytes")
        errors = proof.get("max_abs_errors", {})
        require(set(errors) == {"rust_vs_engine", "python_vs_engine", "rust_vs_python"}, "incomplete source parity proof")
        require(type(proof.get("holdout_rows")) is int and proof["holdout_rows"] > 0, "no held-out source execution")
        require(all(np.isfinite(v) and 0 <= v for v in errors.values()), "source parity failed")
        if not actual_run:
            require(all(v <= 1e-10 for v in errors.values()), "source parity failed")
        source = source_path.read_text()
        lines = [line for line in source.splitlines() if line.startswith(("const TARGET_MIN:", "const TARGET_SCALE:"))]
        require(len(lines) == 2, "expected actual source constants")
        source_rows = source.splitlines()
        first = next((i for i, line in enumerate(source_rows) if line.startswith("pub fn predict_numeric(")), None)
        require(first is not None, "missing actual emitted prediction entry point")
        last = next((i for i in range(first + 1, len(source_rows)) if source_rows[i] == "}"), None)
        require(last is not None and last - first <= 20, "unexpected emitted prediction function extent")
        model_first = next((i for i, line in enumerate(source_rows) if line == "struct Model {"), None)
        declaration = []
        if model_first is not None:
            model_last = next((i for i in range(model_first + 1, len(source_rows)) if source_rows[i] == "}"), None)
            require(model_last is not None and model_last - model_first <= 20, "unexpected emitted model declaration")
            declaration = source_rows[model_first:model_last + 1]
        excerpt = lines + [""] + declaration + [""] + source_rows[first:last + 1]
        item = {"case_id": case["id"], "source_sha256": digest(source_path), "source_lines": lines,
                "source_path": str(source_path.resolve()),
                "source_excerpt_lines": excerpt,
                "verification": {"engine_rows": proof.get("engine_parity_rows", proof["holdout_rows"]),
                                 "embedded_self_check_rows": 64, "max_abs_error": max(errors.values())},
                "engine_run_id": proof["run_id"], "scope": "same actual native Engine run" if actual_run else
                "separate signed-engine run; same dataset and FIT context; diagnostic slate differs"}
        if case["id"] in models:
            model_dir = Path(models[case["id"]]["model_directory"])
            if actual_run:
                require(case["source"]["model_id"] == model_dir.name and model_dir.resolve().is_relative_to(root.resolve()),
                        "GP model belongs to a different run or custody root")
            manifest_path = model_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            require(model_dir.name == "sha256-" + digest(manifest_path), "GP model content address differs")
            require(manifest.get("schema") == "jope.model-object/v1" and manifest.get("primary") == "pipeline.json",
                    "unsupported GP model artifact")
            require(len(manifest["files"]) == 2 and {v["path"] for v in manifest["files"]} == {"input.json", "pipeline.json"},
                    "incomplete or duplicate GP model payload inventory")
            for payload in manifest["files"]:
                require(payload["path"] in {"pipeline.json", "input.json"}, "unexpected model payload path")
                file_path = model_dir / payload["path"]
                require(file_path.stat().st_size == payload["bytes"] and digest(file_path) == payload["sha256"],
                        "GP model payload integrity failure")
            pipeline_path = model_dir / "pipeline.json"
            pipeline = json.loads(pipeline_path.read_text())
            dsl = pipeline["dsl"]
            if actual_run:
                require(models[case["id"]]["run_id"] == case["run_id"]
                        and digest(manifest_path) == runproof["model_manifest_sha256"]
                        and hashlib.sha256(dsl.encode()).hexdigest() == source_sha(case), "winning GP/model/source binding differs")
            item.update({"model_object_id": model_dir.name, "model_manifest_sha256": digest(manifest_path),
                         "pipeline_sha256": digest(pipeline_path), "gp_dsl": dsl, "gp_tree": gp_sexpression(dsl),
                         "gp_program_sha256": hashlib.sha256(dsl.encode()).hexdigest(),
                         "gp_scope": "literal winning model syntax; temporary internal debug preview"})
        elif actual_run:
            raise ValueError("Actual winning model is missing from the source inventory")
        if metric_root:
            metric_path = metric_root / f"{case['id']}-emit-metric.json"
            if metric_path.exists():
                metric = json.loads(metric_path.read_text())
                require(metric.get("schema") == "jope.visual-emit-batch-metric/v1" and metric.get("state") == "measured",
                        "unqualified emission metric")
                require(metric["source_sha256"] == item["source_sha256"] and metric["source_bytes_equal_verified_artifact"] is True,
                        "emission rate source differs from executed source")
                if "model_object_id" in item:
                    require(metric["model_id"] == item["model_object_id"], "GP model/emitted source identity differs")
                require(metric["lexer"] == "jope.source-lex/v1" and metric["repeated_artifact"] is True,
                        "unqualified emission lexer or scope")
                elapsed, tokens = int(metric["batch_elapsed_ns"]), int(metric["total_emitted_tokens"])
                require(elapsed >= max(100_000_000, int(metric["minimum_reporting_interval_ns"])) and tokens > 0,
                        "emission interval below reporting threshold")
                rate = tokens * 1e9 / elapsed
                require(np.isfinite(metric["value"]) and abs(rate - metric["value"]) <= rate * 1e-10,
                        "emission rate arithmetic differs")
                # Preserve the full receipt. Native benchmark verifies exact emitted
                # bytes against this executed source before and after its batch.
                item["emission_metric"] = metric
                item["emission_receipt_sha256"] = digest(metric_path)
        if "gp_tree" in item:
            require("emission_metric" in item or (actual_run and case["validation"]["actual_run_binding"] is True),
                    "GP preview requires native model-to-executed-source binding")
        result.append(item)
    return result


def source_sha(case):
    return case["source"]["champion_program_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--engine-proof", type=Path)
    parser.add_argument("--emit-metrics", type=Path)
    args = parser.parse_args()
    data, report = validate(args.trace)
    if data["schema"] == RUN_SCHEMA:
        proofs = engine_evidence(args.engine_proof or args.trace.parent,data,args.emit_metrics)
        report["source_semantics"] = {v["case_id"]: {"status":"PASS", "source_sha256":v["source_sha256"],
                                     "execution":v["verification"]} for v in proofs}
    result = json.dumps(report, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
