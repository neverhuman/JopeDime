#!/usr/bin/env python3
"""Fit/export the actual G-selected DSL and retain the whole native Rust file.

This is a bounded local research route, not production ranker admission. It
never substitutes a different program when the selected program cannot export.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from code_pages import make_pages
from validate_trace import require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(case):
    """The new G admission chooses only actual, unevaluated proposal indices."""
    programs = case["programs"]
    name = next((name for name in ("final_k0_cost", "ensemble_cost_final", "cost_final") if name in case["outputs"]), None)
    require(name is not None, "missing observed complete G cost; residual alone cannot select a GP")
    costs = np.asarray(case["outputs"][name], dtype=np.float64).reshape(-1)
    require(len(costs) == len(programs) and len(costs) > 4 and bool(np.isfinite(costs).all()),
            "invalid observed proposal costs")
    index = min(range(4, len(costs)), key=lambda i: (float(costs[i]), i))
    return index, programs[index], name, costs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--native-proof", type=Path, required=True)
    parser.add_argument("--fit-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-pages-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.trace.read_text())
    require(raw["schema"] == "jain.h256-stage2-capture/v1" and raw.get("status") == "complete",
            "GP selection requires an actual complete Stage 2 capture")
    require(not args.output_dir.exists() and not args.code_pages_dir.exists(), "output reuse refused")
    require(args.output_dir.is_absolute() and args.code_pages_dir.is_absolute(), "outputs must be external absolute paths")
    # Check the immutable model, inputs, scalar arithmetic and golden forward
    # before allowing any new algorithm/source attribution.
    from h256_stage2_validate import validate
    _, neural_validation = validate(args.trace)
    source_trace_sha = sha(args.trace)
    fit_binary_sha = sha(args.fit_binary)
    original = json.loads((args.native_proof / "trace.json").read_text())
    native_cases = {case["id"]: case for case in original["cases"]}
    models = json.loads((args.native_proof / "model-paths.json").read_text())
    if isinstance(models, dict):
        models = models["cases"]
    by_model = {case["case_id"]: case for case in models}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.code_pages_dir.mkdir(parents=True, exist_ok=False)
    results, page_results = [], []
    rendered = copy.deepcopy(raw)
    for case, shown in zip(raw["cases"], rendered["cases"]):
        cid = case["id"]
        require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cid)), "unsafe case name")
        reference = native_cases[cid]
        require(case["dataset_sha256"] == reference["dataset_sha256"]
                and case["fit_row_ids"] == reference["fit_row_ids"]
                and case["holdout_row_ids"] == reference["holdout_row_ids"], "source FIT/dataset differs from G input")
        index, dsl, cost_name, costs = choose(case)
        selected = case["selection"]
        require(selected["selected_slate_index"] == index and selected["selected_program"] == dsl,
                "controller selection differs from observed neural selection")
        dsl_path = args.output_dir / (cid + ".selected.dsl")
        dsl_path.write_text(dsl)
        csv = args.native_proof / (cid + ".csv")
        require(sha(csv) == case["dataset_sha256"], "source CSV differs from captured table")
        target = args.output_dir / cid
        command = [str(args.fit_binary.resolve()), str(dsl_path.resolve()), str(csv.resolve()),
                   str(by_model[cid]["model_directory"]),
                   str((args.native_proof / cid / "proof/split.json").resolve()), str(target.resolve())]
        started = time.perf_counter_ns()
        result = subprocess.run(command, capture_output=True, timeout=180, check=False)
        (args.output_dir / (cid + ".stdout")).write_bytes(result.stdout)
        (args.output_dir / (cid + ".stderr")).write_bytes(result.stderr)
        require(result.returncode == 0, f"selected native GP fit/export failed for {cid}; retain failure, do not reroll")
        proof_path = target / "receipt.json"
        proof = json.loads(proof_path.read_text())
        dsl_sha = hashlib.sha256(dsl.encode()).hexdigest()
        require(proof["state"] == "pass" and proof["selected_dsl"]["sha256"] == dsl_sha
                and proof["selected_dsl"]["canonical_sha256"] == dsl_sha,
                "native refit changed the neural-selected canonical program")
        require(proof["input_csv"]["sha256"] == case["dataset_sha256"]
                and proof["split"]["fit"] == case["fit_row_ids"]
                and proof["split"]["holdout"] == case["holdout_row_ids"], "native refit changed dataset/FIT")
        rust = target / "predictor.rs"
        require(sha(rust) == proof["source_sha256"]["rust"], "executed Rust identity differs")
        manifest = make_pages(rust, args.code_pages_dir / cid, font_size=21, columns=48, rows=24)
        source = {"case_id": cid, "rust_sha256": proof["source_sha256"]["rust"],
                  "python_sha256": proof["source_sha256"]["python"],
                  "proof_path": str(proof_path.relative_to(args.output_dir)), "proof_sha256": sha(proof_path),
                  "rust_path": str(rust.relative_to(args.output_dir)),
                  "selected_program_sha256": dsl_sha, "selected_program": dsl,
                  "selected_index": index, "cost_output_name": cost_name,
                  "neural_capture_sha256": source_trace_sha, "model_id": proof["model"]["model_id"],
                  "causality": "actual G pre-reveal proposal selection followed by native supplied-DSL fit/export; no registered Engine champion claim"}
        shown["source"] = source
        results.append({"case_id": cid, "selection": source, "fit_command": command,
                        "fit_exit_code": result.returncode, "fit_proof_wall_ns": time.perf_counter_ns()-started,
                        "fit_binary_sha256": fit_binary_sha, "costs": costs.tolist()})
        page_results.append({"case_id": cid, "source_sha256": source["rust_sha256"],
                             "page_count": manifest["page_count"]})
        print(json.dumps({"case": cid, "selected_index": index, "rust_sha256": source["rust_sha256"],
                          "whole_source_pages": manifest["page_count"]}), flush=True)
    # Copy immutable snapshot inputs; leave the raw neural archive untouched.
    for directory in ("assets", "source"):
        src = args.trace.parent / directory
        if src.is_dir():
            shutil.copytree(src, args.output_dir / directory)
    rendered["gp_binding"] = {"raw_capture_path": str(args.trace.resolve()), "raw_capture_sha256": source_trace_sha,
                              "controller_sha256": sha(__file__), "fit_binary_sha256": fit_binary_sha}
    rendered["coverage"]["source_or_gp_causality"] = True
    out = args.output_dir / "trace.json"
    out.write_text(json.dumps(rendered, separators=(",", ":"), allow_nan=False) + "\n")
    receipt = {"schema": "jain.h256-gp-source-controller/v1", "status": "PASS",
               "raw_neural_capture_sha256": source_trace_sha, "render_trace_sha256": sha(out),
               "controller_sha256": sha(__file__), "neural_validation": neural_validation,
               "cases": results, "complete_source_pages": page_results,
               "scope": "G-selected native GP refits, full unchanged source execution and full source raster; no production admission/quality claim"}
    (args.output_dir / "controller.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
