#!/usr/bin/env python3
"""Check the saved H256 encoder snapshot without attributing GP output to it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from validate_trace import affine_arrays, array, require

SCHEMA = "jain.h256-stage1-capture/v1"
ROLES = {"problem_encoder", "solution_encoder"}
FROZEN_DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
OUTPUT_COUNTS = {
    "dope629": 629, "hidden256": 256, "context_tokens_92x32": 92 * 32,
    "context_mask_92": 92, "block_log_norms_11": 11, "scale_side_8": 8,
    "context_key_92x32": 92 * 32, "context_value_92x32": 92 * 32,
    "context_summary_64": 64, "context_bilinear_24": 24,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def f32_bytes(values):
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def typed_bytes(values, dtype):
    formats = {"float32": "<f4", "float64": "<f8", "bool": "?", "int64": "<i8", "int32": "<i4"}
    require(dtype in formats, "unknown tensor datatype")
    return np.ascontiguousarray(values, dtype=formats[dtype]).tobytes()


def safe_file(root, relative):
    path = Path(relative)
    require(not path.is_absolute(), "snapshot file must be relative")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(),
            "snapshot file escapes capture or is missing")
    return resolved


def tensor_values(record):
    dtype = record.get("dtype")
    require(dtype in {"float32", "float64", "bool", "int64", "int32"}, "unknown tensor datatype")
    if record["shape"] == []:
        value = np.asarray(record["values"], dtype=np.float64)
        require(value.size == 1 and bool(np.isfinite(value).all()), "invalid scalar tensor")
        value = value.reshape(())
    else:
        value = array(record["values"], record["shape"], record.get("name", record["id"]))
    if dtype == "bool":
        require(bool(np.isin(value, [0, 1]).all()), "invalid boolean tensor")
    return value


def check_graph(case, parameters):
    tensors = case["tensors"]
    by_tensor = {v["id"]: v for v in tensors}
    require(len(by_tensor) == len(tensors), "duplicate tensor identity")
    for tensor in tensors:
        tensor_values(tensor)
        require(tensor.get("role") in ROLES, "unknown tensor encoder role")
    operations = case["operations"]
    by_op = {v["id"]: v for v in operations}
    require(bool(by_op) and len(by_op) == len(operations), "missing/duplicate operation identity")
    prior = set()
    for op in operations:
        require(op.get("role") in ROLES and bool(op.get("kind")), "invalid operation role/kind")
        require(set(op.get("parent_op_ids", [])) <= prior, "operation parents are missing or not topological")
        require(set(op.get("parameter_ids", [])) <= parameters.keys(), "foreign operation parameter")
        actual_parents = []
        for tid in op["input_tensor_ids"]:
            require(tid in by_tensor or tid in parameters, "foreign operation input")
            if tid in by_tensor:
                producer = by_tensor[tid].get("producer_op_id")
                require(producer is None or producer in prior, "input producer is not earlier in graph")
                if producer is not None and producer not in actual_parents:
                    actual_parents.append(producer)
        require(op.get("parent_op_ids", []) == actual_parents, "parents differ from actual input producers")
        require(bool(op["output_tensor_ids"]), "operation has no observed output")
        for tid in op["output_tensor_ids"]:
            require(tid in by_tensor, "operation output tensor missing")
            require(by_tensor[tid].get("producer_op_id") == op["id"], "output producer differs")
        prior.add(op["id"])
    for tensor in tensors:
        producer = tensor.get("producer_op_id")
        require(producer is None or producer in by_op, "tensor producer missing")
        require(producer is None or tensor["id"] in by_op[producer]["output_tensor_ids"],
                "tensor is absent from its producer outputs")
    if "functional_dependencies" in case:
        expected = {(parent, op["id"]) for op in operations for parent in op["parent_op_ids"]}
        actual = [(dep["from_operation_id"], dep["to_operation_id"])
                  for dep in case["functional_dependencies"]]
        require(len(actual) == len(set(actual)) and set(actual) == expected,
                "functional routes differ from observed tensor dependencies")
        for dep in case["functional_dependencies"]:
            target = by_op[dep["to_operation_id"]]
            require(dep["kind"] == target["kind"] and dep["source_ref"] == target["call_source"],
                    "functional route type/source differs")
    return by_tensor, by_op


def check_readouts(case, by_tensor):
    require(set(case["outputs"]) == set(case["outputs_tensor_ids"]), "readout inventory differs")
    for name, values in case["outputs"].items():
        tid = case["outputs_tensor_ids"][name]
        require(tid in by_tensor, "named readout tensor is absent")
        tensor = by_tensor[tid]
        require(typed_bytes(values, tensor["dtype"]) == typed_bytes(tensor["values"], tensor["dtype"]),
                "named readout differs from observed tensor")


def check_affines(case, by_tensor, by_op, parameters):
    maximum = 0.
    require(bool(case["trace"]["affine"]), "no measured learned operations")
    for op in case["trace"]["affine"]:
        x, w, b, y = affine_arrays(op)
        opid, pids = op["operation_id"], op["parameter_ids"]
        require(opid in by_op, "affine lacks actual graph operation")
        graph_op = by_op[opid]
        require(graph_op["kind"] == "Linear" and len(graph_op["input_tensor_ids"]) == 1
                and len(graph_op["output_tensor_ids"]) == 1, "affine graph is not an observed Linear")
        for side, tid in (("input", graph_op["input_tensor_ids"][0]),
                          ("output", graph_op["output_tensor_ids"][0])):
            tensor = by_tensor[tid]
            require(tensor["shape"] == op[side + "_shape"]
                    and f32_bytes(tensor["values"]) == f32_bytes(op[side]),
                    "affine snapshot differs from graph tensor")
        require(bool(pids) and set(pids) <= parameters.keys(), "affine parameter missing")
        weight = parameters[pids[0]]
        require(weight["shape"] == op["weight_shape"]
                and f32_bytes(weight["values"]) == f32_bytes(op["weights"]),
                "affine does not use saved weight matrix")
        require(pids == graph_op["parameter_ids"], "affine graph parameter differs")
        if len(pids) > 1:
            require(f32_bytes(parameters[pids[1]]["values"]) == f32_bytes(op["bias"]),
                    "affine does not use saved bias")
        else:
            require(bool((b == 0).all()), "unbound nonzero affine bias")
        error = float(np.max(np.abs(x @ w.T + b - y) / (np.abs(x) @ np.abs(w).T + np.abs(b) + 1e-9)))
        require(error <= 1e-5, "encoder affine closure failed")
        maximum = max(maximum, error)
    return maximum


def check_native_inputs(case, cache):
    inputs = case["inputs"]
    for prefix in ("native_trace", "native_journal"):
        p = Path(inputs[prefix + "_path"])
        if str(p) not in cache:
            require(p.is_file() and p.stat().st_size <= 256 * 1024 * 1024, "native input archive missing/oversize")
            cache[str(p)] = (sha(p), json.loads(p.read_text()))
        require(cache[str(p)][0] == inputs[prefix + "_sha256"], "upstream native input archive differs")
    native = cache[inputs["native_trace_path"]][1]
    reference = next(c for c in native["cases"] if c["id"] == case["id"])
    for name in ("dataset_sha256", "fit_row_ids", "holdout_row_ids", "input_header", "table_preview"):
        require(case[name] == reference[name], "native dataset/table/FIT identity differs")
    journal = cache[inputs["native_journal_path"]][1]
    require(journal["run_id"] == reference["run_id"], "native journal belongs to another run")
    call = next(c for c in journal["calls"] if c["identity"]["phase"] == "fit_conditioning")
    require(call["identity"]["fit_matrix_sha256"] == inputs["fit_matrix_sha256"], "native FIT matrix differs")
    by_name = {t["name"]: t for t in case["tensors"]}
    for name, key in (("fit.v256", "v256"), ("fit.col64x17", "col")):
        require(name in by_name and by_name[name].get("producer_op_id") is None
                and f32_bytes(by_name[name]["values"]) == f32_bytes(call["outputs"][key]),
                "encoder root differs from native FIT descriptor")


def parameter_inventory_sha(parameters):
    inventory = [{k: p[k] for k in ("id", "shape", "sha256", "checkpoint_origin", "role")}
                 for p in sorted(parameters.values(), key=lambda p: p["id"])]
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def check_saved_parameters(root, data, parameters):
    """Read the real saved state with a restricted loader, outside Blender."""
    import codecs
    import torch
    allowed = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, np.dtypes.Float32DType, codecs.encode]
    checkpoint = data["checkpoint"]
    problem = data["dope"]["state_checkpoint"]
    loaded = {}
    for description in (checkpoint, problem):
        path = safe_file(root, description["path"])
        require(sha(path) == description["sha256"] and path.stat().st_size == description["bytes"],
                "parameter origin checkpoint differs")
        with torch.serialization.safe_globals(allowed):
            loaded[description["path"]] = torch.load(path, map_location="cpu", weights_only=True)
    stage1 = loaded[checkpoint["path"]]
    state = stage1["encoder_state_dict"]
    require(len(state) == checkpoint["tensor_count"]
            and sum(v.numel() for v in state.values()) == checkpoint["parameter_count"],
            "actual saved Stage 1 inventory differs")
    require(stage1["dope_sha256"] == FROZEN_DOPE, "saved encoder was trained with another DOPE")
    saved_solution_keys = set()
    for parameter in parameters.values():
        origin_file, origin_key = parameter["checkpoint_origin"].split("::", 1)
        require(origin_file in loaded, "parameter originates in an unverified checkpoint")
        group, key = origin_key.split("/", 1)
        original = loaded[origin_file][group][key]
        require(list(original.shape) == parameter["shape"]
                and f32_bytes(original.detach().cpu().numpy()) == f32_bytes(parameter["values"]),
                "trace parameter differs from actual saved state")
        if parameter["role"] == "solution_encoder":
            require(origin_file == checkpoint["path"] and group == "encoder_state_dict",
                    "solution weights do not originate in saved Stage 1")
            saved_solution_keys.add(key)
        else:
            require(origin_file == problem["path"] and group == "state_dict" and key.startswith("dope_model."),
                    "problem weights do not originate in frozen DOPE")
    require(saved_solution_keys == set(state), "saved solution weights omitted from inventory")
    scaler = data["scaler"]
    require(scaler["checkpoint_path"] == checkpoint["path"] and scaler["policy"] == "scaler_policy_b",
            "Stage 1 uses the wrong saved scaler")
    originals = stage1["scaler_policy_b"]
    require({b["name"] for b in scaler["buffers"]} == set(originals), "saved scaler buffers omitted")
    for buffer in scaler["buffers"]:
        original = originals[buffer["name"]]
        require(list(original.shape) == buffer["shape"]
                and f32_bytes(original) == f32_bytes(tensor_values(buffer))
                and hashlib.sha256(f32_bytes(original)).hexdigest() == buffer["sha256"],
                "scaler differs from actual saved state")


def validate(path, *, saved_receipt=None):
    path = Path(path)
    require(path.stat().st_size <= 256 * 1024 * 1024, "encoder trace exceeds 256 MiB")
    data = json.loads(path.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    require(data.get("schema") == SCHEMA, "wrong encoder snapshot schema")
    coverage = data["coverage"]
    require(coverage.get("stage1_only") is True and coverage.get("stage2") is False
            and coverage.get("query_or_basis_heads_available") is False,
            "saved Stage 1 snapshot incorrectly claims missing trained heads")
    require(coverage.get("dropped_records") == 0, "encoder observation loss")
    checkpoint = data["checkpoint"]
    saved = safe_file(path.parent, checkpoint["path"])
    require(sha(saved) == checkpoint["sha256"] and saved.stat().st_size == checkpoint["bytes"],
            "saved checkpoint identity differs")
    require(checkpoint["tensor_count"] == 39 and checkpoint["parameter_count"] == 15005,
            "unsupported Stage 1 checkpoint inventory")
    sources = data["sources_manifest"]
    require(isinstance(sources, list) and bool(sources), "no pinned research source")
    for source in sources:
        require(sha(safe_file(path.parent, source["path"])) == source["sha256"], "research source differs")
    dope = data["dope"]
    require(dope["composite_expected"] == FROZEN_DOPE
            and dope["before"] == dope["after"] == dope["composite_expected"],
            "frozen problem encoder identity differs")
    entries = data["parameters"]
    parameters = {v["id"]: v for v in entries}
    require(bool(entries) and len(parameters) == len(entries), "missing/duplicate parameter identity")
    for parameter in entries:
        require(parameter.get("role") in ROLES, "unknown parameter encoder role")
        require(parameter.get("dtype") == "float32", "unsupported parameter datatype")
        values = tensor_values(parameter)
        require(hashlib.sha256(f32_bytes(values)).hexdigest() == parameter["sha256"],
                "parameter bytes differ")
        require(bool(parameter.get("checkpoint_origin")), "parameter has no saved origin")
    inventory_sha = parameter_inventory_sha(parameters)
    scaler_sha = hashlib.sha256(json.dumps(data["scaler"], sort_keys=True).encode()).hexdigest()
    if saved_receipt is None:
        check_saved_parameters(path.parent, data, parameters)
    else:
        verified = json.loads(Path(saved_receipt).read_text())
        require(verified.get("status") == "PASS" and verified.get("saved_parameters_verified") is True
                and verified.get("trace_sha256") == sha(path)
                and verified.get("parameter_inventory_sha256") == inventory_sha
                and verified.get("scaler_inventory_sha256") == scaler_sha
                and verified.get("verifier_sha256") == sha(__file__),
                "saved checkpoint verification receipt differs")
        problem = dope["state_checkpoint"]
        require(sha(safe_file(path.parent, problem["path"])) == problem["sha256"],
                "frozen DOPE state changed after verification")
    cases = data["cases"]
    require(1 <= len(cases) <= 16, "encoder case count outside bounded contract")
    ids, results, native_cache = set(), [], {}
    for case in cases:
        cid = case["id"]
        require(isinstance(cid, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", cid)
                and cid not in ids, "duplicate or invalid case ID")
        ids.add(cid)
        require("functional_dependencies" in case, "missing functional route inventory")
        require(case.get("source") is None and case.get("tokens_per_second") is None,
                "encoder-only capture cannot claim GP source or source rate")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", case["dataset_sha256"])), "dataset identity missing")
        fit, hold = case["fit_row_ids"], case["holdout_row_ids"]
        require(bool(fit) and bool(hold) and len(set(fit)) == len(fit) and len(set(hold)) == len(hold),
                "duplicate or empty row partition")
        require(not set(fit) & set(hold) and set(fit) | set(hold) == set(range(case["rows"])),
                "FIT/HOLD partition is not disjoint and complete")
        by_tensor, by_op = check_graph(case, parameters)
        check_readouts(case, by_tensor)
        check_native_inputs(case, native_cache)
        affine = case["trace"]["affine"]
        max_error = check_affines(case, by_tensor, by_op, parameters)
        for name, count in OUTPUT_COUNTS.items():
            value = np.asarray(case["outputs"][name], dtype=np.float64)
            require(value.size == count and bool(np.isfinite(value).all()), f"{name}: wrong readout")
        mask = np.asarray(case["outputs"]["context_mask_92"])
        require(bool(np.isin(mask, [0, 1]).all()), "invalid 92-token mask")
        results.append({"id": cid, "affine_operations": len(affine), "graph_operations": len(by_op),
                        "max_normalized_closure_error": max_error})
    receipt = {"schema": "jain.h256-validation/v1", "status": "PASS", "trace_sha256": sha(path),
               "checkpoint_sha256": checkpoint["sha256"], "cases": results,
               "saved_parameters_verified": True, "parameter_inventory_sha256": inventory_sha,
               "scaler_inventory_sha256": scaler_sha,
               "verifier_sha256": sha(__file__),
               "scope": "saved Stage 1 context encoder only; no Stage 2 scoring, GP source or emission rate"}
    return data, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    _, receipt = validate(args.trace)
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.receipt:
        with args.receipt.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")
