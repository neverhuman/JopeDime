#!/usr/bin/env python3
"""Admit actual saved five-fold H256 inference and optional causally bound GP exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

import h256_validate as shared
import validate_trace
import h256_functional
from validate_trace import require

SCHEMA = "jain.h256-stage2-capture/v1"
RECEIPT_SCHEMA = "jain.h256-stage2-validation/v1"
ENSEMBLE_SHA = "ee40e191e4ba8f2b0a3e0a34a44746984ea2aebafc277c588f71f73f7de323c8"
ROUND_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
CRITIC_SHA = "ba5678e1097550d54ec5339cf42b341339af7d65e566cd3a2bcf28f16c596755"
GOLDEN_SHA = "de8f0ae10b0c38413cfd9171e6d627aff97764b38ffca8d9ccda48fdefe5a0bb"
SCALER_NAMES = {"dope629_mean", "dope629_std", "hidden256_mean", "hidden256_std", "col18_mean", "col18_std"}
REQUIRED_GUARDS = ["saved_parameters", "saved_buffers", "saved_scalers", "source_files",
                   "native_inputs", "graph", "readouts", "affine_closure", "bridge_artifacts",
                   "query_binding", "scaler_arithmetic", "ensemble_arithmetic", "selection", "golden_proof"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path, maximum=512 * 1024 * 1024):
    path = Path(path)
    require(path.is_file() and path.stat().st_size <= maximum, "missing/oversize JSON artifact")
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def helper_hashes():
    return {"h256_validate.py": shared.sha(shared.__file__),
            "validate_trace.py": shared.sha(validate_trace.__file__),
            "h256_functional.py": shared.sha(h256_functional.__file__),
            "h256_stage2_inference.py": shared.sha(Path(__file__).with_name("h256_stage2_inference.py"))}


def artifact(root, record, expected=None):
    path = shared.safe_file(root, record["path"])
    require(shared.sha(path) == record["sha256"], "saved artifact hash differs: " + record["path"])
    if "bytes" in record:
        require(path.stat().st_size == record["bytes"], "saved artifact size differs")
    if expected is not None:
        require(record["sha256"] == expected, "unsupported saved artifact identity")
    return path


def bound_values(record, count, cache):
    require(isinstance(record, dict) and bool(record.get("source_ref")), "bridge input has no source")
    path = Path(record["artifact_path"])
    key = str(path)
    if key not in cache:
        cache[key] = (shared.sha(path), read_json(path))
    require(cache[key][0] == record["artifact_sha256"], "bridge artifact hash differs")
    dtype = record.get("dtype", "float32")
    require(dtype in {"float32", "float64"}, "bridge input dtype differs")
    values = np.asarray(record["values"], dtype=np.float32 if dtype == "float32" else np.float64).reshape(-1)
    require(values.size == count and bool(np.isfinite(values).all()), "bridge input count/finiteness differs")
    raw = shared.typed_bytes(values, dtype)
    require(hashlib.sha256(raw).hexdigest() == record["values_sha256"], "bridge numeric hash differs")
    original = cache[key][1]
    for part in record["json_path"]:
        original = original[part]
    require(raw == shared.typed_bytes(np.asarray(original).reshape(-1), dtype), "bridge cells differ from bound artifact")
    return values, cache[key][1]


def saved_inventory(root, data, parameters):
    """Restricted CPU-only checkpoint loading; never called by receipt-backed Blender."""
    from h256_stage2_inference import load_saved
    ensemble = load_saved(root / data["checkpoint"]["path"])
    round_state = load_saved(root / data["dope"]["state_checkpoint"]["path"])["state_dict"]
    require(ensemble["dope_sha256"] == shared.FROZEN_DOPE and len(ensemble["fold_state_dicts"]) == 5,
            "ensemble lineage/fold count differs")
    folds = ensemble["fold_state_dicts"]
    for state in folds:
        require(len(state) == 84 and sum(v.numel() for v in state.values()) == 183668,
                "saved fold parameter inventory differs")
    seen = [set() for _ in range(5)]
    for parameter in parameters.values():
        origin = parameter["checkpoint_origin"]
        pid = parameter["id"]
        if pid.startswith("h256.fold"):
            match = re.fullmatch(r"h256\.fold([0-4])\.(.+)", pid)
            require(match is not None, "invalid fold parameter identity")
            index, key = int(match[1]), match[2]
            require(origin == f"assets/stage2-ensemble.pt::fold_state_dicts/{index}/{key}"
                    and parameter["role"] == "solution_encoder", "fold checkpoint origin differs")
            require(key in folds[index], "unknown saved fold parameter")
            original = folds[index][key]
            seen[index].add(key)
        else:
            if pid.startswith("dope."):
                key = "dope_model." + pid.removeprefix("dope.")
                require(parameter["role"] == "problem_encoder", "DOPE role differs")
            else:
                require(pid.startswith("base.") and parameter["role"] == "solution_encoder", "base role/identity differs")
                key = "adapter_10tok." + pid.removeprefix("base.")
            require(origin == "assets/round-882.pt::state_dict/" + key and key in round_state,
                    "base/DOPE checkpoint origin differs")
            original = round_state[key]
        require(str(original.dtype) == "torch.float32" and list(original.shape) == parameter["shape"]
                and shared.f32_bytes(original.detach().cpu().numpy()) == shared.f32_bytes(parameter["values"]),
                "parameter differs from actual checkpoint bytes")
    require(all(seen[i] == set(folds[i]) for i in range(5)), "saved fold parameters omitted")
    for case in data["cases"]:
        for tensor in case["tensors"]:
            if "checkpoint_origin" not in tensor:
                continue
            prefix = "assets/round-882.pt::state_dict/"
            require(tensor["checkpoint_origin"].startswith(prefix), "unknown saved buffer checkpoint")
            key = tensor["checkpoint_origin"].removeprefix(prefix)
            require(key in round_state, "saved buffer is absent from checkpoint")
            original = round_state[key]
            require(list(original.shape) == tensor["shape"] and tensor["dtype"] == "float32"
                    and shared.f32_bytes(original.detach().cpu().numpy()) == shared.f32_bytes(tensor["values"]),
                    "root buffer differs from saved state")
    require(data["scaler_state"]["mode"] == "canonical_full_train"
            and data["scaler_state"]["key"] == "full_train_scaler", "require canonical shared training scaler")
    scaler = ensemble["full_train_scaler"]
    require(set(scaler) == SCALER_NAMES, "checkpoint scaler keys differ")
    require(float(ensemble["ensemble_alpha_prior"]) == float(data["checkpoint"]["alpha_prior"])
            and float(ensemble["ensemble_alpha_traj"]) == float(data["checkpoint"]["alpha_trajectory"]),
            "ensemble alpha differs from saved checkpoint")
    return [{name: np.asarray(scaler[name], dtype=np.float32).tolist() for name in sorted(scaler)} for _ in range(5)]


def check_scalers(data, scalers):
    require(len(scalers) == 5 and all(set(s) == SCALER_NAMES for s in scalers), "verified scaler inventory differs")
    state = data["scaler_state"]
    require(state["checkpoint_path"] == data["checkpoint"]["path"] and state["mode"] == "canonical_full_train"
            and state["key"] == "full_train_scaler", "wrong scaler contract")
    buffers = state["buffers"]
    require(len(buffers) == 30 and {(b["fold"], b["name"]) for b in buffers}
            == {(i, name) for i in range(5) for name in SCALER_NAMES}, "saved scaler inventory omitted/duplicated")
    for buffer in buffers:
        value = np.asarray(scalers[buffer["fold"]][buffer["name"]], dtype=np.float32)
        require(list(value.shape) == buffer["shape"] and bool(np.isfinite(value).all())
                and hashlib.sha256(shared.f32_bytes(value)).hexdigest() == buffer["sha256"], "saved scaler buffer differs")
        if buffer["name"].endswith("_std"):
            require(bool((value >= 1e-4).all()), "invalid saved scaler standard deviation")


def check_scaler_arithmetic(case, by_tensor, scalers):
    raw = np.asarray(case["outputs"]["dope629"], dtype=np.float32).reshape(629)
    hidden = np.asarray(case["outputs"]["hidden256"], dtype=np.float32).reshape(256)
    columns = next(t for t in case["tensors"] if t["name"] == "fit.col64x17")
    col = np.asarray(columns["values"], dtype=np.float32).reshape(64, 17)
    mask = (np.arange(64) < case["features"]).astype(np.float32)
    col18 = np.concatenate([col, raw[565:629, None]], axis=-1)
    operations = [o for o in case["operations"] if o["kind"] == "saved_scaler_standardize_clip15_and_col18_recipe"]
    require(len(operations) == 5, "missing saved-scaler native calls")
    for index, scaler in enumerate(scalers):
        scale = {name: np.asarray(value, dtype=np.float32) for name, value in scaler.items()}
        dope = np.clip((raw - scale["dope629_mean"]) / scale["dope629_std"], -15., 15.)
        hid = np.clip((hidden - scale["hidden256_mean"]) / scale["hidden256_std"], -15., 15.)
        cols = np.clip((col18 - scale["col18_mean"][None, :]) / scale["col18_std"][None, :], -15., 15.) * mask[:, None]
        rms = lambda x: float(np.log1p(np.sqrt(np.mean(x * x) + 1e-8)))
        maximum = lambda x: float(np.log1p(np.max(np.abs(x))))
        norms = np.asarray([rms(dope[:256]), maximum(dope[:256]), rms(dope[256:384]), maximum(dope[256:384]),
                            rms(dope[384:512]), maximum(dope[384:512]), rms(dope[512:526]), rms(dope[526:549]),
                            rms(dope[549:565]), rms(dope[565:629]), maximum(dope[565:629])], dtype=np.float32)
        op = next(o for o in operations if o["name"] == f"h256.fold{index}.FoldScaler.transform_context")
        require(len(op["output_tensor_ids"]) == 5, "scaler output inventory differs")
        for tid, expected in zip(op["output_tensor_ids"], [dope, hid, cols, mask, norms]):
            tensor = by_tensor[tid]
            require(tensor["dtype"] == "float32" and shared.f32_bytes(tensor["values"]) == shared.f32_bytes(expected),
                    "saved scaler source arithmetic differs")


def check_case(case, parameters, scalers, cache):
    require(case["scope"] == "exact_recorded_training_contract" and case["trajectory_executed"] is False,
            "this gate requires genuine zero-reveal Stage 2 inference")
    fit, hold = case["fit_row_ids"], case["holdout_row_ids"]
    require(bool(fit) and bool(hold) and len(set(fit)) == len(fit) and len(set(hold)) == len(hold)
            and not set(fit) & set(hold) and set(fit) | set(hold) == set(range(case["rows"])), "bad FIT partition")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", case["dataset_sha256"])), "missing dataset digest")
    require(case.get("tokens_per_second") is None, "unverified neural token throughput")
    require("functional_dependencies" in case, "functional dependency inventory absent")
    by_tensor, by_op = shared.check_graph(case, parameters)
    shared.check_readouts(case, by_tensor)
    shared.check_native_inputs(case, cache)
    # PyTorch MHA uses this exact nn.Linear subclass for out_proj. The frozen
    # Stage 1 checker predates that path; normalize its type only in the checker
    # view, leaving every observed operand, parameter, identity and DAG unchanged.
    affine_ops = dict(by_op)
    for opid, operation in by_op.items():
        if operation["kind"] == "NonDynamicallyQuantizableLinear":
            require(operation["observation"] == "native-module-hook", "unobserved MHA output projection")
            affine_ops[opid] = {**operation, "kind": "Linear"}
    closure = shared.check_affines(case, by_tensor, affine_ops, parameters)
    functional, functional_receipt = h256_functional.derived_mha_affines(
        {"schema": SCHEMA, "parameters": list(parameters.values())}, case)
    check_scaler_arithmetic(case, by_tensor, scalers)
    binding = case["bridge_inputs"]
    require(binding["id"] == case["id"] and binding["dataset_sha256"] == case["dataset_sha256"]
            and binding["programs"] == case["programs"], "bridge input identity differs")
    require(binding.get("trajectory") is None and binding["reveal_count"] == 0
            and binding["base_capture"]["reveals"] == 0
            and binding["base_capture"]["history_costs_used"] is False, "fabricated reveal history")
    require(not any("delta_trajectory" in name for name in case["outputs"])
            and not any(op.get("execution_scope") == "original_stage2_trajectory_forward" for op in case["operations"]),
            "zero-reveal capture invents a trajectory invocation")
    programs, outputs = case["programs"], case["outputs"]
    count = len(programs)
    require(5 <= count <= 128 and len(set(programs)) == count, "candidate slate is invalid")
    arrays = {}
    critic = None
    for name in ("gate_risk", "sp_prior", "initial_cost_before_g", "r882_k0_total_cost"):
        arrays[name], current = bound_values(binding[name], count, cache)
        require(current["critic_sha256"] == CRITIC_SHA and current["frozen_round882_sha256"] == ROUND_SHA,
                "critic/base input checkpoint differs")
        require(critic is None or current is critic, "bridge inputs were spliced across artifacts")
        critic = current
    critic_case = next(c for c in critic["cases"] if c["id"] == case["id"])
    require(critic_case["dataset_sha256"] == case["dataset_sha256"], "critic input dataset differs")
    base = np.asarray(outputs["r882_k0_total_cost"], dtype=np.float64).reshape(-1)
    require(np.max(np.abs(base - arrays["r882_k0_total_cost"])) < 2e-5, "fresh base differs from bound critic base")
    h_cost = np.asarray(critic_case["arrays"]["h_cost"], dtype=np.float64).reshape(-1)
    require(np.allclose(arrays["initial_cost_before_g"], arrays["r882_k0_total_cost"] + .005 * h_cost,
                        rtol=1e-12, atol=1e-12), "initial cost recipe differs")
    journal = cache[case["inputs"]["native_journal_path"]][1]
    call = next(c for c in journal["calls"] if c["call_id"] == binding["native_call_id"])
    identity = call["identity"]
    require(programs == identity["revealed_programs"] + identity["proposal_programs"], "candidate native slate differs")
    native_query = next(a for a in call["trace"]["affine"] if a["name"] == "cand_query_proj.0")
    query = next(a for a in case["trace"]["affine"] if a["name"] == "base.cand_query_proj.0")
    require(query["input_shape"] == [1, count, 66] and native_query["input_shape"] == [count, 66], "native query shape differs")
    query_error = float(np.max(np.abs(np.asarray(query["input"], dtype=np.float32).reshape(count, 66)
                                    - np.asarray(native_query["input"], dtype=np.float32).reshape(count, 66))))
    require(query_error < 3e-5, "fresh original candidate query differs from native reference")
    feature_path = Path(critic["native_features_path"])
    feature_key = str(feature_path)
    if feature_key not in cache:
        cache[feature_key] = (shared.sha(feature_path), read_json(feature_path))
    require(cache[feature_key][0] == critic["native_features_sha256"], "native program feature archive differs")
    features = next(c for c in cache[feature_key][1]["cases"] if c["id"] == case["id"])
    require(features["dataset_sha256"] == case["dataset_sha256"] and features["programs"] == programs
            and features["native_call_id"] == binding["native_call_id"], "native feature case identity differs")
    def root_reaches(root_id, target_id):
        todo, seen = [target_id], set()
        while todo:
            tid = todo.pop()
            if tid == root_id:
                return True
            if tid in seen or tid not in by_tensor:
                continue
            seen.add(tid)
            producer = by_tensor[tid]["producer_op_id"]
            if producer is not None:
                todo.extend(by_op[producer]["input_tensor_ids"])
        return False
    raw_bytes = shared.f32_bytes(features["raw_program_features412"])
    query_input_id = by_op[query["operation_id"]]["input_tensor_ids"][0]
    require(any(t["producer_op_id"] is None and t["shape"] == [1, count, 412]
                and shared.f32_bytes(t["values"]) == raw_bytes and root_reaches(t["id"], query_input_id)
                for t in case["tensors"]), "fresh base raw412 root differs from native feature artifact")
    projection_bytes = shared.typed_bytes(features["projection16_f64"], "float64")
    require(any(t["producer_op_id"] is None and t["dtype"] == "float64" and t["shape"] == [1, count, 16]
                and shared.typed_bytes(t["values"], "float64") == projection_bytes
                and root_reaches(t["id"], case["outputs_tensor_ids"]["r882_k0_total_cost"]) for t in case["tensors"]),
            "fresh base frozen projection16 input differs")
    require(bool((np.asarray(query["input"]).reshape(count, 66)[:, -2:] == 0).all()), "SOS source contract differs")
    candidate = np.asarray(outputs["candidate_query"], dtype=np.float32).reshape(count, 32)
    priors = []
    for index in range(5):
        prefix = f"fold{index}."
        prior = np.asarray(outputs[prefix + "delta_prior"], dtype=np.float32).reshape(-1)
        require(prior.size == count and bool(np.isfinite(prior).all()), "invalid fold prior")
        priors.append(prior)
        mask = np.asarray(outputs[prefix + "context_mask_92"])
        expected_mask = np.concatenate([np.ones(28), np.arange(64) < case["features"]])
        require(mask.size == 92 and np.array_equal(mask.reshape(-1), expected_mask), "invalid context mask")
        require(np.asarray(outputs[prefix + "context_tokens_92x32"]).size == 92 * 32, "wrong context token shape")
        affine = next(a for a in case["trace"]["affine"] if a["name"] == f"h256.fold{index}.prior_residual_mlp.0")
        cells = np.asarray(affine["input"], dtype=np.float32).reshape(count, 98)
        require(shared.f32_bytes(cells[:, -2]) == shared.f32_bytes(arrays["gate_risk"])
                and shared.f32_bytes(cells[:, -1]) == shared.f32_bytes(arrays["sp_prior"])
                and shared.f32_bytes(cells[:, 56:88]) == shared.f32_bytes(candidate), "bridge/query values do not reach actual native prior input")
    mean = np.stack(priors).astype(np.float64).mean(axis=0)
    alpha = float(binding["alpha_prior"])
    weighted, final = alpha * mean, arrays["initial_cost_before_g"] + alpha * mean
    for name, expected in (("ensemble_delta_prior", mean), ("ensemble_weighted_residual", weighted), ("final_k0_cost", final)):
        value = np.asarray(outputs[name], dtype=np.float64).reshape(-1)
        require(value.size == count and np.allclose(value, expected, rtol=1e-12, atol=1e-12), "fivefold ensemble arithmetic differs: " + name)
    selection = case["selection"]
    selected = 4 + int(np.argmin(final[4:]))
    require(selection["selected_slate_index"] == selected and selection["selected_proposal_index"] == selected - 4
            and selection["selected_program"] == programs[selected] and selection["outcome_based_selection"] is False
            and selection["reveal_count"] == 0, "G minimum-cost proposal selection differs")
    validation = case["validation"]
    require(set(validation["observer_on_off_exact"]) == set(outputs)
            and all(value == 0 for value in validation["observer_on_off_exact"].values())
            and validation["corrupted_output_negative_control"] == "rejected", "capture observation controls failed")
    return {"id": case["id"], "affine_operations": len(case["trace"]["affine"]), "graph_operations": len(by_op),
            "additional_functional_matrix_operations": len(functional), "functional_matrix_proof": functional_receipt,
            "tensors": len(by_tensor), "max_normalized_closure_error": closure, "native_query_max_abs_error": query_error,
            "selected_slate_index": selected, "selected_program_sha256": hashlib.sha256(programs[selected].encode()).hexdigest()}


def validate(path, saved_receipt=None, gp_root=None):
    path = Path(path).resolve()
    data = read_json(path)
    require(data.get("schema") == SCHEMA and data.get("status") == "complete", "wrong/incomplete Stage 2 capture")
    require(data["inference_scope"] == "exact_recorded_training_contract", "adapted diagnostic cannot satisfy strict Stage 2 gate")
    root, coverage = path.parent, data["coverage"]
    require(coverage["stage2"] is True and coverage["stage1_only"] is False and coverage["dropped_records"] == 0,
            "wrong/lossy capture scope")
    require(data["runtime"]["training"] is False and data["runtime"]["optimizer"] is None
            and data["validation"]["parameter_before_equals_after"] is True, "inference changed trained parameters")
    checkpoint = data["checkpoint"]
    artifact(root, checkpoint, ENSEMBLE_SHA)
    artifact(root, data["dope"]["state_checkpoint"], ROUND_SHA)
    artifact(root, data["critic_state_checkpoint"], CRITIC_SHA)
    golden = data["golden_fixture"]
    artifact(root, golden, GOLDEN_SHA)
    golden_proof_path = shared.safe_file(root, golden["proof_receipt_path"])
    require(shared.sha(golden_proof_path) == golden["proof_receipt_sha256"], "golden proof bytes differ")
    proof = read_json(golden_proof_path)
    require(proof["status"] == "PASS" and proof["checkpoint_sha256"] == ENSEMBLE_SHA
            and proof["critic_sha256"] == CRITIC_SHA and proof["golden_sha256"] == GOLDEN_SHA, "golden proof lineage differs")
    for name, value in proof["checks"].items():
        require(value is True if name.endswith("_exact") else isinstance(value, (int, float)) and 0 <= value <= 2e-6,
                "golden inference guard failed")
    require(checkpoint["fold_count"] == 5 and checkpoint["tensor_count_per_fold"] == 84
            and checkpoint["parameter_count_per_fold"] == 183668 and checkpoint["scaler_mode"] == "canonical_full_train",
            "unsupported saved Stage 2 inventory")
    dope = data["dope"]
    require(dope["composite_expected"] == dope["before"] == dope["after"] == shared.FROZEN_DOPE, "frozen DOPE changed")
    sources = data["sources_manifest"]
    require(bool(sources) and len({s["path"] for s in sources}) == len(sources), "missing/duplicate sources")
    for source in sources:
        artifact(root, source)
    entries = data["parameters"]
    parameters = {p["id"]: p for p in entries}
    require(len(entries) == len(parameters) == 590, "saved parameter inventory differs")
    require(sum(p["id"].startswith("dope.") for p in entries) == 56
            and sum(p["id"].startswith("base.") for p in entries) == 114, "DOPE/base parameter inventory differs")
    for parameter in entries:
        require(parameter["role"] in shared.ROLES and parameter["dtype"] == "float32", "parameter role/dtype differs")
        require(hashlib.sha256(shared.f32_bytes(shared.tensor_values(parameter))).hexdigest() == parameter["sha256"], "parameter numeric hash differs")
    inventory_sha = shared.parameter_inventory_sha(parameters)
    scaler_sha, sources_sha = digest(data["scaler_state"]), digest(sources)
    helpers, trace_sha = helper_hashes(), shared.sha(path)
    if saved_receipt is None:
        scalers = saved_inventory(root, data, parameters)
    else:
        verified = read_json(saved_receipt, 4 * 1024 * 1024)
        require(verified.get("schema") == RECEIPT_SCHEMA and verified.get("status") == "PASS"
                and verified.get("trace_sha256") == trace_sha and verified.get("verifier_sha256") == shared.sha(__file__)
                and verified.get("shared_helpers_sha256") == helpers
                and verified.get("parameter_inventory_sha256") == inventory_sha
                and verified.get("scaler_inventory_sha256") == scaler_sha and verified.get("source_inventory_sha256") == sources_sha
                and verified.get("saved_parameters_verified") is True
                and verified.get("guards") == {key: "PASS" for key in REQUIRED_GUARDS}, "saved Stage 2 verification receipt differs")
        scalers = verified["verified_scalers"]
        require(digest(scalers) == verified["verified_scalers_sha256"], "verified scaler payload differs")
    check_scalers(data, scalers)
    cases = data["cases"]
    require(1 <= len(cases) <= 16 and len({c["id"] for c in cases}) == len(cases), "invalid case inventory")
    cache, results, executed = {}, [], set()
    for case in cases:
        require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", case["id"]), "invalid case ID")
        require(float(case["bridge_inputs"]["alpha_prior"]) == float(checkpoint["alpha_prior"])
                and float(case["bridge_inputs"]["alpha_trajectory"]) == float(checkpoint["alpha_trajectory"]), "case ensemble alpha differs")
        results.append(check_case(case, parameters, scalers, cache))
        executed.update(p for op in case["operations"] for p in op["parameter_ids"])
    require(executed == set(coverage["executed_parameter_ids"])
            and set(parameters) - executed == set(coverage["unused_parameter_ids"])
            and all(p["executed"] == (p["id"] in executed) for p in entries), "executed parameter coverage differs")
    gp_results = None
    if gp_root is not None or any(case.get("source") is not None for case in cases):
        require(gp_root is not None, "source-bearing capture requires GP proof root")
        from h256_gp_validate import validate_gp_sources
        gp_results = validate_gp_sources(data, path, gp_root)
    report = {"schema": RECEIPT_SCHEMA, "status": "PASS", "trace_sha256": trace_sha,
              "verifier_sha256": shared.sha(__file__), "shared_helpers_sha256": helpers,
              "checkpoint_sha256": ENSEMBLE_SHA, "critic_sha256": CRITIC_SHA, "golden_sha256": GOLDEN_SHA,
              "golden_proof_sha256": golden["proof_receipt_sha256"],
              "saved_parameters_verified": True, "parameter_inventory_sha256": inventory_sha,
              "scaler_inventory_sha256": scaler_sha, "source_inventory_sha256": sources_sha,
              "verified_scalers": scalers, "verified_scalers_sha256": digest(scalers),
              "guards": {key: "PASS" for key in REQUIRED_GUARDS}, "cases": results,
              "gp_sources": gp_results,
              "scope": "saved five-fold H256 zero-reveal prior inference; optional GP source requires independent causal proof"}
    return data, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--saved-receipt", type=Path)
    parser.add_argument("--gp-root", type=Path)
    args = parser.parse_args()
    _, receipt = validate(args.trace, args.saved_receipt, args.gp_root)
    if args.receipt:
        with args.receipt.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "verified_scalers"}, indent=2))
