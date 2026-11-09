#!/usr/bin/env python3
"""Observe saved H256 Stage 2 folds with explicitly supplied, provenance-bound inputs.

The Stage 2 checkpoint omits the separately fitted hurdle/SP input predictors.
This observer therefore refuses to invent those priors or trajectory observations.
Its required bridge-input manifest must declare their actual source and scope.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

import h256_inference as stage1

SCHEMA = "jain.h256-stage2-capture/v1"
ENSEMBLE_SHA = "ee40e191e4ba8f2b0a3e0a34a44746984ea2aebafc277c588f71f73f7de323c8"
UNIFIED_SHA = "b91046e6bd1f9ec70efdb546d0ceb1203fb48f82d406b5ed19eee457b1ab3de8"
QUERY_SOURCE_SHA = "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a"


def load_saved(path):
    import _codecs
    allowed = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
               np.dtypes.BoolDType, np.dtypes.Int64DType, np.dtypes.Int32DType, _codecs.encode]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def original_query_module(path, state):
    stage1.require(stage1.sha(path) == QUERY_SOURCE_SHA, "original query source differs")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlywheelBlockFormerAdapterNet")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Attribute) and t.attr == "cand_query_proj" for t in n.targets))
    holder = nn.Module()
    # Execute the unchanged original module-construction statement, without unrelated imports.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), str(path), "exec"),
         {"self": holder, "nn": nn, "d_token": 32})
    prefix = "adapter_10tok.cand_query_proj."
    holder.cand_query_proj.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    holder.eval()
    for parameter in holder.parameters():
        parameter.requires_grad_(False)
    return holder


def attach_stage1b(model, root, state):
    """Bind unchanged original descriptor/landmark/token projection methods and saved buffers."""
    for key, value in state.items():
        if "." not in key and not hasattr(model, key):
            model.register_buffer(key, value.clone())
    path = root / "source/cpu_exact_fused.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedJopeNetwork")
    names = {"_compute_lm_smart_80d_gpu", "_svd_slot_project", "_project_stage1b_tokens_batched_gpu"}
    functions = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    for name in names:
        setattr(model, name, types.MethodType(namespace[name], model))


def attach_sync_free_attention(adapter, root):
    """Use the unchanged deployed fused MHA wrapper, including its finite -1e4 mask."""
    path = root / "source/cpu_exact_fused.py"
    stage1.require(stage1.sha(path) == stage1.EXPECTED["source/cpu_exact_fused.py"], "fused attention source differs")
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sync_free_mha_forward")
    namespace = {"torch": torch, "F": torch.nn.functional}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    for module in adapter.modules():
        if isinstance(module, nn.MultiheadAttention):
            module.forward = types.MethodType(namespace[fn.name], module)


def source_files(root, custody):
    files = {p: root / p for p in stage1.EXPECTED if p.startswith("source/")}
    for name in ("pre_condense_models.py", "run_h256_flywheel_ablation.py", "pretrain_full_s3_7split_bridge.py",
                 "phase12_multimodal_model.py", "flywheel_model.py", "common_h252.py", "stage1_hurdle_sos_critic.py", "part_a_causal_2plus2_neural.py"):
        files["source/stage2_" + name] = custody / "source" / name
    return files


def bound_array(record, count, name):
    """Require an immutable input artifact, exact numeric cells and an explicit source scope."""
    stage1.require(isinstance(record, dict) and bool(record.get("source_ref")), name + " has no source reference")
    source = Path(record["artifact_path"])
    stage1.require(source.is_file() and stage1.sha(source) == record["artifact_sha256"], name + " input artifact differs")
    dtype = record.get("dtype", "float32")
    stage1.require(dtype in {"float32", "float64"}, name + " has unsupported input dtype")
    np_dtype = np.float64 if dtype == "float64" else np.float32
    little_dtype = "<f8" if dtype == "float64" else "<f4"
    a = np.asarray(record["values"], dtype=np_dtype).reshape(-1)
    stage1.require(a.size == count and bool(np.isfinite(a).all()), name + " has invalid numeric cells")
    stage1.require(hashlib.sha256(a.astype(little_dtype).tobytes()).hexdigest() == record["values_sha256"], name + " numeric hash differs")
    source_values = json.loads(source.read_text())
    for key in record["json_path"]:
        source_values = source_values[key]
    stage1.require(np.array_equal(np.asarray(source_values, dtype=np_dtype).reshape(-1), a),
                   name + " values do not match bound artifact cells")
    return torch.tensor(a).reshape(1, count)


def inventory(model, base_adapter, bridges):
    entries, ids, tensors = [], {}, {}
    groups = [("dope", model.dope_model, "problem_encoder", "assets/round-882.pt::state_dict/dope_model."),
              ("base", base_adapter, "solution_encoder", "assets/round-882.pt::state_dict/adapter_10tok.")]
    groups += [(f"h256.fold{i}", bridge, "solution_encoder",
                f"assets/stage2-ensemble.pt::fold_state_dicts/{i}/") for i, bridge in enumerate(bridges)]
    for prefix, module, role, origin in groups:
        for name, parameter in module.named_parameters():
            pid = prefix + "." + name
            ids[id(parameter)] = pid
            tensors[pid] = parameter
            entries.append({"id": pid, "name": pid, "role": role, "shape": list(parameter.shape),
                "dtype": "float32", "values": stage1.values(parameter), "sha256": stage1.f32_sha(parameter),
                "checkpoint_origin": origin + name, "executed": False})
    return entries, ids, tensors


def case_forward(case, bridge_input, native_case, model, dope_fn, base_adapter, bridges, scalers,
                 parameter_ids, observe):
    observer = stage1.Observer(parameter_ids) if observe else None
    hidden = []
    hidden_handle = model.dope_model.router.task_latent_head.register_forward_pre_hook(lambda _m, args: hidden.append(args[0]))
    named_readouts = {}
    output_handles = []
    if observer:
        observer.attach(model.dope_model, "dope")
        observer.attach(base_adapter, "base")
        for index, bridge in enumerate(bridges):
            observer.attach(bridge, f"h256.fold{index}")
    for index, bridge in enumerate(bridges):
        # The context pack is observed directly at the actual method boundary.
        original_encode = bridge.encode_context_tokens
        def observed_encode(ctx, original_encode=original_encode, index=index):
            result = original_encode(ctx)
            named_readouts[f"fold{index}.context_tokens_92x32"] = result[0]
            named_readouts[f"fold{index}.context_mask_92"] = result[1]
            named_readouts[f"fold{index}.scale_side_8"] = result[2]
            return result
        bridge.encode_context_tokens = observed_encode
        output_handles.append((bridge, original_encode))
    journal_path = Path(case["provenance"]["native_journal_path"])
    stage1.require(stage1.sha(journal_path) == case["provenance"]["native_journal_sha256"], "native journal differs")
    journal = json.loads(journal_path.read_text())
    conditioning = next(c for c in journal["calls"] if "v256" in c["outputs"])
    if observer:
        observer.conditioning_call_id = conditioning["call_id"]
    slate = next(c for c in journal["calls"] if c["call_id"] == bridge_input["native_call_id"])
    identity = slate["identity"]
    programs = identity["revealed_programs"] + identity["proposal_programs"]
    stage1.require(programs == bridge_input["programs"], "candidate identities/order differ")
    query_affine = next(a for a in slate["trace"]["affine"] if a["name"] == "cand_query_proj.0")
    native_query = torch.tensor(query_affine["input"], dtype=torch.float32).reshape(query_affine["input_shape"])
    m = len(programs)
    stage1.require(native_query.shape == (m, 66), "candidate query shape differs")
    stage1.require(bool((native_query[:, -2:] == 0).all()), "serving query SOS contract is not source-prescribed zero")
    q_input = native_query.unsqueeze(0)
    v = torch.tensor(case["v256"], dtype=torch.float32).reshape(1, 256)
    col = torch.tensor(case["col64x17"], dtype=torch.float32).reshape(1, 64, 17)
    if observer:
        observer.tensor(v, name="fit.v256")
        observer.tensor(col, name="fit.col64x17")
    gate_risk = bound_array(bridge_input["gate_risk"], m, "gate_risk")
    sp_prior = bound_array(bridge_input["sp_prior"], m, "sp_prior")
    initial_cost_before_g = bound_array(bridge_input["initial_cost_before_g"], m, "initial_cost_before_g")
    trajectory = bridge_input.get("trajectory")
    if trajectory is not None:
        rev_idx_values = trajectory["revealed_indices"]
        stage1.require(len(rev_idx_values) == 4 and len(set(rev_idx_values)) == 4
                       and all(type(i) is int and 0 <= i < m for i in rev_idx_values), "invalid actual reveal indices")
        rev_idx = torch.tensor(rev_idx_values, dtype=torch.long).reshape(1, 4)
        y_comb = bound_array(trajectory["combined_observations"], 4, "combined_observations")
        y_f = bound_array(trajectory["fitness_observations"], 4, "fitness_observations")
        rev_gate = bound_array(trajectory["gate_observations"], 4, "gate_observations")
        base_cost = bound_array(trajectory["base_cost_before_prior"], m, "base_cost_before_prior")
        stage1.require(bool(((rev_gate == 0) | (rev_gate == 1)).all()), "reveal gate is not boolean")
    def call(function, *args, **kwargs):
        if observer:
            with observer:
                return function(*args, **kwargs)
        return function(*args, **kwargs)
    try:
        with torch.no_grad():
            dope629, fresh_lm48 = call(dope_fn, model, v, col, use_landmarks=True)
            h256 = hidden[-1]
            if observer:
                observer.role = "solution_encoder"
                observer.execution_scope = "original_legacy_context_projection"
                observer.call_source = "source/cpu_exact_fused.py::_project_stage1b_tokens_batched_gpu"
            fresh_tokens, fresh_mask, fresh_lm48_z = call(model._project_stage1b_tokens_batched_gpu, fresh_lm48, dope629)
            if observer:
                observer.reference_checks = {}
                for key, tensor in (("dope629", dope629), ("lm48", fresh_lm48), ("lm48_z", fresh_lm48_z),
                                    ("tokens", fresh_tokens), ("mask", fresh_mask)):
                    error = float(np.max(np.abs(stage1.tensor_array(tensor).reshape(-1)
                        - np.asarray(conditioning["outputs"][key], dtype=np.float32).reshape(-1))))
                    stage1.require(error < 3e-4, "fresh original conditioning differs from native reference " + key)
                    observer.reference_checks["native_" + key + "_max_abs_error"] = error
            if observer:
                observer.execution_scope = "original_r882_base_zero_reveal"
                observer.call_source = "source/stage2_phase12_multimodal_model.py::FlywheelBlockFormerAdapterNet.predict_costs K_rev0"
            stage1.require(native_case["programs"] == programs, "raw native candidate identities differ")
            raw_program = torch.tensor(native_case["raw_program_features412"], dtype=torch.float32).reshape(1, m, 412)
            projection = torch.tensor(native_case["projection16_f64"], dtype=torch.float64).reshape(1, m, 16)
            base = call(base_adapter.predict_costs, ctx_feat=torch.zeros((1, 584)), prog_412=raw_program,
                proj_16=projection, base_prior_cost=torch.zeros((1, m), dtype=torch.float64),
                scale=torch.tensor([.4208], dtype=torch.float64), sos_prior=torch.zeros((1, m, 2)),
                mm_tokens_10x32=fresh_tokens, mm_mask_10=fresh_mask,
                revealed_indices=torch.empty((1, 0), dtype=torch.long), revealed_raw_f=torch.empty((1, 0), dtype=torch.float64),
                revealed_gate=torch.empty((1, 0)))
            cand_q, sem24 = base["cand_q"], base["sem24"]
            expected_base = np.asarray(bridge_input["r882_k0_total_cost"]["values"], dtype=np.float64)
            base_error = float(np.max(np.abs(stage1.tensor_array(base["total_cost"]).reshape(-1) - expected_base)))
            stage1.require(base_error < 2e-5,
                           "fresh observed zero-reveal base differs from critic feature source")
            if observer:
                observer.reference_checks["critic_source_base_cost_max_abs_error"] = base_error
            named_readouts.update({"dope629": dope629, "hidden256": h256, "candidate_query": cand_q,
                                   "semantic_signature_24": sem24, "fresh_descriptor_projection_lm48": fresh_lm48,
                                   "fresh_lm48_z": fresh_lm48_z, "base_tokens_10x32": fresh_tokens, "base_mask_10": fresh_mask,
                                   "r882_k0_total_cost": base["total_cost"], "r882_k0_effective_prior": base["eff_prior"]})
            entry = {"dope629": stage1.tensor_array(dope629).reshape(629), "hidden256": stage1.tensor_array(h256).reshape(256),
                "col_lm_64x17": stage1.tensor_array(col).reshape(64, 17), "col_mask_64": np.arange(64) < case["features"],
                "tokens_10x32": stage1.tensor_array(fresh_tokens).reshape(10, 32),
                "mask_10": stage1.tensor_array(fresh_mask).reshape(10)}
            priors, attns, bilins = [], [], []
            for index, (bridge, scaler) in enumerate(zip(bridges, scalers)):
                ctx = scaler.transform_context(entry, device=torch.device("cpu"))
                if observer:
                    observer.execution_scope = "original_stage2_prior_forward"
                    observer.call_source = "source/stage2_pre_condense_models.py::FoldScaler.transform_context"
                    observer.record(f"h256.fold{index}.FoldScaler.transform_context", "saved_scaler_standardize_clip15_and_col18_recipe",
                        (dope629, h256, col), {k: ctx[k] for k in ("dope629_z", "hidden256_z", "col18_z", "col_mask_64", "block_log_norms")},
                        observation="executed-source-function")
                    observer.call_source = "source/stage2_pre_condense_models.py::PreCondenseResidualBridge.forward_prior_and_context"
                dp, attn, bilin = call(bridge.forward_prior_and_context, ctx, cand_q, sem24, gate_risk, sp_prior)
                priors.append(dp)
                attns.append(attn)
                bilins.append(bilin)
                named_readouts.update({f"fold{index}.delta_prior": dp, f"fold{index}.attention_context": attn,
                                       f"fold{index}.bilinear_24": bilin,
                                       f"fold{index}.block_log_norms_11": ctx["block_log_norms"]})
            if observer:
                observer.execution_scope = "saved_five_fold_prediction_mean"
                observer.call_source = "source/stage2_run_h256_flywheel_ablation.py::evaluate_context_with_model"
            prior_mean = call(lambda: torch.stack(priors, dim=0).double().mean(dim=0))
            if trajectory is None:
                # K_rev=0: execute the genuine pre-reveal prior API. No trajectory
                # tensor or zero-valued prediction is fabricated for unused heads.
                combined_residual = call(lambda: float(bridge_input["alpha_prior"]) * prior_mean)
                final_k0_cost = call(lambda: initial_cost_before_g + combined_residual.double())
                named_readouts.update({"ensemble_delta_prior": prior_mean,
                                       "ensemble_weighted_residual": combined_residual,
                                       "final_k0_cost": final_k0_cost})
                return named_readouts, observer
            base_plus_prior = call(lambda: (base_cost + float(bridge_input["alpha_prior"]) * prior_mean).float())
            trajectories = []
            for index, bridge in enumerate(bridges):
                if observer:
                    observer.execution_scope = "original_stage2_trajectory_forward"
                    observer.call_source = "source/stage2_pre_condense_models.py::PreCondenseResidualBridge.forward_causal_trajectory_residual"
                dt = call(bridge.forward_causal_trajectory_residual, attns[index], bilins[index], cand_q, sem24,
                          base_plus_prior, rev_idx, y_comb, y_f, rev_gate)
                trajectories.append(dt)
                named_readouts[f"fold{index}.delta_trajectory"] = dt
            if observer:
                observer.execution_scope = "saved_five_fold_prediction_mean"
                observer.call_source = "source/stage2_run_h256_flywheel_ablation.py::evaluate_context_with_model"
            trajectory_mean = call(lambda: torch.stack(trajectories, dim=0).double().mean(dim=0))
            combined_residual = call(lambda: float(bridge_input["alpha_prior"]) * prior_mean
                                    + float(bridge_input["alpha_trajectory"]) * trajectory_mean)
            named_readouts.update({"ensemble_delta_prior": prior_mean, "ensemble_delta_trajectory": trajectory_mean,
                                   "ensemble_weighted_residual": combined_residual, "base_cost_plus_prior": base_plus_prior})
            return named_readouts, observer
    finally:
        hidden_handle.remove()
        for bridge, original_encode in output_handles:
            bridge.encode_context_tokens = original_encode
        if observer:
            observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original H256 custody root")
    parser.add_argument("--stage2-custody", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="actual native FIT descriptor manifest")
    parser.add_argument("--bridge-inputs", type=Path, required=True, help="actual provenance-bound gate/SP inputs and optional real4reveal trajectory; no invented defaults")
    parser.add_argument("--native-features", type=Path, required=True, help="exact raw412/projection16 candidate artifact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scaler-mode", choices=["canonical_full_train", "foldwise"], default="canonical_full_train")
    parser.add_argument("--limit", type=int, default=6, choices=range(1, 17))
    args = parser.parse_args()
    stage1.require(not args.output_dir.exists(), "output exists; refusing overwrite")
    root, custody = args.root.resolve(), args.stage2_custody.resolve()
    for rel, expected in stage1.EXPECTED.items():
        stage1.require(stage1.sha(root / rel) == expected, "original pinned custody differs: " + rel)
    stage1.require(stage1.sha(custody / "assets/stage2-ensemble.pt") == ENSEMBLE_SHA, "Stage 2 checkpoint differs")
    stage1.require(stage1.sha(custody / "assets/unified.pt") == UNIFIED_SHA, "unified checkpoint differs")
    stage1.require(stage1.sha(custody / "source/pre_condense_models.py") == stage1.EXPECTED["source/pre_condense_models.py"], "Stage 2 architecture differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261103)
    ensemble = load_saved(custody / "assets/stage2-ensemble.pt")
    unified = load_saved(custody / "assets/unified.pt")
    stage1.require(ensemble["arm"] == stage1.ARM and len(ensemble["fold_state_dicts"]) == 5
                   and ensemble["dope_sha256"] == stage1.DOPE_COMPOSITE, "wrong saved ensemble")
    state = torch.load(root / "assets/round-882.pt", map_location="cpu", weights_only=True)["state_dict"]
    other = unified["base_solution_ranker_state_dict"]
    stage1.require(set(state) == set(other) and all(torch.equal(v, other[k]) for k, v in state.items()), "unified base differs from verified R882")
    model, dope_fn, freeze, before = stage1.load_dope(root, state)
    attach_stage1b(model, root, state)
    from h256_stage2_inputs import original_base
    base_adapter = original_base(custody / "source", state)
    attach_sync_free_attention(base_adapter, root)
    original = stage1.load_original(custody / "source/pre_condense_models.py", "h256_saved_stage2_original")
    bridges = []
    for saved in ensemble["fold_state_dicts"]:
        stage1.require(len(saved) == 84 and sum(v.numel() for v in saved.values()) == 183668, "unexpected full fold inventory")
        bridge = original.PreCondenseResidualBridge(arm=stage1.ARM)
        bridge.load_state_dict(saved, strict=True)
        bridge.eval()
        for parameter in bridge.parameters():
            parameter.requires_grad_(False)
        bridges.append(bridge)
    scaler_states = ensemble["fold_scalers"] if args.scaler_mode == "foldwise" else [ensemble["full_train_scaler"]] * 5
    scalers = [original.FoldScaler(**{k: np.asarray(v, dtype=np.float32) for k, v in saved.items()}) for saved in scaler_states]
    parameters, parameter_ids, parameter_tensors = inventory(model, base_adapter, bridges)
    before_hashes = {p["id"]: p["sha256"] for p in parameters}
    inputs = json.loads(args.inputs.read_text())
    bridge_manifest = json.loads(args.bridge_inputs.read_text())
    stage1.require(bridge_manifest["schema"] == "jain.h256-stage2-inputs/v1", "wrong Stage 2 input schema")
    scope = bridge_manifest["scope"]
    stage1.require(scope in {"exact_recorded_training_contract", "explicit_adapted_diagnostic"}, "input scope must be explicit")
    bindings = {c["id"]: c for c in bridge_manifest["cases"]}
    native_features = json.loads(args.native_features.read_text())
    native_cases = {c["id"]: c for c in native_features["cases"]}
    cases, executed, checks = [], set(), []
    for case in inputs["cases"][:args.limit]:
        binding = bindings[case["id"]]
        stage1.require(binding["dataset_sha256"] == case["dataset_sha256"], "Stage 2 input dataset differs")
        stage1.require(abs(float(binding["alpha_prior"]) - float(ensemble["ensemble_alpha_prior"])) <= 1e-15
                       and abs(float(binding["alpha_trajectory"]) - float(ensemble["ensemble_alpha_traj"])) <= 1e-15,
                       "ensemble alpha differs")
        baseline, _ = case_forward(case, binding, native_cases[case["id"]], model, dope_fn, base_adapter, bridges, scalers, parameter_ids, False)
        observed, observer = case_forward(case, binding, native_cases[case["id"]], model, dope_fn, base_adapter, bridges, scalers, parameter_ids, True)
        by_tensor_id = {t["id"]: t for t in observer.tensors}
        for module, prefix in ((model, ""), (base_adapter, "adapter_10tok.")):
            for name, buffer in module.named_buffers():
                tid = observer.by_object.get(id(buffer))
                if tid in by_tensor_id:
                    by_tensor_id[tid]["name"] = "saved_buffer." + prefix + name
                    by_tensor_id[tid]["checkpoint_origin"] = "assets/round-882.pt::state_dict/" + prefix + name
                    by_tensor_id[tid]["sha256"] = stage1.f32_sha(buffer)
        check = {"id": case["id"], "observer_on_off_exact": stage1.parity(baseline, observed),
                 **stage1.affine_controls(observer.affine), **observer.reference_checks}
        executed |= observer.executed
        output_ids = {name: observer.tensor(value, name=name) for name, value in observed.items()}
        selection = None
        if "final_k0_cost" in observed:
            scores = stage1.tensor_array(observed["final_k0_cost"]).reshape(-1)
            selected_slate_index = 4 + int(np.argmin(scores[4:]))
            selection = {"selected_slate_index": selected_slate_index,
                "selected_proposal_index": selected_slate_index - 4,
                "selected_program": binding["programs"][selected_slate_index],
                "rule": "minimum actual G pre-reveal final cost among proposal rows>=4; stable first-index tie",
                "outcome_based_selection": False, "reveal_count": 0}
        by_op = {o["id"]: o for o in observer.operations}
        dependencies = [{"from": by_op[parent]["name"], "to": op["name"], "from_operation_id": parent,
                         "to_operation_id": op["id"], "kind": op["kind"], "caption": "observed tensor dependency",
                         "source_ref": op["call_source"]} for op in observer.operations for parent in op["parent_op_ids"]]
        cases.append({"id": case["id"], "dataset_sha256": case["dataset_sha256"], "rows": len(case["fit_row_ids"]) + len(case["holdout_row_ids"]),
            "features": case["features"], "fit_row_ids": case["fit_row_ids"], "holdout_row_ids": case["holdout_row_ids"],
            "table_preview": case["table_preview"], "table_preview_row_ids": case["fit_row_ids"][:len(case["table_preview"])],
            "input_header": case["input_header"], "inputs": case["provenance"], "bridge_inputs": binding,
            "programs": binding["programs"], "scope": scope, "trajectory_executed": binding.get("trajectory") is not None,
            "selection": selection,
            "tensors": observer.tensors, "operations": observer.operations,
            "trace": {"affine": observer.affine, "tensors": []}, "functional_dependencies": dependencies,
            "outputs": {k: stage1.values(v) for k, v in observed.items()}, "outputs_tensor_ids": output_ids,
            "source": None, "tokens_per_second": None, "validation": check})
        cases[-1]["readout_provenance"] = {
            "fresh_descriptor_projection_lm48": {"source_ref": "source/cpu_exact_fused.py::_run_dope_mlc_v3_and_lm48",
                "meaning": "48D projection of standardized FIT descriptor256 using saved step2_svd_basis_256x48; not direct measured landmark outputs",
                "formula": "clamp((v256-step2_lm_mean)/step2_lm_std,-6,6) @ step2_svd_basis"},
            "fresh_lm48_z": {"source_ref": "source/cpu_exact_fused.py::_project_stage1b_tokens_batched_gpu",
                "formula": "clamp((fresh_descriptor_projection_lm48-mu_lm48)/sd_lm48,-6,6)"},
            "base_tokens_10x32": {"source_ref": "source/cpu_exact_fused.py::_project_stage1b_tokens_batched_gpu",
                "meaning": "fresh legacy R882 context path parallel to new G92-token bypass"},
            "native_conditioning_reference": {"native_call_id": observer.conditioning_call_id,
                "native_journal_sha256": case["provenance"]["native_journal_sha256"],
                "meaning": "same-input native conditioning used only for numerical reference; current displayed readouts are fresh CPU forwards"}}
        checks.append(check)
        print(json.dumps({"case": case["id"], "status": "PASS", "affine": len(observer.affine), "operations": len(observer.operations)}), flush=True)
    after = freeze.snapshot_dope_weights(model)
    stage1.require(after.composite_sha256 == before.composite_sha256, "DOPE mutated")
    stage1.require(all(stage1.f32_sha(parameter_tensors[k]) == v for k, v in before_hashes.items()), "saved neural parameter mutated")
    for p in parameters:
        p["executed"] = p["id"] in executed
    sources = source_files(root, custody)
    data = {"schema": SCHEMA, "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "inference_scope": scope, "checkpoint": {"path": "assets/stage2-ensemble.pt", "sha256": ENSEMBLE_SHA,
            "bytes": (custody / "assets/stage2-ensemble.pt").stat().st_size, "fold_count": 5,
            "tensor_count_per_fold": 84, "parameter_count_per_fold": 183668,
            "alpha_prior": ensemble["ensemble_alpha_prior"], "alpha_trajectory": ensemble["ensemble_alpha_traj"],
            "inference": "mean of five fold predictions; parameter-soup state_dict is unused", "scaler_mode": args.scaler_mode},
        "dope": {"composite_expected": stage1.DOPE_COMPOSITE, "before": before.composite_sha256, "after": after.composite_sha256,
            "state_checkpoint": {"path": "assets/round-882.pt", "sha256": stage1.EXPECTED["assets/round-882.pt"], "bytes": (root / "assets/round-882.pt").stat().st_size}},
        "critic_state_checkpoint": {"path": "assets/critic.pt", "sha256": stage1.sha(custody / "assets/critic.pt"),
            "bytes": (custody / "assets/critic.pt").stat().st_size, "fold_seed": 256140,
            "feature_builder": "source/stage2_stage1_hurdle_sos_critic.py::build_candidate_features"},
        "golden_fixture": {"path": "assets/golden.pt", "sha256": stage1.sha(custody / "assets/golden.pt"),
            "proof_receipt_path": "assets/golden-proof.receipt.json", "proof_receipt_sha256": stage1.sha(root / "stage2-golden-proof.receipt.json"),
            "scope": "authorised unseen fixture inference-only smoke; no learning or selection for displayed synthetic cases"},
        "scaler_state": {"checkpoint_path": "assets/stage2-ensemble.pt", "key": "full_train_scaler" if args.scaler_mode == "canonical_full_train" else "fold_scalers",
            "mode": args.scaler_mode,
            "buffers": [{"fold": i, "name": k, "shape": list(np.asarray(v).shape),
                "sha256": hashlib.sha256(np.ascontiguousarray(v, dtype="<f4").tobytes()).hexdigest()}
                for i, saved in enumerate(scaler_states) for k, v in saved.items()]},
        "sources_manifest": [{"path": rel, "sha256": stage1.sha(path)} for rel, path in sources.items()],
        "parameters": parameters, "cases": cases, "coverage": {"stage1_only": False, "stage2": True,
            "dropped_records": 0, "executed_parameter_ids": sorted(executed), "unused_parameter_ids": sorted(set(parameter_tensors) - executed),
            "source_or_gp_causality": False, "separate_hurdle_sp_predictors_in_checkpoint": True,
            "legacy_base": "R882 still uses legacy context condensation; the 92-token G path is additive and separately observed"},
        "runtime": {"python": sys.version, "torch": torch.__version__, "device": "cpu", "threads": 1,
            "training": False, "optimizer": None, "argv": sys.argv, "capture_source_sha256": stage1.sha(__file__),
            "stage1_observer_source_sha256": stage1.sha(stage1.__file__), "inputs_manifest_sha256": stage1.sha(args.inputs),
            "bridge_inputs_manifest_sha256": stage1.sha(args.bridge_inputs)}, "validation": {"cases": checks, "parameter_before_equals_after": True}}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    copies = {**sources, "assets/stage2-ensemble.pt": custody / "assets/stage2-ensemble.pt",
              "assets/round-882.pt": root / "assets/round-882.pt", "assets/unified.pt": custody / "assets/unified.pt",
              "assets/critic.pt": custody / "assets/critic.pt", "assets/golden.pt": custody / "assets/golden.pt",
              "assets/golden-proof.receipt.json": root / "stage2-golden-proof.receipt.json"}
    for rel, source in copies.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    trace = out / "trace.json"
    with trace.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    receipt = {"schema": "jain.h256-stage2-capture-receipt/v1", "status": "PASS", "trace_sha256": stage1.sha(trace),
               "scope": scope, "capture_source_sha256": stage1.sha(__file__), "checks": checks}
    with (out / "capture.receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "trace": str(trace.resolve()), "sha256": receipt["trace_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
