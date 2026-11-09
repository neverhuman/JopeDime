#!/usr/bin/env python3
"""Apply saved H256 hurdle/SP critics to real outcome-free native slate features.

No fitting occurs. The copied original feature builder and original common-pool
semantic signature run on supplied raw program412, R882 costs, and FIT context.
K_rev=0 emits a pre-reveal bridge manifest and invents no trajectory observations.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import h256_inference as stage1
import h256_stage2_inference as stage2

CRITIC_SHA = "ba5678e1097550d54ec5339cf42b341339af7d65e566cd3a2bcf28f16c596755"
FEATURE_SOURCE_SHA = "fc0dd0e2fa543f0540d677609989c1785ef33b07f6cf7e54f4f8bedbd0e5e797"
COMMON_SOURCE_SHA = "abe974ed10a0cd0d4b08440015a72a7b0adcebb22118c234732b812e9f0b0f92"
BASE_SOURCE_SHA = "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a"


def original_base(source, state):
    model_path = source / "flywheel_model.py"
    phase_path = source / "phase12_multimodal_model.py"
    stage1.require(stage1.sha(model_path) == BASE_SOURCE_SHA
                   and stage1.sha(phase_path) == stage2.QUERY_SOURCE_SHA, "original R882 base source differs")
    namespace = {"np": np, "torch": torch, "nn": nn, "F": F}
    stage1.execute_ast(model_path, ["extract_head_and_stage_signatures", "SwiGLUResBlock", "FlywheelGameNet"], namespace)
    tree = ast.parse(phase_path.read_text())
    vocabulary = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                      and isinstance(n.target, ast.Name) and n.target.id == "OPERATOR_VOCAB")
    namespace["OPERATOR_VOCAB"] = ast.literal_eval(vocabulary.value)
    stage1.execute_ast(phase_path, ["compute_semantic_struct_sig_24d_torch", "transform_program_features_torch",
                                  "FlywheelBlockFormerAdapterNet"], namespace)
    anchor = namespace["FlywheelGameNet"]()
    adapter = namespace["FlywheelBlockFormerAdapterNet"](anchor,
        state["adapter_10tok.prog_mean"].numpy(), state["adapter_10tok.prog_std"].numpy())
    prefix = "adapter_10tok."
    adapter.load_state_dict({k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def derive_k0(case, fit_case, adapter, model, dope_fn):
    """Actual original R882 base forward at zero reveals; recorded history costs are never used."""
    journal_path = Path(fit_case["provenance"]["native_journal_path"])
    stage1.require(stage1.sha(journal_path) == fit_case["provenance"]["native_journal_sha256"], "native journal changed")
    journal = json.loads(journal_path.read_text())
    condition = next(c for c in journal["calls"] if "v256" in c["outputs"])
    slate = next(c for c in journal["calls"] if c["call_id"] == case["native_call_id"])
    stage1.require(slate["identity"]["revealed_programs"] + slate["identity"]["proposal_programs"] == case["programs"],
                   "native program row binding differs")
    n = len(case["programs"])
    prog = torch.tensor(case["raw_program_features412"], dtype=torch.float32).reshape(1, n, 412)
    proj = torch.tensor(case["projection16_f64"], dtype=torch.float64).reshape(1, n, 16)
    # These zeros are the actual ServingAdmissionSlate::canonical_inputs policy,
    # not missing learned data substituted with a fallback.
    with torch.no_grad():
        dope629, lm48 = dope_fn(model, torch.tensor(fit_case["v256"]).reshape(1, 256),
                               torch.tensor(fit_case["col64x17"]).reshape(1, 64, 17), use_landmarks=True)
        tokens, mask, lm48_z = model._project_stage1b_tokens_batched_gpu(lm48, dope629)
        out = adapter.predict_costs(ctx_feat=torch.zeros((1, 584), dtype=torch.float32),
            prog_412=prog, proj_16=proj, base_prior_cost=torch.zeros((1, n), dtype=torch.float64),
            scale=torch.tensor([.4208], dtype=torch.float64), sos_prior=torch.zeros((1, n, 2), dtype=torch.float32),
            mm_tokens_10x32=tokens, mm_mask_10=mask, revealed_indices=torch.empty((1, 0), dtype=torch.long),
            revealed_raw_f=torch.empty((1, 0), dtype=torch.float64), revealed_gate=torch.empty((1, 0), dtype=torch.float32))
    reference = next(a for a in slate["trace"]["affine"] if a["name"] == "cand_query_proj.0")
    actual_input = torch.cat([out["sem24"], out["sem24"] - out["sem24"].mean(dim=1, keepdim=True),
        adapter.prog_412_bottleneck(torch.clamp((torch.sign(prog) * torch.log1p(prog.abs()) - adapter.prog_mean) / adapter.prog_std, -8., 8.)),
        torch.zeros((1, n, 2))], dim=-1)
    error = float(np.max(np.abs(actual_input.detach().numpy().reshape(-1) - np.asarray(reference["input"], dtype=np.float32))))
    stage1.require(error < 2e-5, "fresh original base program/query input differs from actual native serving input")
    return {**case, "r882_total_cost": out["total_cost"].detach().numpy().reshape(-1).tolist(),
        "r882_base_prior_cost": [0.] * n, "r882_effective_prior": out["eff_prior"].detach().numpy().reshape(-1).tolist(),
        "lm48_z": lm48_z.numpy().reshape(-1).tolist(), "base_query_input_max_abs_error": error,
        "base_capture": {"kind": "fresh_original_cpu_forward", "reveals": 0,
            "legacy_context_tokens": "fresh original DOPE wrapper and original Stage1B projection on exact recorded FIT descriptors",
            "source": "FlywheelBlockFormerAdapterNet.predict_costs -> FlywheelGameNet.predict_costs K_rev0",
            "context_policy": "native serving 584 zeros; base prior zeros; SOS zeros", "history_costs_used": False}}


def raw_hash(a, dtype):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def original_builders(source):
    feature_path = source / "stage1_hurdle_sos_critic.py"
    common_path = source / "common_h252.py"
    stage1.require(stage1.sha(feature_path) == FEATURE_SOURCE_SHA and stage1.sha(common_path) == COMMON_SOURCE_SHA,
                   "original critic feature source differs")
    namespace = {"np": np}
    stage1.execute_ast(feature_path, ["build_candidate_features"], namespace)
    stage1.execute_ast(common_path, ["compute_sem24_numpy"], namespace)
    return namespace["build_candidate_features"], namespace["compute_sem24_numpy"]


def predict(critic, raw_program412, rcost, anchor_prior, lm48_z, source):
    build, semantic = original_builders(source)
    program = np.asarray(raw_program412, dtype=np.float32).reshape(-1, 412)
    n = len(program)
    stage1.require(n >= 9, "original top8 critic feature contract needs at least nine actual candidates")
    rcost = np.asarray(rcost, dtype=np.float32).reshape(n)
    anchor_prior = np.asarray(anchor_prior, dtype=np.float32).reshape(n)
    lm48 = np.asarray(lm48_z, dtype=np.float32).reshape(48)
    stage1.require(all(bool(np.isfinite(v).all()) for v in (program, rcost, anchor_prior, lm48)), "nonfinite native critic input")
    sem = semantic(program)
    cid = "native-input"
    feature64 = build([{ "context_sha256": cid }], {cid: {"program": program, "sem24": sem}},
        {cid: {"total_cost": rcost, "anchor_total": rcost, "anchor_prior": anchor_prior}},
        {cid: {"lm48_z": lm48}})[cid]
    x65 = np.concatenate([feature64.astype(np.float64), np.ones((n, 1), dtype=np.float64)], axis=1)
    pc = x65 @ np.asarray(critic["w_c_full"], dtype=np.float64)
    pg = x65 @ np.asarray(critic["w_g_full"], dtype=np.float64)
    cont = ((pc - float(np.mean(pc))) / (float(np.std(pc)) + 1e-6)).astype(np.float32)
    gate = ((pg - float(np.mean(pg))) / (float(np.std(pg)) + 1e-6)).astype(np.float32)
    sp = -(sem.astype(np.float64) @ np.asarray(critic["sp_dir_full"], dtype=np.float64))
    sp = ((sp - float(np.mean(sp))) / (float(np.std(sp)) + 1e-6)).astype(np.float32)
    unsafe = gate.astype(np.float64) > (float(gate[int(np.argmin(rcost))]) + .10)
    h_cost = (.65 * cont.astype(np.float64) + .35 * gate.astype(np.float64)
              + .15 * sp.astype(np.float64) + np.where(unsafe, 10., 0.))
    return {"common_semantic_signature24": sem, "feature64": feature64, "feature65": x65,
            "continuous_cost_z": cont, "gate_risk": gate, "sp_prior": sp, "h_cost": h_cost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="pinned Stage2 dependency source directory")
    parser.add_argument("--native-features", type=Path, required=True, help="real raw412/basecost/FIT context artifact")
    parser.add_argument("--fit-inputs", type=Path, help="actual native FIT manifest; derive genuine K_rev0 R882 costs")
    parser.add_argument("--round882", type=Path, help="frozen R882 checkpoint for fresh zero-reveal base forward")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    stage1.require(stage1.sha(args.critic) == CRITIC_SHA, "saved critic bytes differ")
    stage1.require(not args.output_dir.exists(), "output already exists")
    critic = stage2.load_saved(args.critic)
    stage1.require(critic["dope_sha256"] == stage1.DOPE_COMPOSITE and critic["fold_seed"] == 256140,
                   "critic training identity differs")
    native = json.loads(args.native_features.read_text())
    stage1.require(native["schema"] == "jain.h256-native-slate-features/v1", "wrong native feature artifact")
    stage1.require(1 <= len(native["cases"]) <= 16, "native case count outside bound")
    if args.fit_inputs:
        stage1.require(args.round882 and stage1.sha(args.round882) == stage1.EXPECTED["assets/round-882.pt"],
                       "fresh K0 base needs exact frozen checkpoint")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        state = torch.load(args.round882, map_location="cpu", weights_only=True)["state_dict"]
        adapter = original_base(args.source, state)
        dope_root = args.round882.parent.parent
        stage2.attach_sync_free_attention(adapter, dope_root)
        for relative, expected in stage1.EXPECTED.items():
            stage1.require(stage1.sha(dope_root / relative) == expected, "original DOPE custody source differs")
        model, dope_fn, freeze, before = stage1.load_dope(dope_root, state)
        stage2.attach_stage1b(model, dope_root, state)
        fits = {c["id"]: c for c in json.loads(args.fit_inputs.read_text())["cases"]}
        native["cases"] = [derive_k0(c, fits[c["id"]], adapter, model, dope_fn) for c in native["cases"]]
        stage1.require(freeze.snapshot_dope_weights(model).composite_sha256 == before.composite_sha256,
                       "DOPE mutated in K0 source preparation")
    results, bindings = [], []
    for case in native["cases"]:
        n = len(case["programs"])
        stage1.require(len(case["raw_program_features412"]) == n * 412, "raw program feature cells missing")
        result = predict(critic, case["raw_program_features412"], case["r882_total_cost"],
                         case["r882_base_prior_cost"], case["lm48_z"], args.source)
        result["r882_k0_total_cost"] = np.asarray(case["r882_total_cost"], dtype=np.float64)
        result["initial_cost_before_g"] = result["r882_k0_total_cost"] + .005 * result["h_cost"]
        results.append({"id": case["id"], "dataset_sha256": case["dataset_sha256"],
            "arrays": {k: np.asarray(v).reshape(-1).tolist() for k, v in result.items()},
            "shapes": {k: list(v.shape) for k, v in result.items()},
            "hashes": {k: raw_hash(v, "<f8" if np.asarray(v).dtype == np.float64 else "<f4") for k, v in result.items()}})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    predictions_path = out / "critic-predictions.json"
    prediction_artifact = {"schema": "jain.h256-critic-predictions/v1", "critic_sha256": CRITIC_SHA,
        "input_builder_source_sha256": stage1.sha(__file__), "base_source_sha256": BASE_SOURCE_SHA,
        "frozen_round882_sha256": stage1.EXPECTED["assets/round-882.pt"],
        "native_features_path": str(args.native_features.resolve()), "native_features_sha256": stage1.sha(args.native_features),
        "feature_source_sha256": FEATURE_SOURCE_SHA, "common_semantic_source_sha256": COMMON_SOURCE_SHA, "cases": results}
    with predictions_path.open("x") as stream:
        json.dump(prediction_artifact, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    predictions_sha = stage1.sha(predictions_path)
    for index, (case, result) in enumerate(zip(native["cases"], results)):
        def binding(name):
            cells = result["arrays"][name]
            dtype = "float64" if name in {"initial_cost_before_g", "r882_k0_total_cost"} else "float32"
            return {"values": cells, "dtype": dtype,
                "values_sha256": raw_hash(cells, "<f8" if dtype == "float64" else "<f4"),
                "artifact_path": str(predictions_path.resolve()), "artifact_sha256": predictions_sha,
                "source_ref": "source/stage2_stage1_hurdle_sos_critic.py::build_candidate_features + saved full critic prediction formulas",
                "json_path": ["cases", index, "arrays", name]}
        bindings.append({"id": case["id"], "dataset_sha256": case["dataset_sha256"],
            "native_call_id": case["native_call_id"], "programs": case["programs"], "gate_risk": binding("gate_risk"),
            "sp_prior": binding("sp_prior"), "alpha_prior": .032999999999999995, "alpha_trajectory": .01,
            "r882_k0_total_cost": binding("r882_k0_total_cost"),
            "trajectory": None, "reveal_count": 0,
            "initial_cost_before_g": {**binding("initial_cost_before_g"),
                "recipe": "fresh R882 K_rev0 total_cost + .005 * saved-hurdle h_cost"},
            "base_capture": case.get("base_capture"),
            "critic_prediction_index": index, "native_feature_provenance": case["provenance"]})
    manifest = {"schema": "jain.h256-stage2-inputs/v1", "scope": "exact_recorded_training_contract",
        "inference_scope": "Saved full critics on outcome-free native inputs; genuine G pre-reveal prior with K_rev0; trajectory not executed",
        "critic_sha256": CRITIC_SHA, "predictions_path": str(predictions_path.resolve()),
        "input_builder_source_sha256": stage1.sha(__file__),
        "predictions_sha256": predictions_sha, "native_features_sha256": stage1.sha(args.native_features), "cases": bindings}
    manifest_path = out / "bridge-inputs.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.fit_inputs:
        with (out / "native-k0-features.json").open("x") as stream:
            json.dump(native, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path.resolve()), "sha256": stage1.sha(manifest_path),
                      "cases": len(bindings), "reveal_count": 0}), flush=True)


if __name__ == "__main__":
    main()
