#!/usr/bin/env python3
"""Replay the pinned, saved H256 Stage 1 encoder on observed synthetic FIT inputs.

This is a CPU-only research snapshot observer, not a product GP inference adapter.
It executes the copied original classes/functions, captures tensor-object lineage,
and never substitutes initialized, unsaved parameters for missing trained heads.
"""
from __future__ import annotations

import argparse
import ast
import _codecs
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

sys.dont_write_bytecode = True

EXPECTED = {
    "assets/stage1_encoder.pt": "7e5257741409575b237f0312ee8a7c5f6e1fc48d831056b6166ceea455528a44",
    "assets/round-882.pt": "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64",
    "assets/stage1_receipt.json": "f6e8c67a0022f034d751ed4114cffd0872eecc5476ad5d2e175a5bac43e23657",
    "source/pre_condense_models.py": "c091bdc6c33758a7f169534707cba8cea615338be8b65092d4199d92212ea795",
    "source/pretrain_full_s3_7split_bridge.py": "7b74b8cb456f638a32519e621f9799e53409dd2b3195e16b873d7d67bb6500b9",
    "source/run_h256_flywheel_ablation.py": "4995a974fd397d1cbf852842ba4294a003a595081279f1c929169ffb526ea0ab",
    "source/extract_pre_condense_cache.py": "a05b8e8ea6dab47588784db0043bdf131cc0617d0add799eb4e5af791ee55cd8",
    "source/multilevel_model.py": "39ce38d17a87a2f9cf1a02dbad6efb6606100d46c66ab00fcd56c0416e9643b6",
    "source/multiclass_head.py": "d4c9e750330487b4c9c2cd6c702ed58201eb9322735b911e45ec7cff883f4cda",
    "source/dope_freeze_guard.py": "38cbed30f8361a5e95e05d768e16051437b22a629ec56c9ac5938654685b4ea2",
    "source/cpu_exact_fused.py": "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab",
}
DOPE_COMPOSITE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
ARM = "G_FULL_S3_REG_7SPLIT_11121_PRETRAINED"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_array(tensor):
    with _disable_current_modes():
        return tensor.detach().cpu().contiguous().numpy().copy()


def f32_sha(tensor):
    return hashlib.sha256(np.ascontiguousarray(tensor_array(tensor), dtype="<f4").tobytes()).hexdigest()


def values(tensor):
    a = tensor_array(tensor)
    require(bool(np.isfinite(a).all()), "non-finite observed tensor")
    return a.reshape(-1).tolist()


def load_original(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute_ast(path, names, namespace):
    """Compile unchanged selected class definitions, avoiding unrelated import side effects."""
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    require({n.name for n in selected} == set(names), "original source definition missing")
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + selected,
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


def load_dope(root, state):
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "math": math,
                 "MAX_COLS": 64, "_LEVEL_INDEX": {"low": 0, "small": 0, "medium": 0, "high": 1, "ultra": 2}}
    execute_ast(root / "source/multiclass_head.py", ["InterLandmarkAttentionBlock"], namespace)
    execute_ast(root / "source/multilevel_model.py",
                ["LandmarkHyperRouter", "ConditionalHyperTargetNet", "CovariateRefinementNet", "MultiLevelLandmarkModel"], namespace)
    container = nn.Module()
    container.dope_model = namespace["MultiLevelLandmarkModel"]()
    dope_state = {k.removeprefix("dope_model."): v for k, v in state.items() if k.startswith("dope_model.")}
    container.dope_model.load_state_dict(dope_state, strict=True)
    freeze = load_original(root / "source/dope_freeze_guard.py", "h256_original_freeze_guard")
    for name in freeze.DOPE_NORMALIZER_BUFFERS:
        require(name in state, "missing frozen normalizer " + name)
        container.register_buffer(name, state[name].clone())
    tree = ast.parse((root / "source/cpu_exact_fused.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedJopeNetwork")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_run_dope_mlc_v3_and_lm48")
    gates = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                 and n.target.id == "STEP3_GATE_KEYS_14")
    container.step3_gate_keys_14 = ast.literal_eval(gates.value)
    fn_namespace = {"torch": torch}
    original_fn = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(original_fn), str(root / "source/cpu_exact_fused.py"), "exec"), fn_namespace)
    forward = fn_namespace[fn.name]
    freeze.lock_dope_weights_and_eval_mode(container)
    before = freeze.snapshot_dope_weights(container)
    require(before.composite_sha256 == DOPE_COMPOSITE, "copied DOPE composite does not match Stage 1 freeze")
    return container, forward, freeze, before


def load_stage1(path):
    # Explicit aliases for this pinned older NumPy checkpoint; no unrestricted pickle load.
    allowed = [(np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
               np.ndarray, np.dtype, np.dtypes.Float32DType, _codecs.encode]
    with torch.serialization.safe_globals(allowed):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["dope_sha256"] == DOPE_COMPOSITE, "pretraining used a different frozen problem encoder")
    saved = payload["encoder_state_dict"]
    require(len(saved) == 39 and sum(v.numel() for v in saved.values()) == 15005,
            "unexpected saved Stage 1 inventory")
    return payload


def source_summary_function(root):
    """Use the unchanged query-independent pooling statements from the original pretrainer."""
    path = root / "source/pretrain_full_s3_7split_bridge.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "S3PretrainBasisModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "encode_summary")
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "mask_exp")
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "q")
    body = fn.body[start:end]
    wrapper = ast.FunctionDef(name="observed_summary", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg="ctx_tokens"), ast.arg(arg="ctx_mask")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body + [ast.Return(value=ast.Name(id="ctx_summary_64", ctx=ast.Load()))], decorator_list=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["observed_summary"]


class Observer(TorchDispatchMode):
    """Observe actual tensor identities. Leaf modules own their outputs; ATen captures joins."""
    def __init__(self, parameter_ids):
        super().__init__()
        self.parameter_ids = parameter_ids
        self.tensors, self.operations, self.affine = [], [], []
        self.by_object, self.held, self.hooks = {}, [], []
        self.suppression = 0
        self.role = "problem_encoder"
        self.execution_scope = "original_dope_forward"
        self.call_source = "source/cpu_exact_fused.py::_run_dope_mlc_v3_and_lm48"
        self.executed = set()

    def tensor(self, tensor, producer=None, name=None):
        key = id(tensor)
        if key in self.parameter_ids:
            return self.parameter_ids[key]
        if key in self.by_object:
            return self.by_object[key]
        tid = f"t{len(self.tensors):05d}"
        dtype = str(tensor.dtype).removeprefix("torch.")
        require(dtype in {"float32", "float64", "int64", "int32", "bool"}, "unsupported tensor dtype " + dtype)
        record = {"id": tid, "name": name or tid, "role": self.role, "shape": list(tensor.shape),
                  "dtype": dtype, "values": values(tensor), "producer_op_id": producer}
        self.by_object[key] = tid
        self.held.append(tensor)
        self.tensors.append(record)
        require(len(self.tensors) <= 5000, "tensor capture budget exceeded")
        return tid

    def walk(self, value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from self.walk(child)
        elif isinstance(value, dict):
            for child in value.values():
                yield from self.walk(child)

    def record(self, name, kind, inputs, outputs, parameter_ids=(), observation="native-functional-dispatch"):
        opid = f"op{len(self.operations):05d}"
        ins = list(dict.fromkeys(self.tensor(t) for t in self.walk(inputs)))
        pids = list(dict.fromkeys(list(parameter_ids) + [v for v in ins if v in self.parameter_ids.values()]))
        ins = [v for v in ins if v not in pids]
        parents = list(dict.fromkeys(next(t["producer_op_id"] for t in self.tensors if t["id"] == tid)
                                    for tid in ins))
        parents = [p for p in parents if p is not None]
        outs = []
        for t in self.walk(outputs):
            # Functional no-op casts may return the identical Tensor. They add no new producer.
            if id(t) in self.by_object or id(t) in self.parameter_ids:
                continue
            outs.append(self.tensor(t, opid, name + ".output"))
        if not outs:
            return None
        self.operations.append({"id": opid, "name": name, "kind": kind, "role": self.role,
            "input_tensor_ids": ins, "output_tensor_ids": outs, "parameter_ids": pids,
            "parent_op_ids": parents, "observation": observation,
            "execution_scope": self.execution_scope, "call_source": self.call_source})
        self.executed.update(pids)
        require(len(self.operations) <= 5000, "operation capture budget exceeded")
        return opid

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if not self.suppression:
            self.record(str(func), str(func), (args, kwargs), result)
        return result

    def attach(self, model, prefix):
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding, nn.SiLU)):
                continue
            full_name = prefix + "." + name

            def before(mod, args, full_name=full_name):
                for parameter in mod.parameters(recurse=False):
                    require(id(parameter) in self.parameter_ids, "attempted execution of unsaved parameter " + full_name)
                self.suppression += 1

            def after(mod, args, output, full_name=full_name):
                self.suppression -= 1
                pids = [self.parameter_ids[id(p)] for p in mod.parameters(recurse=False)]
                opid = self.record(full_name, type(mod).__name__, args, output, pids, "native-module-hook")
                if isinstance(mod, nn.Linear):
                    require(opid is not None, "linear output must have a producer")
                    x, y, w = args[0], output, mod.weight
                    self.affine.append({"sequence": len(self.affine), "network": "dope" if prefix == "dope" else "h256",
                        "role": self.role, "name": full_name, "input_shape": list(x.shape), "weight_shape": list(w.shape),
                        "output_shape": list(y.shape), "weight_layout": "out,in", "input": values(x), "weights": values(w),
                        "bias": values(mod.bias) if mod.bias is not None else [0.] * mod.out_features,
                        "output": values(y), "parameter_ids": pids, "operation_id": opid,
                        "execution_scope": self.execution_scope, "call_source": self.call_source})
            self.hooks.append(module.register_forward_pre_hook(before))
            self.hooks.append(module.register_forward_hook(after))

    def close(self):
        for hook in self.hooks:
            hook.remove()
        require(self.suppression == 0, "unbalanced observer leaf scope")


def inventory(dope, bridge, saved):
    records, ids, tensors = [], {}, {}
    for prefix, named, role, origin in (
        ("dope", dope.dope_model.named_parameters(), "problem_encoder", "assets/round-882.pt::state_dict/dope_model."),
        ("h256", ((k, dict(bridge.named_parameters())[k]) for k in saved), "solution_encoder", "assets/stage1_encoder.pt::encoder_state_dict/"),
    ):
        for name, tensor in named:
            pid = prefix + "." + name
            ids[id(tensor)] = pid
            tensors[pid] = tensor
            records.append({"id": pid, "name": pid, "role": role, "shape": list(tensor.shape), "dtype": "float32",
                "values": values(tensor), "sha256": f32_sha(tensor), "checkpoint_origin": origin + name,
                "executed": False})
    return records, ids, tensors


def parity(a, b):
    require(a.keys() == b.keys(), "readout set changed with observation")
    errors = {}
    for key in a:
        left, right = tensor_array(a[key]), tensor_array(b[key])
        require(np.array_equal(left, right), "observation changed native tensor " + key)
        errors[key] = float(np.max(np.abs(left.astype(np.float64) - right)))
    return errors


def affine_controls(affine):
    errors, rejected = [], False
    for op in affine:
        x = np.asarray(op["input"], dtype=np.float64).reshape(-1, op["weight_shape"][1])
        w = np.asarray(op["weights"], dtype=np.float64).reshape(op["weight_shape"])
        b = np.asarray(op["bias"], dtype=np.float64)
        y = np.asarray(op["output"], dtype=np.float64).reshape(-1, op["weight_shape"][0])
        denom = np.abs(x) @ np.abs(w).T + np.abs(b) + 1e-9
        error = float(np.max(np.abs(x @ w.T + b - y) / denom))
        require(error <= 1e-5, "affine closure failed: " + op["name"])
        errors.append(error)
        bad = y.copy()
        bad.flat[0] += float(denom.flat[0] + 1.)
        rejected |= bool(np.max(np.abs(x @ w.T + b - bad) / denom) > 1e-5)
    require(rejected, "corrupted output negative control did not reject")
    return {"max_normalized_affine_error": max(errors), "corrupted_output_negative_control": "rejected"}


def capture_case(case, root, model, dope_forward, bridge, scaler, summary_fn, parameter_ids, observe):
    observer = Observer(parameter_ids) if observe else None
    hidden = []
    def hidden_hook(_mod, args):
        hidden.append(args[0])
    handle = model.dope_model.router.task_latent_head.register_forward_pre_hook(hidden_hook)
    if observer:
        observer.attach(model.dope_model, "dope")
        observer.attach(bridge, "h256")
    v = torch.tensor(case["v256"], dtype=torch.float32).reshape(1, 256)
    col = torch.tensor(case["col64x17"], dtype=torch.float32).reshape(1, 64, 17)
    journal_path = Path(case["provenance"]["native_journal_path"])
    require(sha(journal_path) == case["provenance"]["native_journal_sha256"], "native journal identity changed")
    journal = json.loads(journal_path.read_text())
    conditioning = next(c for c in journal["calls"] if "v256" in c["outputs"])
    require(np.array_equal(np.asarray(conditioning["outputs"]["v256"], dtype=np.float32).reshape(-1), np.asarray(case["v256"], dtype=np.float32)),
            "FIT descriptor is not the observed native input")
    if observer:
        observer.tensor(v, name="fit.v256")
        observer.tensor(col, name="fit.col64x17")
    try:
        with torch.no_grad():
            if observer:
                with observer:
                    dope629, _unused_lm48 = dope_forward(model, v, col, use_landmarks=True)
            else:
                dope629, _unused_lm48 = dope_forward(model, v, col, use_landmarks=True)
            h256 = hidden[-1]
            dope_error = float(np.max(np.abs(tensor_array(dope629).reshape(-1) - np.asarray(case["reference_dope629"], dtype=np.float32))))
            hidden_error = float(np.max(np.abs(tensor_array(h256).reshape(-1) - np.asarray(case["reference_hidden256"], dtype=np.float32))))
            require(dope_error <= 3e-4 and hidden_error <= 3e-4, "fresh Python DOPE differs from same-weights native reference")
            entry = {"dope629": tensor_array(dope629).reshape(629), "hidden256": tensor_array(h256).reshape(256),
                "col_lm_64x17": tensor_array(col).reshape(64, 17), "col_mask_64": np.arange(64) < case["features"],
                "tokens_10x32": np.asarray(conditioning["outputs"]["tokens"], dtype=np.float32).reshape(10, 32),
                "mask_10": np.asarray(conditioning["outputs"]["mask"], dtype=np.float32).reshape(10)}
            ctx = scaler.transform_context(entry, device=torch.device("cpu"))
            relevant = {k: ctx[k] for k in ("dope629_z", "hidden256_z", "col18_z", "col_mask_64", "block_log_norms")}
            if observer:
                observer.role = "solution_encoder"
                observer.execution_scope = "original_encoder_api"
                observer.call_source = "source/pre_condense_models.py::FoldScaler.transform_context"
                observer.record("h256.FoldScaler.transform_context", "saved_policy_b_standardize_clip15_and_col18_recipe",
                                (dope629, h256, col), relevant, observation="executed-source-function")
                observer.call_source = "source/pre_condense_models.py::PreCondenseResidualBridge.encode_context_tokens"
                with observer:
                    tokens, mask, scale = bridge.encode_context_tokens(ctx)
                observer.execution_scope = "diagnostic_saved_projection"
                observer.call_source = "source/pre_condense_models.py::SyncFreeCrossAttention.k_proj/v_proj (partial; no query)"
                with observer:
                    key = bridge.ctx_cross_attn.k_proj(tokens)
                    value = bridge.ctx_cross_attn.v_proj(tokens)
                observer.call_source = "source/pretrain_full_s3_7split_bridge.py::S3PretrainBasisModel.encode_summary (query-independent prefix only)"
                with observer:
                    summary = summary_fn(tokens, mask)
                    bilinear = bridge.ctx_bilinear_u(summary)
            else:
                tokens, mask, scale = bridge.encode_context_tokens(ctx)
                key = bridge.ctx_cross_attn.k_proj(tokens)
                value = bridge.ctx_cross_attn.v_proj(tokens)
                summary = summary_fn(tokens, mask)
                bilinear = bridge.ctx_bilinear_u(summary)
            out = {"dope629": dope629, "hidden256": h256, "context_tokens_92x32": tokens,
                   "context_mask_92": mask, "block_log_norms_11": ctx["block_log_norms"], "scale_side_8": scale,
                   "context_key_92x32": key, "context_value_92x32": value,
                   "context_summary_64": summary, "context_bilinear_24": bilinear}
            extra = {"native_reference_dope629_max_abs_error": dope_error,
                     "native_reference_hidden256_max_abs_error": hidden_error}
            return out, observer, extra
    finally:
        handle.remove()
        if observer:
            observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="custodied h256 asset directory")
    parser.add_argument("--inputs", type=Path, required=True, help="native-fit-inputs.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="new, exclusive capture directory")
    parser.add_argument("--limit", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    for relative, expected in EXPECTED.items():
        require(sha(root / relative) == expected, "pinned custody asset differs: " + relative)
    require(not args.output_dir.exists(), "capture output already exists; refusing overwrite")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261103)  # unused initialized branches are never invoked or recorded
    payload = load_stage1(root / "assets/stage1_encoder.pt")
    checkpoint = torch.load(root / "assets/round-882.pt", map_location="cpu", weights_only=True)
    model, dope_forward, freeze, before = load_dope(root, checkpoint["state_dict"])
    original = load_original(root / "source/pre_condense_models.py", "h256_original_pre_condense")
    bridge = original.PreCondenseResidualBridge(arm=ARM)
    loaded = bridge.load_pretrained_s3_context_encoder(payload["encoder_state_dict"])
    require(loaded == 39, "not all saved Stage 1 tensors loaded")
    for p in bridge.parameters():
        p.requires_grad_(False)
    bridge.eval()
    scaler = original.FoldScaler(**payload["scaler_policy_b"])
    summary_fn = source_summary_function(root)
    parameters, parameter_ids, parameter_tensors = inventory(model, bridge, payload["encoder_state_dict"])
    before_parameters = {p["id"]: p["sha256"] for p in parameters}
    inputs = json.loads(args.inputs.read_text())
    require(inputs["schema"] == "jain.h256-fit-inputs/v1", "wrong FIT input manifest")
    cases, executed, checks = [], set(), []
    for case in inputs["cases"][:args.limit]:
        require(sha(case["dataset_file"]) == case["dataset_sha256"], "synthetic CSV identity changed")
        require(sha(case["provenance"]["native_trace_path"]) == case["provenance"]["native_trace_sha256"], "native trace identity changed")
        with Path(case["dataset_file"]).open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            csv_rows = np.asarray(list(reader), dtype=np.float32)
        preview_ids = case["fit_row_ids"][:len(case["table_preview"])]
        require(header == case["input_header"] and np.array_equal(
            csv_rows[preview_ids], np.asarray(case["table_preview"], dtype=np.float32)),
            "table preview differs from exact FIT CSV rows")
        baseline, _, _ = capture_case(case, root, model, dope_forward, bridge, scaler, summary_fn, parameter_ids, False)
        observed, observer, extra = capture_case(case, root, model, dope_forward, bridge, scaler, summary_fn, parameter_ids, True)
        check = {"id": case["id"], "observer_on_off_exact": parity(baseline, observed), **extra,
                 **affine_controls(observer.affine)}
        executed |= observer.executed
        output_ids = {k: observer.tensor(v, name=k) for k, v in observed.items()}
        tensor_lookup = {t["id"]: t for t in observer.tensors}
        deps = []
        op_by_id = {o["id"]: o for o in observer.operations}
        for op in observer.operations:
            for parent in op["parent_op_ids"]:
                deps.append({"from": op_by_id[parent]["name"], "to": op["name"], "from_operation_id": parent,
                    "to_operation_id": op["id"], "kind": op["kind"], "caption": "observed tensor dependency",
                    "source_ref": op["call_source"]})
        cases.append({"id": case["id"], "dataset_sha256": case["dataset_sha256"],
            "rows": len(case["fit_row_ids"]) + len(case["holdout_row_ids"]), "fit_rows": len(case["fit_row_ids"]),
            "input_rows": len(case["fit_row_ids"]) + len(case["holdout_row_ids"]), "features": case["features"],
            "fit_row_ids": case["fit_row_ids"], "holdout_row_ids": case["holdout_row_ids"],
            "input_header": case["input_header"], "table_preview": case["table_preview"],
            "table_preview_row_ids": preview_ids,
            "inputs": {**case["provenance"], "manifest_sha256": sha(args.inputs),
                "col18_recipe": "original FoldScaler.transform_context: concat col64x17 with fresh DOPE629[565:629]; PolicyB standardize/clip15/mask",
                "col18_recipe_source_ref": "source/pre_condense_models.py:99"},
            "tensors": observer.tensors, "operations": observer.operations,
            "trace": {"affine": observer.affine, "tensors": [
                {"network": "dope" if tensor_lookup[tid]["role"] == "problem_encoder" else "h256",
                 "name": name, "shape": tensor_lookup[tid]["shape"], "values": tensor_lookup[tid]["values"]}
                for name, tid in output_ids.items()]},
            "outputs": {k: values(v) for k, v in observed.items()}, "outputs_tensor_ids": output_ids,
            "functional_dependencies": deps, "source": None, "tokens_per_second": None,
            "inference_scope": "saved Stage 1 G context encoder; saved K/V and bilinear are separately labeled partial diagnostics",
            "validation": check})
        checks.append(check)
        print(json.dumps({"case": case["id"], "status": "PASS", "affine": len(observer.affine),
                          "operations": len(observer.operations), "tensors": len(observer.tensors), **extra}), flush=True)
    after = freeze.snapshot_dope_weights(model)
    require(after.composite_sha256 == before.composite_sha256, "problem encoder mutated")
    require(all(f32_sha(parameter_tensors[pid]) == expected for pid, expected in before_parameters.items()), "saved parameter mutated")
    for parameter in parameters:
        parameter["executed"] = parameter["id"] in executed
    saved = payload["encoder_state_dict"]
    data = {"schema": "jain.h256-stage1-capture/v1", "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inference_scope": "CPU Stage 1 saved context encoder snapshot; diagnostic saved projections separated; no Stage 2 or GP guidance",
        "checkpoint": {"path": "assets/stage1_encoder.pt", "sha256": EXPECTED["assets/stage1_encoder.pt"],
            "bytes": (root / "assets/stage1_encoder.pt").stat().st_size, "tensor_count": len(saved),
            "parameter_count": sum(t.numel() for t in saved.values()), "state_dict_sha256": {k: f32_sha(v) for k, v in saved.items()},
            "state_dict_shapes": {k: list(v.shape) for k, v in saved.items()}, "arm": ARM, "epochs": 16},
        "dope": {"path": "assets/round-882.pt", "sha256": EXPECTED["assets/round-882.pt"],
            "state_checkpoint": {"path": "assets/round-882.pt", "sha256": EXPECTED["assets/round-882.pt"],
                "bytes": (root / "assets/round-882.pt").stat().st_size}, "composite_expected": DOPE_COMPOSITE,
            "before": before.composite_sha256, "after": after.composite_sha256},
        "scaler": {"checkpoint_path": "assets/stage1_encoder.pt", "policy": "scaler_policy_b",
            "buffers": [{"id": "scaler_policy_b." + k, "name": k, "shape": list(v.shape),
                "dtype": "float32", "values": v.reshape(-1).tolist(),
                "sha256": hashlib.sha256(np.ascontiguousarray(v, dtype="<f4").tobytes()).hexdigest(),
                "checkpoint_origin": "assets/stage1_encoder.pt::scaler_policy_b/" + k}
                for k, v in payload["scaler_policy_b"].items()],
            "source_ref": "source/pre_condense_models.py::FoldScaler.transform_context"},
        "sources_manifest": [{"path": p, "sha256": h} for p, h in EXPECTED.items() if p.startswith("source/")],
        "parameters": parameters, "cases": cases,
        "coverage": {"stage1_only": True, "stage2": False, "query_or_basis_heads_available": False, "dropped_records": 0,
            "executed_parameter_ids": sorted(executed), "unused_parameter_ids": sorted(set(parameter_tensors) - executed),
            "diagnostic_projections": ["ctx_cross_attn.k_proj", "ctx_cross_attn.v_proj", "ctx_bilinear_u"],
            "omitted": ["missing pretrained query_tokens and basis661/denoise629 heads", "unsaved Stage 2 residual heads",
                "existing 10-token/SVD representation is not an input to G", "legacy auxiliary lm48 computed by unchanged DOPE wrapper but not passed to G"],
            "scaler_internals": "original NumPy FoldScaler captured at executed-function boundary; Torch native functional operations retain object lineage"},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "training": False, "optimizer": None, "argv": sys.argv,
            "capture_source_sha256": sha(__file__), "inputs_manifest_sha256": sha(args.inputs)},
        "validation": {"parameter_before_equals_after": True, "cases": checks}}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    for relative in EXPECTED:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)
        require(sha(target) == EXPECTED[relative], "copied capture asset differs")
    trace_path = out / "trace.json"
    with trace_path.open("x") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    receipt = {"schema": "jain.h256-capture-receipt/v1", "status": "PASS", "trace_path": str(trace_path.resolve()),
        "trace_sha256": sha(trace_path), "capture_source_sha256": sha(__file__), "argv": sys.argv,
        "inputs_manifest_sha256": sha(args.inputs), "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": EXPECTED["assets/stage1_encoder.pt"], "dope_before": before.composite_sha256,
        "dope_after": after.composite_sha256, "checks": checks,
        "scope": data["inference_scope"]}
    with (out / "capture.receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "trace": str(trace_path), "sha256": receipt["trace_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
