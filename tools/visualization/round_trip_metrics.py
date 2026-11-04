#!/usr/bin/env python3
"""Measure a warm CSV -> full saved H256 inference -> fitted, complete Rust trip.

One perf_counter_ns interval encloses native fresh FIT landmark encoding, the
original frozen DOPE/R882/five-fold G/critic computation, proposal selection,
native GP fitting, and receipt of the complete Rust source. Checkpoints and the
outcome-free candidate bank are resident. Compilation, source parity proofs,
output-source hashing, token counting and telemetry follow the interval. This is an
equivalent Rust-source throughput measurement, not neural token decoding.

The stdlib-only load_metrics entry point also runs in Blender without Torch.
Raw data, saved parameters, complete generated source and receipts stay outside
Git. Existing scientific capture and converter files are never modified.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import struct
import sys
import time

SCHEMA = "jain.h256-data-to-rust-metrics/v1"
LEXER = "jope.source-lex/v1"
SCOPE = "warm resident CPU; fresh raw CSV/FIT landmarks -> frozen DOPE + R882 K0 + fresh critics + five G priors -> selected native GP fit -> complete Rust bytes"
MAX_LINE = 16 * 1024 * 1024
MAX_CASES = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def case_catalog(cases, label):
    """Require one bounded actual catalog without silently collapsing duplicate IDs."""
    require(isinstance(cases, list) and 1 <= len(cases) <= MAX_CASES,
            label + " must contain 1..16 actual cases")
    require(all(isinstance(case, dict) and isinstance(case.get("id"), str)
                and 0 < len(case["id"]) <= 128 for case in cases), label + " has an invalid case identity")
    result = {case["id"]: case for case in cases}
    require(len(result) == len(cases), label + " has duplicate case identities")
    return result


def _close(actual, expected, name):
    require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9),
            "invalid metric arithmetic: " + name)


def _fit_identity(proof):
    """Independently check the helper's byte-hash domain using the bound contract.

    The earlier complete-source proof hashes normalized arrays as JSON. The
    resident helper hashes their IEEE little-endian f64 bytes. Keep those domains
    separate, and recompute the latter from the exact same CSV/FIT/contract here.
    This function executes after timing and also works without NumPy in Blender.
    """
    contract_path = Path(proof["original_contract_model"]["path"]) / "input.json"
    require(sha(contract_path) == proof["original_contract_model"]["input_sha256"], "original input contract changed")
    contract = read_json(contract_path)
    csv_path = Path(proof["input_csv"]["path"])
    require(sha(csv_path) == proof["input_csv"]["sha256"], "original FIT CSV changed")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fit = proof["split"]["fit"]
    columns, target = contract["columns"], contract["target"]
    x = [[float(rows[i][column["name"]]) for column in columns] for i in fit]
    y = [float(rows[i][target["name"]]) for i in fit]
    require(all(finite(v) for row in x for v in row) and all(finite(v) for v in y), "nonfinite raw FIT")
    normalized = [[0. if columns[j]["constant"] else max(0., min(1.,
                       (row[j] - columns[j]["min"]) / (columns[j]["max"] - columns[j]["min"])))
                   for j in contract["selected"]] for row in x]
    scaled_y = [(v - target["min"]) / target["scale"] for v in y]
    def digest(values, code):
        return hashlib.sha256(b"".join(struct.pack("<" + code, v) for v in values)).hexdigest()
    return {"input_contract_sha256": sha(contract_path),
        "original_model_id": proof["original_contract_model"]["model_id"],
        "raw_fit_x_f32_sha256": digest((v for row in x for v in row), "f"),
        "raw_fit_y_f32_sha256": digest(y, "f"),
        "normalized_fit_x_f64_sha256": digest((v for row in normalized for v in row), "d"),
        "normalized_fit_y_f64_sha256": digest(scaled_y, "d")}


def _sample(sample, expected_source):
    ns = sample.get("total_ns")
    tokens = sample.get("rust_tokens")
    require(type(ns) is int and ns > 0 and type(tokens) is int and tokens > 0,
            "timing and canonical token count must be positive integers")
    _close(sample.get("total_ms"), ns / 1e6, "total_ms")
    _close(sample.get("equivalent_rust_tokens_per_s"), tokens * 1e9 / ns, "tokens/sec")
    require(sample.get("rust_sha256") == expected_source and sample.get("source_bytes_exact") is True,
            "measured Rust bytes differ from the proven source")
    require(sample.get("selection_exact") is True and sample.get("descriptor_bitwise") is True,
            "fresh selection or FIT descriptors differ")
    require(sample.get("lexer") == LEXER, "canonical Rust lexer identity differs")
    require(sample.get("native_release") is True, "native fitting must use the proven release profile")
    require(sample.get("telemetry_event_id") == 0x012A, "missing canonical metric telemetry")
    require(sample.get("telemetry_denominator_ns") == ns,
            "internal metric denominator differs from the outer round-trip span")
    require(sample.get("telemetry_languages") == ["rust"], "metric includes a non-Rust source")
    native = sample.get("native_metric", {})
    require(native.get("lexer") == LEXER and native.get("rust_tokens") == tokens
            and native.get("all_languages_tokens") == tokens and native.get("output_languages") == ["rust"]
            and native.get("round_trip_ns") == ns and native.get("metric_id") == "0x012a"
            and native.get("rust_timed_alone") is False and native.get("neural_decoding_tokens_per_second") is None,
            "native Rust-only canonical metric contract differs")
    _close(native.get("rust_tokens_per_second"), tokens * 1e9 / ns, "native Rust tokens/sec")
    _close(native.get("total_ms"), ns / 1e6, "native total_ms")
    events = dict(native.get("metric_events", []))
    _close(events.get(0x012A), tokens * 1e9 / ns, "telemetry 0x012a")
    _close(events.get(0x012B), tokens * 1e9 / ns, "Rust-only compatibility telemetry 0x012b")
    require(finite(sample.get("final_cost_max_abs_error")) and sample["final_cost_max_abs_error"] <= 2e-5,
            "fresh complete neural cost differs from the proven pass")
    require(finite(sample.get("critic_max_abs_error")) and sample["critic_max_abs_error"] <= 2e-5,
            "fresh critic outputs differ from the bound original critic")


def load_metrics(path, trace_path, data=None):
    """Strictly load an artifact-bound receipt; no Torch import or recomputation.

    Returns the full JSON, including cases[cid]. A consumer must pass the exact
    source-enriched trace it renders. Source/helper/checkpoint/input bytes are
    rehashed; changes require a new benchmark, rather than a stale cached label.
    """
    path, trace_path = Path(path).resolve(), Path(trace_path).resolve()
    receipt = read_json(path)
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "PASS", "unqualified round-trip receipt")
    require(receipt.get("trace_sha256") == sha(trace_path), "metric source trace changed")
    data = read_json(trace_path) if data is None else data
    require(data.get("schema") == "jain.h256-stage2-capture/v1", "metrics require the actual H256 Stage2 trace")
    require(receipt.get("lexer") == LEXER and receipt.get("metric_kind") == "equivalent_rust_source_throughput",
            "round-trip metric scope or canonical lexer differs")
    runtime = receipt.get("runtime", {})
    require(runtime.get("device") == "cpu" and runtime.get("torch_threads") == 1
            and runtime.get("torch_interop_threads") == 1 and runtime.get("training") is False,
            "round-trip runtime contract differs")
    method = receipt.get("method", {})
    require(method.get("outer_clock") == "time.perf_counter_ns" and method.get("warmup_per_case") == 1
            and method.get("samples_per_case") == 3 and method.get("statistic") == "paired_median_duration_sample"
            and method.get("one_contiguous_interval") is True
            and method.get("proof_compilation_in_timed_interval") is False
            and method.get("token_count_after_clock") is True,
            "round-trip method is incomplete or differs")
    require(receipt.get("dope_before") == data["dope"]["composite_expected"] == receipt.get("dope_after"),
            "frozen DOPE mutation or lineage mismatch")
    bindings = receipt.get("artifacts", [])
    require(len(bindings) >= 8, "missing benchmark source/checkpoint/input bindings")
    for binding in bindings:
        item = Path(binding["path"])
        require(item.is_file() and item.stat().st_size == binding["bytes"] and sha(item) == binding["sha256"],
                "round-trip bound artifact changed: " + str(item))
    require(any(Path(b["path"]).resolve() == Path(__file__).resolve() for b in bindings),
            "round-trip Python implementation is unbound")
    expected = case_catalog(data.get("cases"), "render trace")
    cases = receipt.get("cases", {})
    require(set(cases) == set(expected) and receipt.get("case_count") == len(expected), "round-trip case catalog differs")
    chosen = receipt.get("input_artifacts", {})
    require(set(chosen) == {"fit_inputs", "native_features", "critic_inputs"}, "input artifact catalog bindings are incomplete")
    for label, binding in chosen.items():
        require(binding in bindings, "chosen input artifact is outside the bound inventory: " + label)
        catalog = case_catalog(read_json(binding["path"])["cases"], label)
        require(set(catalog) == set(expected), "chosen input case identities differ: " + label)
        if label == "fit_inputs":
            require(all(catalog[cid]["dataset_sha256"] == case["dataset_sha256"]
                        and catalog[cid]["fit_row_ids"] == case["fit_row_ids"]
                        and catalog[cid]["holdout_row_ids"] == case["holdout_row_ids"]
                        for cid, case in expected.items()), "chosen FIT dataset/partition differs")
        if label == "native_features":
            require(all(catalog[cid]["programs"] == case["programs"] for cid, case in expected.items()),
                    "chosen native candidate programs differ")
    request_ids = set()
    for cid, result in cases.items():
        case = expected[cid]
        source = case["source"]
        require(result.get("dataset_sha256") == case["dataset_sha256"]
                and result.get("fit_row_ids") == case["fit_row_ids"]
                and result.get("selected_index") == case["selection"]["selected_slate_index"]
                and result.get("selected_program_sha256") == source["selected_program_sha256"]
                and result.get("rust_sha256") == source["rust_sha256"]
                and result.get("model_id") == source["model_id"], "round-trip case/source identity mismatch: " + cid)
        rust = trace_path.parent / source["rust_path"]
        require(sha(rust) == source["rust_sha256"], "rendered Rust source bytes changed")
        source_proof = read_json(trace_path.parent / source["proof_path"])
        fit_identity = _fit_identity(source_proof)
        samples = result.get("samples", [])
        require(len(samples) == 3, "round-trip needs all three paired samples")
        for sample in [result["warmup"], *samples]:
            _sample(sample, source["rust_sha256"])
            rid = sample.get("request_id")
            require(isinstance(rid, str) and rid not in request_ids, "duplicate round-trip request identity")
            request_ids.add(rid)
            require(sample.get("selected_index") == result["selected_index"]
                    and sample.get("model_id") == result["model_id"], "per-sample selection/model drift")
            identity = sample.get("native_identity", {})
            require(identity.get("dataset_sha256") == case["dataset_sha256"]
                    and identity.get("fit_row_ids") == case["fit_row_ids"]
                    and identity.get("holdout_row_ids") == case["holdout_row_ids"]
                    and identity.get("split_sha256") == source_proof["split"]["sha256"]
                    and all(identity.get(key) == value for key, value in fit_identity.items()),
                    "fresh native FIT normalization differs from the proven source fit")
        middle = sorted(enumerate(samples), key=lambda item: (item[1]["total_ns"], item[0]))[1][0]
        require(result.get("representative_sample_index") == middle, "unpaired or cherry-picked median")
        selected = samples[middle]
        for key in ("total_ns", "total_ms", "rust_tokens", "equivalent_rust_tokens_per_s", "lexer"):
            require(result.get(key) == selected[key], "display metric is not one actual paired sample")
        require(result.get("min_total_ns") == min(s["total_ns"] for s in samples)
                and result.get("max_total_ns") == max(s["total_ns"] for s in samples), "hidden timing range")
    return receipt


class NativeHelper:
    """Resident JSONL process. Read complete UTF-8 code before stopping the timer."""
    def __init__(self, binary, log_path):
        self.log = Path(log_path).open("wb")
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, env=os.environ.copy())

    def request(self, value, stop_clock=False):
        wire = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        self.process.stdin.write(wire)
        self.process.stdin.flush()
        line = self.process.stdout.readline(MAX_LINE + 1)
        ended = time.perf_counter_ns() if stop_clock else None
        require(line.endswith(b"\n") and len(line) <= MAX_LINE, "native helper EOF or oversized response")
        result = json.loads(line)
        require(result.get("request_id") == value["request_id"] and result.get("case_id") == value["case_id"],
                "native helper response identity differs")
        require(result.get("phase") != "error" and result.get("status") != "ERROR",
                "native helper rejected request: " + str(result))
        return result, ended

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=10)
            require(code == 0, "native helper exited unsuccessfully")
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.log.close()


class ResidentInference:
    """Original saved computation, with only recording/hashing removed from the span."""
    def __init__(self, root, custody, trace, native_features, fit_inputs, critic_inputs):
        import numpy as np
        import torch
        import h256_inference as stage1
        import h256_stage2_inference as stage2
        import h256_stage2_inputs as inputs
        self.np, self.torch, self.stage1 = np, torch, stage1
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261103)
        for relative, expected in stage1.EXPECTED.items():
            require(sha(root / relative) == expected, "original DOPE custody differs: " + relative)
        require(sha(custody / "assets/stage2-ensemble.pt") == stage2.ENSEMBLE_SHA, "saved Stage2 differs")
        require(sha(custody / "assets/unified.pt") == stage2.UNIFIED_SHA, "saved unified identity differs")
        require(sha(custody / "assets/critic.pt") == inputs.CRITIC_SHA, "saved critics differ")
        ensemble = stage2.load_saved(custody / "assets/stage2-ensemble.pt")
        unified = stage2.load_saved(custody / "assets/unified.pt")
        require(ensemble["arm"] == stage1.ARM and len(ensemble["fold_state_dicts"]) == 5
                and ensemble["dope_sha256"] == stage1.DOPE_COMPOSITE, "wrong saved ensemble")
        state = torch.load(root / "assets/round-882.pt", map_location="cpu", weights_only=True)["state_dict"]
        other = unified["base_solution_ranker_state_dict"]
        require(set(state) == set(other) and all(torch.equal(v, other[k]) for k, v in state.items()),
                "unified saved base differs from original R882")
        self.model, self.dope_fn, self.freeze, self.before = stage1.load_dope(root, state)
        stage2.attach_stage1b(self.model, root, state)
        self.base = inputs.original_base(custody / "source", state)
        stage2.attach_sync_free_attention(self.base, root)
        require(sha(custody / "source/pre_condense_models.py") == stage1.EXPECTED["source/pre_condense_models.py"],
                "original Stage2 architecture differs")
        original = stage1.load_original(custody / "source/pre_condense_models.py", "h256_round_trip_original")
        self.bridges = []
        for saved in ensemble["fold_state_dicts"]:
            require(len(saved) == 84 and sum(v.numel() for v in saved.values()) == 183668, "fold inventory differs")
            bridge = original.PreCondenseResidualBridge(arm=stage1.ARM)
            bridge.load_state_dict(saved, strict=True)
            bridge.eval().requires_grad_(False)
            self.bridges.append(bridge)
        self.scalers = [original.FoldScaler(**{k: np.asarray(v, dtype=np.float32)
                        for k, v in ensemble["full_train_scaler"].items()}) for _ in range(5)]
        self.alpha = float(ensemble["ensemble_alpha_prior"])
        require(self.alpha == trace["checkpoint"]["alpha_prior"], "G prior scale differs")
        self.critic = stage2.load_saved(custody / "assets/critic.pt")
        require(self.critic["dope_sha256"] == stage1.DOPE_COMPOSITE and self.critic["fold_seed"] == 256140,
                "saved critic training identity differs")
        self.build_features, self.semantic = inputs.original_builders(custody / "source")
        self.cases = case_catalog(trace["cases"], "neural trace")
        self.fit_inputs = case_catalog(read_json(fit_inputs)["cases"], "FIT input catalog")
        self.reference_critics = case_catalog(read_json(critic_inputs)["cases"], "critic input catalog")
        self.native = case_catalog(read_json(native_features)["cases"], "native program feature catalog")
        require(set(self.cases) == set(self.fit_inputs) == set(self.reference_critics) == set(self.native),
                "trace/FIT/critic/program case inventories differ")
        self.resident = {}
        for cid, case in self.cases.items():
            bank, fit = self.native[cid], self.fit_inputs[cid]
            require(bank["programs"] == case["programs"] and len(bank["programs"]) == 12
                    and fit["dataset_sha256"] == case["dataset_sha256"]
                    and fit["fit_row_ids"] == case["fit_row_ids"], "static candidate/FIT identity differs")
            self.resident[cid] = {
                "raw": torch.tensor(bank["raw_program_features412"], dtype=torch.float32).reshape(1, 12, 412),
                "projection": torch.tensor(bank["projection16_f64"], dtype=torch.float64).reshape(1, 12, 16),
                "program_numpy": np.asarray(bank["raw_program_features412"], dtype=np.float32).reshape(12, 412),
            }
        self.hidden = []
        self.hidden_hook = self.model.dope_model.router.task_latent_head.register_forward_pre_hook(
            lambda _module, args: self.hidden.append(args[0]))

    def critics(self, program, base_cost, lm48):
        """Exact original predict arithmetic; original AST builders were loaded once."""
        np = self.np
        rcost = np.asarray(base_cost, dtype=np.float32).reshape(12)
        sem = self.semantic(program)
        cid = "native-input"
        feature64 = self.build_features([{ "context_sha256": cid }],
            {cid: {"program": program, "sem24": sem}},
            {cid: {"total_cost": rcost, "anchor_total": rcost, "anchor_prior": np.zeros(12, dtype=np.float32)}},
            {cid: {"lm48_z": np.asarray(lm48, dtype=np.float32).reshape(48)}})[cid]
        x65 = np.concatenate([feature64.astype(np.float64), np.ones((12, 1), dtype=np.float64)], axis=1)
        pc = x65 @ np.asarray(self.critic["w_c_full"], dtype=np.float64)
        pg = x65 @ np.asarray(self.critic["w_g_full"], dtype=np.float64)
        cont = ((pc - float(np.mean(pc))) / (float(np.std(pc)) + 1e-6)).astype(np.float32)
        gate = ((pg - float(np.mean(pg))) / (float(np.std(pg)) + 1e-6)).astype(np.float32)
        sp = -(sem.astype(np.float64) @ np.asarray(self.critic["sp_dir_full"], dtype=np.float64))
        sp = ((sp - float(np.mean(sp))) / (float(np.std(sp)) + 1e-6)).astype(np.float32)
        unsafe = gate.astype(np.float64) > (float(gate[int(np.argmin(rcost))]) + .10)
        hcost = (.65 * cont.astype(np.float64) + .35 * gate.astype(np.float64)
                 + .15 * sp.astype(np.float64) + np.where(unsafe, 10., 0.))
        return gate, sp, hcost

    def forward(self, cid, encoded):
        np, torch = self.np, self.torch
        bank, case = self.resident[cid], self.cases[cid]
        self.hidden.clear()
        with torch.no_grad():
            v = torch.tensor(encoded["v256"], dtype=torch.float32).reshape(1, 256)
            col = torch.tensor(encoded["col64x17"], dtype=torch.float32).reshape(1, 64, 17)
            dope629, lm48 = self.dope_fn(self.model, v, col, use_landmarks=True)
            h256 = self.hidden[-1]
            tokens, mask, lm48_z = self.model._project_stage1b_tokens_batched_gpu(lm48, dope629)
            base = self.base.predict_costs(ctx_feat=torch.zeros((1, 584)), prog_412=bank["raw"],
                proj_16=bank["projection"], base_prior_cost=torch.zeros((1, 12), dtype=torch.float64),
                scale=torch.tensor([.4208], dtype=torch.float64), sos_prior=torch.zeros((1, 12, 2)),
                mm_tokens_10x32=tokens, mm_mask_10=mask, revealed_indices=torch.empty((1, 0), dtype=torch.long),
                revealed_raw_f=torch.empty((1, 0), dtype=torch.float64), revealed_gate=torch.empty((1, 0)))
            gate, sp, hcost = self.critics(bank["program_numpy"], base["total_cost"].numpy(), lm48_z.numpy())
            gate_t = torch.tensor(gate, dtype=torch.float32).reshape(1, 12)
            sp_t = torch.tensor(sp, dtype=torch.float32).reshape(1, 12)
            entry = {"dope629": dope629.numpy().reshape(629), "hidden256": h256.numpy().reshape(256),
                "col_lm_64x17": col.numpy().reshape(64, 17), "col_mask_64": np.arange(64) < case["features"],
                "tokens_10x32": tokens.numpy().reshape(10, 32), "mask_10": mask.numpy().reshape(10)}
            priors = []
            for bridge, scaler in zip(self.bridges, self.scalers):
                ctx = scaler.transform_context(entry, device=torch.device("cpu"))
                prior, _attention, _bilinear = bridge.forward_prior_and_context(
                    ctx, base["cand_q"], base["sem24"], gate_t, sp_t)
                priors.append(prior)
            mean = torch.stack(priors, dim=0).double().mean(dim=0)
            initial = base["total_cost"].double() + .005 * torch.tensor(hcost, dtype=torch.float64).reshape(1, 12)
            final = (initial + self.alpha * mean).numpy().reshape(12)
            selected = 4 + int(np.argmin(final[4:]))
        return {"selected_index": selected, "selected_dsl": case["programs"][selected],
                "final_cost": final, "gate_risk": gate, "sp_prior": sp, "h_cost": hcost}

    def check(self, cid, encoded, result):
        """Outside the timer: reject descriptor drift and altered learned selection."""
        np = self.np
        case, fit = self.cases[cid], self.fit_inputs[cid]
        for key, field in (("v256", "v256_f32_sha256"), ("col64x17", "col64x17_f32_sha256")):
            expected = np.asarray(fit[key], dtype="<f4").tobytes()
            actual = np.asarray(encoded[key], dtype="<f4").tobytes()
            require(actual == expected and hashlib.sha256(actual).hexdigest() == encoded[field],
                    "fresh native descriptor bits differ: " + cid + "/" + key)
        identity = encoded["identity"]
        require(identity["dataset_sha256"] == case["dataset_sha256"]
                and identity["fit_row_ids"] == case["fit_row_ids"]
                and identity["holdout_row_ids"] == case["holdout_row_ids"], "fresh FIT/CSV identity differs")
        require(result["selected_index"] == case["selection"]["selected_slate_index"]
                and result["selected_dsl"] == case["source"]["selected_program"], "fresh G selection differs")
        ref = np.asarray(case["outputs"]["final_k0_cost"], dtype=np.float64)
        cost_error = float(np.max(np.abs(result["final_cost"] - ref)))
        require(bool(np.isfinite(result["final_cost"]).all()) and cost_error <= 2e-5, "fresh final G cost differs")
        arrays = self.reference_critics[cid]["arrays"]
        critic_error = max(float(np.max(np.abs(result[name] - np.asarray(arrays[name]))))
                           for name in ("gate_risk", "sp_prior", "h_cost"))
        require(critic_error <= 2e-5, "fresh critic features/predictions differ")
        return cost_error, critic_error

    def close(self):
        self.hidden_hook.remove()
        after = self.freeze.snapshot_dope_weights(self.model).composite_sha256
        require(after == self.before.composite_sha256, "frozen DOPE weights mutated")
        return after


def benchmark(args):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    root, custody = args.root.resolve(), args.stage2_custody.resolve()
    trace_path, output = args.trace.resolve(), args.output_dir.resolve()
    require(not output.exists(), "output already exists; refusing to overwrite evidence")
    require(args.native_helper.is_file() and os.access(args.native_helper, os.X_OK), "native helper is not executable")
    trace = read_json(trace_path)
    require(trace.get("schema") == "jain.h256-stage2-capture/v1", "benchmark requires an actual H256 Stage2 capture")
    trace_cases = case_catalog(trace.get("cases"), "benchmark trace")
    input_path = (args.fit_inputs or root / "inputs/native-fit-inputs.json").resolve()
    feature_path = (args.native_features or root / "native-program-features.json").resolve()
    critic_path = (args.critic_inputs or root / "native-stage2-inputs-v3/critic-predictions.json").resolve()
    original_root = args.original_run.resolve()
    model_entries = read_json(original_root / "model-paths.json")
    require(isinstance(model_entries, list) and len(model_entries) == len(trace_cases), "original model catalog count differs")
    models = {m["case_id"]: m for m in model_entries}
    require(len(models) == len(model_entries) and set(models) == set(trace_cases), "original model case identities differ")
    implementation = Path(__file__).resolve()
    sources = [implementation, implementation.with_name("h256_inference.py"),
               implementation.with_name("h256_stage2_inference.py"), implementation.with_name("h256_stage2_inputs.py"),
               args.native_helper.resolve(), args.native_helper_source.resolve(), input_path, feature_path, critic_path,
               original_root / "model-paths.json", root / "assets/round-882.pt",
               custody / "assets/stage2-ensemble.pt", custody / "assets/critic.pt", custody / "assets/unified.pt"]
    sources.extend(root / rel for rel in read_json(root / "custody.receipt.json").get("files", {})
                   if isinstance(rel, str) and rel.startswith("source/") and (root / rel).is_file())
    # The exact AST files actually executed are bound even if a custody receipt
    # uses a different listing shape. Include every pinned original source.
    sources.extend(sorted((root / "source").glob("*.py")))
    sources.extend(sorted((custody / "source").glob("*.py")))
    jobs = []
    for case in trace["cases"]:
        cid, source = case["id"], case["source"]
        csv = original_root / (cid + ".csv")
        split = original_root / cid / "proof/split.json"
        rust = trace_path.parent / source["rust_path"]
        proof = trace_path.parent / source["proof_path"]
        require(sha(csv) == case["dataset_sha256"] and sha(rust) == source["rust_sha256"]
                and sha(proof) == source["proof_sha256"], "bound source/data proof changed")
        proof_data = read_json(proof)
        require(proof_data["state"] == "pass" and proof_data["source_unchanged"] is True
                and proof_data["model"]["model_id"] == source["model_id"], "unqualified native fit/source proof")
        model_dir = Path(models[cid]["model_directory"])
        sources.extend([csv, split, split.with_name("summary.json"), rust, proof])
        sources.extend(sorted(p for p in model_dir.rglob("*") if p.is_file()))
        jobs.append((cid, case, {"case_id": cid, "csv_path": str(csv), "split_path": str(split),
                               "original_model_dir": str(model_dir)}))
    before = [artifact(p) for p in dict.fromkeys(sources)]
    resident = ResidentInference(root, custody, trace, feature_path, input_path, critic_path)
    output.mkdir(parents=True)
    helper = NativeHelper(args.native_helper.resolve(), output / "native-helper.stderr.log")
    results = {}
    try:
        for cid, case, common in jobs:
            measured, warmup = [], None
            for repeat in range(4):
                request_id = f"{cid}-{'warmup' if repeat == 0 else 'sample-' + str(repeat)}"
                common_request = {**common, "request_id": request_id}
                started = time.perf_counter_ns()
                encoded, _ = helper.request({**common_request, "mode": "encode"})
                require(encoded.get("phase") == "encoded", "native helper failed fresh encoding")
                neural = resident.forward(cid, encoded)
                ready, ended = helper.request({**common_request, "mode": "fit-rust",
                                               "selected_dsl": neural["selected_dsl"]}, stop_clock=True)
                # Nothing below contributes to the outer interval. The complete
                # code_ready UTF-8 response was already read when ended was taken.
                elapsed = ended - started
                require(ready.get("phase") == "code_ready", "native source is not ready")
                rust_bytes = ready["rust_source"].encode("utf-8")
                require(len(rust_bytes) == ready["rust_source_bytes"]
                        and hashlib.sha256(rust_bytes).hexdigest() == case["source"]["rust_sha256"],
                        "timed complete source differs from the proven Rust file")
                require(ready["canonical_dsl"] == case["source"]["selected_program"]
                        and ready["fitted_model_id"] == case["source"]["model_id"]
                        and ready["identity"] == encoded["identity"], "fit/export identity changed within round trip")
                cost_error, critic_error = resident.check(cid, encoded, neural)
                proof, _ = helper.request({"mode": "proof", "request_id": request_id, "case_id": cid,
                    "expected_rust_sha256": case["source"]["rust_sha256"], "round_trip_ns": elapsed})
                require(proof.get("phase") == "proof", "native token/hash/telemetry proof failed")
                for response in (encoded, ready, proof):
                    require(response.get("schema") == "jain.local-inference-helper/v1"
                            and response.get("helper_source_sha256") == sha(args.native_helper_source)
                            and response.get("native_debug_assertions") is False,
                            "native release helper source identity differs")
                require(proof["source_bytes_equal_verified_artifact"] is True
                        and proof["fitted_model_id"] == ready["fitted_model_id"]
                        and proof["canonical_dsl"] == ready["canonical_dsl"], "native source proof identity differs")
                metric = proof["metric"]
                # Native helper names are intentionally explicit; no Python lexer
                # or synthesized token count is accepted as a fallback.
                tokens = metric["rust_tokens"]
                sample = {"request_id": request_id, "total_ns": elapsed, "total_ms": elapsed / 1e6,
                    "rust_tokens": tokens, "equivalent_rust_tokens_per_s": tokens * 1e9 / elapsed,
                    "lexer": metric["lexer"], "rust_sha256": proof["source_sha256"],
                    "source_bytes_exact": True, "descriptor_bitwise": True, "selection_exact": True,
                    "native_release": True,
                    "selected_index": neural["selected_index"], "model_id": ready["fitted_model_id"],
                    "final_cost_max_abs_error": cost_error, "critic_max_abs_error": critic_error,
                    "telemetry_event_id": int(metric["metric_id"], 16),
                    "telemetry_denominator_ns": metric["round_trip_ns"],
                    "telemetry_languages": metric["output_languages"],
                    "native_fit_ns": int(ready["fit_ns"]), "native_export_ns": int(ready["export_ns"]),
                    "native_identity": encoded["identity"], "native_metric": metric}
                _sample(sample, case["source"]["rust_sha256"])
                _close(metric["rust_tokens_per_second"], sample["equivalent_rust_tokens_per_s"], "native telemetry rate")
                if repeat == 0:
                    warmup = sample
                else:
                    measured.append(sample)
                print(f"{request_id}: {sample['total_ms']:.3f} ms; {tokens} Rust tokens; {sample['equivalent_rust_tokens_per_s']:.1f} equivalent Rust tokens/s", flush=True)
            middle = sorted(range(3), key=lambda index: (measured[index]["total_ns"], index))[1]
            results[cid] = {"dataset_sha256": case["dataset_sha256"], "fit_row_ids": case["fit_row_ids"],
                "selected_index": case["selection"]["selected_slate_index"],
                "selected_program_sha256": case["source"]["selected_program_sha256"],
                "rust_sha256": case["source"]["rust_sha256"], "model_id": case["source"]["model_id"],
                "warmup": warmup, "samples": measured, "representative_sample_index": middle,
                "min_total_ns": min(s["total_ns"] for s in measured),
                "max_total_ns": max(s["total_ns"] for s in measured),
                **{key: measured[middle][key] for key in ("total_ns", "total_ms", "rust_tokens", "equivalent_rust_tokens_per_s", "lexer")}}
        after = resident.close()
        require(before == [artifact(item["path"]) for item in before], "benchmark bound artifact mutated during measurement")
        receipt = {"schema": SCHEMA, "status": "PASS", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "trace_path": str(trace_path), "trace_sha256": sha(trace_path),
            "metric_kind": "equivalent_rust_source_throughput", "lexer": LEXER, "scope": SCOPE,
            "case_count": len(trace_cases),
            "input_artifacts": {"fit_inputs": artifact(input_path), "native_features": artifact(feature_path),
                                "critic_inputs": artifact(critic_path)},
            "method": {"outer_clock": "time.perf_counter_ns", "one_contiguous_interval": True,
                "start": "before native encode JSONL request; CSV is freshly read in native helper",
                "stop": "immediately after reading the complete code_ready UTF-8 JSONL response, before parsing that response",
                "includes": ["helper IPC/JSON input and output", "native input identity hashes and contract/range guards",
                    "fresh raw CSV reads and exact FIT parsing/normalization",
                    "fresh native high landmark256 and col64x17 encoding", "original frozen DOPE and fresh R882 K0 base",
                    "fresh saved hurdle/SP critics", "five saved G prior forwards and eligible proposal minimum",
                    "native GP fitting and complete Rust-only source export"],
                "resident_outside_interval": ["checkpoint/model/scaler loads", "original contract static checks",
                    "static outcome-free 12-program bank raw412/projection16", "original AST feature builders"],
                "excludes": ["hash/identity/reference validation after response", "canonical token counting and telemetry",
                    "compiler/selfcheck/training/parity proofs", "Python source export", "Blender/rendering"],
                "warmup_per_case": 1, "samples_per_case": 3, "statistic": "paired_median_duration_sample",
                "cache_state": "warm resident models and contract cache; OS filesystem cache is not flushed",
                "proof_compilation_in_timed_interval": False, "token_count_after_clock": True,
                "interpretation": "Rust lexical tokens in complete source divided by the actual whole data-to-source wall time; not neural token decoding"},
            "runtime": {"device": "cpu", "torch_threads": 1, "torch_interop_threads": 1,
                "native_profile": "release", "native_debug_assertions": False,
                "training": False, "python": sys.version, "torch": resident.torch.__version__,
                "platform": platform.platform(), "cpu": platform.processor(), "argv": sys.argv},
            "dope_before": resident.before.composite_sha256, "dope_after": after,
            "artifacts": before, "cases": results}
        target = output / "catalog.receipt.json"
        target.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        load_metrics(target, trace_path, trace)
        return target
    except Exception as error:
        (output / "failure.receipt.json").write_text(json.dumps({"status": "FAIL", "error": str(error),
            "completed_cases": list(results), "trace_sha256": sha(trace_path)}, indent=2) + "\n")
        raise
    finally:
        helper.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="original H256 custody root")
    parser.add_argument("--stage2-custody", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True, help="same-case CSV/split/original model-paths root")
    parser.add_argument("--trace", type=Path, required=True, help="source-enriched actual H256 trace")
    parser.add_argument("--fit-inputs", type=Path, help="actual catalog FIT descriptors; defaults to ROOT/inputs/native-fit-inputs.json")
    parser.add_argument("--native-features", type=Path, help="actual candidate feature catalog; defaults to ROOT/native-program-features.json")
    parser.add_argument("--critic-inputs", type=Path, help="actual critic reference catalog; defaults to ROOT/native-stage2-inputs-v3/critic-predictions.json")
    parser.add_argument("--native-helper", type=Path, required=True, help="release resident visual_inference_metric executable")
    parser.add_argument("--native-helper-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(benchmark(args))


if __name__ == "__main__":
    main()
