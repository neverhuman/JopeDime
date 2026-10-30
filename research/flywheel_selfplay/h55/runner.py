#!/usr/bin/env python3
"""Leakage-guarded H55 adapter averaging cell runner."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
H48_PATH = ROOT / "research/flywheel_selfplay/h48/run.py"
H49_SCORE_PATH = ROOT / "research/flywheel_selfplay/h49/score.py"
H53B_PATH = ROOT / "research/flywheel_selfplay/h53b/runner.py"
DEV_PATH = ROOT / "research/flywheel_selfplay/dev_incumbent.py"
TRAINING_PATH = ROOT / "research/flywheel_selfplay/automatic_promotion.py"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
TRAINABLE_PARAMS = 29_211
FOLD_SEED = 4900
SEEDS = (12, 13, 14)
K_STEPS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1219
SLATE_SIZE = 192
SWA_MEMBERS = 8
EMA_ROUNDS = 25
EMA_DECAY = 0.9
R882_HARNESS_UTILITY = 1.1536152769912449


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes(input_dir: Path) -> dict[str, str]:
    paths = (
        H48_PATH, H49_SCORE_PATH, H53B_PATH, DEV_PATH, TRAINING_PATH,
        ROOT / "research/flywheel_selfplay/run_fused_closed_loop.py",
        ROOT / "research/flywheel_selfplay/h46/run.py",
        ROOT / "research/flywheel_selfplay/h55/runner.py",
        ROOT / "research/flywheel_selfplay/h55/run_wave.py",
        ROOT / "research/flywheel_selfplay/h55/audit.py",
        ROOT / "research/flywheel_selfplay/h55/build_metrics.py",
        input_dir / "cpu_exact_fused.py",
    )
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path)
            for path in paths}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def family_folds(rows: list[dict], h48) -> list[set[str]]:
    folds = h48.family_folds(rows, count=5, seed=FOLD_SEED)
    if len(folds) != 5 or any(not fold for fold in folds):
        raise RuntimeError("seed-4900 split must contain five nonempty folds")
    flat = [family for fold in folds for family in fold]
    if len(flat) != len(set(flat)) or set(flat) != {r["family_id"] for r in rows}:
        raise RuntimeError("seed-4900 split is not a disjoint family partition")
    return folds


def load_rows(input_dir: Path, rows: list[dict], h48) -> tuple[dict, dict]:
    """Load payloads only for the supplied fold's rows."""
    base = h48.base
    pools: dict[str, dict] = {}
    tables: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        if row.get("dl_split") != "DL-Train" or row.get("source_split") not in ("train", "inner"):
            raise RuntimeError("fold includes data outside the approved DL-Train closure")
        if row.get("task") != "regression":
            raise RuntimeError("H55 supports only regression contexts")
        cid = row["context_sha256"]
        pool_path = input_dir / "pools" / f"{cid}.npz"
        table_path = input_dir / "tables" / f"{cid}.jtt1"
        if base.sha(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"pool hash mismatch for {cid}")
        if base.sha(table_path) != row["train_only_jtt1_sha256"]:
            raise RuntimeError(f"train-only table hash mismatch for {cid}")
        with np.load(pool_path, allow_pickle=False) as source:
            programs = source["program"].astype(np.float32)
            pids = [v.decode() if isinstance(v, bytes) else str(v) for v in source["program_sha256"].tolist()]
            pools[cid] = {
                "rec": row,
                "pids": pids,
                "program": programs,
                "proj16": programs.astype(np.float64) @ base.evaluator.get_w1_sign_matrix() / np.sqrt(412),
                "struct_sig": base.compute_semantic_struct_sig_24d(programs),
                "fitness": source["fitness"].astype(np.float64),
                "logratio": source["logratio"].astype(np.float64),
                "gate": source["gate"].astype(bool),
                "prior_cost": np.zeros(len(programs), dtype=np.float64),
                "scale": 0.4208,
            }
        raw = table_path.read_bytes()
        nt, nq, width = base.struct.unpack_from("<III", raw, 4)
        if nq or raw[:4] != b"JTT1" or raw[16] != 2:
            raise RuntimeError("H55 requires train-only two-fold JTT1 inputs")
        values = np.frombuffer(raw, dtype="<f8", offset=17 + nt).reshape(nt, width + 1).copy()
        tables[cid] = (values[:, :-1], values[:, -1])
    return pools, tables


def input_metadata(input_dir: Path, h48) -> tuple[dict, list[dict], list[dict]]:
    manifest_path = input_dir / "source_manifest.json"
    if sha256(manifest_path) != MANIFEST_SHA:
        raise RuntimeError("source manifest file SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_ID
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("source manifest identity/scope mismatch")
    selected = json.loads((input_dir / "selected.json").read_text())
    train_rows = selected.get("train", [])
    dev_rows = selected.get("dev", [])
    if len(train_rows) != 1219 or len({r["context_sha256"] for r in train_rows}) != 1219:
        raise RuntimeError("DL-Train context inventory mismatch")
    if len({r["family_id"] for r in train_rows}) != 205:
        raise RuntimeError("DL-Train family inventory mismatch")
    if len(dev_rows) != 91 or len({r["family_id"] for r in dev_rows}) != 19:
        raise RuntimeError("DL-Val metadata inventory mismatch")
    if {r["family_id"] for r in train_rows} & {r["family_id"] for r in dev_rows}:
        raise RuntimeError("DL-Train and DL-Val family metadata overlap")
    return selected, train_rows, dev_rows


def build_model(input_dir: Path, checkpoint: Path, device: torch.device, h48):
    if sha256(checkpoint) != R882_SHA:
        raise RuntimeError("r882 checkpoint SHA mismatch")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint_state.get("round") != 882 or checkpoint_state.get("dope_sha256") != DOPE_SHA:
        raise RuntimeError("r882 checkpoint metadata mismatch")
    fused = import_module("h55_fused", input_dir / "cpu_exact_fused.py")
    base = h48.base
    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for name, parameter in net.adapter_10tok.named_parameters():
        if not name.startswith("anchor."):
            parameter.requires_grad_(True)
    base.guard.lock_dope_weights_and_eval_mode(net)
    net.load_state_dict(checkpoint_state["state_dict"], strict=True)
    if h48.h46.checkpoint_guard(base.guard, net, expected=DOPE_SHA) != DOPE_SHA:
        raise RuntimeError("DOPE checkpoint guard mismatch")
    trainable = [p for p in net.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != TRAINABLE_PARAMS:
        raise RuntimeError("requires_grad parameter count is not 29,211")
    names = [name for name, p in net.named_parameters() if p.requires_grad]
    if any(not name.startswith("adapter_10tok.") or ".anchor." in name for name in names):
        raise RuntimeError("non-adapter or frozen-anchor parameter is trainable")
    return net, trainable, base.guard, checkpoint_state


def adapter_state(net) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in net.named_parameters()
        if parameter.requires_grad
    }


def adapter_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_adapter(net, state: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in net.named_parameters():
            if parameter.requires_grad:
                if name not in state or tuple(state[name].shape) != tuple(parameter.shape):
                    raise RuntimeError(f"adapter tensor inventory mismatch at {name}")
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def live_objective(prediction: torch.Tensor, pools: dict, cids: list[str], indexes: list[np.ndarray]) -> torch.Tensor:
    target_np = np.stack([pools[cid]["fitness"][ix] for cid, ix in zip(cids, indexes, strict=True)])
    gate_np = np.stack([pools[cid]["gate"][ix] for cid, ix in zip(cids, indexes, strict=True)])
    target = torch.as_tensor(target_np, dtype=torch.float32, device=prediction.device)
    gates = torch.as_tensor(gate_np, dtype=torch.bool, device=prediction.device)
    target = torch.where(gates, target, torch.full_like(target, 2.0))
    row_huber = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    row_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(-prediction / 0.2, dim=-1),
        torch.nn.functional.softmax(-target / 0.2, dim=-1), reduction="none",
    ).sum(dim=1)
    return (row_huber + 0.1 * row_kl).mean()


def prepare_training(input_dir: Path, checkpoint: Path, device: torch.device):
    h48 = import_module("h55_h48", H48_PATH)
    h48.CHECKPOINT_SHA = R882_SHA
    selected, train_rows, dev_rows = input_metadata(input_dir, h48)
    folds = family_folds(train_rows, h48)
    fold_b_families = folds[4]
    fold_a_families = set().union(*folds[:4])
    if fold_a_families & fold_b_families:
        raise RuntimeError("fold A/B family overlap")
    a_rows = [r for r in train_rows if r["family_id"] in fold_a_families]
    b_rows = [r for r in train_rows if r["family_id"] in fold_b_families]
    if len(a_rows) + len(b_rows) != len(train_rows) or not a_rows or not b_rows:
        raise RuntimeError("seed-4900 fold assignment coverage mismatch")
    pools_a, tables_a = load_rows(input_dir, a_rows, h48)
    net, trainable, guard, checkpoint_state = build_model(input_dir, checkpoint, device, h48)
    tokens_a, masks_a = h48.h46.make_tokens(net, tables_a, device)
    predictor_a = h48.make_predictor(net, pools_a, tokens_a, masks_a, device)
    # Exact-semantics speedup copied from H53b; prove equivalence on the A-only roster.
    h53b = import_module("h55_h53b_runner", H53B_PATH)
    training = import_module("h55_training_indices", TRAINING_PATH)
    dev = import_module("h55_dev_incumbent", DEV_PATH)
    sorted_orders = {cid: np.argsort(pool["fitness"]) for cid, pool in pools_a.items()}
    saved_state = np.random.get_state()
    np.random.seed(61353)
    probe_ids = sorted(pools_a)[:min(16, len(pools_a))]
    reference = [training.training_indices(pools_a[cid], SLATE_SIZE) for cid in probe_ids]
    np.random.seed(61353)
    cached = [h53b.cached_training_indices(pools_a[cid], sorted_orders[cid], SLATE_SIZE) for cid in probe_ids]
    np.random.set_state(saved_state)
    if any(not np.array_equal(x, y) for x, y in zip(reference, cached, strict=True)):
        raise RuntimeError("cached H53b sampler differs from the pinned live sampler")
    family_by_cid = {r["context_sha256"]: r["family_id"] for r in a_rows}
    all_families = {r["family_id"] for r in train_rows}
    fold_assignment = {
        "seed": FOLD_SEED,
        "folds": [sorted(f) for f in folds],
        "fold_a_families": sorted(fold_a_families),
        "fold_b_families": sorted(fold_b_families),
        "train_contexts_a": len(a_rows),
        "train_contexts_b": len(b_rows),
    }
    fold_sha = canonical_sha(fold_assignment)
    return {
        "h48": h48, "selected": selected, "a_rows": a_rows, "b_rows": b_rows,
        "dev_rows": dev_rows, "folds": folds, "fold_sha": fold_sha,
        "fold_assignment": fold_assignment, "fold_a_families": fold_a_families,
        "fold_b_families": fold_b_families, "all_families": all_families,
        "pools_a": pools_a, "tables_a": tables_a, "tokens_a": tokens_a,
        "masks_a": masks_a, "predictor_a": predictor_a, "family_by_cid": family_by_cid,
        "net": net, "trainable": trainable, "guard": guard,
        "checkpoint_state": checkpoint_state, "sorted_orders": sorted_orders,
        "dev_module": dev, "h53b": h53b,
        "source_sha256": source_hashes(input_dir),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }


def run_update_unit(
    ctx: dict,
    seed: int,
    unit: int,
    sampler_key: int,
    optimizer: torch.optim.Optimizer | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> dict:
    h48 = ctx["h48"]
    net, trainable = ctx["net"], ctx["trainable"]
    pools = ctx["pools_a"]
    train_ids = sorted(pools)
    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    np.random.seed((sampler_key + unit * 104729) % (2**32))
    grad_norms = []
    step_seconds = []
    rotation = unit % len(train_ids)
    round_train = train_ids[rotation:] + train_ids[:rotation]
    for step in range(K_STEPS):
        step_seed = ctx["dev_module"].batch_seed(sampler_key, unit, step)
        cids = random.Random(step_seed).choices(round_train, k=BATCH_SIZE)
        indexes = [
            ctx["h53b"].cached_training_indices(pools[cid], ctx["sorted_orders"][cid], SLATE_SIZE)
            for cid in cids
        ]
        optimizer.zero_grad(set_to_none=True)
        start = time.monotonic()
        loss = live_objective(ctx["predictor_a"](cids, indexes), pools, cids, indexes)
        loss.backward()
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(trainable, 1.0)))
        optimizer.step()
        step_seconds.append(time.monotonic() - start)
        if ema_state is not None:
            with torch.no_grad():
                for name, parameter in net.named_parameters():
                    if parameter.requires_grad:
                        ema_state[name].mul_(EMA_DECAY).add_(parameter.detach().cpu(), alpha=1.0 - EMA_DECAY)
        if h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA) != DOPE_SHA:
            raise RuntimeError("DOPE changed during H55 update")
    return {"optimizer": optimizer, "grad_norms": grad_norms, "step_seconds": step_seconds}


def train_swa(ctx: dict, seed: int, timing_only: bool) -> tuple[dict, dict]:
    if timing_only:
        raise ValueError("SWA timing pilot is not preregistered; use EMA-25 timing cell")
    net, checkpoint_state = ctx["net"], ctx["checkpoint_state"]
    member_states = []
    member_timings = []
    for member in range(SWA_MEMBERS):
        net.load_state_dict(checkpoint_state["state_dict"], strict=True)
        if ctx["h48"].h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA) != DOPE_SHA:
            raise RuntimeError("DOPE mismatch before SWA member")
        sampler_key = seed * 1_000_003 + (member + 1) * 10_007
        start = time.monotonic()
        step = run_update_unit(ctx, seed, 1, sampler_key)
        member_timings.append({
            "member": member,
            "sampler_key": sampler_key,
            "optimizer_step_seconds": step["step_seconds"],
            "gradient_norms": step["grad_norms"],
            "wall_seconds": time.monotonic() - start,
        })
        member_states.append(adapter_state(net))
    names = set(member_states[0])
    if any(set(state) != names for state in member_states):
        raise RuntimeError("SWA member adapter tensor inventories differ")
    averaged = {
        name: torch.stack([state[name].to(torch.float64) for state in member_states]).mean(0).to(member_states[0][name].dtype)
        for name in sorted(names)
    }
    return averaged, {"members": member_timings, "member_states": member_states}


def run_ema_trajectory(ctx: dict, seed: int) -> tuple[dict, dict]:
    """Run the fixed 25-round trajectory and update the EMA after each optimizer step."""
    h48 = ctx["h48"]
    net, trainable, checkpoint_state = ctx["net"], ctx["trainable"], ctx["checkpoint_state"]
    net.load_state_dict(checkpoint_state["state_dict"], strict=True)
    raw_initial = adapter_state(net)
    ema = {name: value.clone() for name, value in raw_initial.items()}
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    rounds = []
    for round_index in range(1, EMA_ROUNDS + 1):
        round_start = time.monotonic()
        step = run_update_unit(
            ctx, seed, round_index, seed,
            optimizer=optimizer, ema_state=ema,
        )
        rounds.append({"round": round_index, "optimizer_step_seconds": step["step_seconds"],
                       "gradient_norms": step["grad_norms"], "wall_seconds": time.monotonic() - round_start})
    raw_final = adapter_state(net)
    return ema, {"raw_final": raw_final, "ema_final": ema, "rounds": rounds}


def bootstrap_family_gains(gains: dict[str, float], seed: int) -> dict:
    families = sorted(gains)
    values = np.asarray([gains[f] for f in families], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(10_000, len(values)))
    mean_boot = values[indexes].mean(axis=1)
    median_boot = np.median(values[indexes], axis=1)
    return {
        "family_count": len(values),
        "mean_gain": float(values.mean()),
        "mean_lcb_one_sided_95": float(np.quantile(mean_boot, 0.05)),
        "median_gain": float(np.median(values)),
        "median_ci95": [float(np.quantile(median_boot, 0.025)), float(np.quantile(median_boot, 0.975))],
        "draws": 10_000,
    }


def score_fold_b(ctx: dict, candidate_states: list[dict[str, torch.Tensor]], admission_state: dict[str, torch.Tensor]) -> dict:
    """Load and score fold B only after all A-only training is complete."""
    h48, net = ctx["h48"], ctx["net"]
    if set(ctx["fold_a_families"]) & set(ctx["fold_b_families"]):
        raise RuntimeError("fold A/B overlap before admission scoring")
    # Drop A payloads/tables before loading B to keep RSS bounded.
    del ctx["predictor_a"], ctx["pools_a"], ctx["tables_a"], ctx["tokens_a"], ctx["masks_a"], ctx["sorted_orders"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pools_b, tables_b = load_rows(ctx["input_dir"], ctx["b_rows"], h48)
    device = next(net.parameters()).device
    tokens_b, masks_b = h48.h46.make_tokens(net, tables_b, device)
    predict_b = h48.make_predictor(net, pools_b, tokens_b, masks_b, device)
    ids_b = [r["context_sha256"] for r in ctx["b_rows"]]
    families_b = {r["context_sha256"]: r["family_id"] for r in ctx["b_rows"]}
    # The baseline and each permitted candidate/control are each scored once on B.
    net.load_state_dict(ctx["checkpoint_state"]["state_dict"], strict=True)
    h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    net.eval()
    baseline = h48.evaluate_utility(ids_b, pools_b, predict_b)
    scored_candidates = []
    for state in candidate_states:
        set_adapter(net, state)
        net.eval()
        scored_candidates.append(h48.evaluate_utility(ids_b, pools_b, predict_b))
        h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    set_adapter(net, admission_state)
    net.eval()
    admission = h48.evaluate_utility(ids_b, pools_b, predict_b)
    h48.h46.checkpoint_guard(ctx["guard"], net, expected=DOPE_SHA)
    return {
        "baseline": baseline,
        "admission": admission,
        "candidate_controls": scored_candidates,
        "families": families_b,
        "fold_b_context_ids": ids_b,
        "fold_b_scoring_passes": 2 + len(candidate_states),
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
    }


def elo_diagnostic(families: dict[str, str], baseline: dict, candidate: dict) -> dict:
    elo = import_module("h55_elo", ROOT / "python/src/jope/north_star/elo_tracker.py")
    records = [
        elo.DatasetEvalRecord(
            dataset_id=cid, family_id=families[cid], task_type="regression",
            scores_by_contender={"r882": float(baseline["per_context"][cid]), "H55": float(candidate["per_context"][cid])},
        )
        for cid in sorted(families)
    ]
    return elo.compute_family_grouped_bradley_terry_elo(records, anchor_contender="r882", anchor_elo=1500.0)


def train_and_score(args) -> dict:
    if args.seed not in SEEDS or args.arm not in ("swa8", "ema25"):
        raise RuntimeError("unregistered H55 arm or seed")
    if args.device != "cuda:0" or not args.lock_held:
        raise RuntimeError("outcome/timing cells require cuda:0 with qualification.lock held")
    if args.mode == "timing" and (args.arm, args.seed) != ("ema25", 12):
        raise RuntimeError("only EMA-25 seed12 is preregistered as timing-only")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    ctx = prepare_training(args.input, args.checkpoint, torch.device(args.device))
    ctx["input_dir"] = args.input
    dope_before = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if args.mode == "timing":
        if args.arm == "ema25":
            _, timing = run_ema_trajectory(ctx, args.seed)
        else:
            raise RuntimeError("timing mode only supports EMA-25")
        dope_after = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
        payload = {
            "schema": "jope.h55.timing/v1", "status": "complete", "arm": "ema25", "seed": 12,
            "rounds_completed": EMA_ROUNDS, "optimizer_updates_completed": EMA_ROUNDS * K_STEPS,
            "timing_seconds_by_round": [r["wall_seconds"] for r in timing["rounds"]],
            "wall_seconds": time.monotonic() - started, "device": "cuda:0", "fold_seed": FOLD_SEED,
            "fold_assignment_sha256": ctx["fold_sha"], "fold_b_scored": False, "dl_val_reads": 0,
            "dope_sha256_before": dope_before, "dope_sha256_after": dope_after,
            "source_sha256": ctx["source_sha256"], "source_commit": ctx["source_commit"],
            "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
        }
        atomic_json(args.out / "timing.json", payload)
        return payload

    if args.arm == "swa8":
        admission_state, training = train_swa(ctx, args.seed, False)
        member_states = training.pop("member_states")
        candidate_states = member_states
        training_report = training
    else:
        admission_state, training = run_ema_trajectory(ctx, args.seed)
        candidate_states = [training["raw_final"]]
        training_report = {"rounds": training["rounds"]}
    # Persist the final registered candidate before fold B opens.
    out_artifact = args.out / "candidate_adapter.pt"
    ctx["h48"].save_adapter_snapshot(out_artifact, admission_state, R882_SHA)
    scored = score_fold_b(ctx, candidate_states, admission_state)
    baseline = scored["baseline"]
    candidate = scored["admission"]
    gains = {f: float(candidate["per_family"][f]) - float(baseline["per_family"][f]) for f in baseline["per_family"]}
    if set(gains) != set(scored["baseline"]["per_family"]):
        raise RuntimeError("fold-B candidate/control family identities differ")
    positive = sum(v > 0 for v in gains.values())
    ties = sum(v == 0 for v in gains.values())
    negative = sum(v < 0 for v in gains.values())
    stats = bootstrap_family_gains(gains, 55_000 + args.seed + (0 if args.arm == "swa8" else 100))
    payload = {
        "schema": "jope.h55.cell/v1", "status": "complete", "mode": "outcome",
        "arm": args.arm, "seed": args.seed, "device": "cuda:0",
        "candidate_definition": "uniform mean of eight independent k=2 adapter updates" if args.arm == "swa8" else "EMA-0.9 adapter after exactly 25 rounds; final EMA only",
        "rounds_completed": 1 if args.arm == "swa8" else EMA_ROUNDS,
        "optimizer_updates_completed": SWA_MEMBERS * K_STEPS if args.arm == "swa8" else EMA_ROUNDS * K_STEPS,
        "swa_member_count": SWA_MEMBERS if args.arm == "swa8" else 0,
        "ema_decay": EMA_DECAY if args.arm == "ema25" else None,
        "ema_scored_final_only": args.arm == "ema25",
        "round_selection_used": False,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "k_steps_per_update": K_STEPS,
        "batch_size": BATCH_SIZE, "slate_size": SLATE_SIZE,
        "fold_seed": FOLD_SEED, "fold_assignment": ctx["fold_assignment"],
        "fold_assignment_sha256": ctx["fold_sha"],
        "fold_a_family_count": len(ctx["fold_a_families"]), "fold_b_family_count": len(ctx["fold_b_families"]),
        "fold_a_context_count": len(ctx["a_rows"]), "fold_b_context_count": len(ctx["b_rows"]),
        "fold_a_b_family_disjoint": not bool(ctx["fold_a_families"] & ctx["fold_b_families"]),
        "fold_b_scoring_passes": scored["fold_b_scoring_passes"],
        "r882_fold_b_family_utility": baseline["mean"],
        "candidate_fold_b_family_utility": candidate["mean"],
        "r882_fold_b_family_utility_by_id": baseline["per_family"],
        "candidate_fold_b_family_utility_by_id": candidate["per_family"],
        "family_gain_vs_r882": gains, "family_bootstrap": stats,
        "w_t_l_family": {"wins": positive, "ties": ties, "losses": negative},
        "elo_vs_r882_diagnostic": elo_diagnostic(scored["families"], baseline, candidate),
        "worst_five_family_gains": sorted(gains.items(), key=lambda x: x[1])[:5],
        "matched_control": ({"type": "mean of eight single-candidate fold-B utilities",
                             "member_family_utilities": [r["mean"] for r in scored["candidate_controls"]],
                             "mean_family_utility": float(np.mean([r["mean"] for r in scored["candidate_controls"]]))}
                            if args.arm == "swa8" else
                            {"type": "raw final iterate at round 25", "family_utility": scored["candidate_controls"][0]["mean"]}),
        "candidate_adapter_sha256": adapter_digest(admission_state),
        "candidate_adapter_artifact": str(out_artifact),
        "source_sha256": ctx["source_sha256"], "source_commit": ctx["source_commit"],
        "r882_checkpoint_sha256": R882_SHA, "source_manifest_sha256": MANIFEST_SHA,
        "manifest_identity": MANIFEST_ID, "dope_sha256_before": dope_before,
        "dope_sha256_after": ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA),
        "trainable_parameter_count": sum(p.numel() for p in ctx["trainable"]),
        "requires_grad_parameter_count": sum(p.numel() for p in ctx["net"].parameters() if p.requires_grad),
        "training_timing": training_report,
        "wall_seconds": time.monotonic() - started,
        "validation_read_count": 0, "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
    }
    if payload["dope_sha256_before"] != payload["dope_sha256_after"]:
        raise RuntimeError("DOPE changed across H55 cell")
    atomic_json(args.out / "summary.json", payload)
    return payload


def run_dry(args) -> dict:
    torch.set_num_threads(1)
    ctx = prepare_training(args.input, args.checkpoint, torch.device("cpu"))
    cids = sorted(ctx["pools_a"])[:2]
    indexes = [np.arange(min(SLATE_SIZE, len(ctx["pools_a"][cid]["pids"]))) for cid in cids]
    with torch.no_grad():
        output = ctx["predictor_a"](cids, indexes)
    if tuple(output.shape) != (2, min(SLATE_SIZE, min(len(ctx["pools_a"][cid]["pids"]) for cid in cids))):
        raise RuntimeError("two-context dry-run output shape mismatch")
    result = {"dry_run": "pass", "contexts": 2, "fold_seed": FOLD_SEED,
              "fold_assignment_sha256": ctx["fold_sha"], "fold_b_scored": False,
              "dl_val_read": False, "optimizer_updates": 0,
              "trainable_parameter_count": sum(p.numel() for p in ctx["trainable"]),
              "source_sha256": ctx["source_sha256"], "source_commit": ctx["source_commit"],
              "dope_sha256": ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)}
    print(json.dumps(result, sort_keys=True))
    return result


def run_terminal_val(args) -> dict:
    admission = json.loads(args.admission.read_text())
    arm_result = admission.get("arms", {}).get(args.arm, {})
    if arm_result.get("admitted") is not True:
        raise RuntimeError("terminal DL-Val is forbidden unless this arm admitted on fold B")
    if args.seed not in SEEDS or args.arm not in ("swa8", "ema25"):
        raise RuntimeError("unregistered terminal validation arm/seed")
    if args.out.exists():
        raise RuntimeError("refusing a duplicate terminal DL-Val score")
    if not args.lock_held or args.device != "cuda:0":
        raise RuntimeError("terminal DL-Val requires the CUDA qualification lock")
    h48 = import_module("h55_val_h48", H48_PATH)
    h48.CHECKPOINT_SHA = R882_SHA
    h49 = import_module("h55_h49_score", H49_SCORE_PATH)
    net, trainable, guard, checkpoint_state = build_model(args.input, args.checkpoint, torch.device(args.device), h48)
    dev_rows, pools, tables = h49.load_validation_packet(args.input)
    if len(dev_rows) != 91 or len({r["family_id"] for r in dev_rows}) != 19:
        raise RuntimeError("terminal DL-Val inventory mismatch")
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device(args.device))
    predictor = h48.make_predictor(net, pools, tokens, masks, torch.device(args.device))
    artifact = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if artifact.get("base_checkpoint_sha256") != R882_SHA or artifact.get("dope_sha256") != DOPE_SHA or artifact.get("trainable_params") != TRAINABLE_PARAMS:
        raise RuntimeError("terminal adapter artifact identity mismatch")
    set_adapter(net, artifact["adapter_trainable_state"])
    net.eval()
    ids = [r["context_sha256"] for r in dev_rows]
    result = h48.evaluate_utility(ids, pools, predictor)
    if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
        raise RuntimeError("DOPE changed during terminal validation")
    payload = {
        "schema": "jope.h55.terminal-val/v1", "arm": args.arm, "seed": args.seed,
        "terminal_dl_val_score_count": 1, "validation_payloads_opened": True,
        "family_utility": result["mean"], "context_utility": result["context_mean"],
        "family_delta_vs_r882_harness": result["mean"] - R882_HARNESS_UTILITY,
        "family_utilities": result["per_family"], "context_utilities": result["per_context"],
        "r882_harness_family_utility": R882_HARNESS_UTILITY,
        "candidate_adapter_sha256": sha256(args.adapter), "dope_sha256_before": DOPE_SHA,
        "dope_sha256_after": h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA),
        "source_sha256": source_hashes(args.input),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
    }
    atomic_json(args.out, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "timing", "outcome", "terminal-val"), required=True)
    parser.add_argument("--arm", choices=("swa8", "ema25"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--checkpoint", type=Path, default=Path.home() / "jope-data/h51/ckpt/round-882.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()
    if args.mode == "dry-run":
        if args.device != "cpu":
            raise RuntimeError("dry run is CPU-only and must not read outcomes")
        result = run_dry(args)
        atomic_json(args.out, result)
    elif args.mode == "terminal-val":
        if args.adapter is None or args.admission is None or args.arm is None or args.seed is None:
            raise RuntimeError("terminal validation requires arm, seed, adapter, and admission receipt")
        result = run_terminal_val(args)
        print(json.dumps({"arm": args.arm, "seed": args.seed, "terminal_dl_val_score_count": 1,
                          "family_utility": result["family_utility"]}, sort_keys=True), flush=True)
    else:
        if args.arm is None or args.seed is None:
            raise RuntimeError("timing/outcome modes require arm and seed")
        if args.out.exists():
            raise RuntimeError("refusing to overwrite a cell receipt directory")
        args.out.mkdir(parents=True, exist_ok=False)
        result = train_and_score(args)
        if args.mode == "timing":
            print(json.dumps({"mode": "timing", "rounds": result["rounds_completed"],
                              "wall_seconds": result["wall_seconds"], "fold_b_scored": False}, sort_keys=True), flush=True)
        else:
            atomic_json(args.out / "summary.json", result)
            print(json.dumps({"mode": "outcome", "arm": args.arm, "seed": args.seed,
                              "fold_b_mean_gain": result["family_bootstrap"]["mean_gain"],
                              "wall_seconds": result["wall_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
