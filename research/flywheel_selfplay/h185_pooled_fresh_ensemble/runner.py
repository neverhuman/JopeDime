"""One monitored H185 CPU attempt: power-aware 205-family pooled cross-fit confirmatory test of FRESH160, ENS5, and RENS5 vs R882."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

import analysis as stat
from treatment import (
    PARAMETERS,
    TENSORS,
    adapter_forward,
    adapter_forward_static,
    adapter_l2_drift,
    flatten_grad_with_none_as_zero,
    grad_norms,
    precompute_static_slate,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO
H185_INPUTS = Path("/home/ubuntu/jope-data/h185/inputs")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            p = REPO / path if kind == "parent_sources" else Path(path)
            if sha(p) != digest:
                raise RuntimeError(f"{kind} pin mismatch: {p}")
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def runtime():
    path = REPO / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h185_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt, provenance):
    packet = json.loads((rt.INPUT / "selected.json").read_text(encoding="utf-8"))
    train, val = packet["train"], packet["dev"]
    train_only = json.loads(
        (H185_INPUTS / "train_only_selected_1219.json").read_text(encoding="utf-8")
    )["train"]
    if train != train_only:
        raise RuntimeError("train_only_selected_1219 mismatch")
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
        or len({r["family_id"] for r in val}) != 19
        or len({r["context_sha256"] for r in train + val}) != 1310
    ):
        raise RuntimeError("cohort size or duplicate context")
    for rows, split in ((train, "DL-Train"), (val, "DL-Val")):
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected context or wrong split")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("TRAIN/DL-Val family overlap")
    ids = sorted({r["dataset_id"] for r in train + val})
    if ids != provenance["dataset_ids"] or provenance["real_catalog_joined"] != 276:
        raise RuntimeError("catalog provenance closure")
    if provenance["status"] != "PASS" or not provenance["zero_synthetic"]:
        raise RuntimeError("real-S3 source provenance")
    return train, val


class Engine:
    def __init__(self, net, rt, names, frozen, guard, fused):
        self.net, self.rt, self.names, self.frozen, self.guard = (
            net,
            rt,
            names,
            frozen,
            guard,
        )
        self.pools, self.cached, self.eval_batch_cache = {}, {}, {}
        self.tok, self.mask = {}, {}
        self.projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)

    def load_features(self, path, expected_rows):
        expected_cids = [r["context_sha256"] for r in expected_rows]
        with np.load(path, allow_pickle=False) as z:
            cids = list(map(str, z["cids"]))
            if cids != expected_cids:
                raise RuntimeError(f"feature cache cid order mismatch for {path}")
            tokens, masks = z["tokens_10x32"], z["mask_10"]
            for i, c in enumerate(cids):
                self.tok[c] = torch.as_tensor(tokens[i], dtype=torch.float32)
                self.mask[c] = torch.as_tensor(masks[i], dtype=torch.bool)

    def load_pools(self, rows):
        for r in rows:
            cid = r["context_sha256"]
            if cid in self.pools:
                continue
            p = self.rt.INPUT / "pools" / (cid + ".npz")
            if sha(p) != r["pool_sha256"]:
                raise RuntimeError("pool hash mismatch: " + cid)
            with np.load(p, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                self.pools[cid] = dict(
                    program=prog,
                    proj16=(prog.astype(np.float64) @ self.projection).astype(
                        np.float32
                    ),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

    def batch(self, cids, ixs):
        def stack(k):
            return np.stack(
                [self.pools[c][k][ix] for c, ix in zip(cids, ixs, strict=True)]
            )

        b, k = len(cids), len(ixs[0])
        fitness = torch.as_tensor(stack("fitness"), dtype=torch.float32)
        gate = torch.as_tensor(stack("gate"), dtype=torch.bool)
        kwargs = dict(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=torch.as_tensor(stack("program"), dtype=torch.float32),
            proj_16=torch.as_tensor(stack("proj16"), dtype=torch.float32),
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            revealed_indices=torch.arange(4).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, :4],
            revealed_gate=gate[:, :4].float(),
        )
        return (
            kwargs,
            torch.stack([self.tok[c] for c in cids]),
            torch.stack([self.mask[c] for c in cids]),
            fitness,
            gate,
        )

    def anchored(self, kwargs):
        with torch.no_grad():
            out = self.net.adapter_10tok.anchor.predict_costs(
                **kwargs,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
        return out["total_cost"].detach().float(), out["eff_prior"].detach().float()

    def forward(self, kwargs, tok, mask, anchor):
        return adapter_forward(
            self.net,
            kwargs["prog_412"],
            kwargs["sos_prior"],
            tok,
            mask,
            kwargs["scale"],
            kwargs["revealed_indices"],
            kwargs["revealed_raw_f"],
            kwargs["revealed_gate"],
            *anchor,
        )

    def check(self):
        self.rt.check_frozen(self.net, self.guard, self.names, self.frozen)
        if (
            self.net.dope_model.training
            or self.net.adapter_10tok.training
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")

    def prove_static_gradient_parity(self, batch_cids, params, initial):
        kwargs, tok, mask, fitness, gate = self.batch(
            batch_cids, [np.arange(24) for _ in batch_cids]
        )
        anchor = self.anchored(kwargs)
        static_slate = precompute_static_slate(
            self.net.adapter_10tok, kwargs["prog_412"], kwargs["sos_prior"], mask
        )
        gen = torch.Generator(device="cpu").manual_seed(185777)
        for mode in ("initial", "perturbed"):
            if mode == "perturbed":
                with torch.no_grad():
                    for p in params:
                        p.add_(
                            0.01
                            * torch.randn(
                                p.shape, generator=gen, dtype=p.dtype, device=p.device
                            )
                        )
            for p in params:
                p.grad = None
            pred_dyn = self.forward(kwargs, tok, mask, anchor)
            loss_dyn = self.rt.loss(pred_dyn, fitness, gate)
            loss_dyn.backward()
            grad_dyn, act_dyn, none_dyn = flatten_grad_with_none_as_zero(params)
            none_mask_dyn = [p.grad is None for p in params]

            for p in params:
                p.grad = None
            pred_stat = adapter_forward_static(
                self.net,
                tok,
                static_slate,
                kwargs["scale"],
                kwargs["revealed_indices"],
                kwargs["revealed_raw_f"],
                kwargs["revealed_gate"],
                *anchor,
            )
            loss_stat = self.rt.loss(pred_stat, fitness, gate)
            loss_stat.backward()
            grad_stat, act_stat, none_stat = flatten_grad_with_none_as_zero(params)
            none_mask_stat = [p.grad is None for p in params]

            if (
                not torch.equal(pred_dyn.detach(), pred_stat.detach())
                or not torch.equal(grad_dyn, grad_stat)
                or none_mask_dyn != none_mask_stat
                or act_dyn != 41
                or none_dyn != 6
                or act_stat != 41
                or none_stat != 6
            ):
                raise RuntimeError(
                    f"static vs dynamic forward/grad parity failed in mode={mode}"
                )
        with torch.no_grad():
            for n, p in self.net.named_parameters():
                if n in initial:
                    p.copy_(initial[n])
        for p in params:
            p.grad = None
        self.check()

    def clear_eval_cache(self, keep_cids=()):
        keep = set(keep_cids)
        self.eval_batch_cache = {
            k: v for k, v in self.eval_batch_cache.items() if all(c in keep for c in k)
        }

    def evaluate(self, rows, native=False, prove_identity=False, cache_batches=True):
        groups, result, raw = {}, {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = tuple(cids[start : start + 8])
                    if native:
                        kwargs, tok, mask, _, _ = self.batch(
                            batch, [np.arange(k) for _ in batch]
                        )
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tok, mm_mask_10=mask
                        )["total_cost"].float()
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        static_slate = precompute_static_slate(
                            self.net.adapter_10tok,
                            kwargs["prog_412"],
                            kwargs["sos_prior"],
                            mask,
                        )
                        fast_scores = adapter_forward_static(
                            self.net,
                            tok,
                            static_slate,
                            kwargs["scale"],
                            kwargs["revealed_indices"],
                            kwargs["revealed_raw_f"],
                            kwargs["revealed_gate"],
                            *anchor,
                        )
                        if prove_identity and (
                            not torch.equal(
                                scores, self.forward(kwargs, tok, mask, anchor)
                            )
                            or not torch.equal(scores, fast_scores)
                        ):
                            raise RuntimeError(
                                "step0 eval/static forward differs from native r882 bits"
                            )
                        if cache_batches:
                            self.eval_batch_cache[batch] = (
                                tok,
                                static_slate,
                                kwargs["scale"],
                                kwargs["revealed_indices"],
                                kwargs["revealed_raw_f"],
                                kwargs["revealed_gate"],
                                anchor[0],
                                anchor[1],
                            )
                    else:
                        if batch not in self.eval_batch_cache:
                            kwargs, tok, mask, _, _ = self.batch(
                                batch, [np.arange(k) for _ in batch]
                            )
                            missing = [c for c in batch if c not in self.cached]
                            if missing:
                                anchor = self.anchored(kwargs)
                                for i, cid in enumerate(batch):
                                    self.cached[cid] = tuple(
                                        t[i].clone() for t in anchor
                                    )
                            anchor = tuple(
                                torch.stack([self.cached[c][j] for c in batch])
                                for j in range(2)
                            )
                            static_slate = precompute_static_slate(
                                self.net.adapter_10tok,
                                kwargs["prog_412"],
                                kwargs["sos_prior"],
                                mask,
                            )
                            entry = (
                                tok,
                                static_slate,
                                kwargs["scale"],
                                kwargs["revealed_indices"],
                                kwargs["revealed_raw_f"],
                                kwargs["revealed_gate"],
                                anchor[0],
                                anchor[1],
                            )
                            if cache_batches:
                                self.eval_batch_cache[batch] = entry
                        else:
                            entry = self.eval_batch_cache[batch]
                        scores = adapter_forward_static(self.net, *entry)
                    costs = scores.numpy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite prediction")
                    for cid, cost in zip(batch, costs, strict=True):
                        choice = int(np.argmin(cost))
                        pool = self.pools[cid]
                        result[cid] = dict(
                            choice=choice,
                            utility=float(
                                -pool["fitness"][choice]
                                if pool["gate"][choice]
                                else -2.0
                            ),
                            invalid=not bool(pool["gate"][choice]),
                        )
                        raw[cid] = cost.copy()
        self.check()
        return dict(
            context_ids=[r["context_sha256"] for r in rows],
            values=[result[r["context_sha256"]]["utility"] for r in rows],
            choices=[result[r["context_sha256"]]["choice"] for r in rows],
            invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        ), raw


def evaluate_score_ensemble(engine, rows, member_raw_list, k=5):
    """Average per-candidate cost vectors across k members before argmin and return (values, choices, invalid)."""
    if len(member_raw_list) != k:
        raise ValueError(
            f"expected {k} member raw score dicts, got {len(member_raw_list)}"
        )
    values, choices, invalid = [], [], []
    for r in rows:
        cid = r["context_sha256"]
        stacked = np.stack([member_raw_list[m][cid] for m in range(k)], axis=0)
        avg_cost = stat.score_ensemble(stacked, k=k)
        choice = int(np.argmin(avg_cost))
        pool = engine.pools[cid]
        is_valid = bool(pool["gate"][choice])
        util = float(-pool["fitness"][choice] if is_valid else -2.0)
        values.append(util)
        choices.append(choice)
        invalid.append(not is_valid)
    return dict(
        context_ids=[r["context_sha256"] for r in rows],
        values=values,
        choices=choices,
        invalid=invalid,
    )


def adapter_state(net, names):
    return {name: net.state_dict()[name].detach().cpu().clone() for name in names}


def restore(net, names, state):
    with torch.no_grad():
        params = dict(net.named_parameters())
        for name in names:
            params[name].copy_(state[name])


def random_adapter_state(net, names, seed):
    torch.manual_seed(int(seed))
    old = net.adapter_10tok
    fresh = (
        type(old)(
            frozen_anchor=old.anchor,
            prog_mean=old.prog_mean.detach().cpu().numpy(),
            prog_std=old.prog_std.detach().cpu().numpy(),
            d_token=old.d_token,
            n_modalities=old.n_modalities,
            modality_dropout_p=old.modality_dropout_p,
        )
        .cpu()
        .eval()
    )
    params = dict(fresh.named_parameters())
    state = {}
    for name in names:
        suffix = name.removeprefix("adapter_10tok.")
        if suffix not in params or not params[suffix].requires_grad:
            raise RuntimeError("fresh adapter trainable inventory mismatch")
        state[name] = params[suffix].detach().cpu().clone()
    if len(state) != TENSORS or sum(t.numel() for t in state.values()) != PARAMETERS:
        raise RuntimeError("fresh adapter must have 47 tensors/29,211 parameters")
    del fresh
    return state


def run_cell(
    net,
    rt,
    engine,
    names,
    frozen,
    guard,
    state,
    fit_rows,
    excluded_cids,
    seed,
    updates,
    plan=None,
    optimizer=None,
    start_step=1,
):
    fit_cid_set = {r["context_sha256"] for r in fit_rows}
    if fit_cid_set & set(excluded_cids):
        raise RuntimeError("FIT context set overlaps B/A/DL-Val excluded contexts")
    if optimizer is None:
        if state is None:
            raise ValueError("initial state required for a new trajectory")
        restore(net, names, state)
        params = [p for p in net.parameters() if p.requires_grad]
        if len(params) != TENSORS or sum(p.numel() for p in params) != PARAMETERS:
            raise RuntimeError("optimizer parameter inventory mismatch")
        optimizer = torch.optim.AdamW(
            params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
        )
    else:
        params = [p for p in net.parameters() if p.requires_grad]
        if state is not None:
            raise ValueError("continuation must retain in-memory optimizer and weights")
    if plan is None:
        cids = sorted(fit_cid_set)
        orders = {cid: np.argsort(engine.pools[cid]["fitness"]) for cid in cids}
        plan = stat.batch_plan(cids, orders, seed, updates=updates)
    if len(plan) < updates:
        raise RuntimeError("batch plan shorter than requested update budget")

    traces, times = [], []
    for step in range(start_step, updates + 1):
        stream, cids, indexes = plan[step - 1]
        if not set(cids) <= fit_cid_set or (set(cids) & set(excluded_cids)):
            raise RuntimeError("FIT-only batch leak detected")
        t0 = time.monotonic()
        kwargs, tok, mask, fitness, gate = engine.batch(cids, indexes)
        anchor = engine.anchored(kwargs)
        torch.manual_seed(stream)
        optimizer.zero_grad(set_to_none=True)
        prediction = engine.forward(kwargs, tok, mask, anchor)
        objective = rt.loss(prediction, fitness, gate)
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite objective")
        objective.backward()
        _, gnorms = grad_norms(params, max_norm=1.0, expected_numel=PARAMETERS)
        if gnorms["active_grad_tensors"] != 41 or gnorms["none_grad_tensors"] != 6:
            raise RuntimeError("active/None grad tensor count mismatch")
        optimizer.step()
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        engine.check()
        traces.append(
            dict(
                step=step,
                stream=stream,
                fit_context_ids=cids,
                loss=float(objective.detach()),
                raw_grad_l2=gnorms["raw_grad_l2"],
                clipped_grad_l2=gnorms["clipped_grad_l2"],
                active_grad_tensors=gnorms["active_grad_tensors"],
                none_grad_tensors=gnorms["none_grad_tensors"],
                learning_rate=1e-4,
                elapsed_seconds=elapsed,
            )
        )
    return dict(
        state=adapter_state(net, names),
        traces=traces,
        elapsed_update_seconds=times,
        optimizer=optimizer,
        batch_plan=plan,
    )


def save_state_npz(path, state):
    np.savez_compressed(path, **{n: t.numpy() for n, t in state.items()})


def load_state_npz(path, names):
    with np.load(path, allow_pickle=False) as z:
        return {n: torch.as_tensor(z[n]) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")

    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    verify(pins)

    h178_receipt_path = (
        REPO / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
    )
    h178_receipt = json.loads(h178_receipt_path.read_text(encoding="utf-8"))
    h177_analysis_path = (
        REPO / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
    )
    h177_analysis_sha = sha(h177_analysis_path)
    if h178_receipt.get("h177_analysis_sha256") != h177_analysis_sha:
        raise RuntimeError("h178_receipt h177_analysis_sha256 mismatch")

    launch = json.loads(args.launch.read_text(encoding="utf-8"))
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        sha(h178_receipt_path),
        smoke=args.smoke,
    )

    rt = runtime()
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE composite pin mismatch")
    train, val = cohorts(rt, provenance)
    coverage_proof = stat.verify_pooled_b_coverage(train, fold_seed=stat.FOLD_SEED)
    fold_manifest = json.loads(
        (HERE / "fold_manifest.json").read_text(encoding="utf-8")
    )
    if stat.folds(train, fold_seed=stat.FOLD_SEED) != fold_manifest["families"]:
        raise RuntimeError("fold_manifest families mismatch")

    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(args.launch.resolve()), str(h178_receipt_path.resolve())}
    )
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text(encoding="utf-8")) != launch:
        raise RuntimeError("launch mutation after audit hook installation")

    updates, started = 0, time.monotonic()
    stage_guard = stat.StageOrderGuard()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        initial = adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params) != PARAMETERS
            or len(names) != TENSORS
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
        ):
            raise RuntimeError("adapter/mode inventory mismatch")
        rt.check_frozen(net, guard, names, frozen)

        if args.smoke:
            # Verify distinct member inits and RENS5==ENS5 init identity on constructor in smoke mode
            smoke_member_hashes = {}
            for s in stat.SEEDS:
                m_seeds = stat.member_seeds_for(s)
                m_hashes = tuple(
                    rt.tensor_hash(random_adapter_state(net, names, ms))
                    for ms in m_seeds
                )
                f160_hash = rt.tensor_hash(
                    random_adapter_state(net, names, stat.fresh160_init_seed(s, 0))
                )
                stat.validate_cell_init_hashes(initial_hash, f160_hash, m_hashes)
                stat.validate_rens5_matches_ens5_inits(m_hashes, m_hashes)
                smoke_member_hashes[str(s)] = list(m_hashes)
            restore(net, names, initial)
            rt.check_frozen(net, guard, names, frozen)
            save(
                out / "smoke.json",
                dict(
                    wave="H185",
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=TENSORS,
                    trainable_parameters=PARAMETERS,
                    prespecified_candidate_step=stat.PRESPECIFIED_CANDIDATE_STEP,
                    pooled_b_coverage=coverage_proof,
                    h178_receipt_verified=True,
                    smoke_member_init_hashes=smoke_member_hashes,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    initial_adapter_sha256=initial_hash,
                ),
            )
            return

        ref = json.loads(Path(pins["reference"]).read_text(encoding="utf-8"))
        expected_contract = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            ref["evaluation_contract"]["contexts"] != expected_contract
            or hashlib.sha256(
                json.dumps(ref["evaluation_contract"], sort_keys=True).encode()
            ).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation contract mismatch")

        engine = Engine(net, rt, names, frozen, guard, fused)
        # Gate 0: Load ONLY DL-Val features and pools before exact 91/91 r882 baseline gate passes.
        engine.load_features(H185_INPUTS / "val_only_context_features_91.npz", val)
        engine.load_pools(val)
        val_cids = {r["context_sha256"] for r in val}
        baseline, costs = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91 r882 baseline mismatch")
        engine.prove_static_gradient_parity(
            [r["context_sha256"] for r in val[:8]], params, initial
        )
        if rt.tensor_hash(adapter_state(net, names)) != initial_hash:
            raise RuntimeError("adapter mutated during static gradient parity check")
        np.savez_compressed(out / "baseline_costs.npz", **costs)
        save(
            out / "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=1.1377941898164525,
                eval_forward_bit_exact=True,
                static_grad_parity_bit_exact=True,
                static_grad_parity_perturbed_bit_exact=True,
            ),
        )
        print(
            "BASELINE PASS 91/91; eval/static forward & grad parity BIT-EXACT",
            flush=True,
        )
        del costs

        engine.load_features(
            H185_INPUTS / "train_only_context_features_1219.npz", train
        )
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}

        # Stage 1: Train on FIT_k only per fold, score ONLY B_k at the single prespecified final step per arm,
        # and pool B across all 205 families BEFORE opening or scoring any A_k outcome.
        pooled_b_panel = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        pooled_b_choices = np.full((3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64)
        pooled_b_ens5_member_utils = np.full((3, 1219, 5), np.nan)
        pooled_b_ens5_member_choices = np.full((3, 1219, 5), -1, dtype=np.int64)
        pooled_b_rens5_member_utils = np.full((3, 1219, 5), np.nan)
        pooled_b_rens5_member_choices = np.full((3, 1219, 5), -1, dtype=np.int64)

        pilot_record = None
        execution = None
        fresh160_updates = stat.FRESH160_DEFAULT_UPDATES
        drifts_by_arm = {"FRESH160": [], "ENS5_member_mean": [], "RENS5": []}
        cell_stage1_records = []

        stage_guard.mark_b_read()
        for outer in range(5):
            fit_rows, b_rows, a_rows = stat.partition(
                train, outer, fold_seed=stat.FOLD_SEED
            )
            fit_cids = {r["context_sha256"] for r in fit_rows}
            b_cids = {r["context_sha256"] for r in b_rows}
            a_cids = {r["context_sha256"] for r in a_rows}
            excluded_cids = b_cids | a_cids | val_cids
            if (fit_cids & excluded_cids) or (b_cids & a_cids):
                raise RuntimeError("family/context partition leakage")

            # Load FIT_k pools first (before B_k pools on fold 0 so the 4-update timing pilot runs before any TRAIN evaluation).
            engine.load_pools(fit_rows)
            orders = {
                c: np.argsort(engine.pools[c]["fitness"]) for c in sorted(fit_cids)
            }
            b_indices = [positions[r["context_sha256"]] for r in b_rows]
            selbase = None

            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-seed{seed}"
                f160_init_seed = stat.fresh160_init_seed(seed, outer)
                f160_init_state = random_adapter_state(net, names, f160_init_seed)
                f160_init_hash = rt.tensor_hash(f160_init_state)

                m_seeds = stat.member_seeds_for(seed)
                m_init_states = [random_adapter_state(net, names, ms) for ms in m_seeds]
                m_init_hashes = tuple(rt.tensor_hash(st) for st in m_init_states)
                stat.validate_cell_init_hashes(
                    initial_hash, f160_init_hash, m_init_hashes
                )

                f160_batch_seed = stat.stream_seed(
                    seed, outer, "FRESH160", "h185-h156-batches-v1"
                )
                f160_plan = stat.batch_plan(
                    sorted(fit_cids),
                    orders,
                    f160_batch_seed,
                    updates=stat.FRESH160_DEFAULT_UPDATES,
                )

                # Timing pilot on the first 4 updates of (outer=0, seed=185101, FRESH160) BEFORE any B or A pool is scored
                if pilot_record is None:
                    if (outer, seed) != (0, stat.SEEDS[0]):
                        raise RuntimeError("timing pilot order changed")
                    pilot = run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        f160_init_state,
                        fit_rows,
                        excluded_cids,
                        f160_batch_seed,
                        stat.PILOT_UPDATES,
                        plan=f160_plan,
                    )
                    updates += stat.PILOT_UPDATES
                    sec_per_update = float(
                        np.median(pilot["elapsed_update_seconds"][1:])
                    )
                    execution = stat.execution_plan(sec_per_update)
                    fresh160_updates = int(execution["fresh160_updates"])
                    stage_guard = stat.StageOrderGuard(
                        {
                            "FRESH160": fresh160_updates,
                            "ENS5": stat.ENS5_MEMBER_UPDATES,
                            "RENS5": stat.RENS5_MEMBER_UPDATES,
                        }
                    )
                    stage_guard.mark_b_read()
                    pilot_record = dict(
                        elapsed_update_seconds=pilot["elapsed_update_seconds"],
                        seconds_per_update=sec_per_update,
                        **execution,
                        rule="timing only; first 4 updates of A0-seed185101 FRESH160 on FIT_0 before any B or A outcome is scored",
                    )
                    save(out / "pilot_budget.json", pilot_record)
                    tail = run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        None,
                        fit_rows,
                        excluded_cids,
                        f160_batch_seed,
                        fresh160_updates,
                        plan=f160_plan[:fresh160_updates],
                        optimizer=pilot["optimizer"],
                        start_step=stat.PILOT_UPDATES + 1,
                    )
                    updates += fresh160_updates - stat.PILOT_UPDATES
                    f160_cell = dict(
                        state=tail["state"],
                        traces=pilot["traces"] + tail["traces"],
                        elapsed_update_seconds=pilot["elapsed_update_seconds"]
                        + tail["elapsed_update_seconds"],
                        batch_plan=f160_plan[:fresh160_updates],
                    )
                else:
                    f160_cell = run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        f160_init_state,
                        fit_rows,
                        excluded_cids,
                        f160_batch_seed,
                        fresh160_updates,
                        plan=f160_plan[:fresh160_updates],
                    )
                    updates += fresh160_updates

                # Load B_outer pools now that pilot is locked; A_outer remains unloaded in Stage 1.
                if selbase is None:
                    engine.load_pools(b_rows)
                    engine.clear_eval_cache(keep_cids=val_cids | b_cids)
                    restore(net, names, initial)
                    selbase, _ = engine.evaluate(b_rows, cache_batches=True)

                pooled_b_panel[si, b_indices, 0] = selbase["values"]
                pooled_b_choices[si, b_indices, 0] = selbase["choices"]

                # Evaluate FRESH160 at its single prespecified step on B_outer
                if stage_guard.candidate_step_for("FRESH160") != fresh160_updates:
                    raise RuntimeError("FRESH160 candidate step mismatch")
                f160_final_hash = rt.tensor_hash(f160_cell["state"])
                f160_drift = adapter_l2_drift(
                    f160_cell["state"], f160_init_state, names
                )
                if f160_final_hash == f160_init_hash or f160_drift <= 0.0:
                    raise RuntimeError("FRESH160 trajectory failed to update")
                drifts_by_arm["FRESH160"].append(f160_drift)
                save_state_npz(
                    out / f"{cellname}-FRESH160-step{fresh160_updates}.npz",
                    f160_cell["state"],
                )
                restore(net, names, f160_cell["state"])
                f160_b_eval, _ = engine.evaluate(b_rows, cache_batches=True)
                pooled_b_panel[si, b_indices, 1] = f160_b_eval["values"]
                pooled_b_choices[si, b_indices, 1] = f160_b_eval["choices"]

                # Evaluate RENS5 (0 updates) on B_outer using the exact 5 member initial states,
                # then train each member for 40 updates on FIT_outer to form ENS5.
                rens5_b_raws = []
                rens5_hashes_checked = []
                ens5_b_raws = []
                ens5_final_hashes = []
                ens5_member_drifts = []
                ens5_member_records = []

                for mi, (m_seed, m_init) in enumerate(
                    zip(m_seeds, m_init_states, strict=True), 1
                ):
                    # RENS5 member mi at 0 updates
                    restore(net, names, m_init)
                    r_hash = rt.tensor_hash(adapter_state(net, names))
                    rens5_hashes_checked.append(r_hash)
                    save_state_npz(
                        out / f"{cellname}-RENS5-member{mi}-step0.npz", m_init
                    )
                    r_b_eval, r_b_raw = engine.evaluate(b_rows, cache_batches=True)
                    rens5_b_raws.append(r_b_raw)
                    pooled_b_rens5_member_utils[si, b_indices, mi - 1] = r_b_eval[
                        "values"
                    ]
                    pooled_b_rens5_member_choices[si, b_indices, mi - 1] = r_b_eval[
                        "choices"
                    ]

                    # Train ENS5 member mi for 40 updates on FIT_outer from the exact same m_init
                    m_batch_seed = stat.stream_seed(
                        seed,
                        outer,
                        f"MEMBER{mi}",
                        m_seed,
                        "h185-h156-batches-v1",
                    )
                    m_plan = stat.batch_plan(
                        sorted(fit_cids),
                        orders,
                        m_batch_seed,
                        updates=stat.ENS5_MEMBER_UPDATES,
                    )
                    m_cell = run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        m_init,
                        fit_rows,
                        excluded_cids,
                        m_batch_seed,
                        stat.ENS5_MEMBER_UPDATES,
                        plan=m_plan,
                    )
                    updates += stat.ENS5_MEMBER_UPDATES
                    m_final_hash = rt.tensor_hash(m_cell["state"])
                    m_drift = adapter_l2_drift(m_cell["state"], m_init, names)
                    if m_final_hash == r_hash or m_drift <= 0.0:
                        raise RuntimeError(f"ENS5 member {mi} failed to update")
                    ens5_final_hashes.append(m_final_hash)
                    ens5_member_drifts.append(m_drift)
                    save_state_npz(
                        out / f"{cellname}-ENS5-member{mi}-step40.npz",
                        m_cell["state"],
                    )
                    restore(net, names, m_cell["state"])
                    e_b_eval, e_b_raw = engine.evaluate(b_rows, cache_batches=True)
                    ens5_b_raws.append(e_b_raw)
                    pooled_b_ens5_member_utils[si, b_indices, mi - 1] = e_b_eval[
                        "values"
                    ]
                    pooled_b_ens5_member_choices[si, b_indices, mi - 1] = e_b_eval[
                        "choices"
                    ]
                    ens5_member_records.append(
                        dict(
                            member_index=mi,
                            member_seed=m_seed,
                            init_sha256=r_hash,
                            final_sha256=m_final_hash,
                            drift_l2=m_drift,
                            updates=stat.ENS5_MEMBER_UPDATES,
                        )
                    )

                stat.validate_rens5_matches_ens5_inits(
                    rens5_hashes_checked, m_init_hashes
                )
                if len(set(ens5_final_hashes)) != 5:
                    raise RuntimeError(
                        "ENS5 trained member states are not all distinct"
                    )

                ens5_b_eval = evaluate_score_ensemble(engine, b_rows, ens5_b_raws, k=5)
                rens5_b_eval = evaluate_score_ensemble(
                    engine, b_rows, rens5_b_raws, k=5
                )
                pooled_b_panel[si, b_indices, 2] = ens5_b_eval["values"]
                pooled_b_choices[si, b_indices, 2] = ens5_b_eval["choices"]
                pooled_b_panel[si, b_indices, 3] = rens5_b_eval["values"]
                pooled_b_choices[si, b_indices, 3] = rens5_b_eval["choices"]

                drifts_by_arm["ENS5_member_mean"].append(
                    float(np.mean(ens5_member_drifts))
                )
                drifts_by_arm["RENS5"].append(0.0)

                cell_rec = dict(
                    cell=cellname,
                    seed=seed,
                    outer_fold=outer,
                    b_fold=(outer + 1) % 5,
                    fit_families=len({r["family_id"] for r in fit_rows}),
                    fit_contexts=len(fit_rows),
                    b_families=len({r["family_id"] for r in b_rows}),
                    b_contexts=len(b_rows),
                    r882_sha256=initial_hash,
                    fresh160=dict(
                        init_seed=f160_init_seed,
                        init_sha256=f160_init_hash,
                        final_sha256=f160_final_hash,
                        updates=fresh160_updates,
                        drift_l2=f160_drift,
                    ),
                    ens5_members=ens5_member_records,
                    rens5_matches_ens5_inits=True,
                )
                cell_stage1_records.append(cell_rec)
                save(out / f"{cellname}.b_only.json", cell_rec)
                restore(net, names, initial)
                engine.check()
                print(
                    f"STAGE-1 B-ONLY {cellname} updates={updates}",
                    flush=True,
                )

        if not np.isfinite(pooled_b_panel).all():
            raise RuntimeError("incomplete pooled_b_panel")

        b_minus2 = {
            arm: [
                int(np.count_nonzero(pooled_b_panel[s, :, j] == -2.0)) for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }
        b_agree = {
            arm: [
                float(np.mean(pooled_b_choices[s, :, j] == pooled_b_choices[s, :, 0]))
                for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }
        pooled_b_decisions = stat.pooled_b_select_h185(
            train,
            pooled_b_panel[:, :, 0],
            {
                arm: pooled_b_panel[:, :, col]
                for col, arm in enumerate(stat.CANDIDATES, 1)
            },
            seed=stat.BOOT_SEED,
            prespecified_steps=stage_guard.expected_steps,
        )
        pooled_b_summary = stat.summarize_panel(
            train,
            pooled_b_panel,
            minus2_by_arm=b_minus2,
            agreement_by_arm=b_agree,
            seed=stat.BOOT_SEED,
        )
        lock_path = out / "pooled_b_selection_lock.json"
        save(
            lock_path,
            dict(
                wave="H185",
                pooled_b_families=205,
                pooled_b_contexts=1219,
                fold_seed=stat.FOLD_SEED,
                boot_seed=stat.BOOT_SEED,
                holm_multiplicity=stat.HOLM_MULTIPLICITY,
                prespecified_candidate_step=stage_guard.expected_steps,
                decisions=pooled_b_decisions,
                pooled_b_summary=pooled_b_summary,
                locked_before_a_confirmation=True,
            ),
        )
        stage_guard.mark_b_locked(lock_path)

        # Stage 2: Separate confirmation readout on A families (correlated cross-fold readout) and descriptive DL-Val.
        locked = stage_guard.require_b_locked()
        confirm_a_selected = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        confirm_a_uncond = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        confirm_a_choices_uncond = np.full(
            (3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64
        )
        confirm_a_choices_selected = np.full(
            (3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64
        )
        confirm_a_ens5_member_utils = np.full((3, 1219, 5), np.nan)
        confirm_a_ens5_member_choices = np.full((3, 1219, 5), -1, dtype=np.int64)
        confirm_a_rens5_member_utils = np.full((3, 1219, 5), np.nan)
        confirm_a_rens5_member_choices = np.full((3, 1219, 5), -1, dtype=np.int64)

        val_selected = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        val_uncond = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        val_choices_uncond = np.full((3, 5, 91, len(stat.ALL_ARMS)), -1, dtype=np.int64)

        for outer in range(5):
            _, _, a_rows = stat.partition(train, outer, fold_seed=stat.FOLD_SEED)
            a_cids = {r["context_sha256"] for r in a_rows}
            a_indices = [positions[r["context_sha256"]] for r in a_rows]
            engine.load_pools(a_rows)
            engine.clear_eval_cache(keep_cids=val_cids | a_cids)
            restore(net, names, initial)
            abase, _ = engine.evaluate(a_rows, cache_batches=True)

            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-seed{seed}"
                confirm_a_selected[si, a_indices, 0] = abase["values"]
                confirm_a_uncond[si, a_indices, 0] = abase["values"]
                confirm_a_choices_uncond[si, a_indices, 0] = abase["choices"]
                confirm_a_choices_selected[si, a_indices, 0] = abase["choices"]
                val_selected[si, outer, :, 0] = baseline["values"]
                val_uncond[si, outer, :, 0] = baseline["values"]
                val_choices_uncond[si, outer, :, 0] = baseline["choices"]

                # FRESH160 on A_outer and DL-Val
                f160_snap = load_state_npz(
                    out / f"{cellname}-FRESH160-step{fresh160_updates}.npz", names
                )
                restore(net, names, f160_snap)
                f160_a_eval, _ = engine.evaluate(a_rows, cache_batches=True)
                f160_v_eval, _ = engine.evaluate(val, cache_batches=True)

                # ENS5 and RENS5 on A_outer and DL-Val
                ens5_a_raws, ens5_v_raws = [], []
                rens5_a_raws, rens5_v_raws = [], []
                for mi in range(1, 6):
                    e_snap = load_state_npz(
                        out / f"{cellname}-ENS5-member{mi}-step40.npz", names
                    )
                    restore(net, names, e_snap)
                    e_a_eval, e_a_raw = engine.evaluate(a_rows, cache_batches=True)
                    _, e_v_raw = engine.evaluate(val, cache_batches=True)
                    ens5_a_raws.append(e_a_raw)
                    ens5_v_raws.append(e_v_raw)
                    confirm_a_ens5_member_utils[si, a_indices, mi - 1] = e_a_eval[
                        "values"
                    ]
                    confirm_a_ens5_member_choices[si, a_indices, mi - 1] = e_a_eval[
                        "choices"
                    ]

                    r_snap = load_state_npz(
                        out / f"{cellname}-RENS5-member{mi}-step0.npz", names
                    )
                    restore(net, names, r_snap)
                    r_a_eval, r_a_raw = engine.evaluate(a_rows, cache_batches=True)
                    _, r_v_raw = engine.evaluate(val, cache_batches=True)
                    rens5_a_raws.append(r_a_raw)
                    rens5_v_raws.append(r_v_raw)
                    confirm_a_rens5_member_utils[si, a_indices, mi - 1] = r_a_eval[
                        "values"
                    ]
                    confirm_a_rens5_member_choices[si, a_indices, mi - 1] = r_a_eval[
                        "choices"
                    ]

                ens5_a_eval = evaluate_score_ensemble(engine, a_rows, ens5_a_raws, k=5)
                ens5_v_eval = evaluate_score_ensemble(engine, val, ens5_v_raws, k=5)
                rens5_a_eval = evaluate_score_ensemble(
                    engine, a_rows, rens5_a_raws, k=5
                )
                rens5_v_eval = evaluate_score_ensemble(engine, val, rens5_v_raws, k=5)

                evals_by_arm = {
                    "FRESH160": (f160_a_eval, f160_v_eval),
                    "ENS5": (ens5_a_eval, ens5_v_eval),
                    "RENS5": (rens5_a_eval, rens5_v_eval),
                }
                for col, arm in enumerate(stat.CANDIDATES, 1):
                    a_ev, v_ev = evals_by_arm[arm]
                    confirm_a_uncond[si, a_indices, col] = a_ev["values"]
                    confirm_a_choices_uncond[si, a_indices, col] = a_ev["choices"]
                    val_uncond[si, outer, :, col] = v_ev["values"]
                    val_choices_uncond[si, outer, :, col] = v_ev["choices"]

                    admitted = bool(locked["decisions"][arm]["admitted"])
                    confirm_a_selected[si, a_indices, col] = (
                        a_ev["values"] if admitted else abase["values"]
                    )
                    confirm_a_choices_selected[si, a_indices, col] = (
                        a_ev["choices"] if admitted else abase["choices"]
                    )
                    val_selected[si, outer, :, col] = (
                        v_ev["values"] if admitted else baseline["values"]
                    )

                save(
                    out / f"{cellname}.a_confirm.json",
                    dict(
                        cell=cellname,
                        seed=seed,
                        outer_fold=outer,
                        a_families=len({r["family_id"] for r in a_rows}),
                        a_contexts=len(a_rows),
                    ),
                )

        restore(net, names, initial)
        engine.check()

        np.savez_compressed(
            out / "panels.npz",
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            val_context_ids=np.array([r["context_sha256"] for r in val]),
            val_family_ids=np.array([r["family_id"] for r in val]),
            seed_ids=np.array(stat.SEEDS),
            arm_names=np.array(stat.ALL_ARMS),
            pooled_b=pooled_b_panel,
            pooled_b_choices=pooled_b_choices,
            pooled_b_ens5_member_utils=pooled_b_ens5_member_utils,
            pooled_b_ens5_member_choices=pooled_b_ens5_member_choices,
            pooled_b_rens5_member_utils=pooled_b_rens5_member_utils,
            pooled_b_rens5_member_choices=pooled_b_rens5_member_choices,
            confirm_a_selected=confirm_a_selected,
            confirm_a_unconditional=confirm_a_uncond,
            confirm_a_choices_unconditional=confirm_a_choices_uncond,
            confirm_a_choices_selected=confirm_a_choices_selected,
            confirm_a_ens5_member_utils=confirm_a_ens5_member_utils,
            confirm_a_ens5_member_choices=confirm_a_ens5_member_choices,
            confirm_a_rens5_member_utils=confirm_a_rens5_member_utils,
            confirm_a_rens5_member_choices=confirm_a_rens5_member_choices,
            val_selected=val_selected,
            val_unconditional=val_uncond,
            val_choices_unconditional=val_choices_uncond,
        )

        a_uncond_minus2 = {
            arm: [
                int(np.count_nonzero(confirm_a_uncond[s, :, j] == -2.0))
                for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }
        a_uncond_agree = {
            arm: [
                float(
                    np.mean(
                        confirm_a_choices_uncond[s, :, j]
                        == confirm_a_choices_uncond[s, :, 0]
                    )
                )
                for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }
        a_sel_minus2 = {
            arm: [
                int(np.count_nonzero(confirm_a_selected[s, :, j] == -2.0))
                for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }
        a_sel_agree = {
            arm: [
                float(
                    np.mean(
                        confirm_a_choices_selected[s, :, j]
                        == confirm_a_choices_selected[s, :, 0]
                    )
                )
                for s in range(3)
            ]
            for j, arm in enumerate(stat.ALL_ARMS)
        }

        confirm_a_uncond_summary = stat.summarize_panel(
            train,
            confirm_a_uncond,
            minus2_by_arm=a_uncond_minus2,
            agreement_by_arm=a_uncond_agree,
            seed=stat.BOOT_SEED,
        )
        confirm_a_selected_summary = stat.summarize_panel(
            train,
            confirm_a_selected,
            minus2_by_arm=a_sel_minus2,
            agreement_by_arm=a_sel_agree,
            seed=stat.BOOT_SEED,
        )
        val_uncond_summary = stat.summarize_panel(
            val, val_uncond.mean(axis=1), seed=stat.BOOT_SEED
        )
        val_selected_summary = stat.summarize_panel(
            val, val_selected.mean(axis=1), seed=stat.BOOT_SEED
        )

        # Pairwise member choice disagreement rates on B and A (reproducible from panels.npz)
        def member_disagreement(choices_3d):
            rates = []
            for s in range(3):
                for c in range(choices_3d.shape[1]):
                    ch = choices_3d[s, c]
                    rates.append(
                        sum(ch[i] != ch[j] for i in range(5) for j in range(i + 1, 5))
                        / 10.0
                    )
            return float(np.mean(rates))

        verdict_info = stat.resolve_h185_verdict(
            pooled_b_decisions,
            confirm_a_uncond_summary,
            h178_receipt=h178_receipt,
            expected_h177_analysis_sha256=h177_analysis_sha,
        )

        audit_summary = dict(
            status="PASS",
            opened=sorted(audit.opened),
            refused=audit.refused,
            protected_reads=0,
            optimizer_updates=updates,
            wall_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            threads_at_end=torch.get_num_threads(),
            cuda_initialized=torch.cuda.is_initialized(),
        )
        save(out / "audit.json", audit_summary)

        metrics = dict(
            wave="H185",
            status="COMPLETE",
            arms=list(stat.ALL_ARMS),
            candidates=list(stat.CANDIDATES),
            holm_multiplicity=stat.HOLM_MULTIPLICITY,
            seeds=list(stat.SEEDS),
            member_seeds_by_seed={
                str(k): list(v) for k, v in stat.MEMBER_SEEDS_BY_SEED.items()
            },
            fold_seed=stat.FOLD_SEED,
            boot_seed=stat.BOOT_SEED,
            prespecified_candidate_step=stage_guard.expected_steps,
            pilot_budget=pilot_record,
            optimizer_updates=updates,
            trained_trajectories=15 * 6,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            static_grad_parity_bit_exact=True,
            static_grad_parity_perturbed_bit_exact=True,
            rens5_uses_exact_ens5_inits=True,
            pooled_b_coverage=coverage_proof,
            h178_calibration_receipt=h178_receipt,
            selection_caveat_disclosure=(
                "Candidate arms (FRESH160, ENS5, RENS5) were nominated after H179/H180 on the same 205 DL-Train families "
                "(evaluated here under fresh cyclic folds fold_seed=185140 and fresh seeds 185101-185103); Stage-2 pooled A "
                "is a correlated cross-fold confirmation readout over the same 205 families."
            ),
            pooled_b_selection=dict(
                decisions=pooled_b_decisions,
                summary=pooled_b_summary,
            ),
            confirmation_a_unconditional=confirm_a_uncond_summary,
            confirmation_a_selected=confirm_a_selected_summary,
            descriptive_val_unconditional=val_uncond_summary,
            descriptive_val_selected=val_selected_summary,
            member_pairwise_choice_disagreement=dict(
                pooled_b_ens5=member_disagreement(pooled_b_ens5_member_choices),
                pooled_b_rens5=member_disagreement(pooled_b_rens5_member_choices),
                confirm_a_ens5=member_disagreement(confirm_a_ens5_member_choices),
                confirm_a_rens5=member_disagreement(confirm_a_rens5_member_choices),
                independently_reproducible_from_panels_npz=True,
            ),
            l2_drift_summary={
                k: dict(
                    median=float(np.median(v)),
                    min=float(np.min(v)),
                    max=float(np.max(v)),
                )
                for k, v in drifts_by_arm.items()
            },
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            trainable_parameters=PARAMETERS,
            active_grad_tensors=41,
            none_grad_tensors=6,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            elapsed_seconds=time.monotonic() - started,
            **verdict_info,
        )
        save(out / "metrics.json", metrics)
        print("H185 COMPLETE verdict=" + metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "incomplete.json",
            dict(
                wave="H185",
                status="INCOMPLETE",
                optimizer_updates=updates,
                reason=repr(exc),
                refused=audit.refused,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
