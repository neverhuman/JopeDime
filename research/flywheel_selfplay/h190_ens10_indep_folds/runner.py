"""One monitored H190 CPU attempt on xbabe1: ENS10 vs RENS10 confirmation with independent fold seeds per outer seed (190141, 190142, 190143)."""

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
H190_INPUTS = Path("/home/ubuntu/lane_work/h190-agy/inputs")


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
    spec = importlib.util.spec_from_file_location("h190_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt, provenance):
    packet = json.loads((rt.INPUT / "selected.json").read_text(encoding="utf-8"))
    train, val = packet["train"], packet["dev"]
    train_only = json.loads(
        (H190_INPUTS / "train_only_selected_1219.json").read_text(encoding="utf-8")
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
        gen = torch.Generator(device="cpu").manual_seed(190777)
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


def evaluate_score_ensemble(engine, rows, member_raw_list, k=stat.DEFAULT_ENSEMBLE_K):
    """Average per-candidate cost vectors across k members before argmin and return (values, choices, invalid)."""
    if len(member_raw_list) < k:
        raise ValueError(
            f"expected at least {k} member raw score dicts, got {len(member_raw_list)}"
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
    host = os.uname().nodename.split(".")[0]
    allowed_hosts = ("xbabe1", "xbabe2") if args.smoke else ("xbabe1",)
    if (
        host not in allowed_hosts
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError(f"CPU {allowed_hosts} nice19 required, got host={host}")

    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    verify(pins)
    stat.validate_member_seed_disjointness()

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
    coverage_proof = stat.verify_independent_fold_partitions(train)
    fold_manifest = json.loads(
        (HERE / "fold_manifest.json").read_text(encoding="utf-8")
    )
    for s in stat.SEEDS:
        fseed = stat.fold_seed_for(s)
        expected_folds = stat.folds(train, fold_seed=fseed)
        if (
            fold_manifest["families_by_seed"][str(s)] != expected_folds
            or fold_manifest["families_by_fold_seed"][str(fseed)] != expected_folds
        ):
            raise RuntimeError(f"fold_manifest families mismatch for seed {s}")

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
            smoke_member_hashes = {}
            for s in stat.SEEDS:
                m_seeds = stat.member_seeds_for(s, k=stat.DEFAULT_ENSEMBLE_K)
                m_hashes = tuple(
                    rt.tensor_hash(random_adapter_state(net, names, ms))
                    for ms in m_seeds
                )
                stat.validate_cell_init_hashes(
                    initial_hash, m_hashes, k=stat.DEFAULT_ENSEMBLE_K
                )
                stat.validate_rens_matches_ens_inits(
                    m_hashes, m_hashes, k=stat.DEFAULT_ENSEMBLE_K
                )
                smoke_member_hashes[str(s)] = list(m_hashes)
            restore(net, names, initial)
            rt.check_frozen(net, guard, names, frozen)
            save(
                out / "smoke.json",
                dict(
                    wave="H190",
                    host=host,
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=TENSORS,
                    trainable_parameters=PARAMETERS,
                    prespecified_candidate_step=stat.PRESPECIFIED_CANDIDATE_STEP,
                    independent_fold_coverage=coverage_proof,
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
        engine.load_features(H190_INPUTS / "val_only_context_features_91.npz", val)
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
            H190_INPUTS / "train_only_context_features_1219.npz", train
        )
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}

        # Stage 1: Each outer seed s uses its OWN independent fold_seed (190141, 190142, 190143).
        # Train on FIT_{s, outer} only, score ONLY B_{s, outer} at the single prespecified step per arm,
        # and pool B across all 205 families per seed BEFORE opening or scoring any A_{s, outer} outcome.
        pooled_b_panel = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        pooled_b_choices = np.full((3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64)
        pooled_b_k5_panel = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        pooled_b_k5_choices = np.full((3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64)

        pilot_record = None
        execution = None
        active_k = stat.DEFAULT_ENSEMBLE_K
        pooled_b_ens_member_utils = None
        pooled_b_ens_member_choices = None
        pooled_b_rens_member_utils = None
        pooled_b_rens_member_choices = None

        drifts_by_arm = {"ENS10_member_mean": [], "RENS10": []}
        cell_stage1_records = []

        stage_guard.mark_b_read()
        for si, seed in enumerate(stat.SEEDS):
            fseed = stat.fold_seed_for(seed)
            for outer in range(5):
                fit_rows, b_rows, a_rows = stat.partition(train, outer, fold_seed=fseed)
                fit_cids = {r["context_sha256"] for r in fit_rows}
                b_cids = {r["context_sha256"] for r in b_rows}
                a_cids = {r["context_sha256"] for r in a_rows}
                excluded_cids = b_cids | a_cids | val_cids
                if (fit_cids & excluded_cids) or (b_cids & a_cids):
                    raise RuntimeError("family/context partition leakage")

                engine.load_pools(fit_rows)
                orders = {
                    c: np.argsort(engine.pools[c]["fitness"]) for c in sorted(fit_cids)
                }
                b_indices = [positions[r["context_sha256"]] for r in b_rows]
                cellname = f"seed{seed}-fseed{fseed}-A{outer}"

                m_seeds_full = stat.member_seeds_for(seed, k=stat.DEFAULT_ENSEMBLE_K)
                m_init_states_full = [
                    random_adapter_state(net, names, ms) for ms in m_seeds_full
                ]
                m_init_hashes_full = tuple(
                    rt.tensor_hash(st) for st in m_init_states_full
                )
                stat.validate_cell_init_hashes(
                    initial_hash, m_init_hashes_full, k=stat.DEFAULT_ENSEMBLE_K
                )

                # Timing pilot on the first 4 updates of member 1 of (seed=190101, outer=0) BEFORE any B or A pool is scored
                pilot_m1_cell = None
                if pilot_record is None:
                    if (si, outer) != (0, 0):
                        raise RuntimeError("timing pilot order changed")
                    m1_seed = m_seeds_full[0]
                    m1_init = m_init_states_full[0]
                    m1_batch_seed = stat.stream_seed(
                        seed,
                        fseed,
                        outer,
                        "MEMBER1",
                        m1_seed,
                        "h190-h156-batches-v1",
                    )
                    m1_plan = stat.batch_plan(
                        sorted(fit_cids),
                        orders,
                        m1_batch_seed,
                        updates=stat.MEMBER_UPDATES,
                    )
                    pilot = run_cell(
                        net,
                        rt,
                        engine,
                        names,
                        frozen,
                        guard,
                        m1_init,
                        fit_rows,
                        excluded_cids,
                        m1_batch_seed,
                        stat.PILOT_UPDATES,
                        plan=m1_plan,
                    )
                    updates += stat.PILOT_UPDATES
                    sec_per_update = float(
                        np.median(pilot["elapsed_update_seconds"][1:])
                    )
                    execution = stat.execution_plan(sec_per_update)
                    active_k = int(execution["active_ensemble_k"])
                    stage_guard = stat.StageOrderGuard(
                        stat.PRESPECIFIED_CANDIDATE_STEP, ensemble_k=active_k
                    )
                    stage_guard.mark_b_read()
                    pooled_b_ens_member_utils = np.full((3, 1219, active_k), np.nan)
                    pooled_b_ens_member_choices = np.full(
                        (3, 1219, active_k), -1, dtype=np.int64
                    )
                    pooled_b_rens_member_utils = np.full((3, 1219, active_k), np.nan)
                    pooled_b_rens_member_choices = np.full(
                        (3, 1219, active_k), -1, dtype=np.int64
                    )
                    pilot_record = dict(
                        elapsed_update_seconds=pilot["elapsed_update_seconds"],
                        seconds_per_update=sec_per_update,
                        **execution,
                        rule="timing only; first 4 updates of seed190101-fseed190141-A0 MEMBER1 on FIT_0 before any B or A outcome is scored",
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
                        m1_batch_seed,
                        stat.MEMBER_UPDATES,
                        plan=m1_plan,
                        optimizer=pilot["optimizer"],
                        start_step=stat.PILOT_UPDATES + 1,
                    )
                    updates += stat.MEMBER_UPDATES - stat.PILOT_UPDATES
                    pilot_m1_cell = dict(
                        state=tail["state"],
                        traces=pilot["traces"] + tail["traces"],
                        elapsed_update_seconds=pilot["elapsed_update_seconds"]
                        + tail["elapsed_update_seconds"],
                        batch_plan=m1_plan,
                    )

                # Load B_{s, outer} pools now that pilot is locked; A_{s, outer} remains unloaded/unscored in Stage 1.
                engine.load_pools(b_rows)
                engine.clear_eval_cache(keep_cids=val_cids | b_cids)
                restore(net, names, initial)
                selbase, _ = engine.evaluate(b_rows, cache_batches=True)

                pooled_b_panel[si, b_indices, 0] = selbase["values"]
                pooled_b_choices[si, b_indices, 0] = selbase["choices"]
                pooled_b_k5_panel[si, b_indices, 0] = selbase["values"]
                pooled_b_k5_choices[si, b_indices, 0] = selbase["choices"]

                m_seeds = m_seeds_full[:active_k]
                m_init_states = m_init_states_full[:active_k]
                m_init_hashes = m_init_hashes_full[:active_k]

                rens_b_raws = []
                rens_hashes_checked = []
                ens_b_raws = []
                ens_final_hashes = []
                ens_member_drifts = []
                ens_member_records = []

                for mi, (m_seed, m_init) in enumerate(
                    zip(m_seeds, m_init_states, strict=True), 1
                ):
                    restore(net, names, m_init)
                    r_hash = rt.tensor_hash(adapter_state(net, names))
                    rens_hashes_checked.append(r_hash)
                    save_state_npz(
                        out / f"{cellname}-RENS-member{mi}-step0.npz", m_init
                    )
                    r_b_eval, r_b_raw = engine.evaluate(b_rows, cache_batches=True)
                    rens_b_raws.append(r_b_raw)
                    pooled_b_rens_member_utils[si, b_indices, mi - 1] = r_b_eval[
                        "values"
                    ]
                    pooled_b_rens_member_choices[si, b_indices, mi - 1] = r_b_eval[
                        "choices"
                    ]

                    if pilot_m1_cell is not None and mi == 1:
                        m_cell = pilot_m1_cell
                    else:
                        m_batch_seed = stat.stream_seed(
                            seed,
                            fseed,
                            outer,
                            f"MEMBER{mi}",
                            m_seed,
                            "h190-h156-batches-v1",
                        )
                        m_plan = stat.batch_plan(
                            sorted(fit_cids),
                            orders,
                            m_batch_seed,
                            updates=stat.MEMBER_UPDATES,
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
                            stat.MEMBER_UPDATES,
                            plan=m_plan,
                        )
                        updates += stat.MEMBER_UPDATES

                    m_final_hash = rt.tensor_hash(m_cell["state"])
                    m_drift = adapter_l2_drift(m_cell["state"], m_init, names)
                    if m_final_hash == r_hash or m_drift <= 0.0:
                        raise RuntimeError(f"ENS member {mi} failed to update")
                    ens_final_hashes.append(m_final_hash)
                    ens_member_drifts.append(m_drift)
                    save_state_npz(
                        out / f"{cellname}-ENS-member{mi}-step40.npz",
                        m_cell["state"],
                    )
                    restore(net, names, m_cell["state"])
                    e_b_eval, e_b_raw = engine.evaluate(b_rows, cache_batches=True)
                    ens_b_raws.append(e_b_raw)
                    pooled_b_ens_member_utils[si, b_indices, mi - 1] = e_b_eval[
                        "values"
                    ]
                    pooled_b_ens_member_choices[si, b_indices, mi - 1] = e_b_eval[
                        "choices"
                    ]
                    ens_member_records.append(
                        dict(
                            member_index=mi,
                            member_seed=m_seed,
                            init_sha256=r_hash,
                            final_sha256=m_final_hash,
                            drift_l2=m_drift,
                            updates=stat.MEMBER_UPDATES,
                        )
                    )

                stat.validate_rens_matches_ens_inits(
                    rens_hashes_checked, m_init_hashes, k=active_k
                )
                if len(set(ens_final_hashes)) != active_k:
                    raise RuntimeError("ENS trained member states are not all distinct")

                ens_b_eval = evaluate_score_ensemble(
                    engine, b_rows, ens_b_raws, k=active_k
                )
                rens_b_eval = evaluate_score_ensemble(
                    engine, b_rows, rens_b_raws, k=active_k
                )
                pooled_b_panel[si, b_indices, 1] = ens_b_eval["values"]
                pooled_b_choices[si, b_indices, 1] = ens_b_eval["choices"]
                pooled_b_panel[si, b_indices, 2] = rens_b_eval["values"]
                pooled_b_choices[si, b_indices, 2] = rens_b_eval["choices"]

                ens5_b_eval = evaluate_score_ensemble(
                    engine, b_rows, ens_b_raws[:5], k=5
                )
                rens5_b_eval = evaluate_score_ensemble(
                    engine, b_rows, rens_b_raws[:5], k=5
                )
                pooled_b_k5_panel[si, b_indices, 1] = ens5_b_eval["values"]
                pooled_b_k5_choices[si, b_indices, 1] = ens5_b_eval["choices"]
                pooled_b_k5_panel[si, b_indices, 2] = rens5_b_eval["values"]
                pooled_b_k5_choices[si, b_indices, 2] = rens5_b_eval["choices"]

                drifts_by_arm["ENS10_member_mean"].append(
                    float(np.mean(ens_member_drifts))
                )
                drifts_by_arm["RENS10"].append(0.0)

                cell_rec = dict(
                    cell=cellname,
                    seed=seed,
                    fold_seed=fseed,
                    outer_fold=outer,
                    b_fold=(outer + 1) % 5,
                    active_ensemble_k=active_k,
                    fit_families=len({r["family_id"] for r in fit_rows}),
                    fit_contexts=len(fit_rows),
                    b_families=len({r["family_id"] for r in b_rows}),
                    b_contexts=len(b_rows),
                    r882_sha256=initial_hash,
                    ens_members=ens_member_records,
                    rens_matches_ens_inits=True,
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
        pooled_b_decisions = stat.pooled_b_select_h190(
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
                wave="H190",
                pooled_b_families=205,
                pooled_b_contexts=1219,
                fold_seeds_by_seed={
                    str(k): int(v) for k, v in stat.FOLD_SEED_BY_SEED.items()
                },
                boot_seed=stat.BOOT_SEED,
                holm_multiplicity=stat.HOLM_MULTIPLICITY,
                active_ensemble_k=active_k,
                prespecified_candidate_step=stage_guard.expected_steps,
                decisions=pooled_b_decisions,
                pooled_b_summary=pooled_b_summary,
                locked_before_a_confirmation=True,
            ),
        )
        stage_guard.mark_b_locked(lock_path)

        # Stage 2: Separate confirmation readout on A_{s, outer} families and descriptive DL-Val.
        locked = stage_guard.require_b_locked()
        confirm_a_selected = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        confirm_a_uncond = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        confirm_a_choices_uncond = np.full(
            (3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64
        )
        confirm_a_choices_selected = np.full(
            (3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64
        )
        confirm_a_k5_uncond = np.full((3, 1219, len(stat.ALL_ARMS)), np.nan)
        confirm_a_k5_choices_uncond = np.full(
            (3, 1219, len(stat.ALL_ARMS)), -1, dtype=np.int64
        )
        confirm_a_ens_member_utils = np.full((3, 1219, active_k), np.nan)
        confirm_a_ens_member_choices = np.full((3, 1219, active_k), -1, dtype=np.int64)
        confirm_a_rens_member_utils = np.full((3, 1219, active_k), np.nan)
        confirm_a_rens_member_choices = np.full((3, 1219, active_k), -1, dtype=np.int64)

        val_selected = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        val_uncond = np.full((3, 5, 91, len(stat.ALL_ARMS)), np.nan)
        val_choices_uncond = np.full((3, 5, 91, len(stat.ALL_ARMS)), -1, dtype=np.int64)

        for si, seed in enumerate(stat.SEEDS):
            fseed = stat.fold_seed_for(seed)
            for outer in range(5):
                _, _, a_rows = stat.partition(train, outer, fold_seed=fseed)
                a_cids = {r["context_sha256"] for r in a_rows}
                a_indices = [positions[r["context_sha256"]] for r in a_rows]
                engine.load_pools(a_rows)
                engine.clear_eval_cache(keep_cids=val_cids | a_cids)
                restore(net, names, initial)
                abase, _ = engine.evaluate(a_rows, cache_batches=True)

                cellname = f"seed{seed}-fseed{fseed}-A{outer}"
                confirm_a_selected[si, a_indices, 0] = abase["values"]
                confirm_a_uncond[si, a_indices, 0] = abase["values"]
                confirm_a_choices_uncond[si, a_indices, 0] = abase["choices"]
                confirm_a_choices_selected[si, a_indices, 0] = abase["choices"]
                confirm_a_k5_uncond[si, a_indices, 0] = abase["values"]
                confirm_a_k5_choices_uncond[si, a_indices, 0] = abase["choices"]
                val_selected[si, outer, :, 0] = baseline["values"]
                val_uncond[si, outer, :, 0] = baseline["values"]
                val_choices_uncond[si, outer, :, 0] = baseline["choices"]

                ens_a_raws, ens_v_raws = [], []
                rens_a_raws, rens_v_raws = [], []
                for mi in range(1, active_k + 1):
                    e_snap = load_state_npz(
                        out / f"{cellname}-ENS-member{mi}-step40.npz", names
                    )
                    restore(net, names, e_snap)
                    e_a_eval, e_a_raw = engine.evaluate(a_rows, cache_batches=True)
                    _, e_v_raw = engine.evaluate(val, cache_batches=True)
                    ens_a_raws.append(e_a_raw)
                    ens_v_raws.append(e_v_raw)
                    confirm_a_ens_member_utils[si, a_indices, mi - 1] = e_a_eval[
                        "values"
                    ]
                    confirm_a_ens_member_choices[si, a_indices, mi - 1] = e_a_eval[
                        "choices"
                    ]

                    r_snap = load_state_npz(
                        out / f"{cellname}-RENS-member{mi}-step0.npz", names
                    )
                    restore(net, names, r_snap)
                    r_a_eval, r_a_raw = engine.evaluate(a_rows, cache_batches=True)
                    _, r_v_raw = engine.evaluate(val, cache_batches=True)
                    rens_a_raws.append(r_a_raw)
                    rens_v_raws.append(r_v_raw)
                    confirm_a_rens_member_utils[si, a_indices, mi - 1] = r_a_eval[
                        "values"
                    ]
                    confirm_a_rens_member_choices[si, a_indices, mi - 1] = r_a_eval[
                        "choices"
                    ]

                ens_a_eval = evaluate_score_ensemble(
                    engine, a_rows, ens_a_raws, k=active_k
                )
                ens_v_eval = evaluate_score_ensemble(
                    engine, val, ens_v_raws, k=active_k
                )
                rens_a_eval = evaluate_score_ensemble(
                    engine, a_rows, rens_a_raws, k=active_k
                )
                rens_v_eval = evaluate_score_ensemble(
                    engine, val, rens_v_raws, k=active_k
                )

                ens5_a_eval = evaluate_score_ensemble(
                    engine, a_rows, ens_a_raws[:5], k=5
                )
                rens5_a_eval = evaluate_score_ensemble(
                    engine, a_rows, rens_a_raws[:5], k=5
                )
                confirm_a_k5_uncond[si, a_indices, 1] = ens5_a_eval["values"]
                confirm_a_k5_choices_uncond[si, a_indices, 1] = ens5_a_eval["choices"]
                confirm_a_k5_uncond[si, a_indices, 2] = rens5_a_eval["values"]
                confirm_a_k5_choices_uncond[si, a_indices, 2] = rens5_a_eval["choices"]

                evals_by_arm = {
                    "ENS10": (ens_a_eval, ens_v_eval),
                    "RENS10": (rens_a_eval, rens_v_eval),
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
                        fold_seed=fseed,
                        outer_fold=outer,
                        a_families=len({r["family_id"] for r in a_rows}),
                        a_contexts=len(a_rows),
                    ),
                )
                print(f"STAGE-2 A-CONFIRM {cellname}", flush=True)

        restore(net, names, initial)
        engine.check()

        np.savez_compressed(
            out / "panels.npz",
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            val_context_ids=np.array([r["context_sha256"] for r in val]),
            val_family_ids=np.array([r["family_id"] for r in val]),
            seed_ids=np.array(stat.SEEDS),
            fold_seeds=np.array(stat.FOLD_SEEDS),
            arm_names=np.array(stat.ALL_ARMS),
            pooled_b=pooled_b_panel,
            pooled_b_choices=pooled_b_choices,
            pooled_b_k5=pooled_b_k5_panel,
            pooled_b_k5_choices=pooled_b_k5_choices,
            pooled_b_ens_member_utils=pooled_b_ens_member_utils,
            pooled_b_ens_member_choices=pooled_b_ens_member_choices,
            pooled_b_rens_member_utils=pooled_b_rens_member_utils,
            pooled_b_rens_member_choices=pooled_b_rens_member_choices,
            confirm_a_selected=confirm_a_selected,
            confirm_a_unconditional=confirm_a_uncond,
            confirm_a_choices_unconditional=confirm_a_choices_uncond,
            confirm_a_choices_selected=confirm_a_choices_selected,
            confirm_a_k5_unconditional=confirm_a_k5_uncond,
            confirm_a_k5_choices_unconditional=confirm_a_k5_choices_uncond,
            confirm_a_ens_member_utils=confirm_a_ens_member_utils,
            confirm_a_ens_member_choices=confirm_a_ens_member_choices,
            confirm_a_rens_member_utils=confirm_a_rens_member_utils,
            confirm_a_rens_member_choices=confirm_a_rens_member_choices,
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
        pooled_b_k5_summary = stat.summarize_panel(
            train, pooled_b_k5_panel, seed=stat.BOOT_SEED
        )
        confirm_a_k5_uncond_summary = stat.summarize_panel(
            train, confirm_a_k5_uncond, seed=stat.BOOT_SEED
        )
        val_uncond_summary = stat.summarize_panel(
            val, val_uncond.mean(axis=1), seed=stat.BOOT_SEED
        )
        val_selected_summary = stat.summarize_panel(
            val, val_selected.mean(axis=1), seed=stat.BOOT_SEED
        )

        verdict_info = stat.resolve_h190_verdict(
            pooled_b_decisions,
            confirm_a_uncond_summary,
            h178_receipt=h178_receipt,
            expected_h177_analysis_sha256=h177_analysis_sha,
        )

        audit_summary = dict(
            status="PASS",
            host=host,
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
            wave="H190",
            host=host,
            status="COMPLETE",
            arms=list(stat.ALL_ARMS),
            candidates=list(stat.CANDIDATES),
            holm_multiplicity=stat.HOLM_MULTIPLICITY,
            active_ensemble_k=active_k,
            seeds=list(stat.SEEDS),
            fold_seeds_by_seed={
                str(k): int(v) for k, v in stat.FOLD_SEED_BY_SEED.items()
            },
            member_seeds_by_seed={
                str(k): list(v[:active_k]) for k, v in stat.MEMBER_SEEDS_BY_SEED.items()
            },
            boot_seed=stat.BOOT_SEED,
            prespecified_candidate_step=stage_guard.expected_steps,
            pilot_budget=pilot_record,
            optimizer_updates=updates,
            trained_trajectories=15 * active_k,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            static_grad_parity_bit_exact=True,
            static_grad_parity_perturbed_bit_exact=True,
            rens10_uses_exact_ens10_inits=True,
            independent_fold_coverage=coverage_proof,
            h178_calibration_receipt=h178_receipt,
            selection_caveat_disclosure=(
                "Candidate arms (ENS10, RENS10) were nominated after H179/H182/H185 on the same 205 DL-Train families; "
                "H190 eliminates shared-partition pseudo-replication by assigning each outer seed (190101, 190102, 190103) "
                "its own independent 3/1/1 family fold partition (fold_seeds 190141, 190142, 190143) and disjoint member seeds."
            ),
            pooled_b_selection=dict(
                decisions=pooled_b_decisions,
                summary=pooled_b_summary,
            ),
            confirmation_a_unconditional=confirm_a_uncond_summary,
            confirmation_a_selected=confirm_a_selected_summary,
            descriptive_val_unconditional=val_uncond_summary,
            descriptive_val_selected=val_selected_summary,
            diagnostic_nested_k5=dict(
                diagnostic_only=True,
                pooled_b=pooled_b_k5_summary,
                confirmation_a_unconditional=confirm_a_k5_uncond_summary,
            ),
            partition_variance_component=dict(
                pooled_b=pooled_b_summary["partition_variance"],
                confirmation_a_unconditional=confirm_a_uncond_summary[
                    "partition_variance"
                ],
                h182_shared_fold_ens10_reference=dict(
                    fold_seed=182140,
                    per_seed_gain=[
                        0.015200267278660685,
                        0.019115990611359168,
                        0.012835414953637455,
                    ],
                    between_seed_sd_ddof1=float(
                        np.std(
                            [
                                0.015200267278660685,
                                0.019115990611359168,
                                0.012835414953637455,
                            ],
                            ddof=1,
                        )
                    ),
                    between_seed_range=float(
                        0.019115990611359168 - 0.012835414953637455
                    ),
                ),
            ),
            member_pairwise_choice_disagreement=dict(
                pooled_b_ens10=stat.pairwise_member_disagreement(
                    pooled_b_ens_member_choices, k=active_k
                ),
                pooled_b_rens10=stat.pairwise_member_disagreement(
                    pooled_b_rens_member_choices, k=active_k
                ),
                confirm_a_ens10=stat.pairwise_member_disagreement(
                    confirm_a_ens_member_choices, k=active_k
                ),
                confirm_a_rens10=stat.pairwise_member_disagreement(
                    confirm_a_rens_member_choices, k=active_k
                ),
                pooled_b_ens5_prefix=stat.pairwise_member_disagreement(
                    pooled_b_ens_member_choices, k=5
                ),
                confirm_a_ens5_prefix=stat.pairwise_member_disagreement(
                    confirm_a_ens_member_choices, k=5
                ),
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
        print("H190 COMPLETE verdict=" + metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "incomplete.json",
            dict(
                wave="H190",
                host=host,
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
