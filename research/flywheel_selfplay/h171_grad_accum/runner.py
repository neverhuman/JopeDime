"""One CPU-only H171 wave: large effective batch via family-stratified gradient accumulation on r882 continuation."""

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
import torch.nn.functional as F
import analysis as stat
from treatment import (
    ACCUM_STEPS,
    DIAG_STEPS,
    PARAMETERS,
    TOTAL_UPDATES,
    accumulate_microbatches,
    adapter_forward,
    adapter_forward_static,
    adapter_l2_drift,
    compute_fit_family_weights,
    flatten_grad_with_none_as_zero,
    grad_cosine,
    grad_norms,
    precompute_static_slate,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H171_INPUTS = Path("/home/ubuntu/jope-data/h171/inputs")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources"):
        for path, digest in pins[kind].items():
            p = REPO / path if kind == "parent_sources" else Path(path)
            if sha(p) != digest:
                raise RuntimeError("pin mismatch: " + str(p))
    for path, digest in pins["source_sha256"].items():
        if sha(HERE / path) != digest:
            raise RuntimeError("source pin mismatch: " + path)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package mismatch: " + name)


def runtime():
    path = REPO / "research/flywheel_selfplay/h146_lookahead_exec/runtime.py"
    spec = importlib.util.spec_from_file_location("h171_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train_only = json.loads((H171_INPUTS / "train_only_selected_1219.json").read_text())
    train, val = packet["train"], packet["dev"]
    if train != train_only["train"]:
        raise RuntimeError("train_only_selected_1219 mismatch")
    if (
        len(train) != 1219
        or len(val) != 91
        or len({r["family_id"] for r in train}) != 205
    ):
        raise RuntimeError("cohort size")
    if len({r["context_sha256"] for r in train + val}) != 1310:
        raise RuntimeError("duplicate context")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected context")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("family overlap")
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
        self.pools, self.cached, self.tokens, self.masks = {}, {}, {}, {}
        self.eval_batch_cache = {}
        self.projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)

    def load_features(self, npz_path, rows):
        expected = {r["context_sha256"] for r in rows}
        with np.load(npz_path, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if len(ids) != len(rows) or set(ids) != expected:
                raise RuntimeError("split feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            for i, c in enumerate(ids):
                self.tokens[c] = torch.as_tensor(tok[i], dtype=torch.float32)
                self.masks[c] = torch.as_tensor(mask[i], dtype=torch.bool)

    def load_pools(self, rows):
        for row in rows:
            cid = row["context_sha256"]
            if cid in self.pools:
                continue
            path = self.rt.INPUT / "pools" / (cid + ".npz")
            if sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool hash")
            with np.load(path, allow_pickle=False) as z:
                p = z["program"].astype(np.float32)
                self.pools[cid] = dict(
                    program=p,
                    proj16=(p.astype(np.float64) @ self.projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

    def batch(self, cids, indexes):
        b, k = len(cids), len(indexes[0])

        def stack(key):
            return np.stack(
                [self.pools[c][key][ix] for c, ix in zip(cids, indexes, strict=True)]
            )

        prog = torch.as_tensor(stack("program"), dtype=torch.float32)
        proj = torch.as_tensor(stack("proj16"), dtype=torch.float32)
        fitness = torch.as_tensor(stack("fitness"), dtype=torch.float32)
        gate = torch.as_tensor(stack("gate"), dtype=torch.bool)
        kwargs = dict(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=prog,
            proj_16=proj,
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, : min(4, k)],
            revealed_gate=gate[:, : min(4, k)].float(),
        )
        return (
            kwargs,
            torch.stack([self.tokens[c] for c in cids]),
            torch.stack([self.masks[c] for c in cids]),
            fitness,
            gate,
        )

    def anchored(self, kwargs):
        with torch.no_grad():
            a = self.net.adapter_10tok.anchor.predict_costs(
                **kwargs,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
        return a["total_cost"].detach().float(), a["eff_prior"].detach().float()

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

    def build_fit_bundle(self, ordered_cids, slates):
        """Precompute fit-only 24-candidate static slates and frozen anchors once per (seed, outer) cell."""
        ixs = [slates[c] for c in ordered_cids]
        kwargs, tok, mask, fitness, gate = self.batch(ordered_cids, ixs)
        anc_totals, anc_priors = [], []
        for start in range(0, len(ordered_cids), 64):
            end = min(start + 64, len(ordered_cids))
            sub_kw = {k: v[start:end] for k, v in kwargs.items()}
            at, ap = self.anchored(sub_kw)
            anc_totals.append(at)
            anc_priors.append(ap)
        anchor_total = torch.cat(anc_totals, dim=0)
        anchor_prior = torch.cat(anc_priors, dim=0)
        static_slate = precompute_static_slate(
            self.net.adapter_10tok, kwargs["prog_412"], kwargs["sos_prior"], mask
        )
        return dict(
            cids=ordered_cids,
            cid_to_idx={c: i for i, c in enumerate(ordered_cids)},
            tok=tok,
            static_slate=static_slate,
            scale=kwargs["scale"],
            rev=kwargs["revealed_indices"],
            raw=kwargs["revealed_raw_f"],
            gate_rev=kwargs["revealed_gate"],
            anchor_total=anchor_total,
            anchor_prior=anchor_prior,
            fitness=fitness,
            gate=gate,
        )

    def forward_fit_slice(self, bundle, idx_list=None):
        if idx_list is None:
            return adapter_forward_static(
                self.net,
                bundle["tok"],
                bundle["static_slate"],
                bundle["scale"],
                bundle["rev"],
                bundle["raw"],
                bundle["gate_rev"],
                bundle["anchor_total"],
                bundle["anchor_prior"],
            )
        ix = torch.as_tensor(idx_list, dtype=torch.long)
        sub_static = {k: v[ix] for k, v in bundle["static_slate"].items()}
        return adapter_forward_static(
            self.net,
            bundle["tok"][ix],
            sub_static,
            bundle["scale"][ix],
            bundle["rev"][ix],
            bundle["raw"][ix],
            bundle["gate_rev"][ix],
            bundle["anchor_total"][ix],
            bundle["anchor_prior"][ix],
        )

    def prove_static_gradient_parity(self, batch_cids, params, initial):
        """Verify bit-exact forward and backward gradient parity between dynamic adapter_forward and adapter_forward_static at both initial and perturbed weights."""
        kwargs, tok, mask, fitness, gate = self.batch(
            batch_cids, [np.arange(24) for _ in batch_cids]
        )
        anchor = self.anchored(kwargs)
        static_slate = precompute_static_slate(
            self.net.adapter_10tok, kwargs["prog_412"], kwargs["sos_prior"], mask
        )
        gen = torch.Generator(device="cpu").manual_seed(171777)
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
            loss_dyn = self.per_context_loss(pred_dyn, fitness, gate).mean()
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
            loss_stat = self.per_context_loss(pred_stat, fitness, gate).mean()
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

    @staticmethod
    def per_context_loss(pred, fitness, gate):
        target = torch.where(
            gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0)
        )
        return F.smooth_l1_loss(pred, target, reduction="none").mean(
            1
        ) + 0.1 * F.kl_div(
            F.log_softmax(-pred / 0.2, dim=-1),
            F.softmax(-target / 0.2, dim=-1),
            reduction="none",
        ).sum(1)

    def check(self):
        self.rt.check_frozen(self.net, self.guard, self.names, self.frozen)
        if (
            self.net.dope_model.training
            or self.net.adapter_10tok.training
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")

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
    out.mkdir()  # Exclusive; never reuse completed/partial study outputs.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    if launch["manifest_sha256"] != sha(HERE / "manifest.json"):
        raise RuntimeError("launch manifest binding")
    if not args.smoke and (
        not launch["merged_source_verified"]
        or launch["source_commit"]
        != subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    ):
        raise RuntimeError("merged exact source required")
    rt = runtime()
    train, val = cohorts(rt)
    provenance = json.loads((HERE / "provenance.json").read_text())
    ids = sorted({r["dataset_id"] for r in train + val})
    if (
        provenance["status"] != "PASS"
        or not provenance["zero_synthetic"]
        or ids != provenance["dataset_ids"]
        or provenance["real_catalog_joined"] != 276
    ):
        raise RuntimeError("real S3 origin closure")
    if (
        stat.folds(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["families"]
    ):
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    # Deliberate exact-path read after hook: regression for H153's launch custody failure.
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    updates, started = 0, time.monotonic()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        initial = {n: net.state_dict()[n].detach().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params) != PARAMETERS
            or len(names) != 47
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
        ):
            raise RuntimeError("adapter/mode inventory")
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        if {id(p) for group in optimizer.param_groups for p in group["params"]} != {
            id(p) for p in params
        }:
            raise RuntimeError("optimizer exclusion")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=47,
                    trainable_parameters=PARAMETERS,
                    launch_exact_path_after_audit=True,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    initial_adapter_sha256=initial_hash,
                    census=census,
                ),
            )
            return
        ref = json.loads(Path(pins["reference"]).read_text())
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
            raise RuntimeError("evaluation contract")
        engine = Engine(net, rt, names, frozen, guard, fused)
        # Load ONLY DL-Val features and pools before exact 91/91 r882 baseline gate passes.
        engine.load_features(H171_INPUTS / "val_only_context_features_91.npz", val)
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
            raise RuntimeError("exact 91 r882 baseline")
        engine.prove_static_gradient_parity(
            [r["context_sha256"] for r in val[:8]], params, initial
        )
        if (
            rt.tensor_hash({n: net.state_dict()[n].detach().clone() for n in names})
            != initial_hash
        ):
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
        # Startup verify(pins) hashed all pinned input files; actual DL-Train feature/pool tensor decoding is deferred until after DL-Val baseline passes.
        engine.load_features(
            H171_INPUTS / "train_only_context_features_1219.npz", train
        )
        engine.load_pools(train)
        primary = np.full((3, 1219, 5), np.nan)
        secondary = np.full((3, 5, 91, 5), np.nan)
        unconstrained = np.full_like(primary, np.nan)
        unconstrained_val = np.full_like(secondary, np.nan)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        selections = []
        raw_b_panels_by_arm = {a: [] for a in stat.ARMS}
        adapter_hashes_by_arm = {a: [] for a in stat.ARMS}
        step40_drifts = {a: [] for a in stat.ARMS}
        selected_drifts = {a: [] for a in stat.ARMS}
        unconstrained_drifts = {a: [] for a in stat.ARMS}
        grad_diag_exact = {a: [] for a in stat.ARMS}
        grad_diag_lagged = {a: [] for a in stat.ARMS}
        grad_diag_step1 = {a: [] for a in stat.ARMS}
        grad_diag_r882 = {a: [] for a in stat.ARMS}
        raw_grad_norms_by_arm = {a: [] for a in stat.ARMS}
        fold_fit_summary = {}
        distinct_from_ctrl_step40 = {a: 0 for a in stat.ARMS[1:]}

        def restore(snapshot):
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in snapshot:
                        p.copy_(snapshot[n])
            engine.check()

        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fitids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            if fitids & excluded:
                raise RuntimeError("family leakage")
            ordered_fit_cids, fit_weights, n_fit_fams = compute_fit_family_weights(
                fit, excluded
            )
            if n_fit_fams != 123:
                raise RuntimeError("expected 123 fit families")
            fold_fit_summary[str(outer)] = dict(
                fit_contexts=len(ordered_fit_cids),
                fit_families=n_fit_fams,
                min_context_weight=float(fit_weights.min()),
                max_context_weight=float(fit_weights.max()),
                weights_sha256=hashlib.sha256(
                    fit_weights.numpy().tobytes()
                ).hexdigest(),
            )
            engine.clear_eval_cache(
                keep_cids=val_cids | {r["context_sha256"] for r in select}
            )
            restore(initial)
            selbase, _ = engine.evaluate(select, cache_batches=True)
            abase, _ = engine.evaluate(assess, cache_batches=False)
            indices = [positions[r["context_sha256"]] for r in assess]
            orders = {
                c: np.argsort(engine.pools[c]["fitness"]) for c in ordered_fit_cids
            }

            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-{seed}"
                slates = stat.fit_slates(fit, orders, seed, outer)
                plan = stat.family_stratified_plan(fit, seed, outer, excluded)
                plan_digest = hashlib.sha256(
                    json.dumps(plan, sort_keys=True).encode()
                ).hexdigest()
                slate_digest = hashlib.sha256(
                    b"".join(slates[c].tobytes() for c in ordered_fit_cids)
                ).hexdigest()
                fit_bundle = engine.build_fit_bundle(ordered_fit_cids, slates)

                def compute_full_fit_grad(bundle=fit_bundle):
                    for p in params:
                        p.grad = None
                    pred = engine.forward_fit_slice(bundle)
                    per_ctx = engine.per_context_loss(
                        pred, bundle["fitness"], bundle["gate"]
                    )
                    w_loss = torch.sum(per_ctx.double() * fit_weights.double())
                    if not torch.isfinite(w_loss):
                        raise RuntimeError("nonfinite full-fit objective")
                    w_loss.backward()
                    vec, act_n, none_n = flatten_grad_with_none_as_zero(params)
                    for p in params:
                        p.grad = None
                    if act_n != 41 or none_n != 6:
                        raise RuntimeError(
                            f"unexpected active/None grad count: {act_n}/{none_n}"
                        )
                    return vec, float(w_loss.detach()), act_n, none_n

                restore(initial)
                g_full_r882, loss_full_r882, _, _ = compute_full_fit_grad()
                g_full_r882_l2 = float(torch.linalg.vector_norm(g_full_r882))

                snapshots, selector_panels, traces, step_drifts = {}, {}, {}, {}
                for arm in stat.ARMS:
                    restore(initial)
                    torch.manual_seed(stat.stream_seed(seed, outer, arm))
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    snapshots[arm] = {0: initial}
                    selector_panels[arm] = [selbase]
                    traces[arm] = []
                    step_drifts[arm] = {"0": 0.0}
                    g_full_current = g_full_r882
                    loss_full_current = loss_full_r882
                    ref_upd_idx = 1

                    if arm in ("CTRL", "ACC4", "ACC16"):
                        m_count = ACCUM_STEPS[arm]
                        for u, step_entry in enumerate(plan, 1):
                            exact_diag = u in DIAG_STEPS[arm]
                            if exact_diag:
                                ref_upd_idx = u
                                if u > 1:
                                    g_full_current, loss_full_current, _, _ = (
                                        compute_full_fit_grad()
                                    )
                            lag_updates = u - ref_upd_idx
                            mbs = step_entry["microbatches"][:m_count]
                            mb_fns = []
                            for mb in mbs:
                                idx_list = [
                                    fit_bundle["cid_to_idx"][c] for c in mb["cids"]
                                ]
                                mb_fns.append(
                                    lambda ix=idx_list, b=fit_bundle: (
                                        engine.per_context_loss(
                                            engine.forward_fit_slice(b, ix),
                                            b["fitness"][ix],
                                            b["gate"][ix],
                                        ).mean()
                                    )
                                )
                            mean_loss = accumulate_microbatches(params, mb_fns)
                            upd_vec, gnorms = grad_norms(params, 1.0)
                            if (
                                gnorms["active_grad_tensors"] != 41
                                or gnorms["none_grad_tensors"] != 6
                            ):
                                raise RuntimeError("active/None grad tensor mismatch")
                            cos_curr = grad_cosine(upd_vec, g_full_current)
                            cos_r882 = grad_cosine(upd_vec, g_full_r882)
                            optimizer.step()
                            updates += 1
                            grad_diag_r882[arm].append(cos_r882)
                            raw_grad_norms_by_arm[arm].append(gnorms["raw_grad_l2"])
                            if u == 1:
                                grad_diag_step1[arm].append(cos_r882)
                            if exact_diag:
                                grad_diag_exact[arm].append(cos_curr)
                            else:
                                grad_diag_lagged[arm].append(cos_curr)
                            traces[arm].append(
                                dict(
                                    update=u,
                                    logical_step=u,
                                    microbatches_accumulated=m_count,
                                    examples_in_update=8 * m_count,
                                    loss=mean_loss,
                                    full_fit_loss=loss_full_current,
                                    exact_instantaneous_full_fit=exact_diag,
                                    reference_full_fit_update=ref_upd_idx,
                                    reference_full_fit_lag_updates=lag_updates,
                                    cos_to_full_fit_grad=(
                                        cos_curr if exact_diag else None
                                    ),
                                    cos_to_lagged_macrostep_full_fit_grad=cos_curr,
                                    cos_to_r882_full_fit_grad=cos_r882,
                                    **gnorms,
                                )
                            )
                            if u in stat.STEPS:
                                engine.check()
                                snapshot = {
                                    n: net.state_dict()[n].detach().clone()
                                    for n in names
                                }
                                snapshots[arm][u] = snapshot
                                step_drifts[arm][str(u)] = adapter_l2_drift(
                                    snapshot, initial, names
                                )
                                np.savez_compressed(
                                    out / f"{cellname}-{arm}-step{u}.npz",
                                    **{n: t.numpy() for n, t in snapshot.items()},
                                )
                                panel, _ = engine.evaluate(select, cache_batches=True)
                                selector_panels[arm].append(panel)
                    else:
                        # CTRL_EX4: 160 live updates on the exact same 160 microbatches (u=1..40, m=0..3) as ACC4.
                        upd_idx = 0
                        for u, step_entry in enumerate(plan, 1):
                            for mb in step_entry["microbatches"][:4]:
                                upd_idx += 1
                                exact_diag = upd_idx in DIAG_STEPS["CTRL_EX4"]
                                if exact_diag:
                                    ref_upd_idx = upd_idx
                                    if upd_idx > 1:
                                        g_full_current, loss_full_current, _, _ = (
                                            compute_full_fit_grad()
                                        )
                                lag_updates = upd_idx - ref_upd_idx
                                idx_list = [
                                    fit_bundle["cid_to_idx"][c] for c in mb["cids"]
                                ]
                                mean_loss = accumulate_microbatches(
                                    params,
                                    [
                                        lambda ix=idx_list, b=fit_bundle: (
                                            engine.per_context_loss(
                                                engine.forward_fit_slice(b, ix),
                                                b["fitness"][ix],
                                                b["gate"][ix],
                                            ).mean()
                                        )
                                    ],
                                )
                                upd_vec, gnorms = grad_norms(params, 1.0)
                                if (
                                    gnorms["active_grad_tensors"] != 41
                                    or gnorms["none_grad_tensors"] != 6
                                ):
                                    raise RuntimeError(
                                        "active/None grad tensor mismatch"
                                    )
                                cos_curr = grad_cosine(upd_vec, g_full_current)
                                cos_r882 = grad_cosine(upd_vec, g_full_r882)
                                optimizer.step()
                                updates += 1
                                grad_diag_r882[arm].append(cos_r882)
                                raw_grad_norms_by_arm[arm].append(gnorms["raw_grad_l2"])
                                if upd_idx == 1:
                                    grad_diag_step1[arm].append(cos_r882)
                                if exact_diag:
                                    grad_diag_exact[arm].append(cos_curr)
                                else:
                                    grad_diag_lagged[arm].append(cos_curr)
                                logical_s = stat.CTRL_EX4_SNAPSHOT_UPDATES.get(upd_idx)
                                traces[arm].append(
                                    dict(
                                        update=upd_idx,
                                        logical_step=logical_s,
                                        microbatches_accumulated=1,
                                        examples_in_update=8,
                                        loss=mean_loss,
                                        full_fit_loss=loss_full_current,
                                        exact_instantaneous_full_fit=exact_diag,
                                        reference_full_fit_update=ref_upd_idx,
                                        reference_full_fit_lag_updates=lag_updates,
                                        cos_to_full_fit_grad=(
                                            cos_curr if exact_diag else None
                                        ),
                                        cos_to_lagged_macrostep_full_fit_grad=cos_curr,
                                        cos_to_r882_full_fit_grad=cos_r882,
                                        **gnorms,
                                    )
                                )
                                if logical_s in stat.STEPS and logical_s > 0:
                                    engine.check()
                                    snapshot = {
                                        n: net.state_dict()[n].detach().clone()
                                        for n in names
                                    }
                                    snapshots[arm][logical_s] = snapshot
                                    step_drifts[arm][str(logical_s)] = adapter_l2_drift(
                                        snapshot, initial, names
                                    )
                                    np.savez_compressed(
                                        out / f"{cellname}-{arm}-step{logical_s}.npz",
                                        **{n: t.numpy() for n, t in snapshot.items()},
                                    )
                                    panel, _ = engine.evaluate(
                                        select, cache_batches=True
                                    )
                                    selector_panels[arm].append(panel)
                        if upd_idx != TOTAL_UPDATES["CTRL_EX4"]:
                            raise RuntimeError("CTRL_EX4 update count mismatch")

                    if (
                        rt.tensor_hash(snapshots[arm][40]) == initial_hash
                        or step_drifts[arm]["40"] <= 0.0
                    ):
                        raise RuntimeError("unchanged adapter")
                    step40_drifts[arm].append(step_drifts[arm]["40"])

                ctrl_h40 = rt.tensor_hash(snapshots["CTRL"][40])
                for arm in stat.ARMS[1:]:
                    if rt.tensor_hash(snapshots[arm][40]) != ctrl_h40:
                        distinct_from_ctrl_step40[arm] += 1

                cell_b_mats = {
                    arm: np.array([p["values"] for p in selector_panels[arm]]).T
                    for arm in stat.ARMS
                }
                decisions = {
                    arm: stat.choose(
                        select,
                        cell_b_mats[arm],
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, arm),
                    )
                    for arm in stat.ARMS
                }
                cell_adapter_hashes = {
                    a: {str(s): rt.tensor_hash(w) for s, w in snaps.items()}
                    for a, snaps in snapshots.items()
                }
                for arm in stat.ARMS:
                    raw_b_panels_by_arm[arm].append(cell_b_mats[arm])
                    adapter_hashes_by_arm[arm].append(cell_adapter_hashes[arm])
                    sel_s = str(decisions[arm]["selected_step"])
                    best_s = str(decisions[arm]["best_step"])
                    selected_drifts[arm].append(step_drifts[arm][sel_s])
                    unconstrained_drifts[arm].append(step_drifts[arm][best_s])
                lock = out / (cellname + "-selection.json")
                save(
                    lock,
                    dict(
                        decisions=decisions,
                        fit_ids=ordered_fit_cids,
                        select_ids=[r["context_sha256"] for r in select],
                        assess_ids=[r["context_sha256"] for r in assess],
                        batch_plan_sha256=plan_digest,
                        fit_slates_sha256=slate_digest,
                        fit_weights_sha256=fold_fit_summary[str(outer)][
                            "weights_sha256"
                        ],
                        r882_full_fit_grad_l2=g_full_r882_l2,
                        step_l2_drifts=step_drifts,
                    ),
                )
                lock_sha = sha(
                    lock
                )  # Durable before ANY trained-policy outer-A/DL-Val assessment.
                primary[si, indices, 0], unconstrained[si, indices, 0] = (
                    abase["values"],
                    abase["values"],
                )
                secondary[si, outer, :, 0], unconstrained_val[si, outer, :, 0] = (
                    baseline["values"],
                    baseline["values"],
                )
                assessment = {}
                for ai, arm in enumerate(stat.ARMS, 1):
                    assessment[arm] = {0: dict(outer=abase, val=baseline)}
                    for step in sorted(
                        {decisions[arm]["selected_step"], decisions[arm]["best_step"]}
                        - {0}
                    ):
                        restore(snapshots[arm][step])
                        a, _ = engine.evaluate(assess, cache_batches=False)
                        v, _ = engine.evaluate(val, cache_batches=True)
                        assessment[arm][step] = dict(outer=a, val=v)
                    chosen = assessment[arm][decisions[arm]["selected_step"]]
                    best = assessment[arm][decisions[arm]["best_step"]]
                    primary[si, indices, ai] = chosen["outer"]["values"]
                    secondary[si, outer, :, ai] = chosen["val"]["values"]
                    unconstrained[si, indices, ai] = best["outer"]["values"]
                    unconstrained_val[si, outer, :, ai] = best["val"]["values"]
                save(
                    out / (cellname + ".json"),
                    dict(
                        outer_fold=outer,
                        seed=seed,
                        decisions=decisions,
                        selection_sha256=lock_sha,
                        fit_summary=fold_fit_summary[str(outer)],
                        batch_plan_sha256=plan_digest,
                        fit_slates_sha256=slate_digest,
                        r882_full_fit_grad_l2=g_full_r882_l2,
                        step_l2_drifts=step_drifts,
                        selector_panels=selector_panels,
                        assessment=assessment,
                        traces=traces,
                        adapter_sha256=cell_adapter_hashes,
                    ),
                )
                selections.append(decisions)
                save(
                    out / "progress.json",
                    dict(
                        complete_cells=len(selections),
                        optimizer_updates=updates,
                        last_cell=cellname,
                    ),
                )
                print(
                    "CELL COMPLETE",
                    cellname,
                    "selected",
                    {a: decisions[a]["selected_step"] for a in stat.ARMS},
                    "best",
                    {a: decisions[a]["best_step"] for a in stat.ARMS},
                    "step40_drift",
                    {a: round(step_drifts[a]["40"], 5) for a in stat.ARMS},
                    "step1_cos",
                    {
                        a: round(traces[a][0]["cos_to_r882_full_fit_grad"], 4)
                        for a in stat.ARMS
                    },
                    flush=True,
                )
                restore(initial)
                del snapshots

        if updates != 4200 or len(selections) != 15:
            raise RuntimeError(
                f"incomplete wave: updates={updates}, cells={len(selections)}"
            )
        val_panel, val_unconstrained = (
            secondary.mean(axis=1),
            unconstrained_val.mean(axis=1),
        )
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            unconstrained=unconstrained,
            unconstrained_val=unconstrained_val,
        )
        ctrl_med_drift = float(np.median(step40_drifts["CTRL"]))
        drift_summary = {
            a: dict(
                median_step40_l2_drift=float(np.median(step40_drifts[a])),
                min_step40_l2_drift=float(np.min(step40_drifts[a])),
                max_step40_l2_drift=float(np.max(step40_drifts[a])),
                ratio_vs_ctrl=float(np.median(step40_drifts[a]) / ctrl_med_drift),
                median_selected_l2_drift=float(np.median(selected_drifts[a])),
                median_unconstrained_l2_drift=float(np.median(unconstrained_drifts[a])),
                distinct_from_ctrl_step40=(
                    15 if a == "CTRL" else distinct_from_ctrl_step40[a]
                ),
            )
            for a in stat.ARMS
        }
        gradient_noise_diagnostic = {
            a: dict(
                microbatches_per_update=ACCUM_STEPS[a],
                examples_per_update=8 * ACCUM_STEPS[a],
                updates_per_cell=TOTAL_UPDATES[a],
                exact_instantaneous_evals=len(grad_diag_exact[a]),
                exact_instantaneous_fraction=float(
                    len(grad_diag_exact[a]) / (15 * TOTAL_UPDATES[a])
                ),
                exact_instantaneous_cos_median=float(np.median(grad_diag_exact[a])),
                exact_instantaneous_cos_mean=float(np.mean(grad_diag_exact[a])),
                exact_instantaneous_cos_q25=float(
                    np.quantile(grad_diag_exact[a], 0.25)
                ),
                exact_instantaneous_cos_q75=float(
                    np.quantile(grad_diag_exact[a], 0.75)
                ),
                exact_instantaneous_cos_min=float(np.min(grad_diag_exact[a])),
                exact_instantaneous_cos_max=float(np.max(grad_diag_exact[a])),
                step1_r882_cos_median=float(np.median(grad_diag_step1[a])),
                step1_r882_cos_mean=float(np.mean(grad_diag_step1[a])),
                step1_r882_cos_min=float(np.min(grad_diag_step1[a])),
                step1_r882_cos_max=float(np.max(grad_diag_step1[a])),
                cos_to_r882_full_fit_median=float(np.median(grad_diag_r882[a])),
                cos_to_r882_full_fit_mean=float(np.mean(grad_diag_r882[a])),
                lagged_offgrid_evals=len(grad_diag_lagged[a]),
                lagged_offgrid_cos_median=(
                    float(np.median(grad_diag_lagged[a]))
                    if grad_diag_lagged[a]
                    else None
                ),
                raw_grad_l2_median=float(np.median(raw_grad_norms_by_arm[a])),
                raw_grad_l2_mean=float(np.mean(raw_grad_norms_by_arm[a])),
            )
            for a in stat.ARMS
        }

        def summarize_selector(arm):
            sel_counts = {str(s): 0 for s in stat.STEPS}
            best_counts = {str(s): 0 for s in stat.STEPS}
            gains, lcbs = [], []
            for d in selections:
                sel_counts[str(d[arm]["selected_step"])] += 1
                best_counts[str(d[arm]["best_step"])] += 1
                gains.append(d[arm]["gain"])
                lcbs.append(d[arm]["ci95"][0])
            return dict(
                strict_selected_step_counts=sel_counts,
                best_step_counts=best_counts,
                median_best_step_gain=float(np.median(gains)),
                max_best_step_gain=float(np.max(gains)),
                max_best_step_lcb95=float(np.max(lcbs)),
            )

        inner_selector_summary = {a: summarize_selector(a) for a in stat.ARMS}
        identity_audit = stat.arm_identity_audit(
            selections,
            raw_b_panels_by_arm,
            primary,
            val_panel,
            adapter_hashes_by_arm,
        )
        for a, id_info in identity_audit.items():
            inner_selector_summary[a].update(id_info)

        metrics = dict(
            wave="H171",
            status="COMPLETE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            optimizer_updates=updates,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            static_grad_parity_bit_exact=True,
            static_grad_parity_perturbed_bit_exact=True,
            primary=stat.summarize(train, primary),
            secondary=stat.summarize(val, val_panel),
            descriptive_unconstrained=stat.summarize(train, unconstrained),
            descriptive_unconstrained_val=stat.summarize(val, val_unconstrained),
            gradient_noise_diagnostic=gradient_noise_diagnostic,
            inner_selector_summary=inner_selector_summary,
            drift_summary=drift_summary,
            fold_fit_summary=fold_fit_summary,
            selections=selections,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            trainable_parameters=PARAMETERS,
            active_grad_tensors=41,
            none_grad_tensors=6,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        metrics["verdict"] = (
            "ADVANCE"
            if any(
                metrics["primary"]["contrasts"][a + "_minus_r882"]["advance"]
                for a in stat.ARMS[1:]
            )
            else "NO_ADVANCE"
        )
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise
    finally:
        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
