"""One CPU-only H177 wave: pooled cross-fit confirmatory selector (205-family pooled B selection + single prespecified candidate per arm + separate A-family confirmation)."""

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
    CANDIDATE_STEP,
    PARAMETERS,
    accumulate_microbatches,
    adapter_forward,
    adapter_forward_static,
    adapter_l2_drift,
    apply_optimizer_settings,
    compute_fit_family_weights,
    flatten_grad_with_none_as_zero,
    grad_norms,
    precompute_static_slate,
    validate_h175_power_anchor,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
H177_INPUTS = Path("/home/ubuntu/jope-data/h177/inputs")


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
    spec = importlib.util.spec_from_file_location("h177_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train_only = json.loads((H177_INPUTS / "train_only_selected_1219.json").read_text())
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
        gen = torch.Generator(device="cpu").manual_seed(177777)
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
    power_anchor = json.loads((HERE / "h175_power_anchor.json").read_text())
    validate_h175_power_anchor(power_anchor, smoke=args.smoke)
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
    coverage_proof = stat.verify_pooled_b_coverage(train)
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
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    updates, started = 0, time.monotonic()
    stage_guard = stat.StageOrderGuard()
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
                    prespecified_candidate_step=stat.PRESPECIFIED_CANDIDATE_STEP,
                    pooled_b_coverage=coverage_proof,
                    h175_power_anchor_status=power_anchor["status"],
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
        engine.load_features(H177_INPUTS / "val_only_context_features_91.npz", val)
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
        engine.load_features(
            H177_INPUTS / "train_only_context_features_1219.npz", train
        )
        engine.load_pools(train)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}

        # Stage 1: Train on FIT_k only per fold, score ONLY B_k at the single prespecified candidate step (step 40), and pool B across all 205 families.
        pooled_b_panel = np.full((3, 1219, 5), np.nan)
        step40_drifts = {a: [] for a in stat.ARMS}
        adapter_hashes_by_cell_arm = {}
        fold_fit_summary = {}

        def restore(snapshot):
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in snapshot:
                        p.copy_(snapshot[n])
            engine.check()

        stage_guard.mark_b_read()
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
            b_indices = [positions[r["context_sha256"]] for r in select]
            orders = {
                c: np.argsort(engine.pools[c]["fitness"]) for c in ordered_fit_cids
            }

            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-{seed}"
                pooled_b_panel[si, b_indices, 0] = selbase["values"]
                slates = stat.fit_slates(fit, orders, seed, outer)
                plan = stat.family_stratified_plan(fit, seed, outer, excluded)
                fit_bundle = engine.build_fit_bundle(ordered_fit_cids, slates)
                cell_b_records = {"r882": selbase}
                cell_hashes = {"r882": initial_hash}
                cell_drifts = {}

                for col, arm in enumerate(stat.ARMS, 1):
                    cand_step = stage_guard.candidate_step_for(arm)
                    if cand_step != CANDIDATE_STEP:
                        raise RuntimeError("prespecified candidate step mismatch")
                    restore(initial)
                    torch.manual_seed(stat.stream_seed(seed, outer, arm))
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    m_count = ACCUM_STEPS[arm]
                    for u, step_entry in enumerate(plan, 1):
                        apply_optimizer_settings(optimizer, arm, u)
                        mbs = step_entry["microbatches"][:m_count]
                        mb_fns = []
                        for mb in mbs:
                            idx_list = [fit_bundle["cid_to_idx"][c] for c in mb["cids"]]
                            mb_fns.append(
                                lambda ix=idx_list, b=fit_bundle: (
                                    engine.per_context_loss(
                                        engine.forward_fit_slice(b, ix),
                                        b["fitness"][ix],
                                        b["gate"][ix],
                                    ).mean()
                                )
                            )
                        accumulate_microbatches(params, mb_fns)
                        _, gnorms = grad_norms(params, 1.0)
                        if (
                            gnorms["active_grad_tensors"] != 41
                            or gnorms["none_grad_tensors"] != 6
                        ):
                            raise RuntimeError("active/None grad tensor mismatch")
                        optimizer.step()
                        updates += 1

                    # Evaluate ONLY the single prespecified candidate step (step 40) on B_outer.
                    engine.check()
                    snapshot = {n: net.state_dict()[n].detach().clone() for n in names}
                    drift40 = adapter_l2_drift(snapshot, initial, names)
                    step40_drifts[arm].append(drift40)
                    cell_drifts[arm] = drift40
                    cell_hashes[arm] = rt.tensor_hash(snapshot)
                    np.savez_compressed(
                        out / f"{cellname}-{arm}-step{cand_step}.npz",
                        **{n: t.numpy() for n, t in snapshot.items()},
                    )
                    b_eval, _ = engine.evaluate(select, cache_batches=True)
                    pooled_b_panel[si, b_indices, col] = b_eval["values"]
                    cell_b_records[arm] = b_eval

                adapter_hashes_by_cell_arm[cellname] = cell_hashes
                save(
                    out / f"{cellname}.b_only.json",
                    dict(
                        cell=cellname,
                        seed=seed,
                        outer_fold=outer,
                        b_fold=(outer + 1) % 5,
                        prespecified_candidate_step=CANDIDATE_STEP,
                        adapter_sha256=cell_hashes,
                        step40_l2_drift=cell_drifts,
                        b_evaluations=cell_b_records,
                    ),
                )
                del fit_bundle
                print(
                    f"STAGE-1 B-ONLY {cellname} updates={updates}",
                    flush=True,
                )

        if not np.isfinite(pooled_b_panel).all():
            raise RuntimeError("incomplete pooled_b_panel")

        pooled_b_decisions = stat.pooled_b_select(
            train,
            pooled_b_panel[:, :, 0],
            {arm: pooled_b_panel[:, :, col] for col, arm in enumerate(stat.ARMS, 1)},
            seed=stat.BOOT_SEED,
        )
        pooled_b_summary = stat.summarize(train, pooled_b_panel, seed=stat.BOOT_SEED)
        lock_path = out / "pooled_b_selection_lock.json"
        save(
            lock_path,
            dict(
                wave="H177",
                pooled_b_families=205,
                pooled_b_contexts=1219,
                prespecified_candidate_step=stat.PRESPECIFIED_CANDIDATE_STEP,
                h178_calibration_gate=stat.H178_CALIBRATION_GATE,
                decisions=pooled_b_decisions,
                pooled_b_summary=pooled_b_summary,
                locked_before_a_confirmation=True,
            ),
        )
        stage_guard.mark_b_locked(lock_path)

        # Stage 2: Separate confirmation readout on A families (never used for selection in fold k) and descriptive DL-Val.
        locked = stage_guard.require_b_locked()
        confirm_a_selected = np.full((3, 1219, 5), np.nan)
        confirm_a_step40 = np.full((3, 1219, 5), np.nan)
        val_selected = np.full((3, 5, 91, 5), np.nan)
        val_step40 = np.full((3, 5, 91, 5), np.nan)

        for outer in range(5):
            _, _, assess = stat.partition(train, outer)
            a_indices = [positions[r["context_sha256"]] for r in assess]
            engine.clear_eval_cache(
                keep_cids=val_cids | {r["context_sha256"] for r in assess}
            )
            restore(initial)
            abase, _ = engine.evaluate(assess, cache_batches=True)
            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-{seed}"
                confirm_a_selected[si, a_indices, 0] = abase["values"]
                confirm_a_step40[si, a_indices, 0] = abase["values"]
                val_selected[si, outer, :, 0] = baseline["values"]
                val_step40[si, outer, :, 0] = baseline["values"]
                a_records, v_records = {"r882": abase}, {"r882": baseline}

                for col, arm in enumerate(stat.ARMS, 1):
                    cand_step = stage_guard.candidate_step_for(arm)
                    with np.load(
                        out / f"{cellname}-{arm}-step{cand_step}.npz",
                        allow_pickle=False,
                    ) as z:
                        snap = {n: torch.as_tensor(z[n]) for n in names}
                    restore(snap)
                    a_eval, _ = engine.evaluate(assess, cache_batches=True)
                    v_eval, _ = engine.evaluate(val, cache_batches=True)
                    a_records[arm] = a_eval
                    v_records[arm] = v_eval
                    confirm_a_step40[si, a_indices, col] = a_eval["values"]
                    val_step40[si, outer, :, col] = v_eval["values"]
                    admitted = bool(locked["decisions"][arm]["admitted"])
                    confirm_a_selected[si, a_indices, col] = (
                        a_eval["values"] if admitted else abase["values"]
                    )
                    val_selected[si, outer, :, col] = (
                        v_eval["values"] if admitted else baseline["values"]
                    )
                save(
                    out / f"{cellname}.a_confirm.json",
                    dict(
                        cell=cellname,
                        seed=seed,
                        outer_fold=outer,
                        a_confirm_evaluations=a_records,
                        val_evaluations=v_records,
                    ),
                )

        restore(initial)
        np.savez_compressed(
            out / "panels.npz",
            pooled_b=pooled_b_panel,
            confirm_a_selected=confirm_a_selected,
            confirm_a_step40=confirm_a_step40,
            val_selected=val_selected,
            val_step40=val_step40,
        )
        confirm_a_selected_summary = stat.summarize(
            train, confirm_a_selected, seed=stat.BOOT_SEED
        )
        confirm_a_step40_summary = stat.summarize(
            train, confirm_a_step40, seed=stat.BOOT_SEED
        )
        val_selected_summary = stat.summarize(
            val, val_selected.mean(axis=1), seed=stat.BOOT_SEED
        )
        val_step40_summary = stat.summarize(
            val, val_step40.mean(axis=1), seed=stat.BOOT_SEED
        )
        audit_summary = dict(
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
        advance_any = any(
            d["admitted"]
            and confirm_a_selected_summary["contrasts"][arm + "_minus_r882"]["ci95"][0]
            > 0.0
            for arm, d in pooled_b_decisions.items()
            if arm != "CTRL"
        )
        metrics = dict(
            wave="H177",
            status="COMPLETE",
            arms=list(stat.ALL_ARMS),
            seeds=list(stat.SEEDS),
            prespecified_candidate_step=stat.PRESPECIFIED_CANDIDATE_STEP,
            optimizer_updates=updates,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            static_grad_parity_bit_exact=True,
            static_grad_parity_perturbed_bit_exact=True,
            pooled_b_coverage=coverage_proof,
            h175_power_anchor=power_anchor,
            h178_calibration_gate=stat.H178_CALIBRATION_GATE,
            pooled_b_selection=dict(
                decisions=pooled_b_decisions,
                summary=pooled_b_summary,
            ),
            confirmation_a_selected=confirm_a_selected_summary,
            confirmation_a_step40=confirm_a_step40_summary,
            descriptive_val_selected=val_selected_summary,
            descriptive_val_step40=val_step40_summary,
            step40_l2_drift_summary={
                arm: dict(
                    median=float(np.median(step40_drifts[arm])),
                    min=float(np.min(step40_drifts[arm])),
                    max=float(np.max(step40_drifts[arm])),
                )
                for arm in stat.ARMS
            },
            fold_fit_summary=fold_fit_summary,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            trainable_parameters=PARAMETERS,
            active_grad_tensors=41,
            none_grad_tensors=6,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit_summary,
            elapsed_seconds=time.monotonic() - started,
            verdict="ADVANCE" if advance_any else "NO_ADVANCE",
        )
        save(out / "metrics.json", metrics)
        print("H177 COMPLETE verdict=" + metrics["verdict"], flush=True)
    except Exception as exc:
        save(
            out / "incomplete.json",
            dict(
                wave="H177",
                status="INCOMPLETE",
                optimizer_updates=updates,
                reason=str(exc),
                refused=audit.refused,
            ),
        )
        raise


if __name__ == "__main__":
    main()
