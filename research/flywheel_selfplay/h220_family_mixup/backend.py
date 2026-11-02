"""H219 CONTROL and within-family mixed-input/mixed-target adapter training."""

import io
import json
import hashlib
import time
from collections import Counter
import numpy as np
import torch


def restore(net, names, state):
    params = dict(net.named_parameters())
    with torch.no_grad():
        for name in names:
            params[name].copy_(state[name].to(params[name].device))


class Engine:
    def __init__(
        self,
        net,
        rt,
        treatment,
        pools,
        gpu_inputs,
        native,
        check,
        progress,
        store,
        names,
    ):
        self.net, self.rt, self.treatment, self.pools = net, rt, treatment, pools
        self.inputs, self.native, self.check, self.progress, self.store, self.names = (
            gpu_inputs,
            native,
            check,
            progress,
            store,
            names,
        )
        self.initial = {
            n: dict(net.named_parameters())[n].detach().cpu().clone() for n in names
        }
        self.full = {}
        self.labels_cache = {}

    def restore(self, state):
        restore(self.net, self.names, state)

    def labels(self, cid, allowed, role):
        if cid not in allowed or role not in ("FIT", "A"):
            raise RuntimeError("label role refusal")
        if role == "A" and not self.progress.get("locked", False):
            raise RuntimeError("A label before global lock")
        if cid not in self.labels_cache:
            with np.load(io.BytesIO(self.pools.blob(cid)), allow_pickle=False) as z:
                self.labels_cache[cid] = (
                    z["fitness"].astype(np.float64),
                    z["gate"].astype(bool),
                )
        self.pools.outcome_reads.append((role, cid))
        return self.labels_cache[cid]

    def utility(self, cid, allowed, role):
        f, g = self.labels(cid, allowed, role)
        return np.where(g, -f, -2.0)

    def anchor(self, kwargs):
        with torch.no_grad():
            out = self.net.adapter_10tok.anchor.predict_costs(
                **kwargs,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )
        return out["total_cost"].detach().float(), out["eff_prior"].detach().float()

    def full_entry(self, cid):
        if cid not in self.full:
            kwargs, tok, mask = self.inputs[cid]
            anchor = self.anchor(kwargs)
            static = self.treatment.precompute_static_slate(
                self.net.adapter_10tok, kwargs["prog_412"], kwargs["sos_prior"], mask
            )
            self.full[cid] = (
                tok,
                static,
                kwargs["scale"],
                kwargs["revealed_indices"],
                kwargs["revealed_raw_f"],
                kwargs["revealed_gate"],
                *anchor,
            )
        return self.full[cid]

    def score(self, cid):
        self.check()
        with torch.no_grad():
            return (
                self.treatment.adapter_forward_static(self.net, *self.full_entry(cid))[
                    0
                ]
                .float()
                .cpu()
                .numpy()
            )

    def batch(self, cids, indices, allowed):
        if not set(cids) <= allowed:
            raise RuntimeError("FIT-only batch refusal")
        programs, projections, fitness, gates, toks, masks = [], [], [], [], [], []
        for cid, ix in zip(cids, indices, strict=True):
            kwargs, tok, mask = self.inputs[cid]
            ids = torch.as_tensor(ix, device="cuda:0")
            programs.append(kwargs["prog_412"][0, ids])
            projections.append(kwargs["proj_16"][0, ids])
            f, g = self.labels(cid, allowed, "FIT")
            fitness.append(f[ix])
            gates.append(g[ix])
            toks.append(tok[0])
            masks.append(mask[0])
        f = torch.as_tensor(np.stack(fitness), dtype=torch.float32, device="cuda:0")
        g = torch.as_tensor(np.stack(gates), dtype=torch.bool, device="cuda:0")
        kwargs = dict(
            ctx_feat=torch.zeros((len(cids), 584), device="cuda:0"),
            prog_412=torch.stack(programs),
            proj_16=torch.stack(projections),
            base_prior_cost=torch.zeros((len(cids), 24), device="cuda:0"),
            scale=torch.full((len(cids),), 0.4208, device="cuda:0"),
            sos_prior=torch.zeros((len(cids), 24, 2), device="cuda:0"),
            revealed_indices=torch.arange(4, device="cuda:0")[None].expand(
                len(cids), -1
            ),
            revealed_raw_f=f[:, :4],
            revealed_gate=g[:, :4].float(),
        )
        return kwargs, torch.stack(toks), torch.stack(masks), f, g

    def train(
        self, rows, arm, seed, outer, label, legacy, sci, output, checkpoint, steps=40
    ):
        if steps not in (4, 40) or arm not in sci.ARMS:
            raise ValueError("registered timing/full trajectory required")
        allowed = {r["context_sha256"] for r in rows}
        cids = sorted(allowed)
        groups = {
            f: sorted(r["context_sha256"] for r in rows if r["family_id"] == f)
            for f in sorted({r["family_id"] for r in rows})
        }
        plan = legacy.sample_microbatches(
            groups, sci.LEGACY_ARMS[arm], sci.training_seed(seed, outer)
        )
        if len(plan) != 160 or any(
            len(batch) != 16 or not set(batch) <= allowed for batch in plan
        ):
            raise RuntimeError("H109 matched microbatch/FIT plan mismatch")
        staged = legacy.stage_r882_copy(checkpoint, output / "scratch" / label)
        restore(self.net, self.names, self.initial)
        params = [p for p in self.net.parameters() if p.requires_grad]
        if len(params) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("adapter inventory mismatch")
        mix_counts = dict(
            rows=0, mixed_rows=0, singleton_rows=0, same_family_pairs=True
        )
        family_by_cid = {r["context_sha256"]: r["family_id"] for r in rows}
        before = self.rt.tensor_hash(self.initial)
        witness = cids[0]
        before_score = self.score(witness)
        orders = {cid: np.argsort(self.labels(cid, allowed, "FIT")[0]) for cid in cids}
        if any(len(o) < 24 for o in orders.values()):
            raise RuntimeError("24-candidate FIT slate unavailable")
        optimizer = ObservedOptimizer(
            torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
            params,
        )
        losses, rates, durations = [], [], []
        for step in range(steps):
            started = time.monotonic()
            self.check()
            rates.append(
                legacy.set_update_learning_rate(optimizer, sci.LEGACY_ARMS[arm], step)
            )
            micro_losses = []
            for micro in range(4):
                batch = plan[step * 4 + micro]
                indices = []
                for slot, cid in enumerate(batch):
                    rng = np.random.default_rng(
                        sci.stream(seed, outer, step, micro, slot, cid, "candidates")
                    )
                    indices.append(
                        np.r_[
                            orders[cid][:4],
                            rng.choice(orders[cid][4:], 20, replace=False),
                        ]
                    )
                left = self.batch(batch, indices, allowed)
                kwargs, tok, mask, f, g = left
                anchors = self.anchor(kwargs)
                if arm == "MIX":
                    partners, weights = self.mixup.plan(
                        batch,
                        groups,
                        family_by_cid,
                        sci.stream(seed, outer, step, micro, "mixup"),
                    )
                    if any(
                        family_by_cid[a] != family_by_cid[b]
                        for a, b in zip(batch, partners, strict=True)
                    ):
                        raise RuntimeError("cross-family partner refused")
                    partner_indices = []
                    for slot, (cid, partner) in enumerate(
                        zip(batch, partners, strict=True)
                    ):
                        if cid == partner:
                            partner_indices.append(indices[slot])
                        else:
                            rng = np.random.default_rng(
                                sci.stream(
                                    seed,
                                    outer,
                                    step,
                                    micro,
                                    slot,
                                    partner,
                                    "partner-candidates",
                                )
                            )
                            partner_indices.append(
                                np.r_[
                                    orders[partner][:4],
                                    rng.choice(orders[partner][4:], 20, replace=False),
                                ]
                            )
                    right = self.batch(partners, partner_indices, allowed)
                    kwargs, tok, mask, f, g, anchors = self.mixup.batch(
                        left, right, anchors, self.anchor(right[0]), weights
                    )
                    mix_counts["rows"] += len(batch)
                    mix_counts["mixed_rows"] += sum(
                        a != b for a, b in zip(batch, partners, strict=True)
                    )
                    mix_counts["singleton_rows"] += sum(
                        len(groups[family_by_cid[cid]]) == 1 for cid in batch
                    )
                pred = self.treatment.adapter_forward(
                    self.net,
                    kwargs["prog_412"],
                    kwargs["sos_prior"],
                    tok,
                    mask,
                    kwargs["scale"],
                    kwargs["revealed_indices"],
                    kwargs["revealed_raw_f"],
                    kwargs["revealed_gate"],
                    *anchors,
                )
                micro_losses.append(self.rt.loss(pred, f, g))
            losses.append(legacy.accumulate_step(params, optimizer, micro_losses))
            self.progress["updates"] += 1
            durations.append(time.monotonic() - started)
        state = {
            n: dict(self.net.named_parameters())[n].detach().cpu().clone()
            for n in self.names
        }
        after = self.rt.tensor_hash(state)
        changed = not np.array_equal(before_score, self.score(witness))
        if before == after or optimizer.gradient_squared_sum <= 0 or not changed:
            raise RuntimeError("missing optimizer/changed-forward proof")
        family_by_cid = {r["context_sha256"]: r["family_id"] for r in rows}
        counts = Counter(
            family_by_cid[cid] for batch in plan[: steps * 4] for cid in batch
        )
        proof = dict(
            arm=arm,
            seed=seed,
            outer=outer,
            training_seed=sci.training_seed(seed, outer),
            updates=steps,
            microbatches=steps * 4,
            contexts_per_microbatch=16,
            effective_batch=64,
            before_sha256=before,
            after_sha256=after,
            gradient_squared_sum=optimizer.gradient_squared_sum,
            forward_changed=changed,
            actual_device=str(params[0].device),
            active_gradients=41,
            inactive_gradients=6,
            fit_families=sorted(groups),
            loss_per_update=losses,
            learning_rates=rates,
            seconds_per_update=durations,
            sampler_plan_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
            sampled_contexts_by_family=dict(counts),
            mixup=mix_counts,
            staged_checkpoint=str(staged),
            checkpoint_sha256=legacy._sha256(staged),
        )
        self.store.put("proofs/" + label + ".json", (json.dumps(proof) + "\n").encode())
        return state, proof


class ObservedOptimizer:
    """Observe gradients at H109's step hook without changing its accumulation."""

    def __init__(self, optimizer, params):
        self.optimizer, self.params = optimizer, params
        self.param_groups = optimizer.param_groups
        self.gradient_squared_sum = 0.0

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)

    def step(self):
        if sum(p.grad is not None for p in self.params) != 41:
            raise RuntimeError("41 active/6 inactive gradient inventory required")
        self.gradient_squared_sum += sum(
            float(p.grad.detach().square().sum())
            for p in self.params
            if p.grad is not None
        )
        self.optimizer.step()
