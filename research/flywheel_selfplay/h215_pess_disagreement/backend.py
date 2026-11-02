"""GPU adapter-only H190 recipe, with cached outcome-free full-pool inputs."""

import io
import numpy as np
import torch


def random_adapter_state(net, names, seed):
    torch.manual_seed(seed)
    old = net.adapter_10tok
    device = next(old.parameters()).device
    fresh = (
        type(old)(
            frozen_anchor=old.anchor,
            prog_mean=old.prog_mean.detach().cpu().numpy(),
            prog_std=old.prog_std.detach().cpu().numpy(),
            d_token=old.d_token,
            n_modalities=old.n_modalities,
            modality_dropout_p=old.modality_dropout_p,
        )
        .to(device)
        .eval()
    )
    params = dict(fresh.named_parameters())
    state = {
        name: params[name.removeprefix("adapter_10tok.")].detach().clone()
        for name in names
    }
    if len(state) != 47 or sum(t.numel() for t in state.values()) != 29211:
        raise RuntimeError("fresh adapter inventory mismatch")
    return state


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
        self.full = {}
        self.labels_cache = {}

    def restore(self, state):
        restore(self.net, self.names, state)

    def labels(self, cid, allowed, role):
        if cid not in allowed or role not in ("FIT", "B", "A"):
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
            ctx_feat=torch.zeros((8, 584), device="cuda:0"),
            prog_412=torch.stack(programs),
            proj_16=torch.stack(projections),
            base_prior_cost=torch.zeros((8, 24), device="cuda:0"),
            scale=torch.full((8,), 0.4208, device="cuda:0"),
            sos_prior=torch.zeros((8, 24, 2), device="cuda:0"),
            revealed_indices=torch.arange(4, device="cuda:0")[None].expand(8, -1),
            revealed_raw_f=f[:, :4],
            revealed_gate=g[:, :4].float(),
        )
        return kwargs, torch.stack(toks), torch.stack(masks), f, g

    def train(self, rows, seed, label):
        allowed = {r["context_sha256"] for r in rows}
        cids = sorted(allowed)
        orders = {cid: np.argsort(self.labels(cid, allowed, "FIT")[0]) for cid in cids}
        if len(cids) < 8 or any(len(o) < 24 for o in orders.values()):
            raise RuntimeError("H190 batch dimensions unavailable")
        initial = random_adapter_state(self.net, self.names, seed)
        restore(self.net, self.names, initial)
        params = [p for p in self.net.parameters() if p.requires_grad]
        if len(params) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("optimizer inventory mismatch")
        before = self.rt.tensor_hash(initial)
        witness = cids[0]
        before_score = self.score(witness)
        optimizer = torch.optim.AdamW(
            params, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)
        )
        losses, gradient_sum = [], 0.0
        for step in range(40):
            self.check()
            rng = np.random.default_rng(seed * 1000 + step)
            batch = [cids[i] for i in rng.choice(len(cids), 8, replace=False)]
            ix = [
                np.r_[orders[c][:4], rng.choice(orders[c][4:], 20, replace=False)]
                for c in batch
            ]
            kwargs, tok, mask, f, g = self.batch(batch, ix, allowed)
            anchor = self.anchor(kwargs)
            optimizer.zero_grad(set_to_none=True)
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
                *anchor,
            )
            loss = self.rt.loss(pred, f, g)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite objective")
            loss.backward()
            active = sum(p.grad is not None for p in params)
            if active != 41:
                raise RuntimeError("active/None gradient inventory mismatch")
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            gradient_sum += float(norm) ** 2
            optimizer.step()
            self.progress["updates"] += 1
            losses.append(float(loss.detach()))
        state = {
            name: dict(self.net.named_parameters())[name].detach().cpu().clone()
            for name in self.names
        }
        after = self.rt.tensor_hash(state)
        changed = not np.array_equal(before_score, self.score(witness))
        if before == after or gradient_sum <= 0 or not changed:
            raise RuntimeError("missing learned-forward proof")
        proof = dict(
            member_seed=seed,
            updates=40,
            before_sha256=before,
            after_sha256=after,
            gradient_squared_sum=gradient_sum,
            forward_changed=changed,
            actual_device=str(params[0].device),
            losses=losses,
            fit_families=sorted({r["family_id"] for r in rows}),
        )
        # Each fit's proof is durable immediately, including if a later fit aborts.
        self.store.put(
            "proofs/" + label + ".json",
            (__import__("json").dumps(proof) + "\n").encode(),
        )
        return state, proof
