"""Bounded CPU batches using the pinned H144 forward and H146 loss/freeze helpers."""

from collections import OrderedDict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


class Pools:
    def __init__(self, rows, projection, check):
        self.rows = {r["context_sha256"]: r for r in rows}
        self.projection = projection
        self.check = check
        self.cache = OrderedDict()
        self.verified = {}

    def get(self, cid):
        self.check()
        if cid not in self.rows:
            raise RuntimeError("pool outside approved TRAIN/Val roster")
        if cid not in self.cache:
            path = Path("/home/ubuntu/jope-data/h50/input/pools") / (cid + ".npz")
            if cid not in self.verified:
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                if digest != self.rows[cid]["pool_sha256"]:
                    raise RuntimeError("pool content changed")
                self.verified[cid] = digest
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                value = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ self.projection).astype(
                        np.float32
                    ),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
            self.cache[cid] = value
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
        self.cache.move_to_end(cid)
        return self.cache[cid]


class Engine:
    def __init__(
        self, rt, reg, sci, checkpoint, rows, tokens, masks, check, output, progress
    ):
        self.rt, self.reg, self.sci, self.checkpoint = rt, reg, sci, checkpoint
        self.rows, self.tokens, self.masks = rows, tokens, masks
        self.check, self.output, self.progress = check, output, progress
        self.pools = None

    def model(self, arm):
        self.check()
        for path, digest in self.pins.items():
            if self.rt.sha(path) != digest:
                raise RuntimeError("frozen input/source identity changed: " + path)
        net, guard, names, frozen, fused = self.rt.build(self.checkpoint)
        self.rt.check_frozen(net, guard, names, frozen)
        if self.pools is None:
            self.pools = Pools(
                self.rows, fused.get_w1_sign_matrix() / np.sqrt(412.0), self.check
            )
        targets = self.reg.spectralize(net.adapter_10tok) if arm == "SN" else []
        names = tuple(n for n, p in net.named_parameters() if p.requires_grad)
        buffers = tuple(
            n
            for n, _ in net.named_buffers()
            if n.startswith("adapter_10tok.") and ".parametrizations." in n
        )
        mutable = (*names, *buffers)
        if (
            len(names) != 47
            or sum(p.numel() for p in net.parameters() if p.requires_grad) != 29211
        ):
            raise RuntimeError("adapter-only parameter inventory")
        frozen = self.rt.tensor_hash(
            {n: t for n, t in net.state_dict().items() if n not in mutable}
        )
        return net, guard, mutable, frozen, targets

    def predict(self, net, cids, indexes, jac=False):
        pools = [self.pools.get(c) for c in cids]
        progs = np.stack(
            [p["program"][ix] for p, ix in zip(pools, indexes, strict=True)]
        )
        proj = np.stack([p["proj16"][ix] for p, ix in zip(pools, indexes, strict=True)])
        b, k = progs.shape[:2]
        tokens = (
            torch.stack([self.tokens[c] for c in cids]).detach().requires_grad_(jac)
        )
        masks = torch.stack([self.masks[c] for c in cids])
        fitness = torch.as_tensor(
            np.stack([p["fitness"][ix] for p, ix in zip(pools, indexes, strict=True)]),
            dtype=torch.float32,
        )
        gates = torch.as_tensor(
            np.stack([p["gate"][ix] for p, ix in zip(pools, indexes, strict=True)]),
            dtype=torch.bool,
        )
        pred = net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=torch.as_tensor(progs),
            proj_16=torch.as_tensor(proj),
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            mm_tokens_10x32=tokens,
            mm_mask_10=masks,
            revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, : min(4, k)],
            revealed_gate=gates[:, : min(4, k)].float(),
        )["total_cost"].float()
        return pred, fitness, gates, tokens, masks

    def score(self, net, rows, evidence_label=None):
        groups = {}
        result = {}
        cost_records = {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools.get(cid)["fitness"]), []).append(cid)
        with torch.no_grad():
            for size, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    self.check()
                    batch = cids[start : start + 8]
                    scores = self.predict(net, batch, [np.arange(size) for _ in batch])[
                        0
                    ].numpy()
                    if not np.isfinite(scores).all():
                        raise RuntimeError("nonfinite costs")
                    for cid, cost in zip(batch, scores, strict=True):
                        p = self.pools.get(cid)
                        index = int(np.argmin(cost))
                        cost_records[cid] = cost.copy()
                        result[cid] = dict(
                            choice=index,
                            utility=float(
                                -p["fitness"][index] if p["gate"][index] else -2.0
                            ),
                            chosen_fitness=float(p["fitness"][index]),
                            chosen_gate=bool(p["gate"][index]),
                            pool_sha256=self.pools.rows[cid]["pool_sha256"],
                            costs_sha256=hashlib.sha256(cost.tobytes()).hexdigest(),
                        )
        if evidence_label:
            dest = self.output / "predictions" / (evidence_label + ".npz")
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as f:
                np.savez_compressed(f, **cost_records)
        return result

    def save(self, name, value):
        path = self.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")

    def fit(self, rows, arm, lam, seed, label, steps=40):
        if (
            arm not in self.sci.ARMS
            or steps not in (2, 40)
            or (arm == "JAC" and lam not in self.sci.LAMBDAS)
        ):
            raise ValueError("unregistered fit")
        if not rows or any(r["dl_split"] != "DL-Train" for r in rows):
            raise RuntimeError("TRAIN-only fitting")
        allowed = {r["context_sha256"] for r in rows}
        net, guard, mutable, frozen, targets = self.model(arm)
        orders = {
            cid: np.argsort(self.pools.get(cid)["fitness"]) for cid in sorted(allowed)
        }
        plan = self.sci.batch_plan(sorted(allowed), orders, seed)[:steps]
        initial = {n: net.state_dict()[n].detach().clone() for n in mutable}
        before = self.rt.tensor_hash(initial)
        witness = [rows[0]]
        before_scores = self.score(net, witness)
        params = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        losses = []
        penalties = []
        gradient_sum = 0.0
        started = time.time()
        for stream, cids, indexes in plan:
            self.check()
            if not set(cids) <= allowed:
                raise RuntimeError("FIT closure")
            torch.manual_seed(stream)
            optimizer.zero_grad(set_to_none=True)
            self.reg.advance(targets)
            with sdpa_kernel(SDPBackend.MATH):
                pred, fitness, gates, tok, mask = self.predict(
                    net, cids, indexes, arm == "JAC"
                )
                penalty = (
                    self.reg.jacobian_penalty(pred, tok, mask, stream)
                    if arm == "JAC"
                    else pred.new_zeros(())
                )
                objective = self.rt.loss(pred, fitness, gates) + (
                    lam * penalty if arm == "JAC" else 0.0
                )
                if not torch.isfinite(objective):
                    raise RuntimeError("nonfinite objective")
                objective.backward()
            grad = sum(
                float(p.grad.detach().square().sum())
                for p in params
                if p.grad is not None
            )
            if not np.isfinite(grad):
                raise RuntimeError("nonfinite gradient")
            gradient_sum += grad
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            self.check()
            journal = self.output / "optimizer-events.jsonl"
            with journal.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            event="intent",
                            label=label,
                            step=len(losses),
                            at=time.time(),
                        )
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            optimizer.step()
            self.progress["updates"] += 1
            with journal.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            event="completed",
                            label=label,
                            step=len(losses),
                            updates=self.progress["updates"],
                            at=time.time(),
                        )
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())
            losses.append(float(objective.detach()))
            penalties.append(float(penalty.detach()))
        self.rt.check_frozen(net, guard, mutable, frozen)
        state = {n: net.state_dict()[n].detach().clone() for n in mutable}
        after = self.rt.tensor_hash(state)
        after_scores = self.score(net, witness)
        changed = before_scores != after_scores
        if before == after or gradient_sum <= 0 or not changed:
            raise RuntimeError("missing actual training/forward proof")
        if targets and any(norm.steps != steps for _, _, _, norm in targets):
            raise RuntimeError("SN power-step inventory")
        dest = self.output / "states" / (label + ".pt")
        dest.parent.mkdir(exist_ok=True)
        with dest.open("xb") as f:
            torch.save(state, f)
        with dest.open("rb") as f:
            state_hash = hashlib.file_digest(f, "sha256").hexdigest()
        proof = dict(
            label=label,
            arm=arm,
            lambda_=lam,
            seed=seed,
            updates=steps,
            fit_families=sorted({r["family_id"] for r in rows}),
            fit_contexts=len(rows),
            plan_sha256=hashlib.sha256(
                json.dumps(
                    [
                        (int(st), ids, [ix.tolist() for ix in indices])
                        for st, ids, indices in plan
                    ]
                ).encode()
            ).hexdigest(),
            losses=losses,
            jacobian_penalties=penalties,
            gradient_squared_sum=gradient_sum,
            adapter_before_sha256=before,
            adapter_after_sha256=after,
            forward_changed=True,
            dope_sha256_before=self.rt.DOPE,
            dope_sha256_after=guard.snapshot_dope_weights(net).composite_sha256,
            nonadapter_unchanged=True,
            state_sha256=state_hash,
            state_path=str(dest),
            spectral_targets=[name + "." + field for name, field, _, _ in targets],
            power_steps_per_target=[norm.steps for _, _, _, norm in targets],
            initialization_power_steps=1 if targets else 0,
            wall_seconds=time.time() - started,
        )
        self.save("proofs/" + label + ".json", proof)
        self.progress["fits"] += 1
        self.save("progress/" + label + ".json", dict(self.progress))
        print(
            "FIT",
            label,
            "updates",
            steps,
            "wall",
            round(proof["wall_seconds"], 2),
            flush=True,
        )
        del net, optimizer, params, state, initial, guard
        gc.collect()
        return dict(label=label, arm=arm, path=str(dest), sha256=state_hash)

    def evaluate(self, state, rows):
        self.check()
        with Path(state["path"]).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != state["sha256"]:
            raise RuntimeError("state identity changed")
        net, guard, mutable, frozen, targets = self.model(state["arm"])
        saved = torch.load(state["path"], map_location="cpu", weights_only=True)
        if set(saved) != set(mutable):
            raise RuntimeError("state inventory differs")
        current = net.state_dict()
        current.update(saved)
        net.load_state_dict(current, strict=True)
        result = self.score(net, rows, evidence_label=state["label"])
        self.rt.check_frozen(net, guard, mutable, frozen)
        del net, guard, current, saved
        gc.collect()
        return result
