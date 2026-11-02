"""H213's separate head trainer with immutable feature caching and family bags."""

import io
import hashlib
import json
from collections import Counter
import numpy as np
import torch


class Engine:
    def __init__(
        self,
        sci,
        rt,
        pools,
        native,
        tokens,
        masks,
        position,
        check,
        progress,
        store,
        device="cuda:0",
    ):
        self.sci, self.rt, self.pools, self.native = sci, rt, pools, native
        self.tokens, self.masks, self.position = tokens, masks, position
        self.check, self.progress, self.store, self.device = (
            check,
            progress,
            store,
            device,
        )
        self.features, self.utilities = {}, {}
        self.feature_bytes = 0

    def x_for(self, cid):
        self.check()
        if cid not in self.features:
            ix = self.position[cid]
            raw = self.sci.features(
                self.pools.predictors(cid)[0],
                self.tokens[ix],
                self.masks[ix],
                self.native[cid],
            )
            if self.feature_bytes + raw.nbytes > 5 * 2**30:
                raise RuntimeError("fixed feature cache cap")
            self.features[cid] = torch.as_tensor(raw, device=self.device)
            self.feature_bytes += raw.nbytes
        return self.features[cid]

    def utility(self, cid, allowed, role):
        if (
            cid not in allowed
            or role not in ("FIT", "B", "A")
            or (role == "A" and not self.progress["locked"])
        ):
            raise RuntimeError("outcome role/lock refusal before cache access")
        if cid not in self.utilities:
            self.utilities[cid] = self.pools.utility(
                cid, allowed, role, locked=self.progress["locked"]
            )
        self.pools.outcome_reads.append(("CACHE_" + role, cid))
        return self.utilities[cid]

    def score(self, head, cid):
        with torch.no_grad():
            return head(self.x_for(cid), int(self.native[cid].argmin())).cpu().numpy()

    def fit(self, rows, seed, outer, member, shuffled):
        allowed = {r["context_sha256"] for r in rows}
        plan, init, population = self.sci.fit_plan(rows, seed, outer, member)
        family = {r["context_sha256"]: r["family_id"] for r in rows}
        if any(
            cid not in allowed or family[cid] not in population
            for batch in plan
            for cid in batch
        ):
            raise RuntimeError("FIT bootstrap plan closure")
        label = (
            "SINGLE" if member is None else f"{'SHUF' if shuffled else 'BAG'}{member}"
        )
        if member is None and shuffled:
            raise ValueError("no shuffled single arm")
        torch.manual_seed(init)
        head = self.sci.DeltaHead().to(self.device)
        if sum(p.numel() for p in head.parameters()) != 47744:
            raise RuntimeError("H213 head inventory mismatch")
        before = self.rt.tensor_hash(head.state_dict())
        witness = plan[0][0]
        before_score = self.score(head, witness)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.0001)
        losses = []
        gradient_sum = 0.0
        for cids in plan:
            self.check()
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for cid in cids:
                u = self.utility(cid, allowed, "FIT")
                if shuffled:
                    u = np.random.default_rng(
                        self.sci.stream(seed, outer, member, cid, "shuffle")
                    ).permutation(u)
                ref = int(self.native[cid].argmin())
                pred = head(self.x_for(cid), ref)
                loss = (
                    self.sci.balanced_loss(
                        pred,
                        torch.as_tensor(u, device=self.device, dtype=torch.float32),
                        ref,
                    )
                    / 4
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite loss")
                loss.backward()
                loss_value += float(loss.detach())
            grad = sum(
                float(p.grad.square().sum())
                for p in head.parameters()
                if p.grad is not None
            )
            if not np.isfinite(grad):
                raise RuntimeError("nonfinite gradient")
            gradient_sum += grad
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            self.check()
            optimizer.step()
            self.progress["updates"] += 1
            losses.append(loss_value)
        after = self.rt.tensor_hash(head.state_dict())
        if (
            before == after
            or gradient_sum <= 0
            or np.array_equal(before_score, self.score(head, witness))
        ):
            raise RuntimeError("no actual learned forward change")
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        proof = dict(
            seed=seed,
            outer=outer,
            arm=label,
            member=member,
            shuffled=shuffled,
            updates=128,
            gradient_squared_sum=gradient_sum,
            before_sha256=before,
            after_sha256=after,
            forward_changed=True,
            actual_device=str(next(head.parameters()).device),
            losses=losses,
            init_seed=init,
            fit_families=sorted(set(family.values())),
            bootstrap_families=population,
            bootstrap_counts=dict(Counter(population)),
            sampled_contexts_by_family=dict(
                Counter(family[cid] for batch in plan for cid in batch)
            ),
            context_plan_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest(),
        )
        name = f"{seed}-{outer}-{label}"
        b = io.BytesIO()
        torch.save(state, b)
        self.store.put("heads/" + name + ".pt", b.getvalue())
        self.store.put("proofs/" + name + ".json", (json.dumps(proof) + "\n").encode())
        return state, proof

    def predict(self, states, rows):
        heads = []
        for state in states:
            head = self.sci.DeltaHead().to(self.device)
            head.load_state_dict(state, strict=True)
            heads.append(head)
        answer = {}
        for row in rows:
            cid = row["context_sha256"]
            answer[cid] = self.sci.ensemble_mean([self.score(h, cid) for h in heads])
        return answer
