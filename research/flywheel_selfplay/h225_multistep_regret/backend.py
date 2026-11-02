"""H225 adapter proposals using the pinned H222 CPU forward implementation."""

import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def make_engine(base, science):
    class Engine(base.Engine):
        def details(self, net, rows, shuffled=None, record_costs=None):
            groups, result = {}, {}
            for row in rows:
                cid = row["context_sha256"]
                groups.setdefault(len(self.pools.get(cid)["fitness"]), []).append(cid)
            with torch.no_grad():
                for size, ids in sorted(groups.items()):
                    for start in range(0, len(ids), 8):
                        self.check()
                        batch = ids[start : start + 8]
                        costs = self.predict(
                            net, batch, [np.arange(size) for _ in batch]
                        )[0].numpy()
                        if not np.isfinite(costs).all():
                            raise RuntimeError("nonfinite costs")
                        for cid, cost in zip(batch, costs, strict=True):
                            if record_costs is not None:
                                record_costs[cid] = cost.copy()
                            pool = self.pools.get(cid)
                            choice = int(np.argmin(cost))
                            labels = science.gaps(pool["fitness"], pool["gate"])
                            result[cid] = dict(
                                choice=choice,
                                utility=float(
                                    -pool["fitness"][choice]
                                    if pool["gate"][choice]
                                    else -2.0
                                ),
                                regret=science.regret(cost, labels),
                                shuffled_regret=science.regret(cost, shuffled[cid])
                                if shuffled is not None
                                else None,
                                costs_sha256=hashlib.sha256(cost.tobytes()).hexdigest(),
                                pool_sha256=self.pools.rows[cid]["pool_sha256"],
                            )
            return result

        def baseline(self, rows, shuffled=None):
            if not hasattr(self, "reference_cache"):
                self.reference_cache = {}
                self.reference_costs = {}
            missing = [
                r for r in rows if r["context_sha256"] not in self.reference_cache
            ]
            if missing:
                net, guard, mutable, frozen, _ = self.model("CTRL")
                self.reference_cache.update(
                    self.details(net, missing, record_costs=self.reference_costs)
                )
                self.rt.check_frozen(net, guard, mutable, frozen)
                del net, guard
                gc.collect()
            result = {
                r["context_sha256"]: dict(self.reference_cache[r["context_sha256"]])
                for r in rows
            }
            if shuffled is not None:
                for cid, value in result.items():
                    value["shuffled_regret"] = science.regret(
                        self.reference_costs[cid], shuffled[cid]
                    )
            return result

        def save(self, name, value):
            path = self.output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as f:
                json.dump(value, f, indent=2, allow_nan=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

        def fit_proposal(self, rows, config, seed, label, *, smoke=False):
            if (self.output / "selection-lock.json").exists():
                raise RuntimeError("optimizer forbidden after selection lock")
            if (
                config not in science.CONFIGS
                or not rows
                or any(r["dl_split"] != "DL-Train" for r in rows)
            ):
                raise ValueError("registered TRAIN-only proposal required")
            steps = 2 if smoke else config["steps"]
            allowed = {r["context_sha256"] for r in rows}
            net, guard, mutable, frozen, _ = self.model("CTRL")
            orders = {
                cid: np.argsort(self.pools.get(cid)["fitness"])
                for cid in sorted(allowed)
            }
            plan = science.batch_plan(rows, orders, seed, steps)
            before = self.rt.tensor_hash({n: net.state_dict()[n] for n in mutable})
            before_forward = self.details(net, rows[:1])
            params = [p for p in net.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                params, lr=1e-4 * config["lr_scale"], weight_decay=1e-4
            )
            losses, gradient_sum = [], 0.0
            start = time.time()
            for stream, cids, indices in plan:
                self.check()
                if not set(cids) <= allowed:
                    raise RuntimeError("FIT input closure violated")
                torch.manual_seed(stream)
                optimizer.zero_grad(set_to_none=True)
                with sdpa_kernel(SDPBackend.MATH):
                    pred, fitness, gates, _, _ = self.predict(net, cids, indices)
                    loss = self.rt.loss(pred, fitness, gates)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite objective")
                    loss.backward()
                gradient = sum(
                    float(p.grad.detach().square().sum())
                    for p in params
                    if p.grad is not None
                )
                if not np.isfinite(gradient):
                    raise RuntimeError("nonfinite gradient")
                gradient_sum += gradient
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                self.check()
                if self.progress["updates"] >= self.max_updates:
                    raise RuntimeError("registered optimizer-update cap")
                self.journal(label, len(losses), "intent")
                optimizer.step()
                self.progress["updates"] += 1
                self.journal(label, len(losses), "completed")
                losses.append(float(loss.detach()))
            self.rt.check_frozen(net, guard, mutable, frozen)
            state = {n: net.state_dict()[n].detach().clone() for n in mutable}
            after = self.rt.tensor_hash(state)
            after_forward = self.details(net, rows[:1])
            if before == after or gradient_sum <= 0 or before_forward == after_forward:
                raise RuntimeError("actual optimizer/forward-change proof absent")
            dest = self.output / "states" / (label + ".pt")
            dest.parent.mkdir(exist_ok=True)
            with dest.open("xb") as f:
                torch.save(state, f)
                f.flush()
                os.fsync(f.fileno())
            state_sha = self.rt.sha(dest)
            proof = dict(
                label=label,
                config=config,
                stream_seed=seed,
                steps=steps,
                fit_families=sorted({r["family_id"] for r in rows}),
                fit_contexts=len(rows),
                fit_context_ids_sha256=hashlib.sha256(
                    "\n".join(sorted(allowed)).encode()
                ).hexdigest(),
                adapter_before_sha256=before,
                adapter_after_sha256=after,
                gradient_squared_sum=gradient_sum,
                losses=losses,
                forward_changed=True,
                nonadapter_unchanged=True,
                dope_sha256_before=self.rt.DOPE,
                dope_sha256_after=guard.snapshot_dope_weights(net).composite_sha256,
                state_sha256=state_sha,
                state_path=str(dest),
                elapsed_seconds=time.time() - start,
            )
            self.save("proofs/" + label + ".json", proof)
            self.progress["fits"] += 1
            self.save("progress/" + label + ".json", dict(self.progress))
            print(
                "FIT",
                label,
                "updates",
                self.progress["updates"],
                "fits",
                self.progress["fits"],
                flush=True,
            )
            del net, guard, optimizer, params, state
            gc.collect()
            return dict(
                label=label,
                path=str(dest),
                sha256=state_sha,
                config=config,
                fit_families=proof["fit_families"],
            )

        def journal(self, label, step, event):
            with (self.output / "optimizer-events.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            event=event,
                            label=label,
                            step=step,
                            updates=self.progress["updates"],
                            at=time.time(),
                        )
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())

        def evaluate_proposal(self, state, rows, shuffled=None, *, role, forbidden=()):
            locked = bool(
                getattr(self, "selection_lock_sha256", None)
                and self.rt.sha(self.output / "selection-lock.json")
                == self.selection_lock_sha256
            )
            science.require_score_role(
                [r["family_id"] for r in rows],
                state["fit_families"],
                role,
                locked,
                forbidden,
            )
            if self.rt.sha(state["path"]) != state["sha256"]:
                raise RuntimeError("proposal state changed")
            net, guard, mutable, frozen, _ = self.model("CTRL")
            saved = torch.load(state["path"], map_location="cpu", weights_only=True)
            if set(saved) != set(mutable):
                raise RuntimeError("proposal parameter inventory differs")
            current = net.state_dict()
            current.update(saved)
            net.load_state_dict(current, strict=True)
            result = self.details(net, rows, shuffled)
            self.rt.check_frozen(net, guard, mutable, frozen)
            del net, guard, saved, current
            gc.collect()
            return result

    return Engine
