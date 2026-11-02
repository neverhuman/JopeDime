"""Frozen H136 COSWR recipe, bounded lazy inputs, and immutable per-cell predictions."""

import __future__
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import types
import numpy as np

H136 = Path("/home/ubuntu/jope-rescue-20260921/h136-fambal-cosine-run/run_h136_cpu.py")
H136_SHA = "b0eda24a960f31c52fc308f5b667f1b76c9a6a55dd0ce031898beeda4fc21072"
SOURCE = "e3c978d5d36f50a190b82543e592834b549e0844"


def function(raw, names, env):
    tree = ast.parse(raw)
    selected = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if {n.name for n in selected} != set(names):
        raise RuntimeError("recipe function inventory")
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            "frozen-recipe",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        env,
    )


def recipe():
    import torch
    import torch.nn.functional as F

    raw = H136.read_bytes()
    if hashlib.sha256(raw).hexdigest() != H136_SHA:
        raise RuntimeError("H136 recipe identity")
    env = dict(
        np=np,
        torch=torch,
        F=F,
        math=math,
        random=random,
        hashlib=hashlib,
        ROUNDS=20,
        STEPS_PER_ROUND=2,
        UPDATES_PER_CELL=40,
        BATCH_SIZE=16,
        SLATE_SIZE=192,
        LEARNING_RATE=1e-4,
        COSWR_WARMUP_UPDATES=4,
        COSWR_RESTART_PERIOD=40 / 3,
    )
    for lane, names in [
        ("h53b/runner.py", ["cached_training_indices"]),
        ("h61/runner.py", ["compute_candidate_pool_hash"]),
        ("h59/runner.py", ["preregistered_targets"]),
        ("h48/run.py", ["make_predictor"]),
    ]:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                "/home/ubuntu/JopeDime",
                "show",
                SOURCE + ":research/flywheel_selfplay/" + lane,
            ]
        )
        child = dict(env)
        function(source, names, child)
        env[lane.split("/")[0]] = types.SimpleNamespace(**{n: child[n] for n in names})
    function(
        raw,
        [
            "coswr_learning_rate",
            "paired_stream_seed",
            "build_control_or_coswr_batch_plan",
            "per_context_losses",
            "non_trainable_snapshot",
        ],
        env,
    )
    return types.SimpleNamespace(**env)


class LazyPool:
    def __init__(self, pools, cid, h124, projection):
        self.pools, self.cid, self.h124, self.projection = pools, cid, h124, projection

    def __getitem__(self, key):
        if key == "program":
            return self.h124.program(self.pools, self.cid)
        if key == "proj16":
            reader, spans, _ = self.projection
            return reader.rows(*spans[self.cid])
        return self.pools[self.cid][key]


class Controller:
    def __init__(
        self,
        h124,
        h139,
        helper,
        net,
        context,
        pools,
        projection,
        val_context,
        val_pools,
        val_projection,
        rows,
        val,
        root,
        store,
        check,
    ):
        self.h124, self.h139, self.helper, self.original = h124, h139, helper, net
        self.context, self.pools, self.projection = context, pools, projection
        self.val_context, self.val_pools, self.val_projection = (
            val_context,
            val_pools,
            val_projection,
        )
        self.rows, self.val, self.root, self.store, self.check = (
            rows,
            val,
            root,
            store,
            check,
        )
        self.recipe = recipe()
        self.receipts = {}
        self.cache_key, self.cache = None, None
        self.total_updates = 0
        (self.root / "coswr-updates.jsonl").open("x").close()
        self.guard = h124.source_module(
            "python/src/jope/north_star/dope_freeze_guard.py", "h230_dope_guard"
        )
        for p in pools.values():
            with np.load(p["path"], allow_pickle=False) as z:
                p["pids"] = z["pids"].copy()

    def train_cell(self, seed, outer, fit_groups, mapping):
        import torch

        key = (seed, outer)
        if key in self.receipts:
            return self.predictions(seed, outer)
        self.check()
        r, h = self.recipe, self.h124
        directory = self.root / "coswr" / f"seed{seed}-outer{outer}"
        directory.mkdir(parents=True, exist_ok=False)
        net = copy.deepcopy(self.original)
        for p in net.parameters():
            p.requires_grad_(False)
        params = []
        for name, p in net.adapter_10tok.named_parameters():
            if not name.startswith("anchor."):
                p.requires_grad_(True)
                params.append(p)
        if len(params) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("adapter trainability boundary")
        self.guard.lock_dope_weights_and_eval_mode(net)
        before, _, _ = r.non_trainable_snapshot(net)
        initial = torch.cat([p.detach().reshape(-1).clone() for p in params])
        ids = sorted(c for c in self.pools if mapping[c] in fit_groups)
        fit_pools = {c: self.pools[c] for c in ids}
        orders = {c: np.argsort(fit_pools[c]["fitness"]) for c in ids}
        plan = r.build_control_or_coswr_batch_plan(ids, fit_pools, orders, seed)
        if any(
            c not in fit_pools
            for step in plan
            for mb in step["microbatches"]
            for c in mb["context_ids"]
        ):
            raise RuntimeError("training outside FIT")
        tokens = {
            c: torch.as_tensor(self.context[c]["tokens_10x32"], dtype=torch.float32)
            for c in ids
        }
        masks = {
            c: torch.as_tensor(self.context[c]["mask_10"], dtype=torch.bool)
            for c in ids
        }
        lazy = {c: LazyPool(self.pools, c, h, self.projection) for c in ids}
        predict = r.h48.make_predictor(net, lazy, tokens, masks, torch.device("cpu"))
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        self.guard.assert_optimizer_excludes_dope(optimizer, net)
        losses, lrs, norms = [], [], []
        torch.set_num_threads(4)
        for step in plan:
            self.check()
            u = step["update_index"]
            lr = r.coswr_learning_rate(u)
            for group in optimizer.param_groups:
                group["lr"] = lr
            mb = step["microbatches"][0]
            torch.manual_seed(int(mb["stream_seed"]))
            cids, indices = mb["context_ids"], list(mb["candidate_indices"])
            target = r.h59.preregistered_targets(
                torch.as_tensor(
                    np.stack(
                        [fit_pools[c]["fitness"][ix] for c, ix in zip(cids, indices)]
                    ),
                    dtype=torch.float32,
                ),
                torch.as_tensor(
                    np.stack(
                        [fit_pools[c]["gate"][ix] for c, ix in zip(cids, indices)]
                    ),
                    dtype=torch.bool,
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = r.per_context_losses(predict(list(cids), indices), target).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite COSWR loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            self.total_updates += 1
            with (self.root / "coswr-updates.jsonl").open("a") as journal:
                journal.write(
                    json.dumps(
                        dict(
                            seed=seed,
                            outer=outer,
                            update=u + 1,
                            total_updates=self.total_updates,
                            lr=lr,
                            loss=float(loss.detach()),
                        )
                    )
                    + "\n"
                )
                journal.flush()
                os.fsync(journal.fileno())
            losses.append(float(loss.detach()))
            lrs.append(lr)
            norms.append(float(norm))
        after, _, _ = r.non_trainable_snapshot(net)
        dope = self.guard.snapshot_dope_weights(net).composite_sha256
        final = torch.cat([p.detach().reshape(-1) for p in params])
        drift = float(torch.linalg.vector_norm(final - initial))
        if (
            before != after
            or dope != h.DOPE
            or drift <= 0
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen/adapter change boundary")
        torch.save(
            {
                n: p.detach().cpu()
                for n, p in net.adapter_10tok.named_parameters()
                if not n.startswith("anchor.")
            },
            directory / "adapter.pt",
        )
        del optimizer, predict, params, tokens, masks, lazy
        h.release_memory()
        train_cost, _ = self.helper.bounded_threads(
            h.baseline,
            directory,
            self.rows,
            self.context,
            self.pools,
            self.projection,
            net=net,
        )
        val_cost, _ = self.helper.bounded_threads(
            h.baseline,
            directory,
            self.val,
            self.val_context,
            self.val_pools,
            self.val_projection,
            net=net,
        )
        np.savez_compressed(directory / "predictions.npz", **train_cost, **val_cost)
        receipt = dict(
            seed=seed,
            outer=outer,
            updates=40,
            fit_contexts=len(ids),
            fit_context_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            lrs=lrs,
            losses=losses,
            pre_clip_norms=norms,
            non_trainable_before=before,
            non_trainable_after=after,
            dope=dope,
            adapter_drift=drift,
            adapter_sha256=hashlib.sha256(
                (directory / "adapter.pt").read_bytes()
            ).hexdigest(),
            predictions_sha256=hashlib.sha256(
                (directory / "predictions.npz").read_bytes()
            ).hexdigest(),
            batch_pool_hashes=[
                x["microbatches"][0]["candidate_pool_sha256"] for x in plan
            ],
        )
        with (directory / "receipt.json").open("x") as f:
            json.dump(receipt, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        for path in sorted(directory.iterdir()):
            if path.is_file():
                self.store.put(path)
        self.receipts[key] = receipt
        del net, train_cost, val_cost
        h.release_memory()
        print("COSWR", seed, outer, "40 updates", flush=True)
        return self.predictions(seed, outer)

    def predictions(self, seed, outer):
        key = (seed, outer)
        if self.cache_key != key:
            path = self.root / "coswr" / f"seed{seed}-outer{outer}" / "predictions.npz"
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != self.receipts[key]["predictions_sha256"]
            ):
                raise RuntimeError("COSWR predictions changed")
            with np.load(path, allow_pickle=False) as z:
                self.cache = {c: z[c] for c in z.files}
            self.cache_key = key
        return self.cache
