"""One monitored H198 CPU attempt: per-context test-time training (TTT) of a fresh copy of the r882 adapter on revealed
candidates only; Stage 1 (B-only {1,3,5} x {1e-5,3e-5,1e-4} grid selection) -> fsynced lock -> Stage 2 (pooled A for
TTT_K, TTT_SHUF, TTT_L2, plus descriptive DL-Val)."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import boto3
import numpy as np
import torch

import analysis as stat
from s3_arrays import upload
from treatment import (
    GRID,
    GRID_LABELS,
    K_GRID,
    L2_BETA,
    LR_GRID,
    adapter_forward,
    shuf_permutation,
    ttt_context_trajectory,
    validate_launch,
    verify_leakage_mask_invariance,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    with path.open("w") as f:
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
    spec = importlib.util.spec_from_file_location("h198_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cohorts(rt, provenance):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, val = packet["train"], packet["dev"]
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


def adapter_state(net, names):
    return {name: net.state_dict()[name].detach().cpu().clone() for name in names}


class Engine:
    def __init__(self, net, rt, names, frozen, guard, fused, rows):
        self.net, self.rt, self.names, self.frozen, self.guard = (
            net,
            rt,
            names,
            frozen,
            guard,
        )
        self.pdict = dict(net.named_parameters())
        self.params = [self.pdict[n] for n in names]
        self.initial = adapter_state(net, names)
        self.initial_hash = rt.tensor_hash(self.initial)
        self.pools, self.cached = {}, {}
        self.projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        with np.load(rt.CACHE, allow_pickle=False) as z:
            cids = list(map(str, z["cids"]))
            if len(cids) != 1310 or set(cids) != {r["context_sha256"] for r in rows}:
                raise RuntimeError("feature cache closure")
            tokens, masks = z["tokens_10x32"], z["mask_10"]
            self.tokens = {
                c: torch.as_tensor(tokens[i], dtype=torch.float32)
                for i, c in enumerate(cids)
            }
            self.masks = {
                c: torch.as_tensor(masks[i], dtype=torch.bool)
                for i, c in enumerate(cids)
            }

    def load(self, rows):
        for row in rows:
            cid = row["context_sha256"]
            if cid in self.pools:
                continue
            path = self.rt.INPUT / "pools" / (cid + ".npz")
            if sha(path) != row["pool_sha256"]:
                raise RuntimeError("TRAIN/DL-Val pool hash")
            with np.load(path, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                self.pools[cid] = dict(
                    program=prog,
                    proj16=(prog.astype(np.float64) @ self.projection).astype(
                        np.float32
                    ),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )

    def batch(self, cids, indexes):
        def stack(key):
            return np.stack(
                [self.pools[c][key][ix] for c, ix in zip(cids, indexes, strict=True)]
            )

        b, k = len(cids), len(indexes[0])
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
            torch.stack([self.tokens[c] for c in cids]),
            torch.stack([self.masks[c] for c in cids]),
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

    def forward(self, kwargs, tokens, masks, anchor):
        return adapter_forward(
            self.net,
            kwargs["prog_412"],
            kwargs["sos_prior"],
            tokens,
            masks,
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
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")
        if self.rt.tensor_hash({n: self.pdict[n] for n in self.names}) != self.initial_hash:
            raise RuntimeError("adapter state not bit-exact r882")

    def evaluate(self, rows, native=False, prove_identity=False):
        self.check()
        groups, results, raw = {}, {}, {}
        for row in rows:
            cid = row["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    kwargs, tokens, masks, _, _ = self.batch(
                        batch, [np.arange(k) for _ in batch]
                    )
                    if native:
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tokens, mm_mask_10=masks
                        )["total_cost"].float()
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        if prove_identity and not torch.equal(
                            scores, self.forward(kwargs, tokens, masks, anchor)
                        ):
                            raise RuntimeError("native/cached r882 costs differ")
                    else:
                        if any(cid not in self.cached for cid in batch):
                            anchor = self.anchored(kwargs)
                            for i, cid in enumerate(batch):
                                self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        anchor = tuple(
                            torch.stack([self.cached[c][j] for c in batch])
                            for j in range(2)
                        )
                        scores = self.forward(kwargs, tokens, masks, anchor)
                    costs = scores.numpy()
                    if not np.isfinite(costs).all():
                        raise RuntimeError("nonfinite prediction")
                    for cid, cost in zip(batch, costs, strict=True):
                        pool = self.pools[cid]
                        choice = int(np.argmin(cost))
                        results[cid] = float(
                            -pool["fitness"][choice] if pool["gate"][choice] else -2.0
                        )
                        raw[cid] = cost.copy()
        self.check()
        return {r["context_sha256"]: results[r["context_sha256"]] for r in rows}, raw

    def context_inputs(self, cid, fitness_arr=None, gate_arr=None):
        """Build single-context inputs exposing ONLY the 4 revealed candidate outcomes (indices 0..3).
        Unrevealed labels (indices 4..M-1) are never read or included."""
        pool = self.pools[cid]
        fit_src = pool["fitness"] if fitness_arr is None else fitness_arr
        gate_src = pool["gate"] if gate_arr is None else gate_arr
        m = len(pool["program"])
        rev_f = np.asarray(fit_src[:4], dtype=np.float32)
        rev_g = np.asarray(gate_src[:4], dtype=bool)
        if not np.isfinite(rev_f).all():
            raise RuntimeError("nonfinite revealed fitness")
        anchor_total, anchor_prior = self.cached[cid]
        return dict(
            prog_412=torch.as_tensor(pool["program"][None, :, :], dtype=torch.float32),
            sos_prior=torch.zeros((1, m, 2), dtype=torch.float32),
            tokens=self.tokens[cid].unsqueeze(0),
            masks=self.masks[cid].unsqueeze(0),
            scale=torch.full((1,), 0.4208, dtype=torch.float32),
            revealed_indices=torch.arange(4, dtype=torch.long).reshape(1, 4),
            revealed_raw_f=torch.as_tensor(rev_f[None, :], dtype=torch.float32),
            revealed_gate=torch.as_tensor(rev_g[None, :], dtype=torch.float32),
            anchor_total=anchor_total.unsqueeze(0),
            anchor_prior=anchor_prior.unsqueeze(0),
        )

    def realized_utility(self, cid, pick):
        """Read ground-truth outcome ONLY after argmin pick has been finalized."""
        pool = self.pools[cid]
        return float(-pool["fitness"][pick] if pool["gate"][pick] else -2.0)

    def oracle(self, rows):
        out = {}
        for row in rows:
            pool = self.pools[row["context_sha256"]]
            utility = stat.context_utility(pool["fitness"], pool["gate"])
            out[row["context_sha256"]] = float(np.max(utility))
        return out


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
        raise RuntimeError("CPU xbabe2 nice19 only")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = runtime()
    provenance = json.loads((HERE / "provenance.json").read_text())
    if pins["dope_composite_sha256"] != rt.DOPE:
        raise RuntimeError("DOPE composite pin mismatch")
    train, val = cohorts(rt, provenance)
    fold_doc = json.loads((HERE / "fold_manifest.json").read_text())
    folds = fold_doc["families_by_seed"]
    fams = [r["family_id"] for r in train]
    folds_by_seed = {
        s: stat.family_folds(fams, seed=stat.FOLD_SEEDS[s]) for s in stat.SEEDS
    }
    if (
        any(folds[str(s)] != folds_by_seed[s] for s in stat.SEEDS)
        or len({json.dumps(g) for g in folds_by_seed.values()}) != 3
    ):
        raise RuntimeError("fold manifest identity or non-distinct seed partitions")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")

    def interrupted(signum, frame):
        raise RuntimeError(f"signal {signum}; no scientific retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started, updates, reset_checks = time.monotonic(), 0, 0
    remote = {}
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        if len(names) != 47 or sum(p.numel() for p in params) != 29211:
            raise RuntimeError("trainable adapter inventory")
        initial = adapter_state(net, names)
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or any(m["p"] for m in census["adapter_attention_dropout"])
        ):
            raise RuntimeError("dropout/eval mode mismatch")
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    scored_contexts=0,
                    optimizer_updates=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    initial_adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                    frozen_sha256=frozen,
                ),
            )
            return

        client = boto3.client("s3", region_name="us-east-2")

        def store(key, arrays):
            receipt = upload(client, key + ".npz", arrays)
            remote[key] = receipt
            save(output / "remote-arrays.json", remote)
            return receipt

        reference = json.loads(Path(pins["reference"]).read_text())
        engine = Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(val)
        baseline, raw_val = engine.evaluate(val, native=True, prove_identity=True)
        baseline_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        if not np.array_equal(
            baseline_values, np.asarray(reference["utility"]["per_context"])
        ):
            raise RuntimeError("exact native 91 DL-Val r882 baseline")
        if float(np.mean(baseline_values)) != 1.1377941898164525:
            raise RuntimeError("exact native 91 DL-Val mean")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                families=19,
                mean=1.1377941898164525,
                bit_exact=True,
                initial_adapter_sha256=initial_hash,
            ),
        )
        store("native/VAL", raw_val)
        print("BASELINE PASS: 91/91 native r882 exact", flush=True)

        engine.load(train)
        train_r882, raw_train = engine.evaluate(train, native=True, prove_identity=True)
        store("native/TRAIN", raw_train)
        train_oracle = engine.oracle(train)
        val_r882 = [baseline[r["context_sha256"]] for r in val]
        val_oracle = list(engine.oracle(val).values())
        gap = stat.zero_update_gap(
            train,
            [train_r882[r["context_sha256"]] for r in train],
            [train_oracle[r["context_sha256"]] for r in train],
            val,
            val_r882,
            val_oracle,
        )
        save(output / "zero_update_gap.json", gap)

        # Runtime leakage-mask & 0-step bit-exactness self-check on first 8 TRAIN contexts
        leakage_checked = 0
        for row in train[:8]:
            cid = row["context_sha256"]
            c_in = engine.context_inputs(cid)
            with torch.no_grad():
                s0 = adapter_forward(
                    net,
                    c_in["prog_412"],
                    c_in["sos_prior"],
                    c_in["tokens"],
                    c_in["masks"],
                    c_in["scale"],
                    c_in["revealed_indices"],
                    c_in["revealed_raw_f"],
                    c_in["revealed_gate"],
                    c_in["anchor_total"],
                    c_in["anchor_prior"],
                )[0].numpy()
            if not np.array_equal(s0, raw_train[cid]):
                raise RuntimeError("TTT0 single-context forward != batched native r882")
            verify_leakage_mask_invariance(
                net,
                names,
                engine.params,
                engine.pdict,
                engine.initial,
                engine.initial_hash,
                rt,
                engine.context_inputs,
                cid,
                engine.pools[cid],
            )
            updates += 9 * 5
            reset_checks += 9
            leakage_checked += 1
        engine.check()
        print(f"LEAKAGE MASK & TTT0 PROOF PASS ({leakage_checked} contexts)", flush=True)

        # Stage 1: evaluate 3x3 TTT_K grid on B contexts (union of B_j across j=0..4 is the 1,219 TRAIN contexts)
        # No A selection or Stage-2 panel assembly occurs until ttt_selection_lock.json is written and fsynced.
        grid_utilities_by_cid = {}
        grid_drifts_by_cid = {}
        b_grid_cost_arrays = {}
        for idx_r, row in enumerate(train):
            cid = row["context_sha256"]
            c_in = engine.context_inputs(cid)
            u_vec = np.empty(len(GRID), dtype=np.float64)
            d_vec = np.empty(len(GRID), dtype=np.float64)
            for li, lr in enumerate(LR_GRID):
                snaps, steps_run = ttt_context_trajectory(
                    net,
                    names,
                    engine.params,
                    engine.pdict,
                    engine.initial,
                    engine.initial_hash,
                    rt,
                    c_in,
                    lr=lr,
                    k_steps=K_GRID,
                    arm="TTT_K",
                )
                updates += steps_run
                reset_checks += 1
                for ki, k in enumerate(K_GRID):
                    g_idx = ki * len(LR_GRID) + li
                    pick = snaps[k]["pick"]
                    u_vec[g_idx] = engine.realized_utility(cid, pick)
                    d_vec[g_idx] = snaps[k]["drift"]
                    b_grid_cost_arrays[f"{cid}__{GRID_LABELS[g_idx]}"] = snaps[k]["costs"]
            grid_utilities_by_cid[cid] = u_vec
            grid_drifts_by_cid[cid] = d_vec
            if (idx_r + 1) % 300 == 0 or idx_r + 1 == len(train):
                engine.check()
                print(f"STAGE-1 B-GRID {idx_r + 1}/{len(train)} contexts", flush=True)

        store("stage1/B_grid_costs", b_grid_cost_arrays)
        del b_grid_cost_arrays

        fold_sha = sha(HERE / "fold_manifest.json")
        lock = stat.stage1_lock(
            train, grid_utilities_by_cid, folds_by_seed, fold_sha, initial_hash
        )
        lock_path = output / "ttt_selection_lock.json"
        with lock_path.open("xb") as f:
            f.write(stat.lock_bytes(lock))
            f.flush()
            os.fsync(f.fileno())
        locked, lock_sha = stat.verify_stage1_lock(lock_path, fold_sha, initial_hash)
        print(
            "LOCK PASS",
            [c["grid_label"] for c in locked["cells"]],
            lock_sha[:12],
            flush=True,
        )

        # Stage 2: pooled-A evaluation after verified on-disk B lock
        index = {r["context_sha256"]: i for i, r in enumerate(train)}
        shape = (len(stat.SEEDS), len(train))
        panels = {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name in (*stat.ARMS, *GRID_LABELS)
        }
        drifts = {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name in ("TTT_K", "TTT_SHUF", "TTT_L2")
        }
        stage2_cost_arrays = {}

        # Cache TTT_L2 by (cid, k_star, lr_star) since TTT_L2 is deterministic per (cid, k, lr)
        l2_cache = {}
        for cell in locked["cells"]:
            stat.verify_stage1_lock(lock_path, fold_sha, initial_hash)
            seed = int(cell["seed"])
            outer = int(cell["outer"])
            si = stat.SEEDS.index(seed)
            g_idx = int(cell["grid_index"])
            k_star = int(cell["k_star"])
            lr_star = float(cell["lr_star"])
            a_fams = set(cell["a_families"])
            a_rows = [r for r in train if r["family_id"] in a_fams]
            for r in a_rows:
                cid = r["context_sha256"]
                pos = index[cid]
                c_in = engine.context_inputs(cid)
                panels["R882"][si, pos] = train_r882[cid]

                # TTT_K on A at locked (k*, lr*) and descriptive fixed grid
                panels["TTT_K"][si, pos] = float(grid_utilities_by_cid[cid][g_idx])
                drifts["TTT_K"][si, pos] = float(grid_drifts_by_cid[cid][g_idx])
                for gi, glabel in enumerate(GRID_LABELS):
                    panels[glabel][si, pos] = float(grid_utilities_by_cid[cid][gi])

                # TTT_SHUF on A at locked (k*, lr*) with seed-specific derangement
                perm = shuf_permutation(seed, cid)
                shuf_snaps, steps_shuf = ttt_context_trajectory(
                    net,
                    names,
                    engine.params,
                    engine.pdict,
                    engine.initial,
                    engine.initial_hash,
                    rt,
                    c_in,
                    lr=lr_star,
                    k_steps=(k_star,),
                    arm="TTT_SHUF",
                    shuf_perm=perm,
                )
                updates += steps_shuf
                reset_checks += 1
                panels["TTT_SHUF"][si, pos] = engine.realized_utility(
                    cid, shuf_snaps[k_star]["pick"]
                )
                drifts["TTT_SHUF"][si, pos] = shuf_snaps[k_star]["drift"]
                stage2_cost_arrays[f"s{seed}__A__{cid}__TTT_SHUF"] = shuf_snaps[k_star][
                    "costs"
                ]

                # TTT_L2 on A at locked (k*, lr*) with fixed beta=10.0
                l2_key = (cid, k_star, lr_star)
                if l2_key not in l2_cache:
                    l2_snaps, steps_l2 = ttt_context_trajectory(
                        net,
                        names,
                        engine.params,
                        engine.pdict,
                        engine.initial,
                        engine.initial_hash,
                        rt,
                        c_in,
                        lr=lr_star,
                        k_steps=(k_star,),
                        arm="TTT_L2",
                        beta=L2_BETA,
                    )
                    updates += steps_l2
                    reset_checks += 1
                    l2_cache[l2_key] = dict(
                        utility=engine.realized_utility(cid, l2_snaps[k_star]["pick"]),
                        drift=l2_snaps[k_star]["drift"],
                        costs=l2_snaps[k_star]["costs"],
                    )
                panels["TTT_L2"][si, pos] = l2_cache[l2_key]["utility"]
                drifts["TTT_L2"][si, pos] = l2_cache[l2_key]["drift"]
                stage2_cost_arrays[f"s{seed}__A__{cid}__TTT_L2"] = l2_cache[l2_key][
                    "costs"
                ]

            engine.check()
            print(
                f"STAGE-2 A CELL s{seed}-f{outer} ({len(a_rows)} contexts, {cell['grid_label']})",
                flush=True,
            )

        if any(not np.isfinite(p).all() for p in panels.values()):
            raise RuntimeError("incomplete pooled-A panel coverage")
        if any(not np.isfinite(d).all() for d in drifts.values()):
            raise RuntimeError("incomplete pooled-A drift coverage")

        native_panel = np.array([train_r882[r["context_sha256"]] for r in train])
        for si in range(len(stat.SEEDS)):
            if not np.array_equal(panels["R882"][si], native_panel):
                raise RuntimeError("pooled R882 != native r882")

        # Descriptive DL-Val (91 contexts) at each seed's B-modal (k*, lr*)
        val_panels = {
            name: np.full((len(stat.SEEDS), len(val)), np.nan, dtype=np.float64)
            for name in stat.ARMS
        }
        val_drifts = {
            name: np.full((len(stat.SEEDS), len(val)), np.nan, dtype=np.float64)
            for name in ("TTT_K", "TTT_SHUF", "TTT_L2")
        }
        for si, seed in enumerate(stat.SEEDS):
            v_cfg = locked["val_grid_by_seed"][str(seed)]
            vk = int(v_cfg["k_star"])
            vlr = float(v_cfg["lr_star"])
            for vi, r in enumerate(val):
                cid = r["context_sha256"]
                c_in = engine.context_inputs(cid)
                val_panels["R882"][si, vi] = baseline[cid]
                for arm in ("TTT_K", "TTT_SHUF", "TTT_L2"):
                    perm = shuf_permutation(seed, cid) if arm == "TTT_SHUF" else None
                    snaps, st_run = ttt_context_trajectory(
                        net,
                        names,
                        engine.params,
                        engine.pdict,
                        engine.initial,
                        engine.initial_hash,
                        rt,
                        c_in,
                        lr=vlr,
                        k_steps=(vk,),
                        arm=arm,
                        shuf_perm=perm,
                        beta=L2_BETA,
                    )
                    updates += st_run
                    reset_checks += 1
                    val_panels[arm][si, vi] = engine.realized_utility(
                        cid, snaps[vk]["pick"]
                    )
                    val_drifts[arm][si, vi] = snaps[vk]["drift"]
                    stage2_cost_arrays[f"s{seed}__VAL__{cid}__{arm}"] = snaps[vk][
                        "costs"
                    ]
        engine.check()
        store("stage2/A_and_VAL_costs", stage2_cost_arrays)
        del stage2_cost_arrays

        # Statistical analysis
        selector = stat.pooled_select(
            train, panels["R882"], {"TTT_K": panels["TTT_K"]}
        )
        ttt_k_vs_r882 = stat.contrast_stats(train, panels["TTT_K"], panels["R882"])
        ttt_k_vs_r882["holm_multiplicity"] = 1
        ttt_k_vs_r882["holm_p"] = ttt_k_vs_r882["p_one_sided"]
        ttt_shuf_vs_r882 = stat.contrast_stats(
            train, panels["TTT_SHUF"], panels["R882"]
        )
        ttt_l2_vs_r882 = stat.contrast_stats(train, panels["TTT_L2"], panels["R882"])
        ttt_k_minus_shuf = stat.contrast_stats(
            train, panels["TTT_K"], panels["TTT_SHUF"]
        )
        ttt_k_minus_l2 = stat.contrast_stats(train, panels["TTT_K"], panels["TTT_L2"])

        elos = {
            arm: stat.elo_pair(train, panels[arm], panels["R882"])
            for arm in ("TTT_K", "TTT_SHUF", "TTT_L2")
        }
        fixed_grid_descriptive = {
            glabel: stat.contrast_stats(train, panels[glabel], panels["R882"])
            for glabel in GRID_LABELS
        }
        minus2 = {
            k: [int(np.count_nonzero(v[s] == -2.0)) for s in range(len(stat.SEEDS))]
            for k, v in panels.items()
        }
        drift_summary = {
            arm: dict(
                per_seed_mean=[
                    float(drifts[arm][s].mean()) for s in range(len(stat.SEEDS))
                ],
                median_of_seed_means=float(
                    np.median([drifts[arm][s].mean() for s in range(len(stat.SEEDS))])
                ),
                overall_mean=float(drifts[arm].mean()),
                overall_max=float(drifts[arm].max()),
            )
            for arm in ("TTT_K", "TTT_SHUF", "TTT_L2")
        }
        stability = {
            glabel: sum(c["grid_index"] == i for c in locked["cells"])
            for i, glabel in enumerate(GRID_LABELS)
        }
        arm_utilities = {
            arm: dict(
                per_seed_equal_family_utility=[
                    stat.equal_family_mean(train, panels[arm][s])
                    for s in range(len(stat.SEEDS))
                ],
                median_over_seeds=float(
                    np.median(
                        [
                            stat.equal_family_mean(train, panels[arm][s])
                            for s in range(len(stat.SEEDS))
                        ]
                    )
                ),
                minus2_per_seed=minus2[arm],
            )
            for arm in stat.ARMS
        }

        receipt = json.loads(
            (
                REPO
                / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
            ).read_text()
        )
        h178_ok = stat.h178_gate(
            receipt,
            sha(REPO / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"),
        )
        decision = stat.verdict(
            ttt_k_vs_r882,
            ttt_shuf_vs_r882,
            ttt_k_minus_shuf,
            selector,
            h178_ok,
        )

        dl_val_summary = {}
        for arm in stat.ARMS:
            per_seed_u = [
                stat.equal_family_mean(val, val_panels[arm][s])
                for s in range(len(stat.SEEDS))
            ]
            base_u = [
                stat.equal_family_mean(val, val_panels["R882"][s])
                for s in range(len(stat.SEEDS))
            ]
            dl_val_summary[arm] = dict(
                per_seed_equal_family_utility=per_seed_u,
                median_equal_family_utility=float(np.median(per_seed_u)),
                per_seed_gain_vs_r882=[
                    float(a - b) for a, b in zip(per_seed_u, base_u, strict=True)
                ],
                median_gain_vs_r882=float(
                    np.median([a - b for a, b in zip(per_seed_u, base_u, strict=True)])
                ),
                minus2_per_seed=[
                    int(np.count_nonzero(val_panels[arm][s] == -2.0))
                    for s in range(len(stat.SEEDS))
                ],
            )

        final_hash = rt.tensor_hash(adapter_state(net, names))
        if final_hash != initial_hash:
            raise RuntimeError("final adapter hash != initial r882 hash")

        summary = dict(
            wave="H198",
            status="COMPLETE",
            verdict=decision["verdict"],
            disposition=decision["disposition"],
            decision=decision,
            arms=arm_utilities,
            ttt_k_selector=selector["TTT_K"],
            ttt_k_vs_r882=ttt_k_vs_r882,
            ttt_shuf_vs_r882=ttt_shuf_vs_r882,
            ttt_l2_vs_r882=ttt_l2_vs_r882,
            ttt_k_minus_ttt_shuf=ttt_k_minus_shuf,
            ttt_k_minus_ttt_l2=ttt_k_minus_l2,
            elo_vs_r882=elos,
            mean_adapter_drift_per_context=drift_summary,
            minus2_per_seed=minus2,
            grid_selection_stability=stability,
            fixed_grid_descriptive=fixed_grid_descriptive,
            cells=[
                {
                    k: c[k]
                    for k in (
                        "seed",
                        "outer",
                        "grid_index",
                        "grid_label",
                        "k_star",
                        "lr_star",
                        "b_equal_family_utility",
                    )
                }
                for c in locked["cells"]
            ],
            lock_sha256=lock_sha,
            pairwise_fold_jaccards=stat.fold_jaccards(folds_by_seed),
            dl_val_descriptive=dict(
                val_grid_by_seed=locked["val_grid_by_seed"],
                by_arm=dl_val_summary,
            ),
            h178_receipt_gate=h178_ok,
            leakage_mask_verified_contexts=leakage_checked,
            per_context_reset_hash_checks=reset_checks,
            total_ttt_sgd_steps=updates,
            l2_anchor_beta=L2_BETA,
            initial_adapter_sha256=initial_hash,
            final_adapter_sha256=final_hash,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            remote_s3_objects=len(remote),
            zero_update_gap={k: v for k, v in gap.items() if k != "family_values"},
            diagnostic_only=True,
            serving_promotion=False,
        )
        if len(json.dumps(summary, allow_nan=False).encode()) >= 100_000:
            raise RuntimeError("metrics would exceed 100 KB")
        save(output / "metrics.json", summary)
        np.savez_compressed(
            output / "panels.npz",
            **panels,
            TTT_K_drift=drifts["TTT_K"],
            TTT_SHUF_drift=drifts["TTT_SHUF"],
            TTT_L2_drift=drifts["TTT_L2"],
            VAL_R882=val_panels["R882"],
            VAL_TTT_K=val_panels["TTT_K"],
            VAL_TTT_SHUF=val_panels["TTT_SHUF"],
            VAL_TTT_L2=val_panels["TTT_L2"],
            context_ids=np.array([r["context_sha256"] for r in train]),
            family_ids=np.array([r["family_id"] for r in train]),
            seed_ids=np.array(stat.SEEDS),
        )
        store(
            "stage2/panels",
            dict(
                **panels,
                TTT_K_drift=drifts["TTT_K"],
                TTT_SHUF_drift=drifts["TTT_SHUF"],
                TTT_L2_drift=drifts["TTT_L2"],
                VAL_R882=val_panels["R882"],
                VAL_TTT_K=val_panels["TTT_K"],
                VAL_TTT_SHUF=val_panels["TTT_SHUF"],
                VAL_TTT_L2=val_panels["TTT_L2"],
            ),
        )
        save(
            output / "audit.json",
            dict(
                status="PASS",
                opened=audit.receipt()["opened"],
                refused=audit.receipt()["refused"],
                protected_reads=audit.receipt()["protected_reads"],
                total_ttt_sgd_steps=updates,
                per_context_reset_hash_checks=reset_checks,
                initial_adapter_sha256=initial_hash,
                final_adapter_sha256=final_hash,
            ),
        )
        print("COMPLETE", summary["verdict"], "stability", stability, flush=True)
    except BaseException as exc:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                total_ttt_sgd_steps=updates,
                per_context_reset_hash_checks=reset_checks,
                error=repr(exc),
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
