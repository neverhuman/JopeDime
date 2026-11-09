"""One CPU-only H264 wave: zero-update kNN residual memory on frozen r882 embeddings (KNN primary, KNN_SHUF within-family
shuffled-value placebo, CONTROL = r882), merged H256/H247/H170 runtime and frozen-anchor machinery, one fresh fold seed
264140. No optimizer and no parameter update. H239 A1 cooperative pauses and hash-verified per-cell resume. Decision
panel: DL-Val91 single-pick (hw_t1658)."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import analysis as stat
from treatment import adapter_embed, adapter_forward, validate_launch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAUSE = None
OUT = None


def cooperate():
    """H239 A1: block between optimizer steps, evaluation batches and pool loads while a pause is requested."""
    if PAUSE is not None:
        while PAUSE.exists():
            time.sleep(0.25)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def save_cell(name, obj):
    """Completed-cell checkpoint (H239 A1): fsynced JSON whose sha256 is recorded in progress.json."""
    save(OUT / name, obj)
    return sha(OUT / name)


def recover(prior, launch):
    """H239 A1 resume: reuse only hash-verified completed cells of the same manifest (at most two resumes)."""
    if prior is None:
        return {}
    prior = prior.resolve()
    if (
        str(prior) != launch.get("resume_from")
        or not 1 <= launch.get("resume_number", 0) <= 2
    ):
        raise RuntimeError("resume source binding")
    progress = json.loads((prior / "progress.json").read_text())
    if progress.get("manifest_sha256") != launch["manifest_sha256"]:
        raise RuntimeError("resume changed science")
    cells = {}
    for name, digest in progress["files"].items():
        if sha(prior / name) != digest:
            raise RuntimeError("resume cell hash: " + name)
        cells[name.removesuffix(".json")] = json.loads((prior / name).read_text())
    return cells


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
    spec = importlib.util.spec_from_file_location("h170_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Amendment A2: #2006 made the H146 runtime import `research.flywheel_selfplay.runtime_audit` with the JopeDime
    # root first on sys.path, binding `research` to that namespace early. The pinned H50 cpu_exact_fused.py then
    # cannot import JopeDimeMicro's `research.levels`. Drop the early binding; `module.Audit` keeps its class.
    for name in [
        n for n in sys.modules if n == "research" or n.startswith("research.")
    ]:
        del sys.modules[name]
    return module


def cohorts(rt):
    packet = json.loads((rt.INPUT / "selected.json").read_text())
    train, val = packet["train"], packet["dev"]
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
    def __init__(self, net, rt, names, frozen, guard, fused, rows):
        self.net, self.rt, self.names, self.frozen, self.guard = (
            net,
            rt,
            names,
            frozen,
            guard,
        )
        self.pools, self.cached = {}, {}
        self.projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if len(ids) != 1310 or set(ids) != {r["context_sha256"] for r in rows}:
                raise RuntimeError("feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            self.tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            self.masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }

    def load(self, rows):
        for row in rows:
            cooperate()
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

    def check(self):
        self.rt.check_frozen(self.net, self.guard, self.names, self.frozen)
        if (
            self.net.dope_model.training
            or self.net.adapter_10tok.anchor.training
            or torch.cuda.is_initialized()
        ):
            raise RuntimeError("frozen mode/device violation")

    def evaluate(self, rows, native=False, prove_identity=False):
        groups, result, raw = {}, {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    cooperate()
                    batch = cids[start : start + 8]
                    kwargs, tok, mask, _, _ = self.batch(
                        batch, [np.arange(k) for c in batch]
                    )
                    if native:
                        scores = self.net.adapter_10tok.predict_costs(
                            **kwargs, mm_tokens_10x32=tok, mm_mask_10=mask
                        )["total_cost"].float()
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        if prove_identity and not torch.equal(
                            scores, self.forward(kwargs, tok, mask, anchor)
                        ):
                            raise RuntimeError(
                                "step0 cached forward differs from native r882 bits"
                            )
                    else:
                        missing = [c for c in batch if c not in self.cached]
                        if missing:
                            anchor = self.anchored(kwargs)
                            for i, cid in enumerate(batch):
                                self.cached[cid] = tuple(t[i].clone() for t in anchor)
                        anchor = tuple(
                            torch.stack([self.cached[c][j] for c in batch])
                            for j in range(2)
                        )
                        scores = self.forward(kwargs, tok, mask, anchor)
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

    def embed(self, rows, shortlist_size):
        """One frozen r882 pass per context on its full candidate pool (same batches as evaluate): the full cost vector
        (bit-identical to evaluate's) plus the context embedding and the shortlisted candidates' query embeddings."""
        out, groups = {}, {}
        for r in rows:
            cid = r["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.no_grad():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    cooperate()
                    batch = cids[start : start + 8]
                    kwargs, tok, mask, _, _ = self.batch(
                        batch, [np.arange(k) for c in batch]
                    )
                    missing = [c for c in batch if c not in self.cached]
                    if missing:
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                    anchor = tuple(
                        torch.stack([self.cached[c][j] for c in batch])
                        for j in range(2)
                    )
                    costs = self.forward(kwargs, tok, mask, anchor).numpy()
                    pooled, query = adapter_embed(
                        self.net, kwargs["prog_412"], kwargs["sos_prior"], tok, mask
                    )
                    for i, cid in enumerate(batch):
                        cost = costs[i].copy()
                        short = stat.shortlist(cost, shortlist_size)
                        out[cid] = dict(
                            cost=cost,
                            short=short,
                            ctx=pooled[i].numpy().copy(),
                            cand=query[i].numpy()[short].copy(),
                        )
        self.check()
        return out


def bank(rows, emb, fitness_of=None):
    """Concatenated keys, families and owners (context, shortlist position) of the rows' shortlisted pairs; with
    `fitness_of`, also the residual values (FIT memory only)."""
    ks, fams, owners, vals = [], [], [], []
    for r in rows:
        cid = r["context_sha256"]
        e = emb[cid]
        ks.append(stat.keys(e["ctx"], e["cand"]))
        fams += [r["family_id"]] * len(e["short"])
        owners += [(cid, j) for j in range(len(e["short"]))]
        if fitness_of is not None:
            pool = fitness_of[cid]
            s = e["short"]
            vals.append(
                stat.residual_values(e["cost"][s], pool["fitness"][s], pool["gate"][s])
            )
    return (
        np.concatenate(ks).astype(np.float32),
        np.asarray(fams),
        owners,
        np.concatenate(vals) if vals else None,
    )


def shuffled(values, fams, seed, outer):
    """KNN_SHUF placebo values: the FIT residuals permuted within each family (fixed stream per seed and outer fold)."""
    out = values.copy()
    rng = np.random.default_rng(stat.stream_seed(seed, outer, "KNN_SHUF"))
    for f in sorted(set(fams.tolist())):
        ix = np.flatnonzero(fams == f)
        out[ix] = values[ix[rng.permutation(len(ix))]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume-from", type=Path)
    args = ap.parse_args()
    global PAUSE, OUT
    PAUSE = args.output.with_suffix(".pause.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] != "xbabe2"
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("CPU xbabe2 nice19 required")
    out = OUT = args.output.resolve()
    out.mkdir()  # Exclusive; never reuse completed/partial study outputs.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    source_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    validate_launch(
        launch,
        source_head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    recovered = recover(args.resume_from, launch)
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
    if {str(s): stat.folds(train, s) for s in stat.SEEDS} != json.loads(
        (HERE / "fold_manifest.json").read_text()
    )["families_by_seed"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | set(pins["assets"]) | {str(args.launch.resolve())}
    allowed |= {
        str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz")) for r in train + val
    }
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    started = time.monotonic()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        params = [p for p in net.parameters() if p.requires_grad]
        initial = {n: net.state_dict()[n].detach().clone() for n in names}
        initial_hash = rt.tensor_hash(initial)
        census = rt.dropout_census(net)
        if (
            sum(p.numel() for p in params) != 29211
            or len(names) != 47
            or census["adapter_dropout_modules"]
            or census["adapter_training"]
            or census["dope_training"]
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
        ):
            raise RuntimeError("adapter/mode inventory")
        for p in params:
            p.requires_grad_(
                False
            )  # Zero-update study: nothing can receive a gradient.
        rt.check_frozen(net, guard, names, frozen)
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
        engine = Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(
            val
        )  # No TRAIN outcome opens until the exact native DL-Val baseline passes.
        baseline, base_costs = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91 r882 baseline")
        save(
            out / "baseline.json",
            dict(**baseline, status="PASS", matches=91, mean=1.1377941898164525),
        )
        print("BASELINE PASS 91/91; cached/native eval forward BIT-EXACT", flush=True)
        if args.smoke:
            # Smoke: exercise the embedding pass and the exact search on DL-Val keys only (no fitness, no memory values).
            emb_val = engine.embed(val, stat.SHORTLIST)
            identity = all(
                np.array_equal(
                    emb_val[r["context_sha256"]]["cost"],
                    base_costs[r["context_sha256"]],
                )
                for r in val
            )
            if not identity:
                raise RuntimeError("embedding pass cost identity")
            vkeys, vfam, _, _ = bank(val, emb_val)
            vdist, _ = stat.knn_search(
                vkeys, vfam, vkeys, vfam, kmax=stat.K_MAX, cooperate=cooperate
            )
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    embedding_cost_identity=True,
                    search_shape=list(vdist.shape),
                    search_distance_range=[float(vdist.min()), float(vdist.max())],
                    optimizer_updates=0,
                    scored_contexts=91,
                    scored_policy="r882 only (Native91 reproduction)",
                    train_contexts_scored=0,
                    adapter_tensors=47,
                    trainable_parameters=29211,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    initial_adapter_sha256=initial_hash,
                    census=census,
                ),
            )
            return
        engine.load(train)
        emb = engine.embed(train + val, stat.SHORTLIST)
        for r in val:  # The embedding pass reproduces the Native91 costs bit for bit.
            if not np.array_equal(
                emb[r["context_sha256"]]["cost"], base_costs[r["context_sha256"]]
            ):
                raise RuntimeError("embedding pass cost identity")
        del base_costs
        print("EMBEDDINGS", len(emb), "contexts; shortlist", stat.SHORTLIST, flush=True)
        pools = engine.pools

        def scores(rows, emb_rows, dist, idx, values, configs):
            """Per context and config: (utility, choice, invalid) for r882 cost + lambda x kNN residual."""
            res, n = {}, stat.SHORTLIST
            for kt in {(k, t) for k, t, lam in configs if lam}:
                res[kt] = stat.knn_residual(dist, idx, values, *kt)
            table = np.empty((len(rows), len(configs)))
            choices = np.empty((len(rows), len(configs)), dtype=np.int64)
            for i, r in enumerate(rows):
                cid = r["context_sha256"]
                e, pool = emb_rows[cid], pools[cid]
                for j, (k, t, lam) in enumerate(configs):
                    resid = res[(k, t)][i * n : (i + 1) * n] if lam else None
                    choice = stat.pick(e["cost"], e["short"], resid, lam)
                    choices[i, j] = choice
                    table[i, j] = stat.utility(pool["fitness"], pool["gate"], choice)
            return table, choices

        memories, decisions = {}, {}
        # Phase 1: SELECT-fold choices for every cell, then ONE global fsync lock before any ASSESS or DL-Val read.
        for outer in range(5):
            parts = [stat.partition(train, outer, s) for s in stat.SEEDS]
            if any(
                [r["context_sha256"] for r in p[q]]
                != [r["context_sha256"] for r in parts[0][q]]
                for p in parts
                for q in range(3)
            ):
                raise RuntimeError("shared fold seed partition")
            fit, select, assess = parts[0]
            mkeys, mfam, _, mvals = bank(fit, emb, pools)
            if {r["family_id"] for r in fit} & {
                r["family_id"] for r in select + assess + val
            }:
                raise RuntimeError("family leakage")
            memories[outer] = (fit, select, assess, mkeys, mfam, mvals)
            qkeys, qfam, _, _ = bank(select, emb)
            dist, idx = stat.knn_search(qkeys, qfam, mkeys, mfam, cooperate=cooperate)
            panel, _ = scores(select, emb, dist, idx, mvals, stat.CONFIGS)
            for seed in stat.SEEDS:
                decisions[f"A{outer}-{seed}"] = stat.choose_config(
                    select, panel, stat.stream_seed(stat.BOOT_SEED, outer, seed, "KNN")
                )
            print(
                "SELECTED",
                outer,
                {
                    c: decisions[f"A{outer}-{s}"]["selected"]
                    for s in stat.SEEDS
                    for c in [s]
                },
                flush=True,
            )
        lock = out / "selection.json"
        save(
            lock,
            dict(decisions=decisions, configs=stat.CONFIGS, shortlist=stat.SHORTLIST),
        )
        lock_sha = sha(
            lock
        )  # Durable before ANY ASSESS or DL-Val utility of a kNN policy is computed.
        print("SELECTIONS LOCKED", lock_sha, flush=True)

        npol = len(stat.ALL_ARMS)
        primary = np.full((3, 1219, npol), np.nan)
        secondary = np.full((3, 5, 91, npol), np.nan)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        files, cells = {}, []
        for outer in range(5):
            fit, select, assess, mkeys, mfam, mvals = memories[outer]
            qrows = assess + val
            qkeys, qfam, _, _ = bank(qrows, emb)
            dist, idx = stat.knn_search(qkeys, qfam, mkeys, mfam, cooperate=cooperate)
            na = len(assess)
            for si, seed in enumerate(stat.SEEDS):
                name = f"A{outer}-{seed}"
                if name in recovered:
                    cell = recovered[name]
                else:
                    dec = decisions[name]
                    cfg = stat.CONFIGS[dec["selected_config"]]
                    configs = [stat.CONFIGS[0], cfg, cfg]
                    tab, ch = scores(qrows, emb, dist, idx, mvals, configs[:2])
                    stab, sch = scores(
                        qrows, emb, dist, idx, shuffled(mvals, mfam, seed, outer), [cfg]
                    )
                    util = np.column_stack(
                        [tab[:, 0], tab[:, 0], tab[:, 1], stab[:, 0]]
                    )
                    picks = np.column_stack([ch[:, 0], ch[:, 0], ch[:, 1], sch[:, 0]])
                    invalid = np.array(
                        [
                            [not pools[r["context_sha256"]]["gate"][c] for c in row]
                            for r, row in zip(qrows, picks)
                        ]
                    )
                    cell = dict(
                        outer_fold=outer,
                        seed=seed,
                        assess_ids=[r["context_sha256"] for r in assess],
                        decision=dec,
                        selection_sha256=lock_sha,
                        memory_pairs=int(len(mvals)),
                        memory_families=len({r["family_id"] for r in fit}),
                        assess_utility=util[:na].tolist(),
                        val_utility=util[na:].tolist(),
                        changed_choices={
                            a: {
                                "assess": int((picks[:na, ai] != picks[:na, 0]).sum()),
                                "val": int((picks[na:, ai] != picks[na:, 0]).sum()),
                            }
                            for ai, a in enumerate(stat.ALL_ARMS)
                        },
                        invalid_picks={
                            a: {
                                "assess": int(invalid[:na, ai].sum()),
                                "val": int(invalid[na:, ai].sum()),
                            }
                            for ai, a in enumerate(stat.ALL_ARMS)
                        },
                        val_choices=picks[na:].tolist(),
                    )
                    files[name + ".json"] = save_cell(name + ".json", cell)
                    save(
                        out / "progress.json",
                        dict(
                            complete_cells=len(files),
                            last_cell=name,
                            manifest_sha256=launch["manifest_sha256"],
                            files=files,
                        ),
                    )
                cells.append(cell)
                idxs = [positions[c] for c in cell["assess_ids"]]
                primary[si, idxs, :] = np.asarray(cell["assess_utility"])
                secondary[si, outer, :, :] = np.asarray(cell["val_utility"])
                print(
                    "CELL COMPLETE",
                    name,
                    "selected",
                    cell["decision"]["selected"],
                    flush=True,
                )
        if (
            len(cells) != 15
            or not np.isfinite(primary).all()
            or not np.isfinite(secondary).all()
        ):
            raise RuntimeError("incomplete wave")
        if any(not torch.equal(net.state_dict()[n], initial[n]) for n in names):
            raise RuntimeError("adapter changed in a zero-update study")
        if guard.snapshot_dope_weights(net).composite_sha256 != rt.DOPE:
            raise RuntimeError("DOPE drift")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        val_panel = secondary.mean(axis=1)
        metrics = dict(
            wave="H264",
            status="COMPLETE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            optimizer_updates=0,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            shortlist=stat.SHORTLIST,
            configs=stat.CONFIGS,
            primary_dlval=stat.decide(stat.summarize(val, val_panel)),
            secondary_oof=stat.summarize(train, primary),
            selected_configs={
                f"A{c['outer_fold']}-{c['seed']}": c["decision"]["selected"]
                for c in cells
            },
            best_configs={
                f"A{c['outer_fold']}-{c['seed']}": c["decision"]["best"] for c in cells
            },
            selector_gains={
                f"A{c['outer_fold']}-{c['seed']}": [
                    c["decision"]["gain"],
                    c["decision"]["ci95"],
                ]
                for c in cells
            },
            changed_choices={
                f"A{c['outer_fold']}-{c['seed']}": c["changed_choices"] for c in cells
            },
            invalid_picks={
                f"A{c['outer_fold']}-{c['seed']}": c["invalid_picks"] for c in cells
            },
            cliff_context=dict(
                index=stat.CLIFF_CONTEXT,
                context_sha256=val[stat.CLIFF_CONTEXT]["context_sha256"],
                family_id=val[stat.CLIFF_CONTEXT]["family_id"],
                r882=float(baseline["values"][stat.CLIFF_CONTEXT]),
                per_arm_by_seed=val_panel[:, stat.CLIFF_CONTEXT, :].tolist(),
                note="Reported separately and never excluded from any statistic (hw_t1658).",
            ),
            pseudoreplication_note=(
                "One shared fold seed: the KNN memory, search and scores are identical across seeds; seeds differ only in "
                "the selector bootstrap stream and the KNN_SHUF permutation."
            ),
            dope_before=rt.DOPE,
            dope_after=guard.snapshot_dope_weights(net).composite_sha256,
            adapter_unchanged=True,
            resume_number=launch.get("resume_number", 0),
            recovered_cells=sorted(recovered),
            selection_sha256=lock_sha,
            frozen_sha256=frozen,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        metrics["verdict"] = (
            "ADVANCE"
            if metrics["primary_dlval"]["contrasts"]["KNN_minus_r882"]["advance"]
            else "NO_ADVANCE"
        )
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=0,
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
