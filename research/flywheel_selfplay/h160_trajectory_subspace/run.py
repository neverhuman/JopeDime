"""Single H160 wave: trajectory-subspace projected continuation on the H148 family cross-fit harness.

Reuses the hash-pinned H146 inference/freeze/loss implementation exactly as merged H148 does.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import analysis as stat

HERE = Path(__file__).resolve().parent
DATA = Path("/home/ubuntu/jope-data/h160")
INPUTS = DATA / "inputs"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify(pins):
    for kind in ("inputs", "assets"):
        for p, h in pins[kind].items():
            if sha(p) != h:
                raise RuntimeError("identity mismatch: " + p)
    for p, h in pins["source_sha256"].items():
        if sha(HERE / p) != h:
            raise RuntimeError("source identity mismatch: " + p)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version: " + name)


def parent_executor():
    parent = HERE.parent / "h146_lookahead_exec"
    sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("h160_parent", parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module, sys.modules["runtime"]


def chain_basis(torch, names, initial):
    """Adapter vectors of the accepted chain (pinned owned copies), checked frozen-identical to r882 outside the adapter."""
    vectors = []
    for rnd in stat.CHAIN:
        state = torch.load(
            INPUTS / f"R{rnd}.pt", map_location="cpu", weights_only=True
        )["state_dict"]
        if set(state) != set(initial):
            raise RuntimeError(f"R{rnd} key set differs")
        for n, t in initial.items():
            if n not in names and not torch.equal(state[n], t):
                raise RuntimeError(f"R{rnd} non-adapter tensor differs: {n}")
        vectors.append(
            torch.cat([state[n].double().reshape(-1) for n in names]).numpy()
        )
    if not np.array_equal(
        vectors[-1], torch.cat([initial[n].double().reshape(-1) for n in names]).numpy()
    ):
        raise RuntimeError("chain end must be the r882 adapter")
    bases, receipts = {}, {}
    for arm, k in stat.K.items():
        u, info = stat.basis(np.stack(vectors), k)
        bases[arm] = u
        receipts[arm] = dict(
            info,
            basis_sha256=hashlib.sha256(np.ascontiguousarray(u).tobytes()).hexdigest(),
        )
    return bases, receipts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Exclusive, no result/cache reuse or automatic retry.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)  # Before importing model code or opening any pools.
    parent, rt = parent_executor()
    import torch

    torch.set_num_threads(1)
    try:  # The parent import may already have initialized the interop pool; the supervisor caps total threads.
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    train, val = parent.selected()
    if (
        len(train) != 1219
        or len({r["family_id"] for r in train}) != 205
        or len(val) != 91
    ):
        raise RuntimeError("TRAIN/VAL inventory")
    groups = stat.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"])
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
    started = time.monotonic()
    try:
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA forbidden")
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective inference/training mode changed")
        rt.check_frozen(net, guard, names, frozen)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        bases, basis_receipts = chain_basis(torch, names, initial)
        np.savez_compressed(out / "basis.npz", **{a: u for a, u in bases.items()})
        if args.smoke:
            probe = np.random.default_rng(0).standard_normal(29211)
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    trainable_parameters=sum(
                        p.numel() for p in net.parameters() if p.requires_grad
                    ),
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    basis=basis_receipts,
                    projection_idempotence_error={
                        a: float(
                            np.abs(
                                stat.project(u, stat.project(u, probe))
                                - stat.project(u, probe)
                            ).max()
                        )
                        for a, u in bases.items()
                    },
                    torch_threads=torch.get_num_threads(),
                    interop_threads=torch.get_num_interop_threads(),
                    packages=pins["packages"],
                ),
            )
            return
        launch = json.loads(args.launch.read_text())
        if (
            launch["manifest_sha256"] != sha(HERE / "manifest.json")
            or not launch["merged_source_verified"]
            or not launch["argon_review_pass"]
        ):
            raise RuntimeError("merged preregistration and rd_argon PASS required")
        if (
            launch["source_commit"]
            != subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
            ).strip()
        ):
            raise RuntimeError("launch source HEAD mismatch")
        ref = json.loads(Path(pins["reference"]).read_text())
        contract = ref["evaluation_contract"]
        expected = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            contract["contexts"] != expected
            or hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation identity")
        pools = {}
        for row in train + val:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool identity")
            with np.load(path, allow_pickle=False) as z:
                pools[cid] = dict(
                    program=z["program"].astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for p in pools.values():
            p["proj16"] = (p["program"].astype(np.float64) @ projection).astype(
                np.float32
            )
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }

        def evaluate(rows):
            result = parent.evaluate(net, pools, tokens, masks, rows)
            rt.check_frozen(net, guard, names, frozen)
            return result

        # Regression first: exact 91/91 r882 parity before any update or TRAIN scoring.
        baseline = evaluate(val)
        sanity = parent.baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        save(out / "basis.json", basis_receipts)
        params = dict(net.named_parameters())
        plist = [params[n] for n in names]
        sizes = [params[n].numel() for n in names]
        theta0 = torch.cat([initial[n].double().reshape(-1) for n in names])
        torch_bases = {
            a: torch.as_tensor(u, dtype=torch.float64) for a, u in bases.items()
        }

        def load_adapter(vec):
            with torch.no_grad():
                for n, chunk in zip(names, torch.split(vec, sizes), strict=True):
                    params[n].copy_(chunk.reshape(params[n].shape).to(params[n].dtype))

        def adapter_vec():
            return torch.cat([params[n].detach().double().reshape(-1) for n in names])

        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        cols = ("r882",) + stat.ARMS
        primary = np.full((3, len(train), len(cols)), np.nan)
        secondary = np.full((3, 5, len(val), len(cols)), np.nan)
        base_select, base_assess = {}, {}
        cells, drifts, nonzero = (
            [],
            {a: [] for a in stat.ARMS},
            {a: 0 for a in stat.ARMS},
        )
        out_of_span = {a: [] for a in stat.PRIMARY}
        lr_by_arm = {
            "CTRL": stat.BASE_LR,
            "PROJ_K2": stat.BASE_LR,
            "PROJ_K4": stat.BASE_LR,
        }
        for arm in stat.ARMS:
            if arm == "CTRL_DRIFT":
                lr_by_arm[arm] = stat.drift_lr(drifts["PROJ_K4"], drifts["CTRL"])
                save(
                    out / "ctrl_drift_lr.json",
                    dict(
                        lr=lr_by_arm[arm],
                        proj_k4=drifts["PROJ_K4"],
                        ctrl=drifts["CTRL"],
                    ),
                )
            u = torch_bases.get(arm)
            for outer in range(5):
                fit, select, assess = stat.partition(train, outer)
                fit_ids = {r["context_sha256"] for r in fit}
                excluded = {r["context_sha256"] for r in select + assess + val}
                assert not fit_ids & excluded
                load_adapter(theta0)
                rt.check_frozen(net, guard, names, frozen)
                if (
                    outer not in base_select
                ):  # r882 step-0 scores on B and A, computed once per outer.
                    base_select[outer] = evaluate(select)
                    base_assess[outer] = evaluate(assess)
                orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
                for si, seed in enumerate(stat.SEEDS):
                    tick = time.monotonic()
                    net.load_state_dict(initial, strict=True)
                    rt.check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(
                        plist, lr=lr_by_arm[arm], weight_decay=stat.WEIGHT_DECAY
                    )
                    snapshots, selector_scores, losses, batches = {}, [], [], []
                    cellname = f"{arm}-A{outer}-{seed}"

                    def snapshot(step):
                        vec = adapter_vec()
                        snapshots[step] = vec.clone()
                        np.savez_compressed(
                            out / f"{cellname}-step{step}.npz",
                            adapter=vec.float().numpy(),
                        )
                        selector_scores.append(
                            base_select[outer] if step == 0 else evaluate(select)
                        )

                    snapshot(0)
                    plan = stat.batch_plan(fit_ids, orders, seed)
                    if len(plan) != 40:
                        raise RuntimeError("40-update plan required")
                    for step, (stream, cids, ixs) in enumerate(plan, 1):
                        if not set(cids) <= fit_ids or set(cids) & excluded:
                            raise RuntimeError("minibatch family leakage")
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        pred = parent.predict(net, pools, tokens, masks, cids, ixs)
                        fitness = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                        )
                        gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.bool,
                        )
                        objective = rt.loss(pred, fitness, gate)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite objective")
                        objective.backward()
                        if u is not None:  # gradient projected onto span(U_k)
                            g = torch.cat([p.grad.double().reshape(-1) for p in plist])
                            gp = u @ (u.T @ g)
                            for p, chunk in zip(
                                plist, torch.split(gp, sizes), strict=True
                            ):
                                p.grad.copy_(chunk.reshape(p.shape).to(p.grad.dtype))
                        torch.nn.utils.clip_grad_norm_(
                            plist, 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        if (
                            u is not None
                        ):  # displacement kept in r882 + span(U_k) (AdamW scaling/decay would leak)
                            d = adapter_vec() - theta0
                            load_adapter(theta0 + u @ (u.T @ d))
                        updates += 1
                        rt.check_frozen(net, guard, names, frozen)
                        losses.append(float(objective.detach()))
                        batches.append(
                            dict(
                                step=step,
                                stream=stream,
                                context_ids=cids,
                                indices_sha256=hashlib.sha256(
                                    np.asarray(ixs, dtype="<i8").tobytes()
                                ).hexdigest(),
                            )
                        )
                        if step in stat.STEPS:
                            snapshot(step)
                    final = adapter_vec() - theta0
                    drift = float(torch.linalg.vector_norm(final))
                    if drift <= 0:
                        raise RuntimeError("no optimizer parameter drift")
                    drifts[arm].append(drift)
                    if u is not None:
                        out_of_span[arm].append(
                            stat.out_of_span(bases[arm], final.numpy())
                        )
                    panel = np.array([r["values"] for r in selector_scores]).T
                    decision = stat.choose(
                        select, panel, stat.selection_seed(outer, si, arm)
                    )
                    lockpath = out / f"{cellname}-selection.json"
                    save(
                        lockpath, decision
                    )  # Durable lock precedes every outer-A prediction in this cell.
                    step = decision["selected_step"]
                    if step:
                        nonzero[arm] += 1
                        load_adapter(snapshots[step])
                        assessed, validated = evaluate(assess), evaluate(val)
                    else:
                        assessed, validated = base_assess[outer], baseline
                    ai = [train_index[r["context_sha256"]] for r in assess]
                    col = cols.index(arm)
                    primary[si, ai, col] = assessed["values"]
                    primary[si, ai, 0] = base_assess[outer]["values"]
                    secondary[si, outer, :, col] = validated["values"]
                    secondary[si, outer, :, 0] = baseline["values"]
                    cell = dict(
                        arm=arm,
                        outer_fold=outer,
                        selection_fold=(outer + 1) % 5,
                        seed=seed,
                        lr=lr_by_arm[arm],
                        updates=40,
                        fit_contexts=sorted(fit_ids),
                        losses=losses,
                        batches=batches,
                        adapter_l2_drift=drift,
                        out_of_span=out_of_span[arm][-1] if u is not None else None,
                        snapshot_sha256={
                            str(s): hashlib.sha256(v.numpy().tobytes()).hexdigest()
                            for s, v in snapshots.items()
                        },
                        selection_lock_sha256=sha(lockpath),
                        decision=decision,
                        selector_scores=[r["values"] for r in selector_scores],
                        assessment=assessed["values"],
                        validation=validated["values"],
                        frozen_sha256=frozen,
                        dope_sha256=rt.DOPE,
                        wall_seconds=time.monotonic() - tick,
                    )
                    save(out / f"{cellname}.json", cell)
                    cells.append(
                        dict(
                            arm=arm,
                            outer=outer,
                            seed=seed,
                            selected_step=step,
                            drift=drift,
                        )
                    )
                    print(
                        "CELL",
                        cellname,
                        "updates",
                        updates,
                        "pick",
                        step,
                        "drift",
                        round(drift, 6),
                        "seconds",
                        round(cell["wall_seconds"], 1),
                        flush=True,
                    )
        match = stat.drift_match(drifts["PROJ_K4"], drifts["CTRL_DRIFT"])
        psummary = stat.summarize(train, primary)
        ssummary = stat.summarize(val, secondary.mean(axis=1))
        decision = stat.decide(psummary, nonzero, match)
        metrics = dict(
            wave="H160",
            status="COMPLETE",
            verdict=decision["verdict"],
            mechanism=decision["mechanism"],
            decision=decision,
            primary=psummary,
            secondary_dl_val_descriptive_only=ssummary,
            nonzero_selections=nonzero,
            cells_per_arm=15,
            drift_median={a: float(np.median(v)) for a, v in drifts.items()},
            drift_match=match,
            ctrl_drift_lr=lr_by_arm["CTRL_DRIFT"],
            out_of_span_max={a: float(max(v)) for a, v in out_of_span.items()},
            basis=basis_receipts,
            optimizer_updates=updates,
            baseline=sanity,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if updates != 2400 or len(cells) != 60:
            raise RuntimeError("incomplete wave")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        verify(pins)
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], metrics["mechanism"], flush=True)
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
