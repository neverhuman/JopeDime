"""One CPU-only H170 wave. Frozen-anchor caching follows merged H156; no width changes."""

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
from treatment import (
    adapter_forward,
    gradient_norm,
    adapter_drift,
    apply_settings,
    settings,
    MEANMATCH_LR,
    validate_launch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


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
    spec = importlib.util.spec_from_file_location("h170_runtime", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
    expected_schedule = json.loads((HERE / "schedules.json").read_text())
    actual_schedule = {
        a: [
            dict(step=t, lr=settings(a, t)[0], beta1=settings(a, t)[1], beta2=0.999)
            for t in range(1, 41)
        ]
        for a in stat.ARMS
    }
    if (
        expected_schedule["arms"] != actual_schedule
        or expected_schedule["meanmatch_lr"] != MEANMATCH_LR
    ):
        raise RuntimeError("schedule table identity")
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
    # Deliberate exact-path read after hook: regression for H153's launch custody failure.
    if json.loads(args.launch.read_text()) != launch:
        raise RuntimeError("launch mutation")
    updates, started = 0, time.monotonic()
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
                    trainable_parameters=29211,
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
        engine = Engine(net, rt, names, frozen, guard, fused, train + val)
        engine.load(
            val
        )  # No TRAIN outcomes opened until exact native DL-Val baseline passes.
        baseline, costs = engine.evaluate(val, native=True, prove_identity=True)
        if (
            not np.array_equal(
                np.asarray(baseline["values"]),
                np.asarray(ref["utility"]["per_context"]),
            )
            or float(np.mean(baseline["values"])) != 1.1377941898164525
        ):
            raise RuntimeError("exact 91 r882 baseline")
        np.savez_compressed(out / "baseline_costs.npz", **costs)
        save(
            out / "baseline.json",
            dict(
                **baseline,
                status="PASS",
                matches=91,
                mean=1.1377941898164525,
                eval_forward_bit_exact=True,
            ),
        )
        print("BASELINE PASS 91/91; cached/native eval forward BIT-EXACT", flush=True)
        del costs
        engine.load(train)
        primary = np.full((3, 1219, 5), np.nan)
        secondary = np.full((3, 5, 91, 5), np.nan)
        unconstrained = np.full_like(primary, np.nan)
        unconstrained_val = np.full_like(secondary, np.nan)
        positions = {r["context_sha256"]: i for i, r in enumerate(train)}
        selections, cells = [], []

        def restore(snapshot):
            with torch.no_grad():
                for n, p in net.named_parameters():
                    if n in snapshot:
                        p.copy_(snapshot[n])
            engine.check()

        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fitids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            if fitids & excluded:
                raise RuntimeError("family leakage")
            restore(initial)
            selbase, _ = engine.evaluate(select)
            # Baseline assessment is frozen r882 only, independent of candidate selection.
            abase, _ = engine.evaluate(assess)
            indices = [positions[r["context_sha256"]] for r in assess]
            orders = {c: np.argsort(engine.pools[c]["fitness"]) for c in fitids}
            for si, seed in enumerate(stat.SEEDS):
                cellname = f"A{outer}-{seed}"
                batch_cache, plan_digest = [], hashlib.sha256()
                for stream, cids, indexes in stat.batch_plan(fitids, orders, seed):
                    if not set(cids) <= fitids or set(cids) & excluded:
                        raise RuntimeError("fit-only minibatches")
                    kwargs, tok, mask, fitness, gate = engine.batch(cids, indexes)
                    batch_cache.append(
                        (
                            stream,
                            kwargs,
                            tok,
                            mask,
                            engine.anchored(kwargs),
                            fitness,
                            gate,
                        )
                    )
                    plan_digest.update(
                        json.dumps(
                            [stream, cids, [ix.tolist() for ix in indexes]]
                        ).encode()
                    )
                snapshots, selector_panels, traces = {}, {}, {}
                drifts, no_grad_names = {}, {}
                for arm in stat.ARMS:
                    restore(initial)
                    torch.manual_seed(stat.stream_seed(seed, outer, arm))
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    snapshots[arm], selector_panels[arm], traces[arm] = (
                        {0: initial},
                        [selbase],
                        [],
                    )
                    for step, (
                        stream,
                        kwargs,
                        tok,
                        mask,
                        anchor,
                        fitness,
                        gate,
                    ) in enumerate(batch_cache, 1):
                        learning_rate, beta1 = apply_settings(optimizer, arm, step)
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        pred = engine.forward(kwargs, tok, mask, anchor)
                        objective = rt.loss(pred, fitness, gate)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite objective")
                        objective.backward()
                        norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                params, 1.0, error_if_nonfinite=True
                            )
                        )
                        clipped = gradient_norm(params)
                        missing = [
                            n
                            for n, p in net.named_parameters()
                            if n in initial and p.grad is None
                        ]
                        no_grad_names.setdefault(arm, set()).add(tuple(missing))
                        inactive_elements = sum(
                            p.numel()
                            for n, p in net.named_parameters()
                            if n in initial and p.grad is None
                        )
                        optimizer.step()
                        updates += 1
                        engine.check()
                        traces[arm].append(
                            dict(
                                step=step,
                                loss=float(objective.detach()),
                                raw_grad_l2=norm,
                                clipped_grad_l2=clipped,
                                learning_rate=optimizer.param_groups[0]["lr"],
                                beta1=optimizer.param_groups[0]["betas"][0],
                                beta2=optimizer.param_groups[0]["betas"][1],
                                none_gradient_tensors=len(missing),
                                none_gradient_elements=inactive_elements,
                            )
                        )
                        if step in stat.STEPS:
                            snapshot = {
                                n: net.state_dict()[n].detach().clone() for n in names
                            }
                            snapshots[arm][step] = snapshot
                            np.savez_compressed(
                                out / f"{cellname}-{arm}-step{step}.npz",
                                **{n: t.numpy() for n, t in snapshot.items()},
                            )
                            panel, _ = engine.evaluate(select)
                            selector_panels[arm].append(panel)
                    drifts[arm] = adapter_drift(initial, snapshots[arm][40])
                    if rt.tensor_hash(snapshots[arm][40]) == initial_hash:
                        raise RuntimeError("unchanged adapter")
                decisions = {
                    arm: stat.choose(
                        select,
                        np.array([p["values"] for p in selector_panels[arm]]).T,
                        stat.stream_seed(stat.BOOT_SEED, outer, seed, arm),
                    )
                    for arm in stat.ARMS
                }
                lock = out / (cellname + "-selection.json")
                save(
                    lock,
                    dict(
                        decisions=decisions,
                        fit_ids=sorted(fitids),
                        select_ids=[r["context_sha256"] for r in select],
                        assess_ids=[r["context_sha256"] for r in assess],
                        batch_plan_sha256=plan_digest.hexdigest(),
                        optimizer_schedule_sha256=sha(HERE / "schedules.json"),
                        step40_drift=drifts,
                    ),
                )
                lock_sha = sha(
                    lock
                )  # Durable before ANY trained-policy outer-A/DL-Val assessment.
                primary[si, indices, 0], unconstrained[si, indices, 0] = (
                    abase["values"],
                    abase["values"],
                )
                secondary[si, outer, :, 0], unconstrained_val[si, outer, :, 0] = (
                    baseline["values"],
                    baseline["values"],
                )
                assessment = {}
                for ai, arm in enumerate(stat.ARMS, 1):
                    assessment[arm] = {0: dict(outer=abase, val=baseline)}
                    for step in sorted(
                        {decisions[arm]["selected_step"], decisions[arm]["best_step"]}
                        - {0}
                    ):
                        restore(snapshots[arm][step])
                        a, _ = engine.evaluate(assess)
                        v, _ = engine.evaluate(val)
                        assessment[arm][step] = dict(outer=a, val=v)
                    chosen = assessment[arm][decisions[arm]["selected_step"]]
                    best = assessment[arm][decisions[arm]["best_step"]]
                    primary[si, indices, ai] = chosen["outer"]["values"]
                    secondary[si, outer, :, ai] = chosen["val"]["values"]
                    unconstrained[si, indices, ai] = best["outer"]["values"]
                    unconstrained_val[si, outer, :, ai] = best["val"]["values"]
                cell = dict(
                    outer_fold=outer,
                    seed=seed,
                    decisions=decisions,
                    selection_sha256=lock_sha,
                    selector_panels=selector_panels,
                    assessment=assessment,
                    traces=traces,
                    step40_drift=drifts,
                    none_gradient_names={
                        a: [list(group) for group in sorted(groups)]
                        for a, groups in no_grad_names.items()
                    },
                    adapter_sha256={
                        a: {str(s): rt.tensor_hash(w) for s, w in snaps.items()}
                        for a, snaps in snapshots.items()
                    },
                )
                save(out / (cellname + ".json"), cell)
                cells.append(cell)
                selections.append(decisions)
                save(
                    out / "progress.json",
                    dict(
                        complete_cells=len(selections),
                        optimizer_updates=updates,
                        last_cell=cellname,
                    ),
                )
                print(
                    "CELL COMPLETE",
                    cellname,
                    "selected",
                    {a: decisions[a]["selected_step"] for a in stat.ARMS},
                    flush=True,
                )
                restore(initial)
                del batch_cache, snapshots
        if updates != 2400 or len(selections) != 15:
            raise RuntimeError("incomplete wave")
        val_panel, val_unconstrained = (
            secondary.mean(axis=1),
            unconstrained_val.mean(axis=1),
        )
        np.savez_compressed(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            unconstrained=unconstrained,
            unconstrained_val=unconstrained_val,
        )
        metrics = dict(
            wave="H170",
            status="COMPLETE",
            arms=stat.ALL_ARMS,
            seeds=stat.SEEDS,
            optimizer_updates=updates,
            complete_cells=15,
            baseline_mean=1.1377941898164525,
            eval_forward_bit_exact=True,
            primary=stat.summarize(train, primary),
            secondary=stat.summarize(val, val_panel),
            descriptive_unconstrained=stat.summarize(train, unconstrained),
            descriptive_unconstrained_val=stat.summarize(val, val_unconstrained),
            selections=selections,
            meanmatch_learning_rate=MEANMATCH_LR,
            schedule_table=json.loads((HERE / "schedules.json").read_text()),
            step40_drift={
                a: {
                    "per_cell": [c["step40_drift"][a] for c in cells],
                    "median": float(np.median([c["step40_drift"][a] for c in cells])),
                    "min": min(c["step40_drift"][a] for c in cells),
                    "max": max(c["step40_drift"][a] for c in cells),
                }
                for a in stat.ARMS
            },
            pseudoreplication=stat.pseudoreplication(
                primary, val_panel, unconstrained, val_unconstrained, cells
            ),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            trainable_parameters=29211,
            initial_adapter_sha256=initial_hash,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            serving_promotion=False,
            audit=audit.receipt(),
            elapsed_seconds=time.monotonic() - started,
        )
        metrics["verdict"] = (
            "ADVANCE"
            if any(
                metrics["primary"]["contrasts"][a + "_minus_r882"]["advance"]
                for a in stat.ARMS[1:]
            )
            else "NO_ADVANCE"
        )
        save(out / "metrics.json", metrics)
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as exc:
        save(
            out / "failure.json",
            dict(
                status="INCOMPLETE",
                optimizer_updates=updates,
                error=repr(exc),
                audit=audit.receipt(),
            ),
        )
        raise


if __name__ == "__main__":
    main()
