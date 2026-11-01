"""H164 inference and SAM primitives reused from merged H153; fixed cross-fit execution."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
import science as stat
from launch import read_launch
from runtime import (
    HERE,
    INPUT,
    CACHE,
    DOPE,
    Audit,
    build,
    check_frozen,
    dropout_census,
    loss,
    save,
    sha,
    tensor_hash,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def verify(pins):
    for name, version in pins.get("packages", {}).items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version mismatch: " + name)
    for p, h in pins["inputs"].items():
        if sha(p) != h:
            raise RuntimeError("input identity mismatch: " + p)
    for p, h in pins.get("assets", {}).items():
        if sha(p) != h:
            raise RuntimeError("asset identity mismatch: " + p)
    for p, h in pins.get("executor_sha256", {}).items():
        if sha(HERE / p) != h:
            raise RuntimeError("executor identity mismatch: " + p)


def predict(net, pools, tokens, masks, cids, indexes):
    progs = np.stack(
        [pools[c]["program"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    proj = np.stack(
        [pools[c]["proj16"][ix] for c, ix in zip(cids, indexes, strict=True)]
    )
    b, k = progs.shape[:2]
    return net.adapter_10tok.predict_costs(
        ctx_feat=torch.zeros((b, 584)),
        prog_412=torch.as_tensor(progs),
        proj_16=torch.as_tensor(proj),
        base_prior_cost=torch.zeros((b, k)),
        scale=torch.full((b,), 0.4208),
        sos_prior=torch.zeros((b, k, 2)),
        mm_tokens_10x32=torch.stack([tokens[c] for c in cids]),
        mm_mask_10=torch.stack([masks[c] for c in cids]),
        revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
        revealed_raw_f=torch.as_tensor(
            np.stack(
                [
                    pools[c]["fitness"][ix[: min(4, k)]]
                    for c, ix in zip(cids, indexes, strict=True)
                ]
            ),
            dtype=torch.float32,
        ),
        revealed_gate=torch.as_tensor(
            np.stack(
                [
                    pools[c]["gate"][ix[: min(4, k)]]
                    for c, ix in zip(cids, indexes, strict=True)
                ]
            ),
            dtype=torch.float32,
        ),
    )["total_cost"].float()


def evaluate(net, pools, tokens, masks, rows):
    groups = {}
    result = {}
    for r in rows:
        groups.setdefault(len(pools[r["context_sha256"]]["fitness"]), []).append(
            r["context_sha256"]
        )
    with torch.no_grad():
        for size, cids in sorted(groups.items()):
            for start in range(0, len(cids), 8):
                batch = cids[start : start + 8]
                scores = predict(
                    net, pools, tokens, masks, batch, [np.arange(size) for _ in batch]
                ).numpy()
                if not np.isfinite(scores).all():
                    raise RuntimeError("nonfinite predictions")
                for cid, cost in zip(batch, scores, strict=True):
                    i = int(np.argmin(cost))
                    p = pools[cid]
                    valid = bool(p["gate"][i])
                    result[cid] = dict(
                        choice=i,
                        utility=float(-p["fitness"][i] if valid else -2.0),
                        invalid=not valid,
                    )
    return dict(
        values=[result[r["context_sha256"]]["utility"] for r in rows],
        choices=[result[r["context_sha256"]]["choice"] for r in rows],
        invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
        context_ids=[r["context_sha256"] for r in rows],
    )


def _perturbation_deltas(params, rho, adaptive=False, eta=0.01):
    if not np.isfinite(rho) or rho <= 0 or not np.isfinite(eta) or eta <= 0:
        raise ValueError("positive finite perturbation radius and eta required")
    active = [(p, p.grad.detach()) for p in params if p.grad is not None]
    if not active:
        raise RuntimeError("optimizer arm produced no gradients")
    scales = (
        [p.detach().abs() + eta for p, _ in active]
        if adaptive
        else [torch.ones_like(p) for p, _ in active]
    )
    norm_sq = torch.stack(
        [
            ((scale * grad).double().square().sum())
            for (p, grad), scale in zip(active, scales, strict=True)
        ]
    ).sum()
    norm = torch.sqrt(norm_sq)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0:
        raise RuntimeError("nonfinite or zero SAM gradient norm")
    deltas = []
    for (param, grad), scale in zip(active, scales, strict=True):
        numerator = scale.square() * grad if adaptive else grad
        deltas.append((numerator * (rho / norm)).to(dtype=param.dtype).detach())
    return [(param, delta) for (param, _), delta in zip(active, deltas, strict=True)]


def _optimizer_step(arm, optimizer, params, objective_fn, stream):
    optimizer.zero_grad(set_to_none=True)
    objective = objective_fn()
    if not torch.isfinite(objective):
        raise RuntimeError("nonfinite loss")
    objective.backward()
    if arm == "CTRL":
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        return objective.detach(), None, 1
    if arm not in ("SAM01", "SAM05", "ASAM05"):
        raise ValueError("unknown optimizer arm")
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    rho, adaptive = stat.ARMS[arm]
    deltas = _perturbation_deltas(params, rho, adaptive=adaptive, eta=0.01)
    snapshots = [(param, param.detach().clone()) for param, _ in deltas]
    try:
        with torch.no_grad():
            for param, delta in deltas:
                param.add_(delta)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(stream)
        perturbed_objective = objective_fn()
        if not torch.isfinite(perturbed_objective):
            raise RuntimeError("nonfinite perturbed loss")
        perturbed_objective.backward()
    finally:
        with torch.no_grad():
            for param, saved in snapshots:
                param.copy_(saved)
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return objective.detach(), perturbed_objective.detach(), 2


def selected():
    packet = json.loads((INPUT / "selected.json").read_text())
    train = packet["train"]
    val = packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("incomplete cohort")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if len({r["context_sha256"] for r in rows}) != len(rows):
            raise RuntimeError("duplicate context")
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r.get("has_meta_test") is True
            or r.get("source_split") not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("protected row")
    if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
        raise RuntimeError("family overlap")
    return train, val


def baseline_gate(values, reference):
    expected = np.asarray(reference["utility"]["per_context"], dtype=np.float64)
    actual = np.asarray(values, dtype=np.float64)
    if (
        expected.shape != (91,)
        or actual.shape != (91,)
        or not np.isfinite(actual).all()
    ):
        raise RuntimeError("91 finite contexts required")
    err = float(np.max(np.abs(actual - expected)))
    mean_err = abs(float(actual.mean()) - reference["best_utility"])
    if err != 0.0 or mean_err != 0.0:
        raise RuntimeError(f"baseline exact parity failed: {err}, {mean_err}")
    return dict(
        matches=91,
        max_context_error=err,
        mean_error=mean_err,
        mean=float(actual.mean()),
        status="PASS",
    )


def subpanel(panel, rows):
    lookup = {cid: i for i, cid in enumerate(panel["context_ids"])}
    ix = [lookup[r["context_sha256"]] for r in rows]
    return {k: [v[i] for i in ix] for k, v in panel.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    pins = json.loads((HERE / "manifest.json").read_text())
    if sha(INPUT / "selected.json") != pins["inputs"][str(INPUT / "selected.json")]:
        raise RuntimeError("selected identity")
    train, val = selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory")
    if (
        stat.folds(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["families"]
    ):
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"]) | {str(out / "launch.json")}
    allowed.update(
        str(INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = passes = scored = 0
    started = time.monotonic()
    try:
        # Deliberately before the smoke branch: exercise the former H153 failure path.
        launch = read_launch(out, HERE / "manifest.json", args.smoke)
        verify(pins)
        for name, key in [
            ("PREREG.md", "prereg_sha256"),
            ("fold_manifest.json", "fold_manifest_sha256"),
            ("input-provenance.json", "input_provenance_sha256"),
        ]:
            if sha(HERE / name) != pins[key]:
                raise RuntimeError("protocol identity")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA forbidden")
        net, guard, names, frozen, fused = build(pins["checkpoint"])
        census = dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective mode changed")
        params = [p for p in net.parameters() if p.requires_grad]
        # Trigger PyTorch lazy optimizer/JIT imports through the installed audit hook.
        constructor = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
        constructor.zero_grad(set_to_none=True)
        del constructor
        check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    training_forward_backward_passes=0,
                    launch_path=str(out / "launch.json"),
                    launch_read_under_audit=True,
                    optimizer_constructed=True,
                    adapter_names=list(names),
                    adapter_tensors=len(names),
                    adapter_parameters=sum(p.numel() for p in params),
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=DOPE,
                ),
            )
            return
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
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in train + val:
            cid = r["context_sha256"]
            p = INPUT / "pools" / f"{cid}.npz"
            if sha(p) != r["pool_sha256"]:
                raise RuntimeError("pool identity")
            with np.load(p, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=prog,
                    proj16=(prog.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if len(ids) != len(pools) or set(ids) != set(pools):
                raise RuntimeError("cache closure")
            tok = z["tokens_10x32"]
            mask = z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }

        def score(rows):
            nonlocal scored
            result = evaluate(net, pools, tokens, masks, rows)
            scored += len(rows)
            check_frozen(net, guard, names, frozen)
            return result

        baseline = score(val)
        sanity = baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        trainbase = score(train)
        save(out / "baseline_train.json", trainbase)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        # The same immutable baseline is reused, never refitted or approximately reconstructed.
        modes = ("selected", "unconstrained", "nonzero")
        primary = {mode: np.full((3, len(train), 5), np.nan) for mode in modes}
        secondary = {mode: np.full((3, 5, len(val), 5), np.nan) for mode in modes}
        primary_invalid = {
            mode: np.zeros((3, len(train), 5), dtype=bool) for mode in modes
        }
        secondary_invalid = {
            mode: np.zeros((3, 5, len(val), 5), dtype=bool) for mode in modes
        }
        for mode in modes:
            primary[mode][:, :, 0] = np.asarray(trainbase["values"])
            secondary[mode][:, :, :, 0] = np.asarray(baseline["values"])
            primary_invalid[mode][:, :, 0] = np.asarray(trainbase["invalid"])
            secondary_invalid[mode][:, :, :, 0] = np.asarray(baseline["invalid"])
        cells = []
        identity = dict(
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            input_sha256=pins["inputs"],
            fold_sha256=sha(HERE / "fold_manifest.json"),
        )
        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not fit_ids & excluded
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            baseline_select = subpanel(trainbase, select)
            baseline_assess = subpanel(trainbase, assess)
            ai = [train_index[r["context_sha256"]] for r in assess]
            for si, seed in enumerate(stat.SEEDS):
                plan = stat.batch_plan(fit_ids, orders, seed)
                if len(plan) != 40:
                    raise RuntimeError("40-update plan required")
                for col, arm in enumerate(stat.ARMS, 1):
                    tick = time.monotonic()
                    net.load_state_dict(initial, strict=True)
                    check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    cellname = f"A{outer}-{seed}-{arm}"
                    snapshots = {}
                    selector = []
                    losses = []
                    perturbed_losses = []
                    batches = []
                    cell_passes = 0

                    def snapshot(step):
                        state = {n: net.state_dict()[n].detach().clone() for n in names}
                        snapshots[step] = state
                        np.savez_compressed(
                            out / f"{cellname}-step{step}.npz",
                            **{n: t.numpy() for n, t in state.items()},
                        )
                        selector.append(baseline_select if step == 0 else score(select))

                    snapshot(0)
                    for step, (stream, cids, ixs) in enumerate(plan, 1):
                        if not set(cids) <= fit_ids or set(cids) & excluded:
                            raise RuntimeError("batch family leakage")
                        torch.manual_seed(stream)
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

                        def objective_fn():
                            return loss(
                                predict(net, pools, tokens, masks, cids, ixs),
                                fitness,
                                gate,
                            )

                        objective, perturbed, pass_count = _optimizer_step(
                            arm, optimizer, params, objective_fn, stream
                        )
                        updates += 1
                        passes += pass_count
                        cell_passes += pass_count
                        with (out / "updates.jsonl").open("a") as journal:
                            journal.write(
                                json.dumps(
                                    dict(
                                        cell=cellname,
                                        step=step,
                                        confirmed_updates=updates,
                                        training_forward_backward_passes=passes,
                                    )
                                )
                                + "\n"
                            )
                        check_frozen(net, guard, names, frozen)
                        losses.append(float(objective))
                        if perturbed is not None:
                            perturbed_losses.append(float(perturbed))
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
                    drift = float(
                        torch.linalg.vector_norm(
                            torch.cat(
                                [
                                    dict(net.named_parameters())[n].detach().reshape(-1)
                                    for n in names
                                ]
                            )
                            - initial_vec
                        )
                    )
                    if not drift > 0:
                        raise RuntimeError("no actual parameter update")
                    decision = stat.choose(
                        select,
                        np.asarray([p["values"] for p in selector]).T,
                        stat.BOOT_SEED + 100 * outer + 10 * si,
                    )
                    # Only B results enter either selector. Persist the decision before A or DL-Val.
                    lock = out / f"{cellname}-selection.json"
                    save(lock, decision)
                    chosen = {
                        "selected": decision["selected_step"],
                        "unconstrained": decision["best_step"],
                        "nonzero": decision["best_nonzero_step"],
                    }
                    assessment = {0: baseline_assess}
                    validation = {0: baseline}
                    for step in sorted(set(chosen.values()) - {0}):
                        with torch.no_grad():
                            for n, p in net.named_parameters():
                                if n in snapshots[step]:
                                    p.copy_(snapshots[step][n])
                        assessment[step] = score(assess)
                        validation[step] = score(val)
                    for mode, step in chosen.items():
                        primary[mode][si, ai, col] = assessment[step]["values"]
                        secondary[mode][si, outer, :, col] = validation[step]["values"]
                        primary_invalid[mode][si, ai, col] = assessment[step]["invalid"]
                        secondary_invalid[mode][si, outer, :, col] = validation[step][
                            "invalid"
                        ]
                    cell = dict(
                        identity=identity,
                        arm=arm,
                        outer_fold=outer,
                        selection_fold=(outer + 1) % 5,
                        seed=seed,
                        updates=40,
                        training_forward_backward_passes=cell_passes,
                        fit_contexts=sorted(fit_ids),
                        selector_contexts=[r["context_sha256"] for r in select],
                        assessment_contexts=[r["context_sha256"] for r in assess],
                        losses=losses,
                        perturbed_losses=perturbed_losses,
                        batches=batches,
                        adapter_l2_drift=drift,
                        snapshot_sha256={
                            str(s): tensor_hash(v) for s, v in snapshots.items()
                        },
                        selection_lock_sha256=sha(lock),
                        decision=decision,
                        selector_scores=selector,
                        assessment=assessment,
                        validation=validation,
                        frozen_sha256=frozen,
                        dope_sha256=DOPE,
                        wall_seconds=time.monotonic() - tick,
                    )
                    save(out / f"{cellname}.json", cell)
                    cells.append(
                        dict(
                            cell=cellname,
                            arm=arm,
                            outer_fold=outer,
                            seed=seed,
                            selected_step=chosen["selected"],
                            best_step=chosen["unconstrained"],
                            best_nonzero_step=chosen["nonzero"],
                        )
                    )
                    print(
                        "CELL COMPLETE",
                        cellname,
                        "updates",
                        updates,
                        "selected",
                        chosen,
                        flush=True,
                    )
        summaries = {
            mode: stat.summarize(train, panel) for mode, panel in primary.items()
        }
        secondary_summaries = {
            mode: stat.summarize(val, panel.mean(axis=1))
            for mode, panel in secondary.items()
        }
        decision = stat.decide(summaries["selected"])
        metrics = dict(
            wave="H164",
            status="COMPLETE",
            verdict="ADVANCE"
            if any(d["advance"] for d in decision.values())
            else "NO_ADVANCE",
            primary=summaries["selected"],
            secondary_descriptive_only=secondary_summaries["selected"],
            unconstrained_descriptive_only=dict(
                primary=summaries["unconstrained"],
                secondary=secondary_summaries["unconstrained"],
            ),
            best_nonzero_descriptive_only=dict(
                primary=summaries["nonzero"], secondary=secondary_summaries["nonzero"]
            ),
            decision=decision,
            invalid_counts={
                mode: dict(
                    primary=primary_invalid[mode].sum(axis=1).tolist(),
                    secondary=secondary_invalid[mode].sum(axis=(1, 2)).tolist(),
                    secondary_denominator=455,
                )
                for mode in modes
            },
            cells=cells,
            optimizer_updates=updates,
            training_forward_backward_passes=passes,
            baseline=sanity,
            identity=identity,
            dope_sha256=DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            conditional_fixed_policy_bootstrap=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if (
            updates != 2400
            or passes != 4200
            or len(cells) != 60
            or len(json.dumps(metrics).encode()) >= 100000
        ):
            raise RuntimeError("completion inventory/metric size")
        np.savez_compressed(
            out / "panels.npz",
            **{f"primary_{k}": v for k, v in primary.items()},
            **{f"secondary_{k}": v for k, v in secondary.items()},
            **{f"primary_invalid_{k}": v for k, v in primary_invalid.items()},
            **{f"secondary_invalid_{k}": v for k, v in secondary_invalid.items()},
        )
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                training_forward_backward_passes=passes,
                scored_contexts=scored,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
