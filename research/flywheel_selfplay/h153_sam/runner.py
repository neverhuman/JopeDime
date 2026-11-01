"""One H153 attempt: pinned inputs, baseline first, nine sequential 40-step cells."""

import argparse
import hashlib
import importlib.metadata
import resource
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
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
from science import SEEDS, ARMS, folds, batch_plan, contrast, decide, family_matrix

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
    if arm not in ("SAM", "ASAM"):
        raise ValueError("unknown optimizer arm")
    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    rho = 0.05 if arm == "SAM" else 0.5
    adaptive = arm == "ASAM"
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


def _sharpness_proxy(params, objective_fn, stream, radius=0.05):
    for param in params:
        param.grad = None
    torch.manual_seed(stream)
    base = objective_fn()
    if not torch.isfinite(base):
        raise RuntimeError("nonfinite sharpness-probe loss")
    base.backward()
    deltas = _perturbation_deltas(params, radius, adaptive=False)
    snapshots = [(param, param.detach().clone()) for param, _ in deltas]
    try:
        with torch.no_grad():
            for param, delta in deltas:
                param.add_(delta)
        torch.manual_seed(stream)
        with torch.no_grad():
            perturbed = objective_fn()
        if not torch.isfinite(perturbed):
            raise RuntimeError("nonfinite sharpness-probe perturbed loss")
    finally:
        with torch.no_grad():
            for param, saved in snapshots:
                param.copy_(saved)
        for param in params:
            param.grad = None
    return dict(
        base_train_objective=float(base.detach()),
        perturbed_train_objective=float(perturbed.detach()),
        delta=float(perturbed.detach() - base.detach()),
        l2_radius=radius,
        stream=stream,
    )


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Never accepts cached cells or an old attempt.
    pins = json.loads((HERE / "manifest.json").read_text())
    allowed = set(pins["inputs"])
    # Hash-check selected metadata before extending the exact payload allowlist.
    if sha(INPUT / "selected.json") != pins["inputs"][str(INPUT / "selected.json")]:
        raise RuntimeError("selected mismatch")
    train, val = selected()
    allowed.update(
        str(INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
    forward_backward_passes = 0
    start = time.monotonic()
    try:
        verify(pins)
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA initialization forbidden")
        checkpoint = Path(pins["checkpoint"])
        net, guard, names, frozen, fused = build(checkpoint)
        census = dropout_census(net)
        if census["adapter_dropout_modules"] or any(
            m["p"] != 0 for m in census["adapter_attention_dropout"]
        ):
            raise RuntimeError("prospective dropout exclusion invalid")
        if (
            census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective mode changed")
        if args.smoke:
            check_frozen(net, guard, names, frozen)
            assets = {
                p: sha(p)
                for p in sorted(audit.opened)
                if p.startswith("/home/ubuntu/JopeDimeMicro/")
                and Path(p).is_file()
                and not p.endswith(".pyc")
            }
            # Pin imported Python source rather than timestamp-dependent .pyc caches.
            for m in list(sys.modules.values()):
                p = getattr(m, "__file__", None)
                if (
                    p
                    and str(p).startswith("/home/ubuntu/JopeDimeMicro/")
                    and str(p).endswith(".py")
                ):
                    assets[p] = sha(p)
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    census=census,
                    assets=assets,
                    packages={
                        n: importlib.metadata.version(n)
                        for n in ("numpy", "torch", "psutil")
                    },
                    python=sys.version,
                    frozen_sha256=frozen,
                    adapter_names=list(names),
                    dope_sha256=DOPE,
                ),
            )
            return
        launch = json.loads((out.parent / "launch.json").read_text())
        if not launch["merged_source_verified"] or launch["manifest_sha256"] != sha(
            HERE / "manifest.json"
        ):
            raise RuntimeError("merged launch required")
        groups = folds(train)
        if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
            raise RuntimeError("fold identity")
        fit = [r for r in train if r["family_id"] not in groups[0]]
        secondary = (
            train  # H145 panel: all 1219 TRAIN-exposed contexts; descriptive only.
        )
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
            raise RuntimeError("evaluation contract")
        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for r in train + val:
            cid = r["context_sha256"]
            p = INPUT / "pools" / f"{cid}.npz"
            if sha(p) != r["pool_sha256"]:
                raise RuntimeError("pool mismatch")
            with np.load(p, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            tok = z["tokens_10x32"]
            mask = z["mask_10"]
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("cache closure")
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool) for i, c in enumerate(ids)
            }
        baseline = evaluate(net, pools, tokens, masks, val)
        sanity = baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        baseline_secondary = evaluate(net, pools, tokens, masks, secondary)
        save(out / "baseline_secondary.json", baseline_secondary)
        check_frozen(net, guard, names, frozen)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        params = [p for p in net.parameters() if p.requires_grad]
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        orders = {
            r["context_sha256"]: np.argsort(pools[r["context_sha256"]]["fitness"])
            for r in fit
        }
        cells = {}
        identity = dict(
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            input_sha256=pins["inputs"],
            fold_sha256=sha(HERE / "fold_manifest.json"),
        )
        for seed in SEEDS:
            plan = batch_plan(list(orders), orders, seed)
            ph = hashlib.sha256()
            for stream, cids, ixs in plan:
                ph.update(str(stream).encode())
                for cid, ix in zip(cids, ixs, strict=True):
                    ph.update(cid.encode())
                    ph.update(ix.astype("<i8").tobytes())
            for arm in ARMS:
                t = time.monotonic()
                net.load_state_dict(initial, strict=True)
                check_frozen(net, guard, names, frozen)
                optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                losses = []
                perturbed_losses = []
                sharpness_steps = []
                cell_passes = 0

                def make_objective(cids, ixs):
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
                        pred = predict(net, pools, tokens, masks, cids, ixs)
                        return loss(pred, fitness, gate)

                    return objective_fn

                for stream, cids, ixs in plan:
                    torch.manual_seed(stream)
                    objective_fn = make_objective(cids, ixs)
                    objective, perturbed, pass_count = _optimizer_step(
                        arm, optimizer, params, objective_fn, stream
                    )
                    updates += 1
                    forward_backward_passes += pass_count
                    cell_passes += pass_count
                    check_frozen(net, guard, names, frozen)
                    losses.append(float(objective))
                    if perturbed is not None:
                        perturbed_losses.append(float(perturbed))
                        sharpness_steps.append(float(perturbed - objective))
                probe_stream, probe_cids, probe_ixs = plan[0]
                probe = _sharpness_proxy(
                    params,
                    make_objective(probe_cids, probe_ixs),
                    probe_stream,
                    radius=0.05,
                )
                check_frozen(net, guard, names, frozen)
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
                    raise RuntimeError("no parameter update")
                val_result = evaluate(net, pools, tokens, masks, val)
                sec_result = evaluate(net, pools, tokens, masks, secondary)
                check_frozen(net, guard, names, frozen)
                cell = dict(
                    identity=identity,
                    arm=arm,
                    seed=seed,
                    optimizer="AdamW",
                    learning_rate=1e-4,
                    weight_decay=1e-4,
                    rho=0.05 if arm == "SAM" else (0.5 if arm == "ASAM" else None),
                    asam_eta=0.01 if arm == "ASAM" else None,
                    updates=40,
                    training_forward_backward_passes=cell_passes,
                    sharpness_probe_backward_passes=1,
                    sharpness_proxy=probe,
                    perturbed_losses=perturbed_losses,
                    sharpness_step_deltas=sharpness_steps,
                    batch_plan_sha256=ph.hexdigest(),
                    adapter_l2_drift=drift,
                    adapter_sha256=tensor_hash({n: net.state_dict()[n] for n in names}),
                    frozen_sha256=frozen,
                    dope_sha256=DOPE,
                    losses=losses,
                    primary=val_result,
                    secondary=sec_result,
                    wall_seconds=time.monotonic() - t,
                )
                save(out / f"{arm}-{seed}.json", cell)
                np.savez_compressed(
                    out / f"{arm}-{seed}-adapter.npz",
                    **{n: net.state_dict()[n].numpy() for n in names},
                )
                cells[arm, seed] = cell
                print(
                    "CELL COMPLETE",
                    arm,
                    seed,
                    "updates",
                    updates,
                    "seconds",
                    cell["wall_seconds"],
                    flush=True,
                )
        primary = {}
        desc = {}
        for arm in ARMS:
            values = np.asarray([cells[arm, s]["primary"]["values"] for s in SEEDS])
            ctrl = np.asarray([cells["CTRL", s]["primary"]["values"] for s in SEEDS])
            primary[arm] = dict(
                context_mean_utility_by_seed=values.mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(val, values)[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=contrast(val, values, baseline["values"]),
                vs_CTRL=contrast(val, values, ctrl),
                invalid_counts=[
                    sum(cells[arm, s]["primary"]["invalid"]) for s in SEEDS
                ],
                drift=[cells[arm, s]["adapter_l2_drift"] for s in SEEDS],
                median_adapter_l2_drift=float(
                    np.median([cells[arm, s]["adapter_l2_drift"] for s in SEEDS])
                ),
                sharpness_proxy_train_objective_delta_by_seed=[
                    cells[arm, s]["sharpness_proxy"]["delta"] for s in SEEDS
                ],
                median_sharpness_proxy_train_objective_delta=float(
                    np.median(
                        [cells[arm, s]["sharpness_proxy"]["delta"] for s in SEEDS]
                    )
                ),
            )
            sv = np.asarray([cells[arm, s]["secondary"]["values"] for s in SEEDS])
            sc = np.asarray([cells["CTRL", s]["secondary"]["values"] for s in SEEDS])
            desc[arm] = dict(
                vs_r882=contrast(secondary, sv, baseline_secondary["values"]),
                vs_CTRL=contrast(secondary, sv, sc),
                invalid_counts=[
                    sum(cells[arm, s]["secondary"]["invalid"]) for s in SEEDS
                ],
            )
        utility_panels = {
            arm: np.asarray([cells[arm, seed]["primary"]["values"] for seed in SEEDS])
            for arm in ARMS
        }
        decision = decide(
            {arm: primary[arm]["vs_CTRL"] for arm in ("SAM", "ASAM")},
            {arm: primary[arm]["drift"] for arm in ARMS},
            utility_panels,
        )
        breakthrough_arms = [
            arm
            for arm in ("SAM", "ASAM")
            if primary[arm]["vs_r882"]["median_gain"] > 0
            and primary[arm]["vs_r882"]["ci95"][0] > 0
        ]
        metrics = dict(
            wave="H153",
            status="COMPLETE",
            verdict="ADVANCE"
            if any(d["advance"] for d in decision.values())
            else (
                "DOSE_INVALID"
                if all(d["dose"]["status"] == "DOSE_INVALID" for d in decision.values())
                else "NO_ADVANCE"
            ),
            primary=primary,
            decision=decision,
            secondary_descriptive_only=desc,
            baseline=sanity,
            baseline_family_utility=float(
                family_matrix(val, np.tile(baseline["values"], (3, 1)))[1].mean()
            ),
            baseline_invalid_primary=sum(baseline["invalid"]),
            baseline_invalid_secondary=sum(baseline_secondary["invalid"]),
            optimizer_updates=updates,
            training_forward_backward_passes=forward_backward_passes,
            sharpness_probe_backward_passes=9,
            primary_contrast="paired SAM/ASAM versus CTRL on 91 DL-Val contexts, equal-family median across three seeds",
            breakthrough_arms_vs_r882=breakthrough_arms,
            identity=identity,
            seeds=list(SEEDS),
            arms=ARMS,
            dropped=[],
            secondary_scope="1219 DL-Train contexts / 205 families; TRAIN-exposed, descriptive only",
            inference_unit="equal family mean within seed, median across three seeds; shared family-cluster bootstrap",
            bootstrap_seed=153141,
            bootstrap_draws=10000,
            holm_hypotheses=2,
            serving_promotion=False,
            wall_seconds=time.monotonic() - start,
        )
        if (
            updates != 360
            or forward_backward_passes != 600
            or len(json.dumps(metrics).encode()) >= 100000
        ):
            raise RuntimeError("completion inventory/size")
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                wall_seconds=time.monotonic() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
