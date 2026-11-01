"""One H158 attempt: pinned inputs, baseline first, twelve sequential 40-step cells."""

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
    distillation_loss,
    save,
    sha,
    tensor_hash,
)
from science import (
    SEEDS,
    ARMS,
    folds,
    batch_plan,
    contrast,
    decide,
    family_matrix,
    KD_ARMS,
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


def subpanel(panel, rows):
    positions = {cid: i for i, cid in enumerate(panel["context_ids"])}
    if len(positions) != len(panel["context_ids"]):
        raise ValueError("duplicate panel context")
    indexes = [positions[r["context_sha256"]] for r in rows]
    return {
        k: [panel[k][i] for i in indexes]
        for k in ("values", "choices", "invalid", "context_ids")
    }


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
    allowed = set(pins["inputs"]) | set(pins.get("assets", {}))
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
        teacher, teacher_guard, teacher_names, teacher_frozen, _ = build(
            Path(pins["teacher"]), expected_round=782
        )
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if teacher_names != names or teacher_frozen != frozen:
            raise RuntimeError("teacher/student non-adapter mismatch")
        teacher_hash = tensor_hash(teacher.state_dict())
        if args.smoke:
            # Exercise optimizer constructor and lazy PyTorch JIT imports, no step/forward.
            optimizer = torch.optim.AdamW(
                [p for p in net.parameters() if p.requires_grad],
                lr=1e-5,
                weight_decay=1e-4,
            )
            optimizer.zero_grad(set_to_none=True)
            check_frozen(teacher, teacher_guard, teacher_names, teacher_frozen)
            if tensor_hash(teacher.state_dict()) != teacher_hash:
                raise RuntimeError("teacher changed in smoke")
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
                    optimizer_constructed=True,
                    teacher_sha256=teacher_hash,
                    teacher_requires_grad=False,
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
        secondary = train  # Full TRAIN-exposed descriptive panel.
        assess = [r for r in train if r["family_id"] in groups[0]]
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
        # Veto is a context property based on the full-pool inference argmin.
        teacher_panel = evaluate(teacher, pools, tokens, masks, fit)
        save(out / "teacher_fit.json", teacher_panel)
        teacher_bad = dict(
            zip(teacher_panel["context_ids"], teacher_panel["invalid"], strict=True)
        )
        plans = {seed: batch_plan(list(orders), orders, seed) for seed in SEEDS}
        teacher_batches = {}
        for seed, plan in plans.items():
            costs = []
            with torch.no_grad():
                for _, cids, ixs in plan:
                    score = predict(teacher, pools, tokens, masks, cids, ixs)
                    if not torch.isfinite(score).all():
                        raise RuntimeError("nonfinite teacher scores")
                    costs.append(score.detach().numpy().copy())
            teacher_batches[seed] = np.stack(costs)
            np.savez_compressed(
                out / f"teacher-batches-{seed}.npz", costs=teacher_batches[seed]
            )
        check_frozen(teacher, teacher_guard, teacher_names, teacher_frozen)
        if tensor_hash(teacher.state_dict()) != teacher_hash:
            raise RuntimeError("teacher state changed")
        save(
            out / "teacher_audit.json",
            dict(
                state_sha256_before=teacher_hash,
                state_sha256_after=tensor_hash(teacher.state_dict()),
                forward_scope="fit contexts only; full pool for veto and exact paired 192-candidate training slates for KL",
                optimizer_parameters=0,
                fit_contexts=len(fit),
                veto_contexts=sum(teacher_bad.values()),
                plan_sha256={
                    str(seed): sha(out / f"teacher-batches-{seed}.npz")
                    for seed in SEEDS
                },
            ),
        )
        del teacher
        cells = {}
        identity = dict(
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            input_sha256=pins["inputs"],
            fold_sha256=sha(HERE / "fold_manifest.json"),
        )
        for seed in SEEDS:
            plan = plans[seed]
            ph = hashlib.sha256()
            for stream, cids, ixs in plan:
                ph.update(str(stream).encode())
                for cid, ix in zip(cids, ixs, strict=True):
                    ph.update(cid.encode())
                    ph.update(ix.astype("<i8").tobytes())
            for arm, (alpha, veto) in ARMS.items():
                t = time.monotonic()
                net.load_state_dict(initial, strict=True)
                check_frozen(net, guard, names, frozen)
                optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=1e-4)
                losses = []
                for step, (stream, cids, ixs) in enumerate(plan):
                    torch.manual_seed(stream)
                    optimizer.zero_grad(set_to_none=True)
                    pred = predict(net, pools, tokens, masks, cids, ixs)
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
                    objective = distillation_loss(
                        pred,
                        fitness,
                        gate,
                        torch.as_tensor(teacher_batches[seed][step]),
                        torch.as_tensor(
                            [teacher_bad[c] for c in cids], dtype=torch.bool
                        ),
                        alpha,
                        veto,
                    )
                    if not torch.isfinite(objective):
                        raise RuntimeError("nonfinite loss")
                    objective.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    updates += 1
                    with (out / "updates.jsonl").open("a") as progress:
                        progress.write(
                            json.dumps(
                                dict(
                                    arm=arm,
                                    seed=seed,
                                    step=step + 1,
                                    confirmed_updates=updates,
                                )
                            )
                            + "\n"
                        )
                    check_frozen(net, guard, names, frozen)
                    losses.append(float(objective.detach()))
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
                    alpha=alpha,
                    veto=veto,
                    learning_rate=1e-5,
                    weight_decay=1e-4,
                    teacher_sha256=teacher_hash,
                    veto_sampled_contexts=sum(
                        teacher_bad[c] for _, cids, _ in plan for c in cids
                    )
                    if veto
                    else 0,
                    updates=40,
                    batch_plan_sha256=ph.hexdigest(),
                    adapter_l2_drift=drift,
                    adapter_sha256=tensor_hash({n: net.state_dict()[n] for n in names}),
                    frozen_sha256=frozen,
                    dope_sha256=DOPE,
                    losses=losses,
                    primary=val_result,
                    secondary=sec_result,
                    assess=subpanel(sec_result, assess),
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
        assess_primary = {}
        base_assess = subpanel(baseline_secondary, assess)
        for arm in ARMS:
            panels = [cells[arm, seed]["assess"] for seed in SEEDS]
            assess_primary[arm] = dict(
                vs_r882=contrast(
                    assess, [p["values"] for p in panels], base_assess["values"]
                ),
                invalid_counts=[sum(p["invalid"]) for p in panels],
            )
        decision = decide({a: primary[a]["vs_r882"] for a in KD_ARMS})
        metrics = dict(
            wave="H158",
            status="COMPLETE",
            verdict="ADVANCE"
            if any(d["advance"] for d in decision.values())
            else "NO_ADVANCE",
            primary=primary,
            decision=decision,
            secondary_descriptive_only=desc,
            assess_fold=assess_primary,
            baseline_assess_invalid=sum(base_assess["invalid"]),
            teacher_veto_fit_contexts=sum(teacher_bad.values()),
            teacher_state_sha256=teacher_hash,
            baseline=sanity,
            baseline_family_utility=float(
                family_matrix(val, np.tile(baseline["values"], (3, 1)))[1].mean()
            ),
            baseline_invalid_primary=sum(baseline["invalid"]),
            baseline_invalid_secondary=sum(baseline_secondary["invalid"]),
            optimizer_updates=updates,
            identity=identity,
            seeds=list(SEEDS),
            arms=ARMS,
            dropped=[],
            secondary_scope="1219 DL-Train contexts / 205 families; TRAIN-exposed, descriptive only",
            inference_unit="equal family mean within seed, median across three seeds; shared family-cluster bootstrap",
            serving_promotion=False,
            wall_seconds=time.monotonic() - start,
        )
        if updates != 480 or len(json.dumps(metrics).encode()) >= 100000:
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
