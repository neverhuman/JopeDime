"""Single H148 continuation wave; reuse hash-pinned H146 inference/freeze implementation."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import analysis as stat

HERE = Path(__file__).resolve().parent


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
    parent = HERE.parent / "flywheel_selfplay/h146_lookahead_exec"
    sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("h148_parent", parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module, sys.modules["runtime"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Exclusive, no result/cache reuse or automatic retry.
    pins = json.loads((HERE / "manifest.json").read_text())
    # Verify all sources before importing model code or opening any pools.
    verify(pins)
    parent, rt = parent_executor()
    import torch

    train, val = parent.selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory")
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
        if args.smoke:
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
            raise RuntimeError("merged preregistration and Argon PASS required")
        if (
            launch["source_commit"]
            != __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True)
            .strip()
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
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)
        for row in train + val:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool identity")
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            # Check closure before decoding feature arrays.
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

        baseline = evaluate(val)
        sanity = parent.baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        params = [p for p in net.parameters() if p.requires_grad]
        initial_vec = torch.cat([initial[n].reshape(-1) for n in names])
        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        primary = np.full((3, len(train), 3), np.nan)
        secondary = np.full((3, 5, len(val), 3), np.nan)
        selections = []
        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not fit_ids & excluded
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            for si, seed in enumerate(stat.SEEDS):
                tick = time.monotonic()
                net.load_state_dict(initial, strict=True)
                rt.check_frozen(net, guard, names, frozen)
                optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                snapshots, selector_scores, val_scores, losses, batches = (
                    {},
                    [],
                    [],
                    [],
                    [],
                )

                def snapshot(step):
                    state = {n: net.state_dict()[n].detach().clone() for n in names}
                    snapshots[step] = state
                    np.savez_compressed(
                        out / f"A{outer}-{seed}-step{step}.npz",
                        **{n: t.numpy() for n, t in state.items()},
                    )
                    selector_scores.append(evaluate(select))
                    val_scores.append(baseline if step == 0 else evaluate(val))

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
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
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
                if drift <= 0:
                    raise RuntimeError("no optimizer parameter drift")
                decisions = {}
                for rule, rows, results, offset in [
                    ("XFIT", select, selector_scores, 0),
                    ("VALSEL", val, val_scores, 1),
                ]:
                    panel = np.array([r["values"] for r in results]).T
                    decisions[rule] = stat.choose(
                        rows, panel, stat.BOOT_SEED + 100 * outer + 10 * si + offset
                    )
                # Durable lock precedes every outer-A prediction in this cell.
                cellname = f"A{outer}-{seed}"
                lockpath = out / f"{cellname}-selection.json"
                save(lockpath, decisions)
                selected_steps = {0} | {d["selected_step"] for d in decisions.values()}
                assessment = {}
                for step in sorted(selected_steps):
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if n in snapshots[step]:
                                p.copy_(snapshots[step][n])
                    assessment[step] = evaluate(assess)
                ai = [train_index[r["context_sha256"]] for r in assess]
                for col, rule in enumerate(("r882", "XFIT", "VALSEL")):
                    step = 0 if col == 0 else decisions[rule]["selected_step"]
                    primary[si, ai, col] = assessment[step]["values"]
                    secondary[si, outer, :, col] = val_scores[stat.STEPS.index(step)][
                        "values"
                    ]
                cell = dict(
                    outer_fold=outer,
                    selection_fold=(outer + 1) % 5,
                    seed=seed,
                    updates=40,
                    fit_contexts=sorted(fit_ids),
                    selector_contexts=[r["context_sha256"] for r in select],
                    assessment_contexts=[r["context_sha256"] for r in assess],
                    losses=losses,
                    batches=batches,
                    adapter_l2_drift=drift,
                    snapshot_sha256={
                        str(s): rt.tensor_hash(v) for s, v in snapshots.items()
                    },
                    selection_lock_sha256=sha(lockpath),
                    decisions=decisions,
                    selector_scores=selector_scores,
                    validation_scores=val_scores,
                    assessment=assessment,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    wall_seconds=time.monotonic() - tick,
                )
                save(out / f"{cellname}.json", cell)
                selections.append(decisions)
                print(
                    "CELL COMPLETE",
                    cellname,
                    "updates",
                    updates,
                    "picks",
                    {r: d["selected_step"] for r, d in decisions.items()},
                    "seconds",
                    cell["wall_seconds"],
                    flush=True,
                )
        psummary = stat.summarize(train, primary)
        ssummary = stat.summarize(val, secondary.mean(axis=1))
        metrics = dict(
            wave="H148",
            status="COMPLETE",
            verdict=stat.verdict(psummary, selections),
            primary=psummary,
            secondary_descriptive_only=ssummary,
            selections=selections,
            optimizer_updates=updates,
            cells=len(selections),
            baseline=sanity,
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if updates != 600 or len(selections) != 15:
            raise RuntimeError("incomplete wave")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        verify(pins)
        save(out / "metrics.json", metrics)
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
