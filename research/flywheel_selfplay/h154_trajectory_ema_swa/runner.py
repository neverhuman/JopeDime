"""One H154 attempt: pinned inputs, exact R882 baseline gate, three 40-step trajectory seeds producing CTRL + EMA(0.99/0.995/0.999) + SWA(K=5/10/20) with train-only stat recalibration."""

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
from runtime import (
    HERE,
    INPUT,
    CACHE,
    DOPE,
    Audit,
    TrajectoryEMAState,
    TrajectorySWAState,
    apply_stat_recalibration,
    bind_source,
    build,
    check_frozen,
    compute_recal_rms,
    dropout_census,
    guards,
    loss,
    save,
    sha,
    tensor_hash,
)
from science import (
    ARMS,
    CANDIDATE_ARMS,
    EMA_DECAYS,
    LEARNING_RATE,
    RECAL_BOUNDS,
    RECAL_SLATES,
    SEEDS,
    SWA_WINDOWS,
    WEIGHT_DECAY,
    batch_plan,
    contrast,
    decide,
    family_matrix,
    folds,
    study_verdict,
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


def predict_dict(net, pools, tokens, masks, cids, indexes):
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
    )


def predict(net, pools, tokens, masks, cids, indexes):
    return predict_dict(net, pools, tokens, masks, cids, indexes)["total_cost"].float()


def calibration_rms(net, pools, tokens, masks, cal_batches):
    mm_list, traj_list = [], []
    with torch.no_grad():
        for _, cids, ixs in cal_batches:
            out = predict_dict(net, pools, tokens, masks, cids, ixs)
            mm_list.append(out["mm_prior_adj"].detach().cpu().numpy())
            traj_list.append(out["traj_attn_adj"].detach().cpu().numpy())
    return compute_recal_rms(mm_list, traj_list)


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


def subset_eval(full_eval, sub_rows):
    idx = {cid: i for i, cid in enumerate(full_eval["context_ids"])}
    sub_ix = [idx[r["context_sha256"]] for r in sub_rows]
    return dict(
        values=[full_eval["values"][i] for i in sub_ix],
        choices=[full_eval["choices"][i] for i in sub_ix],
        invalid=[full_eval["invalid"][i] for i in sub_ix],
        context_ids=[full_eval["context_ids"][i] for i in sub_ix],
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
    launch_guard = guards(out.parent)
    pins = json.loads((HERE / "manifest.json").read_text())
    allowed = set(pins["inputs"])
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
        if args.smoke:
            check_frozen(net, guard, names, frozen)
            end_guard = guards(out.parent)
            assets = {
                p: sha(p)
                for p in sorted(audit.opened)
                if p.startswith("/home/ubuntu/JopeDimeMicro/")
                and Path(p).is_file()
                and not p.endswith(".pyc")
            }
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
                    candidate_arms_scored=0,
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
                    launch_guard=launch_guard,
                    end_guard=end_guard,
                ),
            )
            return

        launch = json.loads((out.parent / "launch.json").read_text())
        if not launch["merged_source_verified"] or launch["manifest_sha256"] != sha(
            HERE / "manifest.json"
        ):
            raise RuntimeError("merged launch required")
        binding = bind_source(launch, prereg_sha=pins["prereg_sha256"])

        groups = folds(train)
        if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
            raise RuntimeError("fold identity")
        fit = [r for r in train if r["family_id"] not in groups[0]]
        fold0 = [r for r in train if r["family_id"] in groups[0]]
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

        baseline_train = evaluate(net, pools, tokens, masks, train)
        baseline_fold0 = subset_eval(baseline_train, fold0)
        save(out / "baseline_train.json", baseline_train)
        save(out / "baseline_fold0.json", baseline_fold0)
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
            reviewed_head=launch["reviewed_head"],
            binding=binding,
            manifest_sha256=sha(HERE / "manifest.json"),
            input_sha256=pins["inputs"],
            fold_sha256=sha(HERE / "fold_manifest.json"),
        )

        for seed in SEEDS:
            guards(out.parent)
            plan = batch_plan(list(orders), orders, seed)
            cal_batches = plan[:RECAL_SLATES]
            ph = hashlib.sha256()
            for stream, cids, ixs in plan:
                ph.update(str(stream).encode())
                for cid, ix in zip(cids, ixs, strict=True):
                    ph.update(cid.encode())
                    ph.update(ix.astype("<i8").tobytes())

            t_traj = time.monotonic()
            net.load_state_dict(initial, strict=True)
            check_frozen(net, guard, names, frozen)
            ref_rms = calibration_rms(net, pools, tokens, masks, cal_batches)

            optimizer = torch.optim.AdamW(
                params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            ema_trackers = {
                arm: TrajectoryEMAState(initial, names, dec)
                for arm, dec in EMA_DECAYS.items()
            }
            swa_tracker = TrajectorySWAState(names)
            losses = []

            for step_idx, (stream, cids, ixs) in enumerate(plan, start=1):
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
                        [pools[c]["gate"][ix] for c, ix in zip(cids, ixs, strict=True)]
                    ),
                    dtype=torch.bool,
                )
                objective = loss(pred, fitness, gate)
                if not torch.isfinite(objective):
                    raise RuntimeError("nonfinite loss")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                cur_sd = net.state_dict()
                for tracker in ema_trackers.values():
                    tracker.after_fast_step(cur_sd)
                swa_tracker.after_step(step_idx, cur_sd)
                check_frozen(net, guard, names, frozen)
                losses.append(float(objective.detach()))

            traj_seconds = time.monotonic() - t_traj
            ctrl_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}

            raw_states = {"CTRL": ctrl_sd}
            for arm in EMA_DECAYS:
                raw_states[arm] = ema_trackers[arm].export_state_dict(initial)
            for arm, k_win in SWA_WINDOWS.items():
                raw_states[arm] = swa_tracker.export_state_dict(initial, k_win)

            for arm, spec in ARMS.items():
                guards(out.parent)
                t_arm = time.monotonic()
                sd = raw_states[arm]
                recal_receipt = None
                if spec["recal"]:
                    net.load_state_dict(sd, strict=True)
                    check_frozen(net, guard, names, frozen)
                    cand_rms = calibration_rms(net, pools, tokens, masks, cal_batches)
                    sd, recal_receipt = apply_stat_recalibration(
                        sd, names, ref_rms, cand_rms, bounds=RECAL_BOUNDS
                    )
                net.load_state_dict(sd, strict=True)
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
                    raise RuntimeError(f"no parameter update for {arm} seed {seed}")

                val_result = evaluate(net, pools, tokens, masks, val)
                train_result = evaluate(net, pools, tokens, masks, train)
                fold0_result = subset_eval(train_result, fold0)
                check_frozen(net, guard, names, frozen)

                cell = dict(
                    identity=identity,
                    arm=arm,
                    seed=seed,
                    spec=spec,
                    updates=40 if arm == "CTRL" else 0,
                    trajectory_updates_shared=40,
                    batch_plan_sha256=ph.hexdigest(),
                    adapter_l2_drift=drift,
                    adapter_sha256=tensor_hash({n: net.state_dict()[n] for n in names}),
                    frozen_sha256=frozen,
                    dope_sha256=DOPE,
                    recalibration=recal_receipt,
                    losses=losses if arm == "CTRL" else None,
                    primary=val_result,
                    h145_train=train_result,
                    fold0=fold0_result,
                    trajectory_wall_seconds=traj_seconds if arm == "CTRL" else 0.0,
                    eval_wall_seconds=time.monotonic() - t_arm,
                )
                save(
                    out / f"{arm}-{seed}.json",
                    dict(
                        arm=arm,
                        seed=seed,
                        adapter_l2_drift=drift,
                        adapter_sha256=cell["adapter_sha256"],
                        frozen_sha256=frozen,
                        dope_sha256=DOPE,
                        recalibration=recal_receipt,
                        val_mean_utility=float(np.mean(val_result["values"])),
                        h145_train_mean_utility=float(np.mean(train_result["values"])),
                        fold0_mean_utility=float(np.mean(fold0_result["values"])),
                        eval_wall_seconds=cell["eval_wall_seconds"],
                    ),
                )
                cells[arm, seed] = cell
                print(
                    "CELL COMPLETE",
                    arm,
                    seed,
                    "drift",
                    round(drift, 6),
                    "val_mean",
                    round(float(np.mean(val_result["values"])), 7),
                    "train_mean",
                    round(float(np.mean(train_result["values"])), 7),
                    flush=True,
                )

        drifts = {
            arm: [cells[arm, s]["adapter_l2_drift"] for s in SEEDS] for arm in ARMS
        }
        val_panels = {
            arm: np.asarray([cells[arm, s]["primary"]["values"] for s in SEEDS])
            for arm in ARMS
        }
        train_panels = {
            arm: np.asarray([cells[arm, s]["h145_train"]["values"] for s in SEEDS])
            for arm in ARMS
        }
        fold0_panels = {
            arm: np.asarray([cells[arm, s]["fold0"]["values"] for s in SEEDS])
            for arm in ARMS
        }

        ctrl_val = val_panels["CTRL"]
        ctrl_train = train_panels["CTRL"]
        ctrl_fold0 = fold0_panels["CTRL"]

        primary = {}
        h145_train_readout = {}
        fold0_readout = {}

        for arm in ARMS:
            v_val = val_panels[arm]
            primary[arm] = dict(
                context_mean_utility_by_seed=v_val.mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(val, v_val)[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=contrast(val, v_val, baseline["values"]),
                vs_CTRL=contrast(val, v_val, ctrl_val),
                invalid_counts=[
                    sum(cells[arm, s]["primary"]["invalid"]) for s in SEEDS
                ],
                drift=drifts[arm],
                recalibration=[cells[arm, s]["recalibration"] for s in SEEDS],
            )

            v_tr = train_panels[arm]
            h145_train_readout[arm] = dict(
                context_mean_utility_by_seed=v_tr.mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(train, v_tr)[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=contrast(train, v_tr, baseline_train["values"]),
                vs_CTRL=contrast(train, v_tr, ctrl_train),
                invalid_counts=[
                    sum(cells[arm, s]["h145_train"]["invalid"]) for s in SEEDS
                ],
            )

            v_f0 = fold0_panels[arm]
            fold0_readout[arm] = dict(
                context_mean_utility_by_seed=v_f0.mean(axis=1).tolist(),
                family_mean_utility_by_seed=family_matrix(fold0, v_f0)[1]
                .mean(axis=1)
                .tolist(),
                vs_r882=contrast(fold0, v_f0, baseline_fold0["values"]),
                vs_CTRL=contrast(fold0, v_f0, ctrl_fold0),
                invalid_counts=[sum(cells[arm, s]["fold0"]["invalid"]) for s in SEEDS],
            )

        decision = decide(
            {a: primary[a]["vs_r882"] for a in ARMS},
            {a: primary[a]["vs_CTRL"] for a in CANDIDATE_ARMS},
            drifts,
            val_panels,
        )
        top_v, r882_v, ctrl_v, best_arm = study_verdict(decision)

        pseudo = {
            arm: dict(
                identical_primary_utilities_across_seeds=len(
                    {tuple(cells[arm, s]["primary"]["values"]) for s in SEEDS}
                )
                == 1,
                identical_primary_choices_across_seeds=len(
                    {tuple(cells[arm, s]["primary"]["choices"]) for s in SEEDS}
                )
                == 1,
                identical_adapters_across_seeds=len(
                    {cells[arm, s]["adapter_sha256"] for s in SEEDS}
                )
                == 1,
            )
            for arm in ARMS
        }
        cross_arm_pseudo = {}
        for arm in CANDIDATE_ARMS:
            cross_arm_pseudo[f"{arm}_vs_CTRL"] = dict(
                identical_primary_utilities_by_seed=[
                    cells[arm, s]["primary"]["values"]
                    == cells["CTRL", s]["primary"]["values"]
                    for s in SEEDS
                ],
                identical_primary_choices_by_seed=[
                    cells[arm, s]["primary"]["choices"]
                    == cells["CTRL", s]["primary"]["choices"]
                    for s in SEEDS
                ],
                choice_diffs_out_of_91_by_seed=[
                    int(
                        sum(
                            x != y
                            for x, y in zip(
                                cells[arm, s]["primary"]["choices"],
                                cells["CTRL", s]["primary"]["choices"],
                                strict=True,
                            )
                        )
                    )
                    for s in SEEDS
                ],
            )

        end_guard = guards(out.parent)
        metrics = dict(
            wave="H154",
            status="COMPLETE",
            verdict=top_v,
            r882_verdict=r882_v,
            trajectory_vs_ctrl_verdict=ctrl_v,
            best_r882_arm=best_arm,
            primary_dl_val_91ctx_19fam=primary,
            decision=decision,
            secondary_h145_train_1219ctx_205fam=h145_train_readout,
            secondary_fold0_out_of_fold_254ctx_41fam=fold0_readout,
            seed_pseudoreplication=pseudo,
            cross_arm_argmin_pseudoreplication=cross_arm_pseudo,
            baseline_dl_val=sanity,
            baseline_dl_val_family_utility=float(
                family_matrix(val, np.tile(baseline["values"], (3, 1)))[1].mean()
            ),
            baseline_h145_train_context_utility=float(
                np.mean(baseline_train["values"])
            ),
            baseline_h145_train_family_utility=float(
                family_matrix(train, np.tile(baseline_train["values"], (3, 1)))[
                    1
                ].mean()
            ),
            baseline_fold0_family_utility=float(
                family_matrix(fold0, np.tile(baseline_fold0["values"], (3, 1)))[
                    1
                ].mean()
            ),
            baseline_invalid_dl_val=sum(baseline["invalid"]),
            baseline_invalid_h145_train=sum(baseline_train["invalid"]),
            baseline_invalid_fold0=sum(baseline_fold0["invalid"]),
            optimizer_updates=updates,
            identity=identity,
            seeds=list(SEEDS),
            arms=ARMS,
            launch_guard=launch_guard,
            end_guard=end_guard,
            inference_unit="equal family mean within seed, median across three seeds; shared family-cluster bootstrap",
            serving_promotion=False,
            wall_seconds=time.monotonic() - start,
        )
        if updates != 120 or len(json.dumps(metrics).encode()) >= 100000:
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
