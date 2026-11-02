"""H228 fixed 60-fit full-fold execution: no held-out A or DL-Val91 outcome reads before model lock."""

import io
import time
import numpy as np
import torch


def execute(
    engine,
    sci,
    legacy,
    elo,
    rows,
    store,
    rt,
    frozen_guard,
    frozen,
    progress,
    save,
    output,
    check,
    checkpoint,
    val_rows=None,
    val_gpu_inputs=None,
    native_val=None,
):
    # Timing smoke has no held-out utility read and no scientific score.
    fit, _ = sci.partition(rows, sci.SEEDS[0], 0)
    started = time.monotonic()
    _, smoke = engine.train(
        fit,
        "CTRL",
        sci.SMOKE_SEED,
        0,
        "timing-smoke",
        legacy,
        sci,
        output,
        checkpoint,
        steps=4,
    )
    rt.check_frozen(engine.net, frozen_guard, engine.names, frozen)
    smoke_receipt = dict(
        scored=False,
        optimizer_updates=4,
        elapsed_seconds=time.monotonic() - started,
        projected_full_fit_seconds=float(
            np.median(smoke["seconds_per_update"][1:]) * 2400
        ),
        scope="TRAIN FIT-only timing; no held-out labels, selection or promotion; included in single attempt budget",
        proof=smoke,
    )
    save(output / "timing-smoke.json", smoke_receipt)
    store.put("timing-smoke.json", (output / "timing-smoke.json").read_bytes())
    progress["smoke_updates"] = progress["updates"]

    states, proofs = {}, []
    for seed in sci.SEEDS:
        for outer in range(5):
            fit, _ = sci.partition(rows, seed, outer)
            andmask_kept = None
            for arm in sci.ARMS:
                check()
                label = f"{arm}-{seed}-{outer}"
                state, proof = engine.train(
                    fit,
                    arm,
                    seed,
                    outer,
                    label,
                    legacy,
                    sci,
                    output,
                    checkpoint,
                    steps=40,
                    shuf_kept_counts=andmask_kept if arm == "ANDMASK_SHUF" else None,
                )
                if arm == "ANDMASK":
                    andmask_kept = list(proof["invariant"]["kept_counts"])
                elif arm == "ANDMASK_SHUF":
                    if proof["invariant"]["kept_counts"] != andmask_kept:
                        raise RuntimeError(
                            "ANDMASK_SHUF per-step kept_counts diverged from ANDMASK"
                        )
                rt.check_frozen(engine.net, frozen_guard, engine.names, frozen)
                states[(arm, seed, outer)] = state
                proofs.append(proof)
                b = io.BytesIO()
                torch.save(state, b)
                store.put("adapters/" + label + ".pt", b.getvalue())

    if progress["updates"] != 2404 or len(proofs) != 60:
        raise RuntimeError(
            "full matched 60-fit / 2404-update training budget incomplete"
        )

    lock = dict(
        source_commit=progress["source_commit"],
        full_updates=2400,
        smoke_updates=4,
        member_fits=60,
        proofs=proofs,
        global_heldout_reads_before_lock=0,
        dl_val_treatment_reads_before_lock=0,
        protocol="fixed terminal40 across 4 arms x 3 seeds x 5 folds; no B selection or checkpoint search",
    )
    save(output / "selection-lock.json", lock)
    lock_sha = rt.sha(output / "selection-lock.json")
    store.put("selection-lock.json", (output / "selection-lock.json").read_bytes())
    progress["locked"] = True

    panels = {a: np.full((3, len(rows)), np.nan) for a in ("R882",) + sci.ARMS}
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    counts = {a: [0, 0, 0] for a in sci.ARMS}
    for si, seed in enumerate(sci.SEEDS):
        for outer in range(5):
            if rt.sha(output / "selection-lock.json") != lock_sha:
                raise RuntimeError("model lock changed")
            _, held = sci.partition(rows, seed, outer)
            allowed = {r["context_sha256"] for r in held}
            utilities = {cid: engine.utility(cid, allowed, "A") for cid in allowed}
            for row in held:
                cid = row["context_sha256"]
                panels["R882"][si, index[cid]] = utilities[cid][
                    int(engine.native[cid].argmin())
                ]
            for arm in sci.ARMS:
                engine.restore(states[(arm, seed, outer)])
                for row in held:
                    cid = row["context_sha256"]
                    choice = int(engine.score(cid).argmin())
                    panels[arm][si, index[cid]] = utilities[cid][choice]
                    counts[arm][si] += int(choice != int(engine.native[cid].argmin()))
            store.arrays(
                f"A-cell/{seed}-{outer}",
                **{
                    a: p[si, [index[r["context_sha256"]] for r in held]]
                    for a, p in panels.items()
                },
                context_ids=np.array([r["context_sha256"] for r in held]),
            )
    rt.check_frozen(engine.net, frozen_guard, engine.names, frozen)

    val_panels, val_families, val_elo = None, None, None
    val_adoption = None
    if val_rows is not None and val_gpu_inputs is not None and native_val is not None:
        if rt.sha(output / "selection-lock.json") != lock_sha:
            raise RuntimeError("model lock changed before DL-Val91 readout")
        engine.inputs.update(val_gpu_inputs)
        val_allowed = {r["context_sha256"] for r in val_rows}
        val_utilities = {
            cid: engine.utility(cid, val_allowed, "DL-Val") for cid in val_allowed
        }
        val_families = [r["family_id"] for r in val_rows]
        val_panels = {
            a: np.full((3, len(val_rows)), np.nan, dtype=np.float64)
            for a in ("R882",) + sci.ARMS
        }
        val_adoption = {a: [0, 0, 0] for a in sci.ARMS}
        for si, seed in enumerate(sci.SEEDS):
            val_panels["R882"][si] = np.asarray(native_val, dtype=np.float64)
            for arm in sci.ARMS:
                fold_utils = np.zeros((5, len(val_rows)), dtype=np.float64)
                for outer in range(5):
                    engine.restore(states[(arm, seed, outer)])
                    for vi, row in enumerate(val_rows):
                        cid = row["context_sha256"]
                        choice = int(engine.score(cid).argmin())
                        fold_utils[outer, vi] = val_utilities[cid][choice]
                        if outer == 0:
                            val_adoption[arm][si] += int(
                                choice != int(engine.native[cid].argmin())
                            )
                val_panels[arm][si] = fold_utils.mean(axis=0)
        store.arrays(
            "DL-Val91-panels",
            **val_panels,
            context_ids=np.array([r["context_sha256"] for r in val_rows]),
            family_ids=np.array(val_families),
        )
        val_elo = {
            a: elo.elo_vs_reference(
                val_panels[a], val_panels["R882"], val_families, seed=sci.ELO_SEED
            )
            for a in sci.ARMS
        }
        rt.check_frozen(engine.net, frozen_guard, engine.names, frozen)

    families = [r["family_id"] for r in rows]
    result = sci.readout(
        panels, families, val_panels=val_panels, val_families=val_families
    )
    store.arrays(
        "OOF-panels",
        **panels,
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(families),
    )

    andmask_proofs = [p for p in proofs if p["arm"] == "ANDMASK"]
    shuf_proofs = [p for p in proofs if p["arm"] == "ANDMASK_SHUF"]
    gdro_proofs = [p for p in proofs if p["arm"] == "GDRO"]
    invariant_exposure = dict(
        andmask_mean_kept_fraction=float(
            np.mean([p["invariant"]["mean_kept_fraction"] for p in andmask_proofs])
        ),
        andmask_shuf_mean_kept_fraction=float(
            np.mean([p["invariant"]["mean_kept_fraction"] for p in shuf_proofs])
        ),
        exact_per_step_density_match=bool(
            all(
                a["invariant"]["kept_counts"] == b["invariant"]["kept_counts"]
                for a, b in zip(andmask_proofs, shuf_proofs, strict=True)
            )
        ),
        min_families_per_microbatch=int(
            min(p["invariant"]["min_families_per_microbatch"] for p in proofs)
        ),
        gdro_max_final_weight=float(
            max(p["invariant"]["gdro_final_max_weight"] for p in gdro_proofs)
        ),
        gdro_min_final_weight=float(
            min(p["invariant"]["gdro_final_min_weight"] for p in gdro_proofs)
        ),
    )

    result.update(
        status="COMPLETE",
        optimizer_updates=2400,
        smoke_optimizer_updates=4,
        total_optimizer_updates=progress["updates"],
        member_fits=60,
        selection_lock_sha256=lock_sha,
        adoption_counts=counts,
        adoption_rates={a: [v / len(rows) for v in n] for a, n in counts.items()},
        dl_val91_adoption_counts=val_adoption,
        val_reg_inner_oof={
            a: dict(
                context_mean_per_seed=p.mean(1).tolist(),
                equal_family_mean_per_seed=sci.family_mean(p, families)
                .mean(1)
                .tolist(),
            )
            for a, p in panels.items()
        },
        elo_vs_r882={
            a: elo.elo_vs_reference(
                panels[a], panels["R882"], families, seed=sci.ELO_SEED
            )
            for a in sci.ARMS
        },
        dl_val91_elo_vs_r882=val_elo,
        invariant_exposure=invariant_exposure,
        U4=None,
        U4_reason="No multi-proposal budget in this selection protocol",
        serving_promotion=False,
    )
    return result
