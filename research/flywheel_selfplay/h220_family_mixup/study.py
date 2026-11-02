"""H220 fixed full-fold execution: no held-out outcome reads before model lock."""

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
):
    # Timing smoke has no held-out utility read and no scientific score.
    fit, _ = sci.partition(rows, sci.SEEDS[0], 0)
    started = time.monotonic()
    _, smoke = engine.train(
        fit,
        "CONTROL",
        220001,
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
        scope="TRAIN FIT-only timing; no held-out labels, selection or promotion; included in one90min budget",
        proof=smoke,
    )
    save(output / "timing-smoke.json", smoke_receipt)
    store.put("timing-smoke.json", (output / "timing-smoke.json").read_bytes())
    progress["smoke_updates"] = progress["updates"]
    states, proofs = {}, []
    for seed in sci.SEEDS:
        for outer in range(5):
            fit, _ = sci.partition(rows, seed, outer)
            for arm in sci.ARMS:
                check()
                label = f"{arm}-{seed}-{outer}"
                state, proof = engine.train(
                    fit, arm, seed, outer, label, legacy, sci, output, checkpoint
                )
                rt.check_frozen(engine.net, frozen_guard, engine.names, frozen)
                states[(arm, seed, outer)] = state
                proofs.append(proof)
                b = io.BytesIO()
                torch.save(state, b)
                store.put("adapters/" + label + ".pt", b.getvalue())
    if progress["updates"] != 1204 or len(proofs) != 30:
        raise RuntimeError("full matched training budget incomplete")
    lock = dict(
        source_commit=progress["source_commit"],
        full_updates=1200,
        smoke_updates=4,
        proofs=proofs,
        global_heldout_reads_before_lock=0,
        protocol="fixed terminal40, no B selection or checkpoint search",
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
    families = [r["family_id"] for r in rows]
    result = sci.readout(panels, families)
    store.arrays(
        "OOF-panels",
        **panels,
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(families),
    )
    result.update(
        status="COMPLETE",
        optimizer_updates=1200,
        smoke_optimizer_updates=4,
        total_optimizer_updates=progress["updates"],
        member_fits=30,
        selection_lock_sha256=lock_sha,
        adoption_counts=counts,
        adoption_rates={a: [v / len(rows) for v in n] for a, n in counts.items()},
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
            a: elo.elo_vs_reference(panels[a], panels["R882"], families, seed=220151)
            for a in sci.ARMS
        },
        U4=None,
        U4_reason="No multi-proposal budget in this selection protocol",
        serving_promotion=False,
    )
    return result
