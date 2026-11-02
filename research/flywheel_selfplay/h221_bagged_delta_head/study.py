"""Fixed 165-fit, 45-gate protocol. All gates lock before any A outcome read."""

import numpy as np


def execute(engine, sci, rows, store, output, commit, save, file_hash, check, elo):
    states = {}
    proofs = []
    cells = []
    for seed in sci.SEEDS:
        for outer in range(5):
            fit, b_rows, _ = sci.partition(rows, seed, outer)
            fitfamilies = {r["family_id"] for r in fit}
            if len(fitfamilies) != 123:
                raise RuntimeError("FIT closure")
            single, proof = engine.fit(fit, seed, outer, None, False)
            proofs.append(proof)
            bags = {False: [], True: []}
            for member in range(5):
                for shuffled in (False, True):
                    state, proof = engine.fit(fit, seed, outer, member, shuffled)
                    bags[shuffled].append(state)
                    proofs.append(proof)
            allowed = {r["context_sha256"] for r in b_rows}
            for arm, group in [
                ("DELTA_SAFE", [single]),
                ("BAG5_SAFE", bags[False]),
                ("BAG5_SHUF_SAFE", bags[True]),
            ]:
                check()
                states[(seed, outer, arm)] = group
                scores = engine.predict(group, b_rows)
                gains = {t: [] for t in sci.THRESHOLDS}
                for row in b_rows:
                    cid = row["context_sha256"]
                    ref = int(engine.native[cid].argmin())
                    u = engine.utility(cid, allowed, "B")
                    for t in sci.THRESHOLDS:
                        gains[t].append(
                            float(u[sci.pick(scores[cid], ref, t)] - u[ref])
                        )
                tau, grid = sci.select_threshold(
                    gains, [r["family_id"] for r in b_rows]
                )
                store.arrays(f"B/{seed}-{outer}-{arm}", **scores)
                cells.append(
                    dict(
                        seed=seed,
                        outer=outer,
                        arm=arm,
                        threshold=None if np.isinf(tau) else tau,
                        grid=grid,
                        head_hashes=[engine.rt.tensor_hash(s) for s in group],
                    )
                )
    if engine.progress["updates"] != 21120 or len(proofs) != 165 or len(cells) != 45:
        raise RuntimeError("fixed training/gate inventory incomplete")
    lock = dict(
        source_commit=commit,
        updates=engine.progress["updates"],
        proofs=proofs,
        cells=cells,
        A_outcome_reads_before_lock=0,
    )
    save(output / "selection-lock.json", lock)
    lock_hash = file_hash(output / "selection-lock.json")
    store.put("selection-lock.json", (output / "selection-lock.json").read_bytes())
    engine.progress["locked"] = True
    panels = {a: np.full((3, len(rows)), np.nan) for a in sci.ARMS}
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    adoption = {a: [0, 0, 0] for a in sci.ARMS[1:]}
    for cell in cells:
        check()
        if file_hash(output / "selection-lock.json") != lock_hash:
            raise RuntimeError("lock mutated")
        seed, outer, arm = cell["seed"], cell["outer"], cell["arm"]
        _, _, a_rows = sci.partition(rows, seed, outer)
        group = states[(seed, outer, arm)]
        if [engine.rt.tensor_hash(s) for s in group] != cell["head_hashes"]:
            raise RuntimeError("terminal head changed")
        scores = engine.predict(group, a_rows)
        allowed = {r["context_sha256"] for r in a_rows}
        tau = float("inf") if cell["threshold"] is None else cell["threshold"]
        si = sci.SEEDS.index(seed)
        for row in a_rows:
            cid = row["context_sha256"]
            ref = int(engine.native[cid].argmin())
            u = engine.utility(cid, allowed, "A")
            selected = sci.pick(scores[cid], ref, tau)
            panels["R882"][si, index[cid]] = u[ref]
            panels[arm][si, index[cid]] = u[selected]
            adoption[arm][si] += int(selected != ref)
        store.arrays(f"A/{seed}-{outer}-{arm}", **scores)
    if any(not np.isfinite(p).all() for p in panels.values()):
        raise RuntimeError("incomplete endpoint")
    families = [r["family_id"] for r in rows]
    contrasts = {
        a: sci.contrast(panels[a], panels["R882"], families) for a in sci.ARMS[1:]
    }
    store.arrays(
        "panels",
        **panels,
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(families),
    )
    return dict(
        status="COMPLETE",
        verdict=sci.verdict(contrasts["BAG5_SAFE"]),
        primary="BAG5_SAFE",
        holm_multiplicity=1,
        holm_p={"BAG5_SAFE": contrasts["BAG5_SAFE"]["p"]},
        contrasts=contrasts,
        elo_vs_r882={
            a: elo.elo_vs_reference(panels[a], panels["R882"], families, seed=221151)
            for a in sci.ARMS[1:]
        },
        adoption_counts=adoption,
        adoption_rates={a: [x / len(rows) for x in v] for a, v in adoption.items()},
        val_reg_inner_oof={
            a: dict(
                context_mean_per_seed=p.mean(1).tolist(),
                equal_family_mean_per_seed=sci.family_mean(p, families)
                .mean(1)
                .tolist(),
            )
            for a, p in panels.items()
        },
        U4=None,
        U4_status="not measured; no multi-proposal budget",
        contexts_per_seed=len(rows),
        optimizer_updates=engine.progress["updates"],
        member_fits=len(proofs),
        selection_lock_sha256=lock_hash,
    )
