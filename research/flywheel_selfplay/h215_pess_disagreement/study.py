"""H215 two-stage experiment; no A outcomes until all FIT choices/B panels lock."""

import io
import json
import numpy as np
import torch


def execute(
    engine,
    sci,
    selector,
    calibration,
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
):
    names = engine.names
    families = [r["family_id"] for r in rows]
    index = {r["context_sha256"]: i for i, r in enumerate(rows)}
    bpanels = {arm: np.full((3, len(rows)), np.nan) for arm in ("R882",) + sci.ARMS}
    cells = []
    all_proofs = []
    outer_states = {}
    # The exact H178-calibrated rule requires four panels, never a one-arm shortcut.
    qualified = selector.resolve_h177_verdict(
        True, calibration, calibration["h177_analysis_sha256"]
    )
    if qualified["verdict"] != "ADVANCE":
        raise RuntimeError("H178 calibration gate failed")

    def ensemble(fit, evaluation, seed, outer, phase):
        costs = {r["context_sha256"]: [] for r in evaluation}
        states = []
        for member in range(10):
            check()
            ms = sci.member_seed(seed, outer, phase, member)
            label = f"{seed}-{outer}-{phase}-{member}"
            state, proof = engine.train(fit, ms, label)
            all_proofs.append(proof)
            raw = io.BytesIO()
            torch.save(state, raw)
            store.put("members/" + label + ".pt", raw.getvalue())
            states.append(state)
            for cid in costs:
                costs[cid].append(engine.score(cid))
        # Full frozen/DOPE hashes before advancing to another independent ensemble.
        rt.check_frozen(engine.net, frozen_guard, names, frozen)
        sigma = {cid: sci.disagreement(np.stack(v)) for cid, v in costs.items()}
        store.arrays("sigma/" + f"{seed}-{outer}-{phase}", **sigma)
        return states, sigma

    def read_panel(target, si, evaluation, sigma, kappa, seed, outer, role):
        allowed = {r["context_sha256"] for r in evaluation}
        adoption = {arm: 0 for arm in sci.ARMS}
        for row in evaluation:
            cid = row["context_sha256"]
            i = index[cid]
            base = engine.native[cid]
            sig = sigma[cid]
            utility = engine.utility(cid, allowed, role)
            ref = int(base.argmin())
            target["R882"][si, i] = utility[ref]
            shuffled = np.random.default_rng(
                sci.stream(seed, outer, cid, "placebo")
            ).permutation(sig)
            for arm, s, k in [
                ("PESS_NEST", sig, kappa),
                ("PESS_FIX", sig, 1),
                ("PESS_SHUF", shuffled, kappa),
                ("OPT", sig, -1),
            ]:
                pick = sci.pick(base, s, k)
                target[arm][si, i] = utility[pick]
                adoption[arm] += int(pick != ref)
        return adoption

    for si, seed in enumerate(sci.SEEDS):
        for outer in range(5):
            check()
            fit, b, a = sci.partition(rows, seed, outer)
            fit_ids = [r["context_sha256"] for r in fit]
            fit_index = {cid: i for i, cid in enumerate(fit_ids)}
            gains = {k: np.full(len(fit), np.nan) for k in sci.KAPPAS}
            inner_sigmas = []
            for inner in range(3):
                inner_fit, held = sci.inner_partition(fit, seed, outer, inner)
                _, sigma = ensemble(inner_fit, held, seed, outer, inner)
                allowed = {r["context_sha256"] for r in held}
                for row in held:
                    cid = row["context_sha256"]
                    u = engine.utility(cid, allowed, "FIT")
                    base = engine.native[cid]
                    ref = int(base.argmin())
                    for k in sci.KAPPAS:
                        gains[k][fit_index[cid]] = (
                            u[sci.pick(base, sigma[cid], k)] - u[ref]
                        )
                    inner_sigmas.extend(sigma[cid].tolist())
            kappa, grid = sci.choose_kappa(gains, [r["family_id"] for r in fit])
            states, sigma = ensemble(fit, b, seed, outer, 3)
            outer_states[(seed, outer)] = states
            read_panel(bpanels, si, b, sigma, kappa, seed, outer, "B")
            b_indices = [index[r["context_sha256"]] for r in b]
            store.arrays(
                f"B-cell/{seed}-{outer}",
                **{arm: panel[si, b_indices] for arm, panel in bpanels.items()},
                context_ids=np.array([r["context_sha256"] for r in b]),
            )
            cell = dict(
                seed=seed,
                outer=outer,
                kappa=kappa,
                inner_family_mean_gain={str(k): v for k, v in grid.items()},
                FIT_sigma_quantiles=np.quantile(
                    inner_sigmas, [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                member_seeds=[sci.member_seed(seed, outer, 3, m) for m in range(10)],
            )
            cells.append(cell)
            store.put(f"cells/{seed}-{outer}.json", (json.dumps(cell) + "\n").encode())
    if progress["updates"] != 24000 or any(
        not np.isfinite(x).all() for x in bpanels.values()
    ):
        raise RuntimeError("incomplete FIT/B coverage")
    decisions = selector.H177_DECISION_RULE.select(
        rows,
        bpanels["R882"],
        {sci.SELECTOR_NAMES[a]: bpanels[a] for a in sci.ARMS},
        seed=215150,
    )
    store.arrays("B-panels", **bpanels)
    lock = dict(
        source_commit=progress["source_commit"],
        updates=progress["updates"],
        cells=cells,
        proofs=all_proofs,
        pooled_B_decisions=decisions,
        selector_map=sci.SELECTOR_NAMES,
        calibration=calibration,
    )
    save(output / "selection-lock.json", lock)
    lock_sha = rt.sha(output / "selection-lock.json")
    store.put("selection-lock.json", (output / "selection-lock.json").read_bytes())
    progress["locked"] = True
    panels = {arm: np.full((3, len(rows)), np.nan) for arm in ("R882",) + sci.ARMS}
    adoption = {arm: [0, 0, 0] for arm in sci.ARMS}
    for cell in cells:
        check()
        if rt.sha(output / "selection-lock.json") != lock_sha:
            raise RuntimeError("selection lock mutated")
        seed, outer = cell["seed"], cell["outer"]
        si = sci.SEEDS.index(seed)
        _, _, a = sci.partition(rows, seed, outer)
        costs = {r["context_sha256"]: [] for r in a}
        for state in outer_states[(seed, outer)]:
            engine.restore(state)
            for cid in costs:
                costs[cid].append(engine.score(cid))
        sigma = {cid: sci.disagreement(np.stack(v)) for cid, v in costs.items()}
        store.arrays(f"A-sigma/{seed}-{outer}", **sigma)
        counts = read_panel(panels, si, a, sigma, cell["kappa"], seed, outer, "A")
        for arm, n in counts.items():
            adoption[arm][si] += n
    if any(not np.isfinite(x).all() for x in panels.values()):
        raise RuntimeError("incomplete A coverage")
    rt.check_frozen(engine.net, frozen_guard, names, frozen)
    contrasts = {a: sci.contrast(panels[a], panels["R882"], families) for a in sci.ARMS}
    adjusted = selector.holm({a: contrasts[a]["p"] for a in sci.ARMS})
    primary_b = decisions[sci.SELECTOR_NAMES["PESS_NEST"]]["admitted"]
    verdict = sci.verdict(
        contrasts["PESS_NEST"], adjusted["PESS_NEST"], primary_b, True
    )
    store.arrays(
        "A-panels",
        **panels,
        context_ids=np.array([r["context_sha256"] for r in rows]),
        family_ids=np.array(families),
    )
    return dict(
        status="COMPLETE",
        verdict=verdict,
        contrasts=contrasts,
        holm_p=adjusted,
        holm_multiplicity=4,
        pooled_B_decisions=decisions,
        primary_B_admitted=primary_b,
        elo_vs_r882={
            a: elo.elo_vs_reference(panels[a], panels["R882"], families, seed=215151)
            for a in sci.ARMS
        },
        adoption_counts=adoption,
        adoption_rates={
            a: [v / len(rows) for v in counts] for a, counts in adoption.items()
        },
        val_reg_inner_oof={
            a: dict(
                context_mean_per_seed=p.mean(1).tolist(),
                equal_family_mean_per_seed=sci.family_mean(p, families)
                .mean(1)
                .tolist(),
            )
            for a, p in panels.items()
        },
        optimizer_updates=progress["updates"],
        member_fits=len(all_proofs),
        selection_lock_sha256=lock_sha,
        U4=None,
        U4_reason="No verified multi-proposal budget",
        serving_promotion=False,
    )
