"""H303 evaluation: ENS_VOTE2_NESTED (primary) and H302's ENS_VOTE2 (recomputed from the H303 regression run's H302 fold
locks, picks hashes checked against H302's published hashes), with H302's statistics (analysis_h302, unchanged: shared
whole-family bootstrap 302150, 10,000 draws) plus the concentration audit.

Decision (pre-registered; primary ENS_VOTE2_NESTED; ties never win):
  NESTED_CONFIRMED iff (1) median over the 3 root training seeds of the family-equal gain > +0.001;
  (2) per-seed one-sided LCB95 (0.05 quantile) > 0 on 3/3 seeds; (3) harmed-family share (loss > 0.01) <= 35% each seed;
  (4) invalid excess <= 3 and exact one-sided McNemar p >= 0.05 each seed; (5) after dropping the top-5 families by
  gain, the family-equal gain's one-sided LCB95 (whole-family bootstrap of the remaining 200 families, seed 302150)
  > 0 on >= 2/3 seeds. Else NESTED_NOT_CONFIRMED, and H302 is labelled "design-leak-dependent" if it passes (1)-(4)
  while the nested arm does not.
Concentration audit per arm and seed: gain after dropping the top 1/5/10 families; share of the total gain from the
top 10 families; family-median gain with CI (from analysis_h302)."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

H302_PICKS_SHA = {
    30001: "b47088d981afa1a53a8926afc1a025492fafcb9591a9cc52aa191e452c8560d7",
    30002: "0c5761b312cd9366e2fd1e3f49ee7cb37908ff1b8d2228f49d4604d4ece40bed",
    30003: "a4ff743c9b974aa5145cb0aebed51b946bddda47b237dc21024ef2011c95c9ee",
}
H302_SCRATCH = Path(os.environ.get("H302_REGRESS_SCRATCH", "/home/ubuntu/jope-data/h303/h302_regress/scratch"))
OUT = Path(os.environ.get("H303_OUT", "/home/ubuntu/jope-data/h303/out"))
DROP = (1, 5, 10)


def h302_picks(G):
    """H302's ENS_VOTE2 out-of-fold picks, recomputed from the regression run's fold locks exactly as run_h302 scores them."""
    a302, ens, bagged, veto, P8, train_rows = (G[x] for x in ("a302", "ens", "bagged", "veto", "P8", "train_rows"))
    cids = [r["context_sha256"] for r in train_rows]
    picks = np.full((3, len(cids)), -1, dtype=np.int32)
    for k in range(4):
        fd = H302_SCRATCH / f"fold_{k}"
        lock = json.loads((fd / f"fold_{k}_fit_lock.json").read_text())
        test = [i for i, c in enumerate(cids) if G["fam_to_fold"][G["fam_by_cid"][c]] == k]
        X = np.concatenate([P8[cids[i]]["X"] for i in test])
        for s, rseed in enumerate(a302.ROOT_TRAINING_SEEDS):
            outs = [
                ens.score_single_member_from_disk(fd, c, X, len(test), ens.K)
                for c in lock["members_by_root_seed"][str(rseed)]
            ]
            agg = ens.aggregate_ensemble_picks(outs, len(test), q_fixed=bagged.Q_FIXED, tau=veto.TAU)
            for li, gi in enumerate(test):
                picks[s, gi] = int(agg["ENS_VOTE2"][li])
    hashes = {
        rs: hashlib.sha256(np.asarray(picks[s], dtype=np.int32).tobytes()).hexdigest()
        for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)
    }
    return picks, hashes


def panel(G, picks):
    P8, train_rows = G["P8"], G["train_rows"]
    cids = [r["context_sha256"] for r in train_rows]
    n = len(cids)
    u = np.zeros((3, n, 2))
    inv = np.zeros((3, n, 2), dtype=bool)
    for s in range(3):
        for i, c in enumerate(cids):
            j = int(picks[s, i])
            for col, jj in ((0, 0), (1, j)):
                u[s, i, col] = float(P8[c]["utility_k"][jj])
                inv[s, i, col] = (not bool(P8[c]["gate_k"][jj])) or float(P8[c]["utility_k"][jj]) <= -2.0 + 1e-9
    return u, inv


def concentration(D, boot_seed, draws):
    """D: (3, families) family gains. Drop-top-N gains, top-10 share, drop-top-5 LCB95 (bootstrap of the remainder)."""
    out = []
    for d in D:
        order = np.argsort(-d, kind="mergesort")
        rec = {f"gain_drop_top{n}": float(np.delete(d, order[:n]).mean()) for n in DROP}
        total = float(d.sum())
        rec["top10_share_of_total_gain"] = float(d[order[:10]].sum() / total) if total > 0 else None
        rest = np.delete(d, order[:5])
        ix = np.random.default_rng(boot_seed).integers(0, len(rest), (draws, len(rest)))
        b = rest[ix].mean(axis=1)
        rec["drop_top5_ci95"] = np.quantile(b, [0.025, 0.975]).tolist()
        rec["drop_top5_lcb95_one_sided"] = float(np.quantile(b, 0.05))
        out.append(rec)
    return out


def judge(a302, u, inv, families, picks):
    s = a302.summarize_three_seeds(u, inv, families, picks.tolist(), boot_seed=a302.BOOT_SEED)
    ni = a302.invalid_noninferiority(inv, len(set(families)))
    conc = concentration(s["_D"], a302.BOOT_SEED, a302.DRAWS)
    crit = dict(
        c1_median_gain_gt_0_001=bool(s["median_gain_across_seeds"] > 0.001),
        c2_lcb95_gt_0_3of3=bool(all(x > 0 for x in s["per_seed_lcb_one_sided"]["0.05"])),
        c3_harmed_le_35pct_each=bool(all(h <= 0.35 for h in s["harmed_share_by_seed"])),
        c4_invalid_excess_le_3_and_mcnemar=bool(all(x["excess"] <= 3 and x["mcnemar_p"] >= 0.05 for x in ni)),
        c5_drop_top5_lcb95_gt_0_on_2of3=bool(sum(c["drop_top5_lcb95_one_sided"] > 0 for c in conc) >= 2),
    )
    pub = {k: v for k, v in s.items() if not k.startswith("_")}
    return dict(summary=pub, invalid_detail=ni, concentration=conc, criteria=crit, passes=bool(all(crit.values())),
                passes_c1_to_c4=bool(all(crit[k] for k in list(crit)[:4])))  # fmt: skip


def main(G, score_phase, locks):
    a302 = G["a302"]
    assert len(locks) == 12, f"expected 12 fit locks, found {len(locks)}"
    OUT.mkdir(parents=True, exist_ok=True)
    fams = [r["family_id"] for r in G["train_rows"]]
    p302, h302_hashes = h302_picks(G)
    h302_regression = all(h302_hashes[rs] == H302_PICKS_SHA[rs] for rs in H302_PICKS_SHA)
    if not h302_regression:
        (OUT / "REGRESSION_FAIL.json").write_text(json.dumps(dict(got=h302_hashes, want=H302_PICKS_SHA), indent=1))
        raise RuntimeError("H302 picks did not reproduce")
    pn, lock_sha = score_phase(locks)
    nested_hashes = {
        rs: hashlib.sha256(np.asarray(pn[s], dtype=np.int32).tobytes()).hexdigest()
        for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)
    }
    np.savez_compressed(OUT / "picks.npz", ENS_VOTE2_NESTED=pn, ENS_VOTE2_H302=p302)
    res = {}
    for name, p in (("ENS_VOTE2_NESTED", pn), ("ENS_VOTE2_H302", p302)):
        u, inv = panel(G, p)
        res[name] = judge(a302, u, inv, fams, p)
    nested, h302 = res["ENS_VOTE2_NESTED"], res["ENS_VOTE2_H302"]
    verdict = "NESTED_CONFIRMED" if nested["passes"] else "NESTED_NOT_CONFIRMED"
    h302_status = (
        "design-leak-dependent"
        if (not nested["passes"] and h302["passes_c1_to_c4"])
        else ("survives nesting" if nested["passes"] else "not established")
    )
    chosen = [
        dict(
            outer_fold=lk["outer_fold"],
            root_seed=lk["root_seed"],
            **lk["chosen"],
            n_tied=lk["n_tied"],
            elapsed_sec=lk["elapsed_sec"],
        )
        for lk in locks
    ]
    agree = {str(rs): float(np.mean(pn[s] == p302[s])) for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)}
    full = dict(wave="H303", verdict=verdict, h302_status=h302_status, h302_regression_picks_exact=h302_regression, h302_hashes=h302_hashes,
                nested_hashes=nested_hashes, fit_lock_sha256=lock_sha, chosen_configs=chosen, nested_vs_h302_pick_agreement=agree, arms=res,
                inner_tables={f"{lk['outer_fold']}:{lk['root_seed']}": lk["inner_table"] for lk in locks})  # fmt: skip
    (OUT / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    print(json.dumps(dict(verdict=verdict, h302_status=h302_status, h302_regression=h302_regression, chosen=chosen,
                          nested_gain=nested["summary"]["per_seed_family_mean_gain"], h302_gain=h302["summary"]["per_seed_family_mean_gain"],
                          criteria=nested["criteria"])), flush=True)  # fmt: skip
