"""H306 evaluation: ENS_VOTE2_NESTED_XGRID vs slot 0 (H303's judge: H302 statistics + concentration audit) and PAIRED vs
H303's ENS_VOTE2_NESTED (archived picks.npz from S3 H303 tar, hash-checked against H303's published nested hashes).

Decision (pre-registered; ties never win):
  XGRID_ADVANCES iff (1) median over the 3 root seeds of the paired family-equal gain (XGRID - H303 NESTED) > +0.001;
  (2) paired one-sided LCB95 (0.05 quantile, 10,000 whole-family draws, RNG 302150) > 0 on >= 2/3 seeds;
  (3) XGRID harmed-family share (loss > 0.01 vs slot 0) <= 35% each seed; (4) XGRID invalid excess vs slot 0 <= 3 and
  exact one-sided McNemar p >= 0.05 each seed; (5) XGRID drop-top-5 LCB95 > 0 on >= 2/3 seeds.
  Else XGRID_NOT_ADVANCED (H303 grid stands); the chosen-config histogram reports whether choices still pile at an edge.
Elo: MM Bradley-Terry, reference = r882 slot0 (DL-Train OOF) = 1500, NOT the flywheel incumbent r3175 (unmeasured)."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("H306_DATA", "/home/ubuntu/jope-data/h306"))
H303_NESTED_SHA = {
    30001: "0b187de957f692238ee0c21702822baae6f7845f344ecb5c3c0db483a711a7ac",
    30002: "35530bb5db8b9a93e093d759510a575569d3dfb22c9d46da9cd09fc14375c2f8",
    30003: "b8fb9c1476a0e6ddb18c05d0fba63b23c1bf93d99a1492d82c8f2d6da42ffda3",
}


def picks_sha(p):
    return hashlib.sha256(np.asarray(p, dtype=np.int32).tobytes()).hexdigest()


def family_d(u, families):
    """(3, ctx, 2) utilities -> (3, families) arm-minus-slot0 family means, families sorted as in analysis_h302."""
    fams = sorted(set(families))
    f = np.asarray(families)
    return np.stack([np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(3)])


def paired(Dx, Dn, boot_seed=302150, draws=10000):
    """Per-seed paired family-equal gain (XGRID - NESTED) with a whole-family bootstrap CI and one-sided LCB95."""
    n = Dx.shape[1]
    idx = np.random.default_rng(boot_seed).integers(0, n, size=(draws, n))
    out = []
    for s in range(3):
        d = Dx[s] - Dn[s]
        b = d[idx].mean(axis=1)
        out.append(
            dict(
                gain=float(d.mean()),
                ci95=np.quantile(b, [0.025, 0.975]).tolist(),
                lcb95_one_sided=float(np.quantile(b, 0.05)),
                wtl=[int((d > 1e-4).sum()), int((np.abs(d) <= 1e-4).sum()), int((d < -1e-4).sum())],
            )
        )
    return out


def edge_histogram(chosen):
    c = dict(
        tau=dict(sorted(Counter(str(x["tau"]) for x in chosen).items())),
        k=dict(sorted(Counter(str(x["k"]) for x in chosen).items())),
    )
    c["at_new_low_edge_tau_0_025"] = sum(x["tau"] == 0.025 for x in chosen)
    c["at_new_low_edge_k_4"] = sum(x["k"] == 4 for x in chosen)
    c["at_upper_edge_tau_0_10"] = sum(x["tau"] == 0.10 for x in chosen)
    c["at_upper_edge_k_6"] = sum(x["k"] == 6 for x in chosen)
    return c


def decide(x_judge, pair):
    crit = dict(
        c1_median_paired_gain_gt_0_001=bool(np.median([p["gain"] for p in pair]) > 0.001),
        c2_paired_lcb95_gt_0_on_2of3=bool(sum(p["lcb95_one_sided"] > 0 for p in pair) >= 2),
        c3_harmed_le_35pct_each=x_judge["criteria"]["c3_harmed_le_35pct_each"],
        c4_invalid_excess_le_3_and_mcnemar=x_judge["criteria"]["c4_invalid_excess_le_3_and_mcnemar"],
        c5_drop_top5_lcb95_gt_0_on_2of3=x_judge["criteria"]["c5_drop_top5_lcb95_gt_0_on_2of3"],
    )
    return crit, ("XGRID_ADVANCES" if all(crit.values()) else "XGRID_NOT_ADVANCED")


def main(r303, ev, locks):
    G = r303.G
    a302 = G["a302"]
    assert len(locks) == 12, f"expected 12 fit locks, found {len(locks)}"
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "h303_ref" / "out" / "picks.npz") as z:
        pn = np.asarray(z["ENS_VOTE2_NESTED"], dtype=np.int32)
    got = {rs: picks_sha(pn[s]) for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)}
    if got != H303_NESTED_SHA:
        raise RuntimeError(f"archived H303 nested picks hash mismatch: {got}")
    px, lock_sha = r303.score_phase(locks)
    fams = [r["family_id"] for r in G["train_rows"]]
    ux, ix = ev.panel(G, px)
    un, _ = ev.panel(G, pn)
    xj = ev.judge(a302, ux, ix, fams, px)
    pair = paired(family_d(ux, fams), family_d(un, fams))
    crit, verdict = decide(xj, pair)
    chosen = [
        dict(outer_fold=lk["outer_fold"], root_seed=lk["root_seed"], **lk["chosen"], n_tied=lk["n_tied"])
        for lk in locks
    ]
    np.savez_compressed(out / "picks.npz", ENS_VOTE2_NESTED_XGRID=px, ENS_VOTE2_NESTED_H303=pn)
    full = dict(
        wave="H306",
        verdict=verdict,
        criteria=crit,
        elo_reference="r882 slot0 (DL-Train OOF) = 1500; NOT the flywheel incumbent r3175 (r3175-anchored Elo unmeasured)",
        xgrid_picks_sha256={str(rs): picks_sha(px[s]) for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)},
        h303_nested_picks_sha256={str(k): v for k, v in H303_NESTED_SHA.items()},
        xgrid_vs_h303_pick_agreement={
            str(rs): float(np.mean(px[s] == pn[s])) for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)
        },
        fit_lock_sha256=lock_sha,
        chosen_configs=chosen,
        chosen_histogram=edge_histogram(chosen),
        paired_xgrid_minus_h303_nested=dict(
            per_seed=pair,
            median_gain_across_seeds=float(np.median([p["gain"] for p in pair])),
        ),
        xgrid_vs_slot0=xj,
        inner_tables={f"{lk['outer_fold']}:{lk['root_seed']}": lk["inner_table"] for lk in locks},
    )
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    print(json.dumps(dict(verdict=verdict, criteria=crit, paired=pair, hist=full["chosen_histogram"])), flush=True)
