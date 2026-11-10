"""H308 evaluation: H303 NESTED ENS_VOTE2 on new outer partitions (P1, P2) vs r882 slot0, with H303's judge (H302 statistics,
shared whole-family bootstrap RNG 302150 x 10,000, concentration audit), per-family gain correlation across seeds and the
chosen-config histogram. P0 (H303's partition) is reported from H303's archived picks, no refit.

Decision (pre-registered; ties never win). PARTITION_ROBUST iff all of:
  c1 on EVERY new partition: median over the 3 seeds of the family-equal gain > +0.001 AND pooled one-sided LCB95 > 0;
  c2 per-seed one-sided LCB95 > 0 in >= 5/6 (partition, seed) cells;
  c3 drop-top-5 one-sided LCB95 > 0 in >= 4/6 cells;
  c4 harmed-family share (loss > 0.01) <= 35% in every cell;
  c5 invalid excess vs slot 0 <= 3 in every cell.
If P2 had to be dropped for time, c2/c3 become >= 3/3 and >= 2/3 on P1's cells. Else PARTITION_FRAGILE (failed
criteria named). Neither verdict authorizes serving."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from run_h308 import SALTS, outer_fold

DATA = Path(os.environ.get("H308_DATA", "/home/ubuntu/jope-data/h308"))
ELO_LABEL = "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured"


def family_d(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return np.stack([np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(3)])


def seed_corr(D):
    c = np.corrcoef(D)
    return {"30001_30002": float(c[0, 1]), "30001_30003": float(c[0, 2]), "30002_30003": float(c[1, 2])}


def histogram(chosen):
    return {
        "tau": dict(sorted(Counter(str(x["tau"]) for x in chosen).items())),
        "k": dict(sorted(Counter(str(x["k"]) for x in chosen).items())),
        "a": dict(sorted(Counter(str(x["a"]) for x in chosen).items())),
        "h303_design_point_0.10_6_2": sum(x["tau"] == 0.10 and x["k"] == 6 and x["a"] == 2 for x in chosen),
    }


def cells(res, parts):
    return [(p, s) for p in parts for s in range(3)]


def decide(res, parts):
    cs = cells(res, parts)
    need2, need3 = (5, 4) if len(parts) == 2 else (3, 2)
    lcb = [res[p]["summary"]["per_seed_lcb_one_sided"]["0.05"][s] for p, s in cs]
    d5 = [res[p]["concentration"][s]["drop_top5_lcb95_one_sided"] for p, s in cs]
    crit = dict(
        c1_each_partition_median_gt_0_001_and_pooled_lcb95_gt_0=bool(
            all(
                res[p]["summary"]["median_gain_across_seeds"] > 0.001
                and res[p]["summary"]["pooled_lcb_one_sided_95"] > 0
                for p in parts
            )
        ),
        c2_per_seed_lcb95_gt_0_cells=bool(sum(x > 0 for x in lcb) >= need2),
        c3_drop_top5_lcb95_gt_0_cells=bool(sum(x > 0 for x in d5) >= need3),
        c4_harmed_le_35pct_every_cell=bool(all(res[p]["summary"]["harmed_share_by_seed"][s] <= 0.35 for p, s in cs)),
        c5_invalid_excess_le_3_every_cell=bool(all(res[p]["invalid_detail"][s]["excess"] <= 3 for p, s in cs)),
    )
    counts = dict(
        per_seed_lcb95_pos=f"{sum(x > 0 for x in lcb)}/{len(cs)} (need {need2})",
        drop_top5_lcb95_pos=f"{sum(x > 0 for x in d5)}/{len(cs)} (need {need3})",
    )
    verdict = "PARTITION_ROBUST" if all(crit.values()) else "PARTITION_FRAGILE"
    return crit, counts, verdict, [k for k, v in crit.items() if not v]


def main(r303, ev, scored, loader_counters):
    G = r303.G
    a302 = G["a302"]
    fams = [r["family_id"] for r in G["train_rows"]]
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    res, extra = {}, {}
    for part, (picks, lock_sha, chosen, guards) in scored.items():
        u, inv = ev.panel(G, picks)
        res[part] = ev.judge(a302, u, inv, fams, picks)
        extra[part] = dict(
            picks_sha256={
                str(rs): hashlib.sha256(picks[s].astype(np.int32).tobytes()).hexdigest()
                for s, rs in enumerate(a302.ROOT_TRAINING_SEEDS)
            },
            family_gain_corr_across_seeds=seed_corr(family_d(u, fams)),
            chosen_configs=chosen,
            chosen_histogram=histogram(chosen),
            fit_lock_sha256=lock_sha,
            outer_test_families_by_fold=[
                sum(1 for f in G["_fams"] if outer_fold(SALTS[part], f) == k) for k in range(4)
            ],
            guard_receipts=guards,
        )
    # P0 from H303's archived picks (no refit), same judge
    with np.load(DATA / "h303_ref" / "out" / "picks.npz") as z:
        p0 = np.asarray(z["ENS_VOTE2_NESTED"], dtype=np.int32)
    u0, inv0 = ev.panel(G, p0)
    p0_res = ev.judge(a302, u0, inv0, fams, p0)
    h303 = json.loads((DATA / "h303_ref" / "out" / "metrics_full.json").read_text())
    p0_extra = dict(
        family_gain_corr_across_seeds=seed_corr(family_d(u0, fams)),
        chosen_histogram=histogram(h303["chosen_configs"]),
        h303_recomputed_matches_published=bool(
            np.allclose(
                p0_res["summary"]["per_seed_family_mean_gain"],
                h303["arms"]["ENS_VOTE2_NESTED"]["summary"]["per_seed_family_mean_gain"],
                atol=0,
                rtol=0,
            )
        ),
    )
    parts = list(scored)
    crit, counts, verdict, failed = decide(res, parts)
    full = dict(
        wave="H308",
        verdict=verdict,
        criteria=crit,
        criteria_counts=counts,
        failed_criteria=failed,
        partitions_scored=parts,
        elo_label=ELO_LABEL,
        loader_counters=loader_counters,
        partitions={p: dict(judge=res[p], **extra[p]) for p in parts},
        P0_h303_no_refit=dict(judge=p0_res, **p0_extra),
    )
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(out / "picks.npz", **{f"ENS_VOTE2_NESTED_{p}": scored[p][0] for p in parts})
    print(
        json.dumps(
            dict(
                verdict=verdict,
                criteria=crit,
                counts=counts,
                medians={p: res[p]["summary"]["median_gain_across_seeds"] for p in parts},
                gains={p: res[p]["summary"]["per_seed_family_mean_gain"] for p in parts},
            )
        ),
        flush=True,
    )
