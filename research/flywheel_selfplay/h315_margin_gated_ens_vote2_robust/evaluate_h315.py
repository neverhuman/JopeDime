"""H315 evaluation. Regression first: BASE outer picks must equal the archived H303 (P0) / H308 (P1, P2) ENS_VOTE2_NESTED
picks for every (partition, seed), else REGRESSION_FAIL and no MARGIN evaluation. Then per arm and partition: H303's
judge (H302 statistics, whole-family bootstrap RNG 302150, concentration) vs r882 slot0, and the paired decision.

Decision (pre-registered; ties never win):
  MARGIN_ADVANCE iff (1) median over the 9 (partition, seed) cells of the paired drop-top-10 gain difference
  (MARGIN drop-top-10 gain - BASE drop-top-10 gain; each arm drops its own top-10 families) > 0 AND its paired
  whole-family bootstrap LCB95 > 0 (10,000 draws, RNG 315150, same family resample for both arms; statistic = median
  over cells); (2) MARGIN median (over cells) family-equal gain >= 0.9 x BASE's; (3) MARGIN median harmed share <= BASE's.
  Else MARGIN_NOT_ADVANCED."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

ELO_LABEL = "reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN"
PAIRED_SEED, DRAWS, TOPN = 315150, 10000, 10
ARCHIVED = {
    "P0": ("experiments/rd_breakthrough_20261030/H303/picks.npz", "3CUeOo9dmsTM1rZyuAeS9pF7H39Vv80N",
           "0f8435010390c5bff8d6c95a7650163a2ab4c5952e56264fbee96204784d5ea4", "ENS_VOTE2_NESTED"),
    "P1": ("experiments/rd_breakthrough_20261030/H308/picks.npz", "uXsQkA27JSci7XVbEFvbyJvfvzmO3nP0",
           "ac16ccdfd715d000a249dfea87d707159329f189a43475452af038a03517290f", "ENS_VOTE2_NESTED_P1"),
    "P2": ("experiments/rd_breakthrough_20261030/H308/picks.npz", "uXsQkA27JSci7XVbEFvbyJvfvzmO3nP0",
           "ac16ccdfd715d000a249dfea87d707159329f189a43475452af038a03517290f", "ENS_VOTE2_NESTED_P2"),
}  # fmt: skip
PUBLISHED_MEDIAN = {"P0": 0.0643163334285782, "P1": 0.04821480030677507, "P2": 0.04360142642936269}


def fetch(key, version, digest, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["aws", "s3api", "get-object", "--bucket", "veox-jopedime-learning-use2", "--key", key, "--version-id", version, str(dest)],
                       check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, AWS_PROFILE="veox-jepson-prod"))  # fmt: skip
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise RuntimeError("archived picks sha mismatch: " + key)
    return dest


def family_d(u, families):
    fams = sorted(set(families))
    f = np.asarray(families)
    return np.stack(
        [np.array([u[s, f == x, 1].mean() - u[s, f == x, 0].mean() for x in fams]) for s in range(u.shape[0])]
    )


def drop_top(d, n=TOPN):
    return float(np.sort(d)[::-1][n:].mean())


def paired_drop_top(Dm, Db, seed=PAIRED_SEED, draws=DRAWS, n=TOPN):
    """Dm, Db: (cells, families) family gains of MARGIN and BASE. Median over cells of the drop-top-n difference,
    with a paired whole-family bootstrap (same resample for both arms and all cells)."""
    obs = np.array([drop_top(m, n) - drop_top(b, n) for m, b in zip(Dm, Db)])
    F = Dm.shape[1]
    idx = np.random.default_rng(seed).integers(0, F, size=(draws, F))
    diffs = np.zeros((draws, len(Dm)))
    for c in range(len(Dm)):
        sm = -np.sort(-Dm[c][idx], axis=1)
        sb = -np.sort(-Db[c][idx], axis=1)
        diffs[:, c] = sm[:, n:].mean(axis=1) - sb[:, n:].mean(axis=1)
    med = np.median(diffs, axis=1)
    return dict(per_cell=obs.tolist(), median=float(np.median(obs)), ci95=np.quantile(med, [0.025, 0.975]).tolist(),
                lcb95=float(np.quantile(med, 0.05)))  # fmt: skip


def decide(paired, gain_m, gain_b, harm_m, harm_b):
    crit = dict(
        c1_paired_drop_top10_median_gt_0_and_lcb95_gt_0=bool(paired["median"] > 0 and paired["lcb95"] > 0),
        c2_margin_median_gain_ge_90pct_base=bool(np.median(gain_m) >= 0.9 * np.median(gain_b)),
        c3_margin_median_harmed_le_base=bool(np.median(harm_m) <= np.median(harm_b)),
    )
    return crit, ("MARGIN_ADVANCE" if all(crit.values()) else "MARGIN_NOT_ADVANCED")


def main(r303, DATA, scored):
    import evaluate_h303 as ev

    G = r303.G
    a302 = G["a302"]
    fams = [r["family_id"] for r in G["train_rows"]]
    reg = {}
    for part, (key, ver, dig, name) in ARCHIVED.items():
        with np.load(fetch(key, ver, dig, DATA / "archived" / (part[:2] + "_" + Path(key).parent.name + ".npz"))) as z:
            want = np.asarray(z[name], dtype=np.int32)
        got = scored[part][0]["BASE"]
        reg[part] = dict(per_seed_equal=[bool(np.array_equal(got[s], want[s])) for s in range(3)],
                         base_sha256=[hashlib.sha256(got[s].tobytes()).hexdigest() for s in range(3)])  # fmt: skip
    ok = all(all(v["per_seed_equal"]) for v in reg.values())
    out = DATA / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not ok:
        (out / "REGRESSION_FAIL.json").write_text(json.dumps(reg, indent=1))
        raise SystemExit("REGRESSION_FAIL: BASE picks differ from archived H303/H308 picks")
    res, D = {}, {"BASE": [], "MARGIN": []}
    cells = dict(gain={"BASE": [], "MARGIN": []}, harmed={"BASE": [], "MARGIN": []})
    for part in ("P0", "P1", "P2"):
        res[part] = {}
        for arm in ("BASE", "MARGIN"):
            picks = scored[part][0][arm]
            u, inv = ev.panel(G, picks)
            j = ev.judge(a302, u, inv, fams, picks)
            j["summary"] = {k: v for k, v in j["summary"].items() if k != "harmed_families_by_seed"}
            res[part][arm] = j
            D[arm].extend(family_d(u, fams))
            cells["gain"][arm].extend(j["summary"]["per_seed_family_mean_gain"])
            cells["harmed"][arm].extend(j["summary"]["harmed_share_by_seed"])
        reg[part]["base_median_equals_published"] = (
            res[part]["BASE"]["summary"]["median_gain_across_seeds"] == PUBLISHED_MEDIAN[part]
        )
    paired = paired_drop_top(np.array(D["MARGIN"]), np.array(D["BASE"]))
    crit, verdict = decide(
        paired, cells["gain"]["MARGIN"], cells["gain"]["BASE"], cells["harmed"]["MARGIN"], cells["harmed"]["BASE"]
    )
    full = dict(wave="H315", verdict=verdict, criteria=crit, elo_label=ELO_LABEL, regression=reg, paired_drop_top10=paired,
                cell_medians=dict(gain={a: float(np.median(v)) for a, v in cells["gain"].items()},
                                  harmed={a: float(np.median(v)) for a, v in cells["harmed"].items()}),
                cells={k: {a: v for a, v in d.items()} for k, d in cells.items()},
                partitions=res, fit_lock_sha256={p: scored[p][1] for p in scored})  # fmt: skip
    (out / "metrics_full.json").write_text(json.dumps(full, indent=1, allow_nan=False) + "\n")
    np.savez_compressed(
        out / "picks.npz", **{f"{arm}_{p}": scored[p][0][arm] for p in scored for arm in ("BASE", "MARGIN")}
    )
    print(json.dumps(dict(verdict=verdict, criteria=crit, paired=paired, cell_medians=full["cell_medians"],
                          regression={p: v["per_seed_equal"] for p, v in reg.items()})), flush=True)  # fmt: skip
