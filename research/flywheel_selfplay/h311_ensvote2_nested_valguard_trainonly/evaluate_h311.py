"""Paired shared-family bootstrap across nine dependent partition/seed cells."""

from pathlib import Path
import json, hashlib, datetime
import numpy as np
import guard_policy as policy

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h311-astra")
LABEL = "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured"


def interval(point, boot):
    return dict(
        point=float(point),
        ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        lcb95=float(np.quantile(boot, 0.05)),
    )


def family_values(values, fams):
    return np.array([values[fams == f].mean() for f in sorted(set(fams))])


def trimmed(d):
    return np.sort(d, axis=-1)[..., :-5].mean(axis=-1)


def statistics(d, idx):
    sample = d[idx]
    w = int((d > 1e-12).sum())
    l = int((d < -1e-12).sum())
    boots = dict(
        mean=sample.mean(axis=1),
        family_median=np.median(sample, axis=1),
        drop_top5=trimmed(sample),
        elo=1500
        + 400
        * np.log10(
            ((sample > 1e-12).sum(axis=1) + 0.01)
            / ((sample < -1e-12).sum(axis=1) + 0.01)
        ),
    )
    points = dict(
        mean=float(d.mean()),
        family_median=float(np.median(d)),
        drop_top5=float(trimmed(d)),
        elo=float(1500 + 400 * np.log10((w + 0.01) / (l + 0.01))),
    )
    result = {k: interval(points[k], boots[k]) for k in points}
    result.update(
        wtl=[w, len(d) - w - l, l],
        harmed=int((d < -0.01 - 1e-12).sum()),
        top10_net_gain_share=float(np.sort(d)[-10:].sum() / d.sum())
        if abs(d.sum()) > 1e-15
        else None,
    )
    return result, boots


def pool(points, boots):
    return {
        k: interval(
            np.median([v[k]["point"] for v in points]),
            np.median([b[k] for b in boots], axis=0),
        )
        for k in boots[0]
    }


def main():
    assert json.loads((D / "score_receipt.json").read_text())["status"] == "PASS"
    with np.load(D / "scored.npz") as z:
        picks = z["picks"]
        u = z["u"]
        gate = z["gate"]
        fams = z["fams"]
    ids = np.arange(1219)
    families = sorted(set(fams))
    assert len(families) == 205
    rng = np.random.default_rng(302150)
    idx = rng.integers(0, 205, size=(10000, 205))
    base = u[:, 0]
    invalid0 = int((~gate[:, 0]).sum())
    results = {}
    curves = {}
    family_d = {}
    cells = [f"{p}:{s}" for p in ["P0", "P1", "P2"] for s in [30001, 30002, 30003]]
    for arm, ia in [("A", 0), ("B", 1), ("N", 2)]:
        rows = []
        boots = []
        ds = []
        for ip in range(3):
            for ir in range(3):
                pick = picks[ip, ir, ia]
                delta = family_values(u[ids, pick] - base, fams)
                stat, b = statistics(delta, idx)
                stat.update(
                    cell=cells[len(rows)],
                    invalid=int((~gate[ids, pick]).sum()),
                    invalid_slot0=invalid0,
                    invalid_excess=int((~gate[ids, pick]).sum()) - invalid0,
                    switches=int((pick != 0).sum()),
                    picks_sha256=hashlib.sha256(
                        np.ascontiguousarray(pick).tobytes()
                    ).hexdigest(),
                )
                rows.append(stat)
                boots.append(b)
                ds.append(delta)
        results[arm] = dict(
            cells=rows,
            pooled=pool(rows, boots),
            median_invalid=float(np.median([x["invalid"] for x in rows])),
            median_harmed=float(np.median([x["harmed"] for x in rows])),
            partitions={
                p: pool(rows[i * 3 : i * 3 + 3], boots[i * 3 : i * 3 + 3])
                for i, p in enumerate(["P0", "P1", "P2"])
            },
        )
        curves[arm] = boots
        family_d[arm] = np.asarray(ds)
    paired = {}
    for name, left, right in [("B_minus_A", "B", "A"), ("B_minus_N", "B", "N")]:
        rows = []
        boots = []
        for i in range(9):
            d = family_d[left][i] - family_d[right][i]
            bmean = d[idx].mean(axis=1)
            bdrop = curves[left][i]["drop_top5"] - curves[right][i]["drop_top5"]
            boots.append(dict(mean=bmean, drop_top5=bdrop))
            rows.append(
                dict(
                    cell=cells[i],
                    mean=interval(d.mean(), bmean),
                    drop_top5=interval(
                        trimmed(family_d[left][i]) - trimmed(family_d[right][i]), bdrop
                    ),
                    invalid_delta=results[left]["cells"][i]["invalid"]
                    - results[right]["cells"][i]["invalid"],
                    harmed_delta=results[left]["cells"][i]["harmed"]
                    - results[right]["cells"][i]["harmed"],
                    wtl=[
                        int((d > 1e-12).sum()),
                        int((abs(d) <= 1e-12).sum()),
                        int((d < -1e-12).sum()),
                    ],
                )
            )
        paired[name] = dict(
            cells=rows,
            pooled=pool(rows, boots),
            median_invalid_delta=float(np.median([x["invalid_delta"] for x in rows])),
            partitions={
                p: pool(rows[i * 3 : i * 3 + 3], boots[i * 3 : i * 3 + 3])
                for i, p in enumerate(["P0", "P1", "P2"])
            },
        )
    a, b, n = [results[x]["cells"] for x in ["A", "B", "N"]]
    decision = policy.decision(
        [x["invalid"] for x in a],
        [x["invalid"] for x in b],
        [x["invalid"] for x in n],
        [x["harmed"] for x in a],
        [x["harmed"] for x in b],
        paired["B_minus_A"]["pooled"]["mean"]["point"],
        paired["B_minus_A"]["pooled"]["mean"]["lcb95"],
    )
    prereg = json.loads((P / "prereg-published.json").read_text())
    pub = json.loads((P / "implementation-published.json").read_text())
    choices = []
    for path in sorted((D / "tasks").glob("*/fold_*/root_*/fit_lock.json")):
        lock = json.loads(path.read_text())
        choices.append(
            dict(
                partition=lock["partition"],
                outer_fold=lock["outer_fold"],
                root_seed=lock["root_seed"],
                tau_B=lock["arms"]["B"]["tau"],
                tau_N=lock["arms"]["N"]["tau"],
            )
        )
    result = dict(
        wave="H311",
        verdict=decision["verdict"],
        decision=decision,
        arms=results,
        paired=paired,
        thresholds=choices,
        elo_reference=LABEL,
        contexts=1219,
        families=205,
        cells=cells,
        boot_draws=10000,
        boot_rng=302150,
        bootstrap="shared family draws across all nine cells; pooled median; paired drop5 re-trims each arm",
        prereg_commit=prereg["head"],
        source_commit=pub["head"],
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        limitations=[
            "Historical H287/H295/H303 TRAIN design exposure remains.",
            "Guard thresholds use frozen outer-refit selector predictions in-sample on outer_train; guard probabilities are OOF.",
            "Nine cells share the same families and are dependent; no new-population estimate.",
            "P0 seeds30002/30003 near-duplicate.",
            "Regression process follows historical all-TRAIN packet construction; guard fit workers physically read only their outer_train labels.",
            "No dev91, r3175 Elo, fresh-population or serving qualification.",
        ],
    )
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
    assert len(raw.encode()) < 50000
    (P / "source/metrics.json").write_text(raw)
    (D / "metrics_full.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez(
        D / "family_contrasts.npz",
        families=np.array(families),
        A=family_d["A"],
        B=family_d["B"],
        N=family_d["N"],
        bootstrap_indices=idx,
    )
    print(
        json.dumps(
            dict(
                verdict=decision,
                paired=paired["B_minus_A"]["pooled"],
                invalid={
                    x: [v["invalid"] for v in results[x]["cells"]] for x in results
                },
                metrics_bytes=len(raw),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
