"""Publish the fixed six-seed red-team decision, with complete family-level evidence."""

from pathlib import Path
import sys, json, hashlib
import numpy as np
from scipy.stats import t
import inference as inf

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h312-astra")
sys.path.insert(0, str(P / "vendor/h302_ensvote2_trainseed_nested_cv"))
import analysis_h302 as a


def main():
    rec = json.loads((P / "source/score_receipt.json").read_text())
    assert rec["status"] == "PASS"
    path = D / "out/all_seeds.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["output_sha256"]
    with np.load(path, allow_pickle=False) as z:
        u = z["u"]
        invalid = z["invalid"]
        picks = z["picks"]
        families = z["families"]
        roots = z["roots"]
    fams = sorted(set(families))
    assert len(fams) == 19 and u.shape == (6, 91, 2)
    d = np.array(
        [
            [
                u[s, families == f, 1].mean() - u[s, families == f, 0].mean()
                for f in fams
            ]
            for s in range(6)
        ]
    )
    reference = json.loads((P / "source/regression_receipt.json").read_text())
    assert d[:3].mean(axis=1).tolist() == reference["expected_full_precision"]
    rows, pooled = inf.analyze(d)
    for i, row in enumerate(rows):
        row.update(
            root_seed=int(roots[i]),
            elo=float(a.bt_two(row["losses"], row["wins"])),
            invalid=int(invalid[i, :, 1].sum()),
            baseline_invalid=int(invalid[i, :, 0].sum()),
            invalid_excess=int(invalid[i, :, 1].sum() - invalid[i, :, 0].sum()),
            switches=int((picks[i] != 0).sum()),
            harmed_share=row["harmed"] / 19,
            context_gain=float((u[i, :, 1] - u[i, :, 0]).mean()),
        )
    pooled.update(
        elo=float(np.median([x["elo"] for x in rows])),
        median_counts={
            k: float(np.median([x[k] for x in rows]))
            for k in [
                "wins",
                "ties",
                "losses",
                "harmed",
                "invalid",
                "baseline_invalid",
                "switches",
            ]
        },
        count_note="Medians of per-seed counts, not a seventh policy or independent samples.",
    )
    audit = json.loads((P / "source/overlap_audit.json").read_text())
    keep = np.array([f not in audit["overlap_ids"] for f in fams])
    no = d[:, keep].mean(axis=1) if keep.any() else None
    decision = inf.decision(
        rows, audit["overlap_count"], float(np.median(no)) if no is not None else None
    )
    new = d[3:]
    points = new.mean(axis=1)
    median = float(np.median(points))
    sgn = np.random.default_rng(312150).choice([-1.0, 1.0], size=(10000, 19))
    wild = sgn @ (new - points[:, None]).T / 19 + points
    j = inf.jk(np.median((new.sum(axis=1)[:, None] - new) / 18, axis=0), median, 18)
    fresh = dict(
        roots=[30004, 30005, 30006],
        median_gain=median,
        wild=inf.wild_summary(median, np.median(wild, axis=1)),
        jackknife=j,
        t18_ci95=j["ci95"],
        t18_lcb95=float(median - t.ppf(0.95, 18) * j["se"]),
        range=[float(points.min()), float(points.max())],
        sd=float(points.std(ddof=1)),
    )
    metrics = dict(
        wave_id="H312",
        **decision,
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        h310_commit="819a45aa5e30e662ab511cd9962b85b2b8c1e44a",
        elo_label="reference = r3175 slot0 (dev91)",
        utility="-fitness if gate passes else -2.0",
        families=19,
        contexts=91,
        roots=roots.tolist(),
        baseline_context_mean=float(u[0, :, 0].mean()),
        per_seed=rows,
        pooled=pooled,
        fresh_three=fresh,
        overlap=audit,
        nonoverlap=dict(
            families=int(keep.sum()),
            per_seed_gain=no.tolist() if no is not None else None,
            median_gain=float(np.median(no)) if no is not None else None,
            descriptive=True,
        ),
        family_ids_sha256=hashlib.sha256(
            json.dumps(fams, separators=(",", ":")).encode()
        ).hexdigest(),
        family_order="sorted IDs, full list in archived family_gains.npz",
        family_gains=d.tolist(),
        wild_rng=312150,
        wild_draws=10000,
        regression="EXACT PASS",
        dev91_rows_read_before_lock=0,
        serving_authority=False,
        caveats=[
            "19 shared development families; six roots are dependent, not six independent datasets.",
            "Exact sign flips assume family-wise sign symmetry; family dependence and adaptive research exposure are not removed.",
            "Pooled median t/jackknife and wild intervals are descriptive approximations, not guaranteed small-sample coverage.",
            "Even ROBUST requires fresh S3 population H299/H300 confirmation before a breakthrough claim.",
        ],
    )
    raw = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    assert len(raw.encode()) < 50000
    (P / "source/metrics.json").write_text(raw)
    np.savez_compressed(
        D / "out/family_gains.npz", gains=d, families=np.array(fams), roots=roots
    )
    print(
        json.dumps(
            dict(
                verdict=metrics["verdict"],
                criteria=metrics["criteria"],
                gains=[x["gain"] for x in rows],
                exact_p=[x["exact_signflip_p"] for x in rows],
                t_lcb=[x["t_lcb95"] for x in rows],
                lofo=[x["jackknife"]["min_gain"] for x in rows],
                fresh=fresh,
                pooled=pooled,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
