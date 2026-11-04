"""Join already-fsynced metadata classes to archived H312 utilities; no new scoring."""

from pathlib import Path
import json, hashlib, datetime
import numpy as np
from scipy.stats import spearmanr, fisher_exact
import statistics_h314 as stats

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")
M = D / "metadata"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fisher(top, classes, include):
    ids = [i for i in range(19) if include[i]]
    a = sum(top[i] and classes[i] == "NEIGHBORED" for i in ids)
    b = sum(top[i] and classes[i] != "NEIGHBORED" for i in ids)
    c = sum(not top[i] and classes[i] == "NEIGHBORED" for i in ids)
    d = sum(not top[i] and classes[i] != "NEIGHBORED" for i in ids)
    table = [[a, b], [c, d]]
    odds, p = fisher_exact(table, alternative="greater")
    return dict(
        table=table,
        columns=["confirmed_neighbor", "other_included"],
        rows=["top10_of_all19", "remaining"],
        odds_ratio=float(odds) if np.isfinite(odds) else None,
        odds_ratio_nonfinite=not bool(np.isfinite(odds)),
        p_greater=float(p),
        families=len(ids),
    )


def main():
    reg = json.loads((P / "source/regression_receipt.json").read_text())
    assert reg["status"] == "PASS"
    receipt = json.loads((P / "source/classification_receipt.json").read_text())
    assert receipt["status"] == "PASS"
    assert sha(M / "classification.json") == receipt["classification_sha256"]
    classes = json.loads((M / "classification.json").read_text())
    with np.load(D / "references/all_seeds.npz", allow_pickle=False) as z:
        u = z["u"]
        inv = z["invalid"]
        f = z["families"]
        cids = z["cids"]
    fams = sorted(set(f))
    assert set(fams) == set(classes) and len(fams) == 19
    d = np.array(
        [
            [u[s, f == name, 1].mean() - u[s, f == name, 0].mean() for name in fams]
            for s in range(6)
        ]
    )
    assert d.mean(axis=1).tolist() == reg["expected_gains"]
    assert float(u[0, :, 0].mean()) == reg["baseline_mean"]
    groups = {}
    statuses = [classes[name]["status"] for name in fams]
    for group in ["NEIGHBORED", "CLEAN", "UNKNOWN", "ALL"]:
        keep = np.array([group == "ALL" or value == group for value in statuses])
        summary = stats.summarize(d[:, keep])
        ids = [name for name, k in zip(fams, keep) if k]
        mask = np.isin(f, ids)
        for s, row in enumerate(summary["per_seed"]):
            row.update(
                invalid=int(inv[s, mask, 1].sum()),
                baseline_invalid=int(inv[s, mask, 0].sum()),
            )
        if summary["pooled"] is not None:
            summary["pooled"]["median_counts"].update(
                invalid=float(np.median([r["invalid"] for r in summary["per_seed"]])),
                baseline_invalid=float(
                    np.median([r["baseline_invalid"] for r in summary["per_seed"]])
                ),
            )
        summary["family_ids_sha256"] = hashlib.sha256(
            json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        groups[group] = summary
    outcome = stats.decide(groups)
    family_gain = np.median(d, axis=0)
    sim = [classes[name]["max_train_similarity"] for name in fams]
    valid = np.array([x is not None for x in sim])
    x = np.array([s for s in sim if s is not None])
    y = family_gain[valid]
    if len(x) >= 3 and len(set(x)) > 1 and len(set(y)) > 1:
        rho, p = spearmanr(x, y)
        cor = dict(
            rho=float(rho),
            p_two_sided=float(p),
            families=len(x),
            method="Spearman average ranks; asymptotic p; observed similarity lower bounds where incomplete",
        )
    else:
        cor = dict(
            rho=None,
            p_two_sided=None,
            families=len(x),
            unmeasured_reason="fewer than3 available or constant similarity/gain ranks",
        )
    ranking = sorted(range(19), key=lambda i: (-family_gain[i], fams[i]))
    top = np.zeros(19, dtype=bool)
    top[ranking[:10]] = True
    tables = dict(
        confirmed_neighbor_vs_other=fisher(top, statuses, [True] * 19),
        confirmed_neighbor_vs_clean=fisher(
            top, statuses, [v != "UNKNOWN" for v in statuses]
        ),
        unknown_count=statuses.count("UNKNOWN"),
        note="UNKNOWN belongs only to not-confirmed in the all19 detection table; it is never called CLEAN.",
    )
    details = []
    for i, name in enumerate(fams):
        meta = classes[name]
        details.append(
            dict(
                family_id=name,
                status=meta["status"],
                triggered_clauses=meta["triggered_clauses"],
                witness_count=len(meta["witnesses"]),
                max_train_similarity=sim[i],
                similarity_is_lower_bound=meta["similarity_is_lower_bound"],
                pooled_median_gain=float(family_gain[i]),
                per_seed_gains=d[:, i].tolist(),
                top10=bool(top[i]),
            )
        )
    metrics = dict(
        wave_id="H314",
        **outcome,
        initial_prereg_commit=json.loads(
            (P / "initial-prereg-published.json").read_text()
        )["head"],
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        transport_clarification_commit=json.loads(
            (P / "transport-prereg-published.json").read_text()
        )["head"],
        source_commit=json.loads((P / "identityfix-published.json").read_text())[
            "head"
        ],
        regression="EXACT PASS",
        baseline_context_mean=reg["baseline_mean"],
        elo_label="reference = r3175 slot0 (dev91)",
        groups=groups,
        classification_counts={
            g: receipt["groups"][g]["count"] for g in ["NEIGHBORED", "CLEAN", "UNKNOWN"]
        },
        classification_sha256=receipt["classification_sha256"],
        family_results=details,
        association_spearman=cor,
        association_fisher=tables,
        metadata_coverage={
            k: receipt[k]
            for k in [
                "dataset_records",
                "source_registry_entries_read",
                "split_registry_entries_read",
                "metadata_objects_read",
                "missing_by_field",
                "source_registry_datasets_matched",
                "split_registry_datasets_matched",
            ]
        },
        new_fits=0,
        new_scores=0,
        raw_feature_or_label_rows_read=0,
        serving_authority=False,
        caveats=[
            "Metadata neighbor rule detects recorded lineage or structural similarity, not causal contamination.",
            "CLEAN means no rule match in recorded metadata, not guaranteed semantic or historical independence.",
            "Only19 reused development families and dependent seed policies.",
            "Canonical ingested schema/count used for clause(c); raw counts/schema hashes are recorded separately.",
            "No raw source dataset bodies read; full-source registry coverage can be incomplete.",
            "Wild intervals and symmetry-based exact flips do not guarantee coverage with small heterogeneous clusters.",
        ],
    )
    old_classes = json.loads(
        (M / "initial_filename_identity/classification.json").read_text()
    )
    literal_groups = {}
    for name in ["NEIGHBORED", "CLEAN", "UNKNOWN"]:
        keep = np.array([old_classes[f]["status"] == name for f in fams])
        literal_groups[name] = stats.summarize(d[:, keep])
    metrics["literal_filename_sensitivity"] = dict(
        known_false_identity_match=True,
        classification_sha256="5a003c1b86e8145f1d2ad0146b6fbb40306ca887aa725afae68a9953068e45bd",
        groups=literal_groups,
        decision=stats.decide(literal_groups),
        note="Preserved pre-gain correction diagnostic: unrelated Socrata datasets shared only a generic download filename.",
    )
    raw = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    assert len(raw.encode()) < 50000
    (P / "source/metrics.json").write_text(raw)
    (D / "out").mkdir(exist_ok=True)
    np.savez_compressed(
        D / "out/analysis_panel.npz",
        family_gains=d,
        families=np.array(fams),
        classification=np.array(statuses),
        median_family_gain=family_gain,
        top10=top,
    )
    print(
        json.dumps(
            dict(
                verdict=metrics["verdict"],
                counts=metrics["classification_counts"],
                clean=groups["CLEAN"]["pooled"],
                neighbor=groups["NEIGHBORED"]["pooled"],
                spearman=cor,
                fisher=tables,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
