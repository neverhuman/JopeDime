from runtime import *
from statistics_h324 import analyze, decide, config_status


def concentration(x):
    total = float(np.sum(x))
    top = float(np.sort(x)[-10:].sum())
    positive = np.maximum(x, 0)
    gross = float(positive.sum())
    return dict(
        top10_signed_gain=top,
        total_signed_gain=total,
        top10_share=top / total if total > 0 else None,
        undefined_reason=None if total > 0 else "total signed gain <= 0",
        gross_positive_gain=gross,
        top10_gross_positive_share=float(np.sort(positive)[-10:].sum() / gross)
        if gross > 0
        else None,
    )


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    rec = json.loads((P / "source/score_receipt.json").read_text())
    assert rec["status"] == "PASS" and rec["panel_sha256"] == sha(D / "out/panel.npz")
    with np.load(D / "out/panel.npz", allow_pickle=False) as z:
        u = z["u"]
        invalid = z["invalid"]
        families = z["families"]
        picks = z["picks"]
        cids = z["cids"]
    names = sorted(set(families))
    assert len(names) == 205
    ds = {
        a: np.array(
            [
                [
                    u[c, families == f, j].mean() - u[c, families == f, 0].mean()
                    for f in names
                ]
                for c in range(9)
            ]
        )
        for a, j in [("BASE", 1), ("CAP", 2)]
    }
    summaries, paired, secondary, ix = analyze(ds["BASE"], ds["CAP"])
    decision = decide(
        paired, summaries["CAP"]["aggregate"], summaries["BASE"]["aggregate"]
    )
    choices = []
    for part in PARTS:
        for k in range(4):
            for seed in SEEDS:
                base = json.loads(
                    (
                        D
                        / "references"
                        / part
                        / f"fold_{k}"
                        / f"root_{seed}"
                        / "fit_lock.json"
                    ).read_text()
                )["chosen"]
                weighted = json.loads(
                    (
                        D / part / f"fold_{k}" / f"root_{seed}" / "fit_lock_h324.json"
                    ).read_text()
                )["cap_chosen"]
                eq = {x: base[x] == weighted[x] for x in ["tau", "k", "a"]}
                choices.append(
                    dict(
                        partition=part,
                        fold=k,
                        seed=seed,
                        BASE=base,
                        CAP=weighted,
                        parameter_agreement=eq,
                        exact_agreement=all(eq.values()),
                    )
                )
    for arm, j in [("BASE", 1), ("CAP", 2)]:
        for cell, row in enumerate(summaries[arm]["cells"]):
            part = PARTS[cell // 3]
            seed = SEEDS[cell % 3]
            x = ds[arm][cell]
            row.update(
                partition=part,
                root_seed=seed,
                invalid=int(invalid[cell, :, j].sum()),
                baseline_invalid=int(invalid[cell, :, 0].sum()),
                switches=int((picks[cell, :, j - 1] != 0).sum()),
                picks_sha256=hashlib.sha256(
                    picks[cell, :, j - 1].tobytes()
                ).hexdigest(),
                lofo_min=float(((x.sum() - x) / (len(x) - 1)).min()),
                concentration=concentration(x),
            )
        comparisons = []
        for pi, part in enumerate(PARTS):
            for k in range(4):
                ids = np.flatnonzero([fold(part, str(f)) == k for f in families])
                for sa, sb in [(0, 1), (0, 2), (1, 2)]:
                    sets = [
                        {str(cids[n]) for n in ids if picks[pi * 3 + s, n, j - 1] != 0}
                        for s in [sa, sb]
                    ]
                    acts = [
                        {
                            (str(cids[n]), int(picks[pi * 3 + s, n, j - 1]))
                            for n in ids
                            if picks[pi * 3 + s, n, j - 1] != 0
                        }
                        for s in [sa, sb]
                    ]
                    comparisons.append(
                        dict(
                            partition=part,
                            fold=k,
                            seeds=[SEEDS[sa], SEEDS[sb]],
                            switch_context_jaccard=jaccard(*sets),
                            action_jaccard=jaccard(*acts),
                        )
                    )
        summaries[arm]["seed_jaccard"] = dict(
            comparisons=comparisons,
            median=float(np.median([x["switch_context_jaccard"] for x in comparisons])),
            median_action=float(np.median([x["action_jaccard"] for x in comparisons])),
            empty_empty=1,
        )
        summaries[arm]["aggregate"].update(
            median_lofo_min=float(
                np.median([x["lofo_min"] for x in summaries[arm]["cells"]])
            ),
            median_top10_gain_share=float(
                np.median(
                    [x["concentration"]["top10_share"] for x in summaries[arm]["cells"]]
                )
            )
            if all(
                x["concentration"]["top10_share"] is not None
                for x in summaries[arm]["cells"]
            )
            else None,
        )
        summaries[arm]["per_partition"] = {
            part: dict(
                median_gain=float(np.median(ds[arm][pi * 3 : pi * 3 + 3].mean(axis=1))),
                ci95=np.quantile(
                    np.median(
                        np.array(
                            [x[ix].mean(axis=1) for x in ds[arm][pi * 3 : pi * 3 + 3]]
                        ),
                        axis=0,
                    ),
                    [0.025, 0.975],
                ).tolist(),
            )
            for pi, part in enumerate(PARTS)
        }
    ws=json.loads((P/"source/weight_summary.json").read_text())
    m = dict(
        dope=dict(composite_sha256_prefix="c51a56d0",loaded=False,scope="CPU selection layer only"),
        fitting_weight_summary={k:v for k,v in ws.items() if k!="cells"},
        weight_summary_sha256=sha(P/"source/weight_summary.json"),
        config_status=config_status(sum(x["exact_agreement"] for x in choices)),
        wave_id="H324",
        status="COMPLETE",
        **decision,
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        fit_source_commit=json.loads((P / "implementation-published.json").read_text())[
            "head"
        ],
        regression="EXACT PASS",
        elo_label="reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN",
        r3175_train="NOT_MEASURED",
        families=205,
        contexts=1219,
        arms=summaries,
        paired_drop_top10=paired,
        secondary_drop_of_paired_differences=secondary,
        config_agreement=dict(
            cells=choices,
            exact=sum(x["exact_agreement"] for x in choices) / 36,
            parameters={
                p: sum(x["parameter_agreement"][p] for x in choices) / 36
                for p in ["tau", "k", "a"]
            },
        ),
        outer_test_rows_read_before_lock=0,
        outer_test_packets_built_before_lock=0,
        outer_score_calls=36,
        dev91_rows_read=0,
        serving_authority=False,
        bootstrap=dict(
            draws=10000,
            rng=324191,
            shared_family_resampling=True,
            indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        ),
        caveats=[
            "Fixed fitted policies; bootstrap does not rerun model selection.",
            "Reused TRAIN families and dependent partition/seed cells.",
            "Family bootstrap multiplicities count as occurrences in each fitting-set normalization.",
            "Original local H50/H58/H276 input S3 versions not independently established.",
        ],
    )
    raw = json.dumps(m, separators=(",", ":"), allow_nan=False) + "\n"
    assert len(raw.encode()) < 50000
    (P / "source/metrics.json").write_text(raw)
    np.savez_compressed(
        D / "out/family_gains.npz", **ds, families=np.array(names), bootstrap_indices=ix
    )
    print(
        json.dumps(
            dict(
                **decision,
                paired=paired,
                aggregate={a: s["aggregate"] for a, s in summaries.items()},
                config_agreement=m["config_agreement"]["exact"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
