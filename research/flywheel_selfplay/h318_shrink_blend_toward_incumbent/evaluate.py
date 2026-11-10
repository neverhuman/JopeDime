from runtime import *
from statistics_h318 import analyze, decide


def main():
    rec = json.loads((P / "source/score_receipt.json").read_text())
    assert rec["status"] == "PASS" and rec["panel_sha256"] == sha(D / "out/panel.npz")
    with np.load(D / "out/panel.npz", allow_pickle=False) as z:
        u = z["u"]
        invalid = z["invalid"]
        f = z["families"]
        picks = z["picks"]
    names = sorted(set(f))
    assert len(names) == 205 and u.shape == (9, 1219, 3)
    ds = {
        a: np.array(
            [
                [u[s, f == fam, j].mean() - u[s, f == fam, 0].mean() for fam in names]
                for s in range(9)
            ]
        )
        for a, j in [("BASE", 1), ("BLEND", 2)]
    }
    summaries, paired, secondary, ix = analyze(ds["BASE"], ds["BLEND"])
    dec = decide(
        paired, summaries["BLEND"]["aggregate"], summaries["BASE"]["aggregate"]
    )
    for arm, j in [("BASE", 1), ("BLEND", 2)]:
        for cell, r in enumerate(summaries[arm]["cells"]):
            part = PARTS[cell // 3]
            seed = SEEDS[cell % 3]
            choices = [
                json.loads(
                    (
                        D / part / f"fold_{k}" / f"root_{seed}" / "fit_lock_h318.json"
                    ).read_text()
                )["alpha"]
                for k in range(4)
            ]
            r.update(
                partition=part,
                root_seed=seed,
                chosen_alpha_by_fold=choices if arm == "BLEND" else [1] * 4,
                invalid=int(invalid[cell, :, j].sum()),
                baseline_invalid=int(invalid[cell, :, 0].sum()),
                switches=int((picks[cell, :, j - 1] != 0).sum()),
                picks_sha256=hashlib.sha256(
                    picks[cell, :, j - 1].tobytes()
                ).hexdigest(),
            )
        summaries[arm]["per_partition"] = {
            part: dict(
                median_gain=float(np.median(ds[arm][i * 3 : i * 3 + 3].mean(axis=1))),
                ci95=np.quantile(
                    np.median(
                        np.array(
                            [v[ix].mean(axis=1) for v in ds[arm][i * 3 : i * 3 + 3]]
                        ),
                        axis=0,
                    ),
                    [0.025, 0.975],
                ).tolist(),
            )
            for i, part in enumerate(PARTS)
        }
    choices = {
        f"{p}/{k}/{s}": json.loads(
            (D / p / f"fold_{k}" / f"root_{s}" / "fit_lock_h318.json").read_text()
        )["alpha"]
        for p in PARTS
        for k in range(4)
        for s in SEEDS
    }
    m = dict(
        wave_id="H318",
        status="COMPLETE",
        **dec,
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        fit_source_commit=json.loads((P / "implementation-published.json").read_text())[
            "head"
        ],
        regression="EXACT PASS",
        fresh_base_regression="EXACT PASS",
        regression_receipt_sha256=sha(P / "source/regression_receipt.json"),
        elo_label="reference = r882 slot0 (DL-Train outer)",
        r3175_train="NOT_MEASURED",
        families=205,
        contexts=1219,
        arms=summaries,
        paired_drop_top10=paired,
        secondary_drop_of_paired_differences=secondary,
        alpha_choices=choices,
        alpha_histogram={
            str(a): list(choices.values()).count(a) for a in [0, 0.25, 0.5, 0.75, 1]
        },
        score_blend_level="hard-pick one-hot candidate scores, deterministic argmax; alpha<=.5 incumbent, alpha>.5 ENS",
        outer_test_rows_read_before_lock=0,
        outer_test_packets_built_before_lock=0,
        outer_score_calls=36,
        dev91_rows_read=0,
        serving_authority=False,
        bootstrap=dict(
            draws=10000,
            rng=318180,
            shared_family_resampling=True,
            indices_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        ),
        caveats=[
            "Discrete selector has no continuous prediction output: this score blend reduces to two policies, not smooth program-prediction shrinkage.",
            "TRAIN adaptive reuse; nine cells share families/seeds and are not independent replications.",
            "H314 CLEAN denotes absence of metadata rule matches, not proven semantic independence.",
            "No dev91/r3175 comparison or live/serving change.",
        ],
    )
    raw = json.dumps(m, indent=2, allow_nan=False) + "\n"
    assert len(raw.encode()) < 50000
    (P / "source/metrics.json").write_text(raw)
    np.savez_compressed(
        D / "out/family_gains.npz", **ds, families=np.array(names), bootstrap_indices=ix
    )
    print(
        json.dumps(
            dict(
                **dec,
                paired=paired,
                aggregate={a: s["aggregate"] for a, s in summaries.items()},
                alpha=m["alpha_histogram"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
