"""Paired concentration-resistant estimands, with frozen H303 baseline picks."""

import hashlib
import json
import datetime
import numpy as np
from robust_objective import decide

ELO_LABEL = "reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured"
EXPECTED = {
    30001: "0b187de957f692238ee0c21702822baae6f7845f344ecb5c3c0db483a711a7ac",
    30002: "35530bb5db8b9a93e093d759510a575569d3dfb22c9d46da9cd09fc14375c2f8",
    30003: "b8fb9c1476a0e6ddb18c05d0fba63b23c1bf93d99a1492d82c8f2d6da42ffda3",
}


def interval(x):
    return dict(
        ci95=np.quantile(x, [0.025, 0.975]).tolist(), lcb95=float(np.quantile(x, 0.05))
    )


def trim5(x):
    return np.sort(x, axis=-1)[..., :-5].mean(axis=-1)


def paired_statistics(robust, baseline, draws=10000):
    assert (
        robust.shape == baseline.shape and robust.shape[0] == 3 and robust.shape[1] > 5
    )
    ix = np.random.default_rng(302150).integers(
        0, robust.shape[1], (draws, robust.shape[1])
    )
    funcs = {
        "mean": lambda x: x.mean(axis=-1),
        "drop_top5": trim5,
        "family_median": lambda x: np.median(x, axis=-1),
    }
    output = {}
    for name, fn in funcs.items():
        point = fn(robust) - fn(baseline)
        boot = np.stack([fn(r[ix]) - fn(h[ix]) for r, h in zip(robust, baseline)])
        output[name] = dict(
            per_seed=point.tolist(),
            per_seed_intervals=[interval(b) for b in boot],
            median=float(np.median(point)),
            pooled=interval(np.median(boot, axis=0)),
        )
    output["median_familywise_paired_delta_descriptive"] = np.median(
        robust - baseline, axis=1
    ).tolist()
    return output


def family_delta(panel, families):
    f = np.asarray(families)
    return np.stack(
        [
            np.array(
                [
                    u[f == name, 1].mean() - u[f == name, 0].mean()
                    for name in sorted(set(families))
                ]
            )
            for u in panel
        ]
    )


def pooled_additions(D):
    ix = np.random.default_rng(302150).integers(0, D.shape[1], (10000, D.shape[1]))
    med_boot = np.stack([np.median(d[ix], axis=1) for d in D])
    fixed = np.stack([np.delete(d, np.argsort(-d, kind="stable")[:5]) for d in D])
    # Remaining sets differ between seeds; use shared original-family weights for
    # pooled drop-top5, preserving overlapping family identities after deletion.
    counts = np.zeros((10000, D.shape[1]))
    np.add.at(counts, (np.arange(10000)[:, None], ix), 1)
    bdrop = []
    for d in D:
        keep = np.ones(len(d), bool)
        keep[np.argsort(-d, kind="stable")[:5]] = False
        w = counts[:, keep]
        assert np.all(w.sum(1) > 0)
        bdrop.append((w @ d[keep]) / w.sum(1))
    return dict(
        family_median_per_seed_lcb95=np.quantile(med_boot, 0.05, axis=1).tolist(),
        family_median_pooled_lcb95=float(
            np.quantile(np.median(med_boot, axis=0), 0.05)
        ),
        drop_top5_median=float(np.median(fixed.mean(1))),
        drop_top5_pooled=interval(np.median(bdrop, axis=0)),
        drop_top1_median=float(np.median([np.sort(d)[:-1].mean() for d in D])),
        drop_top10_median=float(np.median([np.sort(d)[:-10].mean() for d in D])),
    )


def main(r, locks, picks, data, source):
    import evaluate_h303 as ev
    from run_h309 import save, sha

    g = r.G
    families = [row["family_id"] for row in g["train_rows"]]
    with np.load(data / "h303_reference/out/picks.npz") as z:
        base = np.asarray(z["ENS_VOTE2_NESTED"], dtype=np.int32)
    hashes = {
        seed: hashlib.sha256(base[i].tobytes()).hexdigest()
        for i, seed in enumerate((30001, 30002, 30003))
    }
    assert hashes == EXPECTED
    arms = {}
    delta = {}
    for name, pp in [("ROBUSTSEL", picks), ("H303_NESTED", base)]:
        panel, invalid = ev.panel(g, pp)
        arms[name] = ev.judge(g["a302"], panel, invalid, families, pp)
        delta[name] = family_delta(panel, families)
        arms[name]["pooled_additions"] = pooled_additions(delta[name])
        arms[name]["elo_reference"] = ELO_LABEL
    archived = json.loads((data / "h303_reference/out/metrics_full.json").read_text())
    expected = archived["arms"]["ENS_VOTE2_NESTED"]["summary"][
        "per_seed_family_mean_gain"
    ]
    observed = arms["H303_NESTED"]["summary"]["per_seed_family_mean_gain"]
    assert np.allclose(expected, observed, rtol=0, atol=1e-12)
    paired = paired_statistics(delta["ROBUSTSEL"], delta["H303_NESTED"])
    choices = []
    agreement = 0
    for lock in locks:
        k, s = lock["outer_fold"], lock["root_seed"]
        ref = json.loads(
            (
                data / f"h303_reference/scratch/fold_{k}/root_{s}/fit_lock.json"
            ).read_text()
        )
        same = all(lock["chosen"][n] == ref["chosen"][n] for n in ("tau", "k", "a"))
        agreement += int(same)
        choices.append(
            dict(
                outer_fold=k,
                root_seed=s,
                robust=lock["chosen"],
                h303=ref["chosen"],
                same_config=same,
                mean_table_matches_h303=lock["mean_table_matches_h303"],
                reused_regression_profiles=lock["reused_fresh_regression_profiles"],
            )
        )
    decision = decide(paired, arms["ROBUSTSEL"], arms["H303_NESTED"], agreement)
    np.savez_compressed(
        data / "family_gains.npz",
        families=np.array(sorted(set(families))),
        robust=delta["ROBUSTSEL"],
        h303=delta["H303_NESTED"],
    )
    full = dict(
        wave_id="H309",
        status="COMPLETE",
        verdict=decision["verdict"],
        decision=decision,
        paired=paired,
        arms=arms,
        chosen_configs=choices,
        pick_agreement_by_seed=np.mean(picks == base, axis=1).tolist(),
        baseline_pick_sha256=hashes,
        robust_pick_sha256={
            s: hashlib.sha256(picks[i].tobytes()).hexdigest()
            for i, s in enumerate((30001, 30002, 30003))
        },
        baseline_numerical_replay=dict(
            expected=expected, observed=observed, status="PASS"
        ),
        regression=json.loads((data / "regression_receipt.json").read_text()),
        pre_score_inventory_sha256=sha(data / "pre_score_lock_inventory.json"),
        elo_reference=ELO_LABEL,
        root_seeds=[30001, 30002, 30003],
        outer_partition="P0_H302-outer-v1",
        families=205,
        contexts=1219,
        bootstrap_seed=302150,
        draws=10000,
        neural_updates=0,
        neural_forwards=0,
        gpu_used=False,
        dope_untouched=True,
        loader_counters=tl_counters(),
        finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    save(data / "metrics_full.json", full)
    compact = json.loads(json.dumps(full))
    for arm in compact["arms"].values():
        arm["summary"].pop("harmed_families_by_seed", None)
    compact["prereg_commit"] = json.loads(
        (source.parent / "prereg-published.json").read_text()
    )["head"]
    compact["prereg_push_time"] = json.loads(
        (source.parent / "prereg-published.json").read_text()
    )["time"]
    compact["limitations"] = [
        "Historical H302/H303 TRAIN design exposure remains",
        "Root seeds30002/30003 nearly duplicate",
        "Outer-test TRAIN packets built before locks; fit accesses guarded",
        "Paired trimmed bootstrap re-trims each arm; baseline drop-top5 gate uses fixed exclusions",
        "No serving authorization; r3175 comparison unmeasured",
    ]
    save(source / "metrics.json", compact)
    assert (source / "metrics.json").stat().st_size < 50000
    print(
        json.dumps(
            dict(
                decision=decision,
                paired=paired,
                mean_gains=arms["ROBUSTSEL"]["summary"]["per_seed_family_mean_gain"],
                choices=choices,
            )
        ),
        flush=True,
    )


def tl_counters():
    import trainonly_loader

    return dict(trainonly_loader.COUNTERS)
