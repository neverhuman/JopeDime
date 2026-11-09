"""Replay frozen H352 picks against TRAIN utilities only; no model predictions."""

from pathlib import Path
import hashlib, json
import numpy as np
import run_h372 as r
from evaluate_h372 import family_d, PARTS, SEEDS

S = Path(__file__).resolve().parent
D = S.parent


def main():
    assert (
        json.loads((D / "nl15-fetch-receipt.json").read_text())["status"]
        == "EXACT_VERSION_GET_AND_SHA_PASS"
    )
    model = r.initialize(require_prereg=False)
    r.full_packets()
    G = model.G
    import evaluate_h303 as ev

    families = [x["family_id"] for x in G["train_rows"]]
    panels = []
    locks = {}
    gains = []
    with np.load(D / "nl15_reference/raw_family_gains.npz") as z:
        expected = z["NL15"].copy()
    with np.load(D / "nl15_reference/picks.npz") as picks:
        for part in PARTS:
            pp = np.load(D / "nl15_reference" / f"NL15_{part}.npy")
            assert pp.shape == (3, 1219) and np.array_equal(pp, picks["NL15_" + part])
            lock_hash = json.loads(
                (D / "nl15_reference" / f"NL15_{part}_locks.json").read_text()
            )
            for k in range(4):
                ix = [
                    i
                    for i, f in enumerate(families)
                    if r.outer_fold(part, f, G["a302"]) == k
                ]
                for s, seed in enumerate(SEEDS):
                    td = D / "nl15_reference" / part / f"fold_{k}" / f"root_{seed}"
                    sha = hashlib.sha256(
                        (td / "fit_lock.json").read_bytes()
                    ).hexdigest()
                    assert sha == lock_hash[f"{k}:{seed}"]
                    marker = json.loads((td / "score_once.json").read_text())
                    assert (
                        marker["state"] == "COMPLETE"
                        and marker["fit_lock_sha256"] == sha
                        and marker["contexts"] == len(ix)
                    )
                    assert (
                        marker["picks_sha256"]
                        == hashlib.sha256(
                            pp[s, ix].astype(np.int32).tobytes()
                        ).hexdigest()
                    )
                    locks[f"{part}/{k}/{seed}"] = sha
            u, _ = ev.panel(G, pp)
            d = family_d(u, families)
            gains.extend(d)
            panels.extend(
                [
                    dict(
                        partition=part,
                        root_seed=seed,
                        mean_gain=float(d[s].mean()),
                        picks_sha256=hashlib.sha256(pp[s].tobytes()).hexdigest(),
                    )
                    for s, seed in enumerate(SEEDS)
                ]
            )
    assert np.array_equal(np.asarray(gains), expected)
    old = json.loads((D / "nl15_reference/metrics.json").read_text())
    median = float(np.median(np.asarray(gains).mean(axis=1)))
    assert median == old["aggregate"]["NL15"]["median_gain"] == 0.0584046708039792
    result = dict(
        status="PASS",
        source_commit="02dfee2f7e0b8543324494dd4f99d074cd6b93b6",
        panels=panels,
        count_panels=9,
        locks=locks,
        count_locks=36,
        count_scores=36,
        raw_family_arrays_exact=True,
        median_gain_exact=median,
        BASE_refits=0,
        NL15_refits=0,
        new_predictions=0,
        loader=r.R["loader_counters"],
    )
    r.durable(D / "nl15-regression.json", result)
    print(
        json.dumps(
            dict(status="PASS", panels=9, locks=36, scores=36, median_gain=median)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
