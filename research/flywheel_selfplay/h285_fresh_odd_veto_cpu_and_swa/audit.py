"""Terminal replay of saved evidence, with no neural forward or refit."""

import json
import hashlib
import numpy as np
from dependencies import ROOT
from analysis_h285 import summarize, decide, random_valid_picks, NULL_SEEDS


def main():
    out = ROOT / "attempts/attempt1"
    m = json.loads((out / "metrics-full.json").read_text())
    source = json.loads((ROOT / "prereg/published.json").read_text())["source_commit"]
    assert m["source_commit"] == source
    with np.load(out / "outcomes.npz") as z:
        u = z["utility"]
        bad = z["invalid"]
        p = z["program_picks"]
        families = z["families"].tolist()
        cids = z["cids"].tolist()
    stats = summarize(u, bad, p, families)
    assert stats == m["fresh"]
    assert decide(stats) == m["decision"]
    cohort = json.loads((ROOT / "inputs/fresh-selected.json").read_text())
    prefix = json.loads((out / "prefix-lock.json").read_text())
    selected = json.loads((out / "selected.json").read_text())
    assert prefix["family_keys"] == [
        r["family_key"] for r in cohort[: prefix["planned_prefix"]]
    ]
    expected = {
        r["context"]["context_sha256"] for r in cohort[: prefix["planned_prefix"]]
    }
    assert set(cids) | {r["cid"] for r in selected["excluded"]} == expected
    with np.load(out / "fresh-top8-no-picks.npz") as z:
        assert set(z.files) == {
            "cids",
            "X",
            "indices",
            "r3175_costs",
            "swa_costs_at_r3175_top8",
        }
        assert z["X"].shape == (len(cids), 8, 568)
        indices = z["indices"]
        assert z["cids"].tolist() == cids
    null_gates = []
    for i, c in enumerate(cids):
        with np.load(out / "pools" / f"{c}.npz") as z:
            gate = z["gate"]
            fit = z["fitness"]
            utility = np.where(gate, -fit, -2.0)
        with np.load(out / "costs" / f"{c}.npz") as z:
            assert np.array_equal(
                indices[i],
                np.argsort(z["r3175"].astype(np.float32), kind="mergesort")[:8],
            )
            assert np.all(p[:, i, 0] == indices[i, 0])
            assert np.all(p[:, i, 3] == np.argmin(z["swa"]))
        null_gates.append(gate[indices[i]])
        for s in range(3):
            for a in range(4):
                assert u[s, i, a] == utility[p[s, i, a]] and bad[s, i, a] == (
                    not bool(gate[p[s, i, a]])
                )
    for s, seed in enumerate(NULL_SEEDS):
        assert np.array_equal(
            p[s, :, 2],
            indices[np.arange(len(cids)), random_valid_picks(cids, null_gates, seed)],
        )
    assert np.array_equal(u[0, :, 3], u[1, :, 3]) and np.array_equal(
        u[1, :, 3], u[2, :, 3]
    )
    with np.load(out / "frozen-predictions.npz") as z:
        assert np.array_equal(
            p[:, :, 1], indices[np.arange(len(cids))[None, :], z["slots"]]
        )
    val = json.loads((out / "regression/metrics.json").read_text())
    assert (
        val["regression"]["cost_vectors_bit_exact_to_H278"] == 91
        and val["regression"]["cpu_mean"] == 1.1366829009431036
    )
    d = json.loads((out / "dope.json").read_text())
    assert (
        d["before"]
        == d["after"]
        == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
        and not d["CUDA_initialized"]
        and d["neural_updates"] == 0
    )
    for name, digest in json.loads((ROOT / "source/manifest.json").read_text())[
        "source_sha256"
    ].items():
        assert (
            hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest() == digest
        )
    result = dict(
        status="PASS",
        source_commit=source,
        contexts=len(cids),
        families=len(set(families)),
        recomputed_metrics_equal=True,
        raw_outcome_and_pick_replay=True,
        no_picks_or_labels_in_shared_packet=True,
        prefix_identity=True,
        regression_bit_exact_vectors=91,
        neural_forwards=0,
    )
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
