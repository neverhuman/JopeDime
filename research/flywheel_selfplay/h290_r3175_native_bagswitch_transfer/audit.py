"""Verify saved selections, paired family inference and unchanged model identities."""

from pathlib import Path
import json
import math
import hashlib
import numpy as np
import statistics_h290 as stat

ROOT = Path(__file__).resolve().parent.parent


def main():
    out = ROOT / "attempt1"
    m = json.loads((out / "metrics-full.json").read_text())
    count = 0
    for cohort, section in [("H286", "primary"), ("H283", "secondary")]:
        with np.load(ROOT / "prepared" / (cohort + ".npz")) as z:
            u = z["utility"]
            gate = z["gate"]
            ids = z["cids"]
            fam = z["families"]
        p = json.loads((out / (cohort + "-picks.json")).read_text())
        picks = np.asarray(p["picks"])
        assert p["cids"] == ids.tolist() and p["arms"] == list(stat.ARMS)
        with np.load(out / (cohort + "-panels.npz")) as z:
            values = z["utility"]
            invalid = z["invalid"]
            indices = z["bootstrap_indices"]
            stored_fm = z["family_utilities"]
        expected = np.stack([u[np.arange(len(ids))[:, None], j] for j in picks])
        assert np.array_equal(values, expected)
        assert np.array_equal(
            invalid, np.stack([~gate[np.arange(len(ids))[:, None], j] for j in picks])
        )
        assert np.all(picks[:, :, 0] == 0)
        fs = sorted(set(fam))
        fm = np.array(
            [[values[s, fam == f].mean(axis=0) for f in fs] for s in range(3)]
        )
        assert np.array_equal(fm, stored_fm)
        assert np.array_equal(
            indices,
            np.random.default_rng(290150).integers(0, len(fs), (10000, len(fs))),
        )
        sample = np.array([fm[s, indices].mean(axis=1) for s in range(3)]).transpose(
            1, 0, 2
        )
        for j, arm in enumerate(stat.ARMS):
            d = fm[:, :, j] - fm[:, :, 0]
            bs = sample[:, :, j] - sample[:, :, 0]
            a = m[section]["arms"][arm]
            assert np.allclose(
                d.mean(axis=1), a["gain"]["per_seed"], rtol=0, atol=1e-13
            )
            assert np.allclose(
                np.quantile(bs, [0.025, 0.975], axis=0).T,
                a["gain"]["per_seed_ci95"],
                rtol=0,
                atol=1e-12,
            )
            assert np.allclose(
                np.quantile(np.median(bs, axis=1), [0.025, 0.975]),
                a["gain"]["median_ci95"],
                rtol=0,
                atol=1e-12,
            )
            for i in range(3):
                w = int((d[i] > 1e-4).sum())
                l = int((d[i] < -1e-4).sum())  # noqa: E741
                n = w + l
                exact_p = (
                    1.0
                    if n == 0
                    else sum(math.comb(n, k) for k in range(w, n + 1)) / 2**n
                )
                assert abs(exact_p - a["sign_test"][i]["p_one_sided"]) < 1e-12
                assert (
                    w == a["sign_test"][i]["wins"] and l == a["sign_test"][i]["losses"]
                )
            assert (d < -0.01).sum(axis=1).tolist() == a["harmed_loss_gt_001"]
            assert (values[:, :, j] == -2).sum(axis=1).tolist() == a["raw_minus2"]
        con = m[section]["native_minus_r882_fit"]
        b = sample[:, :, 1] - sample[:, :, 2]
        assert np.allclose(
            np.quantile(np.median(b, axis=1), [0.025, 0.975]),
            con["median_ci95"],
            atol=1e-12,
            rtol=0,
        )
        count += picks.size
    assert stat.verdict(m["primary"]) == m["decision"]
    lock = json.loads((ROOT / "fit/fit_lock.json").read_text())
    reference = json.loads((ROOT / "code-closure/export_manifest.json").read_text())
    models = 0
    for root, cells in [
        (ROOT / "fit/models", lock["cells"]),
        (ROOT / "reference", reference["cells"]),
    ]:
        for c in cells:
            items = [c["ranker"], c["veto"], *c["bags"]]
            if "placebo" in c:
                items += [c["placebo"], *c["placebo_bags"]]
            for item in items:
                assert (
                    hashlib.sha256((root / item["file"]).read_bytes()).hexdigest()
                    == item["sha256"]
                )
                models += 1
    for name in ["fit.py", "dependencies.py", "monitor.py", "custody.py"]:
        assert (
            hashlib.sha256((ROOT / "source" / name).read_bytes()).hexdigest()
            == lock["source_sha256"][name]
        )
    r = dict(
        status="PASS",
        selected_slots_and_outcome_joins_verified=count,
        model_hashes_unchanged=models,
        bootstrap_draws_verified=20000,
        paired_family_statistics_and_exact_sign_test_verified=True,
        source_fit_bindings_unchanged=True,
        metrics_sha256=hashlib.sha256(
            (out / "metrics-full.json").read_bytes()
        ).hexdigest(),
        neural_forwards=0,
        neural_updates=0,
    )
    (out / "audit.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
