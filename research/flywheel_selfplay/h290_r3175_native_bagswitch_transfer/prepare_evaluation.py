"""Validate recorded GPU packets and reconstruct unchanged top-k features, without model scoring."""

from dependencies import ROOT, core, topk
import json
import hashlib
import numpy as np


def priors(root, rows):
    wanted = {r["context_sha256"] for r in rows}
    ap = {}
    pj = {}
    cf = {}
    with np.load(root / "pool_priors.npz") as z:
        ids = list(map(str, z["cids"]))
        offsets = z["offsets"]
        base = z["base_prior_cost"]
        proj = z["proj16"]
        for i, c in enumerate(ids):
            if c in wanted:
                lo, hi = map(int, offsets[i : i + 2])
                ap[c] = base[lo:hi].astype(np.float32)
                pj[c] = proj[lo:hi].astype(np.float32)
    with np.load(root / "context_features.npz") as z:
        ids = list(map(str, z["cids"]))
        lm = z["lm48_z"]
        dope = z["dope629"]
        for i, c in enumerate(ids):
            if c in wanted:
                cf[c] = dict(
                    lm48_z=lm[i].astype(np.float32), dope629=dope[i].astype(np.float32)
                )
    return ap, pj, cf


def main():
    assert (ROOT / "source/fit-custody.json").exists()
    train = json.loads((ROOT / "training/selected.json").read_text())["train"]
    trained_ids = {r["context_sha256"] for r in train}
    trained_datasets = {r["dataset_id"] for r in train}
    out = ROOT / "prepared"
    out.mkdir()
    records = []
    family_sets = []
    for wave, expected in [(286, 200), (283, 29)]:
        root = ROOT / "evaluation" / f"H{wave}"
        rows = json.loads((root / "selected.json").read_text())["fresh"]
        ids = [r["context_sha256"] for r in rows]
        families = [r["family_id"] for r in rows]
        assert len(rows) == len(set(families)) == expected
        assert not trained_ids.intersection(ids) and not trained_datasets.intersection(
            r["dataset_id"] for r in rows
        )
        assert all(
            r["source_split"] == "train"
            and r["dl_split"] == "FRESH"
            and r["task"] == "regression"
            for r in rows
        )
        family_sets.append(set(families))
        if wave == 286:
            cohort = json.loads((root / "cohort.json").read_text())["assignment"][
                "H286"
            ]
            assert (
                set(r["family_key"] for r in rows) == set(cohort["families"])
                and cohort["half"] == "EVEN"
            )
        rec = json.loads((root / "rescore.json").read_text())
        assert (
            rec["outcome"] == "A_EXACT"
            and rec["picks_equal"] == 91
            and rec["per_context_equal"] == 91
            and rec["gpu_mean"] == 1.1389572403748425
        )
        assert (
            rec["dope_before"]
            == rec["dope_after"]
            == "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
        )
        assert (
            hashlib.sha256(
                (root / "fresh-r3175-gpu-costs.npz").read_bytes()
            ).hexdigest()
            == rec["fresh"]["costs_sha256"]
        )
        for r in rows:
            assert (
                hashlib.sha256(
                    (root / "pools" / (r["context_sha256"] + ".npz")).read_bytes()
                ).hexdigest()
                == r["pool_sha256"]
            )
        with np.load(root / "fresh-r3175-gpu-costs.npz") as z:
            costs = {c: z[c].astype(np.float32) for c in ids}
        core.INPUT_POOLS_DIR = root
        ap, pj, cf = priors(root, rows)
        P = topk.load_packets(rows, costs, ap, pj, cf, 8)
        with np.load(root / "fresh_top8_packets.npz") as z:
            assert list(map(str, z["cids"])) == ids
            X = np.stack([P[c]["X"] for c in ids])
            assert np.array_equal(X, z["X"])
            assert np.array_equal(
                z["order_top8"],
                np.stack([np.argsort(costs[c], kind="mergesort")[:8] for c in ids]),
            )
        u = np.stack([P[c]["utility_k"] for c in ids])
        gate = np.stack([P[c]["gate_k"] for c in ids])
        assert abs(float(u[:, 0].mean()) - rec["fresh"]["slot0_mean"]) < 1e-12
        order = np.argsort(ids)
        np.savez_compressed(
            out / f"H{wave}.npz",
            cids=np.array(ids)[order],
            families=np.array(families)[order],
            X=X[order],
            utility=u[order],
            gate=gate[order],
        )
        records.append(
            dict(
                wave=f"H{wave}",
                contexts=expected,
                families=expected,
                requested_families=200 if wave == 286 else 60,
                unbuilt_families=0 if wave == 286 else 31,
                all_features_bit_exact=True,
                all_top8_orders_exact=True,
                recorded_gpu_regression_91_of_91=True,
                recorded_gpu_slot0_mean=rec["fresh"]["slot0_mean"],
                recomputed_slot0_mean=float(u[:, 0].mean()),
                training_context_and_dataset_overlap=0,
                file=f"H{wave}.npz",
                sha256=hashlib.sha256((out / f"H{wave}.npz").read_bytes()).hexdigest(),
            )
        )
    assert not family_sets[0].intersection(family_sets[1])
    result = dict(
        status="PASS",
        cohorts=records,
        semi_fresh=True,
        paired_cpu_gpu_drift=dict(
            status="NOT_MEASURABLE",
            reason="No recorded matched CPU/GPU packets for these test contexts; CPU training and GPU test concern different contexts. No new forwards.",
        ),
        neural_forwards=0,
        neural_updates=0,
    )
    (out / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
