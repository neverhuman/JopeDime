"""Pre-attempt regressions on the prior r882 cell and fixed r3175 CPU DL-Val91."""

import json
import numpy as np
from dependencies import ROOT, topk, core, clusters
import data
from pool_build import save, sha


def run():
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    train, val = selected["train"], selected["dev"]
    core.INPUT_POOLS_DIR = data.POOLS
    costs = {}
    for name in ["r882-train-costs.npz", "r882-secondary-costs.npz"]:
        with np.load(ROOT / "inputs" / name, allow_pickle=False) as z:
            costs.update({c: z[c].astype(np.float32) for c in z.files})
    baseline = data.native91(
        val, costs, json.loads((ROOT / "inputs/best_dev_checkpoint.json").read_text())
    )
    ap, pj, cf = data.priors_for(train)
    packets = topk.load_packets(train, costs, ap, pj, cf, 8)
    import analysis_h272

    fam = {r["context_sha256"]: r["family_id"] for r in train}
    folds = analysis_h272.cids_by_fold(train, 272140)
    got = topk.run_cell(
        272101, 0, "RERANK8", False, folds, packets, fam, 8, salt=272140
    )
    ref = json.loads((ROOT / "inputs/h272-F272140-S272101-K0.json").read_text())
    inv = int(
        sum(not packets[c]["gate_k"][j] for c, j in zip(folds[0], got["assess_picks"]))
    )
    chg = sum(j != 0 for j in got["assess_picks"])
    deep = sum(j >= 4 for j in got["assess_picks"])
    assert (
        got["receipt"] == ref["receipts"]["RERANK8"]
        and inv == ref["invalid"]["RERANK8"]
        and chg == ref["changed"]["RERANK8"]
        and deep == ref["deep_picks"]
    )
    u = []
    equal = 0
    with (
        np.load(ROOT / "prep/val/r882-secondary-costs.npz") as cpu,
        np.load(ROOT / "inputs/r3175-secondary-costs.npz") as cached,
    ):
        for r in val:
            c = r["context_sha256"]
            assert np.array_equal(cpu[c], cached[c])
            equal += 1
            with np.load(data.POOLS / "pools" / f"{c}.npz") as z:
                j = int(np.argmin(cpu[c]))
                u.append(float(-z["fitness"][j] if z["gate"][j] else -2.0))
    assert float(np.mean(u)) == 1.1366829009431036
    assert (
        sha(ROOT / "source/folds.json")
        == "e89f9aa130cf1529a684c83c2dc8bd1658dbae69a5c1e0af87760ed8d92208f2"
    )
    manifest = json.loads((ROOT / "source/folds.json").read_text())
    parts = clusters.partitions_from_manifest(manifest)
    assert all(
        sorted(f for g in groups for f in g) == sorted(set(fam.values()))
        for groups in parts.values()
    )
    result = dict(
        status="PASS",
        h272_cell="F272140-S272101-K0",
        receipt_exact=True,
        invalid=inv,
        changed=chg,
        deep_picks=deep,
        native91=baseline,
        r3175_cpu_mean=float(np.mean(u)),
        r3175_cost_vectors_bit_exact_vs_H278=equal,
        cluster_manifest_sha256=sha(ROOT / "source/folds.json"),
        neural_updates=0,
    )
    save(ROOT / "regression.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    run()
