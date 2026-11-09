"""The H280/H124 CPU incumbent path, with checkpoint argument and frozen numeric1."""

from pathlib import Path
import hashlib
import json
import numpy as np
from dependencies import ROOT
import h124
import data


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_existing_val(checkpoint, out):
    import torch

    torch.set_num_threads(1)
    original = torch.set_num_threads
    torch.set_num_threads = lambda n: original(1)
    h124.REPO = Path("/home/ubuntu/JopeDime")
    h124.RSS_LIMIT_KIB = 4 * 1024 * 1024
    h124.R882 = checkpoint
    h124.FILES[checkpoint] = sha(checkpoint)
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    rows = selected["dev"]
    allrows = selected["train"] + rows
    context, _ = h124.train_context(allrows, [])
    context = {r["context_sha256"]: context[r["context_sha256"]] for r in rows}
    projection = h124.train_projection(allrows, [])
    pools = {}
    for row in rows:
        c = row["context_sha256"]
        path = data.POOLS / "pools" / f"{c}.npz"
        assert sha(path) == row["pool_sha256"]
        with np.load(path, allow_pickle=False) as z:
            pools[c] = dict(
                path=path,
                pool_sha256=row["pool_sha256"],
                u=np.zeros(row["n_candidates"]),
                fitness=z["fitness"][:4].copy(),
                gate=z["gate"][:4].copy(),
            )
    out.mkdir()
    costs, net = h124.baseline(out, rows, context, pools, projection)
    del net
    u = []
    invalid = []
    picks = []
    for row in rows:
        c = row["context_sha256"]
        j = int(np.argmin(costs[c]))
        picks.append(j)
        with np.load(data.POOLS / "pools" / f"{c}.npz") as z:
            u.append(float(-z["fitness"][j] if z["gate"][j] else -2))
            invalid.append(not bool(z["gate"][j]))
    return rows, costs, np.asarray(u), np.asarray(invalid), np.asarray(picks)


def regression_and_swa_val(out):
    out.mkdir()
    rows, costs, base, bad0, pick0 = score_existing_val(
        ROOT / "inputs/checkpoints/round-3175.pt", out / "r3175"
    )
    with np.load(ROOT / "inputs/r3175-secondary-costs.npz") as reference:
        assert all(
            np.array_equal(costs[r["context_sha256"]], reference[r["context_sha256"]])
            for r in rows
        )
    assert float(base.mean()) == 1.1366829009431036
    rows, _, swa, bad1, pick1 = score_existing_val(
        ROOT / "inputs/checkpoints/swa.pt", out / "swa"
    )
    np.savez_compressed(
        out / "swa-val.npz",
        base=base,
        swa=swa,
        base_invalid=bad0,
        swa_invalid=bad1,
        base_picks=pick0,
        swa_picks=pick1,
        cids=np.array([r["context_sha256"] for r in rows]),
    )
    from analysis_h285 import family_values, contrast, BOOT, elo_per_seed

    ids, fm = family_values(
        np.stack([base, swa], axis=-1)[None], [r["family_id"] for r in rows]
    )
    d = fm[:, :, 1] - fm[:, :, 0]
    draws = np.random.default_rng(BOOT).integers(0, len(ids), (10000, len(ids)))
    stats = contrast(d, draws)
    result = dict(
        regression=dict(
            status="PASS",
            cost_vectors_bit_exact_to_H278=91,
            cpu_mean=float(base.mean()),
        ),
        B_reused_DLVal=dict(
            single_run=True,
            reused=True,
            elo_base_swa=elo_per_seed(
                np.stack([base, swa], axis=-1)[None], [r["family_id"] for r in rows]
            )[0].tolist(),
            gain=stats["per_seed_gain"][0],
            family_bootstrap_ci95=stats["per_seed_ci95"][0],
            wtl=stats["wtl_by_seed"][0],
            harm_families_loss_gt_0_01=stats["harm_families_loss_gt_0_01"][0],
            invalid_base=int(bad0.sum()),
            invalid_swa=int(bad1.sum()),
            changed_picks=int((pick0 != pick1).sum()),
        ),
        forward_calls="r3175 CPU91 + SWA CPU91; no refit",
        neural_updates=0,
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
