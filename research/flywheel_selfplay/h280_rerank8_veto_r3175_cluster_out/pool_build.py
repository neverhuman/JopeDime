"""H278 cost lineage (H226/H124 CPU forward), frozen r3175 copy, outcome-blind topk build."""

from pathlib import Path
import json
import sys
import time
import hashlib
import numpy as np
from dependencies import ROOT, core, topk
import data
import h124


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(rows, out):
    import torch

    sys.addaudithook(data.audit_hook)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # H226's operational override holds the unchanged H124 CPU forward at numeric1.
    original = torch.set_num_threads
    torch.set_num_threads = lambda n: original(1)
    h124.REPO = Path("/home/ubuntu/JopeDime")
    h124.R882 = ROOT / "inputs/round-3175.pt"
    h124.FILES[h124.R882] = (
        "1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b"
    )
    h124.RSS_LIMIT_KIB = 4 * 1024 * 1024
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    allrows = selected["train"] + selected["dev"]
    ctx, _ = h124.train_context(allrows, [])
    ctx = {r["context_sha256"]: ctx[r["context_sha256"]] for r in rows}
    projection = h124.train_projection(allrows, [])
    # Only the four already-revealed anchor labels enter the incumbent forward.
    # Full utility, gate and ranker labels are never constructed here.
    pools = {}
    for r in rows:
        cid = r["context_sha256"]
        path = data.POOLS / "pools" / f"{cid}.npz"
        assert sha(path) == r["pool_sha256"]
        with np.load(path, allow_pickle=False) as z:
            pools[cid] = dict(
                path=path,
                pool_sha256=r["pool_sha256"],
                u=np.zeros(r["n_candidates"]),
                fitness=z["fitness"][:4].copy(),
                gate=z["gate"][:4].copy(),
            )
    out.mkdir()
    started = time.monotonic()
    costs, net = h124.baseline(out, rows, ctx, pools, projection)
    elapsed = time.monotonic() - started
    ap, pj, cf = data.priors_for(rows)
    ids = []
    features = []
    indices = []
    gaps = []
    for r in rows:
        cid = r["context_sha256"]
        rc = costs[cid].astype(np.float32)
        with np.load(data.POOLS / "pools" / f"{cid}.npz", allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
        order = np.argsort(rc, kind="mergesort")
        idx, X = topk.build_topk(
            rc,
            ap[cid],
            prog[order[:8]],
            core.compute_sem24_numpy(prog),
            pj[cid],
            cf[cid]["lm48_z"],
            cf[cid]["dope629"],
            8,
        )
        ids.append(cid)
        features.append(X)
        indices.append(idx)
        gaps.append(float(rc[order[1]] - rc[order[0]]))
    np.savez_compressed(
        out / "features.npz",
        cids=np.asarray(ids),
        X=np.stack(features),
        indices=np.stack(indices),
        cpu_top2_gap=np.asarray(gaps),
    )
    assert sha(h124.R882) == h124.FILES[h124.R882]
    receipt = dict(
        contexts=len(rows),
        families=len({r["family_id"] for r in rows}),
        cost_seconds=elapsed,
        seconds_including_features=time.monotonic() - started,
        projected_train_seconds=(time.monotonic() - started) * 1219 / len(rows),
        source_cost_path="H278 cached costs <- H226 <- H124.baseline; exact H124 byte-pinned forward with numeric1",
        checkpoint_sha256=sha(h124.R882),
        dope=json.loads(
            (
                out
                / (
                    "dope-train.json"
                    if rows[0]["dl_split"] == "DL-Train"
                    else "dope-secondary.json"
                )
            ).read_text()
        ),
        utility_outcomes_read=False,
        only_revealed_anchor_labels_used=True,
        features_sha256=sha(out / "features.npz"),
    )
    save(out / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


def main():
    mode = sys.argv[1]
    selected = json.loads((ROOT / "inputs/selected.json").read_text())
    if mode.startswith("smoke"):
        # Fixed outcome-blind systematic sample over sorted TRAIN context IDs, 20.
        rows = sorted(selected["train"], key=lambda r: r["context_sha256"])
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, 20, dtype=int)]
    elif mode == "val":
        rows = selected["dev"]
    else:
        chosen = set(json.loads((ROOT / "source/cohort.json").read_text())["families"])
        rows = [r for r in selected["train"] if r["family_id"] in chosen]
    prepare(rows, ROOT / "prep" / mode)


if __name__ == "__main__":
    main()
