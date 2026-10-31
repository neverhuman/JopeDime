"""One immutable H135 stage per process; monitor.py is the only launch entry."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from unittest.mock import patch

import numpy as np
from science import SEEDS, ARMS, native, sanity, summarize

HERE = Path(__file__).resolve().parent
INPUT = Path("/home/ubuntu/jope-data/h50/input")
CHECKPOINT = Path("/home/ubuntu/jope-data/h134/run-20261031T2036Z/r882-copy.pt")
REFERENCE_SHA = "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a"
CHECKPOINT_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
H129_SHA = "e34d4a3f7ecb82971fa303f05cca665bb1108509abb90021d1a54717fe02cb6c"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def imported():
    path = HERE.parent / "h129_gbm_blend_harm_gate/runner.py"
    if sha(path) != H129_SHA:
        raise RuntimeError("H129 source changed")
    spec = importlib.util.spec_from_file_location("h135_h129", path)
    h129 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h129)
    h124 = h129.load_h124()
    h124.SEEDS = SEEDS
    return h129, h124


def roster(h124, root):
    selected = INPUT / "selected.json"
    if sha(selected) != h124.FILES[selected]:
        raise RuntimeError("selected packet changed")
    packet = json.loads(selected.read_text())
    if set(packet) != {"train", "dev", "protocol"}:
        raise RuntimeError("unexpected packet closure")
    train, val = packet["train"], packet["dev"]
    if len(train) != 1219 or len(val) != 91:
        raise RuntimeError("cohort count mismatch")
    for rows, split in [(train, "DL-Train"), (val, "DL-Val")]:
        if any(
            r["dl_split"] != split
            or r["task"] != "regression"
            or r["source_split"] not in ("train", "inner")
            for r in rows
        ):
            raise RuntimeError("unauthorized row")
    families = {r["family_id"] for r in val}
    kept = [r for r in train if r["family_id"] not in families]
    ref = root / "reference.json"
    if sha(ref) != REFERENCE_SHA:
        raise RuntimeError("reference contract changed")
    reference = json.loads(ref.read_text())
    contract = reference["evaluation_contract"]
    if (
        hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        != reference["evaluation_identity"]
    ):
        raise RuntimeError("contract identity mismatch")
    if contract["contexts"] != [
        dict(
            cid=r["context_sha256"],
            pool_sha256=r["pool_sha256"],
            training_table_sha256=r["train_only_jtt1_sha256"],
        )
        for r in val
    ]:
        raise RuntimeError("ordered validation contract mismatch")
    save(
        root / "cohort.json",
        dict(
            train_contexts=len(kept),
            validation_contexts=len(val),
            overlap_contexts=len(train) - len(kept),
            overlap_families=len({r["family_id"] for r in train} & families),
            validation_rows_in_fit=0,
        ),
    )
    return train, kept, val, reference


def pools_for(h124, rows):
    # This H135 loader explicitly authorizes DL-Val, without spoofing its split.
    pools = {}
    for r in rows:
        cid = r["context_sha256"]
        path = INPUT / "pools" / f"{cid}.npz"
        if sha(path) != r["pool_sha256"]:
            raise RuntimeError("pool identity mismatch")
        with np.load(path, allow_pickle=False) as z:
            fit = z["fitness"].astype(np.float64)
            gate = z["gate"].astype(bool)
        if (
            fit.shape != (r["n_candidates"],)
            or gate.shape != fit.shape
            or not np.isfinite(fit[gate]).all()
        ):
            raise RuntimeError("invalid pool labels")
        pools[cid] = dict(
            path=path,
            pool_sha256=r["pool_sha256"],
            fitness=fit,
            gate=gate,
            u=np.where(gate, -fit, -2.0),
        )
    return pools


def features(h124, train, val):
    # The pinned mixed caches contain exactly these train+val identities, no protected split.
    context, reads = h124.train_context(train + val, [])
    return context, reads


def baseline_one_thread(h124, *args):
    """A2: cap operational Torch threading without editing imported baseline math."""
    import torch

    setter = torch.set_num_threads
    setter(1)
    torch.set_num_interop_threads(1)
    with patch.object(torch, "set_num_threads", lambda requested: setter(1)):
        return h124.baseline(*args)


def baseline(h124, root, train, val, reference):
    # Pin dependencies without opening source_manifest or other excluded metadata.
    for p, digest in h124.FILES.items():
        if p.name in ("source_manifest.json", "cache_manifest.json", "round-882.pt"):
            continue
        if sha(p) != digest:
            raise RuntimeError("dependency hash mismatch: " + str(p))
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("offline checkpoint changed")
    h124.R882 = CHECKPOINT
    h124.FILES[CHECKPOINT] = CHECKPOINT_SHA
    context, reads = features(h124, train, val)
    pools = pools_for(h124, val)
    projection = h124.train_projection(train + val, [])
    # One torch process, BLAS pools capped to one before imports; <=4 total threads.
    costs, net = baseline_one_thread(h124, root, val, context, pools, projection)
    values = [
        native(
            costs[r["context_sha256"]],
            pools[r["context_sha256"]]["fitness"],
            pools[r["context_sha256"]]["gate"],
        )[0]
        for r in val
    ]
    receipt = sanity(values, reference)
    np.savez_compressed(root / "baseline.npz", **costs)
    receipt.update(
        per_context=values,
        baseline_sha256=sha(root / "baseline.npz"),
        reference_sha256=REFERENCE_SHA,
        reads=reads,
    )
    save(root / "sanity.json", receipt)


def fit(h129, h124, root, train, kept, val, seed):
    gate = json.loads((root / "sanity.json").read_text())
    if (
        gate["status"] != "PASS"
        or sha(root / "baseline.npz") != gate["baseline_sha256"]
    ):
        raise RuntimeError("baseline sanity must pass first")
    context, reads = features(h124, train, val)
    pools = pools_for(h124, kept + val)
    rng = np.random.default_rng(135000 + seed)
    samples = {
        r["context_sha256"]: 4
        + rng.choice(len(pools[r["context_sha256"]]["u"]) - 4, 192, replace=False)
        for r in kept
    }
    # Import the actual H129/H124 fit function unchanged. Virtual partition0 holds
    # all allowed training rows; partition4 holds val prediction targets only.
    mapping = {r["context_sha256"]: 0 for r in kept}
    mapping.update({r["context_sha256"]: 4 for r in val})
    from sklearn.ensemble import HistGradientBoostingRegressor

    captured = []

    def capture(**kwargs):
        model = HistGradientBoostingRegressor(**kwargs)
        captured.append(model)
        return model

    from threadpoolctl import threadpool_limits

    with (
        threadpool_limits(limits=1, user_api="blas"),
        threadpool_limits(limits=4, user_api="openmp"),
    ):
        with patch("sklearn.ensemble.HistGradientBoostingRegressor", capture):
            receipt = h124.fit_one(
                (seed, (4,)), root, kept + val, context, pools, samples, mapping
            )
        if len(captured) != 1:
            raise RuntimeError("exactly one full-train fit required")
        model = captured.pop()
        predictions, fitness, gates = [], [], []
        for r in kept:
            cid = r["context_sha256"]
            idx = np.setdiff1d(np.arange(4, len(pools[cid]["u"])), samples[cid])
            cf = np.concatenate(
                [context[cid]["lm48_z"], context[cid]["dope629"]]
            ).astype(np.float32)
            xp = h124.program(pools, cid)[idx]
            x = np.concatenate([xp, np.repeat(cf[None], len(xp), axis=0)], axis=1)
            predictions.append(model.predict(x).astype(np.float32))
            fitness.append(pools[cid]["fitness"][idx])
            gates.append(pools[cid]["gate"][idx])
        a, b = h124.calibration(
            np.concatenate(predictions), np.concatenate(fitness), np.concatenate(gates)
        )
    with np.load(root / "baseline.npz", allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    values = {arm: [] for arm in ARMS}
    choices = {arm: [] for arm in ARMS}
    with np.load(root / f"seed{seed}-exclude4.npz", allow_pickle=False) as z:
        for r in val:
            cid = r["context_sha256"]
            cost = (a - b * z[cid]).astype(np.float32)
            arms = dict(
                BLEND_025=h129.blend(base[cid], cost, 0.25),
                PURE_GBM=cost,
                BLEND_050=h129.blend(base[cid], cost, 0.5),
            )
            for arm, c in arms.items():
                v, i = native(c, pools[cid]["fitness"], pools[cid]["gate"])
                values[arm].append(v)
                choices[arm].append(i)
    save(
        root / f"seed{seed}.json",
        dict(
            seed=seed,
            sampling_seed=135000 + seed,
            sampling_sha256=hashlib.sha256(
                b"".join(
                    samples[r["context_sha256"]].astype("<i8").tobytes() for r in kept
                )
            ).hexdigest(),
            calibration=[a, b],
            calibration_rows=sum(len(x) for x in predictions),
            calibration_scope="non-sampled DL-Train candidates only; same contexts as fitting, not family OOF",
            fit=receipt,
            values=values,
            choices=choices,
            reads=reads,
        ),
    )


def install_audit(root, train, val):
    allowed = {INPUT / "selected.json", INPUT / "cpu_exact_fused.py", CHECKPOINT}
    allowed.update(
        INPUT / "pools" / (r["context_sha256"] + ".npz") for r in train + val
    )
    allowed.update(
        Path("/home/ubuntu/jope-data/h58/cache") / name
        for name in ("context_features.npz", "pool_priors.npz")
    )
    opened = set()

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        lower = str(path).lower()
        if any(
            term in lower
            for term in (
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
                "flywheel-full-20261030",
            )
        ):
            raise RuntimeError("protected path refused")
        if path.is_relative_to(Path("/home/ubuntu/jope-data")) and path not in allowed:
            raise RuntimeError("data outside explicit H135 closure: " + str(path))
        if path in allowed:
            mode = args[1] or ""
            flags = args[2]
            if any(k in mode for k in "wax+") or (
                isinstance(flags, int)
                and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)
            ):
                raise RuntimeError("input mutation refused")
            opened.add(str(path))

    sys.addaudithook(audit)
    return opened


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage", choices=["baseline", "135", "136", "137", "summary"], required=True
    )
    args = p.parse_args()
    root = args.root.resolve()
    launch = json.loads((root.parent / "launch.json").read_text())
    if (
        not launch["merged_source_verified"]
        or not launch["seed_freshness_verified"]
        or os.getpriority(os.PRIO_PROCESS, 0) < 19
    ):
        raise RuntimeError("merge, seed and nice19 launch gates required")
    h129, h124 = imported()
    train, kept, val, reference = roster(h124, root)
    opened = install_audit(root, train, val)
    if args.stage == "baseline":
        baseline(h124, root, train, val, reference)
    elif args.stage == "summary":
        results = [json.loads((root / f"seed{s}.json").read_text()) for s in SEEDS]
        if len({r["sampling_sha256"] for r in results}) != 3:
            raise RuntimeError("identical training samples")
        metrics = summarize(
            val,
            json.loads((root / "sanity.json").read_text())["per_context"],
            {a: [r["values"][a] for r in results] for a in ARMS},
        )
        save(root / "metrics.json", metrics)
        if (root / "metrics.json").stat().st_size >= 100000:
            raise RuntimeError("oversized metrics")
    else:
        fit(h129, h124, root, train, kept, val, int(args.stage))
    save(
        root / ("read-audit-" + args.stage + ".json"),
        dict(opened=sorted(opened), protected_reads=0),
    )
    print("COMPLETE", args.stage, flush=True)


if __name__ == "__main__":
    main()
