"""Independent TRAIN feature builder and frozen LightGBM forward pass for H305.

No research runner imports. Shared NPZ caches include 91 DL-Val rows: read only
verified TRAIN prefixes of NPY members, never materialize their trailing rows.
"""

from pathlib import Path
import hashlib
import json
import math
import zipfile
import numpy as np

SEEDS = (30001, 30002, 30003)
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prefix_array(path, key, rows):
    """Read a C-order NPY prefix without decoding or allocating trailing rows."""
    with zipfile.ZipFile(path) as z, z.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        assert not fortran and not dtype.hasobject and 0 <= rows <= shape[0]
        size = rows * math.prod(shape[1:]) * dtype.itemsize
        raw = f.read(size)
        assert len(raw) == size
        return np.frombuffer(raw, dtype=dtype).reshape((rows, *shape[1:])).copy()


def features(cost, prior, program, projection, lm, dope):
    """568 columns in the published model's ABI, derived independently."""
    cost = cost.astype(np.float64)
    order = np.argsort(cost, kind="stable")[:8]
    p = program[order].astype(np.float32)
    chunks = np.pad(program.astype(np.float32), ((0, 0), (0, 20))).reshape(-1, 18, 24)
    sem = chunks.mean(1) + np.float32(0.5) * chunks.max(1)
    sem = (sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-6)).astype(
        np.float64
    )
    sem8 = sem[order]
    c = cost[order]
    sd = max(float(cost.std()), 1e-12)
    z = (cost - cost.mean()) / sd
    z8 = (c - c.mean()) / max(float(c.std()), 1e-6)
    probabilities = []
    for v, temperature in [(z[order], 1.0), (z[order], 0.2), (z8, 1.0)]:
        e = np.exp(-(v - v.min()) / temperature)
        probabilities.append(e / e.sum())
    prior = prior.astype(np.float64)
    prior_z = (prior - prior.mean()) / max(float(prior.std()), 1e-6)
    standardized = []
    for values in (sem, projection.astype(np.float64)):
        standardized.append(
            np.clip(
                (values[order] - values.mean(0)) / (values.std(0) + 1e-5), -5, 5
            ).astype(np.float32)
        )
    sem_z, proj_z = standardized
    unit = proj_z / np.maximum(np.linalg.norm(proj_z, axis=1, keepdims=True), 1e-6)
    centroid = sem8.mean(0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-6)
    estimators = np.where(p[:, 24:29].max(1) > 0, p[:, 24:29].argmax(1), -1)
    counts = p[:, :23].sum(1).astype(np.float64)
    d = dope.astype(np.float32)
    context = np.concatenate(
        [
            lm.astype(np.float32),
            d[:25],
            np.array(
                [d.mean(), d.std(), np.linalg.norm(d) / math.sqrt(len(d))],
                dtype=np.float32,
            ),
        ]
    )
    rows = []
    for j in range(8):
        others = [i for i in range(8) if i != j]
        geometry = np.array(
            [
                c[j],
                z[order[j]],
                z8[j],
                j,
                j / 7,
                j == 0,
                c[j] - c[0],
                (c[j] - c[0]) / sd,
                (c[1] - c[0]) / sd,
                (c[7] - c[0]) / sd,
                *[v[j] for v in probabilities],
                prior_z[order[j]],
                *[c[j] - c[i] for i in others],
                *[(c[j] - c[i]) / sd for i in others],
            ],
            dtype=np.float32,
        )
        sc = [float(np.clip(np.dot(sem8[j], sem8[i]), -1, 1)) for i in others]
        pc = [float(np.clip(np.dot(unit[j], unit[i]), -1, 1)) for i in others]
        relations = np.array(
            [
                np.clip(np.dot(sem8[j], sem8[0]), -1, 1),
                np.mean(sc),
                max(sc),
                np.clip(np.dot(unit[j], unit[0]), -1, 1),
                np.mean(pc),
                max(pc),
                np.clip(np.dot(sem8[j], centroid), -1, 1),
                np.mean(np.clip(sem[:4] @ sem8[j], -1, 1)),
                counts[j],
                counts[j] - counts[0],
                estimators[j] == estimators[0],
                sum(estimators == estimators[j]),
            ],
            dtype=np.float32,
        )
        rows.append(
            np.concatenate([geometry, relations, p[j], sem_z[j], proj_z[j], context])
        )
    result = np.stack(rows)
    assert result.shape == (8, 568) and result.dtype == np.float32
    return order, result


def build_train(data_root, output):
    root = Path(data_root)
    selected = root / "h50/input/selected.json"
    assert digest(selected) == SELECTED_SHA
    rows = json.loads(selected.read_text())["train"]
    assert len(rows) == 1219 and len({r["family_id"] for r in rows}) == 205
    assert all(
        r["task"] == "regression" and not r.get("has_meta_test", False) for r in rows
    )
    cids = [r["context_sha256"] for r in rows]
    cache = root / "h58/cache"
    with np.load(cache / "pool_priors.npz", allow_pickle=False) as z:
        pc = list(map(str, z["cids"]))
        offsets = z["offsets"]
    with np.load(cache / "context_features.npz", allow_pickle=False) as z:
        cc = list(map(str, z["cids"]))
    assert set(pc[:1219]) == set(cids) and set(cc[:1219]) == set(cids)
    assert not set(pc[1219:]) & set(cids) and not set(cc[1219:]) & set(cids)
    base = prefix_array(
        cache / "pool_priors.npz", "base_prior_cost", int(offsets[1219])
    )
    projection = prefix_array(cache / "pool_priors.npz", "proj16", int(offsets[1219]))
    lm = prefix_array(cache / "context_features.npz", "lm48_z", 1219)
    dope = prefix_array(cache / "context_features.npz", "dope629", 1219)
    pi, ci = (
        {c: i for i, c in enumerate(pc[:1219])},
        {c: i for i, c in enumerate(cc[:1219])},
    )
    X, U, gates, identities = [], [], [], []
    costs_path = root / "h276/inputs/r882-train-costs.npz"
    with np.load(costs_path, allow_pickle=False) as costs:
        assert set(costs.files) == set(cids)
        for n, cid in enumerate(cids):
            path = root / "h50/input/pools" / (cid + ".npz")
            identities.append([cid, digest(path)])
            with np.load(path, allow_pickle=False) as z:
                prog = z["program"].astype(np.float32)
                fitness = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(bool)
            i = pi[cid]
            lo, hi = map(int, offsets[i : i + 2])
            order, x = features(
                costs[cid].astype(np.float32),
                base[lo:hi].astype(np.float32),
                prog,
                projection[lo:hi].astype(np.float32),
                lm[ci[cid]],
                dope[ci[cid]],
            )
            X.append(x)
            U.append(np.where(gate, -fitness, -2.0)[order])
            gates.append(gate[order])
            if n % 100 == 0:
                print("TRAIN feature contexts", n, flush=True)
    np.savez(
        output,
        X=np.stack(X),
        utility=np.stack(U),
        gates=np.stack(gates),
        cids=np.array(cids),
        families=np.array([r["family_id"] for r in rows]),
    )
    return dict(
        contexts=len(cids),
        selected_sha256=digest(selected),
        costs_sha256=digest(costs_path),
        pool_identities_sha256=hashlib.sha256(
            json.dumps(identities).encode()
        ).hexdigest(),
        train_array_sha256={
            k: hashlib.sha256(v.tobytes()).hexdigest()
            for k, v in [
                ("prior", base),
                ("projection", projection),
                ("lm48", lm),
                ("dope629", dope),
            ]
        },
        cache_metadata_rows=1310,
        cache_feature_rows_read=1219,
        excluded_tail_rows=91,
        physical_zip_decompression="May prefetch compressed trailing bytes; no trailing NPY rows requested, decoded or used",
        output_sha256=digest(output),
    )


def outer_fold(family):
    return hashlib.sha256(b"H302-outer-v1\0" + family.encode()).digest()[-1] % 4


def select(scores, margin):
    best = scores.argmax(axis=1)
    if math.isinf(margin):
        return np.zeros(len(best), dtype=np.int64)
    return np.where(scores[np.arange(len(best)), best] - scores[:, 0] > margin, best, 0)


def member_pick(directory, config, X, tau=None, votes=None):
    import lightgbm as lgb

    n = len(X)
    flat = X.reshape(-1, 568)

    def predict(info, iterations=None):
        path = directory / info["file"]
        assert digest(path) == info["sha256"], str(path)
        model = lgb.Booster(model_file=str(path), params={"num_threads": 1})
        assert model.num_feature() == 568
        return model.predict(flat, num_iteration=iterations, num_threads=1).reshape(
            n, 8
        )

    raw = predict(config["ranker"], config["selected_n_estimators"])
    invalid = predict(config["veto"])
    threshold = config["selected_tau"] if tau is None else tau
    mask = invalid >= threshold
    mask[:, 0] = False
    raw[mask] = -np.inf
    margin = float(config["strict_margin"])
    proposed = select(raw, margin)
    count = np.zeros(n, dtype=int)
    for bag in config["bags"]:
        score = predict(bag, config["selected_n_estimators"])
        count += score[np.arange(n), proposed] - score[:, 0] > margin
    q = config["q_fixed"] if votes is None else votes
    return np.where(count >= q, proposed, 0)


def consensus(members, agreement=2):
    members = np.asarray(members)
    answer = np.zeros(members.shape[1], dtype=np.int64)
    for slot in reversed(range(1, 8)):
        answer[(members == slot).sum(0) >= agreement] = slot
    return answer


def replay302(archive_dir, data):
    X, families = data["X"], data["families"]
    folds = np.array([outer_fold(str(f)) for f in families])
    picks = np.full((3, len(X)), -1, dtype=np.int64)
    receipts = []
    for k in range(4):
        directory = Path(archive_dir) / f"fold_{k}"
        lock = json.loads((directory / f"fold_{k}_fit_lock.json").read_text())
        test = np.flatnonzero(folds == k)
        for s, seed in enumerate(SEEDS):
            members = [
                member_pick(directory, c, X[test])
                for c in lock["members_by_root_seed"][str(seed)]
            ]
            picks[s, test] = consensus(members)
        receipts.append(
            dict(
                fold=k,
                test_families=len(set(families[test])),
                test_contexts=len(test),
                fit_families=len(set(families[folds != k])),
                direct_fit_overlap=0,
                fit_lock_sha256=digest(directory / f"fold_{k}_fit_lock.json"),
            )
        )
        print("H302 replay fold", k, flush=True)
    assert np.all(picks >= 0)
    return picks, receipts


def replay303(archive_dir, data):
    """Replay final refits only; caller must wait for the H303 closeout."""
    from audit_structure import check_nested_lock

    X, families = data["X"], data["families"]
    folds = np.array([outer_fold(str(f)) for f in families])
    picks = np.full((3, len(X)), -1, dtype=np.int64)
    receipts, seen = [], set()
    for path in sorted(Path(archive_dir).rglob("fit_lock.json")):
        lock = json.loads(path.read_text())
        if lock.get("schema") != "jope.h303.fit-lock/v1":
            continue
        checked = check_nested_lock(lock)
        k, root = lock["outer_fold"], lock["root_seed"]
        assert (k, root) not in seen
        seen.add((k, root))
        test = np.flatnonzero(folds == k)
        ch = lock["chosen"]
        members = [
            member_pick(path.parent / "refit", c, X[test], tau=ch["tau"], votes=ch["k"])
            for c in lock["members"]
        ]
        picks[SEEDS.index(root), test] = consensus(members, ch["a"])
        checked["fit_lock_sha256"] = digest(path)
        receipts.append(checked)
    assert seen == {(k, s) for k in range(4) for s in SEEDS} and np.all(picks >= 0)
    return picks, receipts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["features", "h302"])
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()
    if args.phase == "features":
        result = build_train("/home/ubuntu/jope-data", args.scratch / "train.npz")
    else:
        with np.load(args.scratch / "train.npz", allow_pickle=False) as data:
            picks, folds = replay302(args.scratch / "h302", data)
            families = data["families"]
            utility = data["utility"]
            means = []
            for p in picks:
                chosen = utility[np.arange(len(p)), p]
                means.append(
                    float(
                        np.mean(
                            [
                                chosen[families == f].mean()
                                - utility[families == f, 0].mean()
                                for f in sorted(set(families))
                            ]
                        )
                    )
                )
        np.save(args.scratch / "h302-picks.npy", picks)
        hashes = [
            hashlib.sha256(np.asarray(p, dtype="<i4").tobytes()).hexdigest()
            for p in picks
        ]
        published = json.loads(
            (args.scratch / "h302/metrics_full_h302.json").read_text()
        )["primary_ENS_VOTE2"]["summary"]
        expected = [
            "b47088d981afa1a53a8926afc1a025492fafcb9591a9cc52aa191e452c8560d7",
            "0c5761b312cd9366e2fd1e3f49ee7cb37908ff1b8d2228f49d4604d4ece40bed",
            "a4ff743c9b974aa5145cb0aebed51b946bddda47b237dc21024ef2011c95c9ee",
        ]
        result = dict(
            status="PASS"
            if hashes == expected
            and np.allclose(
                means, published["per_seed_family_mean_gain"], rtol=0, atol=1e-14
            )
            else "H302_REPRO_FAIL",
            picks_sha256=hashes,
            expected_picks_sha256=expected,
            gains=means,
            published_gains=published["per_seed_family_mean_gain"],
            folds=folds,
        )
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
