"""Feature-only parent; per-task outer_train labels; fsynced guard locks; separate score."""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import datetime, hashlib, json, os, sys
import numpy as np
import trainonly_loader as tl
import guard_policy as policy

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h311-astra")
POOLS = Path("/home/ubuntu/jope-data/h50/input")
SEEDS = (30001, 30002, 30003)
PARTS = ("P0", "P1", "P2")
SALTS = ("H302-outer-v1", "H308-outer-v1", "H308-outer-v2")
G = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    return sha(path)


def fold_of(part, family):
    return (
        hashlib.sha256(
            (SALTS[PARTS.index(part)] + "\0" + str(family)).encode()
        ).digest()[-1]
        % 4
    )


def read_program_only(path):
    with np.load(path, allow_pickle=False) as pool:
        return pool["program"].astype(np.float32)


def verify_sources():
    pins = json.loads((P / "source/dependency_pins.json").read_text())
    for name, h in pins["vendor"].items():
        assert sha(P / "vendor" / name) == h, name
    assert sha(P / "source/trainonly_loader.py") == pins["loader_sha256"]


def modules():
    verify_sources()
    sys.path.insert(0, str(P / "vendor/h303_ensvote2_fully_nested_hparams"))
    import run_h303 as r

    sys.path.insert(0, str(P / "vendor/h302_ensvote2_trainseed_nested_cv"))
    import run_h302 as r302

    r302.install_h302_audit_guard()
    core, topk, veto, bagged, ens = r302.load_pinned_h287_modules(D / "pinned_h287")
    core.INPUT_POOLS_DIR = POOLS
    # H295 imports the same frozen feature/model helper bytes; verify before reuse.
    for name in ["h265_core.py", "topk.py", "veto.py", "bagged.py", "ensemble.py"]:
        assert sha(D / "pinned_h287" / name) == sha(
            P / "vendor/h295_label_free_validity_guard" / name
        ), name
    sys.path.insert(0, str(P / "vendor/h295_label_free_validity_guard"))
    import valguard as vg

    G.update(r=r, core=core, topk=topk, veto=veto, bagged=bagged, ens=ens, vg=vg)
    return r


def features():
    r = modules()
    core, topk = G["core"], G["topk"]
    selected = POOLS / "selected.json"
    assert (
        sha(selected)
        == "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
    )
    rows = json.loads(selected.read_text())["train"]
    cids = [x["context_sha256"] for x in rows]
    fams = [x["family_id"] for x in rows]
    proxy = tl.TrainOnlyNumpy(set(cids))
    cache = Path("/home/ubuntu/jope-data/h58/cache")
    with proxy.load(cache / "pool_priors.npz", allow_pickle=False) as z:
        pc = [str(x) for x in z["cids"]]
        offset = z["offsets"]
        base = z["base_prior_cost"]
        proj = z["proj16"]
        pi = {c: i for i, c in enumerate(pc)}
    with proxy.load(cache / "context_features.npz", allow_pickle=False) as z:
        cc = [str(x) for x in z["cids"]]
        lm = z["lm48_z"]
        dope = z["dope629"]
        ci = {c: i for i, c in enumerate(cc)}
    allx = []
    indices = []
    with np.load(
        "/home/ubuntu/jope-data/h276/inputs/r882-train-costs.npz", allow_pickle=False
    ) as z:
        for cid in cids:
            cost = z[cid].astype(np.float32)
            order = np.argsort(cost, kind="mergesort")
            # No fitness/gate access: np.load is lazy per ZIP member.
            prog = read_program_only(POOLS / "pools" / f"{cid}.npz")
            lo, hi = int(offset[pi[cid]]), int(offset[pi[cid] + 1])
            idx, x = topk.build_topk(
                cost,
                base[lo:hi].astype(np.float32),
                prog[order[:8]],
                core.compute_sem24_numpy(prog),
                proj[lo:hi].astype(np.float32),
                lm[ci[cid]].astype(np.float32),
                dope[ci[cid]].astype(np.float32),
                8,
            )
            allx.append(x)
            indices.append(idx)
    X = np.stack(allx)
    indices = np.array(indices, dtype=np.int32)
    assert X.shape == (1219, 8, 568) and len(set(fams)) == 205
    out = D / "features.npz"
    np.savez(out, X=X, indices=indices, cids=np.array(cids), fams=np.array(fams))
    save(
        P / "source/feature_receipt.json",
        dict(
            sha256=sha(out),
            X_sha256=array_sha(X),
            indices_sha256=array_sha(indices),
            loader_counters=tl.COUNTERS,
            physical_validity_or_utility_arrays_read=0,
            contexts=1219,
            families=205,
            source_sha256=sha(__file__),
        ),
    )
    print("FEATURES_ONLY_COMPLETE", flush=True)


class ScopedLabels:
    """Only allowed context files can be opened; outer labels cannot enter a fit worker."""

    def __init__(self, cids, indices, allowed, root=POOLS):
        self.cids = cids
        self.indices = indices
        self.allowed = set(int(i) for i in allowed)
        self.root = Path(root)
        self.denied = 0
        self.reads = 0

    def read(self, ids):
        ids = np.asarray(ids, dtype=int)
        if any(int(i) not in self.allowed for i in ids):
            self.denied += 1
            raise RuntimeError("outer_test_labels_read_before_lock forbidden")
        u = []
        gate = []
        for i in ids:
            with np.load(
                self.root / "pools" / f"{self.cids[i]}.npz", allow_pickle=False
            ) as z:
                f = z["fitness"].astype(np.float64)
                g = z["gate"].astype(bool)
            idx = self.indices[i]
            u.append(np.where(g[idx], -f[idx], -2.0))
            gate.append(g[idx])
            self.reads += 1
        return np.asarray(u), np.asarray(gate)


def prepare():
    modules()
    assert (
        json.loads((P / "source/regression_receipt.json").read_text())["status"]
        == "PASS"
    )
    rec = json.loads((P / "source/feature_receipt.json").read_text())
    assert sha(D / "features.npz") == rec["sha256"]
    with np.load(D / "features.npz", allow_pickle=False) as z:
        G.update({k: z[k] for k in z.files})
    G["aug"] = G["vg"].build_candidate_guard_features(G["X"])


def reference(part, k, seed):
    root = D / "references" / ("h303/scratch" if part == "P0" else "h308/" + part)
    path = root / f"fold_{k}/root_{seed}/fit_lock.json"
    return path, json.loads(path.read_text())


def selector(path, lock, ids):
    r = G["r"]
    ch = lock["chosen"]
    outs = r.score_members(
        path.parent / "refit",
        lock["members"],
        G["X"][ids].reshape(-1, 568),
        len(ids),
        G["ens"],
    )
    members = np.asarray(
        [
            r.member_picks(o, ch["tau"], ch["k"], G["veto"], G["core"], G["bagged"])
            for o in outs
        ],
        dtype=np.int32,
    )
    primary = np.asarray(r.ens_picks(members, ch["a"], G["ens"]), dtype=np.int32)
    return (
        primary,
        members,
        np.array([o["votes"] for o in outs]),
        np.array([o["raw_scores"] for o in outs]),
    )


def task(args):
    part, k, seed = args
    fams = G["fams"]
    folds = np.array([fold_of(part, f) for f in fams])
    tr = np.flatnonzero(folds != k)
    te = np.flatnonzero(folds == k)
    labels = ScopedLabels(G["cids"], G["indices"], tr)
    u, gate = labels.read(tr)
    path, ref = reference(part, k, seed)
    primary, members, votes, raw = selector(path, ref, tr)
    vg = G["vg"]
    ss = G["core"].stream_seed
    X = G["X"][tr]
    aug = G["aug"][tr]
    fam = fams[tr]
    f = list(sorted(set(fam)))
    inner_seed = ss(seed, "H311-inner", part, k)
    np.random.default_rng(inner_seed).shuffle(f)
    fm = {v: i % 3 for i, v in enumerate(f)}
    inner = np.array([fm[x] for x in fam])
    out = D / "tasks" / part / f"fold_{k}/root_{seed}"
    out.mkdir(parents=True)
    lock = dict(
        partition=part,
        outer_fold=k,
        root_seed=seed,
        reference_lock_sha256=sha(path),
        reference_config=ref["chosen"],
        train_cids_sha256=array_sha(G["cids"][tr]),
        train_indices=tr.tolist(),
        outer_test_indices=te.tolist(),
        inner_fold_seed=inner_seed,
        arms={},
        outer_test_labels_read_before_lock=0,
        denied_outer_label_accesses=0,
        physical_label_context_reads=len(tr),
        feature_sha256=sha(D / "features.npz"),
        source_sha256={f.name: sha(f) for f in (P / "source").glob("*.py")},
    )
    for arm in ["B", "N"]:
        model_seed = seed if arm == "B" else 311201 + SEEDS.index(seed)
        prob = np.zeros((len(tr), 8))
        streams = []
        for fold in range(3):
            it = np.flatnonzero(inner != fold)
            iv = np.flatnonzero(inner == fold)
            y = (~gate[it]).reshape(-1)
            fitseed = ss(model_seed, "H311", arm, part, k, "inner", fold)
            permseed = ss(model_seed, "H311", arm, part, k, "perm", fold)
            if arm == "N":
                y = np.random.default_rng(permseed).permutation(y)
            mod = vg.fit_hybrid_valguard(
                X[it].reshape(-1, 568), aug[it].reshape(-1, 652), y, fitseed
            )
            prob[iv] = vg.predict_hybrid_valguard(
                mod, X[iv].reshape(-1, 568), aug[iv].reshape(-1, 652), len(iv)
            )
            streams.append(
                dict(
                    inner_fold=fold,
                    fit_seed=fitseed,
                    permutation_seed=permseed if arm == "N" else None,
                    train_families=sorted(set(fam[it])),
                    validation_families=sorted(set(fam[iv])),
                    label_sha256=array_sha(y),
                )
            )
        tau, table = policy.choose_tau(primary, members, votes, raw, prob, u, gate, fam)
        y = (~gate).reshape(-1)
        fitseed = ss(model_seed, "H311", arm, part, k, "outer")
        permseed = ss(model_seed, "H311", arm, part, k, "perm", "outer")
        if arm == "N":
            y = np.random.default_rng(permseed).permutation(y)
        mod = vg.fit_hybrid_valguard(
            X.reshape(-1, 568), aug.reshape(-1, 652), y, fitseed
        )
        gbm = out / (arm + ".txt")
        lr = out / (arm + ".npz")
        hg, hl = vg.save_hybrid_valguard(mod, gbm, lr)
        for fp in [gbm, lr]:
            with fp.open("rb") as f:
                os.fsync(f.fileno())
        lock["arms"][arm] = dict(
            tau=tau,
            table=table,
            gbm_sha256=hg,
            lr_sha256=hl,
            fit_seed=fitseed,
            permutation_seed=permseed if arm == "N" else None,
            label_sha256=array_sha(y),
            inner_streams=streams,
        )
    assert labels.denied == 0 and labels.reads == len(tr)
    lock["outer_test_labels_read_before_lock"] = labels.denied
    save(out / "fit_lock.json", lock)
    print(
        json.dumps(
            dict(
                task=args,
                tau_B=lock["arms"]["B"]["tau"],
                tau_N=lock["arms"]["N"]["tau"],
                status="LOCKED",
            )
        ),
        flush=True,
    )
    return str(out / "fit_lock.json")


def fit():
    prepare()
    tasks = [(p, k, s) for p in PARTS for k in range(4) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        paths = list(ex.map(task, tasks))
    assert len(paths) == 36
    save(
        D / "lock_inventory.json",
        dict(
            locks={str(Path(x).relative_to(D)): sha(x) for x in paths},
            locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )


def score():
    prepare()
    inventory = json.loads((D / "lock_inventory.json").read_text())
    assert len(inventory["locks"]) == 36
    for name, h in inventory["locks"].items():
        assert sha(D / name) == h

    def nofit(*a, **k):
        raise RuntimeError("fit forbidden during score")

    G["vg"].fit_hybrid_valguard = nofit
    import lightgbm, sklearn.linear_model

    lightgbm.LGBMClassifier.fit = nofit
    sklearn.linear_model.LogisticRegression.fit = nofit
    picks = np.full(
        (3, 3, 3, 1219), -1, dtype=np.int32
    )  # partition, seed, A/B/N, context
    for part in PARTS:
        ip = PARTS.index(part)
        for k in range(4):
            for seed in SEEDS:
                ir = SEEDS.index(seed)
                out = D / "tasks" / part / f"fold_{k}/root_{seed}"
                lock = json.loads((out / "fit_lock.json").read_text())
                ids = np.asarray(lock["outer_test_indices"])
                refpath, ref = reference(part, k, seed)
                assert sha(refpath) == lock["reference_lock_sha256"]
                primary, members, votes, raw = selector(refpath, ref, ids)
                picks[ip, ir, 0, ids] = primary
                for ia, arm in enumerate(["B", "N"], 1):
                    a = lock["arms"][arm]
                    assert sha(out / (arm + ".txt")) == a["gbm_sha256"]
                    assert sha(out / (arm + ".npz")) == a["lr_sha256"]
                    mod = G["vg"].load_hybrid_valguard(
                        out / (arm + ".txt"), out / (arm + ".npz")
                    )
                    prob = G["vg"].predict_hybrid_valguard(
                        mod,
                        G["X"][ids].reshape(-1, 568),
                        G["aug"][ids].reshape(-1, 652),
                        len(ids),
                    )
                    picks[ip, ir, ia, ids] = policy.fallback(
                        primary, members, votes, raw, prob, a["tau"]
                    )
    assert (picks >= 0).all()
    with np.load(D / "references/h303/out/picks.npz") as z:
        assert np.array_equal(picks[0, :, 0], z["ENS_VOTE2_NESTED"])
    with np.load(D / "references/h308/out/picks.npz") as z:
        for i, part in enumerate(PARTS[1:], 1):
            assert np.array_equal(picks[i, :, 0], z["ENS_VOTE2_NESTED_" + part])
    # All 36 locks exist before this process reads any outcome.
    labels = ScopedLabels(G["cids"], G["indices"], range(1219))
    u, gate = labels.read(np.arange(1219))
    np.savez(
        D / "scored.npz", picks=picks, u=u, gate=gate, fams=G["fams"], cids=G["cids"]
    )
    save(
        D / "score_receipt.json",
        dict(
            status="PASS",
            locks=36,
            reference_picks_exact=True,
            scored_sha256=sha(D / "scored.npz"),
            physical_label_context_reads=labels.reads,
            outer_test_labels_read_before_lock=0,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    phase = sys.argv[1]
    {"features": features, "fit": fit, "score": score}[phase]()
