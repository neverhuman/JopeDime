"""One-shot r3175 dev91 score after all fresh fit locks; frozen H310 feature path."""

from pathlib import Path
import sys, os, json, hashlib, datetime
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h312-astra")
V = P / "vendor"
for slug in [
    "h310_ensvote2_dev91_vs_r3175",
    "h303_ensvote2_fully_nested_hparams",
    "h302_ensvote2_trainseed_nested_cv",
]:
    sys.path.insert(0, str(V / slug))
import run_h310 as h
import evaluate_h310 as ev
import run_h302 as r302
import run_h303 as r303


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def models(lock):
    for m in lock["members"]:
        yield m["ranker"]
        yield m["veto"]
        yield from m["bags"]


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    rec = json.loads((P / "source/lock_receipt.json").read_text())
    assert rec["status"] == "PASS" and rec["dev91_rows_read_before_lock"] == 0
    reviewed = []
    for v in rec["receipts"]:
        root = v["root_seed"]
        lp = D / "lock/fold_4" / f"root_{root}" / "fit_lock.json"
        assert sha(lp) == v["fit_lock_sha256"]
        lk = json.loads(lp.read_text())
        assert (
            v["counters"]["worker_dev_guard_denials"] == 0
            and v["counters"]["worker_forbidden_packet_denials"] == 0
        )
        count = 0
        for m in models(lk):
            fp = lp.parent / "refit" / m["file"]
            assert sha(fp) == m["sha256"]
            count += 1
        reviewed.append(dict(root=root, lock_sha256=sha(lp), model_count=count))
    review = dict(
        status="PASS",
        locks=reviewed,
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    with (P / "source/lock_review.json").open("x") as f:
        json.dump(review, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    # The exclusive marker prevents accidental second scoring, including after a partial failure.
    with (D / "score_started.json").open("x") as f:
        json.dump(review, f)
        f.flush()
        os.fsync(f.fileno())
    h.check_pins()
    ev.install_score_guard()
    core, topk, veto, bagged, ens = r302.load_pinned_h287_modules(D / "score_mods")
    core.INPUT_POOLS_DIR = r302.POOLS
    raw = (r302.POOLS / "selected.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == r302.SELECTED_JSON_SHA256
    dev = json.loads(raw)["dev"]
    cids = [r["context_sha256"] for r in dev]
    fams = [r["family_id"] for r in dev]
    ids = set(cids)
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(ev.CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        pc, off, base, proj = (
            [str(c) for c in zp["cids"]],
            zp["offsets"],
            zp["base_prior_cost"],
            zp["proj16"],
        )
        for i, c in enumerate(pc):
            if c in ids:
                aprior[c] = (
                    base[int(off[i]) : int(off[i + 1])].astype(np.float32).copy()
                )
                proj16[c] = (
                    proj[int(off[i]) : int(off[i + 1])].astype(np.float32).copy()
                )
    with np.load(ev.CACHE / "context_features.npz", allow_pickle=False) as zc:
        lm, dp = zc["lm48_z"], zc["dope629"]
        for i, c in enumerate(str(x) for x in zc["cids"]):
            if c in ids:
                ctxf[c] = {
                    "lm48_z": lm[i].astype(np.float32).copy(),
                    "dope629": dp[i].astype(np.float32).copy(),
                }
    costpath = D / "references/gpu-attempt-001-r3175-gpu-costs.npz"
    assert (
        sha(costpath)
        == "d99363f62c36463e6fe1ca825a87332198e6a817021bc208c6379b2d2c330bab"
    )
    with np.load(costpath, allow_pickle=False) as z:
        costs = {c: z[c].astype(np.float32) for c in cids}
    packets = topk.load_packets(dev, costs, aprior, proj16, ctxf, ens.K)
    with np.load(D / "out/historical.npz", allow_pickle=False) as z:
        old_u = z["u"]
        old_i = z["invalid"]
        old_p = z["picks"]
        assert list(z["cids"]) == cids
        assert np.array_equal(
            np.array([packets[c]["utility_k"] for c in cids]), z["pool_utility"]
        )
    X = np.concatenate([packets[c]["X"] for c in cids])
    picks = np.zeros((3, 91), dtype=np.int32)
    U = np.zeros((3, 91, 2))
    I = np.zeros_like(U, dtype=bool)
    for s, root in enumerate((30004, 30005, 30006)):
        lp = D / "lock/fold_4" / f"root_{root}" / "fit_lock.json"
        lk = json.loads(lp.read_text())
        ch = lk["chosen"]
        outs = r303.score_members(lp.parent / "refit", lk["members"], X, 91, ens)
        ml = [
            r303.member_picks(o, ch["tau"], ch["k"], veto, core, bagged) for o in outs
        ]
        picks[s] = r303.ens_picks(ml, ch["a"], ens)
        for i, c in enumerate(cids):
            j = int(picks[s, i])
            U[s, i] = [
                float(packets[c]["utility_k"][0]),
                float(packets[c]["utility_k"][j]),
            ]
            I[s, i] = [ev.invalid(packets[c], 0), ev.invalid(packets[c], j)]
    assert float(U[0, :, 0].mean()) == h.R3175_MEAN
    np.savez_compressed(
        D / "out/all_seeds.npz",
        u=np.concatenate([old_u, U]),
        invalid=np.concatenate([old_i, I]),
        picks=np.concatenate([old_p, picks]),
        families=np.array(fams),
        cids=np.array(cids),
        roots=np.arange(30001, 30007),
    )
    receipt = dict(
        status="PASS",
        fresh_seeds=[30004, 30005, 30006],
        score_calls_per_fresh_seed=1,
        baseline_mean=float(U[0, :, 0].mean()),
        features_shape=list(X.shape),
        feature_matrix_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        selected_sha256=r302.SELECTED_JSON_SHA256,
        feature_cache_sha256={
            str(f): sha(f)
            for f in [ev.CACHE / "pool_priors.npz", ev.CACHE / "context_features.npz"]
        },
        source_sha256=sha(__file__),
        output_sha256=sha(D / "out/all_seeds.npz"),
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    (P / "source/score_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
