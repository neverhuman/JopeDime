"""Replay historical locked picks with outcomes only; never decode program/cache features."""

from pathlib import Path
import sys, json, hashlib
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h312-astra")
V = P / "vendor"
sys.path.insert(0, str(V / "h310_ensvote2_dev91_vs_r3175"))
import evaluate_h310 as ev
import run_h310 as h


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def idhash(ids):
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode()
    ).hexdigest()


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    assert json.loads((P / "source/input_receipt.json").read_text())["status"] == "PASS"
    h.check_pins()
    ev.install_score_guard()
    selpath = Path("/home/ubuntu/jope-data/h50/input/selected.json")
    sel = json.loads(selpath.read_text())
    sys.path.insert(0, str(V / "h302_ensvote2_trainseed_nested_cv"))
    import run_h302 as r

    assert sha(selpath) == r.SELECTED_JSON_SHA256
    train = {x["family_id"] for x in sel["train"]}
    dev = {x["family_id"] for x in sel["dev"]}
    overlap = sorted(train & dev)
    assert len(train) == 205 and len(dev) == 19
    audit = dict(
        train_families=len(train),
        dev_families=len(dev),
        overlap_count=len(overlap),
        overlap_ids=overlap,
        overlap_ids_sha256=idhash(overlap),
        train_ids_sha256=idhash(train),
        dev_ids_sha256=idhash(dev),
        ens_member_training_closure="Each full refit ranker/veto sees all205 TRAIN families; bag samples are subsets of that closure. Reconstructed from frozen H310/H303/H302 code; historical member locks do not list training family IDs.",
        ens_member_closure_overlap_count=len(overlap),
        ens_member_closure_overlap_ids_sha256=idhash(overlap),
        metadata_only=True,
        selected_sha256=sha(selpath),
    )
    (P / "source/overlap_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    ref = D / "references"
    m = json.loads((ref / "metrics_full.json").read_text())
    lockrec = json.loads((ref / "lock_receipt.json").read_text())
    locks = {}
    for root in (30001, 30002, 30003):
        paths = list((ref / "archive").rglob(f"root_{root}/fit_lock.json"))
        paths = [x for x in paths if "fold_4" in x.parts]
        assert len(paths) == 1
        assert sha(paths[0]) == lockrec["locks"][str(root)]["fit_lock_sha256"]
        locks[str(root)] = str(paths[0])
    with np.load(ref / "picks.npz", allow_pickle=False) as z:
        picks = z["r3175_pool"]
    costs = ref / "gpu-attempt-001-r3175-gpu-costs.npz"
    assert (
        sha(costs) == "d99363f62c36463e6fe1ca825a87332198e6a817021bc208c6379b2d2c330bab"
    )
    storedpath = h.R3175_STORED
    assert (
        sha(storedpath)
        == "7565576d267f83400fee14e369e461d0a177fee097329b41c093420ceb0b6bbc"
    )
    stored = json.loads(storedpath.read_text())["dev_utility"]
    cids = [x["context_sha256"] for x in sel["dev"]]
    fams = [x["family_id"] for x in sel["dev"]]
    U = np.zeros((3, 91, 2))
    I = np.zeros_like(U, dtype=bool)
    pool_u = []
    pool_inv = []
    inputhash = {str(storedpath): sha(storedpath), str(costs): sha(costs)}
    with np.load(costs, allow_pickle=False) as zc:
        for i, c in enumerate(cids):
            path = r.POOLS / "pools" / f"{c}.npz"
            # Only outcome members; no program/cache feature decoding in historical replay.
            with np.load(path, allow_pickle=False) as z:
                fit = z["fitness"].astype(np.float64)
                gate = z["gate"].astype(np.bool_)
            order = np.argsort(
                zc[c].astype(np.float32).astype(np.float64), kind="mergesort"
            )[:8]
            ut = np.where(gate, -fit, -2.0)[order]
            invalid = (~gate[order]) | (ut <= -2 + 1e-9)
            pool_u.append(ut)
            pool_inv.append(invalid)
            for s in range(3):
                U[s, i] = [ut[0], ut[picks[s, i]]]
                I[s, i] = [invalid[0], invalid[picks[s, i]]]
    d = ev.family_d(U, fams)
    gains = d.mean(axis=1)
    expected = m["primary_vs_r3175_slot0"]["summary"]["per_seed_family_mean_gain"]
    rec = dict(
        status="PASS"
        if np.array_equal(gains, np.array(expected))
        and float(U[0, :, 0].mean()) == h.R3175_MEAN
        and np.array_equal(U[0, :, 0], stored["per_context"])
        else "REGRESSION_FAIL",
        per_seed_gain=gains.tolist(),
        expected_full_precision=expected,
        rounded6=[round(float(x), 6) for x in gains],
        baseline_mean=float(U[0, :, 0].mean()),
        baseline_per_context_exact=bool(
            np.array_equal(U[0, :, 0], stored["per_context"])
        ),
        h310_locks=locks,
        historical_feature_rows_decoded=0,
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        input_hashes=inputhash,
    )
    (P / "source/regression_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
    assert rec["status"] == "PASS"
    (P / "source/regression_gate.json").write_text(
        json.dumps(
            dict(
                status="PASS", receipt_sha256=sha(P / "source/regression_receipt.json")
            )
        )
        + "\n"
    )
    out = D / "out"
    out.mkdir(exist_ok=True)
    np.savez_compressed(
        out / "historical.npz",
        u=U,
        invalid=I,
        picks=picks,
        families=np.array(fams),
        cids=np.array(cids),
        family_d=d,
        pool_utility=np.array(pool_u),
        pool_invalid=np.array(pool_inv),
    )


if __name__ == "__main__":
    main()
