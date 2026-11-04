"""Mandatory fresh H308 P0 regression and exact H295 ENS_VOTE2 reference replay."""

from pathlib import Path
import sys, json, hashlib, os, datetime
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h311-astra")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    pins = json.loads((P / "source/dependency_pins.json").read_text())
    for n, h in pins["vendor"].items():
        assert sha(P / "vendor" / n) == h, n
    sys.path.insert(0, str(P / "vendor/h308_ensvote2_trainonly_repartition"))
    import run_h308 as h

    h.DATA = D / "regression"
    h.REF_LOCK = D / "references/h303/scratch/fold_0/root_30001/fit_lock.json"
    h.REF_PICKS = D / "references/h303/out/picks.npz"
    r, ev = h.load_h303()
    r.SCRATCH = h.DATA / "setup"
    h.prepare(r)
    ok = h.regress(r)
    rec = json.loads((h.DATA / "regression_receipt.json").read_text())
    if not ok:
        (P / "source/regression_receipt.json").write_text(
            json.dumps(dict(status="REGRESSION_FAIL", h308=rec), indent=2) + "\n"
        )
        raise RuntimeError("H308 regression failed")
    sys.path.insert(0, str(P / "vendor/h295_label_free_validity_guard"))
    import valguard

    cache = Path("/home/ubuntu/jope-data/h291/cache/dl_train_member_oof.npz")
    panel = Path(
        "/home/ubuntu/jope_plan/evidence/H287/argon-t0005/runs/attempt-001/inner_cv_panel.npz"
    )
    metric = Path(
        "/home/ubuntu/jope_plan/evidence/H287/argon-t0005/capsule/research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/metrics.json"
    )
    ref = valguard.verify_h287_ens_vote2_regression(cache, panel, metric)
    assert ref["per_seed_gain_round6"] == [0.053903, 0.059932, 0.062204] and ref[
        "invalid_by_seed"
    ] == [14, 13, 14]
    with np.load(D / "references/h295/outer_cv_panels.npz", allow_pickle=False) as z:
        arms = list(z["arms"])
        old = z["u_panel"][:, :, arms.index("ENS_VOTE2")]
    with np.load(panel, allow_pickle=False) as z:
        assert np.array_equal(old, z["panel"][:, :, 5])
    result = dict(
        status="PASS",
        h308=rec,
        h295=ref,
        h295_archived_reference_vectors_exact=True,
        reference_files={str(f): sha(f) for f in [cache, panel, metric]},
        finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__),
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
    )
    (P / "source/regression_receipt.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
