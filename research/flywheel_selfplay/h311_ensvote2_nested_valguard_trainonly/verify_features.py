"""Compare the label-free rebuild to H295's frozen TRAIN feature matrix only."""

from pathlib import Path
import json, hashlib
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h311-astra")


def main():
    with np.load(D / "features.npz", allow_pickle=False) as z:
        X = z["X"]
        cids = z["cids"]
    with np.load(
        "/home/ubuntu/jope-data/h291/cache/dl_train_member_oof.npz", allow_pickle=False
    ) as z:
        old = z["X"]
        oldcids = list(z["cids"])
        reordered = old[[oldcids.index(c) for c in cids]]
    assert np.array_equal(X, reordered)
    rec = dict(
        status="PASS",
        feature_arrays_exact=True,
        contexts=len(cids),
        shape=list(X.shape),
        X_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        max_abs_diff=float(abs(X - reordered).max()),
        members_read=["X", "cids"],
        utility_or_validity_members_read=0,
    )
    (P / "source/feature_regression.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
