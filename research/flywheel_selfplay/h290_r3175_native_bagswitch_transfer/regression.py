"""Required CPU baseline and one exact H280 cell, before H290 fitting."""

from dependencies import ROOT, load_train, veto
import json
import hashlib
import numpy as np

rows, packets, families = load_train()
with np.load(ROOT / "training/regression_val_packets.npz") as z:
    assert z["X"].shape == (91, 8, 568)
    val = float(z["utility_k"][:, 0].mean())
assert val == 1.1366829009431036, val
ref = json.loads((ROOT / "training/h280-cell.json").read_text())
folds = {int(k): v for k, v in ref["folds"].items()}
assert not set(folds[0]).intersection(c for f in range(1, 5) for c in folds[f])
rv = veto.run_cell_veto(280101, 0, folds, packets, families, 8, "KM276140-0")
assert rv["receipt"] == ref["policies"]["A"]["config"]
assert list(rv["assess_picks"]) == ref["policies"]["A"]["picks"]
for key, expected in [("model", "model_sha256"), ("gate", "gate_sha256")]:
    assert (
        hashlib.sha256(rv[key].encode()).hexdigest() == ref["policies"]["A"][expected]
    ), key
receipt = dict(
    status="PASS",
    cpu_dlval91=val,
    cell="KM276140_280101_0",
    selected_parameters_exact=True,
    picks_exact=True,
    ranker_and_gate_serialization_sha_exact=True,
    contexts=1219,
    families=205,
    features=568,
    neural_forwards=0,
    neural_updates=0,
)
(ROOT / "regression.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
