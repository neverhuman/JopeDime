"""Hash-pinned unchanged H279 science; neural execution forbidden in H290."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent.parent
CLOSURE = ROOT / "code-closure"
for name, item in json.loads((ROOT / "discovery.json").read_text())[
    "source_pins"
].items():
    assert (
        hashlib.sha256((CLOSURE / name).read_bytes()).hexdigest() == item["sha256"]
    ), name
sys.path.insert(0, str(CLOSURE))
import h265_core as core  # noqa: E402
import topk  # noqa: E402
import veto  # noqa: E402
import bagged  # noqa: E402
import freeze_bagged  # noqa: E402

__all__ = ["ROOT", "core", "topk", "veto", "bagged", "freeze_bagged", "load_train"]


def forbidden(*args, **kwargs):
    raise RuntimeError("H290 forbids neural forwards/checkpoint loads")


core.torch.load = forbidden
core.torch.nn.Module.__call__ = forbidden
core.torch.set_num_threads(1)
core.torch.set_num_interop_threads(1)


def load_train():
    import numpy as np

    selected = json.loads((ROOT / "training/selected.json").read_text())
    rows = selected["train"]
    families = {r["context_sha256"]: r["family_id"] for r in rows}
    assert len(rows) == 1219 and len(set(families.values())) == 205
    assert all(
        r["dl_split"] == "DL-Train"
        and r["source_split"] in ["train", "inner"]
        and not r.get("has_meta_test")
        for r in rows
    )
    with np.load(ROOT / "training/train_packets.npz") as z:
        ids = list(map(str, z["cids"]))
        assert ids == [r["context_sha256"] for r in rows]
        arrays = {k: z[k] for k in ["X", "utility_k", "gate_k", "rel_k"]}
    with np.load(ROOT / "training/train_features.npz") as z:
        assert np.array_equal(z["X"], arrays["X"])
    packets = {
        c: {
            **{k: v[i] for k, v in arrays.items()},
            "r882_u": float(arrays["utility_k"][i, 0]),
        }
        for i, c in enumerate(ids)
    }
    assert arrays["X"].shape == (1219, 8, 568)
    return rows, packets, families
