"""Foundation-expert dataset encoder (V6 Lane A).

Encodes a dataset into a fixed vector using the FROZEN TabPFN-v3 regressor's
per-row 512-d hidden states (the "Starlight" expert from the chimera stack,
pretrained on synthetic priors — no quant-dataset leakage). Only train.csv is
ever read; context/query rows are disjoint subsets of train.csv.

Pooling mirrors the Hyperion signature recipe (vs.py:dataset_signature_from_vs):
per-row hiddens -> concat[mean, std, p10, p90] over rows -> 2048-d raw vector.
Compression to the corpus dim happens downstream (Lane A2).

Sidecars are written to a central cache dir (NOT into the quant tree):
    <out_dir>/<dataset_hash>.npy          float32 (2048,)
    <out_dir>/encode_log.jsonl            one line per dataset (provenance)
"""

from __future__ import annotations

import json
import site
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..prime_config import load_config

FUSION_PY = str(load_config().foundation_fusion_python)
RAW_DIM = 2048  # 512 hidden * [mean, std, p10, p90]
ENCODER_VERSION = "foundation_enc.v1.tabpfn"
CTX_ROWS = 768
QUERY_ROWS = 256
SEED = 20260707


def load_expert(device: str = "cuda"):
    try:
        import safetensors  # noqa: F401, PLC0415
    except ModuleNotFoundError:
        # The current frozen xbabe image exposes safetensors in the user site,
        # while Prime's older virtualenv predates the explicit dependency.
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.append(user_site)
        import safetensors  # noqa: F401, PLC0415
    if FUSION_PY not in sys.path:
        sys.path.insert(0, FUSION_PY)
    # The checked-in Chimera tree imports its frozen regressor as
    # ``python.elo_finetune`` from the sibling fable_god package.
    fable_root = str(Path(FUSION_PY).parents[1])
    if fable_root not in sys.path:
        sys.path.insert(0, fable_root)
    from fusion.experts import TabPFNv3Reg  # noqa: PLC0415

    return TabPFNv3Reg(device=device)


def _load_train_matrix(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Headerless train.csv, target = last column (chimera data.py contract)."""
    raw = np.genfromtxt(dataset_dir / "train.csv", delimiter=",", dtype="float64")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    X, y = raw[:, :-1], raw[:, -1]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = X.min(0, keepdims=True), X.max(0, keepdims=True)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return ((X - lo) / span).astype("float32"), y.astype("float32")


def _split_ctx_query(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = rng.permutation(n)
    n_ctx = min(CTX_ROWS, max(8, (3 * n) // 4))
    n_q = min(QUERY_ROWS, n - n_ctx)
    if n_q < 4:  # tiny dataset: allow query overlap with context tail
        return idx[:n_ctx], idx[max(0, n - 8) :]
    return idx[:n_ctx], idx[n_ctx : n_ctx + n_q]


def encode_dataset(expert, dataset_dir: Path) -> np.ndarray:
    X, y = _load_train_matrix(Path(dataset_dir))
    rng = np.random.default_rng(SEED)
    ctx, qry = _split_ctx_query(X.shape[0], rng)
    ds = SimpleNamespace(
        Xtr=X[ctx],
        ytr=y[ctx],
        Xte=X[qry],
        n_train=len(ctx),
    )
    out = expert.infer(ds)
    hid = out.hidden.numpy()  # (M, 512)
    vec = np.concatenate(
        [
            hid.mean(0),
            hid.std(0),
            np.percentile(hid, 10, axis=0),
            np.percentile(hid, 90, axis=0),
        ]
    ).astype("float32")
    assert vec.shape == (RAW_DIM,), vec.shape
    return vec


def encode_many(
    hashes: list[str],
    root: Path,
    out_dir: Path,
    device: str = "cuda",
    log=print,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending_hashes = [h for h in hashes if not (out_dir / f"{h}.npy").exists()]
    log(f"[foundation] {len(hashes)} requested, {len(pending_hashes)} to encode")
    if not pending_hashes:
        return {"encoded": 0, "skipped": len(hashes), "failed": 0}
    expert = load_expert(device)
    ok = failed = 0
    t0 = time.time()
    with open(out_dir / "encode_log.jsonl", "a") as fh:
        for i, h in enumerate(pending_hashes):
            t = time.time()
            try:
                vec = encode_dataset(expert, Path(root) / h)
                np.save(out_dir / f"{h}.npy", vec)
                ok += 1
                fh.write(
                    json.dumps(
                        {
                            "hash": h,
                            "secs": round(time.time() - t, 3),
                            "encoder": ENCODER_VERSION,
                        }
                    )
                    + "\n"
                )
                fh.flush()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                fh.write(json.dumps({"hash": h, "error": str(exc)[:200]}) + "\n")
                fh.flush()
            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t0)
                log(f"[foundation] {i + 1}/{len(pending_hashes)} ({rate:.2f} ds/s, {failed} failed)")
    return {"encoded": ok, "skipped": len(hashes) - len(pending_hashes), "failed": failed}
