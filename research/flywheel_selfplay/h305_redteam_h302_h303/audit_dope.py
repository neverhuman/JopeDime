"""Independently hash frozen checkpoint DOPE tensors without constructing a model."""

from pathlib import Path
import hashlib
import json
import torch
from independent_replay import digest

NORMALIZERS = (
    "step2_lm_mean_1x256",
    "step2_lm_std_1x256",
    "step2_svd_basis_256x48",
    "lm80_mu",
    "lm80_sd",
    "mu_lm48",
    "sd_lm48",
    "mu_bb20",
    "sd_bb20",
    "mu_dm24",
    "sd_dm24",
)
EXPECTED = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def tensor_hash(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    assert a.dtype.byteorder != ">"
    h = hashlib.sha256(f"{a.dtype.str}:{list(a.shape)}:".encode())
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def inspect(path):
    before = digest(path)
    assert before == "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    names = sorted(k for k in state if k.startswith("dope_model."))
    # The pinned state has weight/bias tensors and the router small_embed parameter.
    # Recomputed composite identity checks this partition against the historical hash.
    assert len(names) == 56 and all(
        n.endswith((".weight", ".bias", ".small_embed")) for n in names
    )
    ph = hashlib.sha256()
    for name in names:
        ph.update(f"{name}={tensor_hash(state[name])}\n".encode())
    bh = hashlib.sha256()
    for name in NORMALIZERS:
        bh.update(f"normalizer.{name}={tensor_hash(state[name])}\n".encode())
    composite = hashlib.sha256(
        f"params:{ph.hexdigest()}|buffers:{bh.hexdigest()}".encode()
    ).hexdigest()
    after = digest(path)
    assert composite == EXPECTED == checkpoint["dope_sha256"] and before == after
    return dict(
        status="PASS",
        checkpoint=str(path),
        checkpoint_sha256_before=before,
        checkpoint_sha256_after=after,
        recomputed_composite_sha256=composite,
        recorded_composite_sha256=checkpoint["dope_sha256"],
        parameter_tensors=len(names),
        normalizer_tensors=len(NORMALIZERS),
        interpretation="Immutable non-live r882 source copy verified; H302/H303 consume cached features and cannot update these tensors. This does not audit unrelated live state.",
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = inspect(Path("/home/ubuntu/jope-data/h276/inputs/round-882.pt"))
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
