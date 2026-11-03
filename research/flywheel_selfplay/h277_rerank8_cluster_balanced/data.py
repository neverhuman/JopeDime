from pathlib import Path
import os
import numpy as np

POOLS = Path("/home/ubuntu/jope-data/h50/input")
CACHE = Path("/home/ubuntu/jope-data/h58/cache")
NATIVE91 = 1.1377941898164525
TOP4_CTX, TOP4_FAM = 1.2403601682250796, 1.2519997168684853


def audit_hook(event, args):
    """Refuse any open of a sealed, holdout or meta-test path."""
    if event == "open" and isinstance(args[0], (str, bytes)):
        s = os.fsdecode(args[0]).lower()
        if any(
            x in s
            for x in (
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
                "qualification.lock",
            )
        ):
            raise RuntimeError("protected path: " + s)


def dope_of(path):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    value = str(ckpt.get("dope_sha256", ""))
    del ckpt
    return value


def native91(val, costs, ref):
    ref_map = {
        c["cid"]: float(u)
        for c, u in zip(
            ref["evaluation_contract"]["contexts"], ref["utility"]["per_context"]
        )
    }
    single, top4, exact = [], [], 0
    for r in val:
        cid = r["context_sha256"]
        with np.load(POOLS / "pools" / f"{cid}.npz", allow_pickle=False) as z:
            fit, gate = z["fitness"].astype(np.float64), z["gate"].astype(bool)
        u = np.where(gate, -fit, -2.0)
        rc = costs[cid]
        u0 = float(u[int(np.argmin(rc))])
        single.append(u0)
        exact += u0 == ref_map[cid]
        t4 = np.argsort(rc, kind="mergesort")[:4]
        top4.append(float(u[int(t4[int(np.argmin(np.where(gate[t4], fit[t4], 2.0)))])]))
    fams = np.array([r["family_id"] for r in val])
    t4a = np.asarray(top4)
    fam_top4 = float(np.mean([t4a[fams == f].mean() for f in sorted(set(fams))]))
    if exact != 91 or float(np.mean(single)) != NATIVE91:
        raise RuntimeError("Native91 regression")
    if abs(float(np.mean(top4)) - TOP4_CTX) > 1e-12 or abs(fam_top4 - TOP4_FAM) > 1e-12:
        raise RuntimeError("TOP4_PICK@r882 regression")
    return dict(
        native91=float(np.mean(single)),
        matches=exact,
        top4_ctx=float(np.mean(top4)),
        top4_fam=fam_top4,
    )


def priors_for(rows):
    ids = {r["context_sha256"] for r in rows}
    aprior, proj16, ctxf = {}, {}, {}
    with np.load(CACHE / "pool_priors.npz", allow_pickle=False) as zp:
        cids, offsets = [str(c) for c in zp["cids"]], zp["offsets"]
        base, proj = zp["base_prior_cost"], zp["proj16"]
        for i, cid in enumerate(cids):
            if cid in ids:
                lo, hi = int(offsets[i]), int(offsets[i + 1])
                aprior[cid] = base[lo:hi].astype(np.float32).copy()
                proj16[cid] = proj[lo:hi].astype(np.float32).copy()
    with np.load(CACHE / "context_features.npz", allow_pickle=False) as zc:
        lm, dope = zc["lm48_z"], zc["dope629"]
        for i, cid in enumerate(str(c) for c in zc["cids"]):
            if cid in ids:
                ctxf[cid] = {
                    "lm48_z": lm[i].astype(np.float32).copy(),
                    "dope629": dope[i].astype(np.float32).copy(),
                }
    return aprior, proj16, ctxf
