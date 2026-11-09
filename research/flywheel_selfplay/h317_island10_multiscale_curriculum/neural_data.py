#!/usr/bin/env python3
"""H317 neural data: TRAIN pools -> candidate vectors and 4-head targets; the featurize/proj16/sem24 parity check.

Candidate vector = [slog(feat412) | proj16 | sem24] with feat412 = pool `program`, proj16 = the H58 TRAIN-prefix extract
(verified equal to `program @ W1 / sqrt(412)`), sem24 = JopeDimeMicro `compute_semantic_struct_sig_24d` (the canonical
function the H58 priors use; Gemini's `common_h252.compute_sem24_numpy` is a different pooled signature and is not used).
Targets (TRAIN pools only):
  u      native utility = -fitness if gate else -2.0 (`common_h317.native_utility`)
  lr     pool `logratio` (= mean_lr), masked unless finite and |lr| < 4.9
  sd     sd_lr = (fitness - 0.9*logratio - 0.0005*complexity) / 0.2 for gated candidates (fitness = 0.8 mean + 0.1 tail
         + 0.1 sd + 0.0005 c and tail = mean + sd), clipped at 0; masked for non-gated; complexity = feat412[411] + 1
  gate   pool gate
`permute_seed` (timing only) permutes the target tuple across the candidates of each context, so no outcome signal exists.

CLI: `python3 neural_data.py parity` (xbabe2-light: 1,219 DSL featurizations + array compares).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402
import multiscale_axial_bridge as MB  # noqa: E402

H58_TRAIN_DIR = C.DATA_ROOT / "neural" / "h58train"
H58_TRAIN_POOL = H58_TRAIN_DIR / "pool_priors_train1219.npz"
M_MAX = 1152

# R882 reference pick (never an input): the per-context TRAIN r882 selection pinned from H189 attempt 2
# (~/jope_plan/ns/results/H189_attempt2/B-costs-fold{0..4}-seed189101-R882.json, copied with sha256 into
# $H317_DATA/neural/r882/; the seed189102/189103 copies are byte-identical). Verified at copy time: TRAIN-only cids,
# 1,219/1,219 coverage with disjoint folds, H189's h50 pools == selected.json pool_sha256 == res-01 pools,
# selected_choice == argmin(costs) for 1,219/1,219, recorded utility bit-equal to the res-01 pool's native utility.
# NOTE: `base_prior_cost` in the H58 extract is the adapter INPUT prior, NOT r882's prediction; it is not used here.
R882_DIR = C.DATA_ROOT / "neural" / "r882"
R882_SOURCE = "/home/ubuntu/jope_plan/ns/results/H189_attempt2"
R882_FILES = {
    "B-costs-fold0-seed189101-R882.json": "416ca4df58691522d0e94c72a6a31e640722f88a3e251bd99ea55a617d5e10ba",
    "B-costs-fold1-seed189101-R882.json": "b63b0a9388fd7d2b7324c69f65087c7cf4569abf1f9940a2db2dd966a3b1d2b1",
    "B-costs-fold2-seed189101-R882.json": "6a96c06eaebac4360250229a6490f3736991e8197b4c360f5a831f3cec91f6a2",
    "B-costs-fold3-seed189101-R882.json": "a2b9b64260a97ab60be6e456908e28082ffcfcb08c855b04e695dd6a64fedefa",
    "B-costs-fold4-seed189101-R882.json": "84be9dc1abab595710dd900303a0cdfe22a4fba6db55b58540689c17743869ef",
}
# PREREG §2.2 TRAIN anchors (H258): family 1.385355 / context 1.423073 at 6 decimals; pinned at full precision,
# tolerance 1e-9 (float64 recomputation from the res-01 pools reproduces them).
R882_ANCHORS = {
    "family": 1.3853554329931828,  # 1.385355
    "context": 1.423073434552299,  # 1.423073
}
R882_ANCHOR_TOL = 1e-9


def load_r882_picks(r882_dir: Optional[Path] = None) -> Dict[str, dict]:
    """cid -> {"choice": int, "utility": float} for the 1,219 TRAIN contexts; every file sha256-pinned, TRAIN-only."""
    d = Path(r882_dir) if r882_dir else R882_DIR
    train = {r["context_sha256"] for r in C.train_rows()}
    out: Dict[str, dict] = {}
    for name, sha in R882_FILES.items():
        p = d / name
        if C.sha256_file(p) != sha:
            raise RuntimeError(f"R882 file sha mismatch: {p}")
        rec = json.loads(p.read_text())
        bad = set(rec) - train
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} non-TRAIN cids")
        for cid, e in rec.items():
            if cid in out:
                raise RuntimeError(f"R882 cid {cid[:12]} in two folds")
            costs = np.asarray(e["costs"], np.float64)
            if int(np.argmin(costs)) != int(e["selected_choice"]):
                raise RuntimeError(f"R882 {cid[:12]}: selected_choice != argmin(costs)")
            out[cid] = {
                "choice": int(e["selected_choice"]),
                "utility": float(e["utility"]),
            }
    if set(out) != train:
        raise RuntimeError(f"R882 coverage {len(out)} != 1,219 TRAIN contexts")
    return out


def r882_anchor_check(meta: dict, tol: float = R882_ANCHOR_TOL) -> dict:
    """Recompute the R882 pick utilities from the TRUE pool utilities (float64) and assert the PREREG anchors."""
    u = {
        c: float(meta["u64"][i, meta["r882_pick"][i]])
        for i, c in enumerate(meta["cids"])
    }
    fam_of = dict(zip(meta["cids"], meta["family_id"]))
    ctx = float(np.mean([u[c] for c in sorted(u)]))
    fam = C.family_equal_mean(u, fam_of)
    rec_ok = all(u[c] == meta["r882_recorded_u"][i] for i, c in enumerate(meta["cids"]))
    res = {
        "context_mean": ctx,
        "family_mean": fam,
        "anchors": R882_ANCHORS,
        "recorded_utility_bit_equal": rec_ok,
        "pass": rec_ok
        and abs(ctx - R882_ANCHORS["context"]) < tol
        and abs(fam - R882_ANCHORS["family"]) < tol,
    }
    if not res["pass"]:
        raise RuntimeError(f"R882 TRAIN regression gate FAILED: {res}")
    return res


def micro_sem24(feat412: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(C.MICRO / "python"))
    from phase12_multimodal_model import compute_semantic_struct_sig_24d

    return np.asarray(
        compute_semantic_struct_sig_24d(np.asarray(feat412, np.float32)), np.float32
    )


def micro_proj16(feat412: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(C.MICRO / "python"))
    import rust_evaluator as ev

    return (
        np.asarray(feat412, np.float64) @ ev.get_w1_sign_matrix() / np.sqrt(412.0)
    ).astype(np.float32)


def load_h58_train_pool() -> Dict[str, np.ndarray]:
    with np.load(H58_TRAIN_POOL, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def targets_from_pool(
    fitness: np.ndarray, logratio: np.ndarray, gate: np.ndarray, feat412: np.ndarray
) -> Dict[str, np.ndarray]:
    fitness = np.asarray(fitness, np.float64)
    logratio = np.asarray(logratio, np.float64)
    gate = np.asarray(gate, bool)
    u = C.native_utility(fitness, gate)
    lr_mask = np.isfinite(logratio) & (np.abs(logratio) < 4.9)
    comp = MB.complexity_from_feat412(feat412)
    sd_raw = (fitness - 0.9 * logratio - 0.0005 * comp) / 0.2
    sd_mask = gate & np.isfinite(sd_raw)
    sd = np.where(sd_mask, np.clip(sd_raw, 0.0, None), 0.0)
    return {
        "u": u,
        "lr": np.where(lr_mask, logratio, 0.0),
        "lr_mask": lr_mask.astype(np.float64),
        "sd": sd,
        "sd_mask": sd_mask.astype(np.float64),
        "gate": gate.astype(np.float64),
        "sd_raw_min_gated": float(sd_raw[gate].min()) if gate.any() else float("nan"),
    }


def build_numpy(
    rows: List[dict], permute_seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Stack every TRAIN context (rows order) into [N, 1152, ...] arrays (float16 candidates, float32 targets)."""
    h58 = load_h58_train_pool()
    pos = {str(c): i for i, c in enumerate(h58["cids"])}
    off = h58["offsets"]
    N = len(rows)
    cand = np.zeros((N, M_MAX, MB.CAND_DIM), np.float16)
    tg = {
        k: np.zeros((N, M_MAX), np.float32)
        for k in ("u", "lr", "lr_mask", "sd", "sd_mask", "gate")
    }
    tg["u"][:] = -1e9
    u64 = np.full(
        (N, M_MAX), np.nan, np.float64
    )  # exact native utilities (float64) for every reported number
    n_valid = np.zeros(N, np.int64)
    proj_maxdiff = 0.0
    sd_min = np.inf
    for i, r in enumerate(rows):
        cid = r["context_sha256"]
        with np.load(C.pool_path(cid), allow_pickle=False) as z:
            prog = z["program"].astype(np.float32)
            fit, lrat, gate = z["fitness"], z["logratio"], z["gate"]
        n = prog.shape[0]
        j = pos[cid]
        a, b = int(off[j]), int(off[j + 1])
        if b - a != n:
            raise RuntimeError(f"{cid}: H58 extract rows {b - a} != pool rows {n}")
        proj = h58["proj16"][a:b].astype(np.float32)
        proj_maxdiff = max(proj_maxdiff, float(np.abs(proj - micro_proj16(prog)).max()))
        cand[i, :n] = MB.candidate_vector(prog, proj, micro_sem24(prog)).astype(
            np.float16
        )
        t = targets_from_pool(fit, lrat, gate, prog)
        sd_min = min(sd_min, t.pop("sd_raw_min_gated"))
        if permute_seed is not None:
            p = np.random.default_rng(
                C.stream_seed("H317-timing-permute", permute_seed, cid)
            ).permutation(n)
            t = {k: v[p] for k, v in t.items()}
        for k in tg:
            tg[k][i, :n] = t[k]
        u64[i, :n] = t["u"]
        n_valid[i] = n
    return {
        "cand": cand,
        "n_valid": n_valid,
        "u64": u64,
        "proj16_max_abs_diff_vs_formula": proj_maxdiff,
        "sd_raw_min_gated": sd_min,
        **tg,
    }


def build_train_tensors(
    device, permute_seed: Optional[int] = None, views_path: Optional[Path] = None
):
    """(TrainTensors on device, meta dict). meta: cids, family_id, fold, n_valid, u64 [N,1152] float64 utilities,
    r882_pick [N] (the pinned TRAIN r882 selection; a reference, never an input), r882_recorded_u [N]."""
    import torch

    import crop_shuffle_synthetic_views as V
    import curriculum_trainer as T

    rows = C.train_rows()
    cids = [r["context_sha256"] for r in rows]
    folds = C.family_folds(rows)
    arr = build_numpy(rows, permute_seed)
    vw = V.load_merged_views(views_path)
    if list(vw["cids"]) != cids:
        raise RuntimeError("views cids order != train_rows order")
    views = vw["feats16"].astype(np.float32)
    views[:, 0] = vw["view0_f32"]
    tt = T.TrainTensors(
        cids=cids,
        cand=torch.as_tensor(arr["cand"]).to(device),
        n_valid=torch.as_tensor(arr["n_valid"]).to(device),
        u=torch.as_tensor(arr["u"]).to(device),
        lr=torch.as_tensor(arr["lr"]).to(device),
        lr_mask=torch.as_tensor(arr["lr_mask"]).to(device),
        sd=torch.as_tensor(arr["sd"]).to(device),
        sd_mask=torch.as_tensor(arr["sd_mask"]).to(device),
        gate=torch.as_tensor(arr["gate"]).to(device),
        views=torch.as_tensor(views).to(device),
    ).finalize()
    meta = {
        "cids": cids,
        "family_id": [r["family_id"] for r in rows],
        "fold": np.array([folds[c] for c in cids], np.int64),
        "u64": arr["u64"],
        "n_valid": arr["n_valid"],
        "u": arr["u"],
        "proj16_max_abs_diff_vs_formula": arr["proj16_max_abs_diff_vs_formula"],
        "sd_raw_min_gated": arr["sd_raw_min_gated"],
        "permuted": permute_seed is not None,
    }
    if permute_seed is None:
        r882 = load_r882_picks()
        meta["r882_pick"] = np.array([r882[c]["choice"] for c in cids], np.int64)
        meta["r882_recorded_u"] = [r882[c]["utility"] for c in cids]
    return tt, meta


def parity_check() -> dict:
    """featurize_dsl_412(champion dsl) == pool program row of that champion; complexity, proj16 and sem24 checks."""
    C.install_guard()
    sys.path.insert(0, str(C.MICRO / "python"))
    import rust_evaluator as ev

    rows = {r["context_sha256"]: r for r in C.train_rows()}
    champs = json.loads((C.RES01 / "champions.json").read_text())
    h58 = load_h58_train_pool()
    pos = {str(c): i for i, c in enumerate(h58["cids"])}
    n = exact = found = comp_ok = proj_exact_h58 = 0
    maxabs = proj_maxabs = proj_h58_maxabs = 0.0
    sem_maxabs = gem_vs_micro = 0.0
    bad: List[str] = []
    for ch in champs:
        cid = ch["context_sha256"]
        if cid not in rows:
            raise RuntimeError("non-TRAIN champion")
        n += 1
        f412, p16 = ev.featurize_dsl_412(ch["dsl"])
        with np.load(C.pool_path(cid), allow_pickle=False) as z:
            pids = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in z["program_sha256"].tolist()
            ]
            if ch["program_sha256"] not in pids:
                bad.append(cid)
                continue
            k = pids.index(ch["program_sha256"])
            row = z["program"][k].astype(np.float32)
        found += 1
        exact += int(np.array_equal(f412.astype(np.float32), row))
        maxabs = max(maxabs, float(np.abs(f412.astype(np.float64) - row).max()))
        comp_ok += int(ch["dsl"].count("(") == int(MB.complexity_from_feat412(f412)))
        proj_maxabs = max(
            proj_maxabs,
            float(np.abs(p16.astype(np.float32) - micro_proj16(row[None])[0]).max()),
        )
        j = pos[cid]
        h = h58["proj16"][int(h58["offsets"][j]) + k]
        proj_exact_h58 += int(np.array_equal(h, micro_proj16(row[None])[0]))
        proj_h58_maxabs = max(
            proj_h58_maxabs, float(np.abs(h.astype(np.float64) - p16).max())
        )
        s_f = micro_sem24(f412[None])[0]
        s_r = micro_sem24(row[None])[0]
        sem_maxabs = max(sem_maxabs, float(np.abs(s_f - s_r).max()))
        gem = _gemini_sem24(row[None])[0]
        gem_vs_micro = max(gem_vs_micro, float(np.abs(gem - s_r).max()))
    rep = {
        "schema": "jope.h317.featurize-parity/v1",
        "champions": n,
        "champion_row_found_in_pool": found,
        "missing_in_pool": bad[:5],
        "feat412_exact": exact,
        "feat412_max_abs_diff": maxabs,
        "complexity_eq_steps_plus_1": comp_ok,
        "proj16_featurize_vs_formula_on_pool_row_max_abs_diff": proj_maxabs,
        "proj16_h58_equal_formula_float32": proj_exact_h58,
        "proj16_h58_vs_featurize_float64_max_abs_diff": proj_h58_maxabs,
        "sem24_from_featurize_vs_pool_row_max_abs_diff": sem_maxabs,
        "gemini_compute_sem24_numpy_vs_micro_max_abs_diff": gem_vs_micro,
        "guard": C.guard_report(),
    }
    C.write_json(C.DATA_ROOT / "neural" / "receipts" / "featurize_parity.json", rep)
    return rep


def _gemini_sem24(prog_412: np.ndarray) -> np.ndarray:
    """Copied from origin/main research/flywheel_selfplay/h256_pre_condense_residual_flywheel/common_h252.py
    (`compute_sem24_numpy`) only to document that it is NOT the canonical Micro signature."""
    x = np.asarray(prog_412, dtype=np.float32)
    flat = x.reshape(-1, 412)
    padded = np.pad(flat, ((0, 0), (0, 20)), mode="constant").reshape(-1, 18, 24)
    sig = padded.mean(axis=1) + 0.5 * padded.max(axis=1)
    norms = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
    return (sig / norms).astype(np.float32)


if __name__ == "__main__":
    if sys.argv[1:] == ["parity"]:
        print(json.dumps(parity_check(), indent=1))
    else:
        raise SystemExit("usage: neural_data.py parity")
