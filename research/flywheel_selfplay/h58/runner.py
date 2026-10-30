#!/usr/bin/env python3
"""
H58 Flywheel Self-Play Multi-Arm Wave Runner (t550)
===================================================
Implements four cumulative arms (each an isolated copy of the flywheel loop):
  - Arm A (CONTROL): Current flywheel harness (revealed 0..3 unmasked in loss & argmin,
    deterministic order[:48] in slots 0..47, zeroed ctx_feat/base_prior_cost/sos_prior,
    constant scale=0.4208, parent_delta_net trainable [29,211 params], HWM-only admission,
    smooth_l1(beta=1.0) + 0.1*kl(tau=0.2) loss with failure target 2.0).
  - Arm B (FIX1): Fix Bug 1 (fixed unsorted seed anchors 0..3, permuted unrevealed slots 4..191,
    mask 0..3 from training loss and evaluation argmin) + Fix Bug 3 (real 584D ctx_feat,
    real prior10 base_prior_cost, real 2D sos_prior, per-context scale) + Fix Bug 4
    (freeze dead parent_delta_net -> 23,546 active trainable params).
  - Arm C (FIX1+2): Arm B + Fix Bug 2 (in every self-play round's live GP proposal/evaluation step
    and round-0 historical journal admission, admit ALL evaluated GP candidates on training folds —
    including non-HWM evaluations, model-proposed false positives, and gate failures — as hard
    negatives in pools[c] and reserve hard-negative slots in training slates).
  - Arm D (FIX1+2+3): Arm C + Fix Rank 3 (shift-invariant listwise top-focused loss:
    dual-temperature Plackett-Luce / ListNet cross-entropy + strict pairwise gate-failure &
    top-choice hinge margins on unrevealed candidates 4..191).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "research/flywheel_selfplay"))
sys.path.insert(0, str(ROOT / "python/src"))
import run_fused_closed_loop as base  # noqa: E402
from jope.north_star.elo_tracker import (  # noqa: E402
    DatasetEvalRecord,
    compute_family_grouped_bradley_terry_elo,
)

MANIFEST_FILE_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
MANIFEST_ID = "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3"
SELECTED_FILE_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
CHAMPIONS_FILE_SHA = "a444047ffb937dc8d69012cbdbf0891faf3e7d743b127e7cac3baed114cf141d"
FUSED_SOURCE_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
WORKER_BIN_SHA = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
STAGE1_STAGE2_PT_SHA = "136b2381c312fd4bf267249fe0543aa44f5b780a05d57f974d43dd4277588b6d"
DOPE_MULTITASK_PT_SHA = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
PROG_FEAT_NAMES_SHA = "c4e4373d1bc7e99bc38e44eb4acda22d0f9ecc9e332fa8cb9c6a0b463fe37962"
RUST_EVALUATOR_PY_SHA = "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900"
MODEL_PY_SHA = "0aa76360a7e5e691e420bd5a88a05b343b5b83edcc645e5f50fb33a3d5756f5a"
PHASE12_MM_PY_SHA = "7f5d30280de5cd47e8af5aa6787e5cb2bff321d984f9cae7d261eabd0a9c0d7a"
MICRO_ROOT = Path("/home/ubuntu/JopeDimeMicro")
FOLD_SEED = 4900
DEFAULT_SEEDS = (12, 13, 14)
ARMS = ("A", "B", "C", "D")
TRAINABLE_PARAMS_A = 29_211
TRAINABLE_PARAMS_BCD = 23_546
BOOTSTRAP_DRAWS = 10_000
GATE_MARGIN_LCB95 = 0.001


def verify_pinned_micro_assets() -> dict[str, str]:
    pins = {
        "stage1_stage2_pt_sha256": (
            MICRO_ROOT / "assets/projections/frozen_stage1_stage2_inference_params_v3.pt",
            STAGE1_STAGE2_PT_SHA,
        ),
        "dope_multitask_pt_sha256": (
            MICRO_ROOT / "assets/models/dope_mlc_v3_multitask_best.pt",
            DOPE_MULTITASK_PT_SHA,
        ),
        "prog_feat_names_sha256": (
            MICRO_ROOT / "assets/projections/program-feature-names.json",
            PROG_FEAT_NAMES_SHA,
        ),
        "rust_evaluator_py_sha256": (
            MICRO_ROOT / "python/rust_evaluator.py",
            RUST_EVALUATOR_PY_SHA,
        ),
        "model_py_sha256": (
            MICRO_ROOT / "python/model.py",
            MODEL_PY_SHA,
        ),
        "phase12_mm_py_sha256": (
            MICRO_ROOT / "python/phase12_multimodal_model.py",
            PHASE12_MM_PY_SHA,
        ),
    }
    verified: dict[str, str] = {}
    for key, (path, expected) in pins.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned asset SHA256 mismatch for {key} ({path}): {actual} != {expected}")
        verified[key] = actual
    return verified


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_qualification_lock_held(lock_path: Path | None = None) -> None:
    """
    Verify via Linux kernel `/proc/self/fd` and `/proc/self/fdinfo/<fd>` that THIS process
    holds an open file descriptor to `lock_path` with an active `FLOCK ADVISORY WRITE` lock
    (either inherited from an enclosing `flock qualification.lock ...` wrapper or acquired
    in-process via `fcntl.flock(..., LOCK_EX)`). Rejects unlocked files, locks held only by
    unrelated processes, and mocked subprocess probes.
    """
    target = (lock_path or (Path.home() / "jope_plan/qualification.lock")).resolve()
    if not target.exists():
        raise RuntimeError(f"qualification.lock does not exist at {target}")
    st = target.stat()
    owned_locked_fd = False
    for fd_entry in Path("/proc/self/fd").iterdir():
        try:
            fst = fd_entry.stat()
            if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
                continue
            fdinfo = Path(f"/proc/self/fdinfo/{fd_entry.name}").read_text(encoding="utf-8")
        except OSError:
            continue
        for line in fdinfo.splitlines():
            if line.startswith("lock:") and "FLOCK" in line and "WRITE" in line:
                owned_locked_fd = True
                break
        if owned_locked_fd:
            break
    if not owned_locked_fd:
        raise RuntimeError(
            f"qualification.lock at {target} is NOT held by an open file descriptor of this process"
        )


def adapter_sha(net: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, param in sorted(net.adapter_10tok.named_parameters()):
        if not name.startswith("anchor."):
            h.update(name.encode("utf-8"))
            h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def verify_dope_guard(net: torch.nn.Module, arm: str, expected_dope: str = DOPE_SHA) -> str:
    snap = base.guard.snapshot_dope_weights(net)
    if snap.composite_sha256 != expected_dope:
        raise RuntimeError(f"DOPE guard failed ({arm}): {snap.composite_sha256} != {expected_dope}")
    counts = base.guard.get_trainable_param_counts_by_submodule(net)
    expected_params = TRAINABLE_PARAMS_A if arm == "A" else TRAINABLE_PARAMS_BCD
    total_tr = counts.get("__global_total__", {}).get("trainable_params")
    adap_tr = counts.get("adapter_10tok", {}).get("trainable_params")
    if total_tr != expected_params or adap_tr != expected_params:
        raise RuntimeError(
            f"trainable param guard failed for arm {arm}: expected {expected_params}, got {counts}"
        )
    return snap.composite_sha256


def reset_adapter_fold_clean(
    net: torch.nn.Module,
    seed: int,
    *,
    freeze_parent_delta: bool,
    scale: float = 0.1,
    fit_pools: dict[str, dict] | None = None,
    fit_cids: list[str] | None = None,
) -> list[torch.nn.Parameter]:
    """
    Deterministic fold-clean initialization of both `gamenet_anchor` (`20000 + seed`, frozen) and
    `adapter_10tok` (`10000 + seed`) with zero `r882` or `DL-Train` validation-fold neural weight
    transfer. If `fit_pools` and `fit_cids` are provided, `prog_mean` and `prog_std` are computed
    strictly from the 4 seed anchors of `fit_cids`.
    """
    init_seed = 10_000 + int(seed)
    anchor_seed = 20_000 + int(seed)
    old = net.adapter_10tok
    cls = type(old)
    device = next(old.parameters()).device

    if fit_pools is not None and fit_cids:
        anchor_progs = np.concatenate(
            [fit_pools[cid]["program"][:4] for cid in fit_cids if cid in fit_pools],
            axis=0,
        ).astype(np.float32)
        prog_mean_np = anchor_progs.mean(axis=0).astype(np.float32)
        prog_std_np = np.maximum(anchor_progs.std(axis=0), 1e-6).astype(np.float32)
    else:
        prog_mean_np = old.prog_mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
        prog_std_np = old.prog_std.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # Fold-clean reinitialization of the frozen `gamenet_anchor` submodule
    anchor_cls = type(net.gamenet_anchor)
    torch.manual_seed(anchor_seed)
    np.random.seed(anchor_seed)
    fresh_anchor = anchor_cls(
        ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4
    ).to(device)
    with torch.no_grad():
        for p in fresh_anchor.parameters():
            if scale != 1.0:
                p.mul_(scale)
            p.requires_grad_(False)
    fresh_anchor.eval()
    net.gamenet_anchor = fresh_anchor

    # Fold-clean reinitialization of `adapter_10tok`
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    fresh = cls(
        frozen_anchor=fresh_anchor,
        prog_mean=prog_mean_np,
        prog_std=prog_std_np,
        d_token=32,
        n_modalities=10,
        modality_dropout_p=0.10,
    ).to(device)
    for method_name in (
        "encode_multimodal_context_and_candidates",
        "predict_parent_child_delta",
    ):
        bound = getattr(old, method_name)
        setattr(fresh, method_name, MethodType(bound.__func__, fresh))
    net.adapter_10tok = fresh

    for param in net.parameters():
        param.requires_grad_(False)
    for name, param in net.adapter_10tok.named_parameters():
        if name.startswith("anchor."):
            param.requires_grad_(False)
            continue
        if scale != 1.0:
            with torch.no_grad():
                param.mul_(scale)
        if freeze_parent_delta and name.startswith("parent_delta_net."):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    base.guard.lock_dope_weights_and_eval_mode(net)
    trainable = [p for p in net.parameters() if p.requires_grad]
    expected = TRAINABLE_PARAMS_BCD if freeze_parent_delta else TRAINABLE_PARAMS_A
    actual = sum(p.numel() for p in trainable)
    if actual != expected:
        raise RuntimeError(f"trainable parameter count mismatch: {actual} != {expected}")
    if any(
        name.startswith("anchor.") and p.requires_grad
        for name, p in net.adapter_10tok.named_parameters()
    ):
        raise RuntimeError("frozen anchor became trainable")
    if freeze_parent_delta and any(
        name.startswith("parent_delta_net.") and p.requires_grad
        for name, p in net.adapter_10tok.named_parameters()
    ):
        raise RuntimeError("parent_delta_net remained trainable in B/C/D")
    return trainable


def compute_candidate_priors_live(
    net: torch.nn.Module,
    prog_412: np.ndarray,
    lm48_z_1d: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute real base_prior_cost [M] and sos_prior [M, 2] for newly proposed GP ASTs."""
    m = prog_412.shape[0]
    sem_24 = base.compute_semantic_struct_sig_24d(prog_412)
    with torch.no_grad():
        c_t = torch.as_tensor(prog_412, dtype=torch.float32, device=device).unsqueeze(0)
        s_t = torch.as_tensor(sem_24, dtype=torch.float32, device=device).unsqueeze(0)
        l_t = torch.as_tensor(lm48_z_1d, dtype=torch.float32, device=device).unsqueeze(0)
        base_p = net._predict_calibrated_prior_batched_gpu(c_t, s_t, l_t, track_prefix="prior10")[0]
        sos_idx = torch.full((1, m), -1, dtype=torch.long, device=device)
        sos_p = net._compute_sos_prior_batched_gpu(c_t, sos_idx)[0]
    return base_p.detach().cpu().numpy().astype(np.float32), sos_p.detach().cpu().numpy().astype(np.float32)


def clone_pools(base_pools: dict[str, dict], cids: list[str] | None = None) -> dict[str, dict]:
    """
    Create an isolated copy-on-write pool view so per-round candidate admissions never cross
    arms/folds while avoiding redundant multi-gigabyte array copies for unmutated contexts.
    """
    target_cids = cids if cids is not None else list(base_pools.keys())
    out = {}
    for cid in target_cids:
        src = base_pools[cid]
        out[cid] = {
            "rec": src["rec"],
            "pids": list(src["pids"]),
            "program": src["program"],
            "proj16": src["proj16"],
            "fitness": src["fitness"],
            "logratio": src["logratio"],
            "gate": src["gate"],
            "base_prior_cost": src["base_prior_cost"],
            "sos_prior": src["sos_prior"],
            "scale": float(src["scale"]),
            "lm48_z": src.get("lm48_z", np.zeros((48,), dtype=np.float32)),
            "jtt1_gz_path": src.get("jtt1_gz_path"),
            "jtt1_gz_sha256": src.get("jtt1_gz_sha256"),
            "champions": list(src.get("champions", [])),
            "table_bytes": int(src.get("table_bytes", 32817)),
            "hard_negatives": set(src.get("hard_negatives", set())),
        }
    return out


def _default_seed_champions_for_cid(cid: str, fam_id: str, lm48_z: np.ndarray) -> list[dict]:
    seed_dsls = [
        "(pipe X (sigmoid append=true) (ridge_gcv))",
        "(pipe X (winsor_clip append=true q=0.1) (ridge_gcv))",
    ]
    out = []
    for dsl in seed_dsls:
        f412, pr16 = base.evaluator.featurize_dsl_412(dsl)
        out.append(
            {
                "context_sha256": cid,
                "family_id": fam_id,
                "dl_split": "DL-Train",
                "dsl": dsl,
                "program_sha256": hashlib.sha256(dsl.encode("utf-8")).hexdigest(),
                "feat412": f412.astype(np.float32).tolist(),
                "proj16": pr16.astype(np.float32).tolist(),
                "landmark_summary": lm48_z[:12].astype(np.float32).tolist(),
            }
        )
    return out


def _ensure_jtt1_gz_for_pool(cid: str, pool: dict) -> Path:
    """
    Resolve and cryptographically verify the two-fold `.jtt1.gz` evaluation envelope for `cid`.
    Never falls back to another dataset's table; fails closed if missing or SHA256-mismatched.
    """
    gz_path_raw = pool.get("jtt1_gz_path")
    expected_sha = pool.get("jtt1_gz_sha256")
    if gz_path_raw is None or not expected_sha:
        raise RuntimeError(
            f"missing jtt1_gz_path or jtt1_gz_sha256 for context {cid}; refusing cross-dataset fallback"
        )
    gz_path = Path(gz_path_raw)
    if not gz_path.is_file():
        raise RuntimeError(f"jtt1_gz envelope file does not exist for context {cid}: {gz_path}")
    actual_sha = sha256(gz_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"jtt1_gz SHA256 mismatch for context {cid}: {actual_sha} != {expected_sha}"
        )
    return gz_path


def make_predictor_arm(
    net: torch.nn.Module,
    pools: dict[str, dict],
    ctx_cache: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
    arm: str,
):
    """
    Build predictor for Arm A vs Arms B/C/D.
      - Arm A: zeroed ctx_feat/base_prior_cost/sos_prior, constant scale=0.4208.
      - Arms B/C/D: real 584D ctx_feat, real prior10 base_prior_cost, real 2D sos_prior,
        real per-context scale = max(std(fitness[:4]), 0.10).
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")

    def predict(cids: list[str], indexes: list[np.ndarray]) -> torch.Tensor:
        progs = np.stack([pools[cid]["program"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        proj = np.stack([pools[cid]["proj16"][ix] for cid, ix in zip(cids, indexes, strict=True)])
        batch, slate = progs.shape[:2]
        n_rev = min(4, slate)
        histf = torch.as_tensor(
            np.stack([pools[cid]["fitness"][ix[:n_rev]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32,
            device=device,
        )
        histg = torch.as_tensor(
            np.stack([pools[cid]["gate"][ix[:n_rev]] for cid, ix in zip(cids, indexes, strict=True)]),
            dtype=torch.float32,
            device=device,
        )
        tokens_b = torch.stack([ctx_cache[cid]["tokens_10x32"] for cid in cids])
        masks_b = torch.stack([ctx_cache[cid]["mask_10"] for cid in cids])
        rev_idx = torch.arange(n_rev, device=device).reshape(1, -1).expand(batch, -1)

        if arm == "A":
            ctx_feat_t = torch.zeros((batch, 584), dtype=torch.float32, device=device)
            base_prior_t = torch.zeros((batch, slate), dtype=torch.float32, device=device)
            sos_prior_t = torch.zeros((batch, slate, 2), dtype=torch.float32, device=device)
            scale_t = torch.full((batch,), 0.4208, dtype=torch.float32, device=device)
        else:
            ctx_feat_t = torch.stack([ctx_cache[cid]["ctx_feat_584"] for cid in cids])
            base_prior_np = np.stack(
                [pools[cid]["base_prior_cost"][ix] for cid, ix in zip(cids, indexes, strict=True)]
            )
            sos_prior_np = np.stack(
                [pools[cid]["sos_prior"][ix] for cid, ix in zip(cids, indexes, strict=True)]
            )
            scales_np = np.asarray([pools[cid]["scale"] for cid in cids], dtype=np.float32)
            base_prior_t = torch.as_tensor(base_prior_np, dtype=torch.float32, device=device)
            sos_prior_t = torch.as_tensor(sos_prior_np, dtype=torch.float32, device=device)
            scale_t = torch.as_tensor(scales_np, dtype=torch.float32, device=device)

        return net.adapter_10tok.predict_costs(
            ctx_feat=ctx_feat_t,
            prog_412=torch.as_tensor(progs, dtype=torch.float32, device=device),
            proj_16=torch.as_tensor(proj, dtype=torch.float32, device=device),
            base_prior_cost=base_prior_t,
            scale=scale_t,
            sos_prior=sos_prior_t,
            mm_tokens_10x32=tokens_b,
            mm_mask_10=masks_b,
            revealed_indices=rev_idx,
            revealed_raw_f=histf,
            revealed_gate=histg,
        )["total_cost"].float()

    return predict


def sample_slate_arm(
    pool: dict,
    slate_size: int,
    rng: np.random.Generator,
    arm: str,
) -> np.ndarray:
    """
    Construct a training slate of size `min(slate_size, total)`:
      - Arm A (`CONTROL`): reproduces `automatic_promotion.py:training_indices` (`k = min(48, target_len, len(order))`,
        placing `order[:k]` first without permutation so `order[0..3]` land at revealed anchor positions `0..3`).
      - Arm B (`FIX1`): fixes slots `0..3` to the 4 unsorted seed anchors `[0, 1, 2, 3]`,
        samples top-48 passing from `idx >= 4` + random remaining from `idx >= 4`,
        and randomly permutes all unrevealed slots `4..target_len-1`.
      - Arms C & D (`FIX1+2` / `FIX1+2+3`): fixes slots `0..3` to `[0, 1, 2, 3]`,
        samples up to 48 top passing from `idx >= 4`, up to 36 evaluated hard negatives
        (gate failures, non-HWM GP candidates, and model-proposed false positives) from `idx >= 4`,
        fills the remainder uniformly from `idx >= 4`, and randomly permutes slots `4..target_len-1`.
    """
    total = len(pool["pids"])
    if total <= 4 or slate_size <= 4:
        raise ValueError(f"pool and slate_size must be > 4, got total={total}, slate_size={slate_size}")
    target_len = min(slate_size, total)

    if arm == "A":
        order = np.argsort(pool["fitness"])
        k = min(48, target_len, len(order))
        rest_pool = order[k:] if len(order) > k else order
        need = target_len - k
        rest = (
            rng.choice(rest_pool, size=need, replace=len(rest_pool) < need)
            if need > 0
            else np.empty(0, dtype=np.int64)
        )
        return np.concatenate([order[:k], rest]).astype(np.int64)

    # Arms B, C, D: slots 0..3 are strictly the 4 seed anchors [0, 1, 2, 3]
    anchors = np.array([0, 1, 2, 3], dtype=np.int64)
    unrev_budget = target_len - 4
    unrev_all = np.arange(4, total, dtype=np.int64)
    unrev_gate = pool["gate"][4:]
    unrev_fit = pool["fitness"][4:]
    passed_u = unrev_all[np.flatnonzero(unrev_gate)]
    order_u = (
        passed_u[np.argsort(pool["fitness"][passed_u])]
        if len(passed_u)
        else unrev_all[np.argsort(unrev_fit)]
    )
    top_u = order_u[: min(48, len(order_u), unrev_budget)]

    if arm == "B":
        remain_u = np.setdiff1d(unrev_all, top_u, assume_unique=False)
        need_u = max(0, unrev_budget - len(top_u))
        rest_u = (
            rng.choice(remain_u, size=min(need_u, len(remain_u)), replace=False)
            if len(remain_u) and need_u > 0
            else np.empty(0, dtype=np.int64)
        )
        unrev_picks = np.concatenate([top_u, rest_u]).astype(np.int64)
    else:
        hn_indices = np.asarray(
            sorted(idx for idx in pool.get("hard_negatives", set()) if 4 <= idx < total),
            dtype=np.int64,
        )
        hn_avail = np.setdiff1d(hn_indices, top_u, assume_unique=False)
        hn_quota = min(36, len(hn_avail), max(0, unrev_budget - len(top_u)))
        hn_picks = (
            rng.choice(hn_avail, size=hn_quota, replace=False)
            if hn_quota > 0
            else np.empty(0, dtype=np.int64)
        )
        chosen_so_far = np.concatenate([top_u, hn_picks]).astype(np.int64)
        remain_u = np.setdiff1d(unrev_all, chosen_so_far, assume_unique=False)
        need_u = max(0, unrev_budget - len(chosen_so_far))
        rest_u = (
            rng.choice(remain_u, size=min(need_u, len(remain_u)), replace=False)
            if len(remain_u) and need_u > 0
            else np.empty(0, dtype=np.int64)
        )
        unrev_picks = np.concatenate([chosen_so_far, rest_u]).astype(np.int64)

    permuted_unrev = rng.permutation(unrev_picks).astype(np.int64)
    return np.concatenate([anchors, permuted_unrev])


def admit_gp_candidate_to_pool(
    pool: dict,
    rec: dict,
    *,
    arm: str,
) -> bool:
    """
    Apply arm-specific candidate admission rule (`promote()`):
      - Execution failures (`rec.get("status", "completed") != "completed"`) are NEVER admitted
        to any pool or counted as gate failures in any arm.
      - Arms A & B: HWM-only (`status == "completed"`, `cv_gate_passed == True`, and `fitness < incumbent - 1e-6`).
      - Arms C & D: Admit ALL completed evaluated GP candidates (`status == "completed"`, including
        non-HWM evaluations and genuine CV gate failures `cv_gate_passed == False`), and register
        non-HWM / gate-failing candidates in `pool["hard_negatives"]`.
    """
    if rec.get("status", "completed") != "completed":
        return False
    psha = str(rec["program_sha256"])
    gate_ok = bool(rec.get("cv_gate_passed", False))
    if gate_ok and not np.isfinite(float(rec["fitness"])):
        return False
    fit = float(rec["fitness"]) if gate_ok else 2.0
    logratio = float(rec.get("logratio", fit)) if np.isfinite(float(rec.get("logratio", fit))) else fit
    passed_mask = pool["gate"]
    incumbent = float(np.min(pool["fitness"][passed_mask])) if np.any(passed_mask) else 0.0
    beats_hwm = gate_ok and (fit < incumbent - 1e-6)

    if arm in ("A", "B"):
        if not beats_hwm or psha in pool["pids"]:
            return False
    else:
        if psha in pool["pids"]:
            existing_idx = pool["pids"].index(psha)
            if not beats_hwm and existing_idx >= 4:
                pool["hard_negatives"].add(existing_idx)
            return False

    new_idx = len(pool["pids"])
    pool["pids"].append(psha)
    pool["program"] = np.vstack([pool["program"], np.asarray(rec["feat412"], dtype=np.float32)])
    pool["proj16"] = np.vstack([pool["proj16"], np.asarray(rec["proj16"], dtype=np.float32)])
    pool["fitness"] = np.append(pool["fitness"], fit)
    pool["logratio"] = np.append(pool["logratio"], logratio)
    pool["gate"] = np.append(pool["gate"], gate_ok)
    pool["base_prior_cost"] = np.append(
        pool["base_prior_cost"], float(rec.get("base_prior_cost", 0.0))
    ).astype(np.float32)
    sos_val = rec.get("sos_prior", [0.0, 0.25])
    pool["sos_prior"] = np.vstack(
        [pool["sos_prior"], np.asarray(sos_val, dtype=np.float32).reshape(1, 2)]
    )
    if arm in ("C", "D") and not beats_hwm and new_idx >= 4:
        pool["hard_negatives"].add(new_idx)
    return True


def select_gp_proposal_indices(
    costs: np.ndarray,
    seed: int,
    max_eval_per_ctx: int = 2,
    explore: bool = True,
) -> list[int]:
    """
    Select greedy + exploration candidate indices within `max_eval_per_ctx` using `base.gp_selection_indices`.
    `base.gp_selection_indices(costs, seed, explore)` returns `order[:2]` (up to 2 greedy picks) followed by
    up to 2 random exploration picks from `order[2:]`. When `max_eval_per_ctx == 2` and `explore=True`,
    this allocates 1 greedy pick (`raw[0]`) + 1 exploration pick (`raw[2]`) rather than truncating away
    the exploration picks. When `max_eval_per_ctx == 4`, it returns 2 greedy + 2 exploration picks.
    """
    n_total = len(costs)
    if max_eval_per_ctx <= 0 or n_total == 0:
        return []
    raw = [int(i) for i in base.gp_selection_indices(costs, int(seed), bool(explore))]
    budget = min(int(max_eval_per_ctx), len(raw))
    if not explore or n_total <= 2 or budget <= 1:
        return raw[:budget]
    n_explore = min(max(1, budget // 2), max(0, len(raw) - 2))
    n_greedy = min(budget - n_explore, min(2, len(raw)))
    selected = raw[:n_greedy] + raw[2 : 2 + n_explore]
    for idx in raw:
        if len(selected) >= budget:
            break
        if idx not in selected:
            selected.append(idx)
    return selected


def init_cell_pools_and_bank(
    base_pools: dict[str, dict],
    fit_cids: list[str],
    eval_cids: list[str],
    journal_by_cid: dict[str, list[dict]],
    arm: str,
) -> tuple[dict[str, dict], list[dict], int]:
    """
    Initialize isolated pools and the live GP mutation parent `bank` for a training cell:
      1. Clone `base_pools` for `fit_cids | eval_cids`.
      2. Apply round-0 historical journal admission (`journal_by_cid`) strictly on `fit_cids`
         using the arm's admission rule (`HWM-only` in A/B vs `all evaluated candidates` in C/D).
      3. Seed the live GP parent `bank` strictly from `fit_cids` champions (plus any passing
         journal admissions on `fit_cids`).
    """
    active_cids = sorted(set(fit_cids) | set(eval_cids))
    pools = clone_pools(base_pools, active_cids)
    bank: list[dict] = []
    known_bank: set[tuple[str, str]] = set()
    journal_admitted = 0

    for cid in fit_cids:
        p = pools[cid]
        fam_id = p["rec"]["family_id"]
        lm48 = np.asarray(p.get("lm48_z", np.zeros((48,), dtype=np.float32)), dtype=np.float32)
        champs = p.get("champions") or _default_seed_champions_for_cid(cid, fam_id, lm48)
        for ch in champs:
            key = (cid, str(ch["program_sha256"]))
            if key not in known_bank:
                bank.append(dict(ch))
                known_bank.add(key)

        for jrec in journal_by_cid.get(cid, []):
            if admit_gp_candidate_to_pool(p, jrec, arm=arm):
                journal_admitted += 1
                if bool(jrec.get("cv_gate_passed", False)) and jrec.get("dsl"):
                    key = (cid, str(jrec["program_sha256"]))
                    if key not in known_bank:
                        bank.append(
                            {
                                "context_sha256": cid,
                                "family_id": fam_id,
                                "dl_split": "DL-Train",
                                "dsl": jrec["dsl"],
                                "program_sha256": str(jrec["program_sha256"]),
                                "feat412": list(jrec["feat412"]),
                                "proj16": list(jrec["proj16"]),
                                "landmark_summary": lm48[:12].tolist(),
                            }
                        )
                        known_bank.add(key)

    return pools, bank, journal_admitted


def execute_gp_selfplay_round(
    pools: dict[str, dict],
    bank: list[dict],
    gp_cids: list[str],
    predict_fn,
    rng: np.random.Generator,
    arm: str,
    *,
    net: torch.nn.Module | None = None,
    device: torch.device = torch.device("cpu"),
    round_idx: int = 1,
    round_seed: int | None = None,
    n_proposals: int = 8,
    gp_exploration: float = 0.25,
    max_eval_per_ctx: int = 2,
    max_workers: int = 2,
) -> dict[str, int]:
    """
    Execute one round of live GP candidate synthesis -> neural adapter ranking -> selection ->
    native Rust two-fold CV evaluation -> arm-specific pool & parent-bank admission across `gp_cids`
    (mirroring `run_fused_closed_loop.py:1156-1306`):
      1. Enforce whole-host CPU ceiling (`<= 85%`) via `base.headroom()` and `GLOBAL_CPU_GUARD`.
      2. Mutate current `bank` parents for `cid` via `base.evaluator.propose_flywheel_candidates(...)`.
      3. Featurize novel proposed ASTs via `base.evaluator.featurize_dsl_412`, compute real
         `base_prior_cost` and `sos_prior`, and rank them with the current neural adapter `predict_fn`.
      4. Select top greedy + exploration candidates within `max_eval_per_ctx` via `select_gp_proposal_indices`.
      5. Score selected ASTs with the native Rust evaluator `base.evaluator.evaluate_gp_batch_rust`.
      6. Separate completed CV evaluations (`status == "completed"`) from execution failures (`status != "completed"`),
         and pass completed evaluations to `admit_gp_candidate_to_pool(..., arm=arm)`.
    """
    base.headroom()
    base.evaluator.GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.5)
    eff_seed = int(round_seed) if round_seed is not None else int(rng.integers(1, 1_000_000_000))

    proposer_calls = 0
    evaluator_calls = 0
    proposed_total = 0
    evaluated_total = 0
    evaluated_completed = 0
    execution_failed = 0
    admitted_total = 0
    admitted_gate_fails = 0
    admitted_non_hwm = 0

    for i, cid in enumerate(gp_cids):
        p = pools[cid]
        fam_id = p["rec"].get("family_id", "fam_0")
        lm48 = np.asarray(p.get("lm48_z", np.zeros((48,), dtype=np.float32)), dtype=np.float32)
        parents = [rec for rec in bank if rec.get("context_sha256") == cid]
        if not parents:
            parents = _default_seed_champions_for_cid(cid, fam_id, lm48)
            bank.extend(parents)
        parent_offset = (round_idx + i) % len(parents)
        parents = parents[parent_offset:] + parents[:parent_offset]

        proposer_calls += 1
        raw_proposals = base.evaluator.propose_flywheel_candidates(
            parents,
            lm48,
            bank,
            fam_id,
            n_proposals=n_proposals,
            seed=eff_seed + i,
        )
        known_pids = set(p["pids"])
        proposals = [cand for cand in raw_proposals if cand["program_sha256"] not in known_pids]
        if not proposals:
            proposals = list(raw_proposals[: max(1, max_eval_per_ctx)])
        proposed_total += len(proposals)
        if not proposals:
            continue

        featurized_pairs = [base.evaluator.featurize_dsl_412(x["dsl"]) for x in proposals]
        f_prop = np.stack([fp[0].astype(np.float32) for fp in featurized_pairs])
        pr_prop = np.stack([fp[1].astype(np.float32) for fp in featurized_pairs])
        if net is not None:
            bp_prop, sp_prop = compute_candidate_priors_live(net, f_prop, lm48, device)
        else:
            bp_prop = np.zeros((len(proposals),), dtype=np.float32)
            sp_prop = np.repeat(np.array([[0.0, 0.25]], dtype=np.float32), len(proposals), axis=0)

        for idx_p, cand in enumerate(proposals):
            cand["feat412"] = f_prop[idx_p]
            cand["proj16"] = pr_prop[idx_p]
            cand["base_prior_cost"] = float(bp_prop[idx_p])
            cand["sos_prior"] = sp_prop[idx_p]

        temp_pool = dict(p)
        temp_pool["program"] = np.concatenate([p["program"][:4], f_prop], axis=0)
        temp_pool["proj16"] = np.concatenate([p["proj16"][:4], pr_prop], axis=0)
        temp_pool["base_prior_cost"] = np.concatenate([p["base_prior_cost"][:4], bp_prop], axis=0)
        temp_pool["sos_prior"] = np.concatenate([p["sos_prior"][:4], sp_prop], axis=0)
        pools[cid] = temp_pool
        try:
            with torch.no_grad():
                rank_scores = (
                    predict_fn([cid], [np.arange(4 + len(proposals), dtype=np.int64)])[0, 4:]
                    .detach()
                    .cpu()
                    .numpy()
                )
        finally:
            pools[cid] = p

        sel_indices = select_gp_proposal_indices(
            rank_scores,
            eff_seed + i,
            max_eval_per_ctx=max_eval_per_ctx,
            explore=bool(gp_exploration),
        )
        chosen = [proposals[int(j)] for j in sel_indices]
        if not chosen:
            continue

        gz_path = _ensure_jtt1_gz_for_pool(cid, p)
        evaluator_calls += 1
        results = base.evaluator.evaluate_gp_batch_rust(
            str(gz_path),
            chosen,
            run_outer_test=False,
            max_workers=max_workers,
            timeout_s=20.0,
        )
        by_sha = {c["program_sha256"]: c for c in chosen}
        for r in results:
            evaluated_total += 1
            if r.get("status", "completed") != "completed":
                execution_failed += 1
                continue
            evaluated_completed += 1
            src_cand = by_sha.get(r["program_sha256"], {})
            if "base_prior_cost" not in r:
                r["base_prior_cost"] = float(src_cand.get("base_prior_cost", 0.0))
            if "sos_prior" not in r:
                r["sos_prior"] = src_cand.get("sos_prior", [0.0, 0.25])

            passed_mask = p["gate"]
            incumbent = float(np.min(p["fitness"][passed_mask])) if np.any(passed_mask) else 0.0
            gate_ok = bool(r.get("cv_gate_passed", False))
            was_gate_fail = not gate_ok
            was_non_hwm = gate_ok and (float(r["fitness"]) >= incumbent - 1e-6)
            if admit_gp_candidate_to_pool(p, r, arm=arm):
                admitted_total += 1
                if was_gate_fail:
                    admitted_gate_fails += 1
                elif was_non_hwm:
                    admitted_non_hwm += 1
                if gate_ok and r.get("dsl"):
                    bank.append(
                        {
                            "context_sha256": cid,
                            "family_id": fam_id,
                            "dl_split": "DL-Train",
                            "dsl": r["dsl"],
                            "program_sha256": str(r["program_sha256"]),
                            "feat412": np.asarray(r["feat412"], dtype=np.float32).tolist(),
                            "proj16": np.asarray(r["proj16"], dtype=np.float32).tolist(),
                            "landmark_summary": lm48[:12].tolist(),
                        }
                    )

    return {
        "proposer_calls": proposer_calls,
        "evaluator_calls": evaluator_calls,
        "proposed": proposed_total,
        "evaluated": evaluated_total,
        "evaluated_completed": evaluated_completed,
        "execution_failed": execution_failed,
        "admitted": admitted_total,
        "admitted_gate_fails": admitted_gate_fails,
        "admitted_non_hwm": admitted_non_hwm,
    }


def seed_initial_hard_negatives_for_cd(pools: dict[str, dict], fit_cids: list[str]) -> int:
    """
    For Arms C & D, initialize `pool["hard_negatives"]` on training contexts (`fit_cids`) with:
      1. All unrevealed gate failures (`idx >= 4` where `gate == False`).
      2. High-prior / high-regret decoy candidates (`idx >= 4` where `gate == True` and `fitness`
         is in the worst 40% of the pool while `base_prior_cost` is in the best 40%).
    """
    count = 0
    for cid in fit_cids:
        pool = pools[cid]
        gate = pool["gate"]
        fit = pool["fitness"]
        bprior = pool["base_prior_cost"]
        hn = pool["hard_negatives"]
        fail_idxs = np.flatnonzero(~gate[4:]) + 4
        for idx in fail_idxs.tolist():
            hn.add(int(idx))
        passed_u = np.flatnonzero(gate[4:]) + 4
        if len(passed_u) >= 16:
            fit_q60 = float(np.quantile(fit[passed_u], 0.60))
            pri_q40 = float(np.quantile(bprior[passed_u], 0.40))
            decoys = passed_u[(fit[passed_u] >= fit_q60) & (bprior[passed_u] <= pri_q40)]
            for idx in decoys.tolist():
                hn.add(int(idx))
        count += len(hn)
    return count


def harvest_round_hard_negatives_cd(
    pools: dict[str, dict],
    batch_cids: list[str],
    indexes: list[np.ndarray],
    pred_tensor: torch.Tensor,
) -> int:
    """
    During training rounds in Arms C & D, inspect the model's top-8 predicted unrevealed candidates
    (`slots 4..191`) for each context in the batch. Any candidate ranked in the model's top-8 that
    either fails the gate (`gate == False`) or has actual fitness worse than the context's top-25%
    passing quantile is recorded in `pool["hard_negatives"]` for subsequent rounds.
    """
    added = 0
    pred_np = pred_tensor.detach().cpu().numpy()
    for cid, ix, row in zip(batch_cids, indexes, pred_np, strict=True):
        if len(ix) <= 4:
            continue
        pool = pools[cid]
        unrev_order = np.argsort(row[4:])[:8]
        top_pool_idxs = ix[4 + unrev_order]
        gate = pool["gate"]
        fit = pool["fitness"]
        passed_all = np.flatnonzero(gate[4:]) + 4
        q25 = float(np.quantile(fit[passed_all], 0.25)) if len(passed_all) else 0.0
        for p_idx in top_pool_idxs.tolist():
            p_idx = int(p_idx)
            if p_idx >= 4 and (not bool(gate[p_idx]) or float(fit[p_idx]) > q25):
                if p_idx not in pool["hard_negatives"]:
                    pool["hard_negatives"].add(p_idx)
                    added += 1
    return added


def compute_loss_arm(
    arm: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    gates: torch.Tensor,
) -> torch.Tensor:
    """
    Compute arm-specific training objective:
      - Arm A (`CONTROL`): reproduces `run_fused_closed_loop.py:979-982` (`F.smooth_l1_loss(pred, target)`
        with default `beta=1.0` + `0.1 * F.kl_div(log_softmax(-pred/0.2), softmax(-target/0.2))` across
        all slots `0..M-1`, including revealed anchor slots `0..3`).
      - Arms B & C (`FIX1` / `FIX1+2`): masks out revealed anchor positions `0..3` and applies
        `smooth_l1(beta=1.0) + 0.1 * kl_div(tau=0.2)` strictly on unrevealed slots `4..M-1`.
      - Arm D (`FIX1+2+3`): masks out revealed anchor positions `0..3` and applies the shift-invariant
        listwise top-focused objective (`L_PL + 0.50 * L_gate + 0.25 * L_pair`) on unrevealed slots `4..M-1`.
    """
    if arm == "A":
        loss_reg = F.smooth_l1_loss(pred, target)
        loss_kl = F.kl_div(
            F.log_softmax(-pred / 0.2, dim=1),
            F.softmax(-target / 0.2, dim=1),
            reduction="batchmean",
        )
        return loss_reg + 0.1 * loss_kl

    pred_u = pred[:, 4:]
    target_u = target[:, 4:]
    gate_u = gates[:, 4:]

    if arm in ("B", "C"):
        loss_reg = F.smooth_l1_loss(pred_u, target_u)
        loss_kl = F.kl_div(
            F.log_softmax(-pred_u / 0.2, dim=1),
            F.softmax(-target_u / 0.2, dim=1),
            reduction="batchmean",
        )
        return loss_reg + 0.1 * loss_kl

    if arm != "D":
        raise ValueError(f"unknown arm: {arm}")

    batch_size = pred_u.shape[0]
    device = pred_u.device

    with torch.no_grad():
        inf_pos = torch.full_like(target_u, 1e9)
        pass_targets = torch.where(gate_u, target_u, inf_pos)
        c_min = pass_targets.min(dim=1, keepdim=True).values
        has_pass = gate_u.any(dim=1, keepdim=True)
        c_min = torch.where(has_pass, c_min, target_u.min(dim=1, keepdim=True).values)

        pass_count = gate_u.sum(dim=1, keepdim=True).clamp_min(1).float()
        pass_mean = torch.where(gate_u, target_u, torch.zeros_like(target_u)).sum(dim=1, keepdim=True) / pass_count
        pass_var = (
            torch.where(gate_u, (target_u - pass_mean) ** 2, torch.zeros_like(target_u)).sum(dim=1, keepdim=True)
            / pass_count
        )
        c_std = pass_var.sqrt().clamp_min(0.05)

        regret_norm = ((target_u - c_min) / c_std).clamp(min=0.0, max=6.0)
        r_u = torch.where(gate_u, regret_norm, torch.full_like(regret_norm, 8.0))

        p_sharp = F.softmax(-r_u / 0.10, dim=1)
        p_broad = F.softmax(-r_u / 0.35, dim=1)

    log_q_sharp = F.log_softmax(-pred_u / 0.15, dim=1)
    log_q_broad = F.log_softmax(-pred_u / 0.40, dim=1)
    loss_pl = 0.7 * F.kl_div(log_q_sharp, p_sharp, reduction="batchmean") + 0.3 * F.kl_div(
        log_q_broad, p_broad, reduction="batchmean"
    )

    gate_losses = []
    pair_losses = []
    for b in range(batch_size):
        g_b = gate_u[b]
        p_b = pred_u[b]
        t_b = target_u[b]
        pass_idx = torch.nonzero(g_b, as_tuple=False).squeeze(-1)
        fail_idx = torch.nonzero(~g_b, as_tuple=False).squeeze(-1)

        if pass_idx.numel() > 0:
            p_pass = p_b[pass_idx]
            t_pass = t_b[pass_idx]
            order_pass = torch.argsort(t_pass, stable=True)
            best_pred = p_pass[order_pass[0]]
            k8 = min(8, order_pass.numel())
            top8_mean_pred = p_pass[order_pass[:k8]].mean()

            if fail_idx.numel() > 0:
                p_fail = p_b[fail_idx]
                margin_best = F.relu(0.50 + best_pred - p_fail).mean()
                margin_top8 = F.relu(0.25 + top8_mean_pred - p_fail).mean()
                gate_losses.append(margin_best + 0.5 * margin_top8)

            if order_pass.numel() >= 8:
                k4 = min(4, order_pass.numel() // 2)
                top_indices = order_pass[:k4]
                bot_indices = order_pass[order_pass.numel() // 2 :]
                t_top = t_pass[top_indices].unsqueeze(1)
                t_bot = t_pass[bot_indices].unsqueeze(0)
                strict_better = t_top < (t_bot - 1e-6)
                if strict_better.any():
                    top4_pred = p_pass[top_indices].unsqueeze(1)
                    bot_pred = p_pass[bot_indices].unsqueeze(0)
                    pair_hinge = F.relu(0.15 + top4_pred - bot_pred)
                    pair_losses.append(pair_hinge[strict_better].mean())

    loss_gate = torch.stack(gate_losses).mean() if gate_losses else torch.zeros((), device=device)
    loss_pair = torch.stack(pair_losses).mean() if pair_losses else torch.zeros((), device=device)
    return loss_pl + 0.50 * loss_gate + 0.25 * loss_pair


def _average_ranks(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64).reshape(-1)
    n = arr.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + (j - 1))
        i = j
    return ranks


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.size < 3 or y_arr.size < 3 or x_arr.size != y_arr.size:
        return 0.0
    if not (np.all(np.isfinite(x_arr)) and np.all(np.isfinite(y_arr))):
        return 0.0
    if float(np.ptp(x_arr)) <= 1e-12 or float(np.ptp(y_arr)) <= 1e-12:
        return 0.0
    rx = _average_ranks(x_arr)
    ry = _average_ranks(y_arr)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(rx, ry)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _score_choice(pool: dict, choice: int) -> tuple[float, float, float, float]:
    gate_all = pool["gate"]
    fit_all = pool["fitness"]
    passed_all = np.flatnonzero(gate_all)
    if not bool(gate_all[choice]) or len(passed_all) == 0:
        return -1.0, 1.0, 4.0, 0.0
    oracle = float(np.min(fit_all[passed_all]))
    spread = max(float(np.std(fit_all[passed_all])), 0.05)
    reg_norm = max(0.0, (float(fit_all[choice]) - oracle) / spread)
    u = float(np.clip(1.0 - reg_norm / 2.0, -1.0, 1.0))
    q05 = float(np.quantile(fit_all[passed_all], 0.05))
    hit5 = 1.0 if float(fit_all[choice]) <= q05 + 1e-12 else 0.0
    return u, 0.0, reg_norm, hit5


def evaluate_utility_both_modes(
    cids: list[str],
    pools: dict[str, dict],
    predict_fn,
    arm: str,
) -> tuple[dict, dict]:
    """
    Evaluate both `native` (unmasked `0..P-1` for Arm A; masked `4..P-1` for Arms B/C/D)
    and leak-free `unrev` (`choice = 4 + argmin(row[4:])` for all arms) in a SINGLE evaluation sweep.
    """
    by_size: dict[int, list[str]] = {}
    for cid in cids:
        by_size.setdefault(len(pools[cid]["pids"]), []).append(cid)

    nat_ctx, unrev_ctx = {}, {}
    nat_fam: dict[str, list[float]] = {}
    unrev_fam: dict[str, list[float]] = {}
    nat_viol, unrev_viol = {}, {}
    nat_reg, unrev_reg = {}, {}
    nat_hit5, unrev_hit5 = {}, {}
    spearman_unrev = {}
    batches_executed = 0

    for size, group in sorted(by_size.items()):
        for start in range(0, len(group), 64):
            batch = group[start : start + 64]
            ixs = [np.arange(size, dtype=np.int64) for _ in batch]
            with torch.no_grad():
                preds = predict_fn(batch, ixs).detach().cpu().numpy()
            batches_executed += 1
            for cid, row in zip(batch, preds, strict=True):
                pool = pools[cid]
                fam = pool["rec"]["family_id"]
                choice_unrev = 4 + int(np.argmin(row[4:]))
                choice_nat = int(np.argmin(row)) if arm == "A" else choice_unrev

                u_n, v_n, r_n, h_n = _score_choice(pool, choice_nat)
                u_u, v_u, r_u, h_u = _score_choice(pool, choice_unrev)

                nat_ctx[cid], nat_viol[cid], nat_reg[cid], nat_hit5[cid] = u_n, v_n, r_n, h_n
                unrev_ctx[cid], unrev_viol[cid], unrev_reg[cid], unrev_hit5[cid] = u_u, v_u, r_u, h_u
                nat_fam.setdefault(fam, []).append(u_n)
                unrev_fam.setdefault(fam, []).append(u_u)

                pass_u = np.flatnonzero(pool["gate"][4:]) + 4
                if len(pass_u) >= 3:
                    spearman_unrev[cid] = _spearman_rank_corr(row[pass_u], pool["fitness"][pass_u])
                else:
                    spearman_unrev[cid] = 0.0

    mean_rho = float(np.mean(list(spearman_unrev.values()))) if spearman_unrev else 0.0

    def pack(ctx_d, fam_d, viol_d, reg_d, hit5_d):
        f_means = {f: float(np.mean(vs)) for f, vs in sorted(fam_d.items())}
        return {
            "mean": float(np.mean(list(f_means.values()))),
            "median": float(np.median(list(f_means.values()))),
            "context_mean": float(np.mean(list(ctx_d.values()))),
            "violation_rate": float(np.mean(list(viol_d.values()))),
            "mean_regret_norm": float(np.mean(list(reg_d.values()))),
            "hit5_rate": float(np.mean(list(hit5_d.values()))),
            "spearman_rho": mean_rho,
            "eval_batches_executed": batches_executed,
            "per_family": f_means,
            "per_context": ctx_d,
        }

    return pack(nat_ctx, nat_fam, nat_viol, nat_reg, nat_hit5), pack(
        unrev_ctx, unrev_fam, unrev_viol, unrev_reg, unrev_hit5
    )


def evaluate_utility_arm(
    cids: list[str],
    pools: dict[str, dict],
    predict_fn,
    arm: str,
) -> dict:
    nat, _ = evaluate_utility_both_modes(cids, pools, predict_fn, arm=arm)
    return nat


def bootstrap_t537(
    gains_by_family: dict[str, float],
    seed: int = 20261030,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    fams = sorted(gains_by_family.keys())
    vals = np.asarray([gains_by_family[f] for f in fams], dtype=np.float64)
    if vals.size == 0:
        raise ValueError("empty family gains")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(draws, vals.size))
    boot_means = vals[idx].mean(axis=1)
    boot_medians = np.median(vals[idx], axis=1)
    mean_gain = float(vals.mean())
    median_gain = float(np.median(vals))
    mean_lcb95 = float(np.quantile(boot_means, 0.05))
    mean_ci95 = [float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))]
    median_ci95 = [float(np.quantile(boot_medians, 0.025)), float(np.quantile(boot_medians, 0.975))]
    pos_fams = int(np.sum(vals > 1e-9))
    zero_fams = int(np.sum(np.abs(vals) <= 1e-9))
    neg_fams = int(np.sum(vals < -1e-9))
    passed = bool((mean_lcb95 > GATE_MARGIN_LCB95) and (median_gain >= 0.0))
    return {
        "family_count": int(vals.size),
        "mean_gain": mean_gain,
        "family_median": median_gain,
        "mean_lcb95": mean_lcb95,
        "mean_ci95": mean_ci95,
        "median_ci95": median_ci95,
        "positive_families": pos_fams,
        "tied_families": zero_fams,
        "negative_families": neg_fams,
        "t537_gate_passed": passed,
    }


def compute_elo_panel(
    cids: list[str],
    family_by_cid: dict[str, str],
    scores_by_arm_and_cid: dict[str, dict[str, float]],
    anchor: str = "A",
    bootstrap_draws: int = 1000,
    seed: int = 20261030,
) -> dict:
    contenders = sorted(scores_by_arm_and_cid.keys())
    records = [
        DatasetEvalRecord(
            dataset_id=cid,
            family_id=family_by_cid[cid],
            task_type="regression",
            scores_by_contender={c: float(scores_by_arm_and_cid[c][cid]) for c in contenders},
        )
        for cid in cids
    ]
    point = compute_family_grouped_bradley_terry_elo(
        records=records,
        anchor_contender=anchor,
        anchor_elo=1500.0,
        prior=0.01,
    )
    by_fam: dict[str, list[DatasetEvalRecord]] = {}
    for rec in records:
        by_fam.setdefault(rec.family_id, []).append(rec)
    fam_names = sorted(by_fam.keys())
    rng = np.random.default_rng(seed)
    boot_ratings: dict[str, list[float]] = {c: [] for c in contenders}
    for _ in range(bootstrap_draws):
        chosen_fams = rng.choice(fam_names, size=len(fam_names), replace=True)
        boot_recs = []
        for b_idx, f_name in enumerate(chosen_fams):
            for r in by_fam[f_name]:
                boot_recs.append(
                    DatasetEvalRecord(
                        dataset_id=f"{r.dataset_id}__b{b_idx}",
                        family_id=f"{f_name}__b{b_idx}",
                        task_type="regression",
                        scores_by_contender=r.scores_by_contender,
                    )
                )
        res = compute_family_grouped_bradley_terry_elo(
            records=boot_recs,
            anchor_contender=anchor,
            anchor_elo=1500.0,
            prior=0.01,
        )
        for c in contenders:
            boot_ratings[c].append(float(res["ratings"][c]))

    elo_ci95 = {
        c: [
            round(float(np.quantile(boot_ratings[c], 0.025)), 2),
            round(float(np.quantile(boot_ratings[c], 0.975)), 2),
        ]
        for c in contenders
    }
    point["elo_ci95"] = elo_ci95
    return point


def family_folds(train_rows: list[dict], count: int = 5, seed: int = FOLD_SEED) -> list[set[str]]:
    families = sorted({row["family_id"] for row in train_rows})
    rng = np.random.default_rng(seed)
    order = list(np.asarray(families, dtype=object)[rng.permutation(len(families))])
    return [set(map(str, chunk)) for chunk in np.array_split(order, count)]


def family_balanced_batch(
    cids: list[str],
    family_by_cid: dict[str, str],
    size: int,
    rng: np.random.Generator,
) -> list[str]:
    by_fam: dict[str, list[str]] = {}
    for cid in cids:
        by_fam.setdefault(family_by_cid[cid], []).append(cid)
    names = sorted(by_fam.keys())
    chosen_fams = rng.integers(0, len(names), size=size)
    return [
        str(by_fam[names[int(fi)]][int(rng.integers(0, len(by_fam[names[int(fi)]])))])
        for fi in chosen_fams
    ]


def run_training_cell(
    *,
    arm: str,
    seed: int,
    cell_id: int,
    net: torch.nn.Module,
    base_pools: dict[str, dict],
    fit_cids: list[str],
    eval_cids: list[str],
    family_by_cid: dict[str, str],
    ctx_cache: dict[str, dict[str, torch.Tensor]],
    journal_by_cid: dict[str, list[dict]],
    rounds: int,
    steps_per_round: int,
    batch_size: int,
    slate_size: int,
    device: torch.device,
    gp_contexts_per_round: int = 1,
    gp_proposals_per_context: int = 8,
    gp_eval_per_context: int = 2,
) -> dict:
    """
    Execute one self-contained flywheel self-play training cell `(arm, seed, fold/terminal)`:
      1. Initialize isolated pools and the live GP mutation parent `bank` strictly on `fit_cids`
         while keeping `eval_cids` untouched.
      2. Reset `net.gamenet_anchor` and `net.adapter_10tok` to fold-clean initialization
         (`20000 + seed` and `10000 + seed`, `scale=0.1`, `freeze_parent_delta=(arm != "A")`,
         and `prog_mean`/`prog_std` computed strictly from `fit_cids`).
      3. In every round `r in 1..rounds`:
         a. Enforce host CPU headroom (`<= 85%`), propose novel GP offspring ASTs via
            `base.evaluator.propose_flywheel_candidates`, rank them with the current neural
            adapter `predict_fn`, select top + exploration candidates via `base.gp_selection_indices`,
            score them with the native Rust evaluator `base.evaluator.evaluate_gp_batch_rust`, and
            apply arm-specific pool admission (`HWM-only` in A/B vs `all evaluated candidates
            including gate failures and non-HWM false positives` in C/D).
         b. Execute `steps_per_round` gradient steps on family-balanced minibatches from `fit_cids`,
            harvesting model-proposed top-8 false positives into `pool["hard_negatives"]` in C & D.
      4. Load EMA shadow weights and evaluate ONCE at the terminal round `R` on `eval_cids`
         (computing both native and leak-free unrevealed metrics in a single evaluation sweep).
    """
    t0 = time.monotonic()
    rng = np.random.default_rng(seed * 1_000_003 + cell_id * 97 + (ord(arm[0]) * 13))
    pools, bank, journal_admitted = init_cell_pools_and_bank(
        base_pools, fit_cids, eval_cids, journal_by_cid, arm
    )

    trainable = reset_adapter_fold_clean(
        net,
        seed=seed,
        freeze_parent_delta=(arm != "A"),
        scale=0.1,
        fit_pools=pools,
        fit_cids=fit_cids,
    )
    dope_before = verify_dope_guard(net, arm)
    init_sha = adapter_sha(net)
    init_snapshot = {
        name: p.detach().clone()
        for name, p in net.named_parameters()
        if p.requires_grad
    }

    initial_hn_count = (
        seed_initial_hard_negatives_for_cd(pools, fit_cids)
        if arm in ("C", "D")
        else 0
    )

    predict_fn = make_predictor_arm(net, pools, ctx_cache, device, arm)
    total_steps = rounds * steps_per_round
    lr_init = 3e-4
    optimizer = torch.optim.AdamW(trainable, lr=lr_init, weight_decay=1e-4)
    base.guard.assert_optimizer_excludes_dope(optimizer, net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=3e-5
    )

    ema_decay = 0.90
    ema_state = {
        name: p.detach().clone()
        for name, p in net.adapter_10tok.named_parameters()
        if p.requires_grad
    }

    losses: list[float] = []
    grad_norms: list[float] = []
    harvested_hn_total = 0
    gp_stats_total = {
        "proposer_calls": 0,
        "evaluator_calls": 0,
        "proposed": 0,
        "evaluated": 0,
        "evaluated_completed": 0,
        "execution_failed": 0,
        "admitted": 0,
        "admitted_gate_fails": 0,
        "admitted_non_hwm": 0,
    }

    # Prefer fast two-fold JTT1 contexts (<= 64 KB) for live per-round native Rust CV evaluation
    fast_gp_cids = [cid for cid in fit_cids if pools[cid].get("table_bytes", 32817) <= 65_536]
    eligible_gp_cids = fast_gp_cids if fast_gp_cids else fit_cids
    eff_gp_ctxs = min(gp_contexts_per_round, len(eligible_gp_cids))

    scratch_root = Path("/dev/shm") / f"h58_rust_eval_{os.getpid()}"
    base.evaluator.EVAL_SCRATCH_ROOT = scratch_root

    try:
        for r_idx in range(1, rounds + 1):
            base.headroom()
            net.eval()
            gp_cids = family_balanced_batch(eligible_gp_cids, family_by_cid, eff_gp_ctxs, rng)
            round_seed = int(seed * 100_000 + cell_id * 1_000 + r_idx * 17)
            r_gp = execute_gp_selfplay_round(
                pools,
                bank,
                gp_cids,
                predict_fn,
                rng,
                arm,
                net=net,
                device=device,
                round_idx=r_idx,
                round_seed=round_seed,
                n_proposals=gp_proposals_per_context,
                gp_exploration=0.25,
                max_eval_per_ctx=gp_eval_per_context,
                max_workers=2,
            )
            for k, v in r_gp.items():
                gp_stats_total[k] += v

            net.train()
            for s_idx in range(steps_per_round):
                base.headroom()
                batch_cids = family_balanced_batch(fit_cids, family_by_cid, batch_size, rng)
                indexes = [sample_slate_arm(pools[cid], slate_size, rng, arm) for cid in batch_cids]

                optimizer.zero_grad(set_to_none=True)
                pred = predict_fn(batch_cids, indexes)

                gates_np = np.stack(
                    [pools[cid]["gate"][ix] for cid, ix in zip(batch_cids, indexes, strict=True)]
                )
                fit_np = np.stack(
                    [pools[cid]["fitness"][ix] for cid, ix in zip(batch_cids, indexes, strict=True)]
                )
                target_np = np.where(gates_np, fit_np, 2.0)
                target_t = torch.as_tensor(target_np, dtype=torch.float32, device=device)
                gates_t = torch.as_tensor(gates_np, dtype=torch.bool, device=device)

                loss = compute_loss_arm(arm, pred, target_t, gates_t)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for name, p in net.adapter_10tok.named_parameters():
                        if p.requires_grad:
                            ema_state[name].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

                if arm in ("C", "D") and s_idx == steps_per_round - 1:
                    harvested_hn_total += harvest_round_hard_negatives_cd(
                        pools, batch_cids, indexes, pred
                    )

                losses.append(float(loss.detach().cpu()))
                grad_norms.append(gnorm)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    if rounds > 0 and (
        gp_stats_total["proposer_calls"] < rounds
        or gp_stats_total["evaluator_calls"] < rounds
        or gp_stats_total["evaluated_completed"] < rounds
    ):
        raise RuntimeError(
            f"live GP self-play failed closed in cell ({arm}, seed={seed}, cell={cell_id}): {gp_stats_total}"
        )

    with torch.no_grad():
        for name, p in net.adapter_10tok.named_parameters():
            if p.requires_grad:
                p.copy_(ema_state[name])

    net.eval()
    dope_after = verify_dope_guard(net, arm)
    final_sha = adapter_sha(net)
    update_l2 = float(
        torch.sqrt(
            sum(
                torch.sum((p.detach() - init_snapshot[name]) ** 2)
                for name, p in net.named_parameters()
                if p.requires_grad
            )
        ).cpu()
    )
    if update_l2 <= 0.0 or final_sha == init_sha:
        raise RuntimeError(f"zero parameter movement in cell ({arm}, seed={seed}, cell={cell_id})")

    eval_native, eval_unrev = evaluate_utility_both_modes(eval_cids, pools, predict_fn, arm=arm)

    return {
        "arm": arm,
        "seed": seed,
        "cell_id": cell_id,
        "rounds": rounds,
        "steps_per_round": steps_per_round,
        "optimizer_steps": total_steps,
        "trainable_params": sum(p.numel() for p in trainable),
        "dope_sha256_before": dope_before,
        "dope_sha256_after": dope_after,
        "adapter_sha256_init": init_sha,
        "adapter_sha256_final": final_sha,
        "update_l2_from_init": update_l2,
        "journal_admitted": journal_admitted,
        "gp_proposer_calls": gp_stats_total["proposer_calls"],
        "gp_evaluator_calls": gp_stats_total["evaluator_calls"],
        "gp_proposed": gp_stats_total["proposed"],
        "gp_evaluated": gp_stats_total["evaluated"],
        "gp_evaluated_completed": gp_stats_total["evaluated_completed"],
        "gp_execution_failed": gp_stats_total["execution_failed"],
        "gp_admitted": gp_stats_total["admitted"],
        "gp_admitted_gate_fails": gp_stats_total["admitted_gate_fails"],
        "gp_admitted_non_hwm": gp_stats_total["admitted_non_hwm"],
        "initial_hard_negatives": initial_hn_count,
        "harvested_hard_negatives": harvested_hn_total,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "terminal_eval_sweeps": 1,
        "eval_batches_executed": eval_unrev["eval_batches_executed"],
        "eval_native": eval_native,
        "eval_unrev": eval_unrev,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }


def load_h58_packet_and_cache(
    input_dir: Path,
    cache_dir: Path,
    device: torch.device,
):
    """Load and cryptographically verify the offline packet (`DL-Train` + `DL-Val`) and caches."""
    if sha256(input_dir / "source_manifest.json") != MANIFEST_FILE_SHA:
        raise RuntimeError("source_manifest.json file SHA mismatch")
    manifest = json.loads((input_dir / "source_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("source_manifest_sha256") != MANIFEST_ID
        or manifest.get("excluded_dl_holdout_count") != 93
        or manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("source_manifest isolation audit failed")

    selected_path = input_dir / "selected.json"
    champions_path = input_dir / "champions.json"
    fused_source_path = input_dir / "cpu_exact_fused.py"
    if sha256(selected_path) != SELECTED_FILE_SHA:
        raise RuntimeError("selected.json file SHA mismatch")
    if sha256(champions_path) != CHAMPIONS_FILE_SHA:
        raise RuntimeError("champions.json file SHA mismatch")
    if sha256(fused_source_path) != FUSED_SOURCE_SHA:
        raise RuntimeError("cpu_exact_fused.py file SHA mismatch")
    if sha256(base.evaluator.WORKER_BIN) != WORKER_BIN_SHA:
        raise RuntimeError("bootstrap.s1main worker binary SHA mismatch")
    pinned_micro = verify_pinned_micro_assets()

    cache_manifest_path = cache_dir / "cache_manifest.json"
    if not cache_manifest_path.exists():
        raise RuntimeError(f"missing cache_manifest.json at {cache_manifest_path}")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    ctx_npz_path = cache_dir / "context_features.npz"
    priors_npz_path = cache_dir / "pool_priors.npz"
    journal_path = cache_dir / "journal_candidates.json"
    champs_cache_path = cache_dir / "champions_featurized.json"
    jtt1_gz_manifest_path = cache_dir / "jtt1_gz_manifest.json"
    if (
        cache_manifest.get("source_manifest_file_sha256") != MANIFEST_FILE_SHA
        or cache_manifest.get("source_manifest_sha256") != MANIFEST_ID
        or cache_manifest.get("selected_file_sha256") != SELECTED_FILE_SHA
        or cache_manifest.get("champions_file_sha256") != CHAMPIONS_FILE_SHA
        or cache_manifest.get("fused_source_sha256") != FUSED_SOURCE_SHA
        or cache_manifest.get("worker_bin_sha256") != WORKER_BIN_SHA
        or cache_manifest.get("dope_sha256_before") != DOPE_SHA
        or cache_manifest.get("dope_sha256_after") != DOPE_SHA
        or cache_manifest.get("stage1_stage2_pt_sha256") != STAGE1_STAGE2_PT_SHA
        or cache_manifest.get("dope_multitask_pt_sha256") != DOPE_MULTITASK_PT_SHA
        or cache_manifest.get("prog_feat_names_sha256") != PROG_FEAT_NAMES_SHA
        or cache_manifest.get("rust_evaluator_py_sha256") != RUST_EVALUATOR_PY_SHA
        or cache_manifest.get("model_py_sha256") != MODEL_PY_SHA
        or cache_manifest.get("phase12_mm_py_sha256") != PHASE12_MM_PY_SHA
        or cache_manifest.get("excluded_payloads_opened") != 0
        or cache_manifest.get("outer_rows_delivered") != 0
        or sha256(ctx_npz_path) != cache_manifest.get("cache_npz_sha256")
        or sha256(priors_npz_path) != cache_manifest.get("pool_priors_npz_sha256")
        or sha256(journal_path) != cache_manifest.get("journal_candidates_sha256")
        or sha256(champs_cache_path) != cache_manifest.get("champions_featurized_sha256")
        or not jtt1_gz_manifest_path.is_file()
        or sha256(jtt1_gz_manifest_path) != cache_manifest.get("jtt1_gz_manifest_sha256")
        or any(cache_manifest.get(k) != v for k, v in pinned_micro.items())
    ):
        raise RuntimeError("cache_manifest cryptographic verification failed")
    jtt1_gz_sha_by_cid = json.loads(jtt1_gz_manifest_path.read_text(encoding="utf-8"))

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    train_rows = selected.get("train", [])
    val_rows = selected.get("dev", [])
    if len(train_rows) != 1219 or len(val_rows) != 91:
        raise RuntimeError("cohort count mismatch")
    train_cids_set = {r["context_sha256"] for r in train_rows}
    val_cids_set = {r["context_sha256"] for r in val_rows}
    train_fams_set = {r["family_id"] for r in train_rows}
    val_fams_set = {r["family_id"] for r in val_rows}
    if len(train_cids_set) != 1219 or len(val_cids_set) != 91:
        raise RuntimeError("duplicate context IDs in selected.json")
    if len(train_fams_set) != 205 or len(val_fams_set) != 19:
        raise RuntimeError("family count mismatch in selected.json")
    if (train_cids_set & val_cids_set) or (train_fams_set & val_fams_set):
        raise RuntimeError("train/val context or family overlap detected")
    if set(jtt1_gz_sha_by_cid.keys()) != train_cids_set:
        raise RuntimeError("jtt1_gz_manifest.json context set mismatch vs DL-Train")
    if any(r.get("dl_split") != "DL-Train" or r.get("task") != "regression" for r in train_rows):
        raise RuntimeError("non-DL-Train or non-regression row in train split")
    if any(r.get("dl_split") != "DL-Val" or r.get("task") != "regression" for r in val_rows):
        raise RuntimeError("non-DL-Val or non-regression row in dev split")

    all_rows = train_rows + val_rows
    cids = [r["context_sha256"] for r in all_rows]
    family_by_cid = {r["context_sha256"]: r["family_id"] for r in all_rows}

    ctx_npz = np.load(ctx_npz_path, allow_pickle=False)
    priors_npz = np.load(priors_npz_path, allow_pickle=False)
    if list(ctx_npz["cids"]) != cids or list(priors_npz["cids"]) != cids:
        raise RuntimeError("cache CID ordering mismatch")

    champions_list = json.loads(champs_cache_path.read_text(encoding="utf-8"))
    champions_by_cid: dict[str, list[dict]] = {}
    for ch in champions_list:
        champions_by_cid.setdefault(ch["context_sha256"], []).append(ch)

    ctx_cache: dict[str, dict[str, torch.Tensor]] = {}
    base_pools: dict[str, dict] = {}

    tokens_all = ctx_npz["tokens_10x32"]
    masks_all = ctx_npz["mask_10"]
    lm48_all = ctx_npz["lm48_z"]
    ctx_feat_all = ctx_npz["ctx_feat_584"]
    offsets_all = priors_npz["offsets"]
    base_prior_all = priors_npz["base_prior_cost"]
    sos_prior_all = priors_npz["sos_prior"]
    scale_all = priors_npz["scale"]
    proj16_all = priors_npz["proj16"]

    jtt1_gz_dir = cache_dir / "jtt1_gz"

    for idx, row in enumerate(all_rows):
        cid = row["context_sha256"]
        lo, hi = int(offsets_all[idx]), int(offsets_all[idx + 1])
        ctx_cache[cid] = {
            "tokens_10x32": torch.as_tensor(tokens_all[idx], dtype=torch.float32, device=device),
            "mask_10": torch.as_tensor(masks_all[idx], dtype=torch.bool, device=device),
            "ctx_feat_584": torch.as_tensor(ctx_feat_all[idx], dtype=torch.float32, device=device),
        }
        pool_path = input_dir / "pools" / f"{cid}.npz"
        if sha256(pool_path) != row["pool_sha256"]:
            raise RuntimeError(f"pool SHA256 mismatch for {cid}")
        with np.load(pool_path, allow_pickle=False) as src:
            programs = src["program"].astype(np.float32)
            pids = [
                v.decode() if isinstance(v, bytes) else str(v)
                for v in src["program_sha256"].tolist()
            ]
            fitness = src["fitness"].astype(np.float64)
            logratio = src["logratio"].astype(np.float64)
            gate = src["gate"].astype(bool)
        if hi - lo != len(pids):
            raise RuntimeError(f"cached prior slice mismatch for {cid}: {hi - lo} != {len(pids)}")
        tpath = input_dir / "tables" / f"{cid}.jtt1"
        tbytes = int(tpath.stat().st_size) if tpath.exists() else 32817
        if cid in train_cids_set:
            gz_path = jtt1_gz_dir / f"{cid}.jtt1.gz"
            expected_gz_sha = jtt1_gz_sha_by_cid[cid]
            if not gz_path.is_file() or sha256(gz_path) != expected_gz_sha:
                raise RuntimeError(f"jtt1.gz missing or SHA256 mismatch for DL-Train context {cid}")
        else:
            gz_path = None
            expected_gz_sha = None
        base_pools[cid] = {
            "rec": row,
            "pids": pids,
            "program": programs,
            "proj16": proj16_all[lo:hi].astype(np.float32),
            "fitness": fitness,
            "logratio": logratio,
            "gate": gate,
            "base_prior_cost": base_prior_all[lo:hi].astype(np.float32),
            "sos_prior": sos_prior_all[lo:hi].astype(np.float32),
            "scale": float(scale_all[idx]),
            "lm48_z": lm48_all[idx].astype(np.float32),
            "jtt1_gz_path": gz_path,
            "jtt1_gz_sha256": expected_gz_sha,
            "champions": champions_by_cid.get(cid, []),
            "table_bytes": tbytes,
            "hard_negatives": set(),
        }

    journal_records = json.loads(journal_path.read_text(encoding="utf-8"))
    journal_by_cid: dict[str, list[dict]] = {}
    for rec in journal_records:
        journal_by_cid.setdefault(rec["context_sha256"], []).append(rec)

    spec = importlib.util.spec_from_file_location("h58_runner_fused", fused_source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load cpu_exact_fused.py")
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)

    net = fused.FusedJopeNetwork(device=device, adapter_variant="h40b")
    for p in net.parameters():
        p.requires_grad_(False)
    base.guard.lock_dope_weights_and_eval_mode(net)
    if base.guard.snapshot_dope_weights(net).composite_sha256 != DOPE_SHA:
        raise RuntimeError("DOPE SHA mismatch after loading frozen backbone")

    return train_rows, val_rows, family_by_cid, base_pools, ctx_cache, journal_by_cid, net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path.home() / "jope-data/h50/input")
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / "jope-data/h58/cache")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "jope-data/h58/runs")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--steps-per-round", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--slate-size", type=int, default=192)
    parser.add_argument("--lock-held", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    if args.device == "cuda:0":
        if not args.lock_held:
            raise RuntimeError("GPU execution of H58 requires --lock-held under qualification.lock")
        verify_qualification_lock_held()
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(0.90)
    torch.set_num_threads(4)

    device = torch.device(args.device)
    train_rows, val_rows, family_by_cid, base_pools, ctx_cache, journal_by_cid, net = (
        load_h58_packet_and_cache(args.input, args.cache_dir, device)
    )
    train_ids = [r["context_sha256"] for r in train_rows]
    val_ids = [r["context_sha256"] for r in val_rows]
    folds = family_folds(train_rows, count=5, seed=FOLD_SEED)

    oof_native_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    oof_unrev_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    oof_native_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_unrev_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_native_reg: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_unrev_reg: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_native_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_unrev_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    oof_spearman_rho: dict[str, list[float]] = {a: [] for a in ARMS}

    val_native_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    val_unrev_ctx: dict[str, dict[int, dict[str, float]]] = {a: {} for a in ARMS}
    val_native_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    val_unrev_viol: dict[str, list[float]] = {a: [] for a in ARMS}
    val_native_reg: dict[str, list[float]] = {a: [] for a in ARMS}
    val_unrev_reg: dict[str, list[float]] = {a: [] for a in ARMS}
    val_native_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    val_unrev_hit5: dict[str, list[float]] = {a: [] for a in ARMS}
    val_spearman_rho: dict[str, list[float]] = {a: [] for a in ARMS}

    cell_receipts: list[dict] = []

    # Phase 1: 5-Fold Family-Grouped Inner CV on DL-Train ONLY (DL-Val untouched)
    for seed in args.seeds:
        for arm in ARMS:
            seed_oof_native: dict[str, float] = {}
            seed_oof_unrev: dict[str, float] = {}
            fold_viols_native, fold_viols_unrev = [], []
            fold_regs_native, fold_regs_unrev = [], []
            fold_hit5_native, fold_hit5_unrev = [], []
            fold_rhos = []

            for fold_idx, held_fams in enumerate(folds):
                fit_cids = [cid for cid in train_ids if family_by_cid[cid] not in held_fams]
                held_cids = [cid for cid in train_ids if family_by_cid[cid] in held_fams]
                res = run_training_cell(
                    arm=arm,
                    seed=seed,
                    cell_id=fold_idx + 1,
                    net=net,
                    base_pools=base_pools,
                    fit_cids=fit_cids,
                    eval_cids=held_cids,
                    family_by_cid=family_by_cid,
                    ctx_cache=ctx_cache,
                    journal_by_cid=journal_by_cid,
                    rounds=args.rounds,
                    steps_per_round=args.steps_per_round,
                    batch_size=args.batch_size,
                    slate_size=args.slate_size,
                    device=device,
                )
                seed_oof_native.update(res["eval_native"]["per_context"])
                seed_oof_unrev.update(res["eval_unrev"]["per_context"])
                fold_viols_native.append(res["eval_native"]["violation_rate"] * len(held_cids))
                fold_viols_unrev.append(res["eval_unrev"]["violation_rate"] * len(held_cids))
                fold_regs_native.append(res["eval_native"]["mean_regret_norm"] * len(held_cids))
                fold_regs_unrev.append(res["eval_unrev"]["mean_regret_norm"] * len(held_cids))
                fold_hit5_native.append(res["eval_native"]["hit5_rate"] * len(held_cids))
                fold_hit5_unrev.append(res["eval_unrev"]["hit5_rate"] * len(held_cids))
                fold_rhos.append(res["eval_unrev"]["spearman_rho"] * len(held_cids))

                receipt = {k: v for k, v in res.items() if k not in ("eval_native", "eval_unrev")}
                receipt["split"] = f"inner_fold_{fold_idx}"
                receipt["native_family_utility"] = res["eval_native"]["mean"]
                receipt["unrev_family_utility"] = res["eval_unrev"]["mean"]
                receipt["native_violation_rate"] = res["eval_native"]["violation_rate"]
                receipt["unrev_violation_rate"] = res["eval_unrev"]["violation_rate"]
                receipt["native_mean_regret_norm"] = res["eval_native"]["mean_regret_norm"]
                receipt["unrev_mean_regret_norm"] = res["eval_unrev"]["mean_regret_norm"]
                receipt["spearman_rho"] = res["eval_unrev"]["spearman_rho"]
                cell_receipts.append(receipt)
                print(
                    f"[H58] seed={seed} arm={arm} fold={fold_idx} "
                    f"native_u={res['eval_native']['mean']:.4f} "
                    f"unrev_u={res['eval_unrev']['mean']:.4f} "
                    f"viol={res['eval_unrev']['violation_rate']*100:.2f}% "
                    f"rho={res['eval_unrev']['spearman_rho']:.4f} "
                    f"gp_eval={res['gp_evaluated']} gp_admit={res['gp_admitted']} "
                    f"({res['elapsed_seconds']:.1f}s)",
                    flush=True,
                )

            if len(seed_oof_native) != len(train_ids):
                raise RuntimeError("incomplete out-of-fold predictions on DL-Train")
            oof_native_ctx[arm][seed] = seed_oof_native
            oof_unrev_ctx[arm][seed] = seed_oof_unrev
            oof_native_viol[arm].append(sum(fold_viols_native) / len(train_ids))
            oof_unrev_viol[arm].append(sum(fold_viols_unrev) / len(train_ids))
            oof_native_reg[arm].append(sum(fold_regs_native) / len(train_ids))
            oof_unrev_reg[arm].append(sum(fold_regs_unrev) / len(train_ids))
            oof_native_hit5[arm].append(sum(fold_hit5_native) / len(train_ids))
            oof_unrev_hit5[arm].append(sum(fold_hit5_unrev) / len(train_ids))
            oof_spearman_rho[arm].append(sum(fold_rhos) / len(train_ids))

    # Phase 2: Single Terminal DL-Val Evaluation (1 evaluation sweep per trained (arm, seed) at round R)
    for seed in args.seeds:
        for arm in ARMS:
            term_res = run_training_cell(
                arm=arm,
                seed=seed,
                cell_id=99,
                net=net,
                base_pools=base_pools,
                fit_cids=train_ids,
                eval_cids=val_ids,
                family_by_cid=family_by_cid,
                ctx_cache=ctx_cache,
                journal_by_cid=journal_by_cid,
                rounds=args.rounds,
                steps_per_round=args.steps_per_round,
                batch_size=args.batch_size,
                slate_size=args.slate_size,
                device=device,
            )
            val_native_ctx[arm][seed] = term_res["eval_native"]["per_context"]
            val_unrev_ctx[arm][seed] = term_res["eval_unrev"]["per_context"]
            val_native_viol[arm].append(term_res["eval_native"]["violation_rate"])
            val_unrev_viol[arm].append(term_res["eval_unrev"]["violation_rate"])
            val_native_reg[arm].append(term_res["eval_native"]["mean_regret_norm"])
            val_unrev_reg[arm].append(term_res["eval_unrev"]["mean_regret_norm"])
            val_native_hit5[arm].append(term_res["eval_native"]["hit5_rate"])
            val_unrev_hit5[arm].append(term_res["eval_unrev"]["hit5_rate"])
            val_spearman_rho[arm].append(term_res["eval_unrev"]["spearman_rho"])

            t_receipt = {k: v for k, v in term_res.items() if k not in ("eval_native", "eval_unrev")}
            t_receipt["split"] = "terminal_dl_val"
            t_receipt["terminal_dl_val_eval_count"] = 1
            t_receipt["native_family_utility"] = term_res["eval_native"]["mean"]
            t_receipt["unrev_family_utility"] = term_res["eval_unrev"]["mean"]
            t_receipt["native_violation_rate"] = term_res["eval_native"]["violation_rate"]
            t_receipt["unrev_violation_rate"] = term_res["eval_unrev"]["violation_rate"]
            t_receipt["native_mean_regret_norm"] = term_res["eval_native"]["mean_regret_norm"]
            t_receipt["unrev_mean_regret_norm"] = term_res["eval_unrev"]["mean_regret_norm"]
            t_receipt["spearman_rho"] = term_res["eval_unrev"]["spearman_rho"]
            cell_receipts.append(t_receipt)
            print(
                f"[H58] seed={seed} arm={arm} TERMINAL DL-Val "
                f"native_u={term_res['eval_native']['mean']:.4f} "
                f"unrev_u={term_res['eval_unrev']['mean']:.4f} "
                f"viol={term_res['eval_unrev']['violation_rate']*100:.2f}% "
                f"rho={term_res['eval_unrev']['spearman_rho']:.4f} "
                f"gp_eval={term_res['gp_evaluated']} gp_admit={term_res['gp_admitted']} "
                f"({term_res['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    # Deferred terminal diagnostic reference for fold-clean initialization (INIT_G0, 0 training steps)
    # Out-of-fold INIT_G0 on DL-Train is evaluated fold-by-fold using each fold's fit_cids normalization.
    init_oof_ctx: dict[int, dict[str, float]] = {}
    init_val_ctx: dict[int, dict[str, float]] = {}
    for seed in args.seeds:
        seed_init_oof: dict[str, float] = {}
        for held_fams in folds:
            fit_cids = [cid for cid in train_ids if family_by_cid[cid] not in held_fams]
            held_cids = [cid for cid in train_ids if family_by_cid[cid] in held_fams]
            reset_adapter_fold_clean(
                net,
                seed=seed,
                freeze_parent_delta=True,
                scale=0.1,
                fit_pools=base_pools,
                fit_cids=fit_cids,
            )
            net.eval()
            pred_fn_fold = make_predictor_arm(net, base_pools, ctx_cache, device, "B")
            _, held_unrev = evaluate_utility_both_modes(held_cids, base_pools, pred_fn_fold, arm="B")
            seed_init_oof.update(held_unrev["per_context"])
        if len(seed_init_oof) != len(train_ids):
            raise RuntimeError("incomplete INIT_G0 OOF predictions on DL-Train")
        init_oof_ctx[seed] = seed_init_oof

        reset_adapter_fold_clean(
            net,
            seed=seed,
            freeze_parent_delta=True,
            scale=0.1,
            fit_pools=base_pools,
            fit_cids=train_ids,
        )
        net.eval()
        pred_fn_val = make_predictor_arm(net, base_pools, ctx_cache, device, "B")
        _, va_unrev = evaluate_utility_both_modes(val_ids, base_pools, pred_fn_val, arm="B")
        init_val_ctx[seed] = va_unrev["per_context"]

    def aggregate_split(
        cids_split: list[str],
        ctx_by_arm_seed: dict[str, dict[int, dict[str, float]]],
        init_by_seed: dict[int, dict[str, float]],
    ):
        mean_ctx_by_arm: dict[str, dict[str, float]] = {}
        mean_fam_by_arm: dict[str, dict[str, float]] = {}
        contenders = list(ARMS) + ["INIT_G0"]
        for c in contenders:
            src = init_by_seed if c == "INIT_G0" else ctx_by_arm_seed[c]
            mean_ctx = {
                cid: float(np.mean([src[s][cid] for s in args.seeds]))
                for cid in cids_split
            }
            mean_ctx_by_arm[c] = mean_ctx
            by_f: dict[str, list[float]] = {}
            for cid, val in mean_ctx.items():
                by_f.setdefault(family_by_cid[cid], []).append(val)
            mean_fam_by_arm[c] = {f: float(np.mean(vs)) for f, vs in sorted(by_f.items())}
        return mean_ctx_by_arm, mean_fam_by_arm

    oof_unrev_mean_ctx, oof_unrev_mean_fam = aggregate_split(train_ids, oof_unrev_ctx, init_oof_ctx)
    oof_native_mean_ctx, oof_native_mean_fam = aggregate_split(train_ids, oof_native_ctx, init_oof_ctx)
    val_unrev_mean_ctx, val_unrev_mean_fam = aggregate_split(val_ids, val_unrev_ctx, init_val_ctx)
    val_native_mean_ctx, val_native_mean_fam = aggregate_split(val_ids, val_native_ctx, init_val_ctx)

    pairs = [("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "C"), ("D", "B"), ("B", "INIT_G0"), ("C", "INIT_G0"), ("D", "INIT_G0")]

    def compute_pair_gates(mean_fam_dict: dict[str, dict[str, float]], seed_base: int) -> dict[str, dict]:
        out = {}
        for idx, (lhs, rhs) in enumerate(pairs):
            fams = sorted(mean_fam_dict[lhs].keys())
            gains = {f: mean_fam_dict[lhs][f] - mean_fam_dict[rhs][f] for f in fams}
            out[f"{lhs}_vs_{rhs}"] = bootstrap_t537(gains, seed=seed_base + idx * 101)
        return out

    oof_gates_unrev = compute_pair_gates(oof_unrev_mean_fam, 20261030)
    oof_gates_native = compute_pair_gates(oof_native_mean_fam, 20262030)
    val_gates_unrev = compute_pair_gates(val_unrev_mean_fam, 20263030)
    val_gates_native = compute_pair_gates(val_native_mean_fam, 20264030)

    oof_elo_unrev = compute_elo_panel(train_ids, family_by_cid, oof_unrev_mean_ctx, anchor="A", seed=20265001)
    oof_elo_native = compute_elo_panel(train_ids, family_by_cid, oof_native_mean_ctx, anchor="A", seed=20265002)
    val_elo_unrev = compute_elo_panel(val_ids, family_by_cid, val_unrev_mean_ctx, anchor="A", seed=20265003)
    val_elo_native = compute_elo_panel(val_ids, family_by_cid, val_native_mean_ctx, anchor="A", seed=20265004)

    summary = {
        "schema": "jope.h58.wave-summary/v3",
        "wave": "H58",
        "task_id": "t550",
        "device": str(device),
        "seeds": list(args.seeds),
        "rounds": args.rounds,
        "steps_per_round": args.steps_per_round,
        "optimizer_steps_per_cell": args.rounds * args.steps_per_round,
        "batch_size": args.batch_size,
        "slate_size": args.slate_size,
        "manifest_file_sha256": MANIFEST_FILE_SHA,
        "manifest_id": MANIFEST_ID,
        "selected_file_sha256": SELECTED_FILE_SHA,
        "champions_file_sha256": CHAMPIONS_FILE_SHA,
        "fused_source_sha256": FUSED_SOURCE_SHA,
        "worker_bin_sha256": WORKER_BIN_SHA,
        "dope_sha256": DOPE_SHA,
        "stage1_stage2_pt_sha256": STAGE1_STAGE2_PT_SHA,
        "dope_multitask_pt_sha256": DOPE_MULTITASK_PT_SHA,
        "prog_feat_names_sha256": PROG_FEAT_NAMES_SHA,
        "rust_evaluator_py_sha256": RUST_EVALUATOR_PY_SHA,
        "model_py_sha256": MODEL_PY_SHA,
        "phase12_mm_py_sha256": PHASE12_MM_PY_SHA,
        "excluded_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "terminal_dl_val_eval_sweeps_per_trained_arm_seed": 1,
        "cpu_throttle_guard": base.evaluator.GLOBAL_CPU_GUARD.summary(),
        "dl_train_oof_205_families": {
            "arm_summary": {
                c: {
                    "unrev_family_utility_mean": float(np.mean(list(oof_unrev_mean_fam[c].values()))),
                    "unrev_family_utility_median": float(np.median(list(oof_unrev_mean_fam[c].values()))),
                    "native_family_utility_mean": float(np.mean(list(oof_native_mean_fam[c].values()))),
                    "native_family_utility_median": float(np.median(list(oof_native_mean_fam[c].values()))),
                    "unrev_violation_rate_mean": (
                        float(np.mean(oof_unrev_viol[c])) if c in ARMS else None
                    ),
                    "native_violation_rate_mean": (
                        float(np.mean(oof_native_viol[c])) if c in ARMS else None
                    ),
                    "unrev_mean_regret_norm": (
                        float(np.mean(oof_unrev_reg[c])) if c in ARMS else None
                    ),
                    "native_mean_regret_norm": (
                        float(np.mean(oof_native_reg[c])) if c in ARMS else None
                    ),
                    "unrev_hit5_rate_mean": (
                        float(np.mean(oof_unrev_hit5[c])) if c in ARMS else None
                    ),
                    "native_hit5_rate_mean": (
                        float(np.mean(oof_native_hit5[c])) if c in ARMS else None
                    ),
                    "spearman_rho_mean": (
                        float(np.mean(oof_spearman_rho[c])) if c in ARMS else None
                    ),
                    "unrev_elo": oof_elo_unrev["ratings"][c],
                    "unrev_elo_ci95": oof_elo_unrev["elo_ci95"][c],
                    "native_elo": oof_elo_native["ratings"][c],
                    "native_elo_ci95": oof_elo_native["elo_ci95"][c],
                }
                for c in list(ARMS) + ["INIT_G0"]
            },
            "t537_gates_unrev": oof_gates_unrev,
            "t537_gates_native": oof_gates_native,
            "bradley_terry_elo_unrev": oof_elo_unrev,
            "bradley_terry_elo_native": oof_elo_native,
        },
        "dl_val_terminal_19_families": {
            "arm_summary": {
                c: {
                    "unrev_family_utility_mean": float(np.mean(list(val_unrev_mean_fam[c].values()))),
                    "unrev_family_utility_median": float(np.median(list(val_unrev_mean_fam[c].values()))),
                    "native_family_utility_mean": float(np.mean(list(val_native_mean_fam[c].values()))),
                    "native_family_utility_median": float(np.median(list(val_native_mean_fam[c].values()))),
                    "unrev_violation_rate_mean": (
                        float(np.mean(val_unrev_viol[c])) if c in ARMS else None
                    ),
                    "native_violation_rate_mean": (
                        float(np.mean(val_native_viol[c])) if c in ARMS else None
                    ),
                    "unrev_mean_regret_norm": (
                        float(np.mean(val_unrev_reg[c])) if c in ARMS else None
                    ),
                    "native_mean_regret_norm": (
                        float(np.mean(val_native_reg[c])) if c in ARMS else None
                    ),
                    "unrev_hit5_rate_mean": (
                        float(np.mean(val_unrev_hit5[c])) if c in ARMS else None
                    ),
                    "native_hit5_rate_mean": (
                        float(np.mean(val_native_hit5[c])) if c in ARMS else None
                    ),
                    "spearman_rho_mean": (
                        float(np.mean(val_spearman_rho[c])) if c in ARMS else None
                    ),
                    "unrev_elo": val_elo_unrev["ratings"][c],
                    "unrev_elo_ci95": val_elo_unrev["elo_ci95"][c],
                    "native_elo": val_elo_native["ratings"][c],
                    "native_elo_ci95": val_elo_native["elo_ci95"][c],
                }
                for c in list(ARMS) + ["INIT_G0"]
            },
            "t537_gates_unrev": val_gates_unrev,
            "t537_gates_native": val_gates_native,
            "bradley_terry_elo_unrev": val_elo_unrev,
            "bradley_terry_elo_native": val_elo_native,
        },
        "cell_count": len(cell_receipts),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out_dir / "wave_summary.json", summary)
    atomic_json(args.out_dir / "cell_receipts.json", {"cells": cell_receipts})
    print(json.dumps(summary["dl_train_oof_205_families"]["arm_summary"], indent=2), flush=True)
    print(json.dumps(summary["dl_val_terminal_19_families"]["arm_summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
