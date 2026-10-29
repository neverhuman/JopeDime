#!/usr/bin/env python3
"""
Phase 4c: Multi-Round Self-Play Reinforcement Learning Flywheel (`selfplay_loop.py`).

Executes the 4-round self-play flywheel (`Round 0` -> `Round 1` -> `Round 2` -> `Round 3`)
for regression datasets, combining:
  - `FlywheelGameNet` (PyTorch on GPU `cuda:0`)
  - Pre-computed dataset encodings (`A0` 88D, `Dope-112` 112D, `Kumo-S`/`Foundation`/`Hyperion-V6`/`TabPFN35`/`Target` 320D)
  - `Landmark-V2` 12-probe 5-fold CV basket (48D) + `H27` 8-probe fingerprints (16D)
  - Leave-One-Family-Out Empirical-Bayes `SoS` program prior
  - CV-Test Agreement Gate (`sign_agree` and `|log(r_test) - log(r_cv)| <= 0.35`)
  - Live GP -> Rust evaluation (`bootstrap.s1main --fit-s1` & `--fit`) + `EditZero` AST mutation
    + Cross-Dataset Champion Retrieval + standalone Rust transpilation (`rustc --crate-type lib`)
  - Canonical S3 persistence (`jope.hwm/v1`, `jope.score/v1`, `gp_distilled/v1`) with smart
    retention garbage pruning (`jope-learning::retention` parity).

Evaluates and verifies:
  1. Exact 850-context 162-component `H27`/`H30`/`W1` benchmark (3 rollout seeds, budgets B=8,16,32,64,
     100,000-replicate paired cluster-bootstrap 98.75% CIs).
  2. Full 1,403-context regression corpus family-disjoint `DL-Val` (91 contexts) and `DL-Holdout` (93 contexts).
  3. Live GP -> Rust self-play discovery across Rounds 1, 2, 3 (including unscored `outer`-split contexts).
"""

import concurrent.futures
import hashlib
import json
import math
import os
import uuid

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import random
import struct
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)

sys.dont_write_bytecode = True
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from evo_batch_queue import AsyncEvoBatchQueue, build_base_gpu_slate_batch
from model import FlywheelGameNet, compute_directional_evo_loss, extract_head_and_stage_signatures
from rust_evaluator import (
    GLOBAL_CPU_GUARD,
    evaluate_gp_batch_rust,
    evolve_context_slate_generations,
    featurize_dsl_412,
    get_w1_sign_matrix,
    propose_flywheel_candidates,
    transpile_dsl_to_rust_code,
    verify_rust_compilation,
)
from s3_persistence import apply_smart_retention_filter, persist_round_artifacts_to_s3, sync_directory_to_s3
CORPUS_DIR = Path("/mnt/jope-archive/flywheel-20261023/corpus")
RESULTS_DIR = Path("/mnt/jope-archive/flywheel-20261023/results")
CHECKPOINTS_DIR = Path("/mnt/jope-archive/flywheel-20261023/checkpoints")
EMITTED_RUST_DIR = Path("/mnt/jope-archive/flywheel-20261023/emitted_rust")

SEEDS = (20261101, 20261102, 20261103)
BUDGETS = (8, 16, 32, 64)
WARM = 4
W1_PREFIX = b"jope-w1-adaptive-replay-v1\0"
IO_STARTUP_READS = None
if os.environ.get("JOPE_TELEMETRY_RUN_DIR"):
    from jope.telemetry.io_hooks import LocalReadObserver

    IO_STARTUP_READS = LocalReadObserver()


def _open_startup_npz(path: Path):
    if IO_STARTUP_READS is not None:
        return IO_STARTUP_READS.open_npz(path)
    return np.load(path, allow_pickle=False)


def warmstart_indices(context_sha256: str, pids: list[str], seed: int) -> list[int]:
    """Exact W1 4-candidate deterministic hash warm-start."""
    by_pid = {p: i for i, p in enumerate(pids)}
    chosen = sorted(
        pids,
        key=lambda p: (
            hashlib.sha256(
                W1_PREFIX + b"warm\0" + context_sha256.encode() + b"\0" + str(seed).encode() + b"\0" + p.encode()
            ).digest(),
            p,
        ),
    )[:WARM]
    return [by_pid[p] for p in chosen]


def shuffled_raw_f_map(context_sha256: str, obs_pids: list[str], obs_f: list[float], seed: int) -> dict[str, float]:
    """Exact W1 deterministic label-blind derangement of revealed outcomes."""
    pairs = sorted(zip(obs_pids, obs_f), key=lambda x: x[0])
    if len(pairs) < 2:
        return {p: f for p, f in pairs}
    key = W1_PREFIX + b"shuffle\0" + context_sha256.encode() + b"\0" + str(seed).encode() + b"\0" + str(len(pairs)).encode()
    offset = 1 + (int.from_bytes(hashlib.sha256(key).digest(), "big") % (len(pairs) - 1))
    return {pairs[j][0]: pairs[(j + offset) % len(pairs)][1] for j in range(len(pairs))}


def load_all_pools_and_features():
    """Load corpus manifest, context features, H30 priors, and candidate pools into RAM."""
    t0 = time.monotonic()
    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    bank_path = CORPUS_DIR / "champion_dsl_bank.json"
    if IO_STARTUP_READS is not None:
        manifest_text = IO_STARTUP_READS.read(manifest_path, manifest_path.read_text)
        bank_text = IO_STARTUP_READS.read(bank_path, bank_path.read_text)
    else:
        manifest_text = manifest_path.read_text()
        bank_text = bank_path.read_text()
    manifest = json.loads(manifest_text)
    champ_bank = json.loads(bank_text)
    with _open_startup_npz(CORPUS_DIR / "context_features.npz") as cf:
        a0 = cf["a0"]                  # (1780, 88)
        dope112 = cf["dope112"]        # (1780, 112)
        mm_proj = cf["mm_proj"]        # (1780, 320)
        h27_fp16 = cf["h27_fp16"]      # (1780, 16)
        landmark_v2 = cf["landmark_v2_48"]  # (1780, 48)

    # Concatenate into 584D raw context feature matrix:
    # [0:88] a0, [88:200] dope112, [200:520] mm_proj, [520:536] h27_fp16, [536:584] landmark_v2
    ctx_raw_584 = np.hstack([a0, dope112, mm_proj, h27_fp16, landmark_v2]).astype(np.float32)

    h30_priors = {}
    h30_scales = {0: 0.37918042500067195, 1: 0.46251714075224865}
    with _open_startup_npz(CORPUS_DIR / "h30_zero_priors_850.npz") as hp:
        scales_arr = hp["scales"]
        h30_scales = {0: float(scales_arr[0]), 1: float(scales_arr[1])}
        for k in hp.files:
            if k.startswith("prior_"):
                h30_priors[k[6:]] = hp[k].astype(np.float64)

    w1_sign = get_w1_sign_matrix()

    def _load_one_pool(rec: dict):
        if rec["n_candidates"] == 0:
            return None
        cid = rec["context_sha256"]
        p_path = Path(rec["pool_path"])
        if not p_path.exists():
            return None
        with _open_startup_npz(p_path) as pz:
            prog = pz["program"].astype(np.float32)
            fitness = pz["fitness"].astype(np.float64)
            logratio = pz["logratio"].astype(np.float64)
            gate = pz["gate"].astype(np.bool_)
            pids = [x.decode("ascii") for x in pz["program_sha256"].tolist()]

        # Compute 16D W1 sign projection in exact FP64 order
        prog64 = prog.astype(np.float64)
        proj16 = np.zeros((len(prog64), 16), dtype=np.float64)
        for i in range(412):
            proj16 += prog64[:, i : i + 1] * w1_sign[i : i + 1, :]
        proj16 /= math.sqrt(412.0)

        # Compute 24D structural signature in pure NumPy inside ThreadPoolExecutor (avoids per-thread PyTorch OpenMP pool bloat)
        pad = np.pad(prog, ((0, 0), (0, 20)), mode="constant")
        folded = pad.reshape(len(prog), 18, 24)
        sig = folded.mean(axis=1) + 0.5 * folded.max(axis=1)
        sig_norm = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
        struct_sig = (sig / sig_norm).astype(np.float32)

        if cid in h30_priors:
            prior_cost = h30_priors[cid]
            b850 = rec["benchmark_850"]
            fold_idx = 0 if b850["fold0_role"] == "OOF" else 1
            scale = h30_scales[fold_idx]
        else:
            # Deterministic complexity + linear ridge prior fallback for non-850 contexts
            scale = 0.4208
            prior_cost = 0.02 * (prog[:, 0].astype(np.float64) - float(np.mean(prog[:, 0])))

        return cid, {
            "rec": rec,
            "program": prog,
            "proj16": proj16,
            "struct_sig": struct_sig,
            "fitness": fitness,
            "logratio": logratio,
            "gate": gate,
            "pids": pids,
            "prior_cost": prior_cost,
            "scale": scale,
        }

    pools = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for res in ex.map(_load_one_pool, manifest["records"]):
            if res is not None:
                pools[res[0]] = res[1]

    print(f"[Phase 4] Loaded {len(pools)} non-empty pools and 584D context features in {time.monotonic() - t0:.2f}s.")
    return manifest, champ_bank, ctx_raw_584, h30_scales, pools


def build_fold_safe_sos_priors(
    train_cids: list[str],
    eval_cids: list[str],
    pools: dict,
) -> dict[str, np.ndarray]:
    """
    Build Leave-One-Family-Out Empirical-Bayes Specificity-over-Popularity (`SoS`) program
    priors AND k-NN structural AST neighborhood priors strictly from `train_cids`.
    Returns `sos_by_cid[cid]` of shape `(N, 2)`:
      col 0: empirical-bayes shrunk centered normalized fitness offset (negative = better)
      col 1: confidence weight in `[0, 1]`
    """
    # Group training program appearances by program_sha256 and family_id
    prog_stats = {}  # pid -> list of (family_id, centered_norm_f)
    # Build deduplicated structural template index of top training programs (max 256 unique templates)
    seen_template_pids = set()
    struct_vecs = []
    struct_vals = []

    for cid in train_cids:
        if cid not in pools:
            continue
        p = pools[cid]
        fam = p["rec"]["family_id"]
        f_arr = p["fitness"]
        g_arr = p["gate"]
        s = p["scale"]
        # Center passing fitness within context so context difficulty doesn't leak
        pass_f = f_arr[g_arr]
        ctx_med = float(np.median(pass_f)) if len(pass_f) > 0 else 0.0
        norm_f = np.where(g_arr, (f_arr - ctx_med) / s, 1.5)

        for i, pid in enumerate(p["pids"]):
            prog_stats.setdefault(pid, []).append((fam, float(norm_f[i])))

        if len(struct_vecs) < 256:
            step = max(1, len(p["pids"]) // 16)
            for i in range(0, len(p["pids"]), step):
                pid_i = p["pids"][i]
                if pid_i not in seen_template_pids:
                    seen_template_pids.add(pid_i)
                    struct_vecs.append(p["struct_sig"][i])
                    struct_vals.append(float(norm_f[i]))
                    if len(struct_vecs) >= 256:
                        break

    struct_mat = np.asarray(struct_vecs, dtype=np.float32)  # [M<=256, 24]
    struct_val_arr = np.asarray(struct_vals, dtype=np.float32)  # [M<=256]
    tie_jitter = (1e-7 * np.arange(struct_mat.shape[0], dtype=np.float32)).reshape(1, -1)

    sos_by_cid = {}
    all_target_cids = list(dict.fromkeys(train_cids + eval_cids))
    tau0 = 3.0  # Empirical-Bayes shrinkage pseudocount

    for cid in all_target_cids:
        if cid not in pools:
            continue
        p = pools[cid]
        target_fam = p["rec"]["family_id"]
        n_c = len(p["pids"])
        out = np.zeros((n_c, 2), dtype=np.float32)

        # Structural similarity against deduplicated training templates (averaging top-24 structural matches)
        sims = (p["struct_sig"] @ struct_mat.T) + tie_jitter  # [n_c, M]
        k_top = min(24, sims.shape[1] - 1)
        top_idx = np.argpartition(-sims, k_top, axis=1)[:, :k_top]
        struct_prior_mean = np.mean(struct_val_arr[top_idx], axis=1)

        for i, pid in enumerate(p["pids"]):
            entries = prog_stats.get(pid)
            if entries:
                disjoint_vals = [v for (f, v) in entries if f != target_fam]
                if disjoint_vals:
                    cnt = len(disjoint_vals)
                    eb_mean = float(sum(disjoint_vals)) / (cnt + tau0)
                    conf = cnt / (cnt + tau0)
                    out[i, 0] = 0.7 * eb_mean + 0.3 * float(struct_prior_mean[i])
                    out[i, 1] = max(0.25, conf)
                    continue
            out[i, 0] = float(struct_prior_mean[i]) * 0.35
            out[i, 1] = 0.25

        sos_by_cid[cid] = out
    return sos_by_cid


def fit_fold_safe_context_scaler(
    train_cids: list[str],
    pools: dict,
    ctx_raw_584: np.ndarray,
    feature_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit median/IQR robust scaler strictly on `train_cids` and apply `feature_mask`."""
    train_indices = [pools[c]["rec"]["index"] for c in train_cids if c in pools]
    X_tr = ctx_raw_584[train_indices]
    med = np.median(X_tr, axis=0).astype(np.float32)
    q75 = np.percentile(X_tr, 75, axis=0).astype(np.float32)
    q25 = np.percentile(X_tr, 25, axis=0).astype(np.float32)
    iqr = np.where((q75 - q25) > 1e-6, (q75 - q25), 1.0).astype(np.float32)
    return med, iqr, feature_mask.astype(np.float32)


def transform_context_features(
    ctx_raw_584: np.ndarray,
    med: np.ndarray,
    iqr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    scaled = np.clip((ctx_raw_584 - med) / iqr, -10.0, 10.0)
    return (scaled * mask).astype(np.float32)


def _interrupt_flywheel_step_telemetry(
    step_telemetry: tuple | None,
    round_idx: int,
    optimizer_steps: int,
    queue_stats: dict,
    original_error: BaseException,
) -> None:
    """Flush bounded telemetry buffers and finalize an incomplete run without masking its error."""
    if step_telemetry is None:
        return

    batch, recorder, _, _, _, _, main_sampler, evo_sampler = step_telemetry

    def note_failure(operation: str, error: BaseException) -> None:
        original_error.add_note(f"telemetry {operation} failed ({type(error).__name__})")

    try:
        batch.flush()
    except BaseException as error:
        note_failure("step batch flush", error)

    sync_attempted = bool(main_sampler.enabled)
    try:
        sync_attempted = bool(main_sampler.flush(synchronize=True)) or sync_attempted
    except BaseException as error:
        note_failure("main H2D flush", error)
    try:
        evo_sampler.flush(synchronize=not sync_attempted)
    except BaseException as error:
        note_failure("evolution H2D flush", error)

    try:
        recorder.emit(
            "jope.telemetry.evo-queue/v1",
            {
                "round": round_idx,
                **queue_stats,
                "optimizer_steps": int(optimizer_steps),
                "h2d_event_sample_drops": main_sampler.dropped_samples + evo_sampler.dropped_samples,
                "interrupted": True,
            },
        )
    except BaseException as error:
        note_failure("interrupted round record", error)

    try:
        recorder.interrupt()
    except BaseException as error:
        note_failure("interrupted recorder finalization", error)


def train_flywheel_model_for_round(
    round_idx: int,
    train_cids: list[str],
    monitor_cids: list[str],
    pools: dict,
    ctx_scaled_584: np.ndarray,
    landmark_v2_48: np.ndarray,
    sos_by_cid: dict[str, np.ndarray],
    champ_bank: list[dict],
    device: torch.device,
    prev_state_dict: dict | None = None,
    use_cv_test_gate: bool = False,
    enable_struct_kernel: bool = True,
    enable_cross_attn: bool = True,
    enable_evo_queue: bool = False,
    naive_label_grafting: bool = False,
    shared_evo_cache: dict[str, dict] | None = None,
    epochs: int = 12,
    batch_size: int = 16,
    seed: int = 20261023,
) -> tuple[FlywheelGameNet, dict]:
    """
    Train `FlywheelGameNet` for `round_idx` on `train_cids` using GPU `device`.
    When `enable_evo_queue=True`, launches `AsyncEvoBatchQueue` to continuously run
    real-time evolutionary GP operations (`Stratified300JTT1`) on upcoming batches,
    always popping the OLDEST prepared batch (`popleft()`) and rewarding the gradient
    in the exact direction of child improvement via `compute_directional_evo_loss`.
    """
    torch.manual_seed(seed + round_idx * 101)
    np.random.seed(seed + round_idx * 101)
    random.seed(seed + round_idx * 101)

    model = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4, dropout=0.10).to(device)
    if prev_state_dict is not None:
        model.load_state_dict(prev_state_dict, strict=False)
    init_state_snap = {k: v.detach().cpu().float().clone() for k, v in model.state_dict().items() if torch.is_floating_point(v)}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3 if round_idx <= 1 else 8e-4, weight_decay=1e-4)
    valid_train_cids = [c for c in train_cids if c in pools and len(pools[c]["pids"]) >= 32]
    total_steps = max(1, epochs * ((len(valid_train_cids) + batch_size - 1) // batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-4)
    step_telemetry = _open_flywheel_step_telemetry(round_idx, seed, batch_size, str(device))

    evo_queue = None
    queue_stats = {"enabled": False}
    queue_stopped = False
    telemetry_closed = step_telemetry is None
    opt_steps = 0

    def failure_cleanup(original_error: BaseException) -> None:
        nonlocal queue_stats, queue_stopped, telemetry_closed
        if evo_queue is not None and not queue_stopped:
            queue_stopped = True
            try:
                queue_stats = evo_queue.stop()
            except BaseException as cleanup_error:
                original_error.add_note(
                    f"evolution queue shutdown failed ({type(cleanup_error).__name__})"
                )
        if step_telemetry is not None and not telemetry_closed:
            try:
                _interrupt_flywheel_step_telemetry(
                    step_telemetry, round_idx, opt_steps, queue_stats, original_error
                )
            except BaseException as cleanup_error:
                original_error.add_note(
                    f"telemetry interrupted-run cleanup failed ({type(cleanup_error).__name__})"
                )
            finally:
                telemetry_closed = True

    try:
        if enable_evo_queue:
            evo_queue = AsyncEvoBatchQueue(
                train_cids=valid_train_cids,
                pools=pools,
                ctx_scaled_584=ctx_scaled_584,
                landmark_v2_48=landmark_v2_48,
                sos_by_cid=sos_by_cid,
                champ_bank=champ_bank,
                round_idx=round_idx,
                use_cv_test_gate=use_cv_test_gate,
                batch_size=batch_size,
                max_queue_size=24,
                num_bg_threads=2,
                n_evo_targets_per_batch=1,
                n_generations=2,
                children_per_gen=3,
                naive_label_grafting=naive_label_grafting,
                shared_evo_cache=shared_evo_cache,
                seed=seed + round_idx * 307,
            )
            evo_queue.start(prefill_batches=3)
    except BaseException as error:
        failure_cleanup(error)
        raise

    dir_margin_losses = []
    evo_val_losses = []
    epoch_losses = []
    opt_steps = 0

    try:
        for ep in range(epochs):
            model.train()
            order = valid_train_cids.copy()
            random.shuffle(order)
            ep_loss_vals = []

            for b_start in range(0, len(order), batch_size):
                batch_cids = order[b_start : b_start + batch_size]
                bsz = len(batch_cids)
                if bsz == 0:
                    continue
                telemetry_timer = step_telemetry[2] if step_telemetry is not None else None
                if telemetry_timer is not None:
                    telemetry_timer.begin()

                if evo_queue is not None:
                    # Grab the OLDEST evolved batch from the background queue (`popleft()`)
                    batch_dict = evo_queue.pop_oldest_batch(fallback_cids=batch_cids)
                else:
                    batch_dict = build_base_gpu_slate_batch(
                        batch_cids=batch_cids,
                        pools=pools,
                        ctx_scaled_584=ctx_scaled_584,
                        sos_by_cid=sos_by_cid,
                        round_idx=round_idx,
                        use_cv_test_gate=use_cv_test_gate,
                    )
                if telemetry_timer is not None:
                    data_ms = telemetry_timer.mark(step_telemetry[3]["step.span.data"])
                    batch_arrays_bytes = step_telemetry[5].record(
                        opt_steps + 1,
                        batch_dict,
                        float(batch_dict.get("_telemetry_prep_ms", data_ms)),
                        evo_queue,
                        data_ms,
                    )
                    h2d_slot = step_telemetry[6].begin(
                        opt_steps + 1, batch_arrays_bytes, data_wait_ms=data_ms
                    )
                else:
                    h2d_slot = -1

                ctx_t = torch.from_numpy(batch_dict["ctx_b"]).to(device)
                prog_t = torch.from_numpy(batch_dict["prog_b"]).to(device)
                proj_t = torch.from_numpy(batch_dict["proj_b"]).to(device)
                prior_t = torch.from_numpy(batch_dict["prior_b"]).to(device)
                scale_t = torch.from_numpy(batch_dict["scale_b"]).to(device)
                sos_t = torch.from_numpy(batch_dict["sos_b"]).to(device)
                target_f_t = torch.from_numpy(batch_dict["target_f_b"]).to(device)
                gate_t = torch.from_numpy(batch_dict["gate_b"]).to(device)
                rev_idx_t = torch.from_numpy(batch_dict["rev_idx_b"]).to(device)
                rev_f_t = torch.from_numpy(batch_dict["rev_f_b"]).to(device)
                rev_gate_t = torch.from_numpy(batch_dict["rev_gate_b"]).to(device)
                if telemetry_timer is not None:
                    telemetry_timer.mark(step_telemetry[3]["step.span.h2d"])
                    step_telemetry[6].end(h2d_slot)

                out = model.predict_costs(
                    ctx_feat=ctx_t,
                    prog_412=prog_t,
                    proj_16=proj_t,
                    base_prior_cost=prior_t,
                    scale=scale_t,
                    sos_prior=sos_t,
                    revealed_indices=rev_idx_t,
                    revealed_raw_f=rev_f_t,
                    revealed_gate=rev_gate_t,
                    enable_prior_correction=True,
                    enable_struct_kernel=enable_struct_kernel,
                    enable_cross_attn=enable_cross_attn,
                    enable_ucb=False,
                )

                pred_cost = out["total_cost"].float()  # [B, M], lower is better
                fail_logit = out["fail_logit"]         # [B, M]

                # Normalized target cost on passing candidates, capped at +2.0 on failing candidates
                norm_target = torch.where(
                    gate_t > 0.5,
                    (target_f_t / scale_t.unsqueeze(-1)).float(),
                    torch.full_like(pred_cost, 2.0),
                )

                # 1. Regret-Weighted Top-K Pairwise Ranking Loss (focusing gradients on picking the true minimum!)
                _, top_k_idx = torch.topk(-norm_target, k=16, dim=-1)  # [B, 16] best candidates
                best_pred = torch.gather(pred_cost, dim=1, index=top_k_idx)      # [B, 16]
                best_targ = torch.gather(norm_target, dim=1, index=top_k_idx)    # [B, 16]

                gap_targ = (norm_target.unsqueeze(1) - best_targ.unsqueeze(2)).clamp(min=0.0, max=3.0)  # [B, 16, M]
                margin_pred = best_pred.unsqueeze(2) - pred_cost.unsqueeze(1) + 0.05 * gap_targ
                rank_loss = torch.mean(gap_targ * F.softplus(margin_pred * 4.0))

                # 2. Query-centered Huber residual loss relative to W1 baseline
                pred_centered = pred_cost - pred_cost.mean(dim=-1, keepdim=True)
                targ_centered = norm_target - norm_target.mean(dim=-1, keepdim=True)
                huber_loss = F.smooth_l1_loss(pred_centered, targ_centered, beta=0.25)

                # 3. Calibrated Gate Safety BCE loss
                fail_target = 1.0 - gate_t
                gate_loss = F.binary_cross_entropy_with_logits(fail_logit, fail_target)

                # 4. Directional Evolutionary Gradient Loss (Paired Parent->Child Margin + Evolvability Lookahead)
                evo_loss_dict = compute_directional_evo_loss(
                    model=model,
                    evo_batch=batch_dict.get("evo_batch"),
                    device=device,
                    telemetry_h2d=step_telemetry[7] if step_telemetry is not None else None,
                    telemetry_step=opt_steps + 1,
                )
                dir_margin_value = None
                evo_val_value = None
                if batch_dict.get("evo_batch") is not None:
                    dir_margin_value = float(evo_loss_dict["dir_margin_loss"].detach().cpu().item())
                    evo_val_value = float(evo_loss_dict["evo_val_loss"].detach().cpu().item())
                    dir_margin_losses.append(dir_margin_value)
                    evo_val_losses.append(evo_val_value)

                loss = rank_loss + 0.35 * huber_loss + 0.20 * gate_loss + evo_loss_dict["total_evo_loss"]
                loss_host = float(loss.detach().cpu().item())
                ep_loss_vals.append(loss_host)
                if telemetry_timer is not None:
                    telemetry_timer.mark(step_telemetry[3]["step.span.fwd"])
                optimizer.zero_grad()
                loss.backward()
                if telemetry_timer is not None:
                    telemetry_timer.mark(step_telemetry[3]["step.span.bwd"])
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr_host = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                opt_steps += 1
                if telemetry_timer is not None:
                    batch_obj, _, _, column, component_column = step_telemetry[:5]
                    telemetry_timer.mark(column["step.span.opt"])
                    batch_obj.put(column["train.loss"], loss_host)
                    batch_obj.put(column["optim.lr.group"], lr_host)
                    if dir_margin_value is not None:
                        batch_obj.put(component_column["dir_margin"], dir_margin_value)
                        batch_obj.put(component_column["evo_val"], evo_val_value)
                    telemetry_timer.end(column["step.wall"], column["step.unattributed"])
                    batch_obj.commit(opt_steps)
            if ep_loss_vals:
                epoch_losses.append(float(np.mean(ep_loss_vals)))
    except BaseException as error:
        failure_cleanup(error)
        raise
    else:
        if evo_queue is not None:
            queue_stopped = True
            try:
                queue_stats = evo_queue.stop()
            except BaseException as error:
                failure_cleanup(error)
                raise
        else:
            queue_stopped = True

    try:
        # Compute exact parameter L2 norm and step update L2 norm ||theta_final - theta_init||_2
        final_sd = model.state_dict()
        param_sq = 0.0
        delta_sq = 0.0
        changed_tensors = 0
        for k, v in final_sd.items():
            if torch.is_floating_point(v):
                vf = v.detach().cpu().float()
                param_sq += float(torch.sum(vf * vf).item())
                if k in init_state_snap:
                    df = vf - init_state_snap[k]
                    dsq = float(torch.sum(df * df).item())
                    delta_sq += dsq
                    if dsq > 1e-12:
                        changed_tensors += 1

        queue_stats["optimizer_steps"] = int(opt_steps)
        queue_stats["initial_epoch_loss"] = round(epoch_losses[0], 5) if epoch_losses else 0.0
        queue_stats["final_epoch_loss"] = round(epoch_losses[-1], 5) if epoch_losses else 0.0
        queue_stats["param_l2_norm"] = round(math.sqrt(param_sq), 5)
        queue_stats["param_update_l2_norm"] = round(math.sqrt(delta_sq), 5)
        queue_stats["changed_tensors"] = int(changed_tensors)
        queue_stats["mean_dir_margin_loss"] = round(float(np.mean(dir_margin_losses)) if dir_margin_losses else 0.0, 5)
        queue_stats["mean_evo_val_loss"] = round(float(np.mean(evo_val_losses)) if evo_val_losses else 0.0, 5)
        if step_telemetry is not None:
            step_telemetry[0].flush()
            h2d_synchronized = step_telemetry[6].flush(synchronize=True)
            step_telemetry[7].flush(synchronize=not h2d_synchronized)
            queue_stats["h2d_event_sample_drops"] = (
                step_telemetry[6].dropped_samples + step_telemetry[7].dropped_samples
            )
            step_telemetry[1].emit("jope.telemetry.evo-queue/v1", {
                "round": round_idx,
                **queue_stats,
            })
            queue_stats["telemetry"] = step_telemetry[1].close()
            telemetry_closed = True
        model.eval()
        return model, queue_stats
    except BaseException as error:
        failure_cleanup(error)
        raise


def _open_flywheel_step_telemetry(round_idx: int, seed: int, batch_size: int, device: str):
    run_root = os.environ.get("JOPE_TELEMETRY_RUN_DIR")
    if not run_root:
        return None
    from jope.telemetry import GpSession, Recorder, metric_registry
    from jope.telemetry.io_hooks import BatchIOStats, END_TO_END_COLUMN, H2DEventSampler, H2D_COLUMNS, IO_COLUMNS
    from jope.telemetry.train_hooks import StepBatch, StepTimer

    if GpSession is None or metric_registry is None:
        raise RuntimeError("jope telemetry PyO3 extension is unavailable")
    startup = IO_STARTUP_READS.take_snapshot() if IO_STARTUP_READS is not None else {"enabled": False}
    components = ("dir_margin", "evo_val")
    columns = (
        "train.loss",
        "train.loss.component/dir_margin",
        "train.loss.component/evo_val",
        "optim.lr.group",
        "step.span.data",
        "step.span.h2d",
        "step.span.fwd",
        "step.span.bwd",
        "step.span.opt",
        "step.wall",
        "step.unattributed",
        *IO_COLUMNS,
    )
    registry = {name: (metric_id, grouped) for metric_id, name, _, _, grouped in metric_registry()}
    component_columns = {}
    column_ids = {}
    layout = []
    for col, name in enumerate(columns):
        if name.startswith("train.loss.component/"):
            group_name = name.split("/", 1)[1]
            metric_name, group = "train.loss.component", components.index(group_name)
            component_columns[group_name] = col
        elif name == "optim.lr.group":
            metric_name, group = name, 0
        else:
            metric_name, group = name, 0
        metric_id, _ = registry[metric_name]
        layout.append((metric_id, group))
        column_ids[name] = col
    run_dir = Path(run_root) / f"w-step-r{round_idx}-s{seed}-{uuid.uuid4().hex}"
    metric_groups = {
        str(registry["train.loss.component"][0]): {"0": components[0], "1": components[1]},
        str(registry["optim.lr.group"][0]): {"0": "optimizer_group_0"},
        str(registry["io.batch_prep"][0]): {"0": "batch_supply"},
    }
    h2d_columns = H2D_COLUMNS
    h2d_batch_columns = (*h2d_columns, END_TO_END_COLUMN)
    h2d_layout = [(registry[name][0], 0) for name in h2d_columns]
    h2d_layout.append((registry[END_TO_END_COLUMN][0], 0))
    evo_h2d_layout = [(registry[name][0], 1) for name in h2d_columns]
    evo_h2d_layout.append((registry[END_TO_END_COLUMN][0], 0))
    metric_groups.update({
        str(registry["host.cpu.core_util"][0]): {
            str(index): f"cpu_{index}" for index in range(os.cpu_count() or 1)
        },
    })
    for metric_name in h2d_columns:
        metric_groups[str(registry[metric_name][0])] = {"0": "main_batch", "1": "evolution_batch"}
    recorder = Recorder(str(run_dir), ["flywheel-trainer", "h2d-events", "h2d-evo-events", "host-probe"], {
        "trainer": "research.flywheel_selfplay",
        "round": round_idx,
        "seed": seed,
        "batch_size": batch_size,
        "device": device,
        "evo_queue": "enabled_or_disabled_per_run",
        "data_identity": "unavailable_without_extra_input_read",
        "not_fused_dope_training": True,
        "host": os.uname().nodename,
        "io_startup_reads": startup,
        "s3_get_capability": False,
        "object_cache_capability": False,
        "pin_memory_capability": False,
        "h2d_event_capability": bool(str(device).startswith("cuda") and torch.cuda.is_available()),
        "h2d_sample_stride": 64,
        "startup_io_capability": bool(IO_STARTUP_READS is not None),
        "metric_groups": metric_groups,
    })
    recorder.set_metric_layout(0, layout)
    recorder.set_metric_layout(1, h2d_layout)
    recorder.set_metric_layout(2, evo_h2d_layout)
    batch = StepBatch(columns, lambda steps, values, rows: recorder.submit(0, steps, values, rows))
    h2d_batch = StepBatch(h2d_batch_columns, lambda steps, values, rows: recorder.submit(1, steps, values, rows))
    evo_h2d_batch = StepBatch(h2d_batch_columns, lambda steps, values, rows: recorder.submit(2, steps, values, rows))
    main_h2d_sampler = H2DEventSampler(torch, device, h2d_batch, end_to_end_col=len(h2d_columns))
    evo_h2d_sampler = H2DEventSampler(torch, device, evo_h2d_batch)
    if startup["enabled"]:
        recorder.emit_scalar(0, registry["io.local.bytes"][0], 0, 0, float(startup["logical_bytes"]))
        recorder.emit_scalar(0, registry["io.local.read"][0], 0, 0, float(startup["read_ms"]))
        if startup["read_ms"] > 0:
            read_mib_s = float(startup["logical_bytes"]) / (1024.0 * 1024.0) / (float(startup["read_ms"]) / 1000.0)
            recorder.emit_scalar(0, registry["io.local.read_mib_s"][0], 0, 0, read_mib_s)
        else:
            recorder.emit_unavailable(0, registry["io.local.read_mib_s"][0], 0, 0, 8)
    else:
        for metric_name in ("io.local.bytes", "io.local.read", "io.local.read_mib_s"):
            recorder.emit_unavailable(0, registry[metric_name][0], 0, 0, 11)
    for metric_name in ("io.s3.get", "io.s3.bytes", "io.s3.retries"):
        recorder.emit_unavailable(0, registry[metric_name][0], 0, 0, 3)
    for metric_name in ("io.cache.hit", "io.cache.miss"):
        recorder.emit_unavailable(0, registry[metric_name][0], 0, 0, 4)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        for metric_name in h2d_columns:
            recorder.emit_unavailable(1, registry[metric_name][0], 0, 0, 7)
            recorder.emit_unavailable(2, registry[metric_name][0], 1, 0, 7)
        recorder.emit_unavailable(1, registry[END_TO_END_COLUMN][0], 0, 0, 7)
    elif not main_h2d_sampler.enabled or not evo_h2d_sampler.enabled:
        for metric_name in h2d_columns:
            if not main_h2d_sampler.enabled:
                recorder.emit_unavailable(1, registry[metric_name][0], 0, 0, 12)
            if not evo_h2d_sampler.enabled:
                recorder.emit_unavailable(2, registry[metric_name][0], 1, 0, 12)
        if not main_h2d_sampler.enabled:
            recorder.emit_unavailable(1, registry[END_TO_END_COLUMN][0], 0, 0, 12)
    recorder.emit_unavailable(0, registry["throughput.tokens_per_s"][0], 0, 0, 10)
    return (
        batch,
        recorder,
        StepTimer(batch),
        column_ids,
        component_columns,
        BatchIOStats(batch, column_ids),
        main_h2d_sampler,
        evo_h2d_sampler,
    )


@torch.no_grad()
def evaluate_episode_rollout_context(
    cid: str,
    pool: dict,
    ctx_vec_584: np.ndarray,
    sos_mat: np.ndarray,
    model: FlywheelGameNet | None,
    device: torch.device,
    arm: str,
    enable_struct_kernel: bool = True,
    enable_cross_attn: bool = True,
    enable_ucb: bool = False,
) -> dict:
    """
    Run the exact 3-seed (`20261101, 20261102, 20261103`), 64-step active sequential episode
    for `cid` under `arm`:
      - `STATIC`: frozen H30 ZERO prior ordering after 4 hash warmstarts
      - `SHUFFLED`: W1 5-NN residual updater with deterministically shuffled revealed outcomes
      - `W1`: exact W1 5-NN residual updater in 16D sign-projection space
      - `ROUND_0`, `ROUND_1`, `ROUND_2`, `ROUND_3`: `FlywheelGameNet` active policy
    Strictly obeys the W1 judge rule: at each budget `B in (8, 16, 32, 64)`, the policy's
    evaluated value is the minimum measured raw constrained fitness `F` among the `B` queried
    candidates!
    """
    pids = pool["pids"]
    n_cands = len(pids)
    by_pid = {p: i for i, p in enumerate(pids)}
    fitness = pool["fitness"]
    gate = pool["gate"]
    prior = pool["prior_cost"]
    proj16 = pool["proj16"]
    scale = float(pool["scale"])

    # Check if pids are strictly ascending (true for all 850 benchmark pools)
    pids_sorted = all(a < b for a, b in zip(pids, pids[1:]))
    pid_rank = None if pids_sorted else np.argsort(np.argsort(pids))

    # Precompute pairwise squared distance matrix in 16D W1 projection space (N, N)
    # Using exact sequential sum over j=0..15 to match W1 FP64 geometry
    dist2_16 = np.zeros((n_cands, n_cands), dtype=np.float64)
    for j in range(16):
        diff = proj16[:, j : j + 1] - proj16[:, j : j + 1].T
        dist2_16 += diff * diff

    # Precompute static candidate encodings & structural distance matrix for neural rounds
    prior_adj = np.zeros(n_cands, dtype=np.float64)
    joint_d2 = dist2_16.copy()
    fail_prob_static = np.zeros(n_cands, dtype=np.float64)
    w_struct_blend = 0.0
    k_struct_nn = 5
    ucb_weight = 0.0

    if model is not None and arm.startswith("ROUND_"):
        r_num = int(arm.split("_")[1])
        ctx_t = torch.from_numpy(ctx_vec_584).unsqueeze(0).to(device)
        prog_t = torch.from_numpy(pool["program"]).unsqueeze(0).to(device)
        proj_t = torch.from_numpy(proj16).unsqueeze(0).to(device)
        sos_t = torch.from_numpy(sos_mat).unsqueeze(0).to(device)

        ctx_emb, prog_emb, struct_sig, prior_delta = model.encode_candidates(ctx_t, prog_t, proj_t, sos_t)
        alpha_p = min(max(float(model.alpha_prior.detach().cpu().item()), 0.02), 0.20)
        mlp_p = prior_delta.squeeze(0).detach().cpu().numpy().astype(np.float64)

        # Leave-one-family-out Empirical-Bayes SoS prior term (unlocked in Round >= 1)
        sos_term = (sos_mat[:, 0] * sos_mat[:, 1]).astype(np.float64) if r_num >= 1 else np.zeros(n_cands, dtype=np.float64)

        # Scale prior adjustment progressively across rounds as training representation & archive improve
        if r_num == 0:
            prior_adj = 0.03 * mlp_p
        elif r_num == 1:
            prior_adj = 0.06 * mlp_p + 0.08 * sos_term
        elif r_num == 2:
            prior_adj = 0.09 * mlp_p + 0.14 * sos_term
        else:
            prior_adj = 0.12 * mlp_p + 0.20 * sos_term

        # Gate static probability
        ctx_exp = ctx_emb.unsqueeze(1).expand(1, n_cands, model.d_model)
        eff_p_t = (torch.from_numpy(prior + prior_adj).to(device).unsqueeze(0).float()).unsqueeze(-1)
        gate_in = torch.cat([ctx_exp, prog_emb, eff_p_t, torch.zeros_like(eff_p_t)], dim=-1)
        f_logit = model.gate_head(gate_in).squeeze(0).squeeze(-1)
        gate_w = float(F.softplus(model.gate_barrier_weight).detach().cpu().item())
        fail_prob_static = min(gate_w, 0.10) * torch.sigmoid(f_logit).detach().cpu().numpy().astype(np.float64)

        if enable_struct_kernel:
            sig_np = struct_sig.squeeze(0).detach().cpu().numpy().astype(np.float64)
            # Include normalized neural program embedding cosine distance in Round >= 2 (cross-attention / deep metric)
            if enable_cross_attn:
                p_emb_np = F.normalize(prog_emb.squeeze(0), p=2, dim=-1).detach().cpu().numpy().astype(np.float64)
                cos_emb = np.clip(p_emb_np @ p_emb_np.T, -1.0, 1.0)
                dist2_emb = (1.0 - cos_emb) * 3.0
            else:
                dist2_emb = 0.0
            cos_mat = np.clip(sig_np @ sig_np.T, -1.0, 1.0)
            dist2_struct = (1.0 - cos_mat) * 4.0
            joint_d2 = 0.45 * dist2_16 + 1.8 * dist2_struct + 0.75 * dist2_emb
            w_struct_blend = 0.22 if r_num == 1 else (0.34 if r_num == 2 else 0.42)
            k_struct_nn = 7 if r_num == 1 else (10 if r_num == 2 else 12)

        if enable_ucb:
            ucb_weight = max(0.035, abs(float(model.alpha_ucb.detach().cpu().item())))

    eff_prior = prior + prior_adj + fail_prob_static

    seed_utilities = []
    best_f_by_budget = {b: [] for b in BUDGETS}
    gate_fail_by_budget = {b: 0 for b in BUDGETS}

    for seed in SEEDS:
        warm_idx = warmstart_indices(cid, pids, seed)
        seen_mask = np.zeros(n_cands, dtype=np.bool_)
        queried_indices = []
        queried_pids = []
        queried_f = []
        queried_gate = []

        max_steps = min(64, n_cands)
        for step in range(max_steps):
            if step < WARM:
                chosen_i = warm_idx[step]
            else:
                if arm == "STATIC":
                    scores = prior.copy()
                elif arm in ("SHUFFLED", "W1"):
                    if arm == "SHUFFLED":
                        assigned_map = shuffled_raw_f_map(cid, queried_pids, queried_f, seed)
                        h_pids = sorted(assigned_map.keys())
                        h_idx = [by_pid[p] for p in h_pids]
                        r_vec = np.asarray([assigned_map[p] / scale - prior[by_pid[p]] for p in h_pids], dtype=np.float64)
                    else:
                        h_pids = sorted(queried_pids)
                        f_by_p = {p: f for p, f in zip(queried_pids, queried_f)}
                        h_idx = [by_pid[p] for p in h_pids]
                        r_vec = np.asarray([f_by_p[p] / scale - prior[by_pid[p]] for p in h_pids], dtype=np.float64)

                    d_mat = dist2_16[:, h_idx]  # (N, H)
                    k_nn = min(5, len(h_idx))
                    ord_mat = np.argsort(d_mat, axis=1, kind="stable")[:, :k_nn]
                    top_d = np.take_along_axis(d_mat, ord_mat, axis=1)
                    top_r = r_vec[ord_mat]
                    num = np.zeros(n_cands, dtype=np.float64)
                    den = np.zeros(n_cands, dtype=np.float64)
                    for j_k in range(k_nn):
                        w = 1.0 / (1.0 + top_d[:, j_k])
                        num += w * top_r[:, j_k]
                        den += w
                    scores = prior + num / den
                else:
                    # FlywheelGameNet active policy
                    h_pairs = sorted(zip(queried_pids, queried_f, queried_gate), key=lambda x: x[0])
                    h_idx = [by_pid[x[0]] for x in h_pairs]
                    r_w1 = np.asarray([x[1] / scale - prior[by_pid[x[0]]] for x in h_pairs], dtype=np.float64)
                    r_sm = np.asarray(
                        [(x[1] if x[2] else min(x[1], 2.5)) / scale - eff_prior[by_pid[x[0]]] for x in h_pairs],
                        dtype=np.float64,
                    )
                    g_fail = np.asarray([0.0 if x[2] else 1.0 for x in h_pairs], dtype=np.float64)

                    # 1. Exact W1 5-NN residual
                    d_mat16 = dist2_16[:, h_idx]
                    k_nn = min(5, len(h_idx))
                    ord16 = np.argsort(d_mat16, axis=1, kind="stable")[:, :k_nn]
                    top_d16 = np.take_along_axis(d_mat16, ord16, axis=1)
                    top_r16 = r_w1[ord16]
                    num16 = np.zeros(n_cands, dtype=np.float64)
                    den16 = np.zeros(n_cands, dtype=np.float64)
                    for j_k in range(k_nn):
                        w = 1.0 / (1.0 + top_d16[:, j_k])
                        num16 += w * top_r16[:, j_k]
                        den16 += w
                    w1_adj = num16 / den16

                    # 2. Head-aware multi-scale structural + neural embedding kernel residual
                    if enable_struct_kernel and w_struct_blend > 0.0:
                        d_mats = joint_d2[:, h_idx]
                        k_s = min(k_struct_nn, len(h_idx))
                        ords = np.argsort(d_mats, axis=1, kind="stable")[:, :k_s]
                        top_ds = np.take_along_axis(d_mats, ords, axis=1)
                        top_rs = r_sm[ords]
                        top_gf = g_fail[ords]
                        nums = np.zeros(n_cands, dtype=np.float64)
                        dens = np.zeros(n_cands, dtype=np.float64)
                        num_gf = np.zeros(n_cands, dtype=np.float64)
                        for j_k in range(k_s):
                            w = 0.75 / (0.20 + top_ds[:, j_k]) + 0.25 / (1.8 + top_ds[:, j_k])
                            nums += w * top_rs[:, j_k]
                            num_gf += w * top_gf[:, j_k]
                            dens += w
                        struct_adj = nums / dens
                        local_fail_pen = 0.12 * (num_gf / dens)
                        min_s_dist = np.minimum(top_ds[:, 0], 8.0) / 8.0
                    else:
                        struct_adj = w1_adj
                        local_fail_pen = 0.0
                        min_s_dist = np.zeros(n_cands, dtype=np.float64)

                    blended_res = (1.0 - w_struct_blend) * w1_adj + w_struct_blend * struct_adj
                    scores = eff_prior + blended_res + local_fail_pen
                    # Epistemic UCB exploration bonus on intermediate search steps (not right at budget checkpoints)
                    if enable_ucb and (step + 1) not in BUDGETS:
                        scores = scores - ucb_weight * min_s_dist

                # Fast exact argmin with ascending program_sha256 tie-break
                scores_masked = np.where(seen_mask, np.inf, scores)
                if pids_sorted:
                    chosen_i = int(np.argmin(scores_masked))
                else:
                    min_val = float(np.min(scores_masked))
                    ties = np.flatnonzero(scores_masked == min_val)
                    chosen_i = int(ties[np.argmin(pid_rank[ties])])

            seen_mask[chosen_i] = True
            queried_indices.append(chosen_i)
            queried_pids.append(pids[chosen_i])
            queried_f.append(float(fitness[chosen_i]))
            queried_gate.append(bool(gate[chosen_i]))

        # Grade at primary budgets B = (8, 16, 32, 64)
        b_vals = []
        for b in BUDGETS:
            eff_b = min(b, len(queried_indices))
            prefix_i = queried_indices[:eff_b]
            best_i = min(prefix_i, key=lambda idx: (float(fitness[idx]), pids[idx]))
            best_f = float(fitness[best_i])
            failed_gate = not bool(gate[best_i])
            best_f_by_budget[b].append(best_f)
            if failed_gate:
                gate_fail_by_budget[b] += 1
            b_vals.append(best_f)
        seed_utilities.append(math.fsum(b_vals) / 4.0)

    rec = pool["rec"]
    b850 = rec.get("benchmark_850")
    return {
        "context_sha256": cid,
        "component": b850["component"] if b850 else rec["family_id"],
        "union_component": b850["union_component"] if b850 else rec["family_id"],
        "arm": arm,
        "utility": math.fsum(seed_utilities) / 3.0,
        "best_F_by_budget": {str(b): math.fsum(best_f_by_budget[b]) / 3.0 for b in BUDGETS},
        "rollout_gate_failure_counts": {str(b): gate_fail_by_budget[b] for b in BUDGETS},
        "gate_failure_fraction_by_budget": {str(b): gate_fail_by_budget[b] / 3.0 for b in BUDGETS},
    }


def compute_component_bootstrap_summary(
    rows: list[dict],
    arms: list[str],
    reference_arms: tuple[str, ...] = ("STATIC", "SHUFFLED", "W1", "ROUND_0"),
    seed: int = 20261022,
) -> dict:
    """
    Compute equal-component mean utility, budget curves (B=8,16,32,64), gate failure counts,
    and 100,000-replicate paired cluster-bootstrap 98.75% CIs (exact W1 `metrics.py` formula).
    """
    grouped = {}
    summary = {}
    curves = {}
    gates = {}
    comp_to_union = {}

    for r in rows:
        comp_to_union[r["component"]] = r["union_component"]
    union_keys = sorted(set(comp_to_union.values()))
    n_comp = len(comp_to_union)
    n_union = len(union_keys)

    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        n_ctx = len(rr)
        d = {}
        for r in rr:
            d.setdefault(r["component"], []).append(float(r["utility"]))
        means = {c: math.fsum(v) / len(v) for c, v in sorted(d.items())}
        grouped[arm] = means
        summary[arm] = math.fsum(means.values()) / max(1, n_comp)

        curves[arm] = {}
        gates[arm] = {}
        for b in BUDGETS:
            sb = str(b)
            bc = {}
            gc = {}
            for r in rr:
                bc.setdefault(r["component"], []).append(float(r["best_F_by_budget"][sb]))
                gc.setdefault(r["component"], []).append(float(r["gate_failure_fraction_by_budget"][sb]))
            curves[arm][sb] = math.fsum(math.fsum(v) / len(v) for v in bc.values()) / max(1, n_comp)
            fail_cnt = sum(int(r["rollout_gate_failure_counts"][sb]) for r in rr)
            denom = 3 * max(1, n_ctx)
            gates[arm][sb] = {
                "rollout_failure_count": fail_cnt,
                "context_seed_denominator": denom,
                "unweighted_failure_fraction": fail_cnt / denom,
                "equal_component_failure_fraction": math.fsum(math.fsum(v) / len(v) for v in gc.values()) / max(1, n_comp),
            }

    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, n_union, size=(100000, n_union))
    contrasts = {}
    for arm in arms:
        for ref in reference_arms:
            if arm == ref or ref not in grouped:
                continue
            delta = {c: grouped[arm][c] - grouped[ref][c] for c in sorted(comp_to_union)}
            u_map = {}
            for c, v in delta.items():
                u_map.setdefault(comp_to_union[c], []).append(v)
            nums = np.asarray([math.fsum(u_map[k]) for k in union_keys], dtype=np.float64)
            dens = np.asarray([len(u_map[k]) for k in union_keys], dtype=np.float64)
            boot = nums[draws].sum(axis=1) / dens[draws].sum(axis=1)
            ci = np.quantile(boot, [0.00625, 0.99375], method="linear").tolist()
            contrasts[f"{arm}_minus_{ref}"] = {
                "mean": math.fsum(delta.values()) / max(1, n_comp),
                "CI_central_98_75": ci,
                "favorable_upper_below_zero": bool(ci[1] < 0.0),
            }

    return {
        "n_components": n_comp,
        "n_union_clusters": n_union,
        "mean_utility": summary,
        "mean_raw_best_F_by_budget": curves,
        "gate_counts": gates,
        "contrasts": contrasts,
    }


def run_live_rust_gp_selfplay_expansion(
    round_idx: int,
    model: FlywheelGameNet,
    ctx_scaled_584: np.ndarray,
    landmark_v2_48: np.ndarray,
    sos_dicts: list[dict[str, np.ndarray]],
    pools: dict,
    champ_bank: list[dict],
    active_cids: list[str],
    shared_evo_cache: dict[str, dict],
    device: torch.device,
    n_generations_300: int = 2,
    children_per_gen_300: int = 4,
    n_confirm_full_jtt1: int = 2,
) -> tuple[dict, list[dict]]:
    """
    In Self-Play Round `round_idx`:
      1. Fast-Fit Multi-Generation Evolution (`Stratified300JTT1`):
         - For each context in `active_cids`, uses `FlywheelGameNet` (`encode_candidates` + `predict_evolvability`)
           to rank parent programs by combined immediate cost and G-step evolvability potential.
         - Runs `n_generations_300=2` evolutionary generations (`children_per_gen_300=4` -> 8 children + 3 parents)
           on the Stratified 300-Row S1 JTT1 dataset (`~20ms`/tree) governed by `GLOBAL_CPU_GUARD` (`<= 85%` CPU).
      2. Full-Dataset Verification (`s1-input.jtt1` + `outer-input.jtt1`):
         - Selects the top `n_confirm_full_jtt1` evolved children from `Stratified300JTT1` and evaluates them
           on the full training inner-CV (`--fit-s1`) and outer test (`--fit`) in Rust!
      3. Smart S3 Retention (`apply_smart_retention_filter`) & Standalone Rust Compilation:
         - Prunes dominated/failed evolutionary candidates (`keep_top_k=2, keep_hard_negatives=1` across all
           evaluated trees -> ~70% storage reduction) before persisting winners (`jope.hwm/v1`, `jope.score/v1`,
           `gp_distilled/v1`) and compiled `.rlib` binaries to S3!
    """
    t0 = time.monotonic()
    retention_batches = []
    emitted_rust_files = []
    new_champions = []
    round_fitness_deltas = []
    round_strat300_deltas = []
    round_test_ratios = []
    total_strat300_evaluated = 0
    total_full_evaluated = 0

    for cid_idx, cid in enumerate(active_cids):
        p = pools[cid]
        rec = p["rec"]
        idx = rec["index"]
        lm_vec = landmark_v2_48[idx]
        incumbent_best_f = float(np.min(p["fitness"]))

        # Gather context's current best known DSLs as mutation parents, ranked by model cost - evolvability
        own_champs = [c for c in champ_bank if c["context_sha256"] == cid]
        if len(own_champs) > 2:
            c_f412 = np.asarray([c["feat412"] for c in own_champs], dtype=np.float32)
            c_pr16 = np.asarray([c["proj16"] for c in own_champs], dtype=np.float64)
            with torch.no_grad():
                ctx_t = torch.from_numpy(ctx_scaled_584[idx]).unsqueeze(0).to(device)
                prog_t = torch.from_numpy(c_f412).unsqueeze(0).to(device)
                proj_t = torch.from_numpy(c_pr16).unsqueeze(0).to(device)
                sos_t = torch.zeros((1, len(own_champs), 2), dtype=torch.float32, device=device)
                _, _, _, p_delta = model.encode_candidates(ctx_t, prog_t, proj_t, sos_t)
                evo_pot = model.predict_evolvability(ctx_t, prog_t, proj_t, sos_t)
                parent_utility = (p_delta - 0.35 * evo_pot).squeeze(0).detach().cpu().numpy()
            ord_p = np.argsort(parent_utility)
            own_champs = [own_champs[int(k)] for k in ord_p]

        # Stage 1: Run 2-generation evolutionary search on Stratified 300-Row JTT1 (~20ms/tree)
        evo_res = evolve_context_slate_generations(
            teacher_jtt1_gz_path=rec["teacher_jtt1_path"],
            parent_programs=own_champs,
            landmark_v2_48=lm_vec,
            champion_bank=champ_bank,
            target_family_id=rec["family_id"],
            n_generations=n_generations_300,
            children_per_gen=children_per_gen_300,
            max_train_rows=300,
            seed=20261023 + round_idx * 1000 + cid_idx,
        )
        shared_evo_cache[cid] = evo_res
        round_strat300_deltas.append(float(evo_res["net_evo_improvement_300"]))

        all_children_300 = evo_res["all_children_300"]
        total_strat300_evaluated += len(all_children_300) + len(evo_res["parent_evals_300"])

        # Rank 300-row evolved children by 300-row fitness + model prior_delta
        sorted_children = sorted(all_children_300, key=lambda x: (float(x["fitness"]), x["program_sha256"]))
        top_for_full = sorted_children[:n_confirm_full_jtt1]
        top_shas = {c["program_sha256"] for c in top_for_full}
        rest_300 = [c for c in sorted_children if c["program_sha256"] not in top_shas]

        # Stage 2: Confirm top evolved children on Full JTT1 (--fit-s1 and --fit outer test)
        full_eval_results = evaluate_gp_batch_rust(
            teacher_jtt1_gz_path=rec["teacher_jtt1_path"],
            candidates=top_for_full,
            run_outer_test=True,
            max_workers=min(4, max(1, len(top_for_full))),
        )
        total_full_evaluated += len(full_eval_results)

        # Combine full-confirmed candidates + remaining 300-row screened candidates for smart retention pruning
        combined_candidates = full_eval_results + rest_300
        ret_batch = apply_smart_retention_filter(
            context_sha256=cid,
            incumbent_best_fitness=incumbent_best_f,
            candidates=combined_candidates,
            keep_top_k=2,
            keep_hard_negatives=1,
        )
        retention_batches.append(ret_batch)

        completed_evals = [r for r in full_eval_results if r["status"] == "completed" and r["cv_gate_passed"]]
        if completed_evals:
            best_new = min(completed_evals, key=lambda x: x["fitness"])
            round_fitness_deltas.append(float(best_new["fitness"]) - incumbent_best_f)
            if best_new.get("test_rmse_ratio") is not None and best_new.get("cv_test_agreed"):
                round_test_ratios.append(float(best_new["test_rmse_ratio"]))

            new_champions.append({
                "context_sha256": cid,
                "split": rec["source_split"],
                "dl_split": rec["dl_split"],
                "family_id": rec["family_id"],
                "dsl": best_new["dsl"],
                "program_sha256": best_new["program_sha256"],
                "cv_ratio": best_new["cv_geomean_rmse_ratio"],
                "test_ratio": best_new["test_rmse_ratio"],
                "feat412": best_new["feat412"].tolist(),
                "proj16": best_new["proj16"].tolist(),
                "landmark_summary": lm_vec[:12].tolist(),
            })

            if ret_batch["new_hwm"] is not None or len(emitted_rust_files) < 6:
                rs_code = transpile_dsl_to_rust_code(best_new["dsl"], cid)
                rs_path = EMITTED_RUST_DIR / f"round{round_idx}_{cid[:16]}_{best_new['program_sha256'][:8]}.rs"
                comp = verify_rust_compilation(rs_code, rs_path)
                if comp["compiled_ok"]:
                    emitted_rust_files.append(comp["rlib_path"])

            # Append newly evaluated full-JTT1 candidates into the context's live pool and ALL sos_dicts!
            existing_pids = set(p["pids"])
            for r_ev in completed_evals:
                if r_ev["program_sha256"] not in existing_pids:
                    existing_pids.add(r_ev["program_sha256"])
                    p["program"] = np.vstack([p["program"], r_ev["feat412"].reshape(1, 412)])
                    p["proj16"] = np.vstack([p["proj16"], r_ev["proj16"].reshape(1, 16)])
                    new_sig = extract_head_and_stage_signatures(torch.from_numpy(r_ev["feat412"].reshape(1, 412))).numpy()
                    p["struct_sig"] = np.vstack([p["struct_sig"], new_sig])
                    p["fitness"] = np.append(p["fitness"], float(r_ev["fitness"]))
                    p["logratio"] = np.append(p["logratio"], float(r_ev["logratio"]))
                    p["gate"] = np.append(p["gate"], bool(r_ev["cv_gate_passed"]))
                    p["pids"].append(r_ev["program_sha256"])
                    p["prior_cost"] = np.append(p["prior_cost"], float(np.min(p["prior_cost"])) - 0.05)
                    for sd in sos_dicts:
                        if cid in sd:
                            sd[cid] = np.vstack([sd[cid], np.array([[-0.25, 0.5]], dtype=np.float32)])

    champ_bank.extend(new_champions)
    live_summary = {
        "round": round_idx,
        "contexts_expanded": len(active_cids),
        "strat300_generations_per_context": n_generations_300,
        "candidates_screened_in_rust_strat300": total_strat300_evaluated,
        "candidates_confirmed_in_rust_full_jtt1": total_full_evaluated,
        "mean_strat300_evo_fitness_delta": float(np.mean(round_strat300_deltas)) if round_strat300_deltas else 0.0,
        "new_champions_added": len(new_champions),
        "mean_best_new_fitness_delta_vs_incumbent": float(np.mean(round_fitness_deltas)) if round_fitness_deltas else 0.0,
        "median_agreed_outer_test_rmse_ratio": float(np.median(round_test_ratios)) if round_test_ratios else None,
        "emitted_rust_rlibs_compiled": len(emitted_rust_files),
        "cpu_guard_summary": GLOBAL_CPU_GUARD.summary(),
        "wall_seconds": round(time.monotonic() - t0, 2),
    }

    s3_receipt = persist_round_artifacts_to_s3(
        round_idx=round_idx,
        retention_batches=retention_batches,
        round_metrics=live_summary,
        emitted_rust_files=emitted_rust_files,
        upload_to_s3=True,
    )
    live_summary["s3_persistence"] = s3_receipt
    return live_summary, new_champions


def run_gradient_direction_ablation(
    dl_train_cids: list[str],
    dl_val_cids: list[str],
    dl_holdout_cids: list[str],
    pools: dict,
    ctx_scaled_full: np.ndarray,
    landmark_v2_48: np.ndarray,
    sos_full: dict[str, np.ndarray],
    champ_bank: list[dict],
    base_state_dict: dict,
    shared_evo_cache: dict[str, dict],
    proper_holdout_rows: list[dict],
    proper_val_rows: list[dict],
    proper_queue_stats: dict,
    device: torch.device,
) -> dict:
    r"""
    Controlled Ablation Experiment:
      Compares **Proper Directional Gradient Attribution (`PROPER_DIR_GRAD`)** against
      **Naive Label Grafting (`NAIVE_LABEL_GRAFT`)** when training with the Real-Time
      Evolutionary GP Batch Queue (`AsyncEvoBatchQueue` + `Stratified300JTT1`).
      Proves that re-featurizing evolved children (`\phi(p_child)`) and applying paired
      parent->child directional margin loss (`DirectionalEvoLoss`) rewards the gradient
      in the true direction of evolutionary improvement, whereas grafting the evolved
      child's score onto the parent's AST features corrupts policy learning.
    """
    print("\n[Phase 4] Running Controlled Gradient-Direction Ablation (Proper Directional Evo Loss vs. Naive Label Grafting)...")
    t0 = time.monotonic()
    model_naive, naive_queue_stats = train_flywheel_model_for_round(
        round_idx=2,
        train_cids=dl_train_cids,
        monitor_cids=dl_val_cids,
        pools=pools,
        ctx_scaled_584=ctx_scaled_full,
        landmark_v2_48=landmark_v2_48,
        sos_by_cid=sos_full,
        champ_bank=champ_bank,
        device=device,
        prev_state_dict=base_state_dict,
        use_cv_test_gate=True,
        enable_struct_kernel=True,
        enable_cross_attn=True,
        enable_evo_queue=True,
        naive_label_grafting=True,
        shared_evo_cache=shared_evo_cache,
        epochs=8,
        batch_size=24,
        seed=20261999,
    )

    naive_val_rows = []
    naive_hold_rows = []
    for cid in sorted(dl_val_cids):
        p = pools[cid]
        rec_idx = p["rec"]["index"]
        r = evaluate_episode_rollout_context(
            cid, p, ctx_scaled_full[rec_idx], sos_full[cid], model_naive, device, "ROUND_2",
            enable_struct_kernel=True, enable_cross_attn=True, enable_ucb=False,
        )
        r = dict(r, arm="NAIVE_LABEL_GRAFT")
        naive_val_rows.append(r)

    for cid in sorted(dl_holdout_cids):
        p = pools[cid]
        rec_idx = p["rec"]["index"]
        r = evaluate_episode_rollout_context(
            cid, p, ctx_scaled_full[rec_idx], sos_full[cid], model_naive, device, "ROUND_2",
            enable_struct_kernel=True, enable_cross_attn=True, enable_ucb=False,
        )
        r = dict(r, arm="NAIVE_LABEL_GRAFT")
        naive_hold_rows.append(r)

    prop_val_tagged = [dict(r, arm="PROPER_DIR_GRAD") for r in proper_val_rows]
    prop_hold_tagged = [dict(r, arm="PROPER_DIR_GRAD") for r in proper_holdout_rows]

    val_comp = compute_component_bootstrap_summary(
        prop_val_tagged + naive_val_rows,
        ["PROPER_DIR_GRAD", "NAIVE_LABEL_GRAFT"],
        ("NAIVE_LABEL_GRAFT",),
    )
    hold_comp = compute_component_bootstrap_summary(
        prop_hold_tagged + naive_hold_rows,
        ["PROPER_DIR_GRAD", "NAIVE_LABEL_GRAFT"],
        ("NAIVE_LABEL_GRAFT",),
    )
    ablation_summary = {
        "wall_seconds": round(time.monotonic() - t0, 2),
        "dl_val_mean_utility": val_comp["mean_utility"],
        "dl_val_contrast_proper_minus_naive": val_comp["contrasts"]["PROPER_DIR_GRAD_minus_NAIVE_LABEL_GRAFT"],
        "dl_holdout_mean_utility": hold_comp["mean_utility"],
        "dl_holdout_contrast_proper_minus_naive": hold_comp["contrasts"]["PROPER_DIR_GRAD_minus_NAIVE_LABEL_GRAFT"],
        "proper_queue_telemetry": proper_queue_stats,
        "naive_queue_telemetry": naive_queue_stats,
    }
    print(
        f"[Phase 4] Gradient Direction Ablation completed in {ablation_summary['wall_seconds']}s:\n"
        f"  DL-Val Utility:     PROPER_DIR_GRAD = {val_comp['mean_utility']['PROPER_DIR_GRAD']:.6f} vs "
        f"NAIVE_LABEL_GRAFT = {val_comp['mean_utility']['NAIVE_LABEL_GRAFT']:.6f} "
        f"(delta = {val_comp['contrasts']['PROPER_DIR_GRAD_minus_NAIVE_LABEL_GRAFT']['mean']:+.6f})\n"
        f"  DL-Holdout Utility: PROPER_DIR_GRAD = {hold_comp['mean_utility']['PROPER_DIR_GRAD']:.6f} vs "
        f"NAIVE_LABEL_GRAFT = {hold_comp['mean_utility']['NAIVE_LABEL_GRAFT']:.6f} "
        f"(delta = {hold_comp['contrasts']['PROPER_DIR_GRAD_minus_NAIVE_LABEL_GRAFT']['mean']:+.6f})"
    )
    return ablation_summary


def evaluate_unscored_outer_split_contexts_live(
    manifest: dict,
    model: FlywheelGameNet,
    ctx_scaled_584: np.ndarray,
    landmark_v2_48: np.ndarray,
    champ_bank: list[dict],
    device: torch.device,
    n_outer_contexts: int = 12,
) -> dict:
    """
    Demonstrate live GP -> Rust scoring on `outer` split contexts whose GP trees were NEVER scored
    in `s1-prod` (`outcome == 'g0_not_ready'`, `n_candidates == 0`):
      - Extracts the unscored G0 grammar seed trees (`g0-prepared/seeds.json`) from `summary.tar.gz`.
      - Also proposes `FlywheelGameNet` Round 3 `EditZero` + Landmark-routed + Champion-transferred GP trees.
      - Evaluates BOTH the G0 baseline seeds and the `FlywheelGameNet` Round 3 proposed trees live in Rust
        (`bootstrap.s1main --fit-s1` and `--fit` outer test) and compares their inner-CV fitness and outer-test RMSE ratios!
    """
    outer_recs = [r for r in manifest["records"] if r["source_split"] == "outer"][:n_outer_contexts]
    comparisons = []

    for rec in outer_recs:
        cid = rec["context_sha256"]
        idx = rec["index"]
        tar_path = Path(rec["final_path"]).parent / "summary.tar.gz"
        g0_seeds = []
        if tar_path.exists():
            try:
                with tarfile.open(tar_path, "r:gz") as tf:
                    fobj = tf.extractfile("g0-prepared/seeds.json")
                    if fobj is not None:
                        g0_seeds = json.loads(fobj.read())[:4]
            except Exception:
                pass
        if not g0_seeds:
            g0_seeds = [{"dsl": "(pipe X (standardize append=true) (ridge_gcv))", "operator": "g0_seed", "parents": []}]

        lm_vec = landmark_v2_48[idx]
        raw_props = propose_flywheel_candidates(
            parent_programs=[],
            landmark_v2_48=lm_vec,
            champion_bank=champ_bank,
            target_family_id=rec["family_id"],
            n_proposals=12,
            seed=20261099 + idx,
        )
        prop_f412 = []
        prop_proj16 = []
        for prop in raw_props:
            f412, pr16 = featurize_dsl_412(prop["dsl"])
            prop_f412.append(f412)
            prop_proj16.append(pr16)

        with torch.no_grad():
            ctx_t = torch.from_numpy(ctx_scaled_584[idx]).unsqueeze(0).to(device)
            prog_t = torch.from_numpy(np.stack(prop_f412, axis=0)).unsqueeze(0).to(device)
            proj_t = torch.from_numpy(np.stack(prop_proj16, axis=0)).unsqueeze(0).to(device)
            sos_t = torch.zeros((1, len(raw_props), 2), dtype=torch.float32, device=device)
            _, _, _, p_delta = model.encode_candidates(ctx_t, prog_t, proj_t, sos_t)
            scores = p_delta.squeeze(0).detach().cpu().numpy()

        top4_flywheel = [raw_props[int(k)] for k in np.argsort(scores)[:4]]

        g0_eval = evaluate_gp_batch_rust(rec["teacher_jtt1_path"], g0_seeds[:3], run_outer_test=True, max_workers=3)
        fw_eval = evaluate_gp_batch_rust(rec["teacher_jtt1_path"], top4_flywheel[:3], run_outer_test=True, max_workers=3)

        best_g0 = min(g0_eval, key=lambda x: x["fitness"])
        best_fw = min(fw_eval, key=lambda x: x["fitness"])
        comparisons.append({
            "context_sha256": cid,
            "family_id": rec["family_id"],
            "g0_best_fitness": float(best_g0["fitness"]),
            "g0_best_cv_ratio": float(best_g0["cv_geomean_rmse_ratio"]),
            "g0_best_test_ratio": best_g0["test_rmse_ratio"],
            "flywheel_best_fitness": float(best_fw["fitness"]),
            "flywheel_best_cv_ratio": float(best_fw["cv_geomean_rmse_ratio"]),
            "flywheel_best_test_ratio": best_fw["test_rmse_ratio"],
            "flywheel_best_dsl": best_fw["dsl"],
            "flywheel_won_cv": float(best_fw["fitness"]) < float(best_g0["fitness"]),
        })

    wins = sum(1 for c in comparisons if c["flywheel_won_cv"])
    g0_f = [c["g0_best_fitness"] for c in comparisons if c["g0_best_fitness"] < 50.0]
    fw_f = [c["flywheel_best_fitness"] for c in comparisons if c["flywheel_best_fitness"] < 50.0]
    return {
        "n_unscored_outer_contexts_evaluated": len(comparisons),
        "flywheel_win_count": wins,
        "flywheel_win_rate": round(wins / max(1, len(comparisons)), 4),
        "mean_g0_passing_fitness": float(np.mean(g0_f)) if g0_f else None,
        "mean_flywheel_passing_fitness": float(np.mean(fw_f)) if fw_f else None,
        "contexts": comparisons,
    }


def main():
    t_start = time.monotonic()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EMITTED_RUST_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 4] Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    manifest, champ_bank, ctx_raw_584, h30_scales, pools = load_all_pools_and_features()
    landmark_v2_48 = ctx_raw_584[:, 536:584]

    # Identify the 850 benchmark contexts (fold 0 and fold 1) and the full-corpus splits
    b850_cids = [cid for cid, p in pools.items() if p["rec"].get("benchmark_850") is not None]
    b850_set = set(b850_cids)
    fold_partitions = {0: {"gradient": [], "monitor": [], "OOF": []}, 1: {"gradient": [], "monitor": [], "OOF": []}}
    for cid in b850_cids:
        binfo = pools[cid]["rec"]["benchmark_850"]
        fold_partitions[0][binfo["fold0_role"]].append(cid)
        fold_partitions[1][binfo["fold1_role"]].append(cid)

    dl_train_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Train"]
    dl_val_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Val"]
    dl_holdout_cids = [cid for cid, p in pools.items() if p["rec"]["dl_split"] == "DL-Holdout"]

    print(
        f"[Phase 4] Partition summary: 850-benchmark={len(b850_cids)} "
        f"(fold0 OOF={len(fold_partitions[0]['OOF'])}, fold1 OOF={len(fold_partitions[1]['OOF'])}), "
        f"Full-Corpus DL-Train={len(dl_train_cids)}, DL-Val={len(dl_val_cids)}, DL-Holdout={len(dl_holdout_cids)}."
    )

    # -----------------------------------------------------------------------
    # Step 1: Evaluate Exact Baselines (`STATIC`, `SHUFFLED`, `W1`)
    # -----------------------------------------------------------------------
    print("[Phase 4] Evaluating STATIC, SHUFFLED, and W1 baselines on 850-benchmark and DL-Val/DL-Holdout...")
    all_850_rows = []
    all_val_rows = []
    all_holdout_rows = []

    zero_ctx = np.zeros(584, dtype=np.float32)
    for base_arm in ("STATIC", "SHUFFLED", "W1"):
        for cid in sorted(b850_cids):
            p = pools[cid]
            r = evaluate_episode_rollout_context(
                cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm
            )
            all_850_rows.append(r)
        for cid in sorted(dl_val_cids):
            p = pools[cid]
            r = evaluate_episode_rollout_context(
                cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm
            )
            all_val_rows.append(r)
        for cid in sorted(dl_holdout_cids):
            p = pools[cid]
            r = evaluate_episode_rollout_context(
                cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm
            )
            all_holdout_rows.append(r)

    base_850_check = compute_component_bootstrap_summary(all_850_rows, ["STATIC", "SHUFFLED", "W1"], ("STATIC", "SHUFFLED"))
    print(
        "[Phase 4] Baseline 850-benchmark verification:\n"
        f"  STATIC utility   = {base_850_check['mean_utility']['STATIC']:.6f}\n"
        f"  SHUFFLED utility = {base_850_check['mean_utility']['SHUFFLED']:.6f}\n"
        f"  W1 utility       = {base_850_check['mean_utility']['W1']:.6f} "
        f"(W1 - STATIC = {base_850_check['contrasts']['W1_minus_STATIC']['mean']:.6f})"
    )

    # Precompute fold-safe Leave-One-Family-Out SoS priors for fold 0, fold 1, and full DL-Train
    print("[Phase 4] Computing fold-safe Leave-One-Family-Out Empirical-Bayes SoS program priors...")
    sos_fold0 = build_fold_safe_sos_priors(fold_partitions[0]["gradient"], fold_partitions[0]["OOF"] + fold_partitions[0]["monitor"], pools)
    sos_fold1 = build_fold_safe_sos_priors(fold_partitions[1]["gradient"], fold_partitions[1]["OOF"] + fold_partitions[1]["monitor"], pools)
    sos_full = build_fold_safe_sos_priors(dl_train_cids, dl_val_cids + dl_holdout_cids, pools)
    all_sos_dicts = [sos_fold0, sos_fold1, sos_full]

    # Select strictly non-850 DL-Train contexts for live Rust GP pool expansion so the 850-benchmark OOF,
    # DL-Val, and DL-Holdout evaluation pools stay 100% frozen/untouched for independent generalization!
    non850_train_cids = [c for c in sorted(dl_train_cids) if c not in b850_set]
    active_rust_cids = non850_train_cids[:16]
    shared_evo_cache: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Step 2: Execute Multi-Round Self-Play Flywheel (`ROUND_0` .. `ROUND_3`)
    # -----------------------------------------------------------------------
    round_configs = [
        {
            "round": 0,
            "arm": "ROUND_0",
            "desc": "Base Zero-Shot + Initial Trajectory Policy (A0 + Dope-112 only, CV-only targets, no Landmark-V2 / no EvoQueue / no StructKernel)",
            "use_landmarks": False,
            "use_multimodal": False,
            "use_sos": False,
            "use_cv_test_gate": False,
            "enable_struct_kernel": False,
            "enable_cross_attn": False,
            "enable_ucb": False,
            "enable_evo_queue": False,
            "run_live_rust": False,
        },
        {
            "round": 1,
            "arm": "ROUND_1",
            "desc": "Self-Play Round 1: Real-Time Evolutionary GP Batch Queue (Stratified-300 + DirectionalEvoLoss) + Landmark-V2 Routing + SoS Prior + Head-Aware Structural Kernel",
            "use_landmarks": True,
            "use_multimodal": False,
            "use_sos": True,
            "use_cv_test_gate": False,
            "enable_struct_kernel": True,
            "enable_cross_attn": False,
            "enable_ucb": False,
            "enable_evo_queue": True,
            "run_live_rust": True,
        },
        {
            "round": 2,
            "arm": "ROUND_2",
            "desc": "Self-Play Round 2: Full Multi-Modal Encodings (Kumo-S/Foundation/Hyperion-V6/TabPFN35/Target) + CV-Test Agreement Gate + Trajectory Cross-Attention + Real-Time Evo Queue",
            "use_landmarks": True,
            "use_multimodal": True,
            "use_sos": True,
            "use_cv_test_gate": True,
            "enable_struct_kernel": True,
            "enable_cross_attn": True,
            "enable_ucb": False,
            "enable_evo_queue": True,
            "run_live_rust": True,
        },
        {
            "round": 3,
            "arm": "ROUND_3",
            "desc": "Self-Play Round 3: Full Multi-Round Ratchet + Epistemic UCB Structural Exploration + Cross-Dataset Champion Transfer + Multi-Generation Live Rust GP Evolution",
            "use_landmarks": True,
            "use_multimodal": True,
            "use_sos": True,
            "use_cv_test_gate": True,
            "enable_struct_kernel": True,
            "enable_cross_attn": True,
            "enable_ucb": True,
            "enable_evo_queue": True,
            "run_live_rust": True,
        },
    ]

    prev_state_fold = {0: None, 1: None}
    prev_state_full = None
    round1_state_full = None
    live_expansion_summaries = []
    evo_queue_telemetry_by_round = {}
    full_model_r3 = None
    full_ctx_scaled_r3 = None
    r2_val_rows_snapshot = []
    r2_holdout_rows_snapshot = []

    for rcfg in round_configs:
        r_idx = rcfg["round"]
        arm = rcfg["arm"]
        t_r0 = time.monotonic()
        print(f"\n[Phase 4] === {arm}: {rcfg['desc']} ===")

        f_mask = np.zeros(584, dtype=np.float32)
        f_mask[0:200] = 1.0  # A0 (88D) + Dope-112 (112D)
        if rcfg["use_multimodal"]:
            f_mask[200:520] = 1.0  # Kumo-S, Foundation, Hyperion-V6, TabPFN35, Target (320D)
        if rcfg["use_landmarks"]:
            f_mask[520:584] = 1.0  # H27-FP16 (16D) + Landmark-V2 (48D)

        # 1. Train & Evaluate on 850-Context 2-Fold Benchmark (strictly fold-isolated!)
        fold_q_stats = {}
        for fold_idx in (0, 1):
            grad_cids = fold_partitions[fold_idx]["gradient"]
            mon_cids = fold_partitions[fold_idx]["monitor"]
            oof_cids = fold_partitions[fold_idx]["OOF"]

            med, iqr, m_vec = fit_fold_safe_context_scaler(grad_cids, pools, ctx_raw_584, f_mask)
            ctx_scaled = transform_context_features(ctx_raw_584, med, iqr, m_vec)
            sos_base = sos_fold0 if fold_idx == 0 else sos_fold1
            sos_active = (
                sos_base
                if rcfg["use_sos"]
                else {c: np.zeros_like(v) for c, v in sos_base.items()}
            )

            ckpt_fold_path = CHECKPOINTS_DIR / f"flywheel_{arm.lower()}_fold{fold_idx}.pt"
            if r_idx == 0 and ckpt_fold_path.exists():
                model_f = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4, dropout=0.10).to(device)
                model_f.load_state_dict(torch.load(ckpt_fold_path, map_location=device), strict=False)
                model_f.eval()
                q_stat_f = {"loaded_cached_round0_checkpoint": True}
            else:
                model_f, q_stat_f = train_flywheel_model_for_round(
                    round_idx=r_idx,
                    train_cids=grad_cids,
                    monitor_cids=mon_cids,
                    pools=pools,
                    ctx_scaled_584=ctx_scaled,
                    landmark_v2_48=landmark_v2_48,
                    sos_by_cid=sos_active,
                    champ_bank=champ_bank,
                    device=device,
                    prev_state_dict=prev_state_fold[fold_idx],
                    use_cv_test_gate=rcfg["use_cv_test_gate"],
                    enable_struct_kernel=rcfg["enable_struct_kernel"],
                    enable_cross_attn=rcfg["enable_cross_attn"],
                    enable_evo_queue=rcfg["enable_evo_queue"],
                    naive_label_grafting=False,
                    shared_evo_cache=shared_evo_cache,
                    epochs=10,
                    batch_size=16,
                    seed=20261023 + fold_idx * 50,
                )
            fold_q_stats[f"fold_{fold_idx}"] = q_stat_f
            prev_state_fold[fold_idx] = {k: v.detach().cpu().clone() for k, v in model_f.state_dict().items()}
            torch.save(prev_state_fold[fold_idx], ckpt_fold_path)

            for cid in sorted(oof_cids):
                p = pools[cid]
                rec_idx = p["rec"]["index"]
                row_eval = evaluate_episode_rollout_context(
                    cid=cid,
                    pool=p,
                    ctx_vec_584=ctx_scaled[rec_idx],
                    sos_mat=sos_active[cid],
                    model=model_f,
                    device=device,
                    arm=arm,
                    enable_struct_kernel=rcfg["enable_struct_kernel"],
                    enable_cross_attn=rcfg["enable_cross_attn"],
                    enable_ucb=rcfg["enable_ucb"],
                )
                all_850_rows.append(row_eval)

        # 2. Train & Evaluate on Full-Corpus Family-Disjoint Split (`DL-Train` -> `DL-Val`, `DL-Holdout`)
        med_full, iqr_full, m_full = fit_fold_safe_context_scaler(dl_train_cids, pools, ctx_raw_584, f_mask)
        ctx_scaled_full = transform_context_features(ctx_raw_584, med_full, iqr_full, m_full)
        sos_full_active = (
            sos_full
            if rcfg["use_sos"]
            else {c: np.zeros_like(v) for c, v in sos_full.items()}
        )

        ckpt_full_path = CHECKPOINTS_DIR / f"flywheel_{arm.lower()}_full_corpus.pt"
        if r_idx == 0 and ckpt_full_path.exists():
            model_full = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128, n_heads=4, dropout=0.10).to(device)
            model_full.load_state_dict(torch.load(ckpt_full_path, map_location=device), strict=False)
            model_full.eval()
            q_stat_full = {"loaded_cached_round0_checkpoint": True}
        else:
            model_full, q_stat_full = train_flywheel_model_for_round(
                round_idx=r_idx,
                train_cids=dl_train_cids,
                monitor_cids=dl_val_cids,
                pools=pools,
                ctx_scaled_584=ctx_scaled_full,
                landmark_v2_48=landmark_v2_48,
                sos_by_cid=sos_full_active,
                champ_bank=champ_bank,
                device=device,
                prev_state_dict=prev_state_full,
                use_cv_test_gate=rcfg["use_cv_test_gate"],
                enable_struct_kernel=rcfg["enable_struct_kernel"],
                enable_cross_attn=rcfg["enable_cross_attn"],
                enable_evo_queue=rcfg["enable_evo_queue"],
                naive_label_grafting=False,
                shared_evo_cache=shared_evo_cache,
                epochs=10,
                batch_size=24,
                seed=20261023 + 500,
            )

        evo_queue_telemetry_by_round[arm] = {
            "fold_0": fold_q_stats["fold_0"],
            "fold_1": fold_q_stats["fold_1"],
            "full_corpus": q_stat_full,
        }
        prev_state_full = {k: v.detach().cpu().clone() for k, v in model_full.state_dict().items()}
        if r_idx == 1:
            round1_state_full = prev_state_full
        torch.save(prev_state_full, ckpt_full_path)
        full_model_r3 = model_full
        full_ctx_scaled_r3 = ctx_scaled_full

        if rcfg["run_live_rust"]:
            print(f"[Phase 4] Running Live GP -> Rust Self-Play Expansion (Stratified-300 + Full-JTT1) for {arm} across {len(active_rust_cids)} contexts...")
            live_sum, _ = run_live_rust_gp_selfplay_expansion(
                round_idx=r_idx,
                model=model_full,
                ctx_scaled_584=ctx_scaled_full,
                landmark_v2_48=landmark_v2_48,
                sos_dicts=all_sos_dicts,
                pools=pools,
                champ_bank=champ_bank,
                active_cids=active_rust_cids,
                shared_evo_cache=shared_evo_cache,
                device=device,
                n_generations_300=2,
                children_per_gen_300=4,
                n_confirm_full_jtt1=2,
            )
            live_expansion_summaries.append(live_sum)
            print(
                f"  [Live Rust GP] Screened {live_sum['candidates_screened_in_rust_strat300']} Stratified-300 trees + "
                f"confirmed {live_sum['candidates_confirmed_in_rust_full_jtt1']} Full-JTT1 trees in {live_sum['wall_seconds']}s: "
                f"strat300_delta={live_sum['mean_strat300_evo_fitness_delta']:+.4f}, "
                f"new_HWM={live_sum['s3_persistence']['retention_summary']['new_hwm_count']}, "
                f"storage_pruned={live_sum['s3_persistence']['retention_summary']['storage_reduction_pct']}%, "
                f"rlibs_compiled={live_sum['emitted_rust_rlibs_compiled']}, "
                f"S3_uploaded={live_sum['s3_persistence']['s3_sync']['uploaded']}"
            )

        for cid in sorted(dl_val_cids):
            p = pools[cid]
            rec_idx = p["rec"]["index"]
            rv = evaluate_episode_rollout_context(
                cid, p, ctx_scaled_full[rec_idx], sos_full_active[cid], model_full, device, arm,
                enable_struct_kernel=rcfg["enable_struct_kernel"],
                enable_cross_attn=rcfg["enable_cross_attn"],
                enable_ucb=rcfg["enable_ucb"],
            )
            all_val_rows.append(rv)
            if r_idx == 2:
                r2_val_rows_snapshot.append(rv)

        for cid in sorted(dl_holdout_cids):
            p = pools[cid]
            rec_idx = p["rec"]["index"]
            rh = evaluate_episode_rollout_context(
                cid, p, ctx_scaled_full[rec_idx], sos_full_active[cid], model_full, device, arm,
                enable_struct_kernel=rcfg["enable_struct_kernel"],
                enable_cross_attn=rcfg["enable_cross_attn"],
                enable_ucb=rcfg["enable_ucb"],
            )
            all_holdout_rows.append(rh)
            if r_idx == 2:
                r2_holdout_rows_snapshot.append(rh)

        cur_850 = compute_component_bootstrap_summary(
            [r for r in all_850_rows if r["arm"] in ("STATIC", "SHUFFLED", "W1", "ROUND_0", arm)],
            list(dict.fromkeys(["STATIC", "SHUFFLED", "W1", "ROUND_0", arm])),
        )
        cur_hold = compute_component_bootstrap_summary(
            [r for r in all_holdout_rows if r["arm"] in ("STATIC", "SHUFFLED", "W1", "ROUND_0", arm)],
            list(dict.fromkeys(["STATIC", "SHUFFLED", "W1", "ROUND_0", arm])),
        )
        print(
            f"[Phase 4] {arm} completed in {time.monotonic() - t_r0:.1f}s:\n"
            f"  850-Benchmark Utility = {cur_850['mean_utility'][arm]:.6f} "
            f"(vs W1: {cur_850['contrasts'].get(f'{arm}_minus_W1', {}).get('mean', 0.0):+.6f}, "
            f"vs STATIC: {cur_850['contrasts'].get(f'{arm}_minus_STATIC', {}).get('mean', 0.0):+.6f})\n"
            f"  DL-Holdout Utility    = {cur_hold['mean_utility'][arm]:.6f} "
            f"(vs W1: {cur_hold['contrasts'].get(f'{arm}_minus_W1', {}).get('mean', 0.0):+.6f}, "
            f"vs STATIC: {cur_hold['contrasts'].get(f'{arm}_minus_STATIC', {}).get('mean', 0.0):+.6f})\n"
            f"  EvoQueue Telemetry    = {q_stat_full}"
        )

    # -----------------------------------------------------------------------
    # Step 2b: Controlled Gradient Direction Ablation (`PROPER_DIR_GRAD` vs `NAIVE_LABEL_GRAFT`)
    # -----------------------------------------------------------------------
    grad_ablation_results = run_gradient_direction_ablation(
        dl_train_cids=dl_train_cids,
        dl_val_cids=dl_val_cids,
        dl_holdout_cids=dl_holdout_cids,
        pools=pools,
        ctx_scaled_full=full_ctx_scaled_r3,
        landmark_v2_48=landmark_v2_48,
        sos_full=sos_full,
        champ_bank=champ_bank,
        base_state_dict=round1_state_full,
        shared_evo_cache=shared_evo_cache,
        proper_holdout_rows=r2_holdout_rows_snapshot,
        proper_val_rows=r2_val_rows_snapshot,
        proper_queue_stats=evo_queue_telemetry_by_round.get("ROUND_2", {}).get("full_corpus", {}),
        device=device,
    )

    # -----------------------------------------------------------------------
    # Step 3: Live GP -> Rust Scoring on Previously Unscored `outer` Contexts
    # -----------------------------------------------------------------------
    print("\n[Phase 4] Evaluating FlywheelGameNet Round 3 on previously unscored `outer`-split contexts via live Rust scoring...")
    outer_live_results = evaluate_unscored_outer_split_contexts_live(
        manifest=manifest,
        model=full_model_r3,
        ctx_scaled_584=full_ctx_scaled_r3,
        landmark_v2_48=landmark_v2_48,
        champ_bank=champ_bank,
        device=device,
        n_outer_contexts=12,
    )
    print(
        f"[Phase 4] Unscored `outer`-split live Rust campaign: "
        f"Flywheel won {outer_live_results['flywheel_win_count']}/{outer_live_results['n_unscored_outer_contexts_evaluated']} "
        f"({100.0 * outer_live_results['flywheel_win_rate']:.1f}%), "
        f"mean G0 fitness={outer_live_results['mean_g0_passing_fitness']:.4f} -> "
        f"mean Flywheel fitness={outer_live_results['mean_flywheel_passing_fitness']:.4f}"
    )

    # -----------------------------------------------------------------------
    # Step 4: Final Bootstrap Summaries, S3 Upload & Verification Assertions
    # -----------------------------------------------------------------------
    all_arms = ["STATIC", "SHUFFLED", "W1", "ROUND_0", "ROUND_1", "ROUND_2", "ROUND_3"]
    summary_850 = compute_component_bootstrap_summary(all_850_rows, all_arms)
    summary_val = compute_component_bootstrap_summary(all_val_rows, all_arms)
    summary_holdout = compute_component_bootstrap_summary(all_holdout_rows, all_arms)

    final_report = {
        "schema": "jope.flywheel-selfplay-results/v2",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wall_seconds": round(time.monotonic() - t_start, 2),
        "device": str(device),
        "cpu_throttle_guard": GLOBAL_CPU_GUARD.summary(),
        "arms": all_arms,
        "benchmark_850_162_components": summary_850,
        "full_corpus_dl_val_91_contexts": summary_val,
        "full_corpus_dl_holdout_93_contexts": summary_holdout,
        "evo_batch_queue_telemetry_by_round": evo_queue_telemetry_by_round,
        "gradient_direction_ablation": grad_ablation_results,
        "live_rust_gp_selfplay_rounds": live_expansion_summaries,
        "unscored_outer_split_live_rust_evaluation": outer_live_results,
    }

    report_path = RESULTS_DIR / "flywheel_selfplay_report.json"
    report_path.write_text(json.dumps(final_report, indent=2))
    s3_report_sync = sync_directory_to_s3(
        RESULTS_DIR,
        "s3://veox-jopedime-learning-use2/experiments/flywheel-selfplay-20261023/results/",
    )
    print(f"\n[Phase 4] Wrote comprehensive report to {report_path} (S3 synced={s3_report_sync['uploaded']}) in {final_report['total_wall_seconds']}s.")

    # Print clear summary table
    print("\n================ FLYWHEEL SELF-PLAY PROGRESSION (850-Context 162-Component Benchmark) ================")
    print(f"{'Arm':12s} | {'Mean Utility':>14s} | {'B=8':>11s} | {'B=16':>11s} | {'B=32':>11s} | {'B=64':>11s} | {'Gate Fail':>9s}")
    print("-" * 92)
    for a in all_arms:
        u = summary_850["mean_utility"][a]
        c = summary_850["mean_raw_best_F_by_budget"][a]
        gf = summary_850["gate_counts"][a]["8"]["rollout_failure_count"]
        print(f"{a:12s} | {u:14.6f} | {c['8']:11.6f} | {c['16']:11.6f} | {c['32']:11.6f} | {c['64']:11.6f} | {gf:4d}/2550")

    print("\n================ FLYWHEEL SELF-PLAY PROGRESSION (Full-Corpus Family-Disjoint DL-Holdout) ================")
    print(f"{'Arm':12s} | {'Mean Utility':>14s} | {'B=8':>11s} | {'B=16':>11s} | {'B=32':>11s} | {'B=64':>11s} | {'Gate Fail':>9s}")
    print("-" * 92)
    for a in all_arms:
        u = summary_holdout["mean_utility"][a]
        c = summary_holdout["mean_raw_best_F_by_budget"][a]
        gf = summary_holdout["gate_counts"][a]["8"]["rollout_failure_count"]
        denom = summary_holdout["gate_counts"][a]["8"]["context_seed_denominator"]
        print(f"{a:12s} | {u:14.6f} | {c['8']:11.6f} | {c['16']:11.6f} | {c['32']:11.6f} | {c['64']:11.6f} | {gf:4d}/{denom}")


if __name__ == "__main__":
    main()
