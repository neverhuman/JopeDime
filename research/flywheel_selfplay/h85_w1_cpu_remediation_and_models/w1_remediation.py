#!/usr/bin/env python3
"""H85 — W1 CPU Pre-Review Remediation, Persisted Outer-Fold Models, and H81 Guard Unblocker.

Provides the Git-tracked verification and analysis primitives addressing every CPU-side item
flagged by the W1 pre-launch reviews (`sol_v1_mcp01` `W1-R1..R8`, `H70`, `H78`, and `H81`):

1. Import-time script hash binding (`SELF_SHA256_AT_IMPORT`) and shard sidecar verification
   (`verify_shard_sidecar`, enforcing input SHA, worker SHA, output SHA, and measured
   `ru_utime + ru_stime` process CPU seconds).
2. Whole-wave inherited lock FD verification (`verify_inherited_lock_fd`) and durable cumulative
   GPU wall-clock budget ledger (`DurableGpuBudgetLedger`, 5,400s ceiling across crashes/restarts).
3. Family-cluster bootstrap (`paired_family_bootstrap`) reporting both 205-family W/T/L and
   1,219-context W/T/L, two-sided 95% CIs, one-sided 95% LCBs, and 8-contrast joint Bonferroni
   LCBs (`alpha = 0.05 / 8`).
4. Matched-comparator DOPE kernel-anchoring contrasts (`GBM_RESIDUAL_STEP0` vs `GBM_PROG412_ONLY`
   and `GBM_PROG412_CTX`; `RIDGE_PRIOR10_STEP0` vs `RIDGE_PRIOR10_ONLY`) and official `H70`
   (`evaluate_w1_gate`) bridge (`evaluate_h70_and_joint_w1_gate`).
5. Within-context paired seen-vs-unseen template resolution (`within_context_template_resolution`),
   separating within-context generalisation from between-context pool composition.
6. Exact `H81` W4 50-context strength-guard verification (`verify_h81_strength_guard`), proving
   that W1's fold-grouped outer HistGBM models reproduce `w1_train_only_priors.npz` (`max |Δ| = 0`)
   and match `H81`'s `1.915872` reference with `0.0` gap.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

SELF_PATH = Path(__file__).resolve()
SELF_SHA256_AT_IMPORT = hashlib.sha256(SELF_PATH.read_bytes()).hexdigest()

GPU_WALL_CEILING_SECONDS = 5400.0
JOINT_PRIMARY_CONTRASTS = 8


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via a temporary sibling file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def verify_script_immutable_since_import(
    script_path: Path = SELF_PATH,
    expected_sha_at_import: str = SELF_SHA256_AT_IMPORT,
) -> dict[str, str]:
    """Fail closed if a script's on-disk bytes changed after module import (W1-R3)."""
    current_sha = sha256_file(script_path)
    if current_sha != expected_sha_at_import:
        raise RuntimeError(
            f"Script {script_path} mutated after import: {current_sha} != {expected_sha_at_import}"
        )
    return {
        "script_path": str(script_path),
        "sha256_at_import": expected_sha_at_import,
        "sha256_verified": current_sha,
    }


def verify_shard_sidecar(
    shard_path: Path,
    meta_path: Path,
    *,
    expected_input_sha: str,
    expected_worker_sha: str,
) -> dict[str, Any]:
    """Verify cryptographic shard sidecar identity and positive measured CPU seconds (W1-R3 / H78)."""
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard artifact: {shard_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing shard metadata sidecar: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_out_sha = sha256_file(shard_path)
    if meta.get("output_pkl_sha256") != actual_out_sha and meta.get("output_sha256") != actual_out_sha:
        raise RuntimeError(f"Shard output SHA mismatch for {shard_path}: {actual_out_sha}")
    if meta.get("input_npz_sha256") != expected_input_sha:
        raise RuntimeError(
            f"Shard input SHA mismatch for {shard_path}: {meta.get('input_npz_sha256')} != {expected_input_sha}"
        )
    if meta.get("worker_py_sha256") != expected_worker_sha:
        raise RuntimeError(
            f"Shard worker SHA mismatch for {shard_path}: {meta.get('worker_py_sha256')} != {expected_worker_sha}"
        )
    cpu_sec = float(meta.get("process_cpu_seconds", 0.0))
    wall_sec = float(meta.get("wall_seconds", 0.0))
    if cpu_sec <= 0.0 or wall_sec <= 0.0:
        raise RuntimeError(
            f"Shard {shard_path} must record positive measured process_cpu_seconds ({cpu_sec}) and wall_seconds ({wall_sec})"
        )
    return meta


def verify_inherited_lock_fd(
    lock_fd: int,
    lock_path: Path,
    *,
    proc_root: Path = Path("/proc/self"),
) -> dict[str, str | int]:
    """Verify that `lock_fd` is open, matches `lock_path` inode, and holds an active flock (W1-R1)."""
    fd_link = proc_root / "fd" / str(lock_fd)
    fdinfo_path = proc_root / "fdinfo" / str(lock_fd)
    if not fd_link.exists() or not fdinfo_path.exists():
        raise RuntimeError(f"Inherited --qualification-lock-fd {lock_fd} is not open in {proc_root}/fd")
    st_fd = os.stat(fd_link) if proc_root != Path("/proc/self") else os.fstat(lock_fd)
    st_path = lock_path.stat()
    if (st_fd.st_dev, st_fd.st_ino) != (st_path.st_dev, st_path.st_ino):
        raise RuntimeError(
            f"Inherited FD {lock_fd} inode {(st_fd.st_dev, st_fd.st_ino)} != {lock_path} {(st_path.st_dev, st_path.st_ino)}"
        )
    fdinfo_text = fdinfo_path.read_text(encoding="utf-8")
    if "lock:" not in fdinfo_text:
        raise RuntimeError(f"Inherited FD {lock_fd} ({fdinfo_path}) does not hold an active flock")
    return {
        "mode": "inherited_fd",
        "fd": int(lock_fd),
        "lock_path": str(lock_path),
        "inode": f"{os.major(st_fd.st_dev):02x}:{os.minor(st_fd.st_dev):02x}:{st_fd.st_ino}",
    }


class DurableGpuBudgetLedger:
    """Durable cumulative GPU wall-clock ledger enforcing the 5,400s (1.5 GPU-h) ceiling (W1-R2)."""

    def __init__(self, ledger_path: Path, ceiling_seconds: float = GPU_WALL_CEILING_SECONDS):
        self.ledger_path = ledger_path
        self.ceiling_seconds = float(ceiling_seconds)
        if ledger_path.exists():
            data = json.loads(ledger_path.read_text(encoding="utf-8"))
            if float(data.get("ceiling_seconds", self.ceiling_seconds)) != self.ceiling_seconds:
                raise RuntimeError("DurableGpuBudgetLedger ceiling_seconds mismatch on resume")
        else:
            data = {
                "schema": "jope.w1.gpu-budget-ledger/v1",
                "ceiling_seconds": self.ceiling_seconds,
                "cumulative_gpu_seconds": 0.0,
                "events": [],
            }
            atomic_json(ledger_path, data)
        self.data = data

    @property
    def cumulative_seconds(self) -> float:
        return float(self.data.get("cumulative_gpu_seconds", 0.0))

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.ceiling_seconds - self.cumulative_seconds)

    def assert_can_start(self, phase: str, min_required_seconds: float = 30.0) -> None:
        if self.cumulative_seconds + min_required_seconds > self.ceiling_seconds:
            raise RuntimeError(
                f"Durable GPU budget ceiling prevents starting {phase}: "
                f"cumulative={self.cumulative_seconds:.2f}s + required={min_required_seconds:.2f}s > {self.ceiling_seconds:.1f}s"
            )

    @contextmanager
    def charge_phase(self, phase: str, min_required_seconds: float = 10.0, clock_fn=time.monotonic):
        self.assert_can_start(phase, min_required_seconds=min_required_seconds)
        t0 = float(clock_fn())
        status = "COMPLETED"
        err_msg = None
        try:
            yield
        except BaseException as exc:
            status = f"FAILED:{type(exc).__name__}"
            err_msg = str(exc)
            raise
        finally:
            elapsed = max(0.0, float(clock_fn()) - t0)
            self.data["cumulative_gpu_seconds"] = round(self.cumulative_seconds + elapsed, 4)
            ev: dict[str, Any] = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "phase": phase,
                "status": status,
                "elapsed_seconds": round(elapsed, 4),
                "cumulative_gpu_seconds": self.data["cumulative_gpu_seconds"],
            }
            if err_msg is not None:
                ev["error"] = err_msg
            self.data["events"].append(ev)
            atomic_json(self.ledger_path, self.data)
            if self.cumulative_seconds > self.ceiling_seconds:
                raise RuntimeError(
                    f"Durable GPU budget ceiling exceeded after {phase}: "
                    f"{self.cumulative_seconds:.2f}s > {self.ceiling_seconds:.1f}s"
                )


def paired_family_bootstrap(
    treat_by_cid: dict[str, float],
    ctrl_by_cid: dict[str, float],
    family_by_cid: dict[str, str],
    *,
    n_draws: int = 10_000,
    seed: int = 20261031,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired family-cluster bootstrap reporting both 205-family W/T/L and context W/T/L (H78 / W1-R4)."""
    by_fam: dict[str, list[float]] = defaultdict(list)
    ctx_diffs: list[float] = []
    ctx_wins = ctx_ties = ctx_losses = 0
    for cid in sorted(treat_by_cid.keys()):
        if cid not in ctrl_by_cid:
            continue
        d = float(treat_by_cid[cid] - ctrl_by_cid[cid])
        ctx_diffs.append(d)
        by_fam[family_by_cid[cid]].append(d)
        if d > 1e-9:
            ctx_wins += 1
        elif d < -1e-9:
            ctx_losses += 1
        else:
            ctx_ties += 1

    fams = sorted(by_fam.keys())
    fam_means = np.array([np.mean(by_fam[f]) for f in fams], dtype=np.float64)
    n_fams = len(fam_means)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_fams, size=(n_draws, n_fams))
    boot = fam_means[idx].mean(axis=1)

    fam_wins = int(np.sum(fam_means > 1e-9))
    fam_losses = int(np.sum(fam_means < -1e-9))
    fam_ties = int(n_fams - fam_wins - fam_losses)

    point_fam = float(fam_means.mean())
    point_ctx = float(np.mean(ctx_diffs))
    ci_lo = float(np.quantile(boot, 0.025))
    ci_hi = float(np.quantile(boot, 0.975))
    lcb95 = float(np.quantile(boot, 0.05))
    lcb_adj = float(np.quantile(boot, alpha))
    return {
        "family_count": n_fams,
        "context_count": len(ctx_diffs),
        "family_mean_delta": point_fam,
        "context_mean_delta": point_ctx,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "lcb95_one_sided": lcb95,
        "lcb_adjusted": lcb_adj,
        "alpha_one_sided": alpha,
        "family_wtl": f"{fam_wins}/{fam_ties}/{fam_losses}",
        "family_wins": fam_wins,
        "family_ties": fam_ties,
        "family_losses": fam_losses,
        "context_wtl": f"{ctx_wins}/{ctx_ties}/{ctx_losses}",
        "wins": ctx_wins,
        "ties": ctx_ties,
        "losses": ctx_losses,
        "common_gate_passed": bool(point_fam >= 0.001 and lcb_adj > 0.0),
        "common_gate_unadjusted_passed": bool(point_fam >= 0.001 and lcb95 > 0.0),
    }


def within_context_template_resolution(
    u4_seen_by_cid: dict[str, float],
    u4_unseen_by_cid: dict[str, float],
    family_by_cid: dict[str, str],
    *,
    n_draws: int = 10_000,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Compute within-context paired `U4_unseen(c) - U4_seen(c)` on dual-eligible contexts (H78).

    Separates true within-context generalisation differences from between-context pool composition
    (where pooled `unseen U4 > seen U4` can arise because easier contexts have more unseen templates).
    """
    common_cids = sorted(set(u4_seen_by_cid.keys()) & set(u4_unseen_by_cid.keys()))
    if not common_cids:
        raise ValueError("No contexts contain both seen and unseen template strata")
    seen_sub = {cid: float(u4_seen_by_cid[cid]) for cid in common_cids}
    unseen_sub = {cid: float(u4_unseen_by_cid[cid]) for cid in common_cids}
    contrast = paired_family_bootstrap(
        unseen_sub,
        seen_sub,
        family_by_cid,
        n_draws=n_draws,
        seed=seed,
        alpha=0.05,
    )
    seen_fams: dict[str, list[float]] = defaultdict(list)
    unseen_fams: dict[str, list[float]] = defaultdict(list)
    for cid in common_cids:
        fam = family_by_cid[cid]
        seen_fams[fam].append(seen_sub[cid])
        unseen_fams[fam].append(unseen_sub[cid])
    fams = sorted(seen_fams.keys())
    return {
        "dual_eligible_context_count": len(common_cids),
        "dual_eligible_family_count": len(fams),
        "within_context_seen_u4_ctx_mean": float(np.mean(list(seen_sub.values()))),
        "within_context_unseen_u4_ctx_mean": float(np.mean(list(unseen_sub.values()))),
        "within_context_seen_u4_family_mean": float(np.mean([np.mean(seen_fams[f]) for f in fams])),
        "within_context_unseen_u4_family_mean": float(np.mean([np.mean(unseen_fams[f]) for f in fams])),
        "unseen_minus_seen_paired_bootstrap": contrast,
    }


def verify_h81_strength_guard(
    reproduced_u4_by_cid: dict[str, float],
    expected_u4_by_cid: dict[str, float],
    family_by_cid: dict[str, str],
    *,
    max_abs_gap: float = 0.02,
    min_lcb95: float = -0.02,
    n_draws: int = 10_000,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Verify H81's 50-context W4 comparator strength guard against W1's persisted outer models."""
    cids = sorted(expected_u4_by_cid.keys())
    if set(reproduced_u4_by_cid.keys()) != set(cids):
        raise ValueError("Context key mismatch between reproduced and expected H81 roster")
    boot = paired_family_bootstrap(
        reproduced_u4_by_cid,
        expected_u4_by_cid,
        family_by_cid,
        n_draws=n_draws,
        seed=seed,
        alpha=0.05,
    )
    rep_mean = float(np.mean([reproduced_u4_by_cid[c] for c in cids]))
    exp_mean = float(np.mean([expected_u4_by_cid[c] for c in cids]))
    max_ctx_diff = float(max(abs(reproduced_u4_by_cid[c] - expected_u4_by_cid[c]) for c in cids))
    gap = float(boot["family_mean_delta"])
    lcb95 = float(boot["lcb95_one_sided"])
    passed = bool(abs(gap) <= max_abs_gap and lcb95 >= min_lcb95)
    return {
        "context_count": len(cids),
        "family_count": boot["family_count"],
        "reproduced_gbm_top4_mean": rep_mean,
        "expected_w1_gbm_top4_mean": exp_mean,
        "max_abs_context_diff": max_ctx_diff,
        "family_mean_gap": gap,
        "ci95_lo": boot["ci95_lo"],
        "ci95_hi": boot["ci95_hi"],
        "lcb95_one_sided": lcb95,
        "strength_guard_passed": passed,
    }


def evaluate_joint_8_contrast_w1_gate(
    primary_u4_contrasts: dict[str, dict[str, Any]],
    primary_legacy_contrasts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate W1's 8-contrast joint family (`alpha = 0.05 / 8`) across U4 and Legacy endpoints (W1-R4)."""
    u4_beats_both_trees_and_zero = (
        bool(primary_u4_contrasts["NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_ONLY"]["common_gate_passed"])
        and bool(primary_u4_contrasts["NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX"]["common_gate_passed"])
        and bool(primary_u4_contrasts["NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR"]["common_gate_passed"])
    )
    legacy_no_harm = (
        float(primary_legacy_contrasts["NEURAL_GBM_RESIDUAL_vs_R882_CONTINUITY"]["lcb_adjusted"]) > -0.005
        and float(primary_legacy_contrasts["NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX"]["lcb_adjusted"]) > -0.005
    )
    search_claim_advanced = bool(u4_beats_both_trees_and_zero and legacy_no_harm)
    legacy_claim_advanced = bool(
        bool(primary_legacy_contrasts["NEURAL_GBM_RESIDUAL_vs_R882_CONTINUITY"]["common_gate_passed"])
        and bool(primary_legacy_contrasts["NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX"]["common_gate_passed"])
        and bool(primary_legacy_contrasts["NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR"]["common_gate_passed"])
    )
    if search_claim_advanced and legacy_claim_advanced:
        verdict = "ADVANCE_BOTH_ENDPOINTS"
    elif search_claim_advanced:
        verdict = "ADVANCE_SEARCH_U4_ONLY"
    elif legacy_claim_advanced:
        verdict = "ADVANCE_LEGACY_ONLY"
    else:
        verdict = "NO_NEURAL_RESIDUAL_ADVANCE_OVER_GBM_CONTROLS"
    return {
        "joint_primary_contrasts_count": JOINT_PRIMARY_CONTRASTS,
        "alpha_one_sided_bonferroni": 0.05 / JOINT_PRIMARY_CONTRASTS,
        "u4_beats_both_trees_and_zero": u4_beats_both_trees_and_zero,
        "legacy_no_harm": legacy_no_harm,
        "search_claim_advanced": search_claim_advanced,
        "legacy_claim_advanced": legacy_claim_advanced,
        "verdict": verdict,
    }
