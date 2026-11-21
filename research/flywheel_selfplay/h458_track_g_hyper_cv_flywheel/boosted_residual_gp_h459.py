#!/usr/bin/env python3
"""Wave H459 / Track S9 Arm A5: Orthogonal Cross-Fitted Functional Gradient Boosted GP (`OC-FGB-GP`)
(`boosted_residual_gp_h459.py`).

Implements `Arm A5` (`OC-FGB-GP`) and its exact worker-call-matched single-stage control (`A0_iso`)
under pre-registration `/home/ubuntu/jope_plan/evidence/H459/PREREG_S9_ARM_A5_OC_FGB_GP.md`:
  1. Effort-Tier Gating (`crates/jope-service/src/discover/effort.rs` Parity):
     - `fast` (`24` evals / `10s`): HARD-REFUSED for multi-stage residual boosting (`medium`/`high`/`ultra` ONLY).
     - `medium` (`5` stages), `high` (`7` stages), `ultra` (`10` stages), supporting:
       * `--budget-mode product` (PRIMARY): hard `WorkerCallBudgetMeter` enforcing `128 / 512 / 4,096` total
         `SUPERSET_BIN --fit-s1` admissions and `60s / 300s / 1,800s` wall-clock limits (counting EVERY Gen-0 seed,
         `CV-2` screen, rotating OOF fold, Tier-B vault fold, Tier-C stack fold, and `PA` refit).
       * `--budget-mode cycle` (SECONDARY): `[5 stg / 100g, 7 stg / 300g, 10 stg / 1000g]` horizon curves.
  2. Six Algorithmic Advances Over Naive Sequential Residual GP:
     - Rotating-Partition OOF Residual Handoff (`rep = 200 + m` on stage winner ONLY — `0` `K x` search blowup,
       `0` in-sample DoF leakage, `0` fold-boundary resonance across stages).
     - Per-Stage `DOPE` + `QHatV4` Residual Re-Contextualization (`(Xs, r^{(m-1)})` re-encoded every stage).
     - 5-Island Multi-Geometry Pseudo-Residuals:
       * Islands 0-1: Exact L2 residual (`op_target_warp` in Stage 1; signed `symlog`/`winsor`/`robust_scale` in m >= 2).
       * Island 2: Fold-causal MAD Pseudo-Huber gradient `delta * asinh(r / delta)` with bounded `sinh` restoration
         (protecting `|kurtosis| > 10` heavy-tail contexts from outlier leverage hijacking).
       * Island 3: Local kernel / heteroscedastic smoother specialist (`gpr`, `krr`, `ksmooth`, `rff_cos`).
       * Island 4: Feature-subspace / oblique-hinge specialist + explicit `QHatV4` AST cosine & OOF residual
         correlation penalty against prior stage winners `{f_1*, ..., f_{m-1}*}`.
     - Stagewise Complexity Annealing (`max_transforms`: `4 -> 3 -> 2` across early/mid/late residual stages).
     - Unseen Tier-B Vault (`rep = 77`) Scale-Invariant Coordinate Backfitting (`w_m in [0, 1]`):
       * Exports BOTH `a5_shrunk` (smooth age-damped ridge-backfitted ensemble) AND `a5_ratchet` (strict monotonic
         Tier-B vault prefix) + stagewise trajectory columns `a5_stg1 .. a5_stgM`.
     - Untouched Tier-C (`rep = 101, 102`) Stack Cross-Fit + `PA` Refit (`_load_col_block` `.npz` + `SIDECAR.json`).
  3. Zero `ya` Bytes Ever Loaded (`assert "ya" not in zb.files`), Zero File Deletions, Zero `/tmp` Usage.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["H335_GROUP"] = "AGG"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from scipy.stats import rankdata

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
for _cand in reversed((
    str(HERE),
    "/home/ubuntu/h450-stage/pub-r6/code",
    "/home/ubuntu/h450-stage/fleet-r10/code",
)):
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)
if sys.path[0] != str(HERE):
    sys.path.insert(0, str(HERE))

import h335_core as K  # noqa: E402
from bagged_neural_seed_h452 import (  # noqa: E402
    BaggedNeuralEncoderH452,
    _stream_seed,
    build_bagged_gen0_population,
    stratified_subsample_80pct,
)
from hyper_cv_test_gp_harvester_h458 import compute_fold_utility_spread  # noqa: E402
from train_qhat_v4_generalizability_h458 import PretrainedQHatV4Prior  # noqa: E402
from train_surrogate_qhat_v3_h454 import featurize_candidate_h454  # noqa: E402
from ultra_gp_evolve_1000g_h453 import PretrainedQHatPrior, TwoTierSlotWorkerRunner, op_target_warp_h453  # noqa: E402
from ultra_gp_evolve_h452 import (  # noqa: E402
    SUPERSET_BIN,
    SUPERSET_SHA,
    _parse_indices,
    _sha256_file,
    featurize_candidate_h452,
    fold_log_utility,
    make_stratified_kfold,
    pack_jtt1_bytes,
)

C, D, G = K.C, K.D, K.G

FORBIDDEN_PATH_TOKENS = ("CONFIRM_FINAL", "FMGAP_B", "FMGAP_C", "SEALED", "/tmp/")

KNOWN_BANKS: Dict[str, Tuple[str, str]] = {
    "/home/ubuntu/jope-data/h430/bank_P201B.npz": (
        "da29e8ec64f623e0fa2ab29b31bbd40b91bef36adf8ddc8e2e773f51c96d01af",
        "P201B",
    ),
    "/home/ubuntu/jope-data/h335-cl/h424/dev/bank_FITSEL.npz": (
        "f8849ccb7343cfec1c0fd6688aaafef7580b85001046f2707fab58c5ecebe069",
        "FITSEL",
    ),
    "/home/ubuntu/jope-data/h458s-quarantine/probe_fresh_rows/board_v1/bank_d91probe_f64.npz": (
        "ffe5f998181fda96bf6c55479d4bc89e9fc671d893b4b592c34cc67505c837cd",
        "dev91-freshprobe-v1",
    ),
}


@dataclass(frozen=True)
class EffortSchedule:
    effort: str
    budget_mode: str
    budget_unit: str  # "candidates" (effort.rs Evaluations) or "worker_calls"
    n_stages: int
    max_candidates: int
    max_calls: int
    wall_seconds: float
    stage_cand_quotas: Tuple[int, ...]
    stage_gens: Tuple[int, ...]
    stage_gen0_pop: int
    eval_per_gen: int
    k_oof: int
    k_tier_c: int
    separate_tier_c: bool
    tier_c_repeats: int


def resolve_effort_schedule(
    effort: str,
    budget_mode: str = "product",
    budget_unit: str = "candidates",
) -> EffortSchedule:
    """Maps `(effort, budget_mode, budget_unit)` to exact stage/candidate/call/wall parameters; refuses `fast`."""
    eff = effort.strip().lower()
    bmode = budget_mode.strip().lower()
    bunit = budget_unit.strip().lower()
    if eff == "fast":
        raise ValueError(
            "Effort 'fast' (24 evaluations / 10s) is explicitly excluded from multi-stage residual GP (S9/Arm A5); "
            "use effort in {'medium', 'high', 'ultra'}."
        )
    if eff not in ("medium", "high", "ultra"):
        raise ValueError(f"Unknown effort tier {effort!r}; expected one of {'medium', 'high', 'ultra'}")
    if bmode not in ("product", "cycle"):
        raise ValueError(f"Unknown budget_mode {budget_mode!r}; expected 'product' or 'cycle'")
    if bunit not in ("candidates", "worker_calls"):
        raise ValueError(f"Unknown budget_unit {budget_unit!r}; expected 'candidates' or 'worker_calls'")

    if bmode == "product":
        if bunit == "candidates":
            # Primary shared S9 unit (`cl_fwriter` 01:10Z + `effort.rs::EvaluationBudget`):
            # 128 / 512 / 4,096 unique admitted candidate programs evaluated; exact worker_calls logged alongside.
            # Notice at EVERY tier: A5_total_worker_calls < A0_1p5x_total_worker_calls!
            if eff == "medium":
                # 128 candidates + 5 * (4 rot_oof + 4 vault + 5 tierC + 1 PA = 14) = 198 worker calls (< A0_1p5x's 202 calls)
                return EffortSchedule(
                    effort="medium",
                    budget_mode="product",
                    budget_unit="candidates",
                    n_stages=5,
                    max_candidates=128,
                    max_calls=210,
                    wall_seconds=60.0,
                    stage_cand_quotas=(28, 26, 26, 24, 24),
                    stage_gens=(16, 14, 14, 12, 12),
                    stage_gen0_pop=12,
                    eval_per_gen=1,
                    k_oof=4,
                    k_tier_c=5,
                    separate_tier_c=True,
                    tier_c_repeats=1,
                )
            if eff == "high":
                # 512 candidates + 7 * (5 rot_oof + 5 vault + 5 tierC + 1 PA = 16) = 624 worker calls (< A0_1p5x's 779 calls)
                return EffortSchedule(
                    effort="high",
                    budget_mode="product",
                    budget_unit="candidates",
                    n_stages=7,
                    max_candidates=512,
                    max_calls=650,
                    wall_seconds=300.0,
                    stage_cand_quotas=(80, 76, 74, 72, 72, 70, 68),
                    stage_gens=(31, 29, 28, 27, 27, 26, 25),
                    stage_gen0_pop=18,
                    eval_per_gen=2,
                    k_oof=5,
                    k_tier_c=5,
                    separate_tier_c=True,
                    tier_c_repeats=1,
                )
            # ultra: 4,096 candidates + 10 * (7 rot_oof + 7 vault + 14 tierC + 1 PA = 29) = 4,386 calls (< A0_1p5x's 6,166 calls)
            return EffortSchedule(
                effort="ultra",
                budget_mode="product",
                budget_unit="candidates",
                n_stages=10,
                max_candidates=4096,
                max_calls=4450,
                wall_seconds=1800.0,
                stage_cand_quotas=(460, 440, 430, 420, 410, 400, 396, 385, 380, 375),
                stage_gens=(145, 138, 134, 130, 127, 124, 122, 118, 117, 115),
                stage_gen0_pop=32,
                eval_per_gen=3,
                k_oof=7,
                k_tier_c=7,
                separate_tier_c=True,
                tier_c_repeats=2,
            )

        # Strict raw subprocess worker_calls cap (`--budget-unit worker_calls`)
        if eff == "medium":
            return EffortSchedule(
                effort="medium",
                budget_mode="product",
                budget_unit="worker_calls",
                n_stages=5,
                max_candidates=98,
                max_calls=128,
                wall_seconds=60.0,
                stage_cand_quotas=(20, 20, 20, 19, 19),
                stage_gens=(14, 14, 13, 12, 12),
                stage_gen0_pop=6,
                eval_per_gen=1,
                k_oof=5,
                k_tier_c=5,
                separate_tier_c=False,
                tier_c_repeats=0,
            )
        if eff == "high":
            return EffortSchedule(
                effort="high",
                budget_mode="product",
                budget_unit="worker_calls",
                n_stages=7,
                max_candidates=400,
                max_calls=512,
                wall_seconds=300.0,
                stage_cand_quotas=(62, 60, 58, 56, 56, 54, 54),
                stage_gens=(26, 24, 22, 20, 18, 16, 14),
                stage_gen0_pop=10,
                eval_per_gen=2,
                k_oof=5,
                k_tier_c=5,
                separate_tier_c=True,
                tier_c_repeats=1,
            )
        return EffortSchedule(
            effort="ultra",
            budget_mode="product",
            budget_unit="worker_calls",
            n_stages=10,
            max_candidates=3800,
            max_calls=4096,
            wall_seconds=1800.0,
            stage_cand_quotas=(400, 390, 385, 380, 380, 375, 375, 375, 370, 370),
            stage_gens=(160, 145, 135, 125, 115, 110, 105, 100, 95, 90),
            stage_gen0_pop=24,
            eval_per_gen=3,
            k_oof=7,
            k_tier_c=7,
            separate_tier_c=True,
            tier_c_repeats=2,
        )

    # Secondary logged `--budget-mode cycle` (`100g / 300g / 1000g` horizon curve)
    if eff == "medium":
        return EffortSchedule(
            effort="medium",
            budget_mode="cycle",
            budget_unit="candidates",
            n_stages=5,
            max_candidates=270,
            max_calls=420,
            wall_seconds=120.0,
            stage_cand_quotas=(64, 58, 54, 50, 44),
            stage_gens=(25, 22, 20, 18, 15),
            stage_gen0_pop=14,
            eval_per_gen=2,
            k_oof=5,
            k_tier_c=5,
            separate_tier_c=True,
            tier_c_repeats=1,
        )
    if eff == "high":
        return EffortSchedule(
            effort="high",
            budget_mode="cycle",
            budget_unit="candidates",
            n_stages=7,
            max_candidates=1026,
            max_calls=1250,
            wall_seconds=480.0,
            stage_cand_quotas=(198, 168, 153, 138, 123, 123, 123),
            stage_gens=(60, 50, 45, 40, 35, 35, 35),
            stage_gen0_pop=18,
            eval_per_gen=3,
            k_oof=7,
            k_tier_c=7,
            separate_tier_c=True,
            tier_c_repeats=1,
        )
    return EffortSchedule(
        effort="ultra",
        budget_mode="cycle",
        budget_unit="candidates",
        n_stages=10,
        max_candidates=3240,
        max_calls=4096,
        wall_seconds=1800.0,
        stage_cand_quotas=(474, 414, 384, 354, 324, 294, 279, 249, 234, 234),
        stage_gens=(150, 130, 120, 110, 100, 90, 85, 75, 70, 70),
        stage_gen0_pop=24,
        eval_per_gen=3,
        k_oof=7,
        k_tier_c=7,
        separate_tier_c=True,
        tier_c_repeats=2,
    )


class WorkerCallBudgetMeter:
    """Thread-safe exact mirror of `EvaluationBudget::admit` in `crates/jope-service/src/discover/effort.rs`.

    Tracks BOTH unique admitted candidate evaluations (`candidates_admitted`, the `effort.rs` unit)
    AND exact `SUPERSET_BIN` subprocess admissions (`worker_calls_admitted`, categorized by role).
    Timed-out or failed evaluations remain counted.
    """

    def __init__(self, max_candidates: int, max_calls: int, wall_seconds: float) -> None:
        self.max_candidates = max(1, int(max_candidates))
        self.max_calls = max(1, int(max_calls))
        self.wall_seconds = float(wall_seconds)
        self.t0 = time.monotonic()
        self._candidates_admitted = 0
        self._admitted = 0
        self._calls_by_kind: Dict[str, int] = {
            "search_screen": 0,
            "rot_oof": 0,
            "vault_rep77": 0,
            "tier_c": 0,
            "pa_refit": 0,
        }
        self._lock = threading.Lock()

    def elapsed_s(self) -> float:
        return float(time.monotonic() - self.t0)

    def candidates_admitted(self) -> int:
        with self._lock:
            return self._candidates_admitted

    def remaining_candidates(self) -> int:
        with self._lock:
            return max(0, self.max_candidates - self._candidates_admitted)

    def admitted(self) -> int:
        with self._lock:
            return self._admitted

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_calls - self._admitted)

    def calls_by_kind(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._calls_by_kind)

    def wall_expired(self, reserve_s: float = 0.0) -> bool:
        return self.elapsed_s() + reserve_s >= self.wall_seconds

    def admit_batch(
        self,
        n_want: int,
        ignore_wall: bool = False,
        is_candidate_eval: bool = False,
        kind: str = "search_screen",
    ) -> int:
        """Admits up to `n_want` worker calls within `max_calls` (and `max_candidates` when `is_candidate_eval`)."""
        if n_want <= 0:
            return 0
        if not ignore_wall and self.wall_expired():
            return 0
        with self._lock:
            avail_calls = max(0, self.max_calls - self._admitted)
            if is_candidate_eval:
                avail_cands = max(0, self.max_candidates - self._candidates_admitted)
                take = min(int(n_want), avail_calls, avail_cands)
                self._candidates_admitted += take
            else:
                take = min(int(n_want), avail_calls)
            self._admitted += take
            self._calls_by_kind[kind] = self._calls_by_kind.get(kind, 0) + take
            return take


class BudgetedSlotRunner:
    """Wraps `TwoTierSlotWorkerRunner` so EVERY `SUPERSET_BIN` invocation passes through `WorkerCallBudgetMeter`."""

    def __init__(self, slot_dir: Path, meter: WorkerCallBudgetMeter, n_slots: int = 3, rayon_threads: int = 1) -> None:
        self.inner = TwoTierSlotWorkerRunner(slot_dir, n_slots=n_slots, rayon_threads=rayon_threads)
        self.meter = meter

    def close(self) -> None:
        self.inner.close()

    def map_s1_budgeted(
        self,
        tasks: Sequence[Tuple[bytes, G.Candidate]],
        timeout_s: float,
        finalizing: bool = False,
        is_candidate_eval: bool = False,
        kind: str = "search_screen",
    ) -> List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]]:
        n_adm = self.meter.admit_batch(
            len(tasks), ignore_wall=finalizing, is_candidate_eval=is_candidate_eval, kind=kind
        )
        if n_adm <= 0:
            return [(None, False) for _ in tasks]
        sub = list(tasks[:n_adm])
        res = self.inner.map_s1(sub, timeout_s=timeout_s)
        if n_adm < len(tasks):
            res.extend([(None, False) for _ in range(len(tasks) - n_adm)])
        return res


PINNED_PRIORS: Dict[str, Tuple[Path, str]] = {
    "dev91-free": (
        Path("/home/ubuntu/jope-data/h452-flywheel/surrogate_v2/qhat_v2.pt"),
        "bf05e0404c9e7a2e4d90a8672e4a5b10312b8c95db142a0cd003f83eb9b208e1",
    ),
    "in-corpus-v2": (
        Path("/home/ubuntu/jope-data/h452-flywheel/surrogate_v2/qhat_v2_all.pt"),
        "d93e1e76f2367cf1287ec449330274daf914c9b249989f6219b9c2483901eb15",
    ),
    "in-corpus-v4": (
        Path("/home/ubuntu/jope-data/h458-flywheel/surrogate_v4/qhat_v4_heldout.pt"),
        "73c1594ec81196288ac959bc8dcd318a7ee44bb08eac283c5ce5315d23d46525",
    ),
}


class UnifiedQHatPrior:
    """Supports pinned `dev91-free` (`qhat_v2.pt`) as default plus `in-corpus-v2` / `in-corpus-v4`."""

    def __init__(self, prior_mode: str = "dev91-free", ckpt_override: Optional[Path] = None) -> None:
        if ckpt_override is not None:
            ckpt_p = ckpt_override
            got_sha = _sha256_file(ckpt_p)
            matched_mode = None
            for pm, (_pp, psha) in PINNED_PRIORS.items():
                if got_sha == psha:
                    matched_mode = pm
                    break
            self.prior_mode = matched_mode or "custom"
            self.ckpt_path = ckpt_p
            self.ckpt_sha256 = got_sha
        else:
            if prior_mode not in PINNED_PRIORS:
                raise ValueError(f"Unknown --prior {prior_mode!r}; expected one of {sorted(PINNED_PRIORS)}")
            ckpt_p, exp_sha = PINNED_PRIORS[prior_mode]
            got_sha = _sha256_file(ckpt_p)
            if got_sha != exp_sha:
                raise RuntimeError(f"Prior SHA mismatch for {prior_mode} ({ckpt_p}): got {got_sha}, expected {exp_sha}")
            self.prior_mode = prior_mode
            self.ckpt_path = ckpt_p
            self.ckpt_sha256 = got_sha

        self._v4: Optional[PretrainedQHatV4Prior] = None
        self._v2: Optional[PretrainedQHatPrior] = None
        self.ok = False
        self.d_ast = 0
        self.net = None

        v4_try = PretrainedQHatV4Prior(self.ckpt_path, require_ok=False)
        if getattr(v4_try, "ok", False):
            self._v4 = v4_try
            self.ok = True
            self.d_ast = int(v4_try.d_ast)
            self.net = v4_try.net
            return

        v2_try = PretrainedQHatPrior(self.ckpt_path)
        if getattr(v2_try, "ok", False):
            self._v2 = v2_try
            self.ok = True
            self.d_ast = int(v2_try.d_ast)
            self.net = v2_try.net
            return

        raise RuntimeError(f"UnifiedQHatPrior failed to load prior checkpoint {self.ckpt_path}")

    def encode_ctx_z(self, enc_dict: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
        if self._v4 is not None:
            return self._v4.encode_ctx_z(enc_dict)
        assert self._v2 is not None
        z = self._v2.encode_ctx_z(enc_dict)
        return z if z is not None else np.zeros((1, 44), dtype=np.float32)

    def predict_components(self, ctx_z: np.ndarray, dsls: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = len(dsls)
        if m == 0:
            z = np.empty(0, dtype=np.float64)
            return z, z, z
        if self._v4 is not None:
            return self._v4.predict_components(ctx_z, dsls)
        assert self._v2 is not None
        X_ast = np.stack([featurize_candidate_h452(G.Candidate.from_dsl(d, "q2")) for d in dsls], axis=0)
        if self._v2.net is not None and X_ast.shape[1] == self._v2.d_ast:
            with torch.no_grad():
                tc = torch.from_numpy(np.repeat(ctx_z, m, axis=0))
                ta = torch.from_numpy(X_ast.astype(np.float32, copy=False))
                q_raw = self._v2.net(tc, ta).numpy().astype(np.float64)
        else:
            q_raw = np.zeros(m, dtype=np.float64)
        zeros = np.zeros(m, dtype=np.float64)
        return q_raw, zeros, zeros


def extract_qhat_ast_embedding(qhat: UnifiedQHatPrior, dsls: Sequence[str]) -> np.ndarray:
    """Extracts L2-normalized AST latent vectors (`160d` on v4, `101d` on v2)."""
    if not dsls:
        return np.zeros((0, 101), dtype=np.float64)
    if qhat.ok and qhat.net is not None and hasattr(qhat.net, "ast_proj") and qhat.d_ast == 128:
        X_ast = np.stack([featurize_candidate_h454(d) for d in dsls], axis=0).astype(np.float32, copy=False)
        with torch.no_grad():
            emb = qhat.net.ast_proj(torch.from_numpy(X_ast)).numpy().astype(np.float64)
    else:
        emb = np.stack(
            [featurize_candidate_h452(G.Candidate.from_dsl(d, "emb")) for d in dsls],
            axis=0,
        ).astype(np.float64)
    nrms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb / nrms


def compute_huber_scale(r_train: np.ndarray) -> Tuple[float, float]:
    """Fold-causal MAD Pseudo-Huber scale `(delta, z_clip)` computed strictly from training-fold residuals."""
    r = np.asarray(r_train, dtype=np.float64)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med))) * 1.4826
    sd = max(float(np.std(r)), 1e-12)
    delta = max(1.65 * mad, 0.25 * sd, 1e-9)
    # Cap restored `delta * sinh(z / delta)` strictly within `+-3.5 * sd` so heavy-tail extrapolation never explodes
    z_clip = float(delta * math.asinh(3.5 * sd / delta))
    return delta, z_clip


def warp_pseudo_huber(r: np.ndarray, delta: float) -> np.ndarray:
    return delta * np.arcsinh(np.asarray(r, dtype=np.float64) / max(delta, 1e-12))


def unwarp_pseudo_huber(z_pred: np.ndarray, delta: float, z_clip: float) -> np.ndarray:
    zc = np.clip(np.asarray(z_pred, dtype=np.float64), -z_clip, z_clip)
    return delta * np.sinh(zc / max(delta, 1e-12))


def enforce_stage_complexity_cap(g: G.Genome, max_transforms: int, stage_idx: int) -> G.Genome:
    """Anneals pipeline depth across stages (`4 -> 3 -> 2`) and strips unsigned target warps on centered residuals."""
    if isinstance(g, G.Portfolio):
        cap_members = 3 if stage_idx == 0 else 2
        ms = [enforce_stage_complexity_cap(m, max_transforms, stage_idx) for m in g.members[:cap_members]]
        return G.Portfolio(tuple(ms), g.folds, g.repeats_auto)
    tr = list(g.transforms)
    if stage_idx >= 1 and tr and tr[0].op in G.TARGET_OPS:
        # On zero-centered signed residuals (m >= 1), strip `log1p`/`rank_gauss` warps that distort additive zero
        kind = str(dict(tr[0].params).get("kind", ""))
        if kind in ("log1p", "rank_gauss"):
            tr = tr[1:]
    if len(tr) > max_transforms:
        tr = tr[:max_transforms]
    return G.Pipe(tuple(tr) + ((g.head,) if g.head is not None else ()))


class StageOrthogonalSurrogate:
    """Blends `UnifiedQHatPrior` + online bootstrap rank-ridge + explicit AST orthogonality to prior stages."""

    def __init__(
        self,
        seed: int,
        qhat: UnifiedQHatPrior,
        ctx_z: np.ndarray,
        prior_stage_embs: Sequence[np.ndarray],
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.qhat = qhat
        self.ctx_z = ctx_z
        self.prior_embs = np.stack(prior_stage_embs, axis=0) if prior_stage_embs else None
        self.X_obs: List[np.ndarray] = []
        self.u_obs: List[float] = []
        self.to_obs: List[float] = []
        self.W_bags: Optional[np.ndarray] = None
        self.w_to: Optional[np.ndarray] = None

    def observe(self, cand: G.Candidate, u_val: float, timed_out: bool = False) -> None:
        if math.isfinite(u_val):
            self.X_obs.append(featurize_candidate_h452(cand))
            self.u_obs.append(float(max(-3.0, min(6.0, u_val))))
            self.to_obs.append(1.0 if timed_out else 0.0)

    def refit(self, n_bags: int = 8, alpha: float = 4.5) -> None:
        n = len(self.u_obs)
        if n < 10:
            self.W_bags = None
            self.w_to = None
            return
        keep_n = min(n, 700)
        X = np.stack(self.X_obs[-keep_n:], axis=0)
        u_arr = np.asarray(self.u_obs[-keep_n:], dtype=np.float64)
        to_arr = np.asarray(self.to_obs[-keep_n:], dtype=np.float64)
        r_tgt = (rankdata(u_arr, method="average") - 0.5) / float(keep_n) - 0.5
        d = X.shape[1]
        X1 = np.column_stack([X, np.ones(keep_n, dtype=np.float64)])
        eye = alpha * np.eye(d + 1, dtype=np.float64)
        eye[-1, -1] = 1e-6
        ws = []
        for _ in range(n_bags):
            idx = self.rng.integers(0, keep_n, size=keep_n)
            try:
                wb = np.linalg.solve(X1[idx].T @ X1[idx] + eye, X1[idx].T @ r_tgt[idx])
            except np.linalg.LinAlgError:
                wb = np.zeros(d + 1, dtype=np.float64)
            ws.append(wb)
        self.W_bags = np.stack(ws, axis=0)
        if float(np.sum(to_arr)) >= 2.0:
            try:
                self.w_to = np.linalg.solve(X1.T @ X1 + 8.0 * eye, X1.T @ to_arr)
            except np.linalg.LinAlgError:
                self.w_to = None
        else:
            self.w_to = None

    def score_ucb(self, cands: Sequence[G.Candidate], beta: float = 0.28) -> np.ndarray:
        m = len(cands)
        if m == 0:
            return np.empty(0, dtype=np.float64)
        dsls = [c.dsl for c in cands]
        q_p, stab_p, gap_p = self.qhat.predict_components(self.ctx_z, dsls)
        raw_prior = q_p + 0.08 * stab_p - 0.22 * gap_p
        if m > 1 and float(np.std(raw_prior)) > 1e-12:
            prior_r = (rankdata(raw_prior, method="average") - 0.5) / float(m) - 0.5
        else:
            prior_r = np.zeros(m, dtype=np.float64)

        ortho_pen = np.zeros(m, dtype=np.float64)
        if self.prior_embs is not None and len(self.prior_embs) > 0:
            c_embs = extract_qhat_ast_embedding(self.qhat, dsls)
            max_cos = np.max(c_embs @ self.prior_embs.T, axis=1)
            ortho_pen = 0.06 * np.maximum(0.0, max_cos - 0.55)

        if self.W_bags is None:
            return 0.80 * prior_r - ortho_pen + self.rng.uniform(-0.06, 0.06, size=m)

        X = np.stack([featurize_candidate_h452(c) for c in cands], axis=0)
        X1 = np.column_stack([X, np.ones(m, dtype=np.float64)])
        preds = X1 @ self.W_bags.T
        mu = np.mean(preds, axis=1)
        sd = np.std(preds, axis=1)
        w_pr = max(0.18, 0.50 * math.exp(-len(self.u_obs) / 180.0))
        sc = (1.0 - w_pr) * mu + w_pr * prior_r + beta * sd - ortho_pen
        if self.w_to is not None:
            sc = sc - 0.45 * np.clip(X1 @ self.w_to, 0.0, 1.0)
        return sc


@dataclass
class StageCandidateRec:
    cand: G.Candidate
    geom: str  # "l2" or "huber"
    island: int
    q_screen: np.ndarray  # residual prediction on screen subset
    u_screen: float
    u_adj: float


def crossfit_residual_program_kfold(
    cand: G.Candidate,
    geom: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    r_target: np.ndarray,
    lab_k: np.ndarray,
    runner: BudgetedSlotRunner,
    include_full_refit: bool = False,
    timeout_s: float = 7.5,
    finalizing: bool = False,
    call_kind: str = "rot_oof",
) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Fold-causally cross-fits `cand` on residual `r_target` (`geom in {'l2', 'huber'}`), returning `(g_oof, g_bag, g_ref)`."""
    n_s, width = Xs.shape
    n_a = Xa.shape[0]
    X_joint = np.vstack([Xs, Xa])
    y_dummy_a = np.linspace(-1.0, 1.0, n_a, dtype=np.float64)
    ones_a = np.ones(n_a, dtype=np.uint8)

    c_fit = G.cap_gpr(cand, width) if "(gpr" in cand.dsl else cand
    uniq_f = sorted(int(x) for x in np.unique(lab_k))
    eff_k = len(uniq_f)
    if eff_k < 2:
        return None

    tasks: List[Tuple[bytes, G.Candidate]] = []
    va_masks: List[np.ndarray] = []
    fold_scales: List[Tuple[float, float]] = []

    for fk in uniq_f:
        va_m = lab_k == fk
        tr_m = ~va_m
        va_masks.append(va_m)
        if geom == "huber":
            d_f, zc_f = compute_huber_scale(r_target[tr_m])
            y_fit_f = warp_pseudo_huber(r_target, d_f)
            fold_scales.append((d_f, zc_f))
        else:
            y_fit_f = r_target
            fold_scales.append((1.0, 0.0))
        fu8 = np.concatenate([va_m.astype(np.uint8), ones_a])
        tasks.append((pack_jtt1_bytes(X_joint, np.concatenate([y_fit_f, y_dummy_a]), fu8), c_fit))

    gots_k = runner.map_s1_budgeted(tasks, timeout_s=timeout_s, finalizing=finalizing, is_candidate_eval=False, kind=call_kind)
    gots_pa: List[Tuple[Optional[Tuple[np.ndarray, np.ndarray]], bool]] = []
    full_scale = (1.0, 0.0)
    if include_full_refit:
        if geom == "huber":
            d_all, zc_all = compute_huber_scale(r_target)
            y_fit_all = warp_pseudo_huber(r_target, d_all)
            full_scale = (d_all, zc_all)
        else:
            y_fit_all = r_target
        fu8_pa = np.concatenate([np.zeros(n_s, dtype=np.uint8), ones_a])
        pa_task = [(pack_jtt1_bytes(X_joint, np.concatenate([y_fit_all, y_dummy_a]), fu8_pa), c_fit)]
        gots_pa = runner.map_s1_budgeted(
            pa_task, timeout_s=timeout_s, finalizing=finalizing, is_candidate_eval=False, kind="pa_refit"
        )
    gots = list(gots_k) + list(gots_pa)
    g_oof = np.empty(n_s, dtype=np.float64)
    g_bag = np.zeros(n_a, dtype=np.float64)

    for va_m, (d_f, zc_f), (got, _) in zip(va_masks, fold_scales, gots[:eff_k]):
        n_va = int(np.sum(va_m))
        if got is None or got[1] is None or len(got[1]) != (n_va + n_a):
            return None
        p_raw = np.asarray(got[1], dtype=np.float64)
        if not np.all(np.isfinite(p_raw)):
            return None
        if geom == "huber":
            p_raw = unwarp_pseudo_huber(p_raw, d_f, zc_f)
        tr_m = ~va_m
        r_tr = r_target[tr_m] if np.any(tr_m) else r_target
        sr_f = max(float(np.std(r_tr)), 1e-12)
        clip_lim_f = 4.0 * max(float(np.max(np.abs(r_tr))), sr_f)
        p_raw = np.clip(p_raw, -clip_lim_f, clip_lim_f)
        g_oof[va_m] = p_raw[:n_va]
        g_bag += p_raw[n_va:] / float(eff_k)

    g_ref: Optional[np.ndarray] = None
    if include_full_refit:
        sr_all = max(float(np.std(r_target)), 1e-12)
        clip_lim_all = 4.0 * max(float(np.max(np.abs(r_target))), sr_all)
        got_pa, _ = gots[eff_k]
        if got_pa is not None and got_pa[1] is not None and len(got_pa[1]) == n_a:
            pa_raw = np.asarray(got_pa[1], dtype=np.float64)
            if np.all(np.isfinite(pa_raw)):
                if geom == "huber":
                    pa_raw = unwarp_pseudo_huber(pa_raw, full_scale[0], full_scale[1])
                g_ref = np.clip(pa_raw, -clip_lim_all, clip_lim_all)
        if g_ref is None:
            g_ref = g_bag.copy()

    return g_oof, g_bag, g_ref


def pooled_fold_blend_utility(pred: np.ndarray, y: np.ndarray, lab: np.ndarray) -> float:
    """Blends 50% pooled log-RMSE utility + 50% per-fold log-RMSE utility on SEARCH rows.

    Prevents single-row heavy-tail blowups (Jensen gap between per-fold geometric RMSE and pooled board RMSE,
    cf. `cl_h458` 02:10Z) while preserving fold-consistency sensitivity on homoscedastic contexts.
    """
    sy = max(float(np.std(y)), 1e-12)
    if not np.all(np.isfinite(pred)):
        return -5.0
    err = pred - y
    rm_pool = max(float(np.sqrt(np.mean(err * err))), 1e-12 * sy)
    u_pool = -math.log(rm_pool / sy)
    u_fold = float(fold_log_utility(pred, y, lab))
    return 0.5 * u_pool + 0.5 * u_fold


def clip_oof_by_train_folds(raw_oof: np.ndarray, ys: np.ndarray, lab: np.ndarray, pad_mult: float = 1.5) -> np.ndarray:
    """Clips each outer OOF fold strictly using training-fold (`lab != fk`) target bounds (`cl_fwriter` 02:20Z)."""
    out = np.empty_like(raw_oof, dtype=np.float64)
    for fk in np.unique(lab):
        va_m = lab == fk
        tr_m = ~va_m
        y_tr = ys[tr_m] if np.any(tr_m) else ys
        lo_f, hi_f = float(np.min(y_tr)), float(np.max(y_tr))
        sy_f = max(float(np.std(y_tr)), 1e-12)
        pad_f = pad_mult * max(hi_f - lo_f, sy_f)
        out[va_m] = np.clip(raw_oof[va_m], lo_f - pad_f, hi_f + pad_f)
    return out


def solve_vault_coordinate_backfit(
    ys: np.ndarray,
    f0_oof: np.ndarray,
    G_vault: np.ndarray,  # shape [n_s, m]
    lab_vault: np.ndarray,
    reg_alpha: float = 0.0025,
) -> Tuple[np.ndarray, float]:
    """Scale-invariant non-negative coordinate backfit `w in [0, 1]^m` on unseen `lab_vault` (`< 2 ms`, `0` worker calls)."""
    m = G_vault.shape[1]
    if m == 0:
        return np.empty(0, dtype=np.float64), float(pooled_fold_blend_utility(f0_oof, ys, lab_vault))

    var_y = max(float(np.var(ys)), 1e-18)
    var_g = np.maximum(np.var(G_vault, axis=0), 1e-18)
    # Age-escalating scale-invariant damping: later residual stages shrink harder toward 0 when SNR drops
    pen_coef = np.array(
        [reg_alpha * (1.0 + 0.14 * j) * (var_g[j] / var_y) for j in range(m)],
        dtype=np.float64,
    )

    w = np.zeros(m, dtype=np.float64)
    cur_pred = f0_oof.copy()

    def _obj(pred: np.ndarray, w_vec: np.ndarray) -> float:
        return float(pooled_fold_blend_utility(pred, ys, lab_vault)) - float(np.sum(pen_coef * (w_vec * w_vec)))

    # Warm-start stagewise greedy step, then run 18 Gauss-Seidel coordinate backfitting sweeps over all `1..m`
    coarse_grid = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    best_val = _obj(cur_pred, w)

    for sweep in range(18):
        changed = False
        step_rad = 0.10 if sweep < 6 else (0.025 if sweep < 12 else 0.006)
        for j in range(m):
            gj = G_vault[:, j]
            base_no_j = cur_pred - w[j] * gj
            if sweep == 0:
                cand_ws = coarse_grid
            else:
                lo_w = max(0.0, w[j] - step_rad)
                hi_w = min(1.0, w[j] + step_rad)
                cand_ws = np.unique(np.concatenate([[0.0, w[j]], np.linspace(lo_w, hi_w, 11, dtype=np.float64)]))
            w_try = w.copy()
            best_wj = w[j]
            for wt in cand_ws:
                w_try[j] = float(wt)
                v = _obj(base_no_j + wt * gj, w_try)
                if v > best_val + 1e-10:
                    best_val = v
                    best_wj = float(wt)
            if abs(best_wj - w[j]) > 1e-9:
                cur_pred = base_no_j + best_wj * gj
                w[j] = best_wj
                changed = True
        if sweep >= 2 and not changed:
            break

    u_raw = float(pooled_fold_blend_utility(cur_pred, ys, lab_vault))
    return w, u_raw


def _propose_stage_offspring(
    arch_by_isl: Dict[int, List[StageCandidateRec]],
    tb: D.TargetBank,
    width: int,
    gen_idx: int,
    n_gens: int,
    stage_idx: int,
    max_transforms: int,
    seen_shas: set[Tuple[str, str]],
    rng: random.Random,
    n_want: int = 48,
) -> List[Tuple[G.Candidate, str, int]]:
    """Generates `n_want` multi-geometry stage proposals (`(cand, geom, island5)`)."""
    out: List[Tuple[G.Candidate, str, int]] = []
    local_seen = set(seen_shas)
    all_recs = [r for recs in arch_by_isl.values() for r in recs]
    if not all_recs:
        return out

    in_exploit = gen_idx >= max(2, int(0.75 * n_gens))

    def _reg(g_raw: Optional[G.Genome], op_tag: str, pars: Sequence[StageCandidateRec], isl: int) -> bool:
        if g_raw is None:
            return False
        g_cap = enforce_stage_complexity_cap(g_raw, max_transforms=max_transforms, stage_idx=stage_idx)
        lin = D.merge_lineage(*(p.cand for p in pars)) if pars else ()
        kw = {"bank_cids": lin} if lin else {}
        c_new = G.safe_candidate(g_cap, f"s{stage_idx}_g{gen_idx}_{op_tag}", [p.cand.sha for p in pars], width, **kw)
        if c_new is None:
            return False
        if c_new.sha in tb.excluded_shas or any(x in tb.excluded_cids for x in D._lineage(c_new)):
            return False
        geom = "huber" if isl == 2 else "l2"
        key = (c_new.sha, geom)
        if key in local_seen:
            return False
        local_seen.add(key)
        out.append((c_new, geom, isl))
        return True

    if in_exploit:
        for isl_k, recs in arch_by_isl.items():
            if not recs:
                continue
            try:
                nbrs = G.line_search_neighbors(recs[0].cand.genome())
                rng.shuffle(nbrs)
                for ng in nbrs[:2]:
                    _reg(ng, "ls", [recs[0]], isl_k)
            except Exception:
                pass

    tries = 0
    while len(out) < n_want and tries < n_want * 16:
        tries += 1
        isl_t = tries % 5
        pool = arch_by_isl.get(isl_t) or all_recs
        p1 = rng.choice(pool[: max(1, min(6, len(pool)))])
        u = rng.random()
        try:
            g1 = p1.cand.genome()
            if stage_idx == 0 and u < 0.10:
                _reg(op_target_warp_h453(g1, rng), "twarp", [p1], isl_t)
            elif isl_t == 3 and u < 0.28:
                # Kernel / local smoother specialist island
                _reg(G.op_insert_stage(g1, rng, group=rng.choice(["kernel_map", "nonlinear", "scale"])), "kern_ins", [p1], 3)
            elif isl_t == 4 and u < 0.30:
                # Subspace / projection / oblique interaction specialist island
                _reg(G.op_insert_stage(g1, rng, group=rng.choice(["projection", "interaction"])), "sub_ins", [p1], 4)
            elif u < 0.32:
                _reg(G.op_parameter(g1, rng), "param", [p1], isl_t)
            elif u < 0.56:
                grp = rng.choice(["interaction", "nonlinear", "scale", "kernel_map", "projection", None])
                _reg(G.op_insert_stage(g1, rng, group=grp), "ins", [p1], isl_t)
            elif u < 0.68:
                _reg(G.op_delete_stage(g1, rng), "del", [p1], isl_t)
            elif u < 0.84 and len(all_recs) >= 2:
                p2 = rng.choice(all_recs)
                _reg(G.op_crossover(g1, p2.cand.genome(), rng), "xover", [p1, p2], isl_t)
            elif u < 0.94:
                _reg(G.op_replace_head(g1, rng, island=min(isl_t, 3)), "hswap", [p1], isl_t)
            else:
                _reg(G.op_synth(min(isl_t, 3), rng), "synth", [], isl_t)
        except Exception:
            continue

    return out[:n_want]


def run_oc_fgb_gp_context(
    cid: str,
    family_id: str,
    Xs: np.ndarray,
    Xa: np.ndarray,
    ys: np.ndarray,
    schedule: EffortSchedule,
    slot_dir: Path,
    encoder: BaggedNeuralEncoderH452,
    qhat: UnifiedQHatPrior,
    anchor_tup: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    single_stage_control: bool = False,
    budget_mult: float = 1.0,
    export_committee_m: int = 0,
    enforce_wall_cap: bool = False,
    seed: int = 45901,
    inner_slots: int = 3,
    rayon_threads: int = 1,
    screen_max_rows: int = 560,
    timeout_cv2: float = 1.6,
    timeout_crossfit: float = 7.5,
    ratchet_margin: float = 0.0015,
) -> Dict:
    """Executes `Arm A5` (`OC-FGB-GP`, `M = schedule.n_stages`) or `A0_iso` (`M = 1`, `budget_mult in {1.0, 1.5}`)."""
    assert qhat.ok, "UnifiedQHatPrior must be loaded!"
    n_s, width = Xs.shape
    n_a = Xa.shape[0]
    ym = float(np.mean(ys))
    sy = max(float(np.std(ys)), 1e-12)
    y_lo, y_hi = float(np.min(ys)), float(np.max(ys))
    pad_y = 1.5 * max(y_hi - y_lo, sy)

    k_eff = min(schedule.k_oof, max(2, n_s // 20))
    k_tc_eff = min(schedule.k_tier_c, max(2, n_s // 20))
    lab_vault = make_stratified_kfold(cid, ys, k=k_eff, rep=77)
    lab_tc0 = make_stratified_kfold(cid, ys, k=k_tc_eff, rep=101)
    lab_tc1 = make_stratified_kfold(cid, ys, k=k_tc_eff, rep=102)

    # Per-stage export call cost so search inside Stage m leaves exact room for cross-fits
    if schedule.separate_tier_c:
        per_stage_cf_calls = 2 * k_eff + k_tc_eff * schedule.tier_c_repeats + 1
    else:
        per_stage_cf_calls = k_eff + 1

    if single_stage_control:
        n_stages_eff = 1
        cf_reserve = per_stage_cf_calls
        if schedule.budget_unit == "candidates":
            eff_max_cands = max(8, int(round(schedule.max_candidates * budget_mult)))
            eff_max_calls = eff_max_cands + cf_reserve + 12
        else:
            eff_max_calls = max(16, int(round(schedule.max_calls * budget_mult)))
            eff_max_cands = max(8, eff_max_calls - cf_reserve)
        product_wall_s = schedule.wall_seconds * budget_mult
        eff_wall = product_wall_s if enforce_wall_cap else 86400.0
        g0_pop = min(36, max(10, eff_max_cands // 5))
        epg = schedule.eval_per_gen if schedule.effort != "medium" else 2
        stage_gens = (max(4, (eff_max_cands - g0_pop + epg - 1) // max(1, epg)),)
        stage_cand_quotas = (eff_max_cands,)
    else:
        n_stages_eff = schedule.n_stages
        eff_max_cands = schedule.max_candidates
        eff_max_calls = schedule.max_calls
        product_wall_s = schedule.wall_seconds
        eff_wall = product_wall_s if enforce_wall_cap else 86400.0
        g0_pop = schedule.stage_gen0_pop
        epg = schedule.eval_per_gen
        stage_gens = schedule.stage_gens
        stage_cand_quotas = schedule.stage_cand_quotas

    meter = WorkerCallBudgetMeter(max_candidates=eff_max_cands, max_calls=eff_max_calls, wall_seconds=eff_wall)
    runner = BudgetedSlotRunner(slot_dir, meter=meter, n_slots=inner_slots, rayon_threads=rayon_threads)

    if anchor_tup is not None:
        f0_oof = np.asarray(anchor_tup[0], dtype=np.float64).copy()
        f0_bag = np.asarray(anchor_tup[1], dtype=np.float64).copy()
        f0_ref = np.asarray(anchor_tup[2], dtype=np.float64).copy()
        anchor_mode = "hybrid_neural"
    else:
        f0_oof = np.full(n_s, ym, dtype=np.float64)
        f0_bag = np.full(n_a, ym, dtype=np.float64)
        f0_ref = np.full(n_a, ym, dtype=np.float64)
        anchor_mode = "pure_gp"

    r_cur = ys - f0_oof
    prior_embs: List[np.ndarray] = []
    prior_oof_units: List[np.ndarray] = []
    stage_winners: List[Dict] = []
    final_stage0_recs: List[StageCandidateRec] = []
    total_screen_evaluated = 0

    G_rot_list: List[np.ndarray] = []
    G_vault_list: List[np.ndarray] = []
    B_vault_list: List[np.ndarray] = []
    R_vault_list: List[np.ndarray] = []

    u_vault_f0 = float(pooled_fold_blend_utility(f0_oof, ys, lab_vault))
    ratchet_best_u_adj = u_vault_f0
    ratchet_w = np.zeros(0, dtype=np.float64)
    shrunk_w = np.zeros(0, dtype=np.float64)
    low_weight_streak = 0

    try:
        for stg_m in range(n_stages_eff):
            if meter.wall_expired(reserve_s=1.5):
                break
            # Minimum calls needed to evaluate at least 4 candidates + cross-fit Stage m
            if meter.remaining() < (4 + per_stage_cf_calls):
                break
            if float(np.std(r_cur)) < 1e-6 * sy:
                break

            max_tr = 4 if (single_stage_control or stg_m <= 1) else (3 if stg_m <= 4 else 2)

            # Re-encode `(Xs, r_cur)` via `DOPE` + `UnifiedQHatPrior` so priors adapt to current residual geometry!
            enc_view = encoder.encode_bag(Xs, r_cur, use_full_dope=(stg_m == 0 or schedule.effort != "medium"))
            ctx_z = qhat.encode_ctx_z(enc_view)

            # Fixed partition seed `45901` (`D5`) so `a0_s0` & `a0_s1` share identical screen rows + fold partitions!
            if n_s > screen_max_rows:
                scr_idx = stratified_subsample_80pct(
                    r_cur,
                    float(screen_max_rows) / float(n_s),
                    np.random.default_rng(_stream_seed("s9_scr", cid, 45901, stg_m)),
                )
            else:
                scr_idx = np.arange(n_s, dtype=np.int64)
            Xs_scr = Xs[scr_idx]
            r_scr = r_cur[scr_idx]
            lab2_scr = make_stratified_kfold(cid, r_scr, k=2, rep=300 + stg_m)
            m0_scr = lab2_scr == 0
            m1_scr = ~m0_scr
            n0_scr, n1_scr = int(m0_scr.sum()), int(m1_scr.sum())

            env_l2_cv2 = pack_jtt1_bytes(Xs_scr, r_scr, lab2_scr.astype(np.uint8))
            d_scr0, zc_scr0 = compute_huber_scale(r_scr[m1_scr])
            d_scr1, zc_scr1 = compute_huber_scale(r_scr[m0_scr])
            d_scr_all, zc_scr_all = compute_huber_scale(r_scr)
            env_hub_cv2 = pack_jtt1_bytes(Xs_scr, warp_pseudo_huber(r_scr, d_scr_all), lab2_scr.astype(np.uint8))

            prior_scr_units = [u_vec[scr_idx] / max(float(np.linalg.norm(u_vec[scr_idx])), 1e-12) for u_vec in prior_oof_units]

            surrogate = StageOrthogonalSurrogate(
                seed=_stream_seed("s9_surr", cid, seed, stg_m),
                qhat=qhat,
                ctx_z=ctx_z,
                prior_stage_embs=prior_embs,
            )

            # Seed population strictly from `(Xs, r_cur)` (`same_family_bank_hits == 0`)
            raw_gen0, tb, gen0_info = build_bagged_gen0_population(
                cid=cid,
                family_id=family_id,
                Xs=Xs,
                ys=r_cur,
                n_pop=max(16, g0_pop * 2),
                encoder=encoder,
                seed=_stream_seed("s9_g0", cid, seed, stg_m),
                has_104m_resid=False,
                anchor_only_ablation=False,
            )
            assert gen0_info["same_family_bank_hits"] == 0, "Same-family bank leak forbidden!"

            capped_gen0: List[Tuple[G.Candidate, str, int]] = []
            seen_stage: set[Tuple[str, str]] = set()
            for idx_c, c0 in enumerate(raw_gen0):
                try:
                    gc = enforce_stage_complexity_cap(c0.genome(), max_transforms=max_tr, stage_idx=stg_m)
                    cc = G.safe_candidate(gc, f"s{stg_m}_seed", list(c0.parents), width)
                except Exception:
                    cc = None
                if cc is None:
                    continue
                isl_c = idx_c % 5
                geom_c = "huber" if (isl_c == 2 and not single_stage_control) else "l2"
                key_c = (cc.sha, geom_c)
                if key_c not in seen_stage:
                    seen_stage.add(key_c)
                    capped_gen0.append((cc, geom_c, isl_c))

            # Rank Gen-0 seeds with `QHatV4` + AST orthogonality so small `g0_pop` evaluates the highest-prior seeds
            if len(capped_gen0) > g0_pop:
                sc0 = surrogate.score_ucb([x[0] for x in capped_gen0], beta=0.15)
                top0 = np.argsort(-sc0)[:g0_pop]
                capped_gen0 = [capped_gen0[int(i)] for i in top0]

            stage_recs: List[StageCandidateRec] = []
            stage_cand_cap = stage_cand_quotas[min(stg_m, len(stage_cand_quotas) - 1)]
            stage_cands_used = 0

            def _eval_screen_batch(batch_items: Sequence[Tuple[G.Candidate, str, int]]) -> None:
                nonlocal stage_cands_used
                allow_calls = max(0, meter.remaining() - per_stage_cf_calls)
                allow_cands = min(meter.remaining_candidates(), max(0, stage_cand_cap - stage_cands_used))
                allow = min(allow_calls, allow_cands)
                if allow <= 0 or not batch_items:
                    return
                sub_items = list(batch_items[:allow])
                tasks_b = [
                    ((env_hub_cv2 if gm == "huber" else env_l2_cv2), c_it)
                    for c_it, gm, _ in sub_items
                ]
                gots_b = runner.map_s1_budgeted(
                    tasks_b, timeout_s=timeout_cv2, is_candidate_eval=True, kind="search_screen"
                )
                stage_cands_used += len(sub_items)
                r_tr0 = r_scr[m1_scr] if n1_scr > 0 else r_scr
                r_tr1 = r_scr[m0_scr] if n0_scr > 0 else r_scr
                clip_s0 = 4.0 * max(float(np.max(np.abs(r_tr0))), max(float(np.std(r_tr0)), 1e-12))
                clip_s1 = 4.0 * max(float(np.max(np.abs(r_tr1))), max(float(np.std(r_tr1)), 1e-12))

                for (c_it, gm, isl_it), (got, t_out) in zip(sub_items, gots_b):
                    if got is None:
                        surrogate.observe(c_it, -2.5, timed_out=t_out)
                        continue
                    p0, p1 = got
                    if len(p0) != n0_scr or len(p1) != n1_scr:
                        continue
                    q_s = np.empty(len(scr_idx), dtype=np.float64)
                    if gm == "huber":
                        q_s[m0_scr] = np.clip(unwarp_pseudo_huber(p0, d_scr0, zc_scr0), -clip_s0, clip_s0)
                        q_s[m1_scr] = np.clip(unwarp_pseudo_huber(p1, d_scr1, zc_scr1), -clip_s1, clip_s1)
                    else:
                        q_s[m0_scr] = np.clip(p0, -clip_s0, clip_s0)
                        q_s[m1_scr] = np.clip(p1, -clip_s1, clip_s1)
                    u_s = float(pooled_fold_blend_utility(f0_oof[scr_idx] + (ys[scr_idx] - f0_oof[scr_idx] - r_scr) + q_s, ys[scr_idx], lab2_scr))
                    # Penalize prediction collinearity & QHatV4 overfit_gap during stage selection
                    _, _, gp_pred = qhat.predict_components(ctx_z, [c_it.dsl])
                    corr_pen = 0.0
                    if prior_scr_units:
                        qn = q_s / max(float(np.linalg.norm(q_s)), 1e-12)
                        max_rho = max(abs(float(np.dot(qn, pu))) for pu in prior_scr_units)
                        corr_pen = 0.05 * max(0.0, max_rho - 0.50)
                    u_adj = u_s - 0.12 * float(gp_pred[0]) - corr_pen
                    surrogate.observe(c_it, u_adj, timed_out=False)
                    stage_recs.append(
                        StageCandidateRec(
                            cand=c_it,
                            geom=gm,
                            island=isl_it,
                            q_screen=q_s,
                            u_screen=u_s,
                            u_adj=u_adj,
                        )
                    )

            _eval_screen_batch(capped_gen0)
            surrogate.refit()

            rng_stg = random.Random(_stream_seed("s9_evo", cid, seed, stg_m))
            n_gens_m = stage_gens[min(stg_m, len(stage_gens) - 1)]

            for gen_i in range(1, n_gens_m + 1):
                if meter.wall_expired(reserve_s=1.5):
                    break
                if stage_cands_used >= stage_cand_cap or meter.remaining_candidates() <= 0:
                    break
                if meter.remaining() <= per_stage_cf_calls:
                    break
                by_isl: Dict[int, List[StageCandidateRec]] = {i: [] for i in range(5)}
                for r in stage_recs:
                    by_isl[r.island].append(r)
                for i in range(5):
                    by_isl[i].sort(key=lambda r: -r.u_adj)
                    by_isl[i] = by_isl[i][:10]

                props = _propose_stage_offspring(
                    arch_by_isl=by_isl,
                    tb=tb,
                    width=width,
                    gen_idx=gen_i,
                    n_gens=n_gens_m,
                    stage_idx=stg_m,
                    max_transforms=max_tr,
                    seen_shas=seen_stage,
                    rng=rng_stg,
                    n_want=48,
                )
                if not props:
                    continue
                beta_g = 0.30 if gen_i <= int(0.75 * n_gens_m) else 0.12
                ucb = surrogate.score_ucb([p[0] for p in props], beta=beta_g)
                pick = np.argsort(-ucb)[: min(epg, len(props))]
                _eval_screen_batch([props[int(pi)] for pi in pick])
                if gen_i % 2 == 0:
                    surrogate.refit()

            if not stage_recs:
                break

            total_screen_evaluated += len(stage_recs)
            stage_recs.sort(key=lambda r: -r.u_adj)
            if stg_m == 0:
                final_stage0_recs = list(stage_recs)
            winner_rec = None
            cf_rot = None
            cf_vlt = None

            # Try top-2 candidates in case top-1 times out on full-row K-fold cross-fit
            for cand_try in stage_recs[:2]:
                if schedule.separate_tier_c:
                    lab_rot = make_stratified_kfold(cid, ys, k=k_eff, rep=200 + stg_m)
                    got_rot = crossfit_residual_program_kfold(
                        cand_try.cand,
                        cand_try.geom,
                        Xs,
                        Xa,
                        r_cur,
                        lab_rot,
                        runner,
                        include_full_refit=False,
                        timeout_s=timeout_crossfit,
                        finalizing=True,
                        call_kind="rot_oof",
                    )
                    if got_rot is None:
                        continue
                    got_vlt = crossfit_residual_program_kfold(
                        cand_try.cand,
                        cand_try.geom,
                        Xs,
                        Xa,
                        r_cur,
                        lab_vault,
                        runner,
                        include_full_refit=False,
                        timeout_s=timeout_crossfit,
                        finalizing=True,
                        call_kind="vault_rep77",
                    )
                    if got_vlt is None:
                        continue
                    winner_rec = cand_try
                    cf_rot = got_rot
                    cf_vlt = got_vlt
                    break
                else:
                    # Product `medium` (`worker_calls`): rotating `rep = 77 + stg_m` serves as inner vault + OOF handoff + PA
                    lab_med = make_stratified_kfold(cid, ys, k=k_eff, rep=77 + stg_m)
                    got_med = crossfit_residual_program_kfold(
                        cand_try.cand,
                        cand_try.geom,
                        Xs,
                        Xa,
                        r_cur,
                        lab_med,
                        runner,
                        include_full_refit=True,
                        timeout_s=timeout_crossfit,
                        finalizing=True,
                        call_kind="vault_rep77",
                    )
                    if got_med is None:
                        continue
                    winner_rec = cand_try
                    cf_rot = got_med
                    cf_vlt = got_med
                    break

            if winner_rec is None or cf_rot is None or cf_vlt is None:
                break

            g_rot_oof, _, _ = cf_rot
            g_vlt_oof, g_vlt_bag, g_vlt_ref = cf_vlt

            G_rot_list.append(g_rot_oof)
            G_vault_list.append(g_vlt_oof)
            B_vault_list.append(g_vlt_bag)
            R_vault_list.append(g_vlt_ref if g_vlt_ref is not None else g_vlt_bag)

            # Joint coordinate backfitting across all stages `1..stg_m+1` on unseen `lab_vault`
            G_v_mat = np.column_stack(G_vault_list)
            w_joint, u_vlt_joint = solve_vault_coordinate_backfit(
                ys=ys,
                f0_oof=f0_oof,
                G_vault=G_v_mat,
                lab_vault=lab_vault,
                reg_alpha=0.0025 if not single_stage_control else 0.0010,
            )
            shrunk_w = w_joint.copy()

            _, _, gap_w = qhat.predict_components(ctx_z, [winner_rec.cand.dsl])
            spread_vlt = compute_fold_utility_spread(f0_oof + G_v_mat @ w_joint, ys, lab_vault)
            u_vlt_adj = u_vlt_joint - 0.12 * float(gap_w[0]) * float(w_joint[-1]) - 0.05 * spread_vlt

            ratcheted_here = False
            if u_vlt_adj >= ratchet_best_u_adj + ratchet_margin:
                ratchet_best_u_adj = u_vlt_adj
                ratchet_w = w_joint.copy()
                ratcheted_here = True

            # Update rotating OOF residual `r_cur` for Stage `stg_m + 1` using backfitted weights `w_joint`!
            G_r_mat = np.column_stack(G_rot_list)
            r_cur = ys - (f0_oof + G_r_mat @ w_joint)

            prior_embs.append(extract_qhat_ast_embedding(qhat, [winner_rec.cand.dsl])[0])
            prior_oof_units.append(g_rot_oof)

            stage_winners.append(
                {
                    "stage": stg_m + 1,
                    "dsl": winner_rec.cand.dsl,
                    "geom": winner_rec.geom,
                    "island": winner_rec.island,
                    "u_screen": round(winner_rec.u_screen, 5),
                    "u_vault_joint": round(u_vlt_joint, 5),
                    "u_vault_adj": round(u_vlt_adj, 5),
                    "w_stage_last": round(float(w_joint[-1]), 5),
                    "w_joint": [round(float(x), 5) for x in w_joint.tolist()],
                    "ratcheted": ratcheted_here,
                    "calls_so_far": meter.admitted(),
                }
            )

            if float(w_joint[-1]) < 0.02:
                low_weight_streak += 1
                if low_weight_streak >= 2:
                    break
            else:
                low_weight_streak = 0

        # --- Final Untouched Tier-C (`rep = 101, 102`) + `PA` Export ---
        m_done = len(stage_winners)
        if m_done == 0:
            cols_out = {
                "shrunk": (f0_oof, f0_bag, f0_ref),
                "ratchet": (f0_oof, f0_bag, f0_ref),
            }
            stage_cols_out: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        else:
            if schedule.separate_tier_c:
                G_tc_list: List[np.ndarray] = []
                B_tc_list: List[np.ndarray] = []
                R_tc_list: List[np.ndarray] = []
                # Replay exact residual sequence causally on untouched `rep=101` (and `rep=102` for `ultra`)!
                r_tc0 = ys - f0_oof
                r_tc1 = ys - f0_oof
                for j, sw in enumerate(stage_winners):
                    c_j = G.Candidate.from_dsl(sw["dsl"], f"tc_s{j+1}")
                    gm_j = sw["geom"]
                    got0 = crossfit_residual_program_kfold(
                        c_j,
                        gm_j,
                        Xs,
                        Xa,
                        r_tc0,
                        lab_tc0,
                        runner,
                        include_full_refit=True,
                        timeout_s=timeout_crossfit,
                        finalizing=True,
                        call_kind="tier_c",
                    )
                    if got0 is None:
                        g_tc = G_vault_list[j]
                        b_tc = B_vault_list[j]
                        rf_tc = R_vault_list[j]
                    else:
                        g0, b0, rf0 = got0
                        if schedule.tier_c_repeats >= 2:
                            got1 = crossfit_residual_program_kfold(
                                c_j,
                                gm_j,
                                Xs,
                                Xa,
                                r_tc1,
                                lab_tc1,
                                runner,
                                include_full_refit=False,
                                timeout_s=timeout_crossfit,
                                finalizing=True,
                                call_kind="tier_c",
                            )
                            if got1 is not None:
                                g1, b1, _ = got1
                                r_tc1 = r_tc1 - shrunk_w[j] * g1
                                g_tc = 0.5 * (g0 + g1)
                                b_tc = 0.5 * (b0 + b1)
                                rf_tc = rf0 if rf0 is not None else b_tc
                            else:
                                g_tc, b_tc, rf_tc = g0, b0, (rf0 if rf0 is not None else b0)
                        else:
                            g_tc, b_tc, rf_tc = g0, b0, (rf0 if rf0 is not None else b0)
                        r_tc0 = r_tc0 - shrunk_w[j] * g0
                    G_tc_list.append(g_tc)
                    B_tc_list.append(b_tc)
                    R_tc_list.append(rf_tc)

                G_exp = np.column_stack(G_tc_list)
                B_exp = np.column_stack(B_tc_list)
                R_exp = np.column_stack(R_tc_list)
            else:
                G_exp = np.column_stack(G_vault_list)
                B_exp = np.column_stack(B_vault_list)
                R_exp = np.column_stack(R_vault_list)

            lab_out = lab_tc0 if schedule.separate_tier_c else lab_vault

            def _compose(w_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                k_w = len(w_vec)
                if k_w == 0:
                    return f0_oof.copy(), f0_bag.copy(), f0_ref.copy()
                q_c = clip_oof_by_train_folds(f0_oof + G_exp[:, :k_w] @ w_vec, ys, lab_out, pad_mult=1.5)
                b_c = np.clip(f0_bag + B_exp[:, :k_w] @ w_vec, y_lo - pad_y, y_hi + pad_y)
                r_c = np.clip(f0_ref + R_exp[:, :k_w] @ w_vec, y_lo - pad_y, y_hi + pad_y)
                return q_c, b_c, r_c

            cols_out = {
                "shrunk": _compose(shrunk_w),
                "ratchet": _compose(ratchet_w),
            }
            stage_cols_out = {}
            for k_s in range(1, m_done + 1):
                w_k = stage_winners[k_s - 1]["w_joint"]
                stage_cols_out[k_s] = _compose(np.asarray(w_k, dtype=np.float64))

        u_shrunk_oof = float(pooled_fold_blend_utility(cols_out["shrunk"][0], ys, lab_tc0))
        u_ratchet_oof = float(pooled_fold_blend_utility(cols_out["ratchet"][0], ys, lab_tc0))

        # Optional post-hoc `A6` committee-without-residuals (`cl_mtc` 01:32Z):
        # Takes up to `export_committee_m` diverse programs from `a0_s0`'s single-stage archive (0 residual stages,
        # 0 extra search candidates) and runs the EXACT same Vault Backfit + Tier-C OOF export on a separate post-hoc
        # runner so `a0_s0`'s own `meter` and `shrunk` prediction remain 100% untouched.
        a6_comm_weights: List[float] = []
        a6_comm_calls = 0
        if single_stage_control and export_committee_m > 0:
            if m_done == 0 or not final_stage0_recs:
                cols_out["committee"] = cols_out["shrunk"]
            else:
                picked_recs: List[StageCandidateRec] = []
                picked_units: List[np.ndarray] = []
                for rc in final_stage0_recs:
                    if len(picked_recs) >= export_committee_m:
                        break
                    qn = rc.q_screen / max(float(np.linalg.norm(rc.q_screen)), 1e-12)
                    if picked_units and max(abs(float(np.dot(qn, pu))) for pu in picked_units) > 0.92:
                        continue
                    picked_recs.append(rc)
                    picked_units.append(qn)
                if len(picked_recs) < min(export_committee_m, len(final_stage0_recs)):
                    seen_dsl = {r.cand.dsl for r in picked_recs}
                    for rc in final_stage0_recs:
                        if len(picked_recs) >= export_committee_m:
                            break
                        if rc.cand.dsl not in seen_dsl:
                            picked_recs.append(rc)
                            seen_dsl.add(rc.cand.dsl)

                comm_meter = WorkerCallBudgetMeter(
                    max_candidates=1, max_calls=max(64, export_committee_m * 36), wall_seconds=max(60.0, eff_wall)
                )
                comm_runner = BudgetedSlotRunner(
                    slot_dir / "a6_comm", meter=comm_meter, n_slots=inner_slots, rayon_threads=rayon_threads
                )
                try:
                    r_s0 = ys - f0_oof
                    G_cv_list: List[np.ndarray] = []
                    B_cv_list: List[np.ndarray] = []
                    R_cv_list: List[np.ndarray] = []
                    valid_recs: List[StageCandidateRec] = []
                    w0_dsl = stage_winners[0]["dsl"]
                    for rc in picked_recs:
                        if rc.cand.dsl == w0_dsl and G_vault_list:
                            G_cv_list.append(G_vault_list[0])
                            B_cv_list.append(B_vault_list[0])
                            R_cv_list.append(R_vault_list[0])
                            valid_recs.append(rc)
                            continue
                        got_cv = crossfit_residual_program_kfold(
                            rc.cand,
                            rc.geom,
                            Xs,
                            Xa,
                            r_s0,
                            lab_vault,
                            comm_runner,
                            include_full_refit=not schedule.separate_tier_c,
                            timeout_s=timeout_crossfit,
                            finalizing=True,
                            call_kind="vault_rep77",
                        )
                        if got_cv is not None:
                            gv, bv, rv = got_cv
                            G_cv_list.append(gv)
                            B_cv_list.append(bv)
                            R_cv_list.append(rv if rv is not None else bv)
                            valid_recs.append(rc)
                    if not G_cv_list:
                        cols_out["committee"] = cols_out["shrunk"]
                    else:
                        G_cv_mat = np.column_stack(G_cv_list)
                        w_comm, _ = solve_vault_coordinate_backfit(
                            ys=ys, f0_oof=f0_oof, G_vault=G_cv_mat, lab_vault=lab_vault, reg_alpha=0.0025
                        )
                        a6_comm_weights = [round(float(x), 5) for x in w_comm.tolist()]
                        if schedule.separate_tier_c:
                            G_ct_list: List[np.ndarray] = []
                            B_ct_list: List[np.ndarray] = []
                            R_ct_list: List[np.ndarray] = []
                            for j_c, rc in enumerate(valid_recs):
                                if w_comm[j_c] <= 1e-6:
                                    G_ct_list.append(G_cv_list[j_c])
                                    B_ct_list.append(B_cv_list[j_c])
                                    R_ct_list.append(R_cv_list[j_c])
                                    continue
                                got_ct0 = crossfit_residual_program_kfold(
                                    rc.cand,
                                    rc.geom,
                                    Xs,
                                    Xa,
                                    r_s0,
                                    lab_tc0,
                                    comm_runner,
                                    include_full_refit=True,
                                    timeout_s=timeout_crossfit,
                                    finalizing=True,
                                    call_kind="tier_c",
                                )
                                if got_ct0 is None:
                                    G_ct_list.append(G_cv_list[j_c])
                                    B_ct_list.append(B_cv_list[j_c])
                                    R_ct_list.append(R_cv_list[j_c])
                                else:
                                    gt0, bt0, rt0 = got_ct0
                                    if schedule.tier_c_repeats >= 2:
                                        got_ct1 = crossfit_residual_program_kfold(
                                            rc.cand,
                                            rc.geom,
                                            Xs,
                                            Xa,
                                            r_s0,
                                            lab_tc1,
                                            comm_runner,
                                            include_full_refit=False,
                                            timeout_s=timeout_crossfit,
                                            finalizing=True,
                                            call_kind="tier_c",
                                        )
                                        if got_ct1 is not None:
                                            gt1, bt1, _ = got_ct1
                                            gt0 = 0.5 * (gt0 + gt1)
                                            bt0 = 0.5 * (bt0 + bt1)
                                    G_ct_list.append(gt0)
                                    B_ct_list.append(bt0)
                                    R_ct_list.append(rt0 if rt0 is not None else bt0)
                            G_cexp = np.column_stack(G_ct_list)
                            B_cexp = np.column_stack(B_ct_list)
                            R_cexp = np.column_stack(R_ct_list)
                        else:
                            G_cexp = G_cv_mat
                            B_cexp = np.column_stack(B_cv_list)
                            R_cexp = np.column_stack(R_cv_list)
                        lab_cm = lab_tc0 if schedule.separate_tier_c else lab_vault
                        q_cm = clip_oof_by_train_folds(f0_oof + G_cexp @ w_comm, ys, lab_cm, pad_mult=1.5)
                        b_cm = np.clip(f0_bag + B_cexp @ w_comm, y_lo - pad_y, y_hi + pad_y)
                        r_cm = np.clip(f0_ref + R_cexp @ w_comm, y_lo - pad_y, y_hi + pad_y)
                        cols_out["committee"] = (q_cm, b_cm, r_cm)
                    a6_comm_calls = comm_meter.admitted()
                finally:
                    comm_runner.close()

        return {
            "cid": cid,
            "family_id": family_id,
            "effort": schedule.effort,
            "budget_mode": schedule.budget_mode,
            "budget_unit": schedule.budget_unit,
            "budget_mult": round(float(budget_mult), 3),
            "anchor_mode": anchor_mode,
            "single_stage_control": bool(single_stage_control),
            "stages_planned": n_stages_eff,
            "stages_executed": len(stage_winners),
            "candidates_admitted": meter.candidates_admitted(),
            "candidates_evaluated": int(total_screen_evaluated),
            "candidates_limit": eff_max_cands,
            "worker_calls_admitted": meter.admitted(),
            "worker_calls_limit": eff_max_calls,
            "calls_by_kind": meter.calls_by_kind(),
            "timeouts": runner.inner.timeouts,
            "inner_slots": int(inner_slots),
            "slot_seconds": round(meter.elapsed_s() * max(1, int(inner_slots)), 3),
            "wall_s": round(meter.elapsed_s(), 3),
            "wall_limit_s": round(eff_wall, 3),
            "product_wall_cap_s": round(product_wall_s, 3),
            "enforce_wall_cap": bool(enforce_wall_cap),
            "wall_bound": bool(enforce_wall_cap and meter.wall_expired(reserve_s=1.5)),
            "would_exceed_product_wall": bool(meter.elapsed_s() > product_wall_s),
            "u_vault_f0": round(u_vault_f0, 5),
            "u_oof_shrunk": round(u_shrunk_oof, 5),
            "u_oof_ratchet": round(u_ratchet_oof, 5),
            "shrunk_weights": [round(float(x), 5) for x in shrunk_w.tolist()],
            "ratchet_weights": [round(float(x), 5) for x in ratchet_w.tolist()],
            "a6_comm_weights": a6_comm_weights,
            "a6_comm_posthoc_calls": a6_comm_calls,
            "stages": stage_winners,
            "columns": cols_out,
            "stage_columns": stage_cols_out,
        }
    finally:
        runner.close()


def save_colblock_npz(
    dst_npz: Path,
    slot_dir: Path,
    named_cols: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    wall_s: float,
) -> None:
    """Writes strictly whitelisted `(ok, wall_s, q_*, bag_*, ref_*)` `.npz` compatible with `score_h458s.py` & `score_probe_h458s.py`."""
    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    slot_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "ok": np.array(True),
        "wall_s": np.array(float(wall_s), dtype=np.float64),
    }
    for name, (q_v, b_v, r_v) in sorted(named_cols.items()):
        payload[f"q_{name}"] = np.ascontiguousarray(q_v, dtype=np.float64)
        payload[f"bag_{name}"] = np.ascontiguousarray(b_v, dtype=np.float64)
        payload[f"ref_{name}"] = np.ascontiguousarray(r_v, dtype=np.float64)
    tmp_p = slot_dir / "colblock_export.tmp.npz"
    np.savez(tmp_p, **payload)
    os.replace(tmp_p, dst_npz)


def lock_or_verify_settings(block_dir: Path, settings: Dict[str, object]) -> None:
    """Enforces `cl_fwriter` D1: claims `block_dir` with `SETTINGS.json` and refuses any mismatched reuse."""
    block_dir.mkdir(parents=True, exist_ok=True)
    sp = block_dir / "SETTINGS.json"
    if sp.is_file():
        old = json.loads(sp.read_text())
        for k, v in settings.items():
            if old.get(k) != v:
                raise RuntimeError(
                    f"SETTINGS.json mismatch in {block_dir} on {k!r}: existing={old.get(k)!r} vs new={v!r}"
                )
    else:
        tmp = block_dir / "SETTINGS.tmp.json"
        tmp.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, sp)


def write_sidecar(
    block_dir: Path,
    names: Sequence[str],
    closure: str,
    bank_sha256: str,
    weights_sha256: str,
    prior_mode: str,
    notes: str,
) -> Path:
    if closure.startswith("fair") and prior_mode != "dev91-free":
        raise ValueError(f"Refusing closure={closure!r} with non-dev91-free prior_mode={prior_mode!r} (F1 guard)!")
    block_dir.mkdir(parents=True, exist_ok=True)
    sc = {
        "names": list(names),
        "closure": closure,
        "prior_mode": prior_mode,
        "prior_in_corpus": bool(prior_mode != "dev91-free"),
        "priors": [prior_mode],
        "cross_fitted": True,
        "bank_sha256": bank_sha256,
        "weights_sha256": weights_sha256,
        "producer": "boosted_residual_gp_h459.py",
        "notes": notes,
    }
    p = block_dir / "SIDECAR.json"
    tmp = block_dir / "SIDECAR.tmp.json"
    tmp.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, p)
    return p


def load_search_bank(bank_path: Path) -> Tuple[List[str], Dict[str, str], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], str]:
    """Loads any `(Xs, Xa, ys, lab)` search bank (`P201B`, `FITSEL`, `S7 probe`, `dev91`), hard-asserting `0` `ya` bytes."""
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in str(bank_path):
            raise ValueError(f"Refusing forbidden bank path: {bank_path}")
    b_sha = _sha256_file(bank_path)
    resolved = str(bank_path.resolve())
    if resolved in KNOWN_BANKS:
        exp_sha, b_label = KNOWN_BANKS[resolved]
        if b_sha != exp_sha:
            raise RuntimeError(f"Bank SHA mismatch on {b_label} ({bank_path}): got {b_sha}, expected {exp_sha}")
    rows: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    fam_map: Dict[str, str] = {}
    with np.load(bank_path, allow_pickle=False) as zb:
        assert "ya" not in zb.files and not any(k.endswith("_ya") for k in zb.files), (
            f"AUDIT/PROBE labels forbidden in search bank {bank_path}!"
        )
        cids = [str(x) for x in zb["cids"].tolist()]
        fams = [str(x) for x in zb["family"].tolist()] if "family" in zb.files else [c[:8] for c in cids]
        ss, sa = zb["s_shape"], zb["a_shape"]
        so = ao = lo = 0
        for i, c in enumerate(cids):
            n, w = int(ss[i, 0]), int(ss[i, 1])
            na, f = int(sa[i, 0]), int(sa[i, 1])
            S_mat = np.array(zb["search"][so : so + n * w], np.float64).reshape(n, w)
            Xa = np.array(zb["audit"][ao : ao + na * f], np.float64, copy=True).reshape(na, f)
            rows[c] = (
                np.array(S_mat[:, :-1], np.float64, copy=True),
                Xa,
                np.array(zb["y64"][lo : lo + n], np.float64, copy=True),
                np.array(zb["lab"][lo : lo + n], np.int64, copy=True),
            )
            fam_map[c] = fams[i]
            so += n * w
            ao += na * f
            lo += n
    return sorted(cids), fam_map, rows, b_sha


def load_anchor_col(anchor_dir: Optional[Path], anchor_col: str, cid: str, n_s: int, n_a: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Loads hybrid neural anchor `(q, bag, ref)` or hard-refuses (`H1` guard: zero silent fallback to `pure_gp`)."""
    if anchor_dir is None:
        if anchor_col:
            raise ValueError(f"--anchor-col={anchor_col!r} provided without --anchor-dir")
        return None
    if not anchor_col:
        raise ValueError(f"--anchor-col is required when --anchor-dir={anchor_dir} is set")
    p = anchor_dir / f"{cid}.npz"
    if not p.is_file():
        raise FileNotFoundError(f"Missing hybrid anchor file {p} for cid={cid}")
    with np.load(p, allow_pickle=False) as z:
        assert "ya" not in z.files, f"ya forbidden in anchor file {p}"
        qk, bk, rk = f"q_{anchor_col}", f"bag_{anchor_col}", f"ref_{anchor_col}"
        if qk not in z.files or bk not in z.files:
            raise KeyError(f"Missing required anchor keys ({qk!r}, {bk!r}) in {p}; available={list(z.files)}")
        q = np.asarray(z[qk], dtype=np.float64)
        b = np.asarray(z[bk], dtype=np.float64)
        if len(q) != n_s or len(b) != n_a:
            raise RuntimeError(f"Anchor shape mismatch in {p} ({anchor_col}): got ({len(q)}, {len(b)}) vs ({n_s}, {n_a})")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(b))):
            raise RuntimeError(f"Non-finite q/bag values in anchor {p} ({anchor_col})")
        if rk in z.files:
            r = np.asarray(z[rk], dtype=np.float64)
            if len(r) != n_a or not np.all(np.isfinite(r)):
                raise RuntimeError(f"Invalid or non-finite {rk!r} in anchor {p}: len={len(r)} expected={n_a}")
        else:
            r = b.copy()
        return q, b, r


def run_selftest(slot_root: Path, prior_mode: str = "dev91-free", prior_ckpt: Optional[Path] = None) -> int:
    """Runs synthetic label-free unit verification (`fast`/`F1`/`D1`/`H1` guards, `128`-cand parity, `0` `ya` keys, DOPE sha)."""
    C.resume_stage_v()
    assert _sha256_file(SUPERSET_BIN) == SUPERSET_SHA

    # 1. Verify `fast` effort is hard-refused
    refused_fast = False
    try:
        resolve_effort_schedule("fast", "product", "candidates")
    except ValueError:
        refused_fast = True
    assert refused_fast, "Effort 'fast' must be refused!"

    # 2. Verify `F1` closure guard refuses `fair` with non-dev91-free prior
    refused_f1 = False
    try:
        write_sidecar(slot_root / "f1_guard_chk", ["x"], "fair", "0" * 64, "0" * 64, "in-corpus-v4", "test")
    except ValueError:
        refused_f1 = True
    assert refused_f1, "F1 guard must refuse closure='fair' with in-corpus prior!"

    # 3. Verify `D1` `SETTINGS.json` lock refuses mismatched reuse
    set_dir = slot_root / f"d1_guard_{int(time.time() * 1000)}"
    lock_or_verify_settings(set_dir, {"effort": "medium", "prior": "dev91-free"})
    refused_d1 = False
    try:
        lock_or_verify_settings(set_dir, {"effort": "high", "prior": "dev91-free"})
    except RuntimeError:
        refused_d1 = True
    assert refused_d1, "D1 SETTINGS.json guard must refuse mismatched settings!"

    # 3b. Verify `H1` hybrid anchor guard refuses missing/broken anchor files instead of falling back to pure_gp
    refused_h1 = False
    try:
        load_anchor_col(set_dir, "r4_e8", "00" * 32, 10, 5)
    except FileNotFoundError:
        refused_h1 = True
    assert refused_h1, "H1 guard must refuse missing anchor file!"

    # 4. Synthetic nonlinear multi-component regression check (`0` benchmark data, `0` audit labels; `n_s=640 > 560` exercises subsample!)
    rng = np.random.default_rng(45999)
    n_s, n_a, w = 640, 60, 6
    Xs = rng.standard_normal((n_s, w))
    Xa = rng.standard_normal((n_a, w))
    ys = 1.8 * Xs[:, 0] - 0.9 * (Xs[:, 1] * Xs[:, 2]) + 0.6 * np.sin(Xs[:, 3]) + 0.12 * rng.standard_normal(n_s)
    ys[0] += 9.5  # kurtosis spike to test Island-2 Pseudo-Huber stability

    qhat = UnifiedQHatPrior(prior_mode=prior_mode, ckpt_override=prior_ckpt)
    encoder = BaggedNeuralEncoderH452(dope_threads=1)
    sched = resolve_effort_schedule("medium", "product", "candidates")

    sdir = slot_root / "selftest_a5"
    res_a5 = run_oc_fgb_gp_context(
        cid="00" * 32,
        family_id="selftest_fam",
        Xs=Xs,
        Xa=Xa,
        ys=ys,
        schedule=sched,
        slot_dir=sdir / "a5",
        encoder=encoder,
        qhat=qhat,
        anchor_tup=None,
        single_stage_control=False,
        seed=45901,
        inner_slots=2,
        rayon_threads=1,
    )
    res_a0 = run_oc_fgb_gp_context(
        cid="00" * 32,
        family_id="selftest_fam",
        Xs=Xs,
        Xa=Xa,
        ys=ys,
        schedule=sched,
        slot_dir=sdir / "a0",
        encoder=encoder,
        qhat=qhat,
        anchor_tup=None,
        single_stage_control=True,
        export_committee_m=sched.n_stages,
        seed=45901,
        inner_slots=2,
        rayon_threads=1,
    )
    res_a0_1p5 = run_oc_fgb_gp_context(
        cid="00" * 32,
        family_id="selftest_fam",
        Xs=Xs,
        Xa=Xa,
        ys=ys,
        schedule=sched,
        slot_dir=sdir / "a0_1p5x",
        encoder=encoder,
        qhat=qhat,
        anchor_tup=None,
        single_stage_control=True,
        budget_mult=1.5,
        seed=45901 + 20011,
        inner_slots=2,
        rayon_threads=1,
    )
    assert res_a5["candidates_admitted"] <= 128, f"A5 exceeded 128 cands: {res_a5['candidates_admitted']}"
    assert res_a5["worker_calls_admitted"] <= 210, f"A5 exceeded 210 calls: {res_a5['worker_calls_admitted']}"
    assert res_a0["candidates_admitted"] <= 128, f"A0 exceeded 128 cands: {res_a0['candidates_admitted']}"
    assert res_a0_1p5["candidates_admitted"] <= 192, f"A0_1p5x exceeded 192 cands: {res_a0_1p5['candidates_admitted']}"
    assert res_a5["stages_executed"] >= 2, f"Expected multi-stage execution, got {res_a5['stages_executed']}"
    assert "committee" in res_a0["columns"], "Expected A6 committee column on a0_s0!"

    out_npz = sdir / "blk" / f"{'00'*32}.npz"
    save_colblock_npz(
        out_npz,
        sdir,
        {
            "a5_shrunk": res_a5["columns"]["shrunk"],
            "a5_ratchet": res_a5["columns"]["ratchet"],
            "a6_comm": res_a0["columns"]["committee"],
        },
        res_a5["wall_s"],
    )
    with np.load(out_npz, allow_pickle=False) as z:
        assert "ya" not in z.files
        assert z["q_a5_shrunk"].shape == (n_s,) and z["ref_a5_shrunk"].shape == (n_a,)
        assert np.all(np.isfinite(z["ref_a5_shrunk"])) and np.all(np.isfinite(z["ref_a6_comm"]))

    dope_ok, dope_sha = encoder.verify_dope_unchanged()
    assert dope_ok, f"DOPE mutated: {dope_sha}"
    print(
        json.dumps(
            {
                "selftest": "PASS",
                "prior_mode": qhat.prior_mode,
                "fast_refused": refused_fast,
                "f1_refused": refused_f1,
                "d1_refused": refused_d1,
                "h1_refused": refused_h1,
                "a5_cands": res_a5["candidates_admitted"],
                "a5_calls": res_a5["worker_calls_admitted"],
                "a0_cands": res_a0["candidates_admitted"],
                "a0_calls": res_a0["worker_calls_admitted"],
                "a0_1p5x_cands": res_a0_1p5["candidates_admitted"],
                "a0_1p5x_calls": res_a0_1p5["worker_calls_admitted"],
                "a5_stages": res_a5["stages_executed"],
                "a5_u_oof_shrunk": res_a5["u_oof_shrunk"],
                "a0_u_oof_shrunk": res_a0["u_oof_shrunk"],
                "dope_unchanged": dope_ok,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _summarize_ledger(results: Sequence[Dict], key: str, inner_slots: int = 3) -> Optional[Dict[str, object]]:
    sub = [r[key] for r in results if isinstance(r.get(key), dict) and "candidates_admitted" in r[key]]
    if not sub:
        return None
    cands = [int(x["candidates_admitted"]) for x in sub]
    calls = [int(x["worker_calls_admitted"]) for x in sub]
    walls = [float(x["wall_s"]) for x in sub]
    by_kind_sum: Dict[str, int] = {}
    for x in sub:
        for kk, kv in x.get("calls_by_kind", {}).items():
            by_kind_sum[kk] = by_kind_sum.get(kk, 0) + int(kv)
    return {
        "n_contexts": len(sub),
        "mean_candidates": round(float(np.mean(cands)), 2),
        "max_candidates": int(max(cands)),
        "mean_worker_calls": round(float(np.mean(calls)), 2),
        "max_worker_calls": int(max(calls)),
        "mean_wall_s": round(float(np.mean(walls)), 3),
        "mean_slot_seconds": round(float(np.mean(walls)) * max(1, int(inner_slots)), 3),
        "p95_wall_s": round(float(np.percentile(walls, 95)), 3),
        "total_calls_by_kind": by_kind_sum,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default="/home/ubuntu/jope-data/h430/bank_P201B.npz")
    ap.add_argument("--effort", default="medium", help="medium | high | ultra (fast is refused)")
    ap.add_argument("--budget-mode", default="product", choices=["product", "cycle"])
    ap.add_argument("--budget-unit", default="candidates", choices=["candidates", "worker_calls"],
                    help="Primary shared S9 unit: 'candidates' (effort.rs 128/512/4096) or 'worker_calls'")
    ap.add_argument("--arm-mode", default="paired", choices=["a5", "a0", "paired"],
                    help="a5 (OC-FGB-GP), a0 (single-stage), or paired (a5 + a0_s0 + a0_s1 + a0_1p5x + a6_comm)")
    ap.add_argument("--prior", default="dev91-free", choices=["dev91-free", "in-corpus-v2", "in-corpus-v4"],
                    help="Surrogate prior checkpoint; 'dev91-free' (qhat_v2.pt bf05e040...) is required for fair closure")
    ap.add_argument("--qhat-ckpt", default="", help="Optional explicit prior checkpoint override")
    ap.add_argument("--anchor-dir", default="", help="Optional _load_col_block dir for hybrid_neural anchor")
    ap.add_argument("--anchor-col", default="", help="Column name inside --anchor-dir (e.g. r4_e8 or icl_e8_all_cv7)")
    ap.add_argument(
        "--closure",
        default="",
        choices=["", "fair", "fair-gp-search-only", "in-corpus", "in-corpus-prior", "fresh-probe"],
        help="Required explicit closure label on board runs (H4 guard; never defaulted silently)",
    )
    ap.add_argument(
        "--enforce-wall-cap",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default S9 candidate parity, H2) disables load-dependent wall truncation; 1 enforces 60/300/1800s cap",
    )
    ap.add_argument("--colblock-dir", default="/home/ubuntu/jope-data/h459-s9/p201b_medium_prod_pure")
    ap.add_argument("--slots-root", default="/home/ubuntu/jope-data/h459-s9/slots/local")
    ap.add_argument("--out-receipt", default="/home/ubuntu/jope_plan/evidence/H459/runs/RUN_RECEIPT.json")
    ap.add_argument("--indices", default="")
    ap.add_argument("--cid-sha-top-n", type=int, default=0,
                    help="If >0, select top-N contexts by sha256(cid_ascii) per PREREG_S9.md §4.2 (80 on P201B, 20 ultra)")
    ap.add_argument("--dom-mult", type=float, default=2.0,
                    help="Budget multiplier for single-stage dominating control A0x (default 2.0 per PREREG_S9.md §2)")
    ap.add_argument("--stride-mod", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inner-slots", type=int, default=3)
    ap.add_argument("--rayon-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=45901)
    ap.add_argument("--export-stage-curve", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    try:
        os.nice(19)
    except OSError:
        pass

    qhat_override = Path(a.qhat_ckpt) if a.qhat_ckpt else None
    slots_root = Path(a.slots_root)
    if a.selftest:
        return run_selftest(slots_root, prior_mode=a.prior, prior_ckpt=qhat_override)

    if not a.closure:
        raise ValueError("--closure is required on all non-selftest runs (H4 guard)!")

    C.resume_stage_v()
    assert _sha256_file(SUPERSET_BIN) == SUPERSET_SHA, f"SUPERSET_BIN sha mismatch: {SUPERSET_BIN}"
    sched = resolve_effort_schedule(a.effort, a.budget_mode, a.budget_unit)
    qhat = UnifiedQHatPrior(prior_mode=a.prior, ckpt_override=qhat_override)
    if a.closure.startswith("fair") and qhat.prior_mode != "dev91-free":
        raise ValueError(f"Refusing --closure={a.closure!r} with non-dev91-free prior {qhat.prior_mode!r} (F1 guard)!")
    encoder = BaggedNeuralEncoderH452(dope_threads=2)

    cids_sorted, fam_map, bank_rows, bank_sha = load_search_bank(Path(a.bank))
    if a.cid_sha_top_n > 0:
        sha_ranked_cids = set(
            sorted(cids_sorted, key=lambda c: hashlib.sha256(c.encode("ascii")).hexdigest())[: int(a.cid_sha_top_n)]
        )
        base_idx = [i for i, c in enumerate(cids_sorted) if c in sha_ranked_cids]
    elif a.indices:
        base_idx = [i for i in _parse_indices(a.indices) if 0 <= i < len(cids_sorted)]
    else:
        base_idx = list(range(len(cids_sorted)))

    size_ranked = sorted(
        base_idx,
        key=lambda idx: -(bank_rows[cids_sorted[idx]][0].shape[0] * math.sqrt(max(1, bank_rows[cids_sorted[idx]][0].shape[1]))),
    )
    if a.stride_mod:
        r_s, m_s = a.stride_mod.split("/", 1)
        r_i, m_i = int(r_s), int(m_s)
        target_idx = [idx for pos, idx in enumerate(size_ranked) if (pos % m_i) == r_i]
    else:
        target_idx = size_ranked

    anc_dir = Path(a.anchor_dir) if a.anchor_dir else None
    colblock_dir = Path(a.colblock_dir)
    colblock_dir.mkdir(parents=True, exist_ok=True)
    tag_prefix = f"{sched.effort}_{'hyb' if anc_dir else 'pure'}"
    dom_tag = "2x" if abs(float(a.dom_mult) - 2.0) < 1e-6 else ("1p5x" if abs(float(a.dom_mult) - 1.5) < 1e-6 else "dom")

    exported_col_names: List[str] = []
    if a.arm_mode in ("a5", "paired"):
        exported_col_names.extend([f"a5_{tag_prefix}_shrunk", f"a5_{tag_prefix}_ratchet"])
        if bool(a.export_stage_curve):
            for k_s in range(1, sched.n_stages + 1):
                exported_col_names.append(f"a5_{tag_prefix}_stg{k_s}")
    if a.arm_mode == "a0":
        exported_col_names.append(f"a0_{tag_prefix}_s0")
    elif a.arm_mode == "paired":
        exported_col_names.extend([
            f"a0_{tag_prefix}_s0",
            f"a0_{tag_prefix}_s1",
            f"a0_{tag_prefix}_{dom_tag}",
            f"a6_{tag_prefix}_comm",
            f"a6b_{tag_prefix}_runbag",
        ])

    lock_or_verify_settings(
        colblock_dir,
        {
            "schema": "jope.h459.settings/v1",
            "bank_sha256": bank_sha,
            "cid_sha_top_n": int(a.cid_sha_top_n),
            "effort": sched.effort,
            "budget_mode": sched.budget_mode,
            "budget_unit": sched.budget_unit,
            "arm_mode": a.arm_mode,
            "dom_mult": round(float(a.dom_mult), 4),
            "prior_mode": qhat.prior_mode,
            "qhat_sha256": qhat.ckpt_sha256,
            "anchor_dir": str(anc_dir) if anc_dir else "",
            "anchor_col": a.anchor_col,
            "closure": a.closure,
            "enforce_wall_cap": bool(a.enforce_wall_cap),
            "seed": int(a.seed),
            "exported_columns": exported_col_names,
        },
    )

    def _run_one(idx: int) -> Dict:
        cid = cids_sorted[idx]
        dst_npz = colblock_dir / f"{cid}.npz"
        meta_json = colblock_dir / f"{cid}.meta.json"
        if dst_npz.is_file() and meta_json.is_file():
            try:
                with np.load(dst_npz, allow_pickle=False) as zchk:
                    if bool(zchk["ok"]) and all(f"ref_{nm}" in zchk.files for nm in exported_col_names):
                        prev_rec = json.loads(meta_json.read_text())
                        prev_rec["skipped_existing"] = True
                        return prev_rec
            except Exception:
                pass

        Xs, Xa, ys, _lab = bank_rows[cid]
        anc_tup = load_anchor_col(anc_dir, a.anchor_col, cid, len(ys), len(Xa))
        sdir = slots_root / f"ctx_{idx:03d}"
        t_ctx = time.monotonic()
        named_cols: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        rec_out: Dict[str, object] = {"cid": cid, "family_id": fam_map[cid], "has_anchor": anc_tup is not None}

        if a.arm_mode in ("a5", "paired"):
            r5 = run_oc_fgb_gp_context(
                cid=cid,
                family_id=fam_map[cid],
                Xs=Xs,
                Xa=Xa,
                ys=ys,
                schedule=sched,
                slot_dir=sdir / "a5",
                encoder=encoder,
                qhat=qhat,
                anchor_tup=anc_tup,
                single_stage_control=False,
                enforce_wall_cap=bool(a.enforce_wall_cap),
                seed=a.seed,
                inner_slots=a.inner_slots,
                rayon_threads=a.rayon_threads,
            )
            named_cols[f"a5_{tag_prefix}_shrunk"] = r5["columns"]["shrunk"]
            named_cols[f"a5_{tag_prefix}_ratchet"] = r5["columns"]["ratchet"]
            if bool(a.export_stage_curve):
                last_col = r5["columns"]["shrunk"]
                for k_s in range(1, sched.n_stages + 1):
                    c_k = r5["stage_columns"].get(k_s, last_col)
                    last_col = c_k
                    named_cols[f"a5_{tag_prefix}_stg{k_s}"] = c_k
            rec_out["a5"] = {k: v for k, v in r5.items() if k not in ("columns", "stage_columns")}

        if a.arm_mode in ("a0", "paired"):
            specs_a0 = (
                [("s0", a.seed, 1.0, 0)]
                if a.arm_mode == "a0"
                else [
                    ("s0", a.seed, 1.0, sched.n_stages),
                    ("s1", a.seed + 10007, 1.0, 0),
                    (dom_tag, a.seed + 20011, float(a.dom_mult), 0),
                ]
            )
            for s_tag, s_val, b_mult, comm_m in specs_a0:
                r0 = run_oc_fgb_gp_context(
                    cid=cid,
                    family_id=fam_map[cid],
                    Xs=Xs,
                    Xa=Xa,
                    ys=ys,
                    schedule=sched,
                    slot_dir=sdir / f"a0_{s_tag}",
                    encoder=encoder,
                    qhat=qhat,
                    anchor_tup=anc_tup,
                    single_stage_control=True,
                    budget_mult=b_mult,
                    export_committee_m=comm_m,
                    enforce_wall_cap=bool(a.enforce_wall_cap),
                    seed=s_val,
                    inner_slots=a.inner_slots,
                    rayon_threads=a.rayon_threads,
                )
                named_cols[f"a0_{tag_prefix}_{s_tag}"] = r0["columns"]["shrunk"]
                if comm_m > 0 and "committee" in r0["columns"]:
                    named_cols[f"a6_{tag_prefix}_comm"] = r0["columns"]["committee"]
                rec_out[f"a0_{s_tag}"] = {k: v for k, v in r0.items() if k not in ("columns", "stage_columns")}
            if a.arm_mode == "paired":
                c_s0 = named_cols[f"a0_{tag_prefix}_s0"]
                c_s1 = named_cols[f"a0_{tag_prefix}_s1"]
                named_cols[f"a6b_{tag_prefix}_runbag"] = (
                    0.5 * (c_s0[0] + c_s1[0]),
                    0.5 * (c_s0[1] + c_s1[1]),
                    0.5 * (c_s0[2] + c_s1[2]),
                )

        wall_ctx = round(time.monotonic() - t_ctx, 3)
        rec_out["wall_total_s"] = wall_ctx
        tmp_meta = sdir / "ctx_meta.tmp.json"
        sdir.mkdir(parents=True, exist_ok=True)
        tmp_meta.write_text(json.dumps(rec_out, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_meta, meta_json)
        save_colblock_npz(dst_npz, sdir, named_cols, wall_ctx)
        print(
            json.dumps(
                {
                    "cid": cid[:12],
                    "done": True,
                    "wall_s": wall_ctx,
                    "a5_cands": rec_out.get("a5", {}).get("candidates_admitted") if isinstance(rec_out.get("a5"), dict) else None,
                    "a5_calls": rec_out.get("a5", {}).get("worker_calls_admitted") if isinstance(rec_out.get("a5"), dict) else None,
                    "a0_calls": rec_out.get("a0_s0", {}).get("worker_calls_admitted") if isinstance(rec_out.get("a0_s0"), dict) else None,
                    "a0_dom_calls": rec_out.get(f"a0_{dom_tag}", {}).get("worker_calls_admitted") if isinstance(rec_out.get(f"a0_{dom_tag}"), dict) else None,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return rec_out

    t_all = time.monotonic()
    results: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = [ex.submit(_run_one, idx) for idx in target_idx]
        for f in cf.as_completed(futs):
            results.append(f.result())

    dope_ok, dope_sha = encoder.verify_dope_unchanged()
    assert dope_ok, f"DOPE weights changed: {dope_sha}"

    sc_path = write_sidecar(
        block_dir=colblock_dir,
        names=exported_col_names,
        closure=a.closure,
        bank_sha256=bank_sha,
        weights_sha256=qhat.ckpt_sha256,
        prior_mode=qhat.prior_mode,
        notes=(
            f"H459 Track S9 Arm A5 OC-FGB-GP ({sched.effort}, {sched.budget_mode}, "
            f"unit={sched.budget_unit}, prior={qhat.prior_mode}, anchor={a.anchor_col or 'pure_gp'})"
        ),
    )

    ledger_summary = {
        k: v
        for k, v in {
            "a5": _summarize_ledger(results, "a5", a.inner_slots),
            "a0_s0": _summarize_ledger(results, "a0_s0", a.inner_slots),
            "a0_s1": _summarize_ledger(results, "a0_s1", a.inner_slots),
            f"a0_{dom_tag}": _summarize_ledger(results, f"a0_{dom_tag}", a.inner_slots),
        }.items()
        if v is not None
    }

    doc = {
        "schema": "jope.h459.s9_arm_a5/v1.1",
        "effort": sched.effort,
        "budget_mode": sched.budget_mode,
        "budget_unit": sched.budget_unit,
        "arm_mode": a.arm_mode,
        "prior_mode": qhat.prior_mode,
        "closure": a.closure,
        "bank": a.bank,
        "bank_sha256": bank_sha,
        "sidecar": str(sc_path),
        "exported_columns": exported_col_names,
        "n_contexts_done": len(results),
        "ledger_summary": ledger_summary,
        "dope_unchanged": dope_ok,
        "dope_sha256": dope_sha,
        "superset_sha256": SUPERSET_SHA,
        "qhat_sha256": qhat.ckpt_sha256,
        "elapsed_s": round(time.monotonic() - t_all, 2),
        "utc": time.strftime("%FT%TZ", time.gmtime()),
        "contexts": results,
    }
    out_r = Path(a.out_receipt)
    out_r.parent.mkdir(parents=True, exist_ok=True)
    tmp_r = out_r.with_suffix(".tmp.json")
    tmp_r.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_r, out_r)
    print(json.dumps({k: v for k, v in doc.items() if k != "contexts"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
