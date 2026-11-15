"""Outer-fold leak-free provenance, snapshot CAS, and whole-procedure closure (`FitScopeV2`).

Enforces three invariants across H460 `APEX-EQ`:
1. `FitScopeV2`: Full SHA-256 content-addressed provenance over training features,
   training labels, row weights, incumbent residual vector, optional auxiliary arrays
   (`bank_train`, `bank_val`, `incumbent_train`, `incumbent_val`, `static_ctrl_val`),
   and fold identifier (`cl_fwriter` Finding #7).
2. `EnsembleSnapshot` + `CoordinatorCAS`: Immutable state snapshots committed one at a
   time via compare-and-swap (`base_epoch`), re-verifying stale worker proposals against
   the current covariance before acceptance (`Coalition §4.1`).
3. `learn_whole_procedure(X_train, y_train, X_val, learner_fn)`: Strict outer-fold
   functional boundary where `y_val` is absent, paired with both in-memory and on-disk
   poisoning canaries proving `max_abs_diff == 0.0` under held-out label corruption.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np


def _hash_array_f64(arr: np.ndarray) -> str:
    """Return canonical full 64-hex SHA-256 over dtype, shape, and contiguous float64 bytes."""
    contig = np.ascontiguousarray(arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(f"f64:{contig.shape}:".encode("utf-8"))
    h.update(contig.tobytes())
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class FitScopeV2:
    """Content-addressed fit scope binding a discovery/refit step to exact training inputs."""

    fold_id: str
    n_rows: int
    n_features: int
    x_sha256: str
    y_sha256: str
    weights_sha256: str
    residual_sha256: str
    extra_sha256: str
    scope_sha256: str

    @classmethod
    def from_arrays(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        residual: Optional[np.ndarray] = None,
        row_weights: Optional[np.ndarray] = None,
        extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
        fold_id: str = "full",
    ) -> "FitScopeV2":
        x_arr = np.ascontiguousarray(x_train, dtype=np.float64)
        y_arr = np.ascontiguousarray(y_train, dtype=np.float64).reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_arr.shape}")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Row mismatch: x_train={x_arr.shape[0]} vs y_train={y_arr.shape[0]}")
        res_arr = (
            y_arr
            if residual is None
            else np.ascontiguousarray(residual, dtype=np.float64).reshape(-1)
        )
        w_arr = (
            np.ones(x_arr.shape[0], dtype=np.float64)
            if row_weights is None
            else np.ascontiguousarray(row_weights, dtype=np.float64).reshape(-1)
        )
        if res_arr.shape[0] != x_arr.shape[0] or w_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("residual and row_weights must match x_train row count")

        extra_hashes: Dict[str, str] = {}
        if extra_arrays:
            for k in sorted(extra_arrays.keys()):
                extra_hashes[str(k)] = _hash_array_f64(extra_arrays[k])
        extra_sha = hashlib.sha256(
            json.dumps(extra_hashes, sort_keys=True).encode("utf-8")
        ).hexdigest()

        x_sha = _hash_array_f64(x_arr)
        y_sha = _hash_array_f64(y_arr)
        w_sha = _hash_array_f64(w_arr)
        r_sha = _hash_array_f64(res_arr)
        payload = json.dumps(
            {
                "extra": extra_sha,
                "fold_id": str(fold_id),
                "n_features": int(x_arr.shape[1]),
                "n_rows": int(x_arr.shape[0]),
                "r": r_sha,
                "schema": "jope.h460.fit_scope/v2",
                "w": w_sha,
                "x": x_sha,
                "y": y_sha,
            },
            sort_keys=True,
        ).encode("utf-8")
        return cls(
            fold_id=str(fold_id),
            n_rows=int(x_arr.shape[0]),
            n_features=int(x_arr.shape[1]),
            x_sha256=x_sha,
            y_sha256=y_sha,
            weights_sha256=w_sha,
            residual_sha256=r_sha,
            extra_sha256=extra_sha,
            scope_sha256=hashlib.sha256(payload).hexdigest(),
        )


class ScopedFeatureCache:
    """Provenance-keyed cache that strictly refuses hits across distinct `FitScopeV2` hashes."""

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], np.ndarray] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(self, scope: FitScopeV2, expr_id: str) -> Optional[np.ndarray]:
        key = (scope.scope_sha256, str(expr_id))
        val = self._store.get(key)
        if val is None:
            self.misses += 1
            return None
        self.hits += 1
        return val.copy()

    def put(self, scope: FitScopeV2, expr_id: str, values: np.ndarray) -> None:
        arr = np.ascontiguousarray(values, dtype=np.float64)
        arr.setflags(write=False)
        self._store[(scope.scope_sha256, str(expr_id))] = arr


ALLOWED_EVIDENCE_ROLES = ("outer_whole_procedure", "conditional_fixed_matrix_lfo")


@dataclasses.dataclass(frozen=True)
class EnsembleSnapshot:
    """Immutable coordinator snapshot published to parallel proposal workers (`F04, F08`)."""

    epoch: int
    scope_sha256: str
    active_names: Tuple[str, ...]
    marginal_log_evidence: float
    alpha_Dual: np.ndarray
    state_digest: str
    evidence_role: str = "outer_whole_procedure"

    def __post_init__(self) -> None:
        if self.evidence_role not in ALLOWED_EVIDENCE_ROLES:
            raise ValueError(f"Invalid evidence_role: {self.evidence_role!r}")
        ro_alpha = np.array(self.alpha_Dual, dtype=np.float64, copy=True)
        ro_alpha.setflags(write=False)
        object.__setattr__(self, "alpha_Dual", ro_alpha)


@dataclasses.dataclass(frozen=True)
class WorkerProposal:
    """Asynchronous candidate proposal tagged with the snapshot epoch & scope it conditioned on (`F08`)."""

    base_epoch: int
    name: str
    train_col: np.ndarray
    val_col: Optional[np.ndarray]
    claimed_score_z2: float
    family_tag: str = "generic"
    proposal_scope_sha256: Optional[str] = None
    evidence_role: str = "outer_whole_procedure"

    def __post_init__(self) -> None:
        if self.evidence_role not in ALLOWED_EVIDENCE_ROLES:
            raise ValueError(f"Invalid evidence_role: {self.evidence_role!r}")
        ro_tr = np.array(self.train_col, dtype=np.float64, copy=True)
        ro_tr.setflags(write=False)
        object.__setattr__(self, "train_col", ro_tr)
        if self.val_col is not None:
            ro_va = np.array(self.val_col, dtype=np.float64, copy=True)
            ro_va.setflags(write=False)
            object.__setattr__(self, "val_col", ro_va)


class CoordinatorCAS:
    """Single-writer compare-and-swap coordinator over `EnsembleSnapshot` epochs."""

    def __init__(self, initial_snapshot: EnsembleSnapshot) -> None:
        self._current = initial_snapshot
        self.accepted_epochs: list[Tuple[int, str, bool]] = []

    @property
    def snapshot(self) -> EnsembleSnapshot:
        return self._current

    def try_commit(
        self,
        proposal: WorkerProposal,
        reverify_fn: Callable[[WorkerProposal, EnsembleSnapshot], Optional[EnsembleSnapshot]],
    ) -> Tuple[bool, EnsembleSnapshot]:
        """Commit `proposal`; enforce `proposal_scope_sha256` and re-verify stale proposals (`F08`)."""
        if (
            proposal.proposal_scope_sha256 is not None
            and proposal.proposal_scope_sha256 != self._current.scope_sha256
        ):
            return False, self._current
        if proposal.evidence_role != self._current.evidence_role:
            return False, self._current
        was_stale = proposal.base_epoch != self._current.epoch
        next_snap = reverify_fn(proposal, self._current)
        if next_snap is None:
            return False, self._current
        if next_snap.scope_sha256 != self._current.scope_sha256:
            raise ValueError("Accepted snapshot must preserve scope_sha256")
        if next_snap.epoch != self._current.epoch + 1:
            raise ValueError("Accepted snapshot must increment epoch by exactly 1")
        self._current = next_snap
        self.accepted_epochs.append((next_snap.epoch, proposal.name, was_stale))
        return True, self._current


def learn_whole_procedure(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    learner_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, FitScopeV2], Tuple[np.ndarray, Dict[str, Any]]],
    fold_id: str = "outer_fold",
    extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, FitScopeV2, Dict[str, Any]]:
    """Execute an entire discovery + equilibrium fit strictly inside `(x_train, y_train, x_val)`.

    `y_val` is structurally absent from this interface: outer validation labels cannot enter
    standardization, feature screening, kernel lengthscales, active-set selection, or damping.
    """
    x_tr = np.array(x_train, dtype=np.float64, copy=True)
    y_tr = np.array(y_train, dtype=np.float64, copy=True).reshape(-1)
    x_va = np.array(x_val, dtype=np.float64, copy=True)
    x_tr.setflags(write=False)
    y_tr.setflags(write=False)
    x_va.setflags(write=False)
    scope = FitScopeV2.from_arrays(x_tr, y_tr, extra_arrays=extra_arrays, fold_id=fold_id)
    preds_val, meta = learner_fn(x_tr, y_tr, x_va, scope)
    return np.asarray(preds_val, dtype=np.float64).reshape(-1), scope, meta


def run_outer_fold_poisoning_canary(
    x_all: np.ndarray,
    y_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    sliced_pipeline_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, Dict[str, Any]]],
    poison_offset: float = 1000.0,
) -> Dict[str, Any]:
    """Verify `max_abs_diff == 0.0` when held-out labels `y_all[val_idx]` are poisoned by `+1000`."""
    y_clean = np.array(y_all, dtype=np.float64, copy=True)
    y_poison = np.array(y_all, dtype=np.float64, copy=True)
    y_poison[val_idx] += float(poison_offset)

    pred_clean, meta_clean = sliced_pipeline_fn(x_all, y_clean, train_idx, val_idx)
    pred_poison, meta_poison = sliced_pipeline_fn(x_all, y_poison, train_idx, val_idx)
    max_diff = float(np.max(np.abs(pred_clean - pred_poison)))
    state_match = meta_clean.get("state_digest") == meta_poison.get("state_digest")
    scope_match = meta_clean.get("fit_scope_sha256") == meta_poison.get("fit_scope_sha256")
    return {
        "max_abs_diff": max_diff,
        "passed": bool(max_diff == 0.0 and state_match and scope_match),
        "poison_offset": float(poison_offset),
        "scope_match": bool(scope_match),
        "state_digest_clean": meta_clean.get("state_digest"),
        "state_digest_poison": meta_poison.get("state_digest"),
        "state_match": bool(state_match),
    }
