"""Transitive Leak-Free Provenance, Sovereign `S0` Policy, and Quarantine Guards (`contracts_v3.py`).

Upgrades `H460`'s `FitScopeV2` to `FitClosureV3` (`Learner Compiler §4`, `SYMBIONT §15`, `cl_h458` `06:05Z`):
1. Binds exact training arrays, `dataset_id`, `family_id`, `outer_fold_id`, allowed/excluded row domains,
   checkpoint corpus exposure families, and `hint_protocol_id` under a deterministic SHA-256 closure.
2. Blocks `gp_chain_h458.py --base-block`-style **global-OOF leakage**: any ancestor block consumed inside
   an outer fold (`outer_fold_id != "full_search"`) MUST carry the identical `outer_fold_id` (not `"global_oof"`).
3. Enforces strict sovereign `S0` (`"ours_only"`) contender eligibility, rejecting external foundation model
   columns (`kumo`, `tabpfn`, `tabicl`, `tabdpt`, `mitra`) or external teacher distillation in `S0`.
4. Refuses any path containing quarantined evaluation tokens (`T14`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import PurePath
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

FORBIDDEN_PATH_TOKENS: Tuple[str, ...] = (
    "confirm",
    "fmgap_b",
    "fmgap-b",
    "fmgap_c",
    "fmgap-c",
    "holdout",
    "sealed",
    "dl-val",
    "dl_val",
    "probe_ya",
)

EXTERNAL_FM_TOKENS: Tuple[str, ...] = (
    "kumo",
    "tabpfn",
    "tabicl",
    "tabdpt",
    "mitra",
    "autogluon_fm",
    "priorlabs",
)

ALLOWED_SOVEREIGN_TIERS: Tuple[str, ...] = ("S0", "S0_T_DIAGNOSTIC", "S1_DIAGNOSTIC", "S2_DIAGNOSTIC")
ALLOWED_INFERENCE_TIERS: Dict[str, Tuple[int, float]] = {
    "medium": (128, 60.0),
    "high": (512, 300.0),
    "ultra": (4096, 1800.0),
}


def hash_array_f64(arr: np.ndarray) -> str:
    """Canonical 64-hex SHA-256 over shape and contiguous `float64` bytes."""
    contig = np.ascontiguousarray(arr, dtype=np.float64)
    h = hashlib.sha256()
    h.update(f"f64:{contig.shape}:".encode("utf-8"))
    h.update(contig.tobytes())
    return h.hexdigest()


def assert_path_not_quarantined(path_like: str | PurePath) -> None:
    """Raise `PermissionError` if `path_like` touches any sealed/confirm/holdout board (`T14`)."""
    norm = str(path_like).lower().replace("\\", "/")
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in norm:
            raise PermissionError(
                f"Quarantine refusal (T14): path {str(path_like)!r} matches forbidden token {tok!r}"
            )


def assert_sovereign_s0_candidates(
    candidate_names: Sequence[str],
    sovereign_tier: str = "S0",
    teacher_id: Optional[str] = None,
) -> None:
    """Enforce strict `S0` ('ours only') contender policy."""
    if sovereign_tier not in ALLOWED_SOVEREIGN_TIERS:
        raise ValueError(f"Unknown sovereign_tier: {sovereign_tier!r}")
    if sovereign_tier != "S0":
        return
    if teacher_id is not None:
        low_t = str(teacher_id).lower()
        for ext in EXTERNAL_FM_TOKENS:
            if ext in low_t:
                raise PermissionError(
                    f"S0 violation: external teacher {teacher_id!r} prohibited in sovereign S0 contender"
                )
    for name in candidate_names:
        low = str(name).lower()
        for ext in EXTERNAL_FM_TOKENS:
            if ext in low:
                raise PermissionError(
                    f"S0 violation: external FM column {name!r} prohibited in sovereign S0 contender"
                )


@dataclasses.dataclass(frozen=True)
class FitClosureV3:
    """Transitive content-addressed fit closure (`jope.h462.fit_closure/v3`)."""

    dataset_id: str
    family_id: str
    outer_fold_id: str
    allowed_row_domain: str
    excluded_row_domains: Tuple[str, ...]
    sovereign_tier: str
    inference_tier: str
    hint_protocol_id: str
    x_sha256: str
    y_sha256: str
    extra_sha256: str
    checkpoint_exposed_families: Tuple[str, ...]
    ancestor_closures: Tuple[str, ...]
    closure_sha256: str

    @classmethod
    def create(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        dataset_id: str,
        family_id: str,
        outer_fold_id: str = "full_search",
        allowed_row_domain: str = "SEARCH_TRAIN",
        excluded_row_domains: Sequence[str] = ("AUDIT", "CONFIRM", "SEALED"),
        sovereign_tier: str = "S0",
        inference_tier: str = "medium",
        hint_protocol_id: str = "none",
        checkpoint_exposed_families: Sequence[str] = (),
        ancestors: Sequence["FitClosureV3"] = (),
        extra_arrays: Optional[Mapping[str, np.ndarray]] = None,
        candidate_names: Sequence[str] = (),
        teacher_id: Optional[str] = None,
    ) -> "FitClosureV3":
        if inference_tier not in ALLOWED_INFERENCE_TIERS:
            raise ValueError(f"Unsupported inference_tier: {inference_tier!r}")
        assert_sovereign_s0_candidates(candidate_names, sovereign_tier=sovereign_tier, teacher_id=teacher_id)

        x_arr = np.ascontiguousarray(x_train, dtype=np.float64)
        y_arr = np.ascontiguousarray(y_train, dtype=np.float64).reshape(-1)
        if x_arr.ndim != 2 or x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"Invalid (x_train, y_train) shapes: {x_arr.shape} vs {y_arr.shape}")

        fam_str = str(family_id)
        ds_str = str(dataset_id)
        fold_str = str(outer_fold_id)
        exposed_set = set(str(f) for f in checkpoint_exposed_families)

        for anc in ancestors:
            if anc.dataset_id != ds_str:
                raise ValueError(
                    f"Ancestor dataset_id mismatch: {anc.dataset_id!r} != {ds_str!r}"
                )
            # Block global-OOF base blocks inside an outer CV fold (cl_h458 06:05Z finding!)
            if fold_str != "full_search" and anc.outer_fold_id != fold_str:
                raise PermissionError(
                    f"Nested fold closure violation: outer fold {fold_str!r} cannot consume ancestor "
                    f"fit on {anc.outer_fold_id!r} (global-OOF base-block leak blocked)"
                )
            if anc.sovereign_tier != "S0" and sovereign_tier == "S0":
                raise PermissionError(
                    f"S0 closure cannot depend on non-S0 ancestor ({anc.sovereign_tier!r})"
                )
            exposed_set.update(anc.checkpoint_exposed_families)

        if fam_str and fam_str in exposed_set:
            raise PermissionError(
                f"Corpus exposure leak (be7de9bc guard): target family {fam_str!r} is in checkpoint/ancestor "
                f"exposed families {sorted(exposed_set)!r}"
            )

        extra_map: Dict[str, str] = {}
        if extra_arrays:
            for k in sorted(extra_arrays.keys()):
                extra_map[str(k)] = hash_array_f64(extra_arrays[k])
        extra_sha = hashlib.sha256(json.dumps(extra_map, sort_keys=True).encode("utf-8")).hexdigest()

        x_sha = hash_array_f64(x_arr)
        y_sha = hash_array_f64(y_arr)
        excl_tuple = tuple(sorted(set(str(e) for e in excluded_row_domains)))
        exp_tuple = tuple(sorted(exposed_set))
        anc_tuple = tuple(anc.closure_sha256 for anc in ancestors)

        payload = json.dumps(
            {
                "allowed_row_domain": str(allowed_row_domain),
                "ancestors": list(anc_tuple),
                "checkpoint_exposed_families": list(exp_tuple),
                "dataset_id": ds_str,
                "excluded_row_domains": list(excl_tuple),
                "extra_sha256": extra_sha,
                "family_id": fam_str,
                "hint_protocol_id": str(hint_protocol_id),
                "inference_tier": str(inference_tier),
                "outer_fold_id": fold_str,
                "schema": "jope.h462.fit_closure/v3",
                "sovereign_tier": str(sovereign_tier),
                "x_sha256": x_sha,
                "y_sha256": y_sha,
            },
            sort_keys=True,
        ).encode("utf-8")

        return cls(
            dataset_id=ds_str,
            family_id=fam_str,
            outer_fold_id=fold_str,
            allowed_row_domain=str(allowed_row_domain),
            excluded_row_domains=excl_tuple,
            sovereign_tier=str(sovereign_tier),
            inference_tier=str(inference_tier),
            hint_protocol_id=str(hint_protocol_id),
            x_sha256=x_sha,
            y_sha256=y_sha,
            extra_sha256=extra_sha,
            checkpoint_exposed_families=exp_tuple,
            ancestor_closures=anc_tuple,
            closure_sha256=hashlib.sha256(payload).hexdigest(),
        )
