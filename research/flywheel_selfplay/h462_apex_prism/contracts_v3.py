"""Transitive Leak-Free Provenance, Row-ID Closure, Exposure, and Sovereign `S0` Guards (`contracts_v3.py`).

Implements `HNC-1` (`HINT_NESTING_CONTRACT_HNC1_20261115T1411Z.md` + `Correction 1`, `cl_fwriter`):
1. **H5 Outer Nesting by Row-ID Sets (Finding 5):** Records canonical `train_row_ids`, `eval_row_ids`, and
   transitive `label_closure_row_ids` (plus SHA-256 digests) on every `FitClosureV3`. Refuses any fit or
   ancestor whose `label_closure_row_ids` intersects the consumer's `eval_row_ids` — regardless of
   `outer_fold_id` string labels (`"fold_3"`, `"full_search"`, etc.). A block fitted on all of `S` is valid
   ONLY when `eval_row_ids` is strictly in `A` (`label_closure(S) & eval(A) == empty`).
2. **H6 Mandatory Checkpoint Exposure (Finding 6):** `checkpoint_exposed_families` has **no silent empty
   default** — callers must pass it explicitly (populated via `load_bank_manifest_exposure` when a checkpoint
   is used, or `()` only when `checkpoint_bank_sha256 == "none"`). Also provides
   `count_exposed_families_on_board` so lock boards (`FITSEL`) record how many board families each entrant's
   checkpoints saw.
3. **H7 Protocol Identity Match (Finding 4 / H7):** Stack consumers refuse blocks with differing non-`"none"`
   `hint_protocol_id` values.
4. **Sovereign `S0` & Quarantine (`T14`):** Rejects external foundation-model columns/teachers in `S0` and
   refuses any path matching quarantined board tokens.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path, PurePath
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

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


def normalize_family_token(fam: str) -> str:
    """Normalize bank family tokens (stripping optional `'fresh:'` prefix)."""
    s = str(fam).strip()
    if s.startswith("fresh:"):
        return s[len("fresh:") :]
    return s


def canonicalize_row_ids(row_ids: Sequence[Union[int, str]]) -> Tuple[Tuple[str, ...], str]:
    """Return `(sorted_unique_row_ids, sha256_hex)` and verify no duplicate row IDs were passed."""
    str_ids = [str(r) for r in row_ids]
    uniq = sorted(set(str_ids))
    if len(uniq) != len(str_ids):
        raise ValueError(f"Duplicate row IDs passed to FitClosureV3 ({len(str_ids)} vs {len(uniq)} unique)")
    digest = hashlib.sha256(json.dumps(uniq).encode("utf-8")).hexdigest()
    return tuple(uniq), digest


def assert_path_not_quarantined(path_like: str | PurePath) -> None:
    """Raise `PermissionError` if `path_like` touches any sealed/confirm/holdout board (`T14`)."""
    norm = str(path_like).lower().replace("\\", "/")
    for tok in FORBIDDEN_PATH_TOKENS:
        if tok in norm:
            raise PermissionError(
                f"Quarantine refusal (T14): path {str(path_like)!r} matches forbidden token {tok!r}"
            )


def load_bank_manifest_exposure(manifest_path: str | PurePath) -> Tuple[str, Tuple[str, ...]]:
    """Load `(bank_sha256, normalized_exposed_families)` from a training-bank `.npz.json` manifest (`H6`)."""
    assert_path_not_quarantined(manifest_path)
    p = Path(manifest_path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    bank_sha = str(
        doc.get("bank_sha256")
        or doc.get("npz_sha256")
        or doc.get("sha256")
        or hashlib.sha256(p.read_bytes()).hexdigest()
    )
    fams: Set[str] = set()
    for key in ("exposed_families", "families", "family_keys"):
        if isinstance(doc.get(key), list):
            for v in doc[key]:
                if v:
                    fams.add(normalize_family_token(str(v)))
    for list_key in ("tables", "entries", "contexts", "cohorts"):
        val = doc.get(list_key)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, Mapping):
                    fk = item.get("family_key") or item.get("family") or item.get("family_id")
                    if fk:
                        fams.add(normalize_family_token(str(fk)))
        elif isinstance(val, Mapping):
            for item in val.values():
                if isinstance(item, Mapping):
                    fk = item.get("family_key") or item.get("family") or item.get("family_id")
                    if fk:
                        fams.add(normalize_family_token(str(fk)))
    if not fams:
        # Support `jope.h446.bank/v1` companion `.npz` manifests (`doc["file"]` or `<stem>.npz`) storing `family` array
        cand_npz = []
        if isinstance(doc.get("file"), str) and doc["file"].endswith(".npz"):
            cand_npz.append(Path(doc["file"]))
        if p.name.endswith(".npz.json"):
            cand_npz.append(p.with_name(p.name[: -len(".json")]))
        for npz_p in cand_npz:
            if npz_p.is_file():
                with np.load(npz_p, allow_pickle=False) as zb:
                    for fk_arr in ("family", "families", "family_key"):
                        if fk_arr in zb.files:
                            for v in zb[fk_arr].tolist():
                                if v:
                                    fams.add(normalize_family_token(str(v)))
                if fams:
                    break
    if not fams:
        raise ValueError(f"H6 violation: bank manifest {str(p)!r} yielded zero exposed families")
    return bank_sha, tuple(sorted(fams))


def count_exposed_families_on_board(
    board_families: Sequence[str],
    checkpoint_exposed_families: Sequence[str],
) -> int:
    """Count how many unique families on a lock board (`FITSEL`) appear in `checkpoint_exposed_families` (`H6`)."""
    exp = {normalize_family_token(f) for f in checkpoint_exposed_families if str(f).strip()}
    brd = {normalize_family_token(f) for f in board_families if str(f).strip()}
    return int(len(brd & exp))


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
    """Transitive content-addressed fit & row-ID label closure (`jope.h462.fit_closure/v3`, `HNC-1`)."""

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
    train_row_ids: Tuple[str, ...]
    eval_row_ids: Tuple[str, ...]
    label_closure_row_ids: Tuple[str, ...]
    train_row_ids_sha256: str
    eval_row_ids_sha256: str
    label_closure_row_ids_sha256: str
    checkpoint_bank_sha256: str
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
        *,
        train_row_ids: Sequence[Union[int, str]],
        eval_row_ids: Sequence[Union[int, str]],
        checkpoint_exposed_families: Sequence[str],
        checkpoint_bank_sha256: str = "none",
        outer_fold_id: str = "full_search",
        allowed_row_domain: str = "SEARCH_TRAIN",
        excluded_row_domains: Sequence[str] = ("AUDIT", "CONFIRM", "SEALED"),
        sovereign_tier: str = "S0",
        inference_tier: str = "medium",
        hint_protocol_id: str = "none",
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

        if len(train_row_ids) != x_arr.shape[0]:
            raise ValueError(
                f"H5 violation: train_row_ids length ({len(train_row_ids)}) != x_train rows ({x_arr.shape[0]})"
            )
        if len(eval_row_ids) == 0:
            raise ValueError("H5 violation: eval_row_ids must be non-empty")

        train_tuple, train_sha = canonicalize_row_ids(train_row_ids)
        eval_tuple, eval_sha = canonicalize_row_ids(eval_row_ids)
        train_set = set(train_tuple)
        eval_set = set(eval_tuple)
        direct_overlap = sorted(train_set & eval_set)
        if direct_overlap:
            raise PermissionError(
                f"H5 self-overlap violation: train_row_ids intersects eval_row_ids (e.g. {direct_overlap[:5]!r})"
            )

        bank_sha_str = str(checkpoint_bank_sha256).strip() or "none"
        exposed_set: Set[str] = {normalize_family_token(f) for f in checkpoint_exposed_families if str(f).strip()}
        if bank_sha_str != "none" and not exposed_set:
            raise ValueError(
                "H6 violation: checkpoint_exposed_families must be non-empty when checkpoint_bank_sha256 is set"
            )

        fam_str = normalize_family_token(family_id)
        ds_str = str(dataset_id)
        fold_str = str(outer_fold_id)
        closure_rows: Set[str] = set(train_set)
        non_none_protocols: List[str] = [str(hint_protocol_id)] if str(hint_protocol_id) != "none" else []

        for anc in ancestors:
            if anc.dataset_id != ds_str:
                raise ValueError(f"Ancestor dataset_id mismatch: {anc.dataset_id!r} != {ds_str!r}")
            anc_leak = sorted(set(anc.label_closure_row_ids) & eval_set)
            if anc_leak:
                raise PermissionError(
                    f"H5 outer-nesting violation: ancestor {anc.outer_fold_id!r} label closure intersects "
                    f"consumer {fold_str!r} eval_row_ids (e.g. {anc_leak[:5]!r})"
                )
            if anc.sovereign_tier != "S0" and sovereign_tier == "S0":
                raise PermissionError(
                    f"S0 closure cannot depend on non-S0 ancestor ({anc.sovereign_tier!r})"
                )
            if anc.hint_protocol_id != "none":
                non_none_protocols.append(anc.hint_protocol_id)
            closure_rows.update(anc.label_closure_row_ids)
            exposed_set.update(anc.checkpoint_exposed_families)

        if len(set(non_none_protocols)) > 1:
            raise PermissionError(
                f"H7 protocol mismatch across stack blocks: {sorted(set(non_none_protocols))!r}"
            )

        if fam_str and fam_str in exposed_set:
            raise PermissionError(
                f"H6 corpus exposure leak: target family {fam_str!r} is in checkpoint/ancestor "
                f"exposed families ({len(exposed_set)} families)"
            )

        closure_tuple = tuple(sorted(closure_rows))
        closure_rows_sha = hashlib.sha256(json.dumps(list(closure_tuple)).encode("utf-8")).hexdigest()

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
                "checkpoint_bank_sha256": bank_sha_str,
                "checkpoint_exposed_families": list(exp_tuple),
                "dataset_id": ds_str,
                "eval_row_ids_sha256": eval_sha,
                "excluded_row_domains": list(excl_tuple),
                "extra_sha256": extra_sha,
                "family_id": fam_str,
                "hint_protocol_id": str(hint_protocol_id),
                "inference_tier": str(inference_tier),
                "label_closure_row_ids_sha256": closure_rows_sha,
                "outer_fold_id": fold_str,
                "schema": "jope.h462.fit_closure/v3.1_hnc1",
                "sovereign_tier": str(sovereign_tier),
                "train_row_ids_sha256": train_sha,
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
            train_row_ids=train_tuple,
            eval_row_ids=eval_tuple,
            label_closure_row_ids=closure_tuple,
            train_row_ids_sha256=train_sha,
            eval_row_ids_sha256=eval_sha,
            label_closure_row_ids_sha256=closure_rows_sha,
            checkpoint_bank_sha256=bank_sha_str,
            checkpoint_exposed_families=exp_tuple,
            ancestor_closures=anc_tuple,
            closure_sha256=hashlib.sha256(payload).hexdigest(),
        )


def verify_nested_oof_fn(
    oof_fn,
    x_search: np.ndarray,
    y_search: np.ndarray,
    lab5: np.ndarray,
    *,
    poison_scale: float = 1000.0,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> Dict[str, object]:
    """`HNC-1` `H5` whole-procedure outer-fold label-poison verifier for `SEARCH-OOF` emitters.

    Perturbs `y_search[va]` on every outer fold `fk in unique(lab5)` by large asymmetric offsets
    (`+poison_scale * sd` plus a row-indexed ramp so rank, min, max, mean, and variance on `va` all shift)
    and verifies `oof_fn(x_search, y_poison, lab5)[va]` matches `oof_fn(x_search, y_search, lab5)[va]`
    within `(atol, rtol)`. Any program selection, clipping bound, or variance floor that peeks at `y_search[va]`
    raises `PermissionError`.
    """
    x_s = np.ascontiguousarray(x_search, dtype=np.float64)
    y_s = np.ascontiguousarray(y_search, dtype=np.float64).reshape(-1)
    lab = np.ascontiguousarray(lab5, dtype=np.int64).reshape(-1)
    if x_s.ndim != 2 or x_s.shape[0] != y_s.shape[0] or lab.shape[0] != y_s.shape[0]:
        raise ValueError(f"Shape mismatch in verify_nested_oof_fn: {x_s.shape}, {y_s.shape}, {lab.shape}")

    q_clean = np.asarray(oof_fn(x_s.copy(), y_s.copy(), lab.copy()), dtype=np.float64)
    if q_clean.shape[0] != y_s.shape[0]:
        raise ValueError(f"oof_fn output rows {q_clean.shape[0]} != n_search {y_s.shape[0]}")

    sd = max(float(np.std(y_s)), 1.0)
    folds = sorted(int(k) for k in np.unique(lab))
    max_abs_leak = 0.0
    max_rel_leak = 0.0

    for fk in folds:
        va = lab == fk
        n_va = int(np.sum(va))
        if n_va == 0:
            continue
        y_poison = y_s.copy()
        ramp = np.linspace(1.0, 2.5, n_va, dtype=np.float64)
        signs = np.where((np.arange(n_va) % 2) == 0, 1.0, -0.7)
        y_poison[va] = y_s[va] + float(poison_scale) * sd * ramp * signs
        q_poison = np.asarray(oof_fn(x_s.copy(), y_poison, lab.copy()), dtype=np.float64)

        diff = np.abs(q_poison[va] - q_clean[va])
        denom = np.maximum(np.abs(q_clean[va]), 1e-12)
        abs_err = float(np.max(diff))
        rel_err = float(np.max(diff / denom))
        max_abs_leak = max(max_abs_leak, abs_err)
        max_rel_leak = max(max_rel_leak, rel_err)

        tol_mat = float(atol) + float(rtol) * np.abs(q_clean[va])
        if bool(np.any(diff > tol_mat)):
            raise PermissionError(
                f"H5 nested-OOF violation on outer fold {fk}: validation labels leaked into q_oof[va] "
                f"(max_abs={abs_err:.3e}, max_rel={rel_err:.3e})"
            )

    return {
        "max_abs_leak": max_abs_leak,
        "max_rel_leak": max_rel_leak,
        "n_folds_checked": len(folds),
        "verified": True,
    }
