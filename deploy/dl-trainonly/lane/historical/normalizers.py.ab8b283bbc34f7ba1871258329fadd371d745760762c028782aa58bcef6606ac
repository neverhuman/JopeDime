"""PLAN §5 P6 / §6.4 and owner decision DL-D4a: per-block normalizers refit on train_fit contexts only, under new ABI
IDs (schema ``jope.dl-normalizer/v1``).

One normalizer is fitted per (arm, block) from the block vectors of the train_fit contexts:
1. Standardize each dim by its fit-set mean and population std; a dim whose std is <= 1e-12 * max(1, |mean|) is
   constant: scale 1, listed in dims.constant_dims, and its rotation row is exactly 0, so it contributes 0 to every
   output.
2. PCA of the Gram Z'Z/N (einsum, optimize=False) over the non-constant dims, eigenvalues in descending order. Each
   eigenvector's largest-|entry| is made positive (lowest index on ties). The rank is #{w > 1e-10 * w_max}; the kept
   width is width_full = min(rank, N - 1, max_components).
3. Projections are standardized by their fit-set mean and std (std floored at 1e-12; the number floored is recorded)
   and clipped to +-32.

The same rotation serves every width W: apply computes all width_full components and returns the first W, so
apply(W=k) == apply(FULL)[:k] bitwise. No padding (W > width_full) and no truncation below 1 component.

Eigendecomposition. numpy.linalg.eigh (OpenBLAS LAPACK) returned different bytes for OMP/OPENBLAS thread counts 1, 2,
4 and 8 on the same Gram matrix (probe on 2026-10-05, numpy 2.5.3, scipy-openblas 0.3.34), which breaks the spec's
thread-independence test T20. As that test prescribes, eigh is replaced by a pure-numpy Jacobi method (parallel
round-robin ordering, elementwise updates only): its bytes depend only on the input and the numpy build. The header's
method.eigh names it.

Determinism. All arithmetic is float64; rows are ordered by context_sha256 before any arithmetic, so the result does
not depend on the caller's row order. Arrays are stored as '<f4' in one blob, in this order: mean[d], scale[d],
rotation[d, r] row-major, proj_mean[r], proj_scale[r], eigenvalues[r]. The JSON header is common.canonical with no
timestamps or host data; normalizer_sha256 = sha256(header bytes). Files <normalizer_sha256>.json and
<blob_sha256>.f32le are create-only (mode 0o444).

Fit rule (DL-D4a): only rows whose own split_role and whose partition-map split_role are both train_fit are admitted;
every refusal is raised before any arithmetic. No S3 access: this module imports only numpy and dltrain.common.

Partition-map provenance. fit.partition_map_kind 'p3' is a positive property: it needs the P3 holdout binding sha256
and the sha256 of the partition-map binding record (dltrain.encodings.bind_partition_map: the map's roles recomputed
from the bound holdout.json, plus PASS P3/P5 receipts binding this map). A 'fake' map carries neither, so a fake map's
normalizer can never claim a P3 binding.
"""
import dataclasses
import functools
import json
import math
import re
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np

from dltrain import common

SCHEMA = "jope.dl-normalizer/v1"
EXPERIMENT_ID = "gp-policy-dl-v1-e0"
ABI_PREFIX = "dl-v1-e0"
CLIP_ABS = 32.0
RANK_RTOL = 1e-10
MIN_BLOCK_WIDTH = 16
CONST_RTOL = 1e-12
PROJ_SCALE_FLOOR = 1e-12
PARTITIONS = ("train_fit", "dev", "diag", "heldout")          # as in dltrain.programs
EXCLUDED_ROLES = ("sealed", "prohibited")                     # as in dltrain.programs
FIT_ROLE = "train_fit"
TRAIN_FAMILY_ROLES = ("train_fit", "diag")                   # diag is a context-level carve-out of train_fit families
PARTITION_MAP_KINDS = ("p3", "fake")
FIT_CONTEXT_SET_SCHEMA = "jope.dl-fit-context-set/v1"
FIT_FAMILY_SET_SCHEMA = "jope.dl-fit-family-set/v1"
DTYPE_STORED = "<f4"
ARRAY_ORDER = ("mean", "scale", "rotation", "proj_mean", "proj_scale", "eigenvalues")
JACOBI_METHOD = "dltrain.normalizers.jacobi_eigh/v1 (parallel round-robin Jacobi, pure numpy; replaces numpy.linalg.eigh, " \
                "which is BLAS-thread-count dependent)"
JACOBI_REL_TOL = 1e-15
JACOBI_ABS_TOL = 1e-14
JACOBI_MAX_SWEEPS = 100
METHOD = {
    "standardize": "mean and std (ddof=0) per dim over the fit rows; scale=std if std > 1e-12*max(1,|mean|) else 1 "
                   "(constant dim: zero rotation row)",
    "pca": "eigendecomposition of Z'Z/N over the non-constant dims; descending eigenvalues; largest-|entry| of each "
           "eigenvector positive (lowest index on ties); rank=#{w > rank_rtol*w_max}; width_full=min(rank, N-1, "
           "max_components)",
    "projection_scale": "mean and std (ddof=0) of the fit-row projections; std floored at 1e-12",
    "clip_abs": CLIP_ABS,
    "rank_rtol": RANK_RTOL,
    "dtype_compute": "float64",
    "dtype_stored": DTYPE_STORED,
    "gram": "einsum(optimize=False)",
    "eigh": JACOBI_METHOD,
}

# The arm table (PLAN §6.4). Block order fixes allocate_widths tie-breaking and the abi_id preimage.
BLOCKS = {
    "m": {"in": 88, "max_components": None, "source": "A0 meta-features (dltrain.metafeatures)"},
    "hyperion": {"in": 512, "max_components": None, "source": "Hyperion v5 arm B signature (adopted)"},
    "foundation": {"in": 2048, "max_components": 1024, "source": "Starlight Foundation raw 2048 (adopted)"},
    "tabpfn35": {"in": 2048, "max_components": None, "source": "TabPFN 3.5 embedding; width w3 declared later"},
    "target": {"in": 512, "max_components": None, "source": "fused target encoder (LB-replica only)"},
}
ARMS = {
    "A0": ("m",),
    "A1": ("hyperion", "m"),
    "A2": ("hyperion", "foundation", "m"),
    "A3": ("hyperion", "foundation", "m", "tabpfn35"),
    "LB-replica": ("hyperion", "foundation", "target"),
}
RESEARCH_ONLY_ARMS = ("A2", "A3", "LB-replica")
DEV_ONLY_ARMS = ("LB-replica",)                               # never holdout-reported

_HEX64 = re.compile(r"[0-9a-f]{64}")


class NormalizerError(ValueError):
    """A normalizer cannot be fitted, decoded, verified or applied as asked."""


class NormalizerFitRefused(NormalizerError):
    """The fit set breaks the train_fit rule (or another admission rule); raised before any arithmetic."""

    def __init__(self, reason: str, message: str, *, context_sha256: str | None = None, split_role: str | None = None):
        super().__init__(f"fit refused ({reason}): {message}")
        self.reason = reason
        self.context_sha256 = context_sha256
        self.split_role = split_role


class FitRow(NamedTuple):
    context_sha256: str
    family_id: str
    split_role: str
    vector: object


def _is_hex64(v) -> bool:
    return isinstance(v, str) and _HEX64.fullmatch(v) is not None


def fit_context_set_sha256(contexts) -> str:
    return common.sha256_bytes(common.canonical({"schema": FIT_CONTEXT_SET_SCHEMA, "contexts": sorted(contexts)}))


def fit_family_set_sha256(families) -> str:
    return common.sha256_bytes(common.canonical({"schema": FIT_FAMILY_SET_SCHEMA, "families": sorted(set(families))}))


# ----------------------------------------------------------------------------------------------- Normalizer
@dataclasses.dataclass(frozen=True, eq=False)
class Normalizer:
    """A verified normalizer: its canonical header bytes, its blob and the decoded read-only '<f4' arrays."""

    header: dict
    header_bytes: bytes
    blob: bytes
    arrays: dict

    def __post_init__(self):
        object.__setattr__(self, "_f64", {k: np.asarray(v, dtype=np.float64) for k, v in self.arrays.items()})

    def __eq__(self, other):
        return isinstance(other, Normalizer) and self.header_bytes == other.header_bytes and self.blob == other.blob

    def __hash__(self):
        return hash(self.header_bytes)

    @property
    def sha256(self) -> str:
        return common.sha256_bytes(self.header_bytes)

    @property
    def blob_sha256(self) -> str:
        return self.header["blob"]["sha256"]

    @property
    def block_id(self) -> str:
        return self.header["block_id"]

    @property
    def arm(self) -> str:
        return self.header["arm"]

    @property
    def d(self) -> int:
        return int(self.header["dims"]["in"])

    @property
    def rank(self) -> int:
        return int(self.header["dims"]["rank"])

    @property
    def width_full(self) -> int:
        return int(self.header["dims"]["width_full"])

    @property
    def constant_dims(self) -> list[int]:
        return list(self.header["dims"]["constant_dims"])

    def manifest_entry(self) -> dict:
        """The jope.dl-context-encoding/v1 'normalizers' entry of this block."""
        fit = self.header["fit"]
        return {"block_id": self.block_id, "normalizer_sha256": self.sha256, "blob_sha256": self.blob_sha256,
                "fit_family_set_sha256": fit["fit_family_set_sha256"],
                "fit_context_set_sha256": fit["fit_context_set_sha256"], "rank": self.rank,
                "width_full": self.width_full}


# ----------------------------------------------------------------------------------------------- Jacobi eigh
@functools.lru_cache(maxsize=16)
def _jacobi_schedule(d: int) -> tuple:
    """Round-robin (circle method) schedule: m-1 rounds of disjoint pairs covering every pair (p < q) exactly once,
    with m = d rounded up to even (a dummy index d is skipped for odd d)."""
    m = d + (d % 2)
    others = list(range(1, m))
    rounds = []
    for r in range(m - 1):
        ring = [0] + others[r:] + others[:r]
        ps, qs = [], []
        for i in range(m // 2):
            a, b = ring[i], ring[m - 1 - i]
            if a < d and b < d:
                ps.append(min(a, b))
                qs.append(max(a, b))
        p_arr, q_arr = np.array(ps, dtype=np.intp), np.array(qs, dtype=np.intp)
        p_arr.setflags(write=False)
        q_arr.setflags(write=False)
        rounds.append((p_arr, q_arr))
    return tuple(rounds)


def jacobi_eigh(g):
    """Eigenvalues and eigenvectors (columns) of a real symmetric matrix by the Jacobi method, using only elementwise
    numpy operations (no BLAS/LAPACK), so the bytes do not depend on thread counts. Rotations in one round act on
    disjoint index pairs and are applied together; a pair is skipped when |a_pq| <= max(1e-14*||A||_F,
    1e-15*sqrt(|a_pp*a_qq|)). Iteration stops after a sweep with no rotation. Order: as on the diagonal (unsorted)."""
    a = np.array(g, dtype=np.float64, order="C", copy=True)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise NormalizerError(f"jacobi_eigh needs a square matrix, got {a.shape}")
    d = a.shape[0]
    v = np.eye(d)
    if d <= 1:
        return a.diagonal().copy(), v
    fro = math.sqrt(float(np.sum(a * a)))
    if fro == 0.0:
        return np.zeros(d), v
    abs_tol = JACOBI_ABS_TOL * fro
    upper = np.triu_indices(d, 1)
    lower = (upper[1], upper[0])
    for _ in range(JACOBI_MAX_SWEEPS):
        rotations = 0
        for ps, qs in _jacobi_schedule(d):
            if ps.size == 0:
                continue
            apq = a[ps, qs]
            app = a[ps, ps]
            aqq = a[qs, qs]
            sel = np.abs(apq) > np.maximum(abs_tol, JACOBI_REL_TOL * np.sqrt(np.abs(app * aqq)))
            if not bool(sel.any()):
                continue
            p1, q1, apq, app, aqq = ps[sel], qs[sel], apq[sel], app[sel], aqq[sel]
            theta = (aqq - app) / (2.0 * apq)
            t = np.where(theta >= 0.0, 1.0, -1.0) / (np.abs(theta) + np.hypot(theta, 1.0))
            c = 1.0 / np.hypot(t, 1.0)
            s = t * c
            colp, colq = a[:, p1], a[:, q1]
            a[:, p1] = colp * c - colq * s
            a[:, q1] = colp * s + colq * c
            rowp, rowq = a[p1, :], a[q1, :]
            a[p1, :] = rowp * c[:, None] - rowq * s[:, None]
            a[q1, :] = rowp * s[:, None] + rowq * c[:, None]
            a[p1, p1] = app - t * apq
            a[q1, q1] = aqq + t * apq
            a[p1, q1] = 0.0
            a[q1, p1] = 0.0
            vp, vq = v[:, p1], v[:, q1]
            v[:, p1] = vp * c - vq * s
            v[:, q1] = vp * s + vq * c
            rotations += int(p1.size)
        a[lower] = a[upper]
        if rotations == 0:
            return a.diagonal().copy(), v
    raise NormalizerError(f"Jacobi eigendecomposition did not converge in {JACOBI_MAX_SWEEPS} sweeps (d={d})")


# ----------------------------------------------------------------------------------------------- fit
def _check_args(block_id, arm, block_version, feature_list_sha256, epoch_sha256, partition_map_sha256,
                partition_map_kind, holdout_binding_sha256, max_components, partition_map_binding_sha256=None):
    for name, val in (("block_id", block_id), ("arm", arm), ("block_version", block_version)):
        if not isinstance(val, str) or not val:
            raise NormalizerError(f"{name} must be a non-empty string")
    for name, val, optional in (("feature_list_sha256", feature_list_sha256, True), ("epoch_sha256", epoch_sha256, False),
                                ("partition_map_sha256", partition_map_sha256, False),
                                ("holdout_binding_sha256", holdout_binding_sha256, True),
                                ("partition_map_binding_sha256", partition_map_binding_sha256, True)):
        if not (val is None and optional) and not _is_hex64(val):
            raise NormalizerError(f"{name} must be a 64-hex sha256{' or None' if optional else ''}")
    if partition_map_kind not in PARTITION_MAP_KINDS:
        raise NormalizerError(f"partition_map_kind must be one of {PARTITION_MAP_KINDS}")
    _check_map_provenance(partition_map_kind, holdout_binding_sha256, partition_map_binding_sha256)
    if max_components is not None and (isinstance(max_components, bool) or not isinstance(max_components, int)
                                       or max_components < 1):
        raise NormalizerError("max_components must be None or an int >= 1")


def _check_map_provenance(kind, holdout_binding_sha256, partition_map_binding_sha256):
    """p3 needs both binding hashes (bound, not merely 'not fake-looking'); a fake map carries neither."""
    if kind == "p3" and not (_is_hex64(holdout_binding_sha256) and _is_hex64(partition_map_binding_sha256)):
        raise NormalizerError("partition_map_kind 'p3' needs holdout_binding_sha256 and partition_map_binding_sha256 "
                              "(a bound P3 partition map)")
    if kind == "fake" and (holdout_binding_sha256 is not None or partition_map_binding_sha256 is not None):
        raise NormalizerError("a fake partition map carries no holdout or partition-map binding")


def _admit(rows, partition_map):
    """Every refusal, before any arithmetic: empty set, a row role or a map role other than train_fit, an unmapped
    context, a family that differs from the map, a duplicate, ragged dims, a non-finite value, N < 2.
    Returns (rows ordered by context_sha256, float64 matrix in that order)."""
    if not isinstance(partition_map, Mapping):
        raise NormalizerError("partition_map must be a mapping context_sha256 -> {family_id, split_role}")
    rows = list(rows)
    for i, r in enumerate(rows):
        if not isinstance(r, FitRow):
            if not isinstance(r, (tuple, list)) or len(r) != len(FitRow._fields):
                raise NormalizerError(f"fit row {i} is not a FitRow{FitRow._fields}")
            rows[i] = FitRow(*r)
    if not rows:
        raise NormalizerFitRefused("empty", "no fit rows")
    seen = set()
    for r in rows:
        ctx = r.context_sha256
        if r.split_role != FIT_ROLE:
            raise NormalizerFitRefused("role", f"context {ctx} has split_role {r.split_role!r}, not {FIT_ROLE!r}",
                                       context_sha256=ctx, split_role=r.split_role)
        entry = partition_map.get(ctx) if isinstance(ctx, str) else None
        if not isinstance(entry, Mapping):
            raise NormalizerFitRefused("unmapped", f"context {ctx} has split_role 'unmapped' (not in the partition map)",
                                       context_sha256=ctx, split_role="unmapped")
        role = entry.get("split_role")
        if role != FIT_ROLE:
            raise NormalizerFitRefused("map_role", f"context {ctx} has split_role {role!r} in the partition map, "
                                       f"not {FIT_ROLE!r}", context_sha256=ctx, split_role=role)
        if entry.get("family_id") != r.family_id:
            raise NormalizerFitRefused("family_mismatch", f"context {ctx} names family {r.family_id!r} but the partition "
                                       f"map has {entry.get('family_id')!r} (split_role {role!r})",
                                       context_sha256=ctx, split_role=role)
        if ctx in seen:
            raise NormalizerFitRefused("duplicate", f"context {ctx} appears twice", context_sha256=ctx, split_role=role)
        seen.add(ctx)
    rows.sort(key=lambda r: r.context_sha256)
    vectors, dims = [], set()
    for r in rows:
        try:
            vec = np.asarray(r.vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise NormalizerFitRefused("ragged", f"context {r.context_sha256} has a non-numeric vector: {exc}",
                                       context_sha256=r.context_sha256) from None
        if vec.ndim != 1 or vec.size == 0:
            raise NormalizerFitRefused("ragged", f"context {r.context_sha256} vector has shape {vec.shape}",
                                       context_sha256=r.context_sha256)
        dims.add(vec.size)
        vectors.append(vec)
    if len(dims) != 1:
        raise NormalizerFitRefused("ragged", f"vectors have different dims {sorted(dims)}")
    x = np.stack(vectors)
    finite = np.isfinite(x)
    if not bool(finite.all()):
        bad = rows[int(np.flatnonzero(~finite.all(axis=1))[0])].context_sha256
        raise NormalizerFitRefused("nonfinite", f"context {bad} has a non-finite value", context_sha256=bad)
    if len(rows) < 2:
        raise NormalizerFitRefused("too_few", f"N={len(rows)} < 2 fit rows")
    return rows, x


def _f32(a) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(a, dtype=np.float64).astype(DTYPE_STORED))
    out[out == 0] = 0.0                              # -0.0 -> +0.0, so the stored bytes have one zero
    return out


def _compute(x, max_components):
    """float64 fit of an admitted matrix (rows ordered by context). Returns the stored arrays and the dims."""
    n, d = x.shape
    mean = x.sum(axis=0) / n
    dev = x - mean
    std = np.sqrt((dev * dev).sum(axis=0) / n)
    const = ~(std > CONST_RTOL * np.maximum(1.0, np.abs(mean)))
    scale = np.where(const, 1.0, std)
    z = dev / scale
    z[:, const] = 0.0
    nc = np.flatnonzero(~const)
    if nc.size == 0:
        raise NormalizerError("rank 0: every dim is constant on the fit set")
    znc = np.ascontiguousarray(z[:, nc])
    gram = np.einsum("ni,nj->ij", znc, znc, optimize=False) / n
    w, vecs = jacobi_eigh(gram)
    order = np.argsort(-w, kind="stable")
    w, vecs = w[order], vecs[:, order]
    lead = np.argmax(np.abs(vecs), axis=0)
    vecs = vecs * np.where(vecs[lead, np.arange(vecs.shape[1])] < 0.0, -1.0, 1.0)[None, :]
    w_max = float(w[0])
    rank = int(np.count_nonzero(w > RANK_RTOL * w_max)) if w_max > 0.0 else 0
    width = min(rank, n - 1, max_components if max_components is not None else rank)
    if width < 1:
        raise NormalizerError(f"rank {rank} leaves no component (N={n}, max_components={max_components})")
    rotation = np.zeros((d, width), dtype=np.float64)
    rotation[nc, :] = vecs[:, :width]
    proj = np.einsum("ni,ik->nk", z, rotation, optimize=False)
    proj_mean = proj.sum(axis=0) / n
    pdev = proj - proj_mean
    proj_std = np.sqrt((pdev * pdev).sum(axis=0) / n)
    floored = int(np.count_nonzero(proj_std < PROJ_SCALE_FLOOR))
    proj_scale = np.maximum(proj_std, PROJ_SCALE_FLOOR)
    arrays = {"mean": _f32(mean), "scale": _f32(scale), "rotation": _f32(rotation), "proj_mean": _f32(proj_mean),
              "proj_scale": _f32(proj_scale), "eigenvalues": _f32(w[:width])}
    dims = {"in": int(d), "rank": rank, "width_full": int(width), "constant_dims": [int(i) for i in np.flatnonzero(const)],
            "floored_components": floored}
    return arrays, dims


def _array_table(arrays):
    table, parts, offset = [], [], 0
    for name in ARRAY_ORDER:
        data = arrays[name].tobytes()
        table.append({"name": name, "dtype": DTYPE_STORED, "shape": list(arrays[name].shape), "offset": offset,
                      "bytes": len(data)})
        parts.append(data)
        offset += len(data)
    return table, b"".join(parts)


def fit_block(block_id, rows, *, partition_map, arm, block_version, feature_list_sha256=None, epoch_sha256,
              partition_map_sha256, partition_map_kind, holdout_binding_sha256=None, max_components=None,
              partition_map_binding_sha256=None):
    """Fit one block's normalizer on train_fit rows only (DL-D4a). Refusals come before any arithmetic."""
    _check_args(block_id, arm, block_version, feature_list_sha256, epoch_sha256, partition_map_sha256,
                partition_map_kind, holdout_binding_sha256, max_components, partition_map_binding_sha256)
    ordered, x = _admit(rows, partition_map)
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
            arrays, dims = _compute(x, max_components)
    except FloatingPointError as exc:
        raise NormalizerError(f"normalizer arithmetic failed: {exc}") from exc
    for name in ARRAY_ORDER:
        if not bool(np.all(np.isfinite(arrays[name]))):
            raise NormalizerError(f"{name} is not finite in {DTYPE_STORED}")
    contexts = [r.context_sha256 for r in ordered]
    families = sorted({r.family_id for r in ordered})
    table, blob = _array_table(arrays)
    header = {
        "schema": SCHEMA, "experiment_id": EXPERIMENT_ID, "arm": arm, "block_id": block_id,
        "block_version": block_version, "feature_list_sha256": feature_list_sha256,
        "method": dict(METHOD),
        "dims": dims,
        "fit": {"partition_role": FIT_ROLE, "n_contexts": len(contexts), "n_families": len(families),
                "fit_context_set_sha256": fit_context_set_sha256(contexts),
                "fit_family_set_sha256": fit_family_set_sha256(families),
                "partition_map_sha256": partition_map_sha256, "partition_map_kind": partition_map_kind,
                "partition_map_binding_sha256": partition_map_binding_sha256,
                "epoch_sha256": epoch_sha256, "holdout_binding_sha256": holdout_binding_sha256},
        "arrays": table,
        "blob": {"sha256": common.sha256_bytes(blob), "bytes": len(blob)},
    }
    return from_bytes(common.canonical(header), blob)


# ----------------------------------------------------------------------------------------------- bytes
def to_bytes(n: Normalizer) -> tuple[bytes, bytes]:
    return n.header_bytes, n.blob


def _expect(cond, message):
    if not cond:
        raise NormalizerError(message)


def _int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def from_bytes(json_bytes: bytes, blob: bytes, *, normalizer_sha256: str | None = None) -> Normalizer:
    """Decode and verify: canonical header, schema, header hash (when expected), blob sha256/bytes, the array table
    and the arrays' contracts. Any mismatch raises NormalizerError."""
    json_bytes, blob = bytes(json_bytes), bytes(blob)
    if normalizer_sha256 is not None:
        _expect(common.sha256_bytes(json_bytes) == normalizer_sha256,
                f"header sha256 {common.sha256_bytes(json_bytes)} != expected {normalizer_sha256}")
    try:
        header = json.loads(json_bytes)
        canonical = common.canonical(header)
    except (ValueError, TypeError) as exc:
        raise NormalizerError(f"header is not canonical JSON: {exc}") from None
    _expect(isinstance(header, dict) and canonical == json_bytes, "header bytes are not canonical JSON")
    _expect(header.get("schema") == SCHEMA, f"schema {header.get('schema')!r} != {SCHEMA}")
    keys = {"schema", "experiment_id", "arm", "block_id", "block_version", "feature_list_sha256", "method", "dims",
            "fit", "arrays", "blob"}
    _expect(set(header) == keys, f"header keys {sorted(header)} != {sorted(keys)}")
    _expect(header["experiment_id"] == EXPERIMENT_ID, "experiment_id differs")
    for k in ("arm", "block_id", "block_version"):
        _expect(isinstance(header[k], str) and header[k], f"{k} must be a non-empty string")
    _expect(header["feature_list_sha256"] is None or _is_hex64(header["feature_list_sha256"]), "feature_list_sha256")
    _expect(header["method"] == METHOD, "method differs from this implementation's")
    dims, fit, b = header["dims"], header["fit"], header["blob"]
    _expect(isinstance(dims, dict) and set(dims) == {"in", "rank", "width_full", "constant_dims", "floored_components"},
            "dims keys")
    d, r = dims["in"], dims["width_full"]
    _expect(_int(d) and d >= 1 and _int(r) and r >= 1 and _int(dims["rank"]) and r <= dims["rank"] <= d
            and _int(dims["floored_components"]) and 0 <= dims["floored_components"] <= r, "dims values")
    const = dims["constant_dims"]
    _expect(isinstance(const, list) and all(_int(i) and 0 <= i < d for i in const) and const == sorted(set(const))
            and len(const) < d, "constant_dims")
    fit_keys = {"partition_role", "n_contexts", "n_families", "fit_context_set_sha256", "fit_family_set_sha256",
                "partition_map_sha256", "partition_map_kind", "partition_map_binding_sha256", "epoch_sha256",
                "holdout_binding_sha256"}
    _expect(isinstance(fit, dict) and set(fit) == fit_keys, "fit keys")
    _expect(fit["partition_role"] == FIT_ROLE, "fit.partition_role must be train_fit")
    _expect(_int(fit["n_contexts"]) and fit["n_contexts"] >= 2 and r <= fit["n_contexts"] - 1
            and _int(fit["n_families"]) and 1 <= fit["n_families"] <= fit["n_contexts"], "fit counts")
    for k in ("fit_context_set_sha256", "fit_family_set_sha256", "partition_map_sha256", "epoch_sha256"):
        _expect(_is_hex64(fit[k]), f"fit.{k}")
    for k in ("holdout_binding_sha256", "partition_map_binding_sha256"):
        _expect(fit[k] is None or _is_hex64(fit[k]), f"fit.{k}")
    _expect(fit["partition_map_kind"] in PARTITION_MAP_KINDS, "fit.partition_map_kind")
    try:
        _check_map_provenance(fit["partition_map_kind"], fit["holdout_binding_sha256"], fit["partition_map_binding_sha256"])
    except NormalizerError as exc:
        raise NormalizerError(f"fit.partition_map_kind: {exc}") from None
    _expect(isinstance(b, dict) and set(b) == {"sha256", "bytes"}, "blob keys")
    _expect(b["bytes"] == len(blob), f"blob has {len(blob)} bytes, header says {b['bytes']}")
    _expect(b["sha256"] == common.sha256_bytes(blob), "blob sha256 differs from the header")
    shapes = {"mean": [d], "scale": [d], "rotation": [d, r], "proj_mean": [r], "proj_scale": [r], "eigenvalues": [r]}
    table = header["arrays"]
    _expect(isinstance(table, list) and [t.get("name") if isinstance(t, dict) else None for t in table]
            == list(ARRAY_ORDER), "array table order")
    arrays, offset = {}, 0
    for t in table:
        name = t["name"]
        _expect(set(t) == {"name", "dtype", "shape", "offset", "bytes"} and t["dtype"] == DTYPE_STORED
                and t["shape"] == shapes[name] and t["offset"] == offset and t["bytes"] == 4 * math.prod(shapes[name]),
                f"array table entry {name}")
        a = np.frombuffer(blob, dtype=DTYPE_STORED, count=math.prod(shapes[name]), offset=offset).reshape(shapes[name])
        arrays[name] = a
        offset += t["bytes"]
    _expect(offset == len(blob), "array table does not cover the blob")
    for name, a in arrays.items():
        _expect(bool(np.all(np.isfinite(a))), f"{name} has a non-finite value")
    _expect(bool(np.all(arrays["scale"] > 0)) and bool(np.all(arrays["proj_scale"] > 0)), "scales must be > 0")
    if const:
        _expect(bool(np.all(arrays["scale"][const] == 1.0)) and not bool(np.any(arrays["rotation"][const, :])),
                "constant dims need scale 1 and a zero rotation row")
    return Normalizer(header=header, header_bytes=json_bytes, blob=blob, arrays=arrays)


def write(n: Normalizer, out_dir) -> dict:
    """Create-only, 0o444: <blob_sha256>.f32le, then the header <normalizer_sha256>.json that references it."""
    out = Path(out_dir)
    hb = common.immutable_write(out / f"{n.blob_sha256}.f32le", n.blob)
    hj = common.immutable_write(out / f"{n.sha256}.json", n.header_bytes)
    return {"normalizer_sha256": n.sha256, "header": hj, "blob": hb}


def load(out_dir, normalizer_sha256: str) -> Normalizer:
    out = Path(out_dir)
    json_bytes = (out / f"{normalizer_sha256}.json").read_bytes()
    try:
        blob_sha = json.loads(json_bytes)["blob"]["sha256"]
    except (ValueError, KeyError, TypeError) as exc:
        raise NormalizerError(f"unreadable normalizer header {normalizer_sha256}: {exc}") from None
    _expect(_is_hex64(blob_sha), "blob sha256")
    return from_bytes(json_bytes, (out / f"{blob_sha}.f32le").read_bytes(), normalizer_sha256=normalizer_sha256)


# ----------------------------------------------------------------------------------------------- apply
def _width(n: Normalizer, width) -> int:
    w = n.width_full if width is None else width
    if isinstance(w, bool) or not isinstance(w, (int, np.integer)):
        raise NormalizerError(f"width must be an int, got {width!r}")
    w = int(w)
    if w < 1 or w > n.width_full:
        raise NormalizerError(f"width {w} outside [1, width_full={n.width_full}] (no padding, no truncation below 1)")
    return w


def apply_with_stats(n: Normalizer, x, width=None):
    """(float32[W] output, clip count). All width_full components are computed with the same rotation and the first W
    returned, so apply(W=k) == apply(FULL)[:k]."""
    w = _width(n, width)
    xv = np.asarray(x, dtype=np.float64)
    if xv.shape != (n.d,):
        raise NormalizerError(f"x has shape {xv.shape}, expected ({n.d},)")
    if not bool(np.all(np.isfinite(xv))):
        raise NormalizerError("x has a non-finite value")
    f = n._f64
    try:
        with np.errstate(divide="raise", over="raise", invalid="raise", under="ignore"):
            z = (xv - f["mean"]) / f["scale"]
            full = (np.einsum("i,ik->k", z, f["rotation"], optimize=False) - f["proj_mean"]) / f["proj_scale"]
    except FloatingPointError as exc:
        raise NormalizerError(f"x is outside the float64 range of this normalizer: {exc}") from None
    clipped = np.abs(full) > CLIP_ABS
    out = np.clip(full, -CLIP_ABS, CLIP_ABS)[:w].astype(np.float32)
    return out, int(np.count_nonzero(clipped[:w]))


def apply(n: Normalizer, x, width=None):
    return apply_with_stats(n, x, width)[0]


# ----------------------------------------------------------------------------------------------- arms, widths, ABI
def allocate_widths(W, blocks) -> dict:
    """Split W over [(block_id, width_full_b)] proportionally to width_full_b, at least 16 per block: w_b =
    min(width_full_b, max(16, floor(W*d_b/D))), then the sum is fixed by largest remainder (+1 to the largest
    W*d_b/D - w_b among blocks below width_full_b; -1 from the smallest among blocks above 16; ties by block order)."""
    blocks = [(b, int(d) if isinstance(d, np.integer) else d) for b, d in blocks]
    if isinstance(W, np.integer):
        W = int(W)
    if not blocks:
        raise NormalizerError("allocate_widths needs at least one block")
    ids = [b for b, _ in blocks]
    if len(set(ids)) != len(ids):
        raise NormalizerError(f"duplicate block ids {ids}")
    for b, d in blocks:
        if not _int(d) or d < MIN_BLOCK_WIDTH:
            raise NormalizerError(f"block {b!r} width_full {d!r} must be an int >= {MIN_BLOCK_WIDTH}")
    if not _int(W):
        raise NormalizerError(f"W must be an int, got {W!r}")
    full = sum(d for _, d in blocks)
    if W > full:
        raise NormalizerError(f"W={W} exceeds FULL={full} (no padding)")
    if W < MIN_BLOCK_WIDTH * len(blocks):
        raise NormalizerError(f"W={W} < {MIN_BLOCK_WIDTH} x {len(blocks)} blocks")
    exact = [Fraction(W * d, full) for _, d in blocks]
    w = [min(d, max(MIN_BLOCK_WIDTH, math.floor(e))) for (_, d), e in zip(blocks, exact)]
    diff = W - sum(w)
    while diff > 0:
        i = max((i for i in range(len(w)) if w[i] < blocks[i][1]), key=lambda i: (exact[i] - w[i], -i))
        w[i] += 1
        diff -= 1
    while diff < 0:
        i = min((i for i in range(len(w)) if w[i] > MIN_BLOCK_WIDTH), key=lambda i: (exact[i] - w[i], i))
        w[i] -= 1
        diff += 1
    return {b: wi for (b, _), wi in zip(blocks, w)}


def abi_id(arm: str, normalizers, widths) -> str:
    """'dl-v1-e0:<arm>:' + sha256(canonical({arm, normalizers:[sha], widths}))[:16]."""
    shas = [x.sha256 if isinstance(x, Normalizer) else x for x in normalizers]
    if not isinstance(arm, str) or not arm or not all(_is_hex64(s) for s in shas):
        raise NormalizerError("abi_id needs an arm and 64-hex normalizer sha256 values")
    wd = dict(widths)
    if not all(isinstance(k, str) and _int(v) and v >= 1 for k, v in wd.items()):
        raise NormalizerError("widths must map block ids to ints >= 1")
    digest = common.sha256_bytes(common.canonical({"arm": arm, "normalizers": shas, "widths": wd}))
    return f"{ABI_PREFIX}:{arm}:{digest[:16]}"


def arm_blocks(arm: str) -> tuple:
    if arm not in ARMS:
        raise NormalizerError(f"unknown arm {arm!r}; arms are {sorted(ARMS)}")
    return ARMS[arm]


# ----------------------------------------------------------------------------------------------- closure
def assert_fit_closure(n: Normalizer, partition_map, present_train_fit) -> dict:
    """The normalizer's fit set must equal the present train_fit contexts, each mapped to train_fit, and every fit
    family must be a train_fit family (all its mapped contexts train_fit or diag). The recorded fit-set hashes and
    counts are recomputed and compared. Raises NormalizerFitRefused on any violation."""
    present = list(present_train_fit)
    if len(set(present)) != len(present):
        raise NormalizerFitRefused("duplicate", "present_train_fit lists a context twice")
    for ctx in present:
        entry = partition_map.get(ctx)
        if not isinstance(entry, Mapping):
            raise NormalizerFitRefused("unmapped", f"context {ctx} is not in the partition map", context_sha256=ctx,
                                       split_role="unmapped")
        if entry.get("split_role") != FIT_ROLE:
            raise NormalizerFitRefused("map_role", f"context {ctx} has split_role {entry.get('split_role')!r}",
                                       context_sha256=ctx, split_role=entry.get("split_role"))
    families = sorted({partition_map[c]["family_id"] for c in present})
    fam_set = set(families)
    roles: dict[str, set] = {f: set() for f in families}
    for ctx, entry in partition_map.items():
        if isinstance(entry, Mapping) and entry.get("family_id") in fam_set:
            roles[entry["family_id"]].add(entry.get("split_role"))
    for f in families:
        bad = sorted(str(x) for x in roles[f] - set(TRAIN_FAMILY_ROLES))
        if bad:
            raise NormalizerFitRefused("family_not_train_fit", f"fit family {f} also has contexts with roles {bad}")
    fit = n.header["fit"]
    ctx_sha, fam_sha = fit_context_set_sha256(present), fit_family_set_sha256(families)
    if fit["fit_context_set_sha256"] != ctx_sha or fit["n_contexts"] != len(present):
        raise NormalizerFitRefused("fit_context_set", f"fit context set {fit['fit_context_set_sha256']} "
                                   f"(n={fit['n_contexts']}) != present train_fit {ctx_sha} (n={len(present)})")
    if fit["fit_family_set_sha256"] != fam_sha or fit["n_families"] != len(families):
        raise NormalizerFitRefused("fit_family_set", f"fit family set {fit['fit_family_set_sha256']} != {fam_sha}")
    return {"fit_context_set_sha256": ctx_sha, "fit_family_set_sha256": fam_sha, "n_contexts": len(present),
            "n_families": len(families)}
