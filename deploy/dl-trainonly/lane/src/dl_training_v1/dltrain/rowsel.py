"""Train-only row selection (ROWSEL-TRAINONLY prototype). Reversible prep: nothing here reads S3, the network or the cohort.

The leak this removes. research_2k views were cut from each dataset's T1 rows (train lines then test lines) by the Rust
research sampler (research_2k/rust/research-sampler/src/main.rs, choose_indices): target-rank bins, k-means clusters on
random projections normalised by a mean/scale, and a missing-count bin, all fitted on ALL rows, then <= 2,000 rows taken
per stratum. The outer folds were remapped afterwards from parent folds defined on the full source rows
(materialize.remap_membership). Rows that later became outer-test therefore shaped which training rows exist.

Design, per dataset and outer parent fold f (repeat 0):

    universe U        every candidate row with a stable integer position and a value-free string id, supplied by the
                      caller (T1 lines: universe_from_t1; a raw pre-coreset source: universe_from_raw)
    parent folds      a partition of U fixed before any sampling (split registry folds for T1 universes)
    quotas            (n_train_f, n_test_f) from the parent-fold SIZES only (fold_quotas): the budget (2,000) is
                      apportioned over folds by largest remainder of budget*|F_f|/|U|, which is the expected v1 view
                      proportion (the v1 sample was fold-agnostic); n_train_f = budget - n_test_f. |U| <= budget keeps
                      every row (v1 did the same). Callers that perturb fold sizes pin the quotas explicitly.
    train rows        sampler_select(U, fit_positions = U \\ F_f, max_rows = n_train_f, seed): every fitted quantity
                      (target bins, projection mean/scale, k-means seeds and centroids, missing bins, strata, quotas,
                      priorities) is computed from the fit rows alone. The function slices the fit block first and
                      never touches the universe again.
    test rows         draw_test_rows(dataset_hash, ids of F_f, n_test_f): the n_test_f smallest
                      sha256(dataset_hash NUL 'trainonly-rowsel-v1' NUL row_id). It receives ids only: no values, no
                      target, no fitted state.

counterfactual_report() is the regression check: it perturbs, replaces and drops fold-f rows and lists every train-side
field that is not bit-identical, and it perturbs train rows and lists every test-draw field that changed. v1_view() is
the leaky v1 reconstruction (fit on all rows, then remap); it must FAIL that check (tests/test_rowsel.py).

Port of main.rs choose_indices. Deviations (D1-D9); none changes a selection when fit_positions = every row, positions
equal the CSV line index and both sides use glibc's tanh (verified against v1 results, see the prototype receipt):
    D1 fit_positions: main.rs fits on and selects from every input row; here an explicit fit set is both the only
       source of fitted state and the only candidate pool.
    D2 the priority index is the caller's stable position, not the row's offset in the CSV handed to the sampler. For
       T1 universes the position is materialize.py's concatenation index (train lines, then test lines), which equals
       main.rs's index when every row is fitted. It stays fixed when other rows are perturbed, replaced or dropped.
    D3 run()'s limits (>= 1,000 rows, max_rows in 1,000..2,000) are not enforced: n_train_f is ~1,600 and fit sets
       may be small. max_rows >= 0; allocate_quotas refuses a total above the fit-row count (main.rs would loop).
    D4 selected positions are returned sorted; main.rs's output-shuffle order is SelectionFit.output_order.
    D5 floating point: column sums use np.add.accumulate (sequential, row order, as main.rs's loops), squared
       distances accumulate dimension by dimension, and tanh is math.tanh (libm). numpy's tanh is NOT used: on this
       host it differs from glibc in ~30% of values (SIMD). main.rs and math.tanh both call the platform libm, so bit
       identity with Rust holds on glibc hosts; elsewhere the selection may differ by ULP-level ties.
    D6 f64::max(x, 1e-12) is np.fmax (NaN-ignoring, like Rust); numpy's maximum would propagate NaN.
    D7 numbers: parse_csv_lines mirrors main.rs parse_csv (split on ',', trim, empty = NaN, target must be finite);
       Python float() and Rust parse::<f64> are both correctly rounded, but Python also accepts '_' digit separators,
       which are refused explicitly.
    D8 diagnostics (quantile errors, mean shifts) are not ported: they do not affect the selection.
    D9 test-draw keys must be value-free. v1's source_row_id = sha256(dataset_hash, partition, index, row_bytes)
       hashes the row bytes, the target included; a draw keyed on it changes when a test row's values change. The
       universes built here key the draw on positional_row_id (dataset_hash, partition, index); v1_source_row_id is
       kept only as a provenance join to v1 memberships.

Numpy and the standard library only.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from typing import Callable, Mapping, Sequence

import numpy as np

SCHEMA = "jope.dl-rowsel-trainonly/v1"
SAMPLER_NAME = "target-feature-strata-projection-kmeans"
SAMPLER_VERSION = 1
PORT = "dltrain.rowsel/v1 (port of research_2k/rust/research-sampler/src/main.rs choose_indices)"
DRAW_DOMAIN = "trainonly-rowsel-v1"
SEED_DOMAIN = "trainonly-rowsel-v1"
DEFAULT_BUDGET = 2000
TARGET_BINS = 16
FEATURE_CLUSTERS = 16
PROJECTION_DIMS = 8
KMEANS_ITERATIONS = 12
CHUNK_ROWS = 1 << 16
CHUNK_CELLS = 1 << 20

FOLD_PERTURBATIONS = ("features", "targets", "missingness", "outlier", "replace", "drop")
VALUE_PERTURBATIONS = ("features", "targets", "missingness", "outlier")   # ids and sizes unchanged


class RowselError(ValueError):
    pass


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _u64(values) -> bytes:
    return np.asarray(values, dtype="<u8").tobytes()


def _f64(values) -> bytes:
    return np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()


# ------------------------------------------------------------------------------------------------- identities

def derive_seed(dataset_hash: str) -> str:
    """Per-dataset sampler seed for the train-only views; reads no data (v1: sha256(dataset_hash NUL 'research-2k-v1'))."""
    return _sha((dataset_hash + "\0" + SEED_DOMAIN).encode())


def v1_seed(dataset_hash: str) -> str:
    """The seed materialize.py passed to the Rust sampler."""
    return _sha((dataset_hash + "\0research-2k-v1").encode())


def positional_row_id(dataset_hash: str, partition: str, index: int) -> str:
    """Value-free row id: materialize.source_row_id without the row bytes."""
    return _sha(dataset_hash.encode() + b"\0" + partition.encode() + b"\0" + str(int(index)).encode())


def v1_source_row_id(dataset_hash: str, partition: str, index: int, row: bytes) -> str:
    """materialize.source_row_id (binds the row bytes, target included): a provenance join key, never a draw key (D9)."""
    return _sha(dataset_hash.encode() + b"\0" + partition.encode() + b"\0" + str(int(index)).encode() + b"\0" + row)


def draw_rank(dataset_hash: str, row_id: str) -> bytes:
    return hashlib.sha256(dataset_hash.encode() + b"\0" + DRAW_DOMAIN.encode() + b"\0" + row_id.encode()).digest()


def priority(seed: str, domain: str, index: int, row: bytes) -> bytes:
    """main.rs priority(): sha256(seed NUL domain NUL u64le(index) NUL row)."""
    return hashlib.sha256(seed.encode() + b"\0" + domain.encode() + b"\0" + struct.pack("<Q", int(index)) + b"\0"
                          + row).digest()


# ------------------------------------------------------------------------------------------------- universe

@dataclasses.dataclass(frozen=True, eq=False)
class Universe:
    """Every candidate row of one dataset. positions: stable unique source indices; row_ids: stable unique value-free
    ids (the test-draw key); X: float64 features with NaN for missing; y: finite float64 target; row_bytes: the bytes
    hashed into the train-side priorities (the CSV line for T1 rows)."""
    dataset_hash: str
    positions: np.ndarray
    row_ids: tuple
    X: np.ndarray
    y: np.ndarray
    row_bytes: tuple

    def __post_init__(self):
        pos = np.array(self.positions, dtype=np.int64).reshape(-1)
        X = np.array(self.X, dtype=np.float64)
        y = np.array(self.y, dtype=np.float64).reshape(-1)
        n = pos.size
        if X.ndim != 2 or X.shape[0] != n or y.size != n or len(self.row_ids) != n or len(self.row_bytes) != n:
            raise RowselError(f"universe arrays disagree: positions {n}, X {X.shape}, y {y.size}, "
                              f"ids {len(self.row_ids)}, bytes {len(self.row_bytes)}")
        if X.shape[1] < 1:
            raise RowselError("universe needs at least one feature column")
        if n and int(pos.min()) < 0:
            raise RowselError("positions must be non-negative")
        if np.unique(pos).size != n:
            raise RowselError("positions are not unique")
        if len(set(self.row_ids)) != n:
            raise RowselError("row ids are not unique")
        if not bool(np.all(np.isfinite(y))):
            raise RowselError("non-finite target")
        for a in (pos, X, y):
            a.setflags(write=False)
        object.__setattr__(self, "positions", pos)
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "row_ids", tuple(str(v) for v in self.row_ids))
        object.__setattr__(self, "row_bytes", tuple(bytes(v) for v in self.row_bytes))
        object.__setattr__(self, "_index", {int(p): i for i, p in enumerate(pos)})

    @property
    def n(self) -> int:
        return int(self.positions.size)

    @property
    def features(self) -> int:
        return int(self.X.shape[1])

    def index_of(self, positions) -> np.ndarray:
        try:
            return np.array([self._index[int(p)] for p in positions], dtype=np.int64)
        except KeyError as error:
            raise RowselError(f"position {error.args[0]} is not in the universe") from None


def lines(body: bytes) -> list[bytes]:
    """main.rs parse_csv line split: '\\n', one trailing '\\r' removed, empty lines skipped."""
    out = []
    for raw in body.split(b"\n"):
        line = raw[:-1] if raw.endswith(b"\r") else raw
        if line:
            out.append(line)
    return out


def _parse_f64(text: str) -> float:
    if "_" in text:
        raise RowselError(f"not a Rust f64 literal: {text!r}")
    return float(text)


def parse_csv_lines(body_lines: Sequence[bytes], expected_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """main.rs parse_csv over already split lines: features float64 with NaN for empty cells, finite target last."""
    if expected_columns < 2:
        raise RowselError("need at least one feature and the target")
    rows = np.empty((len(body_lines), expected_columns), dtype=np.float64)
    for r, line in enumerate(body_lines):
        fields = line.decode("utf-8").split(",")
        if len(fields) != expected_columns:
            raise RowselError(f"line {r + 1} has {len(fields)} columns; expected {expected_columns}")
        for c, field in enumerate(fields):
            text = field.strip()
            rows[r, c] = math.nan if not text else _parse_f64(text)
        if not math.isfinite(rows[r, -1]):
            raise RowselError(f"non-finite target on line {r + 1}")
    return rows[:, :-1], rows[:, -1].copy()


def universe_from_t1(dataset_hash: str, train_body: bytes, test_body: bytes, expected_columns: int) -> Universe:
    """T1 universe as materialize.py concatenated it: train lines then test lines; position = concatenation index."""
    train_lines, test_lines = lines(train_body), lines(test_body)
    all_lines = train_lines + test_lines
    X, y = parse_csv_lines(all_lines, expected_columns)
    ids = [positional_row_id(dataset_hash, "train", i) for i in range(len(train_lines))]
    ids += [positional_row_id(dataset_hash, "test", i) for i in range(len(test_lines))]
    return Universe(dataset_hash, np.arange(len(all_lines)), tuple(ids), X, y, tuple(all_lines))


def t1_v1_source_row_ids(dataset_hash: str, train_body: bytes, test_body: bytes) -> list[str]:
    """v1 membership ids of the T1 universe rows, in position order (provenance join only)."""
    out = [v1_source_row_id(dataset_hash, "train", i, line) for i, line in enumerate(lines(train_body))]
    out += [v1_source_row_id(dataset_hash, "test", i, line) for i, line in enumerate(lines(test_body))]
    return out


def canonical_row_bytes(x: np.ndarray, y: float) -> bytes:
    """Row bytes for rows without a source line: repr of every value, '' for missing, target last."""
    cells = ["" if not math.isfinite(v) else repr(float(v)) for v in np.asarray(x, dtype=np.float64)]
    return (",".join(cells + [repr(float(y))])).encode()


def universe_from_raw(dataset_hash: str, X, y, raw_index, row_bytes: Sequence[bytes] | None = None,
                      partition: str = "raw") -> Universe:
    """Raw pre-selection universe (L7a source rows, e.g. before the T1 hybrid_coreset or acquisition cap).

    The caller supplies numeric rows: features float64 with NaN for missing, finite target. Category columns must be
    coded by a rule that reads no outer-test rows (a fixed by-key code, or a code refitted on the fit rows of the
    fold being built); acquisition codes fitted on all rows would carry the all-row fit back in. raw_index is the
    stable row index in the raw file after deterministic row-wise cleanup; it is both the position and, with the
    partition name, the value-free draw id."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    raw_index = np.asarray(raw_index, dtype=np.int64).reshape(-1)
    if row_bytes is None:
        row_bytes = [canonical_row_bytes(X[i], y[i]) for i in range(len(y))]
    ids = [positional_row_id(dataset_hash, partition, int(i)) for i in raw_index]
    return Universe(dataset_hash, raw_index, tuple(ids), X, y, tuple(row_bytes))


# ------------------------------------------------------------------------------------------------- sampler port

def _chunk_rows(columns: int) -> int:
    return max(1, CHUNK_CELLS // max(int(columns), 1))


def _seq_colsum(a: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """Column sums of a 2-d block accumulated row by row, continuing from `init` (default 0.0): main.rs loop order.
    Each column is accumulated along contiguous memory (transposed chunk), never pairwise."""
    a = np.asarray(a, dtype=np.float64)
    s = np.zeros(a.shape[1], dtype=np.float64) if init is None else np.asarray(init, dtype=np.float64)
    step = _chunk_rows(a.shape[1])
    for start in range(0, a.shape[0], step):
        block = np.concatenate([s[:, None], a[start:start + step].T], axis=1)
        s = np.add.accumulate(block, axis=1)[:, -1].copy()
    return s


def _total_order_key(values: np.ndarray) -> np.ndarray:
    """f64::total_cmp as a signed int64 key."""
    bits = np.ascontiguousarray(values, dtype=np.float64).view(np.int64)
    return bits ^ ((bits >> 63).view(np.uint64) >> np.uint64(1)).view(np.int64)


def rank_bins(values: np.ndarray, requested: int) -> np.ndarray:
    """main.rs rank_bins: rank by (total_cmp, index); bins = min(requested, distinct values, n); bin = rank*bins//n."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(n), _total_order_key(values)))
    ordered = values[order]
    unique = int(np.count_nonzero(ordered[1:] != ordered[:-1])) + 1
    bins = min(max(int(requested), 1), unique, n)
    out = np.empty(n, dtype=np.int64)
    out[order] = (np.arange(n, dtype=np.int64) * bins) // n
    return out


def feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """main.rs feature_stats: per-column mean and sample sd (floored at 1e-12; 1.0 below two values) over finite cells."""
    X = np.asarray(X, dtype=np.float64)
    m, features = X.shape
    step = _chunk_rows(features)
    counts = np.zeros(features, dtype=np.int64)
    sums = np.zeros(features, dtype=np.float64)
    ss = np.zeros(features, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        for start in range(0, m, step):                       # pass 1: counts and sums of finite cells, row order
            block = X[start:start + step]
            finite = np.isfinite(block)
            counts += finite.sum(axis=0)
            sums = _seq_colsum(np.where(finite, block, 0.0), sums)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        for start in range(0, m, step):                       # pass 2: squared deviations, row order
            block = X[start:start + step]
            ss = _seq_colsum(np.where(np.isfinite(block), (block - means) ** 2, 0.0), ss)
        scales = np.where(counts > 1, np.fmax(np.sqrt(ss / np.maximum(counts - 1, 1)), 1e-12), 1.0)
    missing = 1.0 - counts / max(m, 1)
    return means, scales, missing


def projection_weights(seed: str, features: int, dims: int) -> np.ndarray:
    """main.rs projection_weight for feature rows 0..2F (F..2F are the missing-indicator weights)."""
    w = np.empty((2 * features, dims), dtype=np.float64)
    denom = float((1 << 64) - 1)
    for j in range(2 * features):
        for d in range(dims):
            digest = hashlib.sha256(seed.encode() + b"projection" + struct.pack("<Q", j) + struct.pack("<Q", d)).digest()
            w[j, d] = 2.0 * (float(int.from_bytes(digest[:8], "little")) / denom) - 1.0
    return w


def project(X: np.ndarray, means: np.ndarray, scales: np.ndarray, seed: str, dims: int) -> np.ndarray:
    """main.rs project: tanh((x - mean) / (3 scale)) . w, plus 0.25 w' per missing cell, over sqrt(F)."""
    X = np.asarray(X, dtype=np.float64)
    m, features = X.shape
    w = projection_weights(seed, features, dims)
    norm = math.sqrt(max(features, 1))
    out = np.empty((m, dims), dtype=np.float64)
    for start in range(0, m, CHUNK_ROWS):
        block = X[start:start + CHUNK_ROWS]
        value = np.zeros((block.shape[0], dims), dtype=np.float64)
        for j in range(features):
            col = block[:, j]
            finite = np.isfinite(col)
            z = np.zeros(block.shape[0], dtype=np.float64)
            if finite.any():
                t = (col[finite] - means[j]) / (3.0 * scales[j])
                z[finite] = np.fromiter(map(math.tanh, t.tolist()), dtype=np.float64, count=t.size)
            value += z[:, None] * w[j][None, :]
            if not finite.all():
                value[~finite] += 0.25 * w[features + j]
        out[start:start + block.shape[0]] = value / norm
    return out


def _sqdist(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    acc = (points[:, None, 0] - centroids[None, :, 0]) ** 2
    for d in range(1, points.shape[1]):
        acc = acc + (points[:, None, d] - centroids[None, :, d]) ** 2
    return acc


def kmeans(projected: np.ndarray, requested: int, seed: str, positions: np.ndarray, row_bytes: Sequence[bytes]):
    """main.rs clusters: centroids seeded at evenly spaced sha256 'centroid' priorities, <= 12 Lloyd iterations,
    ties to the lower cluster, empty clusters keep their centroid. Returns (assignments, k, seed positions, centroids)."""
    m = projected.shape[0]
    k = min(max(int(requested), 1), m)
    ranked = sorted((priority(seed, "centroid", int(positions[i]), row_bytes[i]), int(positions[i]), i) for i in range(m))
    init = [ranked[slot * m // k][2] for slot in range(k)]
    centroids = projected[init].copy()
    assignments = np.zeros(m, dtype=np.int64)
    for _ in range(KMEANS_ITERATIONS):
        best = np.empty(m, dtype=np.int64)
        for start in range(0, m, CHUNK_ROWS):
            best[start:start + CHUNK_ROWS] = np.argmin(_sqdist(projected[start:start + CHUNK_ROWS], centroids), axis=1)
        changed = bool(np.any(assignments != best))
        assignments = best
        for cluster in range(k):
            members = projected[assignments == cluster]
            if members.shape[0]:
                centroids[cluster] = _seq_colsum(members) / members.shape[0]
        if not changed:
            break
    return assignments, k, [int(positions[i]) for i in init], centroids


def missing_bins(X: np.ndarray) -> np.ndarray:
    features = X.shape[1]
    missing = (~np.isfinite(X)).sum(axis=1)
    return np.where(missing == 0, 0, np.where(missing * 4 <= features, 1, np.where(missing * 2 <= features, 2, 3)))


def allocate_quotas(sizes: Mapping[tuple, int], total: int) -> dict[tuple, int]:
    """main.rs allocate_quotas: one row per stratum when they fit, then largest remainder, round-robin top-up."""
    if total > sum(sizes.values()):
        raise RowselError("quota total exceeds the rows available")
    keys = sorted(sizes)
    quotas: dict[tuple, int] = {}
    if len(keys) <= total:
        quotas = {key: 1 for key in keys}
    base = sum(quotas.values())
    remaining = total - base
    if remaining == 0:
        return quotas
    weights = sum(sizes[key] - quotas.get(key, 0) for key in keys)
    remainders = []
    assigned = base
    for key in keys:
        prior = quotas.get(key, 0)
        numerator = remaining * (sizes[key] - prior)
        add = 0 if weights == 0 else numerator // weights
        quotas[key] = prior + add
        assigned += add
        remainders.append((0 if weights == 0 else numerator % weights, key))
    remainders.sort(key=lambda item: (-item[0], item[1]))
    cursor = 0
    while assigned < total:
        key = remainders[cursor % len(remainders)][1]
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            assigned += 1
        cursor += 1
    return quotas


@dataclasses.dataclass(frozen=True)
class SelectionFit:
    positions: tuple            # selected universe positions, ascending
    output_order: tuple         # main.rs output-shuffle order of the same positions
    passthrough: bool           # fit rows <= max_rows: every fit row kept, nothing fitted
    fingerprints: dict
    identity: str


def sampler_select(universe: Universe, fit_positions, max_rows: int, seed: str, *, target_bins: int = TARGET_BINS,
                   feature_clusters: int = FEATURE_CLUSTERS, projection_dims: int = PROJECTION_DIMS) -> SelectionFit:
    """Fit the research sampler on fit_positions ONLY and select <= max_rows of them (main.rs choose_indices, D1-D9)."""
    if max_rows < 0:
        raise RowselError("max_rows must be >= 0")
    fit_sorted = sorted({int(p) for p in fit_positions})
    idx = universe.index_of(fit_sorted)
    # ---- the fit block; nothing below reads `universe`
    pos = np.asarray(fit_sorted, dtype=np.int64)
    Xf = universe.X[idx]
    yf = universe.y[idx]
    bf = [universe.row_bytes[i] for i in idx]
    m = int(pos.size)
    params = {"seed": seed, "max_rows": int(max_rows), "target_bins": int(target_bins),
              "feature_clusters": int(feature_clusters), "projection_dims": int(projection_dims),
              "kmeans_iterations": KMEANS_ITERATIONS, "sampler": SAMPLER_NAME, "sampler_version": SAMPLER_VERSION}
    fp = {"params": params, "fit_rows": m, "fit_positions_sha256": _sha(_u64(pos)),
          "fit_values_sha256": _sha(_f64(Xf) + _f64(yf) + b"".join(struct.pack("<Q", len(b)) + b for b in bf))}
    if m <= max_rows:
        local = list(range(m))
        fp["passthrough"] = True
    else:
        tb = rank_bins(yf, target_bins)
        means, scales, _ = feature_stats(Xf)
        projected = project(Xf, means, scales, seed, projection_dims)
        assignments, k, init_positions, centroids = kmeans(projected, feature_clusters, seed, pos, bf)
        mb = missing_bins(Xf)
        strata: dict[tuple, list[int]] = {}
        for i in range(m):
            strata.setdefault((int(tb[i]), int(assignments[i]), int(mb[i])), []).append(i)
        quotas = allocate_quotas({key: len(v) for key, v in strata.items()}, max_rows)
        local = []
        for key in sorted(strata):
            ranked = sorted((priority(seed, "within-stratum", int(pos[i]), bf[i]), int(pos[i]), i) for i in strata[key])
            local.extend(i for _, _, i in ranked[:quotas[key]])
        if len(local) != max_rows:
            raise RowselError(f"selected {len(local)} rows, expected {max_rows}")
        fp.update({
            "passthrough": False,
            "target_bins_sha256": _sha(_u64(tb)), "target_bin_count": int(tb.max()) + 1,
            "feature_means_sha256": _sha(_f64(means)), "feature_scales_sha256": _sha(_f64(scales)),
            "projection_sha256": _sha(_f64(projected)),
            "centroid_seed_positions": init_positions, "clusters": int(k),
            "centroids_sha256": _sha(_f64(centroids)), "cluster_assignments_sha256": _sha(_u64(assignments)),
            "missing_bins_sha256": _sha(_u64(mb)),
            "strata_sha256": _sha(_canon([[list(key), len(strata[key]), quotas[key]] for key in sorted(strata)])),
            "nonempty_strata": len(strata),
        })
    selected = sorted(int(pos[i]) for i in local)
    byte_of = {int(pos[i]): bf[i] for i in local}
    output = sorted(selected, key=lambda p: (priority(seed, "output-shuffle", p, byte_of[p]), p))
    fp["selected_positions_sha256"] = _sha(_u64(selected))
    fp["selected"] = len(selected)
    return SelectionFit(tuple(selected), tuple(output), bool(fp["passthrough"]), fp, _sha(_canon(fp)))


def selected_source_indices_sha256(output_order: Sequence[int]) -> str:
    """main.rs selected_source_indices_sha256 (u64le, output-shuffle order); comparable with v1 manifests."""
    return _sha(_u64(list(output_order)))


# ------------------------------------------------------------------------------------------------- folds, quotas, views

def validate_parent_folds(universe: Universe, parent_folds: Mapping[int, Sequence[int]]) -> dict[int, np.ndarray]:
    """Parent folds as {fold: positions}; they must partition the universe positions exactly."""
    folds = {int(f): np.array(sorted(int(p) for p in v), dtype=np.int64) for f, v in parent_folds.items()}
    seen = np.concatenate([v for v in folds.values()]) if folds else np.zeros(0, dtype=np.int64)
    if seen.size != universe.n or np.unique(seen).size != seen.size or not np.array_equal(np.sort(seen),
                                                                                        np.sort(universe.positions)):
        raise RowselError("parent folds do not partition the universe positions exactly once")
    return folds


def fold_quotas(fold_sizes: Mapping[int, int], budget: int = DEFAULT_BUDGET) -> dict[int, tuple[int, int]]:
    """(n_train_f, n_test_f) per fold from parent-fold sizes only. |U| <= budget keeps every row (v1 behaviour);
    otherwise n_test_f is the largest-remainder share of budget*|F_f|/|U| (ties to the lower fold) and
    n_train_f = budget - n_test_f, so every view holds `budget` rows in the v1 expected proportion."""
    if budget < 1:
        raise RowselError("budget must be >= 1")
    sizes = {int(f): int(s) for f, s in fold_sizes.items()}
    total = sum(sizes.values())
    if total <= budget:
        return {f: (total - s, s) for f, s in sizes.items()}
    base = {f: budget * s // total for f, s in sizes.items()}
    rem = sorted(sizes, key=lambda f: (-(budget * sizes[f] % total), f))
    for f in rem[:budget - sum(base.values())]:
        base[f] += 1
    return {f: (budget - base[f], base[f]) for f in sizes}


def draw_test_rows(dataset_hash: str, row_ids: Sequence[str], n_test: int) -> list[str]:
    """The n_test ids with the smallest sha256(dataset_hash NUL DRAW_DOMAIN NUL id). Ids only: no values, no fit."""
    ranked = sorted((draw_rank(dataset_hash, rid), rid) for rid in row_ids)
    return [rid for _, rid in ranked[:max(int(n_test), 0)]]


@dataclasses.dataclass(frozen=True)
class View:
    dataset_hash: str
    fold: int
    budget: int
    quotas: tuple               # (n_train, n_test) requested
    seed: str
    train_positions: tuple
    test_positions: tuple
    train_row_ids: tuple
    test_row_ids: tuple
    train_fit: SelectionFit
    train_values_sha256: str
    test_values_sha256: str
    test_draw_sha256: str
    identity: str


def _values_sha(universe: Universe, positions) -> str:
    idx = universe.index_of(positions)
    return _sha(_f64(universe.X[idx]) + _f64(universe.y[idx])
                + b"".join(struct.pack("<Q", len(universe.row_bytes[i])) + universe.row_bytes[i] for i in idx))


def make_view(universe: Universe, fold: int, budget: int, quotas, seed: str, train_fit: SelectionFit,
              test_positions) -> View:
    train = tuple(int(p) for p in train_fit.positions)
    test = tuple(sorted(int(p) for p in test_positions))
    if set(train) & set(test):
        raise RowselError("train and test rows overlap")
    train_ids = tuple(universe.row_ids[i] for i in universe.index_of(train))
    test_ids = tuple(universe.row_ids[i] for i in universe.index_of(test))
    draw_sha = _sha(_canon({"domain": DRAW_DOMAIN, "ids": sorted(test_ids)}))
    body = {"schema": SCHEMA, "dataset_hash": universe.dataset_hash, "fold": int(fold), "budget": int(budget),
            "quotas": [int(q) for q in quotas], "seed": seed, "port": PORT, "draw_domain": DRAW_DOMAIN,
            "train_fit_identity": train_fit.identity, "train_positions_sha256": _sha(_u64(train)),
            "test_positions_sha256": _sha(_u64(test)), "test_draw_sha256": draw_sha}
    return View(universe.dataset_hash, int(fold), int(budget), tuple(int(q) for q in quotas), seed, train, test,
                train_ids, test_ids, train_fit, _values_sha(universe, train), _values_sha(universe, test), draw_sha,
                _sha(_canon(body)))


def select_view(universe: Universe, parent_folds: Mapping[int, Sequence[int]], fold: int,
                budget: int = DEFAULT_BUDGET, seed: str | None = None, quotas=None) -> View:
    """Train-only view for parent fold `fold`: train rows by the sampler fitted on the complement only, test rows by
    the id-keyed draw from the fold. quotas=(n_train, n_test) pins the counts (default: fold_quotas over sizes)."""
    folds = validate_parent_folds(universe, parent_folds)
    if int(fold) not in folds:
        raise RowselError(f"fold {fold} not in the parent folds")
    seed = derive_seed(universe.dataset_hash) if seed is None else seed
    if quotas is None:
        quotas = fold_quotas({f: v.size for f, v in folds.items()}, budget)[int(fold)]
    n_train, n_test = (int(q) for q in quotas)
    if n_train < 0 or n_test < 0:
        raise RowselError("quotas must be non-negative")
    test_universe = folds[int(fold)]
    held = set(test_universe.tolist())
    train_universe = [int(p) for p in universe.positions if int(p) not in held]
    train_fit = sampler_select(universe, train_universe, n_train, seed)
    test_ids = [universe.row_ids[i] for i in universe.index_of(test_universe)]
    drawn = set(draw_test_rows(universe.dataset_hash, test_ids, n_test))
    test_positions = [int(p) for p, rid in zip(test_universe, test_ids) if rid in drawn]
    return make_view(universe, fold, budget, (n_train, n_test), seed, train_fit, test_positions)


def v1_view(universe: Universe, parent_folds: Mapping[int, Sequence[int]], fold: int,
            budget: int = DEFAULT_BUDGET, seed: str | None = None, quotas=None) -> View:
    """The LEAKY v1 construction, kept as a negative control and for faithfulness checks: the sampler is fitted on
    every universe row, `budget` rows are kept, and the fold is remapped (train = kept \\ F_f, test = kept & F_f).
    `quotas` is accepted for signature compatibility and ignored, as v1 had no per-fold quota."""
    folds = validate_parent_folds(universe, parent_folds)
    seed = v1_seed(universe.dataset_hash) if seed is None else seed
    fit = sampler_select(universe, universe.positions.tolist(), budget, seed)
    held = set(folds[int(fold)].tolist())
    kept = list(fit.positions)
    train = [p for p in kept if p not in held]
    test = [p for p in kept if p in held]
    remapped = dataclasses.replace(fit, positions=tuple(train))
    return make_view(universe, fold, budget, (len(train), len(test)), seed, remapped, test)


# ------------------------------------------------------------------------------------------------- counterfactual check

def train_side(view: View) -> dict:
    return {"n_train": len(view.train_positions), "quotas_train": view.quotas[0],
            "train_positions": view.train_positions, "train_row_ids": view.train_row_ids,
            "train_fit_identity": view.train_fit.identity, "train_fit_fingerprints": view.train_fit.fingerprints,
            "train_values_sha256": view.train_values_sha256, "seed": view.seed}


def test_draw_side(view: View) -> dict:
    return {"n_test": len(view.test_positions), "test_positions": view.test_positions,
            "test_row_ids": view.test_row_ids, "test_draw_sha256": view.test_draw_sha256}


def _differs(a: dict, b: dict) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        if key == "train_fit_fingerprints":
            fa, fb = a.get(key) or {}, b.get(key) or {}
            out += [f"{key}.{k}" for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
        elif a.get(key) != b.get(key):
            out.append(key)
    return out


def _rng(salt: str, kind: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f"{salt}|{kind}".encode()).digest()[:8], "little"))


def _rebuild(u: Universe, X, y, changed_idx, positions=None, row_ids=None, row_bytes=None) -> Universe:
    rb = list(u.row_bytes if row_bytes is None else row_bytes)
    for i in changed_idx:
        rb[i] = canonical_row_bytes(X[i], y[i])
    return Universe(u.dataset_hash, u.positions if positions is None else positions,
                    u.row_ids if row_ids is None else row_ids, X, y, tuple(rb))


def perturb_rows(universe: Universe, parent_folds: Mapping[int, Sequence[int]], target_positions, kind: str,
                 salt: str = "rowsel-cf-v1") -> tuple[Universe, dict]:
    """Counterfactual universe: the rows at target_positions get their features, targets or missingness changed, one
    of them becomes an extreme outlier, they are replaced by new rows (new positions and ids), or half are dropped.
    Rows outside target_positions are untouched (values, positions, ids, bytes)."""
    folds = {int(f): [int(p) for p in v] for f, v in parent_folds.items()}
    targets = sorted(int(p) for p in target_positions)
    idx = universe.index_of(targets)
    rng = _rng(salt, kind)
    X = universe.X.copy()
    y = universe.y.copy()
    if kind == "features":
        finite = np.isfinite(X[idx])
        X[idx] = np.where(finite, rng.normal(50.0, 1e3, size=finite.shape), X[idx])
        return _rebuild(universe, X, y, idx), folds
    if kind == "targets":
        y[idx] = rng.normal(1e7, 1e6, size=idx.size)
        return _rebuild(universe, X, y, idx), folds
    if kind == "missingness":
        block = X[idx]
        finite = np.isfinite(block)
        flip = rng.random(block.shape) < 0.35
        block = np.where(finite & flip, np.nan, block)
        block = np.where(~finite, 0.5, block)
        X[idx] = block
        return _rebuild(universe, X, y, idx), folds
    if kind == "outlier":
        i = idx[:1]
        X[i] = 1e12
        y[i] = -1e12
        return _rebuild(universe, X, y, i), folds
    if kind == "drop":
        drop = set(targets[::2])
        keep = np.array([int(p) not in drop for p in universe.positions])
        new = Universe(universe.dataset_hash, universe.positions[keep],
                       tuple(r for r, k in zip(universe.row_ids, keep) if k), X[keep], y[keep],
                       tuple(b for b, k in zip(universe.row_bytes, keep) if k))
        return new, {f: [p for p in v if p not in drop] for f, v in folds.items()}
    if kind == "replace":
        keep = np.ones(universe.n, dtype=bool)
        keep[idx] = False
        donors = rng.choice(np.flatnonzero(keep), size=idx.size, replace=True)
        newX = X[donors] * rng.normal(1.0, 0.3, size=X[donors].shape) + rng.normal(0.0, 5.0, size=X[donors].shape)
        newy = y[donors] * -3.0 + 11.0
        start = int(universe.positions.max()) + 1
        newpos = np.arange(start, start + idx.size, dtype=np.int64)
        newids = tuple(positional_row_id(universe.dataset_hash, f"replaced-{salt}", int(p)) for p in newpos)
        newbytes = tuple(canonical_row_bytes(newX[i], newy[i]) for i in range(idx.size))
        new = Universe(universe.dataset_hash, np.concatenate([universe.positions[keep], newpos]),
                       tuple(r for r, k in zip(universe.row_ids, keep) if k) + newids,
                       np.concatenate([X[keep], newX]), np.concatenate([y[keep], newy]),
                       tuple(b for b, k in zip(universe.row_bytes, keep) if k) + newbytes)
        removed = set(targets)
        out = {f: [p for p in v if p not in removed] for f, v in folds.items()}
        for f, v in folds.items():
            if removed & set(v):
                out[f] = out[f] + newpos.tolist()
        return new, out
    raise RowselError(f"unknown perturbation {kind!r}")


SelectFn = Callable[..., View]


def counterfactual_report(select_fn: SelectFn, universe: Universe, parent_folds: Mapping[int, Sequence[int]],
                          fold: int, budget: int = DEFAULT_BUDGET, seed: str | None = None,
                          salt: str = "rowsel-cf-v1") -> dict:
    """Rebuild the view after each fold-f perturbation and list every train-side field that is not bit-identical
    (quotas pinned from the unperturbed view); for value-only perturbations and for train-row perturbations, list
    every test-draw field that changed. A clean selector reports nothing; the v1 selector must report differences."""
    base = select_fn(universe, parent_folds, fold, budget, seed, quotas=None)
    pinned = base.quotas
    fold_positions = list(validate_parent_folds(universe, parent_folds)[int(fold)])
    cases = {}
    for kind in FOLD_PERTURBATIONS:
        u2, f2 = perturb_rows(universe, parent_folds, fold_positions, kind, salt)
        v2 = select_fn(u2, f2, fold, budget, seed, quotas=pinned)
        case = {"train_side_differs": _differs(train_side(base), train_side(v2))}
        if kind in VALUE_PERTURBATIONS:
            case["test_draw_differs"] = _differs(test_draw_side(base), test_draw_side(v2))
        cases[f"fold:{kind}"] = case
    held = set(int(p) for p in fold_positions)
    train_universe = [int(p) for p in universe.positions if int(p) not in held]
    for kind in ("features", "targets", "missingness"):
        u3, f3 = perturb_rows(universe, parent_folds, train_universe, kind, salt)
        v3 = select_fn(u3, f3, fold, budget, seed, quotas=pinned)
        cases[f"train:{kind}"] = {"test_draw_differs": _differs(test_draw_side(base), test_draw_side(v3))}
    return {
        "schema": SCHEMA + "#counterfactual",
        "dataset_hash": universe.dataset_hash, "fold": int(fold), "budget": int(budget), "quotas": list(pinned),
        "base_view_identity": base.identity,
        "cases": cases,
        "train_side_invariant": all(not c["train_side_differs"] for k, c in cases.items() if k.startswith("fold:")),
        "test_draw_invariant": all(not c.get("test_draw_differs") for c in cases.values()),
    }
