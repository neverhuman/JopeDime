"""Dataset-side descriptors: the `ds_` scalars and the dense per-dataset embedding blocks.

This module is deliberately NOT pure-genome. It reads per-dataset artifacts (league.db
`dataset_bank`, `meta.json`, baseline `reports/*.json`, and the attention-small_4168 `.vs`
vectors) and returns one row per DATASET, not per genome.

Why these columns cannot help on their own
------------------------------------------
Evaluation is GroupKFold by dataset. Every column in here is CONSTANT within a dataset, so its
main effect is mathematically zero on a cold-start fold. These exist only to be crossed with the
genome-side features (the G2 interaction block, built elsewhere): the measured effect that
motivates the whole module is that head value *flips* with dataset nonlinearity -- histgbm goes
0.373 -> 0.509 from linear to nonlinear datasets while ensemble INVERTS 0.576 -> 0.509. You cannot
represent a flip without a dataset axis to flip along.

Canonical algo set
------------------
Corpus-wide there are 296 distinct `reports/<algo>.json` filenames but only 128 are present for
100% of datasets. The dense `difficulty_*` blocks use ONLY the universal 128. A ragged block would
encode which algorithms had been *run* by the time a dataset was encoded, and that missingness
pattern correlates with encode date, not with difficulty -- it would leak calendar time into a
cold-start model.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sqlite3
import statistics

import numpy as np

from . import R2_CLIP_FLOOR

# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

CANON_ALGOS_PATH = "prime/runs/genome_perf_v1/cache/reports_canon.json"
CANON_ALGOS_ABS = os.path.join(REPO_ROOT, CANON_ALGOS_PATH)

LEAGUE_DB = os.environ.get("JOPE_LEAGUE_DB", os.path.join(REPO_ROOT, "prime/runs/league/league.db"))
QUANT_ROOT = os.environ.get("JOPE_QUANT_ROOT", "/home/ubuntu/quant/regression")
A4168_ROOT = os.environ.get(
    "JOPE_A4168_ROOT",
    "/home/ubuntu/remote_super/quant/regression-embeddings/attention-small_4168",
)
A4168_MANIFEST_DIR = os.path.join(A4168_ROOT, "manifests")

HYPERION_DIM = 1536
TABPFN_DIM = 512
A4168_DIM = 4168

# --------------------------------------------------------------------------------------------
# attention-small_4168 layout. INTERLEAVED, not contiguous.
#
#   sketch                     offsets    0 .. 855                       (856 dims)
#   for candidate i in 0..23:
#       hidden_state           offsets  856 + 138*i .. +128              (128 dims)
#       action                 offsets  984 + 138*i .. +10               ( 10 dims)
#   856 + 24*138 == 4168
# --------------------------------------------------------------------------------------------

N_SKETCH = 856
N_CANDIDATES = 24
CAND_STRIDE = 138
N_HIDDEN = 128
N_ACTION = 10
HIDDEN_BASE = 856
ACTION_BASE = 984

IDX_SKETCH = np.arange(N_SKETCH, dtype=np.int32)
IDX_HIDDEN = np.concatenate([
    np.arange(HIDDEN_BASE + CAND_STRIDE * i, HIDDEN_BASE + CAND_STRIDE * i + N_HIDDEN,
              dtype=np.int32)
    for i in range(N_CANDIDATES)
])
IDX_ACTION = np.concatenate([
    np.arange(ACTION_BASE + CAND_STRIDE * i, ACTION_BASE + CAND_STRIDE * i + N_ACTION,
              dtype=np.int32)
    for i in range(N_CANDIDATES)
])

_SKETCH_RE = re.compile(r"^sketch\..+\.(\d+)$")
_HIDDEN_RE = re.compile(r"^candidate\.(.+)\.hidden_state\.(\d+)$")
_ACTION_RE = re.compile(r"^candidate\.(.+)\.action\.(.+)$")


class A4168LayoutError(RuntimeError):
    """The .vs header does not match the documented interleaved layout."""


# --------------------------------------------------------------------------------------------
# Baseline-algorithm family grouping.
#
# Derived by hand from the 128 universal `reports/<algo>.json` NAMES and pinned here so the
# grouping is auditable and diffable rather than re-inferred per run. Rules are (kind, pattern):
#   "exact"  -- full algo name equality
#   "prefix" -- algo name starts with the pattern
# First matching family wins, in TREE -> KERNEL -> LINEAR order.
#
# Deliberate exclusions, documented so they are choices and not oversights:
#   knn*                        -- instance-based, not a kernel machine; in no family.
#   mlp, tabnet, tabpfn*, tabm*, tabicl, tabr, torch_frame_*, pytorch_tabular_*, ngboost,
#   autogluon_300s, flaml_automl_300s, and the in-house *_regressor entries
#   (fable5, jain*, chimera, jable, gpt5p6sol, hyp3_*)
#                               -- neural / AutoML / proprietary; they inform ds_best_r2 and
#                                  ds_mean_r2 but do not define a "ceiling" for any one family.
#   poly2_ridge                 -- LINEAR: it is a ridge on an explicit feature expansion, so it
#                                  belongs with the linear ceiling, not the kernel one.
#   ebm, ebm_interactions       -- TREE: explainable boosting machines are boosted tree ensembles.
#   xgboost_gblinear*           -- LINEAR despite the xgboost prefix; checked before the tree
#                                  prefix rule (see ALGO_FAMILY_OVERRIDES).
# --------------------------------------------------------------------------------------------

ALGO_FAMILY_OVERRIDES = {
    "xgboost_gblinear": "linear",
    "xgboost_gblinear_alpha0p001_lambda1": "linear",
    "poly2_ridge": "linear",
    "pca_lgbm": "tree",
    "stacking_trees_meta": "tree",
    "gpboost_regressor": "tree",
    "ebm": "tree",
    "ebm_interactions": "tree",
}

ALGO_FAMILY_RULES = (
    ("tree", "prefix", ("catboost", "lightgbm", "xgboost", "random_forest", "extra_trees",
                        "gradient_boosting", "hist_gradient_boosting", "histgbm", "ada_boost")),
    ("tree", "exact", ("decision_tree", "bagging")),
    ("kernel", "prefix", ("rbf_svr", "nu_svr", "kernel_ridge", "linear_svr", "nystroem",
                          "rbfsampler", "gaussian_process")),
    ("linear", "prefix", ("ridge", "lasso", "elastic_net", "linear_regression", "bayesian_ridge",
                          "ard_regression", "huber", "sgd_regressor", "theil_sen", "ransac",
                          "quantile_median", "poisson_regressor", "tweedie_regressor",
                          "orthogonal_matching_pursuit", "passive_aggressive")),
)

# ds_nonlinearity is defined against these two exact baselines, per the brief.
NONLINEARITY_HI = "catboost"
NONLINEARITY_LO = "ridge"


def algo_family(algo: str) -> str:
    """-> 'tree' | 'kernel' | 'linear' | '' for one baseline algorithm name."""
    fam = ALGO_FAMILY_OVERRIDES.get(algo)
    if fam is not None:
        return fam
    for family, kind, patterns in ALGO_FAMILY_RULES:
        if kind == "exact":
            if algo in patterns:
                return family
        elif any(algo.startswith(p) for p in patterns):
            return family
    return ""


# --------------------------------------------------------------------------------------------
# Canonical algo list
# --------------------------------------------------------------------------------------------

def build_canonical_algo_list(datasets, persist: bool = True) -> list[str]:
    """Algorithms whose report exists for 100% of `datasets`. Sorted. Persisted to CANON_ALGOS_PATH.

    Measured on the 2,090 `dataset_bank` datasets: 296 distinct report filenames, of which exactly
    128 are universal. (The brief said 295 distinct; the true count is 296. The 128 universal
    figure is confirmed.) The cliff is sharp -- relaxing to >=99% coverage adds only one algorithm
    and >=90% adds three, so 100% costs almost nothing in width and buys a block with no
    missingness pattern at all.
    """
    datasets = list(datasets)
    counts: dict[str, int] = {}
    n_seen = 0
    for ds in datasets:
        try:
            files = os.listdir(os.path.join(QUANT_ROOT, ds, "reports"))
        except OSError:
            continue
        n_seen += 1
        for f in files:
            if f.endswith(".json"):
                counts[f[:-5]] = counts.get(f[:-5], 0) + 1
    canon = sorted(a for a, n in counts.items() if n == n_seen) if n_seen else []

    if persist:
        os.makedirs(os.path.dirname(CANON_ALGOS_ABS), exist_ok=True)
        payload = {
            "n_datasets_requested": len(datasets),
            "n_datasets_with_reports": n_seen,
            "n_distinct_algos": len(counts),
            "n_universal": len(canon),
            "algos": canon,
        }
        tmp = CANON_ALGOS_ABS + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, CANON_ALGOS_ABS)  # atomic; a half-written canon list is silent corruption
    return canon


def load_canonical_algo_list() -> list[str]:
    """-> the persisted canon list, or [] if it has not been built yet."""
    try:
        with open(CANON_ALGOS_ABS) as fh:
            return list(json.load(fh)["algos"])
    except (OSError, KeyError, ValueError):
        return []


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _bank() -> dict[str, tuple]:
    """-> {dataset: (hyperion f4 (1536,), tabpfn f4 (512,))}. Whole table, ~17 MB."""
    out: dict[str, tuple] = {}
    try:
        con = sqlite3.connect(f"file:{LEAGUE_DB}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        rows = con.execute(
            "select dataset, hyperion, hyperion_dim, tabpfn_v3, tabpfn_v3_dim from dataset_bank"
        ).fetchall()
    except sqlite3.Error:
        return out
    finally:
        con.close()
    for ds, hb, hd, tb, td in rows:
        h = np.frombuffer(hb, dtype="<f4") if hb else None
        t = np.frombuffer(tb, dtype="<f4") if tb else None
        if h is not None and (len(h) != HYPERION_DIM or (hd and int(hd) != HYPERION_DIM)):
            h = None
        if t is not None and (len(t) != TABPFN_DIM or (td and int(td) != TABPFN_DIM)):
            t = None
        out[ds] = (h, t)
    return out


@functools.lru_cache(maxsize=8192)
def _meta(dataset: str) -> dict | None:
    try:
        with open(os.path.join(QUANT_ROOT, dataset, "meta.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


@functools.lru_cache(maxsize=8192)
def _reports(dataset: str, canon: tuple[str, ...]) -> dict[str, tuple[float, float, float]]:
    """-> {algo: (r2, training_time_seconds, inference_rows_per_second)} over canon only.

    Absent / malformed / non-finite entries are simply omitted, never defaulted to 0.0 -- an r2 of
    0.0 is a real and common value (it means "no better than the mean"), so imputing it would be
    indistinguishable from a genuinely useless baseline.
    """
    base = os.path.join(QUANT_ROOT, dataset, "reports")
    out: dict[str, tuple[float, float, float]] = {}
    for algo in canon:
        try:
            with open(os.path.join(base, algo + ".json")) as fh:
                obj = json.load(fh)
        except (OSError, ValueError):
            continue
        r2 = (obj.get("metrics") or {}).get("r2")
        if r2 is None:
            continue
        try:
            r2f = float(r2)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(r2f):
            continue
        out[algo] = (
            r2f,
            _safe_float(obj.get("training_time_seconds")),
            _safe_float(obj.get("inference_rows_per_second")),
        )
    return out


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if np.isfinite(f) else default


@functools.lru_cache(maxsize=1)
def zero_feature_datasets() -> frozenset:
    """Datasets the attention-small_4168 manifests record as having 0 usable feature columns.

    Measured: exactly 7 of 2,326 manifest records. NOTE these do NOT show up as
    `meta.json:n_features == 0` -- no dataset in the bank has that -- so the manifest is the only
    place this is recorded, which is why we read it here rather than deriving it from meta.
    """
    out: set[str] = set()
    try:
        files = os.listdir(A4168_MANIFEST_DIR)
    except OSError:
        return frozenset()
    for f in sorted(files):
        if not f.endswith(".json"):
            continue
        try:
            with open(os.path.join(A4168_MANIFEST_DIR, f)) as fh:
                obj = json.load(fh)
        except (OSError, ValueError):
            continue
        for rec in obj.get("records") or []:
            if not rec.get("features") and rec.get("dataset_id"):
                out.add(rec["dataset_id"])
    return frozenset(out)


# --------------------------------------------------------------------------------------------
# .vs reading + header validation
# --------------------------------------------------------------------------------------------

def validate_a4168_header(names) -> None:
    """Raise A4168LayoutError unless `names` matches the documented interleaved layout exactly.

    This is loud on purpose. Every downstream block slices by fixed integer offset; a silently
    reordered or regenerated artifact would produce a full-rank, finite, entirely wrong feature
    matrix with no error anywhere. Cheap check, unbounded downside if skipped.
    """
    names = list(names)
    if len(names) != A4168_DIM:
        raise A4168LayoutError(f"expected {A4168_DIM} component names, got {len(names)}")

    for j in range(N_SKETCH):
        m = _SKETCH_RE.match(names[j])
        if not m or int(m.group(1)) != j:
            raise A4168LayoutError(f"sketch offset {j}: unexpected component name {names[j]!r}")

    for i in range(N_CANDIDATES):
        h0 = HIDDEN_BASE + CAND_STRIDE * i
        m = _HIDDEN_RE.match(names[h0])
        if not m:
            raise A4168LayoutError(
                f"candidate {i}: offset {h0} is {names[h0]!r}, expected a hidden_state component"
            )
        cand = m.group(1)
        for k in range(N_HIDDEN):
            mm = _HIDDEN_RE.match(names[h0 + k])
            if not mm or mm.group(1) != cand or int(mm.group(2)) != k:
                raise A4168LayoutError(
                    f"candidate {i} ({cand}) hidden {k}: offset {h0 + k} is {names[h0 + k]!r}"
                )
        a0 = ACTION_BASE + CAND_STRIDE * i
        for k in range(N_ACTION):
            mm = _ACTION_RE.match(names[a0 + k])
            if not mm or mm.group(1) != cand:
                raise A4168LayoutError(
                    f"candidate {i} ({cand}) action {k}: offset {a0 + k} is {names[a0 + k]!r}"
                )


_A4168_HEADER_SHA: list[str] = []


def a4168_path(dataset: str) -> str:
    return os.path.join(A4168_ROOT, f"{dataset}_attention-small_4168.vs")


def read_a4168(dataset: str) -> np.ndarray | None:
    """-> (4168,) float32, or None when the .vs is missing/unreadable.

    Reading the 2-line .vs is ~5x faster than the equivalent .json. The ~182 KB header is
    identical across every file, so it is fully validated once and thereafter only compared by
    digest -- a changed digest re-runs the full structural validation and raises on mismatch.
    """
    path = a4168_path(dataset)
    try:
        with open(path) as fh:
            header = fh.readline().rstrip("\n")
            values = fh.readline().rstrip("\n")
    except OSError:
        return None
    if not header or not values:
        return None

    digest = hashlib.sha256(header.encode()).hexdigest()
    if not _A4168_HEADER_SHA:
        validate_a4168_header(header.split(","))
        _A4168_HEADER_SHA.append(digest)
    elif digest != _A4168_HEADER_SHA[0]:
        # Different header bytes: re-validate structurally rather than trusting it.
        validate_a4168_header(header.split(","))

    try:
        vec = np.array(values.split(","), dtype=np.float32)
    except ValueError:
        return None
    if vec.shape[0] != A4168_DIM:
        raise A4168LayoutError(
            f"{path}: expected {A4168_DIM} values, got {vec.shape[0]}"
        )
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------------------------
# ds_ scalar features
# --------------------------------------------------------------------------------------------

DS_FEATURE_NAMES = (
    "ds_nonlinearity",
    "ds_linear_ceiling",
    "ds_tree_ceiling",
    "ds_kernel_ceiling",
    "ds_best_r2",
    "ds_mean_r2",
    "ds_std_r2",
    "ds_frac_algos_above_0",
    "ds_best_algo_index",
    "ds_log_rows",
    "ds_log_cols",
    "ds_cat_frac",
    "ds_pn_ratio",
    "ds_target_only",
    "ds_median_train_time",
    "ds_meta_missing",
    "ds_reports_missing",
    "ds_hyperion_missing",
    "ds_tabpfn_missing",
    "ds_a4168_missing",
)


def dataset_feature_names() -> list[str]:
    """Fixed, deterministic column order for `dataset_features`. Independent of canon and data."""
    return list(DS_FEATURE_NAMES)


def _clip_r2(v: float) -> float:
    """Baselines can score r2 of -1e5 on a degenerate fit; the house floor is -1.0 (R2_CLIP_FLOOR).

    Without this, ds_mean_r2 / ds_std_r2 on a single blown-up baseline would swamp every real
    signal in the vector.
    """
    return max(R2_CLIP_FLOOR, min(1.0, v))


def dataset_features(dataset: str, canon: list[str]) -> dict[str, float]:
    """Scalar descriptors for one dataset. Never raises; missing sources -> 0.0 + a _missing flag."""
    canon_t = tuple(canon)
    out = {k: 0.0 for k in DS_FEATURE_NAMES}

    # --- baseline reports -------------------------------------------------------------------
    try:
        rep = _reports(dataset, canon_t)
    except Exception:
        rep = {}
    if not rep:
        out["ds_reports_missing"] = 1.0
    else:
        r2 = {a: _clip_r2(v[0]) for a, v in rep.items()}
        vals = list(r2.values())

        hi = r2.get(NONLINEARITY_HI)
        lo = r2.get(NONLINEARITY_LO)
        if hi is not None and lo is not None:
            out["ds_nonlinearity"] = float(hi - lo)
        if lo is not None:
            out["ds_linear_ceiling"] = float(lo)

        tree = [r2[a] for a in r2 if algo_family(a) == "tree"]
        kern = [r2[a] for a in r2 if algo_family(a) == "kernel"]
        if tree:
            out["ds_tree_ceiling"] = float(max(tree))
        if kern:
            out["ds_kernel_ceiling"] = float(max(kern))

        best_algo = max(r2, key=lambda a: (r2[a], a))  # name tiebreak keeps this deterministic
        out["ds_best_r2"] = float(r2[best_algo])
        out["ds_mean_r2"] = float(sum(vals) / len(vals))
        out["ds_std_r2"] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
        out["ds_frac_algos_above_0"] = float(sum(1 for v in vals if v > 0.0) / len(vals))
        try:
            out["ds_best_algo_index"] = float(canon.index(best_algo))
        except ValueError:
            out["ds_best_algo_index"] = -1.0

        times = [v[1] for v in rep.values() if v[1] > 0.0]
        if times:
            out["ds_median_train_time"] = float(np.log1p(statistics.median(times)))

    # --- meta.json --------------------------------------------------------------------------
    meta = _meta(dataset)
    if meta is None:
        out["ds_meta_missing"] = 1.0
    else:
        counts = meta.get("counts") or {}
        rows = _safe_float(counts.get("original_rows"))
        n_feat = _safe_float(meta.get("n_features", counts.get("n_features")))
        out["ds_log_rows"] = float(np.log1p(max(rows, 0.0)))
        out["ds_log_cols"] = float(np.log1p(max(n_feat, 0.0)))
        out["ds_pn_ratio"] = float(n_feat / max(rows, 1.0))

        enc = meta.get("feature_encodings")
        if isinstance(enc, dict) and enc:
            n_cat = sum(1 for v in enc.values()
                        if not (isinstance(v, dict) and v.get("kind") == "numeric"))
            out["ds_cat_frac"] = float(n_cat / len(enc))

    # --- block availability -----------------------------------------------------------------
    h, t = _bank().get(dataset, (None, None))
    out["ds_hyperion_missing"] = 0.0 if h is not None else 1.0
    out["ds_tabpfn_missing"] = 0.0 if t is not None else 1.0
    out["ds_a4168_missing"] = 0.0 if os.path.exists(a4168_path(dataset)) else 1.0

    out["ds_target_only"] = 1.0 if dataset in zero_feature_datasets() else 0.0

    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}


# --------------------------------------------------------------------------------------------
# Dense blocks
# --------------------------------------------------------------------------------------------

def dataset_blocks(datasets, canon: list[str] | None = None) -> dict[str, np.ndarray]:
    """-> dense float32 blocks, one row per entry of `datasets`, IN THAT ORDER.

    Keys: hyperion (N,1536), tabpfn (N,512), a4168 (N,4168),
          difficulty_r2 (N,K), difficulty_time (N,K)   where K == len(canon).

    Missing sources become all-zero rows; the corresponding `ds_*_missing` flag in
    `dataset_features` is what tells a model the row is absent rather than genuinely zero.

    hyperion is NOT L2-normalized upstream (norm ~33.5) and tabpfn_v3 IS. Both are passed through
    unchanged -- normalization is a modelling decision for the consumer, and silently rescaling one
    of them here would make the two blocks incomparable to any other reader of the same table.
    """
    datasets = list(datasets)
    n = len(datasets)
    if canon is None:
        canon = load_canonical_algo_list()
    canon = list(canon)
    canon_t = tuple(canon)
    k = len(canon)

    hyp = np.zeros((n, HYPERION_DIM), dtype=np.float32)
    tab = np.zeros((n, TABPFN_DIM), dtype=np.float32)
    a41 = np.zeros((n, A4168_DIM), dtype=np.float32)
    dr2 = np.zeros((n, k), dtype=np.float32)
    dtm = np.zeros((n, k), dtype=np.float32)

    bank = _bank()
    for i, ds in enumerate(datasets):
        h, t = bank.get(ds, (None, None))
        if h is not None:
            hyp[i] = h
        if t is not None:
            tab[i] = t

        vec = read_a4168(ds)
        if vec is not None:
            a41[i] = vec

        rep = _reports(ds, canon_t)
        if rep:
            for j, algo in enumerate(canon):
                v = rep.get(algo)
                if v is None:
                    continue
                dr2[i, j] = _clip_r2(v[0])
                dtm[i, j] = np.log1p(max(v[1], 0.0))

    return {
        "hyperion": hyp,
        "tabpfn": tab,
        "a4168": a41,
        "difficulty_r2": dr2,
        "difficulty_time": dtm,
    }
