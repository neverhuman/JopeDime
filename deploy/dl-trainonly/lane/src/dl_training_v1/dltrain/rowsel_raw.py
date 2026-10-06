"""Raw-universe row-selection rebuild (owner decisions ROWSEL-TRAINONLY and ROWSEL-SPLIT). Pure functions; the CLI and
all file I/O live in tools/v2_rowsel_raw.py. Nothing here reads S3, the network or the cohort.

Per dataset whose L7a raw source matched (t1raw.match_dataset):

  universe U   every t1raw.read_source data row (the t1raw convention: one raw file, or the one table member of a zip/gzip
               container, parsed by read_source; blank and over-long records are skipped by read_source and never numbered)
               after a ROW-WISE cleanup that reads only the row itself: a row without a target (NaN after t1raw's numeric
               parse) is not a candidate (t1raw.match never matches such a row either), and a row with a categorical cell
               whose text names no key of the meta mapping (t1raw.UNMAPPED_CODE) is not a candidate (its category is lost
               in read_source's coding, and t1raw.refit refuses the code).
  position     (PARTITION, index): PARTITION = "raw" (rowsel.universe_from_raw's default; the file is pinned by sha256 in the
               record), index = the row's 0-based index in read_source's output BEFORE cleanup, so it does not move when
               another row is cleaned out. Row id = rowsel.positional_row_id(h, "raw", index): value-free.
  pre-cap      |U| > PRECAP_ROWS: keep the PRECAP_ROWS rows with the smallest
               sha256(h NUL 'trainonly-rowsel-precap-v1' NUL partition NUL decimal(index)) (ties impossible; index breaks
               them anyway). It receives positions only.
  outer split  5 folds (ROWSEL-SPLIT): rows are grouped by their exact parsed FEATURE bytes (feature_bytes: canonical
               float64 for numeric cells, -0.0 == 0.0 and one NaN; the meta key text for categorical cells; the target is
               never part of it); groups are ordered by sha256(h NUL 'trainonly-rowsel-split-v1' NUL group key hex) and each
               goes whole into the currently smallest fold (ties: lower fold index). No target is read.
  per fold f   quotas = rowsel.fold_quotas(fold sizes, 2000) (largest remainder of 2000*|F_f|/|U|; |U| <= 2000 keeps every
               row); train rows = rowsel.sampler_select fitted on U \\ F_f only; test rows = rowsel.draw_test_rows over the
               positional ids of F_f. The sampler's categorical columns are coded by t1prep.fit_category_codes over the
               keys of the FIT rows (missing = NaN), never by the acquisition's all-row code map.
  values       L7a on the selected rows: t1raw.prepare (outer-train median imputation, outer-train category codes and
               target-copy detection, train-only min-max, inner folds from train rows) -> encoder JTT1; teacher JTT1 = the
               refitted, unscaled rows (as tools/v2_build_l7a.py); A0 = metafeatures.from_raw(encoder JTT1).
  row order    JTT1 train rows in the sampler's output-shuffle order (SelectionFit.output_order; v1's sample order), test
               rows by ascending draw rank.
  gate         per context: rowsel.counterfactual_report (train side and test draw bit-identical when fold-f rows are
               perturbed, replaced or dropped, and draw unchanged when train rows are perturbed) and value_counterfactual
               (outer-test raw values perturbed: scaler, encoder/teacher train rows, inner labels, A0 and the train-side
               refit report bit-identical). A dataset is admitted only when every fold passes.

Numpy and the standard library only (plus the lane's own dltrain modules).
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import heapq
import json
import math
import struct
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from dltrain import jtt1, metafeatures, rowsel, t1prep, t1raw, trainonly

CONTEXT_SCHEMA = "jope.dl-v2-rowsel-raw-context/v1"
IDENTITY_SCHEMA = "jope.dl-rowsel-raw/v1"
SPLIT_SCHEMA = "rowsel-raw-split-v1"
PRECAP_DOMAIN = "trainonly-rowsel-precap-v1"
SPLIT_DOMAIN = "trainonly-rowsel-split-v1"
VALUE_CF_SALT = "rowsel-raw-cf-v1"
PRECAP_ROWS = 200_000
OUTER_FOLDS = 5
PARTITION = "raw"
BUDGET = rowsel.DEFAULT_BUDGET
VIEW_TAG = "rowselraw1"
ROW_ORDER = "train: sampler output-shuffle order (SelectionFit.output_order); test: ascending rowsel.draw_rank"
VALUE_PERTURBATIONS = ("features", "targets", "missingness", "categories", "outlier", "drop", "replace_with_train_copies")


class RowselRawError(ValueError):
    """A dataset that cannot be rebuilt; `reason` is a short stable code for the receipts."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _u64(values) -> bytes:
    return np.asarray(values, dtype="<u8").tobytes()


# ------------------------------------------------------------------------------------------------- universe

def _cat_specs(specs):
    return [s for s in specs[:-1] if s["kind"] == "categorical_codes"]


def _num_index(specs):
    return [s["index"] for s in specs[:-1] if s["kind"] != "categorical_codes"]


def clean_rows(values, specs) -> tuple[np.ndarray, dict]:
    """Row-wise candidate mask over read_source rows: a finite target and no UNMAPPED categorical cell. Reads each row
    alone (no statistic over rows). Any other non-key code is a bug upstream and raises."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != len(specs):
        raise RowselRawError("values_width", f"{v.shape} for {len(specs)} columns")
    no_target = ~np.isfinite(v[:, -1])
    unmapped = np.zeros(v.shape[0], dtype=bool)
    for s in _cat_specs(specs):
        col = v[:, s["index"]]
        valid = np.isin(col, np.asarray(sorted(s["inverse"]) + [t1raw.MISSING_CODE, t1raw.UNMAPPED_CODE], dtype=np.float64))
        if not bool(np.all(valid)):
            raise RowselRawError("values_bad_code", s["name"])
        unmapped |= col == t1raw.UNMAPPED_CODE
    keep = ~no_target & ~unmapped
    return keep, {"data_rows": int(v.shape[0]), "no_target": int(no_target.sum()),
                  "unmapped_category_with_target": int((unmapped & ~no_target).sum()), "candidates": int(keep.sum()),
                  "rule": "row-wise: finite target and no categorical cell outside the meta mapping (t1raw.UNMAPPED_CODE)"}


def precap_rank(dataset_hash: str, index: int, partition: str = PARTITION) -> bytes:
    return hashlib.sha256(dataset_hash.encode() + b"\0" + PRECAP_DOMAIN.encode() + b"\0" + partition.encode() + b"\0"
                          + str(int(index)).encode()).digest()


def precap(dataset_hash: str, positions, cap: int = PRECAP_ROWS, partition: str = PARTITION) -> tuple[np.ndarray, dict]:
    """Value-free pre-cap: positions only. |positions| <= cap keeps every one; otherwise the cap smallest precap_rank.
    Streams the ranks through a bounded heap, so a 12M-row universe never holds 12M digests."""
    pos = np.asarray(positions, dtype=np.int64).reshape(-1)
    n = int(pos.size)
    if np.unique(pos).size != n:
        raise RowselRawError("precap_positions_not_unique")
    rep = {"cap": int(cap), "rows_before": n, "domain": PRECAP_DOMAIN, "partition": partition,
           "rule": "keep the cap smallest sha256(h NUL domain NUL partition NUL decimal(index))"}
    if n <= cap:
        kept = np.sort(pos)
        return kept, dict(rep, applied=False, rows_after=n)

    def ranks():
        for start in range(0, n, 1 << 16):
            for p in pos[start:start + (1 << 16)].tolist():
                yield precap_rank(dataset_hash, p, partition), p
    kept = np.sort(np.fromiter((p for _, p in heapq.nsmallest(int(cap), ranks())), dtype=np.int64, count=int(cap)))
    return kept, dict(rep, applied=True, rows_after=int(kept.size))


def _key_bytes(key) -> bytes:
    if key is None:
        return b"\x00"
    kb = str(key).encode("utf-8")
    return b"\x01" + struct.pack("<I", len(kb)) + kb


def _canonical_u64(block) -> np.ndarray:
    b = np.array(block, dtype="<f8", copy=True, order="C")
    b[np.isnan(b)] = np.nan
    b = b + 0.0                                                     # -0.0 -> 0.0
    return np.ascontiguousarray(b).view("<u8")


def feature_bytes(values, specs) -> list[bytes]:
    """Exact parsed feature bytes per row, the target excluded: canonical float64 of the numeric/datetime cells (spec
    order), then per categorical column (spec order) the meta key text (length-prefixed) or a missing marker."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    num = _canonical_u64(v[:, _num_index(specs)]) if _num_index(specs) else np.zeros((n, 0), dtype="<u8")
    out = [num[i].tobytes() for i in range(n)]
    for s in _cat_specs(specs):
        col = v[:, s["index"]].astype(np.int64)
        table = {c: _key_bytes(None if c == t1raw.MISSING_CODE else s["inverse"][c]) for c in np.unique(col).tolist()}
        out = [a + table[c] for a, c in zip(out, col.tolist())]
    return out


def group_keys(values, specs) -> list[str]:
    return [sha(b) for b in feature_bytes(values, specs)]


def row_bytes(values, specs) -> list[bytes]:
    """Sampler priority bytes: the row's own feature bytes and canonical target (rowsel hashes them for fit rows only)."""
    y = _canonical_u64(np.asarray(values, dtype=np.float64)[:, -1:])
    return [f + b"\x02" + y[i].tobytes() for i, f in enumerate(feature_bytes(values, specs))]


@dataclasses.dataclass(frozen=True, eq=False)
class Prep:
    """One dataset's universe after cleanup and pre-cap: positions ascending, values aligned with them."""
    dataset_hash: str
    specs: list
    positions: np.ndarray
    values: np.ndarray
    row_ids: tuple
    row_bytes: tuple
    report: dict

    def rows_of(self, positions) -> np.ndarray:
        p = np.asarray(positions, dtype=np.int64)
        i = np.searchsorted(self.positions, p)
        if p.size and (int(i.max()) >= self.positions.size or not np.array_equal(self.positions[i], p)):
            raise RowselRawError("position_not_in_universe")
        return i


def prepare_universe(dataset_hash: str, values, specs, cap: int = PRECAP_ROWS) -> Prep:
    """read_source values (every data row, t1raw coding) -> the candidate universe: row-wise cleanup, then the value-free
    pre-cap. The caller may drop `values` afterwards; Prep holds only the kept rows."""
    keep, clean_rep = clean_rows(values, specs)
    kept, cap_rep = precap(dataset_hash, np.flatnonzero(keep), cap)
    v = np.ascontiguousarray(np.asarray(values, dtype=np.float64)[kept])
    v.setflags(write=False)
    ids = tuple(rowsel.positional_row_id(dataset_hash, PARTITION, p) for p in kept.tolist())
    rb = tuple(row_bytes(v, specs))
    rep = {"cleanup": clean_rep, "precap": cap_rep, "rows": int(kept.size), "partition": PARTITION,
           "positions_sha256": sha(_u64(kept)), "position_basis": "0-based t1raw.read_source data-row index, before cleanup"}
    kept.setflags(write=False)
    return Prep(dataset_hash, list(specs), kept, v, ids, rb, rep)


# ------------------------------------------------------------------------------------------------- outer split

def split_rank(dataset_hash: str, group_key: str) -> bytes:
    return hashlib.sha256(dataset_hash.encode() + b"\0" + SPLIT_DOMAIN.encode() + b"\0" + group_key.encode()).digest()


def outer_split(dataset_hash: str, positions, values, specs, folds: int = OUTER_FOLDS) -> dict[int, list[int]]:
    """ROWSEL-SPLIT: exact-duplicate feature groups, ordered by split_rank, each into the currently smallest fold (ties to
    the lower fold). Reads feature bytes only; the target column is never touched."""
    pos = [int(p) for p in np.asarray(positions, dtype=np.int64).reshape(-1)]
    keys = group_keys(values, specs)                                # feature columns only (feature_bytes)
    groups: dict[str, list[int]] = {}
    for p, g in zip(pos, keys):
        groups.setdefault(g, []).append(p)
    sizes = [0] * folds
    out: dict[int, list[int]] = {f: [] for f in range(folds)}
    for g in sorted(groups, key=lambda k: (split_rank(dataset_hash, k), k)):
        f = min(range(folds), key=lambda i: (sizes[i], i))
        out[f].extend(groups[g])
        sizes[f] += len(groups[g])
    return {f: sorted(v) for f, v in out.items()}


def split_sha256(dataset_hash: str, folds: Mapping[int, Sequence[int]]) -> str:
    return sha(canon({"schema": SPLIT_SCHEMA, "dataset_hash": dataset_hash, "partition": PARTITION,
                      "folds": [[int(p) for p in folds[f]] for f in sorted(folds)]}))


def split_report(prep: Prep, folds: Mapping[int, Sequence[int]]) -> dict:
    """Split identity plus its structural checks: a partition of the universe, and no duplicate group in two folds."""
    fold_of = {}
    for f, v in folds.items():
        for p in v:
            if p in fold_of:
                raise RowselRawError("split_not_partition", f"position {p} in folds {fold_of[p]} and {f}")
            fold_of[p] = f
    if sorted(fold_of) != prep.positions.tolist():
        raise RowselRawError("split_not_partition", "folds do not cover the universe exactly")
    keys = group_keys(prep.values, prep.specs)
    seen: dict[str, int] = {}
    straddle = 0
    sizes: dict[str, int] = {}
    for p, g in zip(prep.positions.tolist(), keys):
        f = fold_of[p]
        if seen.setdefault(g, f) != f:
            straddle += 1
        sizes[g] = sizes.get(g, 0) + 1
    return {"schema": SPLIT_SCHEMA, "sha256": split_sha256(prep.dataset_hash, folds), "folds": len(folds),
            "fold_sizes": [len(folds[f]) for f in sorted(folds)], "groups": len(seen), "largest_group": max(sizes.values()),
            "duplicate_rows": int(sum(n - 1 for n in sizes.values())), "groups_straddling_folds": straddle,
            "domain": SPLIT_DOMAIN, "rule": "exact parsed-feature groups (target excluded), ordered by sha256(h NUL domain "
            "NUL group key), greedy into the smallest fold, ties to the lower fold; no target stratification"}


def split_target_check(split_fn: Callable, prep: Prep, salt: str = VALUE_CF_SALT) -> dict:
    """The split must not read targets: re-run split_fn with the target column permuted, replaced and given an outlier,
    and list every variant whose fold assignment differs. split_fn(h, positions, values, specs) -> {fold: positions}."""
    base = split_fn(prep.dataset_hash, prep.positions, prep.values, prep.specs)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{salt}|split".encode()).digest()[:8], "little"))
    variants = {}
    y = prep.values[:, -1]
    for kind in ("permuted", "replaced", "outlier", "constant"):
        v = np.array(prep.values, dtype=np.float64, copy=True)
        if kind == "permuted":
            v[:, -1] = y[rng.permutation(y.size)]
        elif kind == "replaced":
            v[:, -1] = rng.normal(1e6, 1e5, size=y.size)
        elif kind == "outlier":
            v[: max(1, y.size // 10), -1] = -1e12
        else:
            v[:, -1] = 0.0
        got = split_fn(prep.dataset_hash, prep.positions, v, prep.specs)
        variants[kind] = {f: list(map(int, got[f])) for f in sorted(got)} != {f: list(map(int, base[f])) for f in sorted(base)}
    return {"variants_differing": sorted(k for k, d in variants.items() if d), "target_invariant": not any(variants.values())}


# ------------------------------------------------------------------------------------------------- per fold

def sampler_matrix(prep: Prep, fit_mask) -> tuple[np.ndarray, dict]:
    """Feature matrix the sampler sees for one fold: numeric cells as parsed (NaN = missing); categorical columns coded by
    t1prep.fit_category_codes over the keys present in the FIT rows only (missing = NaN; keys absent from the fit rows,
    which only non-fit rows can hold, get len(codes): the sampler never reads non-fit rows)."""
    v = prep.values
    fit_mask = np.asarray(fit_mask, dtype=bool)
    X = np.array(v[:, :-1], dtype=np.float64, copy=True)
    maps = {}
    for s in _cat_specs(prep.specs):
        j = s["index"]
        col = v[:, j].astype(np.int64)
        fit_codes = [c for c in np.unique(col[fit_mask]).tolist() if c != t1raw.MISSING_CODE]
        cmap = t1prep.fit_category_codes([s["inverse"][c] for c in fit_codes])
        lut = {c: (math.nan if c == t1raw.MISSING_CODE else float(cmap.get(s["inverse"][c], len(cmap))))
               for c in np.unique(col).tolist()}
        X[:, j] = np.fromiter((lut[c] for c in col.tolist()), dtype=np.float64, count=col.size)
        maps[s["name"]] = sha(repr(sorted(cmap.items())).encode())
    return X, maps


def fold_universe(prep: Prep, fit_mask) -> tuple[rowsel.Universe, dict]:
    X, maps = sampler_matrix(prep, fit_mask)
    return rowsel.Universe(prep.dataset_hash, prep.positions, prep.row_ids, X, prep.values[:, -1], prep.row_bytes), maps


def l7a_artifacts(V, specs, n_train: int, prepare: Callable = t1raw.prepare) -> dict:
    """tools/v2_build_l7a.build_l7a on rows V (outer-train rows first, then outer-test rows), in memory."""
    R = list(range(int(n_train)))
    T = list(range(int(n_train), int(V.shape[0])))
    split = prepare(V, specs, R, T)
    fitted = split["fitted"]
    teacher = trainonly.jtt1_bytes(fitted[R], fitted[T].reshape(-1, fitted.shape[1]), split["inner_labels"])
    a0 = metafeatures.from_raw(split["jtt1"])[0]
    return {"encoder_raw": split["jtt1"], "teacher_raw": teacher, "a0": a0,
            "scaler_sha256": sha(trainonly.scaler_bytes(split)),
            "inner_labels_sha256": sha(np.asarray(split["inner_labels"], np.uint8).tobytes()),
            "inner_loads": split["inner_cv"]["loads"], "t1raw": split["t1raw"]}


def _refit_train_report(rep: Mapping[str, Any]) -> dict:
    imp = rep["imputation"]
    return {"kept_columns": rep["kept_columns"], "target_copy_dropped": rep["target_copy_dropped"],
            "imputed_train": imp["imputed_cells"]["train"], "no_train_value_columns": imp["no_train_value_columns"],
            "train_medians": {k: [c["train_missing"], c["train_median"]] for k, c in sorted(imp["columns"].items())
                              if c["train_missing"]},
            "categorical_train": {k: c["train"] for k, c in sorted(rep["categorical"].items())}}


def values_train_side(art: Mapping[str, Any]) -> dict:
    return {"scaler": art["scaler_sha256"], "encoder_train_rows": jtt1.train_rows_sha256(art["encoder_raw"]),
            "teacher_train_rows": jtt1.train_rows_sha256(art["teacher_raw"]), "inner_labels": art["inner_labels_sha256"],
            "a0": sha(art["a0"]), "refit_train": sha(canon(_refit_train_report(art["t1raw"])))}


def perturb_test_rows(V, specs, n_train: int, kind: str, salt: str = VALUE_CF_SALT) -> np.ndarray:
    """Rows n_train.. (the outer-test rows of one context) changed; rows before n_train are untouched."""
    W = np.array(V, dtype=np.float64, copy=True)
    n = W.shape[0]
    T = np.arange(int(n_train), n)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{salt}|{kind}".encode()).digest()[:8], "little"))
    num = _num_index(specs)
    cats = _cat_specs(specs)
    if kind == "features":
        if num:
            block = W[np.ix_(T, num)]
            W[np.ix_(T, num)] = np.where(np.isnan(block), block, rng.normal(50.0, 1e3, size=block.shape))
        for s in cats:
            codes = np.asarray(sorted(s["inverse"]), dtype=np.float64)
            col = W[T, s["index"]]
            W[T, s["index"]] = np.where(col == t1raw.MISSING_CODE, col, rng.choice(codes, size=T.size))
    elif kind == "targets":
        W[T, -1] = rng.normal(1e7, 1e6, size=T.size)
    elif kind == "missingness":
        if num:
            block = W[np.ix_(T, num)]
            flip = rng.random(block.shape) < 0.35
            W[np.ix_(T, num)] = np.where(np.isnan(block), 0.5, np.where(flip, np.nan, block))
        for s in cats:
            col = W[T, s["index"]]
            flip = rng.random(T.size) < 0.35
            W[T, s["index"]] = np.where(col == t1raw.MISSING_CODE, float(min(s["inverse"])),
                                        np.where(flip, float(t1raw.MISSING_CODE), col))
    elif kind == "categories":
        for s in cats:                                   # test-only categories: the highest meta code everywhere
            W[T, s["index"]] = float(max(s["inverse"]))
        if num and T.size:
            W[T[0], num[0]] = -7.0
    elif kind == "outlier":
        if T.size:
            if num:
                W[T[0], num] = 1e12
            W[T[0], -1] = -1e12
    elif kind == "drop":
        keep = np.ones(n, dtype=bool)
        keep[T[::2]] = False
        return W[keep]
    elif kind == "replace_with_train_copies":
        if T.size and n_train:
            W[T] = V[rng.integers(0, int(n_train), size=T.size)]
            W[T, -1] = W[T, -1] * -3.0 + 11.0
    else:
        raise RowselRawError("unknown_perturbation", kind)
    return W


def value_counterfactual(V, specs, n_train: int, base: Mapping[str, Any] | None = None,
                         prepare: Callable = t1raw.prepare) -> dict:
    """Rebuild the L7a artifacts with the outer-test raw values perturbed; list every train-side field that is not
    bit-identical (scaler, encoder and teacher train rows, inner labels, A0, train-side refit report)."""
    base = base if base is not None else l7a_artifacts(V, specs, n_train, prepare)
    want = values_train_side(base)
    cases = {}
    for kind in VALUE_PERTURBATIONS:
        W = perturb_test_rows(V, specs, n_train, kind)
        got = values_train_side(l7a_artifacts(W, specs, n_train, prepare))
        cases[kind] = sorted(k for k in want if got[k] != want[k])
    return {"cases": cases, "train_side_invariant": not any(cases.values()), "train_side": want}


def _cached_select(select_fn: Callable, universe, view):
    def fn(u, parent_folds, fold, budget=BUDGET, seed=None, quotas=None):
        if u is universe and quotas is None and seed is None:
            return view
        return select_fn(u, parent_folds, fold, budget, seed, quotas)
    return fn


def build_fold(prep: Prep, folds: Mapping[int, Sequence[int]], fold: int, universe_ref: Mapping[str, Any],
               split_ref: Mapping[str, Any], select_fn: Callable = rowsel.select_view, prepare: Callable = t1raw.prepare,
               counterfactual: bool = True) -> dict:
    """One context in memory: the selection, its identity and view_id, the L7a artifacts, and both counterfactual gates.
    Nothing is written. result['passed'] is the admission bit."""
    h = prep.dataset_hash
    held = np.isin(prep.positions, np.asarray(folds[fold], dtype=np.int64))
    U, cat_maps = fold_universe(prep, ~held)
    view = select_fn(U, folds, fold, BUDGET)
    id_of = dict(zip(prep.positions.tolist(), prep.row_ids))
    train_order = [int(p) for p in view.train_fit.output_order]
    if sorted(train_order) != sorted(view.train_positions):
        train_order = sorted(int(p) for p in view.train_positions)       # a selector without an output order (controls)
    test_order = sorted((int(p) for p in view.test_positions), key=lambda p: (rowsel.draw_rank(h, id_of[p]), p))
    V = np.ascontiguousarray(prep.values[prep.rows_of(train_order + test_order)])
    art = l7a_artifacts(V, prep.specs, len(train_order), prepare)
    body = {"schema": IDENTITY_SCHEMA, "route": "raw", "dataset_hash": h, "fold": int(fold), "budget": BUDGET,
            "universe": dict(universe_ref), "split": {"schema": split_ref["schema"], "sha256": split_ref["sha256"],
                                                      "fold_sizes": split_ref["fold_sizes"]},
            "view_identity": view.identity, "row_order": ROW_ORDER, "values_mode": "L7a"}
    identity = sha(canon(body))
    out = {"fold": int(fold), "view": view, "identity": identity, "identity_body": body,
           "view_id": f"{h}__r0__f{int(fold)}__{VIEW_TAG}_{identity[:12]}", "train_order": train_order,
           "test_order": test_order, "values": V, "artifacts": art, "sampler_category_maps": cat_maps}
    if counterfactual:
        rcf = rowsel.counterfactual_report(_cached_select(select_fn, U, view), U, folds, fold, BUDGET)
        vcf = value_counterfactual(V, prep.specs, len(train_order), art, prepare)
        out["rowsel_counterfactual"] = rcf
        out["value_counterfactual"] = vcf
        reasons = []
        if rcf["base_view_identity"] != view.identity:
            reasons.append("counterfactual_base_differs")
        if not rcf["train_side_invariant"]:
            reasons.append("rowsel_train_side_not_invariant")
        if not rcf["test_draw_invariant"]:
            reasons.append("rowsel_test_draw_not_invariant")
        if not vcf["train_side_invariant"]:
            reasons.append("values_train_side_not_invariant")
        out["failures"] = reasons
        out["passed"] = not reasons
    else:
        out["failures"], out["passed"] = ["counterfactual_not_run"], False
    return out


def context_files(result: Mapping[str, Any]) -> dict[str, bytes]:
    art = result["artifacts"]
    return {"encoder.jtt1.gz": gzip.compress(art["encoder_raw"], compresslevel=3, mtime=0),
            "teacher.jtt1.gz": gzip.compress(art["teacher_raw"], compresslevel=3, mtime=0),
            "a0.f32": art["a0"]}


def counterfactual_summary(result: Mapping[str, Any]) -> dict:
    rcf, vcf = result.get("rowsel_counterfactual"), result.get("value_counterfactual")
    out = {"passed": bool(result.get("passed")), "failures": list(result.get("failures") or [])}
    if rcf:
        out["rowsel"] = {"schema": rcf["schema"], "train_side_invariant": rcf["train_side_invariant"],
                         "test_draw_invariant": rcf["test_draw_invariant"], "base_view_identity": rcf["base_view_identity"],
                         "quotas": rcf["quotas"], "cases": sorted(rcf["cases"]),
                         "differing": {k: c for k, c in rcf["cases"].items()
                                       if c.get("train_side_differs") or c.get("test_draw_differs")}}
    if vcf:
        out["values"] = {"train_side_invariant": vcf["train_side_invariant"], "cases": sorted(vcf["cases"]),
                         "differing": {k: v for k, v in vcf["cases"].items() if v}, "train_side": vcf["train_side"]}
    return out


def context_record(result: Mapping[str, Any], prep: Prep, files: Mapping[str, bytes], common: Mapping[str, Any]) -> dict:
    """context.json body (without utc_us). `common` carries the dataset-level fields (raw source, match, epoch0_replaced,
    code_sha256, t1_meta_sha256, ...)."""
    view, art = result["view"], result["artifacts"]
    R, T = result["train_order"], result["test_order"]
    fmeta = {"encoder.jtt1.gz": {"sha256": sha(files["encoder.jtt1.gz"]), "raw_sha256": sha(art["encoder_raw"]),
                                 "bytes": len(files["encoder.jtt1.gz"])},
             "teacher.jtt1.gz": {"sha256": sha(files["teacher.jtt1.gz"]), "raw_sha256": sha(art["teacher_raw"]),
                                 "bytes": len(files["teacher.jtt1.gz"])},
             "a0.f32": {"sha256": sha(files["a0.f32"]), "bytes": len(files["a0.f32"])}}
    tr_sha, te_sha = sha(json.dumps(R).encode()), sha(json.dumps(T).encode())
    ctx_body = {"schema": CONTEXT_SCHEMA, "view_id": result["view_id"], "dataset_hash": prep.dataset_hash,
                "fold": result["fold"], "rowsel_identity": result["identity"], "files": {k: v.get("raw_sha256", v["sha256"])
                                                                                       for k, v in fmeta.items()},
                "scaler_sha256": art["scaler_sha256"], "inner_labels_sha256": art["inner_labels_sha256"],
                "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha}
    rec = dict(common)
    rec.update({
        "schema": CONTEXT_SCHEMA, "view_id": result["view_id"], "dataset_hash": prep.dataset_hash, "fold": result["fold"],
        "context_sha256": sha(canon(ctx_body)),
        "context_sha256_basis": "sha256 of canonical JSON {schema, view_id, dataset_hash, fold, rowsel_identity, files "
                                "(raw sha256), scaler, inner labels, outer train/test indices}; a NEW context identity, "
                                "not an epoch-0 context_sha256",
        "values_mode": "L7a", "published": False, "row_selection_leak": ["clean"],
        "outer_train_indices_sha256": tr_sha, "outer_test_indices_sha256": te_sha,
        "indices_basis": "json list of raw positions (t1raw.read_source row index) in JTT1 row order",
        "n_train": len(R), "n_test": len(T), "files": fmeta,
        "scaler_sha256": art["scaler_sha256"], "inner_labels_sha256": art["inner_labels_sha256"],
        "inner_loads": art["inner_loads"], "t1raw": art["t1raw"],
        "values_sha256": sha(np.ascontiguousarray(result["values"], "<f8").tobytes()),
        "rowsel": {"route": "raw", "identity": result["identity"], "identity_body": result["identity_body"],
                   "view_identity": view.identity, "universe": dict(common["rowsel_universe"]),
                   "split": dict(common["rowsel_split"]), "quotas": {"n_train": view.quotas[0], "n_test": view.quotas[1]},
                   "budget": view.budget, "seed": view.seed, "port": rowsel.PORT, "draw_domain": rowsel.DRAW_DOMAIN,
                   "train_fit_identity": view.train_fit.identity, "passthrough": view.train_fit.passthrough,
                   "selected_positions_sha256": view.train_fit.fingerprints["selected_positions_sha256"],
                   "test_draw_sha256": view.test_draw_sha256, "row_order": ROW_ORDER,
                   "sampler_category_maps_sha256": result["sampler_category_maps"],
                   "code_sha256": dict(common["code_sha256"]), "outer_split_changed": True,
                   "counterfactual": counterfactual_summary(result)},
    })
    rec.pop("rowsel_universe", None)
    rec.pop("rowsel_split", None)
    return rec
