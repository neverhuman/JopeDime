"""CORESET-NONNUMERIC: raw text columns the acquisition tool label-coded before T1 (owner ruling CORESET-NONNUMERIC,
ENG:watcher delegated, 2026-10-06T00:02Z).

Finding (frozen all_rows_coreset datasets of models.tools.acquire_regression_extra_sources; the tool's source is not on
this host, so the rule is established by content match against T1, never by reading the tool):

  staging    a source column pandas read as text was replaced, before meta.json was written, by the RANK of each cell's
             text among the column's sorted (Python str order) distinct non-missing texts over EVERY source data row.
             pandas NA text (t1raw.NA_STRINGS, matched exactly; 'None' included, pandas >= 2) stayed missing. The meta
             records no mapping for it.
  meta       the coded column was then processed like any number: feature_encodings kind numeric (median impute of the
             codes, min-max over the cleaned rows), or categorical_codes keyed by the code's float text ('0.0', '1.0',
             ..., '__MISSING__').
  exception  a column that is mostly numbers with some text ('5;5' among 1..10) was read as numbers, text -> missing:
             t1raw's own numeric reading. A text column the content match blames is re-read that way once, when the
             meta's own range is a range of its numbers.

The staging code is an ALL-ROW FITTED encoder (a category held only by some rows shifts the code of every category sorted
after it). It is used for one thing only: to re-identify every T1 row in the raw source with t1raw.match, unchanged (same
tolerance, one value class per T1 row, one-to-one). It never becomes a value.

The universe handed to the raw builder (tools/v2_rowsel_raw.run_dataset `matcher` hook: every read_source data row) turns
each text column into a categorical_codes spec whose codes are ROW-WISE pointers to the cell's own text: pointer(text) =
48 bits of sha256, no statistic over rows; missing -> t1raw.MISSING_CODE. Downstream code maps a pointer back to its text
(spec['inverse']) and fits t1prep.fit_category_codes on outer-train rows only (rowsel_raw.sampler_matrix, t1raw.refit):
missing and outer-train-unseen text share one extra code, as for every other category. A test row's text therefore
cannot move a train row's stored value, sampler code or refitted code.

Ambiguity: t1raw refuses a T1 row that fits two raw value classes. That happens where a raw row with a missing cell and
a raw row holding the impute value are otherwise equal (T1's own imputation made them indistinguishable).
ambiguity_blocks() grades such a dataset without widening any tolerance:
  matched_set_level  a capacity-respecting one-to-one assignment of every ambiguous T1 row exists AND it consumes every
                     raw row of every candidate class: the multiset of raw rows behind T1 is fixed (only the pairing
                     inside a block of indistinguishable rows is not).
  matched_covered    the assignment exists but candidate classes keep spare raw rows: every T1 row is in the raw source,
                     which of two indistinguishable raw rows it came from is not determined.
Both are reported separately from t1raw's strict 'matched'; the caller decides which tiers to admit.

Numpy, the standard library and the lane's own t1raw / t1prep (imported read-only).
"""
from __future__ import annotations

import copy
import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dltrain import t1prep, t1raw

SCHEMA = "jope.dl-nonnumeric/v1"
RANK = "staging_rank"            # text column: acquisition rank code (match) / row-wise text pointer (universe)
NUMERIC = "numeric"              # text column read as numbers, text -> missing (t1raw's reading)
POINTER_DOMAIN = b"jope-nonnumeric-pointer-v1\0"
POINTER_BYTES = 6                # 48 bits: exact in float64, collisions refused
STAGING_RULE = ("code = rank of the cell text among sorted (Python str order) distinct non-NA texts of the column over "
                "every read_source data row; NA = exact t1raw.NA_STRINGS")
UNIVERSE_RULE = ("text column -> categorical_codes; code = pointer(text) (row-wise, sha256 48 bits), NA -> MISSING_CODE; "
                 "downstream codes fitted on outer-train rows only (t1prep.fit_category_codes)")


class NonnumericError(t1raw.T1RawError):
    """A dataset this module cannot reconstruct; `reason` is a short stable code (caught with T1RawError)."""


def is_na(text: str) -> bool:
    """pandas read_csv default NA text, matched exactly (pandas does not strip a field before the NA test)."""
    return text in t1raw.NA_STRINGS


def pointer(text: str) -> int:
    """Row-wise code of one text cell: depends on this text only."""
    return int.from_bytes(hashlib.sha256(POINTER_DOMAIN + text.encode("utf-8", "surrogatepass")).digest()[:POINTER_BYTES], "big")


def _numeric_keyed(spec: Mapping[str, Any]) -> bool:
    keys = [k for k in spec["inverse"].values() if k != t1prep.MISSING_KEY]
    return bool(keys) and all(t1prep._num(k) is not None for k in keys)


def target_name_conflict(meta: Mapping[str, Any]) -> dict | None:
    """The meta names two different target columns (top level vs acquisition record), or None."""
    top = meta.get("target_original_name") or meta.get("source_target_column_original")
    acq = (meta.get("acquisition") or {}).get("source_target_column_original")
    if isinstance(acq, str) and acq and isinstance(top, str) and top and acq != top:
        return {"used_by_t1raw": top, "acquisition_record": acq}
    return None


# ------------------------------------------------------------------------------------------------- reading

def text_columns(specs: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> list[str]:
    """CANDIDATE text columns: a numeric spec (not datetime-derived) with non-numeric, non-bool text (read_source's
    nonnumeric_cells minus bool_columns), or a categorical spec keyed by numbers ('0.0', '1.0', ...) whose raw cells name
    no key (unmapped_category_cells). confirm_text() keeps a categorical candidate only when an unmapped cell is text: a
    number absent from a value-keyed map ('18' for keys '12.0', '20.0') is source drift, not text, and stays t1raw's."""
    nn = report.get("nonnumeric_cells") or {}
    bools = set(report.get("bool_columns") or [])
    unm = report.get("unmapped_category_cells") or {}
    out = []
    for s in specs:
        if s.get("target"):
            continue
        if s["kind"] == "numeric" and s.get("derive") is None and s["name"] in nn and s["name"] not in bools:
            out.append(s["name"])
        elif s["kind"] == "categorical_codes" and s["name"] in unm and _numeric_keyed(s):
            out.append(s["name"])
    return out


def read_text(raw_path, meta: Mapping[str, Any], specs: Sequence[Mapping[str, Any]], report: Mapping[str, Any],
              names: Sequence[str]) -> dict[str, list[str]]:
    """Raw text of the named spec columns, one entry per read_source data row (same file layer, encoding, delimiter,
    header record, implicit index; blank and over-long records skipped, short records padded with '')."""
    hint = t1raw._container(meta)
    factory, layers = t1raw._opener(Path(raw_path), hint["member"])
    if layers != report["layers"]:
        raise NonnumericError("text_read_drift", f"layers {layers} != {report['layers']}")
    by_name = {s["name"]: s for s in specs}
    csv.field_size_limit(2 ** 31 - 1)
    stream = t1raw._text(factory, report["encoding"])
    try:
        reader = csv.reader(stream, delimiter=report["delimiter"])
        for _ in range(report["header_record"]):
            next(reader)
        header = next(reader)
        got, missing = t1raw._resolve(header, [by_name[n]["source_name"] for n in names])
        if missing or len(header) != report["header_width"]:
            raise NonnumericError("text_read_drift", f"header: missing {missing[:4]}, width {len(header)}")
        shift = 1 if report["implicit_index"] else 0
        width = report["header_width"] + shift
        idx = [(n, got[by_name[n]["source_name"]] + shift) for n in names]
        cols: dict[str, list[str]] = {n: [] for n in names}
        for rec in reader:
            k = len(rec)
            if k == 0 or (k == 1 and rec[0] == ""):
                continue
            if k > width:
                continue
            for n, i in idx:
                cols[n].append(rec[i] if i < k else "")
    finally:
        stream.close()
    for n in names:
        if len(cols[n]) != report["data_rows"]:
            raise NonnumericError("text_read_drift", f"{n}: {len(cols[n])} rows != read_source {report['data_rows']}")
    return cols


def holds_text(cells: Sequence[str]) -> bool:
    """Some non-NA cell is not a number (t1raw's numeric parse fails)."""
    return any(not is_na(t) and math.isnan(t1raw._to_float(t)) for t in cells)


def confirm_text(specs: Sequence[Mapping[str, Any]], text: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Candidates -> text columns: every numeric-spec candidate; a categorical candidate only when it holds text."""
    by_name = {s["name"]: s for s in specs}
    return {n: cells for n, cells in text.items() if by_name[n]["kind"] == "numeric" or holds_text(cells)}


def check_alignment(values, specs: Sequence[Mapping[str, Any]], text: Mapping[str, Sequence[str]]) -> None:
    """Every text cell must reproduce read_source's own value for its row (numeric parse or category lookup), so the
    text rows are read_source's rows."""
    v = np.asarray(values, dtype=np.float64)
    by_name = {s["name"]: s for s in specs}
    for n, cells in text.items():
        s = by_name[n]
        if s["kind"] == "numeric":
            want = np.fromiter((t1raw._to_float(t) for t in cells), dtype=np.float64, count=len(cells))
        else:
            lookup = t1raw._category_lookup(s)
            want = np.fromiter((float(lookup(t)) for t in cells), dtype=np.float64, count=len(cells))
        got = v[:, s["index"]]
        if not bool(np.all((got == want) | (np.isnan(got) & np.isnan(want)))):
            raise NonnumericError("text_read_drift", f"{n}: text rows differ from read_source values")


# ------------------------------------------------------------------------------------------------- the two codings

def staging_rank(cells: Sequence[str]) -> dict[str, int]:
    """The acquisition's all-row code map of one text column (MATCH ONLY: it is fitted on every row)."""
    return {k: i for i, k in enumerate(sorted({t for t in cells if not is_na(t)}))}


def numeric_reading_fits(spec: Mapping[str, Any], cells: Sequence[str]) -> bool:
    """The meta's range [lo, hi] is a range of the column's own numbers (each end is a parsed cell value or the impute
    value): the acquisition may have read this column as numbers."""
    if spec["kind"] != "numeric":
        return False
    nums = {v for v in (t1raw._to_float(t) for t in cells) if not math.isnan(v)}
    if spec.get("impute_value") is not None:
        nums.add(float(spec["impute_value"]))
    tol = t1raw.MATCH_TOL_REL * max(abs(spec["hi"] - spec["lo"]), 1e-300) + t1raw.F64_SLACK * max(abs(spec["lo"]), abs(spec["hi"]))
    return bool(nums) and any(abs(x - spec["lo"]) <= tol for x in nums) and any(abs(x - spec["hi"]) <= tol for x in nums)


def match_matrix(values, specs: Sequence[Mapping[str, Any]], text: Mapping[str, Sequence[str]],
                 readings: Mapping[str, str]) -> np.ndarray:
    """read_source values with every RANK text column replaced by the acquisition's code: a numeric spec gets the rank
    (NaN for NA); a numerically keyed categorical spec gets the meta code whose key equals the rank (MISSING_CODE for NA,
    UNMAPPED_CODE for a rank the mapping lacks). NUMERIC columns keep read_source's reading."""
    V = np.array(values, dtype=np.float64, copy=True)
    by_name = {s["name"]: s for s in specs}
    for n, cells in text.items():
        if readings[n] != RANK:
            continue
        s = by_name[n]
        rank = staging_rank(cells)
        if s["kind"] == "numeric":
            col = [math.nan if is_na(t) else float(rank[t]) for t in cells]
        else:
            key_code = {t1prep._num(k): c for c, k in s["inverse"].items() if k != t1prep.MISSING_KEY}
            col = [float(t1raw.MISSING_CODE) if is_na(t) else float(key_code.get(float(rank[t]), t1raw.UNMAPPED_CODE)) for t in cells]
        V[:, s["index"]] = col
    return V


def universe(values, specs: Sequence[Mapping[str, Any]], text: Mapping[str, Sequence[str]],
             readings: Mapping[str, str]) -> tuple[np.ndarray, list[dict], dict]:
    """(values, specs) for the raw builder: every read_source row; each RANK text column becomes a categorical_codes spec
    with row-wise pointer codes into its own text (`inverse`: pointer -> text). No code depends on another row."""
    V = np.array(values, dtype=np.float64, copy=True)
    out_specs = [copy.deepcopy(dict(s)) for s in specs]
    cols = {}
    for n, cells in text.items():
        if readings[n] != RANK:
            continue
        j = next(i for i, s in enumerate(out_specs) if s["name"] == n)
        s = out_specs[j]
        inverse: dict[int, str] = {}
        codes = np.empty(len(cells), dtype=np.float64)
        for r, t in enumerate(cells):
            if is_na(t):
                codes[r] = t1raw.MISSING_CODE
                continue
            p = pointer(t)
            if inverse.setdefault(p, t) != t:
                raise NonnumericError("pointer_collision", n)
            codes[r] = float(p)
        V[:, s["index"]] = codes
        out_specs[j] = {"index": s["index"], "name": n, "source_name": s["source_name"], "kind": "categorical_codes",
                        "lo": None, "hi": None, "derive": None, "impute_code": None, "inverse": inverse,
                        "nonnumeric": {"meta_kind": s["kind"], "reading": RANK, "rule": UNIVERSE_RULE}}
        cols[n] = {"meta_kind": s["kind"], "texts": len(inverse), "missing_cells": int(np.sum(codes == t1raw.MISSING_CODE))}
    rep = {"schema": SCHEMA, "rule": UNIVERSE_RULE, "columns": cols,
           "numeric_read_columns": sorted(n for n in text if readings[n] == NUMERIC)}
    return V, out_specs, rep


def read_universe(raw_path, meta: Mapping[str, Any], width: int, readings: Mapping[str, str]) -> tuple[np.ndarray, list[dict], dict]:
    """The universe of one raw file from the readings a match receipt pinned (no T1 needed): read_source, the same text
    columns (refused when they differ from the pinned set), universe(). For a rebuild and for text-level counterfactuals."""
    specs = t1raw.columns(meta, width)
    src = t1raw.read_source(raw_path, meta, specs)
    cand = text_columns(specs, src["report"])
    text = confirm_text(specs, read_text(raw_path, meta, specs, src["report"], cand)) if cand else {}
    names = list(text)
    if sorted(names) != sorted(readings):
        raise NonnumericError("text_columns_differ_from_pinned", f"{sorted(names)[:6]} != {sorted(readings)[:6]}")
    check_alignment(src["values"], specs, text)
    return universe(src["values"], specs, text, readings)


# ------------------------------------------------------------------------------------------------- ambiguity

def _candidates(A, specs, U, rows, tol_rel):
    """t1raw.match's candidate classes for the given A rows (same tolerance, impute and code rules)."""
    d, _ = t1raw.t1_cells(A, specs)
    Af = np.asarray(A, dtype=np.float64)
    dec9 = [s["kind"] == "numeric" and t1raw.printed_decimal9(Af[:, s["index"]]) for s in specs]
    out = {}
    for i in rows:
        ok = np.ones(U.shape[0], dtype=bool)
        for s in specs:
            j = s["index"]
            if s["kind"] == "numeric":
                tol = float(t1raw.tolerance(s, d[i:i + 1, j], tol_rel, dec9[j])[0])
                imp = s.get("impute_value") is not None and abs(s["impute_value"] - d[i, j]) <= tol
            else:
                tol, imp = 0.0, d[i, j] == s["impute_code"]
            ok &= t1raw._cell_ok(U[:, j], s, d[i, j], tol, imp)
        out[int(i)] = np.flatnonzero(ok).tolist()
    return out


def _assign(cands: Mapping[int, Sequence[int]], cap: Mapping[int, int]) -> dict[int, int]:
    """Capacity-respecting bipartite assignment of rows to classes (augmenting paths); maximal size."""
    load: dict[int, list[int]] = {}
    got: dict[int, int] = {}

    def augment(i, seen):
        for k in cands[i]:
            if k in seen or cap.get(k, 0) <= 0:
                continue
            seen.add(k)
            holders = load.setdefault(k, [])
            if len(holders) < cap[k]:
                holders.append(i)
                got[i] = k
                return True
            for r in list(holders):
                if augment(r, seen):
                    holders.remove(r)
                    holders.append(i)
                    got[i] = k
                    return True
        return False
    for i in sorted(cands):
        augment(i, set())
    return got


def ambiguity_blocks(A, specs, m: Mapping[str, Any], tol_rel: float = t1raw.MATCH_TOL_REL) -> dict:
    """Are t1raw.match's ambiguous rows SET-determined? Residual capacity of a class = its raw multiplicity minus the
    claims of uniquely matched rows. Tight = an assignment of every ambiguous row exists AND it uses all residual
    capacity of every candidate class, so every raw row behind those classes is a T1 row."""
    st = m["stats"]
    amb = [int(i) for i in np.flatnonzero(np.asarray(m["assign"]) < 0)]
    rep = {"ambiguous_rows": st["ambiguous_rows"], "unmatched_rows": st["unmatched_rows"], "oversubscribed_rows": st["oversubscribed_rows"]}
    if st["unmatched_rows"] or st["oversubscribed_rows"] or not amb:
        return dict(rep, tight=False, complete=False, reason="not_only_ambiguity" if amb else "no_ambiguity")
    U, counts, assign = m["classes"], np.asarray(m["class_counts"]), np.asarray(m["assign"])
    claimed = np.bincount(assign[assign >= 0], minlength=U.shape[0])
    resid = {k: int(counts[k] - claimed[k]) for k in range(U.shape[0])}
    cands = _candidates(A, specs, U, amb, tol_rel)
    if any(len(c) < 2 for c in cands.values()):
        raise NonnumericError("ambiguity_drift", "an ambiguous row has fewer than two candidate classes")
    got = _assign(cands, resid)
    union = sorted({k for c in cands.values() for k in c})
    capacity = sum(max(resid[k], 0) for k in union)
    Af = np.asarray(A, dtype=np.float64)
    blocks: dict[tuple, list[int]] = {}
    for i in amb:
        blocks.setdefault(tuple(cands[i]), []).append(i)
    identical = all(bool(np.all(Af[rows] == Af[rows[0]])) for rows in blocks.values())
    complete = len(got) == len(amb)
    tight = complete and capacity == len(amb)
    return dict(rep, tight=tight, complete=complete, assigned=len(got), candidate_classes=len(union),
                residual_capacity=capacity, blocks=len(blocks), largest_block=max(len(r) for r in blocks.values()),
                t1_rows_identical_within_blocks=identical,
                reason=None if tight else ("no_complete_assignment" if not complete else "spare_capacity"))


# ------------------------------------------------------------------------------------------------- one dataset

def _stats(m):
    keys = ("rows", "matched_rows", "unmatched_rows", "ambiguous_rows", "oversubscribed_rows", "raw_rows",
            "raw_rows_without_target", "raw_value_classes", "blame_columns_first_unmatched", "residual", "matched")
    return {k: m["stats"][k] for k in keys}


def _reason(m, text_names, readings, report, conflict=None):
    st = m["stats"]
    if st["unmatched_rows"]:
        blame = st["blame_columns_first_unmatched"]                       # the most-blamed column names the reason
        top = max(blame, key=lambda c: (blame[c], c)) if blame else None
        if top == "target":
            return "meta_target_name_conflict" if conflict else "rows_unmatched_target"
        if top in {n for n in text_names if readings[n] == RANK}:
            return "rows_unmatched_text_column"
        if top in set(report.get("nonnumeric_cells") or {}):
            return "rows_unmatched_nonnumeric_raw"
        return "rows_unmatched"
    if st["ambiguous_rows"]:
        return "rows_ambiguous"
    return "rows_oversubscribed"


def match_dataset(raw_path, meta: Mapping[str, Any], A, tol_rel: float = t1raw.MATCH_TOL_REL) -> dict:
    """read_source + text columns coded the acquisition's way + t1raw.match. Never raises for a data problem. status:
    'matched' (t1raw's strict rule), 'matched_set_level' / 'matched_covered' (only ambiguity: ambiguity_blocks) or
    'unmatched' with a reason. Every matched status carries the universe (values, specs) over EVERY read_source row."""
    A = np.asarray(A, dtype=np.float64)
    conflict = target_name_conflict(meta)
    base = {"schema": SCHEMA, "staging_rule": STAGING_RULE, "target_name_conflict": conflict}
    try:
        specs = t1raw.columns(meta, A.shape[1])
        src = t1raw.read_source(raw_path, meta, specs)
        rep = src["report"]
        if "target" in (rep.get("nonnumeric_cells") or {}):
            raise NonnumericError("meta_target_name_conflict" if conflict else "target_nonnumeric",
                                  f"{rep['nonnumeric_cells']['target']} text cells in target column {specs[-1]['source_name']!r}")
        cand = text_columns(specs, rep)
        text = confirm_text(specs, read_text(raw_path, meta, specs, rep, cand)) if cand else {}
        names = list(text)
        if not names:
            raise NonnumericError("no_text_columns", f"candidates {cand[:6]} hold no text")
        check_alignment(src["values"], specs, text)
        readings = {n: RANK for n in names}
        attempts = []
        m = t1raw.match(A, specs, match_matrix(src["values"], specs, text, readings), tol_rel)
        attempts.append({"readings": dict(readings), "stats": _stats(m)})
        if m["stats"]["unmatched_rows"]:
            by_name = {s["name"]: s for s in specs}
            flip = [n for n in names if n in m["stats"]["blame_columns_first_unmatched"]
                    and numeric_reading_fits(by_name[n], text[n])]
            if flip:
                for n in flip:
                    readings[n] = NUMERIC
                m = t1raw.match(A, specs, match_matrix(src["values"], specs, text, readings), tol_rel)
                attempts.append({"readings": dict(readings), "flipped_to_numeric": flip, "stats": _stats(m)})
    except t1raw.T1RawError as e:
        return dict(base, status="unmatched", reason=e.reason, detail=e.detail[:300], stats=None, read=None)
    columns = {}
    by_name = {s["name"]: s for s in specs}
    for n in names:
        s = by_name[n]
        nonna = [t for t in text[n] if not is_na(t)]
        columns[n] = {"meta_kind": s["kind"], "reading": readings[n], "distinct_texts": len(set(nonna)),
                      "na_cells": len(text[n]) - len(nonna), "meta_lo": s["lo"], "meta_hi": s["hi"],
                      "meta_impute": s.get("impute_value") if s["kind"] == "numeric" else s.get("impute_code"),
                      "examples": sorted(set(nonna))[:3]}
    out = dict(base, read=rep, stats=_stats(m), attempts=attempts, text_columns=columns, specs=specs)
    if m["matched"]:
        out.update(status="matched", reason=None, detail="")
    else:
        blocks = ambiguity_blocks(A, specs, m, tol_rel)
        out["ambiguity"] = blocks
        if blocks["tight"]:
            out.update(status="matched_set_level", reason="rows_ambiguous_set_determined", detail="")
        elif blocks["complete"]:
            out.update(status="matched_covered", reason="rows_ambiguous_covered", detail="")
        else:
            out.update(status="unmatched", reason=_reason(m, names, readings, rep, conflict), detail="")
    if out["status"] != "unmatched":
        V, uspecs, urep = universe(src["values"], specs, text, readings)
        out.update(values=V, universe_specs=uspecs, universe=urep)
    return out
