"""PLAN §5 P2 near-duplicate screen (spec SCREEN, survey wf_a43390a0-83c): near_dup_screen() and its table loaders.

The screen is deterministic and keeps no RNG state. It uses numpy, plus pyarrow for the BeyondArena loader only.
It never touches S3 or SQLite: the core builder (dltrain.families) pins every input first and passes local paths.

Tables (features only):
- epoch identities: the local research data object (gzip CSV, headerless, last column = target, dropped)
  -> load_data_object();
- sealed references: T1 train + test (headerless CSV, plain or gzip, last column = target, dropped)
  -> load_sealed_ref();
- BeyondArena seed sources: the parquet's numeric columns minus the target, each scaled as
  (v - min) / (max - min) over the full table; non-finite values are dropped -> load_beyondarena().
A Table keeps unrounded float64 values; NaN marks a dropped (non-finite) value.

Screen steps (policy keys in brackets; defaults in DEFAULT_SCREEN_POLICY):
1. tokens: tok = np.round(np.float64(x), ROUND) + 0.0 (the + 0.0 folds -0.0 into 0.0);
2. content: row_multiset_sha256 = sha256(b''.join(sorted(sha256(row.tobytes()).digest()))) over the token
   rows ('<f8' bytes; rows holding a dropped value are skipped); equal digests give 'content' links;
3. informative column: at least INFORMATIVE_MIN_DISTINCT distinct finite tokens;
4. MinHash over each informative column's distinct tokens:
   h(t) = int.from_bytes(blake2b(struct.pack('<d', t), digest_size=8).digest(), 'big');
   for j < MINHASH_K: d = sha256(NDS_DOMAIN + j.to_bytes(2, 'big')).digest(),
   a_j = int(d[0:8]) % p (1 if that is 0), b_j = int(d[8:16]) % p (big-endian), p = 2^61 - 1;
   sig_j = min over tokens of (a_j * h + b_j) % p (exact uint64 arithmetic, tested against Python ints);
5. LSH with LSH = [bands, rows] gives candidate column pairs across different identities; each candidate is
   verified by exact set Jaccard, and a match needs J >= COL_J (exact rational comparison);
6. stop column: a column whose matches reach columns of more than STOP_T other identities;
7. per identity pair, a greedy 1:1 mapping of non-stop matches sorted by (-J, col_a, col_b), where a is the
   lexicographically smaller identity. weak = at least MIN_COLS mapped columns. strong = weak and
   containment = |T_A & T_B| / min(|T_A|, |T_B|) >= CONTAIN, where T_X is the set of distinct token tuples
   projected on the first MAX_JOIN_COLS mapped columns (highest J first), and min(|T_A|, |T_B|) >= MIN_TUPLES;
8. names: norm(n) = re.sub('[^a-z0-9]+', '_', n.lower()).strip('_'); N(i) = {norm(n)} minus GENERIC_NAMES,
   minus names matching GENERIC_NAME_RE, minus ''; a link needs |N & N'| >= NAME_MIN_SHARED and
   Jaccard >= NAME_J. Epoch and sealed names come from meta feature_columns + [target_column]
   (names_from_meta); BeyondArena names are the parquet column names;
8a. sealed references with fewer than NAME_MIN_SHARED non-generic names (review fix SC-1): an identity whose name set
   holds every name of such a reference gets a 'nds:name_subset' link to it (2 names: superset; 1 name: identical
   set). It is review evidence (provisional seal in the core builder), never a union, and carries name hashes only;
8b. sealed-reference pairs re-checked without the stop-column exclusion (SC-1): a pair the primary pass did not link
   that is strong by the joint-row rule once stop columns are kept is reported as 'nds:value_weak' with
   evidence.stop_relaxed = true (provisional seal, owner-reversible; generic grid columns make these false positives
   more often than true ones, so they never union);
9. sensitivity: stop columns and weak/strong counts recomputed at every SENS_T threshold (report only);
10. BeyondArena pair check (report only; the spec's optional raw-domain check), for every pair of a BeyondArena
    table and another table that the screen links or the caller lists in ba_pairs (e.g. curated alias members):
    - scaled domain, always: both sides' tokens as above (the BeyondArena table scaled over its full table);
    - raw domain, only when the other table's meta lengths validate: the meta feature_scaling (aligned to
      output_headers minus target_column, mapped to data columns by feature_columns) relates the domains as
      raw = x * (max - min) + min. The check maps the BeyondArena raw values onto the other table's grid,
      (raw - min) / (max - min), and tokenizes both sides with ROUND decimals. (Un-scaled values would carry the
      float32 storage error, about 6e-8 x the column range in absolute terms, which defeats any fixed
      significant-digit rounding of small values in wide-range columns.)
    A BeyondArena table is the full source while an epoch data object is a <= 2k-row sample, so column Jaccard is
    capped near |sample distinct| / |source distinct| even for a true copy (measured 0.15-0.77 on the houses alias
    members, whose per-column containment is 0.96-1.0). The check therefore maps informative non-stop columns by
    containment of the smaller distinct set >= COL_J (greedy by (-containment, col_a, col_b)), then applies the same
    joint-row rule (CONTAIN on <= MAX_JOIN_COLS columns, >= MIN_TUPLES tuples). It never creates a union link; its
    class (strong | weak | none) is evidence for the core builder's curated-alias confirmation and for review.

Screen coverage (SC-1) counts informative columns that are NOT stop columns (n_informative_nonstop): pairs() drops stop
columns, so a table whose informative columns are all stop columns can never get a value link and is not values-screened.

near_dup_screen() returns a ScreenResult. Its content_equal, value_strong and name_links are the links the spec
unions; value_weak (weak and not strong) and name_subsets are review triggers. ScreenResult.files() renders screen/{links.jsonl,
stop-columns.json, sensitivity.json, coverage.json}. Screen evidence is test-informed (data objects hold outer-test
rows), so it stays outside every learner input closure. In memory, name evidence is plaintext; files() and as_dict()
write it as sha256 values whenever a link touches a sealed reference or an identity the caller redacts. Nothing here
prints.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import re
import struct
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from dltrain import common

SCHEMA = "jope.dl-family-screen/v1"
MERSENNE_P = (1 << 61) - 1
TABLE_KINDS = ("epoch", "sealed_ref", "beyondarena")
LINK_KINDS = ("content", "nds:value_strong", "nds:value_weak", "nds:name_jaccard", "nds:name_subset")
UNION_LINK_KINDS = ("content", "nds:value_strong", "nds:name_jaccard")
COVERAGE_CLASSES = ("full", "names_only", "values_only", "exact_only")
SCREEN_FILES = ("links.jsonl", "stop-columns.json", "sensitivity.json", "coverage.json")

DEFAULT_SCREEN_POLICY = {
    "NDS_DOMAIN": "jope.dl-family-nds/v1",
    "ROUND": 5,
    "INFORMATIVE_MIN_DISTINCT": 12,
    "MINHASH_K": 64,
    "LSH": [16, 4],
    "COL_J": 0.8,
    "MIN_COLS": 2,
    "STOP_T": 5,
    "SENS_T": [10, 20],
    "CONTAIN": 0.5,
    "MIN_TUPLES": 50,
    "MAX_JOIN_COLS": 8,
    "NAME_J": 0.8,
    "NAME_MIN_SHARED": 3,
    "GENERIC_NAMES": ["date", "day", "id", "index", "label", "month", "target", "time", "unnamed_0", "value", "x", "y",
                      "year"],
    "GENERIC_NAME_RE": r"^(x|v|var|col|column|feature|f)_?[0-9]+$",
    "MAX_CANDIDATE_PAIRS": 5_000_000,
}

_P64 = np.uint64(MERSENNE_P)
_M32 = np.uint64(0xFFFFFFFF)
_M29 = np.uint64((1 << 29) - 1)
_U3, _U29, _U32, _U61 = np.uint64(3), np.uint64(29), np.uint64(32), np.uint64(61)


class ScreenInputError(ValueError):
    """A screen input is malformed, fails its pin, or breaks a policy invariant (STOP). Messages never echo data."""


class ScreenLimitError(RuntimeError):
    """The LSH candidate set grew past MAX_CANDIDATE_PAIRS (STOP for review; results are never silently capped)."""


# ----------------------------------------------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------------------------------------------

def _as_int(name, v, lo, hi=None) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
        raise ScreenInputError(f"policy {name}={v!r} must be an integer in [{lo}, {hi if hi is not None else 'inf'}]")
    return v


def _as_ratio(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ScreenInputError(f"policy {name}={v!r} must be a number")
    if not (0 < Fraction(str(v)) <= 1):
        raise ScreenInputError(f"policy {name}={v!r} must be in (0, 1]")
    return v


def screen_policy(policy: Mapping | None = None) -> dict:
    """The effective screen policy: DEFAULT_SCREEN_POLICY overridden by the screen keys of `policy`.

    `policy` may be the whole alias-sources.json['policy'] block; keys this screen does not use are ignored.
    The result is JSON-serializable and is recorded in every screen output. Ratios compare as exact rationals of
    their shortest decimal form (0.8 means 4/5).
    """
    p = dict(DEFAULT_SCREEN_POLICY)
    for k in DEFAULT_SCREEN_POLICY:
        if policy is not None and k in policy:
            p[k] = policy[k]
    dom = p["NDS_DOMAIN"]
    if isinstance(dom, (bytes, bytearray)):
        dom = bytes(dom).decode("utf-8")
    if not isinstance(dom, str) or not dom:
        raise ScreenInputError("policy NDS_DOMAIN must be a non-empty string")
    lsh = p["LSH"]
    if isinstance(lsh, Mapping):
        lsh = [lsh.get("bands"), lsh.get("rows")]
    if not isinstance(lsh, (list, tuple)) or len(lsh) != 2:
        raise ScreenInputError("policy LSH must be [bands, rows]")
    bands, rows = _as_int("LSH.bands", lsh[0], 1), _as_int("LSH.rows", lsh[1], 1)
    k = _as_int("MINHASH_K", p["MINHASH_K"], 1, 4096)
    if bands * rows != k:
        raise ScreenInputError(f"policy LSH {bands}x{rows} must cover MINHASH_K={k}")
    sens = p["SENS_T"]
    if isinstance(sens, int) and not isinstance(sens, bool):
        sens = [sens]
    if not isinstance(sens, (list, tuple)):
        raise ScreenInputError("policy SENS_T must be a list of integers")
    generic = p["GENERIC_NAMES"]
    if isinstance(generic, str) or not all(isinstance(g, str) for g in generic):
        raise ScreenInputError("policy GENERIC_NAMES must be a list of strings")
    gre = p["GENERIC_NAME_RE"]
    try:
        re.compile(gre)
    except (re.error, TypeError) as exc:
        raise ScreenInputError(f"policy GENERIC_NAME_RE does not compile: {exc}") from None
    return {
        "NDS_DOMAIN": dom,
        "ROUND": _as_int("ROUND", p["ROUND"], 0, 15),
        "INFORMATIVE_MIN_DISTINCT": _as_int("INFORMATIVE_MIN_DISTINCT", p["INFORMATIVE_MIN_DISTINCT"], 1),
        "MINHASH_K": k,
        "LSH": [bands, rows],
        "COL_J": _as_ratio("COL_J", p["COL_J"]),
        "MIN_COLS": _as_int("MIN_COLS", p["MIN_COLS"], 1),
        "STOP_T": _as_int("STOP_T", p["STOP_T"], 0),
        "SENS_T": sorted({_as_int("SENS_T", t, 0) for t in sens}),
        "CONTAIN": _as_ratio("CONTAIN", p["CONTAIN"]),
        "MIN_TUPLES": _as_int("MIN_TUPLES", p["MIN_TUPLES"], 1),
        "MAX_JOIN_COLS": _as_int("MAX_JOIN_COLS", p["MAX_JOIN_COLS"], 1),
        "NAME_J": _as_ratio("NAME_J", p["NAME_J"]),
        "NAME_MIN_SHARED": _as_int("NAME_MIN_SHARED", p["NAME_MIN_SHARED"], 1),
        "GENERIC_NAMES": sorted(set(generic)),
        "GENERIC_NAME_RE": gre,
        "MAX_CANDIDATE_PAIRS": _as_int("MAX_CANDIDATE_PAIRS", p["MAX_CANDIDATE_PAIRS"], 1),
    }


def _ratio(v) -> Fraction:
    return Fraction(str(v))


def _ge(num: int, den: int, q: Fraction) -> bool:
    """num / den >= q, exactly (den > 0)."""
    return num * q.denominator >= den * q.numerator


# ----------------------------------------------------------------------------------------------------------------
# Tables and loaders
# ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Table:
    """One screened table, features only. values: float64 (rows, cols), NaN marks a dropped (non-finite) value.

    column_names: the kept feature columns' names when the source carries them (BeyondArena only).
    names: the source column names for the name screen (BeyondArena: every parquet column).
    raw_values: the unscaled values of a BeyondArena table (optional raw-domain check).
    source: the pinned files read ({role: {path, sha256, bytes}}).
    """
    identity: str
    kind: str
    values: np.ndarray
    source: dict
    column_names: tuple | None = None
    names: tuple | None = None
    raw_values: np.ndarray | None = None

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.values.shape[1])


def make_table(identity: str, kind: str, values, *, source: dict | None = None, column_names=None, names=None,
               raw_values=None) -> Table:
    """Validate and normalize a table: 2-D float64, C-contiguous, read-only, every non-finite value set to NaN."""
    if not isinstance(identity, str) or not identity:
        raise ScreenInputError("table identity must be a non-empty string")
    if kind not in TABLE_KINDS:
        raise ScreenInputError(f"{identity}: table kind {kind!r} not in {TABLE_KINDS}")
    arr = np.array(values, dtype=np.float64, copy=True, order="C")
    if arr.ndim != 2:
        raise ScreenInputError(f"{identity}: table values must be 2-D, got {arr.ndim}-D")
    arr[~np.isfinite(arr)] = np.nan
    if column_names is not None:
        column_names = tuple(str(c) for c in column_names)
        if len(column_names) != arr.shape[1]:
            raise ScreenInputError(f"{identity}: {len(column_names)} column names for {arr.shape[1]} columns")
    if names is not None:
        names = tuple(names)
    if raw_values is not None:
        raw_values = np.array(raw_values, dtype=np.float64, copy=True, order="C")
        if raw_values.shape != arr.shape:
            raise ScreenInputError(f"{identity}: raw_values shape {raw_values.shape} != values shape {arr.shape}")
        raw_values[~np.isfinite(raw_values)] = np.nan
        raw_values.setflags(write=False)
    arr.setflags(write=False)
    return Table(identity, kind, arr, dict(source or {}), column_names, names, raw_values)


def read_pinned(path, expected_sha256: str | None = None, expected_bytes: int | None = None) -> tuple[bytes, dict]:
    """Read a local file and check it against its pin; return (bytes, {path, sha256, bytes})."""
    data = Path(path).read_bytes()
    digest = common.sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ScreenInputError(f"{path}: sha256 {digest} != pinned {expected_sha256} (re-pin)")
    if expected_bytes is not None and len(data) != expected_bytes:
        raise ScreenInputError(f"{path}: {len(data)} bytes != pinned {expected_bytes} (re-pin)")
    return data, {"path": str(path), "sha256": digest, "bytes": len(data)}


def parse_csv_matrix(data: bytes, label: str = "csv") -> np.ndarray:
    """Parse a headerless numeric CSV (gzip detected by its magic bytes) into a float64 matrix.

    Values are parsed as numbers and never compared as text. A header, a ragged row or a non-numeric cell is a
    STOP; the error never echoes cell contents, which could carry a sealed name.
    """
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            raise ScreenInputError(f"{label}: corrupt gzip stream") from None
    if not data.strip():
        raise ScreenInputError(f"{label}: empty table")
    try:
        arr = np.loadtxt(io.BytesIO(data), delimiter=",", dtype=np.float64, ndmin=2, comments=None)
    except ValueError:
        raise ScreenInputError(f"{label}: not a headerless rectangular numeric CSV (cell contents withheld)") from None
    return arr


def load_data_object(identity: str, path, *, expected_sha256: str | None = None,
                     expected_bytes: int | None = None) -> Table:
    """An epoch identity's local research data object: gzip CSV, last column = target (dropped)."""
    data, src = read_pinned(path, expected_sha256, expected_bytes)
    arr = parse_csv_matrix(data, identity)
    if arr.shape[1] < 2:
        raise ScreenInputError(f"{identity}: data object has {arr.shape[1]} column(s); need features + target")
    return make_table(identity, "epoch", arr[:, :-1], source={"data": src})


def load_sealed_ref(identity: str, train_path, test_path, *, train_sha256: str | None = None,
                    test_sha256: str | None = None, train_bytes: int | None = None,
                    test_bytes: int | None = None) -> Table:
    """A sealed reference: T1 train + test rows (headerless CSV, last column = target, dropped), train first."""
    tr, src_tr = read_pinned(train_path, train_sha256, train_bytes)
    te, src_te = read_pinned(test_path, test_sha256, test_bytes)
    a = parse_csv_matrix(tr, identity + " train")
    b = parse_csv_matrix(te, identity + " test")
    if a.shape[1] != b.shape[1]:
        raise ScreenInputError(f"{identity}: train has {a.shape[1]} columns, test has {b.shape[1]}")
    if a.shape[1] < 2:
        raise ScreenInputError(f"{identity}: sealed reference has {a.shape[1]} column(s); need features + target")
    return make_table(identity, "sealed_ref", np.vstack([a, b])[:, :-1], source={"train": src_tr, "test": src_te})


def _arrow_numeric(t) -> bool:
    import pyarrow as pa
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_decimal(t)


def load_beyondarena(identity: str, path, target: str, *, expected_sha256: str | None = None,
                     expected_bytes: int | None = None) -> Table:
    """A BeyondArena seed source parquet, features only.

    Keeps the numeric columns (integer, floating, decimal; never boolean, dictionary, string or temporal) other than
    `target`, scales each as (v - min) / (max - min) over the full table's finite values and drops every non-finite
    value; a column left with no finite value (a constant or empty column) is dropped. names are all parquet column
    names; column_names are the kept columns; raw_values keep the unscaled values.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    data, src = read_pinned(path, expected_sha256, expected_bytes)
    try:
        table = pq.read_table(pa.BufferReader(data))
    except (pa.ArrowException, OSError):
        raise ScreenInputError(f"{identity}: unreadable parquet") from None
    all_names = tuple(table.column_names)
    if len(set(all_names)) != len(all_names):
        raise ScreenInputError(f"{identity}: duplicate parquet column names")
    if not isinstance(target, str) or target not in all_names:
        raise ScreenInputError(f"{identity}: target column is not in the parquet schema")
    kept, scaled, raw = [], [], []
    for field in table.schema:
        if field.name == target or not _arrow_numeric(field.type):
            continue
        try:
            col = pc.fill_null(pc.cast(table.column(field.name), pa.float64()), float("nan"))
        except pa.ArrowException:
            raise ScreenInputError(f"{identity}: a numeric parquet column does not cast to float64") from None
        x = np.array(col.to_numpy(), dtype=np.float64, copy=True)
        x[~np.isfinite(x)] = np.nan
        fin = np.isfinite(x)
        if not fin.any():
            continue
        lo, hi = float(np.min(x[fin])), float(np.max(x[fin]))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            s = (x - lo) / (hi - lo)
        s[~np.isfinite(s)] = np.nan
        if not np.isfinite(s).any():
            continue
        kept.append(field.name)
        scaled.append(s)
        raw.append(np.where(np.isfinite(s), x, np.nan))
    n = table.num_rows
    values = np.column_stack(scaled) if scaled else np.empty((n, 0), dtype=np.float64)
    raw_values = np.column_stack(raw) if raw else np.empty((n, 0), dtype=np.float64)
    src = dict(src, n_parquet_columns=len(all_names), n_numeric_features=len(kept))
    return make_table(identity, "beyondarena", values, source={"parquet": src}, column_names=kept, names=all_names,
                      raw_values=raw_values)


def names_from_meta(meta: Mapping | None) -> list[str] | None:
    """meta.feature_columns + [meta.target_column], or None when the meta is missing or malformed (names unknown)."""
    if not isinstance(meta, Mapping):
        return None
    fc, tc = meta.get("feature_columns"), meta.get("target_column")
    if not isinstance(fc, list) or not all(isinstance(c, str) for c in fc):
        return None
    return list(fc) + ([tc] if isinstance(tc, str) else [])


# ----------------------------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------------------------

def normalize_name(n: str) -> str:
    return re.sub("[^a-z0-9]+", "_", n.lower()).strip("_")


def _name_set(names, generic: frozenset, gre: re.Pattern) -> frozenset | None:
    if names is None:
        return None
    out = set()
    for n in names:
        if isinstance(n, str):
            m = normalize_name(n)
            if m and m not in generic and not gre.match(m):
                out.add(m)
    return frozenset(out)


def name_set(names: Iterable | None, policy: Mapping | None = None) -> frozenset | None:
    """N(i): normalized names minus GENERIC_NAMES, GENERIC_NAME_RE matches and ''. None stays None (unknown)."""
    pol = screen_policy(policy)
    return _name_set(names, frozenset(pol["GENERIC_NAMES"]), re.compile(pol["GENERIC_NAME_RE"]))


# ----------------------------------------------------------------------------------------------------------------
# Tokens, MinHash and exact arithmetic
# ----------------------------------------------------------------------------------------------------------------

def tokens(values: np.ndarray, digits: int) -> np.ndarray:
    """tok = np.round(np.float64(x), digits) + 0.0; non-finite becomes the canonical NaN (dropped)."""
    with np.errstate(over="ignore", invalid="ignore"):
        t = np.round(np.asarray(values, dtype=np.float64), digits) + 0.0
    t[~np.isfinite(t)] = np.nan
    return np.ascontiguousarray(t, dtype="<f8")


def _row_bytes(rows: np.ndarray) -> list[bytes]:
    rows = np.ascontiguousarray(rows, dtype="<f8")
    w = rows.shape[1] * 8
    buf = rows.tobytes()
    return [buf[i:i + w] for i in range(0, len(buf), w)]


def row_multiset_sha256(tok: np.ndarray) -> tuple[str | None, int]:
    """sha256 over the sorted sha256 digests of the complete token rows; (None, 0) without features or rows."""
    if tok.ndim != 2 or tok.shape[1] == 0:
        return None, 0
    rows = tok[~np.isnan(tok).any(axis=1)]
    if rows.shape[0] == 0:
        return None, 0
    digests = sorted(hashlib.sha256(r).digest() for r in _row_bytes(rows))
    return hashlib.sha256(b"".join(digests)).hexdigest(), int(rows.shape[0])


def minhash_params(domain: str | bytes, k: int) -> tuple[list[int], list[int]]:
    """(a_j, b_j) for j < k: d = sha256(domain + j.to_bytes(2, 'big')); a = int(d[0:8]) % p or 1; b = int(d[8:16]) % p."""
    dom = domain.encode("utf-8") if isinstance(domain, str) else bytes(domain)
    a_list, b_list = [], []
    for j in range(k):
        d = hashlib.sha256(dom + j.to_bytes(2, "big")).digest()
        a = int.from_bytes(d[0:8], "big") % MERSENNE_P
        a_list.append(a if a != 0 else 1)
        b_list.append(int.from_bytes(d[8:16], "big") % MERSENNE_P)
    return a_list, b_list


def token_hash64(t: float) -> int:
    """h(t) = int.from_bytes(blake2b(struct.pack('<d', t), digest_size=8).digest(), 'big') (reference form)."""
    return int.from_bytes(hashlib.blake2b(struct.pack("<d", float(t)), digest_size=8).digest(), "big")


def _hash_tokens(u: np.ndarray) -> np.ndarray:
    """token_hash64 over a float64 array, as uint64."""
    buf = np.ascontiguousarray(u, dtype="<f8").tobytes()
    digests = b"".join(hashlib.blake2b(buf[i:i + 8], digest_size=8).digest() for i in range(0, len(buf), 8))
    return np.frombuffer(digests, dtype=">u8").astype(np.uint64)


def mulmod61(x: np.ndarray, a: int) -> np.ndarray:
    """(a * x) mod (2^61 - 1) for uint64 x < p and 0 <= a < p, exact in uint64 (2^61 = 1 and 2^64 = 8 mod p)."""
    x = np.asarray(x, dtype=np.uint64)
    a_lo, a_hi = np.uint64(a & 0xFFFFFFFF), np.uint64(a >> 32)
    x_lo, x_hi = x & _M32, x >> _U32
    ll = x_lo * a_lo                                    # < 2^64
    mid = x_hi * a_lo + x_lo * a_hi                     # < 2^62
    hh = x_hi * a_hi                                    # < 2^58
    s = (hh << _U3) + (mid >> _U29) + ((mid & _M29) << _U32) + (ll & _P64) + (ll >> _U61)    # < 2^63
    s = (s & _P64) + (s >> _U61)
    return np.where(s >= _P64, s - _P64, s)


def affine_mod61(h: np.ndarray, a: int, b: int) -> np.ndarray:
    """(a * h + b) mod (2^61 - 1) for any uint64 h (h is reduced mod p first; the result is unchanged)."""
    t = mulmod61(np.asarray(h, dtype=np.uint64) % _P64, a) + np.uint64(b)
    return np.where(t >= _P64, t - _P64, t)


def _signatures(universe: np.ndarray, ids: list[np.ndarray], domain, k: int) -> np.ndarray:
    if not ids:
        return np.empty((0, k), dtype=np.uint64)
    hm = _hash_tokens(universe) % _P64
    flat = np.concatenate(ids)
    offsets = np.zeros(len(ids), dtype=np.intp)
    offsets[1:] = np.cumsum([c.size for c in ids[:-1]])
    a_list, b_list = minhash_params(domain, k)
    sig = np.empty((len(ids), k), dtype=np.uint64)
    for j in range(k):
        sig[:, j] = np.minimum.reduceat(affine_mod61(hm, a_list[j], b_list[j])[flat], offsets)
    return sig


def minhash_signatures(columns: list, domain: str | bytes, k: int) -> np.ndarray:
    """MinHash signatures (len(columns), k) over each column's distinct tokens (every column non-empty, finite)."""
    cols = [np.unique(np.asarray(c, dtype=np.float64)) for c in columns]
    if any(c.size == 0 or not np.isfinite(c).all() for c in cols):
        raise ScreenInputError("MinHash columns must be non-empty and finite")
    if not cols:
        return np.empty((0, k), dtype=np.uint64)
    universe = np.unique(np.concatenate(cols))
    return _signatures(universe, [np.searchsorted(universe, c) for c in cols], domain, k)


def _intersect_count(a: np.ndarray, b: np.ndarray) -> int:
    """|a & b| for sorted unique arrays (token ids or finite tokens)."""
    if a.size > b.size:
        a, b = b, a
    if a.size == 0:
        return 0
    pos = np.searchsorted(b, a)
    pos[pos == b.size] = 0
    return int(np.count_nonzero(b[pos] == a))


def _greedy(matches: list[tuple]) -> list[tuple]:
    """Greedy 1:1 mapping over (J, col_a, col_b) sorted by (-J, col_a, col_b)."""
    used_a, used_b, mapped = set(), set(), []
    for j, ca, cb in sorted(matches, key=lambda m: (-m[0], m[1], m[2])):
        if ca not in used_a and cb not in used_b:
            used_a.add(ca)
            used_b.add(cb)
            mapped.append((j, ca, cb))
    return mapped


def _tuple_set(tok: np.ndarray, cols: list[int]) -> set:
    proj = tok[:, cols]
    return set(_row_bytes(proj[~np.isnan(proj).any(axis=1)]))


# ----------------------------------------------------------------------------------------------------------------
# The screen
# ----------------------------------------------------------------------------------------------------------------

def _link(a: str, b: str, kind: str, kinds: tuple[str, str], evidence: dict, value: str | None = None) -> dict:
    rec = {"a": a, "b": b, "kind": kind, "kinds": [kinds[0], kinds[1]], "evidence": evidence}
    if value is not None:
        rec["value"] = value
    return rec


def _components(pairs: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[str, list[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), []).append(x)
    return sorted(sorted(g) for g in groups.values())


def coverage_class(values_ok: bool, names_ok: bool) -> str:
    """full = values and names; names_only; values_only; exact_only (neither screen can certify the table)."""
    if values_ok and names_ok:
        return "full"
    if names_ok:
        return "names_only"
    if values_ok:
        return "values_only"
    return "exact_only"


def _sort_links(links: list[dict]) -> list[dict]:
    return sorted(links, key=lambda r: (r["kind"], r["a"], r["b"]))


@dataclass(eq=False)
class ScreenResult:
    """near_dup_screen() output. Each link is {a, b, kind, kinds, evidence[, value]} with a < b (string order).

    content_equal ('content', value = row_multiset_sha256), value_strong and name_links are unioned by the core
    builder; value_weak (weak and not strong, plus the stop-relaxed sealed-reference pairs) and name_subsets
    ('nds:name_subset', sealed references with fewer than NAME_MIN_SHARED names) are review triggers. tables holds
    per-identity statistics including n_informative, n_informative_nonstop, n_names and screen_coverage.
    beyondarena_checks holds the report-only BeyondArena pair checks ({a, b, kinds, requested, class, scaled, raw}).
    Name evidence here is plaintext: never print it; files() and as_dict() redact it.
    """
    schema: str
    policy: dict
    tables_digest: str
    tables: dict
    row_multiset_sha256: dict
    content_equal: list
    value_strong: list
    value_weak: list
    name_links: list
    stop_columns: dict
    stop_column_detail: list
    sensitivity: dict
    beyondarena_checks: list
    counts: dict
    name_subsets: list = None

    def beyondarena_check(self, a: str, b: str) -> dict | None:
        """The BeyondArena pair check for {a, b} (either order), or None when the pair was not checked."""
        lo, hi = sorted((a, b))
        for r in self.beyondarena_checks:
            if r["a"] == lo and r["b"] == hi:
                return r
        return None

    def union_links(self) -> list[dict]:
        """The links the spec unions: content, nds:value_strong and nds:name_jaccard."""
        return _sort_links(self.content_equal + self.value_strong + self.name_links)

    def links(self) -> list[dict]:
        return _sort_links(self.content_equal + self.value_strong + self.value_weak + self.name_links
                           + list(self.name_subsets or []))

    def coverage(self) -> dict:
        return {i: t["screen_coverage"] for i, t in self.tables.items()}

    def redacted_links(self, redact: Iterable[str] = ()) -> list[dict]:
        """links() with name evidence as sha256 values on every link touching a sealed reference or `redact`."""
        redact = frozenset(redact)
        out = []
        for r in self.links():
            r = dict(r)
            if r["kind"] == "nds:name_jaccard" and (r["a"] in redact or r["b"] in redact or "sealed_ref" in r["kinds"]):
                ev = {k: v for k, v in r["evidence"].items() if k != "shared"}
                ev["shared_sha256"] = sorted(common.sha256_bytes(n.encode("utf-8")) for n in r["evidence"]["shared"])
                r["evidence"] = ev
            out.append(r)
        return out

    def files(self, redact: Iterable[str] = ()) -> dict[str, bytes]:
        """screen/{links.jsonl, stop-columns.json, sensitivity.json, coverage.json} as canonical bytes (redacted)."""
        head = {"schema": SCHEMA, "policy": self.policy, "tables_digest": self.tables_digest}
        links = b"".join(common.canonical(r) + b"\n" for r in self.redacted_links(redact))
        stop = dict(head, kind="stop-columns", T=self.policy["STOP_T"], columns=self.stop_column_detail,
                    counts={"stop_columns": len(self.stop_column_detail),
                            "identities_with_stop_columns": len(self.stop_columns)})
        sens = dict(head, kind="sensitivity", by_T=self.sensitivity)
        cov = dict(head, kind="coverage", tables=self.tables, counts=self.counts,
                   beyondarena_checks=self.beyondarena_checks)
        return {"links.jsonl": links, "stop-columns.json": common.canonical(stop) + b"\n",
                "sensitivity.json": common.canonical(sens) + b"\n", "coverage.json": common.canonical(cov) + b"\n"}

    def as_dict(self, redact: Iterable[str] = ()) -> dict:
        """A JSON-serializable view; name evidence is redacted exactly as in files()."""
        links = self.redacted_links(redact)
        return {"schema": self.schema, "policy": self.policy, "tables_digest": self.tables_digest,
                "tables": self.tables, "row_multiset_sha256": self.row_multiset_sha256,
                "content_equal": [r for r in links if r["kind"] == "content"],
                "value_strong": [r for r in links if r["kind"] == "nds:value_strong"],
                "value_weak": [r for r in links if r["kind"] == "nds:value_weak"],
                "name_links": [r for r in links if r["kind"] == "nds:name_jaccard"],
                "name_subsets": [r for r in links if r["kind"] == "nds:name_subset"],
                "stop_columns": self.stop_columns, "stop_column_detail": self.stop_column_detail,
                "sensitivity": self.sensitivity, "beyondarena_checks": self.beyondarena_checks,
                "counts": self.counts}


class _Screen:
    def __init__(self, tables, names, policy, metas, ba_pairs):
        self.pol = screen_policy(policy)
        if isinstance(tables, Mapping):
            for key, t in tables.items():
                if not isinstance(t, Table) or t.identity != key:
                    raise ScreenInputError(f"tables[{key!r}] must be a Table with that identity")
            tabs = list(tables.values())
        else:
            tabs = list(tables)
        seen = set()
        for t in tabs:
            if not isinstance(t, Table):
                raise ScreenInputError("tables must hold Table objects (use make_table or a loader)")
            if t.identity in seen:
                raise ScreenInputError(f"duplicate table identity {t.identity}")
            seen.add(t.identity)
        self.tabs = sorted(tabs, key=lambda t: t.identity)
        self.ids = [t.identity for t in self.tabs]
        self.index = {i: n for n, i in enumerate(self.ids)}
        names = dict(names or {})
        if set(names) - seen:
            raise ScreenInputError(f"names given for {len(set(names) - seen)} identities without a table")
        self.raw_names = {t.identity: (names[t.identity] if t.identity in names else t.names) for t in self.tabs}
        self.metas = dict(metas or {})
        if set(self.metas) - seen:
            raise ScreenInputError(f"metas given for {len(set(self.metas) - seen)} identities without a table")
        self.ba_pairs = sorted({tuple(sorted(p)) for p in (ba_pairs or ())})
        for a, b in self.ba_pairs:
            if a not in seen or b not in seen or a == b:
                raise ScreenInputError("ba_pairs must name two different identities that have tables")
        self.col_j, self.contain, self.name_j = (_ratio(self.pol[k]) for k in ("COL_J", "CONTAIN", "NAME_J"))
        self._tuples: dict = {}

    # -- per-table tokens, content digests and informative columns --
    def tokenize(self):
        self.tok = [tokens(t.values, self.pol["ROUND"]) for t in self.tabs]
        self.rms, self.rms_rows, self.tok_sha = {}, {}, {}
        col_table, self.col_index, self.col_tokens, self.n_informative = [], [], [], []
        for ti, tok in enumerate(self.tok):
            ident = self.ids[ti]
            self.rms[ident], self.rms_rows[ident] = row_multiset_sha256(tok)
            self.tok_sha[ident] = common.sha256_bytes(tok.tobytes())
            n_inf = 0
            for c in range(tok.shape[1]):
                col = tok[:, c]
                u = np.unique(col[~np.isnan(col)])
                if u.size >= self.pol["INFORMATIVE_MIN_DISTINCT"]:
                    col_table.append(ti)
                    self.col_index.append(c)
                    self.col_tokens.append(u)
                    n_inf += 1
            self.n_informative.append(n_inf)
        self.col_table = np.array(col_table, dtype=np.int64)
        self.universe = np.unique(np.concatenate(self.col_tokens)) if self.col_tokens else np.empty(0)
        self.col_ids = [np.searchsorted(self.universe, u) for u in self.col_tokens]

    def content_links(self) -> list[dict]:
        groups: dict[str, list[str]] = {}
        for i in self.ids:
            if self.rms[i] is not None:
                groups.setdefault(self.rms[i], []).append(i)
        out = []
        for digest, members in sorted(groups.items()):
            for a, b in combinations(sorted(members), 2):
                ta, tb = self.tabs[self.index[a]], self.tabs[self.index[b]]
                ev = {"n_complete_rows": self.rms_rows[a], "n_feature_columns": ta.n_cols}
                out.append(_link(a, b, "content", (ta.kind, tb.kind), ev, value=digest))
        return out

    # -- MinHash, LSH and exact verification --
    def candidates(self) -> set:
        bands, rows = self.pol["LSH"]
        sig = _signatures(self.universe, self.col_ids, self.pol["NDS_DOMAIN"], self.pol["MINHASH_K"])
        n, limit = len(self.col_ids), self.pol["MAX_CANDIDATE_PAIRS"]
        cand: set = set()
        for b in range(bands):
            block = sig[:, b * rows:(b + 1) * rows]
            order = np.lexsort(block.T[::-1])
            ordered = block[order]
            cuts = np.flatnonzero(np.any(ordered[1:] != ordered[:-1], axis=1)) + 1
            for grp in np.split(order, cuts):
                if grp.size < 2:
                    continue
                gt = self.col_table[grp]
                if np.all(gt == gt[0]):
                    continue
                members = sorted(grp.tolist())
                for x, y in combinations(members, 2):
                    if self.col_table[x] != self.col_table[y]:
                        cand.add(x * n + y)
                if len(cand) > limit:
                    raise ScreenLimitError(f"LSH candidates exceed MAX_CANDIDATE_PAIRS={limit}; STOP for review")
        self.n_candidates = len(cand)
        return cand

    def verify(self, cand: set) -> list[tuple]:
        """Exact Jaccard of every candidate; keep (ci, cj, inter, union) with J >= COL_J (ci < cj)."""
        n, q = len(self.col_ids), self.col_j
        matches = []
        for code in sorted(cand):
            ci, cj = divmod(code, n)
            na, nb = self.col_ids[ci].size, self.col_ids[cj].size
            if not _ge(min(na, nb), max(na, nb), q):
                continue                                    # J <= min/max < COL_J
            inter = _intersect_count(self.col_ids[ci], self.col_ids[cj])
            union = na + nb - inter
            if _ge(inter, union, q):
                matches.append((ci, cj, inter, union))
        return matches

    # -- stop columns, greedy mapping, weak / strong --
    def stop_set(self, matches, t: int) -> tuple[set, dict]:
        others: dict[int, set] = {}
        for ci, cj, _, _ in matches:
            others.setdefault(ci, set()).add(int(self.col_table[cj]))
            others.setdefault(cj, set()).add(int(self.col_table[ci]))
        return {c for c, s in others.items() if len(s) > t}, {c: len(s) for c, s in others.items()}

    def tuple_set(self, ti: int, cols: tuple) -> set:
        key = (ti, cols)
        if key not in self._tuples:
            self._tuples[key] = _tuple_set(self.tok[ti], list(cols))
        return self._tuples[key]

    def pairs(self, matches, stop: set, t: int) -> tuple[list, list]:
        by_pair: dict[tuple[int, int], list] = {}
        for ci, cj, inter, union in matches:
            if ci in stop or cj in stop:
                continue
            ta, tb = int(self.col_table[ci]), int(self.col_table[cj])
            ca, cb = self.col_index[ci], self.col_index[cj]
            if ta > tb:
                ta, tb, ca, cb = tb, ta, cb, ca
            by_pair.setdefault((ta, tb), []).append((Fraction(inter, union), ca, cb))
        strong, weak = [], []
        for (ta, tb), ms in sorted(by_pair.items()):
            mapped = _greedy(ms)
            if len(mapped) < self.pol["MIN_COLS"]:
                continue
            join = mapped[:self.pol["MAX_JOIN_COLS"]]
            sa = self.tuple_set(ta, tuple(m[1] for m in join))
            sb = self.tuple_set(tb, tuple(m[2] for m in join))
            shared = len(sa & sb)
            smaller = min(len(sa), len(sb))
            enough = smaller >= self.pol["MIN_TUPLES"]
            ev = {"T": t, "pairs": [[m[1], m[2]] for m in mapped], "J": [round(float(m[0]), 6) for m in mapped],
                  "n_mapped": len(mapped), "join_cols": len(join), "n_tuples_a": len(sa), "n_tuples_b": len(sb),
                  "n_shared_tuples": shared, "containment": round(shared / smaller, 6) if smaller else None,
                  "min_tuples_ok": enough}
            rec = _link(self.ids[ta], self.ids[tb], "nds:value_strong", (self.tabs[ta].kind, self.tabs[tb].kind), ev)
            if enough and _ge(shared, smaller, self.contain):
                strong.append(rec)
            else:
                rec["kind"] = "nds:value_weak"
                weak.append(rec)
        return strong, weak

    def summarize(self, stop: set, strong: list, weak: list, t: int) -> dict:
        comps = [c for c in _components((r["a"], r["b"]) for r in strong) if len(c) > 1]
        by_kinds: dict[str, int] = {}
        for r in strong:
            k = "|".join(sorted(r["kinds"]))
            by_kinds[k] = by_kinds.get(k, 0) + 1
        return {"T": t, "primary": t == self.pol["STOP_T"], "stop_columns": len(stop),
                "identities_with_stop_columns": len({int(self.col_table[c]) for c in stop}),
                "weak_pairs_total": len(strong) + len(weak), "weak_only_pairs": len(weak), "strong_pairs": len(strong),
                "strong_components": len(comps), "strong_component_identities": sum(len(c) for c in comps),
                "max_strong_component": max((len(c) for c in comps), default=0),
                "strong_pairs_by_kinds": dict(sorted(by_kinds.items()))}

    # -- names --
    def name_links(self) -> tuple[list, dict]:
        generic, gre = frozenset(self.pol["GENERIC_NAMES"]), re.compile(self.pol["GENERIC_NAME_RE"])
        sets = {i: _name_set(self.raw_names[i], generic, gre) for i in self.ids}
        need = self.pol["NAME_MIN_SHARED"]
        inv: dict[str, list[str]] = {}
        for i in self.ids:
            if sets[i] is not None and len(sets[i]) >= need:
                for nm in sets[i]:
                    inv.setdefault(nm, []).append(i)
        shared: dict[tuple[str, str], int] = {}
        for nm in sorted(inv):
            for a, b in combinations(sorted(inv[nm]), 2):
                shared[(a, b)] = shared.get((a, b), 0) + 1
        out = []
        for (a, b), n_shared in sorted(shared.items()):
            union = len(sets[a] | sets[b])
            if n_shared < need or not _ge(n_shared, union, self.name_j):
                continue
            ev = {"J": round(n_shared / union, 6), "n_shared": n_shared, "n_union": union,
                  "shared": sorted(sets[a] & sets[b])}
            out.append(_link(a, b, "nds:name_jaccard", (self.tabs[self.index[a]].kind, self.tabs[self.index[b]].kind),
                             ev))
        return out, sets

    def name_subsets(self, sets: dict) -> list:
        """Review links to sealed references whose name set is too small for a name link (1 <= |N| < NAME_MIN_SHARED):
        an identity holding every name of the reference (2+ names: N(ref) <= N(i); 1 name: N(i) == N(ref)). Evidence
        carries name hashes only (sealed side)."""
        need = self.pol["NAME_MIN_SHARED"]
        out = []
        for r in self.ids:
            tr = self.tabs[self.index[r]]
            nr = sets[r]
            if tr.kind != "sealed_ref" or nr is None or not (1 <= len(nr) < need):
                continue
            for i in self.ids:
                ti = self.tabs[self.index[i]]
                ni = sets[i]
                if ti.kind == "sealed_ref" or ni is None or not nr <= ni or (len(nr) == 1 and ni != nr):
                    continue
                a, b = sorted((r, i))
                ev = {"n_ref_names": len(nr), "n_names": len(ni), "J": round(len(nr) / len(ni | nr), 6),
                      "rule": "superset" if len(nr) > 1 else "identical",
                      "shared_sha256": sorted(common.sha256_bytes(n.encode("utf-8")) for n in nr)}
                out.append(_link(a, b, "nds:name_subset",
                                 (self.tabs[self.index[a]].kind, self.tabs[self.index[b]].kind), ev))
        return _sort_links(out)

    def stop_relaxed(self, matches, linked: set, t: int) -> list:
        """Sealed-reference pairs that the primary pass did not link but that are strong by the joint-row rule when stop
        columns are kept: reported as nds:value_weak with evidence.stop_relaxed (never a union)."""
        sref = {ti for ti, tab in enumerate(self.tabs) if tab.kind == "sealed_ref"}
        rel = [m for m in matches if int(self.col_table[m[0]]) in sref or int(self.col_table[m[1]]) in sref]
        strong, _ = self.pairs(rel, set(), t)
        out = []
        for r in strong:
            if (r["a"], r["b"]) in linked:
                continue
            r = dict(r, kind="nds:value_weak")
            r["evidence"] = dict(r["evidence"], stop_relaxed=True, class_stop_relaxed="strong")
            out.append(r)
        return out

    # -- BeyondArena pair check (scaled domain always, raw domain when the meta validates; report only) --
    def scaling_from_meta(self, table: Table) -> tuple[tuple | None, str]:
        """Per data column (min, max) from meta.feature_scaling aligned to output_headers minus target_column."""
        meta = self.metas.get(table.identity)
        if not isinstance(meta, Mapping):
            return None, "no_meta"
        fsc = meta.get("feature_scaling")
        oh, fc, tc = meta.get("output_headers"), meta.get("feature_columns"), meta.get("target_column")
        if not isinstance(fsc, Mapping) or not isinstance(oh, list) or not isinstance(fc, list) or not isinstance(tc, str):
            return None, "meta_fields_missing"
        mins, maxs = fsc.get("min"), fsc.get("max")
        heads = [h for h in oh if h != tc]
        if (not isinstance(mins, list) or not isinstance(maxs, list) or len(mins) != len(heads)
                or len(maxs) != len(heads) or len(fc) != table.n_cols or len(set(heads)) != len(heads)
                or len(set(fc)) != len(fc) or not all(isinstance(f, str) and f in heads for f in fc)):
            return None, "lengths_do_not_validate"
        pos = {h: n for n, h in enumerate(heads)}
        try:
            lo = np.array([float(mins[pos[f]]) for f in fc], dtype=np.float64)
            hi = np.array([float(maxs[pos[f]]) for f in fc], dtype=np.float64)
        except (TypeError, ValueError):
            return None, "scaling_not_numeric"
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            return None, "scaling_not_numeric"
        return (lo, hi), "applied"

    def joint(self, mapped: list, ta: np.ndarray, tb: np.ndarray, res: dict) -> dict:
        """Joint-row rule on the first MAX_JOIN_COLS mapped columns of two token matrices (already aligned)."""
        k = min(len(mapped), self.pol["MAX_JOIN_COLS"])
        sa, sb = _tuple_set(ta[:, :k], list(range(k))), _tuple_set(tb[:, :k], list(range(k)))
        shared, smaller = len(sa & sb), min(len(sa), len(sb))
        enough = smaller >= self.pol["MIN_TUPLES"]
        res.update({"join_cols": k, "n_tuples_a": len(sa), "n_tuples_b": len(sb), "n_shared_tuples": shared,
                    "containment": round(shared / smaller, 6) if smaller else None, "min_tuples_ok": enough,
                    "class": "strong" if enough and _ge(shared, smaller, self.contain) else "weak"})
        return res

    def contained_map(self, sets_a: list, sets_b: list, skip_a: set, skip_b: set) -> tuple[list, dict]:
        """Greedy map of informative columns by containment of the smaller distinct set (>= COL_J)."""
        need, q = self.pol["INFORMATIVE_MIN_DISTINCT"], self.col_j
        ms, jac = [], {}
        for ca, sa in enumerate(sets_a):
            if sa is None or sa.size < need or ca in skip_a:
                continue
            for cb, sb in enumerate(sets_b):
                if sb is None or sb.size < need or cb in skip_b:
                    continue
                inter = _intersect_count(sa, sb)
                if _ge(inter, min(sa.size, sb.size), q):
                    ms.append((Fraction(inter, min(sa.size, sb.size)), ca, cb))
                    jac[(ca, cb)] = round(inter / (sa.size + sb.size - inter), 6)
        return _greedy(ms), jac

    @staticmethod
    def _summary(mapped: list, jac: dict) -> dict:
        return {"pairs": [[m[1], m[2]] for m in mapped], "col_containment": [round(float(m[0]), 6) for m in mapped],
                "J": [jac[(m[1], m[2])] for m in mapped], "n_mapped": len(mapped), "containment": None,
                "class": "none"}

    def scaled_check(self, ta: int, tb: int, skip_a: set, skip_b: set) -> dict:
        xa, xb = self.tok[ta], self.tok[tb]
        sets_a = [np.unique(c[~np.isnan(c)]) for c in xa.T]
        sets_b = [np.unique(c[~np.isnan(c)]) for c in xb.T]
        mapped, jac = self.contained_map(sets_a, sets_b, skip_a, skip_b)
        res = self._summary(mapped, jac)
        if len(mapped) < self.pol["MIN_COLS"]:
            return res
        return self.joint(mapped, xa[:, [m[1] for m in mapped]], xb[:, [m[2] for m in mapped]], res)

    def raw_check(self, ta: int, tb: int, skip_a: set, skip_b: set, ba_is_a: bool, lo, hi) -> dict:
        """BeyondArena raw values mapped onto the other table's grid with its meta (min, max), ROUND decimals."""
        r = self.pol["ROUND"]
        bi, oi = (ta, tb) if ba_is_a else (tb, ta)
        raw, other = self.tabs[bi].raw_values, self.tok[oi]

        def onto(col: np.ndarray, co: int) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return tokens((col - lo[co]) / (hi[co] - lo[co]), r)

        sets_o = [np.unique(c[~np.isnan(c)]) if hi[n] > lo[n] else None for n, c in enumerate(other.T)]
        uniq_raw = [np.unique(c[~np.isnan(c)]) for c in raw.T]
        need, q = self.pol["INFORMATIVE_MIN_DISTINCT"], self.col_j
        skip_b_side, skip_o = (skip_a, skip_b) if ba_is_a else (skip_b, skip_a)
        ms, jac = [], {}
        for co, so in enumerate(sets_o):
            if so is None or so.size < need or co in skip_o:
                continue
            for cb, ub in enumerate(uniq_raw):
                if cb in skip_b_side or ub.size < need:
                    continue
                sb = np.unique(onto(ub, co))
                sb = sb[~np.isnan(sb)]
                if sb.size < need:
                    continue
                inter = _intersect_count(sb, so)
                if _ge(inter, min(sb.size, so.size), q):
                    key = (cb, co) if ba_is_a else (co, cb)
                    ms.append((Fraction(inter, min(sb.size, so.size)), key[0], key[1]))
                    jac[key] = round(inter / (sb.size + so.size - inter), 6)
        mapped = _greedy(ms)
        res = self._summary(mapped, jac)
        if len(mapped) < self.pol["MIN_COLS"]:
            return res
        pairs = [(m[1], m[2]) if ba_is_a else (m[2], m[1]) for m in mapped]          # (ba col, other col)
        ba_proj = np.column_stack([onto(raw[:, cb], co) for cb, co in pairs])
        o_proj = other[:, [co for _, co in pairs]]
        return self.joint(mapped, ba_proj if ba_is_a else o_proj, o_proj if ba_is_a else ba_proj, res)

    def beyondarena_checks(self, linked: set, stop_cols: dict) -> list:
        ba = {t.identity for t in self.tabs if t.kind == "beyondarena"}
        requested = set(self.ba_pairs)
        pairs = requested | {p for p in linked if (p[0] in ba) != (p[1] in ba)}
        rank = {"strong": 2, "weak": 1, "none": 0}
        out = []
        for a, b in sorted(pairs):
            ta, tb = self.index[a], self.index[b]
            rec = {"a": a, "b": b, "kinds": [self.tabs[ta].kind, self.tabs[tb].kind], "requested": (a, b) in requested}
            if (a in ba) == (b in ba):
                out.append(dict(rec, status="not_a_beyondarena_pair", **{"class": "none"}))
                continue
            skip_a, skip_b = set(stop_cols.get(a, ())), set(stop_cols.get(b, ()))
            scaled = self.scaled_check(ta, tb, skip_a, skip_b)
            ba_is_a = a in ba
            scaling, status = self.scaling_from_meta(self.tabs[tb] if ba_is_a else self.tabs[ta])
            if scaling is None or self.tabs[ta if ba_is_a else tb].raw_values is None:
                raw = {"status": status if scaling is None else "no_raw_values"}
            else:
                raw = dict({"status": "applied"}, **self.raw_check(ta, tb, skip_a, skip_b, ba_is_a, *scaling))
            best = max((scaled["class"], raw.get("class", "none")), key=lambda c: rank[c])
            out.append(dict(rec, status="checked", scaled=scaled, raw=raw, **{"class": best}))
        return out

    # -- driver --
    def run(self) -> ScreenResult:
        self.tokenize()
        content = self.content_links()
        matches = self.verify(self.candidates())
        t0 = self.pol["STOP_T"]
        sensitivity, by_t = {}, {}
        for t in [t0] + [t for t in self.pol["SENS_T"] if t != t0]:
            stop_t, reach_t = self.stop_set(matches, t)
            strong_t, weak_t = self.pairs(matches, stop_t, t)
            sensitivity[str(t)] = self.summarize(stop_t, strong_t, weak_t, t)
            by_t[t] = (stop_t, reach_t, strong_t, weak_t)
        stop, reach, strong, weak = by_t[t0]
        names, sets = self.name_links()
        stop_cols: dict[str, list[int]] = {}
        detail = []
        for c in sorted(stop, key=lambda c: (int(self.col_table[c]), self.col_index[c])):
            ident = self.ids[int(self.col_table[c])]
            stop_cols.setdefault(ident, []).append(self.col_index[c])
            detail.append({"identity": ident, "col": self.col_index[c], "n_other_identities": reach[c],
                           "n_distinct": int(self.col_tokens[c].size)})
        linked = {(r["a"], r["b"]) for r in strong + weak + names + content}
        checks = self.beyondarena_checks(linked, stop_cols)
        relaxed = self.stop_relaxed(matches, linked, t0)
        subsets = self.name_subsets(sets)
        tables, by_kind, cov = {}, {}, {}
        for ti, t in enumerate(self.tabs):
            i = t.identity
            n_names = None if sets[i] is None else len(sets[i])
            nonstop = self.n_informative[ti] - len(stop_cols.get(i, []))
            values_ok = nonstop >= self.pol["MIN_COLS"]          # pairs() drops stop columns (review fix SC-1)
            names_ok = n_names is not None and n_names >= self.pol["NAME_MIN_SHARED"]
            tables[i] = {"kind": t.kind, "n_rows": t.n_rows, "n_feature_columns": t.n_cols,
                         "n_informative": self.n_informative[ti], "n_stop_columns": len(stop_cols.get(i, [])),
                         "n_informative_nonstop": nonstop,
                         "n_dropped_values": int(np.isnan(t.values).sum()), "tokens_sha256": self.tok_sha[i],
                         "row_multiset_sha256": self.rms[i], "n_complete_rows": self.rms_rows[i],
                         "names_known": sets[i] is not None, "n_names": n_names,
                         "screen_coverage": coverage_class(values_ok, names_ok), "source": t.source}
            by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
            cov[tables[i]["screen_coverage"]] = cov.get(tables[i]["screen_coverage"], 0) + 1
        digest = common.sha256_bytes(common.canonical(
            [[i, tables[i]["kind"], tables[i]["n_rows"], tables[i]["n_feature_columns"], tables[i]["tokens_sha256"],
              tables[i]["source"]] for i in self.ids]))
        counts = {"tables": len(self.tabs), "tables_by_kind": dict(sorted(by_kind.items())),
                  "feature_columns": int(sum(t.n_cols for t in self.tabs)),
                  "informative_columns": len(self.col_ids), "distinct_tokens": int(self.universe.size),
                  "lsh_candidate_pairs": self.n_candidates, "lsh_matches": len(matches),
                  "content_equal_pairs": len(content), "stop_columns": len(detail), "strong_pairs": len(strong),
                  "weak_only_pairs": len(weak), "name_links": len(names), "coverage": dict(sorted(cov.items())),
                  "beyondarena_checks": len(checks),
                  "beyondarena_checks_strong": sum(1 for r in checks if r["class"] == "strong"),
                  "sealed_ref_stop_relaxed_pairs": len(relaxed), "name_subset_links": len(subsets)}
        return ScreenResult(SCHEMA, self.pol, digest, tables, dict(self.rms), content, strong,
                            _sort_links(weak + relaxed), names, stop_cols, detail, sensitivity, checks, counts,
                            name_subsets=subsets)


def near_dup_screen(tables, names: Mapping | None = None, *, policy: Mapping | None = None,
                    metas: Mapping | None = None, ba_pairs: Iterable = ()) -> ScreenResult:
    """Run the P2 near-duplicate screen over `tables` (Table objects, or {identity: Table}).

    names: {identity: raw names or None} for identities that have tables. For epoch identities and sealed references
      pass names_from_meta(meta); None (or absent) means names are unknown. A BeyondArena table brings its own
      parquet column names.
    policy: alias-sources.json['policy'] (only the screen keys are read; see DEFAULT_SCREEN_POLICY).
    metas / ba_pairs: the BeyondArena pair check (report only; module docstring step 10). Every pair of a BeyondArena
      table and another table that the screen links (content, values or names) is checked, plus each pair listed in
      ba_pairs, e.g. [('beyondarena:houses', 'item:<id>'), ...] for the curated alias members. metas maps
      identity -> meta (feature_scaling, output_headers, feature_columns, target_column) and enables the raw domain.
    The result is a pure function of the table values, names, metas and policy (no RNG, no clock, sorted outputs).
    """
    return _Screen(tables, names, policy, metas, ba_pairs).run()


__all__ = [
    "SCHEMA", "MERSENNE_P", "TABLE_KINDS", "LINK_KINDS", "UNION_LINK_KINDS", "COVERAGE_CLASSES", "SCREEN_FILES",
    "DEFAULT_SCREEN_POLICY", "ScreenInputError", "ScreenLimitError", "Table", "ScreenResult", "screen_policy",
    "make_table", "read_pinned", "parse_csv_matrix", "load_data_object", "load_sealed_ref", "load_beyondarena",
    "names_from_meta", "normalize_name", "name_set", "tokens", "row_multiset_sha256", "minhash_params",
    "token_hash64", "mulmod61", "affine_mod61", "minhash_signatures", "coverage_class", "near_dup_screen",
]
