"""Four dedup gates, of which exactly one is load-bearing.

A campaign that runs ~70 roles x 3 epochs against the same corpus will rediscover the same feature
many times. That is not a failure of the roles -- it is what independent search *does*. The failure
mode is the controller counting each rediscovery as a new idea, so the survivor set looks like 40
theories when it is 6 theories written 40 ways. Everything downstream (redundancy feedback,
family coverage, the "novel"/"redundant" token an agent is allowed to see) is then a lie.

The gates, weakest to strongest:

1. `structure_signature` -- shape of the spec. Nearly worthless alone: every spec conforms to the
   same schema, so this collides constantly. It is only ever used ANDed with (2).
2. `semantic_fingerprint` -- hash of the mechanism prose. Catches copy-paste and little else. Two
   agents describing the same arithmetic in different words defeat it completely, and that is the
   common case, not the exotic one.
3. `canonical_formula` -- normalized computation text. Real, but only reaches proposals that ship a
   parseable formula, and is defeated by algebraic restatement (`a*b + a*c` vs `a*(b+c)`).
4. `numeric_signature` -- blake2b over the ACTUAL COMPUTED COLUMN VALUES. This is the gate that
   works. It does not care what the agent called it, how the spec nested, or what words the card
   used; two proposals that compute the same numbers hash the same. Nothing else catches
   convergent rediscovery, and convergent rediscovery is the dominant duplicate mode.

Backing it up is a near-duplicate check: |pearson| > 0.999 against every registered column. Exact
value-hashing misses `x` vs `2*x` vs `x + 1e-12`, all of which carry identical information to the
model. The two together are what make the redundancy signal honest.

**The baseline is seeded first, on purpose.** `seed_from_baseline` loads the numeric signatures of
the champion's 1,331 columns before any candidate is checked. Without that ordering, a role that
reinvents an existing baseline column is registered as novel and celebrated, when the correct
outcome is rejection -- the whole fitness question is *marginal* lift over that baseline.
"""

from __future__ import annotations

import ast
import base64
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import CAMPAIGN_SEED, assert_no_forbidden_inputs

# --- Tunables -----------------------------------------------------------------------------------
VALUE_DECIMALS = 9  # values are rounded to 1e-9 before hashing
NEAR_DUP_R = 0.999  # |pearson| above this is a near-duplicate
SKETCH_ROWS = 2048  # rows kept per column for the durable pearson gate
SKETCH_SEED = CAMPAIGN_SEED ^ 0x5A17  # distinct from every fold seed
FORMULA_KEYS = ("formula", "expr", "expression", "compute", "equation", "body")
MECHANISM_KEYS = ("mechanism", "theory", "rationale", "description", "hypothesis", "summary")

_WORD_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "this",
        "to",
        "we",
        "which",
        "with",
    }
)

# Operators whose operand order carries no meaning. Sorting these is what makes `a+b` and `b+a`
# the same formula; leaving them unsorted is why naive text hashing of formulas barely helps.
_COMMUTATIVE = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
_OP_SYMBOL = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.USub: "neg",
    ast.UAdd: "pos",
    ast.Invert: "inv",
    ast.Not: "not",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.And: "and",
    ast.Or: "or",
}


class NoveltyError(RuntimeError):
    """Raised when the index is asked to do something that would corrupt its own comparisons."""


# --- Gate 1: canonical formula -------------------------------------------------------------------


def _num(value) -> str:
    """One spelling per number. `2`, `2.0` and `2.000` all collapse to `2`."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == 0.0:
        return "0"  # also folds -0.0
    return f"{f:.12g}"


def _canon_node(node) -> str:
    """Prefix s-expression form of an AST node, with commutative operands sorted."""
    if isinstance(node, ast.Expression):
        return _canon_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, (int, float)):
            return _num(node.value)
        if node.value is None:
            return "none"
        return json.dumps(str(node.value).strip().casefold())
    if isinstance(node, ast.Name):
        return node.id.casefold()
    if isinstance(node, ast.Attribute):
        return f"{_canon_node(node.value)}.{node.attr.casefold()}"
    if isinstance(node, ast.BinOp):
        sym = _OP_SYMBOL.get(type(node.op), type(node.op).__name__.casefold())
        operands = _flatten(node, type(node.op))
        parts = [_canon_node(o) for o in operands]
        if isinstance(node.op, _COMMUTATIVE):
            parts.sort()
        return f"({sym} {' '.join(parts)})"
    if isinstance(node, ast.UnaryOp):
        sym = _OP_SYMBOL.get(type(node.op), type(node.op).__name__.casefold())
        return f"({sym} {_canon_node(node.operand)})"
    if isinstance(node, ast.BoolOp):
        sym = _OP_SYMBOL.get(type(node.op), "bool")
        parts = sorted(_canon_node(v) for v in node.values)
        return f"({sym} {' '.join(parts)})"
    if isinstance(node, ast.Compare):
        parts = [_canon_node(node.left)]
        for op, cmp in zip(node.ops, node.comparators):
            parts.append(_OP_SYMBOL.get(type(op), type(op).__name__.casefold()))
            parts.append(_canon_node(cmp))
        return f"(cmp {' '.join(parts)})"
    if isinstance(node, ast.Call):
        fn = _canon_node(node.func)
        args = [_canon_node(a) for a in node.args]
        # keyword order is not meaningful; positional order is
        kw = sorted(f"{k.arg}={_canon_node(k.value)}" for k in node.keywords if k.arg)
        return f"(call {fn} {' '.join(args + kw)})".replace("  ", " ")
    if isinstance(node, (ast.List, ast.Tuple)):
        return f"(seq {' '.join(_canon_node(e) for e in node.elts)})"
    if isinstance(node, ast.Set):
        return f"(set {' '.join(sorted(_canon_node(e) for e in node.elts))})"
    if isinstance(node, ast.Dict):
        pairs = sorted(
            f"{_canon_node(k) if k is not None else 'none'}:{_canon_node(v)}"
            for k, v in zip(node.keys, node.values)
        )
        return f"(map {' '.join(pairs)})"
    if isinstance(node, ast.Subscript):
        return f"(idx {_canon_node(node.value)} {_canon_node(node.slice)})"
    return f"({type(node).__name__.casefold()})"


def _flatten(node, op_type) -> list:
    """`a+(b+c)` -> [a, b, c], so associativity does not change the canonical form."""
    out = []
    for side in (node.left, node.right):
        if isinstance(side, ast.BinOp) and isinstance(side.op, op_type):
            out.extend(_flatten(side, op_type))
        else:
            out.append(side)
    return out


def _canon_text(text: str) -> str:
    """Canonicalize one formula string; fall back to whitespace/case normalization."""
    stripped = str(text).strip()
    if not stripped:
        return ""
    try:
        return _canon_node(ast.parse(stripped, mode="eval"))
    except (SyntaxError, ValueError, RecursionError):
        return " ".join(stripped.casefold().split())


def canonical_formula(spec_output) -> str:
    """Normalized textual form of the computation a spec performs.

    Accepts a bare formula string or a nested spec dict/list. Strings under `FORMULA_KEYS` are
    parsed as expressions and canonicalized (commutative operands sorted, associativity flattened,
    numbers given one spelling); everything else is normalized as text with dict keys sorted.

    Defeated by algebraic restatement -- `(a*b)+(a*c)` and `a*(b+c)` are different strings here.
    That is the reason `numeric_signature` exists and this is not the gate that decides anything.
    """

    def walk(node, in_formula: bool) -> str:
        if isinstance(node, str):
            return _canon_text(node) if in_formula else " ".join(node.casefold().split())
        if isinstance(node, bool):
            return "true" if node else "false"
        if isinstance(node, (int, float, np.integer, np.floating)):
            return _num(node)
        if node is None:
            return "none"
        if isinstance(node, dict):
            parts = []
            for key in sorted(node, key=str):
                kl = str(key).casefold()
                parts.append(f"{kl}={walk(node[key], in_formula or kl in FORMULA_KEYS)}")
            return "{" + ",".join(parts) + "}"
        if isinstance(node, (list, tuple, set, frozenset)):
            items = [walk(v, in_formula) for v in node]
            if isinstance(node, (set, frozenset)):
                items.sort()
            return "[" + ",".join(items) + "]"
        if isinstance(node, np.ndarray):
            return "[" + ",".join(_num(v) for v in node.ravel().tolist()) + "]"
        return " ".join(str(node).casefold().split())

    return walk(spec_output, isinstance(spec_output, str))


# --- Gate 2: semantic fingerprint -----------------------------------------------------------------


def semantic_fingerprint(card) -> str:
    """blake2b over the normalized mechanism prose of a theory card.

    Catches copy-paste and near-verbatim restatement. It does NOT catch two agents describing the
    same arithmetic in different words, which is the ordinary case -- so this is never used as a
    rejection gate on its own (see `NoveltyIndex.check`, which requires it to agree with
    `structure_signature` before calling anything a restatement).
    """
    if isinstance(card, str):
        text = card
    elif isinstance(card, dict):
        chunks = [str(card[k]) for k in MECHANISM_KEYS if card.get(k)]
        # No recognized mechanism field: fall back to the whole card so distinct cards still
        # fingerprint distinctly, rather than every card collapsing onto the empty string.
        text = " ".join(chunks) if chunks else json.dumps(card, sort_keys=True, default=str)
    else:
        text = str(card)

    words = [w for w in _WORD_RE.findall(text.casefold()) if w not in _STOPWORDS]
    normalized = " ".join(words)
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


# --- Gate 3: structure signature -------------------------------------------------------------------


def _shape(node) -> str:
    """Type/key skeleton with all scalar VALUES erased. List lengths are collapsed."""
    if isinstance(node, dict):
        return "{" + ",".join(f"{str(k).casefold()}:{_shape(node[k])}" for k in sorted(node, key=str)) + "}"
    if isinstance(node, (list, tuple, set, frozenset)):
        inner = sorted({_shape(v) for v in node})
        return "[" + "|".join(inner) + "]"
    if isinstance(node, np.ndarray):
        return f"a{node.ndim}"
    if isinstance(node, bool):
        return "b"
    if isinstance(node, (int, np.integer)):
        return "i"
    if isinstance(node, (float, np.floating)):
        return "f"
    if node is None:
        return "n"
    if isinstance(node, str):
        return "s"
    return "o"


def structure_signature(spec) -> str:
    """Deterministic 64-bit signature of the SPEC SHAPE -- keys, nesting and types, not values.

    Two specs that differ only in constants collide here by design. Because every proposal conforms
    to one schema, collisions are the norm, so this can only ever narrow another gate.
    """
    return hashlib.blake2b(_shape(spec).encode("utf-8"), digest_size=8).hexdigest()


# --- Gate 4: numeric signature (the decisive one) ---------------------------------------------------


def numeric_signature(values: np.ndarray) -> str:
    """blake2b over the actual computed column values. THE gate.

    Determinism requires four normalizations that are easy to omit and silently fatal:

    * round to 1e-9, so last-bit float differences do not read as different features;
    * fold -0.0 into 0.0, which round() does not do and which has a distinct byte pattern;
    * hash NaN POSITIONS as a separate mask rather than NaN bytes, since NaN has many payloads;
    * keep +inf/-inf distinct from NaN and from each other.

    Two limits worth stating, because both are inherent and neither is a bug:

    * Rounding is a BUCKETING, so it has edges. Two values differing by 1e-12 that straddle a
      bucket boundary still hash differently -- empirically ~1 value in 1,000, so a 400-row column
      perturbed by 1e-12 splits about a third of the time. No rounding scheme avoids this.
    * Rounding is absolute, not relative, so columns on a very large scale are compared at a
      precision below their float resolution.

    Both are covered by `NoveltyIndex`'s |pearson| > 0.999 near-duplicate gate, which reads 1.0 for
    a float-perturbed rediscovery. That gate also catches rescalings like `x` vs `2*x`, which are
    exact-distinct here but carry identical information to the model. Exact hashing and correlation
    are not redundant with each other; each covers the other's blind spot.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    nan_mask = np.isnan(arr)
    posinf = np.isposinf(arr)
    neginf = np.isneginf(arr)

    finite = np.where(nan_mask | posinf | neginf, 0.0, arr)
    finite = np.round(finite, VALUE_DECIMALS)
    finite = finite + 0.0  # folds -0.0 -> 0.0
    finite[finite == 0.0] = 0.0  # belt and braces on the -0.0 byte pattern

    h = hashlib.blake2b(digest_size=16)
    h.update(f"zyal_numeric_v1|{arr.shape[0]}x{arr.shape[1]}|{VALUE_DECIMALS}".encode())
    h.update(np.ascontiguousarray(finite, dtype="<f8").tobytes())
    h.update(np.packbits(np.ascontiguousarray(nan_mask)).tobytes())
    h.update(np.packbits(np.ascontiguousarray(posinf)).tobytes())
    h.update(np.packbits(np.ascontiguousarray(neginf)).tobytes())
    return h.hexdigest()


# --- Sketching for the durable near-duplicate gate ---------------------------------------------------


def _sketch_indices(n_rows: int) -> np.ndarray:
    """Fixed seeded row subsample. Every column in one index MUST use the same rows or the
    pearson comparison is meaningless, which is why `NoveltyIndex` pins `n_rows`."""
    if n_rows <= SKETCH_ROWS:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.RandomState(SKETCH_SEED)
    return np.sort(rng.choice(n_rows, size=SKETCH_ROWS, replace=False)).astype(np.int64)


def _zsketch(sketch: np.ndarray):
    """Unit-norm centred sketch, or None when the column cannot carry a correlation."""
    x = np.asarray(sketch, dtype=np.float64).ravel()
    good = np.isfinite(x)
    if good.sum() < 2:
        return None
    x = np.where(good, x, x[good].mean())
    x = x - x.mean()
    norm = float(np.linalg.norm(x))
    if norm < 1e-12:
        return None
    return x / norm


def _encode(sketch: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(sketch, dtype="<f4").tobytes()).decode("ascii")


def _decode(blob: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(blob), dtype="<f4").astype(np.float64)


def _column_1d(values: np.ndarray) -> np.ndarray:
    """Collapse a candidate to one comparable vector for the pearson gate.

    A multi-column block is reduced to its row-mean. That is a lossy summary and only ever used to
    RAISE a near-duplicate flag; exact identity always goes through `numeric_signature`.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] == 1:
        return arr[:, 0]
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(np.isinf(arr), np.nan, arr), axis=1)


# --- Durable index ------------------------------------------------------------------------------


class NoveltyIndex:
    """Append-only novelty registries under `<run_root>/novelty/`.

    Files:
      `baseline.jsonl` -- champion columns, seeded before any candidate is checked
      `columns.jsonl`  -- every registered candidate column
      `meta.json`      -- pinned row count, so sketches stay comparable across restarts

    Appends are single `O_APPEND` writes under an exclusive `flock`, then `fsync`. That is the
    atomic primitive for a LOG; the tmp-then-`os.replace` pattern used elsewhere in this package is
    for whole artifacts, and applying it to a growing log would be O(n^2) and would silently drop a
    concurrent writer's records. `meta.json` is a whole artifact and does use tmp-then-replace.
    """

    def __init__(self, run_root, *, near_dup_r: float = NEAR_DUP_R) -> None:
        self.root = Path(run_root) / "novelty"
        self.root.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.root / "baseline.jsonl"
        self.columns_path = self.root / "columns.jsonl"
        self.meta_path = self.root / "meta.json"
        self.near_dup_r = float(near_dup_r)

        self.n_rows: int | None = None
        self._numeric: dict[str, str] = {}  # numeric sig -> feature_id
        self._formula: dict[str, str] = {}
        self._semantic: dict[str, list[str]] = {}  # semantic sig -> feature_ids
        self._structure: dict[str, set[str]] = {}  # feature_id -> structure sig
        self._ids: list[str] = []
        self._zsketches: dict[str, np.ndarray] = {}
        self._stack: np.ndarray | None = None
        self._stack_ids: list[str] = []
        self._load()

    # -- persistence -----------------------------------------------------------------------------
    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                self.n_rows = json.loads(self.meta_path.read_text()).get("n_rows")
            except (json.JSONDecodeError, OSError):
                self.n_rows = None
        for path in (self.baseline_path, self.columns_path):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._absorb(json.loads(line))
                    except json.JSONDecodeError:
                        # A torn final line from a killed process is skipped, not fatal. The log is
                        # append-only, so everything before it is still valid.
                        continue

    def _absorb(self, rec: dict) -> None:
        fid = rec["feature_id"]
        self._numeric.setdefault(rec["numeric"], fid)
        if rec.get("formula"):
            self._formula.setdefault(rec["formula"], fid)
        if rec.get("semantic"):
            self._semantic.setdefault(rec["semantic"], []).append(fid)
        if rec.get("structure"):
            self._structure[fid] = rec["structure"]
        if rec.get("sketch"):
            z = _zsketch(_decode(rec["sketch"]))
            if z is not None:
                self._zsketches[fid] = z
                self._stack = None
        self._ids.append(fid)

    def _append(self, path: Path, rec: dict) -> None:
        payload = (json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _write_meta(self) -> None:
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {"n_rows": self.n_rows, "sketch_rows": SKETCH_ROWS, "sketch_seed": SKETCH_SEED},
                sort_keys=True,
            )
        )
        os.replace(tmp, self.meta_path)

    # -- shape guards ----------------------------------------------------------------------------
    def _pin_rows(self, n: int) -> None:
        if self.n_rows is None:
            self.n_rows = int(n)
            self._write_meta()
        elif int(n) != self.n_rows:
            raise NoveltyError(
                f"column has {n} rows but this index is pinned to {self.n_rows}. Sketches are a "
                "fixed seeded row subsample; comparing across row counts would silently produce "
                "meaningless correlations."
            )

    def _stacked(self):
        if self._stack is None:
            self._stack_ids = [i for i in self._ids if i in self._zsketches]
            if self._stack_ids:
                self._stack = np.vstack([self._zsketches[i] for i in self._stack_ids])
            else:
                self._stack = np.zeros((0, 0))
        return self._stack, self._stack_ids

    # -- public API ------------------------------------------------------------------------------
    def seed_from_baseline(self, matrix: np.ndarray, names, *, check_names: bool = True) -> int:
        """Register the champion's columns BEFORE any candidate is checked.

        Ordering is the whole point. A role that reinvents an existing baseline column must be
        rejected as a duplicate, not registered as a discovery -- the fitness question is marginal
        lift over exactly these columns. Returns the number newly seeded.
        """
        mat = np.asarray(matrix)
        names = [str(n) for n in names]
        if mat.ndim != 2 or mat.shape[1] != len(names):
            raise NoveltyError(f"matrix {mat.shape} does not match {len(names)} names")
        if check_names:
            assert_no_forbidden_inputs(names)
        self._pin_rows(mat.shape[0])

        idx = _sketch_indices(mat.shape[0])
        seeded = 0
        for j, name in enumerate(names):
            col = mat[:, j]
            sig = numeric_signature(col)
            if sig in self._numeric:
                continue
            fid = f"baseline::{name}"
            rec = {
                "feature_id": fid,
                "source": "baseline",
                "name": name,
                "numeric": sig,
                "formula": "",
                "semantic": "",
                "structure": "",
                "sketch": _encode(_column_1d(col)[idx]),
            }
            self._append(self.baseline_path, rec)
            self._absorb(rec)
            seeded += 1
        return seeded

    def check(self, column_values: np.ndarray, spec, card) -> tuple[bool, str, str | None]:
        """-> (is_novel, reason, duplicate_of).

        Gate order is decisiveness order. `structure_signature` never rejects on its own -- every
        spec shares a schema, so it would reject everything; it only narrows the prose gate.
        """
        col = np.asarray(column_values, dtype=np.float64)
        n = col.shape[0] if col.ndim else 1
        self._pin_rows(n)

        flat = _column_1d(col)
        finite = np.isfinite(flat)
        if finite.sum() < 2:
            return False, "degenerate_all_missing", None
        if float(np.nanstd(np.where(finite, flat, np.nan))) == 0.0:
            # A constant column cannot carry a main effect and makes pearson undefined.
            return False, "degenerate_constant", None

        sig = numeric_signature(col)
        if sig in self._numeric:
            return False, "numeric_duplicate", self._numeric[sig]

        z = _zsketch(flat[_sketch_indices(n)])
        if z is not None:
            stack, ids = self._stacked()
            if len(ids):
                r = np.abs(stack @ z)
                k = int(np.argmax(r))
                if float(r[k]) > self.near_dup_r:
                    return False, f"near_duplicate_r{float(r[k]):.5f}", ids[k]

        formula = canonical_formula(spec)
        if formula and formula in self._formula:
            return False, "formula_duplicate", self._formula[formula]

        semantic = semantic_fingerprint(card)
        structure = structure_signature(spec)
        for fid in self._semantic.get(semantic, ()):
            if self._structure.get(fid) == structure:
                return False, "semantic_structure_restatement", fid

        return True, "novel", None

    def register(
        self, *, feature_id: str, column_values: np.ndarray, spec, card, extra: dict | None = None
    ) -> None:
        """Atomically append one column to the registry.

        `feature_id` is REQUIRED and is never minted here. The controller owns every ID; an index
        that invented its own would break the one invariant the campaign audit depends on.
        """
        if not feature_id or not isinstance(feature_id, str):
            raise NoveltyError("feature_id must be supplied by the controller; the index never mints IDs")
        col = np.asarray(column_values, dtype=np.float64)
        n = col.shape[0] if col.ndim else 1
        self._pin_rows(n)

        rec = {
            "feature_id": feature_id,
            "source": "candidate",
            "numeric": numeric_signature(col),
            "formula": canonical_formula(spec),
            "semantic": semantic_fingerprint(card),
            "structure": structure_signature(spec),
            "sketch": _encode(_column_1d(col)[_sketch_indices(n)]),
        }
        if extra:
            rec["extra"] = extra
        self._append(self.columns_path, rec)
        self._absorb(rec)

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def feature_ids(self) -> list[str]:
        return list(self._ids)


# --- Batch reconciliation -------------------------------------------------------------------------


def reconcile_duplicates(matrix: np.ndarray, names, index: NoveltyIndex):
    """-> (kept_matrix, kept_names, rejections). Duplicate columns are PHYSICALLY DROPPED.

    Annotating duplicates instead of removing them is the mistake this function exists to prevent:
    an annotated duplicate still enters the model, still inflates the feature count, and still
    dilutes every per-column importance. The returned matrix is a real subset.

    Columns are checked in order and registered as they survive, so this also deduplicates WITHIN
    the batch, not merely against the existing index. Only the numeric and near-duplicate gates can
    fire here -- reconciliation sees values and names, not specs or cards.
    """
    mat = np.asarray(matrix)
    names = [str(n) for n in names]
    if mat.ndim != 2 or mat.shape[1] != len(names):
        raise NoveltyError(f"matrix {mat.shape} does not match {len(names)} names")

    keep: list[int] = []
    rejections: list[dict] = []
    for j, name in enumerate(names):
        col = mat[:, j]
        spec = {"column": name}
        card = {"mechanism": name}
        is_novel, reason, dup_of = index.check(col, spec, card)
        if is_novel:
            index.register(feature_id=name, column_values=col, spec=spec, card=card)
            keep.append(j)
        else:
            rejections.append(
                {
                    "name": name,
                    "column_index": j,
                    "reason": reason,
                    "duplicate_of": dup_of,
                    "numeric": numeric_signature(col),
                }
            )

    kept_idx = np.array(keep, dtype=np.int64)
    kept_matrix = mat[:, kept_idx] if len(kept_idx) else mat[:, :0]
    return kept_matrix, [names[j] for j in keep], rejections
