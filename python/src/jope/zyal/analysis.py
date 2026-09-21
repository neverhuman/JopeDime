"""Analysis mode: an agent proposes a *measurement*, not a model, and gets contrasts back.

This is the one place in ZYAL where agent-authored code sees real labels, so containment is
structural rather than advisory:

* **Only the SELECT split is ever materialized.** `materialize_select_shard` writes a directory
  literally named ``select`` and refuses (LeakageError) if a single non-select row reaches it. It
  re-derives the split from the dataset key with `split_of` rather than trusting `rows.split`, so a
  doctored RowTable does not become a leak. DEV and TEST are not mounted, so code that goes looking
  for them gets ENOENT, not a policy warning.
* **Dataset keys never leave.** The shard exports an opaque contiguous ``group`` index. Raw
  `dataset` hashes, `robust_score`, `oof_r2` and `finalist_rank` all match
  `FORBIDDEN_INPUT_SUBSTRINGS` and are simply absent from the shard.

The second half of the module exists because of a measured fact about this corpus, not tidiness:
op-pair lift mined on SELECT has split-half reliability r=0.50 with 68% sign agreement, and it did
**not** improve with more support. An unreplicated data-mined pattern here is close to a coin flip.
So a finding is not a finding until `replicate()` re-measures it on a disjoint half of SELECT.

That only works if a "contrast" is machine-executable, which is why `CONTRAST_GRAMMAR` is a closed
language over the parsed genome:

    rate(head_op=gbm) in top_decile vs bottom_decile

`effect` is always the lift RATIO rate_a / rate_b (Haldane-Anscombe smoothed so it stays finite),
which also gives the findings contract its teeth: a prospector cannot smuggle "I trained a GBM and
got 0.31" through a field that has to parse as a rate contrast over grammar ops.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jope.genome_perf import split_of
from jope.genome_perf.parse import COST_CLASSES, parse
from jope.grammars import FAMILY_LIST_V3, grammar_for

from . import SCHEMA_VERSION, LeakageError, assert_no_forbidden_inputs, security

# --- Controller-owned limits ---------------------------------------------------------------------
FINDINGS_BYTE_CAP = 8_192  # the findings payload an agent may return
MAX_FINDINGS = 12
MAX_CONTRASTS = 6  # contrasts must be PRE-COMMITTED in the spec
MAX_QUESTION_CHARS = 400
MIN_SUPPORT_N = 30  # a floor against anecdotes; NOT a reliability fix (see module doc)
REPLICATION_MIN_RATIO = 0.5  # replication half must retain half the log-magnitude
DEFAULT_ENGINE = "featmath"
ANALYSIS_HARNESS = Path(__file__).resolve().parents[3] / "ops" / "zyal" / "analysis_runner.py"

FINDING_KEYS = ("claim", "effect", "support_n", "contrast")

# Tokens that mark a *fitted model result*. Enforcement is token-based and vocabulary-aware: `gbm`
# is a real featmath head op, so "head_op=gbm appears 3.2x more often" is a described contrast and
# must pass, while "I trained a GBM regressor" must not. See `_model_result_tokens`.
MODEL_RESULT_TOKENS = frozenset(
    {
        "auc",
        "r2",
        "rmse",
        "accuracy",
        "trained",
        "model",
        "gbm",
        "regressor",
        "classifier",
        "spearman",
        "correlation",
    }
)

# --- Contrast grammar ----------------------------------------------------------------------------
CONTRAST_GRAMMAR = "rate(<predicate>) in <stratum> vs <stratum>"
_CONTRAST_RE = re.compile(r"^rate\(\s*(?P<pred>[^()]+?)\s*\)\s+in\s+(?P<a>\w+)\s+vs\s+(?P<b>\w+)$")
_PRED_RE = re.compile(r"^(?P<attr>[a-z_][a-z0-9_]*)(?P<op>>=|<=|=)(?P<val>[^\s]+)$")

# Strata are cuts on `y`, the WITHIN-dataset midrank percentile, so they are already normalized per
# dataset -- a "top decile" contrast is not silently dominated by whichever dataset has most rows.
STRATA: dict[str, object] = {
    "top_decile": lambda y: y >= 0.9,
    "bottom_decile": lambda y: y <= 0.1,
    "top_quartile": lambda y: y >= 0.75,
    "bottom_quartile": lambda y: y <= 0.25,
    "top_half": lambda y: y >= 0.5,
    "bottom_half": lambda y: y < 0.5,
    "all": lambda y: np.ones(len(y), dtype=bool),
}

PREDICATE_ATTRS = (
    "head_op",  # op of the terminal stage
    "first_op",  # op of the first stage
    "has_op",  # op appears anywhere
    "op_pair",  # 'a>b': a immediately followed by b
    "terminal_family",  # family declared by the terminal stage
    "has_family",  # family declared by any stage
    "cost_class",  # cost class of any stage
    "n_stages",  # supports = >= <=
)


class ContrastError(ValueError):
    """A contrast string is outside the closed grammar, so it could never be replicated."""


@dataclass(frozen=True)
class Contrast:
    text: str
    attr: str
    op: str
    value: str
    stratum_a: str
    stratum_b: str


def _vocab(engine: str) -> dict[str, frozenset]:
    g = grammar_for(engine)
    ops = frozenset(dict(g.transforms)) | frozenset(dict(g.heads))
    return {
        "ops": ops,
        "families": frozenset(FAMILY_LIST_V3),
        "cost_classes": frozenset(COST_CLASSES),
    }


def parse_contrast(text: str, engine: str = DEFAULT_ENGINE) -> Contrast:
    """Parse one contrast string, validating every token against the grammar. Raises ContrastError.

    Nothing outside this language is accepted, because a contrast that cannot be re-executed on a
    held-out half of SELECT is an assertion, not a measurement.
    """
    if not isinstance(text, str):
        raise ContrastError(f"contrast must be a string, got {type(text).__name__}")
    m = _CONTRAST_RE.match(text.strip())
    if not m:
        raise ContrastError(f"contrast must match {CONTRAST_GRAMMAR!r}, got {text!r}")
    a, b = m.group("a"), m.group("b")
    for name in (a, b):
        if name not in STRATA:
            raise ContrastError(f"unknown stratum {name!r}; expected one of {sorted(STRATA)}")
    if a == b:
        raise ContrastError(f"contrast compares stratum {a!r} against itself")

    pm = _PRED_RE.match(m.group("pred"))
    if not pm:
        raise ContrastError(f"predicate must be <attr>[=|>=|<=]<value>, got {m.group('pred')!r}")
    attr, op, val = pm.group("attr"), pm.group("op"), pm.group("val")
    if attr not in PREDICATE_ATTRS:
        raise ContrastError(f"unknown predicate attribute {attr!r}; expected {list(PREDICATE_ATTRS)}")
    if op != "=" and attr != "n_stages":
        raise ContrastError(f"attribute {attr!r} supports only '='")

    vocab = _vocab(engine)
    if attr == "n_stages":
        if not val.isdigit():
            raise ContrastError(f"n_stages value must be a non-negative integer, got {val!r}")
    elif attr in ("head_op", "first_op", "has_op"):
        if val not in vocab["ops"]:
            raise ContrastError(f"{attr}={val!r} is not an op of engine {engine!r}")
    elif attr == "op_pair":
        parts = val.split(">")
        if len(parts) != 2 or not all(p in vocab["ops"] for p in parts):
            raise ContrastError(f"op_pair must be '<op>><op>' over engine {engine!r}, got {val!r}")
    elif attr in ("terminal_family", "has_family"):
        if val not in vocab["families"]:
            raise ContrastError(f"{attr}={val!r} is not a declared family")
    elif attr == "cost_class":
        if val not in vocab["cost_classes"]:
            raise ContrastError(f"cost_class={val!r} is not one of {list(COST_CLASSES)}")

    # Cheap belt-and-braces: the grammar vocabulary is fixed, but if it ever grows a token that
    # touches the forbidden boundary we want the failure here rather than in a prompt.
    assert_no_forbidden_inputs([attr, val])
    return Contrast(text=text.strip(), attr=attr, op=op, value=val, stratum_a=a, stratum_b=b)


# --- Spec ----------------------------------------------------------------------------------------
def analysis_spec_schema() -> dict:
    """The JSON an analysis role must fill. It names no file, no ID and no parameter.

    `contrasts` is pre-committed on purpose: choosing which contrast to report *after* seeing the
    numbers is the mechanism that turns a 0.50-reliability corpus into a stream of confident noise.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": ["question", "hypothesis", "contrasts", "source"],
        "properties": {
            "question": {
                "type": "string",
                "maxLength": MAX_QUESTION_CHARS,
                "description": "What structural question about genomes are you asking?",
            },
            "hypothesis": {
                "type": "string",
                "maxLength": MAX_QUESTION_CHARS,
                "description": "What do you expect to see, and why? Stated BEFORE measurement.",
            },
            "contrasts": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_CONTRASTS,
                "items": {"type": "string", "pattern": CONTRAST_GRAMMAR},
                "description": (
                    f"Pre-committed contrasts, each of the form {CONTRAST_GRAMMAR!r}. "
                    f"Predicate attributes: {list(PREDICATE_ATTRS)}. "
                    f"Strata: {sorted(STRATA)}. "
                    "effect is always the lift ratio rate_a / rate_b."
                ),
            },
            "source": {
                "type": "string",
                "maxLength": security.MAX_PLUGIN_BYTES,
                "description": (
                    "Python defining analyze(shard) -> list of findings, each "
                    "{'claim': str, 'effect': float, 'support_n': int, 'contrast': str}. "
                    "Report measured contrasts with counts. Do not fit a model."
                ),
            },
        },
        "controller_owned": [
            "id",
            "path",
            "filename",
            "output",
            "seed",
            "split",
            "image_digest",
            "timeout_s",
            "engine",
            "shard",
        ],
    }


def validate_analysis(spec, engine: str = DEFAULT_ENGINE) -> list[str]:
    """-> list of human-readable errors; empty means the spec may be run. Never raises on bad input."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return [f"spec must be an object, got {type(spec).__name__}"]

    schema = analysis_spec_schema()
    allowed = set(schema["properties"])
    reserved = set(schema["controller_owned"])
    for key in spec:
        if key in reserved:
            errors.append(f"{key!r} is controller-owned; an agent may not set it")
        elif key not in allowed:
            errors.append(f"unknown field {key!r}")
    for key in schema["required"]:
        if key not in spec:
            errors.append(f"missing required field {key!r}")

    for key in ("question", "hypothesis"):
        val = spec.get(key)
        if key in spec:
            if not isinstance(val, str) or not val.strip():
                errors.append(f"{key!r} must be a non-empty string")
            elif len(val) > MAX_QUESTION_CHARS:
                errors.append(f"{key!r} exceeds {MAX_QUESTION_CHARS} characters")

    contrasts = spec.get("contrasts")
    if "contrasts" in spec:
        if not isinstance(contrasts, list) or not contrasts:
            errors.append("'contrasts' must be a non-empty list")
        elif len(contrasts) > MAX_CONTRASTS:
            errors.append(f"'contrasts' exceeds {MAX_CONTRASTS} entries")
        else:
            for i, text in enumerate(contrasts):
                try:
                    parse_contrast(text, engine)
                except (ContrastError, LeakageError) as exc:
                    errors.append(f"contrasts[{i}]: {exc}")

    source = spec.get("source")
    if "source" in spec:
        if not isinstance(source, str) or not source.strip():
            errors.append("'source' must be a non-empty string")
        else:
            try:
                security.static_check_plugin(source, entrypoint="analyze")
            except security.SandboxRejection as exc:
                errors.append(f"source: {exc}")
    return errors


# --- Shard ---------------------------------------------------------------------------------------
def materialize_select_shard(rows, out_dir) -> Path:
    """Write ONLY select-split rows to <out_dir>/select/. Raises LeakageError on any other row.

    The directory name is load-bearing: `security.sandbox_command(mode='analysis')` refuses to mount
    anything not named 'select'.
    """
    out_dir = Path(out_dir)
    dataset = np.asarray(rows.dataset)
    declared = np.asarray(rows.split)
    y = np.asarray(rows.y, dtype=np.float64)
    genome = np.asarray(rows.genome)
    engine = getattr(rows, "engine", DEFAULT_ENGINE)

    mask = declared == "select"
    if not mask.any():
        raise LeakageError("no select-split rows to materialize")

    ds_sel = dataset[mask]
    # Do not trust rows.split: re-derive from the dataset key. A RowTable whose split column was
    # rebuilt or reordered would otherwise silently smuggle dev/test rows into the one container
    # that sees labels.
    recomputed = np.asarray([split_of(str(d)) for d in ds_sel])
    bad = recomputed != "select"
    if bad.any():
        raise LeakageError(
            f"{int(bad.sum())} row(s) marked 'select' do not hash to the select split; "
            f"first offender {str(ds_sel[bad][0])!r} -> {recomputed[bad][0]!r}"
        )
    if (declared[~mask] == "select").any():  # pragma: no cover - mask is definitionally exhaustive
        raise LeakageError("select rows outside the select mask")

    # Opaque contiguous group index; raw dataset hashes are a forbidden input and never leave.
    keys = sorted(set(str(d) for d in ds_sel))
    index = {k: i for i, k in enumerate(keys)}
    group = np.asarray([index[str(d)] for d in ds_sel], dtype=np.int32)
    y_sel = y[mask]
    genome_sel = genome[mask]

    target = out_dir / "select"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".select.tmp-", dir=out_dir))
    try:
        lines = "".join(
            json.dumps({"row": i, "group": int(group[i]), "genome": str(genome_sel[i])}, sort_keys=True)
            + "\n"
            for i in range(len(genome_sel))
        ).encode()
        (tmp / "genomes.jsonl").write_bytes(lines)
        with open(tmp / "labels.npz", "wb") as fh:
            np.savez(fh, y=y_sel, group=group)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "split": "select",
            "engine": engine,
            "n_rows": int(len(y_sel)),
            "n_groups": len(keys),
            "genomes_sha256": hashlib.sha256(lines).hexdigest(),
            "columns": ["y", "group"],
        }
        (tmp / "manifest.json").write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return target


# --- Findings contract ---------------------------------------------------------------------------
def _model_result_tokens(text: str, allowed: frozenset) -> list[str]:
    """Blocklist tokens in `text` that are NOT part of the described contrast vocabulary.

    Tokenizing on underscores too is deliberate: it catches `oof_r2` while leaving the real op
    `histgbm` intact.
    """
    tokens = set(re.split(r"[^a-z0-9]+", str(text).casefold())) - {""}
    return sorted((tokens & MODEL_RESULT_TOKENS) - allowed)


def check_finding(finding, engine: str = DEFAULT_ENGINE) -> list[str]:
    """-> list of reasons this finding is not admissible; empty means it is."""
    errors: list[str] = []
    if not isinstance(finding, dict):
        return [f"finding must be an object, got {type(finding).__name__}"]

    extra = sorted(set(finding) - set(FINDING_KEYS))
    if extra:
        errors.append(f"unexpected keys {extra}; exactly {list(FINDING_KEYS)} are allowed")
    for key in FINDING_KEYS:
        if key not in finding:
            errors.append(f"missing key {key!r}")

    # Key NAMES that advertise a fitted-model result are rejected outright.
    for key in finding:
        hits = _model_result_tokens(key, frozenset())
        if hits:
            errors.append(f"key {key!r} reports a fitted-model result ({', '.join(hits)})")

    contrast = None
    if "contrast" in finding:
        try:
            contrast = parse_contrast(finding["contrast"], engine)
        except (ContrastError, LeakageError) as exc:
            errors.append(f"contrast: {exc}")

    claim = finding.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        errors.append("'claim' must be a non-empty string")
    else:
        vocab = _vocab(engine)
        allowed = vocab["ops"] | vocab["families"] | vocab["cost_classes"]
        if contrast is not None:
            allowed = allowed | set(re.split(r"[^a-z0-9]+", contrast.text.casefold()))
        hits = _model_result_tokens(claim, frozenset(allowed))
        if hits:
            errors.append(
                f"'claim' reports a fitted-model result ({', '.join(hits)}); report a measured "
                "contrast with counts, not a fitted score"
            )

    effect = finding.get("effect")
    if isinstance(effect, bool) or not isinstance(effect, (int, float)):
        errors.append("'effect' must be a number (the lift ratio rate_a / rate_b)")
    elif not math.isfinite(float(effect)) or float(effect) <= 0.0:
        errors.append("'effect' must be a finite positive lift ratio")

    support = finding.get("support_n")
    if isinstance(support, bool) or not isinstance(support, int):
        errors.append("'support_n' must be an integer count")
    elif support < MIN_SUPPORT_N:
        errors.append(f"'support_n' below the floor of {MIN_SUPPORT_N}")
    return errors


def validate_findings(payload, engine: str = DEFAULT_ENGINE) -> dict:
    """Apply the findings contract to a raw payload (bytes, str, or parsed object).

    Returns {'accepted': [...], 'rejected': [{'finding':…, 'errors':[…]}], 'errors': [...]}.
    """
    out: dict = {"accepted": [], "rejected": [], "errors": []}
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
        if len(raw) > FINDINGS_BYTE_CAP:
            out["errors"].append(f"findings payload exceeds {FINDINGS_BYTE_CAP} bytes")
            return out
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            out["errors"].append(f"findings payload is not valid JSON: {exc}")
            return out
    elif isinstance(payload, str):
        return validate_findings(payload.encode("utf-8", "replace"), engine)

    findings = payload.get("findings") if isinstance(payload, dict) else payload
    if not isinstance(findings, list):
        out["errors"].append("findings must be a list, or an object with a 'findings' list")
        return out
    if len(findings) > MAX_FINDINGS:
        out["errors"].append(f"more than {MAX_FINDINGS} findings")
        findings = findings[:MAX_FINDINGS]
    for finding in findings:
        errors = check_finding(finding, engine)
        if errors:
            out["rejected"].append({"finding": finding, "errors": errors})
        else:
            out["accepted"].append(finding)
    return out


# --- Sandboxed execution -------------------------------------------------------------------------
def _run_command(cmd: list[str], timeout_s: int):
    """Seam for tests; the real path is a plain subprocess call on the docker invocation."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)


def run_analysis(
    spec_source, shard, image_digest, timeout_s: int = 120, harness=None, engine: str = DEFAULT_ENGINE
) -> dict:
    """Static-check `analyze`, run it against the SELECT shard, and police the findings it returns.

    Every path, the mode, and the timeout are minted here. The agent supplies source and nothing
    else.
    """
    shard = Path(shard)
    result: dict = {
        "ok": False,
        "returncode": None,
        "accepted": [],
        "rejected": [],
        "errors": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    # Checked before anything is written to disk; sandbox_command enforces it again at mount time.
    if shard.name != "select":
        raise security.SandboxRejection(f"analysis may only read the SELECT shard, got {shard.name!r}")
    if not shard.is_dir():
        raise FileNotFoundError(f"select shard not found: {shard}")

    security.static_check_plugin(spec_source, entrypoint="analyze")

    harness = Path(harness) if harness is not None else ANALYSIS_HARNESS
    if not harness.is_file():
        raise FileNotFoundError(f"analysis harness not found: {harness}")

    work = Path(tempfile.mkdtemp(prefix="zyal-analysis-"))
    try:
        plugin = work / "plugin.py"
        plugin.write_text(spec_source, encoding="utf-8")
        output = work / "output"
        output.mkdir()
        cmd = security.sandbox_command(
            plugin=plugin,
            harness=harness,
            data_shard=shard,
            output=output,
            image_digest=image_digest,
            mode="analysis",
        )
        try:
            proc = _run_command(cmd, timeout_s)
        except subprocess.TimeoutExpired:
            result["errors"].append(f"analysis exceeded {timeout_s}s")
            return result
        result["returncode"] = proc.returncode
        result["stdout_tail"] = (proc.stdout or "")[-2000:]
        result["stderr_tail"] = (proc.stderr or "")[-2000:]
        if proc.returncode != 0:
            result["errors"].append(f"analysis exited {proc.returncode}")
            return result

        findings_path = output / "findings.json"
        if not findings_path.is_file():
            result["errors"].append("analysis produced no findings.json")
            return result
        raw = findings_path.read_bytes()
        checked = validate_findings(raw, engine)
        result["accepted"] = checked["accepted"]
        result["rejected"] = checked["rejected"]
        result["errors"].extend(checked["errors"])
        result["ok"] = bool(result["accepted"]) and not result["errors"]
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- Measurement and replication -----------------------------------------------------------------
def _predicate_hits(contrast: Contrast, genomes: np.ndarray, engine: str) -> np.ndarray:
    """Boolean mask of rows whose genome satisfies the predicate. Unparseable genomes are False."""
    attr, value = contrast.attr, contrast.value
    cache: dict[str, bool] = {}

    def hit(text: str) -> bool:
        pg = parse(text, engine)
        if not pg.ok or not pg.stages:
            return False  # '(sym ...)' genomes are not pipelines; never silently counted
        if attr == "head_op":
            return pg.stages[-1].op == value
        if attr == "first_op":
            return pg.stages[0].op == value
        if attr == "has_op":
            return value in pg.ops()
        if attr == "op_pair":
            a, b = value.split(">")
            ops = pg.ops()
            return any(ops[i] == a and ops[i + 1] == b for i in range(len(ops) - 1))
        if attr == "terminal_family":
            return value in (pg.stages[-1].families or ())
        if attr == "has_family":
            return any(value in (s.families or ()) for s in pg.stages)
        if attr == "cost_class":
            return any(s.cost_class == value for s in pg.stages)
        if attr == "n_stages":
            n, target = len(pg.stages), int(value)
            return (
                n >= target if contrast.op == ">=" else (n <= target if contrast.op == "<=" else n == target)
            )
        return False  # pragma: no cover - parse_contrast rejects unknown attributes

    out = np.zeros(len(genomes), dtype=bool)
    for i, text in enumerate(genomes):
        key = str(text)
        val = cache.get(key)
        if val is None:
            val = hit(key)
            cache[key] = val
        out[i] = val
    return out


def measure_contrast(contrast, rows, mask=None, engine: str | None = None) -> dict:
    """Measure one contrast over `rows[mask]`. effect = smoothed rate_a / rate_b.

    Haldane-Anscombe smoothing (+0.5 / +1) keeps the ratio finite when a stratum has zero hits, so a
    single empty cell degrades the effect instead of producing an infinity that survives every
    downstream comparison.
    """
    engine = engine or getattr(rows, "engine", DEFAULT_ENGINE)
    if isinstance(contrast, str):
        contrast = parse_contrast(contrast, engine)
    y = np.asarray(rows.y, dtype=np.float64)
    genomes = np.asarray(rows.genome)
    sel = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    hits = _predicate_hits(contrast, genomes, engine)
    a = sel & STRATA[contrast.stratum_a](y)
    b = sel & STRATA[contrast.stratum_b](y)
    k_a, n_a = int((hits & a).sum()), int(a.sum())
    k_b, n_b = int((hits & b).sum()), int(b.sum())
    rate_a = (k_a + 0.5) / (n_a + 1.0)
    rate_b = (k_b + 0.5) / (n_b + 1.0)
    return {
        "contrast": contrast.text,
        "effect": float(rate_a / rate_b),
        "rate_a": float(rate_a),
        "rate_b": float(rate_b),
        "k_a": k_a,
        "n_a": n_a,
        "k_b": k_b,
        "n_b": n_b,
        "support_n": k_a + k_b,
        "n_rows": int(sel.sum()),
    }


def _select_mask(rows) -> np.ndarray:
    declared = np.asarray(rows.split)
    mask = declared == "select"
    if not mask.any():
        raise ValueError("rows contain no select-split data")
    ds = np.asarray(rows.dataset)[mask]
    bad = np.asarray([split_of(str(d)) for d in ds]) != "select"
    if bad.any():
        raise LeakageError(f"{int(bad.sum())} row(s) labelled 'select' hash to another split")
    return mask


def replicate(finding, rows, seed: int) -> dict:
    """Re-measure a finding's contrast on a DISJOINT HALF of SELECT.

    Not ceremony. Op-pair lift mined on this corpus has split-half reliability r=0.50 with 68% sign
    agreement, and it does not improve with more support -- so a data-mined pattern that has not
    survived an out-of-half re-measurement is roughly a coin flip.

    The halves are split by DATASET, never by row: rows from one dataset share its idiosyncrasies,
    so a row-wise split would let the "held-out" half re-measure the same datasets and replicate
    almost anything. Effects are ratios, so "same sign" means the same side of 1.0 and "half the
    magnitude" is measured in log space.
    """
    errors = check_finding(finding, getattr(rows, "engine", DEFAULT_ENGINE))
    if errors:
        return {"replicated": False, "reason": "finding failed the findings contract", "errors": errors}

    engine = getattr(rows, "engine", DEFAULT_ENGINE)
    contrast = parse_contrast(finding["contrast"], engine)
    sel = _select_mask(rows)

    dataset = np.asarray(rows.dataset)
    keys = np.asarray(sorted(set(str(d) for d in dataset[sel])))
    if len(keys) < 2:
        return {"replicated": False, "reason": "fewer than 2 select datasets; cannot split halves"}
    order = np.random.RandomState(int(seed) % (2**32)).permutation(len(keys))
    half_a_keys = set(keys[order[: len(keys) // 2]])
    in_a = np.asarray([str(d) in half_a_keys for d in dataset])
    mask_a, mask_b = sel & in_a, sel & ~in_a

    full = measure_contrast(contrast, rows, sel, engine)
    obs_a = measure_contrast(contrast, rows, mask_a, engine)
    obs_b = measure_contrast(contrast, rows, mask_b, engine)

    claimed = float(finding["effect"])
    log_claim, log_rep = math.log(claimed), math.log(obs_b["effect"])
    sign_agrees = bool(log_claim * log_rep > 0) or (log_claim == 0.0 and log_rep == 0.0)
    magnitude_ratio = (abs(log_rep) / abs(log_claim)) if log_claim != 0.0 else float("inf")
    holds = bool(sign_agrees and magnitude_ratio >= REPLICATION_MIN_RATIO)
    underpowered = obs_b["support_n"] < MIN_SUPPORT_N

    return {
        "contrast": contrast.text,
        "seed": int(seed),
        "effect_claimed": claimed,
        "effect_select_full": full["effect"],
        "effect_half_a": obs_a["effect"],
        "effect_half_b": obs_b["effect"],
        "sign_agrees": sign_agrees,
        "magnitude_ratio": float(magnitude_ratio),
        "support_claimed": int(finding["support_n"]),
        "support_select_full": full["support_n"],
        "support_half_b": obs_b["support_n"],
        "n_datasets_half_a": len(half_a_keys),
        "n_datasets_half_b": len(keys) - len(half_a_keys),
        # A claim whose own contrast does not reproduce on all of SELECT was never measured as
        # stated; that is a different failure from failing to replicate, so it is reported apart.
        "claim_matches_full": bool(abs(math.log(full["effect"]) - log_claim) <= math.log(1.25)),
        "underpowered": bool(underpowered),
        "replicated": bool(holds and not underpowered),
        "reason": (
            "support below floor on the replication half"
            if underpowered
            else "holds"
            if holds
            else "sign flipped"
            if not sign_agrees
            else "magnitude below half"
        ),
    }
