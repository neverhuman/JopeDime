"""Quote-aware genome parsing and structural features.

The existing regex stage parser (full_cv_meta_selector_v2.parse_pipeline_stages)
drops any stage whose params contain parentheses inside quoted exprs — 13.4% of
labels. This tokenizer walks characters with paren-depth and quote state, so
`(map expr="sin(col(6))")` parses as one stage. Params may be `key=value` with
bare or double-quoted values; bare parenthesized exprs like `expr=sigmoid_col0`
also occur and parse fine.

Stdlib + no third-party imports: shared by the study and the deployable
reference implementation.
"""

from __future__ import annotations

import hashlib

HEAD_OPS = (
    "ridge",
    "elasticnet",
    "logistic",
    "histgbm",
    "histgbm2",
    "gbm",
    "gpr",
    "ksmooth",
    "knn",
    "huber_ridge",
    "portfolio",
    "ridge_gcv",
)

# op -> family; anything absent maps to "other". Deterministic, hand-frozen.
FAMILY = {}
for _op in (
    "standardize",
    "robust_scale",
    "norm_features",
    "winsor_clip",
    "quantile",
    "rank",
    "blom",
    "sigmoid",
    "tanh_scale",
):
    FAMILY[_op] = "scale"
for _op in (
    "rfft_mag",
    "fft_mag",
    "polar",
    "rff_cos",
    "rff_orthogonal",
    "rff_halton",
    "fastfood_rbf",
    "spectral",
    "wavelet",
):
    FAMILY[_op] = "fourier"
for _op in ("cheb", "poly", "legendre", "hermite", "spline", "bspline"):
    FAMILY[_op] = "poly"
for _op in (
    "rand_proj",
    "svd_project",
    "pca",
    "nystrom",
    "nystrom_leverage",
    "count_sketch",
    "tensor_sketch",
):
    FAMILY[_op] = "proj"
for _op in ("pair_prod", "cross_terms", "all_pairs", "top_pairs"):
    FAMILY[_op] = "interact"
for _op in ("row_stat", "reduce", "row_norm", "aggregate"):
    FAMILY[_op] = "rowstat"
for _op in ("map", "derive"):
    FAMILY[_op] = "expr"
for _op in (
    "lag",
    "roll",
    "delay_embedding",
    "iterate",
    "phase_interference",
    "quasicrystal",
    "golden_scale",
    "logistic_map",
    "poincare",
):
    FAMILY[_op] = "dynamical"
FAMILIES = ("scale", "fourier", "poly", "proj", "interact", "rowstat", "expr", "dynamical", "other")

WIDTH_KEYS = (
    "n",
    "k",
    "degree",
    "bins",
    "dims",
    "components",
    "rank",
    "n_features",
    "order",
    "knots",
    "topk",
    "n_paths",
    "m",
)


def split_stages_quoted(genome: str) -> list[str] | None:
    """Split a `(pipe X (stage ..) .. (head ..))` genome into stage strings.

    Returns None for non-pipeline genomes ((sym ...), unparseable text).
    Parens inside double-quoted param values do not affect depth.
    """
    s = genome.strip()
    if not s.startswith("("):
        return None
    depth = 0
    in_quote = False
    escape = False
    stages: list[str] = []
    start = -1
    header_done = False
    for i, ch in enumerate(s):
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == "(":
            depth += 1
            if depth == 2:
                start = i
        elif ch == ")":
            if depth == 2 and start >= 0:
                stages.append(s[start : i + 1])
                start = -1
            depth -= 1
            if depth == 0:
                header_done = True
        if depth < 0:
            return None
    if in_quote or depth != 0 or not header_done:
        return None
    head = s[1 : s.find(" ")] if " " in s else s[1:-1]
    if head != "pipe":
        return None
    return stages


def parse_stage(stage: str) -> tuple[str, dict[str, str]]:
    """`(op k1=v1 k2="v 2" flag)` -> (op, {k1: v1, k2: 'v 2', flag: 'true'})."""
    body = stage.strip()
    if body.startswith("("):
        body = body[1:]
    if body.endswith(")"):
        body = body[:-1]
    tokens: list[str] = []
    cur: list[str] = []
    in_quote = False
    escape = False
    for ch in body:
        if in_quote:
            cur.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            cur.append(ch)
        elif ch.isspace():
            if cur:
                tokens.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    if not tokens:
        return "", {}
    op = tokens[0]
    params: dict[str, str] = {}
    for tok in tokens[1:]:
        if "=" in tok:
            key, val = tok.split("=", 1)
            if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
                val = val[1:-1]
            params[key] = val
        else:
            params[tok] = "true"
    return op, params


def _float_or_none(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def structural_features(genome: str) -> dict[str, float] | None:
    """Candidate-varying structural features for the ridge. None if unparseable."""
    stages = split_stages_quoted(genome)
    if stages is None or not stages:
        return None
    parsed = [parse_stage(st) for st in stages]
    ops = [op for op, _ in parsed]
    head = ops[-1] if ops[-1] in HEAD_OPS else "other"
    feats: dict[str, float] = {
        "n_stages": float(len(ops)),
        "n_transforms": float(len(ops) - 1),
        "expr_char_len": float(sum(len(v) for _, p in parsed for k, v in p.items() if k == "expr")),
        "n_exprs": float(sum(1 for _, p in parsed if "expr" in p)),
        "n_append": float(sum(1 for _, p in parsed if p.get("append") == "true")),
        "n_seeded": float(sum(1 for _, p in parsed if "seed" in p)),
    }
    width = 0.0
    for _, p in parsed:
        for wk in WIDTH_KEYS:
            v = _float_or_none(p.get(wk, ""))
            if v is not None and v > 0:
                width += v
    feats["log1p_width"] = float(__import__("math").log1p(width))
    for fam in FAMILIES:
        feats[f"fam_{fam}"] = 0.0
    for op in ops[:-1]:
        feats[f"fam_{FAMILY.get(op, 'other')}"] += 1.0
    for h in ("ridge", "elasticnet", "logistic", "histgbm", "histgbm2", "gpr", "ksmooth", "other"):
        feats[f"head_{h}"] = 1.0 if head == h else 0.0
    return feats


def op_skeleton(genome: str) -> str | None:
    """Seed-stripped op sequence key (novelty_hash-style identity)."""
    stages = split_stages_quoted(genome)
    if stages is None:
        return None
    return ">".join(parse_stage(st)[0] for st in stages)


def hashed_ngrams(genome: str, buckets: int = 256) -> dict[str, float] | None:
    """blake2b-hashed op n-grams (n=1..3). Ceiling-study features only."""
    stages = split_stages_quoted(genome)
    if stages is None:
        return None
    ops = [parse_stage(st)[0] for st in stages]
    out: dict[str, float] = {}
    for n in (1, 2, 3):
        for i in range(len(ops) - n + 1):
            gram = ">".join(ops[i : i + n])
            b = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "big") % buckets
            key = f"ng{n}_{b}"
            out[key] = out.get(key, 0.0) + 1.0
    return out
