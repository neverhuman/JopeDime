"""G1 (search trajectory / saturation) + G6 (pipeline topology) features.

Both groups are *within-dataset varying* by construction: every column is a function of the genome's
stage list alone, so it separates genomes evaluated on the same dataset. Nothing here touches the
dataset, the label, or any corpus statistic.

Why these two groups share a file: they are the same underlying object (the ordered stage list) read
two ways -- G1 reads its *length* against the grammar's transform cap, G6 reads its *shape*.

G1 exists because pipeline length is the largest single untapped signal in the measured corpus:
74.8% of featmath genomes sit at the 7-stage cap and that band is the worst-performing one
(mean percentile 0.510 vs 0.605 at 3 stages). The cap band is also compositionally different -- it
is dominated by elasticnet/ridge heads while short pipelines are gpr/ensemble -- so
`g1_cap_x_head_<family>` carries the interaction that a length feature and a head feature cannot
express additively.

G6 exists because *where* an op sits flips its sign: fft_mag first is -0.060, blom first is +0.047,
rff_cos first is +0.037, rank later is +0.021. A bag-of-ops encoding averages those away.

Defaults: a genome with `pg.ok is False` ((sym ...) and portfolio genomes) yields every column at its
neutral value -- 0.0 everywhere except the `g6_first_pos_`/`g6_last_pos_` block, whose documented
"absent" sentinel is -1.0. `extract` never raises.
"""

from __future__ import annotations

import functools
import hashlib
import math

from jope.grammars import FAMILY_LIST_V3, grammar_for

from .parse import COST_CLASSES, ParsedGenome, family_vector, max_transforms

#: Number of hashed family-transition-bigram buckets.
FAMTRANS_BUCKETS = 128

#: Position sentinel emitted for an op that does not occur in the genome.
ABSENT_POS = -1.0

_COST_RANK = {c: i for i, c in enumerate(COST_CLASSES)}
_EXPENSIVE = frozenset(("expensive", "extreme"))
_KERNELISH = frozenset(("kernel", "random_features"))


@functools.lru_cache(maxsize=4)
def _engine_ops(engine: str) -> tuple[str, ...]:
    """Every op name in the engine grammar (transforms + heads), sorted, deduped."""
    g = grammar_for(engine)
    return tuple(sorted(set(dict(g.transforms)) | set(dict(g.heads))))


@functools.lru_cache(maxsize=4)
def _names(engine: str) -> tuple[str, ...]:
    ops = _engine_ops(engine)
    out: list[str] = [
        "g1_stage_headroom",
        "g1_at_length_cap",
        "g1_stages_below_cap",
        "g1_stages_over_cap",
        "g1_exceeds_cap",
        "g1_head_only",
        "g1_short_pipeline_band",
    ]
    out += [f"g1_cap_x_head_{fam}" for fam in FAMILY_LIST_V3]
    out += [
        "g1_expensive_slot_frac",
        "g1_redundant_scaler_count",
        "g1_terminal_transform_is_scaler",
    ]
    out += [f"g6_first_pos_{op}" for op in ops]
    out += [f"g6_last_pos_{op}" for op in ops]
    out += [f"g6_famtrans_{i}" for i in range(FAMTRANS_BUCKETS)]
    out += [
        "g6_max_consecutive_same_family",
        "g6_family_entropy",
        "g6_monotone_complexity",
        "g6_scaler_before_kernel",
        "g6_append_chain_depth",
        "g6_n_distinct_families",
        "g6_n_distinct_ops",
        "g6_head_arity_class",
    ]
    return tuple(out)


@functools.lru_cache(maxsize=4)
def _defaults(engine: str) -> dict[str, float]:
    """Neutral row: 0.0 everywhere, except position columns which use the -1.0 absent sentinel."""
    return {
        n: (ABSENT_POS if n.startswith(("g6_first_pos_", "g6_last_pos_")) else 0.0) for n in _names(engine)
    }


@functools.lru_cache(maxsize=4096)
def _famtrans_bucket(bigram: str) -> int:
    digest = hashlib.blake2b(bigram.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % FAMTRANS_BUCKETS


def feature_names(engine: str = "featmath") -> list[str]:
    """Fixed, deterministic column list for `engine`. Depends only on the grammar, never on data."""
    return list(_names(engine))


def _short_pipeline_band(n_stages: int) -> float:
    """Ordinal 0..5 over the measured length bands {1, 2, 3, 4-5, 6, 7+}."""
    if n_stages <= 1:
        return 0.0
    if n_stages == 2:
        return 1.0
    if n_stages == 3:
        return 2.0
    if n_stages <= 5:
        return 3.0
    if n_stages == 6:
        return 4.0
    return 5.0


def _head_arity_class(op: str) -> float:
    """0 single / 1 ensemble / 2 stacked / 3 portfolio. Checked most-specific first."""
    if op.startswith("portfolio"):
        return 3.0
    if op.startswith("stack"):
        return 2.0
    if op.startswith("ensemble"):
        return 1.0
    return 0.0


def extract(pg: ParsedGenome, engine: str = "featmath") -> dict[str, float]:
    """One genome -> {column: float}. Pure, deterministic, finite, never raises."""
    feats = dict(_defaults(engine))
    stages = getattr(pg, "stages", None) or []
    if not getattr(pg, "ok", False) or not stages:
        return feats

    n_stages = len(stages)
    cap = max_transforms(engine)
    n_transforms = n_stages - 1  # a pipeline is transforms + one head

    # ---- G1: length against the grammar cap -------------------------------------------------
    # `>=`, not `==`: the corpus contains genomes that EXCEED the declared cap. newmath declares
    # max_transforms=4 yet 15.1% of its k=5 rows carry more than 5 stages (max observed: 10), while
    # newmath/legacy respects the cap perfectly (0 / 23,467). An equality test silently reported
    # "not at cap" for the longest quarter of the newmath corpus and zeroed the entire
    # g1_cap_x_head_* block for exactly the rows it exists to isolate.
    at_cap = 1.0 if n_transforms >= cap else 0.0
    feats["g1_stage_headroom"] = (n_transforms / cap) if cap else 0.0
    feats["g1_at_length_cap"] = at_cap
    # Over-cap genomes are real, so expose the overshoot rather than letting the "headroom" and
    # "below cap" columns silently invert sign (they reached 2.25 and -5.0 respectively).
    feats["g1_stages_below_cap"] = float(max(cap - n_transforms, 0))
    feats["g1_stages_over_cap"] = float(max(n_transforms - cap, 0))
    feats["g1_exceeds_cap"] = 1.0 if n_transforms > cap else 0.0
    feats["g1_head_only"] = 1.0 if n_transforms == 0 else 0.0
    feats["g1_short_pipeline_band"] = _short_pipeline_band(n_stages)

    head = stages[-1]
    if at_cap:
        # Interaction, not a second main effect: the capped band is elasticnet/ridge-dominated
        # while short pipelines are gpr/ensemble, and the head's value inverts between them.
        for fam in head.families:
            key = f"g1_cap_x_head_{fam}"
            if key in feats:
                feats[key] = 1.0

    n_expensive = 0
    n_scaling = 0
    for s in stages:
        if s.cost_class in _EXPENSIVE:
            n_expensive += 1
        if "scaling" in s.families:
            n_scaling += 1
    feats["g1_expensive_slot_frac"] = n_expensive / n_stages
    feats["g1_redundant_scaler_count"] = float(max(0, n_scaling - 1))
    if n_stages >= 2 and "scaling" in stages[-2].families:
        feats["g1_terminal_transform_is_scaler"] = 1.0

    # ---- G6: shape --------------------------------------------------------------------------
    # first/last normalized position per op. Both are emitted because an op appearing twice
    # (common for scalers) carries different information at its earliest vs latest slot.
    for s in stages:
        pos = s.position_norm
        first_key = f"g6_first_pos_{s.op}"
        if first_key in feats:
            if feats[first_key] == ABSENT_POS:
                feats[first_key] = pos
            feats[f"g6_last_pos_{s.op}"] = pos

    # Dominant family per stage = first declared family; "none" for ops the grammar does not know.
    doms = [s.families[0] if s.families else "none" for s in stages]

    for i in range(n_stages - 1):
        key = f"g6_famtrans_{_famtrans_bucket(doms[i] + '>' + doms[i + 1])}"
        feats[key] += 1.0

    # Longest run of the same dominant family. "none" stages break runs and never form one.
    best_run = 0
    run = 0
    prev = None
    for d in doms:
        if d == "none":
            run = 0
            prev = None
            continue
        run = run + 1 if d == prev else 1
        prev = d
        if run > best_run:
            best_run = run
    feats["g6_max_consecutive_same_family"] = float(best_run)

    fam_counts = [c for c in family_vector(pg).values() if c > 0]
    total = float(sum(fam_counts))
    if total > 0.0:
        ent = 0.0
        for c in fam_counts:
            p = c / total
            ent -= p * math.log2(p)
        feats["g6_family_entropy"] = ent
    feats["g6_n_distinct_families"] = float(len(fam_counts))

    ranks = [_COST_RANK.get(s.cost_class, 1) for s in stages]
    feats["g6_monotone_complexity"] = (
        1.0 if all(ranks[i] <= ranks[i + 1] for i in range(n_stages - 1)) else 0.0
    )

    seen_scaler = False
    for s in stages:
        if seen_scaler and _KERNELISH.intersection(s.families):
            feats["g6_scaler_before_kernel"] = 1.0
            break
        if "scaling" in s.families:
            seen_scaler = True

    best_append = 0
    run = 0
    for s in stages:
        if s.params.get("append") == "true":
            run += 1
            if run > best_append:
                best_append = run
        else:
            run = 0
    feats["g6_append_chain_depth"] = float(best_append)

    feats["g6_n_distinct_ops"] = float(len({s.op for s in stages}))
    feats["g6_head_arity_class"] = _head_arity_class(head.op)
    return feats
