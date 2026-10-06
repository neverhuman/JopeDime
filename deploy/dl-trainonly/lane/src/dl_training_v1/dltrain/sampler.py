"""PLAN §5 P7 sampler: the train-only corpus view, fitted sampler statistics, the counter-based family-uniform macro
sampler with the §6.1 groups (and the lineage-B group of the baseline arm, §6.3.2), consumption-journal records, and
collation into a Batch.

Rules (PLAN §6.1, §6.2, §6.6):
- The fit universe is the train_fit contexts (split_role 'train_fit') of train_fit and train_nonpool families that
  hold at least one labelled row. Dev, diag, heldout and sealed units never enter it and are never drawn, and every
  fitted statistic reads only that universe, so perturbing any other unit leaves the fit byte-identical (P7 test 7).
- Every draw is a pure function of sha256(DOMAIN ‖ run_seed ‖ step ‖ slot ‖ purpose ‖ k), rejection-sampled so it is
  exactly uniform. The sampler cursor therefore fixes every future draw, and no global RNG is consumed.
- A family is uniform over the universe families, a dataset uniform within it and a context uniform within that
  dataset, so pi_c = 1/(F·n_d·n_c). Every journal slot records pi_c, the class counts of every sampled member and
  the class counts of the members that enter L_feas (local backend), which are the pi_group numerators (§6.3.2).
- TrainData.content_sha256 binds everything a training step reads from the universe (encodings, tokens, labels, and
  the grammar masks unless a grammar ref is stamped), so a resume against other data is refused (§6.6 identities).
"""
from __future__ import annotations

import copy
import functools
import hashlib
import math
import operator
from collections import Counter
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from dltrain import common
from dltrain import labels as lbl

TRAIN_ASSIGNMENTS = frozenset({"train_fit", "train_nonpool"})
DOMAIN = b"jope.dl-sampler/v1\0"
SUCCESSOR_V1 = "successor-v1"
BASELINE_V1 = "baseline-v1"
GROUP_MODES = (SUCCESSOR_V1, BASELINE_V1)
FIT_SCHEMA = "jope.dl-sampler-fit/v1"
JOURNAL_SCHEMA = "jope.dl-consumption/v1"
STATE_SCHEMA = "jope.dl-sampler-state/v1"
TRAIN_DATA_SCHEMA = "jope.dl-train-data/v1"
# Input identities a loader stamps on TrainData; each must equal the run plan's ref of the same name (§6.6).
DATA_REF_KEYS = ("corpus", "encoding", "normalizers", "tokenizer", "grammar")
FIT_SPLIT_ROLE = "train_fit"
TRAIN_FAMILY_SPLIT_ROLES = ("train_fit", "diag")      # §6.2: a train family's contexts are train_fit or diag only
RUN_SEED_BYTES = 32
PAD_ID = 0
PLAN_REASONS = ("DEGENERATE_BASELINE", "UNKNOWN_BASELINE", "FEW_SKELETONS", "TOO_FEW_LABELLED")
ROLE_CODES = {"teacher": 0, "champion": 0, "comparator": 1, "failure": 2, "feasibility": 3}
Q_QUANTILES = (0, 25, 50, 75, 100)
EXCLUDED_KEYS = ("diag_contexts", "unlabelled_contexts", "empty_datasets", "empty_families")


class SamplerError(RuntimeError):
    """A corpus, fit, draw or sampler-state invariant failed (PLAN §6.1, §6.2, §6.6)."""


# --------------------------------------------------------------------------------------------------- configuration

@dataclass(frozen=True)
class SamplerConfig:
    """PLAN §6.1: B_ctx contexts per step; a successor group of G = 8 is 4 teachers, 2 comparators and up to 2
    failure or timeout members (comparators otherwise); the feasibility stream draws up to 8 labelled members at a
    declared weight of 0.1 of a context. baseline-v1 draws the lineage-B group: the champion plus one member from
    each of group_size - 1 equal-mass quantile bands (§6.3.2)."""

    group_mode: str = SUCCESSOR_V1
    b_ctx: int = 32
    group_size: int = 8
    n_teachers: int = 4
    n_comparators: int = 2
    max_failures: int = 2
    feas_stream_max: int = 8
    feas_stream_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.group_mode not in GROUP_MODES:
            raise ValueError(f"group_mode must be one of {GROUP_MODES}, got {self.group_mode!r}")
        for name, low in (("b_ctx", 1), ("group_size", 2), ("n_teachers", 1), ("n_comparators", 0),
                          ("max_failures", 0), ("feas_stream_max", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an int >= {low}, got {value!r}")
        w = self.feas_stream_weight
        if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or not 0.0 < w <= 1.0:
            raise ValueError(f"feas_stream_weight must be in (0, 1], got {w!r}")
        object.__setattr__(self, "feas_stream_weight", float(w))
        if self.group_mode == SUCCESSOR_V1 and self.group_size != self.n_teachers + self.n_comparators + self.max_failures:
            raise ValueError("successor-v1 needs group_size == n_teachers + n_comparators + max_failures")

    def to_json(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> SamplerConfig:
        unknown = sorted(set(d) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown SamplerConfig keys {unknown}")
        return cls(**dict(d))

    def sha256(self) -> str:
        return common.sha256_bytes(common.canonical(self.to_json()))


# ---------------------------------------------------------------------------------------------------------- corpus

@dataclass(frozen=True, eq=False)
class Corpus:
    """Train-only corpus view (PLAN §5 P5 record allowlist, §6.3 labels): context rows by sha, training records per
    context sorted by program_sha256, their labels, float32 context encodings and the family holdout assignment.
    Treat every field as read-only; build() takes deep copies and freezes the encodings."""

    contexts: Mapping[str, dict]
    records: Mapping[str, tuple[dict, ...]]
    labels: lbl.LabelTable
    encodings: Mapping[str, np.ndarray]
    assignment: Mapping[str, str]

    def __post_init__(self) -> None:
        tokens: dict[tuple[str, str], tuple[int, ...]] = {}
        for ctx, rows in self.records.items():
            for row in rows:
                key = (ctx, row["program_sha256"])
                if key in tokens:
                    raise SamplerError(f"duplicate program {key[1]} in context {ctx}")
                tokens[key] = tuple(int(t) for t in row["tokens"])
        by_key = {(ctx, lab.program_sha256): lab for ctx, labs in self.labels.records.items() for lab in labs}
        widths = sorted({int(np.shape(z)[0]) for z in self.encodings.values()})
        if len(widths) > 1:
            raise SamplerError(f"encoding widths differ: {widths}")
        object.__setattr__(self, "_tokens", tokens)
        object.__setattr__(self, "_label_index", by_key)
        object.__setattr__(self, "_encoding_dim", widths[0] if widths else None)

    @classmethod
    def build(cls, contexts: Sequence[Mapping], records: Sequence[Mapping], encodings: Mapping[str, Any],
              assignment: Mapping[str, str]) -> Corpus:
        """Validate and copy the inputs, then label them with labels.label_corpus (each context from its own rows)."""
        ctx_map: dict[str, dict] = {}
        for row in contexts:
            sha = row["context_sha256"]
            if sha in ctx_map:
                raise SamplerError(f"duplicate context {sha}")
            ctx_map[sha] = copy.deepcopy(dict(row))
        assign = {str(f): str(a) for f, a in dict(assignment).items()}
        for sha, row in ctx_map.items():
            if row["family_id"] not in assign:
                raise SamplerError(f"context {sha}: family {row['family_id']} has no holdout assignment")
        grouped: dict[str, list[dict]] = {sha: [] for sha in sorted(ctx_map)}
        for rec in records:
            sha = rec["context_sha256"]
            ctx = ctx_map.get(sha)
            if ctx is None:
                raise SamplerError(f"record {rec.get('program_sha256')} names unknown context {sha}")
            for key in ("family_id", "dataset_hash", "split_role"):
                if rec.get(key) != ctx.get(key):
                    raise SamplerError(f"record {rec.get('program_sha256')} disagrees with its context on {key}")
            length = rec["length"]
            if isinstance(length, bool) or not isinstance(length, int) or length < 2 or len(rec["tokens"]) != length:
                raise SamplerError(f"record {rec.get('program_sha256')}: len(tokens) != length or length < 2")
            if any(isinstance(t, bool) or int(t) != t or t < 0 for t in rec["tokens"]):
                raise SamplerError(f"record {rec.get('program_sha256')}: token ids must be non-negative ints")
            grouped[sha].append(copy.deepcopy(dict(rec)))
        rec_map: dict[str, tuple[dict, ...]] = {}
        for sha, rows in grouped.items():
            rows.sort(key=lambda r: r["program_sha256"])
            for a, b in zip(rows, rows[1:]):
                if a["program_sha256"] == b["program_sha256"]:
                    raise SamplerError(f"duplicate program {a['program_sha256']} in context {sha}")
            rec_map[sha] = tuple(rows)
        enc: dict[str, np.ndarray] = {}
        width = None
        for sha in sorted(encodings):
            if sha not in ctx_map:
                raise SamplerError(f"encoding for unknown context {sha}")
            z = np.array(encodings[sha], dtype=np.float32, copy=True)
            if z.ndim != 1 or z.shape[0] < 1:
                raise SamplerError(f"encoding of {sha} must be a non-empty vector, got shape {z.shape}")
            if width is None:
                width = z.shape[0]
            elif z.shape[0] != width:
                raise SamplerError(f"encoding width differs for {sha}: {z.shape[0]} != {width}")
            if not np.isfinite(z).all():
                raise SamplerError(f"non-finite encoding for {sha}")
            z.setflags(write=False)
            enc[sha] = z
        table = lbl.label_corpus(list(ctx_map.values()), [r for rows in rec_map.values() for r in rows])
        return cls(contexts=ctx_map, records=rec_map, labels=table, encodings=enc, assignment=assign)

    @property
    def encoding_dim(self) -> int | None:
        return self._encoding_dim

    def tokens(self, ctx: str, prog: str) -> tuple[int, ...]:
        try:
            return self._tokens[(ctx, prog)]
        except KeyError:
            raise SamplerError(f"no program {prog} in context {ctx}") from None

    def label(self, ctx: str, prog: str) -> lbl.RecordLabel:
        try:
            return self._label_index[(ctx, prog)]
        except KeyError:
            raise SamplerError(f"no label for program {prog} in context {ctx}") from None


# ------------------------------------------------------------------------------------------------------ plans

@dataclass(frozen=True)
class ContextPlan:
    """Per-context member pools (PLAN §6.1). labelled_classes and labelled_feas_valid are aligned with labelled and
    give each member's feasibility class and whether it enters L_feas (local backend, §6.3 backend rule), so a draw
    can record the pi_group numerators without the corpus (§6.3.2)."""

    context_sha256: str
    stream: str
    reason: str | None
    teachers: tuple[str, ...]
    bands: tuple[tuple[str, ...], tuple[str, ...]]
    failures: tuple[str, ...]
    labelled: tuple[str, ...]
    baseline: tuple[str, tuple[tuple[str, ...], ...]] | None
    labelled_classes: tuple[int, ...]
    labelled_feas_valid: tuple[bool, ...]

    def to_json(self) -> dict:
        return {"context_sha256": self.context_sha256, "stream": self.stream, "reason": self.reason,
                "teachers": list(self.teachers), "bands": [list(b) for b in self.bands],
                "failures": list(self.failures), "labelled": list(self.labelled),
                "labelled_classes": list(self.labelled_classes),
                "labelled_feas_valid": list(self.labelled_feas_valid),
                "baseline": None if self.baseline is None else [self.baseline[0], [list(b) for b in self.baseline[1]]]}


def _rank_key(r: lbl.RecordLabel) -> tuple[int, str]:
    return (-r.q_grid, r.program_sha256)


def _baseline_group(recs: Sequence[lbl.RecordLabel],
                    config: SamplerConfig) -> tuple[str, tuple[tuple[str, ...], ...]] | None:
    """Lineage-B group pools (PLAN §6.3.2): champion = max value (then sha asc) over q-known detailed rows; the rest,
    sorted by (-value_grid, sha), in group_size - 1 equal-mass bands (np.array_split order); None if too few rows."""
    rows = [r for r in recs if r.q_status == "known" and r.value is not None]
    if len(rows) < config.group_size:
        return None
    champion = min(rows, key=lambda r: (-r.value, r.program_sha256))
    rest = sorted((r for r in rows if r.program_sha256 != champion.program_sha256),
                  key=lambda r: (-lbl.q_grid(r.value), r.program_sha256))
    parts = np.array_split(np.arange(len(rest)), config.group_size - 1)
    return champion.program_sha256, tuple(tuple(rest[int(i)].program_sha256 for i in part) for part in parts)


def context_plan(corpus: Corpus, context_sha256: str, config: SamplerConfig) -> ContextPlan:
    """Pure per-context plan (PLAN §6.1); evaluate uses it for dev and diag contexts too.

    Teachers: the best program per skeleton among completed, q-known, gate-passing rows, the skeletons ranked by
    (-q_grid, sha) of their best program, top n_teachers, highest q first. Bands ("quantile bands below the teacher
    band"): the completed q-known programs whose q_grid is strictly below the lowest teacher's, sorted (-q_grid, sha),
    first ceil(n/2) then the rest; a grid tie with a teacher is inside the teacher band. The group stream needs status
    OK, n_teachers teachers and enough comparators below the band to fill the group; otherwise the context feeds only
    the feasibility stream.
    """
    try:
        ctx_label = corpus.labels.contexts[context_sha256]
    except KeyError:
        raise SamplerError(f"no labelled context {context_sha256}") from None
    recs = corpus.labels.records.get(context_sha256, ())
    known = sorted((r for r in recs if r.outcome == lbl.COMPLETED and r.q_status == "known"), key=_rank_key)
    best: dict[str, Any] = {}
    for r in known:
        if r.gate_pass is True:
            best.setdefault(r.skeleton_sha256, r)       # `known` is rank-sorted, so the first is the skeleton's best
    teacher_labels = sorted(best.values(), key=_rank_key)[:config.n_teachers]
    teachers = tuple(r.program_sha256 for r in teacher_labels)
    floor = teacher_labels[-1].q_grid if teacher_labels else None     # the lowest teacher (rank-sorted)
    rest = [r.program_sha256 for r in known
            if r.program_sha256 not in teachers and (floor is None or r.q_grid < floor)]
    half = (len(rest) + 1) // 2
    labelled = sorted((r.program_sha256, r.feas_class, bool(r.feas_valid)) for r in recs if r.feas_class is not None)
    failures = tuple(p for p, k, _ in labelled if k in (1, 2))
    need = config.n_comparators + (config.max_failures - min(config.max_failures, len(failures)))
    if ctx_label.status != "OK":
        stream, reason = "feasibility", ctx_label.status
    elif len(teachers) < config.n_teachers:
        stream, reason = "feasibility", "FEW_SKELETONS"
    elif len(rest) < need:
        stream, reason = "feasibility", "TOO_FEW_LABELLED"
    else:
        stream, reason = "group", None
    if reason is not None and reason not in PLAN_REASONS:
        raise SamplerError(f"context {context_sha256}: unexpected status {reason}")
    baseline = _baseline_group(recs, config) if config.group_mode == BASELINE_V1 else None
    return ContextPlan(context_sha256=context_sha256, stream=stream, reason=reason, teachers=teachers,
                       bands=(tuple(rest[:half]), tuple(rest[half:])), failures=failures,
                       labelled=tuple(p for p, _, _ in labelled), baseline=baseline,
                       labelled_classes=tuple(int(k) for _, k, _ in labelled),
                       labelled_feas_valid=tuple(v for _, _, v in labelled))


# -------------------------------------------------------------------------------------------------------- the fit

@dataclass(frozen=True)
class SamplerFit:
    """Fitted sampler statistics over the train_fit universe only (PLAN §6.1, §6.2): the family -> dataset -> context
    tree, per-context plans, inclusion probabilities pi, shared-seed imitation divisors, natural class priors per
    backend (§6.3.2 calibration) and label statistics. Treat every field as read-only."""

    config: SamplerConfig
    families: tuple[str, ...]
    datasets: Mapping[str, tuple[str, ...]]
    contexts: Mapping[str, tuple[str, ...]]
    plans: Mapping[str, ContextPlan]
    pi: Mapping[str, float]
    imitation_divisor: Mapping[str, int]
    class_prior_nat: Mapping[str, tuple[float, float, float]]
    label_stats: Mapping[str, Any]
    excluded: Mapping[str, int]

    def statistics(self) -> dict:
        """Every field as sorted JSON values (floats unrounded)."""
        stats = self.label_stats
        return {
            "schema": FIT_SCHEMA,
            "config": self.config.to_json(),
            "families": list(self.families),
            "datasets": {f: list(v) for f, v in sorted(self.datasets.items())},
            "contexts": {d: list(v) for d, v in sorted(self.contexts.items())},
            "plans": {c: p.to_json() for c, p in sorted(self.plans.items())},
            "pi": dict(sorted(self.pi.items())),
            "imitation_divisor": dict(sorted(self.imitation_divisor.items())),
            "class_prior_nat": {b: list(v) for b, v in sorted(self.class_prior_nat.items())},
            "label_stats": {"outcome_counts": dict(sorted(stats["outcome_counts"].items())),
                            "context_status_counts": dict(sorted(stats["context_status_counts"].items())),
                            "n_rows": stats["n_rows"], "n_labelled": stats["n_labelled"],
                            "q_quantiles": None if stats["q_quantiles"] is None else list(stats["q_quantiles"])},
            "excluded": dict(sorted(self.excluded.items())),
        }

    def sha256(self) -> str:
        return common.sha256_bytes(common.canonical(self.statistics()))

    def weights_bytes(self) -> bytes:
        return common.canonical({"families": list(self.families),
                                 "datasets": {f: list(v) for f, v in sorted(self.datasets.items())},
                                 "contexts": {d: list(v) for d, v in sorted(self.contexts.items())},
                                 "pi": dict(sorted(self.pi.items()))})

    def n_contexts(self) -> int:
        return len(self.pi)


def fit(corpus: Corpus, config: SamplerConfig) -> SamplerFit:
    """Fit the sampler on the train_fit universe (PLAN §6.1, §6.2). Only universe contexts are read for any statistic;
    pi_c = 1.0/(F·n_d·n_c) is one division of an integer product."""
    if not isinstance(config, SamplerConfig):
        raise TypeError("config must be a SamplerConfig")
    excluded = dict.fromkeys(EXCLUDED_KEYS, 0)
    tree: dict[str, dict[str, list[str]]] = {}
    dataset_family: dict[str, str] = {}
    for sha in sorted(corpus.contexts):
        row = corpus.contexts[sha]
        fam, ds, role = row["family_id"], row["dataset_hash"], row["split_role"]
        assigned = corpus.assignment.get(fam)
        if assigned is None:
            raise SamplerError(f"context {sha}: family {fam} has no holdout assignment")
        if assigned not in TRAIN_ASSIGNMENTS:
            if role in TRAIN_FAMILY_SPLIT_ROLES:
                raise SamplerError(f"context {sha} of a {assigned} family has split_role {role}")
            continue
        if role not in TRAIN_FAMILY_SPLIT_ROLES:
            raise SamplerError(f"context {sha} of {assigned} family {fam} has split_role {role}")
        if dataset_family.setdefault(ds, fam) != fam:
            raise SamplerError(f"dataset {ds} appears in families {dataset_family[ds]} and {fam}")
        node = tree.setdefault(fam, {}).setdefault(ds, [])
        if role != FIT_SPLIT_ROLE:
            excluded["diag_contexts"] += 1
            continue
        if sha not in corpus.labels.contexts:
            raise SamplerError(f"train_fit context {sha} has no label table entry")
        if not any(r.labelled for r in corpus.labels.records.get(sha, ())):
            excluded["unlabelled_contexts"] += 1
            continue
        node.append(sha)
    families: list[str] = []
    datasets: dict[str, tuple[str, ...]] = {}
    contexts: dict[str, tuple[str, ...]] = {}
    for fam in sorted(f for f, a in corpus.assignment.items() if a in TRAIN_ASSIGNMENTS):
        kept = []
        for ds in sorted(tree.get(fam, {})):
            members = sorted(tree[fam][ds])
            if not members:
                excluded["empty_datasets"] += 1
                continue
            kept.append(ds)
            contexts[ds] = tuple(members)
        if not kept:
            excluded["empty_families"] += 1
            continue
        families.append(fam)
        datasets[fam] = tuple(kept)
    if not families:
        raise SamplerError("the fit universe is empty")
    n_fam = len(families)
    pi: dict[str, float] = {}
    for fam in families:
        n_d = len(datasets[fam])
        for ds in datasets[fam]:
            n_c = len(contexts[ds])
            for c in contexts[ds]:
                pi[c] = 1.0 / (n_fam * n_d * n_c)
    universe = sorted(pi)
    plans = {c: context_plan(corpus, c, config) for c in universe}
    divisors: Counter = Counter()
    for c in universe:
        if plans[c].stream == "group":
            for p in plans[c].teachers:
                if corpus.label(c, p).shared_seed:
                    divisors[p] += 1
    class_counts: dict[str, list[int]] = {}
    outcomes: Counter = Counter()
    statuses: Counter = Counter()
    n_rows = n_labelled = 0
    q_known: list[float] = []
    for c in universe:
        statuses[corpus.labels.contexts[c].status] += 1
        for r in corpus.labels.records.get(c, ()):
            n_rows += 1
            outcomes[r.outcome] += 1
            n_labelled += int(bool(r.labelled))
            if r.feas_class is not None:
                class_counts.setdefault(r.backend, [0, 0, 0])[r.feas_class] += 1
            if r.q_status == "known":
                q_known.append(float(r.q))
    q_quantiles = None
    if q_known:
        q_quantiles = tuple(float(x) for x in np.percentile(np.asarray(q_known, dtype=np.float64), Q_QUANTILES,
                                                            method="linear"))
    return SamplerFit(
        config=config,
        families=tuple(families),
        datasets=datasets,
        contexts=dict(sorted(contexts.items())),
        plans=plans,
        pi=dict(sorted(pi.items())),
        imitation_divisor=dict(sorted(divisors.items())),
        class_prior_nat={b: tuple(n / sum(v) for n in v) for b, v in sorted(class_counts.items())},
        label_stats={"outcome_counts": dict(sorted(outcomes.items())),
                     "context_status_counts": dict(sorted(statuses.items())),
                     "n_rows": n_rows, "n_labelled": n_labelled, "q_quantiles": q_quantiles},
        excluded=excluded,
    )


# -------------------------------------------------------------------------------------------- counter-based draws

def _check_seed(run_seed: bytes) -> bytes:
    if not isinstance(run_seed, (bytes, bytearray)) or len(run_seed) != RUN_SEED_BYTES:
        raise SamplerError(f"run_seed must be {RUN_SEED_BYTES} bytes")
    return bytes(run_seed)


def uniform_index(run_seed: bytes, step: int, slot: int, purpose: bytes, n: int) -> int:
    """Exactly uniform index in [0, n) from sha256(DOMAIN ‖ run_seed ‖ step(8) ‖ slot(4) ‖ purpose ‖ 0 ‖ k(4)), taking the
    first 64-bit word u_k below the largest multiple of n (PLAN §6.6: the draw is fixed by run_seed, step and slot)."""
    seed = _check_seed(run_seed)
    step, slot, n = operator.index(step), operator.index(slot), operator.index(n)
    if n < 1:
        raise SamplerError(f"uniform_index needs n >= 1, got {n}")
    if not 0 <= step < 1 << 64 or not 0 <= slot < 1 << 32:
        raise SamplerError(f"step {step} or slot {slot} out of range")
    if not isinstance(purpose, bytes) or b"\0" in purpose:
        raise SamplerError("purpose must be bytes without NUL")
    base = hashlib.sha256(DOMAIN + seed + step.to_bytes(8, "big") + slot.to_bytes(4, "big") + purpose + b"\0")
    limit = ((1 << 64) // n) * n
    k = 0
    while True:
        h = base.copy()
        h.update(k.to_bytes(4, "big"))
        u = int.from_bytes(h.digest()[:8], "big")
        if u < limit:
            return u % n
        k += 1


def macro_draw(fit: SamplerFit, run_seed: bytes, step: int, slot: int) -> tuple[str, str, str]:
    """(family, dataset, context): a family uniformly from the universe, a dataset uniformly within it, a context
    uniformly within that dataset (PLAN §6.1 draw steps 1-3)."""
    if not fit.families:
        raise SamplerError("the fit universe is empty")
    family = fit.families[uniform_index(run_seed, step, slot, b"family", len(fit.families))]
    ds_list = fit.datasets[family]
    dataset = ds_list[uniform_index(run_seed, step, slot, b"dataset", len(ds_list))]
    ctx_list = fit.contexts[dataset]
    return family, dataset, ctx_list[uniform_index(run_seed, step, slot, b"context", len(ctx_list))]


@dataclass(frozen=True)
class SlotDraw:
    """One drawn context of a step: its inclusion probability, stream, members (in group order), roles, the
    feasibility-class counts of every member, and feas_class_counts, the class counts of the members that enter
    L_feas (feas_valid: local backend). feas_class_counts are the pi_group numerators of the prior-shift readout,
    since the feasibility head is trained on those members only (PLAN §6.3 backend rule, §6.3.2)."""

    slot: int
    family_id: str
    dataset_hash: str
    context_sha256: str
    pi_c: float
    stream: str
    members: tuple[str, ...]
    roles: tuple[str, ...]
    class_counts: tuple[int, int, int]
    feas_class_counts: tuple[int, int, int]


@dataclass(frozen=True)
class StepDraw:
    """All slots of one step. journal_record() is the deterministic consumption-journal line (no times, pids or
    paths; PLAN §6.1 'realized draws are recorded in the consumption journal')."""

    step: int
    slots: tuple[SlotDraw, ...]

    def journal_record(self) -> dict:
        return {"schema": JOURNAL_SCHEMA, "step": self.step,
                "slots": [{"slot": s.slot, "family_id": s.family_id, "dataset_hash": s.dataset_hash,
                           "context_sha256": s.context_sha256, "pi_c": s.pi_c, "stream": s.stream,
                           "members": list(s.members), "roles": list(s.roles),
                           "class_counts": list(s.class_counts),
                           "feas_class_counts": list(s.feas_class_counts)} for s in self.slots]}


def _partial_fisher_yates(pool: Sequence[str], m: int, run_seed: bytes, step: int, slot: int,
                          tag: bytes) -> tuple[str, ...]:
    arr = list(pool)
    for i in range(m):
        j = i + uniform_index(run_seed, step, slot, b"%s:%d" % (tag, i), len(arr) - i)
        arr[i], arr[j] = arr[j], arr[i]
    return tuple(arr[:m])


def _members(plan: ContextPlan, config: SamplerConfig, run_seed: bytes, step: int,
             slot: int) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """(stream, members, roles) of one slot (PLAN §6.1 groups; §6.3.2 lineage-B group in baseline mode)."""
    if config.group_mode == BASELINE_V1:
        if plan.baseline is None:
            return "skipped", (), ()
        champion, bands = plan.baseline
        picks = tuple(band[uniform_index(run_seed, step, slot, b"band:%d" % i, len(band))]
                      for i, band in enumerate(bands))
        return "group", (champion,) + picks, ("champion",) + ("comparator",) * len(picks)
    if plan.stream == "group":
        n_fail = min(config.max_failures, len(plan.failures))
        failures = _partial_fisher_yates(plan.failures, n_fail, run_seed, step, slot, b"failure")
        n_comp = config.n_comparators + (config.max_failures - n_fail)
        bands = [list(plan.bands[0]), list(plan.bands[1])]
        comparators = []
        for i in range(n_comp):
            b = i % 2
            if not bands[b]:
                b = 1 - b
            if not bands[b]:
                raise SamplerError(f"context {plan.context_sha256}: comparator bands exhausted")
            comparators.append(bands[b].pop(uniform_index(run_seed, step, slot, b"comparator:%d" % i, len(bands[b]))))
        members = plan.teachers + tuple(comparators) + failures
        roles = ("teacher",) * len(plan.teachers) + ("comparator",) * n_comp + ("failure",) * n_fail
        return "group", members, roles
    m = min(config.feas_stream_max, len(plan.labelled))
    return "feasibility", _partial_fisher_yates(plan.labelled, m, run_seed, step, slot, b"feas"), ("feasibility",) * m


def _slot_draw(fit: SamplerFit, config: SamplerConfig, run_seed: bytes, step: int, slot: int) -> SlotDraw:
    family, dataset, ctx = macro_draw(fit, run_seed, step, slot)
    plan = fit.plans[ctx]
    stream, members, roles = _members(plan, config, run_seed, step, slot)
    index = {p: i for i, p in enumerate(plan.labelled)}
    counts, feas_counts = [0, 0, 0], [0, 0, 0]
    for m in members:
        i = index.get(m)
        if i is None:
            raise SamplerError(f"member {m} of context {ctx} is not labelled")
        counts[plan.labelled_classes[i]] += 1
        if plan.labelled_feas_valid[i]:
            feas_counts[plan.labelled_classes[i]] += 1
    return SlotDraw(slot=slot, family_id=family, dataset_hash=dataset, context_sha256=ctx, pi_c=fit.pi[ctx],
                    stream=stream, members=members, roles=roles, class_counts=(counts[0], counts[1], counts[2]),
                    feas_class_counts=(feas_counts[0], feas_counts[1], feas_counts[2]))


class MacroSampler:
    """Counter-based sampler (PLAN §6.6 'sampler'): draw(step) is pure, and draw_next() draws step cursor + 1 and then
    advances the cursor. The cursor is authoritative and is never re-derived from the trainer's step counter."""

    def __init__(self, fit: SamplerFit, run_seed: bytes, config: SamplerConfig) -> None:
        self._run_seed = _check_seed(run_seed)
        if config != fit.config:
            raise SamplerError("sampler config differs from the config the fit was computed with")
        self.fit = fit
        self.config = config
        self.cursor = 0
        self._run_seed_sha256 = common.sha256_bytes(self._run_seed)
        self._fit_sha256 = fit.sha256()
        self._config_sha256 = config.sha256()

    @property
    def run_seed_sha256(self) -> str:
        return self._run_seed_sha256

    def draw(self, step: int) -> StepDraw:
        step = operator.index(step)
        if step < 0:
            raise SamplerError(f"step must be >= 0, got {step}")
        return StepDraw(step=step, slots=tuple(_slot_draw(self.fit, self.config, self._run_seed, step, s)
                                               for s in range(self.config.b_ctx)))

    def draw_next(self) -> StepDraw:
        out = self.draw(self.cursor + 1)
        self.cursor += 1
        return out

    def state_dict(self) -> dict:
        return {"schema": STATE_SCHEMA, "cursor": self.cursor, "run_seed_sha256": self._run_seed_sha256,
                "fit_sha256": self._fit_sha256, "config_sha256": self._config_sha256}

    def load_state_dict(self, d: Mapping[str, Any]) -> None:
        if d.get("schema") != STATE_SCHEMA:
            raise SamplerError(f"sampler state schema {d.get('schema')!r} != {STATE_SCHEMA}")
        for key, mine in (("run_seed_sha256", self._run_seed_sha256), ("fit_sha256", self._fit_sha256),
                          ("config_sha256", self._config_sha256)):
            if d.get(key) != mine:
                raise SamplerError(f"sampler state {key} mismatch: {d.get(key)!r} != {mine}")
        cursor = d.get("cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise SamplerError(f"sampler cursor must be an int >= 0, got {cursor!r}")
        self.cursor = cursor


# ------------------------------------------------------------------------------------------------------ collation

@dataclass(frozen=True)
class TrainData:
    """Everything a training step reads: the corpus, its fit and config, the grammar mask function (rows of allowed
    next tokens, shape (len-1, V)), the vocabulary size and the encoding width. refs holds the input identities a
    loader stamps (DATA_REF_KEYS; s3refs or sha256 strings); the trainer refuses a run plan whose refs differ."""

    corpus: Corpus
    fit: SamplerFit
    config: SamplerConfig
    legal_mask_fn: Callable[[Sequence[int]], np.ndarray]
    vocab_size: int
    ctx_dim: int
    refs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.config != self.fit.config:
            raise SamplerError("TrainData config differs from the fit's config")
        for name in ("vocab_size", "ctx_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an int >= 1, got {value!r}")
        if self.corpus.encoding_dim is not None and self.corpus.encoding_dim != self.ctx_dim:
            raise SamplerError(f"encoding width {self.corpus.encoding_dim} != ctx_dim {self.ctx_dim}")
        if self.refs is not None:
            if not isinstance(self.refs, Mapping):
                raise ValueError("refs must be a mapping or None")
            unknown = sorted(set(self.refs) - set(DATA_REF_KEYS))
            if unknown:
                raise ValueError(f"unknown data refs {unknown}; allowed {DATA_REF_KEYS}")
            refs = copy.deepcopy(dict(self.refs))
            common.canonical(refs)                              # JSON values only, no NaN
            object.__setattr__(self, "refs", refs)

    @functools.cached_property
    def content_sha256(self) -> str:
        """sha256 of everything a training step reads from the train_fit universe (PLAN §6.6 identities): a canonical
        header (schema, vocab_size, ctx_dim, stamped refs), then per universe context in sorted order its float32
        little-endian encoding and, per record in program_sha256 order, its tokens and every label field; each
        program's grammar mask too unless a grammar ref is stamped. Dev, diag, heldout and sealed units never enter
        it, so closure poison leaves it unchanged (P7 test 7)."""
        grammar_stamped = bool(self.refs) and self.refs.get("grammar") is not None
        h = hashlib.sha256()
        h.update(common.canonical({"schema": TRAIN_DATA_SCHEMA, "vocab_size": self.vocab_size, "ctx_dim": self.ctx_dim,
                                   "refs": self.refs, "grammar_masks": not grammar_stamped}) + b"\n")
        for ctx in sorted(self.fit.pi):
            z = self.corpus.encodings.get(ctx)
            h.update(common.canonical({"context_sha256": ctx, "encoding_width": None if z is None else len(z)}) + b"\n")
            if z is not None:
                h.update(np.ascontiguousarray(z, dtype="<f4").tobytes())
            for rec in self.corpus.records.get(ctx, ()):
                prog = rec["program_sha256"]
                tokens = self.corpus.tokens(ctx, prog)
                h.update(common.canonical({"tokens": list(tokens), "label": self.corpus.label(ctx, prog).to_json()})
                         + b"\n")
                if not grammar_stamped:
                    mask = np.asarray(self.legal_mask_fn(tokens), dtype=np.bool_)
                    h.update(common.canonical(list(mask.shape)) + np.packbits(mask, axis=None).tobytes())
        return h.hexdigest()


@dataclass(eq=False)
class Batch:
    """One collated step (PLAN §6.1, §6.3, §6.3.2). Contexts are the slots with members, in slot order; sequences are
    their members in group order. Unknown floats are 0.0 with their validity flag False; feas_class is -1 when None.
    role: 0 teacher or champion, 1 comparator, 2 failure, 3 feasibility."""

    mode: str
    step: int
    tokens: torch.Tensor            # int64 [N, T], PAD 0
    lengths: torch.Tensor           # int64 [N], BOS..EOS inclusive
    legal: torch.Tensor             # bool [N, T-1, V]; row i = allowed tokens[:, i+1]; rows >= lengths-1 all False
    ctx_z: torch.Tensor             # float32 [C, W] (a copy; augmentation may edit it in place)
    seq_ctx: torch.Tensor           # int64 [N] index into the C contexts
    ctx_weight: torch.Tensor        # float32 [C]: 1.0 group, feas_stream_weight feasibility
    ctx_is_group: torch.Tensor      # bool [C]
    role: torch.Tensor              # int64 [N]
    q: torch.Tensor                 # float32 [N]
    q_opt: torch.Tensor
    midrank_pct: torch.Tensor
    value: torch.Tensor
    value_pct: torch.Tensor
    q_grid: torch.Tensor            # int64 [N]
    q_known: torch.Tensor           # bool [N]
    gate_pass: torch.Tensor
    feas_valid: torch.Tensor
    rt_valid: torch.Tensor
    rt_censored: torch.Tensor
    feas_class: torch.Tensor        # int64 [N], -1 when unlabelled
    rt_target: torch.Tensor         # float32 [N], log1p seconds
    imitation_divisor: torch.Tensor  # float32 [N], >= 1.0
    ctx_ids: list[str]
    member_ids: list[tuple[str, str]]
    n_tokens: int


BATCH_TENSOR_FIELDS = ("tokens", "lengths", "legal", "ctx_z", "seq_ctx", "ctx_weight", "ctx_is_group", "role", "q",
                       "q_opt", "midrank_pct", "value", "value_pct", "q_grid", "q_known", "gate_pass", "feas_valid",
                       "rt_valid", "rt_censored", "feas_class", "rt_target", "imitation_divisor")


def _or(value, default):
    return default if value is None else value


def collate(draw: StepDraw, data: TrainData) -> Batch:
    """Deterministic collation of one StepDraw (PLAN §6.1): T = max length, PAD 0, grammar-mask rows from
    data.legal_mask_fn, encodings copied as float32. Reads only the drawn (train_fit universe) contexts."""
    corpus, fit, config = data.corpus, data.fit, data.config
    ctx_ids: list[str] = []
    ctx_rows: list[np.ndarray] = []
    ctx_weight: list[float] = []
    ctx_group: list[bool] = []
    seqs: list[tuple[int, str, str, str]] = []
    for s in draw.slots:
        if not s.members:
            continue
        z = corpus.encodings.get(s.context_sha256)
        if z is None:
            raise SamplerError(f"no encoding for drawn context {s.context_sha256}")
        if z.shape != (data.ctx_dim,):
            raise SamplerError(f"encoding of {s.context_sha256} has shape {z.shape}, expected ({data.ctx_dim},)")
        ci = len(ctx_ids)
        ctx_ids.append(s.context_sha256)
        ctx_rows.append(z)
        is_group = s.stream == "group"
        ctx_weight.append(1.0 if is_group else config.feas_stream_weight)
        ctx_group.append(is_group)
        seqs.extend((ci, s.context_sha256, prog, role) for prog, role in zip(s.members, s.roles))
    if not seqs:
        raise SamplerError(f"step {draw.step} drew no members")
    toks = [corpus.tokens(c, p) for _, c, p, _ in seqs]
    n, t_max, v = len(seqs), max(len(t) for t in toks), data.vocab_size
    tokens = np.full((n, t_max), PAD_ID, dtype=np.int64)
    legal = np.zeros((n, t_max - 1, v), dtype=np.bool_)
    for i, t in enumerate(toks):
        if len(t) < 2 or max(t) >= v:
            raise SamplerError(f"sequence {seqs[i][2]} has length {len(t)} or a token id >= {v}")
        tokens[i, :len(t)] = t
        mask = np.asarray(data.legal_mask_fn(t))
        if mask.dtype != np.bool_ or mask.shape != (len(t) - 1, v):
            raise SamplerError(f"legal_mask_fn returned {mask.dtype} {mask.shape}, expected bool ({len(t) - 1}, {v})")
        legal[i, :len(t) - 1] = mask
    lengths = np.array([len(t) for t in toks], dtype=np.int64)
    labs = [corpus.label(c, p) for _, c, p, _ in seqs]

    def f32(vals: list[float]) -> torch.Tensor:
        return torch.tensor(vals, dtype=torch.float32)

    def i64(vals: list[int]) -> torch.Tensor:
        return torch.tensor(vals, dtype=torch.int64)

    def flag(vals: list[bool]) -> torch.Tensor:
        return torch.tensor(vals, dtype=torch.bool)

    divisors = [float(fit.imitation_divisor.get(lab.program_sha256, 1)) if lab.shared_seed else 1.0 for lab in labs]
    return Batch(
        mode=config.group_mode,
        step=draw.step,
        tokens=torch.from_numpy(tokens),
        lengths=torch.from_numpy(lengths),
        legal=torch.from_numpy(legal),
        ctx_z=torch.from_numpy(np.stack(ctx_rows).astype(np.float32, copy=True)),
        seq_ctx=i64([ci for ci, _, _, _ in seqs]),
        ctx_weight=f32(ctx_weight),
        ctx_is_group=flag(ctx_group),
        role=i64([ROLE_CODES[role] for _, _, _, role in seqs]),
        q=f32([_or(lab.q, 0.0) for lab in labs]),
        q_opt=f32([_or(lab.q_opt, 0.0) for lab in labs]),
        midrank_pct=f32([_or(lab.midrank_pct, 0.0) for lab in labs]),
        value=f32([_or(lab.value, 0.0) for lab in labs]),
        value_pct=f32([_or(lab.value_pct, 0.0) for lab in labs]),
        q_grid=i64([_or(lab.q_grid, 0) for lab in labs]),
        q_known=flag([lab.q_status == "known" for lab in labs]),
        gate_pass=flag([lab.gate_pass is True for lab in labs]),
        feas_valid=flag([bool(lab.feas_valid) for lab in labs]),
        rt_valid=flag([bool(lab.rt_valid) for lab in labs]),
        rt_censored=flag([bool(lab.rt_censored) for lab in labs]),
        feas_class=i64([_or(lab.feas_class, -1) for lab in labs]),
        rt_target=f32([_or(lab.rt_target, 0.0) for lab in labs]),
        imitation_divisor=f32(divisors),
        ctx_ids=ctx_ids,
        member_ids=[(c, p) for _, c, p, _ in seqs],
        n_tokens=int(lengths.sum()) - n,
    )
