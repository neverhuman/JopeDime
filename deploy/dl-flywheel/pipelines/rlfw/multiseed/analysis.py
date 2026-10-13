"""Complete multi-seed L6 panels and independently scored selected teachers.

Inputs are caller-owned measurement declarations. SHA and row-set checks do
not authenticate their bodies; outputs never authorize a reward or promotion.
"""

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from statistics import fmean, stdev


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("full lowercase SHA256 required")


def _number(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError("finite scalar measurement required")
    return float(value)


def _ids(values):
    if type(values) is not tuple or not values or any(type(x) is not str or not x for x in values):
        raise ValueError("nonempty immutable IDs required")
    if len(set(values)) != len(values):
        raise ValueError("duplicate IDs refused")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Context:
    context_id: str
    parent_id: str
    comparison_sha256: str

    def __post_init__(self):
        _ids((self.context_id,))
        _ids((self.parent_id,))
        _sha(self.comparison_sha256)


@dataclass(frozen=True)
class Study:
    seeds: tuple[int, ...]
    contexts: tuple[Context, ...]
    train_parent_ids: tuple[str, ...]
    v1_checkpoint_sha256: str
    v1_recipe_sha256: str
    updates: int

    def __post_init__(self):
        if (
            type(self.seeds) is not tuple
            or len(self.seeds) < 3
            or any(type(s) is not int or s < 0 for s in self.seeds)
        ):
            raise ValueError("at least three preregistered independent seed IDs required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds refused")
        if (
            type(self.contexts) is not tuple
            or not self.contexts
            or any(type(c) is not Context for c in self.contexts)
        ):
            raise ValueError("immutable frozen context roster required")
        if len({c.context_id for c in self.contexts}) != len(self.contexts):
            raise ValueError("duplicate contexts refused")
        _ids(self.train_parent_ids)
        parents = {c.parent_id for c in self.contexts}
        if len(parents) < 2 or parents & set(self.train_parent_ids):
            raise ValueError("at least two evaluation parents, disjoint from training, required")
        _sha(self.v1_checkpoint_sha256)
        _sha(self.v1_recipe_sha256)
        if type(self.updates) is not int or self.updates <= 0:
            raise ValueError("one positive equal update budget required")

    @property
    def sha256(self):
        return _digest(asdict(self))

    def plan(self):
        """A declaration for future approval, never an executable launch plan."""
        return {
            "schema": "jope.j22-plan/v1",
            "study_sha256": self.sha256,
            "study": asdict(self),
            "primary": "mean paired L6 difference over ALL preregistered seeds and contexts",
            "direction": "seed_model - frozen_v1; negative is better",
            "best_seed": "descriptive only; never the primary result",
            "training_authorized": False,
            "power": "UNMEASURED",
            "qualified": False,
        }


@dataclass(frozen=True)
class Observation:
    seed: int
    context_id: str
    comparison_sha256: str
    checkpoint_sha256: str
    training_receipt_sha256: str
    measurement_sha256: str
    updates: int
    l6_cost: float
    v1_l6_cost: float

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0 or type(self.updates) is not int or self.updates <= 0:
            raise ValueError("integer seed and positive updates required")
        _ids((self.context_id,))
        for value in (
            self.comparison_sha256,
            self.checkpoint_sha256,
            self.training_receipt_sha256,
            self.measurement_sha256,
        ):
            _sha(value)
        object.__setattr__(self, "l6_cost", _number(self.l6_cost, nonnegative=True))
        object.__setattr__(self, "v1_l6_cost", _number(self.v1_l6_cost, nonnegative=True))


def _percentile(values, fraction):
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    return ordered[lo] + (index - lo) * (ordered[hi] - ordered[lo])


def _replicate(sums, counts, seed_indices, parent_indices):
    # One seed draw is shared across ALL parents, preserving the crossed design.
    total = math.fsum(sums[s][p] for s in seed_indices for p in parent_indices)
    return total / (len(seed_indices) * sum(counts[p] for p in parent_indices))


def multiseed_report(study, observations, *, draws=2000, bootstrap_seed=7):
    if type(study) is not Study or type(draws) is not int or not 100 <= draws <= 100_000:
        raise ValueError("Study and bounded bootstrap draws required")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("nonnegative bootstrap seed required")
    contexts = {c.context_id: c for c in study.contexts}
    seeds = sorted(study.seeds)
    rows, models, baselines = {}, {}, {}
    for row in observations:
        if type(row) is not Observation or row.seed not in seeds or row.context_id not in contexts:
            raise ValueError("row outside preregistered seed/context panel")
        key = (row.seed, row.context_id)
        if key in rows:
            raise ValueError("duplicate seed/context measurement refused")
        if (
            row.comparison_sha256 != contexts[row.context_id].comparison_sha256
            or row.updates != study.updates
        ):
            raise ValueError("comparison identity or update budget changed")
        model = (row.checkpoint_sha256, row.training_receipt_sha256)
        if row.seed in models and models[row.seed] != model:
            raise ValueError("one frozen model/training receipt per seed required")
        if row.context_id in baselines and baselines[row.context_id] != row.v1_l6_cost:
            raise ValueError("frozen v1 cost changed across seeds")
        rows[key], models[row.seed], baselines[row.context_id] = row, model, row.v1_l6_cost
    if len(rows) != len(seeds) * len(contexts):
        raise ValueError("incomplete panel; no imputation or survivor renormalization")
    if any(len({model[k] for model in models.values()}) != len(seeds) for k in (0, 1)):
        raise ValueError("reused checkpoint/training receipt is not an independent seed")
    if any(model[0] == study.v1_checkpoint_sha256 for model in models.values()):
        raise ValueError("frozen reference relabelled as a trained seed")
    parents = sorted({c.parent_id for c in contexts.values()})
    by_parent = [[c for c in sorted(contexts) if contexts[c].parent_id == p] for p in parents]
    counts = [len(cs) for cs in by_parent]
    sums = [[math.fsum(rows[s, c].l6_cost - baselines[c] for c in cs) for cs in by_parent] for s in seeds]
    per_seed = [
        {
            "seed": s,
            "checkpoint_sha256": models[s][0],
            "training_receipt_sha256": models[s][1],
            "l6_cost": fmean(rows[s, c].l6_cost for c in sorted(contexts)),
            "delta_v1": math.fsum(sums[i]) / len(contexts),
        }
        for i, s in enumerate(seeds)
    ]
    rng = random.Random(bootstrap_seed)
    boot = [
        _replicate(
            sums,
            counts,
            [rng.randrange(len(seeds)) for _ in seeds],
            [rng.randrange(len(parents)) for _ in parents],
        )
        for _ in range(draws)
    ]
    deltas = [r["delta_v1"] for r in per_seed]
    return {
        "schema": "jope.j22-multiseed/v1",
        "study_sha256": study.sha256,
        "seeds": len(seeds),
        "parents": len(parents),
        "contexts": len(contexts),
        "v1_checkpoint_sha256": study.v1_checkpoint_sha256,
        "v1_l6_cost": fmean(baselines.values()),
        "per_seed": per_seed,
        "mean_l6_cost": fmean(r["l6_cost"] for r in per_seed),
        "mean_delta_v1": fmean(deltas),
        "between_seed_sample_sd": stdev(deltas),
        "parent_balanced_delta_sensitivity": fmean(
            sums[s][p] / counts[p] for s in range(len(seeds)) for p in range(len(parents))
        ),
        "crossed_seed_parent_ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "bootstrap": {
            "draws": draws,
            "seed": bootstrap_seed,
            "method": "independent seed and parent resampling; all sibling contexts retained",
        },
        "best_seed_descriptive": min(per_seed, key=lambda r: (r["delta_v1"], r["seed"]))["seed"],
        "direction": "negative delta is lower/better L6",
        "scope": "descriptive; finite-sample coverage unqualified",
        "measurement_authentication": "caller-owned; hashes/metadata checked only",
        "training_authorized": False,
        "qualified": False,
        "claim": None,
    }


@dataclass(frozen=True)
class Candidate:
    program_sha256: str
    inner_q: float
    measurement_sha256: str

    def __post_init__(self):
        _sha(self.program_sha256)
        _sha(self.measurement_sha256)
        object.__setattr__(self, "inner_q", _number(self.inner_q))


@dataclass(frozen=True)
class Selection:
    comparison_sha256: str
    candidates: tuple[Candidate, ...]
    fit_row_ids: tuple[str, ...]
    selection_row_ids: tuple[str, ...]

    def __post_init__(self):
        _sha(self.comparison_sha256)
        _ids(self.fit_row_ids)
        _ids(self.selection_row_ids)
        if not set(self.fit_row_ids) <= set(self.selection_row_ids):
            raise ValueError("selection exposure must include all fitting/transform-fitting rows")
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or any(type(c) is not Candidate for c in self.candidates)
        ):
            raise ValueError("immutable inner-only candidates required")
        if len({c.program_sha256 for c in self.candidates}) != len(self.candidates):
            raise ValueError("duplicate candidate programs refused")

    @property
    def winner(self):
        return min(self.candidates, key=lambda c: (-c.inner_q, c.program_sha256))

    @property
    def sha256(self):
        return _digest(asdict(self))


def score_selected_teacher(
    selection,
    *,
    commitment_sha256,
    comparison_sha256,
    program_sha256,
    fit_row_ids,
    judge_row_ids,
    judge_q,
    incumbent_judge_q,
    measurement_sha256,
    incumbent_measurement_sha256,
):
    """Consume scores for the PREVIOUSLY selected program on independent rows.

    No program is run here. Independence checks cover declared row sets only;
    fit/transform artifacts and score bodies need external authentication.
    """
    if type(selection) is not Selection or commitment_sha256 != selection.sha256:
        raise ValueError("selection commitment changed")
    if comparison_sha256 != selection.comparison_sha256 or program_sha256 != selection.winner.program_sha256:
        raise ValueError("comparison or selected program changed")
    _ids(fit_row_ids)
    _ids(judge_row_ids)
    if set(fit_row_ids) != set(selection.fit_row_ids) or set(judge_row_ids) & set(
        selection.selection_row_ids
    ):
        raise ValueError("fit scope changed or judge rows exposed to fitting/selection")
    _sha(measurement_sha256)
    _sha(incumbent_measurement_sha256)
    q, incumbent = _number(judge_q), _number(incumbent_judge_q)
    return {
        "schema": "jope.j22-heldout-teacher/v1",
        "selection_sha256": selection.sha256,
        "program_sha256": program_sha256,
        "candidates_disclosed": len(selection.candidates),
        "inner_q": selection.winner.inner_q,
        "heldout_q": q,
        "incumbent_heldout_q": incumbent,
        "heldout_advantage": q - incumbent,
        "observed_optimism_gap": selection.winner.inner_q - q,
        "measurement_sha256": measurement_sha256,
        "incumbent_measurement_sha256": incumbent_measurement_sha256,
        "scope": "independent declared rows; no unseen-family or unbiasedness certification",
        "inner_q_is_reward": False,
        "qualified_reward": False,
        "claim": None,
    }


def optimism_bound(candidate_count, *, sigma=None):
    """Spec G12 diagnostic ONLY under a zero-mean sub-Gaussian noise assumption."""
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("positive distinct candidate count required")
    bound = (
        None if sigma is None else _number(sigma, nonnegative=True) * math.sqrt(2 * math.log(candidate_count))
    )
    return {
        "candidate_count": candidate_count,
        "bound": bound,
        "status": "UNMEASURED" if sigma is None else "hypothetical sub-Gaussian proxy, not measured SD/SE",
        "corrected_reward": None,
        "qualified": False,
    }
