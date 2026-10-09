"""Fixed best-of-32 search using predicted heads and no measured scores."""

import math
import time
from dataclasses import dataclass, replace

from .arena.arms import BaselineArms, Candidate, draw_seed

K = 32
RAW_LIMIT = 512
BUDGET = 8
SELECTION = "q_rank_feasibility_0.5_sha_tie"


def select(slate):
    """Pure ranking: feasibility guard, with an all-candidate fallback."""
    if len({c.program.program_sha256 for c in slate}) != len(slate):
        raise ValueError("rerank slate must be distinct")
    if any(c.predicted is None for c in slate):
        raise ValueError("all slate members require predicted heads")
    if len(slate) < BUDGET:
        return ()
    feasible = [c for c in slate if c.predicted["p_complete"] >= 0.5]
    pool = feasible if len(feasible) >= BUDGET else slate
    ranked = sorted(pool, key=lambda c: (-c.predicted["q"], c.program.program_sha256))
    # A raw draw likelihood is not the propensity of a deduplicated selection.
    return tuple(
        replace(c, propensity=None, selection_rule=SELECTION) for c in ranked[:BUDGET]
    )


@dataclass(frozen=True)
class PreparedSlate:
    raw_draws: tuple[Candidate, ...]
    slate: tuple[Candidate, ...]
    selected: tuple[Candidate, ...]
    decode_seconds: float
    head_seconds: float
    selection_seconds: float

    def __post_init__(self):
        if any(
            not isinstance(v, tuple)
            for v in (self.raw_draws, self.slate, self.selected)
        ):
            raise ValueError("immutable candidate collections required")
        if not 1 <= len(self.raw_draws) <= RAW_LIMIT:
            raise ValueError("bounded raw draw packet required")
        first = {}
        for c in self.raw_draws:
            if (
                c.total_logprob is None
                or c.selection_rule != "iid_trie_sample"
                or c.batch_protocol_sha256 is None
            ):
                raise ValueError("raw draws require exact supported likelihood")
            if len(first) < K:
                first.setdefault(c.program.program_sha256, c)
        if tuple(first.values()) != self.slate:
            raise ValueError("slate must contain the first 32 distinct raw draws")
        predictions = {c.program.program_sha256: c.predicted for c in self.slate}
        if any(
            c.predicted != predictions.get(c.program.program_sha256)
            for c in self.raw_draws
        ):
            raise ValueError("raw predictions must match the complete scored slate")
        if len(self.slate) < K and len(self.raw_draws) != RAW_LIMIT:
            raise ValueError("shortfall requires all 512 raw draws")
        if select(self.slate) != self.selected:
            raise ValueError("selection does not match the fixed predicted ranking")
        if any(
            not math.isfinite(t) or t < 0
            for t in (self.decode_seconds, self.head_seconds, self.selection_seconds)
        ):
            raise ValueError("finite nonnegative phase timing required")

    def to_dict(self):
        return {
            "schema": "jope.rlfw-slate/v1",
            "k": K,
            "raw_limit": RAW_LIMIT,
            "budget": BUDGET,
            "shortfall": K - len(self.slate),
            "raw_draws": [c.to_dict() for c in self.raw_draws],
            "slate": [c.to_dict() for c in self.slate],
            "selected": [c.to_dict() for c in self.selected],
            "decode_seconds": self.decode_seconds,
            "head_seconds": self.head_seconds,
            "selection_seconds": self.selection_seconds,
        }

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        if (d.pop("schema"), d.pop("k"), d.pop("raw_limit"), d.pop("budget")) != (
            "jope.rlfw-slate/v1",
            K,
            RAW_LIMIT,
            BUDGET,
        ):
            raise ValueError("incompatible slate protocol")
        shortfall = d.pop("shortfall")
        for name in ("raw_draws", "slate", "selected"):
            d[name] = tuple(Candidate.from_dict(c) for c in d[name])
        result = cls(**d)
        if shortfall != K - len(result.slate):
            raise ValueError("shortfall does not match the logged slate")
        return result


class RerankArms(BaselineArms):
    """Opt-in rerank32:<checkpoint>; baseline arms retain their behavior."""

    def __init__(self, catalog, hwm_programs, sample, sample_many, score_heads):
        super().__init__(catalog, hwm_programs, sample)
        self.sample_many, self.score_heads = sample_many, score_heads

    def prepare_slate(self, context, arm, seed):
        if not arm.name.startswith("rerank32:"):
            raise ValueError("unsupported rerank arm")
        if arm.temperature != 1.0:
            raise ValueError("rerank uses preregistered temperature 1.0")
        raw, distinct = [], {}
        start = time.monotonic()
        while len(distinct) < K and len(raw) < RAW_LIMIT:
            seeds = [
                draw_seed(context.context_id, arm.name, seed, i + 1)
                for i in range(len(raw), min(len(raw) + K, RAW_LIMIT))
            ]
            rows = self.sample_many(context.context_id, arm.name, seeds)
            if len(rows) != len(seeds):
                raise ValueError("decoder must return every requested raw draw")
            for row, s in zip(rows, seeds):
                c = self.policy_candidate(row, arm, s)
                raw.append(c)
                if len(distinct) < K:
                    distinct.setdefault(c.program.program_sha256, c)
        decode_seconds = time.monotonic() - start
        start = time.monotonic()
        members = tuple(distinct.values())
        predictions = self.score_heads(
            context.context_id, arm.name, [c.program.tokens for c in members]
        )
        if len(predictions) != len(members):
            raise ValueError("every slate member requires a head readout")
        by_sha = {
            c.program.program_sha256: dict(p) for c, p in zip(members, predictions)
        }
        raw = tuple(
            replace(c, predicted=by_sha.get(c.program.program_sha256)) for c in raw
        )
        members = tuple(
            replace(c, predicted=by_sha[c.program.program_sha256]) for c in members
        )
        head_seconds = time.monotonic() - start
        start = time.monotonic()
        selected = select(members)
        selection_seconds = time.monotonic() - start
        return PreparedSlate(
            raw, members, selected, decode_seconds, head_seconds, selection_seconds
        )

    def slate(self, context, arm, seed, budget):
        if arm.name.startswith("rerank32:"):
            if budget != BUDGET:
                raise ValueError("rerank execution budget is fixed at eight")
            return self.prepare_slate(context, arm, seed).selected
        return super().slate(context, arm, seed, budget)
