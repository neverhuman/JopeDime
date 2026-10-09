"""Precommitted iid whole-program draws; no measured scores reach arm code."""

import hashlib
import math
import random
from bisect import bisect_left
from dataclasses import dataclass

from .game_spec import digest, require_sha


@dataclass(frozen=True)
class Program:
    dsl: str
    program_sha256: str
    tokens: tuple[int, ...]

    def __post_init__(self):
        require_sha(self.program_sha256, "program")
        if (
            not self.dsl
            or hashlib.sha256(self.dsl.encode()).hexdigest() != self.program_sha256
        ):
            raise ValueError("program bytes do not match identity")
        if (
            not isinstance(self.tokens, tuple)
            or len(self.tokens) < 2
            or any(type(t) is not int or t < 0 for t in self.tokens)
        ):
            raise ValueError("complete immutable BOS/EOS token sequence required")
        if (
            self.tokens[0] != 1
            or self.tokens[-1] != 2
            or 1 in self.tokens[1:]
            or 2 in self.tokens[:-1]
            or 0 in self.tokens
        ):
            raise ValueError("exact deployed BOS/EOS boundaries required")

    @classmethod
    def from_dict(cls, value):
        if set(value) - {"dsl", "program_sha256", "tokens"}:
            raise ValueError("program descriptors cannot carry scores or predictions")
        return cls(value["dsl"], value["program_sha256"], tuple(value["tokens"]))

    def to_dict(self):
        return {
            "dsl": self.dsl,
            "program_sha256": self.program_sha256,
            "tokens": list(self.tokens),
        }


@dataclass(frozen=True)
class Candidate:
    program: Program
    draw_seed: int
    total_logprob: float | None
    per_token_logprob: tuple[float, ...]
    propensity: float | None
    selection_rule: str
    batch_protocol_sha256: str | None = None
    predicted: dict | None = None

    def __post_init__(self):
        if type(self.draw_seed) is not int or not 0 <= self.draw_seed < 2**63:
            raise ValueError("bounded integer draw seed required")
        if self.predicted is not None:
            if set(self.predicted) != {"q", "p_complete", "runtime_seconds"} or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in self.predicted.values()
            ):
                raise ValueError("complete finite predicted head fields required")
            if (
                not 0 <= self.predicted["p_complete"] <= 1
                or self.predicted["runtime_seconds"] <= 0
            ):
                raise ValueError("invalid predicted completion or runtime")
            object.__setattr__(self, "predicted", dict(self.predicted))
        if self.total_logprob is not None:
            if len(self.per_token_logprob) != len(self.program.tokens) - 1:
                raise ValueError("EOS-inclusive per-token trace required")
            if any(not math.isfinite(p) or p > 0 for p in self.per_token_logprob):
                raise ValueError("invalid supported log-probability")
            if (
                not math.isfinite(self.total_logprob)
                or abs(math.fsum(self.per_token_logprob) - self.total_logprob) > 1e-12
            ):
                raise ValueError("total must sum trie token likelihoods")
        if self.propensity is not None and not 0 < self.propensity <= 1:
            raise ValueError("unknown propensity must be null")
        if self.batch_protocol_sha256 is not None:
            require_sha(self.batch_protocol_sha256, "decoder batch protocol")

    def to_dict(self):
        return {
            "program": self.program.to_dict(),
            "draw_seed": self.draw_seed,
            "total_logprob": self.total_logprob,
            "per_token_logprob": list(self.per_token_logprob),
            "propensity": self.propensity,
            "selection_rule": self.selection_rule,
            "batch_protocol_sha256": self.batch_protocol_sha256,
            "predicted": self.predicted,
        }

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        d["program"] = Program.from_dict(d["program"])
        d["per_token_logprob"] = tuple(d["per_token_logprob"])
        return cls(**d)


def draw_seed(context_id, arm, seed, slot):
    return int(digest([context_id, arm, seed, slot])[:16], 16) % (2**63)


class BaselineArms:
    def __init__(self, catalog, hwm_programs, sample):
        self.catalog = tuple(
            sorted(
                (Program.from_dict(p) for p in catalog), key=lambda p: p.program_sha256
            )
        )
        if not self.catalog or len({p.program_sha256 for p in self.catalog}) != len(
            self.catalog
        ):
            raise ValueError("distinct complete whole-program support required")
        if len({p.tokens for p in self.catalog}) != len(self.catalog):
            raise ValueError("one exact program per complete token sequence")
        self.by_sha = {p.program_sha256: p for p in self.catalog}
        self.hwm_programs = {
            key: Program.from_dict(p) for key, p in hwm_programs.items()
        }
        self.sample = sample
        self.sorted_tokens = sorted(p.tokens for p in self.catalog)
        self.sentinel = max(t for p in self.catalog for t in p.tokens) + 1

    def policy_candidate(self, row, arm, seed):
        p = Program.from_dict({**row["program"], "tokens": row["tokens"]})
        if self.by_sha.get(p.program_sha256) != p:
            raise ValueError("sampled program is not an exact catalog leaf")
        if (
            row.get("trie_sha256") != arm.trie_sha256
            or row.get("checkpoint_parameter_sha256") != arm.checkpoint_sha256
            or row.get("decoder_seed", seed) != seed
        ):
            raise ValueError("sample checkpoint/support/seed identity mismatch")
        if (
            row.get("logprob_support") != "trie_children"
            or row.get("logprob_aggregation") != "sum_including_eos"
        ):
            raise ValueError("grammar or mean-token likelihood refused")
        total = row["total_logprob"]
        return Candidate(
            p,
            seed,
            total,
            tuple(row["per_token_logprob"]),
            math.exp(total) or None,
            "iid_trie_sample",
            row.get("batch_protocol_sha256"),
        )

    def uniform_logprob(self, tokens):
        # Lexical prefix ranges count whole leaves, avoiding a second large trie.
        previous = len(self.catalog)
        values = []
        for position in range(2, len(tokens) + 1):
            prefix = tokens[:position]
            count = bisect_left(
                self.sorted_tokens, prefix + (self.sentinel,)
            ) - bisect_left(self.sorted_tokens, prefix)
            if not count or count > previous:
                raise ValueError("catalog prefix is not a leaf path")
            values.append(math.log(count / previous) if count != previous else 0.0)
            previous = count
        return tuple(values)

    def slate(self, context, arm, seed, budget):
        if arm.name == "hwm":
            p = self.hwm_programs.get(context.context_id)
            if p is None:
                if context.hwm_program_sha256 is not None:
                    raise ValueError("registered original incumbent is missing")
                return ()
            if p.program_sha256 != context.hwm_program_sha256:
                raise ValueError("original incumbent identity changed")
            return (
                Candidate(
                    p,
                    draw_seed(context.context_id, arm.name, seed, 1),
                    None,
                    (),
                    None,
                    "original_incumbent",
                ),
            )
        candidates = []
        for slot in range(1, budget + 1):
            s = draw_seed(context.context_id, arm.name, seed, slot)
            if arm.name == "uniform_trie":
                p = self.catalog[random.Random(s).randrange(len(self.catalog))]
                per_token = self.uniform_logprob(p.tokens)
                c = Candidate(
                    p,
                    s,
                    math.fsum(per_token),
                    per_token,
                    1 / len(self.catalog),
                    "iid_uniform_whole_program",
                )
            elif arm.name.startswith("trie_sample:"):
                row = self.sample(context.context_id, arm.name, s)
                c = self.policy_candidate(row, arm, s)
            else:
                raise ValueError(f"unsupported arena arm: {arm.name}")
            candidates.append(c)
        return tuple(candidates)


def decision_value(spec, context, arm, seed, slot, candidate, kind):
    if candidate.total_logprob is None:
        raise ValueError("incumbent has no sampler propensity; use comparator receipt")
    group = digest([spec.sha256, context.context_id, arm.name, seed])
    return {
        "schema": "jope.rlfw-decision/v2",
        "decision_id": digest([group, slot, kind]),
        "kind": kind,
        "context_identity": context.identity_sha256,
        "checkpoint_parameter_sha256": arm.checkpoint_sha256 or "0" * 64,
        "trie_sha256": arm.trie_sha256,
        "normalizer_sha256": arm.normalizer_sha256,
        "tokenizer_sha256": arm.tokenizer_sha256,
        "temperature": arm.temperature,
        "seed": candidate.draw_seed,
        "tokens": list(candidate.program.tokens),
        "per_token_logprob": list(candidate.per_token_logprob),
        "total_logprob": candidate.total_logprob,
        "logprob_support": "trie_children",
        "logprob_aggregation": "sum_including_eos",
        "dedup_group": candidate.program.program_sha256,
        "slate_id": group,
        "selection_rule": candidate.selection_rule,
        "propensity": candidate.propensity,
        "predicted": dict(candidate.predicted or {}),
        "execution_key": None,
        "outcome_class": None,
        "game_spec_sha256": spec.sha256,
        "batch_protocol_sha256": candidate.batch_protocol_sha256,
    }


def pr1_trace_sink(trace, value):
    """Lazy PR1 integration; no deployed continuous package at import time."""
    from rlfw.decision_trace import DecisionRecord
    from rlfw.trie_logprob import TrieLogProb

    d = dict(value)
    d.pop("schema")
    d.pop("logprob_support")
    d.pop("logprob_aggregation")
    d["logprob"] = TrieLogProb(
        d.pop("total_logprob"), tuple(d.pop("per_token_logprob"))
    )
    d["tokens"] = tuple(d["tokens"])
    trace.append(DecisionRecord(**d))
