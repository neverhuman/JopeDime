"""Immutable iid groups; no predicted score or failure utility enters a reward."""

import math
from dataclasses import dataclass, field

from rlfw.research_hwm import ARTIFACT_ROLES, HWMKey, Publication, is_sha, sha256
from rlfw.trie_logprob import TrieLogProb

Q_CAP = math.log(1000)
_VERIFIED = object()


@dataclass(frozen=True)
class VerifiedQ:
    comparison: str
    context: str
    parent: str
    program: str
    execution: str
    q_grid: int
    receipt: str
    _proof: object = field(repr=False, compare=False)

    def __post_init__(self):
        if self._proof is not _VERIFIED:
            raise ValueError("use verify_q with fresh role and closure verification")

    @property
    def raw(self):
        return self.q_grid / 1_000_000


def verify_q(key, publication, *, parent, role_check, verify_closure):
    """Adapter boundary: callbacks verify actual role/parent and immutable bodies.

    role_check(key, parent) must freshly resolve TRAIN and parent ancestry.
    verify_closure(key, publication) must verify seven artifact bodies, parity,
    recomputed q_grid and their join to key/program/execution; metadata is not
    sufficient. The pure library performs no IO. Fixtures use synthetic bodies.
    """
    if not isinstance(key, HWMKey) or not isinstance(publication, Publication):
        raise TypeError("pinned comparison and measured publication required")
    p = publication
    if (
        p.source != "measured"
        or p.outcome != "qualified_complete"
        or p.hwm_key != key.digest
        or type(p.q_grid) is not int
        or not is_sha(p.program_sha256)
        or not p.execution_key
        or not isinstance(parent, str)
        or not parent
    ):
        raise ValueError("only verified comparable completed q is a reward")
    if not isinstance(p.artifacts, dict) or set(p.artifacts) != ARTIFACT_ROLES:
        raise ValueError("seven-artifact closure required")
    # Copy input metadata before callbacks so retained identity cannot drift.
    identity = sha256({"key": key.digest, "publication": p.__dict__, "parent": parent})
    if role_check(key, parent) is not True or verify_closure(key, p) is not True:
        raise ValueError("fresh train-role/closure verification failed")
    if identity != sha256(
        {"key": key.digest, "publication": p.__dict__, "parent": parent}
    ):
        raise ValueError("publication changed during verification")
    return VerifiedQ(
        key.digest,
        key.context_identity,
        parent,
        p.program_sha256,
        p.execution_key,
        p.q_grid,
        identity,
        _VERIFIED,
    )


@dataclass(frozen=True)
class UtilityRule:
    """Samples and incumbent MUST use the same rule; ledger q remains raw."""

    mode: str = "clipped"
    cap: float = Q_CAP

    def __post_init__(self):
        if self.mode not in ("raw", "clipped") or self.cap != Q_CAP:
            raise ValueError("raw or registered +/-log(1000) clipping required")

    def apply(self, q):
        if not isinstance(q, VerifiedQ):
            raise TypeError("VerifiedQ required; numeric/predicted utilities refused")
        return q.raw if self.mode == "raw" else max(-self.cap, min(self.cap, q.raw))


@dataclass(frozen=True)
class PolicyIdentity:
    checkpoint: str
    trie: str
    vocabulary: str
    normalizer: str
    protocol: str
    temperature: float = 1.0

    def __post_init__(self):
        if not all(
            is_sha(x)
            for x in (
                self.checkpoint,
                self.trie,
                self.vocabulary,
                self.normalizer,
                self.protocol,
            )
        ):
            raise ValueError("full sampler identity hashes required")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("positive finite temperature required")

    @property
    def support(self):
        return (
            self.trie,
            self.vocabulary,
            self.normalizer,
            self.protocol,
            self.temperature,
        )


@dataclass(frozen=True)
class Draw:
    draw_id: str
    program: str
    behavior: TrieLogProb

    def __post_init__(self):
        if (
            not self.draw_id
            or not is_sha(self.program)
            or not isinstance(self.behavior, TrieLogProb)
        ):
            raise ValueError(
                "raw draw id, program and summed EOS trie likelihood required"
            )


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    parent: str
    context: str
    comparison: str
    policy: PolicyIdentity
    generated_update: int
    generated_seconds: float
    draws: tuple[Draw, ...]
    best_of: int
    incumbent: VerifiedQ | None
    sample_rule: UtilityRule = UtilityRule()
    incumbent_rule: UtilityRule = UtilityRule()
    sampling: str = "iid_with_replacement"

    def __post_init__(self):
        if not all(
            isinstance(s, str) and s for s in (self.group_id, self.parent, self.context)
        ):
            raise ValueError("group/parent/context required")
        if not is_sha(self.comparison) or not isinstance(self.policy, PolicyIdentity):
            raise ValueError("pinned comparison and behavior checkpoint required")
        if type(self.generated_update) is not int or self.generated_update < 0:
            raise ValueError("nonnegative optimizer update required")
        if not math.isfinite(self.generated_seconds) or self.generated_seconds < 0:
            raise ValueError("monotonic generation time required")
        if (
            type(self.draws) is not tuple
            or not self.draws
            or not all(isinstance(d, Draw) for d in self.draws)
        ):
            raise ValueError("immutable, nonempty raw draws required")
        if len({d.draw_id for d in self.draws}) != len(self.draws):
            raise ValueError("draw IDs unique; duplicate PROGRAMS retain multiplicity")
        if self.sampling != "iid_with_replacement":
            raise ValueError("dedup/rerank/archive/mutation groups are L_PI/L6 only")
        if type(self.best_of) is not int or not 1 <= self.best_of <= len(self.draws):
            raise ValueError("1 <= B <= G required")
        if (
            not isinstance(self.sample_rule, UtilityRule)
            or self.sample_rule != self.incumbent_rule
        ):
            raise ValueError("mixed raw/clipped sample/incumbent utilities refused")
        if self.incumbent is not None:
            self.check_q(self.incumbent)

    def check_q(self, q):
        if not isinstance(q, VerifiedQ) or (q.comparison, q.context, q.parent) != (
            self.comparison,
            self.context,
            self.parent,
        ):
            raise ValueError(
                "reward/incumbent is not verified under the frozen comparison"
            )


@dataclass(frozen=True)
class ResolvedGroup:
    spec: GroupSpec
    outcomes: tuple[VerifiedQ, ...]

    def __post_init__(self):
        if not isinstance(self.spec, GroupSpec) or type(self.outcomes) is not tuple:
            raise TypeError("immutable group and outcomes required")
        if len(self.outcomes) != len(self.spec.draws):
            raise ValueError("incomplete groups cannot be shrunk")
        for draw, q in zip(self.spec.draws, self.outcomes):
            self.spec.check_q(q)
            if q.program != draw.program:
                raise ValueError("outcome must join the recorded draw program")

    @property
    def utilities(self):
        return tuple(self.spec.sample_rule.apply(q) for q in self.outcomes)

    @property
    def h(self):
        return (
            None
            if self.spec.incumbent is None
            else self.spec.incumbent_rule.apply(self.spec.incumbent)
        )
