"""Parent-first replay (4x25%) and context scheduling (50/30/10/10)."""

import random
from collections import Counter, deque
from dataclasses import dataclass

from .groups import VerifiedQ

REPLAY_BUCKETS = ("measured", "recent_gain", "hard_pair", "exploration")
CONTEXT_POOLS = ("comparable_hwm", "no_winner_original", "augmented", "development")


class SamplingUnavailable(ValueError):
    """Keep the unfilled quota pending; never silently redistribute it."""


class _Quota:
    def __init__(self, shares, rng):
        self._shares, self._rng, self._pending = tuple(shares), rng, deque()

    def peek(self):
        if not self._pending:
            block = list(self._shares)
            self._rng.shuffle(block)
            self._pending.extend(block)
        return self._pending[0]

    def commit(self):
        self._pending.popleft()


def _parent_context_choice(items, rng):
    """Siblings cannot increase a parent's chance of being selected."""
    grouped = {}
    for item in items:
        grouped.setdefault(item.parent, {}).setdefault(item.context, []).append(item)
    if not grouped:
        raise SamplingUnavailable("quota has no eligible parent/context")
    parent = rng.choice(sorted(grouped))
    context = rng.choice(sorted(grouped[parent]))
    return rng.choice(grouped[parent][context])


@dataclass(frozen=True)
class ReplayItem:
    q: VerifiedQ
    buckets: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.q, VerifiedQ):
            raise ValueError("replay rewards require verified train-role q")
        if (
            type(self.buckets) is not tuple
            or not self.buckets
            or not set(self.buckets) <= set(REPLAY_BUCKETS)
        ):
            raise ValueError("registered replay buckets required")
        if len(set(self.buckets)) != len(self.buckets):
            raise ValueError("duplicate bucket membership")

    @property
    def parent(self):
        return self.q.parent

    @property
    def context(self):
        return self.q.context


@dataclass(frozen=True)
class ReplayBatch:
    bucket: str
    parent: str
    items: tuple[ReplayItem, ...]


class ReplaySampler:
    def __init__(self, items, *, seed, rolling_window, max_family_batches):
        self._items = tuple(items)
        if not self._items or not all(isinstance(x, ReplayItem) for x in self._items):
            raise ValueError("measured replay items required")
        identities = [(x.q.comparison, x.q.program, x.q.receipt) for x in self._items]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate replay item; consolidate bucket membership")
        if (
            type(rolling_window) is not int
            or type(max_family_batches) is not int
            or not 1 <= max_family_batches <= rolling_window
        ):
            raise ValueError("explicit rolling window and family exposure cap required")
        self._rng = random.Random(seed)
        self._quota = _Quota(REPLAY_BUCKETS, self._rng)
        self._history = deque(maxlen=rolling_window)
        self._cap = max_family_batches

    def batch(self, size, *, role_check):
        """Select bucket→parent→context→row; a batch has exactly one parent.

        role_check(q) resolves current role/parent on each call. Frozen q
        verification is necessary but cannot authorize a later role change.
        A parent's cap counts batches, preventing sibling volume from bypassing
        it. Size is explicit so exposure in rows is also readily recorded.
        """
        if type(size) is not int or size < 1:
            raise ValueError("positive batch size required")
        bucket = self._quota.peek()
        # The oldest batch leaves the window when the new batch is committed.
        history = list(self._history)
        if len(history) == self._history.maxlen:
            history = history[1:]
        counts = Counter(history)
        eligible = tuple(
            x
            for x in self._items
            if bucket in x.buckets
            and counts[x.parent] < self._cap
            and role_check(x.q) is True
        )
        first = _parent_context_choice(eligible, self._rng)
        siblings = tuple(x for x in eligible if x.parent == first.parent)
        selected = (first,) + tuple(
            _parent_context_choice(siblings, self._rng) for _ in range(size - 1)
        )
        self._quota.commit()
        self._history.append(first.parent)
        return ReplayBatch(bucket, first.parent, selected)


@dataclass(frozen=True)
class ScheduledContext:
    parent: str
    context: str
    pool: str
    role: str
    rights_cleared: bool
    original: bool = False
    incumbent: VerifiedQ | None = None

    def __post_init__(self):
        if not self.parent or not self.context or self.pool not in CONTEXT_POOLS:
            raise ValueError("registered context/parent/pool required")
        expected = "development" if self.pool == "development" else "train"
        if self.role != expected or self.rights_cleared is not True:
            raise ValueError("sealed/rights-pending/wrong-role contexts refused")
        if self.pool == "comparable_hwm":
            if not isinstance(self.incumbent, VerifiedQ) or (
                self.incumbent.parent,
                self.incumbent.context,
            ) != (self.parent, self.context):
                raise ValueError("comparable train incumbent required")
        if self.pool == "no_winner_original" and (
            self.original is not True or self.incumbent is not None
        ):
            raise ValueError("no-winner original with NULL incumbent required")
        if self.pool == "augmented" and self.original is not False:
            raise ValueError("augmented pool requires a variant")

    @property
    def optimizer_allowed(self):
        return self.role == "train"


class ContextScheduler:
    def __init__(self, contexts, *, seed):
        self._contexts = tuple(contexts)
        if not self._contexts or not all(
            isinstance(x, ScheduledContext) for x in self._contexts
        ):
            raise ValueError("explicit context roster required")
        if len({x.context for x in self._contexts}) != len(self._contexts):
            raise ValueError("context cannot appear in multiple pools/roles")
        roles = {}
        for x in self._contexts:
            if x.parent in roles and roles[x.parent] != x.role:
                raise ValueError("parent role overlap between train and development")
            roles[x.parent] = x.role
        self._rng = random.Random(seed)
        self._quota = _Quota(
            [CONTEXT_POOLS[0]] * 5 + [CONTEXT_POOLS[1]] * 3 + list(CONTEXT_POOLS[2:]),
            self._rng,
        )

    def next(self, *, role_check):
        """Caller dispatches development ONLY to evaluation, never optimizer."""
        pool = self._quota.peek()
        eligible = tuple(
            x for x in self._contexts if x.pool == pool and role_check(x) is True
        )
        selected = _parent_context_choice(eligible, self._rng)
        self._quota.commit()
        return selected
