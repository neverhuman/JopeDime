from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .corpus import BroadCorpus


class ExactOnceBatchSampler:
    """Round-robin shuffled dataset queues with randomized length buckets."""

    def __init__(
        self,
        corpus: BroadCorpus,
        *,
        batch_size: int,
        seed: int,
        eligible_rows: Sequence[int] | None = None,
        bucket_buffer: int = 2048,
        domain_mix: Mapping[str, float] | None = None,
    ) -> None:
        if batch_size < 1 or bucket_buffer < batch_size:
            raise ValueError("invalid exact-once batch or bucket-buffer size")
        selected = (
            np.arange(len(corpus), dtype=np.int64)
            if eligible_rows is None
            else np.asarray(eligible_rows, dtype=np.int64)
        )
        if len(selected) != len(set(selected.tolist())):
            raise ValueError("eligible rows contain duplicates")
        if np.any(selected < 0) or np.any(selected >= len(corpus)):
            raise ValueError("eligible row is outside corpus")
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for row in selected.tolist():
            grouped[(int(corpus.metadata[row, 1]), int(corpus.metadata[row, 0]))].append(row)
        codes = {"real": 0, "synthetic": 1, "online": 2}
        mix = dict(domain_mix or {})
        if mix and (
            set(mix) - set(codes)
            or any(
                not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0
                for weight in mix.values()
            )
            or sum(float(weight) for weight in mix.values()) <= 0
        ):
            raise ValueError("domain_mix must contain nonnegative real/synthetic/online weights")
        self.corpus = corpus
        self.rows = grouped
        self.domain_mix = {codes[name]: float(weight) for name, weight in mix.items() if weight > 0}
        self.eligible = selected
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_buffer = bucket_buffer
        self.epoch = 0
        self.last_report: dict[str, Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(len(self.eligible) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        shuffle_seed = self.seed + self.epoch
        rng = np.random.default_rng(shuffle_seed)
        groups = list(self.rows)
        rng.shuffle(groups)
        queues: dict[tuple[int, int], deque[int]] = {}
        for group in groups:
            values = np.asarray(self.rows[group], dtype=np.int64)
            rng.shuffle(values)
            queues[group] = deque(values.tolist())
        active_by_domain: dict[int, deque[tuple[int, int]]] = defaultdict(deque)
        for group in groups:
            active_by_domain[group[0]].append(group)
        weights = self.domain_mix or {domain: 1.0 for domain in sorted(active_by_domain)}
        # Smooth weighted round-robin domains; each domain internally
        # round-robins shuffled dataset queues.
        credit = {domain: 0.0 for domain in set(active_by_domain) | set(weights)}
        stream: list[int] = []
        while any(active_by_domain.values()):
            available = [domain for domain, active in active_by_domain.items() if active]
            total_weight = sum(weights.get(domain, 0.0) for domain in available)
            if total_weight <= 0:
                total_weight = float(len(available))
                local_weights = {domain: 1.0 for domain in available}
            else:
                local_weights = {domain: weights.get(domain, 0.0) for domain in available}
            for domain in available:
                credit[domain] += local_weights[domain]
            domain = max(available, key=lambda value: (credit[value], -value))
            credit[domain] -= total_weight
            group = active_by_domain[domain].popleft()
            stream.append(queues[group].popleft())
            if queues[group]:
                active_by_domain[domain].append(group)

        batches: list[list[int]] = []
        for start in range(0, len(stream), self.bucket_buffer):
            buffer = stream[start : start + self.bucket_buffer]
            buffer.sort(key=lambda row: int(self.corpus.metadata[row, 4]))
            local = [
                buffer[index : index + self.batch_size] for index in range(0, len(buffer), self.batch_size)
            ]
            rng.shuffle(local)
            for batch in local:
                rng.shuffle(batch)
                batches.append(batch)
        rng.shuffle(batches)
        flattened = [row for batch in batches for row in batch]
        unique = len(set(flattened))
        duplicates = len(flattened) - unique
        self.last_report = {
            "schema": "BroadEpochCoverage.v1",
            "epoch": self.epoch,
            "shuffle_seed": shuffle_seed,
            "eligible_rows": len(self.eligible),
            "visited_rows": len(flattened),
            "unique_rows": unique,
            "duplicate_rows": duplicates,
            "coverage": unique / max(len(self.eligible), 1),
            "domain_rows": {
                name: sum(int(self.corpus.metadata[row, 1]) == code for row in flattened)
                for name, code in (
                    ("real", 0),
                    ("synthetic", 1),
                    ("online", 2),
                )
            },
            "target_domain_mix": {
                name: self.domain_mix.get(code, 0.0)
                for name, code in (
                    ("real", 0),
                    ("synthetic", 1),
                    ("online", 2),
                )
            },
        }
        if (
            len(flattened) != len(self.eligible)
            or unique != len(self.eligible)
            or duplicates
            or set(flattened) != set(self.eligible.tolist())
        ):
            raise AssertionError("exact-once epoch coverage invariant failed")
        yield from batches
