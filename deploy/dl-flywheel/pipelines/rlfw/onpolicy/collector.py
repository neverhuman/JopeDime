"""Pure round collector. The runner owns durable CAS/checkpoint application."""

from dataclasses import dataclass

from .groups import GroupSpec, ResolvedGroup, VerifiedQ

QUARANTINE = frozenset(
    (
        "infra_unknown",
        "parity_custody",
        "parity_unqualified",
        "qualified_failure",
        "g0_not_ready",
        "fitness_only",
        "predicted",
        "missing_closure",
    )
)


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    groups: tuple[ResolvedGroup, ...]
    quarantined: tuple[tuple[str, tuple[str, ...]], ...]


class RoundCollector:
    def __init__(self, round_id, groups):
        groups = tuple(groups)
        if (
            not round_id
            or not groups
            or not all(isinstance(g, GroupSpec) for g in groups)
        ):
            raise ValueError("predeclared immutable groups required")
        if len({g.group_id for g in groups}) != len(groups):
            raise ValueError("duplicate group ID")
        frozen = {(g.policy, g.generated_update, g.sample_rule) for g in groups}
        if len(frozen) != 1:
            raise ValueError("one frozen behavior checkpoint/update/utility per round")
        self._round_id = round_id
        self._specs = {g.group_id: g for g in groups}
        self._results = {g.group_id: {} for g in groups}
        self._closed = False

    def resolve(self, group_id, draw_id, outcome):
        if self._closed:
            raise ValueError("closed round; no retry or second update")
        spec = self._specs[group_id]
        draw = next((d for d in spec.draws if d.draw_id == draw_id), None)
        if draw is None:
            raise ValueError("unregistered draw")
        if isinstance(outcome, VerifiedQ):
            spec.check_q(outcome)
            if outcome.program != draw.program:
                raise ValueError("wrong draw program")
        elif not isinstance(outcome, str) or outcome not in QUARANTINE:
            raise ValueError("only verified q or a terminal quarantine reason accepted")
        rows = self._results[group_id]
        if draw_id in rows:
            if rows[draw_id] != outcome:
                raise ValueError("terminal outcome cannot be replaced/retried")
            return  # exact duplicate delivery is idempotent
        rows[draw_id] = outcome

    def close(self):
        """Consume once. Pending groups are quarantined whole, never zero-filled."""
        if self._closed:
            raise ValueError("round already consumed; at most one optimizer step")
        ready, quarantined = [], []
        for group_id, spec in self._specs.items():
            rows = self._results[group_id]
            reasons = tuple(
                sorted(
                    {
                        rows.get(d.draw_id, "incomplete")
                        for d in spec.draws
                        if not isinstance(rows.get(d.draw_id), VerifiedQ)
                    }
                )
            )
            if reasons:
                quarantined.append((group_id, reasons))
            else:
                ready.append(
                    ResolvedGroup(spec, tuple(rows[d.draw_id] for d in spec.draws))
                )
        self._closed = True
        return RoundResult(self._round_id, tuple(ready), tuple(quarantined))
