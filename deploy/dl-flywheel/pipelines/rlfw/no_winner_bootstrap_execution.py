"""Coverage-aware dispatch and scheduling for no-winner bootstrap waves."""

import json
import math
import random
from dataclasses import dataclass, replace

from rlfw.arena.arms import Candidate, draw_seed
from rlfw.arena.game_spec import require_sha
from rlfw.arena.oracle import BlindOracle
from rlfw.arena.scheduler import ArenaScheduler
from rlfw.research_hwm import Publication
from rlfw.slate import RerankArms
from rlfw.no_winner_bootstrap import (
    BootstrapSpec,
    CampaignWindow,
    TrainingContext,
    check_training_context,
)


def coverage_selection(slate, probabilities, audit, seed):
    if set(probabilities) != {c.program.program_sha256 for c in slate} or any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probabilities.values()
    ):
        raise ValueError("calibrated P(q>0) required for every distinct slate member")
    if len(slate) < 7:
        return ()
    ranked = sorted(
        slate,
        key=lambda c: (
            -c.predicted["p_complete"] * probabilities[c.program.program_sha256],
            c.program.program_sha256,
        ),
    )
    chosen = [
        replace(
            c,
            propensity=None,
            selection_rule="coverage_exploit_p_complete_times_p_q_positive",
        )
        for c in ranked[:5]
    ]
    remaining = sorted(ranked[5:], key=lambda c: c.program.program_sha256)
    rng = random.Random(seed)
    for _ in range(2):
        index = rng.randrange(len(remaining))
        chosen.append(
            replace(
                remaining[index],
                propensity=1 / len(remaining),
                selection_rule="uniform_remaining_slate",
            )
        )
        remaining.pop(index)
    return tuple(chosen) + (replace(audit, selection_rule="uniform_support_audit"),)


@dataclass(frozen=True)
class CoverageSlate:
    raw_draws: tuple[Candidate, ...]
    slate: tuple[Candidate, ...]
    selected: tuple[Candidate, ...]
    predicted_probabilities: dict
    calibration_sha256: str
    selection_seed: int
    decode_seconds: float
    head_seconds: float
    selection_seconds: float

    def __post_init__(self):
        require_sha(self.calibration_sha256, "coverage calibration")
        if not 2 <= len(self.raw_draws) <= 513:
            raise ValueError("bounded neural packet plus one uniform audit required")
        first = {}
        for c in self.raw_draws[:-1]:
            if (
                c.selection_rule != "iid_trie_sample"
                or c.total_logprob is None
                or c.batch_protocol_sha256 is None
            ):
                raise ValueError("exact neural raw draws required")
            if len(first) < 32:
                first.setdefault(c.program.program_sha256, c)
        if (
            tuple(first.values()) != self.slate
            or len(self.slate) < 32
            and len(self.raw_draws) != 513
        ):
            raise ValueError("first32 slate or preserved512-draw shortfall required")
        audit = self.raw_draws[-1]
        if (
            audit.selection_rule != "iid_uniform_whole_program"
            or audit.predicted is None
        ):
            raise ValueError(
                "uniform whole-support audit with predicted heads required"
            )
        if (
            coverage_selection(
                self.slate, self.predicted_probabilities, audit, self.selection_seed
            )
            != self.selected
        ):
            raise ValueError("mixed selection changed")
        if any(
            not math.isfinite(t) or t < 0
            for t in (self.decode_seconds, self.head_seconds, self.selection_seconds)
        ):
            raise ValueError("finite phase times required")

    def to_dict(self):
        return {
            "schema": "jope.rlfw-coverage-slate/v1",
            "raw_draws": [c.to_dict() for c in self.raw_draws],
            "slate": [c.to_dict() for c in self.slate],
            "selected": [c.to_dict() for c in self.selected],
            "predicted": {"p_q_positive": self.predicted_probabilities},
            "calibration_sha256": self.calibration_sha256,
            "selection_seed": self.selection_seed,
            "shortfall": 32 - len(self.slate),
            "decode_seconds": self.decode_seconds,
            "head_seconds": self.head_seconds,
            "selection_seconds": self.selection_seconds,
        }

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        if d.pop("schema") != "jope.rlfw-coverage-slate/v1":
            raise ValueError("incompatible coverage slate")
        shortfall = d.pop("shortfall")
        predicted = d.pop("predicted")
        if set(predicted) != {"p_q_positive"}:
            raise ValueError("only dedicated coverage probabilities accepted")
        d["predicted_probabilities"] = predicted["p_q_positive"]
        for field in ("raw_draws", "slate", "selected"):
            d[field] = tuple(Candidate.from_dict(c) for c in d[field])
        result = cls(**d)
        if shortfall != 32 - len(result.slate):
            raise ValueError("coverage shortfall changed")
        return result


class CoverageArms(RerankArms):
    def __init__(
        self,
        catalog,
        sample,
        sample_many,
        score_heads,
        probabilities,
        calibration_sha256,
    ):
        super().__init__(catalog, {}, sample, sample_many, score_heads)
        require_sha(calibration_sha256, "calibration")
        self.probabilities, self.calibration_sha256 = probabilities, calibration_sha256

    def bind_window(self, window):
        """Recheck the whole campaign window immediately before every model dispatch."""

        def guarded(call):
            def dispatch(*args, **kwargs):
                window.require_open()
                return call(*args, **kwargs)

            return dispatch

        for name in ("sample", "sample_many", "score_heads", "probabilities"):
            setattr(self, name, guarded(getattr(self, name)))

    def prepare_slate(self, context, arm, seed):
        import time

        # The neural packet retains its actual mixed-arm seeds and support pins.
        neural = super().prepare_neural_slate(context, arm, seed)
        start = time.monotonic()
        probabilities = self.probabilities(
            context.context_id,
            arm.name,
            [c.program.tokens for c in neural.slate],
            self.calibration_sha256,
        )
        if len(probabilities) != len(neural.slate):
            raise ValueError("one qualified coverage probability per slate member")
        predicted = {
            c.program.program_sha256: p for c, p in zip(neural.slate, probabilities)
        }
        audit_seed = draw_seed(
            context.context_id, arm.name, seed, len(neural.raw_draws) + 1
        )
        audit_program = self.catalog[
            random.Random(audit_seed).randrange(len(self.catalog))
        ]
        audit_heads = self.score_heads(
            context.context_id, arm.name, [audit_program.tokens]
        )
        if len(audit_heads) != 1:
            raise ValueError("audit requires one complete head readout")
        per_token = self.uniform_logprob(audit_program.tokens)
        audit = Candidate(
            audit_program,
            audit_seed,
            math.fsum(per_token),
            per_token,
            1 / len(self.catalog),
            "iid_uniform_whole_program",
            predicted=audit_heads[0],
        )
        head_seconds = neural.head_seconds + time.monotonic() - start
        start = time.monotonic()
        selection_seed = draw_seed(context.context_id, arm.name, seed, 0)
        selected = coverage_selection(neural.slate, predicted, audit, selection_seed)
        return CoverageSlate(
            neural.raw_draws + (audit,),
            neural.slate,
            selected,
            predicted,
            self.calibration_sha256,
            selection_seed,
            neural.decode_seconds,
            head_seconds,
            time.monotonic() - start,
        )


class TrainingOracle(BlindOracle):
    def check_role(self, context):
        if context not in self.spec.contexts:
            raise ValueError("context outside frozen train wave")
        check_training_context(context, self.role_check, self.spec.role_map_sha256)

    def _resolve(self, *args, **kwargs):
        self.spec.campaign_window.require_open()
        return super()._resolve(*args, **kwargs)

    def before_attempt(self, context):
        # Before every reservation and again before each evaluator request it
        # authorizes, including retries in one slot.
        self.spec.campaign_window.require_open()


class BootstrapScheduler(ArenaScheduler):
    def __init__(
        self, spec, arms, oracle, sink, ledger, admission, queue_empty, *, workers=4
    ):
        self.ledger, self.admission, self.queue_empty = ledger, admission, queue_empty
        self.context_by_id = {c.context_id: c for c in spec.contexts}
        arms.bind_window(spec.campaign_window)
        super().__init__(spec, arms, oracle, sink, workers=workers)

    def prepares(self, arm):
        return arm.name.startswith("rerank_mix:")

    def load_packet(self, value):
        return CoverageSlate.from_dict(value)

    def _run_context(self, context):
        # No proposal, admission or execution once the injected window has closed.
        self.spec.campaign_window.require_open()
        with self.admission() as granted:
            if granted is not True or self.queue_empty() is not True:
                raise RuntimeError("bootstrap requires reserved empty arena queue")
            # An admission wait never carries an earlier check across the boundary.
            self.spec.campaign_window.require_open()
            super()._run_context(context)

    def commit(self, row, key, result):
        context = self.context_by_id[row.context_id]

        def fresh_role(_):
            self.oracle.check_role(context)
            return True

        event = self.ledger.publish(
            context.hwm_key,
            Publication(
                context.hwm_key.digest,
                row.program_sha256,
                key,
                row.censor_reason if row.outcome == "censored" else row.outcome,
                row.q_grid,
                result.get("artifacts"),
            ),
            role_check=fresh_role,
        )
        if row.qualified and event["event"] == "refused":
            raise ValueError("unqualified closure cannot become a first label")
        super().commit(row, key, result)

    def coverage_report(self):
        attempted = {
            c
            for (c,) in self.db.execute("SELECT DISTINCT context FROM prepared_slates")
        }
        rows = self.rows()
        valid, usable = set(), set()
        for cid in attempted:
            measured = {
                r.program_sha256: r.q_grid
                for r in rows
                if r.context_id == cid and r.qualified
            }
            if measured:
                valid.add(cid)
            if (
                len(measured) >= 2
                and max(measured.values()) - min(measured.values()) > 10_000
            ):
                usable.add(cid)
        parents = {self.context_by_id[c].lineage_id for c in valid}
        unfinished = [
            json.loads(result) if result is not None else None
            for (result,) in self.db.execute(
                "SELECT a.result FROM attempts a LEFT JOIN slots s ON a.context=s.context AND a.arm=s.arm AND a.seed=s.seed AND a.slot=s.slot WHERE s.context IS NULL"
            )
        ]
        unknown_costs = sum(r.unknown_cost_attempts for r in rows) + sum(
            r is None or r.get("cost_unknown", False) for r in unfinished
        )
        extra_wall = sum(r["wall_seconds"] for r in unfinished if r is not None)
        extra_cpu = sum(r["cpu_seconds"] for r in unfinished if r is not None)
        return {
            "schema": "jope.rlfw-bootstrap-yield/v1",
            "spec_sha256": self.spec.sha256,
            "contexts_attempted": len(attempted),
            "contexts_with_valid_program": len(valid),
            "contexts_exit_usable": len(usable),
            "first_label_yield": len(valid) / len(attempted) if attempted else None,
            "exit_usable_yield": len(usable) / len(attempted) if attempted else None,
            "new_parent_families": sorted(
                parents - set(self.spec.previously_labeled_parents)
            ),
            "new_parent_families_covered": len(
                parents - set(self.spec.previously_labeled_parents)
            ),
            "censored_logical_slots": sum(r.outcome == "censored" for r in rows),
            "parity_unqualified_logical_slots": sum(
                r.outcome == "parity_unqualified" for r in rows
            ),
            "unfinished_contexts": sorted(attempted - set(self.closed_contexts())),
            "closed_contexts": self.closed_contexts(),
            "unattempted_contexts_unknown": sorted(set(self.context_by_id) - attempted),
            "evaluator_wall_seconds": None
            if unknown_costs
            else sum(r.wall_seconds for r in rows) + extra_wall,
            "evaluator_cpu_seconds": None
            if unknown_costs
            else sum(r.cpu_seconds for r in rows) + extra_cpu,
            "observed_wall_seconds": sum(r.wall_seconds for r in rows) + extra_wall,
            "observed_cpu_seconds": sum(r.cpu_seconds for r in rows) + extra_cpu,
            "unknown_cost_attempts": unknown_costs,
            "optimizer_updates": 0,
            "teacher_admissions": 0,
        }
