"""Opt-in train-only first-label waves; no optimizer or teacher admission."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rlfw.arena.arms import Candidate, draw_seed, pr1_trace_sink
from rlfw.arena.game_spec import (
    ArmSpec,
    GameSpec,
    canonical,
    digest,
    local_path,
    require_sha,
)
from rlfw.arena.oracle import BlindOracle
from rlfw.arena.scheduler import ArenaScheduler
from rlfw.research_hwm import HWMKey, Publication, ResearchHWM
from rlfw.slate import RerankArms


@dataclass(frozen=True)
class TrainingContext:
    context_id: str
    view_id: str
    dataset_hash: str
    cut_id: str
    fold: str | int
    parent_lineage_sha256: str
    identity_sha256: str
    vector_sha256: str
    prepared_input_sha256: str
    hwm_key: HWMKey
    winner_status: str
    role: str = "train"
    excluded: bool = False
    sealed: bool = False
    rights_pending: bool = False

    def __post_init__(self):
        if type(self.fold) not in (str, int) or (
            type(self.fold) is int and self.fold < 0
        ):
            raise ValueError("original native fold type required; booleans refused")
        if self.role != "train" or any(
            v is not False for v in (self.excluded, self.sealed, self.rights_pending)
        ):
            raise ValueError(
                "only cleared, unsealed, unexcluded train originals are eligible"
            )
        if self.winner_status not in ("g0_not_ready", "no_published_hwm"):
            raise ValueError("no-winner status required; it is never a numeric score")
        if any(
            v is None or v == ""
            for v in (
                self.context_id,
                self.view_id,
                self.dataset_hash,
                self.cut_id,
                self.fold,
                self.parent_lineage_sha256,
            )
        ):
            raise ValueError("complete original native identity required")
        for field in ("identity_sha256", "vector_sha256", "prepared_input_sha256"):
            require_sha(getattr(self, field), field)
        if not isinstance(self.hwm_key, HWMKey) or (
            self.hwm_key.context_identity != self.identity_sha256
            or self.hwm_key.dataset_version != self.dataset_hash
            or self.hwm_key.cut != self.cut_id
            or self.hwm_key.prepared_input_digest != self.prepared_input_sha256
        ):
            raise ValueError("research key must bind the exact original input closure")

    @property
    def lineage_id(self):
        return self.parent_lineage_sha256

    @property
    def hwm_program_sha256(self):
        return None

    def authority_value(self, role_map_sha256):
        value = asdict(self)
        for key in ("hwm_key", "winner_status"):
            value.pop(key)
        return {
            **value,
            "role_map_sha256": role_map_sha256,
            "inputs_verified": True,
            "vector_verified": True,
        }


def check_training_context(context, role_check, role_map_sha256):
    fresh = role_check(context.context_id)
    if not isinstance(fresh, dict) or canonical(fresh) != canonical(
        context.authority_value(role_map_sha256)
    ):
        raise ValueError("fresh original train-role/input/vector authority changed")


def select_wave(contexts, role_check, role_map_sha256):
    """Seeded, score-free selection of at most one original per parent."""
    if any(not isinstance(c, TrainingContext) for c in contexts):
        raise TypeError("verified original train context descriptors required")
    if len({c.context_id for c in contexts}) != len(contexts):
        raise ValueError("duplicate original context")
    order = sorted(contexts, key=lambda c: c.context_id)
    random.Random(20261009).shuffle(order)
    selected, parents = [], set()
    for context in order:
        check_training_context(context, role_check, role_map_sha256)
        if context.lineage_id not in parents:
            selected.append(context)
            parents.add(context.lineage_id)
        if len(selected) == 100:
            break
    return tuple(selected)


@dataclass(frozen=True)
class BootstrapSpec:
    game_id: str
    bench_root: str
    out_dir: str
    state_path: str
    contexts: tuple[TrainingContext, ...]
    arms: tuple[ArmSpec, ...]
    role_map_sha256: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    calibration_sha256: str
    runtime_binding_sha256: str
    policy_code_manifest_sha256: str
    policy_qualification_sha256: str
    runtime_files: tuple[tuple[str, str], ...]
    deadline_unix: float
    previously_labeled_parents: tuple[str, ...] = ()
    cache_snapshot_sha256: str = digest({})
    protocol: str = "jope.flywheel-s1-inner-rmse/v1"
    seeds: tuple[int, ...] = (20261009,)
    budget: int = 8
    slate_k: int = 32
    tau: float = 0.01
    order_seed: int = 20261009
    schedule: str = "shuffle"
    device: str = "cpu"
    device_parity_sha256: str | None = None
    operator: str = "coverage_5_2_1"

    def __post_init__(self):
        if (
            not isinstance(self.contexts, tuple)
            or not 1 <= len(self.contexts) <= 100
            or any(not isinstance(c, TrainingContext) for c in self.contexts)
        ):
            raise ValueError("wave requires 1..100 original train contexts")
        if len({c.context_id for c in self.contexts}) != len(self.contexts) or len(
            {c.lineage_id for c in self.contexts}
        ) != len(self.contexts):
            raise ValueError("one context per original parent per wave")
        if (
            not isinstance(self.arms, tuple)
            or len(self.arms) != 1
            or not self.arms[0].name.startswith("rerank_mix:")
            or self.arms[0].temperature != 1.0
        ):
            raise ValueError("one fixed-temperature mixed coverage arm required")
        if (
            self.seeds,
            self.budget,
            self.slate_k,
            self.tau,
            self.order_seed,
            self.operator,
        ) != ((20261009,), 8, 32, 0.01, 20261009, "coverage_5_2_1"):
            raise ValueError("preregistered wave settings cannot change")
        if (
            self.device not in ("cpu", "cuda")
            or not math.isfinite(self.deadline_unix)
            or self.deadline_unix != 1791550800.0
        ):
            raise ValueError("explicit device and fixed deadline required")
        if self.cache_snapshot_sha256 != digest({}):
            raise ValueError("bootstrap starts with the frozen empty research cache")
        for field in (
            "role_map_sha256",
            "evaluator_sha256",
            "canonicalizer_sha256",
            "calibration_sha256",
            "runtime_binding_sha256",
            "policy_code_manifest_sha256",
            "policy_qualification_sha256",
        ):
            require_sha(getattr(self, field), field)
        if (
            not isinstance(self.runtime_files, tuple)
            or len(dict(self.runtime_files)) != len(self.runtime_files)
            or not self.runtime_files
        ):
            raise ValueError("distinct immutable runtime byte pins required")
        pins = dict(self.runtime_files)
        if (
            self.calibration_sha256 not in pins.values()
            or self.runtime_binding_sha256 not in pins.values()
        ):
            raise ValueError(
                "binding and coverage calibration must be sealed runtime dependencies"
            )
        for name, pin in self.runtime_files:
            if Path(name).is_absolute():
                raise ValueError("runtime files are benchmark-relative")
            local_path(self.bench_root, Path(self.bench_root) / name)
            require_sha(pin, name)
        if (
            self.schedule != "shuffle"
            or self.protocol != "jope.flywheel-s1-inner-rmse/v1"
            or not isinstance(self.previously_labeled_parents, tuple)
        ):
            raise ValueError(
                "fixed native protocol, seeded schedule and immutable prior-parent inventory required"
            )
        local_path(self.bench_root, self.out_dir, "runs")
        local_path(self.bench_root, self.state_path, "state")
        for c in self.contexts:
            key = c.hwm_key
            if (
                key.evaluator_sha256,
                key.canonicalizer_sha256,
                key.protocol,
                key.vocabulary_sha256,
                key.trie_sha256,
            ) != (
                self.evaluator_sha256,
                self.canonicalizer_sha256,
                self.protocol,
                self.arms[0].tokenizer_sha256,
                self.arms[0].trie_sha256,
            ):
                raise ValueError("wave/research comparison identity mismatch")

    def to_dict(self):
        return {"schema": "jope.rlfw-bootstrap-spec/v1", **asdict(self)}

    @property
    def sha256(self):
        return digest(self.to_dict())

    # Reuse byte-custody operations; no development GameSpec is constructed.
    freeze = GameSpec.freeze
    verify_frozen = GameSpec.verify_frozen

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        if d.pop("schema") != "jope.rlfw-bootstrap-spec/v1":
            raise ValueError("incompatible bootstrap schema")
        contexts = []
        for item in d["contexts"]:
            c = dict(item)
            key = dict(c["hwm_key"])
            key["inner_folds"] = tuple(key["inner_folds"])
            c["hwm_key"] = HWMKey(**key)
            contexts.append(TrainingContext(**c))
        d["contexts"] = tuple(contexts)
        d["arms"] = tuple(ArmSpec(**a) for a in d["arms"])
        for field in ("seeds", "previously_labeled_parents"):
            d[field] = tuple(d[field])
        d["runtime_files"] = tuple(tuple(p) for p in d["runtime_files"])
        return cls(**d)


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


class BootstrapScheduler(ArenaScheduler):
    def __init__(
        self, spec, arms, oracle, sink, ledger, admission, queue_empty, *, workers=4
    ):
        self.ledger, self.admission, self.queue_empty = ledger, admission, queue_empty
        self.context_by_id = {c.context_id: c for c in spec.contexts}
        super().__init__(spec, arms, oracle, sink, workers=workers)

    def prepares(self, arm):
        return arm.name.startswith("rerank_mix:")

    def load_packet(self, value):
        return CoverageSlate.from_dict(value)

    def _run_context(self, context):
        with self.admission() as granted:
            if granted is not True or self.queue_empty() is not True:
                raise RuntimeError("bootstrap requires reserved empty arena queue")
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
                row.unknown_reason if row.outcome == "censored" else row.outcome,
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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enable-no-winner-bootstrap", action="store_true")
    p.add_argument("--spec", required=True)
    p.add_argument("--runtime-binding", required=True)
    p.add_argument("--policy-code-root", required=True)
    p.add_argument("--policy-qualification", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--device-parity")
    p.add_argument("--pool-workers", type=int, default=4)
    p.add_argument("--max-contexts", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.enable_no_winner_bootstrap:
        raise ValueError("no-winner bootstrap defaults to off")
    from rlfw.bench_cli import verify_device, verify_runtime_scope
    from rlfw.decision_trace import DecisionTrace
    from rlfw.policy_binding import verify_manifest

    spec = BootstrapSpec.from_dict(json.loads(Path(args.spec).read_text()))
    verify_runtime_scope(spec)
    verify_device(spec, args.device, args.device_parity)
    verify_manifest(
        local_path(spec.bench_root, args.policy_code_root),
        expected_sha256=spec.policy_code_manifest_sha256,
    )
    qpath = local_path(spec.bench_root, args.policy_qualification)
    q = json.loads(qpath.read_text())
    if (
        hashlib.sha256(qpath.read_bytes()).hexdigest()
        != spec.policy_qualification_sha256
        or q.get("passed") is not True
        or q.get("draws", 0) < 1000
        or q.get("determinism_passed") is not True
        or not 0 <= q.get("maximum_logprob_error", float("inf")) <= 1e-6
        or spec.arms[0].checkpoint_sha256
        not in q.get("checkpoint_parameter_sha256s", ())
    ):
        raise ValueError("same-pin 1000-draw policy qualification required")
    calibration_paths = [
        name for name, pin in spec.runtime_files if pin == spec.calibration_sha256
    ]
    if len(calibration_paths) != 1:
        raise ValueError("one sealed coverage-calibration artifact required")
    calibration_path = local_path(
        spec.bench_root, Path(spec.bench_root) / calibration_paths[0]
    )
    calibration = json.loads(calibration_path.read_text())
    if hashlib.sha256(
        calibration_path.read_bytes()
    ).hexdigest() != spec.calibration_sha256 or (
        calibration.get("schema") != "jope.rlfw-q-positive-calibration/v1"
        or calibration.get("passed") is not True
        or calibration.get("partition") != "parent-held-train"
        or calibration.get("checkpoint_parameter_sha256")
        != spec.arms[0].checkpoint_sha256
        or calibration.get("trie_sha256") != spec.arms[0].trie_sha256
        or calibration.get("normalizer_sha256") != spec.arms[0].normalizer_sha256
        or calibration.get("tokenizer_sha256") != spec.arms[0].tokenizer_sha256
    ):
        raise ValueError("qualified same-pin parent-held train calibration required")
    spec.freeze()
    spec.verify_frozen()
    path = local_path(spec.bench_root, args.runtime_binding)
    if (
        hashlib.sha256(path.read_bytes()).hexdigest() != spec.runtime_binding_sha256
        or dict(spec.runtime_files).get(
            str(path.relative_to(Path(spec.bench_root).resolve()))
        )
        != spec.runtime_binding_sha256
    ):
        raise ValueError("bootstrap binding must match sealed runtime bytes")
    definition = importlib.util.spec_from_file_location(
        "rlfw_bootstrap_" + spec.sha256, path
    )
    module = importlib.util.module_from_spec(definition)
    sys.modules[definition.name] = module
    definition.loader.exec_module(module)
    runtime = module.build_bootstrap(spec, device=args.device)
    if len(runtime["catalog"]) != 6640:
        raise ValueError(
            "live uniform audit requires the complete 6640-program support"
        )
    oracle = TrainingOracle(spec, runtime["evaluate"], runtime["role_check"], {})
    original_evaluate = oracle.evaluate

    def evaluate(request):
        if runtime["arena_queue_empty"]() is not True:
            raise RuntimeError("priority arena work arrived; yield bootstrap")
        return original_evaluate(request)

    oracle.evaluate = evaluate
    arms = CoverageArms(
        runtime["catalog"],
        runtime["sample"],
        runtime["sample_many"],
        runtime["score_heads"],
        runtime["coverage_probabilities"],
        spec.calibration_sha256,
    )
    ledger = ResearchHWM(
        Path(spec.bench_root) / "state/ledger.sqlite",
        state_root=Path(spec.bench_root) / "state",
    )
    trace = DecisionTrace(spec.out_dir, runs_root=Path(spec.bench_root) / "runs")
    scheduler = BootstrapScheduler(
        spec,
        arms,
        oracle,
        lambda value: pr1_trace_sink(trace, value),
        ledger,
        runtime["reserve_empty_arena"],
        runtime["arena_queue_empty"],
        workers=args.pool_workers,
    )
    try:
        scheduler.run(max_contexts=args.max_contexts)
        if len(scheduler.closed_contexts()) == len(spec.contexts):
            trace.seal()
    finally:
        report = scheduler.coverage_report()
        (Path(spec.out_dir) / "coverage-report.json").write_text(
            canonical(report) + "\n"
        )
        scheduler.close()
        if runtime.get("close"):
            runtime["close"]()
    print(canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
