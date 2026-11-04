"""Opt-in train-only first-label waves; no optimizer or teacher admission."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from rlfw.arena.arms import Candidate, draw_seed, pr1_trace_sink
from rlfw.arena.game_spec import (
    R5_KPI_VERSION,
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


CAMPAIGN_WINDOW_SCHEMA = "jope.rlfw-bootstrap-campaign-window/v1"
# Owner ruling R21: the window length is fixed; its start is the owner's approval.
CAMPAIGN_WINDOW_HOURS = 72


def campaign_clock():
    """Read wall time, monotonic time and boot identity together."""
    return {
        "unix": time.time(),
        "monotonic": time.monotonic(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


@dataclass(frozen=True)
class CampaignWindow:
    """Bootstrap campaign window injected from its manifest, never a code constant."""

    created_unix: float
    deadline_unix: float
    window_hours: int
    boot_id: str
    created_monotonic: float

    def __post_init__(self):
        if (
            type(self.window_hours) is not int
            or self.window_hours != CAMPAIGN_WINDOW_HOURS
        ):
            raise ValueError("bootstrap campaign window must last 72 hours")
        clocks = (self.created_unix, self.deadline_unix, self.created_monotonic)
        if (
            any(type(v) not in (int, float) or not math.isfinite(v) for v in clocks)
            or self.created_unix <= 0
            or self.created_monotonic < 0
        ):
            raise ValueError("finite campaign window clocks required")
        if self.deadline_unix != self.created_unix + 3600 * self.window_hours:
            raise ValueError("campaign deadline must close the window at its length")
        if (
            type(self.boot_id) is not str
            or not 0 < len(self.boot_id) <= 64
            or self.boot_id != self.boot_id.strip()
        ):
            raise ValueError("campaign window boot identity required")

    @classmethod
    def opened_now(cls):
        """For the owner-approved launcher only; no CLI path opens or renews a window."""
        now = campaign_clock()
        return cls(
            now["unix"],
            now["unix"] + 3600 * CAMPAIGN_WINDOW_HOURS,
            CAMPAIGN_WINDOW_HOURS,
            now["boot_id"],
            now["monotonic"],
        )

    def to_manifest(self):
        return {"schema": CAMPAIGN_WINDOW_SCHEMA, **asdict(self)}

    @classmethod
    def from_manifest(cls, value):
        if not isinstance(value, dict) or value.get("schema") != CAMPAIGN_WINDOW_SCHEMA:
            raise ValueError("bootstrap campaign window manifest required")
        d = dict(value)
        d.pop("schema")
        if set(d) != {f.name for f in fields(cls)}:
            raise ValueError("campaign window manifest fields changed")
        return cls(**d)

    def require_open(self):
        now = campaign_clock()
        if now["boot_id"] != self.boot_id:
            raise RuntimeError(
                "campaign deadline is unverifiable after a reboot; "
                "a new owner-approved window is required"
            )
        if not (
            self.created_unix <= now["unix"] < self.deadline_unix
            and self.created_monotonic
            <= now["monotonic"]
            < self.created_monotonic + 3600 * self.window_hours
        ):
            raise RuntimeError(
                "bootstrap campaign window is closed at its deadline or not yet open"
            )


from rlfw.no_winner_bootstrap_spec import BootstrapSpec



from rlfw.no_winner_bootstrap_execution import (
    BootstrapScheduler,
    CoverageArms,
    CoverageSlate,
    TrainingOracle,
    coverage_selection,
)



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enable-no-winner-bootstrap", action="store_true")
    p.add_argument("--spec", required=True)
    p.add_argument("--campaign-window")
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
    if args.campaign_window is None:
        raise ValueError("bootstrap requires its injected campaign window manifest")
    from rlfw.bench_cli import verify_device, verify_runtime_scope
    from rlfw.decision_trace import DecisionTrace
    from rlfw.policy_binding import verify_manifest

    spec = BootstrapSpec.from_dict(json.loads(Path(args.spec).read_text()))
    window = CampaignWindow.from_manifest(
        json.loads(local_path(spec.bench_root, args.campaign_window).read_text())
    )
    if window != spec.campaign_window:
        raise ValueError("wave spec must bind the injected campaign window")
    window.require_open()
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
