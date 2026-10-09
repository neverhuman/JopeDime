"""Explicit research-only arena and paired comparison commands."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

from .arena.arms import BaselineArms, pr1_trace_sink
from .arena.game_spec import GameSpec, canonical, local_path
from .arena.kpi import LABEL, context_units, paired_difference
from .arena.oracle import BlindOracle
from .arena.report import read_run, write_reports
from .arena.scheduler import ArenaScheduler


def verify_runtime_scope(spec):
    """Real executions require the campaign's separate, bounded user scope."""
    if spec.deadline_unix is None or time.time() >= spec.deadline_unix:
        raise ValueError("live arena requires its unexpired fixed campaign deadline")
    state = Path(spec.bench_root) / "state"
    if (
        not (state / "health-baseline.json").exists()
        or (state / "bench-health-hold.json").exists()
        or (state / "STOP.json").exists()
    ):
        raise ValueError("benchmark health baseline or guard is not qualified")
    entries = Path("/proc/self/cgroup").read_text().splitlines()
    unified = next(
        (line.split(":", 2)[2] for line in entries if line.startswith("0::")), None
    )
    if unified is None or not any(
        "rlfw" in p and p.endswith(".scope") for p in Path(unified).parts
    ):
        raise ValueError("run the live arena in its own rlfw user scope")
    cgroup = Path("/sys/fs/cgroup") / unified.lstrip("/")
    quota, period = (cgroup / "cpu.max").read_text().split()
    memory = (cgroup / "memory.max").read_text().strip()
    swap = (cgroup / "memory.swap.max").read_text().strip()
    if (
        quota == "max"
        or int(quota) / int(period) > 4
        or memory == "max"
        or int(memory) > 6 * 1024**3
        or swap != "0"
    ):
        raise ValueError(
            "benchmark scope requires CPUQuota<=400%, MemoryMax<=6G, no swap"
        )
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise ValueError("benchmark evaluator priority must be nice 10 or lower")


def load_binding(spec, path, *, device="cpu"):
    path = local_path(spec.bench_root, path)
    if (
        spec.runtime_binding_sha256 is None
        or hashlib.sha256(path.read_bytes()).hexdigest() != spec.runtime_binding_sha256
    ):
        raise ValueError("external runtime binding must match its frozen SHA")
    if (
        not spec.runtime_files
        or dict(spec.runtime_files).get(
            str(path.relative_to(Path(spec.bench_root).resolve()))
        )
        != spec.runtime_binding_sha256
    ):
        raise ValueError("GameSpec must pin the binding and its execution helpers")
    spec.verify_frozen()
    sys_name = "rlfw_campaign_binding_" + spec.runtime_binding_sha256
    definition = importlib.util.spec_from_file_location(sys_name, path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module.build_runtime(spec, device=device)


def verify_device(spec, device, parity_path=None):
    """CUDA readouts require byte-bound replay parity on the identical pins."""
    if device != spec.device:
        raise ValueError("CLI device must equal the frozen game device")
    if device == "cpu":
        return
    if device != "cuda" or parity_path is None:
        raise ValueError("CUDA requires an explicit CPU/CUDA parity receipt")
    path = local_path(spec.bench_root, parity_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec.device_parity_sha256:
        raise ValueError("device parity receipt hash mismatch")
    if (
        dict(spec.runtime_files).get(
            str(path.relative_to(Path(spec.bench_root).resolve()))
        )
        != spec.device_parity_sha256
    ):
        raise ValueError("device parity receipt must be pinned among runtime files")
    payload = json.loads(path.read_text())
    receipts = payload.get("receipts", [payload])
    qualified = set()
    for receipt in receipts:
        parity = receipt.get("device_parity", {})
        errors = parity.get("maximum_absolute_errors", {})
        if (
            receipt.get("schema") != "jope.rlfw-checkpoint-replay/v1"
            or receipt.get("passed") is not True
            or parity.get("executed") is not True
            or parity.get("passed") is not True
            or parity.get("rows", 0) < 1
            or {parity.get("reference_device"), parity.get("candidate_device")}
            != {"cpu", "cuda"}
            or parity.get("tolerance") != 1e-6
            or set(errors)
            != {"total_logprob", "token_logprob", "q", "p_complete", "runtime_log1p"}
            or any(
                not isinstance(e, (int, float))
                or not math.isfinite(e)
                or not 0 <= e <= 1e-6
                for e in errors.values()
            )
            or receipt.get("code_manifest_sha256") != spec.policy_code_manifest_sha256
            or receipt.get("trie_sha256") != spec.arms[0].trie_sha256
            or receipt.get("normalizer_sha256") != spec.arms[0].normalizer_sha256
            or receipt.get("tokenizer_sha256") != spec.arms[0].tokenizer_sha256
        ):
            raise ValueError(
                "complete same-pin CPU/CUDA likelihood/head parity required"
            )
        qualified.add(receipt.get("checkpoint_parameter_sha256"))
    if any(
        a.checkpoint_sha256 not in qualified
        for a in spec.arms
        if a.checkpoint_sha256 is not None
    ):
        raise ValueError("each CUDA policy checkpoint requires device parity")


def reuse_incumbents(spec, parent_directory, scheduler, arms, oracle):
    """Reuse only qualified original receipts after fresh authority checks."""
    import sqlite3
    from dataclasses import replace

    parent, rows, _ = read_run(parent_directory)
    spec.require_pair(parent)
    for row in rows:
        if row.arm != "hwm" or not row.qualified:
            continue
        context = next(c for c in spec.contexts if c.context_id == row.context_id)
        oracle.check_role(context)
        spec.verify_frozen()
        with sqlite3.connect(f"file:{parent.state_path}?mode=ro", uri=True) as db:
            evidence = db.execute(
                "SELECT execution_key,evidence FROM slots WHERE context=? AND arm='hwm'",
                (row.context_id,),
            ).fetchone()
        if evidence is None:
            raise ValueError("incumbent reuse requires its original qualified evidence")
        candidate = arms.slate(
            context, next(a for a in spec.arms if a.name == "hwm"), spec.seeds[0], 8
        )[0]
        result = oracle.validate(json.loads(evidence[1]), context, candidate.program)
        key = oracle.execution_key(context, candidate.program)
        if evidence[0] != key or result["q_grid"] != row.q_grid:
            raise ValueError("incumbent reuse execution identity or score mismatch")
        oracle.check_role(context)
        spec.verify_frozen()
        scheduler.commit(
            replace(
                row,
                spec_sha256=spec.sha256,
                wall_seconds=0,
                cpu_seconds=0,
                cache_hit=True,
                unknown_cost_attempts=0,
            ),
            key,
            result,
        )
        oracle.own_results[key] = result


def arena(args):
    spec = GameSpec.from_dict(json.loads(Path(args.spec).read_text()))
    verify_runtime_scope(spec)
    verify_device(spec, args.device, args.device_parity)
    if spec.policy_code_manifest_sha256 is None:
        raise ValueError("live games require a pinned policy-code manifest")
    from .decision_trace import DecisionTrace
    from .policy_binding import verify_manifest

    root = local_path(spec.bench_root, args.policy_code_root)
    verify_manifest(root, expected_sha256=spec.policy_code_manifest_sha256)
    policy_arms = tuple(
        a for a in spec.arms if a.name.startswith(("trie_sample:", "rerank32:"))
    )
    if policy_arms:
        receipt_path = local_path(spec.bench_root, args.policy_qualification)
        if (
            hashlib.sha256(receipt_path.read_bytes()).hexdigest()
            != spec.policy_qualification_sha256
        ):
            raise ValueError("policy qualification receipt hash mismatch")
        qualification = json.loads(receipt_path.read_text())
        if (
            qualification.get("passed") is not True
            or qualification.get("draws", 0) < 1000
            or qualification.get("determinism_passed") is not True
        ):
            raise ValueError("1000-draw same-seed likelihood qualification required")
        if not 0 <= qualification.get("maximum_logprob_error", float("inf")) <= 1e-6:
            raise ValueError("deployed likelihood tolerance is 1e-6")
        parameters = set(qualification.get("checkpoint_parameter_sha256s", ()))
        if any(a.checkpoint_sha256 not in parameters for a in policy_arms):
            raise ValueError("each policy checkpoint requires its own qualification")
    if any(a.temperature != 1.0 for a in policy_arms):
        raise ValueError("policy samples use preregistered temperature 1.0")
    spec.freeze()
    runtime = load_binding(spec, args.runtime_binding, device=args.device)
    cache = runtime["frozen_cache"]
    if any(a.name.startswith("rerank32:") for a in spec.arms):
        from .slate import RerankArms

        arms = RerankArms(
            runtime["catalog"],
            runtime["hwm_programs"],
            runtime["sample"],
            runtime["sample_many"],
            runtime["score_heads"],
        )
    else:
        arms = runtime.get("arms") or BaselineArms(
            runtime["catalog"], runtime["hwm_programs"], runtime["sample"]
        )
    oracle = BlindOracle(spec, runtime["evaluate"], runtime["role_check"], cache)
    trace = DecisionTrace(spec.out_dir, runs_root=Path(spec.bench_root) / "runs")
    scheduler = ArenaScheduler(
        spec,
        arms,
        oracle,
        lambda value: pr1_trace_sink(trace, value),
        workers=args.pool_workers,
    )
    ranking = (
        json.loads(Path(args.ranking_json).read_text()) if args.ranking_json else []
    )
    try:
        if args.reuse_hwm_from:
            reuse_incumbents(spec, args.reuse_hwm_from, scheduler, arms, oracle)

        def checkpoint(current):
            if len(current.closed_contexts()) % args.checkpoint_report_every == 0:
                write_reports(
                    spec,
                    current.rows(),
                    current.hwm_values(),
                    current.closed_contexts(),
                    ranking,
                    current.decode_costs(),
                )

        scheduler.run(max_contexts=args.max_contexts, on_context=checkpoint)
        report = write_reports(
            spec,
            scheduler.rows(),
            scheduler.hwm_values(),
            scheduler.closed_contexts(),
            ranking,
            scheduler.decode_costs(),
        )
        if len(scheduler.closed_contexts()) == len(spec.contexts):
            trace.seal()
        return {
            "spec_sha256": spec.sha256,
            "report": str(Path(spec.out_dir) / "report.json"),
            "closed_contexts": report["closed_contexts"],
            "label": LABEL,
        }
    finally:
        scheduler.close()
        if runtime.get("close"):
            runtime["close"]()


def compare(args):
    left, left_rows, left_report = read_run(args.a)
    right, right_rows, right_report = read_run(args.b)
    left.require_pair(right)

    def units(spec, rows, report, arm):
        if arm not in {a.name for a in spec.arms if a.name != "hwm"}:
            raise ValueError("comparison arm is absent")
        closed = {u["context_id"] for u in report["per_context"]}
        hwm = {r.context_id: r.q for r in rows if r.arm == "hwm"}
        return context_units(
            [r for r in rows if r.arm == arm and r.context_id in closed], hwm, spec.tau
        )

    result = {
        "label": LABEL,
        "a_spec_sha256": left.sha256,
        "b_spec_sha256": right.sha256,
        "pairing_sha256": left.pairing_sha256,
        "a_arm": args.a_arm,
        "b_arm": args.b_arm,
        "bootstrap": 2000,
        "bootstrap_seed": 7,
        "paired": paired_difference(
            units(left, left_rows, left_report, args.a_arm),
            units(right, right_rows, right_report, args.b_arm),
        ),
    }
    path = local_path(left.bench_root, args.out, "runs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(canonical(result) + "\n")
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("arena")
    a.add_argument(
        "--spec", required=True, help="preregistered GameSpec JSON; no scores"
    )
    a.add_argument(
        "--runtime-binding",
        required=True,
        help="BENCH-local byte-pinned build_runtime(spec) adapter",
    )
    a.add_argument("--policy-code-root", required=True)
    a.add_argument("--policy-qualification", required=True)
    a.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a.add_argument("--device-parity", help="byte-pinned PR1 CPU/CUDA replay receipt")
    a.add_argument("--pool-workers", type=int, default=4)
    a.add_argument("--max-contexts", type=int)
    a.add_argument("--checkpoint-report-every", type=int, default=20)
    a.add_argument("--ranking-json")
    a.add_argument("--reuse-hwm-from")
    c = commands.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--a-arm", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--b-arm", required=True)
    c.add_argument("--out", required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "arena" and (
        args.checkpoint_report_every < 1
        or args.max_contexts is not None
        and args.max_contexts < 1
    ):
        raise ValueError("positive context/report limits required")
    result = arena(args) if args.command == "arena" else compare(args)
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
