"""Explicit research checkpoint gate with durable monthly spending and receipts."""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean, median, NormalDist
import time

from .arena.game_spec import canonical, digest, local_path, require_sha
from .arena.kpi import cluster_draws, context_units, LABEL
from .arena.report import read_run


def alpha_schedule(looks=6, alpha=0.05):
    """Lan-DeMets O'Brien-Fleming-style cumulative spend, disjoint look budgets."""
    if looks != 6 or alpha != 0.05:
        raise ValueError("six monthly looks and total alpha .05 are preregistered")
    z = NormalDist().inv_cdf(1 - alpha / 2)
    cumulative = [math.erfc(z / math.sqrt(2 * i / looks)) for i in range(1, looks + 1)]
    return tuple(
        {
            "look": i + 1,
            "cumulative_alpha": value,
            "look_alpha": value - (cumulative[i - 1] if i else 0.0),
        }
        for i, value in enumerate(cumulative)
    )


def bootstrap_interval(values, alpha=0.05):
    """Exactly 2000 parent-cluster draws/seed7; no early-tail extrapolation."""
    import numpy as np

    ordered = np.asarray([values[k] for k in sorted(values)], dtype=np.float64)
    if ordered.size == 0 or not np.isfinite(ordered).all():
        raise ValueError("nonempty finite parent means required")
    samples = ordered[np.asarray(cluster_draws(len(ordered), 2000, 7))].mean(axis=1)
    return {
        "mean": float(ordered.mean()),
        "interval": [
            float(x) for x in np.quantile(samples, [alpha / 2, 1 - alpha / 2])
        ],
        "alpha": alpha,
        "bootstrap": 2000,
        "bootstrap_seed": 7,
        "tail_resolved": alpha / 2 * 2000 >= 1,
    }


def signed_receipt(payload):
    return {
        "payload": payload,
        "sha256": digest(payload),
        "signature_kind": "sha256-integrity; no signer authentication",
    }


def verify_receipt(receipt):
    if (
        set(receipt) != {"payload", "sha256", "signature_kind"}
        or receipt["signature_kind"] != "sha256-integrity; no signer authentication"
        or digest(receipt["payload"]) != receipt["sha256"]
    ):
        raise ValueError("gate receipt integrity mismatch")
    require_sha(receipt["sha256"], "gate receipt")
    return receipt["payload"]


@dataclass(frozen=True)
class GateConfig:
    bench_root: str
    deadline_unix: float
    source_sha256: str
    expected_lineages: int = 54
    apply_research_pointer: bool = False

    def __post_init__(self):
        require_sha(self.source_sha256, "gate source")
        if self.expected_lineages != 54 or not 0 < self.deadline_unix <= 1791568800:
            raise ValueError("fixed campaign deadline and 54 lineages required")

    def guard(self):
        if time.time() >= self.deadline_unix:
            raise RuntimeError("fixed campaign gate deadline expired")


def panel_from_runs(challenger, champion, challenger_arm, champion_arm):
    a, ar, ap = read_run(challenger)
    b, br, bp = read_run(champion)
    a.require_pair(b)
    measured = {}
    receipt_values = {}
    for row in ar + br:
        if row.qualified:
            key = (row.context_id, row.program_sha256)
            value = (row.context_id, row.program_sha256, row.q_grid)
            if key in measured and measured[key] != row.q_grid:
                raise ValueError(
                    "same measured program comparison has conflicting quality"
                )
            if (
                row.receipt_sha256 in receipt_values
                and receipt_values[row.receipt_sha256] != value
            ):
                raise ValueError(
                    "one qualified receipt cannot attest different measurements"
                )
            measured[key] = row.q_grid
            receipt_values[row.receipt_sha256] = value
    if Path(a.bench_root).resolve() != Path(b.bench_root).resolve():
        raise ValueError("paired research games must share benchmark custody")

    def side(spec, rows, report, arm):
        registered = next(
            (x for x in spec.arms if x.name == arm and x.name != "hwm"), None
        )
        if registered is None or registered.checkpoint_sha256 is None:
            raise ValueError("gate requires an explicit checkpoint sampling arm")
        closed = {u["context_id"] for u in report["per_context"]}
        if not closed <= {c.context_id for c in spec.contexts}:
            raise ValueError("unknown reported closed context")
        hwm = {r.context_id: r.q for r in rows if r.arm == "hwm"}
        units = context_units(
            [r for r in rows if r.arm == arm and r.context_id in closed], hwm
        )
        expected = {(c, s) for c in closed for s in spec.seeds}
        if {(u["context_id"], u["seed"]) for u in units} != expected:
            raise ValueError("missing or duplicated matched context-seed units")
        # Reward/comparator identities are derived from slots, not report summaries.
        for r in rows:
            if r.spec_sha256 != spec.sha256:
                raise ValueError("slot belongs to another game")
        originals = {c.context_id: c.hwm_program_sha256 for c in spec.contexts}
        comparators = [r for r in rows if r.arm == "hwm"]
        if len({r.context_id for r in comparators}) != len(comparators) or any(
            r.program_sha256 != originals.get(r.context_id) for r in comparators
        ):
            raise ValueError("original comparator identity mismatch")
        return registered.checkpoint_sha256, units

    candidate, au = side(a, ar, ap, challenger_arm)
    baseline, bu = side(b, br, bp, champion_arm)
    roster = [asdict(c) for c in a.contexts]
    return {
        "schema": "jope.rlfw-gate-panel/v1",
        "label": LABEL + "; " + a.device + " inference",
        "panel_sha256": digest(
            {"contexts": roster, "seeds": a.seeds, "role_map": a.role_map_sha256}
        ),
        "pairing_sha256": a.pairing_sha256,
        "game_spec_sha256s": [a.sha256, b.sha256],
        "checkpoint_sha256s": [candidate, baseline],
        "role_map_sha256": a.role_map_sha256,
        "roster": roster,
        "seeds": a.seeds,
        "arms": [challenger_arm, champion_arm],
        "challenger": au,
        "champion": bu,
        "slot_receipt_sha256s": sorted(
            {r.receipt_sha256 for r in ar + br if r.qualified}
        ),
        "benchmark_valid": False,
    }


def evaluate_panel(panel, look):
    if (
        panel["schema"] != "jope.rlfw-gate-panel/v1"
        or panel["benchmark_valid"] is not False
    ):
        raise ValueError("research-only measured gate panel required")
    for pin in (
        *panel["game_spec_sha256s"],
        *panel["checkpoint_sha256s"],
        panel["role_map_sha256"],
        panel["pairing_sha256"],
        *panel["slot_receipt_sha256s"],
    ):
        require_sha(pin, "paired panel identity")
    if (
        len(panel["game_spec_sha256s"]) != 2
        or len(panel["checkpoint_sha256s"]) != 2
        or not panel["seeds"]
        or len(set(panel["seeds"])) != len(panel["seeds"])
        or any(type(s) is not int for s in panel["seeds"])
        or len({c["context_id"] for c in panel["roster"]}) != len(panel["roster"])
        or any(c["role"] != "development" for c in panel["roster"])
        or panel["panel_sha256"]
        != digest(
            {
                "contexts": panel["roster"],
                "seeds": panel["seeds"],
                "role_map": panel["role_map_sha256"],
            }
        )
    ):
        raise ValueError("frozen development roster/seed identity mismatch")
    schedule = alpha_schedule()
    if type(look) is not int or not 1 <= look <= 6:
        raise ValueError("monthly gate look exhausted")
    spending = schedule[look - 1]
    left = {(u["context_id"], u["seed"]): u for u in panel["challenger"]}
    right = {(u["context_id"], u["seed"]): u for u in panel["champion"]}
    reasons = []
    roster = {c["context_id"]: c["lineage_id"] for c in panel["roster"]}
    expected = {(cid, seed) for cid in roster for seed in panel["seeds"]}
    if set(left) != set(right):
        raise ValueError("paired gate context/seed mismatch")
    if len(left) != len(panel["challenger"]) or len(right) != len(panel["champion"]):
        raise ValueError("duplicated gate units")
    if set(left) != expected:
        reasons.append("panel incomplete")
    if len(set(roster.values())) != 54:
        reasons.append("fewer than 54 frozen lineages")
    metrics = defaultdict(lambda: defaultdict(list))
    costs = [defaultdict(list), defaultdict(list)]
    gains = [defaultdict(list), defaultdict(list)]
    for key in sorted(left):
        x, y = left[key], right[key]
        if (
            x["lineage_id"] != y["lineage_id"]
            or x["lineage_id"] != roster.get(key[0])
            or x["hwm"] != y["hwm"]
        ):
            raise ValueError("paired gate lineage/incumbent mismatch")
        lineage = x["lineage_id"]
        if x["censored"] or y["censored"] or x["hwm"] is None:
            reasons.append("censored or unqualified comparator")
            continue
        if any(u["wall_seconds"] is None or u["unknown_cost_attempts"] for u in (x, y)):
            reasons.append("unknown evaluator cost")
            continue
        if any(
            u["wall_seconds"] < 0 or not math.isfinite(u["wall_seconds"])
            for u in (x, y)
        ):
            raise ValueError("finite nonnegative evaluator cost required")
        for metric in ("positive_gain", "score", "valid8", "qualified_completion_rate"):
            if any(u[metric] is None or not math.isfinite(u[metric]) for u in (x, y)):
                raise ValueError("finite measured paired metric required")
            metrics[metric][lineage].append(x[metric] - y[metric])
        for i, unit in enumerate((x, y)):
            costs[i][unit["context_id"]].append(unit["wall_seconds"])
            gains[i][lineage].append(unit["positive_gain"])
    per_lineage = {
        metric: {k: fmean(v) for k, v in sorted(groups.items())}
        for metric, groups in metrics.items()
    }
    if len(per_lineage.get("positive_gain", {})) != 54:
        reasons.append("not all 54 lineages have qualified paired units")
    stats = {}
    if per_lineage:
        for metric in ("positive_gain", "score"):
            stats[metric] = {
                "ci95": bootstrap_interval(per_lineage[metric]),
                "spent": bootstrap_interval(
                    per_lineage[metric], spending["look_alpha"]
                ),
            }
        if not stats["positive_gain"]["spent"]["tail_resolved"]:
            reasons.append("2000 bootstrap draws do not resolve this look's tail")
    if reasons:
        verdict = "INCONCLUSIVE"
    else:
        median_costs = [median(fmean(v) for v in c.values()) for c in costs]
        gain_means = [fmean(fmean(v) for v in g.values()) for g in gains]
        gain_per_second = [
            g / c if c > 0 else None for g, c in zip(gain_means, median_costs)
        ]
        time_pass = median_costs[0] <= 1.05 * median_costs[1] or (
            None not in gain_per_second and gain_per_second[0] > gain_per_second[1]
        )
        predicates = {
            "delta_J8_lower_positive": stats["positive_gain"]["ci95"]["interval"][0] > 0
            and stats["positive_gain"]["spent"]["interval"][0] > 0,
            "delta_win_lower_nonnegative": stats["score"]["ci95"]["interval"][0] >= 0
            and stats["score"]["spent"]["interval"][0] >= 0,
            "valid8_drop_at_most_two_points": fmean(per_lineage["valid8"].values())
            >= -0.02,
            "completion_drop_at_most_two_points": fmean(
                per_lineage["qualified_completion_rate"].values()
            )
            >= -0.02,
            "evaluator_cost_or_gain_rate": time_pass,
        }
        reasons = [name for name, passed in predicates.items() if not passed]
        verdict = "PROMOTE" if not reasons else "REJECT"
        stats["cost"] = {
            "median_evaluator_seconds": median_costs,
            "mean_J8": gain_means,
            "gain_per_second": gain_per_second,
        }
    return {
        "verdict": verdict,
        "reasons": sorted(set(reasons)),
        "spending": spending,
        "stats": stats,
        "per_lineage_deltas": per_lineage,
        "bootstrap": 2000,
        "bootstrap_seed": 7,
        "benchmark_valid": False,
    }


def create_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o444)
    directory_fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def state_path(bench, path):
    root = Path(bench).resolve()
    raw = Path(path).absolute()
    if any(p.is_symlink() for p in (raw, *raw.parents) if p.is_relative_to(root)):
        raise ValueError("research state links are forbidden")
    resolved = local_path(root, raw)
    if not resolved.is_relative_to(root / "state"):
        raise ValueError("research gate writes require BENCH/state custody")
    return resolved


def run_gate(config, panel):
    """Every valid look spends before inference; only this CLI can move research state."""
    config.guard()
    for pin in (
        *panel["game_spec_sha256s"],
        *panel["checkpoint_sha256s"],
        panel["panel_sha256"],
        panel["pairing_sha256"],
    ):
        require_sha(pin, "gate input identity")
    state = state_path(config.bench_root, Path(config.bench_root) / "state")
    state.mkdir(parents=True, exist_ok=True)
    lock_path = state_path(config.bench_root, state / "promotion.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config.guard()
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        ledger = state_path(config.bench_root, state / "gates" / month)
        ledger.mkdir(parents=True, exist_ok=True)
        reserved = sorted(ledger.glob("look_*.reservation.json"))
        if [p.name for p in reserved] != [
            f"look_{i:02d}.reservation.json" for i in range(1, len(reserved) + 1)
        ]:
            raise ValueError("monthly spend ledger is not contiguous")
        for i, path in enumerate(reserved, 1):
            body = verify_receipt(
                json.loads(state_path(config.bench_root, path).read_text())
            )
            if (
                body["month"] != month
                or body["look"] != i
                or body["spending"] != alpha_schedule()[i - 1]
            ):
                raise ValueError("monthly reservation identity/spending changed")
        look = len(reserved) + 1
        if look > 6:
            raise ValueError("six monthly looks already consumed")
        pointer = state_path(config.bench_root, state / "champion.json")
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
        if previous is not None:
            verify_receipt(previous)
        reservation = {
            "schema": "jope.rlfw-gate-reservation/v1",
            "month": month,
            "look": look,
            "panel_sha256": panel["panel_sha256"],
            "input_sha256": digest(panel),
            "config": asdict(config),
            "previous_pointer_sha256": digest(previous),
            "spending": alpha_schedule()[look - 1],
            "at_unix": time.time(),
        }
        create_json(
            ledger / f"look_{look:02d}.reservation.json", signed_receipt(reservation)
        )
        decision = evaluate_panel(panel, look)
        blocked = (state / "STOP.json").exists() or (
            state / "bench-health-hold.json"
        ).exists()
        if blocked:
            decision.update(
                verdict="INCONCLUSIVE",
                reasons=sorted(set(decision["reasons"] + ["campaign guard active"])),
            )
        if (
            previous
            and previous["payload"]["search_champion_sha256"]
            != panel["checkpoint_sha256s"][1]
        ):
            decision.update(
                verdict="INCONCLUSIVE",
                reasons=sorted(set(decision["reasons"] + ["stale research champion"])),
            )
        applied = decision["verdict"] == "PROMOTE" and config.apply_research_pointer
        receipt = signed_receipt(
            {
                "schema": "jope.rlfw-research-gate/v1",
                "label": panel["label"],
                "reservation_sha256": digest(reservation),
                "source_sha256": config.source_sha256,
                "month": month,
                "look": look,
                "panel_sha256": panel["panel_sha256"],
                "game_spec_sha256s": panel["game_spec_sha256s"],
                "pairing_sha256": panel["pairing_sha256"],
                "checkpoint_sha256s": panel["checkpoint_sha256s"],
                "input_sha256": digest(panel),
                "previous_pointer_sha256": digest(previous),
                "decision": decision,
                "pointer_application_requested": applied,
                "scope": "research only; no serving or teacher admission",
            }
        )
        create_json(ledger / f"look_{look:02d}.receipt.json", receipt)
        application = None
        if applied:
            config.guard()
            if (state / "STOP.json").exists() or (
                state / "bench-health-hold.json"
            ).exists():
                raise RuntimeError(
                    "campaign guard changed before research pointer commit"
                )
            version = signed_receipt(
                {
                    "schema": "jope.rlfw-research-pointer/v1",
                    "search_champion_sha256": panel["checkpoint_sha256s"][0],
                    "anchor_sha256": panel["checkpoint_sha256s"][0],
                    "gate_receipt_sha256": receipt["sha256"],
                    "previous_pointer_sha256": digest(previous),
                    "benchmark_valid": False,
                }
            )
            versions = state_path(config.bench_root, state / "pointer_versions")
            versions.mkdir(exist_ok=True)
            immutable = versions / (version["sha256"] + ".json")
            create_json(immutable, version)
            current = json.loads(pointer.read_text()) if pointer.exists() else None
            if current != previous:
                raise RuntimeError("research pointer changed outside promotion lock")
            # One authoritative version stores both champion and anchor atomically.
            temporary = state_path(
                config.bench_root, state / ("champion." + version["sha256"] + ".new")
            )
            create_json(temporary, version)
            os.replace(temporary, pointer)
            directory_fd = os.open(state, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            application = signed_receipt(
                {
                    "schema": "jope.rlfw-research-pointer-application/v1",
                    "gate_receipt_sha256": receipt["sha256"],
                    "pointer_sha256": version["sha256"],
                    "pointer_applied": json.loads(pointer.read_text()) == version,
                }
            )
            create_json(ledger / f"look_{look:02d}.application.json", application)
        return {"gate_receipt": receipt, "application_receipt": application}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bench-root", required=True)
    p.add_argument("--challenger-run", required=True)
    p.add_argument("--champion-run", required=True)
    p.add_argument("--challenger-arm", required=True)
    p.add_argument("--champion-arm", required=True)
    p.add_argument("--deadline-unix", required=True, type=float)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--apply-research-pointer", action="store_true")
    args = p.parse_args(argv)
    source_root = Path(__file__).parent
    module_sha = digest(
        {
            str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_root.rglob("*.py"))
        }
    )
    if module_sha != args.source_sha256:
        raise ValueError("gate source byte identity mismatch")
    panel = panel_from_runs(
        args.challenger_run, args.champion_run, args.challenger_arm, args.champion_arm
    )
    if (
        Path(args.bench_root).resolve()
        != Path(read_run(args.challenger_run)[0].bench_root).resolve()
    ):
        raise ValueError("gate benchmark differs from paired runs")
    config = GateConfig(
        args.bench_root,
        args.deadline_unix,
        module_sha,
        apply_research_pointer=args.apply_research_pointer,
    )
    result = run_gate(config, panel)
    print(canonical(result))
    return result


if __name__ == "__main__":
    main()
