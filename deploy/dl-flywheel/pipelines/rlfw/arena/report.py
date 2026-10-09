"""Benchmark-local JSON, Markdown and per-context CSV with frozen identities."""

import csv
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

from .game_spec import canonical, local_path
from .kpi import cluster_stat, compute_report


def write_reports(spec, rows, hwm, closed_contexts, ranking=(), decode_costs=None):
    spec.verify_frozen()
    report = compute_report(spec, rows, hwm, closed_contexts, ranking)
    by_arm = defaultdict(lambda: defaultdict(list))
    lineages = {c.context_id: c.lineage_id for c in spec.contexts}
    closed = set(closed_contexts)
    for cost in decode_costs or ():
        if cost["context_id"] in closed:
            by_arm[cost["arm"]][lineages[cost["context_id"]]].append(
                cost["wall_seconds"]
            )
    report["decode_rerank_wall_seconds"] = {
        arm: cluster_stat(values) for arm, values in sorted(by_arm.items())
    }
    report["decode_costs_available"] = decode_costs is not None
    out = local_path(spec.bench_root, spec.out_dir, "runs")
    lines = [
        f"# {spec.game_id}",
        "",
        report["label"],
        "",
        f"GameSpec: `{spec.sha256}`",
        f"Pairing identity: `{spec.pairing_sha256}`",
        "",
        f"Closed contexts: {report['closed_contexts']}; observed lineages: {report['observed_lineages']}.",
        f"Comparable incumbent coverage: {report['comparable_coverage']}; censored units: {report['censored_unit_fraction']}.",
        "",
        "B=8, tau=0.01 q-units; parent-cluster bootstrap 2000, seed 7.",
        "Point win rates exclude censored units; bounds assign every censored unit LOSS/WIN.",
        "Regret and best-q means are conditional on observed validity; unknown outcomes have no numeric reward.",
        "",
        "| Arm | Win rate [95% CI] | Censoring bounds | R | R+ | Valid@8 | Distinct programs | Evaluator seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["tau_sensitivity"][str(spec.tau)]["arms"].items():
        win = m["win_rate"]
        lines.append(
            f"| {name} | {win['mean']} {win['ci95']} | {m['censoring_bounds']} | {m['regret']['mean']} | "
            f"{m['positive_regret']['mean']} | {m['valid8']['mean']} | {m['distinct_shas']['mean']} | {m['wall_seconds']['mean']} |"
        )
    lines += [
        "",
        "Full best-q curves, tau sensitivity, paired differences, costs and separate ranking sources are in report.json.",
        "Decode/rerank wall seconds are reported separately and do not consume execution slots.",
        "Size/coverage checks require separate G0/G1 integrity and deterministic replay receipts before a valid-baseline claim.",
        "No generalization or serving-promotion claim follows from this development panel.",
    ]
    stream = io.StringIO()
    fields = [
        "spec_sha256",
        "label",
        "context_id",
        "lineage_id",
        "arm",
        "seed",
        "hwm",
        "censored",
        "score",
        "outcome",
        "lower_score",
        "upper_score",
        "regret",
        "positive_regret",
        "positive_gain",
        "best_q@1",
        "best_q@2",
        "best_q@4",
        "best_q@8",
        "valid@1",
        "valid@2",
        "valid@4",
        "valid@8",
        "charged_slots",
        "wall_seconds",
        "cpu_seconds",
        "distinct_shas",
    ]
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    for unit in report["per_context"]:
        writer.writerow({**unit, "spec_sha256": spec.sha256, "label": report["label"]})
    manifest = {}
    for name, payload in (
        ("report.json", canonical(report) + "\n"),
        ("report.md", "\n".join(lines) + "\n"),
        ("per_context.csv", stream.getvalue()),
        (
            "slots.json",
            canonical(
                {"spec_sha256": spec.sha256, "slots": [r.to_dict() for r in rows]}
            )
            + "\n",
        ),
    ):
        path = local_path(spec.bench_root, out / name, "runs")
        temporary = local_path(
            spec.bench_root, path.with_name(path.name + ".new"), "runs"
        )
        with temporary.open("w") as f:
            f.write(payload)
        temporary.replace(path)
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = out / "outputs.manifest.json"
    path = local_path(spec.bench_root, path, "runs")
    temporary = local_path(spec.bench_root, path.with_name(path.name + ".new"), "runs")
    temporary.write_text(
        canonical({"spec_sha256": spec.sha256, "files": manifest}) + "\n"
    )
    temporary.replace(path)
    return report


def read_run(directory):
    from .game_spec import GameSpec
    from .kpi import SlotResult

    directory = Path(directory).resolve()
    frozen = json.loads((directory / "game_spec.json").read_text())
    spec = GameSpec.from_dict(frozen["spec"])
    if (
        directory != Path(spec.out_dir).resolve()
        or frozen["spec_sha256"] != spec.sha256
    ):
        raise ValueError("run path or frozen spec identity disagrees")
    spec.verify_frozen()
    manifest = json.loads((directory / "outputs.manifest.json").read_text())
    if manifest["spec_sha256"] != spec.sha256 or set(manifest["files"]) != {
        "report.json",
        "report.md",
        "per_context.csv",
        "slots.json",
    }:
        raise ValueError("output manifest identity mismatch")
    for name, pin in manifest["files"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != pin:
            raise ValueError("output bytes changed after reporting")
    slots = json.loads((directory / "slots.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if slots["spec_sha256"] != spec.sha256 or report["spec_sha256"] != spec.sha256:
        raise ValueError("output GameSpec hash mismatch")
    rows = [SlotResult(**r) for r in slots["slots"]]
    return spec, rows, report
