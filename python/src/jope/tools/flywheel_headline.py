#!/usr/bin/env python3
"""Flywheel headline: one immutable scoreboard receipt per generation and a latest.json feed for the native console.

Rows (owner decision O4, 2026-09-13): primary_heldout (16 meta_validation cuts, underpowered, always first),
training_win_rate (124 meta_train contexts, NOT held out), frozen_panel (per monitored block), high_marks (local record
highs per hour from the champion store; not held-out wins), learning (hold / growth / blocks / serving), publication
(champion git state). Inputs: the scoreboard series (flywheel_series.py rows), the baseline receipt, runs/champions,
learning/progress.json and the champion git state. Every row names its source and carries the claim it does NOT make:
`claims.independent_improvement` is "not_established" unless a verdict receipt with outcome smarter_confirmed is given.

  flywheel_headline.py --root R --manifest M --baseline B --series S --generation N --output DIR [--latest FILE]
writes DIR/generation-<N>.json (parallel.flywheel_headline.v1, immutable) and FILE (parallel.flywheel_latest_headline.v1).
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import hold_release_readout
import promotion_gate
from resume_expired_cycle import atomic, read, require, sha

SCHEMA = "parallel.flywheel_headline.v1"
LATEST_SCHEMA = "parallel.flywheel_latest_headline.v1"
PUBLISHABLE = ("meta_train", "meta_validation")
# Rows a later run of the same generation may refresh in latest.json; the immutable per-generation receipt keeps its bytes.
LIVE_ROWS = ("high_marks", "learning", "publication", "frozen_panel")
VERDICT_SCAN_CAP = (
    512 << 20
)  # journal bytes read (from the end, doubling) while looking for a checkpoint's diagnostic verdict
CLAIMS = dict(
    independent_improvement="not_established",
    rule="only a parallel.flywheel_verdict receipt with outcome smarter_confirmed establishes improvement; training-cut rows are NOT held out; "
    "high marks are local record highs on the adaptive cut-heldout teacher, not held-out wins, not promotions; a promotion decision is operational",
)


def series_rows(path):
    rows = []
    if not Path(path).is_file():
        return rows
    with open(path, "rb") as stream:
        for raw in stream:
            try:
                rows.append(json.loads(raw))
            except ValueError:
                continue
    return rows


def latest_by(rows, kind, key):
    table = {}
    for row in rows:
        if row.get("kind") == kind and row.get(key) is not None:
            table[row[key]] = row
    return table


def wtl(block):
    if not isinstance(block, dict) or not isinstance(block.get("wins"), int):
        return None
    n = block["wins"] + block.get("ties", 0) + block.get("losses", 0)
    return dict(
        wins=block["wins"],
        ties=block.get("ties", 0),
        losses=block.get("losses", 0),
        n_pairs=n,
        win_rate=(block["wins"] / n) if n else None,
        mean_log_ratio=block.get("mean_log_ratio"),
        ci90=block.get("ci90"),
    )


def high_marks(root, evaluator, now, window_hours=24):
    counts = dict(
        events_24h=0,
        events_1h=0,
        by_partition=dict(meta_train=0, meta_validation=0),
        identities=0,
        all_time=0,
    )
    for current_path in glob.glob(str(Path(root) / "runs/champions/*/current.json")):
        current = read(current_path)
        identity = (current or {}).get("identity") or {}
        if identity.get("partition") not in PUBLISHABLE or (
            evaluator and identity.get("evaluator_sha256") != evaluator
        ):
            continue
        counts["identities"] += 1
        for event in glob.glob(str(Path(current_path).parent / "event-*.json")):
            try:
                age = now - os.path.getmtime(event)
            except OSError:
                continue
            counts["all_time"] += 1
            if age < window_hours * 3600:
                counts["events_24h"] += 1
                counts["by_partition"][identity["partition"]] += 1
            if age < 3600:
                counts["events_1h"] += 1
    counts["per_hour_24h"] = counts["events_24h"] / window_hours
    counts["source"] = (
        "runs/champions event files (mtime); live evaluator identities; meta_test never counted"
    )
    return counts


def publication_state(run):
    repo = Path(run) / "successor/champion-git/repo"
    state_path = repo / ".champion-git-state.json"
    state = read(state_path) if state_path.is_file() else None
    if not state:
        return dict(status="no_repository")
    receipts = sorted(glob.glob(str(Path(run) / "successor/champion-git/receipts/*.json")))
    last = read(receipts[-1]) if receipts else {}
    remote = state.get("remote") or {}
    return dict(
        status=remote.get("status"),
        events_committed=len(state.get("published") or {}),
        identities=len(state.get("identities") or {}),
        last_sync=(last or {}).get("at"),
        committed_last_sync=(last or {}).get("committed"),
        verified_head=state.get("verified_head"),
        last_error=(remote.get("last_error") or "")[:120] or None,
        note="an attempted push is not a published milestone",
    )


def journal_verdicts(run, shas, cap=VERDICT_SCAN_CAP):
    """Last diagnostic verdict per checkpoint sha from a bounded, growing tail of the controller journal.

    The tail starts at hold_release_readout.TAIL_BYTES and doubles (up to `cap`) until every sha has a verdict or the
    file start is reached; `exhausted` says whether the whole journal was covered, so a missing verdict is `unmonitored`
    only when it is, and `unknown_within_window` otherwise (never a silent zero).
    """
    journal = Path(run) / "learning/journal.jsonl"
    wanted = {s for s in shas if isinstance(s, str)}
    found = {}
    if not journal.is_file() or not wanted:
        return dict(verdicts=found, bytes=0, exhausted=journal.is_file())
    size = os.path.getsize(journal)
    window = hold_release_readout.TAIL_BYTES
    while True:
        found = {}
        for event in hold_release_readout.journal_tail(journal, max_bytes=window):
            if (
                event.get("event") in ("diagnostic_accepted", "diagnostic_rejected")
                and event.get("checkpoint_sha256") in wanted
            ):
                found[event["checkpoint_sha256"]] = (
                    event["event"],
                    event.get("timestamp"),
                )  # the last verdict wins (re-monitoring)
        exhausted = window >= size
        if wanted <= set(found) or exhausted or window >= cap:
            return dict(verdicts=found, bytes=min(window, size), exhausted=exhausted)
        window = min(window * 2, max(cap, size if size <= cap else cap))


def proposer_status(run, checkpoint_sha256, verdicts=None):
    """The frozen-panel verdict of the checkpoint that proposed a generation, from the controller journal.

    A generation proposed by a `diagnostic_rejected` checkpoint is labelled `rejected_lineage` (its rows are the state of
    record but not evidence for the recipe); an accepted proposer is `accepted`; a checkpoint never monitored (the
    serving/pre-release proposer) is `unmonitored` when the whole journal was read, `unknown_within_window` when only the
    bounded tail was; without a journal the label is null.
    """
    journal = Path(run) / "learning/journal.jsonl"
    if not isinstance(checkpoint_sha256, str) or not journal.is_file():
        return dict(checkpoint_sha256=checkpoint_sha256, verdict=None, label=None, source=None)
    scan = verdicts if verdicts is not None else journal_verdicts(run, [checkpoint_sha256])
    verdict, at = scan["verdicts"].get(checkpoint_sha256, (None, None))
    label = {"diagnostic_rejected": "rejected_lineage", "diagnostic_accepted": "accepted"}.get(
        verdict, "unmonitored" if scan["exhausted"] else "unknown_within_window"
    )
    return dict(
        checkpoint_sha256=checkpoint_sha256,
        verdict=verdict,
        verdict_at=at,
        label=label,
        journal_window_bytes=scan["bytes"],
        source="learning/journal.jsonl diagnostic events (bounded tail)",
    )


def frozen_guard(entry, progress, manifest, verdicts):
    """The guard a monitored checkpoint was judged against: baseline x (1 + maximum_relative_regression), with its journal verdict."""
    baseline = (progress.get("training_diagnostic") or {}).get("baseline_ranking_bt")
    ratio = (manifest.get("training_diagnostic_guard") or {}).get("maximum_relative_regression")
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    guard_bt = baseline * (1 + ratio) if numeric(baseline) and numeric(ratio) else None
    bt = (entry.get("panel") or {}).get("ranking_bt")
    verdict, at = verdicts["verdicts"].get(entry.get("checkpoint_sha256"), (None, None))
    return dict(
        baseline_ranking_bt=baseline,
        maximum_relative_regression=ratio,
        guard_bt=guard_bt,
        verdict=verdict or ("unmonitored" if verdicts["exhausted"] else "unknown_within_window"),
        verdict_at=at,
        exceeds_guard=(bt > guard_bt) if numeric(bt) and numeric(guard_bt) else None,
    )


def rollback_status(root, run):
    """The F11 rollback state for the learning row, recomputed live from the read-out receipt; never null-as-ok."""
    readout_path = Path(run) / "successor/decisions/f11-first-monitor-readout.json"
    if not readout_path.is_file():
        return dict(status="no_readout")
    readout = read(readout_path)
    if readout.get("schema") != hold_release_readout.SCHEMA or readout.get("status") != "decided":
        return dict(status="readout_not_decided", readout_sha256=sha(readout_path))
    if readout.get("outcome") != "rollback_required":
        return dict(
            status="no_rollback_required", outcome=readout.get("outcome"), readout_sha256=sha(readout_path)
        )
    if not (Path(run) / "learning/state.json").is_file():
        return dict(status="no_state", readout_sha256=sha(readout_path))
    v = hold_release_readout.verify_rollback(Path(root), Path(run), readout_path)
    return dict(
        status=v["status"],
        learning_checkpoint={
            k: v["learning_checkpoint"].get(k) for k in ("sha256", "is_target", "is_rejected")
        },
        reparent_target_sha256=v["reparent_target_sha256"],
        rejected_checkpoint_sha256=v["rejected_checkpoint_sha256"],
        deviations=len(v["deviations"]),
        automatic_training=v["manifest"].get("automatic_training"),
        readout_sha256=v["readout"]["sha256"],
        journal_window_bytes=v.get("journal_window_bytes"),
    )


def build(root, run, manifest_path, baseline_path, series_path, generation, verdict_path=None):
    root, run = Path(root), Path(run)
    manifest = read(manifest_path)
    baseline = read(baseline_path)
    evaluator = (manifest.get("bindings") or {}).get(manifest.get("cpu_executable"))
    rows = series_rows(series_path)
    generations = latest_by(rows, "generation", "generation")
    require(generation in generations, "no generation row %d in %s" % (generation, series_path))
    row = generations[generation]
    matched = row.get("matched") or {}
    base_held = ((baseline.get("metrics") or {}).get("matched_heldout") or {}).get("seed_only") or {}
    base_train = (baseline.get("metrics") or {}).get("matched_train")
    base_train = (base_train.get("seed_only") if isinstance(base_train, dict) else None) or {}
    primary = dict(
        seed_only=wtl((matched.get("val") or {}).get("seed_only")),
        trailing10=wtl((matched.get("val") or {}).get("trailing10")),
        trailing_generations=((matched.get("val") or {}).get("trailing10") or {}).get("generations")
        if isinstance((matched.get("val") or {}).get("trailing10"), dict)
        else None,
        vs_shuffled=wtl(((matched.get("val") or {}).get("seed_only") or {}).get("vs_shuffled"))
        if isinstance((matched.get("val") or {}).get("seed_only"), dict)
        else None,
        baseline=wtl(base_held),
        mde=((baseline.get("metrics") or {}).get("matched_heldout") or {}).get("mde"),
        label="HELD-OUT 16 meta_validation cuts (PRIMARY, underpowered n=16); ln(neural/generic), negative favours neural",
        source="flywheel_series generation row matched.val (seed-only best_initial_arm per search)",
    )
    training = dict(
        seed_only=wtl((matched.get("train") or {}).get("seed_only")),
        final=wtl((matched.get("train") or {}).get("final")),
        baseline=wtl(base_train),
        n_contexts=row.get("contexts"),
        label="TRAINING-CUT WIN RATE (adaptive meta_train contexts, NOT held out; final level shares the incumbent tail)",
        derivation=row.get("best_initial_arm_derived"),
        source="flywheel_series generation row matched.train",
    )
    monitors = sorted(
        (r for r in rows if r.get("kind") == "monitor" and isinstance(r.get("evaluated_step"), int)),
        key=lambda r: (r["evaluated_step"], r.get("timestamp") or 0),
    )[-6:]
    progress = read(run / "learning/progress.json") if (run / "learning/progress.json").is_file() else {}
    verdicts = journal_verdicts(
        run, [m.get("checkpoint_sha256") for m in monitors] + [row.get("checkpoint_sha256")]
    )
    frozen = [
        dict(
            evaluated_step=m.get("evaluated_step"),
            checkpoint_sha256=m.get("checkpoint_sha256"),
            panel={
                k: (m.get("panel") or {}).get(k)
                for k in (
                    "sequence_nll",
                    "ranking_bt",
                    "ranking_listwise",
                    "runtime_huber",
                    "nontie_pairwise_accuracy",
                )
            },
            baseline_delta=m.get("baseline_delta"),
            lineage=m.get("lineage"),
            guard=frozen_guard(m, progress, manifest, verdicts),
        )
        for m in monitors
    ]
    growth = progress.get("training_growth") or {}
    learning = dict(
        phase=progress.get("phase"),
        automatic_training=progress.get("automatic_training"),
        hold=(progress.get("training_diagnostic") or {}).get("held"),
        hold_reason=progress.get("training_hold_reason"),
        completed_training_blocks=progress.get("completed_training_blocks"),
        growth=dict(
            eligible=growth.get("eligible"),
            programs=growth.get("newly_admitted_training_programs"),
            searches=growth.get("newly_admitted_current_run_searches"),
        ),
        learning_checkpoint=progress.get("last_committed_checkpoint"),
        serving_checkpoint=progress.get("serving_checkpoint"),
        serving_status=progress.get("serving_status"),
        gpu_task_occupancy=(progress.get("gpu_occupancy") or {}).get("task_occupancy"),
        rollback=rollback_status(root, run),
        source="learning/progress.json",
    )
    now = time.time()
    proposer = proposer_status(run, row.get("checkpoint_sha256"), verdicts)
    verdict = read(verdict_path) if verdict_path and Path(verdict_path).is_file() else None
    claims = dict(CLAIMS)
    if isinstance(verdict, dict) and verdict.get("outcome") == "smarter_confirmed":
        claims["independent_improvement"] = "established_by_verdict:" + sha(verdict_path)
    return dict(
        schema=SCHEMA,
        generation=generation,
        generated_at=now,
        run_identity=manifest.get("run_identity"),
        manifest_sha256=sha(manifest_path),
        evaluator_sha256=evaluator,
        baseline=dict(path=str(baseline_path), sha256=sha(baseline_path)),
        series=dict(path=str(series_path), sha256=sha(series_path)),
        generation_row=dict(
            id=row.get("id"),
            checkpoint_sha256=row.get("checkpoint_sha256"),
            report_sha256=row.get("report_sha256"),
            timestamp=row.get("timestamp"),
            proposer=proposer,
        ),
        rows=dict(
            primary_heldout=primary,
            training_win_rate=training,
            frozen_panel=frozen,
            high_marks=high_marks(root, evaluator, now),
            learning=learning,
            publication=publication_state(run),
        ),
        verdict=dict(path=str(verdict_path), sha256=sha(verdict_path), outcome=verdict.get("outcome"))
        if isinstance(verdict, dict)
        else None,
        claims=claims,
        final_tests_opened=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", default="runs/eight-a100-recovery-20260912")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--generation", type=int, help="default: the newest generation row in the series")
    parser.add_argument("--verdict", type=Path)
    parser.add_argument(
        "--output", type=Path, required=True, help="directory for immutable generation-<N>.json receipts"
    )
    parser.add_argument("--latest", type=Path, help="atomic latest.json for the console")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run = root / args.run
    generation = args.generation
    if generation is None:
        rows = latest_by(series_rows(args.series), "generation", "generation")
        require(rows, "series carries no generation rows")
        generation = max(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / ("generation-%06d.json" % generation)
    fresh = build(root, run, args.manifest, args.baseline, args.series, generation, args.verdict)
    if path.is_file():
        # The per-generation receipt is immutable; a later run of the same generation only refreshes the LIVE_ROWS
        # (high marks, learning, publication, frozen panel) in latest.json and keeps the receipt's scoreboard rows and digest.
        receipt = read(path)
        require(
            receipt.get("schema") == SCHEMA and receipt.get("generation") == generation,
            "existing receipt %s is not this generation's" % path,
        )
        digest = sha(path)
        live = dict(receipt["rows"], **{k: fresh["rows"][k] for k in LIVE_ROWS})
        latest = dict(
            receipt,
            rows=live,
            schema=LATEST_SCHEMA,
            receipt=dict(path=str(path), sha256=digest),
            timestamp=time.time(),
            live_rows=list(LIVE_ROWS),
        )
    else:
        receipt = fresh
        digest = promotion_gate.immutable(path, receipt)
        latest = dict(
            receipt,
            schema=LATEST_SCHEMA,
            receipt=dict(path=str(path), sha256=digest),
            timestamp=time.time(),
            live_rows=[],
        )
    if args.latest:
        atomic(args.latest, latest)
    print(
        json.dumps(
            dict(
                receipt=str(path),
                sha256=digest,
                generation=generation,
                primary=receipt["rows"]["primary_heldout"]["trailing10"],
                training=receipt["rows"]["training_win_rate"]["seed_only"],
                high_marks_per_hour=receipt["rows"]["high_marks"]["per_hour_24h"],
            )
        )
    )
    return receipt


if __name__ == "__main__":
    main()
