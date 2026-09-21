#!/usr/bin/env python3
"""F11 read-out: the first monitored learning checkpoint after a hold release decides the rollback clause.

Reads the controller journal (learning/journal.jsonl) for the acknowledgement event named by the decision receipt and
the first diagnostic_accepted / diagnostic_rejected event after it, loads that monitor receipt (parallel.frozen_monitoring.v1)
and the guard state, and writes an immutable parallel.hold_release_readout.v1 under successor/decisions/ that says, per
F11: `continue` (accepted: the loop runs on under sampler A) or `rollback_required` (rejected: training must stop through
a revision with automatic_training false and re-parent to the last diagnostic_accepted checkpoint named here). Nothing here
changes any state; root implements the revision.
"""

import argparse
import json
import os
from pathlib import Path

import promotion_gate
from resume_expired_cycle import atomic, read, require, sha

SCHEMA = "parallel.hold_release_readout.v1"
DECISION_SCHEMA = "parallel.hold_release_decision.v1"


TAIL_BYTES = 8 << 20  # default bounded window over the controller journal (the live one is 2.8 GB)


def journal_events(path):
    with open(path, "rb") as stream:
        for raw in stream:
            try:
                yield json.loads(raw)
            except ValueError:
                continue


def journal_offset(path, max_bytes=TAIL_BYTES, since=None):
    """Byte offset where a bounded tail read of the journal starts.

    The window is the last `max_bytes` of the file; when `since` (a timestamp) is given the window doubles until the
    first complete line inside it is not newer than `since` (or the whole file is covered), so callers that need "every
    event after T" read exactly enough and never the whole 2.8 GB by default.
    """
    size = os.path.getsize(path)
    window = max(1, int(max_bytes))
    while True:
        offset = max(0, size - window)
        if since is None or offset == 0:
            return offset
        with open(path, "rb") as stream:
            stream.seek(offset)
            stream.readline()  # partial line
            first = stream.readline()
        try:
            stamp = float((json.loads(first) or {}).get("timestamp") or 0)
        except (ValueError, TypeError):
            stamp = float("inf")  # unreadable head: widen
        if stamp <= since:
            return offset
        window *= 2


def journal_tail(path, max_bytes=TAIL_BYTES, since=None):
    """Yield the parsed events of the bounded tail (complete lines only; the first partial line is dropped)."""
    offset = journal_offset(path, max_bytes, since)
    with open(path, "rb") as stream:
        stream.seek(offset)
        if offset > 0:
            stream.readline()
        for raw in stream:
            if not raw.endswith(b"\n"):
                break  # a line still being written
            try:
                yield json.loads(raw)
            except ValueError:
                continue


def readout(root, run, decision_path):
    decision = read(decision_path)
    require(decision.get("schema") == DECISION_SCHEMA, "decision receipt schema")
    nonce = "hold-release-" + sha(decision_path)[:16]
    ack = None
    verdict = None
    last_accepted = None
    for event in journal_events(run / "learning/journal.jsonl"):
        kind = event.get("event")
        if ack is None:
            if kind == "diagnostic_accepted":
                last_accepted = dict(
                    checkpoint_sha256=event.get("checkpoint_sha256"),
                    receipt=event.get("receipt"),
                    timestamp=event.get("timestamp"),
                )
            if (
                kind == "diagnostic_hold_acknowledged"
                and (event.get("acknowledgement") or {}).get("nonce") == nonce
            ):
                ack = event
            continue
        if kind in ("diagnostic_accepted", "diagnostic_rejected"):
            verdict = event
            break
    require(ack is not None, "no diagnostic_hold_acknowledged event with nonce %s" % nonce)
    if verdict is None:
        return dict(
            schema=SCHEMA,
            status="pending",
            decision=dict(path=str(decision_path), sha256=sha(decision_path)),
            acknowledgement=ack,
            last_accepted_before_release=last_accepted,
            final_tests_opened=False,
        )
    receipt_path = root / verdict["receipt"] if isinstance(verdict.get("receipt"), str) else None
    receipt = read(receipt_path) if receipt_path and receipt_path.is_file() else None
    state = read(run / "learning/state.json")
    guard = (state or {}).get("training_diagnostic") or {}
    outcome = "continue" if verdict["event"] == "diagnostic_accepted" else "rollback_required"
    return dict(
        schema=SCHEMA,
        status="decided",
        outcome=outcome,
        rule="F11: first monitored checkpoint after release accepted -> continue under sampler A; rejected -> stop via automatic_training:false and re-parent to the last diagnostic_accepted checkpoint",
        decision=dict(path=str(decision_path), sha256=sha(decision_path)),
        acknowledgement=dict(timestamp=ack.get("timestamp"), nonce=nonce),
        verdict=dict(
            event=verdict["event"],
            timestamp=verdict.get("timestamp"),
            checkpoint_sha256=verdict.get("checkpoint_sha256"),
            receipt=verdict.get("receipt"),
            receipt_sha256=sha(receipt_path) if receipt_path and receipt_path.is_file() else None,
        ),
        monitor={
            k: (receipt or {}).get(k)
            for k in ("evaluated_step", "sequence_nll", "ranking_bt", "ranking_listwise", "runtime_huber")
        }
        if receipt
        else None,
        guard=dict(
            baseline_ranking_bt=guard.get("baseline_ranking_bt"),
            ranking_bt=guard.get("ranking_bt"),
            consecutive_rejections=guard.get("consecutive_rejections"),
            held=guard.get("held"),
            last_step=guard.get("last_step"),
        ),
        reparent_to=last_accepted if outcome == "rollback_required" else None,
        final_tests_opened=False,
    )


def checkpoint_sha(root, path):
    """A learning checkpoint's identity is the sha256 of its state.json (the journal's checkpoint_sha256)."""
    if not isinstance(path, str):
        return None
    directory = Path(path) if path.startswith("/") else root / path
    state = directory / "state.json"
    return sha(state) if state.is_file() else None


def active_manifest(root, run, manifest_sha256, since=None):
    """The manifest file the controller is running (state.manifest_sha256), found by digest: the journal's revision receipts
    name it first (manifest_revision_committed.receipt -> migration receipt), then any *manifest*.json under the run.
    With `since`, only the bounded journal tail back to that timestamp is read (revisions are recent by construction)."""
    candidates = []
    journal = run / "learning/journal.jsonl"
    if journal.is_file():
        for event in journal_tail(journal, since=since) if since is not None else journal_events(journal):
            if (
                event.get("event") == "manifest_revision_committed"
                and event.get("new_manifest_sha256") == manifest_sha256
                and isinstance(event.get("receipt"), str)
            ):
                receipt_path = root / event["receipt"]
                if receipt_path.is_file():
                    receipt = read(receipt_path)
                    for key in ("manifest", "new_manifest", "path", "output"):
                        value = receipt.get(key)
                        if isinstance(value, dict):
                            value = value.get("path")
                        if isinstance(value, str):
                            candidates.append(root / value)
    candidates += (
        sorted(run.glob("recovery-manifest-v*.json"))
        + sorted(run.glob("*manifest*.json"))
        + sorted(run.glob("*/*manifest*.json"))
    )
    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.stat().st_size < (64 << 20)
            and sha(candidate) == manifest_sha256
        ):  # live manifests are ~6 MB
            return dict(path=str(candidate.relative_to(root)), sha256=manifest_sha256, value=read(candidate))
    return None


def verify_rollback(root, run, readout_path):
    """Status of the F11 rollback the read-out receipt demands: pending -> training_disabled -> reparented, plus any deviation.

    A deviation is a train stage started after the verdict from a checkpoint other than the re-parent target (a block
    from the rejected lineage); it is reported, never hidden, and the receipt's rule text says what it means.
    """
    receipt = read(readout_path)
    require(
        receipt.get("schema") == SCHEMA and receipt.get("status") == "decided",
        "%s: not a decided %s" % (readout_path, SCHEMA),
    )
    require(
        receipt.get("outcome") == "rollback_required",
        "%s: outcome is %r, nothing to verify" % (readout_path, receipt.get("outcome")),
    )
    target = (receipt.get("reparent_to") or {}).get("checkpoint_sha256")
    require(isinstance(target, str) and len(target) == 64, "read-out names no re-parent checkpoint")
    verdict_at = float(receipt["verdict"]["timestamp"])
    rejected = receipt["verdict"]["checkpoint_sha256"]
    state = read(run / "learning/state.json")
    learning = state.get("learning_checkpoint")
    learning_sha = checkpoint_sha(root, learning)
    manifest = active_manifest(root, run, state.get("manifest_sha256"), since=verdict_at)
    automatic = (manifest or {}).get("value", {}).get("automatic_training") if manifest else None
    automatic_source = "manifest" if manifest else None
    revisions, stops, deviations, trains = [], [], [], []
    journal = run / "learning/journal.jsonl"
    window_bytes = os.path.getsize(journal) - journal_offset(
        journal, since=verdict_at
    )  # bounded: back to the verdict, not the whole file
    for event in journal_tail(journal, since=verdict_at):
        if float(event.get("timestamp") or 0) <= verdict_at:
            continue
        kind = event.get("event")
        if kind == "manifest_revision_committed":
            row = dict(
                timestamp=event.get("timestamp"),
                new_manifest_sha256=event.get("new_manifest_sha256"),
                receipt=event.get("receipt"),
            )
            receipt_path = root / event["receipt"] if isinstance(event.get("receipt"), str) else None
            if receipt_path and receipt_path.is_file():
                receipt = read(receipt_path)
                row.update(
                    receipt_sha256=sha(receipt_path),
                    schema=receipt.get("schema"),
                    training_disabled=receipt.get("training_disabled"),
                    learning_head_reparented=receipt.get("learning_head_reparented"),
                    rejected_checkpoint_sha256=receipt.get("rejected_checkpoint_sha256"),
                )
                # The migration receipt is the revision's own statement; it stands in when the manifest file is not locatable.
                if (
                    event.get("new_manifest_sha256") == state.get("manifest_sha256")
                    and manifest is None
                    and receipt.get("training_disabled") is True
                ):
                    automatic, automatic_source = False, "migration receipt " + str(event["receipt"])
            revisions.append(row)
        elif kind == "controller_stopped":
            stops.append(dict(timestamp=event.get("timestamp"), event=kind, resumable=event.get("resumable")))
        elif kind == "operator_pause_committed":
            stops.append(
                dict(
                    timestamp=event.get("timestamp"),
                    event=kind,
                    reason=event.get("reason"),
                    evidence=event.get("evidence"),
                    training_restart_authorized=event.get("training_restart_authorized"),
                    control_sha256=event.get("control_sha256"),
                )
            )
        elif kind == "stage_started" and event.get("kind") == "train":
            parent = next(
                (
                    arg
                    for arg in event.get("command") or []
                    if isinstance(arg, str) and "/checkpoints/step-" in arg
                ),
                None,
            )
            parent_sha = checkpoint_sha(root, parent)
            row = dict(timestamp=event.get("timestamp"), parent=parent, parent_sha256=parent_sha)
            trains.append(row)
            if parent_sha != target:
                deviations.append(
                    dict(
                        row,
                        rule="f11_deviation: a block trained from a checkpoint other than the re-parent target after the rollback verdict",
                    )
                )
    paused = any(
        s["event"] == "operator_pause_committed" and s.get("training_restart_authorized") is False
        for s in stops
    )
    if learning_sha == target:
        status = "reparented"
    elif automatic is False:
        status = "training_disabled"
    elif paused:
        status = "paused"  # the operator pause holds training, but the manifest still authorizes it: the revision is still owed
    else:
        status = "pending"
    return dict(
        schema=VERIFICATION_SCHEMA,
        status=status,
        readout=dict(path=str(readout_path), sha256=sha(readout_path)),
        rejected_checkpoint_sha256=rejected,
        reparent_target_sha256=target,
        learning_checkpoint=dict(
            path=learning,
            sha256=learning_sha,
            is_target=learning_sha == target,
            is_rejected=learning_sha == rejected,
        ),
        learning_corpus=state.get("learning_corpus"),
        manifest=dict(
            path=(manifest or {}).get("path"),
            sha256=state.get("manifest_sha256"),
            automatic_training=automatic,
            automatic_training_source=automatic_source,
        ),
        guard={
            k: ((state.get("training_diagnostic") or {}).get(k))
            for k in ("consecutive_rejections", "held", "last_step")
        },
        revisions_after_verdict=revisions,
        controller_stops_after_verdict=stops,
        train_stages_after_verdict=trains,
        deviations=deviations,
        journal_window_bytes=window_bytes,
        rule="F11: rollback is complete when learning_checkpoint is the re-parent target; until then automatic_training must be false (an operator pause with training_restart_authorized false is the interim stop); any train stage from another checkpoint after the verdict is an f11_deviation",
        final_tests_opened=False,
    )


VERIFICATION_SCHEMA = "parallel.rollback_verification.v1"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", default="runs/eight-a100-recovery-20260912")
    parser.add_argument("--decision", type=Path, help="hold-release decision receipt (read-out mode)")
    parser.add_argument(
        "--verify-rollback",
        type=Path,
        metavar="READOUT",
        help="verify the rollback demanded by this read-out receipt instead of reading out",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="read-out: immutable receipt (written only when decided); verify: atomic status file (overwritten)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.verify_rollback is not None:
        result = verify_rollback(root, root / args.run, args.verify_rollback)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return result
    require(args.decision is not None, "--decision is required unless --verify-rollback is given")
    result = readout(root, root / args.run, args.decision)
    if result["status"] == "decided" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(result, receipt_sha256=promotion_gate.immutable(args.output, result))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
