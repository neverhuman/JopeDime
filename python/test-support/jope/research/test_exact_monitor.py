#!/usr/bin/env python3
"""Persistent control-plane monitor for bootstrap, refill, ingest, and training."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from test_exact_common import atomic_json, sha256_file, validate_catalog_manifest

from test_exact_monitor_config import (
    CAMPAIGN, CORPUS, EVALUATOR, LOCK, PROPOSALS, RESERVE_FLOOR, ROOT, SCQ, SCRIPTS,
)


from test_exact_monitor_bootstrap import bootstrap, fetch_proposals, run, scheduler_jobs


def strict_label_count() -> int:
    if not CORPUS.is_file():
        return 0
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    count = int(con.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    con.close()
    return count


def corpus_metrics() -> dict:
    if not CORPUS.is_file():
        return {
            "strict_labels": 0,
            "exact_labels_per_minute": 0.0,
            "evaluator_failures": 0,
            "evaluator_failures_per_minute": 0.0,
            "receipt_ingestion_lag_seconds": None,
            "dl_attributed_wins": 0,
        }
    con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
    now_ms = int(time.time() * 1000)
    total = int(con.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    recent = int(
        con.execute(
            "SELECT COUNT(*) FROM observations WHERE created_at_ms>=?",
            (now_ms - 600_000,),
        ).fetchone()[0]
    )
    failures = int(con.execute("SELECT COUNT(*) FROM failures").fetchone()[0])
    recent_failures = int(
        con.execute(
            "SELECT COUNT(*) FROM failures WHERE created_at_ms>=?",
            (now_ms - 600_000,),
        ).fetchone()[0]
    )
    latest = con.execute("SELECT MAX(created_at_ms) FROM observations").fetchone()[0]
    dl_wins = 0
    wins_path = ROOT / "indexes" / "strict-wins.json"
    if wins_path.is_file():
        winning_semantics = {
            row["semantic_sha256"] for row in json.loads(wins_path.read_text())
        }
        if winning_semantics:
            placeholders = ",".join("?" for _ in winning_semantics)
            dl_wins = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM observations
                    WHERE candidate_source='accepted_dl_checkpoint'
                      AND semantic_sha256 IN ({placeholders})
                    """,
                    tuple(sorted(winning_semantics)),
                ).fetchone()[0]
            )
    con.close()
    return {
        "strict_labels": total,
        "exact_labels_per_minute": recent / 10.0,
        "evaluator_failures": failures,
        "evaluator_failures_per_minute": recent_failures / 10.0,
        "receipt_ingestion_lag_seconds": (
            None if latest is None else max(0.0, (now_ms - int(latest)) / 1000.0)
        ),
        "dl_attributed_wins": dl_wins,
    }


def maybe_submit_trainer() -> dict:
    labels = strict_label_count()
    jobs = scheduler_jobs()
    trainers = [
        row
        for job_id, row in jobs.items()
        if job_id.startswith("jope-testonly-train-")
        and row.get("state")
        in {"queued", "starting", "dispatching", "running", "preempting", "held"}
    ]
    if trainers:
        trainer_id = trainers[0]["id"]
        watermark = int(trainer_id.rsplit("-", 1)[-1])
        return {
            "labels": labels,
            "trainer": trainer_id,
            "active_watermark": watermark,
            "labels_since_active_watermark": max(0, labels - watermark),
            "next_watermark": watermark + 50_000,
            "submitted": False,
        }
    prior_watermarks = [
        int(job_id.rsplit("-", 1)[-1])
        for job_id in jobs
        if job_id.startswith("jope-testonly-train-")
        and job_id.rsplit("-", 1)[-1].isdigit()
    ]
    last = max(prior_watermarks, default=0)
    threshold = 1_000 if last == 0 else last + 50_000
    if labels < threshold:
        return {"labels": labels, "next_watermark": threshold, "submitted": False}
    build = run(
        [
            sys.executable,
            str(SCRIPTS / "test_exact_train.py"),
            "build-corpus",
            "--min-rows",
            str(threshold),
        ],
        timeout=3600,
    )
    if build.returncode:
        return {
            "labels": labels,
            "next_watermark": threshold,
            "submitted": False,
            "corpus_not_ready": (build.stderr or build.stdout).strip()[-1000:],
        }
    pointer = json.loads((ROOT / "training" / "current-corpus.json").read_text())
    render = run(
        [
            sys.executable,
            str(SCRIPTS / "test_exact_train.py"),
            "render-spec",
            "--corpus-manifest",
            pointer["manifest"],
        ]
    )
    if render.returncode:
        raise RuntimeError((render.stderr or render.stdout).strip())
    spec = json.loads(render.stdout)
    check = run([str(SCQ), "check", spec["spec"]])
    if check.returncode:
        raise RuntimeError((check.stderr or check.stdout).strip())
    submit = run([str(SCQ), "submit", spec["spec"]])
    if submit.returncode and "already exists" not in (submit.stderr or submit.stdout).lower():
        raise RuntimeError((submit.stderr or submit.stdout).strip())
    return {"labels": labels, "trainer": spec["job_id"], "submitted": True}


def trainer_inventory() -> dict:
    labels = strict_label_count()
    live_states = {
        "queued",
        "starting",
        "dispatching",
        "running",
        "preempting",
        "held",
    }
    active = [
        row
        for job_id, row in scheduler_jobs().items()
        if (
            job_id.startswith("jope-testonly-train-")
            or job_id.startswith("jope-selfplay-train-")
        )
        and row.get("state") in live_states
    ]
    active.sort(key=lambda row: row["id"])
    return {
        "labels": labels,
        "trainer": active[0]["id"] if active else None,
        "active_trainers": [row["id"] for row in active],
        "legacy_diagnostic_only": [
            row["id"]
            for row in active
            if row["id"].startswith("jope-testonly-train-")
        ],
        "submitted": False,
    }


def active_trainer_metrics(job: dict | None) -> dict | None:
    if not job or not job.get("run_dir"):
        return None
    attempt = int(job.get("attempt") or 0)
    roots = sorted(
        (
            Path(job["run_dir"])
            / "attempts"
            / str(attempt)
            / "outputs"
            / "training"
        ).glob("watermark-*")
    )
    if not roots:
        return None
    metrics_path = roots[-1] / "train_metrics.jsonl"
    progress_path = roots[-1] / "train_progress.json"
    rows = []
    if metrics_path.is_file():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line.strip()
        ]
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else None
    if not rows and progress is None:
        return None
    result = {"progress": progress}
    if rows:
        latest = rows[-1]
        pairwise = [
            (float(row["val_pairwise_win_accuracy"]), int(row["epoch"]))
            for row in rows
            if row.get("val_pairwise_win_accuracy") is not None
        ]
        best_pairwise, best_epoch = max(pairwise) if pairwise else (None, None)
        result.update(
            {
                "completed_epochs": len(rows),
                "latest_epoch": latest.get("epoch"),
                "latest_phase": latest.get("phase"),
                "latest_pairwise_accuracy": latest.get(
                    "val_pairwise_win_accuracy"
                ),
                "best_pairwise_accuracy": best_pairwise,
                "best_pairwise_epoch": best_epoch,
                "pairwise_delta_from_epoch_1": (
                    None
                    if not pairwise
                    else float(latest["val_pairwise_win_accuracy"]) - pairwise[0][0]
                ),
                "latest_val_loss": latest.get("val_loss"),
                "latest_overfit_gap": latest.get("overfit_gap"),
                "latest_task_pairwise_accuracy": {
                    task: latest.get(f"val_{task}_pairwise_win_accuracy")
                    for task in ("regression", "binary")
                    if latest.get(f"val_{task}_pairwise_win_accuracy") is not None
                },
            }
        )
    return result


def metrics(phase: dict, trainer: dict) -> dict:
    jobs = scheduler_jobs()
    eval_jobs = [
        row for job_id, row in jobs.items() if job_id.startswith("jte-")
    ]
    live = [
        row
        for row in eval_jobs
        if row.get("state")
        in {"queued", "starting", "dispatching", "running", "preempting", "held"}
        and not (
            row.get("state") == "dispatching"
            and row.get("pid") is None
            and int(row.get("runtime_secs") or 0) > 120
        )
    ]
    stale_dispatching = sum(
        row.get("state") == "dispatching"
        and row.get("pid") is None
        and int(row.get("runtime_secs") or 0) > 120
        for row in eval_jobs
    )
    nodes = run([str(SCQ), "--json", "nodes"])
    gpus = (
        json.loads(nodes.stdout).get("Nodes", {}).get("gpus", [])
        if nodes.returncode == 0
        else []
    )
    corpus_status = corpus_metrics()
    train_results = sorted((ROOT / "training").glob("**/train-result.json"))
    validation_lift = None
    if train_results:
        latest_train = json.loads(train_results[-1].read_text())
        validation_lift = latest_train.get("best_metric")
    trainer_job = jobs.get(trainer.get("trainer"))
    active_training = active_trainer_metrics(trainer_job)
    if active_training and active_training.get("latest_pairwise_accuracy") is not None:
        validation_lift = active_training["latest_pairwise_accuracy"]
    result = {
        "schema": "JopeTestOnlyOperationalStatus.v1",
        "updated_at_ms": int(time.time() * 1000),
        "phase": phase,
        "queue_depth": len(live),
        "stale_dispatching": stale_dispatching,
        "evaluation_states": {
            state: sum(row.get("state") == state for row in eval_jobs)
            for state in sorted({row.get("state") for row in eval_jobs})
        },
        **corpus_status,
        "latest_validation_metric": validation_lift,
        "trainer": trainer,
        "active_training": active_training,
        "xbabe2_gpu": next(
            (gpu for gpu in gpus if gpu.get("node") == "xbabe2"), None
        ),
    }
    atomic_json(ROOT / "ops" / "status.json", result)
    return result


def cycle() -> dict:
    phase = bootstrap()
    ingest = run(
        [
            sys.executable,
            str(SCRIPTS / "test_exact_ingest.py"),
            "reconcile",
            "--limit",
            "200",
        ],
        timeout=3600,
    )
    ingest_result = (
        json.loads(ingest.stdout)
        if ingest.returncode == 0 and ingest.stdout.strip()
        else {"error": (ingest.stderr or ingest.stdout).strip()[-1000:]}
    )
    controller = run(
        [sys.executable, str(SCRIPTS / "test_exact_controller.py")],
        timeout=3600,
    )
    controller_result = (
        json.loads(controller.stdout)
        if controller.returncode == 0 and controller.stdout.strip()
        else {
            "health_gate_stopped_admissions": True,
            "error": (controller.stderr or controller.stdout).strip()[-2000:],
        }
    )
    trainer = trainer_inventory()
    status = metrics(
        {**phase, "ingest": ingest_result, "flywheel": controller_result},
        trainer,
    )
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("test-only monitor is already running")
        while True:
            try:
                print(json.dumps(cycle(), sort_keys=True), flush=True)
            except Exception as error:
                atomic_json(
                    ROOT / "ops" / "monitor-error.json",
                    {
                        "schema": "JopeTestOnlyMonitorError.v1",
                        "at_ms": int(time.time() * 1000),
                        "error": str(error),
                    },
                )
                print(json.dumps({"error": str(error)}, sort_keys=True), flush=True)
            if args.once:
                break
            time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
