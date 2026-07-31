#!/usr/bin/env python3
"""Durable single-writer controller for the internal self-improvement loop."""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from test_exact_common import (
    atomic_json,
    sha256_file,
    sha256_text,
    validate_catalog_counts,
    validate_catalog_manifest,
)

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
SCQ_RUNS = Path("/home/ubuntu/smart_cluster/var/runs")
PHASES = (
    "freeze_corpus",
    "build_packed_corpus",
    "launch_checkpoint_tournament",
    "fetch_candidate_checkpoints",
    "generate_gate_proposals",
    "evaluate_frozen_gate",
    "build_acceptance_evidence",
    "run_checkpoint_gate",
    "advance_accepted_pointer",
    "seed_next_strict_wave",
    "freeze_winner_scan",
    "run_sotarena_qualification",
)
TERMINAL_PHASES = {"accepted", "rejected"}


class PendingAction(RuntimeError):
    """The action is healthy but its external result is not complete yet."""


def action_key(
    *,
    action: str,
    corpus_watermark: int,
    accepted_ancestor: str,
    recipe: str,
    seed: int,
    catalog_sha256: str,
    evaluator_sha256: str,
) -> str:
    material = {
        "action": action,
        "corpus_watermark": corpus_watermark,
        "accepted_ancestor": accepted_ancestor,
        "recipe": recipe,
        "seed": seed,
        "catalog_sha256": catalog_sha256,
        "evaluator_sha256": evaluator_sha256,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ActionJournal:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "ops" / "controller.sqlite"
        self.events = root / "ops" / "controller-events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS actions(
              idempotency_key TEXT PRIMARY KEY,
              action TEXT NOT NULL,
              status TEXT NOT NULL,
              request_json TEXT NOT NULL,
              result_json TEXT,
              attempts INTEGER NOT NULL,
              updated_at_ms INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _event(self, key: str, action: str, status: str, payload: dict) -> None:
        event = json.dumps(
            {
                "schema": "JopeSelfplayControllerEvent.v1",
                "idempotency_key": key,
                "action": action,
                "status": status,
                "at_ms": int(time.time() * 1000),
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode() + b"\n"
        descriptor = os.open(
            self.events, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            os.write(descriptor, event)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def perform(
        self,
        key: str,
        action: str,
        request: dict,
        callback: Callable[[dict], dict],
    ) -> dict:
        existing = self.db.execute(
            "SELECT * FROM actions WHERE idempotency_key=?", (key,)
        ).fetchone()
        if existing is not None and existing["status"] == "succeeded":
            return json.loads(existing["result_json"])
        attempts = 1 if existing is None else int(existing["attempts"]) + 1
        now = int(time.time() * 1000)
        self.db.execute(
            """
            INSERT INTO actions VALUES(?,?,?,?,?,?,?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
              status=excluded.status,request_json=excluded.request_json,
              attempts=excluded.attempts,updated_at_ms=excluded.updated_at_ms
            """,
            (
                key,
                action,
                "running",
                json.dumps(request, sort_keys=True),
                None,
                attempts,
                now,
            ),
        )
        self.db.commit()
        self._event(key, action, "started", {"attempt": attempts})
        try:
            result = callback(request)
        except PendingAction as error:
            self.db.execute(
                "UPDATE actions SET status='pending',updated_at_ms=? WHERE idempotency_key=?",
                (int(time.time() * 1000), key),
            )
            self.db.commit()
            self._event(key, action, "pending", {"detail": str(error)})
            raise
        except Exception as error:
            self.db.execute(
                "UPDATE actions SET status='failed',updated_at_ms=? WHERE idempotency_key=?",
                (int(time.time() * 1000), key),
            )
            self.db.commit()
            self._event(key, action, "failed", {"error": str(error)})
            raise
        self.db.execute(
            """
            UPDATE actions SET status='succeeded',result_json=?,updated_at_ms=?
            WHERE idempotency_key=?
            """,
            (
                json.dumps(result, sort_keys=True),
                int(time.time() * 1000),
                key,
            ),
        )
        self.db.commit()
        self._event(key, action, "succeeded", result)
        return result


class FlywheelController:
    def __init__(
        self,
        root: Path,
        handlers: dict[str, Callable[[dict], dict]],
        *,
        label_threshold: int = 50_000,
        seed: int = 20260726,
    ):
        self.root = root
        self.handlers = handlers
        self.label_threshold = label_threshold
        self.seed = seed
        self.state_path = root / "ops" / "flywheel-state.json"
        self.journal = ActionJournal(root)

    def close(self) -> None:
        self.journal.close()

    def _state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {
            "schema": "JopeSelfplayControllerState.v1",
            "last_frozen_watermark": 0,
            "consecutive_rejections": 0,
            "tournament": None,
        }

    def cycle(self, snapshot: dict) -> dict:
        counts = validate_catalog_counts(
            snapshot["catalog_entries"], snapshot["catalog_counts"]
        )
        state = self._state()
        tournament = state.get("tournament")
        if (
            tournament is not None
            and int(tournament.get("phase_index", 0)) == 0
            and not tournament.get("artifacts")
            and len(tournament.get("recipes") or []) != 1
        ):
            # Legacy unstarted tournaments used three competing recipes. Preserve
            # their frozen watermark, but normalize them to the single ordinary
            # continuation lane unless the rejection policy requires consolidation.
            full_consolidation = int(state.get("consecutive_rejections", 0)) >= 5
            tournament["recipes"] = ["a" if full_consolidation else "c"]
            tournament["corpus_mode"] = (
                "full_consolidation"
                if full_consolidation
                else "recent_plus_replay"
            )
            state["consecutive_rejections"] = int(
                state.get("consecutive_rejections", 0)
            )
            atomic_json(self.state_path, state)
        if tournament is None:
            labels = int(snapshot["valid_labels"])
            if labels - int(state["last_frozen_watermark"]) < self.label_threshold:
                return {
                    "phase": "collecting_labels",
                    "valid_labels": labels,
                    "next_watermark": int(state["last_frozen_watermark"])
                    + self.label_threshold,
                    "catalog_counts": counts,
                }
            full_consolidation = (
                labels // 500_000
                > int(state.get("last_frozen_watermark", 0)) // 500_000
                or int(state.get("consecutive_rejections", 0)) >= 5
            )
            tournament = {
                "watermark": labels,
                "accepted_ancestor": snapshot["accepted_ancestor"],
                "catalog_sha256": snapshot["catalog_sha256"],
                "evaluator_sha256": snapshot["evaluator_sha256"],
                "phase_index": 0,
                "artifacts": {},
                "recipes": ["a" if full_consolidation else "c"],
                "corpus_mode": (
                    "full_consolidation"
                    if full_consolidation
                    else "recent_plus_replay"
                ),
                "seed": self.seed,
            }
            state["last_frozen_watermark"] = labels
            state["tournament"] = tournament
            atomic_json(self.state_path, state)

        phase = PHASES[int(tournament["phase_index"])]
        request = {
            **tournament,
            "action": phase,
            "artifacts": tournament["artifacts"],
        }
        key = action_key(
            action=phase,
            corpus_watermark=int(tournament["watermark"]),
            accepted_ancestor=tournament["accepted_ancestor"],
            recipe="tournament" if phase != "launch_checkpoint_tournament" else "a+b+c",
            seed=int(tournament["seed"]),
            catalog_sha256=tournament["catalog_sha256"],
            evaluator_sha256=tournament["evaluator_sha256"],
        )
        handler = self.handlers.get(phase)
        if handler is None:
            raise RuntimeError(f"controller has no handler for {phase}")
        try:
            result = self.journal.perform(key, phase, request, handler)
        except PendingAction as error:
            return {
                "phase": phase,
                "pending": True,
                "detail": str(error),
                "watermark": tournament["watermark"],
            }
        tournament["artifacts"][phase] = result
        if phase == "run_checkpoint_gate" and result.get("pass") is False:
            tournament["artifacts"]["advance_accepted_pointer"] = {
                "advanced": False,
                "checkpoint_sha256": tournament["accepted_ancestor"],
            }
            tournament["phase_index"] += 2
        else:
            tournament["phase_index"] += 1
        if tournament["phase_index"] >= len(PHASES):
            gate = tournament["artifacts"]["run_checkpoint_gate"]
            terminal = "accepted" if gate.get("pass") else "rejected"
            completed = {
                **tournament,
                "terminal": terminal,
                "completed_at_ms": int(time.time() * 1000),
            }
            completed_path = (
                self.root
                / "training"
                / "tournaments"
                / f"watermark-{int(tournament['watermark']):09d}.json"
            )
            atomic_json(completed_path, completed)
            state["tournament"] = None
            state["last_terminal"] = terminal
            state["last_tournament"] = str(completed_path)
            state["consecutive_rejections"] = (
                0
                if terminal == "accepted"
                else int(state.get("consecutive_rejections", 0)) + 1
            )
        atomic_json(self.state_path, state)
        return {
            "phase": phase,
            "completed": True,
            "result": result,
            "watermark": tournament["watermark"],
        }


def _file_handler(name: str, root: Path) -> Callable[[dict], dict]:
    """Fail-closed production adapter for externally produced phase evidence."""

    def handler(request: dict) -> dict:
        path = (
            root
            / "training"
            / "tournament-work"
            / f"watermark-{int(request['watermark']):09d}"
            / f"{name}.json"
        )
        if not path.is_file():
            raise PendingAction(f"waiting for {path}")
        value = json.loads(path.read_text())
        value["artifact_path"] = str(path)
        value["artifact_sha256"] = sha256_file(path)
        return value

    return handler


def _scq_json(*args: str) -> dict:
    scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
    result = subprocess.run(
        [str(scq), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _freeze_corpus(request: dict) -> dict:
    """Freeze the corpus AT THE TOURNAMENT'S watermark and describe THAT corpus.

    This used to read `training/current-corpus.json` and return whatever the
    global pointer named. That was only ever correct while the tournament was
    the sole producer of the pointer, which is no longer true: the standalone
    freezer (`jope_selfplay_corpus_freezer`) advances it at the live COUNT(*)
    precisely so a parked tournament can no longer starve the DL trainer.

    With two producers the old read is a silent corpus swap. The downstream
    phases consume `artifacts.freeze_corpus.manifest` verbatim —
    `_build_packed_corpus` (:404) packs it and `_launch_checkpoint_tournament`
    (:427), plus `jope_selfplay_promotion.rs:1301-1310`, render the trainer spec
    from it — so a pointer that had moved on would train the candidate on a
    DIFFERENT corpus from the one the tournament pinned, gate it against that
    corpus's evidence, and record the tournament's watermark against it anyway.

    The snapshot directory is the authority instead. `build_corpus` writes
    `train`/`validation` inside its own snapshot (test_exact_train.py:684-720),
    so the manifest it just returned locates itself, at the tournament's
    watermark, whether it was built now or served from the snapshot cache.
    """
    from test_exact_train import build_corpus

    watermark = int(request["watermark"])
    manifest = build_corpus(
        watermark,
        freeze_watermark=watermark,
        corpus_mode=request.get("corpus_mode", "recent_plus_replay"),
    )
    if int(manifest["source_label_watermark"]) != watermark:
        raise RuntimeError("corpus watermark changed while the tournament was freezing")
    manifest_path = Path(manifest["train"]).parent / "corpus-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"frozen corpus manifest is absent: {manifest_path}")
    frozen = json.loads(manifest_path.read_text())
    if int(frozen["source_label_watermark"]) != watermark:
        raise RuntimeError(
            f"{manifest_path} is watermark {frozen['source_label_watermark']}, "
            f"not the tournament's {watermark}"
        )
    packed = ensure_packed_manifest(manifest_path)
    pointer_path = ROOT / "training" / "current-corpus.json"
    pointer = json.loads(pointer_path.read_text())
    if pointer["manifest_sha256"] != sha256_file(pointer["manifest"]):
        raise RuntimeError("frozen corpus pointer failed hash verification")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packed_manifest": str(packed),
        "packed_manifest_sha256": sha256_file(packed),
        "watermark": watermark,
        "pointer": str(pointer_path),
        "pointer_sha256": sha256_file(pointer_path),
        # The global pointer is REPORTED, never followed. When a standalone
        # freeze has run ahead these two differ, and that is healthy.
        "pointer_watermark": int(pointer.get("watermark") or 0),
        "pointer_manifest": pointer["manifest"],
    }


def ensure_packed_manifest(manifest_path: Path) -> Path:
    """Path of the packed corpus bound to ``manifest_path``, materialising it.

    `_build_packed_corpus` is a separate phase, but `freeze_corpus` must still
    name a packed manifest for the tournament's OWN corpus — the pointer's
    packed manifest belongs to whatever the pointer names, which is no longer
    guaranteed to be the same corpus.
    """
    from test_exact_packed_corpus import materialize

    packed = materialize(manifest_path, ROOT / "training" / "packed")
    return (
        ROOT / "training" / "packed" / packed["binding_sha256"] / "manifest.json"
    )


def _build_packed_corpus(request: dict) -> dict:
    from test_exact_packed_corpus import materialize

    frozen = request["artifacts"]["freeze_corpus"]
    packed = materialize(
        Path(frozen["manifest"]),
        ROOT / "training" / "packed",
    )
    path = (
        ROOT
        / "training"
        / "packed"
        / packed["binding_sha256"]
        / "manifest.json"
    )
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "binding_sha256": packed["binding_sha256"],
        "row_count": packed["row_count"],
    }


def _launch_checkpoint_tournament(request: dict) -> dict:
    from test_exact_train import render_spec

    corpus = Path(request["artifacts"]["freeze_corpus"]["manifest"])
    packed = Path(request["artifacts"]["build_packed_corpus"]["manifest"])
    jobs = []
    for offset, recipe in enumerate(request["recipes"]):
        rendered = render_spec(
            corpus,
            recipe=recipe,
            seed=int(request["seed"]) + offset,
            packed_manifest=packed,
        )
        if rendered["accepted_ancestor_sha256"] != request["accepted_ancestor"]:
            raise RuntimeError("accepted checkpoint changed before tournament dispatch")
        scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
        checked = subprocess.run(
            [str(scq), "check", rendered["spec"]],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        if checked.returncode:
            raise RuntimeError((checked.stderr or checked.stdout).strip())
        submit = subprocess.run(
            [str(scq), "submit", rendered["spec"]],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        detail = (submit.stderr or submit.stdout).strip()
        if submit.returncode and "already exists" not in detail.lower():
            raise RuntimeError(detail)
        jobs.append({**rendered, "spec_sha256": sha256_file(rendered["spec"])})
    return {"jobs": jobs, "job_count": len(jobs)}


def _scheduler_jobs() -> dict[str, dict]:
    value = _scq_json("--json", "ls")
    return {
        row["id"]: row
        for row in value.get("Jobs", {}).get("jobs", [])
        if row.get("id")
    }


def _pruned_job_outputs(job_id: str) -> Path | None:
    """Locate outputs for a job the scheduler has already pruned from its records.

    Terminal job records are evicted from `scq ls` after a retention window, so a
    tournament job that completed while no controller was running becomes invisible
    to the scheduler even though its run directory (and outputs) remain on disk.
    """
    attempts = Path("/home/ubuntu/smart_cluster/var/runs") / job_id / "attempts"
    if not attempts.is_dir():
        return None
    outputs = sorted(
        (path / "outputs" for path in attempts.iterdir() if (path / "outputs").is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for candidate in reversed(outputs):
        if any(candidate.rglob("train-result.json")):
            return candidate
    return None


def _fetch_candidate_checkpoints(request: dict) -> dict:
    jobs = _scheduler_jobs()
    launched = request["artifacts"]["launch_checkpoint_tournament"]["jobs"]
    pruned_outputs = {
        item["job_id"]: _pruned_job_outputs(item["job_id"])
        for item in launched
        if item["job_id"] not in jobs
    }
    pending = [
        item["job_id"]
        for item in launched
        if jobs.get(item["job_id"], {}).get("state") != "completed"
        and not pruned_outputs.get(item["job_id"])
    ]
    failed = {
        item["job_id"]: jobs.get(item["job_id"], {}).get("state")
        for item in launched
        if jobs.get(item["job_id"], {}).get("state") in {"failed", "cancelled"}
    }
    if failed:
        raise RuntimeError(f"checkpoint tournament jobs failed: {failed}")
    if pending:
        raise PendingAction(f"waiting for checkpoint jobs: {','.join(pending)}")
    root = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "fetched"
    )
    root.mkdir(parents=True, exist_ok=True)
    candidates = []
    scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
    for item in launched:
        destination = root / item["recipe"]
        if not destination.is_dir():
            temporary = root / f".{item['recipe']}.fetch-{os.getpid()}"
            temporary.mkdir(exist_ok=False)
            source = pruned_outputs.get(item["job_id"])
            if source is not None:
                shutil.copytree(source, temporary / "outputs", dirs_exist_ok=True)
            else:
                result = subprocess.run(
                    [str(scq), "fetch", item["job_id"], "--output", str(temporary)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3_600,
                )
                if result.returncode:
                    shutil.rmtree(temporary, ignore_errors=True)
                    raise RuntimeError((result.stderr or result.stdout).strip())
            os.replace(temporary, destination)
        results = list(destination.rglob("train-result.json"))
        if len(results) != 1:
            raise RuntimeError(
                f"{item['job_id']} fetch has {len(results)} train-result files"
            )
        summary = json.loads(results[0].read_text())
        if (
            summary.get("accepted_ancestor_sha256") != request["accepted_ancestor"]
            or summary.get("recipe") != item["recipe"]
            or int(summary.get("seed")) != int(item["seed"])
        ):
            raise RuntimeError(f"{item['job_id']} training provenance drift")
        digest = str(summary.get("checkpoint_sha256") or "")
        checkpoint_matches = sorted(
            [
            path
            for path in destination.rglob("*.model.pt")
            if sha256_file(path) == digest
            ],
            key=lambda path: (path.name != f"{digest}.model.pt", str(path)),
        )
        if not checkpoint_matches:
            raise RuntimeError(
                f"{item['job_id']} has no verified checkpoint"
            )
        immutable = root / "checkpoints" / f"{digest}.model.pt"
        immutable.parent.mkdir(exist_ok=True)
        if not immutable.exists():
            shutil.copy2(checkpoint_matches[0], immutable)
        if sha256_file(immutable) != digest:
            raise RuntimeError("fetched immutable checkpoint hash mismatch")
        candidates.append(
            {
                "recipe": item["recipe"],
                "seed": item["seed"],
                "checkpoint": str(immutable),
                "checkpoint_sha256": digest,
                "train_result": str(results[0]),
                "train_result_sha256": sha256_file(results[0]),
            }
        )
    return {"candidates": candidates, "candidate_count": len(candidates)}


def _fetch_job_output(job_id: str, destination: Path) -> Path:
    if destination.is_dir():
        return destination
    temporary = destination.with_name(f".{destination.name}.fetch-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)
    scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
    result = subprocess.run(
        [str(scq), "fetch", job_id, "--output", str(temporary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=3_600,
    )
    if result.returncode:
        shutil.rmtree(temporary, ignore_errors=True)
        raise RuntimeError((result.stderr or result.stdout).strip())
    os.replace(temporary, destination)
    return destination


def _generate_gate_proposals(request: dict) -> dict:
    """Generate equal-budget checkpoint proposals on frozen validation contexts."""

    from test_exact_ops import render_checkpoint_proposal_spec
    from test_exact_train import accepted_checkpoint

    work = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "gate-proposals"
    )
    work.mkdir(parents=True, exist_ok=True)
    frozen_champions = work / "frozen-champions.json"
    if not frozen_champions.is_file():
        shutil.copy2(ROOT / "indexes" / "champions.json", frozen_champions)
    baseline_path, baseline_hash = accepted_checkpoint()
    if baseline_hash != request["accepted_ancestor"]:
        raise RuntimeError("gate proposal baseline changed after corpus freeze")
    checkpoints = [
        {
            "role": "baseline",
            "checkpoint": str(baseline_path),
            "checkpoint_sha256": baseline_hash,
        },
        *[
            {"role": "candidate", **candidate}
            for candidate in request["artifacts"]["fetch_candidate_checkpoints"][
                "candidates"
            ]
        ],
    ]
    rendered = []
    scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
    jobs = _scheduler_jobs()
    for offset, checkpoint in enumerate(checkpoints):
        digest = checkpoint["checkpoint_sha256"]
        spec = render_checkpoint_proposal_spec(
            work / f"{digest}.toml",
            checkpoint=Path(checkpoint["checkpoint"]),
            split="validation",
            seed=int(request["seed"]),
            proposals_per_dataset=64,
        )
        if spec["checkpoint_sha256"] != digest:
            raise RuntimeError("proposal spec checkpoint content drift")
        state = jobs.get(spec["job_id"], {}).get("state")
        if state in {"failed", "cancelled"}:
            raise RuntimeError(
                f"checkpoint proposal job failed: {spec['job_id']}={state}"
            )
        if state is None:
            checked = subprocess.run(
                [str(scq), "check", spec["spec"]],
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
            if checked.returncode:
                raise RuntimeError((checked.stderr or checked.stdout).strip())
            submitted = subprocess.run(
                [str(scq), "submit", spec["spec"]],
                capture_output=True,
                text=True,
                check=False,
                timeout=900,
            )
            detail = (submitted.stderr or submitted.stdout).strip()
            if submitted.returncode and "already exists" not in detail.lower():
                raise RuntimeError(detail)
        rendered.append({**checkpoint, **spec})
    jobs = _scheduler_jobs()
    pending = [
        row["job_id"]
        for row in rendered
        if jobs.get(row["job_id"], {}).get("state") != "completed"
    ]
    if pending:
        raise PendingAction(f"waiting for gate proposal jobs: {','.join(pending)}")
    proposal_sets = []
    for row in rendered:
        output = _fetch_job_output(
            row["job_id"], work / "fetched" / row["checkpoint_sha256"]
        )
        proposals = list(output.rglob("dl-proposals.jsonl"))
        summaries = list(output.rglob("dl-proposals.summary.json"))
        if len(proposals) != 1 or len(summaries) != 1:
            raise RuntimeError(f"{row['job_id']} proposal fetch is incomplete")
        summary = json.loads(summaries[0].read_text())
        if (
            summary.get("model_checkpoint_sha256") != row["checkpoint_sha256"]
            or summary.get("split") != "validation"
            or int(summary.get("seed")) != int(request["seed"])
            or sha256_file(proposals[0]) != summary.get("output_sha256")
        ):
            raise RuntimeError(f"{row['job_id']} proposal provenance drift")
        proposal_sets.append(
            {
                **row,
                "proposals": str(proposals[0]),
                "proposals_sha256": sha256_file(proposals[0]),
                "summary": str(summaries[0]),
                "summary_sha256": sha256_file(summaries[0]),
                "proposal_validity": float(summary["proposal_validity"]),
                "p95_proposal_latency_ms": float(
                    summary["p95_proposal_latency_ms"]
                ),
            }
        )
    return {
        "development_split": "validation",
        "proposal_budget_per_dataset": 64,
        "seed": int(request["seed"]),
        "proposal_sets": proposal_sets,
        "frozen_champions": str(frozen_champions),
        "frozen_champions_sha256": sha256_file(frozen_champions),
    }


def _family_held_accuracy(
    dataset_wins: dict[tuple[str, str], float],
    entries: dict[tuple[str, str], dict],
    task: str,
) -> float:
    by_family: dict[str, list[float]] = collections.defaultdict(list)
    for key, value in dataset_wins.items():
        if key[0] == task:
            by_family[entries[key]["source_family"]].append(value)
    if not by_family:
        return 0.0
    return statistics.fmean(
        statistics.fmean(values) for values in by_family.values()
    )


# One gate lane is 2 checkpoints x |validation cohort| manifests (1,304 at the
# 001805437 watermark), so this bound is chosen to emit a whole tournament in a
# single cycle while still capping a pathological cohort.
GATE_SUBMIT_BATCH = 1600


FOREIGN_PAIR_ABANDON_MS = 30 * 60 * 1000


def _abandoned_foreign_pair(pair: dict, scheduler: dict, corpus, now: int) -> bool:
    """True when a non-gate reservation is dead and nobody will ever finish it.

    The wave lane leaks reservations at scale: SCQ drops terminal jobs from its
    inventory after a rolling window, and any claim not ingested inside that
    window is stranded with its protocol pairs pinned at 'submitted' (501,120
    such pairs across the registry at the time of writing). A gate candidate
    that collides with one of those pairs is skipped at reservation time and
    then waited on forever. This applies the system's own
    `recover_orphaned_submissions` rule - forgotten by the scheduler, with no
    authoritative receipt anywhere - plus a 30 minute age floor so a reservation
    the queue keeper is actively working on is never taken.
    """

    if str(pair["reservation_owner"]).startswith("gate:"):
        return False
    if pair["state"] not in {"submitted", "reserved", "infrastructure_retry"}:
        return False
    if int(pair["updated_at_ms"] or 0) > now - FOREIGN_PAIR_ABANDON_MS:
        return False
    job_id = pair["job_id"]
    if job_id:
        if job_id in scheduler:
            return False
        if corpus.execute(
            "SELECT 1 FROM ingested_jobs WHERE job_id=?", (job_id,)
        ).fetchone():
            return False
    # A materialized receipt is NOT proof of evidence: a receipt whose ingest
    # keeps failing sits in receipts/<job>/ with SHARD_OK forever and produces
    # no observation, which is how 144 pairs stayed both unresolved and
    # unreclaimable. The only evidence that counts is what reached the corpus,
    # so ask the corpus directly (two primary-key seeks on WITHOUT ROWID
    # tables) and re-check it here rather than trusting the per-cycle preload,
    # since ingestion may have landed the row mid-cycle.
    identity = (pair["task"], pair["dataset_hash"], pair["semantic_sha256"])
    if corpus.execute(
        "SELECT 1 FROM observations WHERE task=? AND dataset_hash=? "
        "AND semantic_sha256=?",
        identity,
    ).fetchone():
        return False
    if corpus.execute(
        "SELECT 1 FROM failures WHERE task=? AND dataset_hash=? "
        "AND semantic_sha256=? LIMIT 1",
        identity,
    ).fetchone():
        return False
    return True


def _release_abandoned_pair(registry, pair: dict) -> bool:
    """Hand an abandoned reservation back with `supersede_unstarted`.

    Fail-soft on purpose: this reaches into another lane's bookkeeping, so any
    audit mismatch means "leave it alone", never "abort the tournament".
    """

    manifest_sha256 = pair["manifest_sha256"]
    if not manifest_sha256:
        return False
    job_id = pair["job_id"] or f"unstarted-{manifest_sha256[:24]}"
    try:
        registry.supersede_unstarted(
            manifest_sha256,
            job_id,
            retirement_receipt_sha256=sha256_text(
                f"gate-scheduler-state-lost:{manifest_sha256}:{job_id}"
            ),
        )
    except Exception:  # noqa: BLE001 - fail-soft by design, see docstring
        return False
    return True


def _gate_receipt_exists(job_id: str) -> bool:
    if (ROOT / "receipts" / job_id / "SHARD_OK").is_file():
        return True
    attempts = SCQ_RUNS / job_id / "attempts"
    return attempts.is_dir() and any(
        (attempt / "outputs" / "receipt" / "SHARD_OK").is_file()
        for attempt in attempts.glob("*")
    )


def _dead_gate_jobs(backlog, owners: dict[str, str]) -> dict[str, str]:
    """Gate jobs whose evidence will never arrive, mapped to a failure class.

    Two ways a submitted gate job dies without evidence:

    * SCQ ran it and it terminally failed or was cancelled.
    * SCQ FORGOT it. scqd keeps only a rolling window of terminal jobs (~2.6k,
      about an hour at current throughput). A gate job that completes but is not
      fetched inside that window disappears from `scq ls` and from `scq show`,
      its outputs are unreachable, and `selfplay_ingest::reconcile_scheduler`
      will not even attempt a fetch because its candidate filter requires the
      job to be listed as completed or to have a local run receipt. 167 of the
      first 1,304 gate jobs were lost exactly this way.

    Either way the protocol pairs stay pinned at 'submitted' and the tournament
    waits forever, so both are treated as infrastructure failures and the
    manifest is re-submitted under a fresh job id. Safe against the
    submit/ls race because this runs at the top of a cycle, before that cycle
    submits anything: every claim it can see is at least one poll interval old.
    """

    dead: dict[str, str] = {}
    connection = backlog.state_db()
    claims = []
    for owner in sorted(set(owners.values())):
        claims.extend(
            (str(row["job_id"]), str(row["manifest_sha256"]))
            for row in connection.execute(
                "SELECT job_id,manifest_sha256 FROM claims "
                "WHERE reservation_owner=? AND state='submitted'",
                (owner,),
            )
        )
    connection.close()
    if not claims:
        return dead
    scheduler = {
        str(row["id"]): row for row in backlog.jobs() if row.get("id")
    }
    corpus = sqlite3.connect(
        f"file:{ROOT / 'corpus' / 'strict-test-exact.sqlite'}?mode=ro", uri=True
    )
    corpus.execute("PRAGMA busy_timeout=60000")
    try:
        for job_id, _ in claims:
            job = scheduler.get(job_id)
            if job is None:
                failure = "scheduler-state-lost"
            elif str(job.get("state")) in {"failed", "cancelled"}:
                failure = str(job.get("failure_class") or job.get("state"))
            else:
                continue
            if corpus.execute(
                "SELECT 1 FROM ingested_jobs WHERE job_id=?", (job_id,)
            ).fetchone():
                continue
            if _gate_receipt_exists(job_id):
                continue
            dead[job_id] = failure
    finally:
        corpus.close()
    return dead


def _submit_gate_manifests(
    registry, manifests_root: Path, owners: dict[str, str], limit: int
) -> dict:
    """Render, check and submit the gate manifests this controller reserved.

    THE GATE LANE HAS NO OTHER SUBMITTER. `jope_selfplay_queue` only admits
    `waves/strict-wave-*/manifests` and refuses anything whose evaluation_lane is
    protected (`protected evaluation lane is not adaptive admission`), and
    `test_exact_backlog.manifest_paths()` only globs manifests/{pending,
    binary-repair,supplemental,strict-wave-*}. Before this function existed the
    phase reserved 77,075 protocol pairs, wrote 1,304 manifests, and then waited
    forever on work that nothing in the system was going to run. Reserving
    without submitting is the bug; the owner of a reservation owns its
    submission.

    Idempotent and bounded: only manifests whose pairs are still entirely
    reserved/infrastructure_retry AND owned by this tournament's gate owners are
    submitted, at most `limit` per cycle, so a cycle stays short and the next
    cycle continues where this one stopped.
    """

    import test_exact_backlog as backlog

    evaluator = backlog.stage_evaluator()
    connection = backlog.state_db()
    # Ingestion drains `claims` newest-first (selfplay_ingest.rs orders
    # `state='submitted'` claims by updated_at_ms DESC, for both the SCQ fetch
    # and the local receipt pass). The adaptive wave lane re-stamps claims
    # continuously, so a gate claim written once at submission time sinks to the
    # bottom forever and its receipts are ingested at a trickle even though the
    # jobs finished long ago (observed: 13 gate receipts ingested in 10 minutes
    # while 200 wave receipts went through). While a tournament is pending the
    # gate lane IS the priority - it is the phase everything else is blocked
    # on - so keep its un-ingested claims at the head of that queue. This only
    # moves a scheduling hint: no evidence, state or ownership is touched, and a
    # claim leaves the set the moment ingestion marks it 'ingested'.
    now_ms = int(time.time() * 1000)
    for owner in sorted(set(owners.values())):
        connection.execute(
            "UPDATE claims SET updated_at_ms=? "
            "WHERE reservation_owner=? AND state='submitted'",
            (now_ms, owner),
        )
    connection.commit()
    dead = _dead_gate_jobs(backlog, owners)
    selected: list[tuple[Path, str, str, int]] = []
    recovered = 0
    for checkpoint_hash, owner in sorted(owners.items()):
        directory = manifests_root / checkpoint_hash
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if len(selected) >= limit:
                break
            digest = sha256_file(path)
            if registry.manifest_owner(digest) != owner:
                # Either unreserved or a concurrent generator owns it; that
                # owner is responsible for submitting it.
                continue
            claim = connection.execute(
                "SELECT job_id,state,attempts FROM claims WHERE manifest_sha256=?",
                (digest,),
            ).fetchone()
            attempts = int(claim["attempts"]) if claim else 0
            states = set(registry.manifest_state_counts(digest))
            if not states:
                continue
            if not states.issubset({"reserved", "infrastructure_retry"}):
                # Already submitted. Only take it back when SCQ has terminally
                # given up on the job AND no receipt was ever ingested for it,
                # otherwise a still-running gate job would be duplicated.
                if (
                    states != {"submitted"}
                    or claim is None
                    or claim["job_id"] not in dead
                    or attempts >= 5
                ):
                    continue
                registry.record_manifest_infrastructure_failure(
                    digest, dead[claim["job_id"]]
                )
                attempts += 1
                recovered += 1
            selected.append((path, digest, owner, attempts))
        if len(selected) >= limit:
            break
    if not selected:
        connection.close()
        return {"considered": 0, "submitted": 0, "failed": 0, "recovered": 0}

    rendered = []
    for path, digest, owner, attempts in selected:
        job_id, spec, spec_digest = backlog.render_spec(path, evaluator)
        if spec_digest != digest:
            raise RuntimeError(f"{path.name}: gate manifest changed while rendering")
        if attempts:
            # A retried manifest needs a fresh SCQ identity; reusing the dead
            # job id is silently adopted by `scq submit` and would never run.
            retry_job = f"{job_id}-r{attempts}"
            retry_spec = spec.with_name(f"{retry_job}.toml")
            text = spec.read_text()
            old_line = f"id = {json.dumps(job_id)}"
            if old_line not in text:
                raise RuntimeError(f"cannot retry {job_id}: job id absent from spec")
            retry_spec.write_text(text.replace(old_line, f"id = {json.dumps(retry_job)}", 1))
            job_id, spec = retry_job, retry_spec
        rendered.append((job_id, spec, digest, path, owner, attempts))
    backlog.prestage_rows([row[:4] for row in rendered], workers=32)

    failures = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        future_map = {
            pool.submit(backlog.check_one, row[1]): row for row in rendered
        }
        for future in as_completed(future_map):
            _, error = future.result()
            if error:
                failures.append((future_map[future], error))
    if failures:
        row, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} gate specs failed scq check; {row[3].name}: {error}"
        )

    now = int(time.time() * 1000)
    # Ingestion is claim-driven: `jope_selfplay_status reconcile` only fetches
    # and ingests receipts for job ids that appear in claims with state
    # 'submitted'. A gate job without a claim row is invisible to ingestion, so
    # the claim is written before the scheduler call, not after.
    connection.executemany(
        """
        INSERT OR REPLACE INTO claims(
          manifest_sha256,manifest,job_id,spec,state,attempts,
          reservation_owner,updated_at_ms
        ) VALUES(?,?,?,?,?,?,?,?)
        """,
        [
            (digest, str(path), job_id, str(spec), "checked", attempts, owner, now)
            for job_id, spec, digest, path, owner, attempts in rendered
        ],
    )
    connection.commit()

    submitted = 0
    submit_failures = []
    for job_id, spec, digest, path, owner, _ in rendered:
        # Authorize before the scheduler call so a pair that another writer
        # moved out from under us aborts this manifest instead of orphaning a
        # job, and record the submission only after SCQ has accepted it.
        registry.authorize_submission(digest, owner, job_id=job_id)
        _, accepted, detail = backlog.submit_one((job_id, spec, digest, path))
        if not accepted:
            submit_failures.append({"job_id": job_id, "detail": detail[:300]})
            continue
        registry.record_submission(digest, owner, job_id)
        connection.execute(
            "UPDATE claims SET state='submitted',updated_at_ms=? WHERE job_id=?",
            (int(time.time() * 1000), job_id),
        )
        connection.commit()
        submitted += 1
    connection.close()
    return {
        "considered": len(rendered),
        "submitted": submitted,
        "failed": len(submit_failures),
        "recovered": recovered,
        "failures": submit_failures[:5],
    }


def _checkpoint_conditional_advantage(checkpoint_sha256: str) -> float | None:
    """Conditional advantage the trainer measured for the run that EMITTED this checkpoint.

    The training-history archive records `val_conditional_advantage` per epoch and
    `result_checkpoint_sha256` / `selected_checkpoint_epoch` per run, so a
    checkpoint's own reading is the value at the epoch that was actually selected
    -- not the run's best epoch, which would flatter every checkpoint.

    Returns None when the checkpoint has no training run in the archive (e.g. an
    externally supplied anchor). None is deliberate: `test_exact_accept.py`
    treats an absent reading as fail-closed, and inventing a zero here would
    silently promote an unmeasured checkpoint.
    """

    database = ROOT / "training" / "history" / "db" / "history.sqlite3"
    if not checkpoint_sha256 or not database.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        row = connection.execute(
            """
            SELECT e.val_conditional_advantage
            FROM run r
            JOIN epoch e ON e.run_key = r.run_key
            WHERE r.result_checkpoint_sha256 = ?
              AND e.val_conditional_advantage IS NOT NULL
              AND (
                    r.selected_checkpoint_epoch IS NULL
                    OR e.epoch = r.selected_checkpoint_epoch
                  )
            ORDER BY e.epoch DESC
            LIMIT 1
            """,
            (checkpoint_sha256,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if not row or row[0] is None:
        return None
    value = float(row[0])
    return value if math.isfinite(value) else None


def _evaluate_frozen_gate(request: dict) -> dict:
    """Reserve/evaluate every equal-budget proposal and build frozen gate evidence."""

    from test_exact_common import selfplay_authority
    from test_exact_pair_registry import PairRegistry, protocol_for_manifest

    proposal_artifact = request["artifacts"]["generate_gate_proposals"]
    # The gate development cohort is frozen at PROPOSAL time, not evaluation time:
    # each proposal set binds the catalog it was generated under (summary
    # catalog_sha256), and a catalog activation landing mid-tournament must not
    # move the goalposts. Resolve every bound catalog (current pointer +
    # pointer-history) and take the intersection of their validation splits.
    pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
    catalog_by_sha = {pointer["catalog_sha256"]: Path(pointer["catalog"])}
    history = ROOT / "catalog" / "pointer-history.jsonl"
    if history.is_file():
        for line in history.read_text().splitlines():
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                continue
            # History records wrap the superseded pointer: the catalog fields
            # live under "pointer", not at the top level. Reading them at the
            # top level silently matched nothing, so a proposal set generated
            # minutes before a catalog flip was rejected as "unknown" forever
            # (1,224 consecutive gate failures). Accept both shapes.
            old = old.get("pointer") or old
            if old.get("catalog_sha256") and old.get("catalog"):
                catalog_by_sha.setdefault(
                    old["catalog_sha256"], Path(old["catalog"])
                )
    validation_by_set = []
    bound_catalogs = {}
    for proposal_set in proposal_artifact["proposal_sets"]:
        summary = json.loads(Path(proposal_set["summary"]).read_text())
        bound = summary["catalog_sha256"]
        bound_catalogs[proposal_set["checkpoint_sha256"]] = bound
        path = catalog_by_sha.get(bound)
        if path is None or not path.is_file():
            raise RuntimeError(
                f"gate proposal set binds unknown catalog {bound}"
            )
        catalog = json.loads(path.read_text())
        validation_by_set.append(
            {
                (row["task"], row["dataset_hash"]): row
                for row in catalog["entries"]
                if row["split"] == "validation"
            }
        )
    cohort = set.intersection(*(set(rows) for rows in validation_by_set))
    frozen_source = min(validation_by_set, key=len)
    entries = {key: frozen_source[key] for key in cohort}
    if not entries:
        raise RuntimeError("gate development cohort intersection is empty")
    work = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "gate-evaluation"
    )
    manifests_root = (
        ROOT
        / "manifests"
        / "gates"
        / f"watermark-{int(request['watermark']):09d}"
    )
    work.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)
    registry = PairRegistry()
    expected: list[dict] = []
    wave = f"strict-wave-{int(request['watermark']) % 1_000_000:06d}"
    loaded_sets = []
    for proposal_set in proposal_artifact["proposal_sets"]:
        by_dataset: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        with Path(proposal_set["proposals"]).open() as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["task"], row["dataset_hash"])
                if key in entries:
                    by_dataset[key].append(row)
        if set(entries) - set(by_dataset):
            raise RuntimeError("gate proposal set does not cover frozen development")
        loaded_sets.append((proposal_set, by_dataset))
    # Per-dataset equal budgets: a low-diversity context (deficient emission,
    # floor 8 at generation) must not let one side pick its best from more draws
    # than the other, so both sides are truncated to the shared minimum by slot.
    requested_budget = int(proposal_artifact["proposal_budget_per_dataset"])
    gate_budgets: dict[tuple[str, str], int] = {}
    for key in sorted(entries):
        budget = min(
            min(len(by_dataset[key]) for _, by_dataset in loaded_sets),
            requested_budget,
        )
        if budget < 8:
            raise RuntimeError(
                f"{key} gate proposal budget {budget} is below the floor of 8"
            )
        gate_budgets[key] = budget
    gate_budget_deficits = {
        f"{key[0]}:{key[1]}": budget
        for key, budget in gate_budgets.items()
        if budget < requested_budget
    }
    import test_exact_backlog as _backlog

    scheduler = {str(row["id"]) for row in _backlog.jobs() if row.get("id")}
    evidence_corpus = sqlite3.connect(
        f"file:{ROOT / 'corpus' / 'strict-test-exact.sqlite'}?mode=ro", uri=True
    )
    evidence_corpus.execute("PRAGMA busy_timeout=60000")
    now_ms = int(time.time() * 1000)
    released = 0
    gate_owners: dict[str, str] = {}
    for proposal_set, by_dataset in loaded_sets:
        checkpoint_hash = proposal_set["checkpoint_sha256"]
        owner = (
            f"gate:{int(request['watermark'])}:"
            f"{checkpoint_hash}:{int(request['seed'])}"
        )
        gate_owners[checkpoint_hash] = owner
        for key in sorted(entries):
            entry = entries[key]
            candidates = sorted(by_dataset[key], key=lambda row: row["slot"])[
                : gate_budgets[key]
            ]
            base_manifest = {
                "schema": "JopeFlywheelCorpusEvalShard.v1",
                "evaluation_lane": "checkpoint_gate",
                "task": entry["task"],
                "dataset_hash": entry["dataset_hash"],
                "train_csv": entry["train_csv"],
                "test_csv": entry["test_csv"],
                "train_sha256": entry["train_sha256"],
                "test_sha256": entry["test_sha256"],
                "label_authority": selfplay_authority(
                    task=entry["task"],
                    wave=wave,
                    frozen_seed_sha256=sha256_text(
                        f"{request['watermark']}:{checkpoint_hash}:{key}"
                    ),
                    model_checkpoint_sha256=checkpoint_hash,
                    hall_of_fame_sha256=sha256_file(
                        ROOT / "indexes" / "hall-of-fame.json"
                    ),
                ),
                "candidates": candidates,
            }
            protocol = protocol_for_manifest(base_manifest)
            fresh = []
            for candidate in candidates:
                identity = {
                    "task": key[0],
                    "dataset_hash": key[1],
                    "semantic_sha256": candidate["semantic_sha256"],
                    "protocol_sha256": protocol,
                }
                known = registry.get(**identity)
                row = {
                    **identity,
                    "checkpoint_sha256": checkpoint_hash,
                    "role": proposal_set["role"],
                    "decoder": candidate.get("decoder"),
                }
                expected.append(row)
                # 'available' is the unowned state the freshness controller and
                # queue keeper release an unstarted reservation into. A gate
                # candidate that happens to collide with a wave-lane pair is
                # skipped here as "someone else's work" - and if that someone
                # else then releases it, nobody ever evaluates it and the gate
                # blocks on it forever (186 pairs on the first tournament). An
                # available pair belongs to whoever claims it next, and the gate
                # needs it, so the gate claims it; `reserve()` does that
                # atomically under BEGIN IMMEDIATE, so a concurrent wave
                # reservation simply wins and this side sees a live state again.
                if known is None or known["state"] == "available":
                    fresh.append(candidate)
                elif known["state"] not in {
                    "completed",
                    "terminal_failure",
                } and _abandoned_foreign_pair(known, scheduler, evidence_corpus, now_ms):
                    if _release_abandoned_pair(registry, known):
                        released += 1
                        fresh.append(candidate)
                    else:
                        # The reservation is provably dead but its owning lane's
                        # attempt history does not satisfy the retirement audit
                        # (`supersede_unstarted` also cannot retire an
                        # 'infrastructure_retry'), so it can be neither finished
                        # nor handed back. Blocking the tournament forever on
                        # another lane's bookkeeping is the worse failure: mark
                        # it unevaluable, exclude it from BOTH the wait and the
                        # scored ledger, and account for it in the evidence -
                        # with a hard cap below so a systemic collapse still
                        # stops a verdict instead of silently skewing one.
                        row["unevaluable"] = "reservation_abandoned_by_owner"
            if not fresh:
                continue
            manifest = {**base_manifest, "candidates": fresh}
            # Content-addressed by the candidate set: a later top-up (claiming
            # pairs that were released after the first pass) must never
            # overwrite the manifest whose job is already submitted or ingested,
            # because that manifest_sha256 is the binding its pairs, its claim
            # and its receipt all agree on.
            fresh_digest = sha256_text(
                "\n".join(sorted(row["semantic_sha256"] for row in fresh))
            )
            path = (
                manifests_root
                / checkpoint_hash
                / f"{key[0][0]}-{key[1]}-{fresh_digest[:12]}.json"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, manifest)
            digest = sha256_file(path)
            reserved, reused = registry.reserve_manifest(
                manifest,
                owner=owner,
                manifest_path=str(path),
                manifest_sha256=digest,
            )
            if reused and len(reserved) != len(fresh):
                # A concurrent generator won the reservations. It owns submission;
                # this controller only waits for the authoritative results.
                path.unlink(missing_ok=True)
    evidence_corpus.close()
    submission = _submit_gate_manifests(
        registry, manifests_root, gate_owners, GATE_SUBMIT_BATCH
    )
    submission["released_abandoned_pairs"] = released
    # Completion evidence comes from the CORPUS, not from pairs.state.
    # `pairs.state` only ever reached 'completed'/'terminal_failure' through the
    # retired python ingest (test_exact_ingest.record_result); the live Rust
    # ingestion (`jope_selfplay_status reconcile`) writes corpus observations,
    # corpus failures and ingested_jobs and leaves the pair at 'submitted'
    # forever. Waiting on pairs.state was therefore unsatisfiable by
    # construction - the last authoritative_result in the registry predates the
    # first gate manifest by 13 hours. A protocol pair is resolved when the
    # evaluator has spoken about it: an observation row (scored) or a failures
    # row (terminal semantic failure, e.g. a head that is incompatible with the
    # task). Both are per-candidate receipts written in the same ingest
    # transaction, so neither can be half-applied.
    corpus = sqlite3.connect(
        f"file:{ROOT / 'corpus' / 'strict-test-exact.sqlite'}?mode=ro", uri=True
    )
    corpus.execute("PRAGMA busy_timeout=60000")
    corpus.row_factory = sqlite3.Row
    observed: dict[tuple[str, str], dict[str, dict]] = {}
    failed: dict[tuple[str, str], set[str]] = {}
    for key in sorted(entries):
        # observations/failures are WITHOUT ROWID with PK(task,dataset_hash,
        # semantic_sha256,...), so this is a primary-key range scan per dataset:
        # 2 seeks per dataset instead of one point query per candidate.
        observed[key] = {
            str(row["semantic_sha256"]): dict(row)
            for row in corpus.execute(
                """
                SELECT semantic_sha256,score,receipt_sha256,candidate_source
                FROM observations WHERE task=? AND dataset_hash=?
                """,
                key,
            )
        }
        failed[key] = {
            str(row["semantic_sha256"])
            for row in corpus.execute(
                "SELECT semantic_sha256 FROM failures WHERE task=? AND dataset_hash=?",
                key,
            )
        }
    corpus.close()
    pending = []
    pair_rows = []
    scored = {}
    unevaluable: dict[str, int] = collections.Counter()
    for row in expected:
        key = (row["task"], row["dataset_hash"])
        semantic = row["semantic_sha256"]
        observation = observed.get(key, {}).get(semantic)
        if observation is None and row.get("unevaluable"):
            unevaluable[row["checkpoint_sha256"]] += 1
            continue
        if observation is not None:
            resolution = "observation"
        elif semantic in failed.get(key, ()):
            resolution = "evaluator_failure"
        else:
            # Only pay for a registry seek when the corpus has said nothing;
            # this is the whole cohort on the first poll and almost none of it
            # once the lane has drained.
            pair = registry.get(
                **{
                    field: row[field]
                    for field in (
                        "task",
                        "dataset_hash",
                        "semantic_sha256",
                        "protocol_sha256",
                    )
                }
            )
            if pair is None or pair["state"] not in {"completed", "terminal_failure"}:
                pending.append(row)
                continue
            resolution = "pair_registry"
        pair_rows.append(
            {
                **row,
                "resolved_by": resolution,
                "receipt_sha256": (
                    None if observation is None else observation["receipt_sha256"]
                ),
            }
        )
        scored[(row["checkpoint_sha256"], *key, semantic)] = (
            None
            if observation is None
            else {
                "score": observation["score"],
                "receipt_sha256": observation["receipt_sha256"],
                "candidate_source": observation["candidate_source"],
            }
        )
    if pending:
        registry.close()
        raise PendingAction(
            f"waiting for {len(pending)} frozen gate protocol pairs "
            f"({len(pair_rows)} resolved; submitted {submission['submitted']} "
            f"of {submission['considered']} gate jobs this cycle, "
            f"{submission['failed']} rejected, "
            f"{submission['recovered']} recovered, "
            f"{submission['released_abandoned_pairs']} abandoned pairs reclaimed, "
            f"{sum(unevaluable.values())} unevaluable)"
        )
    unevaluable_total = sum(unevaluable.values())
    if unevaluable_total > 0.01 * len(expected):
        raise RuntimeError(
            f"{unevaluable_total} of {len(expected)} gate protocol pairs are "
            "unevaluable; refusing to score a tournament this incomplete"
        )
    champions_path = Path(proposal_artifact["frozen_champions"])
    if sha256_file(champions_path) != proposal_artifact["frozen_champions_sha256"]:
        raise RuntimeError("frozen gate champions changed")
    champions = {
        (row["task"], row["dataset_hash"]): row
        for row in json.loads(champions_path.read_text())
    }
    metrics = {}
    best_by_checkpoint: dict[str, dict[tuple[str, str], float]] = {}
    greedy_by_checkpoint: dict[str, dict[tuple[str, str], float]] = {}
    for proposal_set in proposal_artifact["proposal_sets"]:
        digest = proposal_set["checkpoint_sha256"]
        rows = [row for row in pair_rows if row["checkpoint_sha256"] == digest]
        best: dict[tuple[str, str], float] = {}
        greedy: dict[tuple[str, str], float] = {}
        failures = 0
        receipts = set()
        for row in rows:
            key = (row["task"], row["dataset_hash"])
            score = scored[(digest, *key, row["semantic_sha256"])]
            if score is None:
                failures += 1
                continue
            value = float(score["score"])
            best[key] = max(best.get(key, float("-inf")), value)
            if row["decoder"] == "greedy":
                greedy[key] = value
            receipts.add(score["receipt_sha256"])
        wins = {}
        for key, score in best.items():
            champion = champions.get(key, {})
            incumbent = (
                champion.get("best_new_strict")
                or champion.get("recorded_incumbent")
            )
            wins[key] = float(
                incumbent is not None and score > float(incumbent["score"])
            )
        task_accuracy = {
            task: _family_held_accuracy(wins, entries, task)
            for task in ("regression", "binary")
        }
        macro = statistics.fmean(task_accuracy.values())
        best_by_checkpoint[digest] = best
        greedy_by_checkpoint[digest] = greedy
        metrics[digest] = {
            "checkpoint_sha256": digest,
            "macro_pairwise_accuracy": macro,
            "regression_pairwise_accuracy": task_accuracy["regression"],
            "binary_pairwise_accuracy": task_accuracy["binary"],
            "failure_rate": failures / max(1, len(rows)),
            "p95_proposal_latency_ms": proposal_set[
                "p95_proposal_latency_ms"
            ],
            "pair_receipt_count": len(receipts),
            "equal_budget_proposal_result": macro,
            # THE gate criterion this dict used to omit. test_exact_accept.py
            # reads CONDITIONAL_ADVANTAGE_KEYS off these nodes and treats a
            # missing reading as fail-closed ("an unmeasured checkpoint is
            # fail-closed"). None of those keys were ever written here, so
            # conditional_advantage() returned None for BOTH baseline and
            # candidate on every run, both *_evidence_missing reasons were
            # appended, and internal_pass = not internal_reasons was
            # unconditionally False. The gate returned pass:false for every
            # checkpoint at every accuracy -- promotion has never once fired in
            # this root (no training/accepted.json has ever existed). The
            # trainer has been recording the value all along; it simply was
            # never plumbed to the consumer. May be None -- that keeps the
            # fail-closed contract intact for genuinely unmeasured checkpoints.
            "val_conditional_advantage": _checkpoint_conditional_advantage(digest),
        }
    baseline_set = next(
        row for row in proposal_artifact["proposal_sets"] if row["role"] == "baseline"
    )
    baseline_hash = baseline_set["checkpoint_sha256"]
    candidates = [
        row
        for row in proposal_artifact["proposal_sets"]
        if row["role"] == "candidate"
    ]
    winner = max(
        candidates,
        key=lambda row: (
            metrics[row["checkpoint_sha256"]]["macro_pairwise_accuracy"],
            row["checkpoint_sha256"],
        ),
    )
    candidate_hash = winner["checkpoint_sha256"]
    candidate_best = best_by_checkpoint[candidate_hash]
    baseline_best = best_by_checkpoint[baseline_hash]
    comparisons = [
        (
            candidate_best[key] > baseline_best[key],
            candidate_best[key] < baseline_best[key],
        )
        for key in sorted(set(candidate_best) & set(baseline_best))
    ]
    candidate_equal = sum(win for win, _ in comparisons) / max(1, len(comparisons))
    baseline_equal = sum(loss for _, loss in comparisons) / max(1, len(comparisons))
    strict_origin_wins = 0
    for key, score in candidate_best.items():
        champion = champions.get(key, {})
        incumbent = (
            champion.get("best_new_strict")
            or champion.get("recorded_incumbent")
        )
        strict_origin_wins += int(
            incumbent is not None and score > float(incumbent["score"])
        )
    ledger = {
        "schema": "JopeFrozenGatePairLedger.v1",
        "corpus_watermark": int(request["watermark"]),
        "catalog_sha256": request["catalog_sha256"],
        "proposal_seed": int(request["seed"]),
        "pairs": pair_rows,
    }
    ledger_path = work / "pair-ledger.json"
    atomic_json(ledger_path, ledger)
    greedy_comparisons = [
        (
            score,
            float(
                (
                    champions[key].get("best_new_strict")
                    or champions[key]["recorded_incumbent"]
                )["score"]
            ),
        )
        for key, score in greedy_by_checkpoint[candidate_hash].items()
        if (
            champions.get(key, {}).get("best_new_strict")
            or champions.get(key, {}).get("recorded_incumbent")
        )
    ]
    evidence = {
        "schema": "JopeSelfplayGateEvidence.v3",
        "baseline": metrics[baseline_hash],
        "candidate": metrics[candidate_hash],
        "proposal_validity": min(
            float(row["proposal_validity"])
            for row in proposal_artifact["proposal_sets"]
        ),
        "verified_checkpoint_origin_strict_wins": strict_origin_wins,
        "candidate_equal_budget_result": candidate_equal,
        "baseline_equal_budget_result": baseline_equal,
        "gate_budgets": {
            "requested_per_dataset": requested_budget,
            "cohort_size": len(entries),
            "per_dataset_deficits": gate_budget_deficits,
            "bound_catalogs": bound_catalogs,
            "expected_pairs": len(expected),
            "unevaluable_pairs_by_checkpoint": dict(unevaluable),
        },
        "replay_verification": {
            "predictions": True,
            "scores": True,
            "evaluator_identity": True,
            "dataset_hashes": True,
            "candidate_origin": True,
        },
        "corpus_watermark": int(request["watermark"]),
        "catalog_sha256": request["catalog_sha256"],
        "evaluator_sha256": request["evaluator_sha256"],
        "proposal_seed": int(request["seed"]),
        "pair_ledger": str(ledger_path),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "checkpoint_metrics": metrics,
        "top1_metrics": {
            "greedy_validity": len(greedy_by_checkpoint[candidate_hash])
            / max(1, len(entries)),
            "greedy_strict_win_rate": sum(
                score
                > float(
                    (
                        champions[key].get("best_new_strict")
                        or champions[key]["recorded_incumbent"]
                    )["score"]
                )
                for key, score in greedy_by_checkpoint[candidate_hash].items()
                if (
                    champions.get(key, {}).get("best_new_strict")
                    or champions.get(key, {}).get("recorded_incumbent")
                )
            )
            / max(1, len(greedy_by_checkpoint[candidate_hash])),
            "incumbent_percentile": statistics.fmean(
                1.0 if score > incumbent else 0.5
                for score, incumbent in greedy_comparisons
            )
            if greedy_comparisons
            else 0.0,
            "regret_to_champion": statistics.fmean(
                max(0.0, incumbent - score)
                for score, incumbent in greedy_comparisons
            )
            if greedy_comparisons
            else 0.0,
            "dl_attributed_wins": strict_origin_wins,
            "equal_budget_candidate_vs_baseline": {
                "candidate": candidate_equal,
                "baseline": baseline_equal,
            },
            "equal_budget_candidate_vs_champion_hof": metrics[candidate_hash][
                "macro_pairwise_accuracy"
            ],
        },
    }
    destination = work / "gate-evidence.json"
    atomic_json(destination, evidence)
    registry.close()
    return {
        "acceptance_evidence": evidence,
        "artifact_path": str(destination),
        "artifact_sha256": sha256_file(destination),
    }


def _build_acceptance_evidence(request: dict) -> dict:
    gate_evaluation = request["artifacts"]["evaluate_frozen_gate"]
    evidence = gate_evaluation.get("acceptance_evidence", gate_evaluation)
    if not isinstance(evidence, dict):
        raise RuntimeError("frozen gate did not produce acceptance evidence")
    for key in ("baseline", "candidate", "proposal_validity", "replay_verification"):
        if key not in evidence:
            raise RuntimeError(f"frozen gate evidence is missing {key}")
    fetched = {
        candidate["checkpoint_sha256"]: candidate
        for candidate in request["artifacts"]["fetch_candidate_checkpoints"][
            "candidates"
        ]
    }
    checkpoint_hash = evidence["candidate"].get("checkpoint_sha256")
    if checkpoint_hash not in fetched:
        raise RuntimeError("gate winner is not one of the fetched tournament checkpoints")
    candidate = fetched[checkpoint_hash]
    evidence["candidate"]["checkpoint_path"] = candidate["checkpoint"]
    if evidence["baseline"].get("checkpoint_sha256") != request["accepted_ancestor"]:
        raise RuntimeError("gate baseline differs from the frozen accepted ancestor")
    destination = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "acceptance-evidence.json"
    )
    atomic_json(destination, evidence)
    return {
        **evidence,
        "artifact_path": str(destination),
        "artifact_sha256": sha256_file(destination),
    }


def _run_checkpoint_gate(request: dict) -> dict:
    from test_exact_accept import evaluate_gate

    evidence = request["artifacts"]["build_acceptance_evidence"]
    result = evaluate_gate(evidence)
    destination = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "checkpoint-gate.json"
    )
    atomic_json(destination, result)
    return {
        **result,
        "artifact_path": str(destination),
        "artifact_sha256": sha256_file(destination),
    }


def _advance_accepted_pointer(request: dict) -> dict:
    from test_exact_accept import promote

    evidence = Path(
        request["artifacts"]["build_acceptance_evidence"]["artifact_path"]
    )
    gate = Path(request["artifacts"]["run_checkpoint_gate"]["artifact_path"])
    result = promote(
        evidence_path=evidence,
        gate_path=gate,
        accepted_pointer=ROOT / "training" / "learning.json",
        immutable_root=ROOT / "training" / "immutable",
    )
    if result.get("pass") is not True:
        raise RuntimeError("controller attempted to promote a rejected checkpoint")
    learning = ROOT / "training" / "learning.json"
    # Compatibility pointer for existing consumers; the learning pointer is
    # authoritative and is always advanced first.
    atomic_json(
        ROOT / "training" / "accepted.json",
        json.loads(learning.read_text()),
    )
    milestone = None
    if result.get("milestone_pass") is True:
        milestone_result = promote(
            evidence_path=evidence,
            gate_path=gate,
            accepted_pointer=ROOT / "training" / "milestone.json",
            immutable_root=ROOT / "training" / "immutable",
            enforce_pointer_ancestor=False,
        )
        milestone = {
            "advanced": True,
            "checkpoint_sha256": milestone_result["checkpoint_sha256"],
            "pointer": str(ROOT / "training" / "milestone.json"),
            "pointer_sha256": sha256_file(
                ROOT / "training" / "milestone.json"
            ),
        }
    return {
        "advanced": True,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "promotion_tier": result["promotion_tier"],
        "learning_pointer": str(learning),
        "learning_pointer_sha256": sha256_file(learning),
        "milestone": milestone,
    }


def _seed_next_strict_wave(request: dict) -> dict:
    # The Rust driver parses this handler's ENTIRE stdout as one JSON document
    # (jope_selfplay_promotion.rs PY_SHIM does `print(json.dumps(handler(...)))`),
    # so anything else a callee prints corrupts the result. `test_exact_wave.build`
    # prints a progress line every 25 datasets (:955) — six lines for a
    # 128-dataset micro-wave. That had never mattered because build() had never
    # run to completion in this tournament; the first time it did, at 00:39:37 on
    # 2026-07-29, the wave published correctly and the phase STILL failed with
    #   Error: python handler seed_next_strict_wave returned invalid JSON
    #   Caused by: trailing characters at line 2 column 1
    # after 27 minutes of work. Send every callee's stdout to stderr (where the
    # journal and scq still capture it) and leave stdout to the shim alone.
    import contextlib
    import sys

    with contextlib.redirect_stdout(sys.stderr):
        return _seed_next_strict_wave_inner(request)


def _seed_next_strict_wave_inner(request: dict) -> dict:
    from test_exact_wave import build, plan_micro_cohort
    from test_exact_ops import render_checkpoint_proposal_spec

    gate = request["artifacts"]["run_checkpoint_gate"]
    expected = (
        gate["checkpoint_sha256"]
        if gate["pass"]
        else request["accepted_ancestor"]
    )
    from test_exact_wave import checkpoint

    _, actual = checkpoint()
    if actual != expected:
        raise RuntimeError("next-wave checkpoint does not match the gate outcome")
    checkpoint_path, _ = checkpoint()
    micro = plan_micro_cohort()
    cohort_path = Path(micro["cohort"])
    cohort_sha256 = micro["cohort_sha256"]
    if micro["checkpoint_sha256"] != actual:
        raise RuntimeError("micro-cohort checkpoint does not match the gate outcome")
    work = (
        ROOT
        / "training"
        / "tournament-work"
        / f"watermark-{int(request['watermark']):09d}"
        / "micro-cohort-proposals"
        / cohort_sha256
    )
    work.mkdir(parents=True, exist_ok=True)
    proposal_archive = (
        ROOT
        / "proposals"
        / "by-checkpoint"
        / f"{actual}-{cohort_sha256}.jsonl"
    )
    # ------------------------------------------------------------------
    # Resume from the published set instead of re-earning it.
    #
    # This phase is two very different pieces of work bolted together: a
    # ~3.5 min GPU proposal job, and a ~20 min CPU `build()` that scans the
    # 22 GB pair registry. Only the first half was idempotent, so ANY
    # interruption of the second half -- a `systemctl restart` of
    # jope-selfplay-promotion (observed 23:27:51 on 2026-07-28, which killed a
    # 16-minute-old build), a spec re-render that moves the job id, a node
    # eviction -- threw away a proposal set that had already been fetched,
    # provenance-checked and archived, and sent the phase back to the queue for
    # a GPU it does not need. With 2,100 jobs queued and every GPU allocated,
    # that requeue is unbounded.
    #
    # The archive name is content-addressed by (accepted checkpoint, frozen
    # cohort), and `proposals/current.json` is only written AFTER the full
    # provenance check below, so the pointer agreeing with a full re-hash of
    # the archive is exactly the evidence the fetch path would have produced.
    # ------------------------------------------------------------------
    published_pointer = ROOT / "proposals" / "current.json"
    reused_published_set = False
    if proposal_archive.is_file() and published_pointer.is_file():
        try:
            published = json.loads(published_pointer.read_text())
        except json.JSONDecodeError:
            published = {}
        reused_published_set = (
            published.get("checkpoint_sha256") == actual
            and published.get("cohort_sha256") == cohort_sha256
            and published.get("proposals") == str(proposal_archive)
            and published.get("proposals_sha256") == sha256_file(proposal_archive)
        )
    if not reused_published_set:
        _seed_next_strict_wave_proposals(
            request=request,
            actual=actual,
            checkpoint_path=checkpoint_path,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            work=work,
            proposal_archive=proposal_archive,
        )
    result = build(
        candidates_per_dataset=256,
        max_shard=100,
        proposals_path=proposal_archive,
        cohort_path=cohort_path,
    )
    return {
        **result,
        "proposal_set": str(proposal_archive),
        "proposal_set_sha256": sha256_file(proposal_archive),
        "proposal_checkpoint_sha256": actual,
        "micro_cohort": str(cohort_path),
        "micro_cohort_sha256": cohort_sha256,
        "reused_published_proposal_set": reused_published_set,
    }


def _seed_next_strict_wave_proposals(
    *,
    request: dict,
    actual: str,
    checkpoint_path: Path,
    cohort_path: Path,
    cohort_sha256: str,
    work: Path,
    proposal_archive: Path,
) -> None:
    """Submit/await/fetch the micro-cohort proposal job and publish the set.

    Raises PendingAction until the job is `completed`. Split out of
    `_seed_next_strict_wave` only so the resume path above can skip it whole;
    the body is unchanged.
    """
    from test_exact_ops import render_checkpoint_proposal_spec

    rendered = render_checkpoint_proposal_spec(
        work / f"{actual}.toml",
        checkpoint=checkpoint_path,
        split=None,
        seed=int(request["seed"]),
        proposals_per_dataset=64,
        cohort=cohort_path,
    )
    jobs = _scheduler_jobs()
    state = jobs.get(rendered["job_id"], {}).get("state")
    if state in {"failed", "cancelled"}:
        raise RuntimeError(f"micro-cohort proposal job failed: {state}")
    scq = Path("/home/ubuntu/smart_cluster/target/release/scq")
    if state is None:
        checked = subprocess.run(
            [str(scq), "check", rendered["spec"]],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        if checked.returncode:
            raise RuntimeError((checked.stderr or checked.stdout).strip())
        submitted = subprocess.run(
            [str(scq), "submit", rendered["spec"]],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        detail = (submitted.stderr or submitted.stdout).strip()
        if submitted.returncode and "already exists" not in detail.lower():
            raise RuntimeError(detail)
        raise PendingAction(f"waiting for micro-cohort proposals: {rendered['job_id']}")
    if state != "completed":
        raise PendingAction(f"waiting for micro-cohort proposals: {rendered['job_id']}")
    output = _fetch_job_output(rendered["job_id"], work / "fetched")
    proposals = list(output.rglob("dl-proposals.jsonl"))
    summaries = list(output.rglob("dl-proposals.summary.json"))
    if len(proposals) != 1 or len(summaries) != 1:
        raise RuntimeError("micro-cohort proposal fetch is incomplete")
    summary = json.loads(summaries[0].read_text())
    if (
        summary.get("model_checkpoint_sha256") != actual
        or summary.get("cohort_sha256") != cohort_sha256
        or summary.get("split") is not None
        or int(summary.get("proposals_per_dataset", 0)) != 64
        or sha256_file(proposals[0]) != summary.get("output_sha256")
    ):
        raise RuntimeError("micro-cohort proposal provenance drift")
    proposal_archive.parent.mkdir(parents=True, exist_ok=True)
    if not proposal_archive.is_file():
        shutil.copy2(proposals[0], proposal_archive)
    if sha256_file(proposal_archive) != summary["output_sha256"]:
        raise RuntimeError("archived micro-cohort proposal hash drift")
    # MERGE, never replace.
    #
    # `proposals/current.json` is a shared pointer with several writers and more
    # readers, and it is the LAST resort of `selfplay_runtime::active_checkpoint`
    # (:82-125): with `training/learning.json` and `training/accepted.json` absent
    # -- they have never existed in this root -- the ONLY thing that resolves an
    # active checkpoint is this file's `legacy_adopted: true`. Replacing the
    # document wholesale dropped that key at 23:11:57 on 2026-07-28, the first
    # time this phase ever got far enough to write it, and
    # jope_selfplay_queue.rs:334 immediately fail-closed with
    # `active checkpoint pointer is absent` -- so micro-wave supply stopped
    # building waves and the GP admission lane stalled behind a 2,100-job queue
    # with nothing replenishing it. `jope_selfplay_propose_driver.rs::rebind`
    # (:775-822) already merges for exactly this reason and says so; this writer
    # was the odd one out.
    pointer_path = ROOT / "proposals" / "current.json"
    document = {}
    if pointer_path.is_file():
        try:
            existing = json.loads(pointer_path.read_text())
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict):
            document.update(existing)
    document.update(
        {
            "checkpoint_sha256": actual,
            "cohort_sha256": cohort_sha256,
            "cohort": str(cohort_path),
            "proposals": str(proposal_archive),
            "proposals_sha256": sha256_file(proposal_archive),
            "summary": str(summaries[0]),
            "summary_sha256": sha256_file(summaries[0]),
        }
    )
    atomic_json(pointer_path, document)


def _freeze_winner_scan(request: dict) -> dict:
    from test_exact_winner_scan import scan

    pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
    previous_pointer = ROOT / "winner-scans" / "current.json"
    prior = None
    if previous_pointer.is_file():
        previous = json.loads(previous_pointer.read_text())
        candidate = Path(previous["ledger"])
        if candidate.is_file():
            prior = candidate
    accepted = request["artifacts"]["advance_accepted_pointer"][
        "checkpoint_sha256"
    ]
    public = scan(
        corpus_path=ROOT / "corpus" / "strict-test-exact.sqlite",
        catalog_path=Path(pointer["catalog"]),
        output_root=ROOT / "winner-scans",
        accepted_checkpoint_sha256=accepted,
        evaluator_sha256=request["evaluator_sha256"],
        prior_ledger=prior,
    )
    frozen = json.loads((ROOT / "winner-scans" / "current.json").read_text())
    return {
        **frozen,
        "dataset_counts": public["dataset_counts"],
        "discovered": public["discovered"],
        "selected_cohort_discovered": public["selected_cohort_discovered"],
    }


def _run_sotarena_qualification(request: dict) -> dict:
    """Freeze a report-only qualification queue; never publish adaptive genomes."""

    scan = request["artifacts"]["freeze_winner_scan"]
    ledger_path = Path(scan["ledger"])
    if sha256_file(ledger_path) != scan["ledger_sha256"]:
        raise RuntimeError("winner scan ledger changed before qualification enqueue")
    ledger = json.loads(ledger_path.read_text())
    selected = [
        {
            "task": row["task"],
            "dataset_hash": row["dataset_hash"],
            "semantic_sha256": (
                row["candidate"]["semantic_sha256"]
                if row.get("candidate")
                else None
            ),
            "receipt_sha256": (
                row["candidate"]["receipt_sha256"]
                if row.get("candidate")
                else None
            ),
            "checkpoint_sha256": (
                row["candidate"]["model_checkpoint_sha256"]
                if row.get("candidate")
                else None
            ),
            "qualification_state": row["qualification_state"],
            "evidence_lane": "adaptive",
            "report_only": True,
        }
        for row in ledger["datasets"]
        if row["qualification_state"] == "discovered"
    ]
    tier = request["artifacts"]["run_checkpoint_gate"]["promotion_tier"]
    sealed = None
    if tier == "milestone":
        from test_exact_sealed_audit import milestone_status

        sealed = milestone_status(
            request["artifacts"]["advance_accepted_pointer"][
                "checkpoint_sha256"
            ]
        )
    record = {
        "schema": "JopeSotarenaQualificationQueue.v1",
        "winner_ledger": str(ledger_path),
        "winner_ledger_sha256": scan["ledger_sha256"],
        "candidate_checkpoint_sha256": request["artifacts"][
            "advance_accepted_pointer"
        ]["checkpoint_sha256"],
        "promotion_tier": tier,
        "adaptive_evidence": True,
        "zero_shot_claim_eligible": False,
        "report_only": True,
        "sealed_audit": sealed,
        "candidates": selected,
    }
    path = (
        ROOT
        / "qualifications"
        / f"watermark-{int(request['watermark']):09d}.json"
    )
    atomic_json(path, record)
    return {
        "queue": str(path),
        "queue_sha256": sha256_file(path),
        "enqueued": len(selected),
        "report_only": True,
    }


def default_handlers(root: Path = ROOT) -> dict[str, Callable[[dict], dict]]:
    handlers = {phase: _file_handler(phase, root) for phase in PHASES}
    if root == ROOT:
        handlers.update(
            {
                "freeze_corpus": _freeze_corpus,
                "build_packed_corpus": _build_packed_corpus,
                "launch_checkpoint_tournament": _launch_checkpoint_tournament,
                "fetch_candidate_checkpoints": _fetch_candidate_checkpoints,
                "generate_gate_proposals": _generate_gate_proposals,
                "evaluate_frozen_gate": _evaluate_frozen_gate,
                "build_acceptance_evidence": _build_acceptance_evidence,
                "run_checkpoint_gate": _run_checkpoint_gate,
                "advance_accepted_pointer": _advance_accepted_pointer,
                "seed_next_strict_wave": _seed_next_strict_wave,
                "freeze_winner_scan": _freeze_winner_scan,
                "run_sotarena_qualification": _run_sotarena_qualification,
            }
        )
    return handlers


def live_snapshot(root: Path) -> dict:
    catalog_pointer_path = root / "catalog" / "current.json"
    catalog_authority = validate_catalog_manifest(
        catalog_pointer_path, verify_contexts=False
    )
    catalog_pointer = json.loads(catalog_pointer_path.read_text())
    catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
    corpus = sqlite3.connect(
        f"file:{root / 'corpus' / 'strict-test-exact.sqlite'}?mode=ro", uri=True
    )
    labels = int(corpus.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    corpus.close()
    from test_exact_train import accepted_checkpoint

    _, ancestor = accepted_checkpoint()
    evaluator = Path("/home/ubuntu/jope-dime/target/release/corpus_eval")
    return {
        "valid_labels": labels,
        "accepted_ancestor": ancestor,
        "catalog_sha256": catalog_pointer["catalog_sha256"],
        "catalog_entries": catalog["entries"],
        "catalog_counts": catalog_authority["task_counts"],
        "evaluator_sha256": sha256_file(evaluator),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    lock_path = args.root / "ops" / "controller.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("self-play controller is already running") from error
        controller = FlywheelController(args.root, default_handlers(args.root))
        try:
            print(json.dumps(controller.cycle(live_snapshot(args.root)), sort_keys=True))
        finally:
            controller.close()


if __name__ == "__main__":
    main()
