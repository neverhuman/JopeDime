#!/usr/bin/env python3
"""Validate, submit, and refill the dedicated 5,000-job strict SCQ backlog."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from test_exact_common import (
    ALLOWED_LIVE_STATES,
    EXPECTED_METRIC,
    USEFUL_QUEUE_STATES,
    atomic_json,
    dynamic_queue_targets,
    manifest_authority_kind,
    queue_refill_needed,
    sha256_file,
    sha256_text,
)
from test_exact_evaluator import active as active_evaluator
from test_exact_pair_registry import PairRegistry
from test_exact_freshness import (
    active_epoch_for_root,
    freshness_scoped_manifest,
    manifest_epoch,
)

SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
MANIFEST_ROOT = ROOT / "manifests"
SUPPLEMENTAL = MANIFEST_ROOT / "supplemental"
BASE_MANIFEST_DIRS = (
    ROOT / "manifests" / "pending",
    ROOT / "manifests" / "binary-repair",
    SUPPLEMENTAL,
)
SPECS = ROOT / "specs" / "evaluations"
RUNNER = Path("/home/ubuntu/jope-flywheel/scripts/fweval_runner.sh")
EVALUATOR_SOURCE = Path("/home/ubuntu/jope-dime/target/release/corpus_eval")
STATE = ROOT / "ops" / "backlog.sqlite"
BLOB_DIR = Path("/home/ubuntu/smart_cluster/var/blobs")
SCQ_RUNS = Path("/home/ubuntu/smart_cluster/var/runs")
_FILE_HASH_CACHE: dict[Path, tuple[int, int, str]] = {}
INFRASTRUCTURE_FAILURES = {
    "rpc-error",
    "storage",
    "storage-error",
    "load-error",
    "dependencyunknown",
    "dependency-unknown",
}
# The checkpoint gate gets its own SCQ family so a tournament is never stuck
# behind the (deliberately lowest-priority, continuously refilled) adaptive
# evaluation backlog. Family weight is a live scq override, not a projects.toml
# edit, so no scqd restart is needed.
GATE_PROJECT = "jope-selfplay-gate"
EVALUATION_PROJECTS = {
    "jope-testonly",
    "jope-selfplay-eval",
    "jope-selfplay-fresh",
    GATE_PROJECT,
}
RETRY_BACKOFF_SECONDS = (1, 2, 4, 8, 16, 30)
SUBMISSION_BATCH_LIMIT = 512
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def command(args: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, check=False
    )


def infrastructure_error(detail: str) -> bool:
    lowered = detail.lower()
    return any(
        marker in lowered
        for marker in (
            "controller dropped request",
            "connect ",
            "connection reset",
            "connection refused",
            "broken pipe",
            "timed out",
            "socket",
        )
    )


def retry_command(
    args: list[str],
    *,
    timeout: int = 900,
    backoff: tuple[int, ...] = RETRY_BACKOFF_SECONDS,
) -> subprocess.CompletedProcess:
    result = command(args, timeout=timeout)
    for delay in backoff:
        detail = (result.stderr or result.stdout).strip()
        if result.returncode == 0 or not infrastructure_error(detail):
            break
        time.sleep(delay)
        result = command(args, timeout=timeout)
    return result


def jobs() -> list[dict]:
    result = retry_command([str(SCQ), "--json", "ls"])
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout).get("Jobs", {}).get("jobs", [])


def strict_live_jobs() -> list[dict]:
    return [
        row
        for row in jobs()
        if str(row.get("id") or "").startswith("jte-")
        and not str(row.get("id") or "").startswith("jte-canary-")
        and row.get("project") in EVALUATION_PROJECTS
        and row.get("state") in ALLOWED_LIVE_STATES
        and not stale_dispatch(row)
    ]


def cluster_slot_inventory() -> dict:
    result = retry_command([str(SCQ), "--json", "nodes"], timeout=30)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    rows = json.loads(result.stdout).get("Nodes", {}).get("nodes", [])
    nodes = []
    for row in rows:
        effective = int(row.get("effective_cores") or 0)
        multiplier = 2.0 if row.get("node") == "xbabe4" else 1.0
        nodes.append(
            {
                "node": row.get("node"),
                "physical_cores": int(row.get("cores") or 0),
                "effective_cores": effective,
                "cpu_job_slot_multiplier": multiplier,
                "slot_target": int(effective * multiplier),
                "running_slots": int(row.get("running_jobs") or 0),
            }
        )
    return {
        "effective_cluster_slots": sum(row["slot_target"] for row in nodes),
        "nodes": nodes,
    }


def stale_dispatch(row: dict, *, grace_seconds: int = 120) -> bool:
    return (
        row.get("state") == "dispatching"
        and row.get("pid") is None
        and int(row.get("runtime_secs") or 0) > grace_seconds
    )


def maximum_queue_age_seconds(rows: list[dict]) -> float:
    ages = []
    current = datetime.now(timezone.utc)
    for row in rows:
        if row.get("state") not in USEFUL_QUEUE_STATES or not row.get("queue_ts"):
            continue
        try:
            queued = datetime.fromisoformat(str(row["queue_ts"]))
        except ValueError:
            continue
        ages.append(max(0.0, (current - queued).total_seconds()))
    return max(ages, default=0.0)


def manifest_paths() -> list[Path]:
    wave_dirs = sorted(
        path
        for path in MANIFEST_ROOT.glob("strict-wave-*")
        if path.is_dir()
    )
    manifests = []
    for directory in (*BASE_MANIFEST_DIRS, *wave_dirs):
        for manifest in directory.glob("*.json"):
            if (
                manifest.parent == SUPPLEMENTAL
                and json.loads(manifest.read_text()).get("task") != "regression"
            ):
                continue
            manifests.append(manifest)
    gates = MANIFEST_ROOT / "gates"
    if gates.is_dir():
        manifests.extend(gates.rglob("*.json"))
    return sorted(manifests)


def manifest_is_current(path: Path) -> bool:
    manifest = json.loads(path.read_text())
    if not freshness_scoped_manifest(manifest):
        return True
    active = active_epoch_for_root(ROOT)
    return active is None or manifest_epoch(manifest) == active


def ready_manifest_count() -> int:
    """Count launch-ready checked manifests held outside SCQ."""

    con = state_db()
    count = int(
        con.execute("SELECT COUNT(*) FROM claims WHERE state='checked'").fetchone()[0]
    )
    con.close()
    return count


def unclaimed_manifest_count() -> int:
    con = state_db()
    claimed = {row["manifest"] for row in con.execute("SELECT manifest FROM claims")}
    con.close()
    return sum(str(path) not in claimed for path in manifest_paths())


def adopt_scheduler_claims(scheduler_rows: list[dict] | None = None) -> dict:
    """Make successfully accepted jobs ingestible before a large refill drains."""
    scheduler = {
        row["id"]: row
        for row in (jobs() if scheduler_rows is None else scheduler_rows)
        if row.get("id") and row.get("project") in EVALUATION_PROJECTS
    }
    con = state_db()
    registry = PairRegistry()
    now = int(time.time() * 1000)
    adopted = 0
    for claim in con.execute(
        "SELECT * FROM claims WHERE state='checked' "
        "OR state LIKE 'submit_failed:%'"
    ).fetchall():
        if claim["job_id"] not in scheduler:
            continue
        owner = claim["reservation_owner"] or registry.manifest_owner(
            claim["manifest_sha256"]
        )
        if owner is None:
            registry.close()
            con.close()
            raise RuntimeError(
                f"{claim['manifest_sha256']}: scheduler claim has no pair reservation"
            )
        registry.record_submission(
            claim["manifest_sha256"],
            owner,
            claim["job_id"],
            commit=False,
        )
        con.execute(
            "UPDATE claims SET state='submitted',reservation_owner=?,"
            "attempts=MAX(attempts,1),updated_at_ms=? WHERE job_id=?",
            (owner, now, claim["job_id"]),
        )
        adopted += 1
    con.commit()
    registry.db.commit()
    con.close()
    registry.close()
    return {"adopted": adopted, "scheduler_jobs": len(scheduler)}


def is_infrastructure_failure(job: dict) -> bool:
    return (
        job.get("state") == "failed"
        and str(job.get("failure_class") or "").lower() in INFRASTRUCTURE_FAILURES
    )


def state_db() -> sqlite3.Connection:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE, timeout=60.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=60000")
    con.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS claims(
          manifest_sha256 TEXT PRIMARY KEY,
          manifest TEXT NOT NULL UNIQUE,
          job_id TEXT NOT NULL UNIQUE,
          spec TEXT NOT NULL,
          state TEXT NOT NULL,
          attempts INTEGER NOT NULL DEFAULT 0,
          reservation_owner TEXT,
          updated_at_ms INTEGER NOT NULL
        );
        """
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(claims)")}
    if "reservation_owner" not in columns:
        con.execute("ALTER TABLE claims ADD COLUMN reservation_owner TEXT")
    con.commit()
    return con


def stage_evaluator() -> Path:
    return Path(active_evaluator(ROOT, EVALUATOR_SOURCE)["evaluator"])


def load_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    task = value.get("task")
    authority = value.get("label_authority") or {}
    if value.get("schema") != "JopeFlywheelCorpusEvalShard.v1":
        raise RuntimeError(f"{path.name}: wrong schema")
    if task not in EXPECTED_METRIC or not 1 <= len(value.get("candidates") or []) <= 100:
        raise RuntimeError(f"{path.name}: invalid task/candidate count")
    if manifest_authority_kind(authority, task=task) is None:
        raise RuntimeError(f"{path.name}: manifest self-play authority is invalid")
    semantics = []
    for candidate in value["candidates"]:
        computed = sha256_text(candidate["canonical_genome"])
        if candidate.get("semantic_sha256") != computed:
            raise RuntimeError(f"{path.name}: candidate semantic hash drift")
        semantics.append(computed)
    if len(semantics) != len(set(semantics)):
        raise RuntimeError(f"{path.name}: duplicate candidate")
    for split in ("train", "test"):
        source = Path(value[f"{split}_csv"])
        if not source.is_file():
            raise RuntimeError(f"{path.name}: {split} content pin drift")
        stat = source.stat()
        cached = _FILE_HASH_CACHE.get(source)
        identity = (stat.st_size, stat.st_mtime_ns)
        digest = (
            cached[2]
            if cached is not None and cached[:2] == identity
            else sha256_file(source)
        )
        _FILE_HASH_CACHE[source] = (*identity, digest)
        if digest != value[f"{split}_sha256"]:
            raise RuntimeError(f"{path.name}: {split} content pin drift")
    return value


def quoted(value: object) -> str:
    return json.dumps(str(value))


def input_lines(name: str, source: Path, target: str, digest: str | None = None) -> list[str]:
    return [
        "[[input]]",
        f"name = {quoted(name)}",
        f"source = {quoted(source)}",
        f"target = {quoted(target)}",
        f"sha256 = {quoted(digest or sha256_file(source))}",
        "",
    ]


def render_spec(manifest_path: Path, evaluator: Path) -> tuple[str, Path, str]:
    manifest = load_manifest(manifest_path)
    authority_kind = manifest_authority_kind(
        manifest["label_authority"], task=manifest["task"]
    )
    lane = str(manifest.get("evaluation_lane") or "")
    project = (
        GATE_PROJECT
        if lane == "checkpoint_gate"
        else "jope-selfplay-fresh"
        if manifest.get("checkpoint_epoch")
        and freshness_scoped_manifest(manifest)
        else "jope-selfplay-eval"
        if authority_kind == "corrected"
        else "jope-testonly"
    )
    # `validate-manifest` is a whole-shard, fail-closed pre-flight: one
    # task-invalid candidate rejects the entire manifest. That is right for the
    # adaptive wave lane (every candidate comes from the GP and must be
    # evaluable), and WRONG for the checkpoint gate, where per-candidate
    # validity is exactly the signal being measured: a checkpoint that proposes
    # a regression-only head for a binary dataset must be charged for it through
    # failure_rate, not have its manifest silently refused. The runtime
    # evaluator already handles this correctly - it emits a per-candidate
    # receipt with valid=false/failure="fit" and still exits 0 - so the gate
    # lane submits on per-candidate semantics instead.
    if lane != "checkpoint_gate":
        validation = command(
            [str(evaluator), "validate-manifest", "--manifest", str(manifest_path)]
        )
        if validation.returncode:
            raise RuntimeError(
                f"{manifest_path.name}: evaluator manifest validation failed: "
                f"{(validation.stderr or validation.stdout).strip()}"
            )
    manifest_hash = sha256_file(manifest_path)
    job_id = f"jte-{manifest_hash[:24]}"
    path = SPECS / f"{job_id}.toml"
    lines = [
        "[job]",
        f"id = {quoted(job_id)}",
        f'project = "{project}"',
        'tier = "normal"',
        'command = "/usr/bin/env"',
        (
            'args = ["{input:runner}", "{input:manifest}", "{input:train}", '
            '"{input:test}", "{input:evaluator}", "{output:receipt}"]'
        ),
        "",
        "[resource]",
        "gpus_per_task = 0",
        "cpus_reserved = 1",
        "omp_threads = 1",
        "ram_mb = 768",
        "gang_size = 1",
        'placement = "single_node"',
        "",
        "[env]",
        'RAYON_NUM_THREADS = "1"',
        'OMP_NUM_THREADS = "1"',
        'OPENBLAS_NUM_THREADS = "1"',
        'MKL_NUM_THREADS = "1"',
        'NUMEXPR_NUM_THREADS = "1"',
        "",
        "[preempt]",
        "preemptible = true",
        'checkpoint = "none"',
        'resume = "rerun"',
        "",
        "[limits]",
        "max_runtime_hours = 24",
        "max_restarts = 1",
        "",
    ]
    for item in (
        ("runner", RUNNER, "runner.sh", None),
        ("manifest", manifest_path, "manifest.json", manifest_hash),
        ("train", Path(manifest["train_csv"]), "train.csv", manifest["train_sha256"]),
        ("test", Path(manifest["test_csv"]), "test.csv", manifest["test_sha256"]),
        ("evaluator", evaluator, "corpus_eval", sha256_file(evaluator)),
    ):
        lines.extend(input_lines(*item))
    lines.extend(
        [
            "[[output]]",
            'name = "receipt"',
            'source = "receipt"',
            "directory = true",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return job_id, path, manifest_hash


def check_one(spec: Path) -> tuple[Path, str | None]:
    result = retry_command([str(SCQ), "check", str(spec)])
    detail = (result.stderr or result.stdout).strip()
    if result.returncode == 0:
        return spec, None
    return spec, detail


def prepare(limit: int) -> list[tuple[str, Path, str, Path]]:
    evaluator = stage_evaluator()
    con = state_db()
    registry = PairRegistry()
    claimed = {row["manifest_sha256"] for row in con.execute("SELECT manifest_sha256 FROM claims")}
    rendered = []
    manifests = manifest_paths()
    for manifest in manifests:
        digest = sha256_file(manifest)
        if digest in claimed:
            continue
        manifest_value = load_manifest(manifest)
        if not manifest_is_current(manifest):
            continue
        owner = registry.manifest_owner(digest)
        if owner is None:
            owner = f"manifest:{digest}"
            reserved, reused = registry.reserve_manifest(
                manifest_value,
                owner=owner,
                manifest_path=str(manifest),
                manifest_sha256=digest,
            )
            if reused:
                registry.close()
                con.close()
                raise RuntimeError(
                    f"{manifest.name}: {len(reused)} protocol pairs were already reserved"
                )
        elif not set(registry.manifest_state_counts(digest)).issubset(
            {"reserved", "infrastructure_retry"}
        ):
            continue
        registry.authorize_submission(digest, owner)
        job_id, spec, digest = render_spec(manifest, evaluator)
        rendered.append((job_id, spec, digest, manifest, owner))
        if len(rendered) == limit:
            break
    prestage_rows([row[:4] for row in rendered], workers=64)
    failures = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        future_map = {pool.submit(check_one, row[1]): row for row in rendered}
        for future in as_completed(future_map):
            _, error = future.result()
            if error:
                failures.append((future_map[future], error))
    if failures:
        first, error = failures[0]
        raise RuntimeError(f"{len(failures)} specs failed scq check; {first[1].name}: {error}")
    now = int(time.time() * 1000)
    con.executemany(
        """
        INSERT INTO claims(
          manifest_sha256,manifest,job_id,spec,state,attempts,
          reservation_owner,updated_at_ms
        ) VALUES(?,?,?,?,?,?,?,?)
        """,
        [
            (digest, str(manifest), job_id, str(spec), "checked", 0, owner, now)
            for job_id, spec, digest, manifest, owner in rendered
        ],
    )
    con.commit()
    con.close()
    registry.close()
    return [row[:4] for row in rendered]


def recover_infrastructure_failures(
    scheduler: dict[str, dict],
) -> list[tuple[str, Path, str, Path]]:
    con = state_db()
    registry = PairRegistry()
    recovered = []
    now = int(time.time() * 1000)
    for claim in con.execute(
        "SELECT * FROM claims WHERE state='submitted' AND attempts BETWEEN 1 AND 5 "
        "ORDER BY job_id"
    ).fetchall():
        if not manifest_is_current(Path(claim["manifest"])):
            continue
        job = scheduler.get(claim["job_id"])
        if not job or not is_infrastructure_failure(job):
            continue
        registry.record_manifest_infrastructure_failure(
            claim["manifest_sha256"],
            str(job.get("failure_class") or "infrastructure"),
        )
        old_job = claim["job_id"]
        new_job = f"{old_job}-r{int(claim['attempts'])}"
        old_spec = Path(claim["spec"])
        new_spec = old_spec.with_name(f"{new_job}.toml")
        text = old_spec.read_text()
        old_line = f"id = {json.dumps(old_job)}"
        new_line = f"id = {json.dumps(new_job)}"
        if old_line not in text:
            raise RuntimeError(f"cannot recover {old_job}: job id is absent from spec")
        new_spec.write_text(text.replace(old_line, new_line, 1))
        checked = command([str(SCQ), "check", str(new_spec)])
        if checked.returncode:
            raise RuntimeError(
                f"recovery spec check failed for {new_job}: "
                f"{(checked.stderr or checked.stdout).strip()}"
            )
        con.execute(
            """
            UPDATE claims SET job_id=?,spec=?,state='checked',updated_at_ms=?
            WHERE manifest_sha256=?
            """,
            (new_job, str(new_spec), now, claim["manifest_sha256"]),
        )
        recovered.append(
            (
                new_job,
                new_spec,
                claim["manifest_sha256"],
                Path(claim["manifest"]),
            )
        )
    con.commit()
    con.close()
    registry.close()
    return recovered


def recover_orphaned_submissions(
    scheduler: dict[str, dict], *, grace_seconds: int = 120
) -> int:
    """Retry claims forgotten by SCQ only when no authoritative receipt exists."""

    con = state_db()
    registry = PairRegistry()
    cutoff = int(time.time() * 1000) - grace_seconds * 1_000
    recovered = 0
    for claim in con.execute(
        "SELECT * FROM claims WHERE "
        "(state='submitted' OR state LIKE 'ingest_failed:%no such job%') "
        "AND updated_at_ms<? ORDER BY job_id",
        (cutoff,),
    ).fetchall():
        job_id = claim["job_id"]
        if job_id in scheduler:
            continue
        local_receipt = (
            SCQ_RUNS
            / job_id
            / "attempts"
        )
        has_receipt = (ROOT / "receipts" / job_id / "SHARD_OK").is_file() or any(
            (attempt / "outputs" / "receipt" / "SHARD_OK").is_file()
            for attempt in local_receipt.glob("*")
        )
        if has_receipt:
            continue
        registry.record_manifest_infrastructure_failure(
            claim["manifest_sha256"],
            "scheduler-state-lost",
        )
        con.execute(
            "UPDATE claims SET state='checked',updated_at_ms=? "
            "WHERE manifest_sha256=?",
            (int(time.time() * 1000), claim["manifest_sha256"]),
        )
        recovered += 1
    con.commit()
    con.close()
    registry.close()
    return recovered


def pending_checked_claims(
    scheduler: dict[str, dict],
) -> list[tuple[str, Path, str, Path]]:
    """Revalidate claims whose bulk submitter stopped before SCQ accepted them."""
    con = state_db()
    evaluator = stage_evaluator()
    evaluator_hash = sha256_file(evaluator)
    rows = []
    now = int(time.time() * 1000)
    for claim in con.execute(
        "SELECT * FROM claims WHERE state='checked' ORDER BY job_id"
    ).fetchall():
        if claim["job_id"] in scheduler:
            continue
        manifest = Path(claim["manifest"])
        if not manifest_is_current(manifest):
            continue
        prior_spec = Path(claim["spec"])
        if (
            prior_spec.is_file()
            and f'sha256 = "{evaluator_hash}"' in prior_spec.read_text()
        ):
            rows.append(
                (
                    claim["job_id"],
                    prior_spec,
                    claim["manifest_sha256"],
                    manifest,
                )
            )
            continue
        try:
            job_id, spec, digest = render_spec(manifest, evaluator)
        except RuntimeError as error:
            if "not binary-task-valid" not in str(error):
                con.close()
                raise
            con.execute(
                "UPDATE claims SET state='rejected_task_invalid',updated_at_ms=? "
                "WHERE manifest_sha256=?",
                (now, claim["manifest_sha256"]),
            )
            continue
        if job_id != claim["job_id"] or digest != claim["manifest_sha256"]:
            con.close()
            raise RuntimeError(f"{manifest.name}: checked claim identity drift")
        rows.append((job_id, spec, digest, manifest))
    con.commit()
    con.close()
    return rows


def submit_one(row: tuple[str, Path, str, Path]) -> tuple[str, bool, str]:
    job_id, spec, _, _ = row
    result = retry_command([str(SCQ), "submit", str(spec)])
    detail = (result.stderr or result.stdout).strip()
    lowered = detail.lower()
    adopted = "already exists" in lowered or "already active" in lowered
    return job_id, result.returncode == 0 or adopted, detail


def bounded_submission_count(needed: int) -> int:
    return max(0, min(int(needed), SUBMISSION_BATCH_LIMIT))


def submission_locality_key(
    row: tuple[str, Path, str, Path]
) -> tuple[str, str, int, int, str]:
    manifest = json.loads(row[3].read_text())
    genomes = [
        str(candidate.get("canonical_genome") or "")
        for candidate in manifest.get("candidates") or []
    ]
    head_cost = max(
        (
            (
                3
                if "(gpr " in genome
                else 2
                if any(
                    head in genome
                    for head in ("(histgbm", "(tab_boost", "(extra_trees")
                )
                else 1
            )
            for genome in genomes
        ),
        default=0,
    )
    input_paths = [
        Path(str(manifest.get(f"{split}_csv") or ""))
        for split in ("train", "test")
        if manifest.get(f"{split}_csv")
    ]
    input_bytes = sum(path.stat().st_size for path in input_paths if path.is_file())
    size_bucket = 0 if input_bytes < 1_000_000 else 1 if input_bytes < 10_000_000 else 2
    return (
        manifest["train_sha256"],
        manifest["test_sha256"],
        head_cost,
        size_bucket,
        row[0],
    )


def stage_blob(source: Path, digest: str, blob_dir: Path = BLOB_DIR) -> tuple[bool, int]:
    destination = blob_dir / digest
    if destination.is_file():
        return False, destination.stat().st_size
    blob_dir.mkdir(parents=True, exist_ok=True)
    temporary = blob_dir / f".{digest}.partial-prestage-{os.getpid()}"
    try:
        shutil.copy2(source, temporary)
        actual = sha256_file(temporary)
        if actual != digest:
            raise RuntimeError(
                f"{source}: content pin drift while staging: expected {digest}, got {actual}"
            )
        os.replace(temporary, destination)
        return True, destination.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def prestage_rows(
    rows: list[tuple[str, Path, str, Path]],
    workers: int,
    blob_dir: Path | None = None,
) -> dict[str, int]:
    cache = blob_dir or BLOB_DIR
    paths: dict[str, Path] = {}
    for _, _, manifest_digest, manifest_path in rows:
        paths.setdefault(manifest_digest, manifest_path)
        manifest = json.loads(manifest_path.read_text())
        for split in ("train", "test"):
            paths.setdefault(
                manifest[f"{split}_sha256"], Path(manifest[f"{split}_csv"])
            )
    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(stage_blob, source, digest, cache)
            for digest, source in paths.items()
            if not (cache / digest).is_file()
        ]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return {
        "unique_inputs": len(paths),
        "staged": sum(staged for staged, _ in outcomes),
        "staged_bytes": sum(size for staged, size in outcomes if staged),
    }


def prestage_checked_inputs(workers: int) -> dict:
    con = state_db()
    paths: dict[str, Path] = {}
    for claim in con.execute(
        "SELECT manifest FROM claims WHERE state='checked' ORDER BY job_id"
    ).fetchall():
        manifest = json.loads(Path(claim["manifest"]).read_text())
        for split in ("train", "test"):
            paths.setdefault(
                manifest[f"{split}_sha256"], Path(manifest[f"{split}_csv"])
            )
    con.close()
    outcomes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(stage_blob, source, digest)
            for digest, source in paths.items()
            if not (BLOB_DIR / digest).is_file()
        ]
        for future in as_completed(futures):
            outcomes.append(future.result())
    return {
        "unique_inputs": len(paths),
        "staged": sum(staged for staged, _ in outcomes),
        "staged_bytes": sum(size for staged, size in outcomes if staged),
    }


def refill(target: int, hard_cap: int, *, synchronous_prepare: bool = False) -> dict:
    scheduler_rows = jobs()
    scheduler = {row["id"]: row for row in scheduler_rows if row.get("id")}
    adoption = adopt_scheduler_claims(scheduler_rows)
    orphaned = recover_orphaned_submissions(scheduler)
    live = [
        row
        for row in scheduler_rows
        if str(row.get("id") or "").startswith("jte-")
        and not str(row.get("id") or "").startswith("jte-canary-")
        and row.get("project") in EVALUATION_PROJECTS
        and row.get("state") in ALLOWED_LIVE_STATES
        and not stale_dispatch(row)
    ]
    state_counts = {
        state: sum(row.get("state") == state for row in live)
        for state in ALLOWED_LIVE_STATES
    }
    useful_depth = sum(state_counts.get(state, 0) for state in USEFUL_QUEUE_STATES)
    needed = queue_refill_needed(
        useful_depth, target=target, hard_cap=hard_cap
    )
    if not needed:
        return {
            "live": len(live),
            "useful_depth": useful_depth,
            "useful_depth_after": useful_depth,
            "queued_starting": useful_depth,
            "queued_starting_after": useful_depth,
            "maximum_queue_age_seconds": maximum_queue_age_seconds(live),
            "running": state_counts.get("running", 0),
            "dispatching": state_counts.get("dispatching", 0),
            "held": state_counts.get("held", 0),
            "needed": 0,
            "submitted": 0,
            "failed": 0,
            "orphaned_recovered": orphaned,
        }
    con = state_db()
    registry = PairRegistry()
    for claim in con.execute(
        "SELECT manifest_sha256 FROM claims WHERE reservation_owner IS NULL"
    ).fetchall():
        owner = registry.manifest_owner(claim["manifest_sha256"])
        if owner is not None:
            con.execute(
                "UPDATE claims SET reservation_owner=? WHERE manifest_sha256=?",
                (owner, claim["manifest_sha256"]),
            )
    con.commit()
    con.close()
    registry.close()
    batch_needed = bounded_submission_count(needed)
    rows = recover_infrastructure_failures(scheduler)
    rows.extend(pending_checked_claims(scheduler))
    rows = [row for row in rows if manifest_is_current(row[3])]
    rows = rows[:batch_needed]
    if synchronous_prepare:
        rows.extend(prepare(max(0, batch_needed - len(rows))))
    rows.sort(key=submission_locality_key)
    con = state_db()
    registry = PairRegistry()
    for job_id, _, manifest_sha256, _ in rows:
        claim = con.execute(
            "SELECT reservation_owner FROM claims WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if claim is None or claim["reservation_owner"] is None:
            registry.close()
            con.close()
            raise RuntimeError(f"{job_id}: launch-ready claim has no reservation owner")
        registry.authorize_submission(
            manifest_sha256,
            claim["reservation_owner"],
            job_id=job_id,
        )
    registry.close()
    con.close()
    outcomes = []
    with ThreadPoolExecutor(max_workers=128) as pool:
        futures = [pool.submit(submit_one, row) for row in rows]
        for future in as_completed(futures):
            outcomes.append(future.result())
    con = state_db()
    registry = PairRegistry()
    now = int(time.time() * 1000)
    for job_id, success, detail in outcomes:
        claim = con.execute(
            "SELECT manifest_sha256,reservation_owner FROM claims WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if success and claim is not None:
            registry.record_submission(
                claim["manifest_sha256"],
                claim["reservation_owner"],
                job_id,
            )
        transient = not success and infrastructure_error(detail)
        con.execute(
            """
            UPDATE claims SET state=?,
              attempts=attempts+CASE WHEN ? THEN 1 ELSE 0 END,
              updated_at_ms=?
            WHERE job_id=?
            """,
            (
                (
                    "submitted"
                    if success
                    else "checked"
                    if transient
                    else f"submit_failed:{detail[:500]}"
                ),
                int(success),
                now,
                job_id,
            ),
        )
    con.commit()
    con.close()
    registry.close()
    failures = [row for row in outcomes if not row[1]]
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(outcomes)} SCQ submissions failed")
    after = strict_live_jobs()
    after_counts = {
        state: sum(row.get("state") == state for row in after)
        for state in ALLOWED_LIVE_STATES
    }
    result = {
        "live_before": len(live),
        "useful_depth_before": useful_depth,
        "queued_starting_before": useful_depth,
        "running_before": state_counts.get("running", 0),
        "dispatching_before": state_counts.get("dispatching", 0),
        "held_before": state_counts.get("held", 0),
        "maximum_queue_age_seconds_before": maximum_queue_age_seconds(live),
        "needed": needed,
        "checked": len(rows),
        "submitted": len(outcomes),
        "failed": len(failures),
        "live_after": len(after),
        "useful_depth_after": sum(
            after_counts.get(state, 0) for state in USEFUL_QUEUE_STATES
        ),
        "queued_starting_after": sum(
            after_counts.get(state, 0) for state in USEFUL_QUEUE_STATES
        ),
        "running_after": after_counts.get("running", 0),
        "dispatching_after": after_counts.get("dispatching", 0),
        "held_after": after_counts.get("held", 0),
        "maximum_queue_age_seconds": maximum_queue_age_seconds(after),
        "ready_after": ready_manifest_count(),
        "unclaimed_after": unclaimed_manifest_count(),
        "adopted": adoption["adopted"],
        "orphaned_recovered": orphaned,
    }
    atomic_json(ROOT / "ops" / "last-refill.json", result)
    return result


def record_burn_rate(scheduler_rows: list[dict]) -> list[float]:
    con = state_db()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS terminal_jobs(
          job_id TEXT PRIMARY KEY,observed_at_ms INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS burn_samples(
          observed_at_ms INTEGER PRIMARY KEY,jobs_per_minute REAL NOT NULL
        ) WITHOUT ROWID;
        """
    )
    timestamp = int(time.time() * 1000)
    terminal = [
        str(row["id"])
        for row in scheduler_rows
        if str(row.get("id") or "").startswith("jte-")
        and not str(row.get("id") or "").startswith("jte-canary-")
        and row.get("project") in EVALUATION_PROJECTS
        and row.get("state") in TERMINAL_STATES
    ]
    prior = {
        row[0]
        for row in con.execute(
            "SELECT job_id FROM terminal_jobs WHERE job_id IN "
            f"({','.join('?' for _ in terminal)})",
            terminal,
        )
    } if terminal else set()
    new = [job_id for job_id in terminal if job_id not in prior]
    con.executemany(
        "INSERT OR IGNORE INTO terminal_jobs VALUES(?,?)",
        [(job_id, timestamp) for job_id in new],
    )
    previous = con.execute(
        "SELECT MAX(observed_at_ms) FROM burn_samples"
    ).fetchone()[0]
    elapsed_minutes = max(
        5.0 / 60.0,
        (timestamp - int(previous)) / 60_000.0 if previous else 1.0,
    )
    con.execute(
        "INSERT OR REPLACE INTO burn_samples VALUES(?,?)",
        (timestamp, 0.0 if previous is None else len(new) / elapsed_minutes),
    )
    cutoff = timestamp - 60 * 60_000
    con.execute("DELETE FROM burn_samples WHERE observed_at_ms<?", (cutoff,))
    values = [
        float(row[0])
        for row in con.execute(
            "SELECT jobs_per_minute FROM burn_samples ORDER BY observed_at_ms"
        )
    ]
    con.commit()
    con.close()
    return values


def dynamic_refill() -> dict:
    scheduler_rows = jobs()
    burn = record_burn_rate(scheduler_rows)
    capacity = cluster_slot_inventory()
    target, hard_cap, p95 = dynamic_queue_targets(
        burn, capacity["effective_cluster_slots"]
    )
    result = refill(target, hard_cap, synchronous_prepare=False)
    return {
        **result,
        "queued_target": target,
        "queued_hard_cap": hard_cap,
        "p95_jobs_per_minute": p95,
        **capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--limit", type=int, default=5000)
    refill_command = commands.add_parser("refill")
    refill_command.add_argument("--target", type=int, default=256)
    refill_command.add_argument("--hard-cap", type=int, default=512)
    refill_command.add_argument("--synchronous-prepare", action="store_true")
    commands.add_parser("dynamic-refill")
    prestage = commands.add_parser("prestage")
    prestage.add_argument("--workers", type=int, default=8)
    commands.add_parser("status")
    commands.add_parser("inventory")
    commands.add_parser("adopt")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"checked": len(prepare(args.limit))}, sort_keys=True))
    elif args.command == "refill":
        print(
            json.dumps(
                refill(
                    args.target,
                    args.hard_cap,
                    synchronous_prepare=args.synchronous_prepare,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "dynamic-refill":
        print(json.dumps(dynamic_refill(), sort_keys=True))
    elif args.command == "status":
        live = strict_live_jobs()
        states: dict[str, int] = {}
        for row in live:
            states[row["state"]] = states.get(row["state"], 0) + 1
        print(json.dumps({"live": len(live), "states": states}, sort_keys=True))
    elif args.command == "inventory":
        print(
            json.dumps(
                {
                    "ready": ready_manifest_count(),
                    "unclaimed": unclaimed_manifest_count(),
                },
                sort_keys=True,
            )
        )
    elif args.command == "prestage":
        print(json.dumps(prestage_checked_inputs(args.workers), sort_keys=True))
    else:
        print(json.dumps(adopt_scheduler_claims(), sort_keys=True))


if __name__ == "__main__":
    main()
