from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from test_exact_common import atomic_json, sha256_file
from test_exact_service_runtime import (
    ROOT,
    SCQ,
    SCRIPTS,
    SCQD,
    burn_targets,
    loaded_bundle,
    mark_micro_wave_request,
    parse_result,
    pending_micro_wave_request,
    run,
    start_wave_generation,
    wave_generation_state,
)

def ensure_current_proposals(cohort: Path | None = None) -> dict:
    """Submit or harvest proposals after the micro-cohort is frozen."""

    from test_exact_controller import _fetch_job_output, _scheduler_jobs
    from test_exact_ops import render_checkpoint_proposal_spec
    from test_exact_wave import checkpoint

    checkpoint_path, checkpoint_hash = checkpoint()
    cohort_hash = sha256_file(cohort) if cohort is not None else None
    pointer_path = ROOT / "proposals" / "current.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text())
        proposal_path = Path(pointer.get("proposals", ""))
        if (
            pointer.get("checkpoint_sha256") == checkpoint_hash
            and pointer.get("cohort_sha256") == cohort_hash
            and proposal_path.is_file()
            and sha256_file(proposal_path) == pointer.get("proposals_sha256")
        ):
            return {
                "ready": True,
                "checkpoint_sha256": checkpoint_hash,
                "proposals": str(proposal_path),
                "proposals_sha256": pointer["proposals_sha256"],
            }
    legacy = ROOT / "proposals" / "dl-proposals.jsonl"
    legacy_summary = legacy.with_suffix(".summary.json")
    if cohort is None and legacy.is_file() and legacy_summary.is_file():
        summary = json.loads(legacy_summary.read_text())
        catalog_pointer = json.loads((ROOT / "catalog" / "current.json").read_text())
        catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
        expected_counts = {
            "regression": int(catalog["regression_count"]) * 64,
            "binary": int(catalog["binary_count"]) * 64,
        }
        if (
            summary.get("model_checkpoint_sha256") == checkpoint_hash
            and summary.get("catalog_sha256") == catalog_pointer["catalog_sha256"]
            and summary.get("counts") == expected_counts
            and sha256_file(legacy) == summary.get("output_sha256")
        ):
            archive = (
                ROOT
                / "proposals"
                / "by-checkpoint"
                / f"{checkpoint_hash}.jsonl"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.is_file():
                shutil.copy2(legacy, archive)
            if sha256_file(archive) != summary["output_sha256"]:
                raise RuntimeError("legacy proposal archive hash drift")
            atomic_json(
                pointer_path,
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "proposals": str(archive),
                    "proposals_sha256": summary["output_sha256"],
                    "summary": str(legacy_summary),
                    "summary_sha256": sha256_file(legacy_summary),
                    "legacy_adopted": True,
                },
            )
            return {
                "ready": True,
                "checkpoint_sha256": checkpoint_hash,
                "proposals": str(archive),
                "proposals_sha256": summary["output_sha256"],
                "legacy_adopted": True,
            }
    work = (
        ROOT
        / "proposals"
        / "jobs"
        / checkpoint_hash
        / (cohort_hash or "legacy-full-catalog")
    )
    work.mkdir(parents=True, exist_ok=True)
    rendered = render_checkpoint_proposal_spec(
        work / "proposal.toml",
        checkpoint=checkpoint_path,
        split=None,
        seed=20260726,
        proposals_per_dataset=64,
        cohort=cohort,
    )
    state = _scheduler_jobs().get(rendered["job_id"], {}).get("state")
    if state in {"failed", "cancelled"}:
        raise RuntimeError(f"micro-cohort proposal job failed: {state}")
    if state is None:
        checked = run([str(SCQ), "check", rendered["spec"]], timeout=900)
        if checked.returncode:
            raise RuntimeError((checked.stderr or checked.stdout).strip())
        submitted = run([str(SCQ), "submit", rendered["spec"]], timeout=900)
        detail = (submitted.stderr or submitted.stdout).strip()
        if submitted.returncode and "already exists" not in detail.lower():
            raise RuntimeError(detail)
        return {
            "pending": True,
            "phase": "submitted_micro_cohort_proposals",
            "job_id": rendered["job_id"],
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
        }
    if state != "completed":
        return {
            "pending": True,
            "phase": "waiting_for_micro_cohort_proposals",
            "job_id": rendered["job_id"],
            "job_state": state,
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
        }
    output = _fetch_job_output(rendered["job_id"], work / "fetched")
    proposals = list(output.rglob("dl-proposals.jsonl"))
    summaries = list(output.rglob("dl-proposals.summary.json"))
    if len(proposals) != 1 or len(summaries) != 1:
        raise RuntimeError("micro-cohort proposal fetch is incomplete")
    summary = json.loads(summaries[0].read_text())
    if (
        summary.get("model_checkpoint_sha256") != checkpoint_hash
        or summary.get("cohort_sha256") != cohort_hash
        or summary.get("split") is not None
        or int(summary.get("proposals_per_dataset", 0)) != 64
        or sha256_file(proposals[0]) != summary.get("output_sha256")
    ):
        raise RuntimeError("micro-cohort proposal provenance drift")
    archive = (
        ROOT
        / "proposals"
        / "by-checkpoint"
        / f"{checkpoint_hash}-{cohort_hash or 'all'}.jsonl"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        shutil.copy2(proposals[0], archive)
    if sha256_file(archive) != summary["output_sha256"]:
        raise RuntimeError("archived micro-cohort proposal hash drift")
    atomic_json(
        pointer_path,
        {
            "checkpoint_sha256": checkpoint_hash,
            "cohort_sha256": cohort_hash,
            "proposals": str(archive),
            "proposals_sha256": sha256_file(archive),
            "summary": str(summaries[0]),
            "summary_sha256": sha256_file(summaries[0]),
        },
    )
    return {
        "ready": True,
        "checkpoint_sha256": checkpoint_hash,
        "cohort_sha256": cohort_hash,
        "proposals": str(archive),
        "proposals_sha256": sha256_file(archive),
    }


def controller_cycle() -> dict:
    from test_exact_evaluator import (
        active as active_evaluator,
        candidate as candidate_evaluator,
        promote_candidate,
    )

    active = active_evaluator()
    candidate = candidate_evaluator()
    canary = None
    if candidate["evaluator_sha256"] != active["evaluator_sha256"]:
        report = (
            ROOT
            / "canary"
            / f"{candidate['evaluator_sha256']}.candidate.json"
        )
        canary = parse_result(
            run(
                [
                    sys.executable,
                    str(SCRIPTS / "test_exact_canary.py"),
                    "--evaluator",
                    candidate["evaluator"],
                    "--report",
                    str(report),
                    "--timeout",
                    "900",
                ],
                timeout=1_200,
            )
        )
        active = promote_candidate(report)
    controller = parse_result(
        run(
            [sys.executable, str(SCRIPTS / "test_exact_controller.py")],
            timeout=3_600,
        )
    )
    return {
        "evaluator": {
            "active": active["evaluator_sha256"],
            "candidate_canary": canary,
        },
        "flywheel": controller,
    }


def watchdog_cycle(state: dict) -> dict:
    ping = run([str(SCQ), "ping"], timeout=5)
    if ping.returncode == 0:
        return {
            "consecutive_failures": 0,
            "recovered": False,
            "ping": "ok",
        }
    failures = int(state.get("consecutive_failures", 0)) + 1
    if failures < 3:
        return {
            "consecutive_failures": failures,
            "recovered": False,
            "ping": "failed",
            "detail": (ping.stderr or ping.stdout).strip()[-1_000:],
        }
    started = time.monotonic()
    recovery = run([str(SCQD), "start"], timeout=65)
    if recovery.returncode:
        raise RuntimeError((recovery.stderr or recovery.stdout).strip())
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if run([str(SCQ), "ping"], timeout=5).returncode == 0:
            adopted = parse_result(
                run(
                    [
                        sys.executable,
                        str(SCRIPTS / "test_exact_backlog.py"),
                        "adopt",
                    ],
                    timeout=900,
                )
            )
            return {
                "consecutive_failures": 0,
                "recovered": True,
                "recovery_seconds": time.monotonic() - started,
                "jobs_re_adopted": adopted,
                "existing_jobs_cancelled": 0,
            }
        time.sleep(1)
    raise RuntimeError("SCQ controller did not recover within 60 seconds")


