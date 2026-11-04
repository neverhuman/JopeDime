"""Q1 FIFO metadata preparation and dry-run helpers."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

import q1
import q1_cut

ROOT = Path(__file__).resolve().parents[4]
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
LAUNCHER = ROOT / "deploy/qualification/q1/harness/fifo_launch.py"

@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    driver: Path = ROOT / "deploy/qualification/q1/harness/q1.py"
    prereg: Path = ROOT / "deploy/qualification/q1/PREREGISTRATION.md"
    fifo: Path = ROOT / "research/flywheel_selfplay/qualification_lock_fifo.py"
    cohort: Path = ROOT / "deploy/qualification/q1/cohort.json"
    input_manifest: Path = ROOT / "deploy/qualification/q1/inputs-manifest.json"
    amendment: Path = ROOT / "deploy/qualification/q1/amendments/t0610.json"
    cut_definitions: Path = ROOT / "deploy/qualification/q1/cuts"
    lock: Path = Path.home() / "jope_plan/qualification.lock"
    queue: Path = Path.home() / "jope_plan/qualification.queue"


def write_new(path, value):
    with Path(path).open("x") as stream:
        q1.append_receipt(stream, value)


def planning(dataset_count=None):
    count = 40 if dataset_count is None else dataset_count
    return {
        "dataset_count": dataset_count,
        "minimum_families": 40,
        "policy_runs": 6 * count,
        "planned_charged_admissions": 6 * count * 24,
        "serial_timeout_seconds": 6 * count * 600,
        "expected_wall_seconds": None,
        "timing_basis": "No qualified backend timing; timeout arithmetic is not an ETA",
    }


def prepare(paths, owner):
    return {
        "schema": "jope.q1-fifo-intent/v1",
        "task": "v1/QLT-04",
        "owner": owner,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "-C", str(paths.root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "preregistration_sha256": q1.sha_file(paths.prereg),
        "launcher_sha256": q1.sha_file(LAUNCHER),
        "status": "prepared_not_enqueued",
        "active_ticket": None,
        "after": "W1 releases qualification.lock; then ordinary live-ticket FIFO order",
        "lock": str(paths.lock),
        "queue": str(paths.queue),
        "planning": planning(),
        "gate": [
            "OWNER signature over the final preregistration, plan and backend",
            "qualified production backend and actual admitted cohort/model proofs",
            "clean merged checkout matching the signed source; frozen DOPE",
            "canonical persistent ledger with no prior candidate consumption",
            "explicit wait/wall caps, CPU load <=85, oldest live ticket and exclusive lock",
        ],
        "owner_signers_pending": "Owner allowed-signers SHA-256: PENDING"
        in paths.prereg.read_text(),
        "study_started": False,
    }


def build_input_manifest(paths):
    cohort = q1.load(paths.cohort)
    q1.require(
        isinstance(cohort, dict)
        and cohort.get("schema") == "jope.q1.held-out-cohort/v1",
        "unsupported QLT-02 candidate cohort metadata",
    )
    datasets = []
    seen_ids = set()
    rows = cohort.get("datasets")
    q1.require(isinstance(rows, list), "invalid QLT-02 candidate dataset list")
    for row in rows:
        q1.require(isinstance(row, dict), "invalid QLT-02 candidate dataset entry")
        dataset_id = row.get("dataset_id")
        canonical_sha = row.get("sha256")
        manifest_sha = row.get("s3_manifest_sha256")
        q1.require(
            isinstance(dataset_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", dataset_id) is not None
            and isinstance(canonical_sha, str)
            and canonical_sha.startswith(dataset_id)
            and q1.sha(canonical_sha)
            and q1.sha(manifest_sha),
            "invalid QLT-02 candidate dataset identity or hash",
        )
        q1.require(
            dataset_id not in seen_ids, "duplicate QLT-02 candidate dataset identity"
        )
        seen_ids.add(dataset_id)
        dataset = {
            "dataset_id": dataset_id,
            "sha256": canonical_sha,
            "s3_manifest_sha256": manifest_sha,
        }
        cut_sha256 = row.get("cut_sha256")
        if cut_sha256 is not None:
            q1.require(q1.sha(cut_sha256), "invalid Q1 cut reference in roster")
            definition = q1_cut.load_content_addressed(
                paths.cut_definitions / f"{cut_sha256}.json",
                cut_sha256,
                expected_dataset_sha256=canonical_sha,
            )
            q1_cut.require_three_fold_fit_partition(definition)
            dataset["cut_sha256"] = cut_sha256
        datasets.append(dataset)
    q1.require(datasets, "no QLT-02 candidate metadata in the source cohort")
    q1.require(
        cohort.get("dataset_count") == len(datasets)
        and len(datasets) >= 40
        and isinstance(cohort.get("source"), dict)
        and q1.sha(cohort["source"].get("catalog_sha256")),
        "incomplete QLT-02 candidate cohort metadata",
    )
    return {
        "schema": "jope.q1-cpu-input-manifest/v1",
        "purpose": "candidate-cohort-identities-only; synthetic dry-run only",
        "source_cohort_sha256": q1.sha_file(paths.cohort),
        "source_catalog_sha256": cohort["source"]["catalog_sha256"],
        "dope_sha256": DOPE,
        "qualification_eligible": False,
        "payloads_included": False,
        "datasets": datasets,
    }


def load_amended_roster_packet(paths=Paths()):
    """Load the T0610 preregistration amendment, roster packet, and all 154 real FIT cuts."""
    amendment_path = Path(paths.amendment)
    q1.require(not amendment_path.is_symlink(), "amendment must not be a symlink")
    amendment = q1.load(amendment_path.resolve(strict=True))
    q1.require(
        isinstance(amendment, dict)
        and amendment.get("schema") == "jope.q1-prereg-amendment/v1"
        and amendment.get("amendment_id") == "QLT-04-T0610-R1-R2",
        "unsupported Q1 preregistration amendment",
    )
    roster_rel = amendment.get("roster_cut_packet_path")
    manifest_rel = amendment.get("fold_manifest_path")
    q1.require(
        isinstance(roster_rel, str)
        and isinstance(manifest_rel, str)
        and q1.sha(amendment.get("roster_cut_packet_sha256"))
        and q1.sha(amendment.get("fold_manifest_sha256")),
        "incomplete Q1 amendment references",
    )
    roster_path = (paths.root / roster_rel).resolve(strict=True)
    manifest_path = (paths.root / manifest_rel).resolve(strict=True)
    q1.require(
        not (paths.root / roster_rel).is_symlink()
        and not (paths.root / manifest_rel).is_symlink(),
        "amendment target files must not be symlinks",
    )
    q1.require(
        q1.sha_file(roster_path) == amendment["roster_cut_packet_sha256"],
        "amended roster packet hash differs from amendment",
    )
    q1.require(
        q1.sha_file(manifest_path) == amendment["fold_manifest_sha256"],
        "fold manifest hash differs from amendment",
    )
    packet = q1.load(roster_path)
    q1.require(
        isinstance(packet, dict)
        and packet.get("schema") == "jope.q1-amended-roster-cut-packet/v1"
        and packet.get("amendment_id") == amendment["amendment_id"],
        "unsupported Q1 amended roster packet",
    )
    rows = packet.get("official_roster")
    q1.require(
        isinstance(rows, list)
        and len(rows) == packet.get("official_roster_count") == 153,
        "invalid Q1 amended official roster count",
    )
    seen_ids = set()
    datasets = []
    for row in rows:
        q1.require(
            isinstance(row, dict) and row.get("owner_roster153_member") is True,
            "invalid official roster entry",
        )
        dataset_id = row.get("dataset_id")
        dataset_sha256 = row.get("dataset_sha256")
        train_blob_sha256 = row.get("train_blob_sha256")
        cut_sha256 = row.get("cut_sha256")
        q1.require(
            isinstance(dataset_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", dataset_id) is not None
            and isinstance(dataset_sha256, str)
            and dataset_sha256.startswith(dataset_id)
            and q1.sha(dataset_sha256)
            and q1.sha(train_blob_sha256)
            and q1.sha(cut_sha256)
            and row.get("cut_definition_path")
            == f"deploy/qualification/q1/cuts/{cut_sha256}.json",
            "invalid amended roster dataset or cut binding",
        )
        q1.require(
            dataset_id not in seen_ids, "duplicate amended roster dataset identity"
        )
        seen_ids.add(dataset_id)
        definition = q1_cut.load_content_addressed(
            paths.cut_definitions / f"{cut_sha256}.json",
            cut_sha256,
            expected_dataset_sha256=dataset_sha256,
            expected_train_blob_sha256=train_blob_sha256,
        )
        q1_cut.require_three_fold_fit_partition(definition)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "sha256": dataset_sha256,
                "train_blob_sha256": train_blob_sha256,
                "cut_sha256": cut_sha256,
            }
        )
    candidate = packet.get("catalog_wide_candidate")
    q1.require(
        isinstance(candidate, dict)
        and candidate.get("official_roster_member") is False
        and q1.sha(candidate.get("cut_sha256"))
        and q1.sha(candidate.get("dataset_sha256")),
        "invalid catalog wide candidate metadata",
    )
    candidate_cut = q1_cut.load_content_addressed(
        paths.cut_definitions / f"{candidate['cut_sha256']}.json",
        candidate["cut_sha256"],
        expected_dataset_sha256=candidate["dataset_sha256"],
    )
    q1_cut.require_three_fold_fit_partition(candidate_cut)
    return {
        "schema": "jope.q1-amended-roster-summary/v1",
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": q1.sha_file(amendment_path),
        "roster_cut_packet_sha256": amendment["roster_cut_packet_sha256"],
        "fold_manifest_sha256": amendment["fold_manifest_sha256"],
        "official_roster_count": len(datasets),
        "datasets": datasets,
        "qualification_eligible": False,
        "q1_status": "NOT_RUN",
    }


def load_input_manifest(path, paths):
    source = Path(path)
    q1.require(not source.is_symlink(), "input manifest must not be a symlink")
    manifest = q1.load(source.resolve(strict=True))
    q1.require(
        isinstance(manifest, dict)
        and manifest.get("schema") == "jope.q1-cpu-input-manifest/v1",
        "unsupported Q1 CPU input manifest",
    )
    q1.require(manifest.get("dope_sha256") == DOPE, "frozen DOPE differs")
    q1.require(
        manifest.get("source_cohort_sha256") == q1.sha_file(paths.cohort),
        "QLT-02 candidate source hash differs",
    )
    q1.require(
        manifest == build_input_manifest(paths),
        "candidate input manifest differs from its pinned cohort metadata",
    )
    return manifest


def dry_run(input_manifest, output, paths=Paths()):
    """Write an unknown Q1 decision skeleton without data, lock or queue access."""
    manifest = load_input_manifest(input_manifest, paths)
    source = subprocess.check_output(
        ["git", "-C", str(paths.root), "rev-parse", "HEAD"], text=True
    ).strip()
    record = {
        "schema": "jope.q1-decision-record/v1",
        "run_type": "synthetic_cpu_dry_run",
        "source_commit": source,
        "inputs_manifest_sha256": q1.sha_file(input_manifest),
        "dope_sha256": manifest["dope_sha256"],
        "q1_status": "NOT_RUN",
        "decision": "UNKNOWN_NOT_RUN",
        "primary_effect": None,
        "paired_interval_95": None,
        "subgroups": {name: None for name in ("small", "large", "narrow", "wide")},
        "failure_inventory": "NOT_RUN",
        "compute": {"q1_runs": "NOT_RUN", "cpu_seconds": None, "gpu_hours": None},
        "controls": {
            "qualification_lock_acquired": False,
            "fifo_ticket_created": False,
            "backend_invoked": False,
            "dataset_payloads_opened": False,
            "default_changed": False,
        },
        "real_s3_metadata_entries": len(manifest["datasets"]),
        "qualification_eligible": False,
    }
    write_new(output, record)
    return record
