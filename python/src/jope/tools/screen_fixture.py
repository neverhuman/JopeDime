"""H05 fixture: freeze identities before opening payloads. No numerical starts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA = "parallel.screen_fixture.v2"
GATES = dict(
    top10_recall=0.99,
    missed_champion=0,
    total_time_gain=1.25,
    candidates_per_context=64,
    retained_by_screen=16,
    deterministic_rejection_audits=5,
    training_families=8,
    existing_cuts_per_family=2,
    aggregate_evaluator_slots=16,
    context_deadline_seconds=600,
    numerical_starts_per_context=1000,
)
AXES = (
    dict(name="same_row_f32", precision="f32", same_rows=True,
         note="authority namespace required; not a teaching score"),
    dict(name="histgbm_proxy", replaces="histgbm2", with_head="histgbm",
         note="retain original and proxy identities separately"),
    dict(name="gpr_proxy", note="shorter optimization or smaller basis; separately identified proxy"),
    dict(name="row_fidelity", rows=[128, 256, "full"],
         note="only after same-row screening qualifies; training-only ordered nested rows"),
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def select_identities(census, limit_per_head=8):
    if census.get("unverified") or not census.get("shares"):
        return []
    chosen = []
    for share in census["shares"]:
        head = share["head"]
        if head in ("unknown",):
            continue
        chosen.append(dict(head=head, observations=share["observations"],
                           unique_programs=share["unique_programs"],
                           selected_for_stratum=True, limit=limit_per_head))
        if len(chosen) >= 8:
            break
    return chosen


def freeze(census, output, payloads_opened=False):
    if payloads_opened:
        raise ValueError("H05 fixture must be frozen before opening payloads")
    identities = select_identities(census)
    manifest = dict(
        schema=SCHEMA,
        payloads_opened=False,
        new_fits=0,
        teaching_admission=False,
        authority_namespace_required=True,
        axes=list(AXES),
        gates=GATES,
        strata=identities,
        qualification_ready=False,
        unavailable_requirements=[
            "metadata-selected sixteen training contexts with native membership evidence",
            "sixty-four distinct eligible programs and retained authoritative references per context",
            "separate canonical reservation and authority screening contract for each condition",
            "independent reference and prediction identity verification",
        ],
        strata_scope="head census only; not the required context/candidate pilot",
        census_unverified=bool(census.get("unverified")),
        census_observations=census.get("observations", 0),
        census_files=census.get("files", 0),
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if target.is_symlink():
        raise ValueError("immutable H05 fixture symlink rejected")
    if target.exists():
        if target.read_bytes() != payload.encode():
            raise ValueError("immutable H05 fixture changed; use a successor path")
    else:
        # Publish only a complete durable file, and never overwrite another
        # freezer's result. Old v1 fixtures remain original historical bytes.
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            staging = Path(stream.name)
            stream.write(payload.encode())
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(staging, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != payload.encode():
                    raise ValueError("immutable H05 fixture changed concurrently")
            fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            staging.unlink()
    manifest["manifest_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest
