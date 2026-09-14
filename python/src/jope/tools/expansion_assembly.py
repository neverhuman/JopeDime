"""New expansion assembly identity. Never writes the live context index."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "parallel.expansion_assembly.v1"
ASSIGNMENT_SEED = 20260907
LIVE_CONTEXTS = "runs/dataset-inclusion-20260909/correction-final/current-contexts.json"
LIVE_CONTEXTS_SHA_PREFIX = "d77e856b"
PARENT_ASSEMBLY = "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb"
PARENT_REGISTRY = "22b5de5f2f847fb6e086d125d6fec945a1b4edb7c00b94b49e1ad95e7540cfc4"
V4_REGISTRY = "manifests/generated/families/qualification-v4.json"
V5_REGISTRY = "manifests/generated/families/qualification-v5-expansion-20260909.json"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def refuse_live_index(output):
    path = Path(output)
    resolved = path.resolve()
    live = Path(__file__).resolve().parents[1] / LIVE_CONTEXTS
    if (LIVE_CONTEXTS in path.as_posix() or LIVE_CONTEXTS in resolved.as_posix()
            or (path.exists() and live.exists() and path.samefile(live))):
        raise ValueError("refusing to write the live current-contexts.json")
    return True


def exclude_live_families(families, live_families):
    """Live 80-family partitions do not change."""
    live = set(live_families)
    kept, skipped = [], []
    for name in families:
        if name in live:
            skipped.append(name)
        else:
            kept.append(name)
    if not kept:
        raise ValueError("all families collide with the live 80")
    return kept, sorted(set(skipped))


def canary_apply_contract():
    """Root integrates admission publication after native verification."""
    return dict(
        schema="parallel.v5_canary_apply_contract.v1",
        owner="Codex root integrator",
        apply_contexts_ran=False,
        writes_live_index=False,
        apply_contexts_outer_flock=False,
        dump_cuts_onto_live_gp=False,
        projection_refitted=False,
        cohort_statistics_refitted=False,
        awaiting_ack=True,
    )


def assign_partitions(*args, **kwargs):
    raise ValueError("legacy sorted-tail assignment retired; use authenticated seed-20260907 component growth")


def canary_v5(output, v5_families):
    refuse_live_index(output)
    raise ValueError("legacy Boolean assembly declaration retired; native publication evidence required")


def v6_plan(expansion_families, output, live_families=()):
    refuse_live_index(output)
    raise ValueError("legacy sorted-tail assignment retired; use authenticated seed-20260907 component growth")


def successor_from_growth(assignment, output, parent_assembly=PARENT_ASSEMBLY,
                          parent_registry=PARENT_REGISTRY):
    """Retain a verified growth proposal; only native publication creates an assembly."""
    from family_growth_policy import verify_assignment

    refuse_live_index(output)
    if not isinstance(assignment, (str, Path)):
        raise ValueError("verified growth assignment file required; Boolean declarations cannot authorize assembly")
    path = Path(assignment).resolve(strict=True)
    verified = verify_assignment(path)
    if not verified["rows"]:
        raise ValueError("grown eligible set is empty")
    if verified["parent_registry"]["sha256"] != parent_registry:
        raise ValueError("growth parent registry differs from preserved lineage")
    receipt = dict(
        schema="parallel.expansion_proposal.v2",
        kind="verified_growth_proposal",
        parent_assembly_sha256=parent_assembly,
        parent_registry_sha256=parent_registry,
        assignment=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        rows=verified["rows"],
        required_next_operation="native membership and immutable context publication verification",
        execution_eligible=False,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(receipt, indent=2) + "\n").encode()
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError("immutable expansion proposal changed")
    else:
        with target.open("xb") as stream:
            stream.write(data)
            stream.flush()
            import os
            os.fsync(stream.fileno())
    return receipt
