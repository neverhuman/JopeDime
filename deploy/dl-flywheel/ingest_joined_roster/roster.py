"""Join retained native inputs with versioned S3 metadata; publish locally last.

This exports data candidates, not optimizer admission or newly qualified labels.
Existing HWM DSL is retained as historical CE evidence. Other measured programs
remain separate and never become self-imitation teachers or numeric rewards.
"""

import fcntl
import hashlib
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from jope_flywheel.protocol import partition

from .roster_support import (
    ARTIFACTS, SCHEMA, _backlog_evidence, _programs, canonical, checked_s3_ref,
    digest, document, read, sha, signed_row,
)


def assemble(plan):
    if plan.get("schema") != "jope.native-ingest-plan/v1" or plan.get("non_qualifying") is not True:
        raise ValueError("explicit nonqualifying ingest plan required")
    refs = plan["sources"]
    bodies = {name: read(ref) for name, ref in refs.items()}
    inv = document(bodies["inventory"])
    if inv.get("schema") != "jope.full-native-corpus-inventory-audit/v1":
        raise ValueError("native inventory schema")
    views = {}
    original = {}
    for row in inv["rows"]:
        view = row["view_id"]
        if not isinstance(view, str) or not view or view in views:
            raise ValueError("distinct native view identities required")
        if type(row["fold"]) is not int or not 0 <= row["fold"] <= 4:
            raise ValueError("native outer-fold type")
        if not isinstance(row["dataset_hash"], str) or not re.fullmatch("[0-9a-f]{16}", row["dataset_hash"]):
            raise ValueError("native dataset version")
        if row["cut_id"] is not None and (not isinstance(row["cut_id"], str) or not row["cut_id"]):
            raise ValueError("explicit native cut")
        views[view] = row
        cid = row.get("context_sha256")
        if cid is not None:
            sha(cid)
            if cid in original:
                raise ValueError("duplicate native context")
            original[cid] = row
    meta = document(bodies["meta"])
    cids = meta["cids"]
    n = meta["n"]
    if (
        meta.get("schema") != "dlhwm.data/v1"
        or type(n) is not int
        or not 0 < n <= 20000
        or len(cids) != n
        or len(set(cids)) != n
    ):
        raise ValueError("retained bundle cardinality")
    for field in ("view_id", "dataset", "fold", "program_sha256"):
        if len(meta[field]) != n:
            raise ValueError("bundle identity dimensions")
    members = document(bodies["bundle_manifest"])["files"]
    for name, field in (
        ("meta.json", "meta"),
        ("hwm_programs.json", "winners"),
        ("blocks_kumo_tabular_s.npy", "features"),
        ("gid.npy", "gid"),
    ):
        if members[name] != refs[field]["sha256"]:
            raise ValueError("bundle manifest member binding")
    features = np.load(io.BytesIO(bodies["features"]), allow_pickle=False)
    gid = np.load(io.BytesIO(bodies["gid"]), allow_pickle=False)
    if features.shape != (n, 2051) or features.dtype != np.dtype("<f4") or not np.isfinite(features).all():
        raise ValueError("finite 2051-d float32 Kumo-only features required")
    if gid.shape != (n,) or gid.dtype.kind not in "iu":
        raise ValueError("retained group vector")
    roles = {}
    for parent in (document(line) for line in bodies["parents"].splitlines() if line.strip()):
        lineage = sha(parent["parent_lineage_sha256"])
        role = partition(lineage, parent["split_namespace_sha256"])
        if (
            parent["partition"] != role
            or parent["bundle_manifest_sha256"] != refs["bundle_manifest"]["sha256"]
            or parent["gid_sha256"] != refs["gid"]["sha256"]
        ):
            raise ValueError("retained family/role binding")
        for cid in parent["retained_lineage_context_sha256"]:
            sha(cid)
            if cid in roles:
                raise ValueError("overlapping retained lineage membership")
            roles[cid] = (lineage, role, parent["retained_group"])
    if set(cids) != set(roles):
        raise ValueError("complete retained lineage map required")
    winner_rows = document(bodies["winners"])["programs"]
    winners = {row["cid"]: row for row in winner_rows}
    if len(winner_rows) != n or set(winners) != set(cids):
        raise ValueError("complete frozen HWM membership required")
    target = document(bodies["target"])
    target_views = set(target["views"])
    object_index = document(bodies["object_index"])
    programs, program_refusals = _programs(object_index, original)
    backlog_evidence = _backlog_evidence(object_index, views)
    normalized_objects = sorted(
        {canonical({k: v for k, v in item.items() if k != "copy"}) for item in object_index}
    )
    train, validation, excluded = [], [], []
    blobs = {}
    for i, cid in enumerate(cids):
        row = original.get(cid)
        if (
            row is None
            or row["view_id"] != meta["view_id"][i]
            or row["dataset_hash"] != meta["dataset"][i]
            or type(meta["fold"][i]) is not int
            or row["fold"] != meta["fold"][i]
        ):
            raise ValueError("exact native input crosswalk required")
        if row["current_role"] != "training_validation_eligible" or row["view_id"] not in target_views:
            excluded.append({"view_id": row["view_id"], "context_id": cid, "reason": "role_excluded"})
            continue
        if row["cut_id"] is None:
            excluded.append({"view_id": row["view_id"], "context_id": cid, "reason": "native_cut_unresolved"})
            continue
        lineage, role, group = roles[cid]
        if type(group) is not int or group != int(gid[i]):
            raise ValueError("retained group identity")
        teacher = winners[cid]
        program_sha = sha(teacher["program_sha256"])
        if program_sha != meta["program_sha256"][i] or digest(teacher["dsl"].encode()) != program_sha:
            raise ValueError("frozen CE teacher DSL identity")
        if (row.get("gp_index") or {}).get("outcome") != "selection_reached":
            raise ValueError("frozen teacher lacks published selection")
        raw_vector = features[i].tobytes()
        vector_sha = digest(raw_vector)
        path = "vectors/" + vector_sha + ".f32"
        blobs[path] = raw_vector
        result = {
            "context_id": cid,
            "view_id": row["view_id"],
            "dataset_hash": row["dataset_hash"],
            "cut_id": row["cut_id"],
            "outer_fold": row["fold"],
            "parent_lineage_sha256": lineage,
            "partition": role,
            "features": {
                "path": path,
                "sha256": vector_sha,
                "bytes": len(raw_vector),
                "dimensions": 2051,
                "dtype": "<f4",
            },
            "frozen_hwm_teacher": {
                "program_sha256": program_sha,
                "dsl": teacher["dsl"],
                "grade": "historical_CE_only",
                "q": None,
            },
            "measured_program_candidates": programs.get(cid, []),
            "actor_targets": None,
            "learner_admitted": False,
            "non_qualifying": True,
        }
        (train if role == "train" else validation).append(signed_row(result))
    train.sort(key=lambda r: r["context_id"])
    validation.sort(key=lambda r: r["context_id"])
    vectors = blobs
    blobs = {}
    for name, rows in (("train-vectors.f32", train), ("validation-vectors.f32", validation)):
        packed = b"".join(vectors[row["features"]["path"]] for row in rows)
        pack_sha = digest(packed)
        blobs[name] = packed
        for i, row in enumerate(rows):
            row["features"].update(
                path=name, offset=i * 2051 * 4, pack_sha256=pack_sha, pack_bytes=len(packed)
            )
            row.pop("row_sha256")
            row["row_sha256"] = digest(canonical(row))
    train_families = {r["parent_lineage_sha256"] for r in train}
    val_families = {r["parent_lineage_sha256"] for r in validation}
    if train_families & val_families:
        raise ValueError("train/validation family overlap")
    backlog = []
    for row in views.values():
        if (
            row["current_role"] != "training_validation_eligible"
            or (row.get("gp_index") or {}).get("outcome") == "selection_reached"
        ):
            continue
        refs_for_run = backlog_evidence.get(row["view_id"], [])
        current_hwm = any(r.get("kind") == "backlog_current_hwm" for r in refs_for_run)
        reasons = ["no_retained_input_lineage", "no_retained_2051_encoding"]
        if not current_hwm:
            reasons.append("no_published_winner_in_captured_sources")
        if row.get("context_sha256") is None:
            reasons.append("native_identity_unresolved")
        backlog.append(
            signed_row(
                {
                    "view_id": row["view_id"],
                    "context_id": row.get("context_sha256"),
                    "dataset_hash": row["dataset_hash"],
                    "cut_id": row["cut_id"],
                    "outer_fold": row["fold"],
                    "historical_outcome": (row.get("gp_index") or {}).get("outcome"),
                    "s3_evidence": refs_for_run,
                    "current_selection_evidence_present": current_hwm,
                    "reasons": reasons,
                    "q": None,
                    "qualified": False,
                    "learner_admitted": False,
                }
            )
        )
    backlog.sort(key=lambda r: r["view_id"])
    tables = {
        "train.jsonl": train,
        "validation.jsonl": validation,
        "backlog.jsonl": backlog,
        "program-refusals.jsonl": [
            signed_row(r) for r in sorted(program_refusals, key=lambda r: canonical(r))
        ],
        "role-exclusions.jsonl": [signed_row(r) for r in excluded],
    }
    for name, rows in tables.items():
        blobs[name] = b"".join(canonical(row) + b"\n" for row in rows)
    summary = {
        "train_contexts": len(train),
        "validation_contexts": len(validation),
        "train_families": len(train_families),
        "validation_families": len(val_families),
        "family_overlap": 0,
        "original_inventory_runs": len(views),
        "backlog_runs": len(backlog),
        "backlog_status": dict(Counter(str(r["historical_outcome"]) for r in backlog)),
        "backlog_with_current_selection_evidence": sum(
            r["current_selection_evidence_present"] for r in backlog
        ),
        "validation_with_multi_program_candidates": sum(
            len({p["program_sha256"] for p in r["measured_program_candidates"]}) >= 2 for r in validation
        ),
        "qualified_new_programs": None,
        "actor_target_contexts": None,
    }
    manifest = {
        "schema": SCHEMA,
        "non_qualifying": True,
        "learner_admitted": False,
        "role_scope": "frozen native development split; current runtime admission unmeasured",
        "source_grade": "copied native inputs and S3 wrapper metadata; no new qualified labels",
        "inputs": {
            name: {"sha256": ref["sha256"], "bytes": ref["bytes"]}
            for name, ref in refs.items()
            if name != "object_index"
        },
        "object_index_identity": {
            "semantic_sha256": digest(b"\n".join(normalized_objects)),
            "distinct_entries": len(normalized_objects),
            "basis": "immutable source identities; local copy paths and duplicate entries excluded; raw input hash retained in run receipt",
        },
        "summary": summary,
        "members": {
            name: {"sha256": digest(body), "bytes": len(body)} for name, body in sorted(blobs.items())
        },
    }
    return manifest, blobs


def publish(plan, output, *, before_manifest=None):
    manifest, blobs = assemble(plan)
    output = Path(output).absolute()
    if ".." in output.parts or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("output symlink")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".publish.lock"
    if lock_path.is_symlink():
        raise ValueError("publication lock symlink")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        body = canonical(manifest) + b"\n"
        path = output / "manifest.json"
        if path.is_symlink():
            raise ValueError("manifest symlink")
        if path.exists() and path.read_bytes() != body:
            raise ValueError("existing manifest drift")
        for name, blob_body in blobs.items():
            dest = output / name
            if any(p.is_symlink() for p in (dest, *dest.parents)):
                raise ValueError("output member symlink")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if dest.read_bytes() != blob_body:
                    raise ValueError("existing output drift")
            else:
                with dest.open("xb") as stream:
                    stream.write(blob_body)
                    stream.flush()
                    os.fsync(stream.fileno())
            with dest.open("rb") as stream:
                os.fsync(stream.fileno())
        for directory in (output,):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if before_manifest is not None:
            before_manifest()
        if path.exists():
            if path.read_bytes() != body:
                raise ValueError("existing manifest drift")
        else:
            pending = output / (".manifest-" + digest(body) + ".prepared")
            if pending.is_symlink():
                raise ValueError("prepared manifest symlink")
            if pending.exists() and pending.read_bytes() != body:
                raise ValueError("prepared manifest drift")
            if not pending.exists():
                with pending.open("xb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
            with pending.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(pending, path)
            try:
                fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except OSError:
                # Remove only this fresh commit link; preserve the prepared bytes.
                path.unlink()
                raise
    return {"manifest_sha256": digest(body), "manifest": manifest}


def read_committed(output, expected_manifest_sha256):
    """Require the pinned final manifest and verify every member before returning.

    A prepared manifest or a set of members alone is never a committed roster.
    This reads local data candidates only; it supplies no admission authority.
    """
    output = Path(output).absolute()
    path = output / "manifest.json"
    if ".." in output.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("committed source symlink")
    sha(expected_manifest_sha256)
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or digest(raw) != expected_manifest_sha256:
        raise ValueError("committed manifest hash/size drift")
    manifest = document(raw)
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("non_qualifying") is not True
        or manifest.get("learner_admitted") is not False
    ):
        raise ValueError("nonqualifying committed roster required")
    tables = (
        "train.jsonl",
        "validation.jsonl",
        "backlog.jsonl",
        "program-refusals.jsonl",
        "role-exclusions.jsonl",
    )
    if set(manifest["members"]) != {*tables, "train-vectors.f32", "validation-vectors.f32"}:
        raise ValueError("closed roster member set required")
    blobs = {
        name: read({"path": str(output / name), **ref}, 256 * 1024**2, allow_empty=True)
        for name, ref in manifest["members"].items()
    }
    rows = {name: [document(line) for line in blobs[name].splitlines() if line] for name in tables}
    for table in rows.values():
        for row in table:
            if sha(row.get("row_sha256")) != digest(
                canonical({k: v for k, v in row.items() if k != "row_sha256"})
            ):
                raise ValueError("committed row hash drift")
    seen = set()
    families = {}
    for role, name in (("train", "train.jsonl"), ("validation", "validation.jsonl")):
        groups = set()
        pack_name = role + "-vectors.f32"
        pack = blobs[pack_name]
        if len(pack) != len(rows[name]) * 2051 * 4:
            raise ValueError("packed feature cardinality")
        for i, row in enumerate(rows[name]):
            cid = sha(row["context_id"])
            if (
                cid in seen
                or row["partition"] != role
                or row["learner_admitted"] is not False
                or row["non_qualifying"] is not True
            ):
                raise ValueError("committed context/role/admission identity")
            seen.add(cid)
            groups.add(sha(row["parent_lineage_sha256"]))
            ref = row["features"]
            if (
                ref["path"] != pack_name
                or type(ref["offset"]) is not int
                or ref["offset"] != i * 2051 * 4
                or ref["bytes"] != 2051 * 4
                or ref["dimensions"] != 2051
                or ref["dtype"] != "<f4"
            ):
                raise ValueError("packed feature row binding")
            vector = pack[ref["offset"] : ref["offset"] + ref["bytes"]]
            if (
                digest(vector) != sha(ref["sha256"])
                or ref["pack_sha256"] != manifest["members"][pack_name]["sha256"]
                or ref["pack_bytes"] != len(pack)
                or not np.isfinite(np.frombuffer(vector, dtype="<f4")).all()
            ):
                raise ValueError("packed feature identity")
        families[role] = groups
    if families["train"] & families["validation"]:
        raise ValueError("committed train/validation family overlap")
    summary = manifest["summary"]
    counts = {
        "train_contexts": len(rows["train.jsonl"]),
        "validation_contexts": len(rows["validation.jsonl"]),
        "train_families": len(families["train"]),
        "validation_families": len(families["validation"]),
        "family_overlap": 0,
        "backlog_runs": len(rows["backlog.jsonl"]),
    }
    if any(type(summary[k]) is not int or summary[k] != v for k, v in counts.items()):
        raise ValueError("committed summary count drift")
    return {"manifest_sha256": expected_manifest_sha256, "manifest": manifest, "rows": rows}
