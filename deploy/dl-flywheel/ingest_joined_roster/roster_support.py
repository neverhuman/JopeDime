"""Canonical validation and evidence projection helpers for joined native rosters."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from jope_flywheel.protocol import partition

SCHEMA = "jope.joined-native-roster/v1"
ARTIFACTS = {
    "input",
    "evaluation",
    "program",
    "rust_source",
    "rust_executable",
    "rust_parity",
    "score_records",
}


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(body):
    return hashlib.sha256(body).hexdigest()


def sha(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("full lowercase SHA256 required")
    return value


def document(body):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out

    def nonfinite(value):
        raise ValueError("nonfinite JSON: " + value)

    return json.loads(body, object_pairs_hook=pairs, parse_constant=nonfinite)


def read(ref, limit=64 * 1024**2, *, allow_empty=False):
    minimum = 0 if allow_empty else 1
    if (
        set(ref) != {"path", "sha256", "bytes"}
        or type(ref["bytes"]) is not int
        or not minimum <= ref["bytes"] <= limit
    ):
        raise ValueError("bounded exact local reference required")
    path = Path(ref["path"])
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("non-symlink absolute source required")
    sha(ref["sha256"])
    with path.open("rb") as stream:
        body = stream.read(ref["bytes"] + 1)
    if len(body) != ref["bytes"] or digest(body) != ref["sha256"]:
        raise ValueError("source bytes/hash drift")
    return body


def checked_s3_ref(ref):
    if set(ref) != {"bucket", "key", "version_id", "sha256", "bytes"}:
        raise ValueError("closed S3 reference required")
    sha(ref["sha256"])
    if type(ref["bytes"]) is not int or ref["bytes"] <= 0:
        raise ValueError("S3 byte identity")
    if not isinstance(ref["version_id"], str) or ref["version_id"] in ("", "null"):
        raise ValueError("immutable S3 version required")
    if (
        not isinstance(ref["bucket"], str)
        or not ref["bucket"]
        or not isinstance(ref["key"], str)
        or not ref["key"]
    ):
        raise ValueError("S3 namespace identity")
    return ref


def signed_row(row):
    return {**row, "row_sha256": digest(canonical(row))}


def _programs(index, inventory):
    groups = {}
    refused = []
    for item in index:
        if item["kind"] != "program_result":
            continue
        if item["state"] != "copied":
            refused.append({"source_key": item["key"], "reason": "source_capture_unmeasured"})
            continue
        ref = checked_s3_ref(item["ref"])
        row = document(
            read({"path": item["copy"], "sha256": ref["sha256"], "bytes": ref["bytes"]}, 4 * 1024**2)
        )
        if row.get("schema") != "jope.native-original-program-result/v1":
            refused.append({"source_ref": ref, "reason": "different_context_kind"})
            continue
        ctx = row.get("context", {})
        cid = sha(ctx.get("identity"))
        original = inventory.get(cid)
        if original is None:
            refused.append({"source_ref": ref, "context_id": cid, "reason": "native_identity_unresolved"})
            continue
        if ctx.get("dataset_hash") != original["dataset_hash"] or ctx.get("cut_id") != original["cut_id"]:
            raise ValueError("native dataset/cut cross-context join")
        if row.get("status") != "completed" or row.get("measurement") != "executed":
            refused.append({"source_ref": ref, "context_id": cid, "reason": "not_completed_measurement"})
            continue
        program = sha(row["program_sha256"])
        evaluation = sha(row["evaluation_key"])
        if ref["key"].split("/")[-2:] != [evaluation, "record.json"]:
            raise ValueError("evaluation lookup identity mismatch")
        artifacts = row["artifacts"]
        if set(artifacts) != ARTIFACTS:
            raise ValueError("seven named artifact references required")
        root = ref["key"].rsplit("/", 2)[0] + "/"
        for role, artifact in artifacts.items():
            checked_s3_ref(artifact)
            if (
                artifact["bucket"] != ref["bucket"]
                or artifact["key"] != root + "objects/" + artifact["sha256"] + "/" + role
            ):
                raise ValueError("artifact namespace/hash/role mismatch")
        if artifacts["program"]["sha256"] != program or artifacts["input"]["sha256"] != ctx.get(
            "input_sha256"
        ):
            raise ValueError("native program/input artifact identity mismatch")
        for field in ("input_sha256", "target_sha256", "split_sha256", "evaluator_sha256"):
            sha(ctx[field])
        payload = {
            "evaluation_key": evaluation,
            "program_sha256": program,
            "context": ctx,
            "artifacts": artifacts,
            "evidence_grade": "wrapper_verified_inner_artifact_closure_unmeasured",
            "q": None,
            "q_opt": None,
            "q_grid": None,
            "qualified": False,
            "actor_target": None,
            "self_imitation_teacher": False,
        }
        key = (cid, evaluation)
        if key in groups:
            previous = {k: v for k, v in groups[key].items() if k != "source_refs"}
            if previous != payload:
                raise ValueError("conflicting duplicate evaluation")
            refs = groups[key]["source_refs"] + [ref]
        else:
            refs = [ref]
        payload["source_refs"] = [document(value) for value in sorted({canonical(r) for r in refs})]
        groups[key] = payload
    by_context = {}
    for (cid, _), row in sorted(groups.items()):
        by_context.setdefault(cid, []).append(row)
    return by_context, [document(row) for row in sorted({canonical(r) for r in refused})]


def _backlog_evidence(index, views):
    evidence = {}
    for item in index:
        if item["kind"] not in {"backlog_terminal", "backlog_current_hwm"}:
            continue
        view = item.get("view_id")
        if view not in views:
            raise ValueError("backlog evidence view identity")
        if item["state"] != "copied":
            evidence.setdefault(view, []).append({"state": "unmeasured", "source_key": item["key"]})
            continue
        ref = checked_s3_ref(item["ref"])
        row = document(
            read({"path": item["copy"], "sha256": ref["sha256"], "bytes": ref["bytes"]}, 4 * 1024**2)
        )
        original = views[view]
        if (
            row.get("schema") != "jope.gp-hwm-evidence/v1"
            or row.get("dataset_hash") != original["dataset_hash"]
            or row.get("cut_id") != original["cut_id"]
            or type(row.get("fold")) is not int
            or row["fold"] != original["fold"]
        ):
            raise ValueError("backlog dataset/cut/fold identity")
        cid = sha(row["context_sha256"])
        expected_ids = (
            {original["context_sha256"]}
            if original.get("context_sha256")
            else {r["context_sha256"] for r in original.get("s1_finals", [])}
        )
        if cid not in expected_ids:
            raise ValueError("backlog native context identity")
        kind = item["kind"]
        if row.get("outcome") != ("g0_not_ready" if kind == "backlog_terminal" else "selection_reached"):
            raise ValueError("backlog outcome identity")
        projected = {
            "kind": kind,
            "source_ref": ref,
            "observed_native_context_id": cid,
            "declared_outcome": row["outcome"],
            "authority": row.get("authority"),
            "g0_qualification": row.get("g0_qualification"),
            "source_family": row.get("family"),
            "source_rights": row.get("rights"),
            "qualified": False,
            "learner_admitted": False,
        }
        if kind == "backlog_current_hwm":
            program = row["selection"]["program"]
            if digest(program["dsl"].encode()) != sha(program["program_sha256"]):
                raise ValueError("backlog selection DSL identity")
            projected["declared_selected_program_sha256"] = program["program_sha256"]
        evidence.setdefault(view, []).append(projected)
    return {
        view: [document(row) for row in sorted({canonical(r) for r in rows})]
        for view, rows in evidence.items()
    }
