#!/usr/bin/env python3
"""Freeze and publish sanitized, cut-specific S1 HWM evidence through the #224 publisher.

This stores measured selection progress, not a teacher grant, HWM improvement decision,
outer score, or fitted checkpoint. It never opens JTT1, tar members, or outer-test files.
"""

import argparse
import collections
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE))
from verify_lane import HANDOFF_SHA, verify

# Authenticate the frozen publication dependencies before importing them. Keep that
# custody release unchanged: this adapter and its tests live outside lane/src.
verify(HERE / "lane")
sys.path.insert(0, str(HERE / "lane/src/dl_training_v1"))
from dltrain import common, s3io
from tools import v2_publish as transport

HEADER_SHA = "e83bb98afe7778926cb115031bb28e7c8e47e183a41733529be1acbb5ec01971"
HEADER_SCHEMA = "jope.dl-s1-prod-ledger-header/v1"
# S1 campaign-2 (owner ENG:watcher 13:58 MT): the G3-appended contexts, one research-only ledger, P3 family and role of the
# appended identities from the G3 decision file bound by sha256 in the header (owner GP-2, ENG:watcher 05:16Z).
C2_HEADER_SCHEMA = "jope.dl-s1-prod-c2-ledger-header/v1"
LANES = ("production", "research")
ROLELESS = {"ROLELESS_BRIDGE": "roleless_bridge"}
G3_DECISIONS_SCHEMA = "jope.dl-family-g3-decisions/v1"
# Owner GP-2 (ENG:watcher 05:16Z) names exactly this G3 decision file as the P3 role source for G3 contexts.
G3_DECISIONS_SHA256 = "796e27247b92dcac54c5fec586ee7f8b69aff4d02bd3783d1157c9cd34c01afb"
# GP HWMs for every admitted catalog context (VP OWNER 7:20am MT 2026-10-07, Jepson ruling): a catalog input list may bind
# sha-pinned catalog-v246 P3 decision files ("catalog_p3": {"decision[_tag]", "sha256[_tag]"}, e.g. v1 and _v2). Each is a role
# authority only when listed here together with its independent approval record, re-read by exact sha256 at freeze (the same
# approval test as the S1 gate v3): Lane A records name decision_sha256 and carry a review_ref; Lane C records name
# candidate_sha256 with verdict APPROVE. The pinned registry/holdout keep precedence; an unknown role is refused.
CATALOG_DECISIONS_SCHEMAS = (
    "jope.dl-family-catalog-v246-decisions/v1",
    "jope.dl-family-catalog-v246-decisions/v2",
    "jope.dl-family-catalog-v246-decisions/v3-outliers",
)
APPROVED_P3_DECISIONS = {
    "cb3e6079364d984c26a2719bc22985c89a69b280093c8f5dce75456bb4608d36": {
        "approval": "/home/ubuntu/scratch/jopedime-dl-training-v1/receipts/owner-relays/"
        "p3-decisions-catalog-v246-v1-20261007T080143Z.LANE-A-APPROVED.json",
        "approval_sha256": "86a73eeb977fd19972ba47cfe11524c285c8c49da92e6ecb8f17bf12da4d777d",
        "reviewer": "lane a",
    },
    "4846b11657ceecf4008de0e6130edeee3296c0889455fc2ccb1181eebc31087f": {
        "approval": "/home/ubuntu/jope-evidence/kernel2/bt/P3_V2_REVIEW/laneC-20261007T1017/approval-record.json",
        "approval_sha256": "7f38d5283805eb8f02cb7aa2a50887f2d8850696152bdd82509a811ed58575a1",
        "reviewer": "lane c",
    },
    # P3 v3 (the 40 Wave-2 Group B outliers, batch w2-B-b06): Lane C APPROVE 2026-10-07T14:56:19Z under the owner's 14:36Z
    # partitioned-screen method approval (owner record 750d3041).
    "c846aa5dbb9c03ce815cebffe4284b42be2731d94fbe41e6bf64cd9e371545f0": {
        "approval": "/home/ubuntu/jope-evidence/kernel2/bt/P3_V3_REVIEW/laneC-20261007/approval-record.json",
        "approval_sha256": "2288883b323f86370a28089a90ae035f2f18db14ab12a4437e9b45e9bad2c5ed",
        "reviewer": "lane c",
    },
}
# ONE-HOLDOUT (Jepson, 2026-10-07 5:10pm MT, via OWNER ENG:watcher; rulings 23:18Z): the sealed meta-test group is the only holdout;
# P3 heldout contexts are training data. A heldout final publishes only when its input list binds ("one_holdout":
# {"role_map_sha256"}) one of these sha-pinned role maps and that map names the context, with the same P3 family and role, as newly
# training. Every other role rule is unchanged: sealed_excluded and unknown roles stay refused.
ONE_HOLDOUT_SCHEMA = "jope.dltrain-one-holdout-role-map/v1"
ONE_HOLDOUT_ROLE_MAPS = {
    "3b9caf36a3cde3f12d0337ebfc633b55a1e989a3f53410df578f6fc9a7aa0109": "/home/ubuntu/scratch/jopedime-dl-training-v1/"
    "receipts/owner-relays/one-holdout-role-map-v1-20261007T231936Z.json",
}
PLAN_SCHEMA = "jope.gp-hwm-publish-plan/v1"
OBJECT_SCHEMA = "jope.gp-hwm-evidence/v1"
MANIFEST_SCHEMA = "jope.gp-hwm-publish-manifest/v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
CUT = re.compile(r"trainonly-split-v2g?:[0-9a-f]{64}\Z")
DSL = re.compile(r"[A-Za-z0-9_().=+\- ]{1,65536}\Z")
BINARIES = {
    "engine": "jope-island-engine.s1main",
    "worker": "bootstrap.s1main",
    "proposal": "jope-s1-proposal.s1main",
    "materializer": "materialize_g0.py",
}


class Refused(ValueError):
    """Refusal before authority-bearing publication; never substitute missing inputs."""


def require(ok, reason):
    if not ok:
        raise Refused(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_field(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid sha256")
    return value


def unique(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key")
        obj[key] = value
    return obj


def decode(data):
    def bad_constant(_):
        raise Refused("nonfinite JSON")

    return json.loads(data, object_pairs_hook=unique, parse_constant=bad_constant)


def read_regular(path, limit=100 << 20, prefix=None):
    """Open every component relative to an owned fd, O_NOFOLLOW throughout.

    Reading and hashing the same descriptor avoids check/load replacement. A frozen
    ledger uses an exact newline-terminated prefix; later appends cannot renew it.
    """
    path = Path(path).absolute()
    require(
        not any(p in ("..", "outer-test", "outer-test-scratch") for p in path.parts),
        "forbidden source path",
    )
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        file_fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        with os.fdopen(file_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode), "source is not a regular file")
            length = info.st_size if prefix is None else prefix
            require(
                type(length) is int and 0 <= length <= limit and info.st_size >= length,
                "source size/prefix refused",
            )
            data = stream.read(length)
            require(len(data) == length, "short source read")
            return data
    finally:
        os.close(fd)


class Sources:
    def __init__(self):
        self.refs = {}

    def read(self, path, expected=None, prefix=None):
        path = str(Path(path).absolute())
        data = read_regular(path, prefix=prefix)
        got = digest(data)
        if expected is not None:
            require(
                got == sha_field(expected), "source hash mismatch: " + Path(path).name
            )
        ref = {
            "path": path,
            "sha256": got,
            "bytes": len(data),
            "prefix": prefix is not None,
        }
        require(
            path not in self.refs or self.refs[path] == ref,
            "inconsistent source reference",
        )
        self.refs[path] = ref
        return data

    def doc(self, path, expected):
        return decode(self.read(path, expected))


def catalog_decision(src, path, sha):
    """One approved catalog-v246 P3 decision file: {dataset: (family, role)}.

    Refused unless the file is pinned in APPROVED_P3_DECISIONS and its approval record is present at the pinned sha256 and
    approves exactly this decision (fail closed: a missing, swapped or non-approving record publishes nothing)."""
    require(isinstance(path, str) and isinstance(sha, str), "invalid catalog P3 pin")
    sha = sha_field(sha)
    pin = APPROVED_P3_DECISIONS.get(sha)
    require(pin is not None, "catalog P3 decision file is not an approved role source")
    try:
        approval = src.doc(pin["approval"], pin["approval_sha256"])
    except OSError:
        raise Refused("catalog P3 approval record missing") from None
    reviewer = str(approval.get("reviewer", "")).lower()
    if pin["reviewer"] == "lane a":
        approves = approval.get("decision_sha256") == sha and bool(approval.get("review_ref"))
    else:
        approves = approval.get("candidate_sha256") == sha and approval.get("verdict") == "APPROVE"
    require(approves and pin["reviewer"] in reviewer, "approval record does not approve this catalog P3 decision")
    decisions = src.doc(path, sha)
    require(decisions.get("schema") in CATALOG_DECISIONS_SCHEMAS, "unsupported catalog decisions schema")
    rows = decisions.get("per_dataset")
    require(isinstance(rows, list) and rows, "catalog decisions carry no datasets")
    roles = {}
    for item in rows:
        ds, fid, role = item.get("dataset_hash"), item.get("family_id"), item.get("role")
        require(
            isinstance(ds, str) and re.fullmatch(r"[0-9a-f]{16}", ds)
            and isinstance(fid, str) and re.fullmatch(r"fam-[0-9a-f]{64}", fid)
            and isinstance(role, str) and re.fullmatch(r"[A-Za-z_]{1,32}", role),
            "invalid catalog decision row",
        )
        require(ds not in roles or roles[ds] == (fid, role), "ambiguous catalog decision")
        roles[ds] = (fid, role)
    return roles


def catalog_roles(src, ref):
    """An input list's catalog P3 binding: every "decision[_tag]"/"sha256[_tag]" pair it names, each an approved decision file.

    Returns {dataset: (family, role, [sources])}; a dataset decided by more than one bound file must agree in family and role."""
    require(isinstance(ref, dict), "invalid catalog P3 pin")
    tags = [k[len("decision"):] for k in ref if re.fullmatch(r"decision(_[a-z0-9]{1,8})?", k)]
    require(tags and all(isinstance(ref.get("sha256" + t), str) for t in tags), "invalid catalog P3 pin")
    merged = {}
    for tag in sorted(tags, key=lambda t: (t != "", t)):
        sha = sha_field(ref["sha256" + tag])
        source = {"path": str(Path(ref["decision" + tag]).absolute()), "sha256": sha}
        for ds, (fid, role) in catalog_decision(src, ref["decision" + tag], sha).items():
            if ds in merged:
                require(merged[ds][:2] == (fid, role), "catalog decision files disagree")
                merged[ds][2].append(source)
            else:
                merged[ds] = (fid, role, [source])
    return merged


def one_holdout(src, ref):
    """An input list's ONE-HOLDOUT binding: ({context: P3} the pinned role map names as newly training, the map's source ref)."""
    require(isinstance(ref, dict) and isinstance(ref.get("role_map_sha256"), str), "invalid ONE-HOLDOUT pin")
    sha = sha_field(ref["role_map_sha256"])
    path = ONE_HOLDOUT_ROLE_MAPS.get(sha)
    require(path is not None, "ONE-HOLDOUT role map is not an approved role source")
    doc = src.doc(path, sha)
    require(doc.get("schema") == ONE_HOLDOUT_SCHEMA, "unsupported ONE-HOLDOUT role map schema")
    groups = doc.get("newly_training")
    require(isinstance(groups, dict) and groups, "ONE-HOLDOUT role map names no training contexts")
    training = {}
    for rows in groups.values():
        require(isinstance(rows, list), "invalid ONE-HOLDOUT group")
        for item in rows:
            cid = sha_field(item.get("context_sha256"))
            require(cid not in training, "ambiguous ONE-HOLDOUT context")
            training[cid] = item.get("p3")
    return training, {"path": str(Path(path).absolute()), "sha256": sha}


def finite(value):
    require(
        type(value) in (int, float) and math.isfinite(value), "invalid measured loss"
    )
    return value


def clean_output(obj):
    """Defense in depth for the explicit metadata-only output schema."""
    common.assert_no_secrets_obj(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            require(
                not re.search(
                    r"(^|_)(test|target|label|prediction|rows?)(_|$)",
                    key,
                    re.IGNORECASE,
                ),
                "forbidden publication field",
            )
            clean_output(value)
    elif isinstance(obj, list):
        for value in obj:
            clean_output(value)
    common.canonical(obj)  # rejects nonfinite values at every serialized boundary


from hwm_publish_inventory import inventory, selected_payload



def freeze(source, header_sha, output):
    plan, _ = inventory(source, header_sha)
    return common.write_receipt(output, plan)


def prepare(plan_path, plan_sha):
    data = read_regular(plan_path)
    require(digest(data) == sha_field(plan_sha), "wrong publication plan")
    plan = decode(data)
    require(plan.get("schema") == PLAN_SCHEMA, "unsupported publication plan")
    actual, objects = inventory(plan["source"], plan["header_sha256"], plan["ledgers"])
    # All other dependency pins and final hashes must remain identical, even if the
    # live producer has appended new records beyond our immutable ledger prefix.
    require(actual == plan, "changed publication closure")
    return plan, objects


def publish(plan_path, plan_sha, work_dir, receipt, dry_run, budget_snapshot=None):
    verify(HERE / "lane")
    plan, objects = prepare(
        plan_path, plan_sha
    )  # complete preflight before any client or ledger construction
    work_dir = Path(work_dir).absolute()
    require(
        not work_dir.is_relative_to(Path(plan["source"])),
        "publication state must be outside input custody",
    )
    for path in (work_dir, Path(receipt).absolute()):
        require(".." not in path.parts, "parent traversal in output path")
        require(
            not path.is_relative_to(Path(plan["source"])),
            "receipt must be outside input custody",
        )
        require(
            not any(part.is_symlink() for part in (path, *path.parents)),
            "symlink in output path",
        )
    for name in ("dry-run-publish.jsonl", "real-publish.jsonl", "dry-run-budget.jsonl"):
        require(
            not (work_dir / name).is_symlink()
            and not (work_dir / (name + ".lock")).is_symlink(),
            "symlink ledger",
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    mode = "dry-run" if dry_run else "real"
    ledger_path = work_dir / (mode + "-publish.jsonl")
    if dry_run:
        budget = work_dir / "dry-run-budget.jsonl"
        if not budget.exists():
            # A COPY, never the active publisher's budget ledger. The exact snapshot
            # is local evidence only, guarded dry-run puts cannot reserve real bytes.
            data = read_regular(budget_snapshot) if budget_snapshot else b""
            common.immutable_write(budget, data, mode=0o600)
        writer, reader, _ = transport.fake_clients(budget)
    else:
        require(budget_snapshot is None, "real writes must use the shared v2 budget")
        writer, reader = transport.real_clients()
    ledger = transport.PublishLedger(ledger_path)
    publisher = transport.Publisher(writer, reader, ledger)
    refs = {"production": [], "research": []}
    completed = 0
    status = "failed"
    manifests = []
    error = None
    try:
        for obj in objects:
            clean_output(obj["body"])
            ref = publisher.publish_bytes(
                obj["key"],
                common.canonical(obj["body"]),
                "application/json",
                "gp_evidence",
                obj["body"]["context_sha256"],
            )
            refs[obj["body"]["rights"]].append(
                {k: ref[k] for k in ("key", "sha256", "bytes", "version_id")}
            )
            completed += 1
        # Dependencies first; a rights-specific complete manifest is the commit.
        for lane in ("production", "research"):
            if not refs[lane]:
                continue
            doc = {
                "schema": MANIFEST_SCHEMA,
                "adapter_sha256": plan["adapter_sha256"],
                "custody_sha256": plan["custody_sha256"],
                "plan_sha256": plan_sha,
                "header_sha256": plan["header_sha256"],
                "ledger_prefixes": plan["ledgers"],
                "rights": lane,
                "authority": "progress_evidence_only",
                "objects": refs[lane],
            }
            clean_output(doc)
            body = common.canonical(doc)
            prefix = (
                s3io.V2_RESEARCH_ONLY_KEY if lane == "research" else s3io.V2_PREFIX_KEY
            )
            key = f"{prefix}manifests/gp-hwm/{digest(body)}.json"
            manifests.append(
                publisher.publish_bytes(
                    key, body, "application/json", "manifest", plan_sha
                )
            )
        status = "verified"
        return {
            "manifests": manifests,
            "completed": completed,
            "counts": plan["counts"],
        }
    except BaseException as exc:
        error = (
            type(exc).__name__
        )  # no arbitrary source strings or credential data in the failure receipt
        raise
    finally:
        ledger.close()
        common.write_receipt(
            receipt,
            {
                "schema": "jope.gp-hwm-publish-run/v1",
                "mode": mode,
                "status": status,
                "plan_sha256": plan_sha,
                "adapter_sha256": digest(read_regular(__file__)),
                "custody_sha256": HANDOFF_SHA,
                "counts": plan["counts"],
                "scheduled": len(objects),
                "completed": completed,
                "remaining": len(objects) - completed,
                "error": error,
                "publish_ledger": str(ledger_path),
                "manifests": manifests,
                "publication_stats": dict(publisher.stats),
                "utc": common.utc_iso(),
            },
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--source", type=Path, required=True)
    f.add_argument("--header-sha256", default=HEADER_SHA)
    f.add_argument("--output", type=Path, required=True)
    u = sub.add_parser("publish")
    u.add_argument("--plan", type=Path, required=True)
    u.add_argument("--plan-sha256", required=True)
    u.add_argument("--work-dir", type=Path, required=True)
    u.add_argument("--receipt", type=Path, required=True)
    u.add_argument("--budget-snapshot", type=Path)
    mode = u.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    a = p.parse_args()
    if a.command == "freeze":
        result = freeze(a.source, a.header_sha256, a.output)
    else:
        result = publish(
            a.plan, a.plan_sha256, a.work_dir, a.receipt, a.dry_run, a.budget_snapshot
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
