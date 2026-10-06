"""Synthetic, deterministic P3 fixtures. Nothing here reads real registry data or touches the network.

- Fixture registries in the jope.dl-family-registry/v1 shape of the P2 spec (per_family_fields), with review.csv, the
  review decision file (jope.dl-family-review-decisions/v1), alias sources, the P3 gate inputs (a P2 packet receipt and
  a jope.dl-owner-decisions/v1 file), and closure inputs (epoch rows, master views and a synthetic intake-registry
  SQLite file). A final registry carries the P2 CL-2 marker decision_deltas.provenance, and the decision file's
  owner_verbatim_message provenance resolves to an owner message stored where holdout.review_gate looks for it
  (holdout.P2_RECEIPTS_DIR, which P3Case points under the test's TMPDIR; review fix P3I-1).
- FakeS3: an in-memory versioned bucket with S3 create-only semantics, delete markers, injectable PUT failures (before
  or after the object is stored) and per-key hooks. Every PutObject, GetObject and ListObjectVersions first passes the
  real lane guard (s3io.Guard.check_params), so the fake accepts only those calls the guarded client would send.
  s3io.Guard allows the exact-key, bounded ListObjectVersions that LP-1 needs; proposed_history_guard re-checks the
  same contract, and FakeBucket(history_supported=False) simulates a guard that refuses it (fail closed).
  registry_bucket() holds what P2 publish wrote: the registry at the P2 receipt's exact VersionId (REGISTRY_VID), which
  commit and publish GET before any write (review fix P3I-2).
- A process audit hook that records any write outside TMPDIR while a test is armed. P3Case arms it around every test
  and fails the test on a violation; P3Case also makes the module's default (real) S3 client factory raise.
"""
from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import boto3
from botocore.exceptions import ClientError
from botocore.response import StreamingBody

from dltrain import common, families, holdout, s3io

FIXTURE = "jope.dl-p3-fixture/v1"
SEALED_CANARY = "SEALED-CANARY-lineage-b-meta-test-name"
# The real owner-record locations, read before any test moves them under TMPDIR (P3Case).
REAL_LOCATIONS = {"P2_RECEIPTS_DIR": holdout.P2_RECEIPTS_DIR, "HOLDOUT_DIR": holdout.HOLDOUT_DIR,
                  "OWNER_DIRECTIVES_PATH": holdout.OWNER_DIRECTIVES_PATH}
REGISTRY_VID = "vRegistry-fixture"                 # the VersionId P2 publish recorded for the fixture registry
OWNER_MESSAGE_UTC = "2026-10-05T08:00:00Z"
TMP_ROOT = os.environ.get("TMPDIR") or ""
if not TMP_ROOT or not os.path.isdir(TMP_ROOT):
    raise RuntimeError("TMPDIR must name an existing directory (RUN/tmp); P3 tests never fall back to /tmp")
ALLOWED_ROOT = os.path.realpath(TMP_ROOT)
if ALLOWED_ROOT == "/tmp" or ALLOWED_ROOT.startswith("/tmp/"):
    raise RuntimeError("TMPDIR must not be under /tmp (it is emptied at boot)")


# ------------------------------------------------------------------------------------------------ write guard
_GUARD = {"armed": False, "busy": False, "violations": []}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
_DIR_FD_ARG = {"os.mkdir": 2, "os.chmod": 2, "os.remove": 1, "os.rmdir": 1, "os.utime": 3, "os.chown": 3}


def under_root(path: str) -> bool:
    return path == ALLOWED_ROOT or path.startswith(ALLOWED_ROOT + os.sep)


def _resolve(path, dir_fd=-1):
    if isinstance(path, int):
        return None                       # fd-based: its path was audited when the descriptor was opened
    path = os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path)
    if not os.path.isabs(path):
        base = os.readlink(f"/proc/self/fd/{dir_fd}") if isinstance(dir_fd, int) and dir_fd >= 0 else os.getcwd()
        path = os.path.join(base, path)
    return os.path.realpath(path)


def write_target(event: str, args):
    """The resolved path an audited event writes, or None for reads and for events that write nothing."""
    try:
        if event == "open":
            path, mode, flags = args
            writes = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
                     (isinstance(flags, int) and flags >= 0 and bool(flags & _WRITE_FLAGS))
            return _resolve(path) if writes else None
        if event in _DIR_FD_ARG:
            i = _DIR_FD_ARG[event]
            return _resolve(args[0], args[i] if len(args) > i else -1)
        if event == "os.rename":
            src, dst = _resolve(args[0], args[2]), _resolve(args[1], args[3])
            return next((p for p in (src, dst) if p is not None and not under_root(p)), dst)
        if event == "os.link":
            return _resolve(args[1], args[3])
        if event == "os.symlink":
            return _resolve(args[1], args[2])
        if event == "os.truncate":
            return _resolve(args[0])
        if event == "shutil.copyfile":
            return _resolve(args[1])
        if event == "sqlite3.connect":
            db = os.fsdecode(args[0]) if isinstance(args[0], bytes) else str(args[0])
            if db.startswith("file:"):
                body, _, query = db[len("file:"):].partition("?")
                return None if "mode=ro" in query.split("&") else _resolve(body)
            return None if db == ":memory:" else _resolve(db)
    except Exception:  # noqa: BLE001 (an unresolvable target is itself a violation)
        return f"<unresolved {event}>"
    return None


def _audit_hook(event, args):
    if not _GUARD["armed"] or _GUARD["busy"]:
        return
    _GUARD["busy"] = True
    try:
        target = write_target(event, args)
        if target is not None and not under_root(target):
            _GUARD["violations"].append((event, target))
    finally:
        _GUARD["busy"] = False


sys.addaudithook(_audit_hook)


def arm() -> None:
    _GUARD["violations"] = []
    _GUARD["armed"] = True


def disarm() -> list:
    _GUARD["armed"] = False
    return list(_GUARD["violations"])


class P3Case(unittest.TestCase):
    """A fresh TemporaryDirectory under TMPDIR per test, the write guard armed, no real S3 client factory, and the
    owner-record locations (RUN/receipts/P2, RUN/holdout, OWNER_DIRECTIVES.md) moved under that directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=ALLOWED_ROOT, prefix="p3-")
        self.tmp = Path(self._tmp.name)
        self.p2_receipts = self.tmp / "RUN" / "receipts" / "P2"
        self.p2_receipts.mkdir(parents=True)
        for name, kw in (("_default_clients",
                          {"side_effect": AssertionError("a real S3 client was requested in an offline test")}),
                         ("P2_RECEIPTS_DIR", {"new": self.p2_receipts}),
                         ("HOLDOUT_DIR", {"new": self.tmp / "RUN" / "holdout"}),
                         ("OWNER_DIRECTIVES_PATH", {"new": self.tmp / "plans" / "OWNER_DIRECTIVES.md"})):
            patcher = mock.patch.object(holdout, name, **kw)
            patcher.start()
            self.addCleanup(patcher.stop)
        arm()

    def tearDown(self):
        violations = disarm()
        self._tmp.cleanup()
        self.assertEqual(violations, [], "the test wrote outside TMPDIR")


# ------------------------------------------------------------------------------------------- registry fixtures
def h(tag: str) -> str:
    return hashlib.sha256((FIXTURE + ":" + tag).encode()).hexdigest()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fid(tag: str) -> str:
    return "fam-" + h("family:" + tag)


def family(tag, *, qualified="auto", rights="unknown", sealed=False, seed=False, n_datasets=1, folds=5,
           cohort="frozen", external=False, excluded=1):
    """One P2-shaped family. Context c of member d at fold j is h('context:<tag>:<d>:<j>'); frozen fold 0 is True only
    for the frozen cohort. Sealed families carry a canary in places a careless copier would leak (display, a
    plaintext union key, an external alias id); P3 outputs must never contain it."""
    f = fid(tag)
    members, contexts, fold0 = [], [], {}
    for d in range(n_datasets):
        ds = h(f"dataset:{tag}:{d}")[:16]
        ident = "item:" + h(f"item:{tag}:{d}")
        key = f"rdatasets:pkg/{tag}-{d}"
        m = {"identity": ident, "kind": "registry_item", "dataset_hash": ds, "canonical_sha256": h(f"canonical:{tag}:{d}"),
             "sample_identity_sha256": h(f"sample:{tag}:{d}"), "cohort": cohort, "epoch_status": "included",
             "registry_state": "frozen" if cohort == "frozen" else "validated",
             "display": {"name": SEALED_CANARY} if sealed else {"name": f"dataset {tag}/{d}"},
             "union_keys": [{"kind": "rdatasets_item", "value": SEALED_CANARY + key if sealed else key}]}
        if cohort == "incoming":
            m["item_id"] = ident[len("item:"):]
        members.append(m)
        for j in range(folds):
            c = h(f"context:{tag}:{d}:{j}")
            contexts.append(c)
            fold0[c] = cohort == "frozen" and j == 0
    rec = {
        "family_id": f, "anchor_identity": min(m["identity"] for m in members), "members": members,
        "union_edges": [], "independence_qualified": qualified, "qualification_basis": {"reason": "fixture"},
        "rights_status": rights, "rights_members": {}, "sealed": sealed,
        "sealed_basis": "lineage_b_member" if sealed else None, "seed_selection_exposed": seed,
        "seed_sources": [f"beyondarena:{tag}"] if seed else [], "seed_evidence": None,
        "seeds_from_sources": 7 if seed else 0, "contexts": sorted(contexts), "frozen_fold0": fold0,
        "excluded_contexts": {h(f"excluded:{tag}:{i}"): "withdrawn" for i in range(excluded)},
        "registry_only_contexts": sorted(h(f"registry-only:{tag}:{d}") for d in range(n_datasets)),
        "n_identities": n_datasets, "n_datasets": n_datasets, "n_contexts": len(contexts), "cohorts": [cohort],
        "folds": {}, "screen_coverage": "full", "review_group_ids": [], "encoder_weight_exposure": {},
        "external_aliases": ([{"source": "beyondarena", "external_id": SEALED_CANARY if sealed else f"ba-{tag}",
                               "key_kind": "curated_alias", "role": "annotation"}] if external or sealed else []),
        "display": None if sealed else {"names": [f"dataset {tag}"],
                                        "platforms": ["rdatasets" if cohort == "frozen" else "socrata"], "urls": []},
        "provenance_refs": {}, "fixture": {"tag": tag, "folds": folds, "excluded": excluded},
    }
    return f, rec


NONPOOL_SPECS = (("u0", dict(qualified="unresolved")), ("u1", dict(qualified="unresolved", cohort="incoming")),
                 ("u2", dict(qualified="unresolved", rights="recorded:MIT")), ("u3", dict(qualified="unresolved", n_datasets=3)),
                 ("s0", dict(seed=True)), ("s1", dict(seed=True, qualified="reviewed", cohort="incoming")),
                 ("c0", dict(qualified="unresolved", folds=0)),
                 ("x0", dict(sealed=True, qualified="unresolved")), ("x1", dict(sealed=True, qualified="unresolved", n_datasets=2)),
                 ("x2", dict(sealed=True, qualified="unresolved", rights="prohibited")),
                 ("q0", dict(rights="prohibited")), ("q1", dict(rights="prohibited", qualified="unresolved", cohort="incoming")))
EXPECTED_NONPOOL = {"u0": "train_nonpool", "u1": "train_nonpool", "u2": "train_nonpool", "u3": "train_nonpool",
                    "s0": "train_nonpool", "s1": "train_nonpool", "c0": "train_nonpool", "x0": "sealed_excluded",
                    "x1": "sealed_excluded", "x2": "sealed_excluded", "q0": "prohibited_excluded",
                    "q1": "prohibited_excluded"}


def standard_families(n_pool: int = 60) -> dict:
    """n_pool selectable families (auto/reviewed, frozen/incoming, some with 2 datasets) plus 12 non-pool families:
    unresolved, seed-exposed, contextless, sealed (one also prohibited) and prohibited."""
    fams = {}
    for i in range(n_pool):
        f, rec = family(f"p{i:03d}", qualified="auto" if i % 3 else "reviewed",
                        cohort="frozen" if i % 2 == 0 else "incoming", n_datasets=2 if i % 7 == 0 else 1,
                        external=i % 10 == 0)
        fams[f] = rec
    for tag, kw in NONPOOL_SPECS:
        f, rec = family(tag, **kw)
        fams[f] = rec
    return fams


def pool_only_families(n: int) -> dict:
    return dict(family(f"n{i:04d}", folds=1, excluded=0) for i in range(n))


def registry_doc(families, *, alias_sha, csv_sha, decisions_sha, phase="final", epoch_ref=None,
                 decision_deltas=None) -> dict:
    doc = {"schema": "jope.dl-family-registry/v1", "experiment_id": "gp-policy-dl-v1-e0", "phase": phase,
           "epoch_ref": epoch_ref, "alias_sources_sha256": alias_sha,
           "registry_snapshot": {"poll_seq": 7, "sha256": h("snapshot")}, "review_csv_sha256": csv_sha,
           "review_decisions_sha256": decisions_sha, "policy": {"fixture": FIXTURE}, "encoders": {},
           "identities": {m["identity"]: f for f, rec in families.items() for m in rec["members"]},
           "families": families, "retired_family_ids": {}, "counts": {}}
    if decision_deltas is not None:            # a final registry's P2 CL-2 marker (families run_build final.deltas)
        doc["decision_deltas"] = decision_deltas
    return doc


def simple_registry_doc(families=None) -> dict:
    return registry_doc(standard_families() if families is None else families, alias_sha=h("alias"),
                        csv_sha=h("csv"), decisions_sha=h("decisions"))


REVIEW_COLUMNS = ("review_group_id", "kind", "trigger_kinds", "trigger_value", "family_ids", "members", "n_families",
                  "n_datasets", "n_contexts", "n_fold0", "cohorts", "evidence", "flags", "impact_if_union",
                  "recommendation", "recommendation_basis", "default_if_undecided", "decision_options")


def review_csv(families, n_seed_rows=10) -> tuple[bytes, list[dict]]:
    seed_fams = sorted(f for f, r in families.items() if r["seed_selection_exposed"]) or [""]
    rows = [{"review_group_id": "rg-" + h(f"seed-row:{i}")[:24], "kind": "seed_source_mapping",
             "trigger_value": f"beyondarena:source-{i}", "family_ids": seed_fams[i % len(seed_fams)],
             "recommendation": "confirm_mapping", "recommendation_basis": "agent rule (agent-generated)",
             "default_if_undecided": "P3 STOP", "decision_options": "confirm_mapping;add_members;reject_mapping"}
            for i in range(n_seed_rows)]
    rows.append({"review_group_id": "rg-" + h("publisher-row")[:24], "kind": "publisher", "trigger_value": "example.org",
                 "recommendation": "independent", "default_if_undecided": "unresolved",
                 "decision_options": "independent;union;union_subset;unresolved"})
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in REVIEW_COLUMNS})
    return buf.getvalue().encode(), rows


def decisions_doc(rows, csv_sha, alias_sha, *, seed_decision="confirm_mapping", use_class=False, extra=(),
                  drop_seed_rows=0, provenance=True, when="recommendation==confirm_mapping") -> dict:
    """provenance: True gives a fixed owner_verbatim_message block (it resolves to no stored message), a dict is used
    as the provenance block itself, False leaves it out."""
    seed_rows = [r for r in rows if r["kind"] == "seed_source_mapping"]
    dec = [] if use_class else [{"review_group_id": r["review_group_id"], "decision": seed_decision}
                                for r in seed_rows[drop_seed_rows:]]
    dec += [{"review_group_id": r["review_group_id"], "decision": "independent"} for r in rows if r["kind"] == "publisher"]
    classes = [{"kind": "seed_source_mapping", "when": when, "decision": seed_decision}] if use_class else []
    doc = {"schema": "jope.dl-family-review-decisions/v1", "reviewer": "owner",
           "binds": {"review_csv_sha256": csv_sha, "registry_pre_review_sha256": h("pre-review"),
                     "alias_sources_sha256": alias_sha},
           "rulings": {"rdatasets_granularity": "item", "provisional_seals": "keep", "related_class_policy": "unresolved",
                       "split_registry_excluded_seals_nothing": True},
           "class_decisions": classes, "decisions": dec + list(extra)}
    if isinstance(provenance, dict):
        doc["provenance"] = provenance
    elif provenance:
        doc["provenance"] = {"kind": "owner_verbatim_message", "session": "fixture session",
                             "message_utc": OWNER_MESSAGE_UTC, "message_sha256": h("owner-message")}
    return doc


def owner_message(csv_sha: str) -> bytes:
    """A fixture owner confirmation in the agreed shape: it names the review.csv sha prefix it answers."""
    return (f"DL-D5 confirmed for review.csv {csv_sha[:12]}: Q1=a, Q2=a, Q3=a, Q4=a, Q5=a; "
            "split-registry excluded seals nothing.\n").encode()


def owner_message_path(message: bytes) -> Path:
    """Where review_gate looks for a stored owner message: holdout.P2_RECEIPTS_DIR/dl-d5-owner-message-<sha16>.txt."""
    return Path(holdout.P2_RECEIPTS_DIR) / families.OWNER_MESSAGE_NAME.format(sha16=sha(message)[:16])


def store_owner_message(message: bytes) -> Path:
    """Store an owner message as the DL-D5 expansion does (under the receipts directory), only ever under TMPDIR."""
    p = owner_message_path(message)
    if not under_root(os.path.realpath(p.parent)):
        raise RuntimeError("tests store owner messages only under TMPDIR (P3Case moves holdout.P2_RECEIPTS_DIR there)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(message)
    return p


def message_provenance(message: bytes, **overrides) -> dict:
    return {"kind": "owner_verbatim_message", "session": "fixture session", "message_utc": OWNER_MESSAGE_UTC,
            "message_sha256": sha(message)} | overrides


def cl2_marker(record: dict | None, *, draft=False, simulation=False) -> dict:
    """decision_deltas.provenance as families run_build writes it into a final registry (the P2 CL-2 marker)."""
    return {"draft": draft, "simulation": simulation, "record": record}


def alias_sources_doc() -> dict:
    return {"schema": "jope.dl-family-alias-sources/v1", "experiment_id": "gp-policy-dl-v1-e0", "created_utc_us": 1,
            "policy": {"fixture": FIXTURE},
            "inputs": [{"name": "beyondarena-seed-pointers", "path": "/fixture/beyondarena", "sha256": h("ba"), "bytes": 1,
                        "role": "seed_source"},
                       {"name": "v6-named-datasets", "path": "/fixture/v6", "sha256": h("v6"), "bytes": 1, "role": "alias"}],
            "s3_reads": [], "registry_snapshot": {"path": "/fixture/snapshot", "sha256": h("snapshot"), "poll_seq": 7}}


# ---------------------------------------------------------------------------------------- P3 gate fixtures
OWNER_DECISIONS = {"DL-D2": "a: unknown-licence datasets may train for research with rights_status recorded",
                   "DL-D4": "a: normalizers and projections refit on train_fit families, with new ABI IDs",
                   "DL-D5": "conservative union-find + value-level near-duplicate screen + human review",
                   "DL-D6": "commitment-derived holdout seed; first freeze wins",
                   "DL-D12": "keep lineage-B meta_validation/meta_test families sealed (excluded everywhere)"}


def owner_decisions_doc(*, decisions=None, drop=(), schema="jope.dl-owner-decisions/v1", provenance=True) -> dict:
    dec = dict(OWNER_DECISIONS) | dict(decisions or {})
    for d in drop:
        dec.pop(d, None)
    doc = {"schema": schema, "decisions": dec, "recorded_utc_us": 1}
    if provenance:
        doc["provenance"] = {"kind": "owner_verbatim_message", "session": "fixture session",
                             "owner_message_utc": "2026-10-05T04:41:16Z"}
    return doc


def registry_s3_key(registry_sha: str) -> str:
    return s3io.PREFIX_KEY + f"objects/family-registry/sha256/{registry_sha}.json"


def p2_receipt_doc(registry_sha: str, registry_bytes: int, *, status="PASS", packet="P2",
                   schema="jope.dl-packet-receipt/v1", with_ref=True, ref_overrides=None) -> dict:
    """A P2 packet receipt (PLAN §5 Format) whose outputs carry the published registry's s3ref."""
    out = {"path": "/fixture/RUN/families/registry.json", "sha256": registry_sha, "bytes": registry_bytes}
    if with_ref:
        out["s3"] = {"bucket": s3io.BUCKET, "key": registry_s3_key(registry_sha), "version_id": REGISTRY_VID,
                     "sha256": registry_sha, "bytes": registry_bytes} | dict(ref_overrides or {})
    return {"schema": schema, "packet": packet, "status": status, "outputs": {"registry": out}}


@dataclasses.dataclass
class Inputs:
    dir: Path
    registry: Path
    registry_sha256: str
    decisions: Path
    review_csv: Path
    alias_sources: Path
    p2_receipt: Path
    owner_decisions: Path
    owner_decisions_sha256: str
    doc: dict
    families: dict
    owner_message: Path | None = None


def write_inputs(d: Path, families=None, *, n_seed_rows=10, phase="final", registry_decisions_sha=None,
                 csv_bytes_override=None, epoch_ref=None, p2=None, p2_bytes_override=None, owner=None,
                 message=None, store_message=True, prov_overrides=None, registry_provenance=None,
                 **decision_kw) -> Inputs:
    """Registry, review files, alias sources and the P3 gate inputs. p2 and owner are keyword overrides for
    p2_receipt_doc and owner_decisions_doc.

    Owner record (review fix P3I-1): unless decision_kw names a provenance, the decision file's provenance is an
    owner_verbatim_message for `message` (default owner_message(review.csv sha)) plus prov_overrides, and the message
    is stored at owner_message_path() unless store_message is False. A final registry carries decision_deltas with
    provenance registry_provenance (default: draft and simulation false and the stored message's record; False leaves
    decision_deltas out)."""
    d.mkdir(parents=True, exist_ok=True)
    families = standard_families() if families is None else families
    alias_b = common.canonical(alias_sources_doc())
    csv_b, rows = review_csv(families, n_seed_rows)
    msg = owner_message(sha(csv_b)) if message is None else message
    msg_path = store_owner_message(msg) if store_message else owner_message_path(msg)
    if decision_kw.get("provenance", True) is True:
        decision_kw["provenance"] = message_provenance(msg, **dict(prov_overrides or {}))
    dec_b = common.canonical(decisions_doc(rows, sha(csv_b), sha(alias_b), **decision_kw))
    deltas = None
    if phase == "final" and registry_provenance is not False:
        marker = registry_provenance if registry_provenance is not None else cl2_marker(
            {"path": str(msg_path), "sha256": sha(msg), "form": "stored_owner_message"})
        deltas = {"decisions": {}, "undecided_rows": 0, "provenance": marker}
    doc = registry_doc(families, alias_sha=sha(alias_b), csv_sha=sha(csv_b),
                       decisions_sha=registry_decisions_sha or sha(dec_b), phase=phase, epoch_ref=epoch_ref,
                       decision_deltas=deltas)
    reg_b = common.canonical(doc)
    p2_b = p2_bytes_override if p2_bytes_override is not None else \
        common.canonical(p2_receipt_doc(sha(reg_b), len(reg_b), **(p2 or {})))
    owner_b = common.canonical(owner_decisions_doc(**(owner or {})))
    paths = {"registry": d / "registry.json", "decisions": d / "review-decisions.json", "review_csv": d / "review.csv",
             "alias_sources": d / "alias-sources.json", "p2_receipt": d / "receipts" / "P2" / "receipt.json",
             "owner_decisions": d / "owner-decisions.json"}
    paths["p2_receipt"].parent.mkdir(parents=True, exist_ok=True)
    paths["registry"].write_bytes(reg_b)
    paths["decisions"].write_bytes(dec_b)
    paths["review_csv"].write_bytes(csv_bytes_override if csv_bytes_override is not None else csv_b)
    paths["alias_sources"].write_bytes(alias_b)
    paths["p2_receipt"].write_bytes(p2_b)
    paths["owner_decisions"].write_bytes(owner_b)
    return Inputs(dir=d, registry_sha256=sha(reg_b), owner_decisions_sha256=sha(owner_b), doc=doc, families=families,
                  owner_message=msg_path, **paths)


# ---------------------------------------------------------------------------------------------------- fake S3
def client_error(status: int, code: str, op: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, op)


HISTORY_PARAMS = frozenset({"Bucket", "Prefix", "MaxKeys"})


def proposed_history_guard(params: dict) -> None:
    """The exact-key, bounded ListObjectVersions contract of LP-1, which s3io.Guard now enforces; kept as an independent
    cross-check of the guard: only Bucket, Prefix and MaxKeys; the lane bucket; Prefix one full key inside PREFIX_KEY;
    1 <= MaxKeys <= 100."""
    key, mk = params.get("Prefix"), params.get("MaxKeys")
    if set(params) != HISTORY_PARAMS or params.get("Bucket") != s3io.BUCKET:
        raise s3io.S3Refused("proposed history guard: only Bucket, Prefix and MaxKeys on the lane bucket")
    if not (isinstance(key, str) and key.startswith(s3io.PREFIX_KEY) and not key.endswith("/")
            and ".." not in key.split("/")):
        raise s3io.S3Refused("proposed history guard: Prefix must be one full key inside PREFIX")
    if not isinstance(mk, int) or isinstance(mk, bool) or not 1 <= mk <= 100:
        raise s3io.S3Refused("proposed history guard: 1 <= MaxKeys <= 100")


class FakeBucket:
    """Versioned objects per (bucket, key), newest last; a delete marker is a version whose body is None.
    preload() plants an object as if another writer won; delete_marker() simulates an out-of-lane delete.
    fail_put() queues a PUT failure for a key (store=True: the object is stored and the response lost);
    after_put[key] runs once after a stored PUT (another writer acting right after ours)."""

    def __init__(self, history_supported: bool = True):
        self.objects: dict[tuple[str, str], list[tuple[str, bytes | None]]] = {}
        self.calls: list[tuple[str, str]] = []
        self.n = 0
        self.history_supported = history_supported
        self.put_faults: dict[str, list[tuple[BaseException, bool]]] = {}
        self.after_put: dict[str, object] = {}

    def _vid(self, body: bytes | None) -> str:
        self.n += 1
        return f"v{self.n:04d}-{sha(body)[:8] if body is not None else 'marker'}"

    def preload(self, key: str, body: bytes) -> str:
        vid = self._vid(body)
        self.objects.setdefault((s3io.BUCKET, key), []).append((vid, body))
        return vid

    def plant(self, key: str, body: bytes | None, version_id: str) -> str:
        """An object another lane wrote (for example P2 publish) at a known VersionId; body None is a delete marker."""
        self.objects.setdefault((s3io.BUCKET, key), []).append((version_id, body))
        return version_id

    def delete_marker(self, key: str) -> str:
        vid = self._vid(None)
        self.objects.setdefault((s3io.BUCKET, key), []).append((vid, None))
        return vid

    def fail_put(self, key: str, error: BaseException, store: bool = False) -> None:
        self.put_faults.setdefault(key, []).append((error, store))

    def versions(self, key: str) -> list:
        return list(self.objects.get((s3io.BUCKET, key), []))

    def current(self, key: str):
        v = self.versions(key)
        return v[-1] if v else None

    def puts(self) -> list:
        return [k for op, k in self.calls if op == "PutObject"]


class FakeS3Client:
    def __init__(self, bucket: FakeBucket, writes: bool = False):
        self.bucket = bucket
        self._dltrain_guard = s3io.Guard(writes=writes)

    def _check(self, op, params):
        self._dltrain_guard.check_params(params, types.SimpleNamespace(name=op))
        self.bucket.calls.append((op, params.get("Key")))

    def put_object(self, **params):
        self._check("PutObject", params)
        key = params["Key"]
        faults = self.bucket.put_faults.get(key)
        fault = faults.pop(0) if faults else None
        if fault is not None and not fault[1]:
            raise fault[0]
        versions = self.bucket.objects.setdefault((params["Bucket"], key), [])
        if params.get("IfNoneMatch") == "*" and versions and versions[-1][1] is not None:
            raise client_error(412, "PreconditionFailed", "PutObject")
        body = bytes(params["Body"])
        vid = self.bucket._vid(body)
        versions.append((vid, body))
        hook = self.bucket.after_put.pop(key, None)
        if hook is not None:
            hook(self.bucket)
        if fault is not None:
            raise fault[0]
        return {"VersionId": vid, "ETag": '"' + hashlib.md5(body, usedforsecurity=False).hexdigest() + '"',
                "ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_object(self, **params):
        self._check("GetObject", params)
        versions = self.bucket.objects.get((params["Bucket"], params["Key"])) or []
        if "VersionId" in params:
            found = [v for v in versions if v[0] == params["VersionId"]]
            if not found:
                raise client_error(404, "NoSuchVersion", "GetObject")
            if found[0][1] is None:
                raise client_error(405, "MethodNotAllowed", "GetObject")
        else:
            found = versions[-1:]
            if not found or found[0][1] is None:
                raise client_error(404, "NoSuchKey", "GetObject")
        vid, body = found[0]
        return {"Body": StreamingBody(io.BytesIO(body), len(body)), "VersionId": vid, "ContentLength": len(body),
                "ETag": '"' + hashlib.md5(body, usedforsecurity=False).hexdigest() + '"'}

    def list_object_versions(self, **params):
        self._check("ListObjectVersions", params)                 # the real lane guard (s3io allows the exact-key form)
        proposed_history_guard(params)
        if not self.bucket.history_supported:                     # simulates a guard that refuses history (fail closed)
            raise s3io.S3Refused("simulated: the lane guard refuses ListObjectVersions")
        self.bucket.calls.append(("ListObjectVersions", params["Prefix"]))
        entries = []
        for (b, k), vs in sorted(self.bucket.objects.items()):
            if b == params["Bucket"] and k.startswith(params["Prefix"]):
                entries += [(k, vid, body, i == 0) for i, (vid, body) in enumerate(reversed(vs))]
        page = entries[:params["MaxKeys"]]
        return {"Versions": [{"Key": k, "VersionId": v, "IsLatest": latest, "Size": len(b)}
                             for k, v, b, latest in page if b is not None],
                "DeleteMarkers": [{"Key": k, "VersionId": v, "IsLatest": latest} for k, v, b, latest in page if b is None],
                "IsTruncated": len(entries) > params["MaxKeys"]}


def fake_clients(bucket: FakeBucket) -> dict:
    return {"writer": FakeS3Client(bucket, writes=True), "reader": FakeS3Client(bucket)}


def registry_bucket(inputs: Inputs, **kw) -> FakeBucket:
    """A FakeBucket holding what P2 publish wrote: the registry at the P2 receipt's exact VersionId (P3I-2)."""
    bucket = FakeBucket(**kw)
    bucket.plant(registry_s3_key(inputs.registry_sha256), inputs.registry.read_bytes(), REGISTRY_VID)
    return bucket


def guarded_client(writes: bool = False):
    """The real lane client (s3io.make_client) on fake credentials, with a tripwire before any send."""
    session = boto3.session.Session(aws_access_key_id="testing", aws_secret_access_key="testing", region_name="us-east-2")
    c = s3io.make_client(writes=writes, session=session)

    def tripwire(**_):
        raise AssertionError("a request reached before-send in an offline test")
    c.meta.events.register("before-send.*.*", tripwire)
    return c


def streaming(body: bytes) -> StreamingBody:
    return StreamingBody(io.BytesIO(body), len(body))


# ----------------------------------------------------------------------------------------------------- CLI glue
def run_cli(argv, clients=None, **kw) -> tuple[int, dict]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = holdout._main([str(a) for a in argv], clients=clients, **kw)
    lines = buf.getvalue().strip().splitlines()
    return code, (json.loads(lines[-1]) if lines else {})


def receipt(summary: dict) -> dict:
    return json.loads(Path(summary["receipt"]["path"]).read_text())


def commit_args(inputs: Inputs, out: Path) -> list:
    return ["commit", "--registry", inputs.registry, "--registry-sha256", inputs.registry_sha256,
            "--decisions", inputs.decisions, "--p2-receipt", inputs.p2_receipt,
            "--owner-decisions", inputs.owner_decisions, "--owner-decisions-sha256", inputs.owner_decisions_sha256,
            "--out", out]


def freeze_args(inputs: Inputs, out: Path) -> list:
    return ["freeze", "--registry", inputs.registry, "--registry-sha256", inputs.registry_sha256,
            "--seed-file", out / "seed.hex", "--out", out]


def verify_args(inputs: Inputs, out: Path, poison=True) -> list:
    return ["verify", "--holdout", out / "holdout.json", "--registry", inputs.registry, "--seed-file", out / "seed.hex"] \
        + (["--assignment-poison"] if poison else [])


def commit_files(src: Path, dst: Path) -> None:
    """Copy what commit wrote (seed excluded) into another holdout directory."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("commitment.json", "commitment-ref.json", "gate.json"):
        (dst / name).write_bytes((src / name).read_bytes())


def pipeline(case: unittest.TestCase, inputs: Inputs, bucket: FakeBucket | None = None, verify_too=True):
    """commit -> freeze (-> verify --assignment-poison) with a FakeS3 that holds the P2-published registry; returns
    (holdout dir, bucket)."""
    bucket = bucket or registry_bucket(inputs)
    out = inputs.dir / "holdout"
    code, s = run_cli(commit_args(inputs, out), clients=fake_clients(bucket))
    case.assertEqual(code, 0, s)
    code, s = run_cli(freeze_args(inputs, out))
    case.assertEqual(code, 0, s)
    if verify_too:
        code, s = run_cli(verify_args(inputs, out))
        case.assertEqual(code, 0, s)
    return out, bucket


def closure_args(inputs: Inputs, out: Path, paths: dict, *extra, pin_epoch=True, pin_master=True) -> list:
    argv = ["closure", "--holdout", out / "holdout.json", "--registry", inputs.registry, "--epoch", paths["epoch"],
            "--master", paths["master"], "--registry-db", paths["db"], "--alias-sources", inputs.alias_sources]
    if pin_epoch:
        argv += ["--epoch-sha256", sha(paths["epoch"].read_bytes())]
    if pin_master:
        argv += ["--master-sha256", sha(paths["master"].read_bytes())]
    return argv + list(extra)


def publish_ready(case: unittest.TestCase, inputs: Inputs, bucket: FakeBucket | None = None):
    """commit -> freeze -> verify --assignment-poison -> closure (PASS); returns (holdout dir, bucket, closure paths)."""
    out, bucket = pipeline(case, inputs, bucket)
    paths = closure_inputs(inputs.dir / "closure-in", inputs)
    code, s = run_cli(closure_args(inputs, out, paths))
    case.assertEqual(code, 0, s)
    return out, bucket, paths


def publish_args(out: Path, *extra) -> list:
    return ["publish", "--holdout", out / "holdout.json", *extra]


# -------------------------------------------------------------------------------------------- closure inputs
def _ref(kind: str, c: str) -> dict:
    return {"bucket": "fixture-bucket", "key": f"fixture/{kind}/{c}", "version_id": "fx" + h(kind + c)[:8],
            "sha256": h(f"{kind}:{c}"), "bytes": 100}


def epoch_row(rec: dict, m: dict, c: str, fold: int, status="included", reason=None) -> dict:
    row = {"cohort": m["cohort"], "view_id": f"{m['dataset_hash']}__fold{fold}", "dataset_hash": m["dataset_hash"],
           "fold": fold, "context_sha256": c, "context_schema": "fixture/v1",
           "sample_identity_sha256": m["sample_identity_sha256"], "history_ref": _ref("history", c),
           "telemetry_ref": _ref("telemetry", c), "input_ref": _ref("input", c), "membership_ref": _ref("membership", c),
           "native_source": {"path": f"/fixture/native/{c}.jsonl", "sha256": h("native:" + c), "bytes": 10,
                             "mtime_ns": "1"},
           "prepared_ref": {"path": f"/fixture/prepared/{c}.json", "sha256": h("prepared:" + c), "bytes": 10},
           "eligibility_source": "fixture", "status": status, "exclusion_reason": reason}
    if m["cohort"] == "incoming":
        row["item_id"] = m["identity"][len("item:"):]
    return row


def closure_docs(families: dict, extra_dataset_link_family: str | None = None) -> tuple[dict, dict, list]:
    """epoch-000-shaped rows for every registry context and excluded context, master views (one equivalent each) for
    frozen contexts, and the intake contexts-table rows (including registry-only rows)."""
    rows, views, db_rows = [], [], []
    for f in sorted(families):
        rec = families[f]
        tag, folds = rec["fixture"]["tag"], rec["fixture"]["folds"]
        for di, m in enumerate(rec["members"]):
            if not m["identity"].startswith("item:"):
                continue                                  # external identities have no epoch rows or registry items
            item = m["identity"][len("item:"):]
            for j in range(folds):
                c = h(f"context:{tag}:{di}:{j}")
                rows.append(epoch_row(rec, m, c, j))
                db_rows.append((c, item))
                if m["cohort"] == "frozen":
                    views.append({"context_sha256": c, "dataset_hash": m["dataset_hash"], "view_id": f"{m['dataset_hash']}__fold{j}",
                                  "equivalent_evaluation_contexts": [
                                      {"context_sha256": h("equivalent:" + c), "prepared_path": f"/fixture/other/{c}.json",
                                       "compatibility_identity": {"input_raw_sha256": h("raw:" + c), "train_rows": 10}}],
                                  "tabpfn35": {"rmse": 1.25, "rmse_ratio_to_train_mean": 0.5}})
            db_rows.append((h(f"registry-only:{tag}:{di}"), item))
        for c in sorted(rec["excluded_contexts"]):
            rows.append(epoch_row(rec, rec["members"][0], c, 4, status="excluded", reason="withdrawn"))
    if extra_dataset_link_family:
        rec = families[extra_dataset_link_family]
        rows.append(epoch_row(rec, rec["members"][0], h("sibling-sample:" + extra_dataset_link_family), 1))
    epoch = {"schema": "jope.dl-corpus-epoch/v1", "experiment_id": "gp-policy-dl-v1-e0", "epoch": 0, "contexts": rows}
    master = {"schema": "fixture-master/v1", "views": views}
    return epoch, master, db_rows


def write_closure_inputs(d: Path, epoch: dict, master: dict, db_rows: list) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    paths = {"epoch": d / "epoch-000.json", "master": d / "input-manifest-master.json", "db": d / "intake.sqlite3"}
    paths["epoch"].write_bytes(common.canonical(epoch))
    paths["master"].write_bytes(common.canonical(master))
    con = sqlite3.connect(str(paths["db"]))
    try:
        con.execute("create table polls (seq integer primary key, utc_ns text, status text, payload text)")
        con.execute("create table contexts (context text primary key, item_id text)")
        con.executemany("insert into polls (seq, utc_ns, status, payload) values (?, ?, 'ok', '{}')",
                        [(s, str(s)) for s in (1, 2, 3)])
        con.executemany("insert into contexts (context, item_id) values (?, ?)", db_rows)
        con.commit()
    finally:
        con.close()
    return paths


def closure_inputs(d: Path, inputs, extra_dataset_link_family: str | None = None) -> dict:
    """Closure input files for an Inputs (or a families dict): epoch, master and a synthetic intake SQLite file."""
    families = inputs.families if isinstance(inputs, Inputs) else inputs
    return write_closure_inputs(d, *closure_docs(families, extra_dataset_link_family))
