"""Fixture-only Lane B seal.

A release happens only when kind is "fixture", the path stays inside the
caller-supplied fixture root, and owner_fixture_release is true. Kind "owner",
a missing kind, or a path outside that root raises RealSealRefusal and leaves
the bytes unchanged. Nothing here scans the machine for a real seal.
"""

import fcntl
import json
import os
import random
import threading
from pathlib import Path

from rlfw.arena.game_spec import canonical, digest, require_sha

from .crosswalk import classify_family, roster_status
from .g_judge import PRIMARY_B, select_program

SCHEMA = "jope.rlfw-judge-seal/v1"
ALLOWED_ROLE = "rights_cleared"
REFUSED_ROLES = frozenset({"development", "sealed", "rights_pending", "train"})
_GUARD = threading.Lock()
_PATH_LOCKS = {}


class RealSealRefusal(PermissionError):
    """A real seal, or a release that was not explicitly a fixture, was refused."""


class SealSpent(RuntimeError):
    """The one release, or the one score of a judge context, has already happened."""


class QuarantineError(PermissionError):
    """Overlap is quarantine or overlap_unknown, so this family is not scored."""


def _thread_lock(path):
    key = str(Path(path).resolve())
    with _GUARD:
        return _PATH_LOCKS.setdefault(key, threading.Lock())


def fit_judge_split(row_ids, seed):
    """80/20 from row ids and the seed only. No feature or label is read."""
    ids = list(row_ids)
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("row ids must be unique and at least two")
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("row ids must be non-empty strings")
    n_fit = max(1, min(len(ids) - 1, (4 * len(ids)) // 5))
    order = list(ids)
    random.Random(seed).shuffle(order)
    return order[:n_fit], order[n_fit:]


def _family(raw, seed):
    role = raw.get("role")
    if role in REFUSED_ROLES or role != ALLOWED_ROLE:
        raise ValueError(f"Lane B allows only {ALLOWED_ROLE}")
    fit_ids, judge_ids = fit_judge_split(raw["row_ids"], seed)
    family = {
        "family_id": raw["family_id"],
        "fit_row_ids": fit_ids,
        "judge_context_id": raw["judge_context_id"],
        "judge_row_ids": judge_ids,
        "lineage_id": raw["lineage_id"],
        "pretraining_disclosure": raw.get("pretraining_disclosure"),
        "role": role,
        "row_ids": list(raw["row_ids"]),
        "source_ids": raw.get("source_ids"),
    }
    if not family["family_id"] or not family["lineage_id"] or not family["judge_context_id"]:
        raise ValueError("family, lineage, and judge context ids are required")
    family["overlap_status"] = classify_family(family)
    return family


def _check_disclosures(disclosures, families, arm):
    if not isinstance(disclosures, dict):
        raise ValueError("disclosures must be keyed by family")
    if set(disclosures) != {family["family_id"] for family in families}:
        raise ValueError("disclosures must cover each family once")
    seen = set()
    judge_contexts = {family["judge_context_id"] for family in families}
    for family in families:
        items = disclosures[family["family_id"]]
        if not isinstance(items, list) or len(items) < PRIMARY_B:
            raise ValueError("each family needs at least 8 inner disclosures")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("disclosure must be an object")
            for banned in ("judge_q", "judge_row_ids", "judge_labels"):
                if banned in item:
                    raise ValueError("judge fields are banned in inner disclosures")
            if item.get("arm") != arm:
                raise ValueError("one shared arm")
            context_id = item.get("context_id")
            if not isinstance(context_id, str) or not context_id or context_id in seen:
                raise ValueError("context ids must be unique")
            if context_id in judge_contexts:
                raise ValueError("inner disclosures must not name the judge context")
            seen.add(context_id)
            require_sha(item.get("program_sha256"), "program")
            if type(item.get("inner_q_grid")) is not int:
                raise ValueError("inner disclosures carry integer micro-q only")


def seal_body(document):
    return {
        "arm": document["arm"],
        "disclosures": document["disclosures"],
        "families": document["families"],
        "kind": document.get("kind"),
        "schema": document["schema"],
    }


def seal_id_of(document):
    return digest(seal_body(document))


def plan_body(document):
    return {
        "arm": document["arm"],
        "encoding_sha256": document["encoding_sha256"],
        "picks": document["picks"],
        "seal_id": document["seal_id"],
    }


def build_document(raw_families, disclosures, *, arm, seed, kind):
    if kind not in ("fixture", "owner"):
        raise ValueError("kind must be fixture or owner")
    if not isinstance(arm, str) or not arm:
        raise ValueError("one shared arm name is required")
    if not isinstance(raw_families, list) or not raw_families:
        raise ValueError("at least one family")
    families = [_family(raw, seed) for raw in raw_families]
    if len({family["family_id"] for family in families}) != len(families):
        raise ValueError("duplicate family id")
    if len({family["judge_context_id"] for family in families}) != len(families):
        raise ValueError("duplicate judge context id")
    _check_disclosures(disclosures, families, arm)
    document = {
        "arm": arm,
        "disclosures": disclosures,
        "encoding_sha256": None,
        "families": families,
        "kind": kind,
        "overlap_status": roster_status([family["overlap_status"] for family in families]),
        "picks": None,
        "plan_sha256": None,
        "dispatches": [],
        "release": None,
        "schema": SCHEMA,
        "scored": [],
        "state": "sealed",
    }
    document["seal_id"] = seal_id_of(document)
    return document


def _inside(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if root == path or not path.is_relative_to(root):
        raise RealSealRefusal("seal path must stay inside the caller-supplied fixture root")
    return path, root


def _atomic_write(path, document):
    payload = (canonical(document) + "\n").encode()
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _load_raw(path):
    document = json.loads(Path(path).read_text())
    if document.get("schema") != SCHEMA or document.get("seal_id") != seal_id_of(document):
        raise ValueError("seal id does not match the pre-plan body")
    if document.get("picks") is not None and document.get("plan_sha256") != digest(plan_body(document)):
        raise ValueError("selection plan was tampered")
    if document.get("state") not in ("sealed", "spent"):
        raise ValueError("unknown seal state")
    return document


def save_new(path, document):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if (
        document.get("state") != "sealed"
        or document.get("release")
        or document.get("picks")
        or document.get("scored")
        or document.get("dispatches")
    ):
        raise ValueError("a new seal is sealed, unreleased, and unpicked")
    if document.get("seal_id") != seal_id_of(document):
        raise ValueError("seal id mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, document)


def _mutate(path, update):
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    with _thread_lock(path):
        with lock_path.open("a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            document = _load_raw(path)
            update(document)
            _atomic_write(path, document)
            return document


def freeze(path, *, picks, encoding_sha256):
    """Record the B=8 pick before any release. Judge fields stay out of the plan."""
    require_sha(encoding_sha256, "encoding")

    def update(document):
        if document["state"] != "sealed" or document.get("release"):
            raise SealSpent("a released seal cannot be refrozen")
        if document["picks"] is not None:
            raise ValueError("selection plan is already frozen")
        expected = {family["family_id"] for family in document["families"]}
        if set(picks) != expected:
            raise ValueError("one pick per family")
        for family in document["families"]:
            chosen = select_program(document["disclosures"][family["family_id"]], PRIMARY_B)
            if picks[family["family_id"]] != chosen:
                raise ValueError("frozen pick is not the B=8 program")
        document["picks"] = {family_id: picks[family_id] for family_id in sorted(picks)}
        document["encoding_sha256"] = encoding_sha256
        document["plan_sha256"] = digest(plan_body(document))

    return _mutate(path, update)


def _refuse_unless_fixture_release(path, fixture_root, owner_fixture_release):
    if owner_fixture_release is not True:
        raise RealSealRefusal("release requires owner_fixture_release=True")
    if fixture_root is None:
        raise RealSealRefusal("release requires a caller-supplied fixture root")
    _inside(path, fixture_root)


def release(path, *, fixture_root, owner_fixture_release=False):
    """Spend a fixture seal once. A second call raises SealSpent."""
    _refuse_unless_fixture_release(path, fixture_root, owner_fixture_release)

    def update(document):
        if document.get("kind") != "fixture":
            raise RealSealRefusal("only a fixture seal can be released")
        if document["state"] != "sealed" or document.get("release"):
            raise SealSpent("the seal is spent after exactly one release")
        if not document.get("picks") or not document.get("plan_sha256"):
            raise ValueError("freeze the selection plan before release")
        document["release"] = {"count": 1, "spent": True}
        document["state"] = "spent"

    return _mutate(path, update)


def _score_record(arm, context_id, program_sha256):
    return {"arm": arm, "context_id": context_id, "program_sha256": program_sha256}


def _same_pick(item, record):
    return (
        item.get("arm") == record["arm"]
        and item.get("context_id") == record["context_id"]
        and item.get("program_sha256") == record["program_sha256"]
    )


def _require_released_pick(document, *, context_id, arm, program_sha256):
    released = document.get("kind") == "fixture" and document.get("state") == "spent"
    if not released or not (document.get("release") or {}).get("spent"):
        raise RealSealRefusal("scoring requires a spent fixture seal")
    if arm != document["arm"]:
        raise ValueError("arm does not match the frozen seal")
    family = next(
        (item for item in document["families"] if item["judge_context_id"] == context_id),
        None,
    )
    if family is None:
        raise ValueError("context is not a judge context on this seal")
    if family["overlap_status"] != "no_overlap":
        raise QuarantineError(family["overlap_status"])
    if document["picks"][family["family_id"]] != program_sha256:
        raise ValueError("only the precommitted pick can be scored")


def reserve_dispatch(path, *, context_id, arm, program_sha256, fixture_root):
    """Consume the pick under the seal lock before the evaluator is called."""
    _inside(path, fixture_root)
    require_sha(program_sha256, "program")

    def update(document):
        _require_released_pick(
            document, context_id=context_id, arm=arm, program_sha256=program_sha256
        )
        record = _score_record(arm, context_id, program_sha256)
        dispatches = document.setdefault("dispatches", [])
        if any(_same_pick(item, record) for item in dispatches) or record in document["scored"]:
            raise SealSpent("this judge context was already dispatched")
        dispatches.append({**record, "state": "reserved"})

    return _mutate(path, update)


def quarantine_dispatch(path, *, context_id, arm, program_sha256, fixture_root):
    """Keep a reserved dispatch consumed when the attempt outcome is unknown."""
    _inside(path, fixture_root)
    require_sha(program_sha256, "program")

    def update(document):
        record = _score_record(arm, context_id, program_sha256)
        for item in document.get("dispatches", []):
            if _same_pick(item, record) and item.get("state") == "reserved":
                item["state"] = "unknown"
                return

    return _mutate(path, update)


def mark_scored(path, *, context_id, arm, program_sha256, fixture_root):
    _inside(path, fixture_root)
    require_sha(program_sha256, "program")

    def update(document):
        _require_released_pick(
            document, context_id=context_id, arm=arm, program_sha256=program_sha256
        )
        record = _score_record(arm, context_id, program_sha256)
        if record in document["scored"]:
            raise SealSpent("this judge context was already scored")
        dispatch = next(
            (item for item in document.get("dispatches", []) if _same_pick(item, record)),
            None,
        )
        if dispatch is None or dispatch.get("state") != "reserved":
            raise SealSpent("this judge context was already dispatched")
        document["scored"] = [*document["scored"], record]
        dispatch["state"] = "scored"

    return _mutate(path, update)


class SealStore:
    """Reads a seal from disk. A forged in-memory dict is not a store."""

    def __init__(self, path, *, fixture_root):
        self.path, self.fixture_root = _inside(path, fixture_root)

    def load(self):
        return _load_raw(self.path)

    def require_spent_fixture(self):
        document = self.load()
        if document.get("kind") != "fixture" or document.get("state") != "spent":
            raise RealSealRefusal("scoring requires a spent fixture seal")
        return document

    def judge_rows(self, family_id):
        document = self.require_spent_fixture()
        family = next((item for item in document["families"] if item["family_id"] == family_id), None)
        if family is None:
            raise KeyError(family_id)
        return list(family["judge_row_ids"])
