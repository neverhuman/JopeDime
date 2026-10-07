"""Load-time admission by P3 holdout assignment (owner ruling HOLDOUT-INPUTS-140, ENG:watcher delegated 2026-10-06T01:38Z).

Every trainer or data loader of the DL lane calls `Admission.check(view_id=..., key=...)` before it uses a context or an
object. Admitted: only the decided P3 roles train_fit and train_nonpool (training) and dev (selection only). Refused: sealed_excluded,
heldout, and ANY other or missing assignment (unknown family, 'no_family', 'roleless_bridge', ...: fail closed, owner ENG:watcher
07:35Z), and any object key listed in the production denylist. Fail closed: the denylist must exist and hash to the pinned
sha256, or every check refuses. The denylist is the additive manifest written by tools/v2_holdout_denylist.py.
"""
import hashlib
import json
from pathlib import Path

REFUSED_ASSIGNMENTS = frozenset({"sealed_excluded", "heldout"})
ADMITTED_ASSIGNMENTS = frozenset({"train_fit", "train_nonpool", "dev"})


class AdmissionRefused(RuntimeError):
    pass


class Admission:
    def __init__(self, denylist_path, denylist_sha256: str, purpose: str = "train"):
        self.purpose = purpose
        self.ok, self.reason = False, None
        try:
            b = Path(denylist_path).read_bytes()
        except OSError as e:
            self.reason = f"denylist unreadable: {type(e).__name__}"
            return
        if hashlib.sha256(b).hexdigest() != denylist_sha256:
            self.reason = "denylist sha256 mismatch"
            return
        doc = json.loads(b)
        self.views = {r["view_id"]: r["assignment"] for r in doc["contexts"]}
        self.keys = set(doc["object_keys"])
        self.assign = doc.get("assignment_by_view") or {}
        self.ok = True

    def check(self, view_id: str, key: str | None = None, assignment: str | None = None) -> str:
        if not self.ok:
            raise AdmissionRefused(f"fail closed: {self.reason}")
        if view_id in self.views:
            raise AdmissionRefused(f"{view_id}: {self.views[view_id]} (denylist)")
        if key is not None and key in self.keys:
            raise AdmissionRefused(f"{key}: object of a refused context (denylist)")
        a = assignment if assignment is not None else self.assign.get(view_id)
        if a is None:
            raise AdmissionRefused(f"{view_id}: no holdout assignment known (fail closed)")
        if a in REFUSED_ASSIGNMENTS:
            raise AdmissionRefused(f"{view_id}: {a}")
        if a not in ADMITTED_ASSIGNMENTS:
            raise AdmissionRefused(f"{view_id}: unknown holdout assignment {a!r} (fail closed)")
        if a == "dev" and self.purpose == "train":
            raise AdmissionRefused(f"{view_id}: dev is for selection only")
        return a
