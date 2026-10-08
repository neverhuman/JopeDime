"""Completed replacement metadata is immutable; stale trees require a new build destination."""
import json
from pathlib import Path


def done_context_matches(old, recomputed):
    """Identity/files alone do not bind the admitted and refused fold set."""
    return old.get("epoch0_replaced") == recomputed.get("epoch0_replaced") and isinstance(recomputed.get("epoch0_replaced"), dict)


def dataset_done_matches(shard, record):
    """The cheap dataset shortcut must check the same completed replacement set as the write boundary.

    An inconsistent record returns False so the builder recomputes its gates. That write boundary refuses stale .done
    metadata; this function never repairs or overwrites it.
    """
    try:
        contexts = record.get("contexts") or []
        rep = record.get("epoch0_replaced") or {}
        if not contexts or rep.get("new_contexts") != len(contexts):
            return False
        folds = [int(c["fold"]) for c in contexts]
        ids = [c["view_id"] for c in contexts]
        if len(set(folds)) != len(folds) or len(set(ids)) != len(ids):
            return False
        refusals = record.get("refused_contexts") or []
        if refusals:
            refused = sorted(({"fold": int(c["fold"]), "reason": c["refused"]} for c in refusals), key=lambda c: c["fold"])
            gone = [c["fold"] for c in refused]
            if (set(folds) & set(gone) or len(set(gone)) != len(gone) or not all(c["reason"] for c in refused)
                    or rep.get("admitted_folds") != sorted(folds) or rep.get("refused_folds") != refused
                    or rep.get("built_folds") != sorted(folds + gone)):
                return False
        elif any(k in rep for k in ("built_folds", "admitted_folds", "refused_folds")):
            return False
        for c in contexts:
            vid = c["view_id"]
            if not isinstance(vid, str) or vid in (".", "..") or Path(vid).name != vid:
                return False
            d = shard / vid
            if not (d / ".done").is_file():
                return False
            old = json.loads((d / "context.json").read_bytes())
            if (not done_context_matches(old, {"epoch0_replaced": rep}) or old.get("fold") != c["fold"]
                    or old.get("view_id") != vid or old.get("dataset_hash") != record.get("dataset_hash")
                    or old.get("context_sha256") != c.get("context_sha256")):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def recovery_detail(view_id):
    return (f"{view_id}: completed epoch0_replaced differs from recomputed membership; preserve this tree and rebuild "
            "the same pinned plan/inputs into a new output tree (see deploy/dl-trainonly/RESUME_RECOVERY.md); "
            "never edit context.json or remove .done to repair a completed context")
