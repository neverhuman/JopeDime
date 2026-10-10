"""Family/source crosswalk for Lane B.

Spec §7.3 A5: overlap means quarantine or overlap_unknown. Never claim unseen
unless a disclosure list is present and disjoint from the family's source ids.
A missing Kumo pretraining disclosure is overlap_unknown, not a clean bill.
"""

from rlfw.arena.game_spec import require_sha


def classify_family(family):
    """Return quarantine, overlap_unknown, or no_overlap. Does not say unseen."""
    source_ids = family.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids or len(set(source_ids)) != len(source_ids):
        return "overlap_unknown"
    if any(not isinstance(item, str) or not item for item in source_ids):
        return "overlap_unknown"
    disclosure = family.get("pretraining_disclosure")
    if not isinstance(disclosure, dict):
        return "overlap_unknown"
    try:
        require_sha(disclosure.get("sha256"), "pretraining disclosure")
    except ValueError:
        return "overlap_unknown"
    disclosed = disclosure.get("source_ids")
    if not isinstance(disclosed, list) or any(not isinstance(item, str) for item in disclosed):
        return "overlap_unknown"
    if set(source_ids) & set(disclosed):
        return "quarantine"
    return "no_overlap"


def unseen_word(status):
    """The word 'unseen' is allowed only after an explicit no_overlap result."""
    if status != "no_overlap":
        raise ValueError("overlap means quarantine or overlap_unknown; never claim unseen")
    return "unseen"


def roster_status(statuses):
    if not statuses:
        return "overlap_unknown"
    if any(status == "quarantine" for status in statuses):
        return "quarantine"
    if any(status != "no_overlap" for status in statuses):
        return "overlap_unknown"
    return "no_overlap"
