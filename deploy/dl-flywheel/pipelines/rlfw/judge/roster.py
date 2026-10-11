"""Family-clean partition proposal. This module does not create or spend a seal.

Roles follow the four boundaries in the 2026-10-11 expert review §8.1.
A family is one parent lineage plus every retained context, cut, and view of
that parent. Adoption, the alpha budget, and any real seal are Jepson decisions.
"""

from __future__ import annotations

ROLES = ("meta_training", "development", "confirmation", "sealed_final")


def proof_disjoint(sealed_families, closure):
    """Return whether every sealed family is absent from the training closure.

    `sealed_families` maps family id to a record with `context_ids`, `view_ids`,
    and `dataset_hashes`. `closure` has those three sets plus `family_ids`.
    A family that shares any of those identifiers with the closure fails.
    """
    if not isinstance(sealed_families, dict) or not sealed_families:
        raise ValueError("a sealed proposal needs at least one family")
    required = ("context_ids", "view_ids", "dataset_hashes", "family_ids")
    missing = [name for name in required if name not in closure]
    if missing:
        raise ValueError("closure is missing " + ",".join(missing))
    failures = []
    for family_id, record in sorted(sealed_families.items()):
        reasons = []
        if family_id in closure["family_ids"]:
            reasons.append("family_id")
        for kind in ("context_ids", "view_ids", "dataset_hashes"):
            overlap = set(record.get(kind, ())) & set(closure[kind])
            if overlap:
                reasons.append(kind)
        members = set(record.get("member_roles", ()) or ())
        if members and members != {"sealed_final"}:
            reasons.append("role_split")
        if reasons:
            failures.append({"family_id": family_id, "reasons": reasons})
    return {
        "passed": not failures,
        "families": len(sealed_families),
        "failures": failures,
        "seal_created": False,
        "seal_spent": False,
        "adopted": False,
    }


def proposal_document(partition, proof, sources):
    """A proposal document. `adopted` stays false. No seal field is a spend."""
    if proof.get("seal_spent") or proof.get("seal_created") or proof.get("adopted"):
        raise ValueError("a proposal cannot record a seal spend or adoption")
    unknown = set(partition) - set(ROLES)
    if unknown:
        raise ValueError("unknown roles: " + ",".join(sorted(unknown)))
    return {
        "schema": "jope.rlfw-sealed-roster-proposal/v1",
        "adopted": False,
        "alpha_budget": "unspent_jepson_decision",
        "seal_created": False,
        "seal_spent": False,
        "qualifying": False,
        "roles": list(ROLES),
        "partition": partition,
        "proof": proof,
        "sources": sources,
        "encoder_pretraining_closure": "unmeasured",
    }
