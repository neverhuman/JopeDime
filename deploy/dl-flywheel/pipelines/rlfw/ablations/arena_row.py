"""Turn one arena slot into a measurement dict. This module does not open files."""

from __future__ import annotations

import math

CATALOG_SIZE = 6640
UNIFORM_MU = 1.0 / CATALOG_SIZE
PROPENSITY_SOURCE = "arm_definition_uniform_trie"


def measurement_from_slot(payload, evidence, tokens, role, subrole):
    """Join slot payload, evidence q, and slate tokens. Mismatched q_grid is refused."""
    if payload.get("q_grid") != evidence.get("q_grid"):
        raise ValueError("slot q_grid does not match evidence q_grid")
    if payload.get("program_sha256") != evidence.get("program_sha256"):
        raise ValueError("slot program does not match evidence program")
    if payload.get("outcome") != evidence.get("outcome_class"):
        raise ValueError("slot outcome does not match evidence outcome_class")
    program = payload.get("program_sha256")
    return {
        "reward_kind": "verified_comparable_q",
        "q": evidence.get("q"),
        "q_grid": evidence.get("q_grid"),
        "q_hat": None,
        "predicted_q": None,
        "fitness": None,
        "outcome": payload.get("outcome"),
        "qualified": payload.get("qualified"),
        "parity_verified": evidence.get("parity_verified"),
        "closure_verified": evidence.get("closure_verified"),
        "parity_max_normalized_error": evidence.get("parity_max_normalized_error"),
        "program_sha256": program,
        "context_id": payload.get("context_id"),
        "lineage_id": payload.get("lineage_id"),
        "role": role,
        "subrole": subrole,
        "arm": payload.get("arm"),
        "seed": payload.get("seed"),
        "slot": payload.get("slot"),
        "tokens": None if tokens is None else tuple(tokens),
        "evaluator_sha256": evidence.get("evaluator_sha256"),
        "protocol": evidence.get("protocol"),
    }


def uniform_mu(stored_propensity, catalog_size=CATALOG_SIZE):
    """Behaviour propensity of uniform_trie is 1/|catalog|, from the arm definition.

    The stored slate field is checked against that value and is not the source.
    """
    if type(catalog_size) is not int or catalog_size < 1:
        raise ValueError("catalog_size must be a positive int")
    expected = 1.0 / catalog_size
    if stored_propensity is None or not isinstance(stored_propensity, (int, float)):
        raise ValueError("uniform_trie slate propensity is missing")
    if not math.isfinite(stored_propensity) or abs(stored_propensity - expected) > 1e-15:
        raise ValueError("stored propensity does not match 1/catalog_size")
    return {
        "mu": expected,
        "propensity_source": PROPENSITY_SOURCE,
        "catalog_size": catalog_size,
    }
