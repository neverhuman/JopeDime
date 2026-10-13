"""J8 matched-feedback control definitions.

Restart-only, offline replay, data volume, and fresh feedback are different
interventions (CLOVER Phase 3). This module checks a declared arm against
those definitions. It does not train, launch, or promote a checkpoint.

L6 is a validation loss. Lower is better. The closed scratch readouts sit
above the frozen v1 full-validation cost, so they are worse, and a
non-qualifying readout is not an improvement either way.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

FROZEN_V1_FULL_VAL_L6 = 0.897327
ACTOR_LR = 3.75e-5
SCHEDULED_UPDATES = 704
ARM_SCHEMA = "jope.j8-control-arm/v1"
ARM_NAMES = (
    "frozen",
    "restart_only",
    "offline_replay",
    "data_volume",
    "fresh_feedback",
)
CONFIG_DIR = Path(__file__).with_name("configs")

# Closed NON-QUAL scratch. Receipt files stay outside git. These are the
# identities already posted on COORD; this module does not reread them.
CLOSED_SCRATCH = {
    "restart_only": {
        "receipt_sha256": "1c7c80e1092c903897a050a4e41e7ec9c7066f05036f82b9ce1fc1a6c53b7bc7",
        "harness_arm": "control",
        "updates": 704,
        "scheduled_updates": SCHEDULED_UPDATES,
        "stop_reason": "max_epochs",
        "full_val_l6": 0.9022044512849824,
        "full_val_contexts": 263,
        "earlier_epoch_full_val_l6": 0.8965906633509965,
        "non_qualifying": True,
        "benchmark_valid": False,
    },
    "offline_replay": {
        "receipt_sha256": "bccf7c88353c87ee25a16745054e21ffa1dde71584540ba28e89301736c0471e",
        "harness_arm": "offline_replay",
        "updates": 352,
        "scheduled_updates": SCHEDULED_UPDATES,
        "stop_reason": "kl_disqualify_all_treatment_arms",
        "full_val_l6": 0.9003862184454733,
        "full_val_contexts": 263,
        "non_qualifying": True,
        "benchmark_valid": False,
    },
}


def load_arm(name):
    """Load one declared arm config. The file is not a launch recipe."""
    if name not in ARM_NAMES:
        raise ValueError("unknown J8 arm")
    spec = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    if spec.get("schema") != ARM_SCHEMA:
        raise ValueError("arm config schema")
    if spec.get("intervention") != name:
        raise ValueError("arm config intervention does not match its file")
    return spec


def _one_weight(spec, key):
    if key not in spec:
        raise ValueError(key + " is required")
    value = spec[key]
    if isinstance(value, (list, tuple, dict)) or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(key + " must be one number")
    if not math.isfinite(value):
        raise ValueError(key + " must be finite")
    return float(value)


def _flag(spec, key):
    if type(spec.get(key)) is not bool:
        raise ValueError(key + " must be bool")
    return spec[key]


def _actor_lr(spec):
    value = _one_weight(spec, "actor_lr")
    if not math.isclose(value, ACTOR_LR, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("actor_lr must be 3.75e-5")


def classify_arm(spec):
    """Return the intervention name, or refuse a mixed or launchable spec."""
    if not isinstance(spec, dict):
        raise ValueError("arm spec must be an object")
    if spec.get("schema") not in (None, ARM_SCHEMA):
        raise ValueError("arm config schema")
    if spec.get("trie_arm") is not False:
        raise ValueError("J8 controls are not trie arms")
    if spec.get("launch_allowed") is not False:
        raise ValueError("this module does not authorize a launch")
    pi = _one_weight(spec, "lambda_pi")
    kl = _one_weight(spec, "lambda_kl")
    knn = _one_weight(spec, "lambda_knn")
    new = _flag(spec, "new_feedback")
    prior = _flag(spec, "preexisting_observations")
    matched = _flag(spec, "observation_count_matched")
    updates = spec.get("max_updates")
    if type(updates) is not int or updates < 0:
        raise ValueError("max_updates must be a non-negative int")
    named = spec.get("intervention")
    if new:
        if named not in (None, "fresh_feedback"):
            raise ValueError("new feedback is the fresh-feedback treatment")
        return "fresh_feedback"
    if matched:
        if prior is not True:
            raise ValueError("data volume matches preexisting observations")
        if named not in (None, "data_volume"):
            raise ValueError("matched observation count is data volume, not offline replay")
        if kl != 0.0 or knn != 0.0:
            raise ValueError("data volume is not a trie or kNN arm")
        return "data_volume"
    if prior:
        if pi != 0.01 or kl != 0.0 or knn != 0.0:
            raise ValueError("offline replay is lambda_pi 0.01, lambda_kl 0, lambda_knn 0")
        if updates != SCHEDULED_UPDATES:
            raise ValueError("offline replay uses the closed 704-update schedule")
        _actor_lr(spec)
        if named not in (None, "offline_replay"):
            raise ValueError("preexisting observations without matched volume are offline replay")
        return "offline_replay"
    if pi != 0.0 or kl != 0.0 or knn != 0.0:
        raise ValueError("restart-only has no feedback weight")
    if updates == 0:
        if named not in (None, "frozen"):
            raise ValueError("zero updates and no feedback is frozen")
        return "frozen"
    if updates != SCHEDULED_UPDATES:
        raise ValueError("restart-only uses the closed 704-update schedule")
    _actor_lr(spec)
    if named not in (None, "restart_only"):
        raise ValueError("scheduled optimization without feedback is restart-only")
    return "restart_only"


def direction(cost):
    """Compare one full-validation L6 with frozen v1. Never an improvement claim."""
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
        raise ValueError("l6 cost must be finite")
    if cost > FROZEN_V1_FULL_VAL_L6:
        return "worse"
    if cost < FROZEN_V1_FULL_VAL_L6:
        return "below_v1_not_a_claim"
    return "tied_not_a_claim"


def require_readout(readout):
    """A J8 scratch readout stays non-qualifying and makes no improvement claim."""
    if not isinstance(readout, dict):
        raise ValueError("readout must be an object")
    if readout.get("non_qualifying") is not True or readout.get("benchmark_valid") is not False:
        raise ValueError("readout stays non-qualifying")
    if readout.get("improvement") is True:
        raise ValueError("no improvement claim")
    label = direction(readout.get("full_val_l6"))
    if readout.get("direction") not in (None, label):
        raise ValueError("direction does not match the frozen v1 comparison")
    contexts = readout.get("full_val_contexts")
    if type(contexts) is not int or contexts != 263:
        raise ValueError("full-val context count is 263")
    return label


def require_closed_scratch(name):
    """Return one closed scratch pin after the non-qualifying checks."""
    if name not in CLOSED_SCRATCH:
        raise ValueError("no closed scratch for " + str(name))
    row = CLOSED_SCRATCH[name]
    if require_readout(row) != "worse":
        raise ValueError("closed scratch readout is above frozen v1")
    if row["updates"] > row["scheduled_updates"]:
        raise ValueError("updates exceed the closed schedule")
    earlier = row.get("earlier_epoch_full_val_l6")
    if earlier is not None:
        if direction(earlier) == "worse":
            raise ValueError("earlier epoch marker is not the above-v1 cap")
        if earlier == row["full_val_l6"]:
            raise ValueError("earlier epoch is not the closed cap")
    return row
