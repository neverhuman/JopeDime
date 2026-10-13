"""Per-context L6 cost rows carried by a G-ABL receipt.

G03 requires every scored context to appear as a row with context_id, arm,
epoch, updates, seed, and l6_cost. This module checks that shape. It does not
compute L6, and it does not promote a checkpoint.
"""

from __future__ import annotations

import math

L6_ROW_FIELDS = ("context_id", "arm", "epoch", "updates", "seed", "l6_cost")


def require_l6_cost_row(row):
    """Refuse a row that omits or mistypes any G03 field."""
    if not isinstance(row, dict):
        raise ValueError("l6 row must be an object")
    missing = [field for field in L6_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError("l6 row missing " + ",".join(missing))
    context_id = row["context_id"]
    arm = row["arm"]
    epoch = row["epoch"]
    updates = row["updates"]
    seed = row["seed"]
    cost = row["l6_cost"]
    if not isinstance(context_id, str) or not context_id:
        raise ValueError("context_id must be a non-empty string")
    if not isinstance(arm, str) or not arm:
        raise ValueError("arm must be a non-empty string")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative int")
    if type(updates) is not int or updates < 0:
        raise ValueError("updates must be a non-negative int")
    if type(seed) is not int:
        raise ValueError("seed must be an int")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost):
        raise ValueError("l6_cost must be finite")
    return row


def require_g03_receipt(receipt):
    """Every receipt carries the per-context rows, not only a mean."""
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    if list(receipt.get("l6_row_fields") or ()) != list(L6_ROW_FIELDS):
        raise ValueError("receipt l6_row_fields must be the G03 field list")
    rows = receipt.get("l6_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("receipt missing l6_rows")
    for row in rows:
        require_l6_cost_row(row)
    return receipt
