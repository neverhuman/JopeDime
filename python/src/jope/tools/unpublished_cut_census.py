"""D01 evidence: recount metadata unpublished cuts. Does not admit identities."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "parallel.unpublished_cut_census.v2"


def recount(census_path):
    path = Path(census_path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "parallel.redteam_metadata_audit.v1":
        raise ValueError("unsupported metadata census")
    if data.get("protected_values_opened") is not False:
        raise ValueError("census must explicitly exclude protected value access")
    rows = data.get("expansion_rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("expansion rows must be a list of metadata objects")
    complete = [r for r in rows if r.get("component_metadata_complete") is True]
    registered = [r for r in rows if r.get("registered_eligible") is True]
    metadata_ok = [r for r in rows if r.get("metadata_eligible_addition") is True]
    regression_complete = [r for r in complete if r.get("task") == "regression"]
    retained = [r for r in rows if r.get("retained_index") is True]
    missing = Counter(name for row in rows for name in row.get("missing_components", []))
    dispositions = Counter(
        disposition.get("partition", "unknown")
        for row in rows for disposition in row.get("historical_registry_dispositions", [])
    )
    # Metadata completeness is not native membership or decision-receipt admission.
    execution_queue = 0
    return dict(
        schema=SCHEMA,
        source=dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()),
        protected_values_opened=False,
        expansion_rows=len(rows),
        component_metadata_complete=len(complete),
        component_metadata_complete_regression=len(regression_complete),
        component_metadata_incomplete=sum(r.get("component_metadata_complete") is False for r in rows),
        component_metadata_unknown=sum(type(r.get("component_metadata_complete")) is not bool for r in rows),
        missing_or_incompatible_components=dict(sorted(missing.items())),
        component_bytes_verified=None,
        component_bytes_verification="unavailable: source is a metadata-only census",
        active_membership=data.get("active"),
        retained_index_rows=len(retained),
        outside_retained_indexes=sum(r.get("retained_index") is False for r in rows),
        retained_membership_unknown=sum(type(r.get("retained_index")) is not bool for r in rows),
        component_complete_without_retained_assembly=sum(r.get("retained_index") is False for r in complete),
        historical_registry_dispositions=dict(sorted(dispositions.items())),
        registered_eligible=len(registered),
        metadata_eligible_additions=len(metadata_ok),
        native_membership_verified=None,
        native_membership_verification="unavailable: no native membership receipts joined",
        actual_eligible_additions=None,
        component_groups_outside_retained_indexes=data.get("component_groups_outside_retained_indexes"),
        component_complete_groups=data.get("component_complete_groups"),
        claimed_4048_certified=False,
        execution_queue=execution_queue,
        empty_queue=execution_queue == 0,
        reason=(f"{len(registered)} metadata-declared registered rows; "
                f"{len(metadata_ok)} metadata-declared additions; "
                "no native component, membership or decision receipts joined"),
        source_errors=data.get("errors", []),
        live_index_written=False,
    )
