"""Capacity arithmetic and fail-closed reporting of unverified sync boundaries.

Presence-only v1 declarations are retained as diagnostics. They have no transfer
or activation authority; the indexed manifests and original verifiers own that.
"""
from __future__ import annotations

BOUNDARIES = ("source", "durable_data", "recovery", "release",
              "deployment_conformance", "replication_lag")
GIB = 1024 ** 3
GROWTH_RESERVE_GIB = 250
CAPACITY_COMPONENTS = ("missing_unique_bytes", "snapshots_bytes", "staging_bytes",
                       "materialization_bytes", "expected_growth_bytes", "metadata_bytes")


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer byte count required: " + name)
    return value


def capacity_required(missing_unique_bytes=None, staging_bytes=None,
                      filesystem_bytes=None, growth_gib=GROWTH_RESERVE_GIB,
                      snapshots_bytes=None, materialization_bytes=None,
                      expected_growth_bytes=None, metadata_bytes=None):
    values = dict(missing_unique_bytes=missing_unique_bytes, snapshots_bytes=snapshots_bytes,
                  staging_bytes=staging_bytes, materialization_bytes=materialization_bytes,
                  expected_growth_bytes=expected_growth_bytes, metadata_bytes=metadata_bytes)
    _bytes(filesystem_bytes, "filesystem_bytes")
    _bytes(growth_gib, "growth_gib")
    for name, value in values.items():
        _bytes(value, name)
    reserve = max(250 * GIB, growth_gib * GIB, (filesystem_bytes + 9) // 10)
    return sum(values.values()) + reserve


def certify_capacity(free_bytes, filesystem_bytes, missing_unique_bytes=None, **requirements):
    _bytes(free_bytes, "free_bytes")
    _bytes(filesystem_bytes, "filesystem_bytes")
    if free_bytes > filesystem_bytes:
        raise ValueError("free space exceeds filesystem size")
    components = dict(missing_unique_bytes=missing_unique_bytes,
                      **{k: requirements.get(k) for k in CAPACITY_COMPONENTS[1:]})
    unknown = [key for key, value in components.items() if value is None]
    for key, value in components.items():
        if value is not None:
            _bytes(value, key)
    required = None if unknown else capacity_required(filesystem_bytes=filesystem_bytes, **components)
    return dict(schema="parallel.sync_capacity.v2", free_bytes=free_bytes,
                filesystem_bytes=filesystem_bytes, components=components,
                unknown_components=unknown, required_bytes=required,
                required_reserve_bytes=max(250 * GIB, (filesystem_bytes + 9) // 10),
                shortfall_bytes=None if required is None else max(0, required - free_bytes),
                usable_before_backfill=None if required is None else free_bytes - required,
                backfill_allowed=required is not None and free_bytes >= required,
                scope="capacity arithmetic only; execution remeasures the destination",
                rate_limit_mib_s=(64, 8))


def certify_boundary(name, present=None, sha256=None, lag_seconds=None, reason=None):
    if name not in BOUNDARIES:
        raise ValueError("unknown sync boundary")
    # A digest or a directory cannot establish complete inventories, independent
    # object verification, recovery, original host acknowledgments or lag events.
    return dict(name=name, certified=False, declared_present=present,
                sha256=sha256, lag_seconds=lag_seconds,
                reason=reason or "original complete boundary evidence has not been verified")


def certify(observations):
    rows = []
    for name in BOUNDARIES:
        row = observations.get(name) or {}
        rows.append(certify_boundary(name, row.get("present"), row.get("sha256"),
                                     row.get("lag_seconds"), row.get("reason")))
    return dict(schema="parallel.sync_certificates.v2", certified=False,
                missing_roots=[row["name"] for row in rows], boundaries=rows,
                one_synced_bit=False, evidence_class="unverified_declarations",
                transfer_prerequisite=False, activation_allowed=False)
