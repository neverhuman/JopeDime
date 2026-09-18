"""Content-addressed real-triplet copy plan. Excludes synthetics and nested extras."""
from __future__ import annotations

import re
from pathlib import Path

TASKS = ("regression", "binary", "multi_label", "timeseries")
TRIPLET = ("meta.json", "train.csv", "test.csv")
DEFECTIVE = "3c288e95bcdaf2fa"
EXPANSION_V9 = (
    "expansion_v9_manifest.json",
    "c698c06a14f2297c0225c45fd9d5faf1b7cb31f4116ddffb32b464d3b56c96b6",
    "424d4d2f843ab8434e53398a91ab63d41d206e6c3aa4f3e5f959ab4f53eadaf4",
)
NESTED = ("reports", "predictions", "hyperion", ".vs", "vectors")


def is_triplet(path):
    if (not isinstance(path, str) or path.startswith('/') or '\\' in path or '\0' in path
            or any(part in ('', '.', '..') for part in path.split('/'))):
        return False
    parts = Path(path).parts
    return (
        len(parts) == 3
        and parts[0] in TASKS
        and parts[2] in TRIPLET
        and DEFECTIVE not in parts[1]
    )


def is_excluded(path):
    text = Path(path).as_posix()
    if any(marker in text for marker in EXPANSION_V9):
        return True
    if any(part.startswith("synth_") or part.startswith("synthetic") for part in Path(path).parts):
        return True
    return any(marker in text for marker in NESTED)


def plan_copy(local_only_paths, remote_hashes=None, local_sizes=None):
    """Copy only missing real triplets whose content hash is absent on the node."""
    remote_hashes = set(remote_hashes or ())
    local_sizes = local_sizes or {}
    selected, skipped_hash, skipped_exclude, skipped_non_triplet = [], [], [], []
    objects = []
    bytes_out = 0
    if len(set(local_only_paths)) != len(local_only_paths):
        raise ValueError('duplicate logical paths')
    for path in local_only_paths:
        if is_excluded(path):
            skipped_exclude.append(path)
            continue
        if not is_triplet(path):
            skipped_non_triplet.append(path)
            continue
        row = local_sizes.get(path, {})
        digest = row.get("sha256")
        if (not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)
                or type(row.get('size')) is not int or row['size'] < 0
                or type(row.get('mode')) is not int or not 0 <= row['mode'] <= 0o777
                or row.get('classification') != 'retained_non_synthetic_triplet'):
            raise ValueError('complete original object record required: ' + path)
        if digest and digest in remote_hashes:
            skipped_hash.append(path)
            continue
        selected.append(path)
        objects.append(dict(path=path, **{key: row[key] for key in ('sha256', 'size', 'mode', 'classification')}))
        bytes_out += row['size']
    return dict(
        schema="parallel.raw_triplet_parity.v1",
        n_selected=len(selected),
        n_skipped_present_hash=len(skipped_hash),
        n_skipped_exclude=len(skipped_exclude),
        n_skipped_non_triplet=len(skipped_non_triplet),
        selected_bytes=bytes_out,
        synthetics_copied=0,
        expansion_v9_preserved=True,
        defective_excluded=True,
        execute=False,
        selected=selected,
        objects=objects,
    )


def execute_allowed(plan, capacity, certificates, fence_up):
    """Legacy Boolean helper has no execution authority.

    Transfer bootstraps parity. It requires sealed source objects, scope and
    live capacity/custody checks; it never requires deployment or parity first.
    """
    return False, ["legacy declarations cannot authorize transfer; use the verified workspace transfer operation"]
