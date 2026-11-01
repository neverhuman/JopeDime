"""H202 fixed family folds, paired sampling and B-only common-step selection."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

SEEDS = (202101, 202102, 202103)
FOLD_SEEDS = (202141, 202142, 202143)
BOOT_SEED = 202150
ARMS = ("HNEG40", "CTRL40", "HNEG_SHUF")
ALL_ARMS = ("R882", *ARMS)
STEPS = (0, 10, 20, 30, 40)
UPDATES = 40
DRAWS = 10_000


def stream(*parts):
    return (
        int.from_bytes(
            hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
        )
        & 0x7FFFFFFF
    )


def folds(family_ids, seed):
    names = sorted(set(family_ids))
    if len(names) != 205 or seed not in FOLD_SEEDS:
        raise ValueError("fresh fivefold partition of 205 TRAIN families required")
    order = np.random.default_rng(seed).permutation(205)
    return [sorted(names[j] for j in group) for group in np.array_split(order, 5)]


def partition(rows, groups, outer):
    a, b = set(groups[outer]), set(groups[(outer + 1) % 5])
    fit = set().union(
        *(set(g) for j, g in enumerate(groups) if j not in (outer, (outer + 1) % 5))
    )
    if (len(fit), len(a), len(b)) != (123, 41, 41) or fit & (a | b) or a & b:
        raise ValueError("FIT123/B41/A41 must be disjoint")
    return tuple([r for r in rows if r["family_id"] in role] for role in (fit, b, a))


def family_probabilities(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["family_id"], []).append(r["context_sha256"])
    return {c: 1 / (len(groups) * len(cids)) for cids in groups.values() for c in cids}


def plan(rows, seed, outer):
    groups = {}
    for r in rows:
        groups.setdefault(r["family_id"], []).append(r["context_sha256"])
    names = sorted(groups)
    groups = {f: sorted(cids) for f, cids in groups.items()}
    result = []
    for step in range(1, 41):
        key = stream(seed, outer, step, "H202-family4-v1")
        rng = np.random.default_rng(key)
        cids, families = [], []
        for _ in range(4):
            f = names[int(rng.integers(len(names)))]
            families.append(f)
            cids.append(groups[f][int(rng.integers(len(groups[f])))])
        result.append(
            dict(step=step, stream=key, context_ids=cids, family_ids=families)
        )
    return result


def choose_steps(rows, panel):
    """One common step per arm; pooled B seed-average equal-family gain; early ties."""
    panel = np.asarray(panel)
    if panel.shape != (3, len(rows), 4, 5) or not np.isfinite(panel).all():
        raise ValueError("all seeds/arms/steps/families must finish pooled B")
    labels = np.asarray([r["family_id"] for r in rows])
    fm = np.stack(
        [panel[:, labels == f].mean(axis=1) for f in sorted(set(labels))], axis=1
    )
    selected, gains = {}, {}
    for j, arm in enumerate(ARMS, 1):
        delta = fm[:, :, j, :] - fm[:, :, 0, :]
        value = delta.mean(axis=(0, 1))
        gains[arm] = value.tolist()
        selected[arm] = STEPS[int(np.argmax(value))]
    return dict(
        selected_steps=selected,
        B_equal_family_seed_average_gains=gains,
        rule="maximize pooled-B seed-average family gain; earliest exact tie; step0 included",
    )


def save_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + ".writing")
        with temporary.open("w") as f:
            json.dump(value, f, separators=(",", ":"), allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)


def write_lock(path, payload):
    save_json(path, payload, exclusive=True)
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_A(path, expected):
    path = Path(path)
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("candidate A/Val requires the unchanged durable B lock")
    value = json.loads(path.read_text())
    if not value.get("locked_before_A") or set(value["selected_steps"]) != set(ARMS):
        raise RuntimeError("invalid B selection lock")
    return value["selected_steps"]
