"""Outcome-free H214 pool selection and preregistered family estimands."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import re

import numpy as np

H212_ANALYSIS_SHA = "e09189fabe68d8933fadd6302243bb70313c7c046ec8c0b3c202777a96138e04"


def load_upstream_analysis():
    path = Path(__file__).resolve().parent.parent / "h212_pool_width/analysis.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != H212_ANALYSIS_SHA:
        raise ValueError("H212 analysis pin differs")
    return importlib.import_module(
        "research.flywheel_selfplay.h212_pool_width.analysis"
    )


_upstream = load_upstream_analysis()
first_argmin_utility = _upstream.first_argmin_utility
holm = _upstream.holm

SEEDS = (214101, 214102, 214103)
ARMS = ("NATIVE16", "FPS16")
WIDTH = 16
BOOT_SEED = 214150
BOOT_DRAWS = 10_000
MIN_GAIN = 0.001546


def stream_seed(seed: int, context_sha256: str) -> int:
    if seed not in SEEDS or re.fullmatch("[0-9a-f]{64}", context_sha256) is None:
        raise ValueError("registered H214 seed and context SHA256 required")
    raw = hashlib.sha256(f"H214:propose:{seed}:{context_sha256}".encode()).digest()
    return int.from_bytes(raw[:8], "little") & 0x7FFFFFFF


def unique_programs(parent: dict, proposals: list[dict]) -> list[dict]:
    result, seen = [], set()
    for row in [parent, *proposals]:
        identity = hashlib.sha256(row["dsl"].encode()).hexdigest()
        if row["program_sha256"] != identity:
            raise ValueError("program SHA256 differs from DSL")
        if identity not in seen:
            seen.add(identity)
            result.append(row)
    if len(result) < WIDTH:
        raise ValueError("fewer than 16 unique programs; no refill or reduced width")
    return result


def native_pool(parent: dict, proposals: list[dict]) -> list[dict]:
    return unique_programs(parent, proposals)[:WIDTH]


def fps_pool(parent: dict, proposals: list[dict], proj16: dict) -> list[dict]:
    rows = unique_programs(parent, proposals)
    matrix = np.asarray(
        [proj16[row["program_sha256"]] for row in rows], dtype=np.float64
    )
    if matrix.shape != (len(rows), 16) or not np.isfinite(matrix).all():
        raise ValueError("one finite native proj16 vector per unique program required")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    vectors = matrix / np.where(norms > 0.0, norms, 1.0)
    selected = [0]
    distance = np.sum((vectors - vectors[0]) ** 2, axis=1)
    while len(selected) < WIDTH:
        distance[selected] = -np.inf
        index = int(np.argmax(distance))  # first generator position breaks exact ties
        selected.append(index)
        distance = np.minimum(distance, np.sum((vectors - vectors[index]) ** 2, axis=1))
    return [rows[index] for index in selected]


def diversity(pool: list[dict], proj16: dict) -> dict:
    ids = [row["program_sha256"] for row in pool]
    if len(ids) != WIDTH or len(set(ids)) != WIDTH:
        raise ValueError("diversity requires exactly 16 unique scored programs")
    x = np.asarray([proj16[key] for key in ids], dtype=np.float64)
    if x.shape != (WIDTH, 16) or not np.isfinite(x).all():
        raise ValueError("finite native proj16 vectors required")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms > 0.0, norms, 1.0)
    distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
    return {
        "unique_programs": WIDTH,
        "mean_pairwise_distance": float(distances[np.triu_indices(WIDTH, 1)].mean()),
    }


def family_gain_stats(gain, families: list[str]) -> dict:
    values = np.asarray(gain, dtype=np.float64)
    labels = np.asarray(families, dtype=str)
    order = sorted(set(families))
    if (
        values.shape != (3, 91)
        or labels.shape != (91,)
        or len(order) != 19
        or not np.isfinite(values).all()
    ):
        raise ValueError("complete three-seed DL-Val91 / 19-family panel required")
    means = np.stack(
        [values[:, labels == family].mean(axis=1) for family in order], axis=1
    )
    indices = np.random.default_rng(BOOT_SEED).integers(0, 19, (BOOT_DRAWS, 19))
    draws = np.median(means[:, indices].mean(axis=2), axis=0)
    return {
        "seed_equal_family_mean_gain": means.mean(axis=1).tolist(),
        "median_gain": float(np.median(means.mean(axis=1))),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "p_margin": float((1 + np.sum(draws <= MIN_GAIN)) / (BOOT_DRAWS + 1)),
    }


def decide(stats: dict) -> dict:
    if set(stats) != {"FPS16"}:
        return {"verdict": "INCOMPLETE", "reason": "registered FPS16 contrast required"}
    for row in stats.values():
        numbers = [row["median_gain"], *row["ci95"], row["p_margin"]]
        if not np.isfinite(numbers).all() or row["ci95"][0] > row["ci95"][1]:
            raise ValueError("finite ordered statistics required")
    adjusted = holm({arm: row["p_margin"] for arm, row in stats.items()})
    accepted = [
        arm
        for arm, row in stats.items()
        if row["median_gain"] > 0.0
        and row["ci95"][0] > MIN_GAIN
        and adjusted[arm] < 0.05
    ]
    return {
        "verdict": "ADVANCE" if accepted else "NO_ADVANCE",
        "accepted_arms": sorted(accepted),
        "holm_m": 1,
        "holm_p": adjusted,
    }
