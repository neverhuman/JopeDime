"""Paired whole-family bootstrap for admitted H75 development point panels."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import score_panel

DRAWS = 10000
SEED = 20261031
ALPHA = 0.05 / 7
CONTRASTS = {
    "legacy": [("D", "C"), ("D", "A"), ("replay_D", "replay_B")],
    "confirmed_U8": [("D", "B"), ("D", "C"), ("D", "G"), ("D", "R")],
}


def _family_pair_totals(point, family_by_context, family_ids):
    """Each original family contributes exactly one game per contender pair."""
    panel = point["contenders"]
    positions = {family: i for i, family in enumerate(family_ids)}
    counts = {family: sum(f == family for f in family_by_context.values()) for family in family_ids}
    totals = np.zeros((len(family_ids), len(panel), len(panel)), dtype=np.float64)
    for cid, scores in point["per_context_seed_mean_utility"].items():
        values = np.array([scores[policy] for policy in panel], dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            differences = values[:, None] - values[None, :]
        wins = np.where(differences > 1e-4, 1.0, np.where(differences < -1e-4, 0.0, 0.5))
        np.fill_diagonal(wins, 0.0)
        family = family_by_context[cid]
        totals[positions[family]] += wins / counts[family]
    return totals


def _elo_samples(family_pairs, indices, anchor_index):
    """Batched canonical 400-step MM updates; no invented bootstrap family IDs."""
    n_families, n_policies, _ = family_pairs.shape
    multiplicities = np.zeros((len(indices), n_families), dtype=np.float64)
    np.add.at(multiplicities, (np.arange(len(indices))[:, None], indices), 1.0)
    pair_wins = (multiplicities @ family_pairs.reshape(n_families, -1)).reshape(
        len(indices), n_policies, n_policies
    )
    wins = pair_wins.sum(axis=2) + 0.01 * (n_policies - 1)
    gamma = np.ones((len(indices), n_policies), dtype=np.float64)
    diagonal = np.arange(n_policies)
    for _ in range(400):
        inverse = 1.0 / np.maximum(gamma[:, :, None] + gamma[:, None, :], 1e-12)
        inverse[:, diagonal, diagonal] = 0.0
        denominator = (n_families + 2.0 * 0.01) * inverse.sum(axis=2)
        updated = np.clip(wins / denominator, 1e-6, 1e6)
        gamma = updated / np.maximum(updated[:, anchor_index, None], 1e-12)
    # Match the canonical helper's two-decimal public ratings for each draw.
    return np.round(1500.0 + 400.0 * np.log10(np.maximum(gamma, 1e-9)), 2)


def family_bootstrap(packet, reference, *, draws=DRAWS, seed=SEED):
    """Return point metrics plus descriptive intervals; never an admission decision.

    The caller must load the actual pinned helper with score_panel.load_reference.
    Manifest provenance and selection-before-confirmation remain caller obligations.
    """
    score_panel.require(type(draws) is int and 1 <= draws <= DRAWS, "draws must be an integer in 1..10000")
    score_panel.require(type(seed) is int and 0 <= seed < 2**32, "seed must be an unsigned 32-bit integer")
    point = score_panel.score(packet, reference)
    max_families = 205 if point["panel"] == "legacy" else 20
    score_panel.require(point["n_families"] <= max_families, "panel exceeds preregistered family bound")
    settings = {
        "status": "INCOMPLETE",
        "required_draws": DRAWS,
        "executed_draws": 0,
        "seed": seed,
        "numpy_version": np.__version__,
        "quantile_method": "linear",
        "unit": "whole family after within-context seed averaging",
        "one_sided_alpha": ALPHA,
        "admission_decision": None,
        "interpretation": "Descriptive conditional development intervals; not untouched-family confirmation or a seven-contrast admission decision.",
    }
    point["bootstrap"] = settings
    if point["status"] != "COMPLETE_POINT_DIAGNOSTICS":
        return point
    families = sorted(point["per_family_mean_utility"])
    family_by_context = {cell["context_sha256"]: cell["family_id"] for cell in packet["manifest"]["roster"]}
    indices = np.random.default_rng(seed).integers(
        0, len(families), size=(draws, len(families)), dtype=np.int32
    )
    index_bytes = indices.astype("<i4", copy=False).tobytes()
    roster_bytes = json.dumps(families, separators=(",", ":")).encode()
    pairs = _family_pair_totals(point, family_by_context, families)
    ratings = _elo_samples(pairs, indices, point["contenders"].index("A"))
    settings.update(
        status="COMPLETE_DESCRIPTIVE_BOOTSTRAP" if draws == DRAWS and seed == SEED else "SMOKE_ONLY",
        executed_draws=draws,
        n_families=len(families),
        family_ids=families,
        family_order_sha256=hashlib.sha256(roster_bytes).hexdigest(),
        shared_indices_sha256=hashlib.sha256(index_bytes).hexdigest(),
        elo_source_sha256=point["elo_source_sha256"],
        elo_interval95={
            policy: np.quantile(ratings[:, i], [0.025, 0.975], method="linear").tolist()
            for i, policy in enumerate(point["contenders"])
        },
        contrasts={},
    )
    for left, right in CONTRASTS[point["panel"]]:
        deltas = np.array(
            [
                score_panel.finite(
                    point["per_family_mean_utility"][family][left]
                    - point["per_family_mean_utility"][family][right]
                )
                for family in families
            ],
            dtype=np.float64,
        )
        sampled = deltas[indices]  # The same whole-family draws for every contrast.
        means, medians = sampled.mean(axis=1), np.median(sampled, axis=1)
        settings["contrasts"][left + "-" + right] = {
            "mean_paired_family_delta": float(deltas.mean()),
            "median_paired_family_delta": float(np.median(deltas)),
            "fraction_families_delta_below_minus_0_01": float(np.mean(deltas < -0.01)),
            "mean_lcb_adjusted": float(np.quantile(means, ALPHA, method="linear")),
            "median_interval95": np.quantile(medians, [0.025, 0.975], method="linear").tolist(),
        }
    # The existing scope string remains explicit about unverified provenance/admission.
    point["scope"] = (
        "Point metrics and descriptive paired-family intervals only. Caller roster/selection/utility provenance and scientific admission remain unverified; r882/A continuity is conditional on prior exposure."
    )
    return point
