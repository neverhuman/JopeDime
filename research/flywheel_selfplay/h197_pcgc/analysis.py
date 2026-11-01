"""H197 equal-family A analysis; inference is paired by family across seeds."""

import importlib.util
from pathlib import Path
import sys

import numpy as np

from design import ALL_ARMS, BOOT_SEED, DRAWS

ROOT = Path(__file__).resolve().parents[3]


def family_matrix(values, family_ids):
    values = np.asarray(values, dtype=np.float64)
    family_ids = np.asarray(family_ids, dtype=object)
    if values.shape != family_ids.shape or not np.isfinite(values).all():
        raise ValueError("finite context values and matching family IDs required")
    ordered = sorted(set(family_ids.tolist()))
    return ordered, np.asarray(
        [values[family_ids == family].mean() for family in ordered]
    )


def family_elo(rows, panel):
    """Paired-family H148 Elo with vectorized 10k family bootstrap draws."""
    path = ROOT / "research/flywheel_selfplay/h177_pooled_selector/analysis.py"
    name = "h197_h177_analysis"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    h177 = sys.modules[name]
    labels = [r["family_id"] for r in rows]
    families = sorted(set(labels))
    if len(families) != 205:
        raise ValueError("Elo requires all 205 families")
    wins = np.zeros((len(families), panel.shape[2], panel.shape[2]), dtype=np.float64)
    for seed in range(3):
        wins += h177.family_wins(panel[seed], labels) / 3.0
    point = h177.bt_ratings(wins.sum(axis=0))
    ix = np.random.default_rng(BOOT_SEED).integers(
        0, len(families), (DRAWS, len(families))
    )
    samples = h177.bt_ratings(wins[ix].sum(axis=1))
    return {
        arm: {
            "elo": float(point[i]),
            "lift_vs_R882": float(point[i] - point[0]),
            "lift_ci95": np.quantile(
                samples[:, i] - samples[:, 0], [0.025, 0.975]
            ).tolist(),
        }
        for i, arm in enumerate(ALL_ARMS)
    }


def summarize(rows, panel):
    panel = np.asarray(panel, dtype=np.float64)
    if panel.shape != (3, len(rows), len(ALL_ARMS)) or not np.isfinite(panel).all():
        raise ValueError("complete H197 A panel required")
    family_ids = np.asarray([r["family_id"] for r in rows], dtype=object)
    ordered = sorted(set(family_ids.tolist()))
    if len(ordered) != 205:
        raise ValueError("pooled A must cover all 205 TRAIN families")
    fam = np.stack(
        [
            family_matrix(panel[s, :, a], family_ids)[1]
            for s in range(3)
            for a in range(len(ALL_ARMS))
        ]
    )
    fam = fam.reshape(3, len(ALL_ARMS), len(ordered)).transpose(0, 2, 1)
    rng = np.random.default_rng(BOOT_SEED)
    ix = rng.integers(0, len(ordered), (DRAWS, len(ordered)))

    def contrast(left, right):
        delta = fam[:, :, left] - fam[:, :, right]
        per_seed = delta.mean(axis=1)
        boot = np.median(delta[:, ix].mean(axis=2), axis=0)
        avg_boot = delta.mean(axis=0)[ix].mean(axis=1)
        return {
            "per_seed": per_seed.tolist(),
            "median_over_seeds": float(np.median(per_seed)),
            "median_bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "median_bootstrap_one_sided_p": float(
                (1 + np.count_nonzero(boot <= 0)) / (DRAWS + 1)
            ),
            "seed_average": float(per_seed.mean()),
            "seed_average_ci95": np.quantile(avg_boot, [0.025, 0.975]).tolist(),
            "positive_seed_count": int(np.count_nonzero(per_seed > 0)),
            "family_deltas_seed_average": delta.mean(axis=0).tolist(),
        }

    contrasts = {
        "PCGC40_vs_R882": contrast(2, 0),
        "PCGC40_vs_CTRL40": contrast(2, 1),
        "PCGC40_vs_PCGC_SHUF40": contrast(2, 4),
    }
    primary = contrasts["PCGC40_vs_R882"]
    primary["holm_multiplicity"] = 1
    primary["holm_p"] = primary["median_bootstrap_one_sided_p"]
    harm_deltas = np.asarray(primary["family_deltas_seed_average"])
    harm_rate = float(np.mean(harm_deltas < 0))
    strictly_beats_shuf = contrasts["PCGC40_vs_PCGC_SHUF40"]["median_over_seeds"] > 0
    advance = bool(
        primary["median_bootstrap_ci95"][0] > 0
        and primary["holm_p"] < 0.05
        and harm_rate <= 0.10
        and strictly_beats_shuf
    )
    elo = family_elo(rows, panel)
    return {
        "contexts": len(rows),
        "families": len(ordered),
        "utility_family_equal": {
            arm: [float(fam[s, :, i].mean()) for s in range(3)]
            for i, arm in enumerate(ALL_ARMS)
        },
        "contrasts": contrasts,
        "harm_rate": {
            "definition": "fraction of 205 equal-family PCGC40-minus-R882 deltas below zero after averaging the three seeds",
            "rate": harm_rate,
        },
        "pcgc_strictly_beats_shuffled_placebo": strictly_beats_shuf,
        "Elo": elo,
        "minus2_count_per_seed": {
            arm: [int(np.count_nonzero(panel[s, :, i] == -2.0)) for s in range(3)]
            for i, arm in enumerate(ALL_ARMS)
        },
        "verdict": "ADVANCE" if advance else "NO_ADVANCE",
        "serving_promotion": False,
        "primary": "PCGC40_vs_R882; median of three seed-specific equal-family means, paired family bootstrap, Holm m=1",
        "bootstrap_draws": DRAWS,
        "bootstrap_seed": BOOT_SEED,
    }
