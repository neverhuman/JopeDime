"""Independent folds, immutable full B lock, median-seed family inference."""

import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from parents import h177, h191
from treatment import admission

SEEDS = (196101, 196102, 196103)
FOLD_SEEDS = dict(zip(SEEDS, (196141, 196142, 196143)))
BOOT_SEED, DRAWS = 196150, 10000
ARMS = ("R882", "TTA_NOISE16", "TTA_PERM8", "NOISE1")
h191.SEEDS, h191.FOLD_SEEDS = SEEDS, FOLD_SEEDS
partition, fold_manifest = h191.partition, h191.fold_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jaccards(manifest):
    result = {}
    for a, b in itertools.combinations(map(str, SEEDS), 2):
        matrix = [
            [
                len(set(x) & set(y)) / len(set(x) | set(y))
                for y in manifest[b]["families"]
            ]
            for x in manifest[a]["families"]
        ]
        result[a + "-" + b] = dict(
            matrix=matrix, diagonal_mean=float(np.trace(matrix) / 5)
        )
    return result


class StageGuard:
    def __init__(self, fold_sha, adapter_sha):
        self.path = self.digest = None
        self.fold_sha, self.adapter_sha = fold_sha, adapter_sha

    def lock(self, path):
        if self.path is not None:
            raise RuntimeError("already locked")
        p = Path(path)
        data = json.loads(p.read_text())
        if (
            data["wave"] != "H196"
            or not data["locked_before_a_and_treatment_val"]
            or data["sigma"] != 0.05
            or data["fold_manifest_sha256"] != self.fold_sha
            or data["adapter_sha256"] != self.adapter_sha
            or set(data["cells"]) != {f"s{s}-f{k}" for s in SEEDS for k in range(5)}
            or data["b_panel_sha256"] != sha(p.parent / "B-panel.npz")
        ):
            raise RuntimeError("incomplete/wrong B lock")
        for key, cell in data["cells"].items():
            scale = np.asarray(cell["scale"])
            means = np.asarray(cell["b_family_means"])
            groups = [
                set(cell[key]) for key in ("fit_families", "b_families", "a_families")
            ]
            if (
                key != f"s{cell['seed']}-f{cell['outer']}"
                or scale.shape != (10, 32)
                or not np.isfinite(scale).all()
                or (scale < 0).any()
                or means.shape != (4,)
                or not np.isfinite(means).all()
                or list(map(len, groups)) != [123, 41, 41]
                or any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
                or not cell["noise_receipt"]["version_id"]
            ):
                raise RuntimeError("invalid B cell")
        for seed in SEEDS:
            expected = np.median(
                [data["cells"][f"s{seed}-f{k}"]["scale"] for k in range(5)], axis=0
            )
            if not np.array_equal(expected, data["val_scales"][str(seed)]):
                raise RuntimeError("Val scale not locked FIT median")
        self.path, self.digest = p, sha(p)

    def require(self):
        if self.path is None or sha(self.path) != self.digest:
            raise RuntimeError("A/Val requires immutable full B lock")
        data = json.loads(self.path.read_text())
        if data["b_panel_sha256"] != sha(self.path.parent / "B-panel.npz"):
            raise RuntimeError("B panel mutation after lock")
        return data


def summarize(rows, panel):
    values = np.asarray(panel, np.float64)
    if values.shape != (3, len(rows), 4) or not np.isfinite(values).all():
        raise ValueError("complete finite 3-seed 4-arm panel required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([h177.family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    contrasts = {}
    for col, arm in enumerate(ARMS[1:], 1):
        delta = fm[:, :, col] - fm[:, :, 0]
        per_seed = delta.mean(axis=1)
        boot = np.median(delta[:, ix].mean(axis=2), axis=0)
        contrasts[arm] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            harm_rate=float((delta.mean(axis=0) < 0).mean()),
            harm_rate_by_seed=(delta < 0).mean(axis=1).tolist(),
            harm_below_minus_001=float((delta.mean(axis=0) < -0.01).mean()),
            role="primary" if col == 1 else "descriptive",
        )
    primary = contrasts["TTA_NOISE16"]
    primary["holm_p_m1"] = primary["p_one_sided"]
    wins = np.stack([h177.family_wins(v, families) for v in values])
    elo = h177.bt_ratings(wins.sum(axis=1))
    ei = np.random.default_rng(BOOT_SEED + 1).integers(0, n, (1000, n))
    boot_elo = np.stack([h177.bt_ratings(w[ei].sum(axis=1)) for w in wins])
    return dict(
        contexts=len(rows),
        families=n,
        contrasts=contrasts,
        family_utility_by_seed=fm.mean(axis=1).tolist(),
        context_utility_by_seed=values.mean(axis=1).tolist(),
        elo_by_seed=elo.tolist(),
        median_elo=np.median(elo, axis=0).tolist(),
        elo_lift_by_seed=(elo - 1500).tolist(),
        elo_ci95=np.quantile(
            np.median(boot_elo, axis=0), [0.025, 0.975], axis=0
        ).T.tolist(),
        exact_minus2_by_seed=(values == -2.0).sum(axis=1).tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
        bootstrap_seed=BOOT_SEED,
        harm_definition="fraction of families with seed-mean delta strictly below zero",
    )


def verdict(summary):
    p = summary["contrasts"]["TTA_NOISE16"]
    advance = admission(p["ci95"], p["holm_p_m1"], p["harm_rate"])
    return dict(
        verdict="ADVANCE" if advance else "NO_ADVANCE",
        disposition="CANDIDATE; ENG decides adoption"
        if advance
        else "reject treatment",
        serving_promotion=False,
        holm_multiplicity=1,
    )
