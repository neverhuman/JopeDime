"""B-only nested choices; one imported H177 pooled-A primary decision."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np
from parents import h177
from treatment import GRID, GRID_ARMS

SEEDS = (191101, 191102, 191103)
FOLD_SEEDS = dict(zip(SEEDS, (191141, 191142, 191143)))
BOOT_SEED, DRAWS = 191150, 10000
ARMS = GRID_ARMS[1:]  # Consumed by the unchanged H186 scorer.
READOUT_ARMS = ("R882", "KXSEL", "KX2.5", "KX3.0", "KX4.0")
H177_DECISION_RULE = h177.H177_DECISION_RULE
pooled_b_select = h177.pooled_b_select
verify_pooled_b_coverage = h177.verify_pooled_b_coverage
resolve_h177_verdict = h177.resolve_h177_verdict


@contextmanager
def bindings(seed=SEEDS[0]):
    if seed not in FOLD_SEEDS:
        raise ValueError("unknown fold reshuffle seed")
    names = ("ARMS", "PRESPECIFIED_CANDIDATE_STEP", "FOLD_SEED")
    previous = {key: getattr(h177, key) for key in names}
    h177.ARMS = ("KXSEL",)
    h177.PRESPECIFIED_CANDIDATE_STEP = {"KXSEL": 1}
    h177.FOLD_SEED = FOLD_SEEDS[seed]
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(h177, key, value)


def partition(rows, seed, outer):
    with bindings(seed):
        return h177.partition(rows, outer)


def fold_manifest(rows):
    result = {}
    for seed in SEEDS:
        with bindings(seed):
            result[str(seed)] = dict(
                fold_seed=FOLD_SEEDS[seed],
                families=h177.folds(rows),
                coverage=verify_pooled_b_coverage(rows),
            )
    if len({json.dumps(v["families"]) for v in result.values()}) != 3:
        raise RuntimeError("three distinct family reshuffles required")
    return result


def choose(rows, utility):
    """Exactly B-local equal-family argmax; grid order implements smaller-KX ties."""
    values = np.asarray(utility, dtype=np.float64)
    if values.shape != (len(rows), len(GRID)) or not np.isfinite(values).all():
        raise ValueError("finite B grid required")
    family_means = h177.family_matrix(values, [r["family_id"] for r in rows])[1]
    means = family_means.mean(axis=0)
    index = int(np.argmax(means))
    return dict(grid_index=index, multiplier=GRID[index], family_means=means.tolist())


def val_choice(indices):
    if len(indices) != 5 or any(
        type(i) is not int or i not in range(6) for i in indices
    ):
        raise ValueError("five valid locked choices required")
    return int(np.argmax(np.bincount(indices, minlength=6)))


def project(values, selected):
    if type(selected) is not int or selected not in range(6):
        raise ValueError("invalid selected grid index")
    return np.asarray(values)[..., [0, selected, 3, 4, 5]]


def lock_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SelectionGuard:
    """H186 scorer guard interface with a nested-selection lock, not an A result."""

    def __init__(self, folds_sha, initial_hash):
        self.folds_sha, self.initial_hash = folds_sha, initial_hash
        self.b_locked_path = None
        self.b_reads = 0
        self.digest = None

    def mark_b_read(self):
        if self.b_locked_path is not None:
            raise RuntimeError("B read after selection lock")
        self.b_reads += 1

    def mark_b_locked(self, path):
        if self.b_reads != 15 or self.b_locked_path is not None:
            raise RuntimeError("all fifteen B cells must finish before lock")
        value = json.loads(Path(path).read_text())
        expected = {f"s{s}-f{k}" for s in SEEDS for k in range(5)}
        if (
            value["wave"] != "H191"
            or value["fold_manifest_sha256"] != self.folds_sha
            or value["adapter_sha256"] != self.initial_hash
            or value["b_panel_sha256"] != lock_hash(Path(path).parent / "B-panel.npz")
            or set(value["cells"]) != expected
            or value["locked_before_a_and_treatment_val"] is not True
        ):
            raise RuntimeError("incomplete or wrong-identity selection lock")
        for key, cell in value["cells"].items():
            means = np.asarray(cell["family_means"], dtype=np.float64)
            if means.shape != (6,) or not np.isfinite(means).all():
                raise RuntimeError("invalid B means")
            index = int(np.argmax(means))
            if (
                type(cell["grid_index"]) is not int
                or cell["grid_index"] != index
                or cell["multiplier"] != GRID[index]
            ):
                raise RuntimeError("B argmax or smaller-KX tie rule violated")
        for seed in SEEDS:
            indices = [value["cells"][f"s{seed}-f{k}"]["grid_index"] for k in range(5)]
            if value["val_grid_index"][str(seed)] != val_choice(indices):
                raise RuntimeError("DL-Val policy was not derived from B choices")
        self.b_locked_path, self.digest = Path(path), lock_hash(path)

    def require_b_locked(self):
        if self.b_locked_path is None or lock_hash(self.b_locked_path) != self.digest:
            raise RuntimeError("A/Val requires the complete immutable B choice lock")


def select_a(rows, panel):
    """Imported H177 function names retain B wording; input here is pooled A."""
    with bindings():
        decision = H177_DECISION_RULE.select(
            rows,
            panel[:, :, 0],
            {"KXSEL": panel[:, :, 1]},
            seed=BOOT_SEED,
        )["KXSEL"]
    decision["h177_literal_metadata_m"] = decision.pop("holm_multiplicity")
    decision["holm_multiplicity"] = 1
    decision["selection_stage"] = "pooled A, single prespecified nested policy"
    decision["candidate_version"] = decision.pop("prespecified_candidate_step")
    decision["selected_candidate_version"] = decision.pop("selected_step")
    decision["unadjusted_selected_candidate_version"] = decision.pop(
        "selected_step_unadjusted"
    )
    decision.pop("holm_p_treatments3")
    decision.pop("admitted_treatments3")
    decision["optimizer_updates"] = 0
    return decision


def summarize(rows, panel):
    values = np.asarray(panel, dtype=np.float64)
    if values.shape != (3, len(rows), 5) or not np.isfinite(values).all():
        raise ValueError("complete 3-seed five-policy readout required")
    families = [r["family_id"] for r in rows]
    fm = np.stack([h177.family_matrix(v, families)[1] for v in values])
    n = fm.shape[1]
    ix = np.random.default_rng(BOOT_SEED).integers(0, n, (DRAWS, n))
    contrasts = {}
    for col, arm in enumerate(READOUT_ARMS[1:], 1):
        d = fm[:, :, col] - fm[:, :, 0]
        per_seed = d.mean(axis=1)
        boot = np.median(d[:, ix].mean(axis=2), axis=0)
        contrasts[arm] = dict(
            per_seed_gain=per_seed.tolist(),
            median_gain=float(np.median(per_seed)),
            ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
            positive_seeds=int((per_seed > 0).sum()),
            p_one_sided=float((1 + (boot <= 0).sum()) / (DRAWS + 1)),
            role="primary" if arm == "KXSEL" else "descriptive dose only",
        )
    contrasts["KXSEL"]["holm_p_m1"] = contrasts["KXSEL"]["p_one_sided"]
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
        elo_ci95=np.quantile(
            np.median(boot_elo, axis=0), [0.025, 0.975], axis=0
        ).T.tolist(),
        exact_minus2_by_seed=(values == -2.0).sum(axis=1).tolist(),
        utility_bootstrap_draws=DRAWS,
        elo_bootstrap_draws=1000,
    )


def verdict(decision, summary, receipt, analysis_sha):
    primary = summary["contrasts"]["KXSEL"]
    confirmed = bool(
        decision["admitted"]
        and primary["median_gain"] > 0.001 + 1e-12
        and primary["ci95"][0] > 0
    )
    result = resolve_h177_verdict(confirmed, receipt, analysis_sha)
    return dict(**result, confirmed=confirmed, serving_promotion=False)
