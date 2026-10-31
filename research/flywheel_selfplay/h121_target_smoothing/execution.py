"""H121 adapter over the reviewed H97 cell machinery; all runtime hooks are explicit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from runner import (
    ARMS,
    DOPE_SHA256,
    EPSILON,
    FOLD_SEED,
    R882_SHA256,
    SEEDS,
    family_folds,
    smooth_targets,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
H97_PATH = HERE.parent / "h97_reg_ema" / "runner.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reviewed runtime module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_runtime():
    """Load H97's guarded cell runner and bind H121 arms, fold and target transform."""
    h97 = load_module("h121_h97_runtime", H97_PATH)
    h97.HERE = HERE
    h97.ROOT = ROOT
    h97.ARMS = ARMS
    h97.SEEDS = SEEDS
    state = {
        "folds": None,
        "train_rows": None,
        "score_rows": None,
        "active_arm": None,
        "h48": None,
    }
    h55 = h97.h55
    original_metadata = h55.input_metadata
    original_load_rows = h55.load_rows
    original_load = h97.load
    original_loss = h97.h59.loss_by_arm
    original_atomic_json = h97.atomic_json
    original_evaluate = None

    def input_metadata(input_dir, h48):
        selected, all_train, _dev_rows = original_metadata(input_dir, h48)
        state["h48"] = h48
        fold = family_folds(all_train)
        state["folds"] = fold
        train_set = set(fold["train_families"])
        score_set = set(fold["score_families"])
        train_rows = [r for r in all_train if r["family_id"] in train_set]
        score_rows = [r for r in all_train if r["family_id"] in score_set]
        if not train_rows or not score_rows:
            raise RuntimeError("H121 fold produced an empty training or scoring roster")
        state["train_rows"] = train_rows
        state["score_rows"] = score_rows
        return selected, train_rows, score_rows

    def recorded_folds(_rows, _h48):
        if state["folds"] is None:
            raise RuntimeError("H121 family fold must be fixed before loading pools")
        return [set(fold) for fold in state["folds"]["folds"]]

    def h121_loss(arm, prediction, targets):
        epsilon = EPSILON[state["active_arm"]]
        if epsilon:
            targets = smooth_targets(targets.detach(), epsilon)
        return original_loss(arm, prediction, targets)

    def load_h121(name, path):
        module = original_load(name, path)
        if name == "h97_h49":
            score_rows = state["score_rows"]
            if not score_rows:
                raise RuntimeError("H121 score fold was not captured")

            def load_fresh_score_fold(input_dir):
                pools, tables = original_load_rows(input_dir, score_rows, state["h48"])
                return score_rows, pools, tables

            module.load_validation_packet = load_fresh_score_fold
        elif name == "h97_h61":
            nonlocal original_evaluate
            original_evaluate = module.evaluate_pool_predictions
            h61 = module

            def evaluate_with_all_candidate_regret(
                cids, pools, predictions, *, eval_mode=h61.EVAL_MODE_UNREV
            ):
                report = original_evaluate(
                    cids, pools, predictions, eval_mode=eval_mode
                )
                by_family: dict[str, list[float]] = {}
                for cid in sorted(cids):
                    pool = pools[cid]
                    true_u = h61.clip_context_utility(pool["fitness"], pool["gate"])
                    predicted_u = -np.asarray(predictions[cid], dtype=np.float64)
                    if (
                        true_u.shape != predicted_u.shape
                        or not np.isfinite(predicted_u).all()
                    ):
                        raise RuntimeError(
                            "H121 continuous-regret vector mismatch/nonfinite"
                        )
                    regret = float(np.mean(float(np.max(true_u)) - predicted_u))
                    by_family.setdefault(str(pool["rec"]["family_id"]), []).append(
                        regret
                    )
                family_regret = {
                    f: float(np.mean(v)) for f, v in sorted(by_family.items())
                }
                report["per_family_mean_regret_all_candidates"] = family_regret
                report["mean_regret_all_candidates"] = float(
                    np.mean(list(family_regret.values()))
                )
                return report

            module.evaluate_pool_predictions = evaluate_with_all_candidate_regret
        return module

    h55.input_metadata = input_metadata
    h55.family_folds = recorded_folds
    h97.load = load_h121
    h97.h59.loss_by_arm = h121_loss

    def atomic_json(path, value):
        if Path(path) == state.get("active_out"):
            value["wave"] = "H121"
            value["epsilon"] = EPSILON[value["active_arm"]]
            value["fold_seed"] = FOLD_SEED
            value["fold_assignment_sha256"] = state["folds"]["assignment_sha256"]
            value["scoring_families"] = state["folds"]["score_families"]
        return original_atomic_json(path, value)

    h97.atomic_json = atomic_json
    return h97, state, original_loss, original_atomic_json


def run_cell(args):
    h97, state, original_loss, original_atomic = build_runtime()
    state["active_arm"] = args.arm
    state["active_out"] = Path(args.out)
    try:
        result = h97.run_cell(args)
    finally:
        h97.h59.loss_by_arm = original_loss
        h97.atomic_json = original_atomic
    if result.get("wave") != "H121" or result.get("updates") != 40:
        raise RuntimeError("H121 cell failed wave/update identity guard")
    if (
        result.get("dope_before") != DOPE_SHA256
        or result.get("dope_after") != DOPE_SHA256
    ):
        raise RuntimeError("H121 frozen DOPE identity mismatch")
    if result.get("dev", {}).get("per_family_mean_regret_all_candidates") is None:
        raise RuntimeError("H121 continuous all-candidate regret missing")
    return result


def summarize(cells, draws=10000):
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    indexed = {(c.get("arm"), c.get("seed")): c for c in cells}
    if len(cells) != len(expected) or set(indexed) != expected:
        raise ValueError("H121 requires all nine unique arm/seed cells")
    families = sorted(indexed["CONTROL", SEEDS[0]]["r882_dev"]["per_family"])
    if len(families) < 5:
        raise ValueError("H121 scoring fold must contain at least five families")
    for cell in cells:
        if cell.get("wave") != "H121" or cell.get("updates") != 40:
            raise ValueError("H121 result identity/update mismatch")
        if (
            cell.get("dope_before") != DOPE_SHA256
            or cell.get("dope_after") != DOPE_SHA256
        ):
            raise ValueError("H121 result DOPE guard mismatch")
        if cell.get("checkpoint_sha256") != R882_SHA256:
            raise ValueError("H121 checkpoint identity mismatch")
        for name in ("dev",):
            report = cell[name]
            if set(report["per_family"]) != set(families) or set(
                report["per_family_mean_regret_all_candidates"]
            ) != set(families):
                raise ValueError("H121 family roster/regret mismatch")
    identities = set()
    for seed in SEEDS:
        paired_initials = {indexed[arm, seed]["initial_adapter_sha256"] for arm in ARMS}
        if len(paired_initials) != 1:
            raise ValueError(
                "H121 paired arms did not start from identical fresh r882 adapter state"
            )
    for cell in cells:
        identities.add(
            (
                cell.get("prereg_sha256"),
                cell.get("fold_assignment_sha256"),
                str(cell.get("folds")),
            )
        )
    if len(identities) != 1:
        raise ValueError("H121 cells disagree on preregistration/fold/source identity")
    ref = np.asarray(
        [
            [indexed["CONTROL", s]["r882_dev"]["per_family"][f] for f in families]
            for s in SEEDS
        ]
    )
    samples = np.random.default_rng(121045).integers(
        0, len(families), size=(draws, len(families))
    )
    rng_boot = {}
    for arm in ARMS:
        candidate = np.asarray(
            [[indexed[arm, s]["dev"]["per_family"][f] for f in families] for s in SEEDS]
        )
        rng_boot[arm] = candidate - ref
    result = {
        "schema": "jope.h121.metrics/v1",
        "complete": True,
        "seeds": list(SEEDS),
        "arms": {},
        "family_count": len(families),
        "bootstrap_draws": draws,
        "bootstrap_seed": 121045,
        "reference_elo": 1500,
        "nominal_treatment_comparisons": 2,
        "holm": {
            "method": "one-sided family-cluster bootstrap p-values for margin 0.001; Holm step-down over two treatments"
        },
        "auc": None,
    }
    treatment_stats = []
    for arm in ARMS:
        gains = rng_boot[arm]
        seed_means = gains.mean(axis=1)
        boot = np.median(gains[:, samples].mean(axis=2), axis=0)
        point = float(np.median(seed_means))
        lcb = float(np.quantile(boot, 0.05))
        p_margin = float((np.count_nonzero(boot <= 0.001) + 1) / (draws + 1))
        harms = float(np.mean(gains.mean(axis=0) < -0.01))
        win_vectors = (gains > 0).astype(float) + 0.5 * (gains == 0)
        win_rates = win_vectors.mean(axis=1)
        elo_seed = [elo_rating(float(p)) for p in win_rates]
        elo_seed_ci = [
            [
                elo_rating(float(p))
                for p in np.quantile(
                    win_vectors[i, samples].mean(axis=1), [0.025, 0.975]
                )
            ]
            for i in range(len(SEEDS))
        ]
        median_elo_boot = np.median(win_vectors[:, samples].mean(axis=2), axis=0)
        pooled_elo_boot = win_vectors[:, samples].mean(axis=(0, 2))
        regret = [indexed[arm, s]["dev"]["mean_regret_all_candidates"] for s in SEEDS]
        ref_regret = [
            indexed["CONTROL", s]["r882_dev"]["mean_regret_all_candidates"]
            for s in SEEDS
        ]
        record = {
            "per_seed_family_delta_vs_r882": seed_means.tolist(),
            "median_delta": point,
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": lcb,
            "bootstrap_p_margin_0_001": p_margin,
            "harm_fraction_delta_below_minus_0_01": harms,
            "family_grouped_elo_vs_r882_per_seed": dict(
                zip(SEEDS, elo_seed, strict=True)
            ),
            "family_grouped_elo_ci95_vs_r882_per_seed": dict(
                zip(SEEDS, elo_seed_ci, strict=True)
            ),
            "median_elo_vs_r882": elo_rating(float(np.median(win_rates))),
            "median_elo_ci95_vs_r882": [
                elo_rating(float(p))
                for p in np.quantile(median_elo_boot, [0.025, 0.975])
            ],
            "pooled_elo_vs_r882": elo_rating(float(win_vectors.mean())),
            "pooled_elo_ci95_vs_r882": [
                elo_rating(float(p))
                for p in np.quantile(pooled_elo_boot, [0.025, 0.975])
            ],
            "mean_regret_all_candidates_per_seed": regret,
            "mean_regret_all_candidates_vs_r882_per_seed": (
                np.asarray(ref_regret) - np.asarray(regret)
            ).tolist(),
            "admitted_pre_holm": arm != "CONTROL"
            and point > 0
            and lcb > 0.001
            and harms <= 0.10,
        }
        result["arms"][arm] = record
        if arm != "CONTROL":
            treatment_stats.append((arm, p_margin))
    ordered = sorted(treatment_stats, key=lambda x: x[1])
    adjusted = {}
    running = 0.0
    for rank, (arm, pval) in enumerate(ordered):
        value = min(1.0, (len(ordered) - rank) * pval)
        running = max(running, value)
        adjusted[arm] = running
    result["holm_adjusted_p"] = adjusted
    for arm, _pval in treatment_stats:
        result["arms"][arm]["admitted"] = bool(
            result["arms"][arm]["admitted_pre_holm"] and adjusted[arm] <= 0.05
        )
    return result


def h121_elo_delta(win_probability):
    if win_probability <= 0:
        return "-Infinity"
    if win_probability >= 1:
        return "Infinity"
    return 400.0 * np.log10(win_probability / (1.0 - win_probability))


def elo_rating(win_probability):
    delta = h121_elo_delta(win_probability)
    return delta if isinstance(delta, str) else 1500.0 + delta
