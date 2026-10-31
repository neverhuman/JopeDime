#!/usr/bin/env python3
"""H105 fixed shrinkage math, train-only cell and data-free preparation CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("CONTROL", "INTERP025", "INTERP050", "ANCHOR")
SEEDS = (42, 43, 44)
FOLD_SEED = 105042
BOOTSTRAP_SEED = 105000
ADAPTER_PARAMS = 29211
MARGIN = 0.001
TIE_TOL = 1e-12
PRIOR_DRIFT = 0.34640736440899245
ANCHOR_LAMBDA = 2.8867746553428666
PRIOR_SHA = "109e030f7f8f57b604ad5f2d8df576a817fca0ae482c08026398eca3338d3054"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
FUSED_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
# Filled from the frozen publication base before any H105 timing or outcome.
SOURCE_PINS = {
    "research/flywheel_selfplay/h97_reg_ema/runner.py": "4411ad39b85a2c3d568c03b70d358e40a4fdca7f4affa3315e2856f4f8660a67",
    "research/flywheel_selfplay/h76_sam_steps/runner.py": "3556fe81fe654b148d301244d6d0460bf2363b5a97db0a0b644507168e97d24f",
    "research/flywheel_selfplay/h59/runner.py": "604d02ed72a6149284792cba315124efd1ffe8cac79e5ace22613db86c11b4c3",
    "research/flywheel_selfplay/h55/runner.py": "56fd220689805f07877ae403c2f00a4b34831cc4b2919f4bd69807ff3c21f4bc",
    "research/flywheel_selfplay/h48/run.py": "756960a0e2bfdf23e75012fb347ce772fb6fdeaa4fb4fc2d4178dfaeb9eac9da",
    "research/flywheel_selfplay/h46/run.py": "645bf1470789c4f5bcc97809f703b04d9a594a21dd9d3aa8c2fd4e98810d0320",
    "research/flywheel_selfplay/h61/runner.py": "a1654add23b76e54b22afe5a7f42bf543420b12ca0d31d52b5c25c409498d037",
    "research/flywheel_selfplay/h61/run_wave.py": "b665589255015d6dcc014ef0b6dd86953d336849914fe5278361bd5d76cda426",
    "research/flywheel_selfplay/qualification_lock_fifo.py": "8e3150a7ab9b53612f99db9757f8bfa32938afa5d90c9bbc9db60c6752e79a84",
    "research/flywheel_selfplay/run_fused_closed_loop.py": "564c441a61654bfd2249d5003873a450f45abe31e15cc9fd704ffafd8b48b737",
    "research/flywheel_selfplay/automatic_promotion.py": "261d2d1865c2f50e1c7388dfd2ac404e85f599df3a5b7aaaab69415cf250d8a0",
    "research/flywheel_selfplay/dev_incumbent.py": "81ebdafa5fe1accb1da56a424bee669f0e381fee2bd5e23c751c17618079523f",
    "research/flywheel_selfplay/h53b/runner.py": "85aa59e9c4bee62805ed6c388364611b3a7cdb772a19017eeae848819c2cb318",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime():
    """Lazy source-only import; the data-free dry-run never calls this."""
    global h97, h59, h55, science
    if "h97" not in globals():
        for relative, expected in SOURCE_PINS.items():
            if sha256(ROOT / relative) != expected:
                raise RuntimeError(f"shared scientific source changed: {relative}")
        h97 = load("h105_h97", HERE.parent / "h97_reg_ema/runner.py")
        h59, h55, science = h97.h59, h97.h55, h97.science
    return h97


def safe_path(path):
    path = Path(path)
    for candidate in (path.absolute(), path.resolve()):
        if any(
            any(
                word in part.lower()
                for word in (
                    "sealed",
                    "holdout",
                    "meta-test",
                    "meta_test",
                    "flywheel-full-20261030",
                )
            )
            for part in candidate.parts
        ):
            raise RuntimeError("protected/live path rejected")
    return path


def atomic_json(path, value):
    usage = shutil.disk_usage("/")
    if usage.free < 1024**3 or 100 * (usage.total - usage.free) / usage.total >= 95:
        raise RuntimeError("unsafe filesystem / before H105 write")
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def protocol():
    return {
        "wave": "H105",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "fold_seed": FOLD_SEED,
        "fold_count": 5,
        "assessment_fold": 0,
        "assessment_families": 41,
        "train_families": 164,
        "train_contexts": None,
        "assessment_contexts": None,
        "updates": 40,
        "rounds": 20,
        "updates_per_round": 2,
        "batch": 16,
        "slate": 192,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "clip": 1.0,
        "base_loss": "H97_CONTROL_H59_SmoothL1_plus_0.1KL_temp0.2",
        "alpha": {"CONTROL": 1.0, "INTERP025": 0.25, "INTERP050": 0.5, "ANCHOR": 1.0},
        "anchor_lambda": ANCHOR_LAMBDA,
        "anchor_penalty": "lambda/2 * raw_squared_L2_to_r882",
        "prior_drift": PRIOR_DRIFT,
        "prior_metrics_sha256": PRIOR_SHA,
        "margin": MARGIN,
        "bootstrap_draws": 10000,
        "bootstrap_rng": BOOTSTRAP_SEED,
        "holm_comparisons": 4,
        "harm_loss_threshold": 0.01,
        "maximum_harm_fraction": 0.1,
        "GPU_cap_seconds": 2700,
        "predecessors": ["H99", "H100"],
        "adapter_parameters": ADAPTER_PARAMS,
        "checkpoint_sha256": R882_SHA,
        "dope_sha256": DOPE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "selected_sha256": SELECTED_SHA,
    }


def protocol_hash():
    return hashlib.sha256(
        json.dumps(
            protocol(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def adapter_inventory(state):
    if not state or sum(t.numel() for t in state.values()) != ADAPTER_PARAMS:
        raise ValueError("exactly29,211 adapter parameters required")
    if any(
        not n.startswith("adapter_10tok.") or ".anchor." in n or n.endswith(".anchor")
        for n in state
    ):
        raise ValueError("DOPE/anchor or non-adapter parameter rejected")
    if any(not torch.isfinite(t).all() for t in state.values()):
        raise ValueError("nonfinite adapter")


def interpolate(initial, fine, alpha):
    adapter_inventory(initial)
    adapter_inventory(fine)
    if set(initial) != set(fine) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("invalid interpolation inventory/alpha")
    result = {}
    for name, base in initial.items():
        value = fine[name]
        if base.shape != value.shape or base.dtype != value.dtype:
            raise ValueError("interpolation shape/dtype mismatch")
        result[name] = (
            base.clone()
            if alpha == 0
            else value.clone()
            if alpha == 1
            else base + alpha * (value - base)
        ).detach()
    return result


def anchor_penalty(current, initial):
    adapter_inventory(current)
    adapter_inventory(initial)
    if set(current) != set(initial):
        raise ValueError("anchor inventory mismatch")
    terms = []
    for name, value in current.items():
        if value.shape != initial[name].shape:
            raise ValueError("anchor shape mismatch")
        terms.append((value - initial[name].detach().to(value.device)).square().sum())
    return (ANCHOR_LAMBDA / 2) * sum(terms)


def candidate_loss(base_loss, current, initial, arm):
    if arm not in ARMS:
        raise ValueError("unregistered arm")
    return (
        base_loss + anchor_penalty(current, initial) if arm == "ANCHOR" else base_loss
    )


def family_assignment(rows):
    if not rows:
        raise ValueError("nonempty DL-Train metadata required")
    cids = []
    for row in rows:
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("source_split") not in ("train", "inner")
            or row.get("has_meta_test") is True
            or row.get("origin") not in (None, "real")
        ):
            raise ValueError("only real regression DL-Train train/inner rows allowed")
        if not isinstance(row.get("family_id"), str) or not row["family_id"]:
            raise ValueError("family identity required")
        cid = row.get("context_sha256")
        if (
            not isinstance(cid, str)
            or len(cid) != 64
            or any(c not in "0123456789abcdef" for c in cid)
        ):
            raise ValueError("context identity required")
        cids.append(cid)
    if len(set(cids)) != len(cids):
        raise ValueError("duplicate context")
    families = sorted({r["family_id"] for r in rows})
    folds = [
        sorted(x.tolist())
        for x in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), 5
        )
    ]
    if any(not x for x in folds):
        raise ValueError("five nonempty family folds required")
    result = {
        "seed": FOLD_SEED,
        "folds": folds,
        "assessment_families": folds[0],
        "train_families": sorted(set(families) - set(folds[0])),
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def attempt_scratch(root, arm, seed, attempt, *, timing=False):
    if arm not in ARMS or seed not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError("invalid registered attempt")
    p = (
        safe_path(root)
        / "scratch"
        / f"{'timing-' if timing else ''}{arm}-seed{seed}-attempt{attempt}"
    )
    if p.exists() or p.is_symlink():
        raise RuntimeError("attempt scratch already exists")
    return p


def mark_phase(args, stage, *, outcome_started=False):
    old = json.loads(args.phase.read_text())
    if old.get("attempt_token") != args.attempt_token or (
        old.get("outcome_started") and not outcome_started
    ):
        raise RuntimeError("phase token/scoring boundary mismatch")
    atomic_json(args.phase, dict(old, stage=stage, outcome_started=outcome_started))


def elo(gains, samples):
    wins = gains > 0
    ties = gains == 0
    score = wins + 0.5 * ties

    def rating(p):
        return (
            "-Infinity"
            if p == 0
            else "Infinity"
            if p == 1
            else float(1500 + 400 * np.log10(p / (1 - p)))
        )

    boot = score[:, samples].mean(axis=2)
    med = np.median(boot, axis=0)
    return {
        "per_seed": [
            {
                "seed": s,
                "wins": int(wins[i].sum()),
                "ties": int(ties[i].sum()),
                "losses": int((gains[i] < 0).sum()),
                "elo": rating(score[i].mean()),
                "ci95": [rating(q) for q in np.quantile(boot[i], [0.025, 0.975])],
            }
            for i, s in enumerate(SEEDS)
        ],
        "median_elo": rating(np.median(score.mean(axis=1))),
        "median_ci95": [rating(q) for q in np.quantile(med, [0.025, 0.975])],
    }


def seed_groups(values):
    values = np.asarray(values, dtype=float)
    if values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("three finite seed vectors required")
    result = []
    for i, seed in enumerate(SEEDS):
        for group in result:
            if np.array_equal(values[i], values[SEEDS.index(group[0])]):
                group.append(seed)
                break
        else:
            result.append([seed])
    return result


def holm_bounds(bootstraps):
    if set(bootstraps) != set(ARMS):
        raise ValueError("all four comparisons required")
    draws = len(bootstraps[ARMS[0]])
    if draws < 1 or any(
        len(x) != draws or not np.isfinite(x).all() for x in bootstraps.values()
    ):
        raise ValueError("finite common bootstrap draws required")
    tails = {
        a: float((1 + np.count_nonzero(bootstraps[a] <= MARGIN)) / (draws + 1))
        for a in ARMS
    }
    order = sorted(ARMS, key=lambda a: (tails[a], ARMS.index(a)))
    continuing = True
    adjusted = 0.0
    result = {}
    for rank, arm in enumerate(order):
        alpha = 0.05 / (4 - rank)
        lcb = float(np.quantile(bootstraps[arm], alpha))
        adjusted = max(adjusted, min(1.0, tails[arm] * (4 - rank)))
        continuing = bool(continuing and tails[arm] <= alpha and lcb > MARGIN + TIE_TOL)
        result[arm] = {
            "rank": rank + 1,
            "tail_alpha": alpha,
            "raw_tail_area": tails[arm],
            "holm_adjusted_tail_area": adjusted,
            "holm_lcb": lcb,
            "passes": continuing,
        }
    return result


def summarize(cells, draws=10000):
    if draws != 10000:
        raise RuntimeError("INCOMPLETE: fixed10000bootstrap draws required")
    indexed = {(c["arm"], c["seed"]): c for c in cells}
    if len(cells) != 12 or set(indexed) != {(a, s) for a in ARMS for s in SEEDS}:
        raise RuntimeError("INCOMPLETE: all12unique paired cells required")
    first = indexed["CONTROL", 42]
    families = sorted(first["assessment"]["per_family"])
    contexts = sorted(first["assessment"]["per_context"])
    if len(families) != 41:
        raise RuntimeError("INCOMPLETE: all41assessment families required")
    identities = set()
    for c in cells:
        if (
            c["wave"] != "H105"
            or c["timing_only"]
            or c["updates"] != 40
            or c["trainable_count"] != ADAPTER_PARAMS
            or c["dope_before"] != DOPE_SHA
            or c["dope_after"] != DOPE_SHA
            or c["checkpoint_sha256"] != R882_SHA
            or c["manifest_sha256"] != MANIFEST_SHA
            or c["selected_sha256"] != SELECTED_SHA
            or c["protocol_sha256"] != protocol_hash()
            or c["holdout_payloads_opened"] != 0
            or c["outer_rows_delivered"] != 0
            or c["DL_Val_payloads_opened"] != 0
            or c["initial_adapter_sha256"] == c["final_adapter_sha256"]
        ):
            raise RuntimeError("INCOMPLETE: cell identity/optimizer/boundary mismatch")
        if (
            sorted(c["assessment"]["per_family"]) != families
            or sorted(c["r882_assessment"]["per_family"]) != families
            or len(c["train"]["per_family"]) != 164
            or set(c["train"]["per_family"]) & set(families)
        ):
            raise RuntimeError("INCOMPLETE: family roster mismatch")
        if len(c["train"]["per_context"]) + len(
            c["assessment"]["per_context"]
        ) != 1219 or set(c["train"]["per_context"]) & set(
            c["assessment"]["per_context"]
        ):
            raise RuntimeError("INCOMPLETE: full1219context partition required")
        for packet in ("train", "assessment", "r882_assessment"):
            p = c[packet]
            fmap = p.get("family_by_context", {})
            if fmap.keys() != p["per_context"].keys() or set(fmap.values()) != set(
                p["per_family"]
            ):
                raise RuntimeError("INCOMPLETE: context/family binding missing")
            for family, value in p["per_family"].items():
                actual = np.mean(
                    [v for cid, v in p["per_context"].items() if fmap[cid] == family]
                )
                if not math.isclose(float(actual), value, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(
                        "INCOMPLETE: family aggregate does not match contexts"
                    )
            if not math.isclose(
                float(np.mean(list(p["per_family"].values()))),
                p["mean"],
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("INCOMPLETE: equal-family scalar mismatch")
            if (
                p["per_context"].keys() != first[packet]["per_context"].keys()
                or not np.isfinite(list(p["per_context"].values())).all()
                or not np.isfinite(list(p["per_family"].values())).all()
                or not math.isfinite(p["mean"])
            ):
                raise RuntimeError("INCOMPLETE: context roster/nonfinite score")
        if c["r882_assessment"] != indexed["CONTROL", c["seed"]]["r882_assessment"]:
            raise RuntimeError("INCOMPLETE: r882 reference differs")
        if (
            c["arm"] in ("INTERP025", "INTERP050")
            and c["fine_tuned_adapter_sha256"]
            != indexed["CONTROL", c["seed"]]["fine_tuned_adapter_sha256"]
        ):
            raise RuntimeError("INCOMPLETE: interpolation fine-tune not paired")
        if not math.isfinite(c["adapter_l2"]) or c["adapter_l2"] <= 0:
            raise RuntimeError("INCOMPLETE: adapter distance invalid")
        if c["source_hashes"] != SOURCE_PINS:
            raise RuntimeError("INCOMPLETE: scientific source hashes differ")
        split = c["split"]
        all_families = sorted(set(c["train"]["per_family"]) | set(families))
        expected = [
            sorted(x.tolist())
            for x in np.array_split(
                np.random.default_rng(FOLD_SEED).permutation(all_families), 5
            )
        ]
        if (
            split.get("seed") != FOLD_SEED
            or split.get("folds") != expected
            or split.get("assessment_families") != families
            or split.get("train_families") != sorted(c["train"]["per_family"])
        ):
            raise RuntimeError("INCOMPLETE: fresh family split mismatch")
        identities.add(
            json.dumps(
                [
                    c["source_commit"],
                    c["prereg_sha256"],
                    c["split"],
                    c["initial_adapter_sha256"],
                    c["source_hashes"],
                ],
                sort_keys=True,
            )
        )
    if len(identities) != 1:
        raise RuntimeError("INCOMPLETE: mixed source/prereg/split")
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 41, size=(draws, 41))
    arms = {}
    boots = {}
    for arm in ARMS:
        rows = [indexed[arm, s] for s in SEEDS]
        utility = np.asarray(
            [[c["assessment"]["per_family"][f] for f in families] for c in rows]
        )
        reference = np.asarray(
            [[c["r882_assessment"]["per_family"][f] for f in families] for c in rows]
        )
        gain = utility - reference
        per_seed = gain.mean(axis=1)
        boot = np.median(gain[:, samples].mean(axis=2), axis=0)
        boots[arm] = boot
        harmed = gain.mean(axis=0) < -0.01
        groups = {
            name: seed_groups(values)
            for name, values in [
                (
                    "context",
                    [
                        [c["assessment"]["per_context"][cid] for cid in contexts]
                        for c in rows
                    ],
                ),
                ("family", utility),
                ("scalar", utility.mean(axis=1)),
            ]
        }
        arms[arm] = {
            "per_seed_gain_vs_r882": per_seed.tolist(),
            "median_gain": float(np.median(per_seed)),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "lcb95": float(np.quantile(boot, 0.05)),
            "harm_fraction": float(harmed.mean()),
            "harmed_families": [
                f for f, bad in zip(families, harmed, strict=True) if bad
            ],
            "per_seed_harm_fraction": (gain < -0.01).mean(axis=1).tolist(),
            "any_negative_family_fraction": float((gain.mean(axis=0) < 0).mean()),
            "elo_vs_r882": elo(gain, samples),
            "seed_utility_groups": groups,
            "identical_seed_utility_flag": any(
                len(g) > 1 for grouping in groups.values() for g in grouping
            ),
            "independent_seed_confirmations_claimed": 0,
            "adapter_l2_per_seed": [c["adapter_l2"] for c in rows],
            "seeds": [
                {
                    "seed": c["seed"],
                    "train_reg": c["train"]["mean"],
                    "assessment_reg": c["assessment"]["mean"],
                    "gap": c["train"]["mean"] - c["assessment"]["mean"],
                }
                for c in rows
            ],
        }
    holm = holm_bounds(boots)
    for arm, a in arms.items():
        a["admitted"] = bool(
            holm[arm]["passes"] and a["median_gain"] >= 0 and a["harm_fraction"] <= 0.1
        )
    return {
        "schema": "jope.h105.metrics/v1",
        "complete": True,
        "reference": "r882",
        "families": families,
        "bootstrap_draws": draws,
        "bootstrap_rng": BOOTSTRAP_SEED,
        "holm": holm,
        "arms": arms,
    }


def cpu_dry_run():
    """Synthetic in-memory mathematical controls, not H105 scientific outcomes."""
    base = {"adapter_10tok.fixture": torch.zeros(ADAPTER_PARAMS, dtype=torch.float64)}
    fine = {
        "adapter_10tok.fixture": torch.full(
            (ADAPTER_PARAMS,),
            PRIOR_DRIFT / math.sqrt(ADAPTER_PARAMS),
            dtype=torch.float64,
            requires_grad=True,
        )
    }
    assert torch.equal(
        interpolate(base, fine, 0)["adapter_10tok.fixture"],
        base["adapter_10tok.fixture"],
    )
    assert torch.equal(
        interpolate(base, fine, 1)["adapter_10tok.fixture"],
        fine["adapter_10tok.fixture"],
    )
    penalty = anchor_penalty(fine, base)
    gradient = torch.autograd.grad(penalty, tuple(fine.values()))[0]
    assert math.isclose(float(gradient.norm()), 1.0, abs_tol=1e-12)
    fixture_rows = [
        {
            "context_sha256": f"{i:064x}",
            "family_id": f"fixture-family-{i:03d}",
            "task": "regression",
            "dl_split": "DL-Train",
            "source_split": "train",
            "origin": "real",
        }
        for i in range(205)
    ]
    split = family_assignment(fixture_rows)
    assert (
        len(split["assessment_families"]) == 41 and len(split["train_families"]) == 164
    )
    return {
        "schema": "jope.h105.cpu-preparation/v1",
        "status": "CPU_DRY_RUN_PASS",
        "protocol": protocol(),
        "protocol_sha256": protocol_hash(),
        "synthetic_math_controls": True,
        "synthetic_fold_assignment_sha256": split["sha256"],
        "synthetic_fixture_family_counts": {"train": 164, "assessment": 41},
        "data_reads": 0,
        "checkpoint_reads": 0,
        "optimizer_updates": 0,
        "timing_cells_run": 0,
        "GPU_started": False,
        "GPU_ticket_created": False,
        "outcome_cells_run": 0,
        "admission_assessed": False,
        "execution_authorized_by_this_task": False,
    }


def run_cell(args):
    shared = runtime()
    shared.assert_lease(args.lock_fd)
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise RuntimeError("unregistered H105 cell")
    mark_phase(args, "loading")
    torch.set_num_threads(4)
    safe_path(args.input)
    safe_path(args.checkpoint)
    if (
        args.checkpoint.resolve()
        != (Path.home() / "jope-data/h51/ckpt/round-882.pt").resolve()
    ):
        raise RuntimeError("only the approved offline r882 COPY may be staged")
    if args.input.resolve() != (Path.home() / "jope-data/h50/input").resolve():
        raise RuntimeError("only approved H50 packet allowed")
    for name, expected in [
        ("source_manifest.json", MANIFEST_SHA),
        ("selected.json", SELECTED_SHA),
        ("cpu_exact_fused.py", FUSED_SHA),
    ]:
        if sha256(safe_path(args.input / name)) != expected:
            raise RuntimeError("H50 identity mismatch")
    manifest = json.loads((args.input / "source_manifest.json").read_text())
    if (
        manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("input closure not train-only")
    # Shared identity metadata only; never select/open a DL-Val member or payload.
    selected = json.loads((args.input / "selected.json").read_text())
    rows = selected["train"]
    if len(rows) != 1219 or len({r["family_id"] for r in rows}) != 205:
        raise RuntimeError("full DL-Train inventory required")
    split = family_assignment(rows)
    shared.validate_paths(args.input, rows)
    train_rows = [r for r in rows if r["family_id"] in split["train_families"]]
    assessment_rows = [
        r for r in rows if r["family_id"] in split["assessment_families"]
    ]
    if len(split["assessment_families"]) != 41 or len(split["train_families"]) != 164:
        raise RuntimeError("incorrect whole-family partition")
    h48 = load("h105_h48", h59.H48_PATH)
    scratch = attempt_scratch(
        args.out.parent, args.arm, args.seed, args.attempt, timing=args.timing
    )
    staged = h59.stage_r882_copy(args.checkpoint, scratch)
    pools, tables = h55.load_rows(args.input, train_rows, h48)
    net, params, guard, _ = h55.build_model(
        args.input, staged, torch.device("cuda:0"), h48
    )
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device("cuda:0"))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cuda:0"))
    ctx = {"net": net, "h48": h48, "guard": guard}
    names = h59.verify_cell_integrity(ctx)
    initial = h55.adapter_state(net)
    initial_sha = h55.adapter_digest(initial)
    initial_gpu = {n: t.to(params[0].device) for n, t in initial.items()}
    sampler = load("h105_sampler", h55.H53B_PATH)
    plan = shared.make_update_plan(
        shared.batch_plan(pools, sampler.cached_training_indices, args.seed), "CONTROL"
    )
    if len(plan) != 40:
        raise RuntimeError("exactly40updates required")
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    mark_phase(args, "training")
    for microbatches in plan:
        batch = microbatches[0]
        torch.manual_seed(batch.seed)
        torch.cuda.manual_seed_all(batch.seed)

        def closure():
            prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
            base_loss = h59.loss_by_arm(
                "CONTROL",
                prediction,
                h59._batch_targets(pools, batch, prediction.device),
            )
            current = {n: p for n, p in net.named_parameters() if p.requires_grad}
            return candidate_loss(base_loss, current, initial_gpu, args.arm)

        shared.optimizer_step(net.named_parameters(), optimizer, closure, "CONTROL")
    h59.verify_cell_integrity(ctx)
    fine = h55.adapter_state(net)
    fine_sha = h55.adapter_digest(fine)
    alpha = protocol()["alpha"][args.arm]
    candidate = interpolate(initial, fine, alpha)
    shared.restore_adapter(net, candidate)
    final_sha = h55.adapter_digest(h55.adapter_state(net))
    if final_sha == initial_sha:
        raise RuntimeError("no adapter changed")
    if not args.timing:
        h48.save_adapter_snapshot(
            args.out.with_suffix(".adapter.pt"), candidate, R882_SHA
        )
    drift = math.sqrt(
        sum(
            float((candidate[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )
    for p in params:
        p.requires_grad_(False)
    net.eval()
    h61 = load("h105_h61", HERE.parent / "h61/runner.py")

    def evaluate(packet, predictor):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError("nonfinite terminal score")
        result = h61.evaluate_pool_predictions(packet, packet, scores)
        result["family_by_context"] = {
            cid: p["rec"]["family_id"] for cid, p in packet.items()
        }
        return result

    mark_phase(args, "scoring", outcome_started=True)
    train = evaluate(pools, predict)
    assessment = None
    reference = None
    if not args.timing:
        if sha256(args.input / "selected.json") != SELECTED_SHA:
            raise RuntimeError("roster changed")
        ap, at = h55.load_rows(args.input, assessment_rows, h48)
        tt, tm = h48.h46.make_tokens(net, at, torch.device("cuda:0"))
        predictor = h48.make_predictor(net, ap, tt, tm, torch.device("cuda:0"))
        assessment = evaluate(ap, predictor)
        shared.restore_adapter(net, initial)
        try:
            reference = evaluate(ap, predictor)
        finally:
            shared.restore_adapter(net, candidate)
    for p in params:
        p.requires_grad_(True)
    h59.verify_cell_integrity(ctx)
    shared.assert_lease(args.lock_fd)
    if h55.adapter_digest(h55.adapter_state(net)) != final_sha:
        raise RuntimeError("candidate restore failed")
    result = {
        "wave": "H105",
        "arm": args.arm,
        "seed": args.seed,
        "timing_only": args.timing,
        "updates": len(plan),
        "trainable_count": sum(p.numel() for p in params),
        "trainable_names": list(names),
        "dope_before": DOPE_SHA,
        "dope_after": DOPE_SHA,
        "checkpoint_sha256": R882_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "selected_sha256": SELECTED_SHA,
        "protocol_sha256": protocol_hash(),
        "prereg_sha256": sha256(HERE / "PREREG.md"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {p: sha256(ROOT / p) for p in SOURCE_PINS},
        "split": split,
        "initial_adapter_sha256": initial_sha,
        "fine_tuned_adapter_sha256": fine_sha,
        "final_adapter_sha256": final_sha,
        "adapter_l2": drift,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "DL_Val_payloads_opened": 0,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
    }
    if not args.timing:
        result.update(train=train, assessment=assessment, r882_assessment=reference)
    atomic_json(args.out, result)
    mark_phase(args, "complete", outcome_started=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument(
        "--input", type=Path, default=Path.home() / "jope-data/h50/input"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "jope-data/h51/ckpt/round-882.pt",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--attempt-token")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--lock-fd", type=int)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = cpu_dry_run()
        if args.out:
            if args.out.exists():
                raise RuntimeError("refusing repeated CPU receipt")
            atomic_json(args.out, result)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return
    if (
        args.out is None
        or args.phase != args.out.with_suffix(".phase.json")
        or args.lock_fd is None
        or not args.attempt_token
        or args.out.exists()
    ):
        parser.error("cell needs unique output, phase/token and inherited FIFO lease")
    run_cell(args)


if __name__ == "__main__":
    main()
