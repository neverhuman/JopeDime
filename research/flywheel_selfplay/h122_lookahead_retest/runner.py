#!/usr/bin/env python3
"""H122 Lookahead-only protocol, data-free prep and lock-bound cell worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ("CONTROL", "LA")
SEEDS = (48, 49, 50)
FOLD_SEED = 122042
FOLD_COUNT = 5
ASSESSMENT_FOLD = 0
UPDATES = 40
MARGIN = 0.001
TIE_TOL = 1e-12
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 122000
ELO_BOOTSTRAP_SEED = 122001
H50_INPUT = Path.home() / "jope-data/h50/input"
R882_COPY = Path.home() / "jope-data/h51/ckpt/round-882.pt"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
FUSED_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
SOURCE_PINS = {
    "research/flywheel_selfplay/h93_lookahead_noise/runner.py": "84665188d67a4663839994e165e037f3df540c0e0cdd9839411655dc0db847ab",
    "research/flywheel_selfplay/h76_sam_steps/runner.py": "3556fe81fe654b148d301244d6d0460bf2363b5a97db0a0b644507168e97d24f",
    "research/flywheel_selfplay/h59/runner.py": "604d02ed72a6149284792cba315124efd1ffe8cac79e5ace22613db86c11b4c3",
    "research/flywheel_selfplay/h55/runner.py": "56fd220689805f07877ae403c2f00a4b34831cc4b2919f4bd69807ff3c21f4bc",
    "research/flywheel_selfplay/h61/runner.py": "a1654add23b76e54b22afe5a7f42bf543420b12ca0d31d52b5c25c409498d037",
    "research/flywheel_selfplay/h61/run_wave.py": "b665589255015d6dcc014ef0b6dd86953d336849914fe5278361bd5d76cda426",
}


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load(
    "h122_h93_core", ROOT / "research/flywheel_selfplay/h93_lookahead_noise/runner.py"
)
h59, h55 = core.h59, core.h55
h61 = load("h122_h61", ROOT / "research/flywheel_selfplay/h61/runner.py")
h61wave = load("h122_h61_wave", ROOT / "research/flywheel_selfplay/h61/run_wave.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_pins() -> None:
    for relative, expected in SOURCE_PINS.items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"H122 shared source identity changed: {relative}")
    if (
        h59.R882_SHA,
        h59.DOPE_SHA,
        h59.MANIFEST_SHA,
        core.SELECTED_SHA,
        core.FUSED_SHA,
    ) != (
        R882_SHA,
        DOPE_SHA,
        MANIFEST_SHA,
        SELECTED_SHA,
        FUSED_SHA,
    ):
        raise RuntimeError("H122 pinned H50/r882/DOPE identities changed")


def protocol() -> dict:
    return {
        "wave": "H122",
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "fold_seed": FOLD_SEED,
        "fold_count": FOLD_COUNT,
        "assessment_fold": ASSESSMENT_FOLD,
        "assessment_families": 41,
        "train_families": 164,
        "updates": UPDATES,
        "rounds": 20,
        "updates_per_round": 2,
        "batch": 16,
        "slate": 192,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "clip": 1.0,
        "lookahead_k": 5,
        "lookahead_alpha": 0.5,
        "base_loss": "H82_CONTROL_SmoothL1_plus_0.1KL_temp0.2",
        "primary": "LA minus fresh r882 official unrevealed argmin family utility",
        "margin": MARGIN,
        "median_seed_gain_strictly_positive": True,
        "harm_loss_threshold": 0.01,
        "maximum_harm_fraction": 0.10,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_rng": BOOTSTRAP_SEED,
        "elo_rng": ELO_BOOTSTRAP_SEED,
        "reference_elo": 1500,
        "gpu_cell_wall_cap_seconds": 2700,
        "fifo_predecessors": ["H99", "H100", "H105", "H109", "H121"],
        "checkpoint_sha256": R882_SHA,
        "dope_sha256": DOPE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "selected_sha256": SELECTED_SHA,
        "fused_sha256": FUSED_SHA,
        "execution_authorized_by_cpu_prep": False,
    }


def protocol_sha256() -> str:
    value = json.dumps(
        protocol(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(value.encode()).hexdigest()


def family_assignment(rows: list[dict]) -> dict:
    """Partition only validated DL-Train family identities; never opens payloads."""
    if not rows:
        raise ValueError("nonempty H50 DL-Train metadata required")
    contexts = []
    for row in rows:
        if (
            row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("source_split") not in ("train", "inner")
            or row.get("has_meta_test") is True
            or row.get("origin") not in (None, "real")
        ):
            raise ValueError(
                "H122 accepts only real regression DL-Train train/inner rows"
            )
        family = row.get("family_id")
        context = row.get("context_sha256")
        if not isinstance(family, str) or not family:
            raise ValueError("family identity required")
        if (
            not isinstance(context, str)
            or len(context) != 64
            or any(ch not in "0123456789abcdef" for ch in context)
        ):
            raise ValueError("valid context SHA-256 required")
        contexts.append(context)
    if len(set(contexts)) != len(contexts):
        raise ValueError("duplicate DL-Train context identity")
    families = sorted({row["family_id"] for row in rows})
    if len(families) != 205:
        raise ValueError("H122 requires the pinned 205-family DL-Train roster")
    folds = [
        sorted(part.tolist())
        for part in np.array_split(
            np.random.default_rng(FOLD_SEED).permutation(families), FOLD_COUNT
        )
    ]
    if any(len(fold) != 41 for fold in folds):
        raise RuntimeError(
            "H122 fresh family assignment must have five 41-family folds"
        )
    result = {
        "seed": FOLD_SEED,
        "folds": folds,
        "assessment_families": folds[ASSESSMENT_FOLD],
        "train_families": sorted(set(families) - set(folds[ASSESSMENT_FOLD])),
    }
    if set(result["assessment_families"]) & set(result["train_families"]):
        raise RuntimeError("assessment/train family overlap")
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def split_rows(rows: list[dict], assignment: dict) -> tuple[list[dict], list[dict]]:
    allowed = set(assignment["train_families"]) | set(assignment["assessment_families"])
    if any(row["family_id"] not in allowed for row in rows):
        raise ValueError("row family outside preregistered H122 assignment")
    train = [row for row in rows if row["family_id"] in assignment["train_families"]]
    assessment = [
        row for row in rows if row["family_id"] in assignment["assessment_families"]
    ]
    if len(train) + len(assessment) != len(rows):
        raise RuntimeError("H122 family split lost rows")
    return train, assessment


def safe_path(path: Path) -> Path:
    return core.safe_path(Path(path))


def attempt_scratch(root: Path, arm: str, seed: int, attempt: int) -> Path:
    if arm not in ARMS or seed not in SEEDS or type(attempt) is not int or attempt < 1:
        raise ValueError("invalid H122 physical attempt")
    path = safe_path(root) / "scratch" / f"{arm}-seed{seed}-attempt{attempt}"
    if path.exists() or path.is_symlink():
        raise RuntimeError("H122 attempt scratch already exists")
    return path


def stage_r882_for_attempt(
    root: Path, arm: str, seed: int, attempt: int, lock_fd: int
) -> tuple[Path, dict]:
    """Require the FIFO lease and launch disk check before any checkpoint copy/write."""
    core.assert_lease(lock_fd)
    checkpoint = safe_path(R882_COPY)
    if checkpoint.resolve() != R882_COPY.resolve():
        raise RuntimeError("only the approved offline r882 copy may be staged")
    preflight = h61wave.safe_preflight()
    scratch = attempt_scratch(root, arm, seed, attempt)
    staged = h59.stage_r882_copy(checkpoint, scratch)
    return staged, preflight


def make_update_plan(batches: list, arm: str) -> list[tuple]:
    if arm not in ARMS:
        raise ValueError("unregistered H122 arm")
    plan = core.make_update_plan(batches, arm)
    if len(plan) != UPDATES:
        raise RuntimeError("H122 cells require exactly 40 adapter updates")
    return plan


def atomic_json(path: Path, value: dict) -> None:
    h59.check_writer_disk_budget()
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def mark_phase(args, stage: str, *, outcome_started: bool = False) -> None:
    old = json.loads(args.phase.read_text())
    if old.get("attempt_token") != args.attempt_token or (
        old.get("outcome_started") is True and not outcome_started
    ):
        raise RuntimeError("H122 phase token/scoring boundary mismatch")
    atomic_json(args.phase, dict(old, stage=stage, outcome_started=outcome_started))


def run_cell(args) -> dict:
    """Run one future FIFO-authorized cell; CPU-prep CLI never invokes this worker."""
    core.assert_lease(args.lock_fd)
    if (
        args.arm not in ARMS
        or args.seed not in SEEDS
        or type(args.attempt) is not int
        or args.attempt < 1
    ):
        raise RuntimeError("unregistered H122 cell/attempt")
    out = safe_path(args.out)
    phase = safe_path(args.phase)
    if out.exists() or phase != out.with_suffix(".phase.json"):
        raise RuntimeError("repeated H122 cell or unexpected phase path")
    validate_source_pins()
    # This guard runs before roster/checkpoint access and again immediately before staging.
    h61wave.safe_preflight()
    mark_phase(args, "loading")
    torch.set_num_threads(4)
    input_path = safe_path(args.input)
    checkpoint = safe_path(args.checkpoint)
    if (
        input_path.resolve() != H50_INPUT.resolve()
        or checkpoint.resolve() != R882_COPY.resolve()
    ):
        raise RuntimeError(
            "H122 accepts only the pinned H50 input and offline r882 copy"
        )
    for name, expected in (
        ("source_manifest.json", MANIFEST_SHA),
        ("selected.json", SELECTED_SHA),
        ("cpu_exact_fused.py", FUSED_SHA),
    ):
        if sha256(input_path / name) != expected:
            raise RuntimeError(f"H122 H50 input identity changed: {name}")
    manifest = json.loads((input_path / "source_manifest.json").read_text())
    if (
        manifest.get("excluded_payloads_opened") != 0
        or manifest.get("outer_rows_delivered") != 0
    ):
        raise RuntimeError("H122 pinned input closure is not train-only")
    # Read only the selected DL-Train identities; never access a dev/holdout/sealed member.
    selected = json.loads((input_path / "selected.json").read_text())
    rows = selected["train"]
    if len(rows) != 1219:
        raise RuntimeError("H122 requires all 1,219 pinned DL-Train identities")
    split = family_assignment(rows)
    train_rows, assessment_rows = split_rows(rows, split)
    core.validate_paths(input_path, train_rows + assessment_rows)
    h48 = load("h122_h48", h59.H48_PATH)
    staged, launch_preflight = stage_r882_for_attempt(
        out.parent, args.arm, args.seed, args.attempt, args.lock_fd
    )
    pools, tables = h55.load_rows(input_path, train_rows, h48)
    net, params, guard, _ = h55.build_model(
        input_path, staged, torch.device("cuda:0"), h48
    )
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device("cuda:0"))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device("cuda:0"))
    context = {"net": net, "h48": h48, "guard": guard}
    names = h59.verify_cell_integrity(context)
    initial = h55.adapter_state(net)
    initial_sha = h55.adapter_digest(initial)
    sampler = load("h122_sampler", h55.H53B_PATH)
    batches = core.batch_plan(pools, sampler.cached_training_indices, args.seed)
    plan = make_update_plan(batches, args.arm)
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    lookahead = core.LookaheadState(params) if args.arm == "LA" else None
    mark_phase(args, "training")
    for update_index, microbatches in enumerate(plan):

        def loss_for(batch):
            torch.manual_seed(batch.seed)
            torch.cuda.manual_seed_all(batch.seed)
            prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
            return h59.loss_by_arm(
                "CONTROL",
                prediction,
                h59._batch_targets(pools, batch, prediction.device),
            )

        if args.arm == "CONTROL":
            core.optimizer_step(
                net.named_parameters(),
                optimizer,
                lambda: loss_for(microbatches[0]),
                "CONTROL",
            )
        else:
            core.accumulated_step(
                params,
                optimizer,
                microbatches,
                loss_for,
                update_index=update_index,
                lookahead=lookahead,
            )
    h59.verify_cell_integrity(context)
    candidate = h55.adapter_state(net)
    candidate_sha = h55.adapter_digest(candidate)
    if candidate_sha == initial_sha:
        raise RuntimeError("H122 adapter parameters did not change")
    h59.check_writer_disk_budget()
    h48.save_adapter_snapshot(out.with_suffix(".adapter.pt"), candidate, R882_SHA)
    adapter_l2 = math.sqrt(
        sum(
            float((candidate[name].double() - initial[name].double()).square().sum())
            for name in initial
        )
    )
    for parameter in params:
        parameter.requires_grad_(False)
    net.eval()
    assessment_rows_data, assessment_tables = h55.load_rows(
        input_path, assessment_rows, h48
    )
    assessment_tokens, assessment_masks = h48.h46.make_tokens(
        net, assessment_tables, torch.device("cuda:0")
    )
    assessment_predict = h48.make_predictor(
        net,
        assessment_rows_data,
        assessment_tokens,
        assessment_masks,
        torch.device("cuda:0"),
    )

    def evaluate(packet, predictor):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(value).all() for value in scores.values()):
            raise RuntimeError("nonfinite H122 argmin prediction")
        report = h61.evaluate_pool_predictions(
            packet, packet, scores, eval_mode=h61.EVAL_MODE_UNREV
        )
        report["family_by_context"] = {
            cid: value["rec"]["family_id"] for cid, value in packet.items()
        }
        return report

    # The durable scoring boundary precedes the one assessment evaluation.
    mark_phase(args, "scoring", outcome_started=True)
    assessment = evaluate(assessment_rows_data, assessment_predict)
    core.restore_adapter(net, initial)
    reference = evaluate(assessment_rows_data, assessment_predict)
    core.restore_adapter(net, candidate)
    for parameter in params:
        parameter.requires_grad_(True)
    h59.verify_cell_integrity(context)
    core.assert_lease(args.lock_fd)
    if h55.adapter_digest(h55.adapter_state(net)) != candidate_sha:
        raise RuntimeError("H122 adapter restore failed")
    result = {
        "wave": "H122",
        "arm": args.arm,
        "seed": args.seed,
        "attempt": args.attempt,
        "timing_only": False,
        "updates": len(plan),
        "trainable_count": sum(p.numel() for p in params),
        "trainable_names": list(names),
        "dope_before": DOPE_SHA,
        "dope_after": DOPE_SHA,
        "checkpoint_sha256": R882_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "selected_sha256": SELECTED_SHA,
        "protocol_sha256": protocol_sha256(),
        "prereg_sha256": sha256(HERE / "PREREG.md"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            relative: sha256(ROOT / relative) for relative in SOURCE_PINS
        },
        "fold_seed": FOLD_SEED,
        "split": split,
        "split_sha256": split["sha256"],
        "initial_adapter_sha256": initial_sha,
        "final_adapter_sha256": candidate_sha,
        "adapter_l2": adapter_l2,
        "assessment": assessment,
        "r882_assessment": reference,
        "holdout_payloads_opened": 0,
        "outer_rows_delivered": 0,
        "launch_preflight": launch_preflight,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "threads": torch.get_num_threads(),
        },
    }
    atomic_json(out, result)
    mark_phase(args, "complete", outcome_started=True)
    return result


def _rating(p: float):
    if p <= 0:
        return "-Infinity"
    if p >= 1:
        return "Infinity"
    return float(1500 + 400 * np.log10(p / (1 - p)))


def _elo(gains: np.ndarray, samples: np.ndarray) -> dict:
    score = (gains > 0).astype(float) + 0.5 * (gains == 0)
    boot = score[:, samples].mean(axis=2)
    median_boot = np.median(boot, axis=0)
    per_seed = []
    for index, seed in enumerate(SEEDS):
        per_seed.append(
            {
                "seed": seed,
                "elo": _rating(float(score[index].mean())),
                "ci95": [
                    _rating(float(x)) for x in np.quantile(boot[index], [0.025, 0.975])
                ],
            }
        )
    return {
        "per_seed": per_seed,
        "median_elo": _rating(float(np.median(score.mean(axis=1)))),
        "median_ci95": [
            _rating(float(x)) for x in np.quantile(median_boot, [0.025, 0.975])
        ],
    }


def summarize(cells: list[dict], draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Fail closed unless the six preregistered cells share one exact identity."""
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    indexed = {(cell.get("arm"), cell.get("seed")): cell for cell in cells}
    if draws != BOOTSTRAP_DRAWS or len(cells) != 6 or set(indexed) != expected:
        raise RuntimeError("INCOMPLETE: all six unique H122 paired cells required")
    control = np.asarray(
        [
            [
                indexed[("CONTROL", seed)]["assessment"]["per_family"][family]
                for family in sorted(
                    indexed[("CONTROL", seed)]["assessment"]["per_family"]
                )
            ]
            for seed in SEEDS
        ],
        dtype=np.float64,
    )
    families = sorted(indexed[("CONTROL", SEEDS[0])]["assessment"]["per_family"])
    if (
        len(families) != 41
        or control.shape != (3, 41)
        or not np.isfinite(control).all()
    ):
        raise RuntimeError(
            "INCOMPLETE: all 41 finite assessment-family utilities required"
        )
    identity = None
    references = []
    lookahead = []
    for seed in SEEDS:
        base = indexed[("CONTROL", seed)]
        la = indexed[("LA", seed)]
        if base.get("wave") != "H122" or la.get("wave") != "H122":
            raise RuntimeError("INCOMPLETE: mixed wave identity")
        for cell in (base, la):
            if (
                cell.get("fold_seed") != FOLD_SEED
                or cell.get("protocol_sha256") != protocol_sha256()
                or cell.get("updates") != UPDATES
                or cell.get("checkpoint_sha256") != R882_SHA
                or cell.get("dope_before") != DOPE_SHA
                or cell.get("dope_after") != DOPE_SHA
                or cell.get("holdout_payloads_opened") != 0
                or cell.get("outer_rows_delivered") != 0
            ):
                raise RuntimeError("INCOMPLETE: H122 cell identity/boundary mismatch")
            if (
                sorted(cell["assessment"]["per_family"]) != families
                or sorted(cell["r882_assessment"]["per_family"]) != families
            ):
                raise RuntimeError("INCOMPLETE: assessment family roster mismatch")
        if base["r882_assessment"] != la["r882_assessment"]:
            raise RuntimeError("INCOMPLETE: paired r882 reference changed within seed")
        current_identity = (
            base.get("source_commit"),
            base.get("split_sha256"),
            base.get("initial_adapter_sha256"),
        )
        la_identity = (
            la.get("source_commit"),
            la.get("split_sha256"),
            la.get("initial_adapter_sha256"),
        )
        if current_identity != la_identity or (
            identity is not None and identity != current_identity
        ):
            raise RuntimeError("INCOMPLETE: source/fold/reference identity mismatch")
        identity = current_identity
        references.append(
            [base["r882_assessment"]["per_family"][family] for family in families]
        )
        lookahead.append(
            [la["assessment"]["per_family"][family] for family in families]
        )
    reference = np.asarray(references, dtype=np.float64)
    la_utility = np.asarray(lookahead, dtype=np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(la_utility).all():
        raise RuntimeError("INCOMPLETE: nonfinite family utility")
    primary_gain = la_utility - reference
    paired_control_gain = la_utility - control
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(families), size=(draws, len(families))
    )
    boot = np.median(primary_gain[:, samples].mean(axis=2), axis=0)
    per_seed = primary_gain.mean(axis=1)
    median_gain = float(np.median(per_seed))
    lcb95 = float(np.quantile(boot, 0.05))
    ci95 = np.quantile(boot, [0.025, 0.975]).tolist()
    harmed = primary_gain.mean(axis=0) < -0.01
    elo_samples = np.random.default_rng(ELO_BOOTSTRAP_SEED).integers(
        0, len(families), size=(draws, len(families))
    )
    return {
        "schema": "jope.h122.metrics/v1",
        "complete": True,
        "reference_elo": 1500,
        "families": families,
        "bootstrap_draws": draws,
        "bootstrap_rng": BOOTSTRAP_SEED,
        "primary_vs_r882": {
            "per_seed_mean_gain": per_seed.tolist(),
            "median_seed_gain": median_gain,
            "ci95": ci95,
            "lcb95": lcb95,
            "harm_threshold": -0.01,
            "harm_fraction": float(harmed.mean()),
            "admitted": bool(
                median_gain > 0 and lcb95 > MARGIN + TIE_TOL and harmed.mean() <= 0.10
            ),
            "elo": _elo(primary_gain, elo_samples),
        },
        "secondary_vs_control": {
            "per_seed_mean_gain": paired_control_gain.mean(axis=1).tolist(),
            "median_seed_gain": float(np.median(paired_control_gain.mean(axis=1))),
            "elo": _elo(paired_control_gain, elo_samples),
            "admission_gate": False,
        },
    }


def cpu_dry_run() -> dict:
    """Synthetic protocol checks only; no H50/H51 access, ticket or GPU."""
    validate_source_pins()
    rows = [
        {
            "context_sha256": f"{i:064x}",
            "family_id": f"synthetic-family-{i:03d}",
            "task": "regression",
            "dl_split": "DL-Train",
            "source_split": "train",
            "origin": "real",
        }
        for i in range(205)
    ]
    assignment = family_assignment(rows)
    batches = [
        type("Batch", (), {"round_index": round_no, "step_index": step_no})()
        for round_no in range(1, 21)
        for step_no in range(8)
    ]
    plans = {arm: make_update_plan(batches, arm) for arm in ARMS}
    parameter = torch.tensor([0.0])
    state = core.LookaheadState([parameter])
    for _ in range(5):
        parameter.add_(1.0)
        state.after_fast_step()
    if state.steps != 5 or not torch.isfinite(parameter).all():
        raise RuntimeError("H122 synthetic Lookahead control failed")
    return {
        "schema": "jope.h122.cpu-dry-run/v1",
        "status": "CPU_DRY_RUN_PASS",
        "protocol_sha256": protocol_sha256(),
        "synthetic_fold_sha256": assignment["sha256"],
        "synthetic_families": {
            "train": len(assignment["train_families"]),
            "assessment": len(assignment["assessment_families"]),
        },
        "synthetic_updates_per_arm": {arm: len(plan) for arm, plan in plans.items()},
        "data_reads": 0,
        "checkpoint_reads": 0,
        "optimizer_updates": 0,
        "GPU_started": False,
        "qualification_lock_acquired": False,
        "FIFO_ticket_created": False,
        "outcome_cells_run": 0,
        "admission_assessed": False,
        "execution_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dry-run", action="store_true", required=True)
    args = parser.parse_args()
    if args.cpu_dry_run:
        print(json.dumps(cpu_dry_run(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
