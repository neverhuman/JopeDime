"""H99 GPU cell and Fold-B worker, reusing H80's pinned model and ES primitives."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import torch

import runner as h99

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INPUT = Path.home() / "jope-data/h50/input"
CHECKPOINT = Path.home() / "jope-data/h51/ckpt/round-882.pt"
MANIFEST_SHA = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load H99 dependency {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_execution_paths(run_root: Path, input_dir: Path, checkpoint: Path) -> None:
    h80 = configure_h80()
    h80.safe_path(run_root)
    h80.safe_path(input_dir)
    h80.safe_path(checkpoint)
    if Path(checkpoint).is_symlink() or not Path(checkpoint).is_file():
        raise RuntimeError("H99 r882 checkpoint path must be an existing regular file")


def configure_h80():
    """Pin H80 helpers to H99's fresh seed/arms without editing H80 source."""
    h80 = h99.load_h80()
    h80.FOLD_SEED = h99.FOLD_SEED
    h80.SEEDS = h99.SEEDS
    h80.NULL_SEED = h99.NULL_SEED
    original_import = h80.import_module

    def import_h55_with_h99_folds(name, path):
        module = original_import(name, path)
        if Path(path).resolve() == h80.H55.resolve():
            module.FOLD_SEED = h99.FOLD_SEED
        return module

    h80.import_module = import_h55_with_h99_folds
    original_fitness = h80.compute_fold_a_fitness

    def h99_fitness(candidate, baseline, arm, *, esh_lambda=None):
        if arm == "ES":
            return original_fitness(candidate, baseline, "ES", esh_lambda=1.0)
        lam = float(getattr(h80, "H99_LAMBDA", 2.0) if esh_lambda is None else esh_lambda)
        return original_fitness(candidate, baseline, "ESH", esh_lambda=lam)

    h80.compute_fold_a_fitness = h99_fitness
    h80.score_train_fitness = _h99_score_train_fitness(h80)
    original_disk_check = h80.check_disk_abort

    def h99_disk_check(statvfs_fn=os.statvfs):
        original_disk_check(statvfs_fn=statvfs_fn)
        stats = statvfs_fn("/")
        used = 100.0 * (stats.f_blocks - stats.f_bavail) / stats.f_blocks if stats.f_blocks else 100.0
        if used >= 94.5:
            raise RuntimeError(f"H99 disk abort threshold reached ({used:.2f}% >= 94.5%)")

    h80.check_disk_abort = h99_disk_check
    return h80


def _h99_score_train_fitness(h80):
    def score(ctx, h55, vector, template, names, baseline_per_family=None, arm="ESH", esh_lambda=None):
        h55.set_adapter(ctx["net"], h80.vector_state(vector, template, names))
        ctx["net"].eval()
        ids = sorted(ctx["pools_a"])
        result = ctx["h48"].evaluate_utility(ids, ctx["pools_a"], ctx["predictor_a"])
        per_family = result["per_family"]
        if len(per_family) != len(ctx["fold_a_families"]):
            raise RuntimeError("H99 Fold-A family inventory changed during execution")
        reference = per_family if baseline_per_family is None else baseline_per_family
        fit = h80.compute_fold_a_fitness(per_family, reference, arm, esh_lambda=esh_lambda)
        return fit["fitness"], per_family, fit

    return score


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_cell(cell: dict, staged_checkpoint: Path, scratch: Path, run_root: Path, input_dir: Path = INPUT, attempt: int = 1) -> dict:
    arm, seed = cell["arm"], int(cell["seed"])
    if cell not in [*h99.execution_plan()["candidate_cells"], h99.execution_plan()["null_cell"]]:
        raise ValueError("unregistered H99 cell")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("H99 attempt number must be a positive integer")
    output = run_root / "outcomes" / "cells" / arm / f"seed{seed}" / f"attempt{attempt}"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite H99 attempt output: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("H99 outcome cells require CUDA; use CPU fake-cell tests for local verification")
    h80 = configure_h80()
    validate_execution_paths(run_root, input_dir, staged_checkpoint)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    started = time.monotonic()
    ctx, h55 = h80.make_context(device, input_dir, staged_checkpoint)
    if ctx["fold_assignment"]["seed"] != h99.FOLD_SEED:
        raise RuntimeError("H99 worker did not receive fresh seed-9900 family split")
    template = h55.adapter_state(ctx["net"])
    names = tuple(sorted(template))
    center = h80.adapter_vector(template, names)
    if center.numel() != h99.ADAPTER_PARAMS:
        raise RuntimeError("H99 trainable parameter count differs from 29,211")
    theta = torch.nn.Parameter(center.clone())
    optimizer = torch.optim.Adam([theta], lr=h80.ES_LEARNING_RATE)
    training_arm = "ES" if arm == "NULL" else "ESH"
    lam = None if arm == "NULL" else h99.LAMBDA_BY_ARM[arm]
    h80.H99_LAMBDA = 1.0 if lam is None else lam
    before_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    baseline_score, baseline_families, _ = h99_score_baseline(h80, ctx, h55, center, template, names, training_arm, lam)
    generations = []
    for generation in range(1, h99.GENERATIONS + 1):
        row, _ = h80.run_generation(
            ctx, h55, theta, optimizer, template, names, baseline_families,
            seed=seed, arm=training_arm, generation=generation, null_arm=(arm == "NULL"),
        )
        row["arm"] = arm
        row["esh_lambda"] = lam
        generations.append(row)
    h80.check_disk_abort()
    final_state = h80.vector_state(theta.detach().cpu(), template, names)
    if torch.equal(center, h80.adapter_vector(final_state, names)):
        raise RuntimeError("H99 completed without changing the adapter parameters")
    after_dope = ctx["h48"].h46.checkpoint_guard(ctx["guard"], ctx["net"], expected=DOPE_SHA)
    if before_dope != DOPE_SHA or after_dope != DOPE_SHA:
        raise RuntimeError("H99 DOPE checksum changed during the cell")
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "candidate_adapter.pt"
    h80.check_disk_abort()
    ctx["h48"].save_adapter_snapshot(artifact, final_state, h80.R882_SHA)
    elapsed = time.monotonic() - started
    summary = {
        "schema": "jope.h99.cell/v1", "status": "complete", "arm": arm, "seed": seed,
        "null_diagnostic": arm == "NULL", "generations_completed": len(generations),
        "evaluated_perturbations": len(generations) * h99.POPULATION,
        "population": h99.POPULATION, "antithetic_pairs": h99.POPULATION // 2,
        "sigma": h99.SIGMA, "esh_lambda": lam, "adam_learning_rate": h80.ES_LEARNING_RATE,
        "selection": "final mean-theta only", "fold_seed": h99.FOLD_SEED,
        "fold_assignment": ctx["fold_assignment"], "fold_assignment_sha256": ctx["fold_sha"],
        "fold_a_family_count": len(ctx["fold_a_families"]), "fold_b_scored": False,
        "baseline_fold_a_family_utility": baseline_score, "generations": generations,
        "generation_wall_seconds": [row["generation_wall_seconds"] for row in generations],
        "wall_seconds": elapsed, "candidate_adapter_artifact": str(artifact),
        "candidate_adapter_artifact_sha256": h80.sha256(artifact),
        "adapter_sha256": h55.adapter_digest(final_state), "r882_checkpoint_sha256": h80.R882_SHA,
        "source_manifest_sha256": MANIFEST_SHA, "dope_sha256_before": before_dope,
        "dope_sha256_after": after_dope, "validation_payloads_opened": False,
        "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
        "optimizer_updates": len(generations),
    }
    atomic_json(output / "summary.json", summary)
    return {"summary": str(output / "summary.json"), "artifact_sha256": summary["candidate_adapter_artifact_sha256"], "wall_seconds": elapsed}


def h99_score_baseline(h80, ctx, h55, center, template, names, arm, lam):
    score, families, fit = h80.score_train_fitness(
        ctx, h55, center, template, names, baseline_per_family=None, arm=arm, esh_lambda=lam
    )
    return score, families, fit


def score_fold_b(staged_checkpoint: Path, scratch: Path, run_root: Path, input_dir: Path = INPUT) -> dict:
    """Score the seven fixed final adapters once on fresh seed-9900 Fold B only."""
    if not torch.cuda.is_available():
        raise RuntimeError("H99 Fold-B scoring requires CUDA; use fake-cell fixture tests locally")
    from coordinator import completed_attempt

    plan = h99.execution_plan()
    completed_attempts = {
        (cell["arm"], int(cell["seed"])): completed_attempt(
            run_root, cell["arm"], cell["seed"]
        )
        for cell in [*plan["candidate_cells"], plan["null_cell"]]
    }
    h80 = configure_h80()
    h80.safe_path(input_dir)
    h80.safe_path(staged_checkpoint)
    h55 = h80.import_module("h99_fold_b_h55", h80.H55)
    h55.FOLD_SEED = h99.FOLD_SEED
    h48 = _load_module("h99_fold_b_h48", h80.H48)
    h48.CHECKPOINT_SHA = h80.R882_SHA
    selected, train_rows, _ = h55.input_metadata(input_dir, h48)
    del selected
    h80.validate_roster_rows(input_dir, train_rows, allow_val=False)
    assignment = h99.fresh_family_assignment(train_rows, h48=h48)
    fold_b_ids = set(assignment["fold_b_families"])
    rows_b = [row for row in train_rows if row["family_id"] in fold_b_ids]
    if not rows_b or len(rows_b) != sum(row["family_id"] in fold_b_ids for row in train_rows):
        raise RuntimeError("H99 fresh Fold-B roster is empty or incomplete")
    pools, tables = h55.load_rows(input_dir, rows_b, h48)
    device = torch.device("cuda:0")
    net, _, guard, _ = h55.build_model(input_dir, staged_checkpoint, device, h48)
    h80.validate_trainable_inventory(net.named_parameters())
    tokens, masks = h48.h46.make_tokens(net, tables, device)
    predictor = h48.make_predictor(net, pools, tokens, masks, device)
    ids = [row["context_sha256"] for row in rows_b]
    before = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    net.eval()
    baseline = h48.evaluate_utility(ids, pools, predictor)
    null_cell = plan["null_cell"]
    null_attempt = completed_attempts[(null_cell["arm"], int(null_cell["seed"]))]
    null_dir = run_root / "outcomes" / "cells" / null_cell["arm"] / f"seed{null_cell['seed']}" / f"attempt{null_attempt}"
    null_summary = json.loads((null_dir / "summary.json").read_text())
    if (null_summary.get("arm"), null_summary.get("seed"), null_summary.get("status")) != ("NULL", 36, "complete"):
        raise RuntimeError("H99 NULL diagnostic cell is incomplete")
    expected = {(cell["arm"], cell["seed"]) for cell in plan["candidate_cells"]}
    candidates = []
    seen = set()
    for arm, seed in sorted(expected):
        attempt = completed_attempts[(arm, seed)]
        cell_dir = run_root / "outcomes" / "cells" / arm / f"seed{seed}" / f"attempt{attempt}"
        summary = json.loads((cell_dir / "summary.json").read_text())
        artifact_path = cell_dir / "candidate_adapter.pt"
        if (summary.get("arm"), summary.get("seed"), summary.get("status")) != (arm, seed, "complete"):
            raise RuntimeError(f"H99 cell summary mismatch for {arm}/seed{seed}")
        cell_assignment = summary.get("fold_assignment", {})
        if (
            cell_assignment.get("seed") != h99.FOLD_SEED
            or cell_assignment.get("folds") != assignment["folds"]
            or cell_assignment.get("fold_a_families") != assignment["fold_a_families"]
            or cell_assignment.get("fold_b_families") != assignment["fold_b_families"]
        ):
            raise RuntimeError(f"H99 cell split differs from fresh Fold-B assignment for {arm}/seed{seed}")
        artifact_sha = h80.sha256(artifact_path)
        if summary.get("candidate_adapter_artifact_sha256") != artifact_sha:
            raise RuntimeError(f"H99 candidate artifact digest differs from cell summary: {artifact_path}")
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact.get("base_checkpoint_sha256") != h80.R882_SHA or artifact.get("dope_sha256") != DOPE_SHA or artifact.get("trainable_params") != h99.ADAPTER_PARAMS:
            raise RuntimeError(f"H99 candidate artifact pin mismatch: {artifact_path}")
        if (arm, seed) in seen:
            raise RuntimeError(f"duplicate H99 candidate cell {arm}/seed{seed}")
        seen.add((arm, seed))
        h55.set_adapter(net, artifact["adapter_trainable_state"])
        net.eval()
        h80.check_disk_abort()
        result = h48.evaluate_utility(ids, pools, predictor)
        h80.check_disk_abort()
        if h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA) != DOPE_SHA:
            raise RuntimeError("H99 DOPE checksum changed during Fold-B scoring")
        family_gains = {family: result["per_family"][family] - baseline["per_family"][family] for family in baseline["per_family"]}
        candidates.append({"arm": arm, "seed": seed, "family_utility": result["per_family"], "context_utility": result["per_context"], "family_gain_vs_r882": family_gains, "fold_b_scoring_passes": 1, "candidate_adapter_sha256": artifact_sha})
    if seen != expected:
        raise RuntimeError("H99 Fold-B panel does not contain exactly the six admission cells")
    after = h48.h46.checkpoint_guard(guard, net, expected=DOPE_SHA)
    payload = {
        "schema": "jope.h99.fold-b/v1", "status": "complete", "fold_seed": h99.FOLD_SEED,
        "fold_assignment": assignment, "fold_assignment_sha256": assignment["fold_assignment_sha256"],
        "fold_b_family_count": len(assignment["fold_b_families"]), "source_manifest_sha256": MANIFEST_SHA,
        "r882_checkpoint_sha256": h80.R882_SHA, "dope_sha256_before": before, "dope_sha256_after": after,
        "baseline_family_utility": baseline["per_family"], "candidates": candidates,
        "candidate_count": len(candidates), "fold_b_read_count": len(candidates),
        "null_cell_complete": True, "null_fold_b_scored": False,
        "validation_payloads_opened": False, "holdout_payloads_opened": 0, "outer_rows_delivered": 0,
    }
    atomic_json(run_root / "outcomes" / "fold_b.json", payload)
    return {"fold_b": str(run_root / "outcomes" / "fold_b.json"), "candidate_count": len(candidates)}
