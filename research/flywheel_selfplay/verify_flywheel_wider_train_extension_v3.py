#!/usr/bin/env python3
"""
Phase 7 Bounded Development-Only 256-Context Strict-Train Winner Generation & Matched Acceptance Test
(`jope-bounded-continuation-20261024T091646Z.txt`).

Executes ONE bounded development-only extension testing whether wider `source_split == 'train'`
(`DL-Train`) winner pool augmentation (`256` strict-train contexts across `123` unique families,
vs `5/1,016` in Phase 6) shifts the matched `430`-step training comparison against the frozen-pool
baseline (`BASELINE_FROZEN_POOL_TRAIN_ONLY_430`):
  1. Distinguishes primary-human continuous-improvement authority from the self-imposed per-test
     budget that completed in Phase 6 (`task-1067`), and verifies that Phase 6's negative/inconclusive
     result (`flywheel_winner_pool_integration_v2_report.json`) and all prior artifacts remain intact.
  2. Checks live `xbabe2` CPU/RAM/GPU headroom and enforces a `<=50.0%` whole-host CPU soft target
     (`GLOBAL_CPU_GUARD.soft_cap_pct = 45.0%`) and `<=85.0%` hard ceiling (`nice -n 10`), with a
     finite wall-clock stop (`<=1,080s` / `18.0` minutes).
  3. Selects a deterministic family-stratified cohort of `256` `source_split == 'train'` (`DL-Train`)
     contexts (`25.20%` of the `1,016` strict-train cohort, `123` unique families, `0` family overlap
     with `DL-Val` `26` families or `DL-Holdout` `25` families).
  4. Runs fixed-budget stage-1 `Stratified300JTT1` screening (`6` novel candidates/context = `1,536`
     stage-1 fits: `5` direct `EditZero` mutations of the context's own champion + `1` disjoint-family
     Landmark-V2 neighbor) followed by stage-2 `Full-JTT1` verification on the top-`2` `Stratified300JTT1`
     candidates/context (`512` `Full-JTT1` fits, `run_outer_test=False`).
  5. Materializes versioned training-only pool revisions under
     `/mnt/jope-archive/flywheel-20261023/corpus_v3_wider_strict_train_256/` (`pools_v3a_strict_train_q25`
     and `pools_v3b_strict_train_median`) without mutating `/mnt/jope-archive/flywheel-20261023/corpus/pools/`
     or any evaluation pools.
  6. Executes the matched `430`-step training comparison (`seed=20265000`, `enable_evo_queue=False`,
     `use_cv_test_gate=False`, greedy `enable_ucb=False`) against `BASELINE_FROZEN_POOL_TRAIN_ONLY_430`,
     evaluates once on frozen `DL-Val` (`91` contexts) and frozen `DL-Holdout` (`93` contexts) labeled
     `development_learning_loop_validation` (`development_diagnostic`), and publishes
     `terminal_manifest_v3_20261024.json` to S3 with full SHA-256 readback verification.
"""

from __future__ import annotations

import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from collections import defaultdict
import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from evo_batch_queue import build_base_gpu_slate_batch
from model import FlywheelGameNet
from rust_evaluator import (
    GLOBAL_CPU_GUARD,
    assemble_pipe_dsl,
    evaluate_gp_batch_rust,
    evaluate_gp_batch_rust_strat300,
    featurize_dsl_412,
    mutate_parameter_clause,
    parse_pipe_dsl,
    select_landmark_preferred_pools,
)
from selfplay_loop import (
    build_fold_safe_sos_priors,
    compute_component_bootstrap_summary,
    evaluate_episode_rollout_context,
    fit_fold_safe_context_scaler,
    load_all_pools_and_features,
    transform_context_features,
)

# Enforce <=50% whole-host CPU soft target (set to 45.0%) and <=85% hard ceiling
GLOBAL_CPU_GUARD.soft_cap_pct = 45.0
GLOBAL_CPU_GUARD.hard_cap_pct = 85.0

ARCHIVE_ROOT = Path("/mnt/jope-archive/flywheel-20261023")
CORPUS_DIR = ARCHIVE_ROOT / "corpus"
CHECKPOINTS_DIR = ARCHIVE_ROOT / "checkpoints"
RESULTS_DIR = ARCHIVE_ROOT / "results"

CORPUS_V2_MANIFEST_PATH = ARCHIVE_ROOT / "corpus_v2_train_winner_integrated" / "corpus_v2_manifest.json"
CORPUS_V3_DIR = ARCHIVE_ROOT / "corpus_v3_wider_strict_train_256"
POOLS_V3A_DIR = CORPUS_V3_DIR / "pools_v3a_strict_train_q25"
POOLS_V3B_DIR = CORPUS_V3_DIR / "pools_v3b_strict_train_median"
CORPUS_V3_MANIFEST_PATH = CORPUS_V3_DIR / "corpus_v3_manifest.json"

PHASE6_REPORT_PATH = RESULTS_DIR / "flywheel_winner_pool_integration_v2_report.json"
PHASE7_REPORT_PATH = RESULTS_DIR / "flywheel_wider_train_extension_v3_report.json"

TERMINAL_V1_PATH = ARCHIVE_ROOT / "terminal_manifest_20261024.json"
TERMINAL_V2_PATH = ARCHIVE_ROOT / "terminal_manifest_v2_20261024.json"
TERMINAL_V3_PATH = ARCHIVE_ROOT / "terminal_manifest_v3_20261024.json"
TERMINAL_V3_RECEIPT_PATH = ARCHIVE_ROOT / "terminal_manifest_v3_20261024.s3_receipt.json"

S3_BUCKET = "veox-jopedime-learning-use2"
S3_PREFIX = "experiments/flywheel-selfplay-20261023"
AWS_PROFILE = "veox-jepson-prod"

FINITE_WALL_STOP_S = 1080.0
MAX_EVOLUTION_WALL_S = 600.0


class DeadlineExpiredError(RuntimeError):
    """Raised when `FINITE_WALL_STOP_S` is reached inside any stage."""


def check_wall_deadline(script_t0: float, stage_name: str, reserve_s: float = 30.0) -> None:
    elapsed = time.monotonic() - script_t0
    if elapsed >= (FINITE_WALL_STOP_S - reserve_s):
        raise DeadlineExpiredError(
            f"Finite wall-clock stop reached during {stage_name}: elapsed={elapsed:.2f}s >= "
            f"limit={FINITE_WALL_STOP_S - reserve_s:.2f}s (FINITE_WALL_STOP_S={FINITE_WALL_STOP_S:.2f}s)"
        )

IMMUTABLE_PRIOR_ARTIFACTS = {
    str(CORPUS_DIR / "corpus_manifest.json"): "3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3",
    str(RESULTS_DIR / "flywheel_selfplay_report.json"): "ea0396eedf52adadd93218a8c61c52460c91aa10d71a5004c7b70dbb8016cc5f",
    str(RESULTS_DIR / "flywheel_audit_verification.json"): "1d4a18508ac08400874ddc3128dbbf31f73f40c11a16dbee269abef7ff9d1461",
    str(RESULTS_DIR / "flywheel_qualified_next_phase_audit.json"): "962d0a3f57058866e1eca04ccddc1f5cc10140b4d475cb44e6f34efa7ccb4a77",
    str(RESULTS_DIR / "flywheel_closure_and_next_phase_report.json"): "be7ce4afa8cef95df049c49c942c281ad024f4b9d879f930161918e0f63f17af",
    str(TERMINAL_V1_PATH): "c7af03f9b7539fe1f2e1235f469ebb69647eae4a5811a07930964affecdecd03",
    str(CORPUS_V2_MANIFEST_PATH): "05af9e5930092ab41587b96ce9029f10bf84557ff45b9ceac94348e8e8e510c8",
    str(PHASE6_REPORT_PATH): "957318e0d6fea2dc9df627ebf2d40de4399fdf3cdd4d5471f2dca1b8d27d6bc9",
    str(TERMINAL_V2_PATH): "02d2a7dadcc8d24d2bd2c0c56dd2528cdb56ed013cbe25158993d65169ecfaa2",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sample_host_resources() -> dict:
    mem = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) == 2:
                mem[parts[0].strip()] = int(parts[1].strip().split()[0])
    avail_gib = round(mem.get("MemAvailable", 0) / (1024 * 1024), 3)
    total_gib = round(mem.get("MemTotal", 0) / (1024 * 1024), 3)
    rss_mib = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)

    gpu_info = {}
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) == 4:
            gpu_info = {
                "gpu_name": parts[0],
                "gpu_utilization_pct": float(parts[1]),
                "gpu_memory_used_mib": float(parts[2]),
                "gpu_memory_total_mib": float(parts[3]),
            }
    except Exception:
        pass

    return {
        "host_mem_available_gib": avail_gib,
        "host_mem_total_gib": total_gib,
        "process_max_rss_mib": rss_mib,
        "gpu": gpu_info,
    }


def verify_authority_and_immutable_artifacts() -> dict:
    """
    Step 1: Distinguish primary-human resource authority from self-imposed per-test experimental
    budgets, verify all 9 prior artifacts (including Phase 6's negative/inconclusive report)
    are untouched, and record pre-run live host headroom.
    """
    verified = []
    for p_str, exp_sha in IMMUTABLE_PRIOR_ARTIFACTS.items():
        p = Path(p_str)
        actual_sha = sha256_file(p)
        if actual_sha != exp_sha:
            raise RuntimeError(f"Immutable artifact hash mismatch for {p_str}: expected {exp_sha}, got {actual_sha}")
        verified.append({
            "path": p_str,
            "size_bytes": p.stat().st_size,
            "expected_sha256": exp_sha,
            "actual_sha256": actual_sha,
            "immutable_verified": True,
        })

    phase6_doc = json.loads(PHASE6_REPORT_PATH.read_text())
    val_v2a = phase6_doc["step4_matched_frozen_budget_pool_augmentation_experiment"]["dl_val_91_contexts_summary"]["contrasts"]["WINNER_POOL_V2A_TRAIN_ONLY_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]
    hold_v2a = phase6_doc["step4_matched_frozen_budget_pool_augmentation_experiment"]["dl_holdout_93_contexts_summary"]["contrasts"]["WINNER_POOL_V2A_TRAIN_ONLY_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]
    val_v2b = phase6_doc["step4_matched_frozen_budget_pool_augmentation_experiment"]["dl_val_91_contexts_summary"]["contrasts"]["WINNER_POOL_V2B_DL_TRAIN_510_minus_BASELINE_FROZEN_POOL_DL_TRAIN_510"]
    hold_v2b = phase6_doc["step4_matched_frozen_budget_pool_augmentation_experiment"]["dl_holdout_93_contexts_summary"]["contrasts"]["WINNER_POOL_V2B_DL_TRAIN_510_minus_BASELINE_FROZEN_POOL_DL_TRAIN_510"]

    return {
        "primary_human_grant_vs_per_test_budget_distinction": {
            "primary_human_directive_source_1": (
                "Direct user request (2026-10-23T22:55Z / 2026-10-24T01:40Z): 'Your goal is to demonstrate "
                "proper fly wheel behavior. Please study everything I said carefully and ensure you don't go "
                "over 85% CPU on any compute node. It is important that you do NOT stop until you have been "
                "able to show fly-wheel behavior and we are logging to S3 and getting smarter.'"
            ),
            "primary_human_directive_source_2": (
                "jope-human-followup-20261024T0455.txt: 'Human follow-up via authorized screen monitor: "
                "continue autonomously toward demonstrated continuous-improvement flywheel; ordinary "
                "proceed/next-phase permission is granted. ... After current run, continue the next justified "
                "phase rather than wait for routine approval. ... Maintain whole-host CPU <=85%, resource "
                "fair-share and locked ~/master_log.md coordination. Do not launch duplicate campaigns or "
                "reset existing budgets.'"
            ),
            "monitor_continuation_source_3": (
                "jope-bounded-continuation-20261024T091646Z.txt: 'Original direct-human goal remains autonomous "
                "continuous improvement; routine next phase does not itself require new approval. Your proposed "
                "wider train-context winner generation is a concrete next hypothesis (5/1016 strict-train "
                "contexts augmented is sparse), not a proven explanation. First distinguish an actual "
                "exhausted/expired primary-human resource grant from a self-imposed per-test budget that "
                "simply completed. ... Otherwise freeze ONE bounded development-only extension based on your "
                "proposed>=256 source_train contexts and ~18-minute estimate ...'"
            ),
            "is_primary_human_grant_exhausted_or_expired": False,
            "what_completed_in_phase6": (
                "Self-imposed per-test experimental budget (430 and 510 optimizer steps on the 5/1,016 "
                "strict-train and 16/1,219 DL-Train contexts evolved during the 3-round run) completed "
                "cleanly in 1,114.20s at 13.46% mean host CPU."
            ),
            "consumed_and_remaining_allowance": {
                "external_cloud_compute_spend_usd": 0.0,
                "s3_incremental_put_get_estimated_spend_usd": 0.0004,
                "host_compute_node": "local xbabe2 (32 CPU threads, 125.58 GiB RAM, 1x RTX 3090 24 GiB)",
                "competing_active_claims_on_xbabe2": 0,
                "bounded_phase7_wall_clock_cap_seconds": FINITE_WALL_STOP_S,
                "bounded_phase7_cpu_soft_target_pct": 50.0,
                "bounded_phase7_cpu_guard_configured_soft_cap_pct": GLOBAL_CPU_GUARD.soft_cap_pct,
                "bounded_phase7_cpu_hard_ceiling_pct": GLOBAL_CPU_GUARD.hard_cap_pct,
            },
        },
        "phase6_negative_inconclusive_result_preserved": {
            "report_path": str(PHASE6_REPORT_PATH),
            "report_sha256": IMMUTABLE_PRIOR_ARTIFACTS[str(PHASE6_REPORT_PATH)],
            "pair_a_v2a_minus_baseline_dl_val": val_v2a,
            "pair_a_v2a_minus_baseline_dl_holdout": hold_v2a,
            "pair_b_v2b_minus_baseline_dl_val": val_v2b,
            "pair_b_v2b_minus_baseline_dl_holdout": hold_v2b,
            "status": "preserved_intact_all_4_primary_pool_augmentation_cis_cross_zero",
        },
        "immutable_local_artifacts_verified": verified,
        "pre_run_host_resources": sample_host_resources(),
    }


def select_deterministic_256_strict_train_cohort(manifest: dict, pools_frozen: dict) -> tuple[list[dict], dict]:
    """
    Deterministically select `256` `source_split == 'train'` (`DL-Train`) contexts (`25.20%` of the
    `1,016` strict-train cohort) across `123` unique `DL-Train` families using family-stratified
    round-robin over contexts with `teacher.jtt1.gz` size <= 65 KB.
    Verifies 0 family overlap with `DL-Val` (`26` families) and `DL-Holdout` (`25` families).
    """
    records = manifest["records"]
    strict_train = [
        r for r in records
        if r["dl_split"] == "DL-Train" and r["source_split"] == "train" and r["n_candidates"] > 0
    ]
    dl_val = [r for r in records if r["dl_split"] == "DL-Val" and r["n_candidates"] > 0]
    dl_holdout = [r for r in records if r["dl_split"] == "DL-Holdout" and r["n_candidates"] > 0]

    train_fams = {r["family_id"] for r in strict_train}
    val_fams = {r["family_id"] for r in dl_val}
    hold_fams = {r["family_id"] for r in dl_holdout}
    assert len(train_fams & val_fams) == 0, "Family leakage between strict_train and DL-Val!"
    assert len(train_fams & hold_fams) == 0, "Family leakage between strict_train and DL-Holdout!"

    by_fam = defaultdict(list)
    for r in strict_train:
        cid = r["context_sha256"]
        if cid not in pools_frozen:
            continue
        sz = Path(r["teacher_jtt1_path"]).stat().st_size
        by_fam[r["family_id"]].append((sz, cid, r))

    for fam in by_fam:
        by_fam[fam].sort(key=lambda x: (x[0], x[1]))

    selected_records = []
    seen_cids = set()
    for pass_idx in range(12):
        fams_sorted = sorted(
            by_fam.keys(),
            key=lambda f: (by_fam[f][min(pass_idx, len(by_fam[f]) - 1)][0], f),
        )
        for f in fams_sorted:
            if pass_idx < len(by_fam[f]):
                sz, cid, r = by_fam[f][pass_idx]
                if sz <= 65000 and cid not in seen_cids:
                    selected_records.append(r)
                    seen_cids.add(cid)
                    if len(selected_records) == 256:
                        break
        if len(selected_records) == 256:
            break

    assert len(selected_records) == 256
    sel_fams = {r["family_id"] for r in selected_records}
    jtt1_sizes = [Path(r["teacher_jtt1_path"]).stat().st_size for r in selected_records]

    cohort_spec = {
        "target_context_count": 256,
        "selected_context_count": len(selected_records),
        "fraction_of_1016_strict_train_cohort": round(len(selected_records) / len(strict_train), 5),
        "all_selected_dl_split_eq_dl_train": all(r["dl_split"] == "DL-Train" for r in selected_records),
        "all_selected_source_split_eq_train": all(r["source_split"] == "train" for r in selected_records),
        "unique_families_in_selected_256": len(sel_fams),
        "total_families_in_1016_strict_train": len(train_fams),
        "family_overlap_with_dl_val_26_families": len(sel_fams & val_fams),
        "family_overlap_with_dl_holdout_25_families": len(sel_fams & hold_fams),
        "teacher_jtt1_gz_bytes_min": int(min(jtt1_sizes)),
        "teacher_jtt1_gz_bytes_median": int(sorted(jtt1_sizes)[len(jtt1_sizes) // 2]),
        "teacher_jtt1_gz_bytes_max": int(max(jtt1_sizes)),
        "selected_context_shas_sha256_digest": hashlib.sha256(
            "\n".join(r["context_sha256"] for r in selected_records).encode()
        ).hexdigest(),
    }
    return selected_records, cohort_spec


def propose_6_novel_candidates_for_strict_train_context(
    crec: dict,
    orig_pids_set: set[str],
    champ_entry: dict,
    champ_bank_strict_train: list[dict],
    landmark_v2_48: np.ndarray,
    seed: int,
) -> list[dict]:
    """
    Deterministically propose `6` novel GP DSL candidates for `crec` that are NOT already in
    `orig_pids_set`:
      - 5 direct `EditZero` mutations (`edit_parameter` x2, `edit_insert_stage`, `edit_replace_head`,
        `edit_delete_stage` / `edit_parameter`) of the context's own champion (`champ_entry["dsl"]`),
      - 1 nearest-neighbor Landmark-V2 champion from a strictly disjoint `source_split == 'train'`
        (`DL-Train`) family (`c["family_id"] != crec["family_id"]`).
    """
    rng = random.Random(seed)
    preferred_stages, preferred_heads = select_landmark_preferred_pools(landmark_v2_48)
    parent_dsl = champ_entry["dsl"]
    parent_sha = champ_entry["program_sha256"]
    stages, head = parse_pipe_dsl(parent_dsl)

    proposals = []
    seen_shas = set(orig_pids_set)
    seen_shas.add(parent_sha)

    def try_add(dsl: str, op: str, parents: list[str]) -> bool:
        if not dsl.startswith("(pipe X "):
            return False
        psha = hashlib.sha256(dsl.encode()).hexdigest()
        if psha in seen_shas:
            return False
        seen_shas.add(psha)
        proposals.append({
            "dsl": dsl,
            "program_sha256": psha,
            "operator": op,
            "parents": parents,
        })
        return True

    # 1. Five direct EditZero mutations of the context's own champion DSL
    for attempt in range(40):
        if len(proposals) >= 5:
            break
        mode = attempt % 5
        if mode in (0, 1):
            new_stages = [mutate_parameter_clause(st, rng) for st in stages]
            new_head = mutate_parameter_clause(head, rng)
            try_add(assemble_pipe_dsl(new_stages, new_head), "edit_parameter", [parent_sha])
        elif mode == 2:
            new_stage = mutate_parameter_clause(rng.choice(preferred_stages), rng)
            pos = rng.randint(0, len(stages))
            new_stages = (stages[:pos] + [new_stage] + stages[pos:])[:6]
            try_add(assemble_pipe_dsl(new_stages, head), "edit_insert_stage", [parent_sha])
        elif mode == 3:
            new_head = mutate_parameter_clause(rng.choice(preferred_heads), rng)
            try_add(assemble_pipe_dsl(stages, new_head), "edit_replace_head", [parent_sha])
        else:
            if len(stages) >= 2:
                del_idx = rng.randrange(len(stages))
                new_stages = stages[:del_idx] + stages[del_idx + 1 :]
                try_add(assemble_pipe_dsl(new_stages, mutate_parameter_clause(head, rng)), "edit_delete_stage", [parent_sha])
            else:
                new_stage = mutate_parameter_clause(rng.choice(preferred_stages), rng)
                new_head = mutate_parameter_clause(head, rng)
                try_add(assemble_pipe_dsl(stages + [new_stage], new_head), "edit_insert_stage", [parent_sha])

    # 2. Disjoint-family Landmark-V2 nearest-neighbor champion transfer (strictly from DL-Train `train`)
    target_lm12 = np.asarray(landmark_v2_48[:12], dtype=np.float64)
    target_fam = crec["family_id"]
    scored_neighbors = []
    for c in champ_bank_strict_train:
        if c.get("family_id") == target_fam:
            continue
        c_lm = np.asarray(c["landmark_summary"], dtype=np.float64)
        dist = float(np.sum((target_lm12 - c_lm) ** 2))
        scored_neighbors.append((dist, c["program_sha256"], c))
    scored_neighbors.sort(key=lambda x: (x[0], x[1]))

    for _dist, _psha, nb in scored_neighbors:
        if len(proposals) >= 6:
            break
        if try_add(nb["dsl"], "disjoint_family_champion_transfer", [nb["program_sha256"]]):
            break
        # Or mutate the neighbor's parameters slightly if its exact SHA was already present
        nb_stages, nb_head = parse_pipe_dsl(nb["dsl"])
        mut_nb = assemble_pipe_dsl(
            [mutate_parameter_clause(st, rng) for st in nb_stages],
            mutate_parameter_clause(nb_head, rng),
        )
        if try_add(mut_nb, "disjoint_family_champion_edit_parameter", [nb["program_sha256"]]):
            break

    # Fallback synthesis if fewer than 6
    while len(proposals) < 6:
        n_st = rng.randint(1, 3)
        chosen_stages = [mutate_parameter_clause(rng.choice(preferred_stages), rng) for _ in range(n_st)]
        chosen_head = mutate_parameter_clause(rng.choice(preferred_heads), rng)
        try_add(assemble_pipe_dsl(chosen_stages, chosen_head), "landmark_synthesis", [parent_sha])

    return proposals[:6]


def run_bounded_256_context_winner_generation(
    selected_256_records: list[dict],
    pools_frozen: dict,
    champ_bank: list[dict],
    lm48_all: np.ndarray,
    script_t0: float,
) -> tuple[list[dict], dict]:
    """
    Execute fixed-budget Stage-1 (`Stratified300JTT1`, 6 candidates/context) + Stage-2 (`Full-JTT1`,
    top-2 candidates/context, `run_outer_test=False`) across the 256 `source_split == 'train'`
    contexts under `GLOBAL_CPU_GUARD` (`soft_cap_pct=45.0% <= 50.0%`, `hard_cap_pct=85.0%`) and
    finite wall-clock stop (`MAX_EVOLUTION_WALL_S = 900.0s`).
    """
    champ_by_cid = {c["context_sha256"]: c for c in champ_bank if c.get("context_sha256")}
    champ_bank_strict_train = [
        c for c in champ_bank
        if c.get("dl_split") == "DL-Train" and c.get("split") == "train"
    ]

    evo_t0 = time.monotonic()
    cpu_t0 = time.process_time()

    context_outcomes = []
    qualified_novel_winners = []
    stopped_early_by_wall_cap = False

    def _process_one_context(idx_and_rec: tuple[int, dict]) -> dict:
        idx, crec = idx_and_rec
        if (time.monotonic() - script_t0) >= MAX_EVOLUTION_WALL_S:
            raise DeadlineExpiredError(f"Evolution cap {MAX_EVOLUTION_WALL_S}s reached before context {idx}")
        check_wall_deadline(script_t0, f"evolution_context_{idx}", reserve_s=180.0)
        cid = crec["context_sha256"]
        p_frozen = pools_frozen[cid]
        orig_fit = p_frozen["fitness"]
        orig_gate = p_frozen["gate"]
        orig_pids_set = set(p_frozen["pids"])

        pass_fits = np.sort(orig_fit[orig_gate]) if np.any(orig_gate) else np.sort(orig_fit)
        frozen_best_f = float(pass_fits[0])
        frozen_q25_f = float(np.percentile(pass_fits, 25))
        frozen_median_f = float(np.percentile(pass_fits, 50))
        frozen_max_pass_f = float(pass_fits[-1])

        ch = champ_by_cid[cid]
        lm48 = lm48_all[crec["index"]]

        cands_6 = propose_6_novel_candidates_for_strict_train_context(
            crec=crec,
            orig_pids_set=orig_pids_set,
            champ_entry=ch,
            champ_bank_strict_train=champ_bank_strict_train,
            landmark_v2_48=lm48,
            seed=20267000 + idx,
        )

        # Stage 1: Stratified300JTT1 fast-fit on all 6 novel candidates
        s300_res = evaluate_gp_batch_rust_strat300(
            teacher_jtt1_gz_path=crec["teacher_jtt1_path"],
            candidates=cands_6,
            max_train_rows=300,
            max_workers=3,
            timeout_s=6.0,
        )
        s300_pass = sorted([r for r in s300_res if r.get("cv_gate_passed")], key=lambda r: float(r["fitness"]))
        if len(s300_pass) >= 2:
            top2_for_full = s300_pass[:2]
        elif len(s300_pass) == 1:
            others = sorted([r for r in s300_res if not r.get("cv_gate_passed")], key=lambda r: float(r["fitness"]))
            top2_for_full = s300_pass + others[:1]
        else:
            top2_for_full = sorted(s300_res, key=lambda r: float(r["fitness"]))[:2]

        # Stage 2: Full-JTT1 verification on top-2 (run_outer_test=False so outer test is untouched)
        full_res = evaluate_gp_batch_rust(
            teacher_jtt1_gz_path=crec["teacher_jtt1_path"],
            candidates=top2_for_full,
            run_outer_test=False,
            max_workers=2,
            timeout_s=10.0,
        )

        s300_by_sha = {r["program_sha256"]: r for r in s300_res}
        ctx_winners = []
        for fr in full_res:
            psha = fr["program_sha256"]
            f_val = float(fr["fitness"])
            lr_val = float(fr.get("logratio", 0.0))
            gate_ok = bool(fr.get("cv_gate_passed", False))
            s300_match = s300_by_sha.get(psha, {})
            rec_w = {
                "context_index_in_256": idx,
                "context_sha256": cid,
                "dataset_id": crec["dataset_id"],
                "view_id": crec["view_id"],
                "family_id": crec["family_id"],
                "source_split": crec["source_split"],
                "dl_split": crec["dl_split"],
                "program_sha256": psha,
                "dsl": fr["dsl"],
                "operator": fr.get("operator", "edit_zero"),
                "parents": fr.get("parents", []),
                "strat300_fitness": float(s300_match.get("fitness", 100.0)),
                "strat300_cv_gate_passed": bool(s300_match.get("cv_gate_passed", False)),
                "verified_full_jtt1_fitness": f_val,
                "verified_full_jtt1_logratio": lr_val,
                "verified_full_jtt1_cv_ratio": float(fr.get("cv_geomean_rmse_ratio", 1e9)),
                "verified_full_jtt1_cv_gate_passed": gate_ok,
                "frozen_pool_best_f": frozen_best_f,
                "frozen_pool_q25_f": frozen_q25_f,
                "frozen_pool_median_f": frozen_median_f,
                "frozen_pool_max_pass_f": frozen_max_pass_f,
                "beats_frozen_incumbent_best": bool(gate_ok and f_val < frozen_best_f - 1e-9),
                "beats_frozen_q25": bool(gate_ok and f_val < frozen_q25_f - 1e-9),
                "beats_frozen_median": bool(gate_ok and f_val < frozen_median_f - 1e-9),
                "beats_frozen_worst_passing": bool(gate_ok and f_val < frozen_max_pass_f - 1e-9),
            }
            ctx_winners.append(rec_w)

        return {
            "context_index_in_256": idx,
            "context_sha256": cid,
            "family_id": crec["family_id"],
            "strat300_evaluated_count": len(s300_res),
            "strat300_gate_pass_count": len(s300_pass),
            "full_jtt1_evaluated_count": len(full_res),
            "full_jtt1_gate_pass_count": sum(1 for w in ctx_winners if w["verified_full_jtt1_cv_gate_passed"]),
            "beats_incumbent_count": sum(1 for w in ctx_winners if w["beats_frozen_incumbent_best"]),
            "beats_q25_count": sum(1 for w in ctx_winners if w["beats_frozen_q25"]),
            "beats_median_count": sum(1 for w in ctx_winners if w["beats_frozen_median"]),
            "verified_candidates": ctx_winners,
        }

    # Process in chunks of 16 contexts with max_workers=2 to check finite wall-clock budget and log progress
    chunk_size = 16
    for c_start in range(0, len(selected_256_records), chunk_size):
        if (time.monotonic() - script_t0) >= MAX_EVOLUTION_WALL_S:
            stopped_early_by_wall_cap = True
            print(f"[Phase 7] Reached MAX_EVOLUTION_WALL_S ({MAX_EVOLUTION_WALL_S}s) at {c_start}/256 contexts; stopping evolution cleanly.")
            break
        batch_items = [
            (i, selected_256_records[i])
            for i in range(c_start, min(c_start + chunk_size, len(selected_256_records)))
        ]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                for out_ctx in ex.map(_process_one_context, batch_items):
                    context_outcomes.append(out_ctx)
                    for w in out_ctx["verified_candidates"]:
                        if w["verified_full_jtt1_cv_gate_passed"]:
                            qualified_novel_winners.append(w)
        except DeadlineExpiredError as exc:
            stopped_early_by_wall_cap = True
            print(f"[Phase 7] {exc}; preserving {len(context_outcomes)} completed contexts.")
            break

        done_n = len(context_outcomes)
        if done_n % 32 == 0 or done_n == len(selected_256_records):
            cpu_s = GLOBAL_CPU_GUARD.summary()
            n_q25 = sum(1 for w in qualified_novel_winners if w["beats_frozen_q25"])
            n_med = sum(1 for w in qualified_novel_winners if w["beats_frozen_median"])
            n_inc = sum(1 for w in qualified_novel_winners if w["beats_frozen_incumbent_best"])
            print(
                f"[Phase 7] Evolved {done_n}/{len(selected_256_records)} strict-train contexts in {time.monotonic() - evo_t0:.1f}s "
                f"| Full-JTT1 pass={len(qualified_novel_winners)}, beats_median={n_med}, beats_q25={n_q25}, beats_incumbent={n_inc} "
                f"| Host CPU mean={cpu_s['mean_host_cpu_pct']:.1f}% p95={cpu_s['p95_host_cpu_pct']:.1f}% peak={cpu_s['peak_host_cpu_pct']:.1f}%"
            )

    evo_wall_s = round(time.monotonic() - evo_t0, 2)
    evo_cpu_s = round(time.process_time() - cpu_t0, 2)

    contexts_with_gate_pass = len({w["context_sha256"] for w in qualified_novel_winners})
    contexts_with_median_beat = len({w["context_sha256"] for w in qualified_novel_winners if w["beats_frozen_median"]})
    contexts_with_q25_beat = len({w["context_sha256"] for w in qualified_novel_winners if w["beats_frozen_q25"]})
    contexts_with_incumbent_beat = len({w["context_sha256"] for w in qualified_novel_winners if w["beats_frozen_incumbent_best"]})

    evolution_summary = {
        "contexts_targeted": len(selected_256_records),
        "contexts_completed": len(context_outcomes),
        "stopped_early_by_wall_cap": stopped_early_by_wall_cap,
        "wall_seconds": evo_wall_s,
        "parent_python_cpu_seconds": evo_cpu_s,
        "total_strat300_candidate_fits": sum(c["strat300_evaluated_count"] for c in context_outcomes),
        "total_strat300_gate_passes": sum(c["strat300_gate_pass_count"] for c in context_outcomes),
        "total_full_jtt1_candidate_fits": sum(c["full_jtt1_evaluated_count"] for c in context_outcomes),
        "total_full_jtt1_gate_passing_novel_winners": len(qualified_novel_winners),
        "unique_program_sha256_in_new_winners": len({w["program_sha256"] for w in qualified_novel_winners}),
        "contexts_with_at_least_1_full_jtt1_gate_pass": contexts_with_gate_pass,
        "contexts_with_at_least_1_median_beat": contexts_with_median_beat,
        "contexts_with_at_least_1_q25_beat": contexts_with_q25_beat,
        "contexts_with_at_least_1_incumbent_best_beat": contexts_with_incumbent_beat,
        "new_winners_strictly_beating_incumbent_best": sum(1 for w in qualified_novel_winners if w["beats_frozen_incumbent_best"]),
        "new_winners_strictly_beating_q25": sum(1 for w in qualified_novel_winners if w["beats_frozen_q25"]),
        "new_winners_strictly_beating_median": sum(1 for w in qualified_novel_winners if w["beats_frozen_median"]),
        "new_winners_strictly_beating_worst_passing": sum(1 for w in qualified_novel_winners if w["beats_frozen_worst_passing"]),
        "outer_test_touched_during_evolution": False,
    }
    return qualified_novel_winners, evolution_summary


def build_versioned_v3_strict_train_pools(
    pools_frozen: dict,
    phase7_new_winners: list[dict],
    evolution_summary: dict,
    cohort_spec: dict,
) -> tuple[dict, dict, dict]:
    """
    Combine Phase 6's `28` `Full-JTT1`-verified `source_split == 'train'` novel winners with
    Phase 7's newly verified `source_split == 'train'` novel winners (deduplicated by
    `(context_sha256, program_sha256)`), and materialize two versioned strict-train pool revisions:
      - `pools_v3a_strict_train_q25`: admits `Full-JTT1` gate-passing novel winners on
        `source_split == 'train'` (`DL-Train`) contexts that strictly beat their context's
        frozen-pool top quartile (`Q25 F`).
      - `pools_v3b_strict_train_median`: admits `Full-JTT1` gate-passing novel winners on
        `source_split == 'train'` (`DL-Train`) contexts that strictly beat their context's
        frozen-pool median (`Median F`).
    Original `/mnt/jope-archive/flywheel-20261023/corpus/pools/*.npz` and all evaluation pools
    remain 100% untouched.
    """
    CORPUS_V3_DIR.mkdir(parents=True, exist_ok=True)
    POOLS_V3A_DIR.mkdir(parents=True, exist_ok=True)
    POOLS_V3B_DIR.mkdir(parents=True, exist_ok=True)

    v2_doc = json.loads(CORPUS_V2_MANIFEST_PATH.read_text())
    phase6_strict_train_winners = [
        w for w in v2_doc.get("all_90_qualified_novel_winner_records", [])
        if w.get("dl_split") == "DL-Train" and w.get("source_split") == "train" and w.get("verified_full_jtt1_cv_gate_passed")
    ]

    combined_by_pair = {}
    for w in phase6_strict_train_winners + phase7_new_winners:
        assert w["dl_split"] == "DL-Train" and w["source_split"] == "train"
        key = (w["context_sha256"], w["program_sha256"])
        if key not in combined_by_pair:
            combined_by_pair[key] = w

    all_strict_train_winners = list(combined_by_pair.values())

    w0_rng = np.random.RandomState(42)
    w1_sign = w0_rng.choice([-1.0, 1.0], size=(412, 16))

    def _materialize_revision(target_dir: Path, admitted_records: list[dict], revision_tag: str) -> tuple[dict, list[dict]]:
        by_cid = defaultdict(list)
        for ar in admitted_records:
            by_cid[ar["context_sha256"]].append(ar)

        pools_rev = {}
        for cid, p_entry in pools_frozen.items():
            pools_rev[cid] = copy.deepcopy(p_entry)

        materialized_entries = []
        for cid in sorted(by_cid.keys()):
            cand_list = by_cid[cid]
            orig_p = pools_frozen[cid]
            crec = orig_p["rec"]
            assert crec["dl_split"] == "DL-Train" and crec["source_split"] == "train"

            parent_pool_path = Path(crec["pool_path"])
            parent_pool_sha = sha256_file(parent_pool_path)

            with np.load(parent_pool_path, allow_pickle=False) as npz:
                orig_prog = npz["program"].astype(np.float32)
                orig_fit = npz["fitness"].astype(np.float64)
                orig_lr = npz["logratio"].astype(np.float64)
                orig_gate = npz["gate"].astype(np.bool_)
                orig_pids_raw = npz["program_sha256"]
                orig_pids = [
                    x.decode("ascii") if isinstance(x, (bytes, np.bytes_)) else str(x)
                    for x in orig_pids_raw
                ]
                orig_bmin = npz["birth_seq_min"] if "birth_seq_min" in npz else np.zeros(len(orig_pids), dtype=np.uint32)
                orig_bmax = npz["birth_seq_max"] if "birth_seq_max" in npz else np.zeros(len(orig_pids), dtype=np.uint32)
                orig_gmask = npz["generation_mask"] if "generation_mask" in npz else np.ones(len(orig_pids), dtype=np.uint32)
                orig_deliv = npz["delivery_count"] if "delivery_count" in npz else np.ones(len(orig_pids), dtype=np.uint32)

            orig_prior_by_pid = {
                pid_s: float(orig_p["prior_cost"][i])
                for i, pid_s in enumerate(orig_p["pids"])
            }
            base_prior_offset = float(np.median(orig_p["prior_cost"]))
            base_prog0_mean = float(np.mean(orig_p["program"][:, 0]))

            seen_local = set(orig_pids)
            new_progs, new_fits, new_lrs, new_gates, new_pids_str = [], [], [], [], []
            for ar in cand_list:
                psha = ar["program_sha256"]
                if psha in seen_local:
                    continue
                seen_local.add(psha)
                f412, _p16 = featurize_dsl_412(ar["dsl"])
                new_progs.append(f412.astype(np.float32))
                new_fits.append(float(ar["verified_full_jtt1_fitness"]))
                new_lrs.append(float(ar["verified_full_jtt1_logratio"]))
                new_gates.append(bool(ar["verified_full_jtt1_cv_gate_passed"]))
                new_pids_str.append(psha)

            if not new_pids_str:
                continue

            K_new = len(new_pids_str)
            comb_prog = np.vstack([orig_prog, np.stack(new_progs, axis=0)])
            comb_fit = np.concatenate([orig_fit, np.array(new_fits, dtype=np.float64)])
            comb_lr = np.concatenate([orig_lr, np.array(new_lrs, dtype=np.float64)])
            comb_gate = np.concatenate([orig_gate, np.array(new_gates, dtype=np.bool_)])
            comb_pids_str = orig_pids + new_pids_str
            comb_pids_bytes = np.array([s.encode("ascii") for s in comb_pids_str], dtype="S64")
            comb_bmin = np.concatenate([orig_bmin, np.zeros(K_new, dtype=orig_bmin.dtype)])
            comb_bmax = np.concatenate([orig_bmax, np.zeros(K_new, dtype=orig_bmax.dtype)])
            comb_gmask = np.concatenate([orig_gmask, np.ones(K_new, dtype=orig_gmask.dtype)])
            comb_deliv = np.concatenate([orig_deliv, np.ones(K_new, dtype=orig_deliv.dtype)])

            sort_idx = np.argsort(comb_pids_str)
            comb_prog = comb_prog[sort_idx]
            comb_fit = comb_fit[sort_idx]
            comb_lr = comb_lr[sort_idx]
            comb_gate = comb_gate[sort_idx]
            comb_pids_bytes = comb_pids_bytes[sort_idx]
            comb_bmin = comb_bmin[sort_idx]
            comb_bmax = comb_bmax[sort_idx]
            comb_gmask = comb_gmask[sort_idx]
            comb_deliv = comb_deliv[sort_idx]
            sorted_pids_str = [comb_pids_str[i] for i in sort_idx]

            out_npz_path = target_dir / f"{cid}.npz"
            np.savez_compressed(
                out_npz_path,
                program=comb_prog,
                fitness=comb_fit,
                logratio=comb_lr,
                gate=comb_gate,
                program_sha256=comb_pids_bytes,
                birth_seq_min=comb_bmin,
                birth_seq_max=comb_bmax,
                generation_mask=comb_gmask,
                delivery_count=comb_deliv,
            )
            new_pool_sha = sha256_file(out_npz_path)

            prog64 = comb_prog.astype(np.float64)
            proj16 = np.zeros((len(prog64), 16), dtype=np.float64)
            for j_col in range(412):
                proj16 += prog64[:, j_col : j_col + 1] * w1_sign[j_col : j_col + 1, :]
            proj16 /= math.sqrt(412.0)

            pad = np.pad(comb_prog, ((0, 0), (0, 20)), mode="constant")
            folded = pad.reshape(len(comb_prog), 18, 24)
            sig = folded.mean(axis=1) + 0.5 * folded.max(axis=1)
            sig_norm = np.maximum(np.linalg.norm(sig, axis=-1, keepdims=True), 1e-6)
            struct_sig = (sig / sig_norm).astype(np.float32)

            prior_cost = np.zeros(len(sorted_pids_str), dtype=np.float64)
            for idx_p, pid_s in enumerate(sorted_pids_str):
                if pid_s in orig_prior_by_pid:
                    prior_cost[idx_p] = orig_prior_by_pid[pid_s]
                else:
                    prior_cost[idx_p] = base_prior_offset + 0.02 * (float(comb_prog[idx_p, 0]) - base_prog0_mean)

            new_rec = dict(crec)
            new_rec["pool_path"] = str(out_npz_path)
            new_rec["n_candidates"] = len(sorted_pids_str)

            pools_rev[cid] = {
                "rec": new_rec,
                "program": comb_prog,
                "proj16": proj16,
                "struct_sig": struct_sig,
                "fitness": comb_fit,
                "logratio": comb_lr,
                "gate": comb_gate,
                "pids": sorted_pids_str,
                "prior_cost": prior_cost,
                "scale": float(orig_p["scale"]),
            }

            materialized_entries.append({
                "revision": revision_tag,
                "context_sha256": cid,
                "dataset_id": crec["dataset_id"],
                "view_id": crec["view_id"],
                "family_id": crec["family_id"],
                "source_split": crec["source_split"],
                "dl_split": crec["dl_split"],
                "parent_pool_path": str(parent_pool_path),
                "parent_pool_sha256": parent_pool_sha,
                "parent_n_candidates": len(orig_pids),
                "parent_incumbent_best_f": float(np.min(orig_fit)),
                "new_pool_path": str(out_npz_path),
                "new_pool_sha256": new_pool_sha,
                "new_n_candidates": len(sorted_pids_str),
                "new_incumbent_best_f": float(np.min(comb_fit)),
                "admitted_winner_program_shas": new_pids_str,
                "admitted_winner_count": K_new,
            })

        return pools_rev, materialized_entries

    admitted_v3a = [w for w in all_strict_train_winners if w["beats_frozen_q25"]]
    pools_v3a, entries_v3a = _materialize_revision(POOLS_V3A_DIR, admitted_v3a, "v3a_strict_train_q25")

    admitted_v3b = [w for w in all_strict_train_winners if w["beats_frozen_median"]]
    pools_v3b, entries_v3b = _materialize_revision(POOLS_V3B_DIR, admitted_v3b, "v3b_strict_train_median")

    corpus_v3_manifest = {
        "schema": "jope.flywheel-corpus-v3-wider-strict-train-256/v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_corpus_manifest_path": str(CORPUS_DIR / "corpus_manifest.json"),
        "parent_corpus_manifest_sha256": sha256_file(CORPUS_DIR / "corpus_manifest.json"),
        "parent_corpus_v2_manifest_path": str(CORPUS_V2_MANIFEST_PATH),
        "parent_corpus_v2_manifest_sha256": sha256_file(CORPUS_V2_MANIFEST_PATH),
        "diagnostic_evaluation_pools_frozen": True,
        "epistemic_label": "development_learning_loop_validation (development_diagnostic)",
        "cohort_256_selection_spec": cohort_spec,
        "phase7_256_context_evolution_summary": evolution_summary,
        "combined_strict_train_winner_union_summary": {
            "phase6_strict_train_winners_count": len(phase6_strict_train_winners),
            "phase7_new_strict_train_winners_count": len(phase7_new_winners),
            "combined_unique_context_program_pairs": len(all_strict_train_winners),
            "combined_unique_program_sha256": len({w["program_sha256"] for w in all_strict_train_winners}),
            "combined_unique_strict_train_contexts_with_gate_pass": len({w["context_sha256"] for w in all_strict_train_winners}),
            "combined_strictly_beating_incumbent_best": sum(1 for w in all_strict_train_winners if w["beats_frozen_incumbent_best"]),
            "combined_strictly_beating_q25": len(admitted_v3a),
            "combined_strictly_beating_median": len(admitted_v3b),
        },
        "revisions": {
            "v3a_strict_train_q25": {
                "description": "Augments `source_split == 'train'` (`DL-Train`) contexts with `Full-JTT1`-verified novel winners strictly beating their frozen-pool top quartile (`Q25`).",
                "augmented_context_count": len(entries_v3a),
                "fraction_of_1016_strict_train_augmented": round(len(entries_v3a) / 1016.0, 5),
                "admitted_winner_count": sum(e["admitted_winner_count"] for e in entries_v3a),
                "pools_dir": str(POOLS_V3A_DIR),
                "materialized_pools": entries_v3a,
            },
            "v3b_strict_train_median": {
                "description": "Augments `source_split == 'train'` (`DL-Train`) contexts with `Full-JTT1`-verified novel winners strictly beating their frozen-pool median (`Median`).",
                "augmented_context_count": len(entries_v3b),
                "fraction_of_1016_strict_train_augmented": round(len(entries_v3b) / 1016.0, 5),
                "admitted_winner_count": sum(e["admitted_winner_count"] for e in entries_v3b),
                "pools_dir": str(POOLS_V3B_DIR),
                "materialized_pools": entries_v3b,
            },
        },
    }
    CORPUS_V3_MANIFEST_PATH.write_text(json.dumps(corpus_v3_manifest, indent=2) + "\n")
    return pools_v3a, pools_v3b, corpus_v3_manifest


def train_matched_430_steps_strict_train(
    strict_train_cids: list[str],
    pools_dict: dict,
    ctx_scaled_st: np.ndarray,
    sos_by_cid: dict[str, np.ndarray],
    init_state_dict: dict,
    device: torch.device,
    script_t0: float,
    epochs: int = 10,
    batch_size: int = 24,
    seed: int = 20265000,
    tag: str = "baseline",
) -> tuple[FlywheelGameNet, dict]:
    """
    Strictly deterministic, step-matched, seed-matched 430-step training loop (`enable_evo_queue=False`,
    `use_cv_test_gate=False`, `max_slate=192`, per-step `rng=random.Random(seed * 100000 + opt_steps)`),
    enforcing `check_wall_deadline(script_t0, ...)` at every batch.
    """
    t0 = time.monotonic()
    cpu_t0 = time.process_time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    valid_train_cids = [c for c in strict_train_cids if len(pools_dict[c]["pids"]) >= 64]
    model = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128).to(device)
    model.load_state_dict(copy.deepcopy(init_state_dict))

    opt = torch.optim.AdamW(model.parameters(), lr=5.5e-4, weight_decay=2.5e-4)
    total_steps = epochs * max(1, (len(valid_train_cids) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-5)

    epoch_losses = []
    opt_steps = 0

    for _ep in range(epochs):
        model.train()
        order = valid_train_cids.copy()
        random.shuffle(order)
        ep_loss_vals = []

        for b_start in range(0, len(order), batch_size):
            check_wall_deadline(script_t0, f"train_{tag}_step_{opt_steps}", reserve_s=45.0)
            GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.4)
            batch_cids = order[b_start : b_start + batch_size]
            if not batch_cids:
                continue

            batch_dict = build_base_gpu_slate_batch(
                batch_cids=batch_cids,
                pools=pools_dict,
                ctx_scaled_584=ctx_scaled_st,
                sos_by_cid=sos_by_cid,
                round_idx=2,
                use_cv_test_gate=False,
                rng=random.Random(seed * 100000 + opt_steps),
            )

            ctx_t = torch.from_numpy(batch_dict["ctx_b"]).to(device)
            prog_t = torch.from_numpy(batch_dict["prog_b"]).to(device)
            proj_t = torch.from_numpy(batch_dict["proj_b"]).to(device)
            prior_t = torch.from_numpy(batch_dict["prior_b"]).to(device)
            scale_t = torch.from_numpy(batch_dict["scale_b"]).to(device)
            sos_t = torch.from_numpy(batch_dict["sos_b"]).to(device)
            target_f_t = torch.from_numpy(batch_dict["target_f_b"]).to(device)
            gate_t = torch.from_numpy(batch_dict["gate_b"]).to(device)
            rev_idx_t = torch.from_numpy(batch_dict["rev_idx_b"]).to(device)
            rev_f_t = torch.from_numpy(batch_dict["rev_f_b"]).to(device)
            rev_gate_t = torch.from_numpy(batch_dict["rev_gate_b"]).to(device)

            out = model.predict_costs(
                ctx_feat=ctx_t,
                prog_412=prog_t,
                proj_16=proj_t,
                base_prior_cost=prior_t,
                scale=scale_t,
                sos_prior=sos_t,
                revealed_indices=rev_idx_t,
                revealed_raw_f=rev_f_t,
                revealed_gate=rev_gate_t,
                enable_prior_correction=True,
                enable_struct_kernel=True,
                enable_cross_attn=True,
                enable_ucb=False,
            )

            pred_cost = out["total_cost"].float()
            fail_logit = out["fail_logit"]

            norm_target = torch.where(
                gate_t > 0.5,
                (target_f_t / scale_t.unsqueeze(-1)).float(),
                torch.full_like(pred_cost, 2.0),
            )

            _, top_k_idx = torch.topk(-norm_target, k=16, dim=-1)
            best_pred = torch.gather(pred_cost, dim=1, index=top_k_idx)
            best_targ = torch.gather(norm_target, dim=1, index=top_k_idx)

            gap_targ = (norm_target.unsqueeze(1) - best_targ.unsqueeze(2)).clamp(min=0.0, max=3.0)
            margin_pred = best_pred.unsqueeze(2) - pred_cost.unsqueeze(1) + 0.05 * gap_targ
            rank_loss = torch.mean(gap_targ * F.softplus(margin_pred * 4.0))

            pred_centered = pred_cost - pred_cost.mean(dim=-1, keepdim=True)
            targ_centered = norm_target - norm_target.mean(dim=-1, keepdim=True)
            huber_loss = F.smooth_l1_loss(pred_centered, targ_centered, beta=0.25)

            fail_target = 1.0 - gate_t
            gate_loss = F.binary_cross_entropy_with_logits(fail_logit, fail_target)

            loss = rank_loss + 0.35 * huber_loss + 0.20 * gate_loss
            ep_loss_vals.append(float(loss.detach().cpu().item()))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt_steps += 1

        if ep_loss_vals:
            epoch_losses.append(round(float(np.mean(ep_loss_vals)), 5))

    wall_s = round(time.monotonic() - t0, 2)
    cpu_s = round(time.process_time() - cpu_t0, 2)
    diff_sq = 0.0
    total_sq = 0.0
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            v_f = v.detach().cpu().float()
            v0_f = init_state_dict[k].detach().cpu().float()
            total_sq += float(torch.sum(v_f * v_f).item())
            diff_sq += float(torch.sum((v_f - v0_f) ** 2).item())

    model.eval()
    return model, {
        "arm_tag": tag,
        "train_context_count": len(valid_train_cids),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer_steps": opt_steps,
        "max_slate": 192,
        "use_cv_test_gate": False,
        "enable_evo_queue": False,
        "seed": seed,
        "wall_seconds": wall_s,
        "cpu_process_seconds": cpu_s,
        "initial_epoch_loss": epoch_losses[0],
        "final_epoch_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
        "param_l2_norm": round(float(np.sqrt(total_sq)), 5),
        "param_update_l2_norm": round(float(np.sqrt(diff_sq)), 5),
    }


def run_matched_v3_acceptance_test(
    pools_frozen: dict,
    pools_v3a: dict,
    pools_v3b: dict,
    ctx_raw_584: np.ndarray,
    device: torch.device,
    script_t0: float,
    reuse_saved_checkpoints: bool = False,
) -> dict:
    """
    Execute the matched `430`-step training comparison on the `1,016` `source_split == 'train'`
    (`DL-Train`) contexts and evaluate once on frozen `DL-Val` (`91` contexts) and frozen
    `DL-Holdout` (`93` contexts), enforcing `check_wall_deadline(script_t0, ...)` at every stage.
    """
    check_wall_deadline(script_t0, "matched_acceptance_test_init", reserve_s=45.0)
    dl_train_cids = sorted([cid for cid, p in pools_frozen.items() if p["rec"]["dl_split"] == "DL-Train"])
    dl_val_cids = sorted([cid for cid, p in pools_frozen.items() if p["rec"]["dl_split"] == "DL-Val"])
    dl_holdout_cids = sorted([cid for cid, p in pools_frozen.items() if p["rec"]["dl_split"] == "DL-Holdout"])
    strict_train_cids = sorted([cid for cid in dl_train_cids if pools_frozen[cid]["rec"]["source_split"] == "train"])

    f_mask_all = np.ones(584, dtype=np.float32)
    med_st, iqr_st, m_st = fit_fold_safe_context_scaler(strict_train_cids, pools_frozen, ctx_raw_584, f_mask_all)
    ctx_scaled_st = transform_context_features(ctx_raw_584, med_st, iqr_st, m_st)

    check_wall_deadline(script_t0, "sos_st_frozen_build", reserve_s=45.0)
    sos_st_frozen = build_fold_safe_sos_priors(strict_train_cids, dl_val_cids + dl_holdout_cids, pools_frozen)
    check_wall_deadline(script_t0, "sos_st_v3a_build", reserve_s=45.0)
    sos_st_v3a = build_fold_safe_sos_priors(strict_train_cids, dl_val_cids + dl_holdout_cids, pools_v3a)
    check_wall_deadline(script_t0, "sos_st_v3b_build", reserve_s=45.0)
    sos_st_v3b = build_fold_safe_sos_priors(strict_train_cids, dl_val_cids + dl_holdout_cids, pools_v3b)

    r1_sd = torch.load(CHECKPOINTS_DIR / "flywheel_round_1_full_corpus.pt", map_location=device)
    ckpt_v3a_path = CHECKPOINTS_DIR / "flywheel_winner_pool_v3a_q25_train_only_430.pt"
    ckpt_v3b_path = CHECKPOINTS_DIR / "flywheel_winner_pool_v3b_median_train_only_430.pt"

    m_base_430, tel_base_430 = train_matched_430_steps_strict_train(
        strict_train_cids=strict_train_cids,
        pools_dict=pools_frozen,
        ctx_scaled_st=ctx_scaled_st,
        sos_by_cid=sos_st_frozen,
        init_state_dict=r1_sd,
        device=device,
        script_t0=script_t0,
        epochs=10,
        batch_size=24,
        seed=20265000,
        tag="BASELINE_FROZEN_POOL_TRAIN_ONLY_430",
    )

    if reuse_saved_checkpoints and ckpt_v3a_path.exists() and ckpt_v3b_path.exists():
        m_v3a_430 = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128).to(device)
        m_v3a_430.load_state_dict(torch.load(ckpt_v3a_path, map_location=device))
        m_v3a_430.eval()
        tel_v3a_430 = {
            "arm_tag": "WINNER_POOL_V3A_Q25_430",
            "reused_saved_checkpoint": True,
            "checkpoint_path": str(ckpt_v3a_path),
            "checkpoint_sha256": sha256_file(ckpt_v3a_path),
        }
        m_v3b_430 = FlywheelGameNet(ctx_dim=584, prog_dim=412, proj_dim=16, d_model=128).to(device)
        m_v3b_430.load_state_dict(torch.load(ckpt_v3b_path, map_location=device))
        m_v3b_430.eval()
        tel_v3b_430 = {
            "arm_tag": "WINNER_POOL_V3B_MEDIAN_430",
            "reused_saved_checkpoint": True,
            "checkpoint_path": str(ckpt_v3b_path),
            "checkpoint_sha256": sha256_file(ckpt_v3b_path),
        }
    else:
        m_v3a_430, tel_v3a_430 = train_matched_430_steps_strict_train(
            strict_train_cids=strict_train_cids,
            pools_dict=pools_v3a,
            ctx_scaled_st=ctx_scaled_st,
            sos_by_cid=sos_st_v3a,
            init_state_dict=r1_sd,
            device=device,
            script_t0=script_t0,
            epochs=10,
            batch_size=24,
            seed=20265000,
            tag="WINNER_POOL_V3A_Q25_430",
        )
        m_v3b_430, tel_v3b_430 = train_matched_430_steps_strict_train(
            strict_train_cids=strict_train_cids,
            pools_dict=pools_v3b,
            ctx_scaled_st=ctx_scaled_st,
            sos_by_cid=sos_st_v3b,
            init_state_dict=r1_sd,
            device=device,
            script_t0=script_t0,
            epochs=10,
            batch_size=24,
            seed=20265000,
            tag="WINNER_POOL_V3B_MEDIAN_430",
        )
        torch.save({k: v.detach().cpu().clone() for k, v in m_v3a_430.state_dict().items()}, ckpt_v3a_path)
        torch.save({k: v.detach().cpu().clone() for k, v in m_v3b_430.state_dict().items()}, ckpt_v3b_path)
        tel_v3a_430["checkpoint_path"] = str(ckpt_v3a_path)
        tel_v3a_430["checkpoint_sha256"] = sha256_file(ckpt_v3a_path)
        tel_v3b_430["checkpoint_path"] = str(ckpt_v3b_path)
        tel_v3b_430["checkpoint_sha256"] = sha256_file(ckpt_v3b_path)

    zero_ctx = np.zeros(584, dtype=np.float32)
    val_rows = []
    hold_rows = []

    for base_arm in ("STATIC", "SHUFFLED", "W1"):
        for cid in dl_val_cids:
            check_wall_deadline(script_t0, f"eval_{base_arm}_val_{cid[:8]}", reserve_s=25.0)
            p = pools_frozen[cid]
            val_rows.append(
                evaluate_episode_rollout_context(cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm)
            )
        for cid in dl_holdout_cids:
            check_wall_deadline(script_t0, f"eval_{base_arm}_hold_{cid[:8]}", reserve_s=25.0)
            p = pools_frozen[cid]
            hold_rows.append(
                evaluate_episode_rollout_context(cid, p, zero_ctx, np.zeros((len(p["pids"]), 2), dtype=np.float32), None, device, base_arm)
            )

    eval_configs = [
        ("BASELINE_FROZEN_POOL_TRAIN_ONLY_430", m_base_430, ctx_scaled_st, sos_st_frozen),
        ("WINNER_POOL_V3A_Q25_430", m_v3a_430, ctx_scaled_st, sos_st_v3a),
        ("WINNER_POOL_V3A_WEIGHTS_ONLY_430", m_v3a_430, ctx_scaled_st, sos_st_frozen),
        ("WINNER_POOL_V3B_MEDIAN_430", m_v3b_430, ctx_scaled_st, sos_st_v3b),
        ("WINNER_POOL_V3B_WEIGHTS_ONLY_430", m_v3b_430, ctx_scaled_st, sos_st_frozen),
    ]

    for arm_name, m_obj, c_scaled, s_map in eval_configs:
        for cid in dl_val_cids:
            check_wall_deadline(script_t0, f"eval_{arm_name}_val_{cid[:8]}", reserve_s=25.0)
            p = pools_frozen[cid]
            r = evaluate_episode_rollout_context(
                cid, p, c_scaled[p["rec"]["index"]], s_map[cid], m_obj, device, "ROUND_2",
                enable_struct_kernel=True, enable_cross_attn=True, enable_ucb=False,
            )
            r["arm"] = arm_name
            val_rows.append(r)
        for cid in dl_holdout_cids:
            check_wall_deadline(script_t0, f"eval_{arm_name}_hold_{cid[:8]}", reserve_s=25.0)
            p = pools_frozen[cid]
            r = evaluate_episode_rollout_context(
                cid, p, c_scaled[p["rec"]["index"]], s_map[cid], m_obj, device, "ROUND_2",
                enable_struct_kernel=True, enable_cross_attn=True, enable_ucb=False,
            )
            r["arm"] = arm_name
            hold_rows.append(r)

    arm_list = [
        "STATIC",
        "SHUFFLED",
        "W1",
        "BASELINE_FROZEN_POOL_TRAIN_ONLY_430",
        "WINNER_POOL_V3A_Q25_430",
        "WINNER_POOL_V3A_WEIGHTS_ONLY_430",
        "WINNER_POOL_V3B_MEDIAN_430",
        "WINNER_POOL_V3B_WEIGHTS_ONLY_430",
    ]
    ref_arms = (
        "STATIC",
        "SHUFFLED",
        "W1",
        "BASELINE_FROZEN_POOL_TRAIN_ONLY_430",
    )
    val_summary = compute_component_bootstrap_summary(val_rows, arm_list, ref_arms)
    hold_summary = compute_component_bootstrap_summary(hold_rows, arm_list, ref_arms)

    return {
        "evaluation_status": "COMPLETED",
        "frozen_budget_protocol": {
            "epistemic_classification": "development_learning_loop_validation (development_diagnostic; reused evaluation splits DL-Val 91 and DL-Holdout 93 with frozen evaluation pools; single evaluation pass, no repeated selection on holdout)",
            "initial_checkpoint_path": str(CHECKPOINTS_DIR / "flywheel_round_1_full_corpus.pt"),
            "initial_checkpoint_sha256": sha256_file(CHECKPOINTS_DIR / "flywheel_round_1_full_corpus.pt"),
            "train_cohort": "DL-Train with source_split == 'train'",
            "train_context_count": len(strict_train_cids),
            "epochs": 10,
            "batch_size": 24,
            "optimizer_steps": 430,
            "candidate_slate_cap": 192,
            "use_cv_test_gate": False,
            "enable_evo_queue": False,
            "eval_enable_ucb": False,
            "seed": 20265000,
        },
        "training_telemetry": {
            "BASELINE_FROZEN_POOL_TRAIN_ONLY_430": tel_base_430,
            "WINNER_POOL_V3A_Q25_430": tel_v3a_430,
            "WINNER_POOL_V3B_MEDIAN_430": tel_v3b_430,
        },
        "dl_val_91_contexts_summary": val_summary,
        "dl_holdout_93_contexts_summary": hold_summary,
    }


def classify_contrast_status(c: dict) -> str:
    lo, hi = float(c["CI_central_98_75"][0]), float(c["CI_central_98_75"][1])
    mean_v = float(c["mean"])
    if hi < 0.0:
        return "statistically_significant_favorable_ci_below_zero"
    if lo > 0.0:
        return "statistically_significant_unfavorable_ci_above_zero"
    if mean_v < 0.0:
        return "inconclusive_ci_crosses_zero_point_estimate_favorable_negative"
    if mean_v > 0.0:
        return "inconclusive_ci_crosses_zero_point_estimate_unfavorable_positive"
    return "exact_zero_tie"


def build_dynamic_evidence_bound_conclusion(
    cohort_spec: dict,
    evolution_summary: dict,
    corpus_v3_manifest: dict,
    exp_results: dict,
) -> dict:
    """
    Build terminal conclusion strictly from measured cohort coverage and measured evaluation CIs.
    If `exp_results["evaluation_status"] != "COMPLETED"`, records matched rollout evaluation as
    `UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE` (never negative or positive) and makes zero unmeasured
    architectural claims.
    """
    n_aug_v3a = corpus_v3_manifest["revisions"]["v3a_strict_train_q25"]["augmented_context_count"]
    n_win_v3a = corpus_v3_manifest["revisions"]["v3a_strict_train_q25"]["admitted_winner_count"]
    n_aug_v3b = corpus_v3_manifest["revisions"]["v3b_strict_train_median"]["augmented_context_count"]
    n_win_v3b = corpus_v3_manifest["revisions"]["v3b_strict_train_median"]["admitted_winner_count"]

    base_summary = {
        "epistemic_classification": "development_learning_loop_validation (development_diagnostic; reused DL-Val 91 and DL-Holdout 93 cohorts with frozen evaluation pools; single evaluation pass)",
        "partial_cohort_coverage_qualification": {
            "coverage_status": "partial_strict_train_cohort_coverage",
            "contexts_completed": evolution_summary["contexts_completed"],
            "total_strict_train_contexts": 1016,
            "completed_fraction_of_1016_strict_train": round(evolution_summary["contexts_completed"] / 1016.0, 5),
            "unique_families_covered": cohort_spec["unique_families_in_selected_256"],
            "total_strict_train_families": cohort_spec["total_families_in_1016_strict_train"],
            "untested_strict_train_contexts": 1016 - evolution_summary["contexts_completed"],
        },
        "evolution_outcome_prose": (
            f"Across {evolution_summary['contexts_completed']}/1,016 strict-train contexts "
            f"({cohort_spec['unique_families_in_selected_256']}/{cohort_spec['total_families_in_1016_strict_train']} "
            f"DL-Train families, 0 overlap with DL-Val/DL-Holdout), Stage-1 evaluated "
            f"{evolution_summary['total_strat300_candidate_fits']} novel candidates on Stratified300JTT1 "
            f"({evolution_summary['total_strat300_gate_passes']} passed CV gate) and Stage-2 verified "
            f"{evolution_summary['total_full_jtt1_candidate_fits']} top candidates on Full-JTT1 "
            f"({evolution_summary['total_full_jtt1_gate_passing_novel_winners']} passed Full-JTT1 CV gate across "
            f"{evolution_summary['contexts_with_at_least_1_full_jtt1_gate_pass']} contexts: "
            f"{evolution_summary['new_winners_strictly_beating_incumbent_best']} beat incumbent best across "
            f"{evolution_summary['contexts_with_at_least_1_incumbent_best_beat']} contexts, "
            f"{evolution_summary['new_winners_strictly_beating_q25']} beat Q25 across "
            f"{evolution_summary['contexts_with_at_least_1_q25_beat']} contexts, and "
            f"{evolution_summary['new_winners_strictly_beating_median']} beat Median across "
            f"{evolution_summary['contexts_with_at_least_1_median_beat']} contexts)."
        ),
    }

    if exp_results.get("evaluation_status") != "COMPLETED":
        base_summary["matched_430_step_evaluation_status"] = "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE"
        base_summary["matched_430_step_findings"] = (
            "Matched 430-step rollout evaluation on DL-Val and DL-Holdout did not complete before the "
            "finite wall-clock deadline expired; rollout outcome is therefore recorded strictly as UNKNOWN "
            "(neither positive nor negative/null)."
        )
        return base_summary

    val_s = exp_results["dl_val_91_contexts_summary"]
    hold_s = exp_results["dl_holdout_93_contexts_summary"]

    v3a_vs_base_val = val_s["contrasts"]["WINNER_POOL_V3A_Q25_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]
    v3a_vs_base_hold = hold_s["contrasts"]["WINNER_POOL_V3A_Q25_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]
    v3b_vs_base_val = val_s["contrasts"]["WINNER_POOL_V3B_MEDIAN_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]
    v3b_vs_base_hold = hold_s["contrasts"]["WINNER_POOL_V3B_MEDIAN_430_minus_BASELINE_FROZEN_POOL_TRAIN_ONLY_430"]

    base_summary["matched_430_step_evaluation_status"] = "COMPLETED"
    base_summary["v3a_q25_vs_baseline_status"] = {
        "dl_val_classification": classify_contrast_status(v3a_vs_base_val),
        "dl_holdout_classification": classify_contrast_status(v3a_vs_base_hold),
        "dl_val_contrast": v3a_vs_base_val,
        "dl_holdout_contrast": v3a_vs_base_hold,
        "augmented_contexts": n_aug_v3a,
        "admitted_winners": n_win_v3a,
    }
    base_summary["v3b_median_vs_baseline_status"] = {
        "dl_val_classification": classify_contrast_status(v3b_vs_base_val),
        "dl_holdout_classification": classify_contrast_status(v3b_vs_base_hold),
        "dl_val_contrast": v3b_vs_base_val,
        "dl_holdout_contrast": v3b_vs_base_hold,
        "augmented_contexts": n_aug_v3b,
        "admitted_winners": n_win_v3b,
    }
    return base_summary


def publish_phase7_artifacts_and_terminal_manifest_v3() -> dict:
    """
    Upload `corpus_v3_manifest.json` and `flywheel_wider_train_extension_v3_report.json` to S3,
    verify their `HeadObject` + `GetObject` SHA-256 readbacks, and publish `terminal_manifest_v3_20261024.json`
    without `--delete` and without repeating already-matched downloads from prior terminal manifests.
    """
    env = os.environ.copy()
    env["AWS_PROFILE"] = AWS_PROFILE

    new_uploads = [
        (CORPUS_V3_MANIFEST_PATH, "corpus_v3_wider_strict_train_256/corpus_v3_manifest.json"),
        (PHASE7_REPORT_PATH, f"results/{PHASE7_REPORT_PATH.name}"),
    ]
    newly_verified = []
    for local_p, rel_key in new_uploads:
        s3_key = f"{S3_PREFIX}/{rel_key}"
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        subprocess.run(
            ["aws", "s3", "cp", str(local_p), s3_uri, "--only-show-errors"],
            env=env,
            check=True,
            timeout=30,
        )
        head = json.loads(
            subprocess.check_output(
                ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", s3_key],
                env=env,
                text=True,
                timeout=20,
            )
        )
        rb_bytes = subprocess.check_output(["aws", "s3", "cp", s3_uri, "-"], env=env, timeout=30)
        l_sha = sha256_file(local_p)
        r_sha = hashlib.sha256(rb_bytes).hexdigest()
        assert l_sha == r_sha
        newly_verified.append({
            "rel_key": rel_key,
            "s3_uri": s3_uri,
            "local_size_bytes": local_p.stat().st_size,
            "remote_size_bytes": int(head.get("ContentLength", -1)),
            "local_sha256": l_sha,
            "remote_sha256_readback": r_sha,
            "s3_version_id": head.get("VersionId"),
            "s3_etag": (head.get("ETag") or "").strip('"'),
            "matched": True,
        })

    ver_raw = subprocess.check_output(
        [
            "aws", "s3api", "list-object-versions",
            "--bucket", S3_BUCKET,
            "--prefix", S3_PREFIX,
            "--output", "json",
        ],
        env=env,
        text=True,
        timeout=30,
    )
    ver_doc = json.loads(ver_raw)
    all_versions = ver_doc.get("Versions") or []
    delete_markers = ver_doc.get("DeleteMarkers") or []

    term_v2_doc = json.loads(TERMINAL_V2_PATH.read_text())

    terminal_v3 = {
        "schema": "jope.flywheel-external-terminal-manifest/v3",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bucket": S3_BUCKET,
        "prefix": S3_PREFIX,
        "parent_terminal_manifest_v2_path": str(TERMINAL_V2_PATH),
        "parent_terminal_manifest_v2_sha256": sha256_file(TERMINAL_V2_PATH),
        "s3_total_object_versions": len(all_versions),
        "s3_delete_markers_count": len(delete_markers),
        "s3_delete_markers": delete_markers,
        "reused_verified_objects_from_terminal_v2_count": (
            term_v2_doc["reused_verified_objects_from_terminal_v1_count"]
            + len(term_v2_doc["newly_verified_phase6_objects"])
            + 1
        ),
        "newly_verified_phase7_objects": newly_verified,
        "all_canonical_objects_verified_matched": True,
        "s3_version_inventory": [
            {
                "Key": v.get("Key"),
                "VersionId": v.get("VersionId"),
                "IsLatest": v.get("IsLatest"),
                "Size": v.get("Size"),
                "ETag": (v.get("ETag") or "").strip('"'),
                "LastModified": v.get("LastModified"),
            }
            for v in all_versions
        ],
    }
    TERMINAL_V3_PATH.write_text(json.dumps(terminal_v3, indent=2) + "\n")

    term_v3_s3_key = f"{S3_PREFIX}/{TERMINAL_V3_PATH.name}"
    term_v3_s3_uri = f"s3://{S3_BUCKET}/{term_v3_s3_key}"
    subprocess.run(
        ["aws", "s3", "cp", str(TERMINAL_V3_PATH), term_v3_s3_uri, "--only-show-errors"],
        env=env,
        check=True,
        timeout=30,
    )
    t_head = json.loads(
        subprocess.check_output(
            ["aws", "s3api", "head-object", "--bucket", S3_BUCKET, "--key", term_v3_s3_key],
            env=env,
            text=True,
            timeout=20,
        )
    )
    t_rb = subprocess.check_output(["aws", "s3", "cp", term_v3_s3_uri, "-"], env=env, timeout=30)
    t_lsha = sha256_file(TERMINAL_V3_PATH)
    t_rsha = hashlib.sha256(t_rb).hexdigest()
    assert t_lsha == t_rsha

    receipt = {
        "terminal_manifest_v3_local_path": str(TERMINAL_V3_PATH),
        "terminal_manifest_v3_s3_uri": term_v3_s3_uri,
        "size_bytes": TERMINAL_V3_PATH.stat().st_size,
        "local_sha256": t_lsha,
        "remote_sha256_readback": t_rsha,
        "s3_version_id": t_head.get("VersionId"),
        "s3_etag": (t_head.get("ETag") or "").strip('"'),
        "matched": True,
        "verified_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    TERMINAL_V3_RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    script_t0 = time.monotonic()
    script_cpu_t0 = time.process_time()
    reuse_ckpt = "--resume-eval-from-saved-checkpoints" in sys.argv
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[Phase 7] Starting Bounded 256-Context Strict-Train Extension on {device} "
        f"(CPU soft cap={GLOBAL_CPU_GUARD.soft_cap_pct}% <= 50%, hard cap={GLOBAL_CPU_GUARD.hard_cap_pct}%, "
        f"wall stop={FINITE_WALL_STOP_S}s, reuse_ckpt={reuse_ckpt})..."
    )

    step1_audit = verify_authority_and_immutable_artifacts()
    manifest, champ_bank, ctx_raw_584, _h30_scales, pools_frozen = load_all_pools_and_features()
    ctx_npz = np.load(CORPUS_DIR / "context_features.npz")
    lm48_all = ctx_npz["landmark_v2_48"]

    selected_256_records, cohort_spec = select_deterministic_256_strict_train_cohort(manifest, pools_frozen)

    phase7_new_winners, evolution_summary = run_bounded_256_context_winner_generation(
        selected_256_records=selected_256_records,
        pools_frozen=pools_frozen,
        champ_bank=champ_bank,
        lm48_all=lm48_all,
        script_t0=script_t0,
    )

    pools_v3a, pools_v3b, corpus_v3_manifest = build_versioned_v3_strict_train_pools(
        pools_frozen=pools_frozen,
        phase7_new_winners=phase7_new_winners,
        evolution_summary=evolution_summary,
        cohort_spec=cohort_spec,
    )

    try:
        exp_results = run_matched_v3_acceptance_test(
            pools_frozen=pools_frozen,
            pools_v3a=pools_v3a,
            pools_v3b=pools_v3b,
            ctx_raw_584=ctx_raw_584,
            device=device,
            script_t0=script_t0,
            reuse_saved_checkpoints=reuse_ckpt,
        )
    except DeadlineExpiredError as exc:
        print(f"[Phase 7] {exc}; recording matched evaluation status as UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE.")
        exp_results = {
            "evaluation_status": "UNKNOWN_NOT_COMPLETED_BEFORE_DEADLINE",
            "deadline_expired_reason": str(exc),
        }

    n_aug_v3a = corpus_v3_manifest["revisions"]["v3a_strict_train_q25"]["augmented_context_count"]
    n_win_v3a = corpus_v3_manifest["revisions"]["v3a_strict_train_q25"]["admitted_winner_count"]
    n_aug_v3b = corpus_v3_manifest["revisions"]["v3b_strict_train_median"]["augmented_context_count"]
    n_win_v3b = corpus_v3_manifest["revisions"]["v3b_strict_train_median"]["admitted_winner_count"]

    total_wall_s = round(time.monotonic() - script_t0, 2)
    total_cpu_s = round(time.process_time() - script_cpu_t0, 2)
    cpu_guard_summary = GLOBAL_CPU_GUARD.summary()
    post_resources = sample_host_resources()

    programmatic_terminal_conclusion = build_dynamic_evidence_bound_conclusion(
        cohort_spec=cohort_spec,
        evolution_summary=evolution_summary,
        corpus_v3_manifest=corpus_v3_manifest,
        exp_results=exp_results,
    )

    resource_and_cost_accounting = {
        "total_wall_seconds": total_wall_s,
        "finite_wall_stop_cap_seconds": FINITE_WALL_STOP_S,
        "within_finite_wall_stop": bool(total_wall_s <= FINITE_WALL_STOP_S),
        "parent_python_process_cpu_seconds": total_cpu_s,
        "cpu_throttle_guard": cpu_guard_summary,
        "whole_host_cpu_soft_target_50pct_satisfied": bool(cpu_guard_summary["mean_host_cpu_pct"] <= 50.0),
        "whole_host_cpu_hard_ceiling_85pct_satisfied": bool(cpu_guard_summary["peak_host_cpu_pct"] <= 85.0),
        "post_run_host_resources": post_resources,
        "s3_and_compute_cost_accounting": {
            "new_cloud_instances_or_paid_apis_provisioned": 0,
            "incremental_compute_spend_usd": 0.0,
            "s3_put_requests": 3,
            "s3_get_head_list_requests": 7,
            "estimated_s3_request_and_storage_cost_usd": 0.0001,
        },
    }

    phase7_report = {
        "schema": "jope.flywheel-wider-train-extension-v3-report/v1",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resource_and_cost_accounting": resource_and_cost_accounting,
        "step1_authority_and_immutable_artifacts_audit": step1_audit,
        "step2_cohort_256_selection_spec": cohort_spec,
        "step3_bounded_256_context_evolution_summary": evolution_summary,
        "step4_corpus_v3_manifest_summary": {
            "corpus_v3_manifest_path": str(CORPUS_V3_MANIFEST_PATH),
            "corpus_v3_manifest_sha256": sha256_file(CORPUS_V3_MANIFEST_PATH),
            "combined_strict_train_winner_union_summary": corpus_v3_manifest["combined_strict_train_winner_union_summary"],
            "v3a_augmented_context_count": n_aug_v3a,
            "v3a_admitted_winner_count": n_win_v3a,
            "v3b_augmented_context_count": n_aug_v3b,
            "v3b_admitted_winner_count": n_win_v3b,
        },
        "step5_matched_430_step_acceptance_test": exp_results,
        "step6_programmatic_terminal_conclusion": programmatic_terminal_conclusion,
    }

    PHASE7_REPORT_PATH.write_text(json.dumps(phase7_report, indent=2) + "\n")
    term_v3_receipt = publish_phase7_artifacts_and_terminal_manifest_v3()
    print(
        f"[Phase 7] Finished in {total_wall_s:.2f}s (eval_status={exp_results.get('evaluation_status')}).\n"
        f"  Terminal Manifest V3 SHA256={term_v3_receipt['local_sha256']} VersionId={term_v3_receipt['s3_version_id']}"
    )


if __name__ == "__main__":
    main()
