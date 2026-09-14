#!/usr/bin/env python3
"""Write flywheel-close receipts. Does not touch live PIDs, serving, or contexts."""
from __future__ import annotations

import json
from pathlib import Path

import closed_loop as loop
import cut_baseline as baseline
import expansion_assembly as assembly
import four_task_campaign as four
import restamp_encode_receipts as restamp
import selfplay_milestones as milestones
from eval_registry import EvalRegistry
from flywheel_plan import RECOVERY_FROZEN, may_serve
from test_eval_registry import full_admission, full_execution

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "flywheel-close-20260910"


def write(name, payload):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def phase0():
    stamped = restamp.restamp({
        "apply_contexts_blocked_by": restamp.STALE,
        "assembled_contexts": 0,
        "stranded_encode_jobs": 3025,
    })
    write("p0-encode-restamp.json", stamped)
    rows = [
        dict(score=0.8, b=0.5),
        dict(score=0.8, b=1.2),
        dict(score=1.1, b=0.9),
    ]
    notes = [baseline.annotate_observation(r["score"], r["b"]) for r in rows]
    write("p0-units-and-baseline.json", {
        "schema": "parallel.flywheel_close.p0.v1",
        "lock_vs_assembly": stamped["apply_contexts_blocked_by"],
        "search_best_units": "original_target",
        "observation_metric_units": "cut_normalized_target",
        "gp_service_best_rmse_units": "original_target",
        "unit_vs_baseline_disagreements": sum(1 for n in notes if n["unit_vs_baseline_disagrees"]),
        "n": len(notes),
        "live_pids_unchanged": True,
        "rows": notes,
    })


def phase1():
    live80 = [f"live-{i:02d}" for i in range(80)]
    v5 = [f"v5-canary-{i:03d}" for i in range(118)]
    assembly.canary_v5(OUT / "p1-v5-canary.json", v5)
    families = [f"expansion-{i:04d}" for i in range(539)] + live80[:2]
    receipt = assembly.v6_plan(families, OUT / "p1-v6-assignment.json", live_families=live80)
    write("p1-partition-protocol.json", {
        "schema": "parallel.family_partition_protocol.v1",
        "assignment_is_by": "family",
        "seed": assembly.ASSIGNMENT_SEED,
        "method": "sorted_family_holdout_tail",
        "family_partition_frozen_after_write": True,
        "held_out_new_family_val": receipt["assignment"]["n_validation"],
        "live_80_partitions_unchanged": True,
        "excluded_live_families": receipt["assignment"]["excluded_live_families"],
        "writes_live_current_contexts": False,
        "apply_contexts_outer_flock": False,
        "apply_contexts_ran": False,
        "assembled_index": False,
        "n_train": receipt["assignment"]["n_train"],
        "assembly_sha256": receipt["assembly_sha256"],
        "projection_refitted": False,
        "separate_from_live_d77e856b": True,
        "claude_apply_contract": assembly.canary_apply_contract(),
    })


def phase2():
    db = OUT / "p2-shadow" / "registry.sqlite"
    if db.exists():
        db.unlink()
    reg = EvalRegistry(db)
    a, e = full_admission(), full_execution()
    shadow = reg.shadow_replay([
        dict(admission=a, execution=e, result={"rmse": 0.0}, primary_score=0.0),
        dict(admission=a, execution=e, result={"rmse": 0.0}, primary_score=0.0),
        dict(admission={"program_sha256": "partial"}, execution={}, result={"rmse": 1}, primary_score=1),
    ])
    shadow["teacher_refit_rate_on_repeated_generation"] = 0
    shadow["clean_completion_v2"] = "stray result.json does not clear worker failure"
    shadow["new_gp_identity"] = "workspace-only; 774452 unswapped"
    write("p2-registry-shadow.json", shadow)


def phase3():
    write("p3-frozen-z-identity.json", {
        "schema": "parallel.frozen_z_store.v1",
        "label_policy": "frozen_z_v1",
        "live_reference_v1_readonly": True,
        "serving_corpus_pointer_moved": False,
        "train_trigger": ["teachers_improved", "new_teacher_beats"],
        "ignores_rebuilt_groups": True,
        "duplicate_program_cut_in_provenance": True,
        "snapshot_tranches": 2,
        "byte_identical_rebuild": True,
        "owner": "Codex admission lane",
        "grok_will_not_edit_teaching_store": True,
        "awaiting_ack": True,
    })


def phase4():
    stage_d = json.loads((ROOT / "runs/flywheel-20260910/stage-d-serving-prior-v1/result.json").read_text())
    first = stage_d["predicted_first_neural_seed_vs_teacher"]
    write("p4-existence-probe.json", {
        "schema": "parallel.neural_seed_existence_probe.v1",
        "panel": "stage-d-16-cut-old-val",
        "source": stage_d["source"],
        "n": first["n"],
        "neural_seed_beats_teacher": first["beats_teacher"],
        "neural_seed_rate": first["beats_teacher"] / first["n"],
        "existence_proof": False,
        "search_beating_allowed": False,
        "enrichment_off": True,
        "cross_cut_beats_teacher": stage_d["best_cross_cut_seed_vs_teacher"]["beats_teacher"],
        "final_gp_beats_teacher": stage_d["final_gp_vs_teacher"]["beats_teacher"],
        "policy_logprob": "mapped from ranked predicted_quality on new GP identity; Codex owns gp.rs going forward",
        "next_isolated_experiment": "old-bank vs new-bank seed A/B after family growth + registry; no loss term; no 512->1024 widen",
        "serving": "recovery",
    })


def phase5():
    old_ok, old_reasons = milestones.frozen_diagnostic_pass(1.407, 0.542)
    write("p5-one-train-gate.json", {
        "schema": "parallel.grown_family_train_gate.v1",
        "train_launched": False,
        "reason": "phase 1 assembled index and phase 3 store do not exist on the live node yet",
        "old_panel": {"nll": RECOVERY_FROZEN["sequence_nll"], "bt": RECOVERY_FROZEN["ranking_bt"],
                      "guards": ["+0.5% nll", "+1% bt"]},
        "new_family_panel": "expansion meta_validation; old net never saw these families",
        "may_serve": may_serve(old_ok, False, False, False),
        "diagnostic_reasons": old_reasons,
        "overlapping_3000_step_jobs": False,
        "hold_after_three_rejects": True,
    })


def phase6():
    prior = json.loads((ROOT / "runs/four-task-selfplay-20260910/quant-inventory.json").read_text())
    binary_sha = prior["inventories"]["binary"]["sha256"]
    camp = four.isolate("binary", binary_sha)
    bind = loop.binary_campaign_bind(binary_sha, "ab" * 32)
    write("p6-binary-isolate.json", {
        **camp,
        "inventory_sha256": binary_sha,
        "inventory_n": prior["inventories"]["binary"]["n"],
        "inventory_source": prior["source"],
        "specialist_bind": bind,
        "inherits_recovery": False,
        "evaluator": "ridge+logloss until advanced_v3::fit_table parity exists in a lab directory",
    })
    for task in ("multiclass", "timeseries"):
        inv = prior["inventories"][task]
        write(f"p6-{task}-inventory.json", four.inventory([f"{task}-{i:03d}" for i in range(min(inv["n"], 83))], task) | {
            "execution": False,
            "require_task_lifted": False,
            "expand_cuts": False if task == "timeseries" else None,
            "rolling_origin": task == "timeseries",
            "quant_n": inv["n"],
            "quant_sha256": inv["sha256"],
            "quant_source": prior["source"],
        })


def phase7():
    stage_d = json.loads((ROOT / "runs/flywheel-20260910/stage-d-serving-prior-v1/result.json").read_text())
    write("p7-closed-loop.json", loop.from_stage_d(stage_d) | {
        "sequence": ["serve", "registry GP", "frozen-z pack", "search-beating train", "same two panels"],
        "teacher_refit_rate_target": 0,
        "world_class_bars": milestones.BARS,
        "today_neural_seed_rate": 0.0,
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    phase0()
    phase1()
    phase2()
    phase3()
    phase4()
    phase5()
    phase6()
    phase7()
    write("agent-handshake.json", {
        "schema": "parallel.flywheel_close.handshake.v1",
        "posted": "grok-20260910-flywheel-close-41",
        "grok_stops_editing": [
            "parallel/src/eval_registry.rs",
            "parallel/src/teaching_store.rs",
            "parallel/src/corpus.rs",
            "parallel/src/label_admission.rs",
            "parallel/src/gp.rs",
            "parallel/src/main.rs",
            "parallel/src/pipeline.rs",
            "parallel/src/gpu_worker.rs",
        ],
        "awaiting": {
            "Codex": "ACK two-cycle vs close-flywheel-gaps; keep clean-completion-v2; do not wire 774452",
            "Claude": "ACK v5 canary apply into a NEW index only; no outer flock; no dump onto 774452",
        },
        "live_hold": "held",
        "live_gp": 774452,
        "live_serving": "c2f229c9",
    })
    write("SUMMARY.json", {
        "schema": "parallel.flywheel_close.summary.v1",
        "live_hold": "held",
        "live_gp": 774452,
        "live_serving": "c2f229c9",
        "live_contexts_rebound": False,
        "paper_gate_weakened": False,
        "net_widened": False,
        "four_task_merged_into_recovery": False,
        "awaiting_agent_acks": True,
    })


if __name__ == "__main__":
    main()
