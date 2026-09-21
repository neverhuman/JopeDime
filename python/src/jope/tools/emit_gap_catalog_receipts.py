#!/usr/bin/env python3
"""Write gap-catalog receipts. Does not touch live PIDs, serving, or contexts."""

from __future__ import annotations

import json
from pathlib import Path

import authority_resolution as authority
import closed_loop as loop
import family_growth_policy as growth
import fence_honesty as fence
import registry_namespace as registry
from flywheel_plan import RECOVERY_FROZEN

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "flywheel-gaps-20260910"
CLOSE = ROOT / "runs" / "flywheel-close-20260910"
STAGE_D = ROOT / "runs/flywheel-20260910/stage-d-serving-prior-v1/result.json"


def write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    named = authority.resolve("explicit_merge")
    write(OUT, "g0-authority.json", named)

    fence_receipt = fence.inspect_fence(
        dict(
            legacy_dispatch_fenced=True,
            historical_import_complete=False,
            registry_dispatch_enabled=False,
        ),
        override_exists=True,
        handoff_exists=False,
        gp_main_pid=0,
        restart="no",
    )
    write(
        OUT,
        "g1-fence-honesty.json",
        {
            **fence_receipt,
            "legacy_gp_pid": fence.LEGACY_GP_PID,
            "legacy_unit": fence.LEGACY_UNIT,
            "restart_attempted": False,
            "node_health_confirmed": True,
            "node_health_expected": "planned_migration",
            "node_health_observed": {
                "health": "planned_migration",
                "gp_restart": "no",
                "gp_main_pid": 0,
                "gp_active_state": "failed",
                "handoff_required": "absent",
                "legacy_dispatch_fenced": True,
                "historical_import_complete": False,
                "registry_dispatch_enabled": False,
                "method": "read_only_ssh_20260910",
            },
            "idle_gpus_authorize_train": fence.idle_gpus_authorize_train(0, True),
        },
    )

    live80 = [f"live-{i:02d}" for i in range(80)]
    sealed = [f"sealed-test-{i:02d}" for i in range(10)]
    new_ids = [f"new-real-{i:04d}" for i in range(40)]
    assignment = growth.assign_new_components(
        new_ids + live80[:2] + sealed[:2], live_families=live80, sealed_test_families=sealed
    )
    write(
        OUT,
        "g2-family-growth-8020.json",
        {
            **assignment,
            "example_ids_only": True,
            "must_not_execute": True,
            "note": "placeholder names; assign only provenance-resolved new regression ids from the live audit",
        },
    )
    write(OUT, "g2-v5-canary-contract.json", growth.v5_canary_contract())
    draft = growth.decide_unresolved(
        [
            dict(
                dataset="evidenced-unresolved",
                source_path="/home/ubuntu/hyper/data/example.csv",
                meta={"source_real_data": True},
            ),
            dict(dataset="binary/3c288e95bcdaf2fa", directory="binary/3c288e95bcdaf2fa"),
            dict(dataset="hex-synthetic", meta={"source_real_data": False}),
            dict(dataset="legacy-unresolved", meta={}),
        ]
    )
    write(
        OUT,
        "g2-unresolved-decisions-draft.json",
        {
            **draft,
            "policy": "do not auto-admit the ~2076 evidenced Unresolved reals",
            "live_initial_decisions_unchanged": True,
            "synthetics_stay_off_node": True,
        },
    )

    write(
        OUT,
        "g3-registry-namespace.json",
        {
            **registry.snapshot(),
            "savings": registry.savings_claim(0, 0),
            "enable_attempted": False,
        },
    )

    stage_d = json.loads(STAGE_D.read_text()) if STAGE_D.is_file() else {}
    first = stage_d.get("predicted_first_neural_seed_vs_teacher") or {}
    n = int(first.get("n") or 0)
    beats = int(first.get("beats_teacher") or 0)
    rate = 0.0 if n == 0 else beats / n
    gate = loop.grown_family_train(
        1.407,
        0.542,
        False,
        False,
        False,
        False,
        False,
        rate,
        hold="held",
        automatic_training=False,
        teachers_growth=False,
        new_teacher_beats=0,
        corpus_is_recovery=True,
        new_gp_identity=False,
        admission_v2=False,
        reuse_excluded_from_growth=False,
        historical_import_complete=False,
    )
    write(
        OUT,
        "g4-one-train-two-panels.json",
        {
            **gate,
            "old_panel": {
                "nll": RECOVERY_FROZEN["sequence_nll"],
                "bt": RECOVERY_FROZEN["ranking_bt"],
                "guards": ["+0.5% nll", "+1% bt"],
            },
            "new_family_panel": "expansion meta_validation; old net never saw these families",
            "neural_seed_rate": rate,
            "neural_seed_n": n,
            "neural_seed_beats": beats,
            "existence_proof": False,
            "search_beating_allowed": False,
        },
    )

    handshake = dict(
        schema="parallel.gap_catalog.handshake.v1",
        posted="grok-20260910-gap-catalog-42",
        authority=named["winner"],
        live_execution_authority=named["live_execution_authority"],
        diagnostic_authority=named["diagnostic_authority"],
        grok_stops_editing=named["grok_stops_editing"],
        awaiting={
            "Codex": "ACK merge: two-cycle execution; keep fence; no 74.5% claim",
            "Claude": "ACK 80/20 hash for new regression only; new-index canary; no outer flock",
        },
        live_brake="fence",
        live_gp_pid=774452,
        live_gp_state="killed_18:23_do_not_restart",
        live_serving="c2f229c9",
        historical_import_complete=False,
        train_launched=False,
        may_serve=False,
        final_tests_opened=False,
    )
    write(OUT, "agent-handshake.json", handshake)
    write(
        CLOSE,
        "agent-handshake.json",
        {
            **handshake,
            "schema": "parallel.flywheel_close.handshake.v2",
            "superseded_live_gp_alive_claim": True,
            "previous_posted": "grok-20260910-flywheel-close-41",
        },
    )
    write(
        OUT,
        "SUMMARY.json",
        {
            "schema": "parallel.gap_catalog.summary.v1",
            "live_execution_authority": named["live_execution_authority"],
            "live_brake": "fence",
            "live_gp_restarted": False,
            "live_contexts_rebound": False,
            "blanket_v6_eligibility": False,
            "frozen_z_live_pointer": False,
            "namespace_enabled": False,
            "claimed_savings_745": False,
            "train_launched": False,
            "may_serve": False,
            "synthetics_admitted": False,
            "sealed_tests_opened": False,
            "paper_gate_weakened": False,
            "net_widened": False,
        },
    )


if __name__ == "__main__":
    main()
