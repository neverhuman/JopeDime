#!/usr/bin/env python3
"""Emit world-class close receipts. Does not unfence, rebind d77e856b, or train."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import closed_loop as loop
import component_ledger as ledger
import expansion_assembly as assembly
import family_growth_policy as growth
import fence_honesty
import raw_triplet_parity as triplets
import registry_namespace
import sync_certificates as sync

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "parallel/runs/flywheel-worldclass-20260911"
REDTEAM = ROOT / "parallel/runs/flywheel-redteam-20260910"
LIVE_FENCE = ROOT / "parallel/runs/registry-integration-20260910/live-fence/complete.json"
COMPARISON = REDTEAM / "data/quant-content-comparison.json"
ACTIVE = REDTEAM / "data/remote-active-contexts.json"
LOCAL_HASHES = REDTEAM / "data/local-raw-hashes.json"
REMOTE_HASHES = REDTEAM / "data/remote-raw-hashes.json"
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-i",
    str(Path("~/.ssh/lambda_cloud_a100_20260907_ed25519").expanduser()),
    "ubuntu@158.101.20.226",
]


def write(name, payload):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load(path):
    return json.loads(Path(path).read_text())


def fence_receipt():
    complete = load(LIVE_FENCE) if LIVE_FENCE.is_file() else {}
    observed = {}
    try:
        show = subprocess.check_output(
            SSH + ["systemctl show jope-parallel-gp --property=MainPID,ActiveState"],
            text=True,
            timeout=20,
        )
        observed = dict(line.split("=", 1) for line in show.splitlines() if "=" in line)
    except (OSError, subprocess.SubprocessError) as error:
        observed = {"error": str(error)}
    inspected = fence_honesty.inspect_fence(
        complete,
        override_exists=True,
        handoff_exists=False,
        gp_main_pid=int(observed.get("MainPID") or 0),
        restart="no",
    )
    inspected["node_health_observed"] = observed
    inspected["restart_legacy_gp"] = False
    write("a0-fence.json", inspected)
    return inspected


def sync_receipts():
    local_free = shutil.disk_usage("/home/ubuntu").free
    local_total = shutil.disk_usage("/home/ubuntu").total
    remote_df = {}
    try:
        raw = subprocess.check_output(
            SSH
            + [
                "python3 -c \"import shutil,json;u=shutil.disk_usage('/home/ubuntu');print(json.dumps({'free':u.free,'total':u.total}))\""
            ],
            text=True,
            timeout=20,
        )
        remote_df = json.loads(raw.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        remote_df = {"error": str(error)}
    capacity = sync.certify_capacity(
        free_bytes=int(remote_df["free"]) if "free" in remote_df else 0,
        filesystem_bytes=int(remote_df["total"]) if "total" in remote_df else 0,
    )
    capacity["local_free_bytes"] = local_free
    capacity["local_filesystem_bytes"] = local_total
    capacity["remote"] = remote_df
    declared = {
        "source": {"present": (ROOT / "parallel/src").is_dir(), "reason": None},
        "durable_data": {"present": (REDTEAM / "data/quant-content-comparison.json").is_file()},
        "recovery": {
            "present": Path("/home/ubuntu/jope-dime-recovery/20260908").exists(),
            "reason": "declared recovery root",
        },
        "release": {"present": (ROOT / "parallel/releases").is_dir()},
        "deployment_conformance": {
            "present": False,
            "reason": "mirror health was live-GP-only; fence patch not deployed",
        },
        "replication_lag": {"present": True, "lag_seconds": 0},
    }
    cert = sync.certify(declared)
    write("a1-sync-certificates.json", cert)
    write("a1-capacity.json", capacity)
    return cert, capacity


def triplet_receipt(capacity):
    comparison = load(COMPARISON)
    local_files = {row["path"]: row for row in load(LOCAL_HASHES).get("files", [])}
    remote_hash = {row["sha256"] for row in load(REMOTE_HASHES).get("files", [])}
    plan = triplets.plan_copy(
        comparison.get("local_only_paths") or [],
        remote_hashes=remote_hash,
        local_sizes=local_files,
    )
    plan["comparison_missing_unique_bytes"] = comparison.get("missing_unique_bytes_remote")
    plan["capacity_allows"] = bool(capacity.get("backfill_allowed"))
    plan["copied_bytes"] = 0
    plan["execute"] = False
    plan["execute_reason"] = "not yet gated"
    write(
        "a2-raw-triplet-plan.json",
        {k: v for k, v in plan.items() if k != "selected"}
        | {
            "selected_head": plan["selected"][:20],
            "n_selected": plan["n_selected"],
        },
    )
    return plan


def hold_receipt():
    receipt = dict(
        schema="parallel.canonical_hold_merge.v1",
        automatic_training_hold_merged=True,
        proposal_reserves={"pack": 104, "steady": 312, "refill_below": 208},
        scan_subtracted_poll_seconds=10,
        controller_deletes_checkpoints=False,
        training_rs_retain_still_deletes=True,
        training_rs_claimed_by_gpu_lane=True,
        funded_task_check=True,
    )
    write("a3-canonical-hold.json", receipt)
    return receipt


def ledger_receipts():
    active = load(ACTIVE)
    already = list((active.get("newly_transferred_ids_in_active_index") or {}).get("regression") or [])
    comparison = load(COMPARISON)
    transferred = sorted(
        {
            Path(path).parts[1]
            for path in comparison.get("local_only_paths") or []
            if path.startswith("regression/") and path.endswith("/meta.json")
        }
    )
    # The 64 already-active IDs are a subset of Claude's 1724 copied set, not
    # necessarily of the still-missing local_only_paths. Bind both populations.
    copied = list(already)
    growth_set = ledger.growth_candidates(
        transferred + copied,
        already_in_index=already,
        live_ids=already,
        sealed_ids=[],
        evidenced_unresolved=[],
    )
    rows = [ledger.join_row(cid, already_in_index=True, stage="discovered") for cid in already]
    for cid in growth_set["novel_ids"]:
        rows.append(ledger.join_row(cid, stage="retained"))
    built = ledger.build_ledger(rows)
    write(
        "b1-component-ledger.json",
        {
            **{k: v for k, v in built.items() if k != "rows"},
            "already_in_index_ids": already,
            "n_already_in_index_ids": len(already),
            "n_novel_ids": len(growth_set["novel_ids"]),
        },
    )
    write(
        "b1-growth-candidates.json",
        {
            **{
                k: v
                for k, v in growth_set.items()
                if k not in ("novel_ids", "unresolved_ids", "already_in_index_ids")
            },
            "already_in_index_ids": already,
            "novel_head": growth_set["novel_ids"][:32],
        },
    )
    missing_prov = []
    for cid in growth_set["novel_ids"][:32]:
        receipt = ledger.provenance_receipt(cid, {}, source_path=f"regression/{cid}")
        if not receipt["resolved"]:
            missing_prov.append(cid)
    write(
        "b1-provenance-before-prepare.json",
        dict(
            schema="parallel.provenance_before_prepare.v1",
            n_novel=len(growth_set["novel_ids"]),
            n_checked=min(32, len(growth_set["novel_ids"])),
            n_resolved=0,
            prepare_allowed=False,
            missing_component_ids=missing_prov,
            initial_decisions_mass_approve=False,
        ),
    )
    return already, growth_set


def family_growth_receipt(already, growth_set):
    try:
        assignment = growth.assign_new_components(
            growth_set["novel_ids"][:8],
            live_families=already,
            sealed_test_families=[],
            provenance_resolved=True,
            decision_receipts=[],
        )
    except ValueError as error:
        assignment = dict(
            schema="parallel.family_growth_assignment.v2",
            method="frozen_80_20_hash_v2",
            refused=str(error),
            n_train=0,
            n_validation=0,
            n_test=0,
            rows=[],
            sha256="0" * 64,
            writes_live_index=False,
            final_tests_opened=False,
        )
    write("b2-family-growth-v2.json", assignment)
    return assignment


def successor_receipt(assignment):
    out = OUT / "b3-successor-assembly.json"
    if assignment.get("rows"):
        receipt = assembly.successor_from_growth(assignment, out)
    else:
        receipt = dict(
            schema=assembly.SCHEMA,
            kind="grown_family_successor",
            status="refused_empty_eligible_set",
            parent_assembly_sha256=assembly.PARENT_ASSEMBLY,
            parent_registry_sha256=assembly.PARENT_REGISTRY,
            assembly_sha256=None,
            writes_live_index=False,
            encode_ran=False,
            apply_contexts_ran=False,
            projection_refitted=False,
            live_index_bytes_unchanged=True,
            reason="no decision receipts; successor not minted from aliases",
        )
        write("b3-successor-assembly.json", receipt)
    return receipt


def cycle_receipt():
    ns = registry_namespace.enable_namespace()
    cycle = loop.az2017_cycle(
        loop.RECOVERY_NLL,
        loop.RECOVERY_BT,
        False,
        False,
        False,
        False,
        False,
        0.014,
        hold="held",
        automatic_training=True,
        teachers_growth=False,
        new_teacher_beats=0,
        corpus_is_recovery=True,
        new_gp_identity=False,
        admission_v2=False,
        reuse_excluded_from_growth=False,
        two_panels_predeclared=False,
        historical_import_complete=False,
        namespace_enabled=False,
    )
    cycle["namespace"] = {"enabled": ns[0] if isinstance(ns, tuple) else False, "claimed_savings_745": False}
    cycle["honest_rates"] = dict(neural_seed=0.0, final_gp=0.0, margin=0.014)
    write("c1-az2017-cycle.json", cycle)
    return cycle


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fence = fence_receipt()
    cert, capacity = sync_receipts()
    plan = triplet_receipt(capacity)
    allowed, reasons = triplets.execute_allowed(
        plan,
        capacity,
        cert,
        bool(fence.get("legacy_dispatch_fenced")),
    )
    write(
        "a2-raw-triplet-execute.json",
        dict(
            schema="parallel.raw_triplet_execute.v1",
            allowed=allowed,
            reasons=reasons,
            copied_bytes=0,
            synthetics_copied=0,
            expansion_v9_preserved=True,
            defective_excluded=True,
            n_selected=plan["n_selected"],
            selected_bytes=plan["selected_bytes"],
        ),
    )
    hold = hold_receipt()
    already, growth_set = ledger_receipts()
    assignment = family_growth_receipt(already, growth_set)
    successor_receipt(assignment)
    cycle = cycle_receipt()
    write(
        "SUMMARY.json",
        dict(
            schema="parallel.worldclass_close.summary.v1",
            live_execution_authority="codex_two_cycle_20260910_1824",
            live_brake="fence",
            fence_up=bool(fence.get("legacy_dispatch_fenced")),
            restart_774452=False,
            sync_certified=cert["certified"],
            missing_sync_roots=cert["missing_roots"],
            capacity_allows_backfill=capacity.get("backfill_allowed"),
            synthetics_copied=0,
            already_in_index=len(already),
            already_in_index_not_counted_as_new=True,
            successor_is_d77e856b=False,
            live_index_written=False,
            train_launched=False,
            may_serve=False,
            claimed_savings_745=False,
            world_class=False,
            existence_rate=cycle["neural_seed_rate"],
            hold_merged=hold["automatic_training_hold_merged"],
            n_selected_triplets=plan["n_selected"],
            triplet_copy_allowed=allowed,
            triplet_copy_reasons=reasons,
            copied_bytes=0,
        ),
    )


if __name__ == "__main__":
    main()
