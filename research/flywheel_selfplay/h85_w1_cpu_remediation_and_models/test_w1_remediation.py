"""H85 unit tests: synthetic fixtures only (no external data payloads, GPU, or network)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w1_remediation as rem  # noqa: E402


def test_script_immutable_since_import_passes_and_fails_on_mutation(tmp_path: Path) -> None:
    info = rem.verify_script_immutable_since_import()
    assert info["sha256_verified"] == rem.SELF_SHA256_AT_IMPORT

    fake_script = tmp_path / "runner.py"
    fake_script.write_text("print('v1')\n", encoding="utf-8")
    sha_v1 = hashlib.sha256(fake_script.read_bytes()).hexdigest()
    assert rem.verify_script_immutable_since_import(fake_script, sha_v1)["sha256_verified"] == sha_v1

    fake_script.write_text("print('v2')\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mutated after import"):
        rem.verify_script_immutable_since_import(fake_script, sha_v1)


def test_shard_sidecar_verifies_shas_and_rejects_zero_cpu_seconds(tmp_path: Path) -> None:
    shard = tmp_path / "shard.pkl"
    shard.write_bytes(b"model-bytes")
    out_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    meta_path = tmp_path / "meta.json"
    meta_payload = {
        "output_pkl_sha256": out_sha,
        "input_npz_sha256": "in_sha_123",
        "worker_py_sha256": "wk_sha_456",
        "process_cpu_seconds": 142.5,
        "wall_seconds": 38.2,
    }
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    verified = rem.verify_shard_sidecar(
        shard,
        meta_path,
        expected_input_sha="in_sha_123",
        expected_worker_sha="wk_sha_456",
    )
    assert verified["process_cpu_seconds"] == pytest.approx(142.5)

    with pytest.raises(RuntimeError, match="Shard input SHA mismatch"):
        rem.verify_shard_sidecar(shard, meta_path, expected_input_sha="wrong", expected_worker_sha="wk_sha_456")

    meta_payload["process_cpu_seconds"] = 0.0
    meta_path.write_text(json.dumps(meta_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="positive measured process_cpu_seconds"):
        rem.verify_shard_sidecar(shard, meta_path, expected_input_sha="in_sha_123", expected_worker_sha="wk_sha_456")


def test_inherited_lock_fd_verifies_inode_and_active_lock(tmp_path: Path) -> None:
    lock_file = tmp_path / "qualification.lock"
    lock_file.write_text("holder\n", encoding="utf-8")
    proc_root = tmp_path / "proc_self"
    (proc_root / "fd").mkdir(parents=True)
    (proc_root / "fdinfo").mkdir(parents=True)
    (proc_root / "fd" / "9").symlink_to(lock_file)
    (proc_root / "fdinfo" / "9").write_text("pos:\t0\nflags:\t0100002\nlock:\t1: FLOCK  ADVISORY  WRITE\n", encoding="utf-8")

    res = rem.verify_inherited_lock_fd(9, lock_file, proc_root=proc_root)
    assert res["mode"] == "inherited_fd" and res["fd"] == 9

    (proc_root / "fdinfo" / "9").write_text("pos:\t0\nflags:\t0100002\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not hold an active flock"):
        rem.verify_inherited_lock_fd(9, lock_file, proc_root=proc_root)


def test_durable_gpu_budget_ledger_persists_events_and_enforces_ceiling(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    ticks = iter([100.0, 125.0, 200.0, 290.0])
    led = rem.DurableGpuBudgetLedger(ledger_path, ceiling_seconds=100.0)
    with led.charge_phase("pilot_fold0", min_required_seconds=10.0, clock_fn=lambda: next(ticks)):
        pass
    assert led.cumulative_seconds == pytest.approx(25.0)
    assert led.remaining_seconds == pytest.approx(75.0)

    # Resume from disk and verify failed phases also charge elapsed GPU seconds
    led2 = rem.DurableGpuBudgetLedger(ledger_path, ceiling_seconds=100.0)
    assert led2.cumulative_seconds == pytest.approx(25.0)
    with pytest.raises(RuntimeError, match="exceeded after main_cell"):
        with led2.charge_phase("main_cell", min_required_seconds=10.0, clock_fn=lambda: next(ticks)):
            pass
    assert led2.cumulative_seconds == pytest.approx(115.0)


def test_paired_family_bootstrap_reports_family_and_context_wtl_and_bonferroni_lcb() -> None:
    treat = {}
    ctrl = {}
    fam = {}
    for i in range(60):
        cid = f"c{i:03d}"
        fam[cid] = f"f{i % 20:02d}"
        ctrl[cid] = 1.0
        treat[cid] = 1.05 if i % 20 != 0 else 1.0
    res = rem.paired_family_bootstrap(treat, ctrl, fam, n_draws=500, seed=7, alpha=0.05 / 8)
    assert res["family_count"] == 20 and res["context_count"] == 60
    assert res["family_wtl"] == "19/1/0"
    assert res["context_wtl"] == "57/3/0"
    assert res["common_gate_passed"] is True
    assert res["lcb_adjusted"] <= res["lcb95_one_sided"]


def test_within_context_template_resolution_separates_composition_from_within_ctx_effect() -> None:
    # Simpson's paradox fixture: easy contexts have mostly unseen templates (+2.0), hard contexts have seen (+1.0),
    # but within every dual-eligible context seen templates beat unseen templates by +0.04.
    seen = {}
    unseen = {}
    fam = {}
    for i in range(30):
        cid = f"c{i:02d}"
        fam[cid] = f"f{i % 15:02d}"
        base = 2.0 if i < 15 else 1.0
        seen[cid] = base + 0.04
        unseen[cid] = base
    res = rem.within_context_template_resolution(seen, unseen, fam, n_draws=400, seed=11)
    assert res["dual_eligible_context_count"] == 30
    assert res["unseen_minus_seen_paired_bootstrap"]["family_mean_delta"] == pytest.approx(-0.04)
    assert res["unseen_minus_seen_paired_bootstrap"]["family_wtl"] == "0/0/15"


def test_verify_h81_strength_guard_passes_zero_gap_and_fails_refit_drift() -> None:
    exp = {f"c{i:02d}": 1.915872 for i in range(50)}
    rep_exact = dict(exp)
    fam = {f"c{i:02d}": f"f{i:02d}" for i in range(50)}
    ok = rem.verify_h81_strength_guard(rep_exact, exp, fam, n_draws=200)
    assert ok["strength_guard_passed"] is True and ok["max_abs_context_diff"] == 0.0

    rep_drifted = {c: v - 0.0539 for c, v in exp.items()}
    bad = rem.verify_h81_strength_guard(rep_drifted, exp, fam, n_draws=200)
    assert bad["strength_guard_passed"] is False


def test_evaluate_joint_8_contrast_w1_gate() -> None:
    pass_entry = {"common_gate_passed": True, "lcb_adjusted": 0.012}
    fail_entry = {"common_gate_passed": False, "lcb_adjusted": -0.002}
    u4 = {
        "NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_ONLY": pass_entry,
        "NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX": pass_entry,
        "NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR": pass_entry,
    }
    leg = {
        "NEURAL_GBM_RESIDUAL_vs_R882_CONTINUITY": fail_entry,
        "NEURAL_GBM_RESIDUAL_vs_GBM_PROG412_CTX": fail_entry,
        "NEURAL_GBM_RESIDUAL_vs_NEURAL_ZERO_PRIOR": fail_entry,
    }
    res = rem.evaluate_joint_8_contrast_w1_gate(u4, leg)
    assert res["verdict"] == "ADVANCE_SEARCH_U4_ONLY"
    assert res["search_claim_advanced"] is True and res["legacy_claim_advanced"] is False
