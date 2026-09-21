from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import test_exact_backlog as backlog_module  # noqa: E402
import test_exact_service as service_module  # noqa: E402
from build_selfplay_bundle import sources as bundle_sources  # noqa: E402
from test_exact_accept import evaluate_gate  # noqa: E402
from test_exact_common import (  # noqa: E402
    EXPECTED_METRIC,
    dynamic_queue_targets,
    evaluation_protocol_sha256,
    sha256_text,
    validate_catalog_counts,
    validate_catalog_manifest,
)
from test_exact_controller import default_handlers  # noqa: E402
from test_exact_evaluator import (  # noqa: E402
    active,
    candidate,
    load_pointer,
    promote_candidate,
)
from test_exact_ingest import local_receipt_dir  # noqa: E402
from test_exact_pair_registry import PairRegistry  # noqa: E402
from test_exact_sealed_audit import SealedAuditRegistry  # noqa: E402
from test_exact_service import loaded_bundle, record_starting  # noqa: E402
from test_exact_status import service_is_stale  # noqa: E402
from test_exact_wave import wave_seed  # noqa: E402

import jope.research


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def catalog_fixture(tmp_path: Path) -> Path:
    contexts = [b"a" * 8192, b"b" * 8192]
    data = tmp_path / "contexts.bin"
    data.write_bytes(b"".join(contexts))
    entries = []
    for index, task in enumerate(("regression", "binary")):
        entries.append(
            {
                "catalog_index": index,
                "task": task,
                "dataset_hash": f"d-{index}",
                "metric": "r2" if task == "regression" else "auc",
                "source_family": f"family-{index}",
                "split": "train" if index == 0 else "validation",
                "context_sha256": hashlib.sha256(contexts[index]).hexdigest(),
                "train_sha256": str(index + 1) * 64,
                "test_sha256": str(index + 3) * 64,
            }
        )
    manifest = {
        "schema": "JopeTestOnlyContextCatalog.v1",
        "immutable": True,
        "data_sha256": digest(data),
        "record_stride_bytes": 8192,
        "row_count": 2,
        "regression_count": 1,
        "binary_count": 1,
        "source_family_disjoint": True,
        "entries": entries,
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(manifest))
    pointer = tmp_path / "current.json"
    pointer.write_text(
        json.dumps(
            {
                "catalog": str(catalog),
                "catalog_sha256": digest(catalog),
                "data": str(data),
                "data_sha256": digest(data),
            }
        )
    )
    return pointer


def test_catalog_manifest_is_authority_for_counts_hashes_and_contexts(tmp_path):
    pointer = catalog_fixture(tmp_path)
    result = validate_catalog_manifest(pointer)
    # A pre-four-task manifest declares only the legacy counts; the new tasks
    # read as zero (mirrors jope_selfplay_promotion.rs::validate_catalog_counts).
    assert result["task_counts"] == {
        "regression": 1,
        "binary": 1,
        "timeseries": 0,
        "multiclass": 0,
    }
    assert result["row_count"] == 2

    value = json.loads(pointer.read_text())
    catalog = Path(value["catalog"])
    manifest = json.loads(catalog.read_text())
    manifest["regression_count"] = 2
    catalog.write_text(json.dumps(manifest))
    value["catalog_sha256"] = digest(catalog)
    pointer.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="catalog count mismatch"):
        validate_catalog_manifest(pointer)


def test_expected_metric_matches_the_rust_task_authority():
    """R1/D2 wire values; the Rust twin is expected_catalog_metric /
    TaskKind::primary_metric (asserted from that side by cargo tests)."""

    assert EXPECTED_METRIC == {
        "regression": "r2",
        "binary": "auc",
        "timeseries": "negative_mase",
        "multiclass": "macro_f1",
    }


def test_catalog_counts_default_missing_new_tasks_and_reject_unknown_tasks():
    entries = [
        {"task": "regression", "dataset_hash": "d1"},
        {"task": "binary", "dataset_hash": "d2"},
        {"task": "timeseries", "dataset_hash": "d3"},
        {"task": "multiclass", "dataset_hash": "d4"},
    ]
    # Full four-task declaration.
    assert validate_catalog_counts(
        entries,
        {"regression": 1, "binary": 1, "timeseries": 1, "multiclass": 1},
    ) == {"regression": 1, "binary": 1, "timeseries": 1, "multiclass": 1}
    # Legacy two-task declaration: missing NEW tasks default to zero.
    assert validate_catalog_counts(entries[:2], {"regression": 1, "binary": 1}) == {
        "regression": 1,
        "binary": 1,
        "timeseries": 0,
        "multiclass": 0,
    }
    # ... which still fails closed when the catalog contains new-task entries.
    with pytest.raises(ValueError, match="catalog count mismatch"):
        validate_catalog_counts(entries, {"regression": 1, "binary": 1})
    # Unknown tasks are rejected, in declarations and in entries.
    with pytest.raises(ValueError, match="unsupported task counts"):
        validate_catalog_counts(entries[:2], {"regression": 1, "binary": 1, "multi_label": 1})
    with pytest.raises(ValueError, match="unsupported or anonymous"):
        validate_catalog_counts(
            [{"task": "bogus", "dataset_hash": "d9"}],
            {"regression": 0, "binary": 0},
        )
    # The legacy declarations stay mandatory.
    with pytest.raises(ValueError, match="unsupported task counts"):
        validate_catalog_counts([], {"binary": 0})
    with pytest.raises(ValueError, match="does not declare every per-task count"):
        validate_catalog_counts([], {"regression": -1, "binary": 0})


def protocol(task="regression", version="v1"):
    return evaluation_protocol_sha256(
        task=task,
        metric="r2" if task == "regression" else "auc",
        score_tier="test_exact",
        train_sha256="1" * 64,
        test_sha256="2" * 64,
        version=version,
    )


def test_pair_registry_concurrent_reservation_is_exactly_once(tmp_path):
    path = tmp_path / "pairs.sqlite"
    genome = "(pipe X (ridge))"

    def reserve(index):
        registry = PairRegistry(path)
        try:
            return registry.reserve(
                task="regression",
                dataset_hash="dataset",
                semantic_sha256=sha256_text(genome),
                protocol_sha256=protocol(),
                owner=f"worker-{index}",
                canonical_genome=genome,
            )
        finally:
            registry.close()

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(8)))
    assert sum(outcomes) == 1
    registry = PairRegistry(path)
    assert registry.stats()["pairs"] == 1
    assert registry.stats()["duplicates_prevented"] == 7
    registry.close()


def test_infrastructure_retry_reuses_reservation_and_protocol_migration_is_explicit(
    tmp_path,
):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    genome = "(pipe X (ridge))"
    identity = {
        "task": "regression",
        "dataset_hash": "dataset",
        "semantic_sha256": sha256_text(genome),
        "protocol_sha256": protocol(),
    }
    assert registry.reserve(**identity, owner="one", canonical_genome=genome)
    registry.record_result(
        **identity,
        receipt_sha256="3" * 64,
        failure="rpc-error",
    )
    assert registry.get(**identity)["state"] == "infrastructure_retry"
    assert not registry.reserve(**identity, owner="two", canonical_genome=genome)
    assert registry.migrate_protocol(
        task="regression",
        dataset_hash="dataset",
        semantic_sha256=sha256_text(genome),
        old_protocol_sha256=protocol(),
        new_protocol_sha256=protocol(version="v2"),
        reason="corrected row alignment",
        owner="migration",
    )
    assert registry.stats()["infrastructure_retries"] == 1
    assert registry.stats()["protocol_migrations"] == 1
    registry.close()


def test_submission_requires_live_reservations_and_is_idempotent(tmp_path):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    genome = "(pipe X (ridge))"
    identity = {
        "task": "regression",
        "dataset_hash": "dataset",
        "semantic_sha256": sha256_text(genome),
        "protocol_sha256": protocol(),
    }
    assert registry.reserve(**identity, owner="one", canonical_genome=genome)
    registry.db.execute("UPDATE pairs SET manifest_sha256='manifest' WHERE task='regression'")
    registry.db.commit()
    assert registry.authorize_submission("manifest", "one") == 1
    assert registry.record_submission("manifest", "one", "job") == 1
    assert registry.record_submission("manifest", "one", "job") == 1
    assert registry.authorize_submission("manifest", "one", job_id="job") == 1
    with pytest.raises(RuntimeError, match="different submission"):
        registry.record_submission("manifest", "one", "other-job")
    registry.record_result(
        **identity,
        receipt_sha256="3" * 64,
    )
    with pytest.raises(RuntimeError, match="non-launchable"):
        registry.authorize_submission("manifest", "one")
    registry.close()


def test_pair_registry_batch_reservation_reuses_in_input_order(tmp_path):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    genomes = ["(pipe X (ridge))", "(pipe X (lasso))"]
    rows = [
        {
            "task": "regression",
            "dataset_hash": "dataset",
            "semantic_sha256": sha256_text(genome),
            "protocol_sha256": protocol(),
            "canonical_genome": genome,
        }
        for genome in genomes
    ]
    assert registry.reserve_batch(rows, owner="batch") == [True, True]
    assert registry.reserve_batch(rows, owner="other") == [False, False]
    assert registry.stats()["pairs"] == 2
    registry.close()


def test_queue_horizons_use_observed_p95_cluster_slots_and_clamps():
    assert dynamic_queue_targets([]) == (256, 512, 0.0)
    target, hard_cap, p95 = dynamic_queue_targets([1.0] * 56 + [100.0] * 4)
    assert (target, hard_cap, p95) == (1_500, 3_000, 100.0)
    target, hard_cap, p95 = dynamic_queue_targets([50.0] * 60, 400)
    assert (target, hard_cap, p95) == (800, 1_500, 50.0)


def test_refill_work_is_bounded_without_changing_the_total_target():
    assert backlog_module.bounded_submission_count(-1) == 0
    assert backlog_module.bounded_submission_count(511) == 511
    assert backlog_module.bounded_submission_count(9_000) == 512


def test_service_reports_the_bundle_loaded_at_process_start(tmp_path, monkeypatch):
    monkeypatch.setenv("JOPE_LOADED_BUNDLE_SHA256", "loaded-bundle")
    assert loaded_bundle(tmp_path) == "loaded-bundle"
    state = record_starting("reserve", tmp_path)
    assert state["loaded_bundle_sha256"] == "loaded-bundle"
    assert state["lifecycle"] == "starting"
    assert state["last_success_at_ms"] is None


def test_starting_service_uses_process_start_for_staleness():
    now = 1_000_000
    state = {
        "lifecycle": "starting",
        "last_success_at_ms": 1,
        "process_started_at_ms": now - 1_000,
    }
    assert not service_is_stale(state, now)
    state["process_started_at_ms"] = now - 300_001
    assert service_is_stale(state, now)


def test_wave_generation_is_polled_without_blocking(monkeypatch):
    class FinishedWave:
        pid = 123

        @staticmethod
        def poll():
            return 0

        @staticmethod
        def communicate():
            return ('{"wave":"strict-wave-000006","manifests":42}\n', "")

    monkeypatch.setattr(service_module, "_wave_process", FinishedWave())
    state = service_module.wave_generation_state()
    assert state == {
        "pending": False,
        "phase": "wave_committed",
        "wave": "strict-wave-000006",
        "manifests": 42,
    }
    assert service_module._wave_process is None


def test_ingestion_cycle_uses_bounded_batches(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"ingested": 100}\n', "")

    monkeypatch.setattr(service_module, "ROOT", tmp_path)
    monkeypatch.setattr(service_module, "run", fake_run)
    assert service_module.ingestion_cycle() == {"ingested": 100}
    assert calls[0][-2:] == ["--limit", "100"]


def test_orphaned_submission_retries_only_without_a_receipt(tmp_path, monkeypatch):
    registry_path = tmp_path / "pairs.sqlite"
    backlog_path = tmp_path / "backlog.sqlite"
    runs = tmp_path / "runs"
    monkeypatch.setattr(backlog_module, "STATE", backlog_path)
    monkeypatch.setattr(backlog_module, "ROOT", tmp_path)
    monkeypatch.setattr(backlog_module, "SCQ_RUNS", runs)
    monkeypatch.setattr(
        backlog_module,
        "PairRegistry",
        lambda: PairRegistry(registry_path),
    )
    registry = PairRegistry(registry_path)
    for index in range(2):
        registry.reserve(
            task="regression",
            dataset_hash=f"dataset-{index}",
            semantic_sha256=sha256_text("(pipe X (ridge))"),
            protocol_sha256="protocol",
            owner=f"owner-{index}",
            canonical_genome="(pipe X (ridge))",
            manifest_path=f"/manifest-{index}.json",
            manifest_sha256=f"manifest-{index}",
        )
        registry.record_submission(f"manifest-{index}", f"owner-{index}", f"job-{index}")
    registry.close()
    receipt = runs / "job-1" / "attempts" / "0" / "outputs" / "receipt"
    receipt.mkdir(parents=True)
    (receipt / "SHARD_OK").write_text("{}")
    assert local_receipt_dir("job-1", runs) == receipt
    database = backlog_module.state_db()
    database.executemany(
        "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?)",
        [
            (
                f"manifest-{index}",
                f"/manifest-{index}.json",
                f"job-{index}",
                f"/job-{index}.toml",
                "submitted",
                1,
                f"owner-{index}",
                0,
            )
            for index in range(2)
        ],
    )
    database.commit()
    database.close()
    assert backlog_module.recover_orphaned_submissions({}) == 1
    database = backlog_module.state_db()
    states = dict(database.execute("SELECT job_id,state FROM claims"))
    database.close()
    assert states == {"job-0": "checked", "job-1": "submitted"}
    registry = PairRegistry(registry_path)
    assert registry.manifest_state_counts("manifest-0") == {"infrastructure_retry": 1}
    assert registry.manifest_state_counts("manifest-1") == {"submitted": 1}
    registry.close()


def test_submit_adopts_a_job_that_became_active_during_refill(tmp_path, monkeypatch):
    monkeypatch.setattr(
        backlog_module,
        "retry_command",
        lambda _args: subprocess.CompletedProcess(
            _args,
            1,
            "",
            "error: job id 'jte-race' already active (queued)",
        ),
    )
    job_id, success, detail = backlog_module.submit_one(
        ("jte-race", tmp_path / "job.toml", "manifest", tmp_path / "manifest.json")
    )
    assert job_id == "jte-race"
    assert success
    assert "already active" in detail


@pytest.mark.xbabe2  # needs the runtime-bundle tree on the old state plane (bundle_sources() is empty on a clean checkout)
def test_runtime_bundle_contains_the_complete_prime_python_package():
    targets = {target for _, target in bundle_sources()}
    assert "jope-prime/src/jope/__init__.py" in targets
    assert "jope-prime/src/jope/grammars/spec.py" in targets
    assert "jope-prime/src/jope/policy/model_v3.py" in targets
    assert "jope-prime/prime.toml" in targets
    assert "jope-prime/registry/compatibility/gen31_policy_v3_vocab.json" in targets


def test_deployment_explicitly_restarts_already_active_services():
    deploy = Path(jope.research.__file__).parent / "deploy_selfplay_bundle.py"
    source = deploy.read_text()
    assert '["systemctl", "--user", "enable", *UNITS]' in source
    assert '["systemctl", "--user", "restart", *UNITS]' in source


def test_checkpoint_vocab_compatibility_requires_an_exact_fitting_prefix(monkeypatch):
    from jope.policy.model_v3 import _validate_action_vocab_compatibility

    _validate_action_vocab_compatibility(["PAD", "BOS"], ["PAD", "BOS", "EOS"], 3)
    with pytest.raises(ValueError, match="exact current prefix"):
        _validate_action_vocab_compatibility(["PAD", "old"], ["PAD", "BOS", "EOS"], 3)
    with pytest.raises(ValueError, match="exceeds"):
        _validate_action_vocab_compatibility(["PAD", "BOS"], ["PAD", "BOS", "EOS"], 2)


def test_failed_candidate_canary_leaves_last_good_active(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    good = active(tmp_path, first)
    staged = candidate(tmp_path, second)
    bad_report = tmp_path / "bad.json"
    bad_report.write_text(
        json.dumps(
            {
                "semantic_bit_parity": False,
                "nodes": {
                    node: {"evaluator_sha256": staged["evaluator_sha256"]}
                    for node in ("xbabe1", "xbabe2", "xbabe3", "xbabe4")
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="did not pass"):
        promote_candidate(bad_report, tmp_path)
    assert load_pointer("active", tmp_path)["evaluator_sha256"] == good["evaluator_sha256"]


def gate_evidence(lift: float) -> dict:
    return {
        "baseline": {
            "checkpoint_sha256": "a" * 64,
            "macro_pairwise_accuracy": 0.60,
            "regression_pairwise_accuracy": 0.60,
            "binary_pairwise_accuracy": 0.60,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.02,
        },
        "candidate": {
            "checkpoint_sha256": "b" * 64,
            "macro_pairwise_accuracy": 0.60 + lift,
            "regression_pairwise_accuracy": 0.60 + lift,
            "binary_pairwise_accuracy": 0.60 + lift,
            "failure_rate": 0.01,
            "p95_proposal_latency_ms": 100,
            "val_conditional_advantage": 0.06,
        },
        "proposal_validity": 0.99,
        "verified_checkpoint_origin_strict_wins": 1,
        "candidate_equal_budget_result": 0.51,
        "baseline_equal_budget_result": 0.49,
        "replay_verification": {
            key: True
            for key in (
                "predictions",
                "scores",
                "evaluator_identity",
                "dataset_hashes",
                "candidate_origin",
            )
        },
    }


def test_dual_tier_promotion_keeps_internal_and_milestone_thresholds_separate():
    internal = evaluate_gate(gate_evidence(0.002))
    assert internal["pass"]
    assert internal["promotion_tier"] == "internal"
    assert not internal["milestone_pass"]
    milestone = evaluate_gate(gate_evidence(0.01))
    assert milestone["pass"]
    assert milestone["promotion_tier"] == "milestone"


def test_a_non_conditional_checkpoint_can_never_be_promoted():
    """val_conditional_advantage <= 0 means the head ignores the dataset."""

    dead = gate_evidence(0.05)
    dead["candidate"]["val_conditional_advantage"] = -0.060
    result = evaluate_gate(dead)
    assert not result["pass"]
    assert not result["milestone_pass"]
    assert result["promotion_tier"] == "rejected"
    assert "candidate_policy_is_not_conditional" in result["reasons"]
    assert "candidate_policy_is_not_conditional" in result["milestone_reasons"]
    assert result["candidate_conditional_advantage"] == -0.060

    zero = gate_evidence(0.05)
    zero["candidate"]["val_conditional_advantage"] = 0.0
    assert "candidate_policy_is_not_conditional" in evaluate_gate(zero)["reasons"]


def test_conditionality_margins_separate_internal_from_milestone():
    thin = gate_evidence(0.05)
    thin["candidate"]["val_conditional_advantage"] = 0.004
    thin["baseline"]["val_conditional_advantage"] = -0.02
    result = evaluate_gate(thin)
    assert not result["pass"]
    assert "conditional_advantage_below_half_point" in result["reasons"]

    internal = gate_evidence(0.05)
    internal["candidate"]["val_conditional_advantage"] = 0.03
    internal["baseline"]["val_conditional_advantage"] = 0.02
    result = evaluate_gate(internal)
    assert result["promotion_tier"] == "internal"
    assert result["conditional_advantage_lift"] == pytest.approx(0.01)
    assert "conditional_advantage_lift_below_two_points" in result["milestone_reasons"]


def test_unmeasured_conditionality_is_fail_closed_not_zero():
    for side in ("candidate", "baseline"):
        evidence = gate_evidence(0.05)
        del evidence[side]["val_conditional_advantage"]
        result = evaluate_gate(evidence)
        assert not result["pass"]
        assert f"{side}_conditional_advantage_evidence_missing" in result["reasons"]
        assert result[f"{side}_conditional_advantage"] is None
        assert result["conditional_advantage_lift"] is None
        # Missing evidence is not silently reported as a non-conditional policy.
        assert "candidate_policy_is_not_conditional" not in result["reasons"]

    null_valued = gate_evidence(0.05)
    null_valued["candidate"]["val_conditional_advantage"] = None
    result = evaluate_gate(null_valued)
    assert "candidate_conditional_advantage_evidence_missing" in result["reasons"]


def test_new_task_verdicts_are_advisory_until_anchored():
    """U3: an unanchored branch is recorded (`"advisory": true`) but can never
    alter `pass` / `promotion_tier` / the global reasons; an accepted anchor in
    accepted-by-task.json flips it to gating, exactly like a legacy branch."""

    evidence = gate_evidence(0.05)
    result = evaluate_gate(evidence)
    assert result["pass"] is True and result["promotion_tier"] == "milestone"
    assert result["task_verdicts"]["regression"]["advisory"] is False
    for task in ("timeseries", "multiclass"):
        verdict = result["task_verdicts"][task]
        assert verdict["advisory"] is True
        assert verdict["pass"] is False
        assert verdict["pairwise_lift"] is None
        assert result["task_pairwise_deltas"][task] is None
    assert result["tasks_passing"] == ["regression", "binary"]

    # Advisory evidence, even a hard regression, never blocks the global gate.
    evidence["baseline"]["timeseries_pairwise_accuracy"] = 0.6
    evidence["candidate"]["timeseries_pairwise_accuracy"] = 0.5
    result = evaluate_gate(evidence)
    assert result["pass"] is True and result["promotion_tier"] == "milestone"
    verdict = result["task_verdicts"]["timeseries"]
    assert verdict["advisory"] is True
    assert "timeseries_regressed_more_than_half_point" in verdict["reasons"]
    assert not any(r.startswith("timeseries_") for r in result["milestone_reasons"])

    # Once anchored, the same evidence gates.
    anchor = {"tasks": {"timeseries": {"checkpoint_sha256": "c" * 64}}}
    gated = evaluate_gate(evidence, anchor)
    assert gated["pass"] is False and gated["promotion_tier"] == "rejected"
    assert "timeseries_regressed_more_than_half_point" in gated["reasons"]
    assert gated["task_verdicts"]["timeseries"]["advisory"] is False
    assert gated["task_verdicts"]["multiclass"]["advisory"] is True


PARITY_FIXTURE = (
    Path(__file__).parents[2]
    / "crates"
    / "jope-campaigns"
    / "tests"
    / "fixtures"
    / "conditionality_gate_parity.json"
)


def test_conditionality_gate_parity_fixture_matches_python():
    """The Rust port in src/bin/jope_selfplay_promotion.rs asserts the same file.

    Regenerate after an intentional gate change with
    JOPE_REGENERATE_PARITY_FIXTURE=1, then re-run `cargo test` for the port.
    """

    fixture = json.loads(PARITY_FIXTURE.read_text())
    regenerate = os.environ.get("JOPE_REGENERATE_PARITY_FIXTURE") == "1"
    for case in fixture["cases"]:
        produced = json.dumps(evaluate_gate(case["evidence"]), sort_keys=True, separators=(",", ":"))
        if regenerate:
            case["expected_gate_compact_json"] = produced
            continue
        assert produced == case["expected_gate_compact_json"], case["name"]
    if regenerate:
        PARITY_FIXTURE.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")


def candidate_row(index: int, source: str) -> dict:
    genome = f"(pipe X (cheb degree={index + 2}) (ridge))"
    return {
        "canonical_genome": genome,
        "semantic_sha256": sha256_text(genome),
        "source": source,
    }


def test_every_wave_seed_has_the_exact_256_slot_composition():
    seed = wave_seed(
        task="regression",
        dataset="dataset",
        wave_id="strict-wave-000001",
        local=[candidate_row(index, "local") for index in range(32)],
        dl=[candidate_row(index + 100, "dl") for index in range(64)],
        common=[candidate_row(index + 200, "common") for index in range(96)],
    )
    assert len(seed) == 256
    assert {
        group: sum(row["slot_group"] == group for row in seed)
        for group in ("local_elite", "dl_proposal", "common_winner", "offspring")
    } == {
        "local_elite": 32,
        "dl_proposal": 64,
        "common_winner": 96,
        "offspring": 64,
    }


def test_controller_has_no_file_waiting_production_phases():
    handlers = default_handlers()
    assert handlers["generate_gate_proposals"].__name__ == "_generate_gate_proposals"
    assert handlers["evaluate_frozen_gate"].__name__ == "_evaluate_frozen_gate"
    assert handlers["run_sotarena_qualification"].__name__ == "_run_sotarena_qualification"


def test_sealed_audit_is_one_shot_and_never_training_eligible(tmp_path):
    registry = SealedAuditRegistry(tmp_path / "sealed.sqlite")
    entry = {
        "task": "regression",
        "dataset_hash": "unseen",
        "source_family": "unseen-family",
        "train_sha256": "1" * 64,
        "test_sha256": "2" * 64,
        "incumbent_score": 0.5,
    }
    audit_id = registry.register(
        entry,
        checkpoint_sha256="3" * 64,
        forbidden_datasets=set(),
        forbidden_families=set(),
    )
    registry.emit(audit_id, "(pipe X (ridge))")
    with pytest.raises(RuntimeError, match="exactly one"):
        registry.emit(audit_id, "(pipe X (standardize) (ridge))")
    result = registry.score(
        audit_id,
        score=0.6,
        receipt_sha256="4" * 64,
        compiled_interpreted_parity=True,
        prediction_parity=True,
    )
    assert result["super_human"]
    assert not result["training_ingestion_eligible"]
    registry.close()


def test_systemd_services_are_independently_restartable():
    # ADR-0003 (PR #13/#32) turned every unit into a `.service.in` template rendered by
    # deploy/systemd/render.sh, so the units are named `*.service.in` and spell the bundle
    # directory as @JOPE_BUNDLE@ instead of a literal runtime-bundles path. The guarantee
    # this test exists to defend is unchanged: each service restarts on its own and executes
    # out of the versioned runtime bundle, never an ad-hoc path.
    #
    # This is the template-level half. The rendered half — that substitution leaves no
    # @JOPE_* token, no /home/ literal, and every ExecStart under the rendered roots — is
    # covered in Rust by crates/jope-campaigns/tests/systemd_unit_paths.rs
    # (rendered_units_resolve_only_under_the_rendered_roots), which drives the same
    # jope_paths::systemd::render_tree that render.sh shells out to. Duplicating it here
    # would drag a cargo build into the Python lane for no extra coverage.
    root = Path(__file__).parents[2] / "deploy" / "systemd" / "selfplay"
    required = {
        "jope-selfplay-corpus-freezer.service.in",
        "jope-selfplay-ingestion.service.in",
        "jope-selfplay-promotion.service.in",
        "jope-selfplay-queue.service.in",
        "jope-selfplay-supply.service.in",
        "jope-selfplay-trainer-watchdog.service.in",
    }
    units = {unit.name: unit for unit in root.glob("jope-selfplay-*.service.in")}
    assert required <= units.keys()
    for name in required:
        unit = units[name]
        text = unit.read_text()
        assert "Restart=always" in text
        assert "ExecStart=@JOPE_BUNDLE@/" in text

