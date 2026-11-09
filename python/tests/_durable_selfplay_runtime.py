"""Test group re-exported by test_durable_selfplay for stable test IDs."""

import test_exact_service_runtime as service_runtime_module
from test_durable_selfplay import (
    PairRegistry,
    Path,
    active,
    backlog_module,
    bundle_sources,
    candidate,
    dynamic_queue_targets,
    json,
    load_pointer,
    loaded_bundle,
    local_receipt_dir,
    promote_candidate,
    pytest,
    record_starting,
    service_is_stale,
    service_module,
    sha256_text,
    subprocess,
)


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

    monkeypatch.setattr(service_runtime_module, "_wave_process", FinishedWave())
    state = service_runtime_module.wave_generation_state()
    assert state == {
        "pending": False,
        "phase": "wave_committed",
        "wave": "strict-wave-000006",
        "manifests": 42,
    }
    assert service_runtime_module._wave_process is None


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
def test_runtime_bundle_contains_the_complete_prime_python_package(tmp_path, monkeypatch):
    import build_selfplay_bundle as bundle_builder

    source_root = tmp_path / "source"
    prime_root = tmp_path / "jope-prime"
    monkeypatch.setattr(bundle_builder, "SOURCE", source_root)
    monkeypatch.setattr(bundle_builder, "PRIME", prime_root)
    monkeypatch.setattr(bundle_builder, "SCQ_PROJECTS", source_root / "etc/projects.toml")
    package_files = (
        "prime.toml",
        "src/jope/__init__.py",
        "src/jope/grammars/spec.py",
        "src/jope/policy/model_v3.py",
        "registry/compatibility/gen31_policy_v3_vocab.json",
        "registry/compatibility/prime_arena_suites_v1.json",
    )
    for relative in package_files:
        path = prime_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "")

    targets = {target for _, target in bundle_sources()}
    assert "jope-prime/src/jope/__init__.py" in targets
    assert "jope-prime/src/jope/grammars/spec.py" in targets
    assert "jope-prime/src/jope/policy/model_v3.py" in targets
    assert "jope-prime/prime.toml" in targets
    assert "jope-prime/registry/compatibility/gen31_policy_v3_vocab.json" in targets


def test_deployment_explicitly_restarts_already_active_services():
    deploy = (
        Path(__file__).resolve().parents[1]
        / "test-support"
        / "jope"
        / "research"
        / "deploy_selfplay_bundle.py"
    )
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
