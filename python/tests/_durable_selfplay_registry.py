"""Test group re-exported by test_durable_selfplay for stable test IDs."""

from test_durable_selfplay import (
    EXPECTED_METRIC,
    PairRegistry,
    Path,
    ThreadPoolExecutor,
    catalog_fixture,
    digest,
    json,
    protocol,
    pytest,
    sha256_text,
    validate_catalog_counts,
    validate_catalog_manifest,
)


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
