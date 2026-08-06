from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from test_exact_adaptive import (  # noqa: E402
    attach_ranking_labels,
    evaluate_adaptive,
    write_gene_evidence,
)
from test_exact_capacity import (  # noqa: E402
    node_slot_target,
    ramp_decision,
    validate_selfplay_policy,
)
from test_exact_common import (  # noqa: E402
    dynamic_queue_targets,
    sha256_file,
    sha256_text,
    validate_catalog_manifest,
)
from test_exact_compile_cache import (  # noqa: E402
    compiled_cache_key,
    compiled_finalists,
    exact_parity,
)
from test_exact_expansion import (  # noqa: E402
    TRAIN_ENCODING_NAMES,
    activate_tranche,
    compare_mirror,
    discover,
    materialize_cache,
    quarantine_identities,
    source_family,
    validate_cache,
)
from test_exact_freshness import FreshnessSupervisor, epoch_for_checkpoint  # noqa: E402
from test_exact_freshness_mcp import TOOLS  # noqa: E402
from test_exact_gp_bridge import (  # noqa: E402
    export_gp_seeds,
    inject_promoted_proposals,
)
from test_exact_pair_registry import PairRegistry  # noqa: E402
from test_exact_wave import select_micro_cohort  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_v1_quarantine(cache_root: Path, denied: str) -> None:
    root = cache_root / "incident-quarantine"
    root.mkdir(parents=True)
    identities = [denied, *(f"{index:016x}" for index in range(1, 18))]
    (root / "index.v1.json").write_text(
        json.dumps(
            {
                "schema": "JopeV1EncodingQuarantine.v1",
                "policy": {
                    "catalog_allowed": False,
                    "training_allowed": False,
                    "benchmark_allowed": False,
                    "v2_import_allowed": False,
                    "gc_allowed": False,
                },
                "trees": [
                    {
                        "identity": identity,
                        "state": "complete" if index < 11 else "partial",
                    }
                    for index, identity in enumerate(identities)
                ],
            }
        )
    )


def test_v1_quarantine_is_excluded_from_python_discovery_and_consumption(tmp_path):
    denied = "abcdef0123456789"
    cache_root = tmp_path / "cache"
    write_v1_quarantine(cache_root, denied)
    assert denied in quarantine_identities(cache_root)

    source_root = tmp_path / "source"
    for task_root in ("regression", "binary", "multi_label", "timeseries"):
        (source_root / task_root).mkdir(parents=True)
    source = source_root / "regression" / denied
    source.mkdir()
    for name, contents in (
        ("train.csv", "x,y\n1,2\n"),
        ("test.csv", "x,y\n3,4\n"),
        ("meta.json", "{}"),
    ):
        (source / name).write_text(contents)
    assert discover(
        source_root=source_root,
        mirror_root=tmp_path / "mirror",
        verify_mirror=False,
        cache_root=cache_root,
    ) == []

    row = {
        "task": "regression",
        "dataset_id": denied,
        "dataset_hash": denied,
        "source": str(source),
        "source_family": "family",
        "train_sha256": digest(source / "train.csv"),
        "test_sha256": digest(source / "test.csv"),
    }
    with pytest.raises(RuntimeError, match="denied by quarantine"):
        materialize_cache(row, cache_root=cache_root)
    with pytest.raises(RuntimeError, match="catalog adoption denied"):
        activate_tranche([row], cache_root=cache_root, catalog_root=tmp_path / "catalog")
    with pytest.raises(ValueError, match="quarantined V1 cache"):
        validate_cache(cache_root / "incident-quarantine" / "complete" / denied)


def identity(genome: str = "(pipe X (ridge))") -> dict:
    return {
        "task": "regression",
        "dataset_hash": "dataset",
        "semantic_sha256": sha256_text(genome),
        "protocol_sha256": "1" * 64,
    }


def test_queue_horizon_formula_covers_slots_and_clamps():
    assert dynamic_queue_targets([], 364) == (728, 728, 0.0)
    assert dynamic_queue_targets([1_000], 1) == (2_048, 4_096, 1_000.0)
    with pytest.raises(ValueError):
        dynamic_queue_targets([], -1)


def test_attempt_history_retains_stale_retirement_and_pair_can_be_reused(tmp_path):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    pair = identity()
    assert registry.reserve(**pair, owner="old", canonical_genome="(pipe X (ridge))")
    registry.bind_manifest(
        owner="old",
        manifest_path="/old.json",
        manifest_sha256="2" * 64,
        pairs=[pair],
    )
    registry.record_submission("2" * 64, "old", "old-job")
    assert registry.supersede_unstarted(
        "2" * 64, "old-job", retirement_receipt_sha256="3" * 64
    ) == 1
    assert registry.get(**pair)["state"] == "available"
    assert registry.reserve(**pair, owner="new", canonical_genome="(pipe X (ridge))")
    history = registry.attempt_history(**pair)
    assert [row["state"] for row in history] == [
        "superseded_unstarted",
        "reserved",
    ]
    assert registry.stats()["active_attempts"] == 1
    registry.close()


def test_checked_unsubmitted_reservation_can_be_superseded(tmp_path):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    pair = identity()
    assert registry.reserve(
        **pair, owner="checked", canonical_genome="(pipe X (ridge))"
    )
    registry.bind_manifest(
        owner="checked",
        manifest_path="/checked.json",
        manifest_sha256="5" * 64,
        pairs=[pair],
    )
    assert registry.supersede_unstarted(
        "5" * 64,
        "checked-job",
        retirement_receipt_sha256="6" * 64,
    ) == 1
    assert registry.get(**pair)["state"] == "available"
    history = registry.attempt_history(**pair)
    assert history[0]["state"] == "superseded_unstarted"
    assert history[0]["job_id"] == "checked-job"
    registry.close()


def test_infrastructure_retry_creates_a_new_attempt_on_resubmission(tmp_path):
    registry = PairRegistry(tmp_path / "pairs.sqlite")
    pair = identity()
    registry.reserve(
        **pair,
        owner="owner",
        canonical_genome="(pipe X (ridge))",
        manifest_path="/manifest",
        manifest_sha256="4" * 64,
    )
    registry.record_submission("4" * 64, "owner", "job-1")
    registry.record_manifest_infrastructure_failure("4" * 64, "rpc-error")
    registry.record_submission("4" * 64, "owner", "job-2")
    history = registry.attempt_history(**pair)
    assert [(row["attempt_no"], row["state"]) for row in history] == [
        (1, "infrastructure_retry"),
        (2, "submitted"),
    ]
    registry.close()


class FakeScheduler:
    def __init__(self, jobs):
        self.rows = {row["id"]: dict(row) for row in jobs}
        self.held = []
        self.cancelled = []

    def jobs(self):
        return [dict(row) for row in self.rows.values()]

    def hold(self, job_id):
        assert self.rows[job_id]["state"] in {"queued", "starting"}
        self.rows[job_id]["state"] = "held"
        self.held.append(job_id)

    def cancel(self, job_id):
        assert self.rows[job_id]["state"] == "held"
        self.rows[job_id]["state"] = "cancelled"
        self.cancelled.append(job_id)


def freshness_fixture(tmp_path: Path, initial_state: str = "queued"):
    root = tmp_path / "root"
    (root / "ops").mkdir(parents=True)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"new")
    new_hash = digest(checkpoint)
    (root / "training").mkdir()
    (root / "training" / "learning.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": new_hash})
    )
    old_hash = "a" * 64
    manifest = root / "old-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "JopeFlywheelCorpusEvalShard.v1",
                "checkpoint_epoch": epoch_for_checkpoint(old_hash),
                "checkpoint_sha256": old_hash,
                "task": "regression",
                "dataset_hash": "dataset",
                "candidates": [],
            }
        )
    )
    manifest_hash = digest(manifest)
    registry = PairRegistry(root / "ops" / "pair-registry.sqlite")
    pair = identity()
    registry.reserve(
        **pair,
        owner="owner",
        canonical_genome="(pipe X (ridge))",
        manifest_path=str(manifest),
        manifest_sha256=manifest_hash,
    )
    registry.record_submission(manifest_hash, "owner", "stale-job")
    registry.close()
    backlog = sqlite3.connect(root / "ops" / "backlog.sqlite")
    backlog.execute(
        """
        CREATE TABLE claims(
          manifest_sha256 TEXT PRIMARY KEY,manifest TEXT,job_id TEXT,
          spec TEXT,state TEXT,attempts INTEGER,reservation_owner TEXT,
          updated_at_ms INTEGER
        )
        """
    )
    backlog.execute(
        "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?)",
        (
            manifest_hash,
            str(manifest),
            "stale-job",
            "/spec",
            "submitted",
            1,
            "owner",
            1,
        ),
    )
    backlog.commit()
    backlog.close()
    scheduler = FakeScheduler(
        [
            {
                "id": "stale-job",
                "project": "jope-selfplay-eval",
                "state": initial_state,
            },
            {
                "id": "running-job",
                "project": "jope-testonly",
                "state": "running",
            },
        ]
    )
    supervisor = FreshnessSupervisor(root, scheduler=scheduler)
    return root, scheduler, supervisor, pair, manifest_hash


def test_freshness_holds_and_retires_only_confirmed_unstarted_jobs(tmp_path):
    root, scheduler, supervisor, pair, _ = freshness_fixture(tmp_path)
    supervisor.sync_checkpoint()
    preview = supervisor.preview_stale_retirement()
    # Unclaimed scheduler work is outside the supervisor's cancellation scope.
    assert preview["payload"]["preserved_started"] == []
    result = supervisor.apply_stale_retirement(
        preview["preview_receipt_sha256"], "retire-1"
    )
    assert result["held"] == 1
    assert result["retired"] == 1
    assert result["running_interrupted"] == 0
    assert scheduler.held == ["stale-job"]
    assert scheduler.cancelled == ["stale-job"]
    assert scheduler.rows["running-job"]["state"] == "running"
    registry = PairRegistry(root / "ops" / "pair-registry.sqlite")
    assert registry.get(**pair)["state"] == "available"
    assert registry.attempt_history(**pair)[0]["state"] == "superseded_unstarted"
    registry.close()
    assert (
        sqlite3.connect(root / "ops" / "backlog.sqlite")
        .execute("SELECT state FROM claims")
        .fetchone()[0]
        == "superseded_unstarted"
    )
    assert supervisor.apply_stale_retirement(
        preview["preview_receipt_sha256"], "retire-1"
    ) == result
    supervisor.close()


def test_interrupted_cancelled_retirement_is_resumable(tmp_path):
    root, scheduler, supervisor, pair, _ = freshness_fixture(
        tmp_path, initial_state="cancelled"
    )
    supervisor.sync_checkpoint()
    preview = supervisor.preview_stale_retirement()
    result = supervisor.apply_stale_retirement(
        preview["preview_receipt_sha256"], "resume-cancel"
    )
    assert result["held"] == 0
    assert result["retired"] == 1
    registry = PairRegistry(root / "ops" / "pair-registry.sqlite")
    assert registry.get(**pair)["state"] == "available"
    registry.close()
    supervisor.close()


def test_freshness_cycle_auto_holds_scoped_stale_queue_only(tmp_path):
    _, scheduler, supervisor, _, _ = freshness_fixture(tmp_path)
    scheduler.rows["protected-queue"] = {
        "id": "protected-queue",
        "project": "external",
        "state": "queued",
    }
    status = supervisor.cycle()
    assert status["automatically_held"] == 1
    assert scheduler.rows["stale-job"]["state"] == "held"
    assert scheduler.rows["protected-queue"]["state"] == "queued"
    assert scheduler.rows["running-job"]["state"] == "running"
    supervisor.close()


def test_freshness_mutations_are_receipt_gated_and_idempotent(tmp_path):
    _, _, supervisor, _, _ = freshness_fixture(tmp_path)
    supervisor.sync_checkpoint()
    preview = supervisor.preview_stale_retirement()["preview_receipt_sha256"]
    first = supervisor.pause_admission(preview, "pause-1")
    assert first == supervisor.pause_admission(preview, "pause-1")
    with pytest.raises(RuntimeError, match="different mutation"):
        supervisor.resume_admission(preview, "pause-1")
    assert set(TOOLS) == {
        "get_freshness_status",
        "get_active_checkpoint",
        "list_queue_epochs",
        "explain_job_provenance",
        "preview_stale_retirement",
        "apply_stale_retirement",
        "pause_admission",
        "resume_admission",
        "request_micro_wave",
    }
    supervisor.close()


def datasets():
    return [
        {
            "task": task,
            "dataset_hash": f"{task}-{index:03}",
            "source_family": f"{task}-family-{index:03}",
            "observation_count": index % 7,
            "uncertainty": (index * 13) % 17,
        }
        for task in ("regression", "binary")
        for index in range(100)
    ]


def test_micro_cohort_is_deterministic_balanced_and_lane_exact():
    entries = datasets()
    first = select_micro_cohort(entries, {}, wave_id="strict-wave-000007")
    second = select_micro_cohort(entries, {}, wave_id="strict-wave-000007")
    assert first == second
    assert len(first) == len({(row["task"], row["dataset_hash"]) for row in first}) == 128
    for task in ("regression", "binary"):
        rows = [row for row in first if row["task"] == task]
        assert len(rows) == 64
        assert {
            lane: sum(row["micro_lane"] == lane for row in rows)
            for lane in ("stable_anchor", "least_observed", "hard_uncertain")
        } == {
            "stable_anchor": 16,
            "least_observed": 32,
            "hard_uncertain": 16,
        }


def test_synthetic_families_ignore_seed_and_row_noise_but_keep_recipe():
    one = {
        "origin": "synthetic",
        "generator_recipe": {
            "family": "random_mlp",
            "snr": 4.0,
            "seed": 1,
            "n_train_rows": 100,
        },
    }
    two = {
        "origin": "synthetic",
        "generator_recipe": {
            "family": "random_mlp",
            "snr": 4.0,
            "seed": 99,
            "n_train_rows": 900,
        },
    }
    three = {
        "origin": "synthetic",
        "generator_recipe": {"family": "random_mlp", "snr": 8.0},
    }
    assert source_family(one, "synth_a") == source_family(two, "synth_b")
    assert source_family(one, "synth_a") != source_family(three, "synth_c")


def source_dataset(root: Path, task: str, dataset_id: str, value: str) -> dict:
    source = root / task / dataset_id
    source.mkdir(parents=True)
    (source / "train.csv").write_text(f"0,{value}\n1,{value}\n")
    (source / "test.csv").write_text(f"2,{value}\n3,{value}\n")
    (source / "meta.json").write_text(json.dumps({"name": dataset_id}))
    (source / "signature_manifest.json").write_text("{}\n")
    for name in TRAIN_ENCODING_NAMES:
        (source / name).write_bytes(f"{dataset_id}:{name}".encode())
    return {
        "task": task,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_id,
        "source": str(source),
        "source_family": f"family-{dataset_id}",
        "train_sha256": digest(source / "train.csv"),
        "test_sha256": digest(source / "test.csv"),
    }


def test_expanded_cache_resumes_and_catalog_v2_validates(tmp_path):
    source_root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    rows = [
        source_dataset(source_root, "regression", "a" * 16, "1"),
        source_dataset(source_root, "binary", "b" * 16, "0"),
    ]
    for row in rows:
        first = materialize_cache(row, cache_root=cache_root)
        assert materialize_cache(row, cache_root=cache_root) == first
    pointer = activate_tranche(
        rows,
        cache_root=cache_root,
        catalog_root=tmp_path / "catalog",
        tranche_id="tranche-test",
        context_encoder=lambda cache, task: (
            hashlib.sha256(f"{task}:{cache.name}".encode()).digest()
            * (8192 // 32)
        ),
    )
    authority = validate_catalog_manifest(
        tmp_path / "catalog" / "current.json"
    )
    assert authority["catalog_version"] == 2
    assert authority["task_counts"] == {
        "regression": 1,
        "binary": 1,
        "timeseries": 0,
        "multiclass": 0,
    }
    assert pointer["row_count"] == 2


def test_mirror_conflict_fails_closed_and_writes_receipt(tmp_path):
    source = tmp_path / "source" / "regression" / ("a" * 16)
    mirror = tmp_path / "mirror" / "regression" / ("a" * 16)
    source.mkdir(parents=True)
    mirror.mkdir(parents=True)
    for name in ("train.csv", "test.csv"):
        (source / name).write_text("0,1\n")
        (mirror / name).write_text("0,1\n")
    (mirror / "test.csv").write_text("0,2\n")
    receipts = tmp_path / "receipts"
    with pytest.raises(RuntimeError, match="mirror content conflict"):
        compare_mirror(
            "regression",
            "a" * 16,
            source,
            mirror_root=tmp_path / "mirror",
            receipt_root=receipts,
        )
    assert len(list(receipts.glob("*.json"))) == 1


def test_missing_encoding_failure_does_not_publish_partial_cache(tmp_path):
    row = source_dataset(tmp_path / "source", "binary", "c" * 16, "0")
    source = Path(row["source"])
    (source / TRAIN_ENCODING_NAMES[-1]).unlink()
    cache_root = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="missing train encodings"):
        materialize_cache(row, cache_root=cache_root)
    assert not (cache_root / "binary-expanded" / ("c" * 16)).exists()


class SpyEstimator:
    def __init__(self):
        self.fit_rows = None
        self.predict_rows = None

    def fit(self, features, targets):
        self.fit_rows = (features, targets)

    def predict(self, features):
        self.predict_rows = features
        return [2.0, 3.0]


def test_adaptive_fitness_fits_train_once_and_scores_test_without_cv(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text("0,0\n1,1\n")
    test.write_text("2,2\n3,3\n")
    estimator = SpyEstimator()
    result = evaluate_adaptive(
        "regression", train, test, lambda: estimator
    )
    assert len(estimator.fit_rows[0]) == 2
    assert estimator.predict_rows == [[2.0], [3.0]]
    assert result["score"] == 1.0
    assert not result["cross_validation_fitness"]
    assert result["test_adaptive"] and not result["benchmark_valid"]


def test_ranking_labels_include_rank_percentile_and_incumbent_regret():
    rows = [
        {
            "task": "binary",
            "dataset_hash": "d",
            "semantic_sha256": semantic,
            "score": score,
            "secondary_score": loss,
            "valid": True,
        }
        for semantic, score, loss in (("a", 0.8, 0.4), ("b", 0.9, 0.8))
    ]
    ranked = attach_ranking_labels(rows, {("binary", "d"): 0.95})
    assert ranked[0]["semantic_sha256"] == "b"
    assert ranked[0]["within_dataset_rank"] == 1
    assert ranked[0]["within_dataset_percentile"] == 1.0
    assert ranked[0]["regret_to_incumbent"] == pytest.approx(0.05)


def test_gene_evidence_v2_is_decision_complete_and_hash_pinned(tmp_path):
    output = tmp_path / "gene-evidence.v2.jsonl"
    export = write_gene_evidence(
        [
            {
                "task": "binary",
                "dataset_hash": "dataset",
                "semantic_sha256": "1" * 64,
                "protocol_sha256": "2" * 64,
                "score": 0.9,
                "secondary_metric": "log_loss",
                "secondary_score": 0.2,
                "canonical_genome": "(pipe X (ridge))",
                "receipt_sha256": "3" * 64,
                "checkpoint_epoch": f"checkpoint-{'4' * 64}",
                "model_checkpoint_sha256": "4" * 64,
                "within_dataset_rank": 1,
                "within_dataset_percentile": 1.0,
                "regret_to_incumbent": -0.01,
                "compiled_parity_state": "passed",
            }
        ],
        output,
    )
    row = json.loads(output.read_text())
    assert row["schema"] == "JopeGeneEvidence.v2"
    assert row["incumbent_delta"] == pytest.approx(0.01)
    assert row["test_adaptive"] and not row["benchmark_valid"]
    assert export["output_sha256"] == sha256_file(output)


def test_compile_cache_is_dataset_fitted_and_finalists_are_bounded():
    base = {
        "task": "regression",
        "semantic_sha256": "2" * 64,
        "evaluator_protocol_sha256": "3" * 64,
        "compiler_bundle_sha256": "4" * 64,
        "target_triple": "x86_64-unknown-linux-gnu",
    }
    assert compiled_cache_key(train_sha256="1" * 64, **base) != compiled_cache_key(
        train_sha256="5" * 64, **base
    )
    rows = [
        {
            "task": "regression",
            "dataset_hash": "d",
            "semantic_sha256": str(index),
            "score": score,
            "valid": True,
        }
        for index, score in enumerate((0.1, 0.3, 0.2))
    ]
    finalists = compiled_finalists(rows, {("regression", "d"): 0.15})
    assert [row["score"] for row in finalists] == [0.3, 0.2]
    assert exact_parity([0.0], [0.0], interpreted_score=1.0, compiled_score=1.0)[
        "qualified"
    ]


def test_xbabe4_slot_target_ramp_and_three_sample_rollback():
    policy = (
        Path(__file__).parents[1] / "manifests" / "selfplay" / "scq-nodes.v3.toml"
    )
    assert validate_selfplay_policy(policy)["nodes"]["xbabe4"][
        "cpu_job_slot_multiplier"
    ] == 2.0
    assert node_slot_target(20, 2.0) == 40
    state = {
        "stage": "baseline",
        "effective_cores": 20,
        "stage_started_at_ms": 0,
        "baseline": {
            "failure_rate": 0.01,
            "p95_runtime_seconds": 10,
            "throughput_per_slot": 1,
        },
    }
    state = ramp_decision(state, [], now_at_ms=1)
    assert state["slot_target"] == 30
    state = ramp_decision(state, [], now_at_ms=600_001)
    assert state["slot_target"] == 40
    unsafe = [
        {
            "cpu_psi": 0.25,
            "available_memory_fraction": 0.5,
            "failure_rate": 0.01,
            "p95_runtime_seconds": 10,
            "throughput_per_slot": 1,
        }
    ] * 3
    state = ramp_decision(state, unsafe, now_at_ms=700_000)
    assert state["stage"] == "rolled_back"
    assert state["slot_target"] == 20


def test_gp_export_has_all_64_and_injection_preserves_budget(tmp_path):
    proposals = tmp_path / "proposals.jsonl"
    checkpoint = "8" * 64
    rows = []
    for slot in range(64):
        genome = f"(pipe X (cheb degree={slot + 2}) (ridge))"
        rows.append(
            {
                "task": "regression",
                "dataset_hash": "d",
                "slot": slot,
                "canonical_genome": genome,
                "semantic_sha256": sha256_text(genome),
                "model_checkpoint_sha256": checkpoint,
                "score": slot / 64,
            }
        )
    proposals.write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    exported = export_gp_seeds(
        proposals,
        task="regression",
        dataset_hash="d",
        checkpoint_sha256=checkpoint,
    )
    assert len(exported["proposals"]) == 64
    population = [
        {
            "semantic_sha256": f"old-{index}",
            "canonical_genome": f"old-{index}",
            "score": float(index),
        }
        for index in range(16)
    ]
    injected, receipt = inject_promoted_proposals(
        population,
        rows,
        checkpoint_sha256=checkpoint,
        elite_count=4,
    )
    assert len(injected) == len(population) == 16
    assert receipt["injected"] == 8
