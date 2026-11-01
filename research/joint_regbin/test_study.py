"""Synthetic CPU protocol fixtures; no learner, GPU, dataset or score history."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from research.joint_regbin import candidate_builder, driver, runtime, worker
from research.joint_regbin.admission import (
    PARTIAL_SCHEMA,
    _load_csv,
    build_family_key_pool_manifest,
    review_binary_families,
    verify_family_key_pool_manifest,
)
from research.joint_regbin.driver import (
    collect,
    output_location,
    qualification_lock,
    request,
    write_new,
)
from research.joint_regbin.protocol import (
    ARMS,
    DEFAULTS,
    DOPE,
    SCHEMA,
    admit_result,
    batches,
    digest,
    expected_counts,
    make_plan,
    safe_path,
    validate_manifest,
    verify_catalog,
    verify_plan,
)
from research.joint_regbin.report import comparison, decide, elo, paired_records, render


def _write_family_exclusion(
    tmp_path, *, catalog_sha, keys=(), ids=(), monkeypatch=None
):
    from research.joint_regbin import admission

    artifact = {
        "schema": "jope.jrb01.sealed-family-exclusion/v1",
        "identity_manifest_sha256": admission.PROTECTED_FAMILY_MANIFEST_SHA256,
        "catalog_sha256": catalog_sha,
        "family_keys": [{"kind": kind, "value": value} for kind, value in sorted(keys)],
        "unkeyed_catalog_ids": sorted(ids),
    }
    path = tmp_path / "sealed-family-exclusion.json"
    path.write_text(json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n")
    import hashlib

    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if monkeypatch is not None:
        monkeypatch.setattr(admission, "FAMILY_EXCLUSION_SHA256", artifact_sha)
    return path, artifact_sha


def test_family_key_pool_manifest_certifies_catalog_eligibility_and_assigns_roles(
    tmp_path,
    monkeypatch,
):
    from research.joint_regbin import admission

    rows = []
    for task, prefix, start in (("regression", "r", 1), ("binary", "b", 101)):
        for index in range(6):
            canonical = f"{start + index:064x}"
            if index == 0:
                canonical = "f" * 64
            rows.append(
                {
                    "id": f"{prefix}-{index}",
                    "task": task,
                    "canonical_sha256": canonical,
                    "s3_bucket": "veox-jopedime-datasets-use2",
                    "train_rows": 12,
                    "train_sha256": f"{start + index + 300:064x}",
                }
            )
    catalog = {
        "inventory": {
            "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
            "usable_records": len(rows),
            "catalog_records": len(rows),
            "sealed_name_matches_excluded": [],
        },
        "datasets": rows,
    }

    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", "a" * 64)
    exclusion_path, exclusion_sha = _write_family_exclusion(
        tmp_path, catalog_sha="a" * 64, monkeypatch=monkeypatch
    )
    manifest = build_family_key_pool_manifest(
        catalog,
        "a" * 64,
        family_exclusion_path=exclusion_path,
        family_exclusion_sha256=exclusion_sha,
    )

    assert manifest["schema"] == "jope.joint-regbin.family-key-pools/v1"
    assert (
        manifest["eligibility_verified"] is False
    )  # full worker admission remains separate
    assert manifest["payloads_opened"] == 0
    assert manifest["candidate_count_by_task"] == {"regression": 6, "binary": 6}
    assert manifest["family_by_dataset"]["r-0"] == manifest["family_by_dataset"]["b-0"]
    assert manifest["family_keys_by_dataset"]["r-0"] == [["canonical_sha256", "f" * 64]]
    assert len(manifest["fold_by_family"]) == 11
    assert len(manifest["fold_splits"]) == 5
    assert any(
        split[role][task]
        for split in manifest["fold_splits"].values()
        for role in ("train", "val")
        for task in ("regression", "binary")
    )
    assert manifest["fold_seed"] == 136
    eligibility = manifest["nonsealed_eligibility"]
    assert eligibility["status"] == "CATALOG_USABLE_NONSEALED_MEMBERSHIP_VERIFIED"
    assert eligibility["eligibility_verified"] is True
    assert eligibility["verified_family_count"] == 11
    assert eligibility["excluded_family_count"] == 0
    assert eligibility["excluded_dataset_count"] == 0


def test_family_key_pool_preflight_rederives_and_stays_non_admitted(
    tmp_path, monkeypatch
):
    from research.joint_regbin import admission

    rows = []
    for task, prefix, start in (("regression", "r", 1), ("binary", "b", 101)):
        for index in range(20):
            rows.append(
                {
                    "id": f"{prefix}-{index}",
                    "task": task,
                    "canonical_sha256": f"{start + index:064x}",
                    "s3_bucket": "veox-jopedime-datasets-use2",
                    "train_rows": 12,
                    "train_sha256": f"{start + index + 300:064x}",
                }
            )
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "inventory": {
                    "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
                    "usable_records": len(rows),
                    "catalog_records": len(rows),
                    "sealed_name_matches_excluded": [],
                },
                "datasets": rows,
            }
        )
    )
    from research.joint_regbin.admission import sha256_file

    catalog_sha = sha256_file(catalog_path)
    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", catalog_sha)
    exclusion_path, exclusion_sha = _write_family_exclusion(
        tmp_path, catalog_sha=catalog_sha, monkeypatch=monkeypatch
    )
    manifest = build_family_key_pool_manifest(
        json.loads(catalog_path.read_text()),
        catalog_sha,
        family_exclusion_path=exclusion_path,
        family_exclusion_sha256=exclusion_sha,
    )
    receipt = verify_family_key_pool_manifest(
        manifest,
        catalog_path,
        family_exclusion_path=exclusion_path,
        family_exclusion_sha256=exclusion_sha,
    )
    assert receipt["status"] == "METADATA_SPLIT_VERIFIED"
    assert receipt["family_count"] == 40
    assert receipt["fold_count"] == 5
    assert receipt["eligibility_verified"] is False
    assert receipt["payloads_opened"] == 0
    manifest["fold_splits"] = {"0": {"train": {"binary": ["b-0"]}}}
    with pytest.raises(ValueError, match="differs from pinned catalog split"):
        verify_family_key_pool_manifest(
            manifest,
            catalog_path,
            family_exclusion_path=exclusion_path,
            family_exclusion_sha256=exclusion_sha,
        )


def test_family_key_pool_rejects_reintroduced_sealed_name_exclusion(monkeypatch):
    from research.joint_regbin import admission

    catalog = {
        "inventory": {
            "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
            "catalog_records": 2,
            "usable_records": 1,
            "sealed_name_matches_excluded": ["d-0"],
        },
        "datasets": [
            {
                "id": "d-0",
                "task": "regression",
                "canonical_sha256": "a" * 64,
                "s3_bucket": "veox-jopedime-datasets-use2",
                "train_rows": 12,
                "train_sha256": "b" * 64,
            }
        ],
    }
    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", "a" * 64)
    with pytest.raises(ValueError, match="consistent pinned v246 eligibility metadata"):
        build_family_key_pool_manifest(catalog, "a" * 64)


def test_family_exclusion_artifact_is_pinned_and_closes_union_find_components(
    tmp_path, monkeypatch
):
    from research.joint_regbin import admission

    rows = []
    specifications = [
        ("r-sealed", "regression", "r", "sealed-fp", None, None),
        ("r-bridge", "regression", "r", "sealed-fp", "srcfile:sealed", None),
        ("r-linked", "regression", "r", None, "srcfile:sealed", None),
        ("r-heldout", "regression", "r", "heldout-fp", None, "heldout"),
        ("r-dev", "regression", "r", "dev-fp", None, "dev"),
        ("r-eligible-3", "regression", "r", "reg-fp-3", None, None),
        ("r-eligible-4", "regression", "r", "reg-fp-4", None, None),
        ("r-eligible-5", "regression", "r", "reg-fp-5", None, None),
        ("r-eligible-6", "regression", "r", "reg-fp-6", None, None),
        ("b" * 16, "binary", "b", None, None, None),
        ("b-allowed", "binary", "b", "allowed-fp", None, None),
        ("b-allowed-3", "binary", "b", "bin-fp-3", None, None),
        ("b-allowed-4", "binary", "b", "bin-fp-4", None, None),
        ("b-allowed-5", "binary", "b", "bin-fp-5", None, None),
        ("b-allowed-6", "binary", "b", "bin-fp-6", None, None),
    ]
    for index, (dataset_id, task, _prefix, key, derived, p3_status) in enumerate(
        specifications
    ):
        row = {
            "id": dataset_id,
            "task": task,
            "canonical_sha256": f"{index + 1:064x}",
            "s3_bucket": "veox-jopedime-datasets-use2",
            "train_rows": 12,
            "train_sha256": f"{index + 100:064x}",
        }
        if key:
            row.update(
                family_key=key, family_key_source="provenance.dedupe_fingerprint"
            )
        if derived:
            row.update(family_key_derived=derived)
        if p3_status:
            row["p3_status_fixture_only"] = p3_status
        rows.append(row)
    catalog = {
        "inventory": {
            "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
            "usable_records": len(rows),
            "catalog_records": len(rows),
            "sealed_name_matches_excluded": [],
        },
        "datasets": rows,
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    import hashlib

    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", catalog_sha)
    exclusion_path, exclusion_sha = _write_family_exclusion(
        tmp_path,
        catalog_sha=catalog_sha,
        keys=[("dedupe_fingerprint", "sealed-fp")],
        ids=["b" * 16],
        monkeypatch=monkeypatch,
    )
    manifest = build_family_key_pool_manifest(
        catalog,
        catalog_sha,
        family_exclusion_path=exclusion_path,
        family_exclusion_sha256=exclusion_sha,
    )
    eligible = set(manifest["family_by_dataset"])
    assert {"r-heldout", "r-dev", "b-allowed"} <= eligible
    assert not eligible.intersection({"r-sealed", "r-bridge", "r-linked", "b" * 16})
    role_ids = {
        dataset_id
        for split in manifest["fold_splits"].values()
        for role in ("train", "val")
        for task in admission.protocol.TASKS
        for dataset_id in split[role][task]
    }
    assert {"r-heldout", "r-dev", "b-allowed"} <= role_ids
    assert manifest["nonsealed_eligibility"]["excluded_dataset_count"] == 4
    assert manifest["nonsealed_eligibility"]["excluded_family_count"] == 2


def test_family_exclusion_artifact_missing_or_hash_mismatched_fails_closed(
    tmp_path, monkeypatch
):
    from research.joint_regbin import admission

    rows = [
        {
            "id": "r-0",
            "task": "regression",
            "canonical_sha256": "a" * 64,
            "s3_bucket": "veox-jopedime-datasets-use2",
            "train_rows": 12,
            "train_sha256": "b" * 64,
        }
    ]
    catalog = {
        "inventory": {
            "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
            "usable_records": 1,
            "catalog_records": 1,
            "sealed_name_matches_excluded": [],
        },
        "datasets": rows,
    }
    catalog_sha = "a" * 64
    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", catalog_sha)
    path, actual_sha = _write_family_exclusion(
        tmp_path, catalog_sha=catalog_sha, monkeypatch=monkeypatch
    )
    with pytest.raises(ValueError, match="exclusion artifact"):
        build_family_key_pool_manifest(
            catalog,
            catalog_sha,
            family_exclusion_path=tmp_path / "missing.json",
            family_exclusion_sha256=actual_sha,
        )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="exclusion artifact"):
        build_family_key_pool_manifest(
            catalog,
            catalog_sha,
            family_exclusion_path=path,
            family_exclusion_sha256=actual_sha,
        )


@pytest.mark.parametrize("claimed_nonsealed", [None, False, True])
def test_family_key_pool_uses_owner_eligibility_not_self_reported_flags(
    tmp_path, monkeypatch, claimed_nonsealed
):
    """Synthetic identity graph only; no protected roster or payload is an input."""
    from research.joint_regbin import admission

    rows = []
    for task, prefix, start in (("regression", "r", 1), ("binary", "b", 101)):
        for index in range(12):
            rows.append(
                {
                    "id": f"{prefix}-{index}",
                    "task": task,
                    "canonical_sha256": f"{start + index:064x}",
                    "s3_bucket": "veox-jopedime-datasets-use2",
                    "train_rows": 12,
                    "train_sha256": f"{start + index + 300:064x}",
                    # A self-reported flag in catalog data is not owner evidence.
                    "nonsealed": claimed_nonsealed,
                }
            )
    by_id = {row["id"]: row for row in rows}
    by_id["r-0"].update(canonical_sha256="a" * 64, dataset_hash="d" * 64)
    by_id["b-0"].update(canonical_sha256="a" * 64, dataset_hash="e" * 64)
    by_id["r-1"].update(dataset_hash="e" * 64)
    catalog = {
        "inventory": {
            "source_catalog": "catalog/v246.jsonl (LATEST=v246)",
            "usable_records": len(rows),
            "catalog_records": len(rows),
            "sealed_name_matches_excluded": [],
        },
        "datasets": rows,
    }
    monkeypatch.setattr(admission, "CATALOG_V246_SHA256", "a" * 64)
    exclusion_path, exclusion_sha = _write_family_exclusion(
        tmp_path, catalog_sha="a" * 64, monkeypatch=monkeypatch
    )
    result = build_family_key_pool_manifest(
        catalog,
        "a" * 64,
        family_exclusion_path=exclusion_path,
        family_exclusion_sha256=exclusion_sha,
    )
    family = result["family_by_dataset"]
    assert family["r-0"] == family["b-0"] == family["r-1"]
    assigned = {
        dataset
        for split in result["fold_splits"].values()
        for role in ("train", "val")
        for task in ("regression", "binary")
        for dataset in split[role][task]
    }
    assert assigned, "owner-approved usable catalog rows received no roles"
    assert result["fold_by_family"]
    assert result["nonsealed_eligibility"]["excluded_family_count"] == 0
    assert result["nonsealed_eligibility"]["excluded_dataset_count"] == 0
    assert result["eligibility_verified"] is False
    assert result["payloads_opened"] == 0


@pytest.fixture
def manifest():
    rows = []
    for role, count in (("train", 8), ("val", 5), ("panel", 5)):
        for task in ("regression", "binary"):
            for index in range(count):
                family = f"fixture-{role}-{task}-{index}"
                rows.append(
                    dict(
                        id=family,
                        dataset=family,
                        family=family,
                        task=task,
                        role=role,
                        source_split="train",
                        origin="real",
                        has_meta_test=False,
                        payload=f"fixture-packet/{family}.npz",
                        payload_sha256="1" * 64,
                    )
                )
    return dict(
        schema=SCHEMA,
        eligibility_verified=True,
        catalog_sha256="2" * 64,
        learner_sha256="3" * 64,
        initial_checkpoint_sha256="4" * 64,
        contexts=rows,
    )


@pytest.fixture
def plan(manifest):
    return make_plan(manifest)


def test_implementation_hash_tracks_admission_code(monkeypatch):
    included = []

    def fake_file_sha(path):
        included.append(Path(path).name)
        return "a" * 64

    monkeypatch.setattr("research.joint_regbin.protocol.file_sha", fake_file_sha)
    from research.joint_regbin.protocol import implementation_sha

    implementation_sha()
    assert "admission.py" in included


@pytest.fixture
def preflight_inputs(manifest, tmp_path):
    """Hash-only synthetic files, never a real pool, model or learner."""
    from research.joint_regbin.protocol import file_sha

    packet = tmp_path / "packet"
    packet.mkdir()
    worker, checkpoint = tmp_path / "learner.py", tmp_path / "initial.pt"
    worker.write_bytes(b"# synthetic learner; never executed\n")
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded\n")
    manifest = json.loads(json.dumps(manifest))
    manifest["learner_sha256"] = file_sha(worker)
    manifest["initial_checkpoint_sha256"] = file_sha(checkpoint)
    for row in manifest["contexts"]:
        path = packet / row["payload"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic pool; hash only\n")
        row["payload_sha256"] = file_sha(path)
    return make_plan(manifest), packet, worker, checkpoint


def test_preflight_verifies_inputs_without_qualifying_study(preflight_inputs):
    frozen, packet, worker, checkpoint = preflight_inputs
    result = driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)
    assert result == dict(
        schema="jope.joint-regbin.preflight/v1",
        status="INPUTS_VERIFIED",
        plan_sha256=frozen["sha256"],
        contexts=36,
        jobs=54,
        learner_sha256=frozen["learner_sha256"],
        initial_checkpoint_sha256=frozen["initial_checkpoint_sha256"],
        qualification_status="NOT_QUALIFIED",
        verdict=None,
    )


@pytest.mark.parametrize("changed", ["worker", "checkpoint", "payload"])
def test_preflight_refuses_input_drift(preflight_inputs, changed):
    frozen, packet, worker, checkpoint = preflight_inputs
    path = {
        "worker": worker,
        "checkpoint": checkpoint,
        "payload": packet / frozen["contexts"][-1]["payload"],
    }[changed]
    path.write_bytes(b"changed synthetic input\n")
    with pytest.raises(ValueError, match="drift"):
        driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)


def test_preflight_refuses_missing_input(preflight_inputs):
    frozen, packet, worker, checkpoint = preflight_inputs
    (packet / frozen["contexts"][0]["payload"]).unlink()
    with pytest.raises(FileNotFoundError):
        driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)


@pytest.mark.parametrize("target", ["packet", "worker", "checkpoint", "payload"])
def test_preflight_resolves_all_protected_paths_before_hashing(
    preflight_inputs, tmp_path, monkeypatch, target
):
    frozen, packet, worker, checkpoint = preflight_inputs
    if target == "packet":
        packet = tmp_path / "sealed" / "packet"
    elif target == "worker":
        worker = tmp_path / "sealed" / "learner.py"
    elif target == "checkpoint":
        checkpoint = tmp_path / "sealed" / "initial.pt"
    else:
        # The last allowed name resolves to a nonexistent protected target.
        # Earlier permitted files must remain unopened too.
        path = packet / frozen["contexts"][-1]["payload"]
        path.unlink()
        path.symlink_to(packet / "sealed" / "never-open.npz")

    def forbidden_hash(_path):
        pytest.fail("an input was opened before all path guards passed")

    monkeypatch.setattr(driver, "file_sha", forbidden_hash)
    with pytest.raises(ValueError, match="protected"):
        driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)


@pytest.mark.parametrize("rehash", [False, True])
def test_preflight_rejects_plan_drift_before_input_reads(
    preflight_inputs, monkeypatch, rehash
):
    frozen, packet, worker, checkpoint = preflight_inputs

    def forbidden_hash(_path):
        pytest.fail("input opened for a changed plan")

    monkeypatch.setattr(driver, "file_sha", forbidden_hash)
    changed = dict(frozen, dope_sha256="0" * 64)
    if rehash:
        changed["sha256"] = digest({k: v for k, v in changed.items() if k != "sha256"})
    with pytest.raises(ValueError, match="plan changed|DOPE pin"):
        driver._preflight_verified_inputs(changed, packet, worker, checkpoint)


def test_preflight_cli_has_no_study_side_effects(
    preflight_inputs, tmp_path, monkeypatch, capsys
):
    import sys

    frozen, packet, worker, checkpoint = preflight_inputs
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(frozen))
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")}

    def forbidden(*_args, **_kwargs):
        pytest.fail("CPU preflight attempted a study side effect")

    for name in (
        "qualification_lock",
        "reserve_campaign",
        "worker_call",
        "append_score_store",
    ):
        monkeypatch.setattr(driver, name, forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "preflight",
            str(plan_path),
            "--packet",
            str(packet),
            "--worker",
            str(worker),
            "--checkpoint",
            str(checkpoint),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        driver.main()
    assert stopped.value.code == 2
    assert "worker-bound r882 plan identity mismatch" in capsys.readouterr().err
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")} == before


def test_preflight_rechecks_input_drift_before_gpu_acquisition(
    preflight_inputs, tmp_path
):
    frozen, packet, worker, checkpoint = preflight_inputs
    assert (
        driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)["status"]
        == "INPUTS_VERIFIED"
    )
    (packet / frozen["contexts"][-1]["payload"]).write_bytes(b"later drift")

    with pytest.raises(ValueError, match="payload drift"):
        driver._preflight_verified_inputs(frozen, packet, worker, checkpoint)
    assert not (tmp_path / "campaign").exists()


@pytest.mark.parametrize("missing", ["worker", "protected_plan"])
def test_preflight_cli_refuses_unavailable_inputs(
    preflight_inputs, tmp_path, monkeypatch, capsys, missing
):
    import sys

    frozen, packet, worker, checkpoint = preflight_inputs
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(frozen))
    if missing == "worker":
        worker.unlink()
        diagnostic = "worker-bound r882 plan identity mismatch"
    else:
        plan_path = tmp_path / "sealed" / "never-open.json"
        diagnostic = "protected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "driver",
            "preflight",
            str(plan_path),
            "--packet",
            str(packet),
            "--worker",
            str(worker),
            "--checkpoint",
            str(checkpoint),
        ],
    )
    with pytest.raises(SystemExit) as stopped:
        driver.main()
    assert stopped.value.code == 2
    assert diagnostic in capsys.readouterr().err


def receipt_for(plan, job):
    config = plan["config"]
    orders = list(batches(job, config))
    traces, previous = [], digest("fixture-initial")
    before = previous
    for number in range(1, config["rounds"] + 1):
        after = digest(["fixture-parameters", number])
        end = number * config["steps"]
        traces.append(
            dict(
                round=number,
                optimizer_steps=config["steps"],
                parameters_before=previous,
                parameters_after=after,
                dope_sha256=DOPE,
                checkpoint_sha256=digest(["fixture-model", number]),
                batch_sha256=digest(orders[end - config["steps"] : end]),
            )
        )
        previous = after
    scores = []
    for number in range(
        config["eval_every"], config["rounds"] + 1, config["eval_every"]
    ):
        for row in job["evaluation"]:
            scores.append(
                dict(
                    round=number,
                    id=row["id"],
                    status="ok",
                    metric="rmse" if row["task"] == "regression" else "auc",
                    value=1.1 if row["task"] == "regression" else 0.8,
                    baseline=1.0 if row["task"] == "regression" else 0.7,
                    curve=math.log(1.1) if row["task"] == "regression" else 0.4,
                    model_sha256=digest(["fixture-model", number]),
                    evaluation_sha256=row["payload_sha256"],
                    evaluation_protocol="cached-pool-jrb01/v1",
                )
            )
    return dict(
        plan_sha256=plan["sha256"],
        job_id=job["id"],
        learner_sha256=plan["learner_sha256"],
        initial_checkpoint_sha256=plan["initial_checkpoint_sha256"],
        actual_budget=expected_counts(config),
        dope_before=DOPE,
        dope_after=DOPE,
        parameters_before=before,
        parameters_after=previous,
        round_receipts=traces,
        scores=scores,
    )


def test_plan_is_reproducible_and_all_54_jobs_have_equal_budgets(plan, manifest):
    assert plan == make_plan(manifest)
    assert len(plan["jobs"]) == 54
    assert len({job["id"] for job in plan["jobs"]}) == 54
    expected = expected_counts(plan["config"])
    for job in plan["jobs"]:
        orders = list(batches(job, plan["config"]))
        assert len(orders) == expected["optimizer_steps"] == 100
        assert sum(map(len, orders)) == expected["context_presentations"] == 3200
        assert expected["conditioning_token_slots"] == 32000
        training = {row["id"] for row in job["train"]}
        evaluation_families = {row["family"] for row in job["evaluation"]}
        assert not evaluation_families & {row["family"] for row in job["train"]}
        assert {value for order in orders for value in order} <= training


def test_three_arm_sampling_is_prespecified(plan):
    config = plan["config"]
    for arm in ARMS:
        job = next(
            job for job in plan["jobs"] if job["arm"] == arm and job["fold"] == "full"
        )
        by_id = {row["id"]: row["task"] for row in job["train"]}
        counts = Counter(
            by_id[value] for order in batches(job, config) for value in order
        )
        if arm == "joint":
            assert counts == dict(regression=1600, binary=1600)
        else:
            assert counts == {"regression" if arm == "reg-only" else "binary": 3200}


def test_joint_arm_uses_balanced_sampling_at_the_same_declared_budget(plan):
    assert set(ARMS) == {"joint", "reg-only", "bin-only"}
    expected = expected_counts(plan["config"])
    for arm in ARMS:
        job = next(j for j in plan["jobs"] if j["arm"] == arm and j["fold"] == "full")
        batches_for_job = list(batches(job, plan["config"]))
        assert len(batches_for_job) == expected["optimizer_steps"]
        assert sum(map(len, batches_for_job)) == expected["context_presentations"]
        if arm == "joint":
            tasks = {row["id"]: row["task"] for row in job["train"]}
            counts = Counter(
                tasks[context] for batch in batches_for_job for context in batch
            )
            assert counts == {"regression": 1600, "binary": 1600}


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "DL-Holdout"),
        ("source_split", "outer"),
        ("has_meta_test", True),
        ("origin", "synthetic"),
        ("family", ""),
        ("payload", "sealed/rows.npz"),
        ("payload", "../escape.npz"),
    ],
)
def test_forbidden_metadata_refused_before_payload_open(manifest, field, value):
    manifest["contexts"][0][field] = value
    with pytest.raises(ValueError):
        make_plan(manifest)


def test_all_regression_packet_cannot_be_called_joint(manifest):
    manifest["contexts"] = [
        row for row in manifest["contexts"] if row["task"] == "regression"
    ]
    with pytest.raises(ValueError, match="binary"):
        make_plan(manifest)


def test_train_val_family_overlap_is_refused(manifest):
    manifest["contexts"][16]["family"] = manifest["contexts"][0]["family"]
    with pytest.raises(ValueError, match="overlap"):
        make_plan(manifest)


def test_panel_family_overlap_is_evaluated_only_by_held_family_model(manifest):
    row = next(row for row in manifest["contexts"] if row["role"] == "panel")
    row["family"] = manifest["contexts"][0]["family"]
    plan = make_plan(manifest)
    jobs = [
        job
        for job in plan["jobs"]
        if any(r["id"] == row["id"] for r in job["evaluation"])
    ]
    assert len(jobs) == 9  # three arms × three seeds, exactly once each
    assert all(job["fold"] == plan["folds"][row["family"]] for job in jobs)
    assert all(row["family"] not in {r["family"] for r in job["train"]} for job in jobs)


def test_multiclass_metadata_excluded_without_reading_payload(manifest):
    extra = dict(
        manifest["contexts"][0],
        id="fixture-multiclass",
        dataset="fixture-multiclass",
        family="fixture-multiclass",
        task="multiclass",
        payload="fixture-packet/absent.npz",
    )
    manifest["contexts"].append(extra)
    plan = make_plan(manifest)
    assert plan["excluded_multiclass"] == 1
    assert all(row["task"] != "multiclass" for row in plan["contexts"])


def test_symlink_escape_rejected(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="symlink"):
        safe_path(tmp_path, "link/unknown")


def test_in_packet_symlink_to_protected_target_rejected(tmp_path):
    (tmp_path / "sealed").mkdir()
    (tmp_path / "sealed" / "rows.npz").write_bytes(b"x")
    (tmp_path / "public.npz").symlink_to(tmp_path / "sealed" / "rows.npz")
    with pytest.raises(ValueError, match="protected"):
        safe_path(tmp_path, "public.npz")
    (tmp_path / "data").symlink_to(tmp_path / "sealed")
    with pytest.raises(ValueError, match="protected"):
        safe_path(tmp_path, "data/rows.npz")


def test_metadata_planning_does_not_resolve_payloads_against_cwd(
    manifest, tmp_path, monkeypatch
):
    before = make_plan(manifest)
    (tmp_path / "fixture-packet").symlink_to(tmp_path.parent)
    monkeypatch.chdir(tmp_path)
    assert make_plan(manifest) == before
    with pytest.raises(ValueError, match="symlink"):
        safe_path(tmp_path, manifest["contexts"][0]["payload"])


def test_preregistration_tampering_and_weak_budget_refused(plan, manifest):
    plan["config"]["rounds"] = 10
    with pytest.raises(ValueError, match="changed"):
        verify_plan(plan)
    config = dict(DEFAULTS, rounds=10)
    with pytest.raises(ValueError, match=">=50"):
        make_plan(manifest, config)


def test_actual_pinned_catalog_identity_required(manifest, tmp_path):
    path = tmp_path / "fixture-catalog.json"
    path.write_text(
        json.dumps(
            {
                "datasets": [
                    dict(id=row["dataset"], family_key=row["family"], task=row["task"])
                    for row in manifest["contexts"]
                ]
            }
        )
    )
    from research.joint_regbin.protocol import file_sha

    manifest["catalog_sha256"] = file_sha(path)
    verify_catalog(manifest, path)
    manifest["contexts"][0]["task"] = "binary"
    with pytest.raises(ValueError, match="authoritative"):
        verify_catalog(manifest, path)


def test_training_request_contains_no_evaluation_paths(plan):
    job = plan["jobs"][0]
    payload = json.dumps(request(plan, job, "train"))
    assert not any(row["payload"] in payload for row in job["evaluation"])
    assert "evaluation" not in request(plan, job, "train")


def test_rehashed_plan_still_cannot_admit_training_leakage(plan):
    plan["jobs"][0]["train"].append(plan["jobs"][0]["evaluation"][0])
    plan["sha256"] = digest(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="closure"):
        verify_plan(plan)


def test_score_checkpoint_must_match_training_round(plan):
    job = plan["jobs"][0]
    receipt = receipt_for(plan, job)
    receipt["scores"][0]["model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        admit_result(plan, job, receipt)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation_sha256", "f" * 64),
        ("evaluation_protocol", "different-evaluator/v2"),
    ],
)
def test_evaluation_input_identity_must_match_frozen_context(plan, field, value):
    job = plan["jobs"][0]
    receipt = receipt_for(plan, job)
    receipt["scores"][0][field] = value
    with pytest.raises(ValueError, match="evaluation input"):
        admit_result(plan, job, receipt)


def test_preregistered_source_and_budget_changes_refused(plan, manifest):
    plan["implementation_sha256"] = "f" * 64
    plan["sha256"] = digest(
        {key: value for key, value in plan.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="implementation"):
        verify_plan(plan)
    with pytest.raises(ValueError, match="budget differs"):
        make_plan(manifest, dict(DEFAULTS, wall_seconds=DEFAULTS["wall_seconds"] + 1))


def test_worker_deadline_and_final_output_cap_terminate_only_fixture_worker(tmp_path):
    import os
    import sys

    from research.joint_regbin.driver import worker_call

    with (
        (tmp_path / "fixture-worker.log").open("wb") as log,
        (tmp_path / "fixture-unit.lock").open("a") as lock,
    ):
        with pytest.raises(TimeoutError):
            worker_call(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                dict(os.environ),
                lock.fileno(),
                log,
                0.1,
                tmp_path,
                10000,
            )
        with pytest.raises(RuntimeError, match="output budget"):
            worker_call(
                [sys.executable, "-c", "open('fixture-big','wb').write(b'x'*10001)"],
                dict(os.environ),
                lock.fileno(),
                log,
                5,
                tmp_path,
                10000,
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "budget",
        "frozen",
        "no_update",
        "round",
        "batch",
        "partial",
        "duplicate",
        "metric",
        "nonfinite",
        "source",
    ],
)
def test_result_admission_negative_controls(plan, mutation):
    job = plan["jobs"][0]
    receipt = receipt_for(plan, job)
    assert admit_result(plan, job, receipt) == receipt
    if mutation == "budget":
        receipt["actual_budget"]["optimizer_steps"] -= 1
    if mutation == "frozen":
        receipt["dope_after"] = "f" * 64
    if mutation == "no_update":
        receipt["parameters_after"] = receipt["parameters_before"]
    if mutation == "round":
        receipt["round_receipts"].pop()
    if mutation == "batch":
        receipt["round_receipts"][0]["batch_sha256"] = "0" * 64
    if mutation == "partial":
        receipt["scores"].pop()
    if mutation == "duplicate":
        receipt["scores"].append(receipt["scores"][0])
    if mutation == "metric":
        receipt["scores"][0]["metric"] = "accuracy"
    if mutation == "nonfinite":
        receipt["scores"][0]["value"] = float("nan")
    if mutation == "source":
        receipt["learner_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        admit_result(plan, job, receipt)


def test_complete_collection_and_no_partial_inventory(plan, tmp_path):
    with pytest.raises(ValueError, match="missing/extra"):
        collect(plan, tmp_path)
    for job in plan["jobs"]:
        write_new(
            tmp_path / f"{job['id']}-evaluate.response.json", receipt_for(plan, job)
        )
    rows = collect(plan, tmp_path)
    assert {row["role"] for row in rows} == {"cv", "val", "panel"}
    assert {row["task"] for row in rows} == {"regression", "binary"}
    with pytest.raises(ValueError, match="empty"):
        paired_records(rows, "binary", "panel", 49)


def test_regression_score_store_wiring_is_idempotent_and_task_scoped(tmp_path):
    """The study adapter emits the existing JRB-02 contract without touching history."""
    module = Path(__file__).with_name("score_store.py")
    store = tmp_path / "history.jsonl"
    plan = {"sha256": "f" * 64}
    row = {
        "id": "fixture-regression-1",
        "arm": "joint",
        "seed": 73,
        "fold": 2,
        "job_id": "fixture-regression-job",
        "task": "regression",
        "family": "fixture-family",
        "dataset": "fixture-dataset",
        "role": "cv",
        "metric": "rmse",
        "value": 0.4,
        "curve": 0.8,
        "model_sha256": "a" * 64,
        "round": 5,
    }

    assert driver.append_score_store(plan, [row], module, store) == 2
    assert driver.append_score_store(plan, [row], module, store) == 0
    assert store.with_name("history.jsonl.lock").exists()
    stored = [json.loads(line) for line in store.read_text().splitlines()]
    assert len(stored) == 2
    assert {item["metric"] for item in stored} == {"rmse", "log_rmse_ratio"}
    for item in stored:
        assert set(item) == {
            "schema",
            "ts",
            "model",
            "model_sha",
            "task",
            "dataset",
            "family",
            "split",
            "metric",
            "value",
            "higher_is_better",
            "source",
            "round",
            "host",
        }
        assert item["schema"] == "jope.score-history/v1"
        assert item["ts"].endswith("+00:00")
        assert item["task"] == "regression"
        assert item["dataset"] == "fixture-dataset"
        assert item["family"] == "fixture-family"
        assert item["split"] == "jrb01-cv"
        assert item["higher_is_better"] is False
        assert item["source"] == (
            "jrb01:" + plan["sha256"] + ":fixture-regression-job:"
            "fixture-regression-1:round5"
        )


def test_binary_family_review_promotes_srcfile_and_merges_dedupe_aliases():
    rows = [
        {
            "id": "a",
            "task": "binary",
            "family_key": "a" * 64,
            "family_key_derived": None,
            "canonical_sha256": "1" * 64,
            "source_file": "same_name.csv",
            "source_url": "https://example.test/datasets/same-name.csv",
            "platform": "example",
        },
        {
            "id": "b",
            "task": "binary",
            "family_key": "a" * 64,
            "family_key_derived": None,
            "canonical_sha256": "2" * 64,
            "source_file": "different_name.csv",
            "source_url": "https://example.test/datasets/different-name.csv",
            "platform": "example",
        },
        {
            "id": "c",
            "task": "binary",
            "family_key": None,
            "family_key_derived": "srcfile:same_name",
            "canonical_sha256": "3" * 64,
            "source_file": "same_name.csv",
            "source_url": None,
            "platform": None,
        },
    ]
    reviewed = review_binary_families(rows)
    assert reviewed["schema"] == "jope.joint-regbin.binary-family-review/v1"
    assert reviewed["dataset_count"] == 3
    assert reviewed["family_count"] == 1
    assert len({item["family"] for item in reviewed["records"]}) == 1
    assert ["srcfile", "srcfile:same_name"] in reviewed["families"][0]["union_evidence"]
    assert all("dedupe_tool_policy_sha256" in item for item in reviewed["records"])


def test_family_review_rejects_binary_row_without_any_provisional_family_key():
    with pytest.raises(ValueError, match="lacks a reviewed family key"):
        review_binary_families(
            [
                {
                    "id": "a",
                    "task": "binary",
                    "family_key": None,
                    "family_key_derived": None,
                    "canonical_sha256": None,
                    "source_file": None,
                    "source_url": None,
                    "platform": None,
                }
            ]
        )


def test_binary_csv_loader_uses_only_training_matrix(tmp_path: Path):
    train = tmp_path / "train.csv"
    train.write_text("1,0\n2,1\n3,0\n4,1\n")
    x, y = _load_csv(train, expected_rows=4, expected_cols=2, target_index=-1)
    assert x.tolist() == [[1.0], [2.0], [3.0], [4.0]]
    assert y.tolist() == [0.0, 1.0, 0.0, 1.0]
    assert not (tmp_path / "test.csv").exists()


def test_partial_binary_panel_manifest_fails_closed():
    packet = {
        "schema": SCHEMA,
        "eligibility_verified": False,
        "admission_status": "PARTIAL_PANEL_ONLY",
        "contexts": [
            {
                "id": "synthetic",
                "dataset": "synthetic",
                "family": "srcfile:synthetic",
                "task": "binary",
                "role": "panel",
                "source_split": "train",
                "origin": "real",
                "has_meta_test": False,
                "payload": "binary/synthetic.npz",
                "payload_sha256": "0" * 64,
            }
        ],
    }
    assert PARTIAL_SCHEMA == "jope.joint-regbin.binary-panel-packet/v1"
    with pytest.raises(ValueError, match="explicit eligible real-S3 metadata required"):
        validate_manifest(packet)


def test_family_weighting_seed_pairing_and_elo_orientation():
    records = [
        dict(family="fixture-a", scores={"single": 0.0, "joint": 1.0}),
        dict(family="fixture-b", scores={"single": 1.0, "joint": 0.0}),
    ]
    duplicated = records[:1] * 50 + records[1:]
    assert elo(records, "single")["joint"] == pytest.approx(1500)
    assert elo(duplicated, "single")["joint"] == pytest.approx(1500)
    assert elo(records[:1], "single")["joint"] > 1500
    assert elo(records[1:], "single")["joint"] < 1500


def test_paired_family_bootstrap_is_reproducible():
    records = [
        dict(family=f"fixture-{i}", scores={arm: float(i % 2) for arm in ARMS})
        for i in range(6)
    ]
    first = comparison(records, "regression", 30, 136)
    assert first == comparison(records, "regression", 30, 136)
    assert first["joint"]["delta"] == pytest.approx(0)


def evaluation(reg, binary):
    return dict(
        regression=dict(delta=reg, low=reg - 1, high=reg + 1),
        binary=dict(delta=binary, low=binary - 1, high=binary + 1),
    )


def test_decision_rule_strict_25_boundary_and_two_consecutive_same_task():
    assert decide([evaluation(-25, -25)] * 2) == "JOINT_OK"
    assert decide([evaluation(-30, 0)] * 2) == "SPLIT_RECOMMENDED"
    assert decide([evaluation(-30, 0), evaluation(0, -30)]) == "MITIGATE_AND_RERUN"
    assert decide([evaluation(0, 0), evaluation(-30, 0)]) == "MITIGATE_AND_RERUN"
    ci_zero = evaluation(-30, 0)
    ci_zero["regression"]["high"] = 0
    assert decide([ci_zero] * 2) == "JOINT_OK"


def test_production_output_paths_and_exclusive_lock(tmp_path):
    checkout = tmp_path / "fixture-checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("synthetic marker, not a working copy")
    with pytest.raises(ValueError):
        output_location(checkout / "output")
    with pytest.raises(ValueError):
        output_location("/tmp/jrb01-output")
    # This is a unit lock, never used for GPU execution or substituted in the CLI.
    with qualification_lock(tmp_path / "fixture.lock"):
        with pytest.raises(BlockingIOError):
            with qualification_lock(tmp_path / "fixture.lock"):
                pytest.fail("exclusive lock was stolen")


def test_new_output_path_cannot_reset_existing_campaign_budget(plan, tmp_path):
    from research.joint_regbin.driver import reserve_campaign

    reserve_campaign(plan, tmp_path / "fixture-attempt1", tmp_path / "fixture-ledger")
    ledger = tmp_path / "fixture-ledger" / (plan["sha256"] + ".json")
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="budget reset"):
        reserve_campaign(
            plan, tmp_path / "fixture-attempt2", tmp_path / "fixture-ledger"
        )
    assert ledger.read_bytes() == before


def test_report_produces_standalone_plot_from_explicit_synthetic_fixture(tmp_path):
    tasks = {
        task: {arm: dict(elo=1500, delta=0, low=-1, high=1) for arm in ARMS}
        for task in ("regression", "binary")
    }
    for task in tasks:
        tasks[task]["panel_elo"] = {arm: 1500 for arm in ARMS}
    report = dict(
        verdict=None,
        interpretation="TWO-ROUND SYNTHETIC FIXTURE ONLY; NO STUDY VERDICT",
        comparisons=[dict(tasks=tasks)],
        curves=[
            dict(task=task, arm=arm, round=r, value=0.1)
            for task in tasks
            for arm in ARMS
            for r in (1, 2)
        ],
    )
    render(report, tmp_path)
    assert (tmp_path / "validation-curves.svg").stat().st_size > 1000
    assert "NO VERDICT" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "curves.jsonl").read_text().count("\n") == 12


@pytest.mark.parametrize(
    "token",
    ["holdout", "sealed", "meta-test", "outer_test"],
    ids=["reserved-a", "reserved-b", "reserved-c", "reserved-d"],
)
@pytest.mark.parametrize("kind", ["direct", "resolved", "lexical"])
def test_packet_root_guard(tmp_path, token, kind):
    clean = tmp_path / "clean"
    clean.mkdir()
    public = tmp_path / "public"
    public.symlink_to(clean, target_is_directory=True)
    assert safe_path(clean, "rows.npz") == clean / "rows.npz"
    assert safe_path(public, "rows.npz") == clean / "rows.npz"
    # No payload is created or opened; all paths are synthetic fixture directories.
    protected = tmp_path / token
    protected.mkdir()
    if kind == "direct":
        root = protected
    elif kind == "resolved":
        root = tmp_path / "packet"
        root.symlink_to(protected, target_is_directory=True)
    else:
        root = tmp_path / (token + "-alias")
        root.symlink_to(clean, target_is_directory=True)
    with pytest.raises(ValueError, match="protected"):
        safe_path(root, "rows.npz")


class SyntheticWorkerAdapter(torch.nn.Module):
    """Synthetic stand-in preserving the pinned trainable parameter count."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(29211))

    def predict_costs(self, candidates):
        return {"total_cost": self.weight[0].expand_as(candidates)}


def _worker_context(task="regression", **updates):
    row = {
        "id": "ctx-1",
        "dataset": "dataset-1",
        "family": "family-1",
        "task": task,
        "role": "train",
        "source_split": "train",
        "origin": "real",
        "has_meta_test": False,
        "payload": "packet/ctx-1.npz",
        "payload_sha256": "a" * 64,
    }
    row.update(updates)
    return row


def test_worker_heads_use_explicit_label_and_linear_binary_mapping():
    heads = worker.TwoTaskHeads()
    values = torch.tensor([[1.25], [-0.5]])
    assert torch.equal(heads("regression", values), values.squeeze(-1))
    assert torch.equal(heads("binary", values), values.squeeze(-1))
    with pytest.raises(ValueError, match="unknown task"):
        heads("classification", values)


def test_worker_losses_and_binary_labels_are_strict():
    assert worker.task_loss("regression", torch.tensor([1.0]), torch.tensor([0.0])) > 0
    assert worker.task_loss("binary", torch.tensor([0.0]), torch.tensor([1.0])) > 0
    with pytest.raises(ValueError, match="zero/one"):
        worker.task_loss("binary", torch.tensor([0.0]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="nonfinite"):
        worker.task_loss(
            "regression", torch.tensor([float("nan")]), torch.tensor([0.0])
        )


def test_worker_validates_tel_rows_and_protected_inputs():
    rows = [
        _worker_context(),
        _worker_context("binary", id="ctx-2", family="family-2"),
    ]
    worker.validate_context_rows(rows)
    malformed = [dict(row) for row in rows]
    malformed[0]["task"] = "multiclass"
    with pytest.raises(ValueError, match="explicit regression/binary"):
        worker.validate_context_rows(malformed)
    protected = [dict(row) for row in rows]
    protected[0]["has_meta_test"] = True
    with pytest.raises(ValueError, match="sealed, or holdout"):
        worker.validate_context_rows(protected)
    escaped = [dict(row) for row in rows]
    escaped[0]["payload"] = "../sealed/hidden.npz"
    with pytest.raises(ValueError, match="payload path"):
        worker.validate_context_rows(escaped)


@pytest.mark.parametrize("field", ["id", "dataset", "family"])
def test_worker_refuses_empty_identity_before_packer(field):
    adapter = SyntheticWorkerAdapter()
    heads = worker.TwoTaskHeads()
    optimizer = torch.optim.SGD([*adapter.parameters(), *heads.parameters()], lr=0.1)
    packed = []

    def packer(row):
        packed.append(row)
        return {"candidates": torch.tensor([1.0])}, torch.tensor([1.0])

    core = worker.WorkerCore(adapter, heads, optimizer, packer)
    row = _worker_context(**{field: "  "})

    with pytest.raises(ValueError, match="identity fields must be nonempty"):
        core.train_batch([row])

    assert packed == []


def test_worker_refuses_tel_metadata_candidates_before_payload_packing():
    # TEL role candidates intentionally contain no admitted payload reference.
    candidate = {
        "context": "ctx-1",
        "dataset": "dataset-1",
        "family": "family-1",
        "task": "binary",
        "role": "train",
        "source_split": "train",
        "context_sha256": "a" * 64,
        "pool_sha256": "b" * 64,
        "train_only_jtt1_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="role-manifest schema"):
        worker.validate_context_rows([candidate])


def test_worker_driver_request_checks_roles_and_batch_ids():
    row = _worker_context()
    request_value = {
        "schema": "jope.joint-regbin/v1",
        "phase": "train",
        "contexts": [row],
        "batches": [[row["id"]]],
    }
    worker.validate_request(request_value)
    invalid = {**request_value, "batches": [["unknown-context"]]}
    with pytest.raises(ValueError, match="unknown contexts"):
        worker.validate_request(invalid)
    invalid_rows = [dict(row, role="val")]
    with pytest.raises(ValueError, match="non-training role"):
        worker.validate_request({**request_value, "contexts": invalid_rows})


def test_worker_binary_metrics_use_pinned_task_source():
    module = worker.binary_metrics_module()
    assert callable(module.fit_batched_platt_calibration_gpu)
    result = worker.calibrated_binary_metrics(
        torch.tensor([-2.0, -1.0, 1.0, 2.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
    )
    assert result["log_loss"] >= 0
    assert result["roc_auc"] == pytest.approx(1.0)


def test_worker_updates_shared_adapter_for_regression_and_binary():
    adapter = SyntheticWorkerAdapter()
    heads = worker.TwoTaskHeads()
    optimizer = torch.optim.SGD([*adapter.parameters(), *heads.parameters()], lr=0.1)
    contexts = [
        (
            "reg-ctx",
            "regression",
            {"candidates": torch.tensor([1.0, 2.0])},
            torch.tensor([1.0, 2.0]),
        ),
        (
            "bin-ctx",
            "binary",
            {"candidates": torch.tensor([-1.0, 1.0])},
            torch.tensor([1.0, 1.0]),
        ),
    ]
    result = worker.train_context_batch(adapter, heads, optimizer, contexts)
    assert result["optimizer_steps"] == 1
    assert result["task_contexts"] == {"regression": 1, "binary": 1}
    assert result["parameters_before"] != result["parameters_after"]
    assert result["batch_sha256"] == worker.hash_batch_ids(["reg-ctx", "bin-ctx"])


def test_worker_core_uses_injected_packer_for_synthetic_fixture():
    adapter = SyntheticWorkerAdapter()
    heads = worker.TwoTaskHeads()
    optimizer = torch.optim.SGD([*adapter.parameters(), *heads.parameters()], lr=0.1)

    def packer(_row):
        return {"candidates": torch.tensor([1.0, 2.0])}, torch.tensor([1.0, 2.0])

    core = worker.WorkerCore(adapter, heads, optimizer, packer)
    row = _worker_context()
    result = core.train_batch([row])
    prediction, target = core.predict(row)
    assert result["task_contexts"] == {"regression": 1, "binary": 0}
    assert prediction.shape == target.shape


def test_equal_budget_runner_dispatches_each_arm_through_worker_packer(plan):
    expected = expected_counts(plan["config"])
    rows_by_id = {row["id"]: row for row in plan["contexts"]}
    task_counts = {
        "joint": {"regression": 16, "binary": 16},
        "reg-only": {"regression": 32, "binary": 0},
        "bin-only": {"regression": 0, "binary": 32},
    }

    for arm in ARMS:
        job = next(
            job for job in plan["jobs"] if job["arm"] == arm and job["fold"] == "full"
        )
        train_request = request(plan, job, "train")
        worker.validate_request(train_request)
        assert len(train_request["batches"]) == expected["optimizer_steps"] == 100
        assert all(row["role"] == "train" for row in train_request["contexts"])

        eval_request = request(
            plan, job, "evaluate", previous={"synthetic": "training receipt"}
        )
        worker.validate_request(eval_request)
        assert "training_receipt" in eval_request

        adapter = SyntheticWorkerAdapter()
        heads = worker.TwoTaskHeads()
        optimizer = torch.optim.SGD(
            [*adapter.parameters(), *heads.parameters()], lr=0.1
        )
        packed_ids = []

        def packer(row):
            packed_ids.append(row["id"])
            candidates = torch.linspace(-1.0, 1.0, 16)
            target = (
                torch.tensor([0.0] * 4 + [1.0] * 12)
                if row["task"] == "binary"
                else torch.linspace(0.0, 1.0, 16)
            )
            return {"candidates": candidates}, target

        core = worker.WorkerCore(adapter, heads, optimizer, packer)
        total_steps = 0
        total_contexts = {"regression": 0, "binary": 0}
        for batch in train_request["batches"]:
            contexts = [rows_by_id[context_id] for context_id in batch]
            result = core.train_batch(contexts)
            total_steps += result["optimizer_steps"]
            assert result["task_contexts"] == task_counts[arm]
            for task, count in result["task_contexts"].items():
                total_contexts[task] += count

        assert total_steps == expected["optimizer_steps"] == 100
        assert total_contexts == {
            task: count * expected["optimizer_steps"]
            for task, count in task_counts[arm].items()
        }
        assert len(packed_ids) == expected["optimizer_steps"] * 32


def test_worker_regression_curve_is_log_rmse_over_ridge():
    result = worker.regression_metrics(
        torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0]), ridge_rmse=2.0
    )
    assert result["rmse"] == pytest.approx(2**0.5)
    assert result["log_rmse_ratio"] == pytest.approx(
        torch.log(torch.tensor(2**0.5 / 2.0)).item()
    )


class SplitWorkerAdapter(torch.nn.Module):
    """Two synthetic gradient tensors with the required total parameter count."""

    def __init__(self):
        super().__init__()
        self.left = torch.nn.Parameter(torch.zeros(1))
        self.right = torch.nn.Parameter(torch.zeros(29210))

    def predict_costs(self, candidates):
        return {"total_cost": (self.left[0] + self.right[0]).expand_as(candidates)}


def _worker_update(entry, adapter, heads, optimizer, task):
    packed = {"candidates": torch.ones(2)}
    target = torch.ones(2)
    if entry == "step":
        return worker.train_step(adapter, heads, optimizer, task, packed, target)
    return worker.train_context_batch(
        adapter, heads, optimizer, [("synthetic-context", task, packed, target)]
    )


@pytest.mark.parametrize("entry", ["step", "batch"])
@pytest.mark.parametrize("location", ["adapter", "head"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_worker_gradient_refusal_precedes_any_optimizer_step(
    monkeypatch, entry, location, invalid
):
    adapter = SplitWorkerAdapter()
    heads = worker.TwoTaskHeads()
    parameters = [*adapter.parameters(), *heads.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    step = optimizer.step
    calls = []

    def observed_step(*args, **kwargs):
        calls.append(True)
        return step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", observed_step)
    parameter = adapter.right if location == "adapter" else heads.binary.weight

    def poison_one_gradient(gradient):
        poisoned = gradient.clone()
        poisoned.reshape(-1)[0] = invalid
        return poisoned

    parameter.register_hook(poison_one_gradient)
    before = [parameter.detach().clone() for parameter in parameters]
    with pytest.raises(ValueError, match="nonfinite"):
        _worker_update(
            entry,
            adapter,
            heads,
            optimizer,
            "regression" if location == "adapter" else "binary",
        )
    assert not calls
    assert all(
        torch.equal(original, parameter.detach())
        for original, parameter in zip(before, parameters)
    )
    present = [parameter.grad for parameter in parameters if parameter.grad is not None]
    assert any(bool(torch.isfinite(gradient).all()) for gradient in present)
    assert any(not bool(torch.isfinite(gradient).all()) for gradient in present)


@pytest.mark.parametrize("entry", ["step", "batch"])
@pytest.mark.parametrize("task", ["regression", "binary"])
def test_worker_all_finite_gradients_apply_exactly_one_update(monkeypatch, entry, task):
    adapter = SplitWorkerAdapter()
    heads = worker.TwoTaskHeads()
    parameters = [*adapter.parameters(), *heads.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    step = optimizer.step
    calls = []

    def observed_step(*args, **kwargs):
        calls.append(True)
        return step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", observed_step)
    before = [parameter.detach().clone() for parameter in adapter.parameters()]
    _worker_update(entry, adapter, heads, optimizer, task)
    assert calls == [True]
    assert all(bool(torch.isfinite(parameter).all()) for parameter in parameters)
    assert any(
        not torch.equal(original, parameter.detach())
        for original, parameter in zip(before, adapter.parameters())
    )
    if task == "regression":
        assert all(parameter.grad is None for parameter in heads.parameters())


@pytest.mark.parametrize("entry", ["step", "batch"])
def test_worker_requires_shared_gradient_even_with_finite_head_gradients(
    monkeypatch, entry
):
    class DetachedAdapter(SplitWorkerAdapter):
        def predict_costs(self, candidates):
            return {
                "total_cost": super().predict_costs(candidates)["total_cost"].detach()
            }

    adapter = DetachedAdapter()
    heads = worker.TwoTaskHeads()
    parameters = [*adapter.parameters(), *heads.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    calls = []
    monkeypatch.setattr(optimizer, "step", lambda: calls.append(True))
    before = [parameter.detach().clone() for parameter in parameters]
    with pytest.raises(ValueError, match="shared"):
        _worker_update(entry, adapter, heads, optimizer, "binary")
    assert not calls
    assert all(
        torch.equal(original, parameter.detach())
        for original, parameter in zip(before, parameters)
    )
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert any(parameter.grad is not None for parameter in heads.parameters())


@pytest.mark.parametrize("value", [0.0, 1e-15])
def test_worker_regression_metric_boundary_preserves_rmse_with_finite_log_floor(value):
    result = worker.regression_metrics(
        torch.tensor([value]), torch.zeros(1), ridge_rmse=1.0
    )
    assert result["rmse"] == pytest.approx(value, rel=1e-6, abs=0)
    assert result["log_rmse_ratio"] == pytest.approx(math.log(1e-12))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("field", ["prediction", "target"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_worker_regression_metric_nonfinite_inputs_are_refused(field, invalid):
    values = {"prediction": torch.ones(1), "target": torch.ones(1)}
    values[field] = torch.tensor([invalid])
    with pytest.raises(ValueError, match="nonfinite"):
        worker.regression_metrics(**values, ridge_rmse=1.0)


def test_worker_regression_metric_nonfinite_computation_is_refused():
    with pytest.raises(ValueError, match="nonfinite"):
        worker.regression_metrics(torch.tensor([3e30]), torch.zeros(1), ridge_rmse=1.0)


@pytest.mark.parametrize(
    "ridge", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_worker_regression_metric_bad_ridge_is_refused(ridge):
    with pytest.raises(ValueError, match="positive finite"):
        worker.regression_metrics(torch.ones(1), torch.zeros(1), ridge_rmse=ridge)


def test_source_pinned_runtime_rejects_file_drift_and_symlink(tmp_path):
    source = tmp_path / "runtime-source.py"
    source.write_text("source pin fixture\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert runtime.verify_file_pin(source, expected) == expected
    with pytest.raises(runtime.SourcePinError, match="hash mismatch"):
        runtime.verify_file_pin(source, "0" * 64)
    alias = tmp_path / "runtime-alias.py"
    alias.symlink_to(source)
    with pytest.raises(runtime.SourcePinError, match="not regular"):
        runtime.verify_file_pin(alias, expected)


def test_h144_loader_refuses_hash_mismatch_before_execution(tmp_path, monkeypatch):
    marker = tmp_path / "executed"
    source = tmp_path / "h144-runtime.py"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(runtime, "H144_RUNTIME", source)
    monkeypatch.setattr(runtime, "H144_RUNTIME_SHA256", "0" * 64)

    with pytest.raises(runtime.SourcePinError, match="hash mismatch"):
        runtime._load_pinned_h144_runtime()

    assert not marker.exists()


def test_h144_loader_reads_the_pinned_file_once(tmp_path, monkeypatch):
    source = tmp_path / "h144-runtime.py"
    source.write_text("VALUE = 'verified'\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime, "H144_RUNTIME", source)
    monkeypatch.setattr(runtime, "H144_RUNTIME_SHA256", expected)
    original_open, original_read_bytes = Path.open, Path.read_bytes
    reads = []

    def swap_after_first_read():
        if len(reads) == 1:
            import os

            fd = os.open(source, os.O_WRONLY | os.O_TRUNC)
            os.write(fd, b"VALUE = 'changed after verification'\n")
            os.close(fd)

    def counting_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == source and "r" in mode:
            reads.append("open")
            data = stream.read()
            stream.close()
            swap_after_first_read()
            import io

            return io.BytesIO(data)
        return stream

    def counting_read_bytes(path):
        contents = original_read_bytes(path)
        if path == source:
            reads.append("read_bytes")
            swap_after_first_read()
        return contents

    monkeypatch.setattr(Path, "open", counting_open)
    monkeypatch.setattr(Path, "read_bytes", counting_read_bytes)

    loaded = runtime._load_pinned_h144_runtime()

    assert loaded.VALUE == "verified", reads


def test_worker_adapter_fingerprint_supports_scalar_parameters():
    class ScalarAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scalar = torch.nn.Parameter(torch.tensor(0.25))

    adapter = ScalarAdapter()
    original = worker.adapter_fingerprint(adapter)
    with torch.no_grad():
        adapter.scalar.add_(0.5)
    assert worker.adapter_fingerprint(adapter) != original


def test_source_pinned_runtime_reads_h144_constructor_version_pins(
    tmp_path, monkeypatch
):
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "constructor": {
                    "packages": {
                        "numpy": runtime.EXPECTED_NUMPY,
                        "torch": runtime.EXPECTED_TORCH,
                    },
                    "python": f"{runtime.EXPECTED_PYTHON} (pinned build)",
                }
            }
        )
    )
    monkeypatch.setattr(runtime, "H144_PREFLIGHT", preflight)
    monkeypatch.setattr(runtime, "verify_file_pin", lambda *_: "pinned")
    versions = runtime._verify_runtime_versions()
    assert versions == {
        "python": runtime.EXPECTED_PYTHON,
        "numpy": runtime.EXPECTED_NUMPY,
        "torch": runtime.EXPECTED_TORCH,
    }


def test_source_pinned_runtime_optimizer_is_exactly_adapter_and_heads():
    adapter = torch.nn.Parameter(torch.zeros(3))
    head = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([adapter, head], lr=0.1)
    runtime._verify_optimizer_parameters(optimizer, [adapter, head])
    with pytest.raises(ValueError, match="exactly"):
        runtime._verify_optimizer_parameters(
            torch.optim.SGD([adapter], lr=0.1), [adapter, head]
        )
    extra = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="exactly"):
        runtime._verify_optimizer_parameters(
            torch.optim.SGD([adapter, head, extra], lr=0.1), [adapter, head]
        )


def test_source_pinned_runtime_requires_candidate_builder_packer():
    source_pins = {
        "checkpoint_sha256": runtime.R882_CHECKPOINT_SHA256,
        "dope_sha256": runtime.DOPE_SHA256,
    }
    packer = candidate_builder.make_worker_packer(
        {},
        {},
        {},
        checkpoint_sha256=source_pins["checkpoint_sha256"],
        dope_sha256=source_pins["dope_sha256"],
    )
    runtime._verify_candidate_packer(packer, source_pins)
    with pytest.raises(ValueError, match="not bound"):
        runtime._verify_candidate_packer(lambda _row: None, source_pins)


def test_frozen_worker_core_checks_state_when_synthetic_core_raises():
    class RuntimeGuard:
        def __init__(self):
            self.checks = 0

        def verify_frozen_state(self):
            self.checks += 1

    class FailingCore:
        def train_batch(self, _rows):
            raise RuntimeError("synthetic worker failure")

    guard = RuntimeGuard()
    core = runtime.FrozenWorkerCore(guard, FailingCore())
    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        core.train_batch([])
    assert guard.checks == 2
