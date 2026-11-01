"""Synthetic CPU protocol fixtures; no learner, GPU, dataset or score history."""

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from research.joint_regbin import driver
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
    verify_catalog,
    verify_plan,
)
from research.joint_regbin.report import comparison, decide, elo, paired_records, render


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
    result = driver.preflight(frozen, packet, worker, checkpoint)
    assert result == dict(
        schema="jope.joint-regbin.preflight/v1",
        status="INPUTS_VERIFIED",
        plan_sha256=frozen["sha256"],
        contexts=36,
        jobs=72,
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
        driver.preflight(frozen, packet, worker, checkpoint)


def test_preflight_refuses_missing_input(preflight_inputs):
    frozen, packet, worker, checkpoint = preflight_inputs
    (packet / frozen["contexts"][0]["payload"]).unlink()
    with pytest.raises(FileNotFoundError):
        driver.preflight(frozen, packet, worker, checkpoint)


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
        driver.preflight(frozen, packet, worker, checkpoint)


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
        driver.preflight(changed, packet, worker, checkpoint)


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
    driver.main()
    assert json.loads(capsys.readouterr().out)["status"] == "INPUTS_VERIFIED"
    assert {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")} == before


def test_run_rechecks_input_drift_before_gpu_acquisition(
    preflight_inputs, tmp_path, monkeypatch
):
    frozen, packet, worker, checkpoint = preflight_inputs
    assert (
        driver.preflight(frozen, packet, worker, checkpoint)["status"]
        == "INPUTS_VERIFIED"
    )
    (packet / frozen["contexts"][-1]["payload"]).write_bytes(b"later drift")

    def forbidden(*_args, **_kwargs):
        pytest.fail("drifted inputs reached compute or history mutation")

    for name in (
        "qualification_lock",
        "reserve_campaign",
        "worker_call",
        "append_score_store",
    ):
        monkeypatch.setattr(driver, name, forbidden)
    with pytest.raises(ValueError, match="payload drift"):
        driver.execute(
            frozen, packet, worker, checkpoint, tmp_path / "campaign", None, None
        )
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
        diagnostic = "learner.py"
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


def test_plan_is_reproducible_and_all_72_jobs_have_equal_budgets(plan, manifest):
    assert plan == make_plan(manifest)
    assert len(plan["jobs"]) == 72
    assert len({job["id"] for job in plan["jobs"]}) == 72
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


def test_balanced_and_natural_sampling_are_prespecified(plan):
    config = plan["config"]
    for arm in ARMS:
        job = next(
            job for job in plan["jobs"] if job["arm"] == arm and job["fold"] == "full"
        )
        by_id = {row["id"]: row["task"] for row in job["train"]}
        counts = Counter(
            by_id[value] for order in batches(job, config) for value in order
        )
        if arm.startswith("joint"):
            assert counts == dict(regression=1600, binary=1600)
        else:
            assert counts == {"regression" if arm == "reg-only" else "binary": 3200}


def test_natural_ratio_differs_from_balanced_without_changing_budget(manifest):
    manifest["contexts"] = [
        row
        for row in manifest["contexts"]
        if not (
            row["role"] == "train"
            and row["task"] == "binary"
            and int(row["id"].rsplit("-", 1)[1]) >= 5
        )
    ]
    plan = make_plan(manifest)
    counts = {}
    for arm in ("joint-natural", "joint-balanced"):
        job = next(j for j in plan["jobs"] if j["arm"] == arm and j["fold"] == "full")
        types = {row["id"]: row["task"] for row in job["train"]}
        counts[arm] = Counter(types[c] for b in batches(job, plan["config"]) for c in b)
    assert counts["joint-natural"]["regression"] == 3200 * 8 // 13
    assert counts["joint-balanced"]["regression"] == 1600


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
    assert len(jobs) == 12  # four arms × three seeds, exactly once each
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
        "arm": "joint-balanced",
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
    assert first["joint-balanced"]["delta"] == pytest.approx(0)


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
        verdict="JOINT_OK",
        interpretation="SYNTHETIC FIXTURE ONLY",
        comparisons=[dict(tasks=tasks)],
        curves=[
            dict(task=task, arm=arm, round=r, value=0.1)
            for task in tasks
            for arm in ARMS
            for r in (5, 50)
        ],
    )
    render(report, tmp_path)
    assert (tmp_path / "validation-curves.svg").stat().st_size > 1000
    assert (tmp_path / "curves.jsonl").read_text().count("\n") == 16


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
