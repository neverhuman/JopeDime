"""Explicit synthetic contracts. Tiny CPU optimizer proof is not real-corpus qualification."""

from __future__ import annotations

import copy
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from jope.policy import training_campaign as campaign


@pytest.fixture
def state(monkeypatch):
    # All state, including CPU qualification, obeys the host custody rule.
    root = Path(__file__).resolve().parents[2] / "target" / "training-contract-fixtures"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(campaign, "_runtime_identity_for_test", campaign.runtime_identity, raising=False)
    monkeypatch.setattr(campaign, "runtime_identity", lambda: {"fixture": "explicit-test-runtime"})
    with tempfile.TemporaryDirectory(prefix="campaign-", dir=root) as directory:
        yield Path(directory)


def plan(state, **options):
    return campaign.make_plan(state, synthetic=True, device="cpu", source_sha="a" * 40, **options)


def test_plan_has_no_clock_and_pins_each_category(state):
    value = plan(state)
    assert value["tasks"] == list(campaign.TASKS)
    assert value["qualification_only"] is True
    assert value["lineage"] == campaign.LINEAGE
    assert not (state / "window.json").exists()
    experiment = campaign.experiment_for_plan(value)
    assert experiment["resources"] == {
        "workers": 1,
        "cpus": 20,
        "memory_bytes": 64 * 1024**3,
        "swap_bytes": 0,
        "gpu_tasks": 0,
        "threads_per_worker": 1,
        "total_seconds": 3600,
        "reporting_seconds": 5,
        "per_fit_seconds": 1200,
    }
    assert set(experiment["checkpoints"]) == {
        f"{task}:{name}"
        for task in campaign.TASKS
        for name in ("corpus", "contexts", "encoder", "vocabulary")
    }
    assert campaign.verify_plan(state)[0] == value


def test_cpu_requires_explicit_fixture_and_tmp_state_is_always_rejected(state):
    with pytest.raises(ValueError, match="CPU is explicit synthetic"):
        campaign.make_plan(state, manifest=state / "absent.json", device="cpu", source_sha="a" * 40)
    with pytest.raises(ValueError, match="outside /tmp"):
        campaign.make_plan(
            Path("/tmp/jope-training-refusal-only"),
            synthetic=True,
            device="cpu",
            source_sha="a" * 40,
        )


def test_runtime_and_corpus_drift_refuse_resume(state, monkeypatch):
    value = plan(state, task="binary")
    monkeypatch.setattr(campaign, "runtime_identity", lambda: {"fixture": "changed-runtime"})
    with pytest.raises(ValueError, match="dependencies changed"):
        campaign.verify_plan(state)
    monkeypatch.setattr(campaign, "runtime_identity", lambda: value["runtime"])
    corpus = Path(value["category_inputs"]["binary"]["inputs"]["corpus"]["path"])
    corpus.write_bytes(corpus.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact changed"):
        campaign.verify_plan(state)


def test_original_windows_never_reset_on_resume_or_verification(state, monkeypatch):
    value = plan(state, task="regression")
    monkeypatch.setattr(campaign.time, "time", lambda: 1000.0)
    window = campaign.original_window(state, value, None)
    first = campaign.category_window(state, "regression", window, now=1000000)
    resumed = campaign.category_window(state, "regression", window, now=2200001)
    assert first == resumed
    assert resumed["deadline_unix_ms"] == 2200000
    assert campaign.original_window(state, value, None, admitting=False) == window
    with pytest.raises(ValueError, match="no original admission"):
        campaign.category_window(state, "binary", window, now=1100000, admitting=False)
    assert not (state / "binary" / "window.json").exists()
    corrupt = {**first, "deadline_unix_ms": first["deadline_unix_ms"] + 1}
    campaign.write(state / "regression" / "window.json", corrupt)
    with pytest.raises(ValueError, match="deadline reset"):
        campaign.category_window(state, "regression", window, now=2200001)
    campaign.write(
        state / "window.json",
        {**window, "deadline_unix_ms": window["deadline_unix_ms"] + 1},
    )
    with pytest.raises(ValueError, match="original training window changed"):
        campaign.original_window(state, value, None)


def test_global_reserve_caps_last_category(state):
    window = {
        "experiment_sha256": "e" * 64,
        "started_unix_ms": 1000,
        "deadline_unix_ms": 3601000,
    }
    category = campaign.category_window(state, "multiclass", window, now=3501000)
    assert category["deadline_unix_ms"] == 3596000
    with pytest.raises(ValueError, match="exhausted"):
        campaign.category_window(state, "binary", window, now=3596000)


def test_cuda_requires_exact_experiment_uuid_unit_and_membership(state, monkeypatch):
    value = campaign.make_plan(state, synthetic=True, device="cuda:0", source_sha="a" * 40, task="binary")
    identity = campaign.experiment_identity(campaign.experiment_for_plan(value))
    unit = "jope-asm-" + identity + "-12345678-1234-1234-1234-123456789abc.service"
    window = {
        "experiment_sha256": identity,
        "started_unix_ms": 1000000,
        "deadline_unix_ms": 4600000,
    }
    campaign.write(state / "window.json", window, immutable=True)
    monkeypatch.setattr(campaign.time, "time", lambda: 1000.0)
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "0::/system.slice/" + unit + "/child\n"
            if str(path) == "/proc/self/cgroup"
            else original(path, *args, **kwargs)
        ),
    )
    for invalid in (
        None,
        "jope-asm-" + identity + ".service",
        unit.replace(identity, "f" * 64),
        unit + "extra",
    ):
        with pytest.raises(ValueError, match="supervised"):
            campaign.original_window(state, value, invalid)
    assert campaign.original_window(state, value, unit) == window
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *args, **kwargs: (
            "0::/system.slice/" + unit + "-unrelated\n"
            if str(path) == "/proc/self/cgroup"
            else original(path, *args, **kwargs)
        ),
    )
    with pytest.raises(ValueError, match="outside supervisor"):
        campaign.original_window(state, value, unit)


def test_real_classification_refusal_is_labelled_synthetic_not_renamed_regression(
    state,
):
    inputs = campaign.synthetic_inputs(state / "source", ["regression"])
    admission = state / "empty-admission.json"
    campaign.write(
        admission,
        {"schema": "jope.qualified-teaching-freeze/v1", "entries": []},
        immutable=True,
    )
    manifest = state / "requested-inputs.json"
    campaign.write(
        manifest,
        {
            "schema": "jope.experimental-training-inputs/v1",
            **inputs,
            "admission": campaign.artifact(admission),
        },
        immutable=True,
    )
    value = campaign.make_plan(state / "campaign", manifest=manifest, task="binary", source_sha="a" * 40)
    assert value["device"] == "cuda:0" and value["qualification_only"] is False
    assert value["fallbacks"]["binary"]
    assert value["category_inputs"]["binary"]["data_kind"] == "synthetic_fixture"
    rows = campaign.verify_plan(state / "campaign")[1]
    assert {row["metadata"]["task_type"] for row in rows} == {"binary"}
    assert {row["source"] for row in rows} == {"asm-synthetic-training-fixture"}
    assert all("logit_ridge" in row["genome"] for row in rows)


def test_unused_context_is_rejected_before_training(state):
    inputs = campaign.synthetic_inputs(state, ["regression"])
    index = campaign.read(campaign.checked(inputs["contexts"]))
    index["contexts"]["sealed-unused"] = {"partition": "meta-test"}
    path = state / "extra-context.json"
    campaign.write(path, index)
    inputs["contexts"] = campaign.artifact(path)
    with pytest.raises(ValueError, match="context inventory"):
        campaign.admitted_rows(inputs, "synthetic_fixture", "regression")


def test_unselected_task_cannot_hide_forbidden_authority(state):
    inputs = campaign.synthetic_inputs(state, ["regression", "binary"])
    rows = [campaign.json.loads(line) for line in campaign.checked(inputs["corpus"]).read_text().splitlines()]
    next(row for row in rows if row["metadata"]["task_type"] == "binary")["metadata"]["authority_split"] = (
        "meta-test"
    )
    path = state / "forbidden-other-task.jsonl"
    path.write_bytes(b"".join(campaign.encoded(row) + b"\n" for row in rows))
    inputs["corpus"] = campaign.artifact(path)
    with pytest.raises(ValueError, match="report-only authority"):
        campaign.admitted_rows(inputs, "synthetic_fixture", "regression")


def test_context_missing_dimensions_and_cross_split_family_are_refused(state):
    inputs = campaign.synthetic_inputs(state, ["multiclass"])
    rows = campaign.admitted_rows(inputs, "synthetic_fixture")
    index = campaign.read(campaign.checked(inputs["contexts"]))
    row = copy.deepcopy(rows[0])
    row["embedding"].pop()
    with pytest.raises(ValueError, match="dimensions must be exact"):
        campaign.validate_context(
            row,
            index["contexts"][row["metadata"]["context_id"]],
            inputs["encoder"]["sha256"],
            "synthetic_fixture",
        )
    rows[-1]["metadata"]["source_family"] = rows[0]["metadata"]["source_family"]
    corpus = state / "cross-split.jsonl"
    corpus.write_bytes(b"".join(campaign.encoded(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="family crosses"):
        campaign.admitted_rows({**inputs, "corpus": campaign.artifact(corpus)}, "synthetic_fixture")


@pytest.mark.parametrize("failure", ["unavailable", "timeout", "cancelled"])
def test_worker_failures_cancel_group_and_preserve_accounting(state, monkeypatch, failure):
    child = SimpleNamespace(pid=123, returncode=None)

    def wait(*, timeout):
        if failure == "cancelled":
            raise KeyboardInterrupt()
        raise subprocess.TimeoutExpired("synthetic-helper", timeout)

    child.wait = wait

    def start(*args, **kwargs):
        if failure == "unavailable":
            raise OSError("synthetic missing interpreter")
        return child

    stopped = []
    monkeypatch.setattr(campaign.subprocess, "Popen", start)
    monkeypatch.setattr(campaign, "stop_group", lambda item: stopped.append(item.pid))
    monkeypatch.setattr(campaign.time, "time", lambda: 1.0)
    if failure == "cancelled":
        with pytest.raises(KeyboardInterrupt):
            campaign.launch_worker(["synthetic-helper"], {}, state, 2000)
    else:
        assert (
            campaign.launch_worker(["synthetic-helper"], {}, state, 2000)
            == {"timeout": "budget_exhausted", "unavailable": "unavailable"}[failure]
        )
    outcome = campaign.read(state / "outcome.json")
    assert outcome["cpu_seconds"] is None and outcome["gpu_seconds"] is None
    assert stopped == ([] if failure == "unavailable" else [123])


@pytest.mark.parametrize("task", campaign.TASKS)
def test_real_cpu_optimizer_reload_continuation_and_export(state, monkeypatch, task):
    import torch

    from jope.policy import train_v3

    before_threads = torch.get_num_threads()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    value = plan(state, task=task)

    # Same controller and real optimizer path; no subprocess is needed in this CPU fixture.
    def in_process(arguments, env, attempt, deadline):
        started_ms = int(campaign.time.time() * 1000)
        started = campaign.time.monotonic()
        campaign.worker(state, task, attempt)
        campaign.write(
            attempt / "outcome.json",
            {
                "status": "completed",
                "exit_code": 0,
                "started_unix_ms": started_ms,
                "ended_unix_ms": int(campaign.time.time() * 1000),
                "deadline_unix_ms": deadline,
                "wall_seconds": campaign.time.monotonic() - started,
            },
            immutable=True,
        )
        return "completed"

    monkeypatch.setattr(campaign, "launch_worker", in_process)
    try:
        result = campaign.run(state)
        assert result["success"] is True
        first_window = campaign.read(state / "window.json")
        assert campaign.run(state, resume=True)["success"] is True
        assert campaign.read(state / "window.json") == first_window
        proof = campaign.verify(state)
        head = campaign.read_head(state / task / "head.json")
        assert head["committed_passes"] == 2
        assert head["training_proof"]["optimizer_state_resumed"] is True
        assert proof["tasks"][0]["optimizer_updates"] > 0
        assert (
            head["training_proof"]["parameter_sha256"] != head["training_proof"]["initial_parameter_sha256"]
        )
        exported = campaign.export(state, state / "export")
        assert exported["verification"] == proof
        assert campaign.read(state / "export" / "manifest.json")["serving_eligible"] is False
        # Every execution receipt in the committed chain is byte-bound.
        outcome = Path(head["execution"]["outcome"]["path"])
        campaign.write(outcome, {"status": "failed", "exit_code": 1})
        with pytest.raises(ValueError, match="artifact changed"):
            campaign.verify(state)
        assert value["data_kind"] == "synthetic_fixture"
    finally:
        torch.set_num_threads(before_threads)


def test_campaign_recovers_committed_update_in_original_window(state, monkeypatch):
    import torch

    from jope.policy import train_v3

    threads = torch.get_num_threads()
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(
        train_v3,
        "policy_v3_validation_metrics",
        lambda *args, **kwargs: {"pairwise_win_accuracy": 0.5, "seq_nll": 1.0},
    )
    plan(state, task="regression")
    original_save = train_v3._atomic_torch_save
    interrupted = False

    def interrupt_once(payload, path):
        nonlocal interrupted
        original_save(payload, path)
        if payload.get("strict_progress") and not interrupted:
            interrupted = True
            raise InterruptedError("fixture committed update interruption")

    def in_process(arguments, env, attempt, deadline):
        started_ms = int(campaign.time.time() * 1000)
        status = "completed"
        try:
            campaign.worker(state, "regression", attempt)
        except InterruptedError:
            status = "cancelled"
        campaign.write(
            attempt / "outcome.json",
            {
                "status": status,
                "exit_code": 0 if status == "completed" else 1,
                "started_unix_ms": started_ms,
                "ended_unix_ms": int(campaign.time.time() * 1000),
                "deadline_unix_ms": deadline,
                "wall_seconds": 0.0,
            },
            immutable=True,
        )
        return status

    monkeypatch.setattr(train_v3, "_atomic_torch_save", interrupt_once)
    monkeypatch.setattr(campaign, "launch_worker", in_process)
    try:
        assert campaign.run(state)["success"] is False
        window = campaign.read(state / "window.json")
        category = campaign.read(state / "regression" / "window.json")
        assert campaign.run(state, resume=True)["success"] is True
        assert campaign.read(state / "window.json") == window
        assert campaign.read(state / "regression" / "window.json") == category
        recovered = campaign.read(state / "regression" / "attempt-000002" / "complete.json")["recovered_from"]
        assert recovered and Path(recovered["checkpoint"]["path"]).parent.name == "attempt-000001"
        assert campaign.verify(state)["tasks"][0]["optimizer_updates"] > 0
        Path(recovered["checkpoint"]["path"]).write_bytes(b"changed")
        with pytest.raises(ValueError, match="artifact changed"):
            campaign.verify(state)
    finally:
        torch.set_num_threads(threads)


def test_runtime_pins_installed_bytes_not_only_version(state, monkeypatch):
    package = state / "package" / "policy"
    package.mkdir(parents=True)
    source = package / "training_campaign.py"
    source.write_text("# synthetic source\n")
    monkeypatch.setattr(campaign, "__file__", str(source))
    distributions = []
    for name in ("torch", "numpy", "safetensors"):
        root = state / "installed" / name
        root.mkdir(parents=True)
        (root / "extension.so").write_bytes(b"explicit synthetic extension bytes")
        distributions.append(
            SimpleNamespace(
                metadata={"Name": name},
                version="synthetic1",
                files=[Path("extension.so")],
                locate_file=lambda item, root=root: root / item,
            )
        )
    monkeypatch.setattr(campaign.importlib.metadata, "distributions", lambda: distributions)
    # The state fixture replaces the expensive function; restore its actual implementation.
    original = campaign._runtime_identity_for_test
    first = original()
    assert first["python_invocation"] == str(Path(campaign.sys.executable).absolute())
    assert first["python"]["path"] == str(Path(campaign.sys.executable).resolve())
    assert first["python_prefix"] == str(Path(campaign.sys.prefix).resolve())
    assert first["python_base_prefix"] == str(Path(campaign.sys.base_prefix).resolve())
    configuration = Path(campaign.sys.prefix) / "pyvenv.cfg"
    assert first["venv_configuration"] == (
        campaign.artifact(configuration) if configuration.is_file() else None
    )
    (state / "installed" / "torch" / "extension.so").write_bytes(b"changed same-version extension")
    second = original()
    assert first["packages"]["torch"]["version"] == second["packages"]["torch"]["version"]
    assert first["packages"]["torch"]["files_sha256"] != second["packages"]["torch"]["files_sha256"]


def test_cuda_unavailable_never_falls_back_to_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CPU fallback"):
        campaign.checked_device("cuda:0")


def test_synthetic_classifier_target_width_is512_without_changing_regression(state):
    inputs = campaign.synthetic_inputs(state, list(campaign.TASKS))
    rows = campaign.admitted_rows(inputs, "synthetic_fixture")
    for row in rows:
        target = row["target_embedding"]
        assert len(target) == 512
        if row["metadata"]["task_type"] == "regression":
            assert target == [1.0, 0.5] + [0.0] * 510
        else:
            assert all(value != 0 for value in target)
            assert target == [0.125 if column % 2 == 0 else 0.0625 for column in range(512)]
            assert row["source"] == "asm-synthetic-training-fixture"


def test_runtime_pins_cuda_bindings_and_preprocessing_dependencies(state, monkeypatch):
    distributions = []
    for name in (
        "torch",
        "numpy",
        "safetensors",
        "cuda-bindings",
        "cuda-pathfinder",
        "scipy",
    ):
        folder = state / name
        folder.mkdir()
        (folder / "dependency.bin").write_bytes(name.encode())
        distributions.append(
            SimpleNamespace(
                metadata={"Name": name},
                version="1.0",
                files=["dependency.bin"],
                locate_file=lambda item, folder=folder: folder / item,
            )
        )
    monkeypatch.setattr(campaign.importlib.metadata, "distributions", lambda: distributions)
    first = campaign._runtime_identity_for_test()
    assert set(first["packages"]) == {d.metadata["Name"] for d in distributions}
    for name in ("cuda-bindings", "cuda-pathfinder", "scipy"):
        (state / name / "dependency.bin").write_bytes(b"same version, changed installed bytes")
        changed = campaign._runtime_identity_for_test()
        assert changed["packages"][name]["files_sha256"] != first["packages"][name]["files_sha256"]


def test_requested_coverage_exposes_missing_regression_and_classifier_fallbacks():
    value = {
        "scope": "all-admitted",
        "tasks": ["binary", "multiclass"],
        "category_inputs": {task: {"data_kind": "synthetic_fixture"} for task in ("binary", "multiclass")},
        "unavailable_tasks": ["regression"],
        "unavailable_reasons": {"regression": "no eligible real rows"},
        "fallbacks": {
            "binary": "no eligible real classes",
            "multiclass": "no eligible real classes",
        },
    }
    result = campaign.requested_coverage(value, [])
    assert result["execution_success"] is True
    assert result["success"] is False and result["requested_coverage_complete"] is False
    assert result["requested_tasks"] == list(campaign.TASKS)
    assert result["categories"][-1]["status"] == "unavailable"
    assert [row["data_kind"] for row in result["categories"][:2]] == ["synthetic_fixture"] * 2
    assert result["unavailable_reasons"]["regression"] == "no eligible real rows"


def test_execution_timing_rejects_late_exit_unknown_wall_and_late_validation():
    start = {"started_unix_ms": 1100}
    complete = {"completed_unix_ms": 1800}
    category = {"started_unix_ms": 1000, "deadline_unix_ms": 2000}
    outcome = {
        "status": "completed",
        "exit_code": 0,
        "started_unix_ms": 1200,
        "ended_unix_ms": 1900,
        "deadline_unix_ms": 1950,
        "wall_seconds": 0.7,
    }
    campaign.validate_execution_timing(start, outcome, complete, category, 1951)
    for changed in (
        {**outcome, "ended_unix_ms": 2001},
        {**outcome, "wall_seconds": None},
        {**outcome, "wall_seconds": float("nan")},
        {**outcome, "started_unix_ms": 1801},
        {**outcome, "deadline_unix_ms": 2001},
    ):
        with pytest.raises(ValueError):
            campaign.validate_execution_timing(start, changed, complete, category, 1951)
    with pytest.raises(ValueError, match="controller validation"):
        campaign.validate_execution_timing(start, outcome, complete, category, 2001)


def test_parent_observed_late_zero_exit_is_budget_exhausted(state, monkeypatch):
    clock = iter((1.0, 1.1, 2.001))
    monkeypatch.setattr(campaign.time, "time", lambda: next(clock))
    child = SimpleNamespace(pid=123, returncode=0, wait=lambda **kwargs: 0)
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(campaign, "stop_group", lambda value: None)
    assert campaign.launch_worker(["synthetic-helper"], {}, state, 2000) == "budget_exhausted"
    assert campaign.read(state / "outcome.json")["ended_unix_ms"] == 2001


def test_checkpoint_hashing_cannot_extend_category_deadline(state, monkeypatch):
    path = state / "artifact.bin"
    path.write_bytes(b"synthetic checkpoint bytes")
    reference = campaign.artifact(path)
    monkeypatch.setattr(campaign.time, "time", lambda: 2.001)
    with pytest.raises(ValueError, match="category validation deadline"):
        campaign.checked(reference, 2000)


def test_real_f32_context_identity_refuses_lossy_numbers_and_changed_bytes(state):
    import hashlib
    import struct

    values = [0.25] * 2048
    training = {"selected_training_rows_sha256": "f" * 64}
    context = {
        "lineage": "asm.experimental-regression-context/v1",
        "policy_compatible": False,
        "training": training,
        "values": values,
        "context_sha256": hashlib.sha256(struct.pack("<2048f", *values)).hexdigest(),
    }
    context_path = state / "context.json"
    campaign.write(context_path, context)
    reference = campaign.artifact(context_path)
    producer = {
        "output_sha256": reference["sha256"],
        "fitting_rows_sha256": "f" * 64,
        "encoder_sha256": "e" * 64,
        "inputs": [reference],
        "training": training,
    }
    producer_path = state / "producer.json"
    campaign.write(producer_path, producer)
    row = {
        "dataset_id": "d" * 64,
        "embedding": values[:1536],
        "target_embedding": values[1536:],
        "metadata": {"task_type": "regression"},
    }
    record = {
        "task": "regression",
        "dataset_id": "d" * 64,
        "encoder_sha256": "e" * 64,
        "partition": "fit",
        "fitting_rows_sha256": "f" * 64,
        "lineage": context["lineage"],
        "context_encoding": "f32le/v1",
        "embedding_f32le_sha256": hashlib.sha256(struct.pack("<1536f", *row["embedding"])).hexdigest(),
        "target_embedding_f32le_sha256": hashlib.sha256(
            struct.pack("<512f", *row["target_embedding"])
        ).hexdigest(),
        "context_artifact": reference,
        "producer_receipt": campaign.artifact(producer_path),
    }
    campaign.validate_context(row, record, "e" * 64, "admitted")
    changed = copy.deepcopy(row)
    changed["embedding"][0] = 1.1
    with pytest.raises(ValueError, match="exact finite f32"):
        campaign.validate_context(changed, record, "e" * 64, "admitted")
    changed = copy.deepcopy(row)
    changed["embedding"][0] = 0.5
    with pytest.raises(ValueError, match="f32 bytes changed"):
        campaign.validate_context(changed, record, "e" * 64, "admitted")
