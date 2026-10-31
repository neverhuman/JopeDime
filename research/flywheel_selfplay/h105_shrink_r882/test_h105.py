"""Synthetic CPU controls only; no H105 data, timing or scientific outcomes."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


r = load("h105_unit_runner", HERE / "runner.py")
w = load("h105_unit_wave", HERE / "run_wave.py")


def adapter(value, *, grad=False):
    return {
        "adapter_10tok.fixture": torch.full(
            (r.ADAPTER_PARAMS,), value, dtype=torch.float64, requires_grad=grad
        )
    }


def test_interpolation_endpoints_exact_no_mutation_and_known_distance():
    initial = adapter(0.1)
    fine = adapter(0.3)
    assert torch.equal(
        r.interpolate(initial, fine, 0)["adapter_10tok.fixture"],
        initial["adapter_10tok.fixture"],
    )
    assert torch.equal(
        r.interpolate(initial, fine, 1)["adapter_10tok.fixture"],
        fine["adapter_10tok.fixture"],
    )
    for alpha in (0.25, 0.5):
        candidate = r.interpolate(initial, fine, alpha)
        assert torch.allclose(
            candidate["adapter_10tok.fixture"],
            torch.full((r.ADAPTER_PARAMS,), 0.1 + alpha * 0.2, dtype=torch.float64),
        )
        assert (
            candidate["adapter_10tok.fixture"].data_ptr()
            != fine["adapter_10tok.fixture"].data_ptr()
        )
    assert torch.equal(
        initial["adapter_10tok.fixture"], adapter(0.1)["adapter_10tok.fixture"]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"dope.weight": torch.zeros(29211)},
        {"adapter_10tok.anchor.weight": torch.zeros(29211)},
        {"adapter_10tok.x": torch.zeros(29210)},
        {"adapter_10tok.x": torch.full((29211,), float("nan"))},
    ],
)
def test_parameter_boundary_refuses_dope_anchor_missing_and_nonfinite(bad):
    with pytest.raises(ValueError):
        r.adapter_inventory(bad)


def test_anchor_uses_raw_sum_reference_detached_and_fixed_drift_only_strength():
    initial = adapter(0, grad=True)
    current = adapter(r.PRIOR_DRIFT / np.sqrt(r.ADAPTER_PARAMS), grad=True)
    p = r.anchor_penalty(current, initial)
    p.backward()
    assert initial["adapter_10tok.fixture"].grad is None
    assert current["adapter_10tok.fixture"].grad.norm().item() == pytest.approx(
        1.0, abs=1e-12
    )
    assert r.ANCHOR_LAMBDA == 1 / r.PRIOR_DRIFT
    zero = torch.tensor(2.0, requires_grad=True)
    for arm in ("CONTROL", "INTERP025", "INTERP050"):
        assert r.candidate_loss(zero, current, initial, arm) is zero
    assert r.candidate_loss(zero, current, initial, "ANCHOR") > zero


def rows():
    return [
        {
            "context_sha256": f"{i:064x}",
            "family_id": f"f{i:03d}",
            "task": "regression",
            "dl_split": "DL-Train",
            "source_split": "train",
            "origin": "real",
        }
        for i in range(205)
    ]


def test_fresh_full_family_partition_is_fixed_train_only_and_order_invariant():
    data = rows()
    split = r.family_assignment(data)
    assert split == r.family_assignment(list(reversed(data)))
    assert split["seed"] == 105042 and split["seed"] not in (4900, 9900)
    assert (
        len(split["assessment_families"]) == 41 and len(split["train_families"]) == 164
    )
    assert not set(split["assessment_families"]) & set(split["train_families"])
    assert len({f for fold in split["folds"] for f in fold}) == 205


@pytest.mark.parametrize(
    "change",
    [
        {"dl_split": "DL-Val"},
        {"dl_split": "DL-Holdout"},
        {"source_split": "outer"},
        {"origin": "synthetic"},
        {"has_meta_test": True},
        {"family_id": ""},
    ],
)
def test_disallowed_roster_rows_refused_before_payloads(change):
    data = rows()
    data[0].update(change)
    with pytest.raises(ValueError):
        r.family_assignment(data)


def test_attempt_scoped_staging_never_reuses_existing_or_symlink(tmp_path):
    one = r.attempt_scratch(tmp_path, "CONTROL", 42, 1)
    two = r.attempt_scratch(tmp_path, "CONTROL", 42, 2)
    assert one != two and one.name == "CONTROL-seed42-attempt1"
    one.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        r.attempt_scratch(tmp_path, "CONTROL", 42, 1)
    two.symlink_to(tmp_path / "missing")
    with pytest.raises(RuntimeError):
        r.attempt_scratch(tmp_path, "CONTROL", 42, 2)


def fixture_cells(gain=0.005):
    metadata = [dict(rows()[i % 205], context_sha256=f"{i:064x}") for i in range(1219)]
    split = r.family_assignment(metadata)
    families = split["assessment_families"]
    training = dict.fromkeys(split["train_families"], 1.0)
    assessment_contexts = {
        x["context_sha256"]: 1.0 + gain for x in metadata if x["family_id"] in families
    }
    train_contexts = {
        x["context_sha256"]: 1.0 for x in metadata if x["family_id"] in training
    }
    assessment = {
        "per_family": dict.fromkeys(families, 1.0 + gain),
        "per_context": assessment_contexts,
        "mean": 1.0 + gain,
    }
    reference = {
        "per_family": dict.fromkeys(families, 1.0),
        "per_context": dict.fromkeys(assessment_contexts, 1.0),
        "mean": 1.0,
    }
    train = {"per_family": training, "per_context": train_contexts, "mean": 1.0}
    family_by_context = {x["context_sha256"]: x["family_id"] for x in metadata}
    for packet in (train, assessment, reference):
        packet["family_by_context"] = {
            cid: family_by_context[cid] for cid in packet["per_context"]
        }
    return [
        {
            "wave": "H105",
            "arm": arm,
            "seed": seed,
            "timing_only": False,
            "updates": 40,
            "trainable_count": 29211,
            "dope_before": r.DOPE_SHA,
            "dope_after": r.DOPE_SHA,
            "checkpoint_sha256": r.R882_SHA,
            "manifest_sha256": r.MANIFEST_SHA,
            "selected_sha256": r.SELECTED_SHA,
            "protocol_sha256": r.protocol_hash(),
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "DL_Val_payloads_opened": 0,
            "initial_adapter_sha256": "base",
            "final_adapter_sha256": f"{arm}-{seed}",
            "fine_tuned_adapter_sha256": f"ft-{seed}",
            "source_commit": "source",
            "prereg_sha256": "prereg",
            "source_hashes": dict(r.SOURCE_PINS),
            "split": copy.deepcopy(split),
            "adapter_l2": 0.1,
            "train": copy.deepcopy(train),
            "assessment": copy.deepcopy(assessment),
            "r882_assessment": copy.deepcopy(reference),
        }
        for seed in r.SEEDS
        for arm in r.ARMS
    ]


def test_primary_is_r882_not_control_and_all_four_holm_comparisons_count():
    metrics = r.summarize(fixture_cells(), draws=10000)
    assert metrics["reference"] == "r882" and len(metrics["holm"]) == 4
    assert metrics["holm"]["CONTROL"]["tail_alpha"] == 0.0125
    assert all(
        a["admitted"] and a["median_gain"] == pytest.approx(0.005)
        for a in metrics["arms"].values()
    )
    assert metrics["arms"]["CONTROL"]["identical_seed_utility_flag"]
    assert metrics["arms"]["CONTROL"]["elo_vs_r882"]["per_seed"][0]["wins"] == 41


@pytest.mark.parametrize("gain", [0.0, 0.001, -0.001])
def test_zero_exact_margin_and_negative_gains_never_admit(gain):
    result = r.summarize(fixture_cells(gain), draws=10000)
    assert not any(a["admitted"] for a in result["arms"].values())
    if gain == 0:
        assert result["arms"]["CONTROL"]["elo_vs_r882"]["per_seed"][0]["wins"] == 0
        assert result["arms"]["CONTROL"]["elo_vs_r882"]["median_elo"] == 1500


def test_holm_stepdown_stops_at_first_failure_and_uses_common_full_arm_set():
    boots = {a: np.full(1000, 0.01) for a in r.ARMS}
    boots["CONTROL"] = np.zeros(1000)
    gates = r.holm_bounds(boots)
    assert gates["INTERP025"]["rank"] == 1 and gates["INTERP025"]["passes"]
    assert not gates["CONTROL"]["passes"]
    with pytest.raises(ValueError):
        r.holm_bounds({"CONTROL": np.ones(1000)})


def test_harm_gate_uses_r882_and_rejects_more_than_ten_percent():
    cells = fixture_cells(0.2)
    for c in cells:
        for f in list(c["assessment"]["per_family"])[:5]:
            c["assessment"]["per_family"][f] = 0.98
        packet = c["assessment"]
        for cid, family in packet["family_by_context"].items():
            packet["per_context"][cid] = packet["per_family"][family]
        packet["mean"] = float(np.mean(list(packet["per_family"].values())))
    result = r.summarize(cells, draws=10000)
    assert all(
        a["harm_fraction"] == pytest.approx(5 / 41) and not a["admitted"]
        for a in result["arms"].values()
    )


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "duplicate",
        "seed",
        "family",
        "context",
        "source",
        "dope",
        "updates",
        "dev",
        "unpaired",
        "nonfinite",
    ],
)
def test_incomplete_mixed_or_nonfinite_panel_never_becomes_a_verdict(kind):
    cells = fixture_cells()
    if kind == "missing":
        cells.pop()
    elif kind == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif kind == "seed":
        cells[-1]["seed"] = 39
    elif kind == "family":
        cells[-1]["assessment"]["per_family"].pop(
            next(iter(cells[-1]["assessment"]["per_family"]))
        )
    elif kind == "context":
        cells[-1]["assessment"]["per_context"].pop(
            next(iter(cells[-1]["assessment"]["per_context"]))
        )
    elif kind == "source":
        cells[-1]["source_commit"] = "other"
    elif kind == "dope":
        cells[-1]["dope_after"] = "bad"
    elif kind == "updates":
        cells[-1]["updates"] = 39
    elif kind == "dev":
        cells[-1]["DL_Val_payloads_opened"] = 1
    elif kind == "unpaired":
        cells[1]["fine_tuned_adapter_sha256"] = "other"
    elif kind == "nonfinite":
        cells[-1]["assessment"]["per_family"][
            next(iter(cells[-1]["assessment"]["per_family"]))
        ] = float("nan")
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        r.summarize(cells, draws=10000)


def test_dry_run_cannot_load_data_runtime_or_create_a_ticket(monkeypatch, capsys):
    def forbidden(*a, **k):
        pytest.fail("CPU preparation tried runtime/GPU/data")

    monkeypatch.setattr(w.r, "runtime", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(sys, "argv", ["run_wave.py", "--cpu-dry-run"])
    w.main()
    result = json.loads(capsys.readouterr().out)
    assert (
        result["data_reads"]
        == result["checkpoint_reads"]
        == result["optimizer_updates"]
        == result["outcome_cells_run"]
        == 0
    )
    assert not result["GPU_started"] and not result["GPU_ticket_created"]


def test_predecessors_accept_prior_start_not_cpu_plan_or_missing_predecessor(
    tmp_path, monkeypatch
):
    def parse(name):
        epoch, pid, wave = name.split("-")
        return int(epoch), int(pid), wave

    monkeypatch.setattr(
        w, "fair", SimpleNamespace(fifo_lock=SimpleNamespace(parse_ticket_name=parse))
    )

    def missing(wave):
        raise w.PredecessorPending("not closed")

    def started(wave):
        return {
            "wave": wave,
            "cap_seconds": 9000 if wave == "H99" else 2700,
            "attempts": [
                {"started_utc": "2026-10-31T00:00:00Z", "command": ["python", "worker"]}
            ],
            "in_progress": None,
        }

    assert set(
        w.require_predecessors(
            queue_dir=tmp_path,
            alive_fn=lambda _: True,
            read_fn=missing,
            budget_fn=started,
        )
    ) == {"H99", "H100"}
    with pytest.raises(w.PredecessorPending):
        w.require_predecessors(
            queue_dir=tmp_path,
            alive_fn=lambda _: False,
            read_fn=lambda wave: {
                "complete": True,
                "schema": f"jope.{wave.lower()}.cpu-preparation/v1",
            },
            budget_fn=lambda _: None,
        )
    for i, wave in enumerate(["H99", "H100"]):
        (tmp_path / f"{i + 1}-123-{wave}").write_text(
            f"epoch_ns={i + 1}\npid=123\nwave={wave}\n"
        )
    assert set(
        w.require_predecessors(
            queue_dir=tmp_path,
            alive_fn=lambda _: True,
            read_fn=missing,
            budget_fn=lambda _: None,
        )
    ) == {"H99", "H100"}


def test_budget_refuses_uncertain_launch_negative_unknown_and_exhaustion():
    for ledger in [
        {"attempts": [], "in_progress": {"arm": "CONTROL"}},
        {"attempts": [{"wall_seconds": -1, "wait_seconds": 0}], "in_progress": None},
        {
            "attempts": [{"wall_seconds": float("nan"), "wait_seconds": 0}],
            "in_progress": None,
        },
        {"attempts": [{"wall_seconds": 2691, "wait_seconds": 0}], "in_progress": None},
    ]:
        with pytest.raises(RuntimeError):
            w.remaining(ledger)
    assert (
        w.remaining(
            {
                "attempts": [{"wall_seconds": 100, "wait_seconds": 50}],
                "in_progress": None,
            }
        )
        == 2600
    )


def test_only_load_abort_before_full_scoring_retries_same_attempt_token():
    a = {
        "error_kind": "host_load",
        "attempt": 1,
        "attempt_token": "t",
        "arm": "CONTROL",
        "seed": 42,
    }
    phase = {
        "outcome_started": False,
        "attempt_token": "t",
        "arm": "CONTROL",
        "seed": 42,
    }
    assert w.retryable(a, phase, result_exists=False)
    assert not w.retryable(a, dict(phase, outcome_started=True), result_exists=False)
    assert not w.retryable(dict(a, attempt=3), phase, result_exists=False)
    assert not w.retryable(dict(a, error_kind="fatal"), phase, result_exists=False)


def test_wrong_draw_count_and_unbound_aggregates_are_incomplete():
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        r.summarize(fixture_cells(), draws=9999)
    cells = fixture_cells()
    p = cells[0]["assessment"]
    p["per_family"][next(iter(p["per_family"]))] = 2.0
    with pytest.raises(RuntimeError, match="aggregate"):
        r.summarize(cells)


def test_unleased_worker_refuses_before_any_packet_read(monkeypatch):
    def reject(_):
        raise RuntimeError("no inherited qualification lease")

    monkeypatch.setattr(r, "runtime", lambda: SimpleNamespace(assert_lease=reject))
    with pytest.raises(RuntimeError, match="lease"):
        r.run_cell(SimpleNamespace(lock_fd=None))


def test_retry_driver_never_exceeds_two_extras_or_drops_charged_voids(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(w, "fair", SimpleNamespace(note=lambda *args: None))
    ledger = {"attempts": [], "in_progress": None}
    attempts = []

    def void(root, current, arm, seed, timing, attempt):
        attempts.append(attempt)
        record = {"wall_seconds": 2.0, "wait_seconds": 1.0}
        current["attempts"].append(record)
        return None, record

    with pytest.raises(RuntimeError, match="exhausted"):
        w.execute(tmp_path, ledger, "CONTROL", 42, attempt_fn=void)
    assert (
        attempts == [1, 2, 3]
        and sum(a["wall_seconds"] for a in ledger["attempts"]) == 6
    )
    assert w.remaining(ledger) == 2694


def test_run_namespace_refuses_old_wave_and_symlinked_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(w.Path, "home", lambda: tmp_path)
    authorized = tmp_path / "jope-data/h105/h105-wave"
    w.validate_run_root(authorized)
    old = tmp_path / "jope-data/h90/h90-wave"
    old.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        w.validate_run_root(old)
    authorized.parent.mkdir(parents=True)
    authorized.symlink_to(old, target_is_directory=True)
    with pytest.raises(RuntimeError):
        w.validate_run_root(authorized)


def test_native_h99_completed_gpu_receipt_establishes_prior_start_without_ticket(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        w,
        "fair",
        SimpleNamespace(fifo_lock=SimpleNamespace(parse_ticket_name=lambda _: None)),
    )
    native = {
        "schema": "jope.h99.gpu-budget/v1",
        "cap_seconds": 9000.0,
        "in_progress": None,
        "attempts": [
            {
                "status": "complete",
                "wall_seconds": 100.0,
                "gpu_free_mib": 24000,
                "preflight": {"wave": "H99"},
                "result": {
                    "artifact_sha256": "a" * 64,
                    "summary": str(
                        Path.home()
                        / "jope-data/h99/h99-wave/outcomes/cells/ESH-l2/seed36/attempt1/summary.json"
                    ),
                },
            }
        ],
    }

    def budget(wave):
        return (
            native
            if wave == "H99"
            else {
                "wave": "H100",
                "cap_seconds": 2700,
                "attempts": [
                    {
                        "started_utc": "2026-10-31T00:00:00Z",
                        "command": ["python", "runner.py"],
                    }
                ],
                "in_progress": None,
            }
        )

    def missing(_):
        raise w.PredecessorPending("not closed")

    proof = w.require_predecessors(
        queue_dir=tmp_path, alive_fn=lambda _: False, read_fn=missing, budget_fn=budget
    )
    assert proof["H99"]["prior_gpu_start"]
    for bad in [
        dict(native, schema="jope.h99.cpu-plan/v1"),
        dict(native, attempts=[]),
        dict(native, attempts=[dict(native["attempts"][0], status="failed")]),
    ]:
        with pytest.raises(w.PredecessorPending):
            w.require_predecessors(
                queue_dir=tmp_path,
                alive_fn=lambda _: False,
                read_fn=missing,
                budget_fn=lambda wave: bad if wave == "H99" else budget(wave),
            )


def test_execution_source_checks_actual_owned_and_shared_bytes_not_physical_head(
    tmp_path, monkeypatch
):
    own, shared = tmp_path / "own", tmp_path / "shared"
    own.mkdir()
    shared.mkdir()
    (own / "runner.py").write_bytes(b"owned code\n")
    (shared / "shared.py").write_bytes(b"library code\n")
    import hashlib

    digest = hashlib.sha256(b"library code\n").hexdigest()
    monkeypatch.setattr(r, "HERE", own)
    monkeypatch.setattr(r, "SOURCE_ROOT", shared)
    monkeypatch.setattr(r, "EXECUTION_FILES", ("runner.py",))
    monkeypatch.setattr(r, "SOURCE_PINS", {"shared.py": digest})

    def blob(ref, path):
        assert ref == "a" * 40
        return b"owned code\n" if path.endswith("/runner.py") else b"library code\n"

    monkeypatch.setattr(r, "git_blob", blob)
    proof = r.verify_execution_source("a" * 40)
    assert proof["execution_commit"] == "a" * 40
    (own / "runner.py").write_bytes(b"changed code\n")
    with pytest.raises(RuntimeError, match="source"):
        r.verify_execution_source("a" * 40)
    (own / "runner.py").write_bytes(b"owned code\n")
    (shared / "shared.py").write_bytes(b"changed library\n")
    with pytest.raises(RuntimeError, match="source"):
        r.verify_execution_source("a" * 40)
