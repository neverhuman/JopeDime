"""Synthetic native-receipt controls; never import a model or open a real run."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "h99_h100_readout_test", HERE / "readout.py"
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return r.sha(path.read_bytes())


def load(path):
    return json.loads(path.read_text())


def fixture(wave, root, harmed=0):
    families = [f"family-{i:02}" for i in range(41 if wave == "H99" else 19)]
    baseline = dict.fromkeys(families, 0.5)
    ledger = {
        "cap_seconds": 9000 if wave == "H99" else 2700,
        "in_progress": None,
        "attempts": [],
    }
    metrics = {"arms": {}}
    if wave == "H99":
        folds = [
            [f"fit-family-{fold * 41 + i:03}" for i in range(41)] for fold in range(4)
        ] + [[f"family-{i:02}" for i in range(41)]]
        fold_a_families = sorted(family for fold in folds[:4] for family in fold)
        fold_b_families = folds[4]
        cell_assignment = {
            "seed": 9900,
            "folds": folds,
            "fold_a_families": fold_a_families,
            "fold_b_families": fold_b_families,
            "train_contexts_a": 4,
            "train_contexts_b": 1,
        }
        fold_assignment = {
            "seed": 9900,
            "folds": folds,
            "fold_a_families": fold_a_families,
            "fold_b_families": fold_b_families,
            "source_split": "DL-Train",
            "sealed_meta_test_inputs_opened": False,
        }
        cell_digest = r.sha(
            json.dumps(cell_assignment, sort_keys=True, separators=(",", ":")).encode()
        )
        fold_digest = r.sha(
            json.dumps(
                {"seed": 9900, "folds": folds},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        ledger["schema"] = "jope.h99.gpu-budget/v1"
        candidates = []
        for arm, seed in [(a, s) for a in r.ARMS[wave] for s in r.SEEDS[wave]] + [
            ("NULL", 36)
        ]:
            digest = r.sha(f"{arm}-{seed}".encode())
            summary = root / f"outcomes/cells/{arm}/seed{seed}/attempt1/summary.json"
            write(
                summary,
                {
                    "schema": "jope.h99.cell/v1",
                    "status": "complete",
                    "arm": arm,
                    "seed": seed,
                    "fold_seed": 9900,
                    "fold_assignment": cell_assignment,
                    "fold_assignment_sha256": cell_digest,
                    "fold_b_scored": False,
                    "dope_sha256_before": r.DOPE,
                    "dope_sha256_after": r.DOPE,
                    "r882_checkpoint_sha256": r.R882,
                    "source_manifest_sha256": r.MANIFEST,
                    "candidate_adapter_artifact_sha256": digest,
                },
            )
            ledger["attempts"].append(
                {
                    "arm": arm,
                    "seed": seed,
                    "wall_seconds": 1.0,
                    "wait_seconds": 0.0,
                    "status": "complete",
                    "result": {"summary": str(summary), "artifact_sha256": digest},
                }
            )
            if arm != "NULL":
                gains = {
                    f: -0.02 if i < harmed else 0.02 for i, f in enumerate(families)
                }
                candidates.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "fold_b_scoring_passes": 1,
                        "candidate_adapter_sha256": digest,
                        "family_gain_vs_r882": gains,
                        "family_utility": {f: baseline[f] + gains[f] for f in families},
                    }
                )
        fold = {
            "schema": "jope.h99.fold-b/v1",
            "status": "complete",
            "fold_seed": 9900,
            "source_manifest_sha256": r.MANIFEST,
            "r882_checkpoint_sha256": r.R882,
            "dope_sha256_before": r.DOPE,
            "dope_sha256_after": r.DOPE,
            "candidate_count": 6,
            "fold_b_read_count": 6,
            "fold_b_family_count": 41,
            "null_cell_complete": True,
            "null_fold_b_scored": False,
            "validation_payloads_opened": False,
            "holdout_payloads_opened": 0,
            "outer_rows_delivered": 0,
            "fold_assignment": fold_assignment,
            "fold_assignment_sha256": fold_digest,
            "baseline_family_utility": baseline,
            "candidates": candidates,
        }
        write(root / "outcomes/fold_b.json", fold)
        ledger["attempts"].append(
            {
                "action": "fold_b_scoring",
                "wall_seconds": 1.0,
                "wait_seconds": 0.0,
                "status": "complete",
                "result": {"fold_b": str(root / "outcomes/fold_b.json")},
            }
        )
        metrics.update(
            schema="jope.h99.admission-metrics/v1",
            status="complete",
            fold_seed=9900,
            fold_b_family_count=41,
            fold_b_read_count=6,
            bootstrap_draws=10000,
            strict_margin=0.001,
            source_manifest_sha256=r.MANIFEST,
            fold_assignment_sha256=fold_digest,
        )
        for arm in r.ARMS[wave]:
            metrics["arms"][arm] = {
                "pooled_mean_gain": (41 - 2 * harmed) * 0.02 / 41,
                "pooled_family_median_gain": 0.02,
                "harm_fraction_below_minus_0_01": harmed / 41,
                "mean_lcb95_one_sided": 0.01,
                "holm": {"raw_p_one_sided": 0.0001, "holm_reject_alpha_0_05": True},
                "admitted": harmed / 41 <= 0.1,
                "elo_vs_r882_median_across_seeds": 1600,
                "elo_vs_r882_median_ci95_family_cluster": [1550, 1650],
                "per_seed": [
                    {
                        "seed": s,
                        "elo_vs_r882": 1600,
                        "elo_ci95_family_cluster": [1540, 1660],
                    }
                    for s in r.SEEDS[wave]
                ],
            }
        metric_path = root / "outcomes/admission.json"
    else:
        ledger["wave"] = wave
        for arm in r.ARMS[wave]:
            for seed in r.SEEDS[wave]:
                values = {
                    f: 0.48
                    if i < harmed and arm != "CONTROL"
                    else 0.5 + (0 if arm == "CONTROL" else 0.02)
                    for i, f in enumerate(families)
                }
                cell = {
                    "wave": wave,
                    "arm": arm,
                    "seed": seed,
                    "timing_only": False,
                    "dope_before": r.DOPE,
                    "dope_after": r.DOPE,
                    "checkpoint_sha256": r.R882,
                    "manifest_sha256": r.MANIFEST,
                    "trainable_count": 29211,
                    "updates": 40,
                    "holdout_payloads_opened": 0,
                    "outer_rows_delivered": 0,
                    "source_commit": "c" * 40,
                    "prereg_sha256": "p" * 64,
                    "initial_adapter_sha256": "i" * 64,
                    "source_hashes": {"source": "s" * 64},
                    "folds": [],
                    "dev": {"per_family": values},
                    "r882_dev": {"per_family": baseline},
                }
                digest = write(root / f"{arm}-seed{seed}-attempt1.json", cell)
                ledger["attempts"].append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "timing": False,
                        "attempt": 1,
                        "wall_seconds": 1.0,
                        "wait_seconds": 0.0,
                        "outcome_started": True,
                        "stage": "complete",
                        "receipt_sha256": digest,
                    }
                )
            point = 0 if arm == "CONTROL" else (19 - 2 * harmed) * 0.02 / 19
            harm = 0 if arm == "CONTROL" else harmed / 19
            metrics["arms"][arm] = {
                "median_gain": point,
                "harm_fraction": harm,
                "lcb95": 0 if arm == "CONTROL" else 0.01,
                "admitted": arm != "CONTROL" and harm <= 0.1,
                "elo_vs_r882": {
                    "median_elo": 1500 if arm == "CONTROL" else "Infinity",
                    "median_elo_ci95": [1500, 1500]
                    if arm == "CONTROL"
                    else [1600, "Infinity"],
                    "per_seed": [
                        {
                            "seed": s,
                            "elo": 1500 if arm == "CONTROL" else "Infinity",
                            "elo_ci95": [1500, 1500]
                            if arm == "CONTROL"
                            else [1600, "Infinity"],
                        }
                        for s in r.SEEDS[wave]
                    ],
                },
            }
        timing_digest = write(
            root / "timing-ASAM10-seed39-attempt1.json", {"timing_only": True}
        )
        ledger["attempts"].insert(
            0,
            {
                "arm": "ASAM10",
                "seed": 39,
                "attempt": 1,
                "timing": True,
                "wall_seconds": 1.0,
                "wait_seconds": 0.0,
                "receipt_sha256": timing_digest,
            },
        )
        metrics.update(
            schema="jope.h100.metrics/v1",
            complete=True,
            seeds=list(r.SEEDS[wave]),
            bootstrap_draws=10000,
            ledger=copy.deepcopy(ledger),
            gpu_wall_seconds=13.0,
            holm={
                a: {"tail_area": 0.0001, "passes": True}
                for a in r.ARMS[wave]
                if a != "CONTROL"
            },
        )
        metric_path = root / "metrics.json"
    write(root / "budget.json", ledger)
    write(metric_path, metrics)
    return metric_path


@pytest.mark.parametrize("wave", ["H99", "H100"])
def test_complete_panel_publishes_native_elo_ci_and_wave_specific_common_gate(
    tmp_path, wave
):
    fixture(wave, tmp_path)
    result = r.readout(wave, tmp_path)
    assert result["status"] == "COMPLETE", result
    assert result["verdict"] == "ADMIT"
    assert result["family_count"] == (41 if wave == "H99" else 19)
    for arm, item in result["arms"].items():
        assert item["elo_vs_r882"]["reference_elo"] == 1500
        assert item["common_gate"]["primary_reference"] == (
            "r882" if wave == "H99" else "CONTROL"
        )
        assert item["common_gate"]["verdict"] == (
            "NO_ADMIT" if arm == "CONTROL" else "ADMIT"
        )
    json.dumps(result, allow_nan=False)
    r.publish(result, tmp_path / "published")
    r.publish(result, tmp_path / "published")  # exact replay is idempotent
    assert (
        "Median Elo vs r882" in (tmp_path / f"published/{wave}-readout.md").read_text()
    )


def test_h99_accepts_different_hashes_when_fold_family_lists_match(tmp_path):
    fixture("H99", tmp_path)
    summary_path = next((tmp_path / "outcomes/cells").glob("**/summary.json"))
    summary = load(summary_path)
    fold = load(tmp_path / "outcomes/fold_b.json")

    assert summary["fold_assignment_sha256"] != fold["fold_assignment_sha256"]
    assert r.h99_fold_assignment_matches(summary, fold)
    result = r.readout("H99", tmp_path)
    assert result["status"] == "COMPLETE", result


def test_h99_rejects_cell_fold_family_list_mismatch(tmp_path):
    fixture("H99", tmp_path)
    summary_path = next((tmp_path / "outcomes/cells").glob("**/summary.json"))
    summary = load(summary_path)
    summary["fold_assignment"]["fold_b_families"].append("unassigned-family")
    write(summary_path, summary)

    result = r.readout("H99", tmp_path)
    assert result["status"] == "INVALID", result
    assert "fold/scoring mismatch" in result["reason"]


@pytest.mark.parametrize("wave,harmed", [("H99", 5), ("H100", 2)])
def test_harm_tail_blocks_gain_and_holm_hits(tmp_path, wave, harmed):
    fixture(wave, tmp_path, harmed)
    result = r.readout(wave, tmp_path)
    assert result["status"] == "COMPLETE", result
    assert result["verdict"] == "NO_ADMIT"
    assert all(
        x["common_gate"]["verdict"] == "NO_ADMIT" for x in result["arms"].values()
    )


@pytest.mark.parametrize("wave", ["H99", "H100"])
@pytest.mark.parametrize(
    "state", ["missing", "in_progress", "partial", "missing_metrics"]
)
def test_pending_never_opens_outcomes_before_complete_fifo_panel(
    tmp_path, wave, state, monkeypatch
):
    fixture(wave, tmp_path)
    budget = tmp_path / "budget.json"
    ledger = load(budget)
    if state == "missing":
        budget.rename(tmp_path / "budget.saved")
    elif state == "in_progress":
        ledger["in_progress"] = {"arm": "pending"}
        write(budget, ledger)
    elif state == "partial":
        ledger["attempts"].pop()
        write(budget, ledger)
    else:
        (
            tmp_path / ("outcomes/admission.json" if wave == "H99" else "metrics.json")
        ).rename(tmp_path / "metrics.saved")
    if state != "missing_metrics":
        original = r.Receipts.read

        def audited(self, path, expected_sha=None):
            assert Path(path).name == "budget.json", (
                "partial outcomes must stay unopened"
            )
            return original(self, path, expected_sha)

        monkeypatch.setattr(r.Receipts, "read", audited)
    result = r.readout(wave, tmp_path)
    assert result["status"] == "PENDING", result
    assert result["verdict"] == "UNKNOWN" and result["arms"] == {}


@pytest.mark.parametrize("wave", ["H99", "H100"])
@pytest.mark.parametrize(
    "mutation", ["duplicate", "negative_charge", "nonfinite", "cap", "unknown_score"]
)
def test_invalid_fifo_accounting_cannot_emit_a_verdict(tmp_path, wave, mutation):
    fixture(wave, tmp_path)
    ledger = load(tmp_path / "budget.json")
    row = ledger["attempts"][1]
    if mutation == "duplicate":
        ledger["attempts"].append(copy.deepcopy(row))
    elif mutation == "negative_charge":
        row["wall_seconds"] = -1
    elif mutation == "nonfinite":
        row["wall_seconds"] = float("nan")
    elif mutation == "cap":
        ledger["cap_seconds"] += 1
    else:
        row.update(status="void-or-unknown", error="failed", outcome_started=None)
        if wave == "H99":
            row["action"] = "fold_b_scoring"
    write(tmp_path / "budget.json", ledger)
    result = r.readout(wave, tmp_path)
    assert result["status"] == "INVALID", result
    assert result["verdict"] == "UNKNOWN" and result["arms"] == {}


@pytest.mark.parametrize("wave", ["H99", "H100"])
@pytest.mark.parametrize(
    "mutation",
    ["native_gate", "native_gain", "CI", "seed_inventory", "strict_tie", "holm"],
)
def test_native_report_inconsistency_is_refused(tmp_path, wave, mutation):
    path = fixture(wave, tmp_path)
    native = load(path)
    item = native["arms"][r.ARMS[wave][-1]]
    if mutation == "native_gate":
        item["admitted"] = False
    elif mutation == "native_gain":
        item["pooled_mean_gain" if wave == "H99" else "median_gain"] = 0.123
    elif mutation == "CI":
        if wave == "H99":
            item["elo_vs_r882_median_ci95_family_cluster"] = [1700, 1600]
        else:
            item["elo_vs_r882"]["median_elo_ci95"] = ["Infinity", 1500]
    elif mutation == "seed_inventory":
        rows = item["per_seed"] if wave == "H99" else item["elo_vs_r882"]["per_seed"]
        rows[0]["seed"] = 99
    elif mutation == "strict_tie":
        item["mean_lcb95_one_sided" if wave == "H99" else "lcb95"] = r.MARGIN[wave]
    else:
        target = item["holm"] if wave == "H99" else native["holm"][r.ARMS[wave][-1]]
        target["raw_p_one_sided" if wave == "H99" else "tail_area"] = 0.9
    write(path, native)
    result = r.readout(wave, tmp_path)
    assert result["status"] == "INVALID", result
    assert result["verdict"] == "UNKNOWN" and result["arms"] == {}


def test_h100_stale_metrics_and_changed_receipt_hash_fail(tmp_path):
    path = fixture("H100", tmp_path)
    native = load(path)
    native["ledger"]["attempts"][0]["wall_seconds"] += 1
    write(path, native)
    assert r.readout("H100", tmp_path)["status"] == "INVALID"
    fixture("H100", tmp_path)
    cell = tmp_path / "CONTROL-seed39-attempt1.json"
    data = load(cell)
    data["dope_after"] = "changed"
    write(cell, data)
    assert r.readout("H100", tmp_path)["status"] == "INVALID"


def test_receipt_escape_symlink_and_changing_snapshot_fail_closed(tmp_path):
    fixture("H99", tmp_path)
    budget = load(tmp_path / "budget.json")
    budget["attempts"][0]["result"]["summary"] = str(tmp_path.parent / "outside.json")
    write(tmp_path / "budget.json", budget)
    assert r.readout("H99", tmp_path)["status"] == "INVALID"
    store = r.Receipts(tmp_path)
    with pytest.raises(ValueError, match="escapes run root"):
        store.read(tmp_path / ".." / "outside.json")
    store = r.Receipts(tmp_path)
    store.read(tmp_path / "budget.json")
    write(tmp_path / "budget.json", {})
    with pytest.raises(r.Pending, match="snapshot changed"):
        store.stable()
    alias = tmp_path / "alias.json"
    alias.symlink_to(tmp_path / "budget.json")
    with pytest.raises(ValueError, match="symlinks"):
        store.read(alias)


def test_h100_pre_scoring_rearm_charge_is_retained_without_repeating_the_cell(tmp_path):
    path = fixture("H100", tmp_path)
    budget = load(tmp_path / "budget.json")
    row = next(
        x for x in budget["attempts"] if x["arm"] == "CONTROL" and x["seed"] == 39
    )
    row["attempt"] = 2
    (tmp_path / "CONTROL-seed39-attempt1.json").rename(
        tmp_path / "CONTROL-seed39-attempt2.json"
    )
    budget["attempts"].insert(
        1,
        {
            "arm": "CONTROL",
            "seed": 39,
            "attempt": 1,
            "timing": False,
            "wall_seconds": 15.0,
            "wait_seconds": 0.0,
            "error": "pre-scoring disk abort",
            "outcome_started": False,
            "stage": "loading",
        },
    )
    native = load(path)
    native["ledger"] = budget
    native["gpu_wall_seconds"] = 28.0
    write(tmp_path / "budget.json", budget)
    write(path, native)
    result = r.readout("H100", tmp_path)
    assert result["status"] == "COMPLETE", result
    assert result["gpu_wall_seconds"] == 28.0


def test_holm_preserves_each_native_boundary_and_step_down_stop():
    assert r.holm_decisions({"ESH-l2": 0.025, "ESH-l4": 0.04}, strict=False) == {
        "ESH-l2": True,
        "ESH-l4": True,
    }
    assert not any(
        r.holm_decisions({"ASAM05": 0.025, "ASAM10": 0.04}, strict=True).values()
    )
    assert not any(
        r.holm_decisions({"ESH-l2": 0.03, "ESH-l4": 0.04}, strict=False).values()
    )


def test_watch_cli_publishes_terminal_report_and_never_imports_torch(tmp_path):
    root = tmp_path / "run"
    fixture("H100", root)
    out = tmp_path / "report"
    code = (
        "import runpy,sys;sys.modules['torch']=None;sys.argv="
        + repr(
            [
                str(HERE / "readout.py"),
                "--wave",
                "H100",
                "--root",
                str(root),
                "--out-dir",
                str(out),
                "--watch",
                "--poll-seconds",
                "1",
                "--timeout-seconds",
                "2",
            ]
        )
        + ";runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert load(out / "H100-readout.json")["status"] == "COMPLETE"


def test_timeout_emits_unknown_and_preserves_existing_readout(tmp_path):
    out = tmp_path / "report"
    command = [
        sys.executable,
        str(HERE / "readout.py"),
        "--wave",
        "H99",
        "--root",
        str(tmp_path / "empty"),
        "--out-dir",
        str(out),
        "--watch",
        "--poll-seconds",
        "1",
        "--timeout-seconds",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 3, result.stderr
    value = load(out / "H99-readout.json")
    assert value["verdict"] == "UNKNOWN" and not value["arms"]
    with pytest.raises(ValueError, match="prior readout"):
        r.publish(dict(value, reason="changed"), out)


def test_ties_are_reported_separately_under_each_native_elo_policy():
    values = [0, 0.00005, -0.00005, 0.01, -0.01]
    assert r.win_tie_loss(values, 1e-4) == {"wins": 1, "ties": 3, "losses": 1}
    assert r.win_tie_loss(values, 0) == {"wins": 2, "ties": 1, "losses": 2}
