"""Data-free H229 engineering regressions, resource guards and scientific gates."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load_local(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stat = load_local("analysis")
b = load_local("backend")
protocol = load_local("protocol")


def program(i):
    dsl = f"(pipe X (ridge :alpha {i + 1}))"
    return dict(dsl=dsl, program_sha256=hashlib.sha256(dsl.encode()).hexdigest())


def feature(dsl):
    i = int(hashlib.sha256(dsl.encode()).hexdigest()[:8], 16)
    return np.zeros(412, np.float32), np.random.default_rng(i).normal(size=16).astype(
        np.float32
    )


def proposer(**args):
    return [program(i + 1) for i in range(args["n_proposals"])]


def test_registered_pools_keep_parent_and_equal_cost_fps():
    pools = stat.build_pools(program(0), proposer, {}, feature)
    assert {a: len(x) for a, x in pools.items()} == dict(
        NATIVE16=16, NATIVE64=64, FPS16of64=16
    )
    assert all(x[0] == program(0) for x in pools.values())
    assert {p["program_sha256"] for p in pools["FPS16of64"]} <= {
        p["program_sha256"] for p in pools["NATIVE64"]
    }


def test_fps_ties_follow_generator_order_without_outcomes():
    rows = [program(i) for i in range(64)]
    vectors = {p["program_sha256"]: np.zeros(16) for p in rows}
    assert stat.fps(rows, vectors) == rows[:16]


def test_no_refill_or_duplicate_width():
    with pytest.raises(ValueError):
        stat.unique(program(0), [program(1)] * 15, 16)


def test_first_cost_argmin_and_oracle_are_separate():
    result = stat.score(
        [0.0, 0.0],
        [dict(fitness=1, cv_gate_passed=True), dict(fitness=-9, cv_gate_passed=True)],
    )
    assert result == dict(utility=-1.0, selected_index=0, oracle_utility=9.0)
    assert (
        stat.score([0.0], [dict(fitness=100, cv_gate_passed=False)])["utility"] == -2.0
    )


def test_context_id_is_external_in_actual_h212_prediction(tmp_path):
    import torch

    torch.set_num_threads(1)
    source = subprocess.check_output(
        [
            "git",
            "show",
            json.loads((HERE / "manifest.json").read_text())["source_parent"]
            + ":research/flywheel_selfplay/h212_pool_width/runner.py",
        ],
        cwd=b.REPO,
    )
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "predict_dynamic"
    )
    module_path = tmp_path / "h212_predict_dynamic.py"
    module_path.write_text("import numpy as np\n\n" + ast.unparse(fn) + "\n")
    spec = importlib.util.spec_from_file_location("h212_predict_dynamic", module_path)
    assert spec is not None and spec.loader is not None
    scope = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scope)
    cid = "a" * 64
    net = NS(
        adapter_10tok=NS(
            predict_costs=lambda **k: dict(
                total_cost=torch.zeros((1, k["prog_412"].shape[1]))
            )
        )
    )
    h124 = NS(program=lambda pools, key: np.zeros((4, 412), np.float32))
    context = dict(
        tokens_10x32=np.zeros((10, 32), np.float32), mask_10=np.ones(10, bool)
    )
    pool = dict(fitness=np.ones(4), gate=np.ones(4, bool))
    projection = (
        NS(rows=lambda lo, hi: np.zeros((hi - lo, 16), np.float32)),
        {cid: (0, 4)},
    )
    result = scope.predict_dynamic(
        net,
        h124,
        cid,
        context,
        pool,
        projection,
        [program(1)],
        NS(featurize_dsl_412=feature),
    )
    assert result.shape == (5,) and "context_sha256" not in context


def test_rust_eval_scratch_creation_is_idempotent(tmp_path):
    path = tmp_path / "rust_eval"
    assert b.ensure_scratch(path) == b.ensure_scratch(path) and path.is_dir()


def test_h139_reference_and_checkpoint_are_relocated_before_roster():
    h139 = NS(INPUT="old", ARM_DIR="missing-h139")
    h124 = NS(FILES={})
    b.relocate(h139, h124, Path("input"), Path("copy"), Path("h195-reference"))
    assert h139.ARM_DIR == Path("h195-reference") and h124.R882 == Path("copy")
    assert h124.FILES[Path("copy")] == b.R882_SHA


def test_h58_checks_raw_identity_and_transformed_envelope():
    raw = b"raw TRAIN bytes"

    def envelope(value):
        return b"CV:" + value

    expected = hashlib.sha256(raw).hexdigest()
    b.validate_envelope(raw, envelope(raw), expected, envelope)
    assert hashlib.sha256(envelope(raw)).hexdigest() != expected
    with pytest.raises(ValueError):
        b.validate_envelope(raw, raw, expected, envelope)
    with pytest.raises(ValueError):
        b.validate_envelope(raw, envelope(raw), "0" * 64, envelope)


def test_bit_exact_baseline_rejects_even_sub_tolerance_drift():
    values = np.zeros(91)
    values[-1] = 91 * 1.1377941898164525
    reference = dict(utility=dict(per_context=values.tolist()))
    assert b.baseline_parity(values, reference)["bit_exact_contexts"] == 91
    values[0] = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError):
        b.baseline_parity(values, reference)


def observation():
    return dict(
        host="xbabe1",
        nice=19,
        affinity=4,
        rss=1e9,
        ram=24 * 2**30,
        free=100e9,
        output=0,
        load5=40.0,
    )


def test_load_hysteresis_and_deadline_are_distinct():
    o = observation()
    o["load5"] = 70.0
    assert not protocol.admission(o, now=0)
    o["load5"] = 70.01
    assert protocol.admission(o, now=0)
    o["load5"] = 65.0
    assert protocol.admission(o, paused=True, now=0)
    o["load5"] = 60.0
    assert not protocol.admission(o, paused=True, now=0)
    with pytest.raises(RuntimeError):
        protocol.admission(o, now=protocol.DEADLINE)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rss", 6 * 2**30 + 1),
        ("ram", 6 * 2**30 - 1),
        ("free", 80e9 - 1),
        ("output", 2e9 + 1),
        ("affinity", 5),
        ("nice", 0),
        ("load5", float("nan")),
    ],
)
def test_hard_guards_remain_aborts(field, value):
    o = observation()
    o[field] = value
    with pytest.raises(RuntimeError):
        protocol.admission(o, now=0)


def positive(plateau=0.003, lower=0.001, p=0.0001):
    return dict(
        plateau_median_gain=plateau,
        median_gain=0.003,
        ci95=[lower, 0.004],
        p_one_sided=p,
    )


def test_strict_margin_ties_ci_and_holm():
    stats = {a: positive() for a in stat.CONTRASTS}
    assert stat.decide(stats)["verdict"] == "ADVANCE"
    stats["NATIVE64"] = positive(plateau=0.002)
    assert stat.decide(stats)["verdict"] == "NO_ADVANCE"
    stats["NATIVE64"] = positive(lower=0)
    assert stat.decide(stats)["verdict"] == "NO_ADVANCE"
    stats = {a: positive(p=0.04) for a in stat.CONTRASTS}
    assert stat.decide(stats)["verdict"] == "NO_ADVANCE"
    assert stat.decide({})["verdict"] == "INCOMPLETE"


def test_family_bootstrap_replays_and_requires_complete_panel():
    families = [str(i % 19) for i in range(91)]
    values = np.ones((3, 91)) * 0.01
    a = stat.gain_stats(values, families)
    assert a == stat.gain_stats(values, families)
    assert np.allclose(a["ci95"], [0.01, 0.01])
    with pytest.raises(ValueError):
        stat.gain_stats(values[:2], families)


def test_single_thread_baseline_clamps_legacy_thread_request():
    seen = []
    fake = NS(set_num_threads=lambda n: seen.append(n))
    setter = fake.set_num_threads
    with b.single_thread_torch(fake):
        fake.set_num_threads(4)
    assert seen == [1, 1, 1] and fake.set_num_threads is setter


@pytest.fixture
def treatment(monkeypatch):
    monkeypatch.setattr(
        b,
        "module",
        lambda path, *args, **kwargs: (
            stat if path.endswith("analysis.py") else protocol
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "h229_test_runner", HERE / "runner.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_full_treatment_loop_writes_analysis_and_removes_own_scratch(
    tmp_path, treatment
):
    cid = "a" * 64
    raw = b"TRAIN fixture"
    input_root = tmp_path / "input"
    (input_root / "tables").mkdir(parents=True)
    (input_root / "tables" / f"{cid}.jtt1").write_bytes(raw)
    parent = dict(
        program(0), family_id="bank", dl_split="DL-Train", landmark_summary=np.zeros(12)
    )

    def evaluate(table, programs, **kwargs):
        assert kwargs["run_outer_test"] is False
        return [
            dict(
                program_sha256=p["program_sha256"],
                status="completed",
                fitness=float(i),
                cv_gate_passed=True,
                feat412=np.zeros(412),
            )
            for i, p in enumerate(programs)
        ]

    generator = NS(
        propose_flywheel_candidates=proposer,
        featurize_dsl_412=feature,
        evaluate_gp_batch_rust=evaluate,
    )
    h212 = NS(
        INPUT=input_root,
        predict_dynamic=lambda net, h124, key, context, *rest: np.arange(
            4 + len(rest[-2]), dtype=float
        ),
    )
    runtime = dict(
        phase="smoke",
        generator=generator,
        h212=h212,
        champions=[parent],
        context={cid: dict(lm48_z=np.zeros(48))},
        envelope=NS(training_cv_envelope=lambda x: b"CV:" + x),
        net=None,
        h124=None,
        pools={cid: {}},
        projection=None,
    )
    row = dict(
        context_sha256=cid,
        family_id="target",
        dl_split="DL-Train",
        task="regression",
        train_only_jtt1_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result = treatment.measure(runtime, row, stat.SEEDS[0], tmp_path)
    assert result["logical_evaluation_counts"] == dict(
        NATIVE16=16, NATIVE64=64, FPS16of64=16
    )
    assert set(result["scored"]) == set(stat.ARMS) and not list(
        (tmp_path / "scratch").iterdir()
    )
    assert len((tmp_path / "pool-locks.jsonl").read_text().splitlines()) == 1
    assert (
        json.loads((tmp_path / "panels.jsonl").read_text())["native_results"][0][
            "feat412"
        ]
        == [0.0] * 412
    )


def test_smoke_refuses_non_train_before_proposal_or_payload(treatment, tmp_path):
    with pytest.raises(ValueError):
        treatment.measure(
            dict(phase="smoke"),
            dict(dl_split="DL-Val", task="regression"),
            stat.SEEDS[0],
            tmp_path,
        )


def test_ram_pause_requires_joint_resume_and_preserves_exact_fields(tmp_path):
    o = observation()
    o["ram"] = 11 * 2**30
    assert protocol.admission(o, now=0)
    fields = protocol.violations(o, now=0)[1]
    assert fields == [
        dict(field="ram", value=11 * 2**30, operator="<", threshold=12 * 2**30)
    ]
    protocol.record(tmp_path, "PAUSE_DISPATCH", o, fields)
    saved = json.loads((tmp_path / "supervisor-guard.jsonl").read_text())
    assert saved["violations"] == fields and saved["observation"] == o
    o["ram"] = 15 * 2**30
    assert protocol.admission(o, paused=True, now=0)
    o["ram"] = 16 * 2**30
    o["load5"] = 61
    assert protocol.admission(o, paused=True, now=0)
    o["load5"] = 60
    assert not protocol.admission(o, paused=True, now=0)
    o["ram"] = 12 * 2**30
    assert not protocol.admission(o, paused=False, now=0)


def test_hard_memory_stop_has_exact_value_and_is_distinct_from_pause():
    o = observation()
    o["ram"] = 6 * 2**30 - 1
    with pytest.raises(protocol.HardStop) as caught:
        protocol.admission(o, now=0)
    assert caught.value.fields == [
        dict(field="ram", value=6 * 2**30 - 1, operator="<", threshold=6 * 2**30)
    ]
    assert json.loads(str(caught.value))["observation"]["ram"] == o["ram"]


def test_supervisor_pressure_latch_carries_across_native_unit(tmp_path):
    o = observation()
    o["ram"] = 11 * 2**30
    paused, fields = protocol.shared_admission(tmp_path, o, now=0)
    assert paused and fields[0]["field"] == "ram"
    o["ram"] = 14 * 2**30
    assert protocol.shared_admission(tmp_path, o, now=0)[0]
    o["ram"] = 16 * 2**30
    o["load5"] = 61
    assert protocol.shared_admission(tmp_path, o, now=0)[0]
    o["load5"] = 60
    assert not protocol.shared_admission(tmp_path, o, now=0)[0]
