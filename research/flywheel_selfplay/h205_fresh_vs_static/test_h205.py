"""Data-free unit tests for H205 (fresh vs static data & static teacher, 50 rounds from r882, H178 selector)."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import analysis as stat  # noqa: E402
import monitor as mon  # noqa: E402
import runner as run_mod  # noqa: E402


def _synthetic_205_rows():
    rows = []
    for i in range(1219):
        fam = f"fam_{i % 205:03d}"
        cid = hashlib.sha256(f"ctx_{i}".encode()).hexdigest()
        rows.append(
            dict(
                context_sha256=cid,
                family_id=fam,
                dl_split="DL-Train",
                task="regression",
                has_meta_test=False,
                source_split="train",
            )
        )
    return rows


def test_prereg_and_manifest_authority_and_wave_separation():
    prereg_text = (HERE / "PREREG.md").read_text()
    manifest = json.loads((HERE / "manifest.json").read_text())
    assert stat.AUTHORITY_STRING in prereg_text
    assert manifest["authority"] == stat.AUTHORITY_STRING
    assert manifest["wave"] == "H205"
    assert manifest["rounds"] == 50
    assert manifest["arms"] == ["R882", "STATIC-DATA", "STATIC-TEACHER", "FRESH"]
    assert manifest["r882_sha256"] == stat.R882_SHA256
    assert manifest["dope_sha256"] == stat.DOPE_SHA256
    for w in ("H204", "H205", "H206", "H207"):
        assert w in prereg_text


def test_thread_rule_b_distinguishes_numeric_from_cuda_helper_threads():
    classified = mon.classify_thread_names(
        [
            "python3",
            "python3",
            "cuda-EvtHandlr",
            "cuda00001400006",
            "pt_autograd_0",
            "nv_queue",
            "cudnn_worker",
            "cublas_helper",
        ]
    )
    assert classified["total_threads"] == 8
    assert classified["numeric_threads"] == 2
    assert classified["cuda_helper_threads"] == 6

    base_sample = dict(
        load=10.0,
        cpu=25.0,
        available=32 * 2**30,
        free=100_000_000_000,
        disk_used_fraction=0.91,
        rss=2_000_000_000,
        threads=8,
        numeric_threads=2,
        cuda_helper_threads=6,
        numeric_env_violations=[],
        nice=19,
    )
    assert mon.violations(base_sample) == []

    oversub_sample = dict(base_sample, numeric_threads=5, threads=11)
    assert "numeric threads>4" in mon.violations(oversub_sample)

    env_bad = mon.check_numeric_env_and_torch(
        env={"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "1"},
        torch_intra_op=2,
        torch_inter_op=1,
    )
    assert env_bad
    assert "numeric thread env/torch>4" in mon.violations(
        dict(base_sample, numeric_env_violations=env_bad)
    )

    assert "host CPU>85%" in mon.violations(dict(base_sample, cpu=86.0))
    assert "disk used>93.8%" in mon.violations(
        dict(base_sample, disk_used_fraction=0.939)
    )
    assert "disk free<40GB" in mon.violations(dict(base_sample, free=39_000_000_000))
    assert "aggregate RSS>=6GB" in mon.violations(dict(base_sample, rss=6_000_000_000))
    assert "nice!=19" in mon.violations(dict(base_sample, nice=0))


def test_equal_compute_50_round_schedule_and_205_family_folds():
    rows = _synthetic_205_rows()
    cov = stat.verify_pooled_b_coverage(rows)
    assert cov["total_families"] == 205
    assert cov["b_exact_once"] and cov["a_exact_once"]
    assert cov["zero_fit_b_overlap"] and cov["zero_b_a_same_fold_overlap"]

    fit_rows, b_rows, a_rows = stat.partition(rows, 0)
    plan_by_arm = stat.build_equal_compute_round_plan(fit_rows, 205101, 0)
    assert set(plan_by_arm) == set(stat.ARMS)
    for arm in stat.ARMS:
        rounds_plan = plan_by_arm[arm]
        assert len(rounds_plan) == 50
        for r_info in rounds_plan:
            assert len(r_info["microbatches"]) == 16
            for mb in r_info["microbatches"]:
                assert len(mb["families"]) == 8
                assert len(set(mb["families"])) == 8
                assert len(mb["cids"]) == 8


def test_r882_native_parity_gate_91_exact():
    ref = np.zeros(91, dtype=np.float64)
    ref[0] = stat.R882_NATIVE_DLVAL_MEAN * 91.0
    assert float(ref.mean()) == stat.R882_NATIVE_DLVAL_MEAN
    res = stat.verify_r882_native_parity(ref.copy(), ref)
    assert res["status"] == "PASS" and res["bit_exact"]

    perturbed = ref.copy()
    perturbed[0] += 1e-12
    with pytest.raises(RuntimeError, match="91-context exact native r882"):
        stat.verify_r882_native_parity(perturbed, ref)

    with pytest.raises(RuntimeError, match="91-context exact native r882"):
        stat.verify_r882_native_parity(ref[:90], ref)


def test_h178_per_round_pooled_selector_and_harm_cap():
    rows = _synthetic_205_rows()
    base = np.ones((3, len(rows)), dtype=np.float64)
    good_fresh = base + 0.05
    static_data = base + 0.005
    harmful_teacher = base.copy()
    # Degrade 30 out of 205 families (> 10% cap) by -0.05 while lifting others by +0.08
    harmed_fams = {f"fam_{i:03d}" for i in range(30)}
    for j, r in enumerate(rows):
        if r["family_id"] in harmed_fams:
            harmful_teacher[:, j] -= 0.05
        else:
            harmful_teacher[:, j] += 0.08

    sel = stat.h178_round_select(
        rows,
        base,
        {
            "STATIC-DATA": static_data,
            "STATIC-TEACHER": harmful_teacher,
            "FRESH": good_fresh,
        },
    )
    assert sel["h177_rule_verified"]
    assert sel["decisions"]["FRESH"]["admit"] is True
    assert sel["decisions"]["FRESH"]["decision"] == "ADMIT"
    assert sel["decisions"]["STATIC-TEACHER"]["harm_rate"] > 0.10
    assert sel["decisions"]["STATIC-TEACHER"]["admit"] is False
    assert sel["decisions"]["STATIC-TEACHER"]["decision"] == "REJECT_RETAIN_INCUMBENT"


def test_fresh_arm_gated_on_jrb01_admission_pass(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="missing"):
        stat.verify_fresh_admission_gate(missing)

    bad_overlap = tmp_path / "bad.json"
    bad_overlap.write_text(
        json.dumps(
            dict(
                status="PASS",
                admitted_families=2510,
                d4_overlap_count=1,
                sealed_overlap_count=0,
                manifest_sha256="a" * 64,
            )
        )
    )
    with pytest.raises(RuntimeError, match="non-zero D4 overlap"):
        stat.verify_fresh_admission_gate(bad_overlap)

    valid = tmp_path / "pass.json"
    valid.write_text(
        json.dumps(
            dict(
                status="PASS",
                admitted_families=2510,
                admitted_datasets=3100,
                d4_overlap_count=0,
                sealed_overlap_count=0,
                manifest_sha256="b" * 64,
            )
        )
    )
    gate = stat.verify_fresh_admission_gate(valid)
    assert gate["status"] == "PASS"
    assert gate["admitted_families"] == 2510


def test_runner_smoke_mode(tmp_path):
    out_dir = tmp_path / "smoke_out"
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps(
            dict(
                wave="H205",
                authority=stat.AUTHORITY_STRING,
                manifest_sha256=run_mod.sha(HERE / "manifest.json"),
                numeric_threads=2,
                cuda_runtime_threads_exempt=True,
                live_flywheel_pid_untouched=2297929,
            )
        )
    )
    env = dict(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        PATH=subprocess.os.environ.get("PATH", ""),
    )
    subprocess.run(
        [
            sys.executable,
            str(HERE / "runner.py"),
            "--output",
            str(out_dir),
            "--launch",
            str(launch_path),
            "--smoke",
        ],
        check=True,
        env=env,
    )
    smoke = json.loads((out_dir / "smoke.json").read_text())
    assert smoke["status"] == "PASS"
    assert smoke["authority"] == stat.AUTHORITY_STRING
    assert smoke["optimizer_updates"] == 0
    assert smoke["numeric_threads"] <= 4
