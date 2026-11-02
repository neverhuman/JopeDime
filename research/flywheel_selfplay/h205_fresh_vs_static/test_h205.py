"""Data-free unit tests for H205 (fresh vs static data & static teacher, 50 rounds from r882, H178 selector)."""

import copy
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
                family_aliases=[fam, f"dedupe:static_{i % 205:03d}"],
                dl_split="DL-Train",
                task="regression",
                has_meta_test=False,
                source_split="train",
            )
        )
    return rows


def _synthetic_valid_fresh_rows(n_rows=40, n_families=20):
    rows = []
    eval_bin_sha = hashlib.sha256(b"jopedime-q1-linux-native").hexdigest()
    for i in range(n_rows):
        fam = f"fresh_d4_fam_{i % n_families:03d}"
        ds_id = f"fresh_reg_ds_{i:04d}"
        cid = hashlib.sha256(f"fresh_ctx_{i}".encode()).hexdigest()
        train_sha = hashlib.sha256(f"fresh_train_{i}".encode()).hexdigest()
        fitness = np.linspace(0.8, 1.4, stat.SLATE_CANDIDATES, dtype=np.float64)
        gate = np.ones(stat.SLATE_CANDIDATES, dtype=bool)
        labels_sha = hashlib.sha256(fitness.tobytes() + gate.tobytes()).hexdigest()
        rows.append(
            dict(
                dataset_id=ds_id,
                family_id=fam,
                family_aliases=[fam, f"dedupe:fresh_{i % n_families:03d}"],
                task="regression",
                source_split="train",
                dl_split="DL-Train",
                has_meta_test=False,
                context_sha256=cid,
                train_sha256=train_sha,
                fitness=fitness.tolist(),
                gate=gate.tolist(),
                native_evaluator_receipt=dict(
                    status="PASS",
                    evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
                    evaluator_binary_sha256=eval_bin_sha,
                    labels_sha256=labels_sha,
                    dataset_id=ds_id,
                    context_sha256=cid,
                ),
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


def test_terminal_closeout_and_missing_panel_ledger_recorded():
    metrics = json.loads((HERE / "metrics.json").read_text())
    assert metrics["wave"] == "H205"
    assert metrics["status"] == "INCOMPLETE"
    assert (
        metrics["scientific_disposition"]
        == "EXPLORATORY_NONCONFORMING_HELD_AT_FRESH_BOUNDARY"
    )
    assert metrics["verdict"] == "NO_ADVANCE"
    assert metrics["optimizer_updates"] == 1500
    assert metrics["fresh_updates"] == 0

    limits = metrics["missing_evidence_limits"]
    assert limits["static_b_panel"] is None
    assert limits["static_a_panel"] is None
    assert limits["teacher_b_panel"] is None
    assert limits["teacher_a_panel"] is None
    assert limits["fresh_b_panel"] is None
    assert limits["fresh_a_panel"] is None
    assert limits["elo_ratings"] is None
    assert "not flushed to disk" in limits["missing_reason"]

    budget = metrics["cumulative_budget_ledger"]
    assert budget["budget_reset"] is False
    assert abs(budget["total_elapsed_wall_seconds"] - 1970.9372255061753) < 1e-6
    assert abs(budget["remaining_wall_budget_seconds"] - 23229.062774493825) < 1e-6

    custody = metrics["s3_terminal_custody"]
    assert (
        custody["terminal_custody_manifest_version_id"]
        == "BInrGnjKhZOFhuK5qS9ZOA.EcIsGFwAS"
    )
    assert (
        custody["terminal_custody_manifest_sha256"]
        == "f0860933a6f6a49051a755bf0260a5b59187f08717ccaf241ff4c78cce8e3186"
    )
    by_name = {m["name"]: m for m in custody["verified_members"]}
    assert (
        by_name["disposition_nonconforming_hold.json"]["version_id"]
        == "eR0i4LeBLsPubzVMAXLedPUw73jCI3DK"
    )
    assert by_name["progress.json"]["version_id"] == "N_yw8zfAtYWl8U3d_O8uAKYk06jY8i_4"
    for m in custody["verified_members"]:
        assert m["local_sha256"] == m["downloaded_version_sha256"]
        assert len(m["local_sha256"]) == 64


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


def test_counterexample_fake_fresh_from_old_205_rejected(tmp_path):
    """Counterexample 1: fake-fresh family IDs or aliases drawn from the old 205 static roster cannot pass."""
    static_rows = _synthetic_205_rows()
    valid_fresh = _synthetic_valid_fresh_rows()
    receipt = tmp_path / "admission.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="PASS",
                admitted_families=20,
                admitted_datasets=40,
                d4_overlap_count=0,
                sealed_overlap_count=0,
                manifest_sha256="c" * 64,
            )
        )
    )

    ok_bundle = stat.verify_and_load_fresh_d4_pool(
        receipt, static_rows, fresh_rows_override=valid_fresh
    )
    assert ok_bundle["status"] == "PASS"
    assert ok_bundle["admitted_families"] == 20

    # Fake-fresh case A: all family_ids drawn from old 205 static families
    fake_all_old = copy.deepcopy(valid_fresh)
    for i, r in enumerate(fake_all_old):
        r["family_id"] = f"fam_{i % 205:03d}"
    with pytest.raises(RuntimeError, match="fake-fresh violation"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=fake_all_old
        )

    # Fake-fresh case B: alias overlap with static 205 family_aliases
    fake_alias = copy.deepcopy(valid_fresh)
    fake_alias[0]["family_aliases"].append("dedupe:static_007")
    with pytest.raises(RuntimeError, match="fake-fresh violation"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=fake_alias
        )

    # Fake-fresh case C: schedule builder also refuses overlapping fresh_fit_rows
    fit_rows, _, _ = stat.partition(static_rows, 0)
    with pytest.raises(ValueError, match="strictly disjoint"):
        stat.build_equal_compute_round_plan(
            fit_rows, 205101, 0, fresh_fit_rows=fit_rows[:30]
        )


def test_counterexample_invalid_fresh_receipt_or_native_labels_rejected(tmp_path):
    """Counterexample 2: missing/wrong task, dataset, cut, hash, or native-evaluator receipt cannot create training labels."""
    static_rows = _synthetic_205_rows()
    base_fresh = _synthetic_valid_fresh_rows()
    receipt = tmp_path / "admission.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="PASS",
                admitted_families=20,
                admitted_datasets=40,
                d4_overlap_count=0,
                sealed_overlap_count=0,
                manifest_sha256="d" * 64,
            )
        )
    )

    # Wrong task (binary instead of regression)
    bad_task = copy.deepcopy(base_fresh)
    bad_task[0]["task"] = "binary"
    with pytest.raises(RuntimeError, match="invalid task"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_task
        )

    # Missing dataset_id
    bad_ds = copy.deepcopy(base_fresh)
    bad_ds[0]["dataset_id"] = ""
    with pytest.raises(RuntimeError, match="missing dataset_id"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_ds
        )

    # Wrong cut / meta-test flag
    bad_cut = copy.deepcopy(base_fresh)
    bad_cut[0]["has_meta_test"] = True
    with pytest.raises(RuntimeError, match="invalid cut/split"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_cut
        )

    # Invalid hash
    bad_hash = copy.deepcopy(base_fresh)
    bad_hash[0]["train_sha256"] = "not_a_64_hex_sha"
    with pytest.raises(RuntimeError, match="missing 64-hex"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_hash
        )

    # Wrong native evaluator identifier
    bad_eval = copy.deepcopy(base_fresh)
    bad_eval[0]["native_evaluator_receipt"]["evaluator"] = "synthetic_mock"
    with pytest.raises(RuntimeError, match="invalid native evaluator"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_eval
        )

    # Tampered labels_sha256
    bad_label_sha = copy.deepcopy(base_fresh)
    bad_label_sha[0]["fitness"][0] += 0.125
    with pytest.raises(RuntimeError, match="native labels_sha256 mismatch"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_label_sha
        )


def test_counterexample_favorable_mean_with_ci_crossing_zero_or_excess_harm_rejected():
    """Counterexample 3: favorable mean with 95% CI crossing 0 or excess family harm (>10%) cannot advance."""
    rows = _synthetic_205_rows()
    base = np.ones((3, len(rows)), dtype=np.float64)

    # Case A: Favorable mean gain (+0.008 > 0.001) with high family variance so 95% CI crosses 0 (lcb95 <= 0)
    ci_cross_zero = base.copy()
    for j, r in enumerate(rows):
        fam_idx = int(r["family_id"].split("_")[1])
        if fam_idx == 0:
            ci_cross_zero[:, j] += 2.20
        elif fam_idx < 18:
            ci_cross_zero[:, j] -= 0.009  # within -0.01 harm threshold so harm_rate==0
    res_ci = stat.admit_round_checkpoint_pooled(rows, base, ci_cross_zero)
    assert res_ci["gain"] > stat.MIN_GAIN_MARGIN
    assert res_ci["harm_rate"] == 0.0
    assert res_ci["lcb95"] <= 0.0
    assert res_ci["admit"] is False
    assert res_ci["decision"] == "REJECT_RETAIN_INCUMBENT"

    # Case B: Favorable mean gain and positive LCB95, but >10% of families harmed (>20/205 degraded by >0.01)
    excess_harm = base.copy()
    for j, r in enumerate(rows):
        fam_idx = int(r["family_id"].split("_")[1])
        if fam_idx < 25:
            excess_harm[:, j] -= 0.025
        else:
            excess_harm[:, j] += 0.04
    res_harm = stat.admit_round_checkpoint_pooled(rows, base, excess_harm)
    assert res_harm["gain"] > stat.MIN_GAIN_MARGIN
    assert res_harm["lcb95"] > 0.0
    assert res_harm["harm_rate"] > stat.MAX_HARM_FRACTION
    assert res_harm["admit"] is False
    assert res_harm["decision"] == "REJECT_RETAIN_INCUMBENT"


def test_counterexample_evaluation_or_sealed_family_alias_leakage_rejected(tmp_path):
    """Counterexample 4: evaluation family/alias or sealed D4 family/alias cannot enter training."""
    static_rows = _synthetic_205_rows()
    valid_fresh = _synthetic_valid_fresh_rows()
    receipt = tmp_path / "admission.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="PASS",
                admitted_families=20,
                admitted_datasets=40,
                d4_overlap_count=0,
                sealed_overlap_count=0,
                manifest_sha256="e" * 64,
            )
        )
    )

    eval_rows = [
        dict(
            context_sha256="f" * 64,
            family_id="dl_val_family_01",
            family_aliases=["dl_val_family_01", "dedupe:eval_secret_alias"],
        )
    ]
    leaked_eval = copy.deepcopy(valid_fresh)
    leaked_eval[3]["family_aliases"].append("dedupe:eval_secret_alias")
    with pytest.raises(RuntimeError, match="leakage violation"):
        stat.verify_and_load_fresh_d4_pool(
            receipt,
            static_rows,
            eval_rows=eval_rows,
            fresh_rows_override=leaked_eval,
        )

    leaked_sealed = copy.deepcopy(valid_fresh)
    leaked_sealed[5]["family_aliases"].append("sealed_d4_key_99")
    with pytest.raises(RuntimeError, match="leakage violation"):
        stat.verify_and_load_fresh_d4_pool(
            receipt,
            static_rows,
            sealed_family_keys=["sealed_d4_key_99"],
            fresh_rows_override=leaked_sealed,
        )


def test_counterexample_frozen_teacher_identity_and_shared_fresh_schedule():
    """Counterexample 5: frozen-teacher identity stays fixed and STATIC-TEACHER shares the exact FRESH schedule."""
    static_rows = _synthetic_205_rows()
    fresh_rows = _synthetic_valid_fresh_rows(n_rows=60, n_families=30)
    fit_rows, _, _ = stat.partition(static_rows, 0)
    fresh_fit = stat.partition_fresh_rows(fresh_rows, 0)

    plans = stat.build_equal_compute_round_plan(
        fit_rows, 205101, 0, fresh_fit_rows=fresh_fit
    )
    teacher_plan = plans["STATIC-TEACHER"]
    fresh_plan = plans["FRESH"]
    static_plan = plans["STATIC-DATA"]

    assert len(teacher_plan) == 50 and len(fresh_plan) == 50
    for r_t, r_f, r_s in zip(teacher_plan, fresh_plan, static_plan, strict=True):
        assert r_t["stream"] == r_f["stream"]
        assert r_t["microbatches"] == r_f["microbatches"]
        for mb_f, mb_s in zip(r_f["microbatches"], r_s["microbatches"], strict=True):
            assert set(mb_f["families"]).isdisjoint(set(mb_s["families"]))

    digest_a = "1" * 64
    digest_b = "2" * 64
    ok = stat.verify_frozen_teacher_identity(
        stat.R882_SHA256, stat.DOPE_SHA256, digest_a, digest_a
    )
    assert ok["immutable"] is True

    with pytest.raises(RuntimeError, match="parameter digest mutated"):
        stat.verify_frozen_teacher_identity(
            stat.R882_SHA256, stat.DOPE_SHA256, digest_a, digest_b
        )
    with pytest.raises(RuntimeError, match="training mode"):
        stat.verify_frozen_teacher_identity(
            stat.R882_SHA256,
            stat.DOPE_SHA256,
            digest_a,
            digest_a,
            teacher_training_mode=True,
        )
    with pytest.raises(RuntimeError, match="requires_grad=True"):
        stat.verify_frozen_teacher_identity(
            stat.R882_SHA256,
            stat.DOPE_SHA256,
            digest_a,
            digest_a,
            any_requires_grad=True,
        )


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
