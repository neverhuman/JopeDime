"""Data-free unit tests for H205 (fresh vs static data & static teacher, 50 rounds from r882, H178 selector)."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

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
        fitness = np.linspace(
            0.8 + 0.001 * i, 1.4 + 0.001 * i, stat.SLATE_CANDIDATES, dtype=np.float64
        )
        gate = np.ones(stat.SLATE_CANDIDATES, dtype=bool)
        program = np.full(
            (stat.SLATE_CANDIDATES, stat.PROGRAM_DIM),
            fill_value=0.01 * (i + 1),
            dtype=np.float32,
        )
        tokens_10x32 = np.full(
            (stat.TOKEN_STEPS, stat.TOKEN_DIM),
            fill_value=0.05 * (i + 1),
            dtype=np.float32,
        )
        mask_10 = np.ones(stat.TOKEN_STEPS, dtype=bool)

        labels_sha = hashlib.sha256(fitness.tobytes() + gate.tobytes()).hexdigest()
        programs_sha = hashlib.sha256(program.tobytes()).hexdigest()
        features_sha = hashlib.sha256(
            tokens_10x32.tobytes() + mask_10.tobytes()
        ).hexdigest()

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
                program=program.tolist(),
                tokens_10x32=tokens_10x32.tolist(),
                mask_10=mask_10.tolist(),
                native_evaluator_receipt=dict(
                    status="PASS",
                    evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
                    evaluator_binary_sha256=eval_bin_sha,
                    labels_sha256=labels_sha,
                    programs_sha256=programs_sha,
                    features_sha256=features_sha,
                    dataset_id=ds_id,
                    context_sha256=cid,
                ),
            )
        )
    return rows


class _MockInheritedEngine:
    """Reproduces inherited h170 Engine.batch / tokens / masks / pools structure on CPU."""

    def __init__(self, historical_cids):
        self.projection = np.eye(stat.PROGRAM_DIM, 16, dtype=np.float64) / np.sqrt(
            float(stat.PROGRAM_DIM)
        )
        self.pools = {}
        self.tokens = {
            c: torch.zeros((stat.TOKEN_STEPS, stat.TOKEN_DIM), dtype=torch.float32)
            for c in historical_cids
        }
        self.masks = {
            c: torch.ones((stat.TOKEN_STEPS,), dtype=torch.bool)
            for c in historical_cids
        }

    def batch(self, cids, indexes):
        b, k = len(cids), len(indexes[0])

        def stack(key):
            return np.stack(
                [self.pools[c][key][ix] for c, ix in zip(cids, indexes, strict=True)]
            )

        prog = torch.as_tensor(stack("program"), dtype=torch.float32)
        proj = torch.as_tensor(stack("proj16"), dtype=torch.float32)
        fitness = torch.as_tensor(stack("fitness"), dtype=torch.float32)
        gate = torch.as_tensor(stack("gate"), dtype=torch.bool)
        kwargs = dict(
            ctx_feat=torch.zeros((b, 584)),
            prog_412=prog,
            proj_16=proj,
            base_prior_cost=torch.zeros((b, k)),
            scale=torch.full((b,), 0.4208),
            sos_prior=torch.zeros((b, k, 2)),
            revealed_indices=torch.arange(min(4, k)).reshape(1, -1).expand(b, -1),
            revealed_raw_f=fitness[:, : min(4, k)],
            revealed_gate=gate[:, : min(4, k)].float(),
        )
        return (
            kwargs,
            torch.stack([self.tokens[c] for c in cids]),
            torch.stack([self.masks[c] for c in cids]),
            fitness,
            gate,
        )

    def evaluate(self, rows, native=False):
        vals = np.array(
            [float(np.mean(self.pools[r["context_sha256"]]["fitness"])) for r in rows],
            dtype=np.float64,
        )
        raw = {
            r["context_sha256"]: -np.asarray(
                self.pools[r["context_sha256"]]["fitness"], dtype=np.float64
            )
            for r in rows
        }
        return dict(values=vals), raw


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

    bad_task = copy.deepcopy(base_fresh)
    bad_task[0]["task"] = "binary"
    with pytest.raises(RuntimeError, match="invalid task"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_task
        )

    bad_ds = copy.deepcopy(base_fresh)
    bad_ds[0]["dataset_id"] = ""
    with pytest.raises(RuntimeError, match="missing dataset_id"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_ds
        )

    bad_cut = copy.deepcopy(base_fresh)
    bad_cut[0]["has_meta_test"] = True
    with pytest.raises(RuntimeError, match="invalid cut/split"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_cut
        )

    bad_hash = copy.deepcopy(base_fresh)
    bad_hash[0]["train_sha256"] = "not_a_64_hex_sha"
    with pytest.raises(RuntimeError, match="missing 64-hex"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_hash
        )

    bad_eval = copy.deepcopy(base_fresh)
    bad_eval[0]["native_evaluator_receipt"]["evaluator"] = "synthetic_mock"
    with pytest.raises(RuntimeError, match="invalid native evaluator"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_eval
        )

    bad_label_sha = copy.deepcopy(base_fresh)
    bad_label_sha[0]["fitness"][0] += 0.125
    with pytest.raises(RuntimeError, match="native labels_sha256 mismatch"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_label_sha
        )

    bad_prog_sha = copy.deepcopy(base_fresh)
    bad_prog_sha[0]["program"][0][0] += 1.0
    with pytest.raises(RuntimeError, match="native programs_sha256 mismatch"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_prog_sha
        )

    bad_feat_sha = copy.deepcopy(base_fresh)
    bad_feat_sha[0]["tokens_10x32"][0][0] += 1.0
    with pytest.raises(RuntimeError, match="native features_sha256 mismatch"):
        stat.verify_and_load_fresh_d4_pool(
            receipt, static_rows, fresh_rows_override=bad_feat_sha
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


def test_fresh_engine_ingestion_and_disconnected_npz_rejection(tmp_path):
    """Counterexample 6 (review-a004012ce P1-1 & P1-2):
    - Disjoint fresh contexts ingested into engine succeed through batch, fit_slates, teacher ordering, and treatment.
    - A hash-valid NPZ whose fitness/program differs from the verified receipt/row is rejected.
    - Historical cohort cache closure (1,310 contexts) is preserved and collisions are rejected.
    """
    static_rows = _synthetic_205_rows()
    val_cids = [hashlib.sha256(f"val_{i}".encode()).hexdigest() for i in range(91)]
    historical_cids = [r["context_sha256"] for r in static_rows] + val_cids

    engine = _MockInheritedEngine(historical_cids)
    fresh_rows = _synthetic_valid_fresh_rows(n_rows=40, n_families=20)
    first_cid = fresh_rows[0]["context_sha256"]

    # Before ingestion, disjoint fresh CID raises KeyError in engine.batch
    with pytest.raises(KeyError):
        engine.batch([first_cid], [np.arange(stat.SLATE_CANDIDATES)])

    # Write a valid NPZ for row 0 and bind pool_npz_path + pool_sha256
    valid_npz = tmp_path / f"{first_cid}.npz"
    np.savez(
        valid_npz,
        program=np.asarray(fresh_rows[0]["program"], dtype=np.float32),
        fitness=np.asarray(fresh_rows[0]["fitness"], dtype=np.float64),
        gate=np.asarray(fresh_rows[0]["gate"], dtype=bool),
        tokens_10x32=np.asarray(fresh_rows[0]["tokens_10x32"], dtype=np.float32),
        mask_10=np.asarray(fresh_rows[0]["mask_10"], dtype=bool),
    )
    fresh_rows[0]["pool_npz_path"] = str(valid_npz)
    fresh_rows[0]["pool_sha256"] = run_mod.sha(valid_npz)

    ing = run_mod.ingest_verified_fresh_rows_into_engine(
        engine, fresh_rows, historical_cids=historical_cids
    )
    assert ing["status"] == "PASS"
    assert ing["ingested_contexts"] == 40
    assert ing["historical_contexts_preserved"] == 1310

    # Exercise fresh teacher ordering, fit_slates, and treatment batching on ingested fresh rows
    _, fresh_teacher_raw = engine.evaluate(fresh_rows, native=False)
    fresh_teacher_orders = {
        cid: np.argsort(cost, kind="mergesort")
        for cid, cost in fresh_teacher_raw.items()
    }
    fresh_gt_orders = {
        r["context_sha256"]: np.argsort(
            np.where(
                engine.pools[r["context_sha256"]]["gate"],
                engine.pools[r["context_sha256"]]["fitness"],
                2.0,
            ),
            kind="mergesort",
        )
        for r in fresh_rows
    }
    fresh_fit = stat.partition_fresh_rows(fresh_rows, 0)
    slates_t = stat.h177_analysis.fit_slates(fresh_fit, fresh_teacher_orders, 205101, 0)
    slates_f = stat.h177_analysis.fit_slates(fresh_fit, fresh_gt_orders, 205101, 0)
    sample_cids = sorted(slates_f.keys())[:8]
    kwargs, tok, mask, fit_t, gate_t = engine.batch(
        sample_cids, [slates_f[c] for c in sample_cids]
    )
    assert tok.shape == (8, stat.TOKEN_STEPS, stat.TOKEN_DIM)
    assert mask.shape == (8, stat.TOKEN_STEPS)
    assert fit_t.shape == (8, stat.SLATE_CANDIDATES)
    assert gate_t.shape == (8, stat.SLATE_CANDIDATES)
    assert set(slates_t.keys()) == set(slates_f.keys())

    # Counterexample: NPZ with valid file hash but disconnected fitness ([0, -1, ...] vs verified row) is rejected
    bad_rows = copy.deepcopy(fresh_rows)
    bad_npz = tmp_path / "disconnected.npz"
    disconnected_fitness = np.asarray(bad_rows[1]["fitness"], dtype=np.float64)
    disconnected_fitness[0] = 0.0
    disconnected_fitness[1] = -1.0
    np.savez(
        bad_npz,
        program=np.asarray(bad_rows[1]["program"], dtype=np.float32),
        fitness=disconnected_fitness,
        gate=np.asarray(bad_rows[1]["gate"], dtype=bool),
        tokens_10x32=np.asarray(bad_rows[1]["tokens_10x32"], dtype=np.float32),
        mask_10=np.asarray(bad_rows[1]["mask_10"], dtype=bool),
    )
    bad_rows[1]["pool_npz_path"] = str(bad_npz)
    bad_rows[1]["pool_sha256"] = run_mod.sha(bad_npz)
    with pytest.raises(RuntimeError, match="mismatch between receipt/row"):
        run_mod.ingest_verified_fresh_rows_into_engine(
            _MockInheritedEngine(historical_cids),
            bad_rows,
            historical_cids=historical_cids,
        )

    # Also test NPZ-only row (inline fitness omitted) where NPZ fitness disagrees with receipt labels_sha256
    bad_npz_only = copy.deepcopy(bad_rows)
    del bad_npz_only[1]["fitness"]
    with pytest.raises(RuntimeError, match="native labels_sha256 mismatch"):
        run_mod.ingest_verified_fresh_rows_into_engine(
            _MockInheritedEngine(historical_cids),
            bad_npz_only,
            historical_cids=historical_cids,
        )


def test_stage_order_b_lock_persisted_before_heldout_a_evaluation(tmp_path):
    """Counterexample 7 (review-a004012ce P2): held-out A evaluation is refused unless pooled_b_selection_lock.json
    is persisted on disk with matching per-cell checkpoint parameter digests.
    """
    static_rows = _synthetic_205_rows()
    historical_cids = [r["context_sha256"] for r in static_rows]
    engine = _MockInheritedEngine(historical_cids)
    for i, r in enumerate(static_rows):
        cid = r["context_sha256"]
        engine.pools[cid] = dict(
            program=np.zeros(
                (stat.SLATE_CANDIDATES, stat.PROGRAM_DIM), dtype=np.float32
            ),
            proj16=np.zeros((stat.SLATE_CANDIDATES, 16), dtype=np.float32),
            fitness=np.full(
                stat.SLATE_CANDIDATES, 1.0 + 0.001 * (i % 7), dtype=np.float64
            ),
            gate=np.ones(stat.SLATE_CANDIDATES, dtype=bool),
        )

    cid_to_idx = {r["context_sha256"]: i for i, r in enumerate(static_rows)}
    r882_train_values = np.ones(len(static_rows), dtype=np.float64)
    cells = {}
    arm_digests = {}
    for si, seed in enumerate(stat.SEEDS):
        for outer in range(5):
            _, _, a_rows = stat.partition(static_rows, outer)
            state = {
                "w": torch.tensor(
                    [float(si + 1), float(outer + 1)], dtype=torch.float32
                )
            }
            cells[(si, outer)] = dict(
                seed=seed,
                outer=outer,
                a_rows=a_rows,
                a_indices=[cid_to_idx[r["context_sha256"]] for r in a_rows],
                incumbent_state=state,
            )
            arm_digests[f"seed_{seed}_outer_{outer}"] = run_mod.parameter_digest(state)

    expected_digests = {
        "STATIC-DATA": arm_digests,
        "STATIC-TEACHER": arm_digests,
        "FRESH": arm_digests,
    }
    lock_path = tmp_path / "pooled_b_selection_lock.json"

    # Calling evaluate_heldout_a_panel BEFORE pooled_b_selection_lock.json exists on disk is refused
    with pytest.raises(RuntimeError, match="stage-order violation"):
        run_mod.evaluate_heldout_a_panel(
            engine,
            lambda _s: None,
            cells,
            r882_train_values,
            lock_path,
            expected_digests,
            "FRESH",
        )

    # Persist pooled_b_selection_lock.json and verify held-out A evaluation succeeds
    run_mod.save(
        lock_path,
        dict(
            wave="H205",
            decisions={"FRESH": {"admit": False}},
            selected_checkpoint_digests=expected_digests,
        ),
    )
    a_panel = run_mod.evaluate_heldout_a_panel(
        engine,
        lambda _s: None,
        cells,
        r882_train_values,
        lock_path,
        expected_digests,
        "FRESH",
    )
    assert a_panel.shape == (3, len(static_rows))
    assert np.isfinite(a_panel).all()

    # Mutating a cell's incumbent_state after the lock is persisted is refused
    cells[(0, 0)]["incumbent_state"]["w"] = torch.tensor(
        [99.0, 99.0], dtype=torch.float32
    )
    with pytest.raises(
        RuntimeError, match="incumbent digest differs from persisted B lock"
    ):
        run_mod.evaluate_heldout_a_panel(
            engine,
            lambda _s: None,
            cells,
            r882_train_values,
            lock_path,
            expected_digests,
            "FRESH",
        )


def test_cumulative_budget_shared_deadline_and_preconsumed_refusal():
    """Counterexample 6 (1723-budget-counterexample.json): monitor AST predicate, shared process-start deadline, and pre-consumed budget refusal."""
    import ast

    # 1. Verify stdlib AST predicate in monitor.py directly stops at sample["elapsed"] = 23230.062774493825
    # (where prior_attempt_seconds = 1970.9372255061753 gives cumulative 25201.0s > 25200.0s).
    mon_src = (HERE / "monitor.py").read_text(encoding="utf-8")
    tree = ast.parse(mon_src)
    elapsed_if_tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            src_seg = ast.get_source_segment(mon_src, node.test) or ""
            if 'sample["elapsed"]' in src_seg and "23229.062774493825" in src_seg:
                elapsed_if_tests.append((node.lineno, node.test, src_seg))
    assert elapsed_if_tests, (
        'monitor.py must contain explicit AST predicate `sample["elapsed"] > 23229.062774493825`'
    )
    _, test_node, src_seg = elapsed_if_tests[0]
    assert src_seg.strip() == 'sample["elapsed"] > 23229.062774493825'
    code_obj = compile(ast.Expression(test_node), "<monitor_ast>", "eval")
    assert (
        eval(
            code_obj,
            {"__builtins__": {}},
            {"sample": {"elapsed": 23230.062774493825}},
        )
        is True
    )
    assert (
        eval(
            code_obj,
            {"__builtins__": {}},
            {"sample": {"elapsed": 23229.0}},
        )
        is False
    )

    # 2. Verify shared process-start deadline binds monitor, runner, and JRB-01 wait to ONE deadline
    deadline = mon.compute_shared_process_deadline(process_start_monotonic=1000.0)
    assert np.isclose(
        deadline["remaining_budget_seconds"],
        23229.062774493825,
        rtol=0.0,
        atol=1e-9,
    )
    assert np.isclose(
        deadline["hard_deadline_monotonic"],
        1000.0 + 23229.062774493825,
        rtol=0.0,
        atol=1e-9,
    )
    ok_check = mon.enforce_shared_process_deadline(
        1000.0 + 1800.0, deadline, stage="stage1_static_data"
    )
    assert np.isclose(
        ok_check["remaining_seconds"],
        23229.062774493825 - 1800.0,
        rtol=0.0,
        atol=1e-9,
    )
    with pytest.raises(RuntimeError, match="cumulative 7h hard wall deadline exceeded"):
        mon.enforce_shared_process_deadline(
            1000.0 + 23230.062774493825,
            deadline,
            stage="stage2_wait_jrb01",
        )

    # 3. Verify pre-consumed budget refusal in monitor and runner.validate_launch
    with pytest.raises(RuntimeError, match="prelaunch cumulative budget refusal"):
        mon.compute_remaining_wall_budget_seconds(
            prior_elapsed_seconds=25200.0, campaign_cap_seconds=25200.0
        )
    with pytest.raises(RuntimeError, match="prelaunch cumulative budget refusal"):
        mon.compute_shared_process_deadline(
            process_start_monotonic=500.0,
            prior_elapsed_seconds=25201.0,
            campaign_cap_seconds=25200.0,
        )
    with pytest.raises(RuntimeError, match="cannot reduce prior_elapsed_seconds"):
        mon.compute_remaining_wall_budget_seconds(prior_elapsed_seconds=0.0)
    with pytest.raises(RuntimeError, match="cannot increase campaign_cap_seconds"):
        mon.compute_remaining_wall_budget_seconds(campaign_cap_seconds=30000.0)

    manifest_sha = run_mod.sha(HERE / "manifest.json")
    base_launch = dict(
        wave="H205",
        authority=stat.AUTHORITY_STRING,
        manifest_sha256=manifest_sha,
        numeric_threads=2,
        cuda_runtime_threads_exempt=True,
        live_flywheel_pid_untouched=2297929,
    )
    with pytest.raises(RuntimeError, match="prelaunch cumulative budget refusal"):
        run_mod.validate_launch(
            dict(base_launch, prior_attempts_elapsed_seconds=25200.0),
            manifest_sha,
            process_start_monotonic=100.0,
        )
    with pytest.raises(RuntimeError, match="prelaunch cumulative budget refusal"):
        run_mod.validate_launch(
            dict(base_launch, remaining_wall_budget_seconds=0.0),
            manifest_sha,
            process_start_monotonic=100.0,
        )
    with pytest.raises(RuntimeError, match="fresh_wait_seconds"):
        run_mod.validate_launch(
            dict(base_launch, fresh_wait_seconds=24000.0),
            manifest_sha,
            process_start_monotonic=100.0,
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
