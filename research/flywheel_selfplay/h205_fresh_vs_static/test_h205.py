"""Data-free unit tests for H205 (fresh vs static data & static teacher, 50 rounds from r882, H178 selector)."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _run_h205_deadline_controls(scratch):
    """Execute only the committed stdlib/fake-clock tests, before research imports."""
    import ast
    import hashlib
    import inspect
    import itertools
    import json
    from pathlib import Path
    import tempfile
    import sys

    import pytest

    source = Path(__file__).resolve()
    tree = ast.parse(source.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and (n.name.startswith("_h205_f5_") or n.name.startswith("test_h205_f5_"))
    ]
    scope = {
        "__file__": str(source),
        "Path": Path,
        "json": json,
        "hashlib": hashlib,
        "pytest": pytest,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    results = []
    for name, function in scope.items():
        if not name.startswith("test_h205_f5_"):
            continue
        arguments = []
        for mark in getattr(function, "pytestmark", []):
            if mark.name == "parametrize":
                names = [value.strip() for value in mark.args[0].split(",")]
                arguments.append(
                    [
                        dict(zip(names, (value,) if len(names) == 1 else value))
                        for value in mark.args[1]
                    ]
                )
        for parameter_sets in itertools.product(*arguments):
            params = {
                key: value for group in parameter_sets for key, value in group.items()
            }
            with tempfile.TemporaryDirectory(
                prefix="h205-f5-", dir=scratch
            ) as directory:
                if "tmp_path" in inspect.signature(function).parameters:
                    params["tmp_path"] = Path(directory)
                try:
                    function(**params)
                    results.append({"test": name, "status": "PASS"})
                except KeyboardInterrupt:
                    raise
                except BaseException as exc:
                    results.append(
                        {
                            "test": name,
                            "status": "FAIL",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    passed = sum(row["status"] == "PASS" for row in results)
    prohibited = [
        name
        for name in sys.modules
        if name.split(".")[0] in {"torch", "numpy", "analysis", "runner", "psutil"}
    ]
    if prohibited:
        results.append(
            {
                "test": "prohibited_import_guard",
                "status": "FAIL",
                "error": str(prohibited),
            }
        )
        passed = sum(row["status"] == "PASS" for row in results)
    print(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "scope": "AST-only committed fake-clock controls; no research/model/native/S3 imports or calls",
                "prohibited_imports": prohibited,
                "passed": passed,
                "failed": len(results) - passed,
                "cases": results,
            },
            indent=2,
        )
    )
    return 0 if results and passed == len(results) else 1


if __name__ == "__main__":
    import sys as _deadline_control_sys

    if (
        len(_deadline_control_sys.argv) == 3
        and _deadline_control_sys.argv[1] == "--deadline-controls"
    ):
        raise SystemExit(_run_h205_deadline_controls(_deadline_control_sys.argv[2]))


import numpy as np  # noqa: E402 -- standalone controls exit before research imports
import pytest  # noqa: E402
import torch  # noqa: E402

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
    bridge_proto = manifest["train_padding_bridge_protocol"]
    assert bridge_proto["schema_version"] == stat.TRAIN_PADDING_BRIDGE_SCHEMA
    assert bridge_proto["query_padding_rule"] == stat.TRAIN_PADDING_QUERY_RULE
    assert bridge_proto["protocol_sha256"] == stat.TRAIN_PADDING_PROTOCOL_SHA256
    assert bridge_proto["evaluator_binary_sha256"] == stat.EVALUATOR_BINARY_SHA256
    assert bridge_proto["native_rust_source_sha256"] == stat.NATIVE_RUST_SOURCE_SHA256
    assert bridge_proto["metric_source_sha256"] == stat.METRIC_EVALUATOR_SOURCE_SHA256
    assert (
        bridge_proto["original_jtt1_nq0_sha256_1db3750d4ceccab9"]
        == stat.TARGET_REGRESSION_NQ0_JTT1_SHA256
    )
    assert (
        bridge_proto["padded_jtt1_nq10_sha256_1db3750d4ceccab9"]
        == stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256
    )
    assert (
        bridge_proto["target_same_s3_train_context_sha256_1db3750d4ceccab9"]
        == stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256
    )
    assert bridge_proto["padding_examples_added_to_learner"] == 0
    assert bridge_proto["padding_families_added"] == 0
    for w in ("H204", "H205", "H206", "H207"):
        assert w in prereg_text
    assert stat.TRAIN_PADDING_PROTOCOL_SHA256 in prereg_text
    assert stat.H206_NATIVE_FIXTURE_RECEIPT_SHA256 in prereg_text
    assert stat.H206_SCIENTIFIC_REPLAY_RECEIPT_SHA256 in prereg_text


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
    assert abs(budget["gpu_attempts_1_2_elapsed_seconds"] - 1970.9372255061753) < 1e-6
    assert abs(budget["regression_seam_cpu_elapsed_seconds"] - 50.10207135695964) < 1e-6
    assert abs(budget["total_elapsed_wall_seconds"] - 2021.039296863135) < 1e-6
    assert abs(budget["remaining_wall_budget_seconds"] - 23178.960703136865) < 1e-6

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

    seam = metrics["single_regression_context_milestone"]
    assert seam["status"] == "HISTORICAL_LOCAL_S1PROD_REPLAY_CONTROL"
    assert seam["seam_qualified"] is False
    assert seam["same_s3_train_seam_status"] == "BLOCKED_NQ0_PROTOCOL"
    assert seam["causal_h205_claim_status"] == "NOT_QUALIFIED_NO_ADVANCE"
    assert seam["dataset_id"] == "1db3750d4ceccab9"
    assert (
        seam["context_sha256"]
        == "fa480de30f3fba6fda1d7e9a1df65019acab3de752ded46b96448d0848f86d45"
    )
    assert seam["fpool_disjoint_datasets_count"] == 1772
    assert seam["transitive_disjoint_datasets_count"] == 1550
    assert seam["allowed_roster_datasets"] == 1336
    assert seam["allowed_roster_families"] == 1336
    assert (
        seam["roster_sha256"]
        == "c7ea7148129937d516f19a87c05c771b34d250f10ce9a2e53931e8f13760a504"
    )
    nq0_chk = seam["pure_s3_train_csv_nq0_protocol_check"]
    assert nq0_chk["nq0_refused_by_header_guard"] is True
    assert nq0_chk["nt"] == 1000
    assert nq0_chk["has_header"] is False
    assert (
        nq0_chk["jtt1_nq0_sha256"]
        == "a6d997e39cec56995f579c380a0a1e960d3e887aebb5d40b8a0aae88f613483c"
    )
    assert (
        nq0_chk["train_matrix_sha256"]
        == "cb04fa33aed43118c9c15a794bfac1324e049d56b3833adef56913d26583dff7"
    )
    bridge_rec = seam["prospective_train_padding_bridge"]
    assert bridge_rec["schema_version"] == stat.TRAIN_PADDING_BRIDGE_SCHEMA
    assert bridge_rec["protocol_sha256"] == stat.TRAIN_PADDING_PROTOCOL_SHA256
    assert (
        bridge_rec["original_jtt1_nq0_sha256"] == stat.TARGET_REGRESSION_NQ0_JTT1_SHA256
    )
    assert (
        bridge_rec["padded_jtt1_nq10_sha256"]
        == stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256
    )
    assert (
        bridge_rec["original_jtt1_nq0_sha256"] != bridge_rec["padded_jtt1_nq10_sha256"]
    )
    assert (
        bridge_rec["target_same_s3_train_context_sha256"]
        == stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256
    )
    assert bridge_rec["padding_examples_added_to_learner"] == 0
    assert bridge_rec["padding_families_added"] == 0
    assert bridge_rec["h206_readiness_claimed_from_tokens_alone"] is False
    assert (
        bridge_rec["h206_measured_fixture_evidence"]["real_TRAIN_or_r882_qualification"]
        is False
    )
    assert seam["s1_prod_context_bundle"]["outer_test_rows_decoded"] == 0
    assert (
        seam["engine_consumption"]["s3_persisted_and_replayed_before_optimizer_step"]
        is False
    )
    assert (
        seam["engine_consumption"]["adapter_sha256_before"]
        != seam["engine_consumption"]["adapter_sha256_after"]
    )
    assert (
        seam["engine_consumption"]["dope_sha256_before"]
        == seam["engine_consumption"]["dope_sha256_after"]
    )
    for m in seam["s3_bundle_custody"]["verified_members"]:
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
    """Counterexample 6 (1723-budget-counterexample.json + review1758 Blocker 5): monitor AST predicate, shared process-start deadline, and pre-consumed budget refusal."""
    import ast

    # 1. Verify stdlib AST predicate in monitor.py directly stops at sample["elapsed"] = 23179.960703136865
    # (where prior_attempt_seconds = 2021.039296863135 gives cumulative 25201.0s > 25200.0s).
    mon_src = (HERE / "monitor.py").read_text(encoding="utf-8")
    tree = ast.parse(mon_src)
    elapsed_if_tests = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            src_seg = ast.get_source_segment(mon_src, node.test) or ""
            if 'sample["elapsed"]' in src_seg and "23178.960703136865" in src_seg:
                elapsed_if_tests.append((node.lineno, node.test, src_seg))
    assert elapsed_if_tests, (
        'monitor.py must contain explicit AST predicate `sample["elapsed"] > 23178.960703136865`'
    )
    _, test_node, src_seg = elapsed_if_tests[0]
    assert src_seg.strip() == 'sample["elapsed"] > 23178.960703136865'
    code_obj = compile(ast.Expression(test_node), "<monitor_ast>", "eval")
    assert (
        eval(
            code_obj,
            {"__builtins__": {}},
            {"sample": {"elapsed": 23179.960703136865}},
        )
        is True
    )
    assert (
        eval(
            code_obj,
            {"__builtins__": {}},
            {"sample": {"elapsed": 23178.0}},
        )
        is False
    )

    # 2. Verify shared process-start deadline binds monitor, runner, and JRB-01 wait to ONE deadline
    deadline = mon.compute_shared_process_deadline(process_start_monotonic=1000.0)
    assert np.isclose(
        deadline["prior_elapsed_seconds"],
        2021.039296863135,
        rtol=0.0,
        atol=1e-9,
    )
    assert np.isclose(
        deadline["remaining_budget_seconds"],
        23178.960703136865,
        rtol=0.0,
        atol=1e-9,
    )
    assert np.isclose(
        deadline["hard_deadline_monotonic"],
        1000.0 + 23178.960703136865,
        rtol=0.0,
        atol=1e-9,
    )
    ok_check = mon.enforce_shared_process_deadline(
        1000.0 + 1800.0, deadline, stage="stage1_static_data"
    )
    assert np.isclose(
        ok_check["remaining_seconds"],
        23178.960703136865 - 1800.0,
        rtol=0.0,
        atol=1e-9,
    )
    with pytest.raises(RuntimeError, match="cumulative 7h hard wall deadline exceeded"):
        mon.enforce_shared_process_deadline(
            1000.0 + 23179.960703136865,
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
    with pytest.raises(RuntimeError, match="cannot reduce prior_elapsed_seconds"):
        mon.compute_remaining_wall_budget_seconds(
            prior_elapsed_seconds=1970.9372255061753
        )
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
    assert np.isclose(
        smoke["prior_attempts_elapsed_seconds"],
        2021.039296863135,
        rtol=0.0,
        atol=1e-9,
    )
    assert np.isclose(
        smoke["remaining_wall_budget_seconds"],
        23178.960703136865,
        rtol=0.0,
        atol=1e-9,
    )


def test_allowed_regression_train_roster_and_exclusion_refusals():
    """Verify real-schema exclusion parsing, per-source missing/count-mismatch refusal, transitive closure, and mandatory real-metadata disjointness."""
    with pytest.raises(RuntimeError, match="empty exclusion set refused"):
        stat.build_allowed_regression_train_roster(
            caller_excluded_families=set(), verify_hashes=False
        )
    with pytest.raises(RuntimeError, match="empty exclusion set refused"):
        stat.build_allowed_regression_train_roster(
            caller_excluded_datasets=[], verify_hashes=False
        )

    syn_catalog = [
        dict(
            id="ds_allowed_1",
            name="Allowed 1",
            task="regression",
            family_key="fk_ok_1",
            train_rows=100,
            features=5,
            categorical_features=0,
            train_sha256="1" * 64,
        ),
        dict(
            id="ds_null_fk",
            name="Null FK",
            task="regression",
            family_key=None,
            train_rows=100,
            features=5,
            categorical_features=0,
            train_sha256="2" * 64,
        ),
        dict(
            id="ds_binary",
            name="Binary Task",
            task="binary",
            family_key="fk_bin",
            train_rows=100,
            features=5,
            categorical_features=0,
            train_sha256="3" * 64,
        ),
        dict(
            id="ds_excl_fam",
            name="Excluded Fam",
            task="regression",
            family_key="fk_sealed",
            train_rows=100,
            features=5,
            categorical_features=0,
            train_sha256="4" * 64,
        ),
    ]
    syn_fpool = {
        "ds_allowed_1": "fam_ok_1",
        "ds_null_fk": "fam_null_fk",
        "ds_binary": "fam_bin",
        "ds_excl_fam": "fam_sealed",
    }
    roster_syn = stat.build_allowed_regression_train_roster(
        verify_hashes=False,
        caller_excluded_families={"fam_sealed"},
        caller_excluded_datasets={"ds_explicit_excl"},
        catalog_override=syn_catalog,
        family_by_dataset_override=syn_fpool,
    )
    assert roster_syn["status"] == "PASS"
    assert roster_syn["allowed_datasets_count"] == 1
    assert roster_syn["skipped_missing_family_count"] == 1
    assert roster_syn["records"][0]["dataset_id"] == "ds_allowed_1"

    # Candidate checks: valid passes; null family, binary, bad role, and excluded family fail
    ok_cand = stat.verify_single_regression_train_candidate(
        dict(
            dataset_id="ds_allowed_1",
            family_id="fam_ok_1",
            family_key="fk_ok_1",
            task="regression",
            role="regression_train",
            source_split="train",
            dl_split="DL-Train",
            has_meta_test=False,
        ),
        roster_syn,
    )
    assert ok_cand["status"] == "PASS"

    with pytest.raises(RuntimeError, match="missing or null family_id"):
        stat.verify_single_regression_train_candidate(
            dict(
                dataset_id="ds_allowed_1",
                family_id=None,
                family_key="fk_ok_1",
                task="regression",
            ),
            roster_syn,
        )
    with pytest.raises(RuntimeError, match="non-regression task refused"):
        stat.verify_single_regression_train_candidate(
            dict(
                dataset_id="ds_allowed_1",
                family_id="fam_ok_1",
                family_key="fk_ok_1",
                task="binary",
                role="regression_train",
                source_split="train",
                dl_split="DL-Train",
            ),
            roster_syn,
        )
    with pytest.raises(RuntimeError, match="invalid role or split cut"):
        stat.verify_single_regression_train_candidate(
            dict(
                dataset_id="ds_allowed_1",
                family_id="fam_ok_1",
                family_key="fk_ok_1",
                task="regression",
                role="panel",
                source_split="train",
                dl_split="DL-Train",
            ),
            roster_syn,
        )
    with pytest.raises(RuntimeError, match="forbidden family/dataset overlap"):
        stat.verify_single_regression_train_candidate(
            dict(
                dataset_id="ds_excl_fam",
                family_id="fam_sealed",
                family_key="fk_sealed",
                task="regression",
                role="regression_train",
                source_split="train",
                dl_split="DL-Train",
            ),
            roster_syn,
        )

    # Per-source real-schema missing-key / empty / count-mismatch refusals
    valid_jrb01 = {
        "family_keys": [{"kind": "dedupe_fingerprint", "value": "fk_jrb_1"}],
        "unkeyed_catalog_ids": ["0123456789abcdef"],
    }
    valid_h150 = {
        "families": {"fam_h150_1": {"dataset_ids": ["fedcba9876543210"]}},
        "catalog_sealed_ids": ["fedcba9876543210"],
        "dataset_hashes": ["fedcba9876543210"],
        "context_ids": ["a" * 64],
        "view_ids": ["v1"],
        "counts": {
            "families_total": 1,
            "catalog_sealed_ids": 1,
            "dataset_hashes": 1,
            "context_ids": 1,
            "view_ids": 1,
        },
    }
    valid_d4 = {
        "role_counts": {
            "binary_train": 1,
            "panel": 1,
            "regression_train": 1,
            "regression_val": 1,
        },
        "roles": {
            "binary_train": [
                {"dataset": "ds_b1", "family": "fam_b1", "status": "D4_ELIGIBLE"}
            ],
            "panel": [
                {"dataset": "ds_p1", "family": "fam_p1", "status": "D4_ELIGIBLE"}
            ],
            "regression_train": [
                {
                    "dataset": "ds_rt1",
                    "family": "fam_rt1",
                    "context_sha256": "b" * 64,
                }
            ],
            "regression_val": [
                {
                    "dataset": "ds_rv1",
                    "family": "fam_rv1",
                    "context_sha256": "c" * 64,
                }
            ],
        },
    }
    valid_selected = {
        "train": [
            {
                "context_sha256": "d" * 64,
                "dataset_hash": "ds_sel_tr",
                "family_id": "fam_sel_tr",
            }
        ],
        "dev": [
            {
                "context_sha256": "e" * 64,
                "dataset_hash": "ds_sel_dv",
                "family_id": "fam_sel_dv",
            }
        ],
    }
    valid_cohort = [
        {
            "context_sha256": "f" * 64,
            "dataset_hash": "ds_coh_1",
            "family_id": "fam_coh_1",
        }
    ]
    valid_rk = ["0" * 64]

    def _call_with_overrides(**kw):
        base_kw = dict(
            verify_hashes=False,
            catalog_override=syn_catalog,
            family_by_dataset_override=syn_fpool,
            jrb01_override=valid_jrb01,
            h150_override=valid_h150,
            d4_roles_override=valid_d4,
            selected_override=valid_selected,
            cohort_override=valid_cohort,
            ranking_cids_override=valid_rk,
        )
        base_kw.update(kw)
        return stat.build_allowed_regression_train_roster(**base_kw)

    assert _call_with_overrides()["status"] == "PASS"

    with pytest.raises(RuntimeError, match="JRB01_SEALED_FAMILY_EXCLUSION.json"):
        _call_with_overrides(jrb01_override={"families": ["wrong_key"]})
    with pytest.raises(RuntimeError, match="JRB01_SEALED_FAMILY_EXCLUSION.json"):
        _call_with_overrides(
            jrb01_override={"family_keys": [], "unkeyed_catalog_ids": ["1" * 16]}
        )
    with pytest.raises(RuntimeError, match="sealed_family_manifest.json"):
        _call_with_overrides(h150_override={"sealed_families": ["wrong_key"]})
    with pytest.raises(RuntimeError, match="count mismatch"):
        bad_h150 = copy.deepcopy(valid_h150)
        bad_h150["counts"]["families_total"] = 999
        _call_with_overrides(h150_override=bad_h150)
    with pytest.raises(RuntimeError, match="d4-role-candidate-manifest.json"):
        _call_with_overrides(
            d4_roles_override={"roles": {"panel": []}, "role_counts": {}}
        )
    with pytest.raises(RuntimeError, match="count-mismatched role"):
        bad_d4 = copy.deepcopy(valid_d4)
        bad_d4["role_counts"]["panel"] = 999
        _call_with_overrides(d4_roles_override=bad_d4)
    with pytest.raises(RuntimeError, match="missing required 'family' field"):
        bad_d4_fam = copy.deepcopy(valid_d4)
        del bad_d4_fam["roles"]["panel"][0]["family"]
        _call_with_overrides(d4_roles_override=bad_d4_fam)

    # Mandatory real-metadata disjointness verification (no skip)
    real_roster = stat.build_allowed_regression_train_roster(verify_hashes=True)
    assert real_roster["fpool_disjoint_datasets_count"] == 1772
    assert real_roster["transitive_disjoint_datasets_count"] == 1550
    assert real_roster["allowed_datasets_count"] == 1336
    assert real_roster["allowed_families_count"] == 1336
    assert (
        real_roster["roster_sha256"]
        == "c7ea7148129937d516f19a87c05c771b34d250f10ce9a2e53931e8f13760a504"
    )

    # Independently load all real exclusion artifacts using their true schemas and prove 0 overlap
    jrb_real = json.loads(stat.JRB01_SEALED_EXCLUSION_PATH.read_text())
    h150_real = json.loads(stat.H150_SEALED_FAMILY_MANIFEST_PATH.read_text())
    d4_real = json.loads(stat.D4_ROLE_CANDIDATE_MANIFEST_PATH.read_text())
    sel_real = json.loads(stat.SELECTED_JSON_PATH.read_text())
    coh_real = json.loads(stat.HISTORICAL_COHORT_JSON_PATH.read_text())
    fpool_real = json.loads(stat.FAMILY_KEY_POOL_MANIFEST_PATH.read_text())
    fby_ds = fpool_real["family_by_dataset"]

    protected_ds = set(jrb_real["unkeyed_catalog_ids"])
    protected_ds.update(h150_real["catalog_sealed_ids"])
    protected_ds.update(h150_real["dataset_hashes"])
    protected_fams = {kv["value"] for kv in jrb_real["family_keys"]}
    for fk in h150_real["families"].keys():
        protected_fams.add(fk)
        if fk.startswith("fam-"):
            protected_fams.add(fk.removeprefix("fam-"))
    for rname in ("binary_train", "panel", "regression_train", "regression_val"):
        for e in d4_real["roles"][rname]:
            protected_ds.add(e["dataset"])
            if "family" in e and e["family"]:
                protected_fams.add(e["family"])
    for sp in ("train", "dev"):
        for r in sel_real[sp]:
            protected_ds.add(r.get("dataset_hash") or r["dataset_id"])
            if r.get("family_id"):
                protected_fams.add(r["family_id"])
            if r.get("family_key"):
                protected_fams.add(r["family_key"])
    for r in coh_real:
        protected_ds.add(r.get("dataset_hash") or r["dataset_id"])
        if r.get("family_id"):
            protected_fams.add(r["family_id"])
        if r.get("family_key"):
            protected_fams.add(r["family_key"])
    for d_id in list(protected_ds):
        if d_id in fby_ds:
            protected_fams.add(fby_ds[d_id])

    allowed_ds = {r["dataset_id"] for r in real_roster["records"]}
    allowed_fams = {r["family_id"] for r in real_roster["records"]} | {
        r["family_key"] for r in real_roster["records"]
    }
    assert allowed_ds.isdisjoint(protected_ds)
    assert set(real_roster["fpool_disjoint_dataset_ids"]).isdisjoint(protected_ds)
    assert set(real_roster["transitive_disjoint_dataset_ids"]).isdisjoint(protected_ds)
    assert allowed_fams.isdisjoint(protected_fams)

    target_ok = stat.verify_single_regression_train_candidate(
        dict(
            dataset_id=stat.TARGET_REGRESSION_DATASET_ID,
            family_id=stat.TARGET_REGRESSION_FAMILY_ID,
            family_key=stat.TARGET_REGRESSION_FAMILY_KEY,
            context_sha256=stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256,
            task="regression",
            role="regression_train",
            source_split="train",
            dl_split="DL-Train",
            has_meta_test=False,
        ),
        real_roster,
    )
    assert target_ok["status"] == "PASS"

    # Historical local s1-prod replay context fa480de3... is explicitly refused as a same-S3-TRAIN context
    with pytest.raises(RuntimeError, match="historical local s1-prod replay"):
        stat.verify_single_regression_train_candidate(
            dict(
                dataset_id=stat.TARGET_REGRESSION_DATASET_ID,
                family_id=stat.TARGET_REGRESSION_FAMILY_ID,
                family_key=stat.TARGET_REGRESSION_FAMILY_KEY,
                context_sha256=stat.HISTORICAL_LOCAL_S1PROD_REPLAY_CONTEXT_SHA256,
                task="regression",
                role="regression_train",
                source_split="train",
                dl_split="DL-Train",
                has_meta_test=False,
            ),
            real_roster,
        )


def test_encode_train_only_jtt1_and_nq0_protocol_check(tmp_path):
    """Verify headerless (`has_header=False`) vs headered (`has_header=True`) CSV encoding with `manifest.json` checks and `nq = 0` protocol refusal."""
    # 1. Headered CSV (`has_header=True`)
    csv_hdr = tmp_path / "train_hdr.csv"
    lines = ["x0,x1,x2,x3,x4,target"]
    for i in range(20):
        lines.append(f"{i * 0.1:.4f},{i * 0.2:.4f},1.0,2.0,3.0,{i * 1.5 + 0.5:.4f}")
    csv_hdr.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_hdr_sha = run_mod.sha(csv_hdr)
    man_hdr = tmp_path / "manifest_hdr.json"
    man_hdr.write_text(
        json.dumps(
            {
                "task": "regression",
                "files": {
                    "train": {
                        "bytes": csv_hdr.stat().st_size,
                        "columns": 6,
                        "has_header": True,
                        "rows": 20,
                        "sha256": csv_hdr_sha,
                    }
                },
                "shape": {"features": 5},
                "split": {"train_rows": 20},
                "target": {"index": 5},
            }
        ),
        encoding="utf-8",
    )

    jtt1_hdr = tmp_path / "train_hdr_nq0.jtt1"
    enc_hdr = stat.encode_train_only_jtt1_from_csv(
        csv_hdr,
        out_jtt1_path=jtt1_hdr,
        expected_csv_sha256=csv_hdr_sha,
        manifest_path=man_hdr,
    )
    assert enc_hdr["status"] == "PASS"
    assert enc_hdr["has_header"] is True
    assert enc_hdr["nt"] == 20
    assert enc_hdr["nq"] == 0
    assert enc_hdr["p"] == 5

    # 2. Headerless CSV (`has_header=False`) — row 0 is numeric data and MUST NOT be dropped!
    csv_nohdr = tmp_path / "train_nohdr.csv"
    csv_nohdr.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    csv_nohdr_sha = run_mod.sha(csv_nohdr)
    man_nohdr = tmp_path / "manifest_nohdr.json"
    man_nohdr.write_text(
        json.dumps(
            {
                "task": "regression",
                "files": {
                    "train": {
                        "bytes": csv_nohdr.stat().st_size,
                        "columns": 6,
                        "has_header": False,
                        "rows": 20,
                        "sha256": csv_nohdr_sha,
                    }
                },
                "shape": {"features": 5},
                "split": {"train_rows": 20},
                "target": {"index": 5},
            }
        ),
        encoding="utf-8",
    )
    enc_nohdr = stat.encode_train_only_jtt1_from_csv(
        csv_nohdr,
        expected_csv_sha256=csv_nohdr_sha,
        manifest_path=man_nohdr,
    )
    assert enc_nohdr["has_header"] is False
    assert enc_nohdr["nt"] == 20
    assert enc_nohdr["train_matrix_sha256"] == enc_hdr["train_matrix_sha256"]
    assert enc_nohdr["jtt1_sha256"] == enc_hdr["jtt1_sha256"]

    # Mismatched row count, has_header, or SHA256 against manifest is refused
    with pytest.raises(RuntimeError, match="contradicts manifest rows"):
        stat.encode_train_only_jtt1_from_csv(
            csv_nohdr, manifest_path=man_nohdr, expected_rows=19
        )
    with pytest.raises(RuntimeError, match="contradicts manifest has_header"):
        stat.encode_train_only_jtt1_from_csv(
            csv_nohdr, manifest_path=man_nohdr, has_header=True
        )
    with pytest.raises(RuntimeError, match="train CSV SHA256 mismatch"):
        stat.encode_train_only_jtt1_from_csv(csv_hdr, expected_csv_sha256="f" * 64)

    # 3. Verify real 1,000-row headerless 1db3750d4ceccab9 S3 train.csv + manifest.json
    real_intake_dir = Path(
        "/home/ubuntu/jope-data/h205/regression-seam-20261102T0752Z/raw_s3_intake/1db3750d4ceccab9"
    )
    if (real_intake_dir / "train.csv").is_file():
        real_enc = stat.encode_train_only_jtt1_from_csv(
            real_intake_dir / "train.csv",
            expected_csv_sha256=stat.TARGET_REGRESSION_TRAIN_SHA256,
            manifest_path=real_intake_dir / "manifest.json",
        )
        assert real_enc["has_header"] is False
        assert real_enc["nt"] == 1000
        assert real_enc["nq"] == 0
        assert real_enc["p"] == 5
        assert real_enc["jtt1_bytes_len"] == 49017
        assert (
            real_enc["jtt1_sha256"]
            == "a6d997e39cec56995f579c380a0a1e960d3e887aebb5d40b8a0aae88f613483c"
        )
        assert (
            real_enc["train_matrix_sha256"]
            == "cb04fa33aed43118c9c15a794bfac1324e049d56b3833adef56913d26583dff7"
        )

    if stat.EVALUATOR_BINARY_PATH.is_file():
        probe = stat.probe_s1main_nq0_protocol_blocker(jtt1_hdr)
        assert probe["header"] == dict(nt=20, nq=0, p=5, k=2)
        assert probe["returncode"] != 0
        assert probe["nq0_refused_by_header_guard"] is True


def test_train_only_padding_bridge_and_h206_fixture_evidence(tmp_path):
    """Verify `build_train_only_padded_native_jtt1`, `strip_s1_timing_fields`, `verify_scientific_inner_cv_invariance`,
    and `verify_h206_native_protocol_fixture_evidence` plus all negative controls.
    """
    assert (
        stat.TRAIN_PADDING_PROTOCOL_SHA256
        == "cb967d3c4baa4e855f3b893ccfff2d1045e64435277f89530c11ae4d3a6aa0ed"
    )
    assert run_mod.sha(stat.NATIVE_RUST_SOURCE_PATH) == stat.NATIVE_RUST_SOURCE_SHA256
    assert (
        run_mod.sha(stat.METRIC_EVALUATOR_SOURCE_PATH)
        == stat.METRIC_EVALUATOR_SOURCE_SHA256
    )

    # 1. Synthetic 60-row headerless CSV -> nq=0 JTT1 -> nq=10 TRAIN-only padded JTT1
    csv_60 = tmp_path / "train_60.csv"
    lines_60 = [
        f"{i * 0.05:.4f},{(i % 7) * 0.2:.4f},1.25,-0.5,{i * 0.01:.4f},{i * 0.75 + 0.1:.4f}"
        for i in range(60)
    ]
    csv_60.write_text("\n".join(lines_60) + "\n", encoding="utf-8")
    csv_60_sha = run_mod.sha(csv_60)
    man_60 = tmp_path / "manifest_60.json"
    man_60.write_text(
        json.dumps(
            {
                "task": "regression",
                "files": {
                    "train": {
                        "bytes": csv_60.stat().st_size,
                        "columns": 6,
                        "has_header": False,
                        "rows": 60,
                        "sha256": csv_60_sha,
                    }
                },
                "shape": {"features": 5},
                "split": {"train_rows": 60},
                "target": {"index": 5},
            }
        ),
        encoding="utf-8",
    )
    jtt1_60_nq0 = tmp_path / "train_60_nq0.jtt1"
    enc_60 = stat.encode_train_only_jtt1_from_csv(
        csv_60,
        out_jtt1_path=jtt1_60_nq0,
        expected_csv_sha256=csv_60_sha,
        manifest_path=man_60,
    )
    assert enc_60["nt"] == 60 and enc_60["nq"] == 0
    assert len(enc_60["cut_sha256"]) == 64

    jtt1_60_nq10 = tmp_path / "train_60_padded_nq10.jtt1"
    bridge_60 = stat.build_train_only_padded_native_jtt1(
        jtt1_60_nq0,
        out_padded_jtt1_path=jtt1_60_nq10,
        expected_original_jtt1_nq0_sha256=enc_60["jtt1_sha256"],
        expected_train_matrix_sha256=enc_60["train_matrix_sha256"],
        expected_cut_sha256=enc_60["cut_sha256"],
        expected_nt=60,
        expected_p=5,
    )
    assert bridge_60["status"] == "PASS"
    assert bridge_60["original_jtt1_nq0_sha256"] != bridge_60["padded_jtt1_nq10_sha256"]
    assert bridge_60["original_header"] == dict(nt=60, nq=0, p=5, k=2)
    assert bridge_60["padded_header"] == dict(nt=60, nq=10, p=5, k=2)
    assert bridge_60["cut_sha256"] == enc_60["cut_sha256"]
    assert bridge_60["train_matrix_sha256"] == enc_60["train_matrix_sha256"]
    assert bridge_60["padding_examples_added_to_learner"] == 0
    assert bridge_60["padding_families_added"] == 0

    # Negative controls on build_train_only_padded_native_jtt1:
    # (a) Non-zero source nq refused (passing already-padded nq=10 file as source)
    with pytest.raises(RuntimeError, match="requires source nq == 0"):
        stat.build_train_only_padded_native_jtt1(jtt1_60_nq10)
    # (b) Altered original nq=0 SHA refused
    with pytest.raises(RuntimeError, match="original nq=0 JTT1 SHA256 mismatch"):
        stat.build_train_only_padded_native_jtt1(
            jtt1_60_nq0, expected_original_jtt1_nq0_sha256="d" * 64
        )
    # (c) Altered train_matrix_sha256 refused
    with pytest.raises(RuntimeError, match="train_matrix_sha256 mismatch"):
        stat.build_train_only_padded_native_jtt1(
            jtt1_60_nq0, expected_train_matrix_sha256="f" * 64
        )
    # (d) Altered cut_sha256 refused
    with pytest.raises(RuntimeError, match="cut_sha256 mismatch"):
        stat.build_train_only_padded_native_jtt1(
            jtt1_60_nq0, expected_cut_sha256="e" * 64
        )
    # (e) Fewer than 50 train rows refused
    raw_60 = bytearray(jtt1_60_nq0.read_bytes())
    small_nt_jtt1 = tmp_path / "small_nt.jtt1"
    # Build a 20-row nq=0 JTT1
    csv_20 = tmp_path / "train_20.csv"
    csv_20.write_text("\n".join(lines_60[:20]) + "\n", encoding="utf-8")
    stat.encode_train_only_jtt1_from_csv(
        csv_20, out_jtt1_path=small_nt_jtt1, has_header=False
    )
    with pytest.raises(RuntimeError, match="requires nt >= 50"):
        stat.build_train_only_padded_native_jtt1(small_nt_jtt1)
    # (f) Corrupted fold cut (all fold 0 -> fold 1 has 0 < 25 rows) refused
    bad_cut_bytes = bytearray(raw_60)
    bad_cut_bytes[17 : 17 + 60] = b"\x00" * 60
    bad_cut_jtt1 = tmp_path / "bad_cut.jtt1"
    bad_cut_jtt1.write_bytes(bytes(bad_cut_bytes))
    with pytest.raises(RuntimeError, match="each inner fold requires >= 25 rows"):
        stat.build_train_only_padded_native_jtt1(bad_cut_jtt1)

    # 2. Real 1db3750d4ceccab9 1,000-row headerless train.csv -> exact nq=0 and nq=10 bridge hashes
    real_intake_dir = Path(
        "/home/ubuntu/jope-data/h205/regression-seam-20261102T0752Z/raw_s3_intake/1db3750d4ceccab9"
    )
    if (real_intake_dir / "train.csv").is_file():
        real_nq0 = tmp_path / "real_1db3750d4ceccab9_nq0.jtt1"
        real_nq10 = tmp_path / "real_1db3750d4ceccab9_nq10.jtt1"
        real_enc = stat.encode_train_only_jtt1_from_csv(
            real_intake_dir / "train.csv",
            out_jtt1_path=real_nq0,
            expected_csv_sha256=stat.TARGET_REGRESSION_TRAIN_SHA256,
            manifest_path=real_intake_dir / "manifest.json",
        )
        assert real_enc["jtt1_sha256"] == stat.TARGET_REGRESSION_NQ0_JTT1_SHA256
        assert real_enc["cut_sha256"] == stat.TARGET_REGRESSION_CUT_SHA256
        real_bridge = stat.build_train_only_padded_native_jtt1(
            real_nq0,
            out_padded_jtt1_path=real_nq10,
            expected_original_jtt1_nq0_sha256=stat.TARGET_REGRESSION_NQ0_JTT1_SHA256,
            expected_train_matrix_sha256=stat.TARGET_REGRESSION_TRAIN_MATRIX_SHA256,
            expected_cut_sha256=stat.TARGET_REGRESSION_CUT_SHA256,
            expected_nt=1000,
            expected_p=5,
        )
        assert (
            real_bridge["padded_jtt1_nq10_sha256"]
            == stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256
        )
        assert (
            real_bridge["padding_queries_sha256"]
            == stat.TARGET_REGRESSION_PADDING_QUERIES_SHA256
        )
        derived_cid = stat.derive_same_s3_train_context_sha256(
            dataset_id=stat.TARGET_REGRESSION_DATASET_ID,
            train_sha256=stat.TARGET_REGRESSION_TRAIN_SHA256,
            original_jtt1_nq0_sha256=real_bridge["original_jtt1_nq0_sha256"],
            padded_jtt1_nq10_sha256=real_bridge["padded_jtt1_nq10_sha256"],
            cut_sha256=real_bridge["cut_sha256"],
        )
        assert derived_cid == stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256
        assert derived_cid != stat.HISTORICAL_LOCAL_S1PROD_REPLAY_CONTEXT_SHA256

    # 3. Verify scientific inner-CV invariance helper and measured H206 fixture receipts (0 new native calls)
    ev_a = {
        "status": "completed",
        "fold_protocol": "jope.s1-two-fold/v1",
        "total_ms": 11,
        "inner_cv": [
            {
                "fold_id": "s1_fold_0",
                "rows_train": 30,
                "rows_test": 30,
                "fit_ms": 5,
                "predict_ms": 1,
                "metrics": {"rmse": 0.25, "mae": 0.2, "r2": 0.9, "resid_std": 0.25},
                "baseline_metrics": {
                    "rmse": 1.0,
                    "mae": 0.8,
                    "r2": 0.0,
                    "resid_std": 1.0,
                },
            },
            {
                "fold_id": "s1_fold_1",
                "rows_train": 30,
                "rows_test": 30,
                "fit_ms": 4,
                "predict_ms": 1,
                "metrics": {"rmse": 0.27, "mae": 0.21, "r2": 0.88, "resid_std": 0.27},
                "baseline_metrics": {
                    "rmse": 1.0,
                    "mae": 0.8,
                    "r2": 0.0,
                    "resid_std": 1.0,
                },
            },
        ],
    }
    ev_b = copy.deepcopy(ev_a)
    ev_b["total_ms"] = 19
    ev_b["inner_cv"][0]["fit_ms"] = 9
    inv_ok = stat.verify_scientific_inner_cv_invariance(ev_a, ev_b)
    assert inv_ok["invariance_raw_equal"] is False
    assert inv_ok["scientific_inner_cv_equal"] is True
    assert inv_ok["max_abs_scientific_metric_diff"] == 0.0

    ev_bad = copy.deepcopy(ev_b)
    ev_bad["inner_cv"][0]["metrics"]["rmse"] += 1e-6
    with pytest.raises(RuntimeError, match="scientific inner-CV invariance failed"):
        stat.verify_scientific_inner_cv_invariance(ev_a, ev_bad)

    h206_ev = stat.verify_h206_native_protocol_fixture_evidence()
    assert h206_ev["status"] == "PASS"
    assert h206_ev["native_calls_total"] == 2
    assert h206_ev["invariance_raw_equal"] is False
    assert h206_ev["scientific_inner_cv_equal"] is True
    assert h206_ev["max_abs_scientific_metric_diff"] == 0.0
    assert h206_ev["real_TRAIN_or_r882_qualification"] is False


def test_single_context_regression_seam_bundle_verifier_and_negative_counterexamples(
    tmp_path,
):
    """Verify `verify_pre_consumption_s3_replay_bundle`, `verify_single_context_regression_seam_bundle`, and all negative rejection counterexamples."""
    bdir = tmp_path / "bundle_ok"
    bdir.mkdir(parents=True, exist_ok=True)

    cid = hashlib.sha256(b"fresh_reg_ctx_001").hexdigest()
    train_sha = hashlib.sha256(b"fresh_reg_train_csv_001").hexdigest()
    train_mat_sha = hashlib.sha256(b"fresh_reg_train_mat_001").hexdigest()
    cut_sha = hashlib.sha256(b"fresh_reg_cut_001").hexdigest()
    orig_nq0_sha = hashlib.sha256(b"fresh_reg_jtt1_nq0_001").hexdigest()
    padded_nq10_sha = hashlib.sha256(b"fresh_reg_jtt1_nq10_001").hexdigest()
    ds_id = "fresh_ds_001"
    fam_id = "fresh_fam_001"
    fam_key = "fresh_fk_001"

    dsl_list = [
        f"(pipe X (standardize append=true) (winsor_clip q={0.05 + 0.005 * i:.4f}) (ridge_gcv))"
        for i in range(stat.SLATE_CANDIDATES)
    ]
    feat_names = [f"f_{i}" for i in range(stat.PROGRAM_DIM)]
    feat_names[0] = "steps"
    feat_names[1] = "dsl_characters"
    feat_names[2] = "estimator.ridge_gcv"
    prog_mat, proj16_mat = run_mod.featurize_dsl_412_and_proj16(
        dsl_list, feature_names=feat_names
    )
    fitness = np.linspace(0.6, 1.2, stat.SLATE_CANDIDATES, dtype=np.float64)
    gate_arr = np.ones(stat.SLATE_CANDIDATES, dtype=bool)
    tok = np.full((stat.TOKEN_STEPS, stat.TOKEN_DIM), 0.25, dtype=np.float32)
    msk = np.ones(stat.TOKEN_STEPS, dtype=bool)
    d629 = np.zeros(629, dtype=np.float32)
    lm48 = np.zeros(48, dtype=np.float32)
    ctx584 = np.zeros(584, dtype=np.float32)

    pool_path = bdir / "pool.npz"
    feat_path = bdir / "features.npz"
    np.savez(
        pool_path,
        program=prog_mat,
        proj16=proj16_mat,
        fitness=fitness,
        gate=gate_arr,
        tokens_10x32=tok,
        mask_10=msk,
        train_rows=np.asarray(1000, dtype=np.int64),
        train_sha256=np.asarray(train_sha),
        train_matrix_sha256=np.asarray(train_mat_sha),
        cut_sha256=np.asarray(cut_sha),
        original_jtt1_nq0_sha256=np.asarray(orig_nq0_sha),
        padded_jtt1_nq10_sha256=np.asarray(padded_nq10_sha),
    )
    np.savez(
        feat_path,
        tokens_10x32=tok,
        mask_10=msk,
        dope629=d629,
        lm48_z=lm48,
        ctx_feat_584=ctx584,
        train_rows=np.asarray(1000, dtype=np.int64),
        train_sha256=np.asarray(train_sha),
        train_matrix_sha256=np.asarray(train_mat_sha),
        cut_sha256=np.asarray(cut_sha),
        original_jtt1_nq0_sha256=np.asarray(orig_nq0_sha),
        padded_jtt1_nq10_sha256=np.asarray(padded_nq10_sha),
    )

    labels_sha = hashlib.sha256(fitness.tobytes() + gate_arr.tobytes()).hexdigest()
    programs_sha = hashlib.sha256(prog_mat.tobytes()).hexdigest()
    features_sha = hashlib.sha256(tok.tobytes() + msk.tobytes()).hexdigest()
    pool_sha = run_mod.sha(pool_path)
    feat_npz_sha = run_mod.sha(feat_path)

    eval_rec = dict(
        status="PASS",
        evaluator=stat.NATIVE_RUST_EVALUATOR_ID,
        evaluator_binary_sha256=stat.EVALUATOR_BINARY_SHA256,
        native_rust_source_sha256=stat.NATIVE_RUST_SOURCE_SHA256,
        metric_source_sha256=stat.METRIC_EVALUATOR_SOURCE_SHA256,
        context_sha256=cid,
        dataset_id=ds_id,
        train_sha256=train_sha,
        train_matrix_sha256=train_mat_sha,
        cut_sha256=cut_sha,
        train_rows=1000,
        has_header=False,
        feature_columns=5,
        original_jtt1_nq0_sha256=orig_nq0_sha,
        padded_jtt1_nq10_sha256=padded_nq10_sha,
        train_padding_bridge_schema=stat.TRAIN_PADDING_BRIDGE_SCHEMA,
        train_padding_bridge_rule=stat.TRAIN_PADDING_QUERY_RULE,
        train_padding_bridge_protocol_sha256=stat.TRAIN_PADDING_PROTOCOL_SHA256,
        padding_examples_added_to_learner=0,
        padding_families_added=0,
        s1_input_sha256=padded_nq10_sha,
        training_block_sha256=padded_nq10_sha,
        labels_sha256=labels_sha,
        programs_sha256=programs_sha,
        features_sha256=features_sha,
        dope_sha256=stat.DOPE_SHA256,
        r882_sha256=stat.R882_SHA256,
        candidates_count=24,
        evaluated_candidates_count=24,
        outer_test_rows_decoded=0,
        live_s1main_slate24_verification=dict(
            status="PASS",
            evaluated_candidates=24,
            max_abs_fitness_diff=0.0,
            gate_exact_match=True,
        ),
    )
    manifest = dict(
        dataset_id=ds_id,
        family_id=fam_id,
        family_key=fam_key,
        family_aliases=[fam_id, fam_key],
        context_sha256=cid,
        task="regression",
        role="regression_train",
        source_split="train",
        dl_split="DL-Train",
        has_meta_test=False,
        outer_test_rows_decoded=0,
        train_sha256=train_sha,
        train_matrix_sha256=train_mat_sha,
        cut_sha256=cut_sha,
        train_rows=1000,
        has_header=False,
        feature_columns=5,
        original_jtt1_nq0_sha256=orig_nq0_sha,
        padded_jtt1_nq10_sha256=padded_nq10_sha,
        train_padding_bridge_schema=stat.TRAIN_PADDING_BRIDGE_SCHEMA,
        train_padding_bridge_rule=stat.TRAIN_PADDING_QUERY_RULE,
        train_padding_bridge_protocol_sha256=stat.TRAIN_PADDING_PROTOCOL_SHA256,
        padding_examples_added_to_learner=0,
        padding_families_added=0,
        s1_input_sha256=padded_nq10_sha,
        training_block_sha256=padded_nq10_sha,
        pool_sha256=pool_sha,
        features_npz_sha256=feat_npz_sha,
    )
    run_mod.save(bdir / "context_manifest.json", manifest)
    run_mod.save(bdir / "native_evaluator_receipt.json", eval_rec)
    cm_sha = run_mod.sha(bdir / "context_manifest.json")
    nr_sha = run_mod.sha(bdir / "native_evaluator_receipt.json")

    pre_s3_rec = dict(
        status="PASS",
        persisted_and_replayed_before_optimizer_step=True,
        source_train_object=dict(
            bucket="veox-jopedime-datasets-use2",
            key=f"blobs/sha256/{train_sha}",
            version_id="ver_src_001",
            sha256=train_sha,
        ),
        uploaded_bundle_objects=[
            dict(
                name="context_manifest.json",
                version_id="ver_cm_001",
                local_sha256=cm_sha,
                downloaded_version_sha256=cm_sha,
            ),
            dict(
                name="native_evaluator_receipt.json",
                version_id="ver_nr_001",
                local_sha256=nr_sha,
                downloaded_version_sha256=nr_sha,
            ),
            dict(
                name="pool.npz",
                version_id="ver_pool_001",
                local_sha256=pool_sha,
                downloaded_version_sha256=pool_sha,
            ),
            dict(
                name="features.npz",
                version_id="ver_feat_001",
                local_sha256=feat_npz_sha,
                downloaded_version_sha256=feat_npz_sha,
            ),
        ],
    )
    s3_rec = dict(
        status="PASS",
        source_train_object=pre_s3_rec["source_train_object"],
        uploaded_bundle_objects=pre_s3_rec["uploaded_bundle_objects"],
    )
    eng_rec = dict(
        status="PASS",
        ingested_context_sha256=cid,
        dataset_id=ds_id,
        s3_persisted_and_replayed_before_optimizer_step=True,
        consumed_pool_s3_version_id="ver_pool_001",
        consumed_features_s3_version_id="ver_feat_001",
        used_old_1310_fallback=False,
        optimizer_steps=1,
        trainable_parameters=29211,
        adapter_sha256_before="a" * 64,
        adapter_sha256_after="b" * 64,
        dope_sha256_before=stat.DOPE_SHA256,
        dope_sha256_after=stat.DOPE_SHA256,
        loss_value=0.45,
    )

    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", pre_s3_rec)
    run_mod.save(bdir / "s3_custody_receipt.json", s3_rec)
    run_mod.save(bdir / "engine_consumption_receipt.json", eng_rec)

    res = stat.verify_single_context_regression_seam_bundle(
        bdir,
        expected_train_sha256=train_sha,
        expected_train_matrix_sha256=train_mat_sha,
        expected_cut_sha256=cut_sha,
        expected_original_jtt1_nq0_sha256=orig_nq0_sha,
        expected_padded_jtt1_nq10_sha256=padded_nq10_sha,
        expected_train_rows=1000,
        forbidden_cids={"9" * 64},
        forbidden_families={"old_static_fam"},
        forbidden_datasets={"old_static_ds"},
    )
    assert res["status"] == "PASS"
    assert res["context_sha256"] == cid
    assert res["original_jtt1_nq0_sha256"] == orig_nq0_sha
    assert res["padded_jtt1_nq10_sha256"] == padded_nq10_sha
    assert res["cut_sha256"] == cut_sha
    assert res["s3_persisted_and_replayed_before_optimizer_step"] is True

    # 1. Empty exclusion sets refused
    with pytest.raises(RuntimeError, match="empty exclusion set refused"):
        stat.verify_single_context_regression_seam_bundle(bdir, forbidden_cids=set())
    with pytest.raises(RuntimeError, match="empty exclusion set refused"):
        stat.verify_single_context_regression_seam_bundle(
            bdir, forbidden_families=set()
        )

    # 2. Old-1310 CID / family / dataset overlap refused, AND historical local replay fa480de3... refused
    with pytest.raises(RuntimeError, match="overlaps forbidden/historical CIDs"):
        stat.verify_single_context_regression_seam_bundle(bdir, forbidden_cids={cid})
    with pytest.raises(RuntimeError, match="overlaps forbidden families"):
        stat.verify_single_context_regression_seam_bundle(
            bdir, forbidden_families={fam_id}
        )
    with pytest.raises(RuntimeError, match="overlaps forbidden datasets"):
        stat.verify_single_context_regression_seam_bundle(
            bdir, forbidden_datasets={ds_id}
        )

    # Historical local s1-prod replay hash fa480de3... is always refused even when not passed in forbidden_cids
    hist_cid = stat.HISTORICAL_LOCAL_S1PROD_REPLAY_CONTEXT_SHA256
    bad_hist_manifest = dict(manifest, context_sha256=hist_cid)
    bad_hist_eval = dict(eval_rec, context_sha256=hist_cid)
    run_mod.save(bdir / "context_manifest.json", bad_hist_manifest)
    run_mod.save(bdir / "native_evaluator_receipt.json", bad_hist_eval)
    hist_cm_sha = run_mod.sha(bdir / "context_manifest.json")
    hist_nr_sha = run_mod.sha(bdir / "native_evaluator_receipt.json")
    bad_pre_s3_hist = copy.deepcopy(pre_s3_rec)
    bad_pre_s3_hist["uploaded_bundle_objects"][0]["local_sha256"] = hist_cm_sha
    bad_pre_s3_hist["uploaded_bundle_objects"][0]["downloaded_version_sha256"] = (
        hist_cm_sha
    )
    bad_pre_s3_hist["uploaded_bundle_objects"][1]["local_sha256"] = hist_nr_sha
    bad_pre_s3_hist["uploaded_bundle_objects"][1]["downloaded_version_sha256"] = (
        hist_nr_sha
    )
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", bad_pre_s3_hist)
    with pytest.raises(RuntimeError, match="overlaps forbidden/historical CIDs"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )
    run_mod.save(bdir / "context_manifest.json", manifest)
    run_mod.save(bdir / "native_evaluator_receipt.json", eval_rec)
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", pre_s3_rec)

    # 2b. Padding bridge identity failures: identical nq0/nq10 hash or non-zero padding_examples_added_to_learner refused
    bad_same_jtt1_manifest = dict(manifest, padded_jtt1_nq10_sha256=orig_nq0_sha)
    run_mod.save(bdir / "context_manifest.json", bad_same_jtt1_manifest)
    bad_cm_sha = run_mod.sha(bdir / "context_manifest.json")
    bad_pre_s3_jtt1 = copy.deepcopy(pre_s3_rec)
    bad_pre_s3_jtt1["uploaded_bundle_objects"][0]["local_sha256"] = bad_cm_sha
    bad_pre_s3_jtt1["uploaded_bundle_objects"][0]["downloaded_version_sha256"] = (
        bad_cm_sha
    )
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", bad_pre_s3_jtt1)
    with pytest.raises(RuntimeError, match="failed TRAIN-only padding bridge identity"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )

    bad_pad_ex_manifest = dict(manifest, padding_examples_added_to_learner=10)
    run_mod.save(bdir / "context_manifest.json", bad_pad_ex_manifest)
    bad_cm_sha2 = run_mod.sha(bdir / "context_manifest.json")
    bad_pre_s3_pad = copy.deepcopy(pre_s3_rec)
    bad_pre_s3_pad["uploaded_bundle_objects"][0]["local_sha256"] = bad_cm_sha2
    bad_pre_s3_pad["uploaded_bundle_objects"][0]["downloaded_version_sha256"] = (
        bad_cm_sha2
    )
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", bad_pre_s3_pad)
    with pytest.raises(RuntimeError, match="failed TRAIN-only padding bridge identity"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )
    run_mod.save(bdir / "context_manifest.json", manifest)
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", pre_s3_rec)

    # 3. Local 1,600-row cut vs 1,000-row S3 train.csv mismatch refused
    with pytest.raises(RuntimeError, match="manifest train_rows"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            expected_train_rows=1600,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )

    # 4. Partial candidate evaluation (24 live vs 1,152 cached) refused
    bad_partial_eval = dict(eval_rec, evaluated_candidates_count=12)
    run_mod.save(bdir / "native_evaluator_receipt.json", bad_partial_eval)
    bad_nr_sha = run_mod.sha(bdir / "native_evaluator_receipt.json")
    bad_pre_s3 = copy.deepcopy(pre_s3_rec)
    bad_pre_s3["uploaded_bundle_objects"][1]["local_sha256"] = bad_nr_sha
    bad_pre_s3["uploaded_bundle_objects"][1]["downloaded_version_sha256"] = bad_nr_sha
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", bad_pre_s3)
    with pytest.raises(
        RuntimeError, match="all pool candidates must be natively evaluated"
    ):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )
    run_mod.save(bdir / "native_evaluator_receipt.json", eval_rec)
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", pre_s3_rec)

    # 5. Wrong evaluator / encoder hash refused
    bad_eval = dict(eval_rec, dope_sha256="c" * 64)
    run_mod.save(bdir / "native_evaluator_receipt.json", bad_eval)
    bad_nr_sha2 = run_mod.sha(bdir / "native_evaluator_receipt.json")
    bad_pre_s3_2 = copy.deepcopy(pre_s3_rec)
    bad_pre_s3_2["uploaded_bundle_objects"][1]["local_sha256"] = bad_nr_sha2
    bad_pre_s3_2["uploaded_bundle_objects"][1]["downloaded_version_sha256"] = (
        bad_nr_sha2
    )
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", bad_pre_s3_2)
    with pytest.raises(
        RuntimeError, match="failed same-TRAIN row/matrix/encoder/identity check"
    ):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )
    run_mod.save(bdir / "native_evaluator_receipt.json", eval_rec)
    run_mod.save(bdir / "s3_pre_consumption_custody_receipt.json", pre_s3_rec)

    # 6. Posthoc S3 upload (s3_persisted_and_replayed_before_optimizer_step=False) or unchanged adapter digest refused
    bad_eng_order = dict(eng_rec, s3_persisted_and_replayed_before_optimizer_step=False)
    run_mod.save(bdir / "engine_consumption_receipt.json", bad_eng_order)
    with pytest.raises(RuntimeError, match="failed pre-consumption S3 replay binding"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )

    bad_eng = dict(eng_rec, adapter_sha256_after=eng_rec["adapter_sha256_before"])
    run_mod.save(bdir / "engine_consumption_receipt.json", bad_eng)
    with pytest.raises(RuntimeError, match="parameter-delta check"):
        stat.verify_single_context_regression_seam_bundle(
            bdir,
            forbidden_cids={"9" * 64},
            forbidden_families={"old_static_fam"},
            forbidden_datasets={"old_static_ds"},
        )
    run_mod.save(bdir / "engine_consumption_receipt.json", eng_rec)


def test_construct_and_consume_single_regression_context_ast_and_whole_path_controls(
    tmp_path, monkeypatch
):
    """Verify AST ordering and whole-path controls for `construct_and_consume_single_regression_context`:
    (1) AST order: `_s3_put_create_only_and_verify` and `verify_pre_consumption_s3_replay_bundle` precede
        `consume_single_fresh_context_in_inherited_engine`, and local `teacher.jtt1.gz` / `summary.tar.gz` are absent.
    (2) Control A (`BLOCKED_NQ0_PROTOCOL`): when `bootstrap.s1main` refuses `nq == 0`, stops fail-closed with
        `optimizer_steps == 0` and zero S3 learning-store uploads.
    (3) Control B (S3 upload/replay failure negative control): if S3 upload/replay fails, raises `RuntimeError`
        before `consume_single_fresh_context_in_inherited_engine` is ever called (`optimizer_steps == 0`).
    (4) Control C (Same-TRAIN persist -> exact-VersionId replay -> consume): uploads and replays all 4 bundle
        files into `verified_s3_replay/` before calling `consume_single_fresh_context_in_inherited_engine`.
    """
    import ast
    import shutil

    runner_src = (HERE / "runner.py").read_text(encoding="utf-8")
    tree = ast.parse(runner_src)
    fn_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "construct_and_consume_single_regression_context"
    )
    exec_seg = "\n".join(
        ast.get_source_segment(runner_src, stmt) or "" for stmt in fn_node.body[1:]
    )
    assert "teacher.jtt1.gz" not in exec_seg
    assert "summary.tar.gz" not in exec_seg

    call_lines = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Call):
            s = ast.get_source_segment(runner_src, node.func) or ""
            if s in (
                "_s3_put_create_only_and_verify",
                "stat.verify_pre_consumption_s3_replay_bundle",
                "consume_single_fresh_context_in_inherited_engine",
                "stat.verify_single_context_regression_seam_bundle",
            ):
                call_lines.setdefault(s, []).append(node.lineno)

    assert (
        call_lines["_s3_put_create_only_and_verify"][0]
        < call_lines["stat.verify_pre_consumption_s3_replay_bundle"][0]
        < call_lines["consume_single_fresh_context_in_inherited_engine"][0]
        < call_lines["stat.verify_single_context_regression_seam_bundle"][0]
    )

    real_intake_dir = Path(
        "/home/ubuntu/jope-data/h205/regression-seam-20261102T0752Z/raw_s3_intake/1db3750d4ceccab9"
    )
    assert (real_intake_dir / "train.csv").is_file()
    assert (real_intake_dir / "manifest.json").is_file()

    def _mock_s3_get(
        bucket, key, dest_path, aws_profile="veox-jepson-prod", deadline_spec=None
    ):
        dest_p = Path(dest_path)
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("manifest.json"):
            shutil.copy2(real_intake_dir / "manifest.json", dest_p)
            ver = "9nTZwIaM1feUD0HpgMd1sktu.vmZd6f0"
        else:
            shutil.copy2(real_intake_dir / "train.csv", dest_p)
            ver = "xJwIN6naH0sfM_uMuEBwd63AV2A8H.6_"
        return dict(
            bucket=bucket,
            key=key,
            version_id=ver,
            size_bytes=int(dest_p.stat().st_size),
            sha256=run_mod.sha(dest_p),
        )

    monkeypatch.setattr(run_mod, "_s3_get_object_with_version", _mock_s3_get)

    manifest_obj = json.loads((HERE / "manifest.json").read_text())
    launch_obj = dict(
        wave="H205",
        authority=stat.AUTHORITY_STRING,
        manifest_sha256=run_mod.sha(HERE / "manifest.json"),
        numeric_threads=2,
        cuda_runtime_threads_exempt=True,
        live_flywheel_pid_untouched=2297929,
    )
    deadline_spec = mon.compute_shared_process_deadline(
        process_start_monotonic=run_mod.time.monotonic()
    )

    # Control A1: require_hold_release_for_live=True without release receipt -> HOLD_ACTIVE_PRE_SCORING, 0 native calls, 0 uploads, 0 optimizer steps
    put_calls_a = []
    consume_calls_a = []
    monkeypatch.setattr(
        run_mod,
        "_s3_put_create_only_and_verify",
        lambda *a, **kw: put_calls_a.append((a, kw)),
    )
    orig_consume = run_mod.consume_single_fresh_context_in_inherited_engine
    monkeypatch.setattr(
        run_mod,
        "consume_single_fresh_context_in_inherited_engine",
        lambda *a, **kw: consume_calls_a.append((a, kw)),
    )

    out_hold = tmp_path / "out_hold_active"
    res_hold = run_mod.construct_and_consume_single_regression_context(
        out_hold,
        launch_obj,
        manifest_obj,
        deadline_spec=deadline_spec,
        use_train_padding_bridge=True,
        require_hold_release_for_live=True,
    )
    assert res_hold["status"] == "HOLD_ACTIVE_PRE_SCORING"
    assert res_hold["seam_qualified"] is False
    assert res_hold["optimizer_steps"] == 0
    assert res_hold["native_calls_total"] == 0
    assert (
        res_hold["train_padding_bridge"]["original_jtt1_nq0_sha256"]
        == stat.TARGET_REGRESSION_NQ0_JTT1_SHA256
    )
    assert (
        res_hold["train_padding_bridge"]["padded_jtt1_nq10_sha256"]
        == stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256
    )
    assert res_hold["context_sha256"] == stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256
    assert put_calls_a == []
    assert consume_calls_a == []

    # Control A2: Real bootstrap.s1main refuses nq == 0 when use_train_padding_bridge=False -> fail-closed BLOCKED_NQ0_PROTOCOL, 0 uploads, 0 optimizer steps
    out_a = tmp_path / "out_blocked_nq0"
    res_a = run_mod.construct_and_consume_single_regression_context(
        out_a,
        launch_obj,
        manifest_obj,
        deadline_spec=deadline_spec,
        use_train_padding_bridge=False,
    )
    assert res_a["status"] == "BLOCKED_NQ0_PROTOCOL"
    assert res_a["seam_qualified"] is False
    assert res_a["optimizer_steps"] == 0
    assert res_a["local_s1prod_fallback_used"] is False
    assert (
        res_a["pure_s3_train_csv_nq0_protocol_check"]["jtt1_nq0_sha256"]
        == "a6d997e39cec56995f579c380a0a1e960d3e887aebb5d40b8a0aae88f613483c"
    )
    assert res_a["pure_s3_train_csv_nq0_protocol_check"]["nt"] == 1000
    assert put_calls_a == []
    assert consume_calls_a == []

    # Create a deterministic nq=10-checking test worker script for Controls B and C (verifies padded nq=10 JTT1 is passed)
    mock_worker = tmp_path / "mock_s1main_nq10_ok.py"
    mock_worker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, hashlib, struct\n"
        "_, flag, jtt1_p, cand_p, out_p = sys.argv\n"
        "assert flag == '--fit-s1'\n"
        "raw = open(jtt1_p, 'rb').read()\n"
        "magic, nt, nq, p, k = struct.unpack('<4sIIIb', raw[:17])\n"
        "assert magic == b'JTT1' and nt == 1000 and nq == 10 and p == 5 and k == 2\n"
        "cand = json.loads(open(cand_p).read())\n"
        "h = int(hashlib.sha256(cand['dsl'].encode()).hexdigest()[:8], 16)\n"
        "r0 = 0.70 + (h % 150) * 0.001\n"
        "r1 = 0.72 + ((h >> 8) % 150) * 0.001\n"
        "ev = {\n"
        "  'status': 'completed',\n"
        "  'fold_protocol': 'jope.s1-two-fold/v1',\n"
        "  'inner_cv': [\n"
        "    {'metrics': {'rmse': r0}, 'baseline_metrics': {'rmse': 1.0}},\n"
        "    {'metrics': {'rmse': r1}, 'baseline_metrics': {'rmse': 1.0}},\n"
        "  ]\n"
        "}\n"
        "open(out_p, 'w').write(json.dumps(ev))\n",
        encoding="utf-8",
    )
    mock_worker.chmod(0o755)

    # Control B: Upload / VersionId replay failure occurs BEFORE consume_single_fresh_context_in_inherited_engine
    consume_calls_b = []
    monkeypatch.setattr(
        run_mod,
        "consume_single_fresh_context_in_inherited_engine",
        lambda *a, **kw: consume_calls_b.append((a, kw)),
    )

    def _failing_s3_put(
        bucket,
        key,
        src_path,
        aws_profile="veox-jepson-prod",
        replay_dest_path=None,
        deadline_spec=None,
    ):
        if key.endswith("pool.npz"):
            raise RuntimeError("simulated S3 VersionId replay failure on pool.npz")
        src_p = Path(src_path)
        if replay_dest_path is not None:
            Path(replay_dest_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, replay_dest_path)
        s = run_mod.sha(src_p)
        return dict(
            name=src_p.name,
            bucket=bucket,
            key=key,
            version_id=f"ver_{src_p.name}",
            size_bytes=int(src_p.stat().st_size),
            local_sha256=s,
            downloaded_version_sha256=s,
        )

    monkeypatch.setattr(run_mod, "_s3_put_create_only_and_verify", _failing_s3_put)
    with pytest.raises(RuntimeError, match="simulated S3 VersionId replay failure"):
        run_mod.construct_and_consume_single_regression_context(
            tmp_path / "out_fail_s3",
            launch_obj,
            manifest_obj,
            deadline_spec=deadline_spec,
            worker_bin=mock_worker,
            verify_worker_sha256=False,
            use_train_padding_bridge=True,
        )
    assert consume_calls_b == []

    # Control C: Full valid same-TRAIN bridge -> persist -> exact-VersionId replay -> consume control
    events_c = []

    def _recording_s3_put(
        bucket,
        key,
        src_path,
        aws_profile="veox-jepson-prod",
        replay_dest_path=None,
        deadline_spec=None,
    ):
        src_p = Path(src_path)
        events_c.append(f"s3_put:{src_p.name}")
        if replay_dest_path is not None:
            Path(replay_dest_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_p, replay_dest_path)
        s = run_mod.sha(src_p)
        return dict(
            name=src_p.name,
            bucket=bucket,
            key=key,
            version_id=f"ver_{src_p.name}",
            size_bytes=int(src_p.stat().st_size),
            local_sha256=s,
            downloaded_version_sha256=s,
        )

    def _recording_consume(
        engine, net, rt, fresh_row, historical_cids, deadline_spec=None
    ):
        events_c.append(f"consume:{Path(fresh_row['pool_npz_path']).parent.name}")
        return orig_consume(
            engine,
            net,
            rt,
            fresh_row,
            historical_cids=historical_cids,
            deadline_spec=deadline_spec,
        )

    monkeypatch.setattr(run_mod, "_s3_put_create_only_and_verify", _recording_s3_put)
    monkeypatch.setattr(
        run_mod,
        "consume_single_fresh_context_in_inherited_engine",
        _recording_consume,
    )

    out_c = tmp_path / "out_valid_same_train"
    res_c = run_mod.construct_and_consume_single_regression_context(
        out_c,
        launch_obj,
        manifest_obj,
        deadline_spec=deadline_spec,
        worker_bin=mock_worker,
        verify_worker_sha256=False,
        use_train_padding_bridge=True,
    )
    assert res_c["status"] == "PENDING_FINAL_ACCOUNTING"
    assert res_c["seam_qualified"] is False
    assert res_c["operation_succeeded"] is True
    assert res_c["context_sha256"] == stat.TARGET_SAME_S3_TRAIN_CONTEXT_SHA256
    assert res_c["context_sha256"] != stat.HISTORICAL_LOCAL_S1PROD_REPLAY_CONTEXT_SHA256
    assert res_c["s1_prod_context_bundle"]["train_rows"] == 1000
    assert (
        res_c["s1_prod_context_bundle"]["original_jtt1_nq0_sha256"]
        == stat.TARGET_REGRESSION_NQ0_JTT1_SHA256
    )
    assert (
        res_c["s1_prod_context_bundle"]["padded_jtt1_nq10_sha256"]
        == stat.TARGET_REGRESSION_PADDED_NQ10_JTT1_SHA256
    )
    assert (
        res_c["s1_prod_context_bundle"]["cut_sha256"]
        == stat.TARGET_REGRESSION_CUT_SHA256
    )
    assert res_c["s1_prod_context_bundle"]["padding_examples_added_to_learner"] == 0
    assert res_c["s1_prod_context_bundle"]["padding_families_added"] == 0
    assert res_c["s1_prod_context_bundle"]["candidates_count"] == 24
    assert res_c["s1_prod_context_bundle"]["evaluated_candidates_count"] == 24
    assert res_c["h206_readiness_claimed_from_tokens_alone"] is False
    assert res_c["family_novelty_ledger"]["padding_examples_added_to_learner"] == 0
    assert res_c["family_novelty_ledger"]["padding_families_added"] == 0
    assert (
        res_c["engine_consumption"]["s3_persisted_and_replayed_before_optimizer_step"]
        is True
    )
    assert res_c["engine_consumption"]["optimizer_steps"] == 1
    assert (
        res_c["engine_consumption"]["adapter_sha256_before"]
        != res_c["engine_consumption"]["adapter_sha256_after"]
    )
    assert events_c[:6] == [
        "s3_put:context_manifest.json",
        "s3_put:native_evaluator_receipt.json",
        "s3_put:pool.npz",
        "s3_put:features.npz",
        "s3_put:s3_pre_consumption_custody_receipt.json",
        "consume:verified_s3_replay",
    ]


def _h205_f5_sources():
    """Load exact source AST with stdlib only; never import the research/model modules."""
    import ast
    import types

    here = Path(__file__).resolve().parent
    tree = ast.parse((here / "monitor.py").read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import) and any(
                n.name == "psutil" for n in node.names
            ):
                continue
            keep.append(node)
    mon_ns = {"__file__": str(here / "monitor.py")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "monitor.py", "exec"), mon_ns)
    monitor = types.SimpleNamespace(**mon_ns)
    runner_tree = ast.parse((here / "runner.py").read_text())
    selected = [
        n
        for n in runner_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "sha",
            "_run_bounded_subprocess",
            "_s3_get_object_with_version",
            "_s3_put_create_only_and_verify",
            "consume_single_fresh_context_in_inherited_engine",
        }
    ]
    import os
    import tempfile

    runner = {
        "mon": types.SimpleNamespace(**mon_ns),
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "tempfile": tempfile,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "runner.py", "exec"), runner
    )
    return mon_ns, runner, runner_tree, monitor


def _h205_f5_ledger(tmp_path):
    state = {
        "schema": "h205-campaign-accounting/v1",
        "history_complete": True,
        "baseline_elapsed_seconds": 2021.039296863135,
        "campaign_cap_seconds": 25200.0,
        "source_evidence": ["SYNTHETIC_RECONCILED_HISTORY"],
        "completed": {},
        "pending": None,
    }
    path = tmp_path / "accounting.json"
    path.write_text(json.dumps(state))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.0])
def test_h205_f5_invalid_timeout_refuses_before_call(invalid):
    mon_ns, _, _, _ = _h205_f5_sources()
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )
    with pytest.raises(RuntimeError):
        mon_ns["remaining_subprocess_timeout"](101.0, deadline, invalid)


@pytest.mark.parametrize("elapsed", [1.0, 9.0, 10.0])
def test_h205_f5_native_aws_timeout_is_remaining_window(elapsed):
    import types

    mon_ns, run_ns, _, _ = _h205_f5_sources()
    clock = [100.0 + elapsed]
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )
    calls = []
    run_ns["time"] = types.SimpleNamespace(monotonic=lambda: clock[0])
    run_ns["subprocess"] = types.SimpleNamespace(
        run=lambda *a, **kw: calls.append(kw) or "OK"
    )
    if elapsed == 10.0:
        with pytest.raises(RuntimeError):
            run_ns["_run_bounded_subprocess"](["INERT"], deadline_spec=deadline)
        assert calls == []
    else:
        assert (
            run_ns["_run_bounded_subprocess"](["INERT"], deadline_spec=deadline) == "OK"
        )
        assert calls[0]["timeout"] == 10.0 - elapsed


def test_h205_f5_subprocess_late_completion_never_returns_success():
    import types

    mon_ns, run_ns, _, _ = _h205_f5_sources()
    clock = [109.0]
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )
    run_ns["time"] = types.SimpleNamespace(monotonic=lambda: clock[0])

    def fake_call(*args, **kwargs):
        assert kwargs["timeout"] == 1.0
        clock[0] = 111.0
        return "LATE_SUCCESS"

    run_ns["subprocess"] = types.SimpleNamespace(run=fake_call)
    with pytest.raises(RuntimeError):
        run_ns["_run_bounded_subprocess"](["INERT"], deadline_spec=deadline)


@pytest.mark.parametrize(
    "operation,delay", [("get", 0.1), ("put", 0.1), ("get", 2.0), ("put", 2.0)]
)
def test_h205_f5_actual_aws_helpers_keep_identity_and_bound_calls(
    tmp_path, operation, delay
):
    import types

    mon_ns, run_ns, _, _ = _h205_f5_sources()
    clock = [109.0]
    calls = []
    body = b"SYNTHETIC_ONLY"
    source = tmp_path / "source"
    source.write_bytes(body)
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )

    def fake_aws(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        assert 0.0 < kwargs["timeout"] <= 110.0 - clock[0]
        if "get-object" in command:
            Path(command[-1]).write_bytes(body)
        clock[0] += delay
        return types.SimpleNamespace(
            stdout=json.dumps({"VersionId": "SYNTHETIC_VERSION"})
        )

    run_ns["subprocess"] = types.SimpleNamespace(run=fake_aws)
    run_ns["time"] = types.SimpleNamespace(monotonic=lambda: clock[0])

    def invoke():
        if operation == "get":
            return run_ns["_s3_get_object_with_version"](
                "INERT", "INERT", tmp_path / "replay", deadline_spec=deadline
            )
        return run_ns["_s3_put_create_only_and_verify"](
            "INERT",
            "INERT",
            source,
            replay_dest_path=tmp_path / "replay",
            deadline_spec=deadline,
        )

    if delay > 1.0:
        with pytest.raises(RuntimeError):
            invoke()
    else:
        result = invoke()
        assert result["version_id"] == "SYNTHETIC_VERSION"
        if operation == "put":
            assert "--if-none-match" in calls[0][0] and "--version-id" in calls[1][0]
            assert result["local_sha256"] == hashlib.sha256(body).hexdigest()


def test_h205_f5_failed_terminal_commit_preserves_pending_and_blocks_restart(tmp_path):
    mon_ns, _, _, _ = _h205_f5_sources()
    path, digest = _h205_f5_ledger(tmp_path)
    mon_ns["begin_accounted_invocation"](path, digest, "crashed", 100.0)
    before = path.read_bytes()

    def failed_fsync(*args):
        raise OSError("SYNTHETIC_FSYNC_FAILURE")

    mon_ns["_accounting_save"] = failed_fsync
    with pytest.raises(OSError):
        mon_ns["finish_accounted_invocation"](path, "crashed", 111.0, "INCOMPLETE")
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError):
        mon_ns["begin_accounted_invocation"](
            path, hashlib.sha256(before).hexdigest(), "next", 120.0
        )


@pytest.mark.parametrize("bad", ["json", "flags"])
def test_h205_f5_bootstrap_refusal_is_unknown_not_a_zero_cost_pass(tmp_path, bad):
    import types

    mon_ns, _, _, _ = _h205_f5_sources()
    launch = tmp_path / "malformed.json"
    launch.write_text("{" if bad == "json" else "{}")
    output = tmp_path / "work"
    args = types.SimpleNamespace(
        output=output,
        launch=launch,
        smoke=bad == "flags",
        construct_single_regression_context=True,
    )
    parser = types.SimpleNamespace(
        add_argument=lambda *a, **kw: None, parse_args=lambda: args
    )
    mon_ns["argparse"] = types.SimpleNamespace(ArgumentParser=lambda: parser)
    mon_ns["PROCESS_START_MONOTONIC"] = 100.0
    mon_ns["time"] = types.SimpleNamespace(monotonic=lambda: 105.0)
    with pytest.raises(json.JSONDecodeError if bad == "json" else RuntimeError):
        mon_ns["main"]()
    refusal = json.loads(output.with_suffix(".admission-refusal.json").read_text())
    assert refusal["observed_elapsed_lower_bound_seconds"] == 5.0
    assert (
        refusal["terminal_tail_seconds"] is None
        and refusal["history_reconciliation_required"] is True
    )
    assert refusal["seam_qualified"] is False


@pytest.mark.parametrize(
    "field",
    [
        "process_start_monotonic",
        "prior_elapsed_seconds",
        "campaign_cap_seconds",
        "remaining_budget_seconds",
        "hard_deadline_monotonic",
    ],
)
def test_h205_f5_nonfinite_deadline_identity_refuses(field):
    mon_ns, _, _, _ = _h205_f5_sources()
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )
    deadline[field] = float("nan")
    with pytest.raises(RuntimeError):
        mon_ns["enforce_shared_process_deadline"](101.0, deadline)


@pytest.mark.parametrize("failure", [False, True])
def test_h205_f5_full_startup_failure_tail_and_once_only_carry(tmp_path, failure):
    import types

    mon_ns, _, _, _ = _h205_f5_sources()
    path, digest = _h205_f5_ledger(tmp_path)
    clock = [110.0]  # Ten seconds of imports/setup precede ledger admission.
    mon_ns["time"] = types.SimpleNamespace(monotonic=lambda: clock[0])
    launch = {
        "campaign_ledger_path": str(path),
        "campaign_ledger_sha256": digest,
        "accounting_invocation_id": "one",
        "remaining_wall_budget_seconds": 100.0,
    }

    def fake_observe(deadline, identity):
        assert deadline["process_start_monotonic"] == 100.0 and identity == "one"
        clock[0] += 3.0  # Fake forward/transport.
        clock[0] += 7.0  # Fake final upload, summary printing, child exit/log closure.
        if failure:
            raise ValueError("SYNTHETIC_FAILURE")
        return {"operation_succeeded": True, "exit_code": 0}

    if failure:
        with pytest.raises(ValueError, match="SYNTHETIC_FAILURE"):
            mon_ns["run_accounted_observation"](launch, 100.0, fake_observe)
    else:
        result = mon_ns["run_accounted_observation"](launch, 100.0, fake_observe)
        assert result["status"] == "PENDING_FINAL_ACCOUNTING"
        assert result["seam_qualified"] is False
    state = json.loads(path.read_text())
    record = state["completed"]["one"]
    assert record["elapsed_seconds"] == 20.0
    assert (
        state["history_complete"] is False
        and state["unreconciled_tail"]["seconds"] is None
    )
    before = path.read_bytes()
    assert (
        mon_ns["finish_accounted_invocation"](path, "one", 999.0, "ignored") == record
    )
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError):
        mon_ns["begin_accounted_invocation"](
            path, hashlib.sha256(before).hexdigest(), "two", 130.0
        )
    # An explicit synthetic owner reconciliation charges the known two-second
    # supervisor tail; it does not reset the prior window or create a production ledger.
    state["completed"]["one"]["elapsed_seconds"] += 2.0
    state["source_evidence"].append("SYNTHETIC_FULL_EXIT_OBSERVATION_PLUS_2S")
    state["history_complete"] = True
    state["unreconciled_tail"] = None
    path.write_text(json.dumps(state))
    next_admission = mon_ns["begin_accounted_invocation"](
        path, hashlib.sha256(path.read_bytes()).hexdigest(), "two", 130.0
    )
    assert next_admission["prior_elapsed_seconds"] == 2021.039296863135 + 22.0


@pytest.mark.parametrize(
    "bad", ["missing", "stale", "pending", "history", "nan", "floor", "cap"]
)
def test_h205_f5_unreconciled_or_invalid_ledger_refuses(tmp_path, bad):
    mon_ns, _, _, _ = _h205_f5_sources()
    path, digest = _h205_f5_ledger(tmp_path)
    state = json.loads(path.read_text())
    if bad == "missing":
        path.unlink()
    elif bad == "stale":
        digest = "0" * 64
    else:
        if bad == "pending":
            state["pending"] = {"invocation_id": "old", "process_start_monotonic": 1.0}
        elif bad == "history":
            state["history_complete"] = False
        elif bad == "nan":
            state["baseline_elapsed_seconds"] = float("nan")
        elif bad == "floor":
            state["baseline_elapsed_seconds"] = 0.0
        elif bad == "cap":
            state["campaign_cap_seconds"] = 99999.0
        path.write_text(json.dumps(state))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        mon_ns["begin_accounted_invocation"](path, digest, "new", 100.0)


def test_h205_f5_expiry_at_worker_exit_is_charged_and_unqualified(tmp_path):
    import types

    mon_ns, _, _, _ = _h205_f5_sources()
    path, digest = _h205_f5_ledger(tmp_path)
    clock = [100.0]
    mon_ns["time"] = types.SimpleNamespace(monotonic=lambda: clock[0])
    launch = {
        "campaign_ledger_path": str(path),
        "campaign_ledger_sha256": digest,
        "accounting_invocation_id": "late",
        "remaining_wall_budget_seconds": 10.0,
    }

    def late_exit(*args):
        clock[0] = 111.0
        return {"exit_code": 0, "operation_succeeded": True}

    with pytest.raises(RuntimeError):
        mon_ns["run_accounted_observation"](launch, 100.0, late_exit)
    state = json.loads(path.read_text())
    assert state["completed"]["late"]["elapsed_seconds"] == 11.0
    assert state["completed"]["late"]["status"] == "INCOMPLETE"


@pytest.mark.parametrize("delay", [0.1, 20.0])
def test_h205_f5_actual_final_upload_tail_checks_deadline_before_publication(
    tmp_path, delay
):
    import ast
    import types

    mon_ns, _, runner_tree, _ = _h205_f5_sources()
    function = next(
        n
        for n in runner_tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "construct_and_consume_single_regression_context"
    )
    begin = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "custody_upload" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "summary" for t in n.targets)
    )
    clock = [109.0]

    def fake_upload(*a, **kw):
        clock[0] += delay
        return {"SYNTHETIC": True}

    scope = dict(
        mon=types.SimpleNamespace(**mon_ns),
        time=types.SimpleNamespace(monotonic=lambda: clock[0]),
        deadline_spec=mon_ns["compute_shared_process_deadline"](
            100.0, requested_remaining_seconds=10.0
        ),
        _s3_put_create_only_and_verify=fake_upload,
        s3_bucket="INERT",
        s3_prefix="INERT",
        bundle_dir=tmp_path,
        replay_dir=tmp_path,
        t_start=100.0,
        stat=types.SimpleNamespace(
            verify_single_context_regression_seam_bundle=lambda *a, **kw: {
                "SYNTHETIC": True
            }
        ),
        nq0_enc={"csv_sha256": "fake", "train_matrix_sha256": "fake", "nt": 1},
        expected_eval_bin_sha="fake",
        roster={
            "forbidden_cids": [],
            "forbidden_families": [],
            "forbidden_datasets": [],
        },
    )
    code = compile(
        ast.Module(body=function.body[begin:end], type_ignores=[]),
        "actual_final_tail",
        "exec",
    )
    if delay > 1.0:
        with pytest.raises(RuntimeError):
            exec(code, scope)
    else:
        exec(code, scope)
        assert scope["verified_bundle"] == {"SYNTHETIC": True}
    assert (
        "summary" not in scope
    )  # No qualified result was constructed by this prepublication slice.


@pytest.mark.parametrize("expiry", ["forward", "backward", "none"])
def test_h205_f5_actual_consumer_pre_step_checks_without_model_or_optimizer(expiry):
    import contextlib
    import types

    mon_ns, run_ns, _, _ = _h205_f5_sources()
    clock = [100.0]
    steps = []

    class FakeParameter:
        requires_grad = True

        def __init__(self, value=1.0):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def clone(self):
            return FakeParameter(self.value)

        def numel(self):
            return 29211

        @property
        def device(self):
            return "INERT"

        def to(self, device):
            return self

        def copy_(self, other):
            self.value = other.value

    parameter = FakeParameter()

    class FakeLoss:
        def backward(self):
            if expiry == "backward":
                clock[0] = 111.0

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 0.5

    class FakeOptimizer:
        def zero_grad(self, **kwargs):
            pass

        def step(self):
            steps.append("FAKE_STEP")
            parameter.value += 1.0

    def forward(*args):
        if expiry == "forward":
            clock[0] = 111.0
        return "INERT"

    engine = types.SimpleNamespace(
        pools={"new": {"gate": [], "fitness": []}},
        guard=None,
        names=None,
        frozen=None,
        batch=lambda *a: ({}, None, None, None, None),
        anchored=lambda *a: None,
        forward=forward,
    )
    run_ns.update(
        time=types.SimpleNamespace(monotonic=lambda: clock[0]),
        np=types.SimpleNamespace(
            where=lambda *a: [], argsort=lambda *a, **kw: list(range(24))
        ),
        stat=types.SimpleNamespace(
            SLATE_CANDIDATES=24,
            SEEDS=[1],
            h177_analysis=types.SimpleNamespace(
                fit_slates=lambda *a: {"new": list(range(24))}
            ),
        ),
        torch=types.SimpleNamespace(
            optim=types.SimpleNamespace(AdamW=lambda *a, **kw: FakeOptimizer()),
            isfinite=lambda *a: types.SimpleNamespace(all=lambda: True),
            no_grad=contextlib.nullcontext,
        ),
        parameter_digest=lambda values: str(next(iter(values.values())).value),
        ingest_verified_fresh_rows_into_engine=lambda *a, **kw: {
            "historical_contexts_preserved": True
        },
    )
    args = (
        engine,
        types.SimpleNamespace(named_parameters=lambda: [("adapter", parameter)]),
        types.SimpleNamespace(
            check_frozen=lambda *a: None, DOPE="INERT", loss=lambda *a: FakeLoss()
        ),
        {"context_sha256": "new", "dataset_id": "synthetic", "family_id": "synthetic"},
        set(),
    )
    deadline = mon_ns["compute_shared_process_deadline"](
        100.0, requested_remaining_seconds=10.0
    )
    if expiry == "none":
        result = run_ns["consume_single_fresh_context_in_inherited_engine"](
            *args, deadline_spec=deadline
        )
        assert (
            result["optimizer_steps"] == 1
            and steps == ["FAKE_STEP"]
            and parameter.value == 1.0
        )
    else:
        with pytest.raises(RuntimeError):
            run_ns["consume_single_fresh_context_in_inherited_engine"](
                *args, deadline_spec=deadline
            )
        assert steps == [] and parameter.value == 1.0


def test_default_same_train_candidate_records_native_s1main_schema(tmp_path):
    records = run_mod._default_same_train_candidate_records(feature_columns=5)
    assert len(records) == stat.SLATE_CANDIDATES
    for rec in records:
        prog = rec["program"]
        assert set(prog.keys()) == {
            "dsl",
            "program_sha256",
            "birth_seq",
            "parents",
            "operator",
        }
        assert prog["dsl"].startswith("(pipe X ")
        assert (
            hashlib.sha256(prog["dsl"].encode("utf-8")).hexdigest()
            == prog["program_sha256"]
        )
        assert isinstance(prog["birth_seq"], int) and prog["birth_seq"] >= 0
        assert prog["parents"] == []
        assert prog["operator"] == "init_slate24"
    if stat.EVALUATOR_BINARY_PATH.is_file():
        rng = np.random.default_rng(205)
        x = rng.normal(size=(60, 5))
        y = 0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.1 * rng.normal(size=60)
        csv_p = tmp_path / "syn.csv"
        rows = [",".join(f"{v:.6f}" for v in list(r) + [yt]) for r, yt in zip(x, y)]
        csv_p.write_text("\n".join(rows) + "\n")
        nq0_p = tmp_path / "syn_nq0.jtt1"
        enc = stat.encode_train_only_jtt1_from_csv(csv_p, nq0_p, has_header=False)
        nq10_p = tmp_path / "syn_nq10.jtt1"
        stat.build_train_only_padded_native_jtt1(
            nq0_p,
            out_padded_jtt1_path=nq10_p,
            expected_nq0_sha256=enc["jtt1_sha256"],
            expected_cut_sha256=enc["cut_sha256"],
            expected_train_matrix_sha256=enc["train_matrix_sha256"],
            expected_nt=60,
            expected_p=5,
        )
        deadline = mon.compute_shared_process_deadline(
            process_start_monotonic=run_mod.time.monotonic()
        )
        res = run_mod.evaluate_slate24_live_s1main(
            nq10_p, records, deadline_spec=deadline
        )
        assert res["status"] == "PASS"
        assert res["evaluated_candidates"] == 24
        assert res["fitness_array"].shape == (24,)
        assert res["gate_array"].shape == (24,)
