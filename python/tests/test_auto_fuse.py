"""Unit tests for the NS2 auto-fuse harness (`tools/auto_fuse.py`, NS24).

CPU-only synthetic fixtures: tensor-state hashing (0-D tensors, byte identity), the DOPE freeze
summary, the released-checkpoint loader's validation, and `main()` driven end to end through fake
gpu_bench modules (runtime identity, evaluator labels). The real GPU P2 evaluation is not exercised.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

TOOL = Path(__file__).resolve().parents[2] / "tools" / "auto_fuse.py"
# The tool points TMPDIR and the compiler caches at its lane target when imported; keep that out of
# the rest of the test session.
_TOOL_ENV = ("TMPDIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")


@pytest.fixture(scope="module")
def auto_fuse():
    saved = {key: os.environ.get(key) for key in _TOOL_ENV}
    spec = importlib.util.spec_from_file_location("auto_fuse_under_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


def _kat_state() -> dict[str, torch.Tensor]:
    return {
        "w": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 4,
        "idx": torch.tensor([[1, -2], [3, 4]], dtype=torch.int64),
        "mask": torch.tensor([True, False, True]),
        "half": torch.tensor([0.5, -1.25], dtype=torch.bfloat16),
        "bytes": torch.tensor([0, 255, 7], dtype=torch.uint8),
    }


def _reference_sha(state: dict[str, torch.Tensor]) -> str:
    """The digest format written out independently of the tool's byte view."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        meta = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode()
        start = cpu.storage_offset() * cpu.element_size()
        raw = bytes(cpu.untyped_storage().tolist())[start : start + cpu.numel() * cpu.element_size()]
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


class _ToyFused(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dope_model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
        self.head = nn.Linear(2, 1)
        self.scale = nn.Parameter(torch.ones(4))


def test_tensor_state_sha_is_unchanged_for_tensors_main_already_hashed(auto_fuse) -> None:
    # Known answer from tools/auto_fuse.py before the 0-D fix (main e3dfb2c94).
    expected = "c90ac156ea142b572a9cbf7af7e7323cd10964b782396bbee6ce54655a2a8fde"
    assert auto_fuse.tensor_state_sha(_kat_state()) == expected
    assert _reference_sha(_kat_state()) == expected


def test_tensor_state_sha_hashes_zero_dim_non_byte_tensors(auto_fuse) -> None:
    state = {
        "num_batches_tracked": torch.tensor(7, dtype=torch.int64),
        "temperature": torch.tensor(0.25, dtype=torch.float64),
        "flag": torch.tensor(True),
    }
    digest = auto_fuse.tensor_state_sha(state)
    assert digest == _reference_sha(state)
    # The shape is part of the identity: a 0-D tensor and its one-element vector differ.
    widened = dict(state, num_batches_tracked=torch.tensor([7], dtype=torch.int64))
    assert auto_fuse.tensor_state_sha(widened) != digest


def test_tensor_state_sha_binds_names_dtypes_and_values_not_order_or_layout(auto_fuse) -> None:
    state = _kat_state()
    digest = auto_fuse.tensor_state_sha(state)
    assert auto_fuse.tensor_state_sha(dict(reversed(list(state.items())))) == digest
    transposed = dict(state, w=state["w"].t().contiguous().t())
    assert not transposed["w"].is_contiguous()
    assert auto_fuse.tensor_state_sha(transposed) == digest

    renamed = {("weight" if k == "w" else k): v for k, v in state.items()}
    same_bytes_other_dtype = dict(state, idx=state["idx"].view(torch.float64))
    changed_value = dict(state, bytes=torch.tensor([0, 255, 8], dtype=torch.uint8))
    for variant in (renamed, same_bytes_other_dtype, changed_value):
        assert auto_fuse.tensor_state_sha(variant) != digest


def test_tensor_state_sha_rejects_non_tensor_items(auto_fuse) -> None:
    with pytest.raises(TypeError, match="^non-tensor state item: step$"):
        auto_fuse.tensor_state_sha({"w": torch.zeros(2), "step": 3})


def test_freeze_dope_hashes_batchnorm_state_and_counts_trainables(auto_fuse) -> None:
    torch.manual_seed(0)
    model = _ToyFused()
    model.train()
    freeze = auto_fuse.freeze_dope(model)

    assert freeze["state_sha256"] == auto_fuse.tensor_state_sha(model.dope_model.state_dict())
    assert freeze["state_sha256"] == _reference_sha(model.dope_model.state_dict())
    assert freeze["parameter_tensors"] == 4
    assert freeze["parameter_elements"] == 3 * 2 + 2 + 2 + 2
    # running_mean, running_var and the 0-D num_batches_tracked
    assert freeze["buffer_tensors"] == 3
    assert freeze["buffer_elements"] == 2 + 2 + 1
    assert not any(m.training for m in model.dope_model.modules())
    assert auto_fuse.trainable_counts(model) == {
        "dope_model": 0,
        "head": 3,
        "<root-only>": 4,
        "<total>": 7,
    }


def test_dope_sha_requires_a_dope_submodule(auto_fuse) -> None:
    with pytest.raises(RuntimeError, match="^fused model is missing dope_model$"):
        auto_fuse.dope_sha(nn.Linear(2, 2))


def _checkpoint(tmp_path: Path, monkeypatch, auto_fuse, envelope, *, pin: bool = True) -> Path:
    path = tmp_path / "dope.pt"
    torch.save(envelope, path)
    if pin:
        monkeypatch.setattr(auto_fuse, "RELEASED_DOPE_SHA256", auto_fuse.file_sha(path))
    return path


def test_load_released_dope_refuses_an_unpinned_checkpoint(tmp_path, monkeypatch, auto_fuse) -> None:
    path = _checkpoint(
        tmp_path, monkeypatch, auto_fuse, {"model_state_dict": {"w": torch.ones(2)}}, pin=False
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match=f"^refusing non-released DOPE checkpoint with SHA-256 {digest}$"):
        auto_fuse.load_released_dope(path)


def test_load_released_dope_returns_a_finite_pinned_state(tmp_path, monkeypatch, auto_fuse) -> None:
    state = dict(_kat_state(), num_batches_tracked=torch.tensor(3, dtype=torch.int64))
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, {"model_state_dict": state, "epoch": 4})
    loaded, digest = auto_fuse.load_released_dope(path)
    assert digest == auto_fuse.file_sha(path)
    assert auto_fuse.tensor_state_sha(loaded) == auto_fuse.tensor_state_sha(state)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.complex64])
def test_load_released_dope_refuses_non_finite_tensors(tmp_path, monkeypatch, auto_fuse, bad, dtype) -> None:
    poisoned = torch.tensor([1.0, bad, 2.0]).to(dtype)
    state = {"a": torch.ones(2), "z": poisoned, "m": torch.tensor(float("nan")), "idx": torch.arange(3)}
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, {"model_state_dict": state})
    with pytest.raises(ValueError, match="^released DOPE checkpoint has non-finite tensors: m, z$"):
        auto_fuse.load_released_dope(path)


@pytest.mark.parametrize(
    "envelope",
    [
        {"model_state_dict": {}},
        {"model_state_dict": {"w": torch.ones(2), "step": 3}},
        {"state_dict": {"w": torch.ones(2)}},
        [torch.ones(2)],
    ],
    ids=["empty", "non-tensor-item", "missing-key", "not-a-dict"],
)
def test_load_released_dope_refuses_malformed_envelopes(tmp_path, monkeypatch, auto_fuse, envelope) -> None:
    path = _checkpoint(tmp_path, monkeypatch, auto_fuse, envelope)
    with pytest.raises(ValueError, match="^released DOPE checkpoint has no tensor model_state_dict$"):
        auto_fuse.load_released_dope(path)


def test_runtime_identity_hashes_every_input(tmp_path, monkeypatch, auto_fuse) -> None:
    files = {"fused/a.py": tmp_path / "a.py", "micro/b.pt": tmp_path / "b.pt"}
    files["fused/a.py"].write_bytes(b"print(1)\n")
    files["micro/b.pt"].write_bytes(bytes(range(256)))
    monkeypatch.setattr(auto_fuse, "RUNTIME_INPUTS", files)
    assert auto_fuse.runtime_identity() == {
        name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
    }


def test_runtime_identity_refuses_missing_inputs(tmp_path, monkeypatch, auto_fuse) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    files = {
        "fused/z.so": tmp_path / "z.so",
        "fused/a.py": tmp_path / "a.py",
        "micro/m.pt": tmp_path / "m.pt",
    }
    monkeypatch.setattr(auto_fuse, "RUNTIME_INPUTS", files)
    with pytest.raises(
        FileNotFoundError, match="^required read-only runtime inputs missing: fused/z.so, micro/m.pt$"
    ):
        auto_fuse.runtime_identity()


def test_runtime_inputs_are_the_read_only_selection_inputs(auto_fuse) -> None:
    names = {name.split("/", 1)[1] for name in auto_fuse.RUNTIME_INPUTS}
    # The gpu_bench Phase 2 submission pins (DOPE, the cohort manifest and the chained reference are
    # pinned separately), the Phase 2 harness and the helper module it imports.
    assert {
        "fused_jope_net.py",
        "gpu_stage1a_and_slate.py",
        "fused_gpu_kernels.cu",
        "libfused_gpu_kernels.so",
        "distilled_stage1a_bridge_real_s3.pt",
        "bridge_provenance_audit.json",
        "gpu_slate_tables.pt",
        "expanded_gpu_bank_1766.pt",
        "reviewer_infer_emit_rust_only",
        "flywheel_winner_pool_v3a_q25_train_only_430.pt",
        "h45_full_s3_15151_gate_disjoint_all_dl_train_1100_10tok.pt",
        "h40b_round2_dope_self_contained_10tok.pt",
        "frozen_stage1_stage2_inference_params_v3.pt",
        "run_phase2_shard_and_fleet.py",
        "run_parity_and_quick10_benchmark.py",
    } <= names
    for name, path in auto_fuse.RUNTIME_INPUTS.items():
        root = auto_fuse.FUSED if name.startswith("fused/") else auto_fuse.MICRO_ASSETS
        assert path.is_relative_to(root) and path.name == name.split("/", 1)[1]
        assert not path.is_relative_to(auto_fuse.REPO)
    assert auto_fuse.CHAINED_REPORT.is_relative_to(auto_fuse.GPU_BENCH)
    assert len(set(auto_fuse.QUICK10_DATASET_IDS)) == 10


def _chained(ids_ok: list[str], ids_failed: list[str]) -> dict:
    rows = [{"dataset_id": i, "status": "ok", "selected_dsl": f"(pipe X (ridge) #{i})"} for i in ids_ok]
    rows += [{"dataset_id": i, "status": "failed", "selected_dsl": None} for i in ids_failed]
    return {"datasets": rows}


def _phase2_rows(chained: dict, wrong: set[str]) -> list[dict]:
    rows = []
    for c in chained["datasets"]:
        pick = c["selected_dsl"] or "(pipe X (ridge_gcv))"
        if c["dataset_id"] in wrong:
            pick = "(pipe X (histgbm depth=4))"
        rows.append(
            {"dataset_id": c["dataset_id"], "selected_dsl_h45_fp16": pick, "selected_dsl_h45_fp32": pick}
        )
    return rows


def _cohort_ids(auto_fuse) -> tuple[list[str], list[str]]:
    others = [f"ds{i:04d}" for i in range(912 - len(auto_fuse.QUICK10_DATASET_IDS))]
    return list(auto_fuse.QUICK10_DATASET_IDS) + others, ["failed0", "failed1", "failed2"]


def test_count_matches_counts_paired_h45_picks_and_quick10(auto_fuse) -> None:
    ok, failed = _cohort_ids(auto_fuse)
    chained = _chained(ok, failed)
    wrong = {ok[0], ok[20], ok[21]}
    rows = _phase2_rows(chained, wrong)
    rows[30]["selected_dsl_h45_fp32"] = "(pipe X (ridge lambda_frac=0.1))"
    counted = auto_fuse.count_matches(rows, chained)
    assert counted == {
        "paired": 912,
        "fp16": 909,
        "fp32": 908,
        "quick10": 9,
        "mismatched_ids": sorted(wrong),
    }


def test_count_matches_refuses_repeated_or_missing_quick10_rows(auto_fuse) -> None:
    ok, failed = _cohort_ids(auto_fuse)
    chained = _chained(ok, failed)
    rows = _phase2_rows(chained, set())
    with pytest.raises(ValueError, match="^Phase 2 report repeats a dataset$"):
        auto_fuse.count_matches(rows + rows[-1:], chained)
    with pytest.raises(ValueError, match="^Phase 2 report is missing quick10 datasets$"):
        auto_fuse.count_matches(rows[1:], chained)


class _FakeFusedNet(nn.Module):
    def __init__(self, *, adapter_variant: str = "h40b", device=None) -> None:
        super().__init__()
        self.adapter_variant = adapter_variant
        self.dope_model = nn.Sequential(nn.Linear(3, 2), nn.BatchNorm1d(2))
        self.adapter_10tok = nn.Linear(2, 2)
        with torch.no_grad():
            self.adapter_10tok.weight.fill_(45.0 if adapter_variant == "h45" else 40.0)


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch, auto_fuse):
    """`main()` wired to a fake gpu_bench Phase 2 harness, with every side effect kept under tmp_path."""
    root = tmp_path / "runtime"
    root.mkdir()
    inputs = {f"fused/input{i}.bin": root / f"input{i}.bin" for i in range(3)}
    for i, path in enumerate(inputs.values()):
        path.write_bytes(bytes([i]) * 64)
    bank = root / "expanded_gpu_bank_1766.pt"
    torch.save({"bank": torch.zeros(2)}, bank)
    inputs["fused/expanded_gpu_bank_1766.pt"] = bank
    manifest = root / "infer_real_cohort_no_friedman_manifest.json"
    manifest.write_text("{}\n")
    ok, failed = _cohort_ids(auto_fuse)
    chained = _chained(ok, failed)
    chained_path = root / "chained.json"
    chained_path.write_text(json.dumps(chained))

    torch.manual_seed(0)
    released = _FakeFusedNet(adapter_variant="h45").dope_model.state_dict()
    checkpoint = tmp_path / "dope.pt"
    torch.save({"model_state_dict": released}, checkpoint)

    state: dict[str, object] = {
        "during_shard": None,
        "wrong": set(),
        "harness_fp16": None,
        "drop_first_row": False,
        "calls": [],
    }
    built: list[float] = []
    fused = types.ModuleType("fake_fused_jope_net")
    fused.EXPANDED_BANK_PATH = bank
    fused.FusedJopeNetwork = _FakeFusedNet
    phase2 = types.ModuleType("fake_run_phase2_shard_and_fleet")

    def run_shard(shard_index: int, host_name: str, scratch_dir: Path, num_shards: int = 3) -> None:
        state["calls"].append((shard_index, host_name, num_shards))
        for variant in ("h45", "h40b"):
            net = phase2.FusedJopeNetwork(adapter_variant=variant, device=None)
            built.append(float(net.adapter_10tok.weight.detach()[0, 0]))
        if state["during_shard"] is not None:
            state["during_shard"]()
        rows = _phase2_rows(chained, state["wrong"])
        if state["drop_first_row"]:  # the first row is a quick10 dataset
            rows = rows[1:]
        fp16 = 912 - len(state["wrong"]) if state["harness_fp16"] is None else state["harness_fp16"]
        npz = scratch_dir / f"single_gpu_915_{host_name}_holdout_preds.npz"
        npz.write_bytes(b"npz")
        report = {
            "paired_chained_compared_count": 912,
            "paired_exact_dsl_match_h45_fp16_count": fp16,
            "holdout_preds_npz_path": str(npz),
            "holdout_preds_npz_sha256": auto_fuse.file_sha(npz),
            "datasets": rows,
        }
        (scratch_dir / f"single_gpu_915_{host_name}_report.json").write_text(json.dumps(report))

    phase2.run_shard = run_shard
    out = tmp_path / "out"
    for name, value in {
        "RUNTIME_INPUTS": inputs,
        "RELEASED_DOPE_SHA256": auto_fuse.file_sha(checkpoint),
        "GPU_BENCH": root,
        "COHORT_SHA256": auto_fuse.file_sha(manifest),
        "CHAINED_REPORT": chained_path,
        "CHAINED_REPORT_SHA256": auto_fuse.file_sha(chained_path),
        "OUTPUT_ROOT": out,
        "QUALIFICATION_LOCK": tmp_path / "qualification.lock",
        "import_runtime": lambda: (fused, phase2),
    }.items():
        monkeypatch.setattr(auto_fuse, name, value)
    monkeypatch.setattr(auto_fuse.socket, "gethostname", lambda: "xbabe9")
    monkeypatch.setattr(auto_fuse.os, "nice", lambda increment: 0)
    monkeypatch.setattr(auto_fuse.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(auto_fuse.torch.cuda, "is_available", lambda: True)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setattr(sys, "argv", ["tools/auto_fuse.py", str(checkpoint)])

    def run() -> tuple[dict, dict]:
        assert auto_fuse.main() == 0
        (run_dir,) = out.iterdir()
        receipt = json.loads((run_dir / "p2_eval_receipt.json").read_text())
        manifest_out = json.loads((run_dir / "auto_fuse_manifest.json").read_text())
        return receipt, manifest_out

    return types.SimpleNamespace(run=run, inputs=inputs, state=state, built=built, ok=ok)


def test_main_runs_the_phase2_shard_and_binds_identity_and_labels(fake_runtime, auto_fuse) -> None:
    expected = {name: auto_fuse.file_sha(path) for name, path in fake_runtime.inputs.items()}
    receipt, manifest = fake_runtime.run()

    assert fake_runtime.state["calls"] == [(0, "xbabe9", 1)]
    assert receipt["status"] == "INCOMPLETE" and receipt["evaluation_ran"] is True
    assert receipt["host"] == "xbabe9"
    assert receipt["cohort912"] == {
        "matches": 912,
        "denominator": 912,
        "threshold": 903,
        "fp32_router_matches": 912,
        "mismatched_dataset_ids": [],
    }
    assert receipt["quick10"] == {"matches": 10, "denominator": 10}
    assert receipt["harness_counts"] == {"paired": 912, "fp16": 912} and receipt["harness_counts_agree"]
    assert receipt["reports"]["phase2_single915"]["path"].endswith("single_gpu_915_xbabe9_report.json")
    assert receipt["reports"]["holdout_predictions"]["sha256"] == hashlib.sha256(b"npz").hexdigest()
    assert receipt["runtime_identity_sha256"] == manifest["runtime_identity_sha256"] == expected
    assert receipt["runtime_identity_unchanged"] is True
    assert receipt["dope_checksum_unchanged"] is True
    # The harness builds h45 and h40b networks; both run the fused h45 artifact.
    assert receipt["p2_evaluator_constructed_adapter_variants"] == ["h45", "h40b"]
    assert receipt["p2_evaluator_state"] == "fused_h45 artifact"
    assert fake_runtime.built == [45.0, 45.0]


def test_main_fails_the_gate_below_903_and_names_the_misses(fake_runtime) -> None:
    wrong = set(fake_runtime.ok[100:110])
    fake_runtime.state["wrong"] = wrong
    receipt, _ = fake_runtime.run()

    assert receipt["status"] == "FAIL"
    assert receipt["cohort912"]["matches"] == 902
    assert receipt["cohort912"]["mismatched_dataset_ids"] == sorted(wrong)
    assert receipt["quick10"] == {"matches": 10, "denominator": 10}


def test_main_fails_when_the_harness_count_disagrees(fake_runtime) -> None:
    fake_runtime.state["wrong"] = {fake_runtime.ok[500]}
    fake_runtime.state["harness_fp16"] = 912
    receipt, _ = fake_runtime.run()

    assert receipt["status"] == "FAIL"
    assert receipt["cohort912"]["matches"] == 911
    assert receipt["harness_counts"]["fp16"] == 912 and receipt["harness_counts_agree"] is False


def test_main_refuses_a_runtime_that_changes_during_evaluation(fake_runtime) -> None:
    target = fake_runtime.inputs["fused/input1.bin"]
    fake_runtime.state["during_shard"] = lambda: target.write_bytes(b"phase-1 tree")
    receipt, _ = fake_runtime.run()

    assert receipt["status"] == "NOT_RUN" and receipt["evaluation_ran"] is False
    assert receipt["reason"] == "gpu_bench runtime inputs changed during evaluation"
    assert receipt["runtime_identity_unchanged"] is False
    assert "cohort912" not in receipt


def test_main_reports_an_unreadable_phase2_report_as_not_run(fake_runtime) -> None:
    fake_runtime.state["drop_first_row"] = True
    receipt, _ = fake_runtime.run()

    assert receipt["status"] == "NOT_RUN" and receipt["evaluation_ran"] is False
    assert (
        receipt["reason"]
        == "Phase 2 report unreadable: ValueError: Phase 2 report is missing quick10 datasets"
    )
    assert "cohort912" not in receipt


def test_main_refuses_a_changed_chained_reference_before_evaluating(fake_runtime, auto_fuse) -> None:
    auto_fuse.CHAINED_REPORT.write_text("{}")
    with pytest.raises(RuntimeError, match="^chained cohort915 reference identity changed$"):
        auto_fuse.main()
    assert fake_runtime.state["calls"] == []


def test_main_refuses_a_missing_runtime_input_before_evaluating(fake_runtime, auto_fuse) -> None:
    fake_runtime.inputs["fused/input2.bin"].unlink()
    with pytest.raises(
        FileNotFoundError, match="^required read-only runtime inputs missing: fused/input2.bin$"
    ):
        auto_fuse.main()
    assert fake_runtime.built == []
