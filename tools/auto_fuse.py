#!/usr/bin/env python3
"""Assemble released DOPE into the fused inference model and run real-data P2 checks.

Internal HPC harness using read-only sibling gpu_bench/ and JopeDimeMicro sources. P2 runs gpu_bench's
Phase 2 harness (`run_phase2_shard_and_fleet.run_shard`, one shard over all 915 datasets), the code path
behind the signed 912/912 receipts, with every network it builds pinned to this run's fused artifact.
Run at nice 19 with an interpreter that has torch and tiktoken. The P2 GPU phase takes
qualification.lock nonblockingly.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import shutil
import socket
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()
GPU_BENCH = HOME / "gpu_bench"
FUSED = GPU_BENCH / "fused"
MICRO_PYTHON = HOME / "JopeDimeMicro" / "python"
OUTPUT_ROOT = REPO / "target" / "auto-fuse"
QUALIFICATION_LOCK = HOME / "jope_plan" / "qualification.lock"
RELEASED_DOPE_SHA256 = "a36510b9e63050c01d182ccce7e4b489d9fb6ed7e39f8f2948cff420114e8952"
COHORT_SHA256 = "95b3f54042e5f8bb405080b8e169bf0af33008a0f86ce66477479635ce7e4139"
CHAINED_REPORT = GPU_BENCH / "runs" / "infer_v1_cohort915" / "infer_v1_cohort915_combined_report.json"
# The pinned external chained (5-net CPU) selections that every match is counted against.
CHAINED_REPORT_SHA256 = "9478a6f2c374455cb1cc3578eb09e34e17da2e4120f6da057574b3337c36bdca"
# quick10 (gpu_bench reviewer_infer_quick10_manifest.json, sha256 7e6229ab...): all ten are cohort915 members.
QUICK10_DATASET_IDS = (
    "2cac5f3ad025291f",
    "d50018658be71d1e",
    "f2d83d253c91dffe",
    "c6c5a1e929d039b9",
    "dbc501fc7f6a40b5",
    "c02354d05cfc50ca",
    "6d6c749c3ff27710",
    "1125c70822c5b44b",
    "4c562924720c9e19",
    "a76d20c7ab4f79b3",
)
COHORT_GATE = (903, 912)
MICRO_ASSETS = HOME / "JopeDimeMicro" / "assets"
# Read-only inputs that decide the fused selection (the gpu_bench Phase 2 submission pins, the harness and
# its helpers). The receipt records their bytes so it can be joined to a signed runtime identity: the
# 2026-10-29 run scored 893/912 on the Phase 1 tree and its receipt could not say so.
RUNTIME_INPUTS = {
    "fused/fused_jope_net.py": FUSED / "fused_jope_net.py",
    "fused/gpu_stage1a_and_slate.py": FUSED / "gpu_stage1a_and_slate.py",
    "fused/fused_gpu_kernels.cu": FUSED / "fused_gpu_kernels.cu",
    "fused/libfused_gpu_kernels.so": FUSED / "libfused_gpu_kernels.so",
    "fused/distilled_stage1a_bridge_real_s3.pt": FUSED / "distilled_stage1a_bridge_real_s3.pt",
    "fused/gpu_slate_tables.pt": FUSED / "gpu_slate_tables.pt",
    "fused/expanded_gpu_bank_1766.pt": FUSED / "expanded_gpu_bank_1766.pt",
    "fused/bridge_provenance_audit.json": FUSED / "bridge_provenance_audit.json",
    "fused/reviewer_infer_emit_rust_only": FUSED / "reviewer_infer_emit_rust_only",
    "fused/run_phase2_shard_and_fleet.py": FUSED / "run_phase2_shard_and_fleet.py",
    "fused/run_parity_and_quick10_benchmark.py": FUSED / "run_parity_and_quick10_benchmark.py",
    "micro/frozen_stage1_stage2_inference_params_v3.pt": MICRO_ASSETS
    / "projections"
    / "frozen_stage1_stage2_inference_params_v3.pt",
    "micro/flywheel_winner_pool_v3a_q25_train_only_430.pt": MICRO_ASSETS
    / "models"
    / "flywheel_winner_pool_v3a_q25_train_only_430.pt",
    "micro/h45_full_s3_15151_gate_disjoint_all_dl_train_1100_10tok.pt": MICRO_ASSETS
    / "models"
    / "h45_full_s3_15151_gate_disjoint_all_dl_train_1100_10tok.pt",
    "micro/h40b_round2_dope_self_contained_10tok.pt": MICRO_ASSETS
    / "models"
    / "h40b_round2_dope_self_contained_10tok.pt",
}

# Keep compiler, Triton, and temporary outputs inside this lane's reusable target.
TORCH_TMP = REPO / "target" / "auto-fuse-tmp"
TORCH_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(TORCH_TMP)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(REPO / "target" / "torchinductor")
os.environ["TRITON_CACHE_DIR"] = str(REPO / "target" / "triton")

import torch  # noqa: E402
from torch import nn  # noqa: E402


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_identity() -> dict[str, str]:
    # A missing slate table would also be rebuilt into gpu_bench, so refuse before any work.
    missing = sorted(name for name, path in RUNTIME_INPUTS.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(f"required read-only runtime inputs missing: {', '.join(missing)}")
    return {name: file_sha(path) for name, path in RUNTIME_INPUTS.items()}


def tensor_state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"non-tensor state item: {name}")
        cpu = tensor.detach().cpu().contiguous()
        meta = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode()
        # A byte view of a 0-D tensor with a wider dtype (BatchNorm's int64 num_batches_tracked) is
        # refused, so flatten first; the bytes of every other tensor stay the same and the shape is in meta.
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        h.update(len(meta).to_bytes(8, "little"))
        h.update(meta)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def dope_sha(model: nn.Module) -> str:
    if not hasattr(model, "dope_model") or not isinstance(model.dope_model, nn.Module):
        raise RuntimeError("fused model is missing dope_model")
    return tensor_state_sha(model.dope_model.state_dict())


def freeze_dope(model: nn.Module) -> dict[str, Any]:
    dope = model.dope_model
    for p in dope.parameters():
        p.requires_grad_(False)
        p.grad = None
    dope.eval()
    if any(p.requires_grad or p.grad is not None for p in dope.parameters()):
        raise RuntimeError("DOPE parameters are not frozen")
    if any(m.training for m in dope.modules()):
        raise RuntimeError("a DOPE submodule is not in eval mode")
    return {
        "state_sha256": dope_sha(model),
        "parameter_tensors": len(list(dope.parameters())),
        "parameter_elements": sum(p.numel() for p in dope.parameters()),
        "buffer_tensors": len(list(dope.buffers())),
        "buffer_elements": sum(b.numel() for b in dope.buffers()),
    }


def trainable_counts(model: nn.Module) -> dict[str, int]:
    counts = {
        name: sum(p.numel() for p in child.parameters() if p.requires_grad)
        for name, child in model.named_children()
    }
    counts["<root-only>"] = sum(
        p.numel() for name, p in model.named_parameters() if "." not in name and p.requires_grad
    )
    counts["<total>"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return counts


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        dfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_runtime() -> tuple[Any, Any]:
    for path in (FUSED, MICRO_PYTHON):
        if not path.is_dir():
            raise FileNotFoundError(f"required read-only runtime missing: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    fused = importlib.import_module("fused_jope_net")
    phase2 = importlib.import_module("run_phase2_shard_and_fleet")
    return fused, phase2


def count_matches(rows: list[dict[str, Any]], chained: dict[str, Any]) -> dict[str, Any]:
    """Exact DSL matches of the h45 picks against the chained selections, over the paired datasets."""
    reference = {
        r["dataset_id"]: r["selected_dsl"] for r in chained.get("datasets", []) if r.get("status") == "ok"
    }
    ids = [r["dataset_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Phase 2 report repeats a dataset")
    paired = [r for r in rows if r["dataset_id"] in reference]
    quick = [r for r in rows if r["dataset_id"] in QUICK10_DATASET_IDS]
    if len(quick) != len(QUICK10_DATASET_IDS):
        raise ValueError("Phase 2 report is missing quick10 datasets")
    return {
        "paired": len(paired),
        "fp16": sum(r["selected_dsl_h45_fp16"] == reference[r["dataset_id"]] for r in paired),
        "fp32": sum(r["selected_dsl_h45_fp32"] == reference[r["dataset_id"]] for r in paired),
        "quick10": sum(r["selected_dsl_h45_fp16"] == reference.get(r["dataset_id"]) for r in quick),
        "mismatched_ids": sorted(
            r["dataset_id"] for r in paired if r["selected_dsl_h45_fp16"] != reference[r["dataset_id"]]
        ),
    }


def load_released_dope(path: Path) -> tuple[dict[str, torch.Tensor], str]:
    path = path.resolve(strict=True)
    digest = file_sha(path)
    if digest != RELEASED_DOPE_SHA256:
        raise ValueError(f"refusing non-released DOPE checkpoint with SHA-256 {digest}")
    envelope = torch.load(path, map_location="cpu", weights_only=True)
    state = envelope.get("model_state_dict") if isinstance(envelope, dict) else None
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError("released DOPE checkpoint has no tensor model_state_dict")
    nonfinite = sorted(
        name
        for name, value in state.items()
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())
    )
    if nonfinite:
        raise ValueError(f"released DOPE checkpoint has non-finite tensors: {', '.join(nonfinite)}")
    return state, digest


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: tools/auto_fuse.py <pinned_released_dope_checkpoint>")
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    torch.set_num_threads(4)

    checkpoint = Path(sys.argv[1])
    run_dir = OUTPUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    source_state, source_sha = load_released_dope(checkpoint)
    source_file_sha_before = file_sha(checkpoint)
    runtime_before = runtime_identity()
    fused, phase2 = import_runtime()

    # Keep dynamic bank-registration writes inside this run's output directory.
    bank = run_dir / "expanded_gpu_bank_1766.pt"
    shutil.copy2(fused.EXPANDED_BANK_PATH, bank)
    fused.EXPANDED_BANK_PATH = bank
    model = fused.FusedJopeNetwork(adapter_variant="h45", device=torch.device("cpu"))
    model.dope_model.load_state_dict(source_state, strict=True)
    freeze = freeze_dope(model)
    if freeze["state_sha256"] != tensor_state_sha(source_state):
        raise RuntimeError("loaded DOPE state differs from released input")
    counts = trainable_counts(model)
    if counts.get("dope_model") != 0:
        raise RuntimeError("DOPE parameters remain trainable")

    artifact = run_dir / "fused_h45.pt"
    payload = {
        "schema": "jope.fused-inference-checkpoint/v1",
        "model_class": "FusedJopeNetwork",
        "adapter_variant": "h45",
        "dope_checkpoint_sha256": source_sha,
        "dope_state_sha256": freeze["state_sha256"],
        "model_state_dict": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        "trainable_parameter_counts": counts,
    }
    atomic_torch_save(payload, artifact)
    artifact_sha = file_sha(artifact)
    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    if saved.get("dope_state_sha256") != freeze["state_sha256"]:
        raise RuntimeError("fused checkpoint readback has a DOPE identity mismatch")

    instances: list[nn.Module] = []

    class PinnedFusedNetwork(fused.FusedJopeNetwork):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.constructed_adapter_variant = getattr(self, "adapter_variant", None)
            self.load_state_dict(saved["model_state_dict"], strict=True)
            freeze_dope(self)
            if dope_sha(self) != freeze["state_sha256"]:
                raise RuntimeError("P2 evaluator loaded a different DOPE state")
            instances.append(self)

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir()
    # run_shard writes only under its scratch directory; its gpu_bench paths stay read-only inputs.
    phase2.FusedJopeNetwork = PinnedFusedNetwork
    cohort_manifest = GPU_BENCH / "infer_real_cohort_no_friedman_manifest.json"
    if file_sha(cohort_manifest) != COHORT_SHA256:
        raise RuntimeError("cohort915 manifest identity changed")
    if file_sha(CHAINED_REPORT) != CHAINED_REPORT_SHA256:
        raise RuntimeError("chained cohort915 reference identity changed")
    host = socket.gethostname()
    phase2_report = eval_dir / f"single_gpu_915_{host}_report.json"

    eval_reason = None
    try:
        with QUALIFICATION_LOCK.open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("qualification.lock is held; refusing to wait for GPU") from exc
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("P2 requires allocated CUDA; CPU fallback is refused")
                phase2.run_shard(0, host, eval_dir, num_shards=1)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except Exception as exc:
        eval_reason = f"{type(exc).__name__}: {exc}"

    dope_checksum_main_after = dope_sha(model)
    source_file_sha_after = file_sha(checkpoint)
    if dope_checksum_main_after != freeze["state_sha256"]:
        eval_reason = eval_reason or "DOPE state changed in the assembled model"
    if source_file_sha_after != source_file_sha_before or source_file_sha_after != source_sha:
        eval_reason = eval_reason or "released DOPE checkpoint file changed during evaluation"
    try:
        runtime_after: dict[str, str] | None = runtime_identity()
    except FileNotFoundError as exc:
        runtime_after = None
        eval_reason = eval_reason or f"{type(exc).__name__}: {exc}"
    if runtime_after is not None and runtime_after != runtime_before:
        eval_reason = eval_reason or "gpu_bench runtime inputs changed during evaluation"

    counted: dict[str, Any] = {}
    report: dict[str, Any] = {}
    if eval_reason is None:
        try:
            report = json.loads(phase2_report.read_text())
            counted = count_matches(report["datasets"], json.loads(CHAINED_REPORT.read_text()))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            eval_reason = f"Phase 2 report unreadable: {type(exc).__name__}: {exc}"

    p2: dict[str, Any] = {
        "schema": "jope.auto-fuse-p2-evaluation/v1",
        "evaluation_ran": eval_reason is None,
        "utc": datetime.now(UTC).isoformat(),
        "dope_checkpoint_sha256": source_sha,
        "fused_checkpoint_sha256": artifact_sha,
        "dope_checksum_before": freeze["state_sha256"],
        "dope_checksums_after": {
            "assembled_model": dope_checksum_main_after,
            "p2_evaluator_models": [dope_sha(m) for m in instances],
        },
        "dope_checksum_unchanged": (
            dope_checksum_main_after == freeze["state_sha256"]
            and all(dope_sha(m) == freeze["state_sha256"] for m in instances)
        ),
        "source_file_sha256_before": source_file_sha_before,
        "source_file_sha256_after": source_file_sha_after,
        "source_file_unchanged": source_file_sha_after == source_file_sha_before == source_sha,
        "runtime_identity_sha256": runtime_before,
        "runtime_identity_unchanged": runtime_after == runtime_before,
        # Every evaluator instance carries this run's fused h45 artifact, including the one the harness
        # builds as h40b.
        "p2_evaluator_constructed_adapter_variants": [m.constructed_adapter_variant for m in instances],
        "p2_evaluator_state": "fused_h45 artifact",
        "trainable_parameter_counts": counts,
        "paired_holdout_r2_no_regression": "UNVERIFIED: the shard harness retains GPU holdout predictions; "
        "the paired CPU-chain comparison is a separate step",
        "all10_language_selected_program_correctness": "UNVERIFIED: codegen timing is not correctness evidence",
    }
    if eval_reason is not None:
        p2.update({"status": "NOT_RUN", "reason": eval_reason})
    else:
        threshold, denominator = COHORT_GATE
        # The harness counts the same pairs itself; both counts must agree.
        harness = {
            "paired": report.get("paired_chained_compared_count"),
            "fp16": report.get("paired_exact_dsl_match_h45_fp16_count"),
        }
        agree = harness == {"paired": counted["paired"], "fp16": counted["fp16"]}
        passed = (
            agree
            and counted["paired"] == denominator
            and counted["fp16"] >= threshold
            and counted["quick10"] == len(QUICK10_DATASET_IDS)
        )
        p2.update(
            {
                "status": "INCOMPLETE" if passed else "FAIL",
                "harness": "gpu_bench run_phase2_shard_and_fleet.run_shard(0, host, num_shards=1)",
                "host": host,
                "cohort912": {
                    "matches": counted["fp16"],
                    "denominator": counted["paired"],
                    "threshold": threshold,
                    "fp32_router_matches": counted["fp32"],
                    "mismatched_dataset_ids": counted["mismatched_ids"],
                },
                "quick10": {"matches": counted["quick10"], "denominator": len(QUICK10_DATASET_IDS)},
                "harness_counts": harness,
                "harness_counts_agree": agree,
                "reports": {
                    "phase2_single915": {"path": str(phase2_report), "sha256": file_sha(phase2_report)},
                    "holdout_predictions": {
                        "path": report.get("holdout_preds_npz_path"),
                        "sha256": report.get("holdout_preds_npz_sha256"),
                    },
                },
            }
        )
    receipt = run_dir / "p2_eval_receipt.json"
    receipt.write_text(json.dumps(p2, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "jope.auto-fuse-manifest/v1",
        "dope_checkpoint_sha256": source_sha,
        "dope_state_sha256": freeze["state_sha256"],
        "runtime_identity_sha256": runtime_before,
        "fused_checkpoint": str(artifact),
        "fused_checkpoint_sha256": artifact_sha,
        "trainable_parameter_counts": counts,
        "dope_freeze": freeze,
        "p2_receipt": str(receipt),
        "p2_status": p2["status"],
    }
    (run_dir / "auto_fuse_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"fused_checkpoint": str(artifact), "p2_receipt": str(receipt), "p2_status": p2["status"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
