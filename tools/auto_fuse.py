#!/usr/bin/env python3
"""Assemble released DOPE into the fused inference model and run real-data P2 checks.

Internal HPC harness using read-only sibling gpu_bench/ and JopeDimeMicro sources.
Run at nice 19. The P2 GPU phase takes qualification.lock nonblockingly.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import json
import os
import shutil
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


def tensor_state_sha(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"non-tensor state item: {name}")
        cpu = tensor.detach().cpu().contiguous()
        meta = json.dumps([name, str(cpu.dtype), list(cpu.shape)], separators=(",", ":")).encode()
        raw = cpu.view(torch.uint8).numpy().tobytes()
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


def import_runtime() -> tuple[Any, Any, Any]:
    for path in (FUSED, MICRO_PYTHON):
        if not path.is_dir():
            raise FileNotFoundError(f"required read-only runtime missing: {path}")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    fused = importlib.import_module("fused_jope_net")
    parity = importlib.import_module("run_parity_and_quick10_benchmark")
    cohort = importlib.import_module("run_batch_vram_and_cohort915_benchmark")
    return fused, parity, cohort


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
    fused, parity, cohort = import_runtime()

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
            self.load_state_dict(saved["model_state_dict"], strict=True)
            freeze_dope(self)
            if dope_sha(self) != freeze["state_sha256"]:
                raise RuntimeError("P2 evaluator loaded a different DOPE state")
            instances.append(self)

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir()
    parity.FUSED_DIR = eval_dir
    cohort.FUSED_DIR = eval_dir
    parity.FusedJopeNetwork = PinnedFusedNetwork
    cohort.FusedJopeNetwork = PinnedFusedNetwork
    cohort_manifest = GPU_BENCH / "infer_real_cohort_no_friedman_manifest.json"
    if file_sha(cohort_manifest) != COHORT_SHA256:
        raise RuntimeError("cohort915 manifest identity changed")

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
                parity.main()
                cohort.main()
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

    quick_path = eval_dir / "quick10_and_parity_report.json"
    cohort_path = eval_dir / "batch_sweep_and_cohort915_report.json"
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
        "trainable_parameter_counts": counts,
        "paired_holdout_r2_no_regression": "UNVERIFIED: no paired CPU-chain R2 receipt in this harness",
        "all10_language_selected_program_correctness": "UNVERIFIED: codegen timing is not correctness evidence",
    }
    if eval_reason is not None:
        p2.update({"status": "NOT_RUN", "reason": eval_reason})
    else:
        quick = json.loads(quick_path.read_text())
        cohort_report = json.loads(cohort_path.read_text())
        q = quick["quick10_parity_and_fitter_accuracy"]
        c = cohort_report["cohort915_summary"]["landmarks_on"]
        matches = int(c["exact_dsl_match_vs_chained_count"])
        paired = int(c["exact_dsl_match_vs_chained_paired_count"])
        quick_matches = int(q["default_path_lm_exact_dsl_matches"])
        quick_total = int(q["total_datasets"])
        passed = matches >= 903 and paired == 912 and quick_matches == 10 and quick_total == 10
        p2.update(
            {
                "status": "INCOMPLETE" if passed else "FAIL",
                "cohort912": {"matches": matches, "denominator": paired, "threshold": 903},
                "quick10": {"matches": quick_matches, "denominator": quick_total},
                "reports": {
                    "quick10": {"path": str(quick_path), "sha256": file_sha(quick_path)},
                    "cohort915": {"path": str(cohort_path), "sha256": file_sha(cohort_path)},
                },
            }
        )
    receipt = run_dir / "p2_eval_receipt.json"
    receipt.write_text(json.dumps(p2, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "jope.auto-fuse-manifest/v1",
        "dope_checkpoint_sha256": source_sha,
        "dope_state_sha256": freeze["state_sha256"],
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
