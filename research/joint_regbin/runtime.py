"""Source-pinned, read-only R882 initialization for the joint reg/bin worker.

This module loads the accepted H40B R882 implementation and weights, freezes
DOPE and every non-adapter parameter, and exposes the existing task heads/core
with an explicitly supplied optimizer and candidate-pool packer. It does not
admit role manifests, open dataset payloads, choose optimizer hyperparameters,
or start a study.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch
from torch import nn

from . import candidate_builder, protocol, worker

REPO_ROOT = Path(__file__).resolve().parents[2]
H144_DIR = REPO_ROOT / "research/flywheel_selfplay/h144_reg_sweep"
H144_MANIFEST = H144_DIR / "manifest.json"
H144_MANIFEST_SHA256 = (
    "ee12c095a27dff630cffaa79d72f0de2a06e348fb9dfd9b685d5c81feabd7a09"
)
H144_PREFLIGHT = H144_DIR / "preflight.json"
H144_PREFLIGHT_SHA256 = (
    "cd969f1f3c1cae4804d2c35c462939a93c4976946e248980aeb2ecca2302a785"
)
H144_RUNTIME = H144_DIR / "runtime.py"
H144_RUNTIME_SHA256 = "24d4eac98da23791f706341fd6c7c3f54cc5f693b0b435852049b319dd996ecc"
R882_CHECKPOINT = Path("/home/ubuntu/jope-data/H141-t1815/r882_init.pt")
R882_COPY_RECEIPT = Path(
    "/home/ubuntu/jope_plan/evidence/H141/t1815/r882_copy_receipt.json"
)
R882_COPY_RECEIPT_SHA256 = (
    "0abd37c6eee7b811a51a43d6e5b6e593e3bf743f62d446563cc4c5efe9740740"
)
R882_CHECKPOINT_SHA256 = (
    "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
)
DOPE_SHA256 = protocol.DOPE
FUSED_SOURCE = Path("/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py")
FUSED_SOURCE_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
EXPANDED_BANK_SIDECAR = FUSED_SOURCE.with_name("expanded_gpu_bank_1766.pt")
MICRO_PYTHON = Path("/home/ubuntu/JopeDimeMicro/python")
EXPECTED_PYTHON = "3.12.3"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_TORCH = "2.7.0"
TRAINABLE_ADAPTER_PARAMETERS = 29_211
TRAINABLE_ADAPTER_TENSORS = 47


class SourcePinError(ValueError):
    """Raised when a runtime input differs from the reviewed source bundle."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_source_sha256() -> str:
    digest = hashlib.sha256()
    for name, path in (
        ("runtime.py", Path(__file__)),
        ("worker.py", Path(worker.__file__)),
        ("candidate_builder.py", Path(candidate_builder.__file__)),
        ("protocol.py", Path(protocol.__file__)),
        ("admission.py", Path(candidate_builder.admission.__file__)),
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def verify_file_pin(path: Path, expected_sha256: str) -> str:
    """Verify a regular, non-symlink file before its contents are loaded."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SourcePinError(f"pinned runtime file is missing or not regular: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SourcePinError(f"pinned runtime file hash mismatch: {path}")
    return actual


def _runtime_manifest() -> dict:
    verify_file_pin(H144_MANIFEST, H144_MANIFEST_SHA256)
    try:
        manifest = json.loads(H144_MANIFEST.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePinError("pinned H144 source manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("wave") != "H144":
        raise SourcePinError("pinned H144 source manifest schema mismatch")
    if manifest.get("executor_sha256", {}).get("runtime.py") != H144_RUNTIME_SHA256:
        raise SourcePinError("H144 manifest does not pin its initialization runtime")
    return manifest


def _verify_runtime_closure(manifest: dict) -> dict:
    """Verify only model-runtime code/assets; do not inspect H144 study inputs."""
    verify_file_pin(H144_RUNTIME, H144_RUNTIME_SHA256)
    inputs = manifest.get("inputs", {})
    if inputs.get(str(FUSED_SOURCE)) != FUSED_SOURCE_SHA256:
        raise SourcePinError("H144 manifest does not pin the fused model source")
    verify_file_pin(FUSED_SOURCE, FUSED_SOURCE_SHA256)
    if EXPANDED_BANK_SIDECAR.exists() or EXPANDED_BANK_SIDECAR.is_symlink():
        raise SourcePinError("unregistered fused-model candidate-bank sidecar exists")

    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise SourcePinError("H144 model runtime asset pins are absent")
    verified_assets = {}
    for raw_path, expected in sorted(assets.items()):
        path = Path(raw_path)
        if not path.is_absolute() or not isinstance(expected, str):
            raise SourcePinError("H144 model runtime asset pin is malformed")
        verified_assets[str(path)] = verify_file_pin(path, expected)
    return verified_assets


def _load_pinned_h144_runtime() -> ModuleType:
    """Load H144 initialization code only from the bytes covered by its pin."""
    if H144_RUNTIME.is_symlink() or not H144_RUNTIME.is_file():
        raise SourcePinError("pinned H144 runtime file is missing or not regular")
    source = H144_RUNTIME.read_bytes()
    if hashlib.sha256(source).hexdigest() != H144_RUNTIME_SHA256:
        raise SourcePinError("pinned H144 runtime hash mismatch")
    module = ModuleType("jrb01_h144_runtime_pinned")
    module.__file__ = str(H144_RUNTIME)
    exec(compile(source, str(H144_RUNTIME), "exec"), module.__dict__)
    return module


def _load_pinned_fused_source() -> ModuleType:
    """Load fused model code only from bytes covered by its source pin."""
    if FUSED_SOURCE.is_symlink() or not FUSED_SOURCE.is_file():
        raise SourcePinError("pinned fused source is missing or not regular")
    source = FUSED_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != FUSED_SOURCE_SHA256:
        raise SourcePinError("pinned fused source hash mismatch")
    module = ModuleType("h144_fused")
    module.__file__ = str(FUSED_SOURCE)
    sys.modules[module.__name__] = module
    exec(compile(source, str(FUSED_SOURCE), "exec"), module.__dict__)
    return module


def _bind_pinned_fused_loader(h144_runtime: ModuleType) -> None:
    """Replace H144's path reread with the source-pinned in-memory loader."""
    if not callable(getattr(h144_runtime, "load_fused", None)):
        raise SourcePinError("pinned H144 runtime has no fused-source loader")
    h144_runtime.load_fused = _load_pinned_fused_source


def _expose_pinned_model_namespace() -> None:
    """Expose H144's already-verified research modules beside this namespace."""
    micro_research = MICRO_PYTHON / "research"
    if not micro_research.is_dir() or micro_research.is_symlink():
        raise SourcePinError("pinned model-code research package is unavailable")
    import research

    package_path = getattr(research, "__path__", None)
    if package_path is None:
        raise SourcePinError("product research namespace cannot load pinned model code")
    if str(micro_research) not in {str(path) for path in package_path}:
        package_path.append(str(micro_research))


def _verify_runtime_versions() -> dict:
    import numpy

    verify_file_pin(H144_PREFLIGHT, H144_PREFLIGHT_SHA256)
    preflight = json.loads(H144_PREFLIGHT.read_text())
    python = ".".join(map(str, sys.version_info[:3]))
    numpy_version = numpy.__version__
    torch_version = torch.__version__.split("+", 1)[0]
    if (python, numpy_version, torch_version) != (
        EXPECTED_PYTHON,
        EXPECTED_NUMPY,
        EXPECTED_TORCH,
    ):
        raise SourcePinError(
            "Python/NumPy/PyTorch versions differ from H144 runtime pin"
        )
    constructor = preflight.get("constructor", {})
    pinned_packages = constructor.get("packages", {})
    if (
        pinned_packages.get("numpy") != numpy_version
        or pinned_packages.get("torch") != torch_version
    ):
        raise SourcePinError("H144 preflight package versions differ")
    if not str(constructor.get("python", "")).startswith(python + " "):
        raise SourcePinError("H144 preflight Python build differs")
    return {"python": python, "numpy": numpy_version, "torch": torch_version}


def _verify_checkpoint_copy(path: Path) -> None:
    verify_file_pin(R882_COPY_RECEIPT, R882_COPY_RECEIPT_SHA256)
    try:
        receipt = json.loads(R882_COPY_RECEIPT.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePinError("R882 custody receipt is unreadable") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("H141_init_copy") != str(path)
        or receipt.get("sha256") != R882_CHECKPOINT_SHA256
        or receipt.get("source_live_directory_access") is not False
        or receipt.get("model_loaded") is not False
        or receipt.get("GPU_started") is not False
    ):
        raise SourcePinError("R882 copy does not match the pinned custody receipt")


def _configure_adapter_trainability(net: nn.Module) -> tuple[str, ...]:
    names = []
    for name, parameter in net.named_parameters():
        trainable = name.startswith("adapter_10tok.") and not name.startswith(
            "adapter_10tok.anchor."
        )
        parameter.requires_grad_(trainable)
        if trainable:
            names.append(name)
    if (
        len(names) != TRAINABLE_ADAPTER_TENSORS
        or sum(net.state_dict()[name].numel() for name in names)
        != TRAINABLE_ADAPTER_PARAMETERS
    ):
        raise SourcePinError("R882 trainable adapter inventory mismatch")
    return tuple(names)


class SourcePinnedWorker:
    """R882-initialized model and task heads, ready for an admitted packer."""

    def __init__(
        self,
        net: nn.Module,
        guard: ModuleType,
        h144_runtime: ModuleType,
        trainable_names: tuple[str, ...],
        frozen_state_sha256: str,
        source_pins: dict,
    ):
        self.net = net
        self.adapter = net.adapter_10tok
        self.heads = worker.TwoTaskHeads(shared_width=1)
        self.guard = guard
        self.h144_runtime = h144_runtime
        self.trainable_names = trainable_names
        self.frozen_state_sha256 = frozen_state_sha256
        self.source_pins = source_pins
        self.initial_adapter_sha256 = worker.adapter_fingerprint(self.adapter)
        self.verify_frozen_state()

    def verify_frozen_state(self) -> None:
        self.h144_runtime.check_frozen(
            self.net,
            self.guard,
            self.trainable_names,
            self.frozen_state_sha256,
        )
        snapshot = self.guard.snapshot_dope_weights(self.net)
        if snapshot.composite_sha256 != DOPE_SHA256:
            raise SourcePinError("frozen DOPE identity changed")
        actual = tuple(
            name for name, value in self.net.named_parameters() if value.requires_grad
        )
        if actual != self.trainable_names:
            raise SourcePinError("trainable parameter surface changed after R882 load")

    def build_core(
        self,
        optimizer: torch.optim.Optimizer,
        packer: Callable[[dict], tuple[dict, torch.Tensor]],
    ) -> "FrozenWorkerCore":
        _verify_candidate_packer(packer, self.source_pins)
        expected = [
            parameter
            for parameter in (*self.adapter.parameters(), *self.heads.parameters())
            if parameter.requires_grad
        ]
        _verify_optimizer_parameters(optimizer, expected)
        return FrozenWorkerCore(
            self, worker.WorkerCore(self.adapter, self.heads, optimizer, packer)
        )

    def make_candidate_packer(
        self,
        pools_by_context_id: dict[str, dict],
        tokens_by_context_id: dict[str, torch.Tensor],
        masks_by_context_id: dict[str, torch.Tensor],
        *,
        device: str | torch.device = "cpu",
    ) -> Callable[[dict], tuple[dict, torch.Tensor]]:
        """Bind candidate pools to this exact read-only R882/DOPE initialization."""
        return candidate_builder.make_worker_packer(
            pools_by_context_id,
            tokens_by_context_id,
            masks_by_context_id,
            checkpoint_sha256=self.source_pins["checkpoint_sha256"],
            dope_sha256=self.source_pins["dope_sha256"],
            device=device,
        )


class FrozenWorkerCore:
    """WorkerCore wrapper that checks frozen DOPE/model state around each call."""

    def __init__(self, runtime: SourcePinnedWorker, core: worker.WorkerCore):
        self.runtime = runtime
        self.core = core

    def train_batch(self, rows: list[dict]) -> dict:
        self.runtime.verify_frozen_state()
        try:
            result = self.core.train_batch(rows)
        finally:
            self.runtime.verify_frozen_state()
        result["dope_sha256"] = DOPE_SHA256
        result["worker_source_sha256"] = self.runtime.source_pins[
            "worker_source_sha256"
        ]
        return result

    def predict(self, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
        self.runtime.verify_frozen_state()
        try:
            result = self.core.predict(row)
        finally:
            self.runtime.verify_frozen_state()
        return result


def _verify_optimizer_parameters(
    optimizer: torch.optim.Optimizer, expected: list[nn.Parameter]
) -> None:
    expected_ids = [id(parameter) for parameter in expected]
    actual_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if (
        len(expected_ids) != len(set(expected_ids))
        or len(actual_ids) != len(set(actual_ids))
        or set(actual_ids) != set(expected_ids)
    ):
        raise ValueError(
            "optimizer must contain exactly the trainable adapter and task head"
        )


def _verify_candidate_packer(packer: Callable, source_pins: dict) -> None:
    expected = {
        "checkpoint_sha256": source_pins["checkpoint_sha256"],
        "dope_sha256": source_pins["dope_sha256"],
    }
    if not callable(packer) or getattr(packer, "_jrb_source_pins", None) != expected:
        raise ValueError("candidate-pool packer is not bound to this R882/DOPE source")


def initialize_r882_worker(
    checkpoint_path: Path = R882_CHECKPOINT,
) -> SourcePinnedWorker:
    """Load the exact read-only R882 model and frozen DOPE source closure on CPU."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_sha = verify_file_pin(checkpoint_path, R882_CHECKPOINT_SHA256)
    if checkpoint_sha != R882_CHECKPOINT_SHA256:
        raise SourcePinError("initial checkpoint is not the pinned R882")
    _verify_checkpoint_copy(checkpoint_path)
    versions = _verify_runtime_versions()
    manifest = _runtime_manifest()
    assets = _verify_runtime_closure(manifest)
    h144_runtime = _load_pinned_h144_runtime()
    _bind_pinned_fused_loader(h144_runtime)
    _expose_pinned_model_namespace()
    net, guard, h144_names, frozen_sha, fused = h144_runtime.build(checkpoint_path)
    verify_file_pin(checkpoint_path, R882_CHECKPOINT_SHA256)
    _verify_runtime_closure(manifest)
    if fused.__file__ != str(FUSED_SOURCE):
        raise SourcePinError("H144 loader did not use the pinned fused model source")
    trainable_names = _configure_adapter_trainability(net)
    if trainable_names != tuple(h144_names):
        raise SourcePinError("JRB and H144 trainable adapter inventories differ")
    snapshot = guard.snapshot_dope_weights(net)
    if snapshot.composite_sha256 != DOPE_SHA256:
        raise SourcePinError("loaded R882 does not contain the frozen DOPE identity")
    source_pins = {
        "checkpoint_sha256": checkpoint_sha,
        "dope_sha256": snapshot.composite_sha256,
        "h144_manifest_sha256": H144_MANIFEST_SHA256,
        "h144_runtime_sha256": H144_RUNTIME_SHA256,
        "fused_source_sha256": FUSED_SOURCE_SHA256,
        "runtime_assets": assets,
        "runtime_versions": versions,
        "trainable_adapter_tensors": len(trainable_names),
        "trainable_adapter_parameters": TRAINABLE_ADAPTER_PARAMETERS,
        "worker_source_sha256": _worker_source_sha256(),
    }
    return SourcePinnedWorker(
        net,
        guard,
        h144_runtime,
        trainable_names,
        frozen_sha,
        source_pins,
    )
