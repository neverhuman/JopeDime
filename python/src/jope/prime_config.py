"""Single path authority for read-only references and Prime-owned outputs."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrimeConfig:
    legacy_access: str
    legacy_root: Path
    warm_start_checkpoint: Path
    jain_binary_reference: Path
    jain_binary: Path
    regression_dataset_root: Path
    binary_dataset_root: Path
    newmath_winners_dir: Path
    hyperion_binary_candidates: tuple[Path, ...]
    hyperion_weights: Path
    hyperion_manifest: Path
    validation_benchmark_root: Path
    fable_root: Path
    nomad_root: Path
    quant_models_root: Path
    foundation_fusion_python: Path
    target_maker_exporter: Path
    target_encoder_model: Path
    validation_dataset_manifest: Path
    root: Path
    registry_snapshot: Path
    generation31_vocab_snapshot: Path
    artifacts: Path
    runs: Path
    cache: Path
    ssh_control_dir: Path
    checkpoints: Path
    models: Path
    gpu_cache: Path
    remote_scratch_root: str
    mode: str
    discovery_backend: str
    promotion_backend: str
    user_artifact: str
    max_width: int


def _resolved(path: str | Path, *, relative_to: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    return candidate.resolve(strict=False)


def _owned_path(root: Path, value: str | Path, *, field: str) -> Path:
    candidate = _resolved(value, relative_to=root)
    if not candidate.is_relative_to(root):
        raise ValueError(f"prime.{field} must resolve below prime.root")
    return candidate


def load_config(path: str | Path | None = None) -> PrimeConfig:
    configured = path if path is not None else os.environ.get("JOPE_PRIME_CONFIG")
    config_path = (
        Path(configured)
        if configured is not None
        else Path(__file__).resolve().parents[2] / "prime.toml"
    )
    config_path = config_path.expanduser().resolve(strict=False)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    legacy = data["legacy"]
    prime = data["prime"]
    remote = data["remote"]
    evaluation = data["evaluation"]
    access = str(legacy.get("access", ""))
    if access != "read_only":
        raise ValueError("legacy.access must be 'read_only'")
    config_dir = config_path.parent
    root = _resolved(prime["root"], relative_to=config_dir)
    legacy_root = _resolved(legacy["root"], relative_to=config_dir)
    warm_start_checkpoint = _resolved(
        legacy["warm_start_checkpoint"], relative_to=config_dir
    )
    jain_binary_reference = _resolved(
        legacy["jain_binary_reference"], relative_to=config_dir
    )
    regression_dataset_root = _resolved(
        legacy["regression_dataset_root"], relative_to=config_dir
    )
    binary_dataset_root = _resolved(legacy["binary_dataset_root"], relative_to=config_dir)
    newmath_winners_dir = _resolved(legacy["newmath_winners_dir"], relative_to=config_dir)
    hyperion_binary_candidates = tuple(
        _resolved(value, relative_to=config_dir)
        for value in legacy["hyperion_binary_candidates"]
    )
    if not hyperion_binary_candidates:
        raise ValueError("legacy.hyperion_binary_candidates must not be empty")
    hyperion_weights = _resolved(legacy["hyperion_weights"], relative_to=config_dir)
    hyperion_manifest = _resolved(legacy["hyperion_manifest"], relative_to=config_dir)
    validation_benchmark_root = _resolved(
        legacy["validation_benchmark_root"], relative_to=config_dir
    )
    fable_root = _resolved(legacy["fable_root"], relative_to=config_dir)
    nomad_root = _resolved(legacy["nomad_root"], relative_to=config_dir)
    quant_models_root = _resolved(legacy["quant_models_root"], relative_to=config_dir)
    foundation_fusion_python = _resolved(
        legacy["foundation_fusion_python"], relative_to=config_dir
    )
    target_maker_exporter = _resolved(
        legacy["target_maker_exporter"], relative_to=config_dir
    )
    target_encoder_model = _resolved(
        legacy["target_encoder_model"], relative_to=config_dir
    )
    validation_dataset_manifest = _resolved(
        legacy["validation_dataset_manifest"], relative_to=config_dir
    )
    read_only_inputs = {
        "root": legacy_root,
        "warm_start_checkpoint": warm_start_checkpoint,
        "jain_binary_reference": jain_binary_reference,
        "regression_dataset_root": regression_dataset_root,
        "binary_dataset_root": binary_dataset_root,
        "newmath_winners_dir": newmath_winners_dir,
        "hyperion_weights": hyperion_weights,
        "hyperion_manifest": hyperion_manifest,
        "validation_benchmark_root": validation_benchmark_root,
        "fable_root": fable_root,
        "nomad_root": nomad_root,
        "quant_models_root": quant_models_root,
        "foundation_fusion_python": foundation_fusion_python,
        "target_maker_exporter": target_maker_exporter,
        "target_encoder_model": target_encoder_model,
        "validation_dataset_manifest": validation_dataset_manifest,
        **{
            f"hyperion_binary_candidates[{index}]": candidate
            for index, candidate in enumerate(hyperion_binary_candidates)
        },
    }
    for field, candidate in read_only_inputs.items():
        if candidate.is_relative_to(root) or root.is_relative_to(candidate):
            raise ValueError(f"legacy.{field} must not overlap prime.root")
    discovery_backend = str(evaluation.get("discovery_backend", "auto"))
    promotion_backend = str(
        evaluation.get("promotion_backend", evaluation.get("backend", "exact_f64"))
    )
    user_artifact = str(evaluation.get("user_artifact", "portable_python"))
    if discovery_backend not in {"auto", "fast_cuda", "fast_cpu", "exact"}:
        raise ValueError("evaluation.discovery_backend is invalid")
    if promotion_backend != "exact_f64":
        raise ValueError("evaluation.promotion_backend must be 'exact_f64'")
    if user_artifact != "portable_python":
        raise ValueError("evaluation.user_artifact must be 'portable_python'")
    return PrimeConfig(
        legacy_access=access,
        legacy_root=legacy_root,
        warm_start_checkpoint=warm_start_checkpoint,
        jain_binary_reference=jain_binary_reference,
        regression_dataset_root=regression_dataset_root,
        binary_dataset_root=binary_dataset_root,
        newmath_winners_dir=newmath_winners_dir,
        hyperion_binary_candidates=hyperion_binary_candidates,
        hyperion_weights=hyperion_weights,
        hyperion_manifest=hyperion_manifest,
        validation_benchmark_root=validation_benchmark_root,
        fable_root=fable_root,
        nomad_root=nomad_root,
        quant_models_root=quant_models_root,
        foundation_fusion_python=foundation_fusion_python,
        target_maker_exporter=target_maker_exporter,
        target_encoder_model=target_encoder_model,
        validation_dataset_manifest=validation_dataset_manifest,
        root=root,
        registry_snapshot=_owned_path(root, prime["registry_snapshot"], field="registry_snapshot"),
        generation31_vocab_snapshot=_owned_path(
            root,
            prime["generation31_vocab_snapshot"],
            field="generation31_vocab_snapshot",
        ),
        artifacts=_owned_path(root, prime["artifacts"], field="artifacts"),
        runs=_owned_path(root, prime["runs"], field="runs"),
        cache=_owned_path(root, prime["cache"], field="cache"),
        ssh_control_dir=_owned_path(
            root, prime["ssh_control_dir"], field="ssh_control_dir"
        ),
        checkpoints=_owned_path(root, prime["checkpoints"], field="checkpoints"),
        models=_owned_path(root, prime["models"], field="models"),
        jain_binary=_owned_path(root, prime["jain_binary"], field="jain_binary"),
        gpu_cache=_owned_path(root, prime["gpu_cache"], field="gpu_cache"),
        remote_scratch_root=_remote_scratch_root(remote["scratch_root"]),
        mode=str(prime["mode"]),
        discovery_backend=discovery_backend,
        promotion_backend=promotion_backend,
        user_artifact=user_artifact,
        max_width=int(evaluation["max_width"]),
    )


def _remote_scratch_root(value: str | Path) -> str:
    """Validate the configured POSIX prefix used only on worker hosts."""
    text = str(value).rstrip("/")
    if (
        not text.startswith("/")
        or text in {"", "/", "/tmp"}
        or text.startswith("/tmp/")
    ):
        raise ValueError(
            "remote.scratch_root must be a durable dedicated absolute path outside /tmp"
        )
    if any(character.isspace() for character in text):
        raise ValueError("remote.scratch_root must not contain whitespace")
    return text


def ssh_control_path(host: str, config: PrimeConfig | None = None) -> Path:
    """Return an owned SSH multiplexing socket path for a simple host name."""
    config = config or load_config()
    safe = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in str(host)
    )
    if not safe:
        raise ValueError("SSH host must be non-empty")
    config.ssh_control_dir.mkdir(parents=True, exist_ok=True)
    return config.ssh_control_dir / f"jope-ssh-{safe}.sock"
