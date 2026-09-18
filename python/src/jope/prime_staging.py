"""Prime-owned workspaces for read-only legacy datasets."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .prime_config import PrimeConfig, load_config

_OWNED_DIRECTORY_NAMES = ("reports", "outputs")


@dataclass(frozen=True)
class StagedDataset:
    source_dir: Path
    dataset_dir: Path
    reports_dir: Path
    outputs_dir: Path
    staged: bool


def prime_owned_path(path: str | Path, config: PrimeConfig | None = None) -> Path:
    """Resolve ``path`` and reject any destination outside ``prime.root``."""
    config = config or load_config()
    root = config.root.resolve(strict=False)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise ValueError(f"destination must resolve below Prime root {root}")
    return candidate


def ensure_prime_owned_directory(
    path: str | Path, config: PrimeConfig | None = None
) -> Path:
    """Create a real Prime-owned directory, never an output symlink."""
    config = config or load_config()
    requested = Path(path).expanduser()
    if not requested.is_absolute():
        requested = config.root.resolve(strict=False) / requested
    if requested.is_symlink():
        raise ValueError(f"Prime-owned directory must not be a symlink: {requested}")
    destination = prime_owned_path(requested, config)
    destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir():
        raise ValueError(f"Prime-owned directory is not a directory: {destination}")
    return destination


def _link_input(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve(strict=False) == source.resolve(strict=True):
            return
        raise ValueError(f"staged input symlink collision: {destination}")
    if destination.exists():
        raise ValueError(f"staged input path already exists: {destination}")
    try:
        destination.symlink_to(source)
    except FileExistsError:  # another staging worker won the same deterministic race
        if not destination.is_symlink() or (
            destination.resolve(strict=False) != source.resolve(strict=True)
        ):
            raise ValueError(f"staged input path collision: {destination}") from None


def stage_read_only_dataset(
    source_dir: str | Path,
    *,
    config: PrimeConfig | None = None,
    destination: str | Path | None = None,
    output_names: Iterable[str] = (),
) -> StagedDataset:
    """Stage top-level regular inputs while keeping every writable path Prime-owned.

    Legacy directories and known output filenames are never linked. Repeated calls
    reuse the deterministic workspace without replacing prior owned outputs.
    """
    config = config or load_config()
    source = Path(source_dir).expanduser().resolve(strict=True)
    if not source.is_dir():
        raise ValueError(f"dataset source is not a directory: {source}")

    cache = prime_owned_path(config.cache, config)
    key = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:16]
    if destination is None:
        workspace = cache / "staged_datasets" / key / source.name
    else:
        requested = Path(destination).expanduser()
        workspace = requested if requested.is_absolute() else cache / requested
    requested_workspace = workspace
    resolved_workspace = prime_owned_path(requested_workspace, config)
    if not resolved_workspace.is_relative_to(cache):
        raise ValueError(f"staged dataset must resolve below Prime cache {cache}")
    if (
        resolved_workspace == source
        or resolved_workspace.is_relative_to(source)
        or source.is_relative_to(resolved_workspace)
    ):
        raise ValueError("staged dataset source and destination must not overlap")
    workspace = ensure_prime_owned_directory(requested_workspace, config)

    excluded = set(output_names) | set(_OWNED_DIRECTORY_NAMES)
    for entry in sorted(source.iterdir(), key=lambda path: path.name):
        if entry.name in excluded:
            continue
        try:
            mode = entry.stat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            _link_input(entry, workspace / entry.name)

    reports = ensure_prime_owned_directory(workspace / "reports", config)
    outputs = ensure_prime_owned_directory(workspace / "outputs", config)
    return StagedDataset(
        source_dir=source,
        dataset_dir=workspace,
        reports_dir=reports,
        outputs_dir=outputs,
        staged=True,
    )


def prime_dataset_workspace(
    source_dir: str | Path,
    *,
    config: PrimeConfig | None = None,
    output_names: Iterable[str] = (),
) -> StagedDataset:
    """Use an owned dataset in place, otherwise stage its read-only inputs."""
    config = config or load_config()
    source = Path(source_dir).expanduser().resolve(strict=True)
    root = config.root.resolve(strict=False)
    if source.is_relative_to(root):
        dataset = prime_owned_path(source, config)
        reports = ensure_prime_owned_directory(dataset / "reports", config)
        outputs = ensure_prime_owned_directory(dataset / "outputs", config)
        return StagedDataset(source, dataset, reports, outputs, staged=False)
    return stage_read_only_dataset(
        source,
        config=config,
        output_names=output_names,
    )
