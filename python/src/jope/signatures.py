"""Signature service: generate the Hyperion `.vs` sidecars for any dataset dir.

Wraps the ``hyperion_vs`` Rust CLI (Hyperion MatrixGhostUnified encoder) with the exact
pinned flags/seed/weights that produced the deployed regression signatures, so we can make
signatures for datasets that don't have them yet — chiefly the ~24.8k binary datasets and
any synthetic data.

Anti-corruption invariant (the hyp30/hyp128 lesson): a sidecar that EXISTS but whose row
count differs from ``train.csv`` is **corrupt, not missing**. ``verify_sidecars`` reports it
as corrupt; ``make_signatures`` refuses to feed corrupt sidecars downstream and never
silently overwrites a good one without ``force``. Generation is idempotent (a complete dir
is skipped).

Cluster fan-out is deterministic host-sharding (``host_index``/``host_count``) — this is
per-dataset batch work, NOT the ssh-lane bridge protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .prime_config import PrimeConfig, load_config
from .prime_staging import (
    ensure_prime_owned_directory,
    prime_dataset_workspace,
    prime_owned_path,
)

# The feature-enabled CUDA build from the deployed pipeline (5s/dataset on GPU); the
# repo's default target/release/hyperion_vs is a feature-less stub. Override via env.
_PRIME_CONFIG = load_config()
_CUDA_HINTS = [str(path) for path in _PRIME_CONFIG.hyperion_binary_candidates]
HYPERION_WEIGHTS = str(_PRIME_CONFIG.hyperion_weights)
HYPERION_MANIFEST = str(_PRIME_CONFIG.hyperion_manifest)
HYPERION_SEED = 20260623
HYPERION_SWEEPS = 3
HYPERION_SCALES = "4,7,11,15"
SIGNATURE_ENCODER = "hyperion_vs.v5_armB.s3.sc4-7-11-15.seed20260623"

ROW_TRAIN = "hyperion3train128dim.vs"
ROW_TEST = "hyperion3test128dim.vs"
CELL_TRAIN = "train3.vs"
CELL_TEST = "test3.vs"
ROW_DIM = 128
SIGNATURE_MANIFEST = "signature_manifest.json"


def hyperion_binary() -> str:
    env = os.environ.get("JOPE_HYPERION_VS_BIN")
    if env and Path(env).exists():
        return env
    for hint in _CUDA_HINTS:
        if Path(hint).exists():
            return hint
    return _CUDA_HINTS[-1]


@dataclass
class SignatureReport:
    dataset_dir: str
    dataset_hash: str
    status: str  # complete | generated | failed | skipped
    errors: list[str] = field(default_factory=list)
    rows_train: int | None = None
    timings_ms: float | None = None


def csv_row_count(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def _vs_shape(path: Path) -> tuple[int, int]:
    """(rows, cols) of a .vs file from its first line + line count (cheap)."""
    rows = 0
    cols = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if rows == 0:
                cols = len(line.split(","))
            rows += 1
    return rows, cols


def verify_sidecars(dataset_dir: Path, *, deep: bool = False) -> dict[str, Any]:
    """Report missing vs corrupt sidecars. A wrong-row-count file is CORRUPT, not missing."""
    dataset_dir = Path(dataset_dir)
    train_csv = dataset_dir / "train.csv"
    result: dict[str, Any] = {"missing": [], "corrupt": [], "rows": {}}
    if not train_csv.exists():
        result["missing"].append("train.csv")
        result["complete"] = False
        return result
    train_rows = csv_row_count(train_csv)
    result["rows"]["train.csv"] = train_rows
    row_sidecars = [(ROW_TRAIN, train_rows, ROW_DIM)]
    if (dataset_dir / "test.csv").exists():
        test_rows = csv_row_count(dataset_dir / "test.csv")
        row_sidecars.append((ROW_TEST, test_rows, ROW_DIM))
    for name, expect_rows, expect_cols in row_sidecars:
        path = dataset_dir / name
        if not path.exists():
            result["missing"].append(name)
            continue
        rows, cols = _vs_shape(path)
        result["rows"][name] = rows
        if rows != expect_rows or cols != expect_cols:
            result["corrupt"].append(f"{name} (got {rows}x{cols}, want {expect_rows}x{expect_cols})")
        elif deep:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip() and not all(_finite(v) for v in line.split(",")):
                        result["corrupt"].append(f"{name} (non-finite cell)")
                        break
    result["complete"] = not result["missing"] and not result["corrupt"]
    return result


def _finite(token: str) -> bool:
    try:
        v = float(token)
        return v == v and v not in (float("inf"), float("-inf"))
    except ValueError:
        return False


def make_signatures(
    dataset_dir: Path,
    *,
    force: bool = False,
    hyperion_bin: str | None = None,
    timeout_s: float = 300.0,
    scratch: Path | None = None,
    config: PrimeConfig | None = None,
) -> SignatureReport:
    """Generate sidecars without writing through to a legacy dataset directory."""
    config = config or load_config()
    source_dir = Path(dataset_dir).expanduser().resolve(strict=False)
    dataset_hash = source_dir.name
    started = time.time()
    train_csv = source_dir / "train.csv"
    if not train_csv.exists():
        return SignatureReport(str(source_dir), dataset_hash, "failed", ["no train.csv"])

    source_pre = verify_sidecars(source_dir)
    if source_pre["complete"] and not force:
        return SignatureReport(
            str(source_dir),
            dataset_hash,
            "skipped",
            rows_train=source_pre["rows"].get("train.csv"),
        )

    output_names = {
        ROW_TRAIN,
        ROW_TEST,
        CELL_TRAIN,
        CELL_TEST,
        SIGNATURE_MANIFEST,
        "_discard_test3.vs",
        "_discard_test128.vs",
    }
    workspace = prime_dataset_workspace(
        source_dir,
        config=config,
        output_names=output_names,
    )
    dataset_dir = workspace.dataset_dir
    train_csv = dataset_dir / "train.csv"
    pre = verify_sidecars(dataset_dir)
    if pre["complete"] and not force:
        return SignatureReport(
            str(dataset_dir),
            dataset_hash,
            "skipped",
            rows_train=pre["rows"].get("train.csv"),
        )

    binary = hyperion_bin or hyperion_binary()
    test_csv = dataset_dir / "test.csv"
    scratch = ensure_prime_owned_directory(scratch, config) if scratch is not None else dataset_dir
    # hyperion_vs always needs --test; for train-only dirs, replay train as test and discard.
    test_arg = str(test_csv) if test_csv.exists() else str(train_csv)
    test_from_train = not test_csv.exists()
    cell_test = str(
        scratch
        / ("_discard_test3.vs" if test_from_train else CELL_TEST if scratch == dataset_dir else CELL_TEST)
    )
    row_test = str(
        scratch
        / ("_discard_test128.vs" if test_from_train else ROW_TEST if scratch == dataset_dir else ROW_TEST)
    )
    cmd = [
        binary,
        "--train",
        str(train_csv),
        "--test",
        test_arg,
        "--out-cell-train",
        str(dataset_dir / CELL_TRAIN),
        "--out-cell-test",
        cell_test,
        "--out-row-train",
        str(dataset_dir / ROW_TRAIN),
        "--out-row-test",
        row_test,
        "--weights",
        HYPERION_WEIGHTS,
        "--manifest",
        HYPERION_MANIFEST,
        "--sweeps",
        str(HYPERION_SWEEPS),
        "--scales",
        HYPERION_SCALES,
        "--seed",
        str(HYPERION_SEED),
        "--batch",
        "256",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return SignatureReport(
            str(dataset_dir), dataset_hash, "failed", [f"hyperion_vs timeout after {timeout_s}s"]
        )
    if proc.returncode != 0:
        return SignatureReport(
            str(dataset_dir), dataset_hash, "failed", [f"hyperion_vs: {proc.stderr.strip()[:300]}"]
        )

    if test_from_train:
        for junk in (Path(cell_test), Path(row_test)):
            junk.unlink(missing_ok=True)
    elif scratch != dataset_dir:
        Path(cell_test).replace(dataset_dir / CELL_TEST)
        Path(row_test).replace(dataset_dir / ROW_TEST)

    post = verify_sidecars(dataset_dir, deep=True)
    if not post["complete"]:
        # refuse to leave corrupt sidecars: remove what we just wrote
        for name in (ROW_TRAIN, ROW_TEST, CELL_TRAIN, CELL_TEST):
            (dataset_dir / name).unlink(missing_ok=True)
        return SignatureReport(str(dataset_dir), dataset_hash, "failed", post["missing"] + post["corrupt"])

    (dataset_dir / SIGNATURE_MANIFEST).write_text(
        json.dumps(
            {
                "dataset_hash": dataset_hash,
                "signature_encoder": SIGNATURE_ENCODER,
                "rows_train": post["rows"].get("train.csv"),
                "test_from_train": test_from_train,
                "binary": binary,
                "source_dataset": str(source_dir),
                "staged": workspace.staged,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return SignatureReport(
        str(dataset_dir),
        dataset_hash,
        "generated",
        rows_train=post["rows"].get("train.csv"),
        timings_ms=(time.time() - started) * 1000.0,
    )


def discover_datasets(root: Path) -> list[Path]:
    """Dataset dirs (with train.csv), skipping dot / .incoming partials, sorted for stable sharding."""
    root = Path(root)
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith(".") and (child / "train.csv").exists():
            out.append(child)
    return out


def make_signatures_many(
    dirs: list[Path],
    *,
    host_index: int = 0,
    host_count: int = 1,
    force: bool = False,
    hyperion_bin: str | None = None,
    report_jsonl: Path | None = None,
    config: PrimeConfig | None = None,
) -> Iterator[SignatureReport]:
    """Generate signatures for this host's deterministic shard of ``dirs``."""
    config = config or load_config()
    shard = [d for i, d in enumerate(dirs) if i % host_count == host_index]
    report_path = prime_owned_path(report_jsonl, config) if report_jsonl else None
    if report_path is not None:
        ensure_prime_owned_directory(report_path.parent, config)
    handle = report_path.open("a", encoding="utf-8") if report_path else None
    try:
        for d in shard:
            report = make_signatures(
                d,
                force=force,
                hyperion_bin=hyperion_bin,
                config=config,
            )
            if handle:
                handle.write(json.dumps(report.__dict__, sort_keys=True) + "\n")
                handle.flush()
            yield report
    finally:
        if handle:
            handle.close()
