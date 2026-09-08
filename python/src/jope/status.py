from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def artifact_freshness(
    *,
    art_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    source = _newest_source(root)
    artifacts: dict[str, Any] = {}
    if art_dir is not None:
        art = Path(art_dir)
        artifacts["checkpoint"] = _artifact_status(art / "checkpoints" / "model.pt", source)
        artifacts["active_rows"] = _artifact_status(art / "active_rows.jsonl", source)
    if run_dir is not None:
        run = Path(run_dir)
        artifacts["evaluation"] = _artifact_status(run / "evaluation.jsonl", source)
        artifacts["train_metrics"] = _artifact_status(run / "train_metrics.jsonl", source)
    stale = [name for name, row in artifacts.items() if row.get("stale")]
    return {
        "project_root": str(root),
        "source_mtime": source.get("mtime"),
        "newest_source": source.get("path"),
        "artifacts": artifacts,
        "stale_artifacts": stale,
        "fresh": not stale,
        "warning": None if not stale else "source files are newer than: " + ", ".join(stale),
    }


def write_status_json(
    *,
    art_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> str:
    return json.dumps(
        artifact_freshness(art_dir=art_dir, run_dir=run_dir, project_root=project_root),
        indent=2,
        sort_keys=True,
    )


def _newest_source(root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for rel in ("src/jope", "rust/jope-validate/src"):
        base = root / rel
        if base.exists():
            candidates.extend(path for path in base.rglob("*") if path.suffix in {".py", ".rs"})
    newest_path: Path | None = None
    newest_mtime = 0.0
    for path in candidates:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = path
    return {
        "path": str(newest_path) if newest_path is not None else None,
        "mtime": newest_mtime if newest_path is not None else None,
    }


def _artifact_status(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {"path": str(path), "exists": False, "mtime": None, "age_sec": None, "stale": True}
    source_mtime = source.get("mtime")
    stale = bool(source_mtime is not None and mtime < float(source_mtime))
    return {
        "path": str(path),
        "exists": True,
        "mtime": mtime,
        "age_sec": time.time() - mtime,
        "stale": stale,
    }
