from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .artifact_pins import artifact_sha256
from .fused import BUNDLE_FORMAT, BUNDLE_VERSION, FusedTrainingConfig, _jsonable_config


def _write_bundle_manifest(
    bundle_dir: Path,
    cfg: FusedTrainingConfig,
    *,
    feature_names: Sequence[str],
    vectorizer_artifact: str,
    vectorizer: Mapping[str, object],
    checkpoint_best: str,
    checkpoint_final: str,
    classical_bundles: Mapping[str, Mapping[str, object]],
    vector_weights: Sequence[float],
    scorer: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _jsonable_config(cfg),
        "window_size": int(cfg.window_size),
        "signature_views": int(cfg.signature_views),
        "seed": int(cfg.seed),
        "include_cache_profile": bool(cfg.include_cache_profile),
        "vector_dim": int(cfg.vector_dim),
        "classical_component_dim": int(cfg.classical_component_dim),
        "feature_names": list(feature_names),
        "signature_feature_names": [
            f"{agg}:{name}" for agg in ("mean", "std", "min", "max") for name in feature_names
        ],
        "vectorizer": {
            "artifact": vectorizer_artifact,
            "signature_dim": int(vectorizer.get("signature_dim", 0)),
            "component_dim": int(vectorizer.get("component_dim", 0)),
            "actual_dim": int(vectorizer.get("actual_dim", 0)),
        },
        "neural": {
            "best": checkpoint_best,
            "final": checkpoint_final,
        },
        "classical_models": {
            name: os.path.relpath(str(bundle["model_path"]), str(bundle_dir))
            for name, bundle in classical_bundles.items()
        },
        "vector_weights": [float(x) for x in vector_weights],
        "scorer": dict(scorer),
        "report": dict(report),
    }
    # Bind all joblib load paths, including an optional candidate meta scorer.
    artifact_paths = {vectorizer_artifact, *manifest["classical_models"].values()}
    artifact_paths.update(
        str(scorer[key]) for key in ("artifact", "candidate_meta_artifact") if scorer.get(key)
    )
    manifest["artifact_sha256"] = {
        rel_path: artifact_sha256(bundle_dir / rel_path) for rel_path in sorted(artifact_paths)
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

def _copy_checkpoints(bundle_dir: Path, best_checkpoint: Path, final_checkpoint: Path) -> tuple[str, str]:
    neural_dir = bundle_dir / "neural"
    neural_dir.mkdir(parents=True, exist_ok=True)
    best_dest = neural_dir / "best.pt"
    final_dest = neural_dir / "final.pt"
    shutil.copy2(best_checkpoint, best_dest)
    shutil.copy2(final_checkpoint, final_dest)
    return os.path.relpath(best_dest, bundle_dir), os.path.relpath(final_dest, bundle_dir)

def _relative_artifact(bundle_dir: Path, path: Path) -> str:
    return os.path.relpath(str(path), str(bundle_dir))

