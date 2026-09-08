"""Prime-owned shadow factory for the certified CountSketch cohort.

Candidate model execution is delegated exclusively to ``PrimeNativeSearch``.
This module expands the canonical registry bins, ranks returned native CV
scores, exact-promotes finalists, and writes only the winning Python artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from itertools import product
from pathlib import Path
from typing import Any

from .grammars.prime import registry_document
from .prime_config import PrimeConfig, load_config
from .prime_search import NativeSearchError, PrimeNativeSearch


def _registry_value(value: Any) -> Any:
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError(f"malformed registry value {value!r}")
    return next(iter(value.values()))


def count_sketch_genomes() -> list[str]:
    """Exhaustive canonical certification cohort in stable registry-bin order."""
    entries = [
        entry
        for entry in registry_document()["operators"]
        if entry.get("name") == "count_sketch"
    ]
    if len(entries) != 1:
        raise ValueError("canonical registry must contain one CountSketch descriptor")
    entry = entries[0]
    if not entry.get("lifecycle", {}).get("quarantined"):
        raise ValueError("shadow factory is only valid while CountSketch is quarantined")
    params = {param["name"]: param for param in entry["params"]}
    widths = [_registry_value(value) for value in params["n"]["bins"]]
    seeds = [_registry_value(value) for value in params["seed"]["bins"]]
    if not widths or not seeds:
        raise ValueError("CountSketch registry bins must not be empty")
    genomes = ["(pipe X (ridge))"]
    for width, seed, append in product(widths, seeds, (False, True)):
        append_text = "true" if append else "false"
        genomes.append(
            f"(pipe X (count_sketch n={int(width)} seed={int(seed)} "
            f"append={append_text}) (ridge))"
        )
    return genomes


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


class PrimeCountSketchFactory:
    """Run one train-side CountSketch certification/champion batch."""

    def __init__(
        self,
        search: PrimeNativeSearch,
        *,
        output_root: str | Path | None = None,
        config: PrimeConfig | None = None,
    ) -> None:
        self.search = search
        self.config = config or load_config()
        root = (
            Path(output_root).expanduser().resolve(strict=False)
            if output_root is not None
            else self.config.runs / "count_sketch_factory"
        )
        if not root.is_relative_to(self.config.root):
            raise ValueError("Prime factory output must resolve below prime.root")
        self.output_root = root

    def run_dataset(
        self,
        *,
        dataset_hash: str,
        train_csv: str | Path,
        folds: int = 5,
        seed: int = 0,
        fold_seed: int = 0,
        exact_top_k: int = 3,
        emit_winner: bool = True,
    ) -> dict[str, Any]:
        if not dataset_hash or any(character.isspace() for character in dataset_hash):
            raise ValueError("dataset_hash must be a non-empty token")
        output_dir = (self.output_root / dataset_hash).resolve(strict=False)
        if not output_dir.is_relative_to(self.output_root):
            raise ValueError("dataset_hash escapes the Prime factory output root")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", dataset_hash):
            raise ValueError("dataset_hash must be a simple path-free token")
        if exact_top_k <= 0:
            raise ValueError("exact_top_k must be positive")
        train_path = Path(train_csv).expanduser().resolve(strict=True)
        genomes = count_sketch_genomes()
        rows = [
            {
                "engine": "prime",
                "dataset_hash": dataset_hash,
                "train_csv": str(train_path),
                "genome": genome,
                "folds": folds,
                "fold_seed": fold_seed,
            }
            for genome in genomes
        ]
        fast_results = self.search.evaluate_ranked_rows(
            rows,
            seed=seed,
            suite_hash="prime-count-sketch-shadow-v1",
        )
        candidates: list[dict[str, Any]] = []
        for index, result in fast_results:
            candidates.append(
                {
                    "candidate_index": index,
                    "genome": rows[index]["genome"],
                    "semantic_plan_hash": result.get("semantic_plan_hash"),
                    "native_score_robust": result.get("native_robust_score"),
                    "native_result": result,
                }
            )
        ranked = [
            candidate
            for candidate in candidates
            if candidate["native_result"].get("valid")
            and candidate["native_score_robust"] is not None
        ]
        if not ranked:
            raise NativeSearchError("CountSketch shadow batch produced no finite native scores")

        promoted_candidates = ranked[: min(exact_top_k, len(ranked))]
        promotion_rows: list[dict[str, Any]] = []
        for candidate in promoted_candidates:
            row = dict(rows[candidate["candidate_index"]])
            row["semantic_plan_hash"] = candidate["semantic_plan_hash"]
            promotion_rows.append(row)
        exact_results = self.search.promote_exact_rows(
            promotion_rows,
            seed=seed,
            suite_hash="prime-count-sketch-exact-v1",
        )
        finalists: list[dict[str, Any]] = []
        for promoted_index, exact in exact_results:
            candidate = promoted_candidates[promoted_index]
            finalists.append(
                {
                    **candidate,
                    "exact_score_robust": exact.get("_native_robust_score"),
                    "exact_result": exact,
                    "row": promotion_rows[promoted_index],
                }
            )
        if not finalists:
            raise NativeSearchError("CountSketch finalists produced no finite exact scores")
        winner = finalists[0]
        bundle = None
        if emit_winner:
            bundle = self.search.emit_user_python(winner["row"], winner["exact_result"])
            self._write_python_bundle(output_dir, bundle)

        report = {
            "schema": "jope-prime-count-sketch-factory-v1",
            "dataset_hash": dataset_hash,
            "train_csv": str(train_path),
            "candidate_count": len(candidates),
            "exact_finalist_count": len(finalists),
            "evaluation_seed": seed,
            "fold_seed": fold_seed,
            "search_backend_policy": self.search.backend,
            "require_cuda": self.search.require_cuda,
            "winner": {
                "genome": winner["genome"],
                "semantic_plan_hash": winner["semantic_plan_hash"],
                "native_score_robust": winner["native_score_robust"],
                "exact_score_robust": winner["exact_score_robust"],
                "artifact_id": bundle.get("artifact_id") if bundle else None,
            },
            "candidates": candidates,
            "finalists": [
                {
                    "genome": finalist["genome"],
                    "semantic_plan_hash": finalist["semantic_plan_hash"],
                    "native_score_robust": finalist["native_score_robust"],
                    "exact_score_robust": finalist["exact_score_robust"],
                }
                for finalist in finalists
            ],
        }
        _atomic_text(
            output_dir / "factory_report.json",
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        )
        return report

    @staticmethod
    def _write_python_bundle(output_dir: Path, bundle: dict[str, Any]) -> None:
        if bundle.get("kind") != "portable_python":
            raise NativeSearchError("winner bundle is not portable Python")
        artifact_id = bundle.get("artifact_id")
        if not isinstance(artifact_id, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_id):
            raise NativeSearchError("winner bundle has an invalid artifact identity")
        files = bundle.get("files")
        if not isinstance(files, list):
            raise NativeSearchError("winner bundle has no files")
        validated_files: list[tuple[str, bytes]] = []
        for file in files:
            if not isinstance(file, dict):
                raise NativeSearchError("winner bundle contains a malformed file")
            relative = file.get("relative_path")
            contents = file.get("contents")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise NativeSearchError("winner bundle contains an unsafe path")
            if not isinstance(contents, list) or not all(
                isinstance(value, int) and 0 <= value <= 255 for value in contents
            ):
                raise NativeSearchError("winner bundle contains invalid bytes")
            content_bytes = bytes(contents)
            content_hash = file.get("sha256")
            if (
                not isinstance(content_hash, str)
                or hashlib.sha256(content_bytes).hexdigest() != content_hash
            ):
                raise NativeSearchError("winner bundle file hash does not match its bytes")
            validated_files.append((relative, content_bytes))
        manifest = bundle.get("manifest")
        if not isinstance(manifest, str):
            raise NativeSearchError("winner bundle has no manifest")
        manifest_hash = bundle.get("manifest_sha256")
        if (
            not isinstance(manifest_hash, str)
            or hashlib.sha256(manifest.encode("utf-8")).hexdigest() != manifest_hash
        ):
            raise NativeSearchError("winner manifest hash does not match its contents")
        try:
            manifest_document = json.loads(manifest)
        except json.JSONDecodeError as error:
            raise NativeSearchError("winner manifest is not valid JSON") from error
        if (
            not isinstance(manifest_document, dict)
            or manifest_document.get("artifact_hash") != artifact_id
            or manifest_document.get("artifact_kind") != "portable_python"
        ):
            raise NativeSearchError("winner manifest identity does not match the bundle")
        for relative, content_bytes in validated_files:
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(content_bytes)
            temporary.replace(destination)
        _atomic_text(output_dir / "manifest.json", manifest + "\n")


__all__ = ["PrimeCountSketchFactory", "count_sketch_genomes"]
