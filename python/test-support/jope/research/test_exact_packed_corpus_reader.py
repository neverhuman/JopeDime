from __future__ import annotations

from test_exact_packed_corpus_storage import (
    argparse,
    hashlib,
    json,
    os,
    shutil,
    sys,
    time,
    Counter,
    Path,
    Callable,
    Iterable,
    Iterator,
    Sequence,
    np,
    atomic_json,
    prime_source_root,
    sha256_file,
    PACK_SCHEMA,
    LEGACY_PACK_SCHEMA,
    PACK_LAYOUT,
    LABEL_SHARD_ROWS,
    TASK_CODE,
    SPLIT_CODE,
    _jsonl,
    _default_tokenizer,
    _open_array,
    _packed_records,
    _LabelShardWriter,
    _write_causal_pair_index,
    _write_recipe_e_evidence_index,
)

def verify(
    manifest: dict,
    root: Path,
    *,
    source_manifest_path: Path | None = None,
    source_paths: dict[str, Path] | None = None,
) -> None:
    if manifest.get("schema") not in {PACK_SCHEMA, LEGACY_PACK_SCHEMA}:
        raise ValueError("unsupported packed corpus schema")
    if manifest.get("test_adaptive") is not True or manifest.get("benchmark_valid") is not False:
        raise ValueError("packed corpus provenance labels are invalid")
    binding_path = manifest.get("catalog_binding")
    if binding_path is not None:
        binding_path = Path(binding_path)
        if (
            not binding_path.is_file()
            or sha256_file(binding_path) != manifest["catalog_binding_sha256"]
        ):
            raise ValueError("packed corpus catalog binding hash mismatch")
    source_manifest = source_manifest_path or Path(manifest["source_manifest"])
    if (
        not source_manifest.is_file()
        or sha256_file(source_manifest) != manifest["source_manifest_sha256"]
    ):
        raise ValueError("packed corpus source manifest hash mismatch")
    source = json.loads(source_manifest.read_text())
    for key in ("train", "validation"):
        expected = manifest[f"{key}_sha256"]
        if source.get(f"{key}_sha256") != expected:
            raise ValueError(f"packed corpus source {key} binding mismatch")
        # A packed corpus is a complete immutable training input. Requiring its
        # 9+ GiB source JSONL files here made every packed trainer stage and
        # re-hash those files before Python could start, defeating the packed
        # boundary and exceeding the scheduler's dispatch window. Callers may
        # still provide source_paths for an explicit full-source audit.
        if source_paths is not None:
            path = source_paths[key]
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"packed corpus source {key} hash mismatch")
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"packed corpus file hash mismatch: {name}")
    row_count = int(manifest["row_count"])
    for name in ("dataset_indices", "tasks", "splits", "scores", "runtime_ms"):
        if len(np.load(root / f"{name}.npy", mmap_mode="r")) != row_count:
            raise ValueError(f"packed corpus row count mismatch: {name}")
    for name in ("genome_offsets", "token_offsets"):
        offsets = np.load(root / f"{name}.npy", mmap_mode="r")
        if len(offsets) != row_count + 1 or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError(f"packed corpus offsets are invalid: {name}")
    token_offsets = np.load(root / "token_offsets.npy", mmap_mode="r")
    if (root / "token_ids.u16.bin").stat().st_size != int(token_offsets[-1]) * 2:
        raise ValueError("packed corpus token byte length is invalid")
    if manifest["schema"] == LEGACY_PACK_SCHEMA:
        with (root / manifest["reversible_index"]).open(encoding="utf-8") as handle:
            indexed = sum(1 for line in handle if line.strip())
        if indexed != row_count:
            raise ValueError("packed corpus reversible-index row count mismatch")
    else:
        selected = root / manifest["selected_row_index"]["path"]
        if selected.name == "observations.jsonl" or selected.stat().st_size != row_count * 32:
            raise ValueError("packed selected-row digest index is invalid")
        shard_rows = 0
        previous = 0
        for shard in manifest["label_shards"]:
            if (
                int(shard["first_receipt"]) <= previous
                or int(shard["last_receipt"]) < int(shard["first_receipt"])
                or sha256_file(root / shard["path"]) != shard["sha256"]
            ):
                raise ValueError("packed content-addressed label shard is invalid")
            rows = sum(1 for _ in _jsonl(root / shard["path"]))
            if rows != int(shard["rows"]):
                raise ValueError("packed label-shard row count mismatch")
            shard_rows += rows
            previous = int(shard["last_receipt"])
        if shard_rows != row_count:
            raise ValueError("packed label-shard total differs from selected rows")
    causal_index = manifest.get("causal_pair_index")
    if causal_index is not None:
        causal_rows = 0
        seen_pairs = set()
        with (root / causal_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json_loads(line)
                pair_id = str(row.get("pair_id") or "")
                direct = int(row.get("direct_ordinal", -1))
                comparator = int(row.get("comparator_ordinal", -1))
                if (
                    len(pair_id) != 64
                    or any(character not in "0123456789abcdef" for character in pair_id)
                    or pair_id in seen_pairs
                    or not 0 <= direct < row_count
                    or not 0 <= comparator < row_count
                    or direct == comparator
                ):
                    raise ValueError("packed causal pair index is invalid")
                seen_pairs.add(pair_id)
                causal_rows += 1
        if causal_rows != int(manifest.get("causal_pair_count") or 0):
            raise ValueError("packed causal pair count mismatch")
    recipe_e_index = manifest.get("recipe_e_evidence_index")
    if recipe_e_index is not None:
        recipe_e_rows = 0
        with (root / recipe_e_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = _json_loads(line)
                lane = row.get("lane")
                ordinal = int(row.get("ordinal", -1))
                if (
                    row.get("task") not in TASK_CODE
                    or lane
                    not in {
                        "completed_descendant_trajectories",
                        "validity_instability_hard_negatives",
                    }
                    or not 0 <= ordinal < row_count
                ):
                    raise ValueError("packed recipe E evidence index is invalid")
                if lane == "completed_descendant_trajectories":
                    receipt = str(
                        row.get("descendant_credit_receipt_sha256") or ""
                    )
                    if (
                        len(receipt) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in receipt
                        )
                        or row.get("descendant_horizon") not in {1, 2, 4, 8}
                    ):
                        raise ValueError(
                            "packed recipe E descendant evidence is invalid"
                        )
                recipe_e_rows += 1
        expected_rows = (
            manifest.get("recipe_e_evidence_sidecar") or {}
        ).get("rows")
        if recipe_e_rows != int(expected_rows or 0):
            raise ValueError("packed recipe E evidence count mismatch")


class Float32View(Sequence[float]):
    """List-compatible view over one mmap context row without owning its floats."""

    def __init__(self, values: np.ndarray):
        self._values = values

    def __bool__(self) -> bool:
        return len(self._values) > 0

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[float]:
        return (float(value) for value in self._values)

    def __getitem__(self, item):
        value = self._values[item]
        if isinstance(item, slice):
            return [float(number) for number in value]
        return float(value)


def load_training_rows(
    manifest_path: Path,
    *,
    task: str | None = None,
    source_manifest_path: Path | None = None,
    source_train: Path | None = None,
    source_validation: Path | None = None,
    catalog_data: Path | None = None,
):
    """Return TrainingRows sharing mmap contexts plus their packed token cache."""

    if task not in {None, "binary", "regression"}:
        raise ValueError("packed corpus task must be binary or regression")
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    source_paths = None
    if source_train is not None or source_validation is not None:
        if source_train is None or source_validation is None:
            raise ValueError("both packed source row paths must be provided")
        source_paths = {"train": source_train, "validation": source_validation}
    verify(
        manifest,
        root,
        source_manifest_path=source_manifest_path,
        source_paths=source_paths,
    )
    prime = prime_source_root()
    if str(prime) not in sys.path:
        sys.path.insert(0, str(prime))
    from jope.policy.automaton import replay_masks
    from jope.policy.tokens import build_prime_action_vocab
    from jope.schema import TrainingRow

    catalog_path = catalog_data or Path(manifest["catalog_data"])
    if sha256_file(catalog_path) != manifest["catalog_data_sha256"]:
        raise ValueError("packed corpus catalog data hash mismatch")
    catalog_rows = catalog_path.stat().st_size // (2_048 * 4)
    catalog = np.memmap(
        catalog_path,
        dtype="<f4",
        mode="r",
        shape=(catalog_rows, 2_048),
    )
    dataset_indices = np.load(root / "dataset_indices.npy", mmap_mode="r")
    scores = np.load(root / "scores.npy", mmap_mode="r")
    runtime_ms = np.load(root / "runtime_ms.npy", mmap_mode="r")
    genome_offsets = np.load(root / "genome_offsets.npy", mmap_mode="r")
    token_offsets = np.load(root / "token_offsets.npy", mmap_mode="r")
    genomes = np.memmap(root / "genomes.bin", dtype="u1", mode="r")
    token_ids = np.memmap(root / "token_ids.u16.bin", dtype="<u2", mode="r")
    vocab = build_prime_action_vocab()
    rows = []
    rows_by_ordinal = {}
    artifact_cache = {}
    for ordinal, record in enumerate(_packed_records(manifest, root)):
        if ordinal and ordinal % 25_000 == 0:
            print(
                "packed-loader "
                f"rows={ordinal}/{manifest['row_count']} "
                f"percent={100.0 * ordinal / int(manifest['row_count']):.1f}",
                flush=True,
            )
        if task is not None and record.get("task") != task:
            continue
        context = catalog[int(dataset_indices[ordinal])]
        genome_start, genome_end = map(
            int, genome_offsets[ordinal : ordinal + 2]
        )
        token_start, token_end = map(int, token_offsets[ordinal : ordinal + 2])
        genome = bytes(genomes[genome_start:genome_end]).decode()
        row = object.__new__(TrainingRow)
        row.dataset_id = record["dataset_hash"]
        row.source = record["source"]
        row.embedding = Float32View(context[:1_536])
        row.target_embedding = Float32View(context[1_536:])
        row.genome = genome
        row.canonical_genome = genome
        row.valid = bool(record["valid"])
        row.metric = record["metric"]
        row.score = float(scores[ordinal])
        row.rank = None
        row.elo = None
        row.cost_ms = float(runtime_ms[ordinal])
        row.family_bitmap = {}
        row.complexity = None
        row.runtime_class = None
        row.feedback_origin = None
        row.engine = record["engine"]
        row.metadata = record["metadata"]
        rows.append(row)
        rows_by_ordinal[ordinal] = row
        tokens = [int(value) for value in token_ids[token_start:token_end]]
        artifact_cache[id(row)] = (tokens, replay_masks(tokens, vocab))
    causal_index = manifest.get("causal_pair_index")
    if causal_index is not None:
        with (root / causal_index).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                pair = _json_loads(line)
                if task is not None and pair.get("task") != task:
                    continue
                direct = rows_by_ordinal.get(int(pair["direct_ordinal"]))
                comparator = rows_by_ordinal.get(int(pair["comparator_ordinal"]))
                if direct is None or comparator is None:
                    raise ValueError("packed causal pair row is absent from task view")
                common = {
                    "causal_pair_id": pair["pair_id"],
                    "causal_pair_index": int(pair["pair_index"]),
                    "causal_accepted_parent_checkpoint_sha256": pair[
                        "accepted_parent_checkpoint_sha256"
                    ],
                    "causal_wave": int(pair["wave"]),
                    "causal_score_gap": float(pair["score_gap"]),
                    "causal_pair_hard": bool(pair["hard"]),
                    "causal_pair_novel": bool(pair["novel"]),
                }
                direct.metadata.update(
                    {**common, "causal_lane": "native_dl_direct"}
                )
                comparator.metadata.update(
                    {**common, "causal_lane": "matched_gp_comparator"}
                )
    expected_rows = (
        manifest["row_count"]
        if task is None
        else int((manifest.get("task_counts") or {}).get(task) or 0)
    )
    if len(rows) != expected_rows:
        raise ValueError("packed observation index row count mismatch")
    return rows, artifact_cache
