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

def materialize(
    manifest_path: Path,
    output_root: Path,
    *,
    tokenizer: Callable[[dict], list[int]] = _default_tokenizer,
    catalog_pointer_path: Path = Path(
        "/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json"
    ),
    catalog_binding_path: Path | None = None,
) -> dict:
    source = json.loads(manifest_path.read_text())
    train_path = Path(source["train"])
    validation_path = Path(source["validation"])
    if catalog_binding_path is not None:
        catalog_binding = json.loads(catalog_binding_path.read_text())
        if catalog_binding.get("schema") != "CatalogBinding.v1":
            raise ValueError("unsupported immutable catalog binding")
        catalog_binding_sha256 = sha256_file(catalog_binding_path)
        if catalog_binding_sha256 != source.get(
            "catalog_binding_sha256", catalog_binding_sha256
        ):
            raise ValueError("training corpus catalog binding hash mismatch")
        catalog_pointer = {
            "catalog": catalog_binding["catalog_path"],
            "catalog_sha256": catalog_binding["catalog_sha256"],
            "data": catalog_binding["data_path"],
            "data_sha256": catalog_binding["data_sha256"],
        }
        for path_key, hash_key in (
            ("catalog", "catalog_sha256"),
            ("data", "data_sha256"),
        ):
            if sha256_file(Path(catalog_pointer[path_key])) != catalog_pointer[hash_key]:
                raise ValueError(f"catalog binding {path_key} hash mismatch")
    else:
        catalog_binding = None
        catalog_pointer = json.loads(catalog_pointer_path.read_text())
        catalog_binding_sha256 = hashlib.sha256(
            json.dumps(
                catalog_pointer, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    catalog = json.loads(Path(catalog_pointer["catalog"]).read_text())
    if source["catalog_sha256"] != catalog_pointer["catalog_sha256"]:
        raise ValueError("training corpus and packed catalog identities differ")
    train_sha256 = sha256_file(train_path)
    validation_sha256 = sha256_file(validation_path)
    if train_sha256 != source["train_sha256"]:
        raise ValueError("training JSONL changed after corpus freeze")
    if validation_sha256 != source["validation_sha256"]:
        raise ValueError("validation JSONL changed after corpus freeze")
    catalog_index = {
        (entry["task"], entry["dataset_hash"]): int(entry["catalog_index"])
        for entry in catalog["entries"]
    }
    binding = hashlib.sha256(
        json.dumps(
            {
                "train_sha256": train_sha256,
                "validation_sha256": validation_sha256,
                "catalog_sha256": source["catalog_sha256"],
                "catalog_binding_sha256": catalog_binding_sha256,
                "schema": PACK_SCHEMA,
                "layout": PACK_LAYOUT,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    final = output_root / binding
    manifest_final = final / "manifest.json"
    if manifest_final.is_file():
        value = json.loads(manifest_final.read_text())
        verify(value, final)
        return value

    row_count = sum(
        1
        for path in (train_path, validation_path)
        for _ in _jsonl(path)
    )
    temporary = output_root / f".{binding}.tmp-{os.getpid()}"
    output_root.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(exist_ok=False)
    try:
        arrays = {
            "dataset_indices": _open_array(
                temporary / "dataset_indices.npy", "<u4", row_count
            ),
            "tasks": _open_array(temporary / "tasks.npy", "u1", row_count),
            "splits": _open_array(temporary / "splits.npy", "u1", row_count),
            "scores": _open_array(temporary / "scores.npy", "<f4", row_count),
            "runtime_ms": _open_array(
                temporary / "runtime_ms.npy", "<f4", row_count
            ),
            "genome_offsets": _open_array(
                temporary / "genome_offsets.npy", "<u8", row_count + 1
            ),
            "token_offsets": _open_array(
                temporary / "token_offsets.npy", "<u8", row_count + 1
            ),
        }
        arrays["genome_offsets"][0] = 0
        arrays["token_offsets"][0] = 0
        genome_cursor = token_cursor = 0
        task_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        causal_members = []
        recipe_e_members = []
        shard_writer = _LabelShardWriter(
            temporary, output_root / "_label_shards_v3"
        )
        with (
            (temporary / "genomes.bin").open("wb") as genomes,
            (temporary / "token_ids.u16.bin").open("wb") as token_output,
            (temporary / "selected_row_index.sha256.bin").open("wb") as selected_index,
        ):
            rows = (
                (row, split)
                for path, split in (
                    (train_path, "train"),
                    (validation_path, "validation"),
                )
                for row in _jsonl(path)
            )
            for ordinal, (row, split) in enumerate(rows):
                metadata = row.get("metadata") or {}
                task = str(metadata.get("task_type") or "")
                dataset = str(metadata.get("dataset_hash") or row.get("dataset_id") or "")
                key = (task, dataset)
                if key not in catalog_index:
                    raise ValueError(f"packed row is absent from the catalog: {task}:{dataset}")
                if metadata.get("test_adaptive") is not True:
                    raise ValueError("packed row does not disclose test adaptation")
                if metadata.get("benchmark_valid") is not False:
                    raise ValueError("packed row incorrectly claims benchmark validity")
                genome = str(row.get("canonical_genome") or row.get("genome") or "")
                genome_bytes = genome.encode()
                raw_tokens = tokenizer(row)
                if any(
                    not isinstance(token, (int, np.integer))
                    or int(token) < 0
                    or int(token) > 65_535
                    for token in raw_tokens
                ):
                    raise ValueError("policy token is outside the packed uint16 vocabulary")
                tokens = np.asarray(raw_tokens, dtype="<u2")
                if not len(tokens):
                    raise ValueError("packed row has an empty policy token sequence")
                genomes.write(genome_bytes)
                tokens.tofile(token_output)
                genome_cursor += len(genome_bytes)
                token_cursor += len(tokens)
                arrays["dataset_indices"][ordinal] = catalog_index[key]
                arrays["tasks"][ordinal] = TASK_CODE[task]
                arrays["splits"][ordinal] = SPLIT_CODE[split]
                arrays["scores"][ordinal] = float(row["score"])
                arrays["runtime_ms"][ordinal] = float(row.get("cost_ms") or 0.0)
                arrays["genome_offsets"][ordinal + 1] = genome_cursor
                arrays["token_offsets"][ordinal + 1] = token_cursor
                record = {
                    "row": ordinal,
                    "task": task,
                    "dataset_hash": dataset,
                    "engine": row.get("engine"),
                    "semantic_sha256": metadata.get("semantic_plan_hash"),
                    "source": row.get("source"),
                    "receipt_sha256": metadata.get("receipt_sha256"),
                    "model_checkpoint_sha256": metadata.get(
                        "model_checkpoint_sha256"
                    ),
                    "generation": metadata.get("generation"),
                    "metric": row.get("metric"),
                    "valid": row.get("valid", True),
                    "metadata": metadata,
                }
                shard_writer.write(record, ordinal)
                receipt_identity = str(metadata.get("receipt_sha256") or "")
                if len(receipt_identity) == 64 and all(
                    character in "0123456789abcdef"
                    for character in receipt_identity
                ):
                    selected_index.write(bytes.fromhex(receipt_identity))
                else:
                    selected_index.write(
                        hashlib.sha256(
                            json.dumps(
                                record,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).digest()
                    )
                if metadata.get("causal_pair_id"):
                    causal_members.append(
                        {
                            "pair_id": str(metadata["causal_pair_id"]),
                            "lane": str(metadata.get("causal_lane") or ""),
                            "ordinal": ordinal,
                            "split": split,
                            "task": task,
                            "dataset_hash": dataset,
                            "accepted_parent_checkpoint_sha256": str(
                                metadata.get(
                                    "causal_accepted_parent_checkpoint_sha256"
                                )
                                or ""
                            ),
                            "wave": int(metadata.get("causal_wave") or 0),
                            "pair_index": int(
                                metadata.get("causal_pair_index") or 0
                            ),
                            "semantic_sha256": str(
                                metadata.get("semantic_plan_hash") or ""
                            ),
                            "score": float(row["score"]),
                        }
                    )
                descendant_receipt = str(
                    metadata.get("descendant_credit_receipt_sha256") or ""
                )
                if descendant_receipt and not metadata.get(
                    "descendant_censored", False
                ):
                    if len(descendant_receipt) != 64 or any(
                        character not in "0123456789abcdef"
                        for character in descendant_receipt
                    ):
                        raise ValueError(
                            "recipe E descendant receipt has an invalid identity"
                        )
                    if metadata.get("descendant_horizon") not in {1, 2, 4, 8}:
                        raise ValueError(
                            "recipe E descendant row has an invalid horizon"
                        )
                    recipe_e_members.append(
                        {
                            "ordinal": ordinal,
                            "task": task,
                            "lane": "completed_descendant_trajectories",
                            "descendant_credit_receipt_sha256": descendant_receipt,
                            "descendant_horizon": int(
                                metadata["descendant_horizon"]
                            ),
                        }
                    )
                elif any(
                    bool(metadata.get(name))
                    for name in (
                        "validity_hard_negative",
                        "instability_hard_negative",
                        "adversarial_hard_case",
                        "candidate_invalid",
                    )
                ):
                    recipe_e_members.append(
                        {
                            "ordinal": ordinal,
                            "task": task,
                            "lane": "validity_instability_hard_negatives",
                        }
                    )
                task_counts[task] += 1
                split_counts[split] += 1
        label_shards = shard_writer.close(row_count)
        causal_pair_stats = _write_causal_pair_index(
            temporary / "causal_pairs.jsonl", causal_members
        )
        recipe_e_evidence_stats = _write_recipe_e_evidence_index(
            temporary / "recipe_e_evidence.jsonl", recipe_e_members
        )
        for array in arrays.values():
            array.flush()
        del arrays

        files = {}
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json":
                continue
            files[path.name] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        packed = {
            "schema": PACK_SCHEMA,
            "layout": PACK_LAYOUT,
            "watermark": source["source_label_watermark"],
            "binding_sha256": binding,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": sha256_file(manifest_path),
            "train_sha256": source["train_sha256"],
            "validation_sha256": source["validation_sha256"],
            "source_label_watermark": source["source_label_watermark"],
            "catalog_sha256": source["catalog_sha256"],
            "catalog_binding": (
                str(catalog_binding_path) if catalog_binding_path is not None else None
            ),
            "catalog_binding_sha256": catalog_binding_sha256,
            "catalog_data": catalog_pointer["data"],
            "catalog_data_sha256": catalog_pointer["data_sha256"],
            "row_count": row_count,
            "task_counts": dict(task_counts),
            "split_counts": dict(split_counts),
            "test_adaptive": True,
            "benchmark_valid": False,
            "selection_bias": "test-exact-reused",
            "selected_row_index": {
                "path": "selected_row_index.sha256.bin",
                "sha256": files["selected_row_index.sha256.bin"]["sha256"],
                "dtype": "sha256",
                "rows": row_count,
            },
            "selected_row_index_dtype": "sha256",
            "causal_pair_index": "causal_pairs.jsonl",
            "exact_pair_sidecar": {
                "path": "causal_pairs.jsonl",
                "sha256": files["causal_pairs.jsonl"]["sha256"],
                "dtype": "jsonl",
                "rows": causal_pair_stats["complete_pairs"],
            },
            "causal_pair_count": causal_pair_stats["complete_pairs"],
            "causal_pair_stats": causal_pair_stats,
            "recipe_e_evidence_index": "recipe_e_evidence.jsonl",
            "recipe_e_evidence_sidecar": {
                "path": "recipe_e_evidence.jsonl",
                "sha256": files["recipe_e_evidence.jsonl"]["sha256"],
                "dtype": "jsonl",
                "rows": recipe_e_evidence_stats["rows"],
            },
            "recipe_e_evidence_stats": recipe_e_evidence_stats,
            "task_columns": {
                "task_code": "tasks.npy",
                "split_code": "splits.npy",
            },
            "task_column": {
                "path": "tasks.npy",
                "sha256": files["tasks.npy"]["sha256"],
                "dtype": "u1",
                "rows": row_count,
            },
            "split_column": {
                "path": "splits.npy",
                "sha256": files["splits.npy"]["sha256"],
                "dtype": "u1",
                "rows": row_count,
            },
            "loader": {
                "deterministic_workers": 4,
                "pinned_memory_prefetch": True,
            },
            "label_shards": label_shards,
            "created_at_ms": int(time.time() * 1_000),
            "files": files,
        }
        atomic_json(temporary / "manifest.json", packed)
        os.replace(temporary, final)
        verify(packed, final)
        return packed
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
