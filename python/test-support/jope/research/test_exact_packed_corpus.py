#!/usr/bin/env python3
"""Content-addressed mmap corpus for internal test-exact policy training."""

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
from test_exact_packed_corpus_materialize import materialize as _materialize
from test_exact_packed_corpus_reader import Float32View, load_training_rows, verify


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
    return _materialize(
        manifest_path,
        output_root,
        tokenizer=tokenizer,
        catalog_pointer_path=catalog_pointer_path,
        catalog_binding_path=catalog_binding_path,
        label_shard_rows=LABEL_SHARD_ROWS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--catalog-pointer",
        type=Path,
        default=Path(
            "/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json"
        ),
    )
    parser.add_argument("--catalog-binding", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.manifest,
                args.output_root,
                catalog_pointer_path=args.catalog_pointer,
                catalog_binding_path=args.catalog_binding,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
