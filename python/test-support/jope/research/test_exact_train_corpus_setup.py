from __future__ import annotations

from test_exact_train import *

def prepare_corpus(
    min_rows, freeze_watermark, corpus_mode, catalog_binding_path,
    corpus_db, root, sampler_tag, create_catalog_binding, sha256_file,
    complete_causal_members,
):
    if not corpus_db.is_file():
        raise RuntimeError("strict corpus database does not exist yet")
    if corpus_mode not in {"recent_plus_replay", "full_consolidation"}:
        raise ValueError("unsupported continuation corpus mode")
    catalog_binding_path = (
        catalog_binding_path or create_catalog_binding()
    ).resolve()
    binding = json.loads(catalog_binding_path.read_text())
    if binding.get("schema") != "CatalogBinding.v1":
        raise RuntimeError("unsupported catalog binding")
    binding_sha256 = sha256_file(catalog_binding_path)
    con = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    # The frozen CTE sorts ~2-3M wide rows by a 5-column key and the per-dataset
    # ROW_NUMBER() window re-sorts that output. On SQLite's default ~2 MB page
    # cache every one of those sorts spills to disk: a build was observed doing
    # 122 GB of reads and 46 GB of temp writes against an 8 GB database and still
    # not finishing after 17 minutes. Give the sorter real memory instead.
    con.execute("PRAGMA cache_size = -4000000")  # ~4 GB, negative = KiB
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA mmap_size = 8589934592")  # 8 GB mmap of the DB file
    pointer = {
        "catalog": binding["catalog_path"],
        "catalog_sha256": binding["catalog_sha256"],
        "data": binding["data_path"],
        "data_sha256": binding["data_sha256"],
    }
    catalog = json.loads(Path(binding["catalog_path"]).read_text())
    entries = {
        (entry["task"], entry["dataset_hash"]): entry
        for entry in catalog["entries"]
    }
    total = int(con.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
    if total < min_rows:
        raise RuntimeError(f"strict corpus has {total} rows; needs {min_rows}")
    watermark = total if freeze_watermark is None else int(freeze_watermark)
    if watermark < min_rows or watermark > total:
        raise RuntimeError(
            f"cannot freeze watermark {watermark} from {total} strict labels"
        )
    causal_members, causal_pair_stats = complete_causal_members(con)
    # SAMPLER_TAG is part of the identity: the snapshot used to be keyed on the
    # watermark alone, so changing the sampler and rebuilding at an unchanged
    # watermark returned the OLD corpus from the early-return below without a
    # word of warning.
    legacy_snapshot = (
        root
        / "training"
        / f"snapshot-{watermark:09d}-{corpus_mode}-{sampler_tag}"
    )
    snapshot = legacy_snapshot.with_name(
        f"{legacy_snapshot.name}-c{binding_sha256[:12]}"
    )
    if legacy_snapshot.exists():
        legacy_manifest = legacy_snapshot / "corpus-manifest.json"
        if (
            legacy_manifest.is_file()
            and json.loads(legacy_manifest.read_text()).get("catalog_sha256")
            == binding["catalog_sha256"]
        ):
            snapshot = legacy_snapshot
    return {
        'con': con,
        'catalog_binding_path': catalog_binding_path,
        'binding': binding,
        'binding_sha256': binding_sha256,
        'pointer': pointer,
        'catalog': catalog,
        'entries': entries,
        'total': total,
        'watermark': watermark,
        'causal_members': causal_members,
        'causal_pair_stats': causal_pair_stats,
        'legacy_snapshot': legacy_snapshot,
        'snapshot': snapshot,
    }
