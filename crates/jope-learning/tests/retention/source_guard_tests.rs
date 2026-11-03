use super::*;

fn collect_retention_sources(directory: &Path, root: &Path, files: &mut Vec<String>) {
    for entry in fs::read_dir(directory).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            collect_retention_sources(&path, root, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(
                path.strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
}

#[test]
fn retention_sources_have_no_s3_write_or_delete_path() {
    // The brain and the write-side library are read-only/offline by construction: no S3 client,
    // no process spawning, no filesystem deletion. An apply path must be a separate reviewed tool.
    let sources = [
        ("mod.rs", include_str!("../../src/retention/mod.rs")),
        (
            "classify.rs",
            include_str!("../../src/retention/classify.rs"),
        ),
        (
            "distilled.rs",
            include_str!("../../src/retention/distilled.rs"),
        ),
        (
            "executor_candidate.rs",
            include_str!("../../src/retention/executor_candidate.rs"),
        ),
        (
            "executor_candidate/mock_namespaces.rs",
            include_str!("../../src/retention/executor_candidate/mock_namespaces.rs"),
        ),
        (
            "executor_candidate/tombstone.rs",
            include_str!("../../src/retention/executor_candidate/tombstone.rs"),
        ),
        ("groups.rs", include_str!("../../src/retention/groups.rs")),
        ("index.rs", include_str!("../../src/retention/index.rs")),
        ("inputs.rs", include_str!("../../src/retention/inputs.rs")),
        (
            "inventory.rs",
            include_str!("../../src/retention/inventory.rs"),
        ),
        (
            "lifecycle.rs",
            include_str!("../../src/retention/lifecycle.rs"),
        ),
        ("md5.rs", include_str!("../../src/retention/md5.rs")),
        (
            "obligations.rs",
            include_str!("../../src/retention/obligations.rs"),
        ),
        ("pack.rs", include_str!("../../src/retention/pack.rs")),
        ("parquet.rs", include_str!("../../src/retention/parquet.rs")),
        ("plan.rs", include_str!("../../src/retention/plan.rs")),
        ("policy.rs", include_str!("../../src/retention/policy.rs")),
        ("refs.rs", include_str!("../../src/retention/refs.rs")),
        ("snappy.rs", include_str!("../../src/retention/snappy.rs")),
        ("tags.rs", include_str!("../../src/retention/tags.rs")),
        ("value.rs", include_str!("../../src/retention/value.rs")),
        (
            "bin/jope_retention.rs",
            include_str!("../../src/bin/jope_retention.rs"),
        ),
    ];
    // Every module of the retention tree is in the list, so a new file cannot dodge the guard.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/retention");
    let mut on_disk = Vec::new();
    collect_retention_sources(&root, &root, &mut on_disk);
    on_disk.sort();
    let mut listed: Vec<String> = sources
        .iter()
        .map(|(n, _)| n.to_string())
        .filter(|n| !n.starts_with("bin/"))
        .collect();
    listed.sort();
    assert_eq!(listed, on_disk, "guard list != src/retention/*.rs");
    let forbidden = [
        "object_store",
        "jope_datasets",
        "remove_file",
        "remove_dir",
        "process::Command",
        "DeleteObject",
        "delete_object",
        "PutObject",
        "put_object",
        "PutBucketLifecycle",
        "CopyObject",
    ];
    for (name, text) in sources {
        for f in forbidden {
            assert!(!text.contains(f), "{name} contains forbidden {f:?}");
        }
    }
}
