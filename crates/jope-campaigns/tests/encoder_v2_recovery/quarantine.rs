use super::*;

#[test]
fn quarantine_is_exact_and_denied_for_every_consumption_class() {
    let quarantine = quarantine();
    quarantine.validate().unwrap();
    assert!(quarantine.digest().is_ok());
    for operation in [
        "discovery",
        "catalog",
        "training",
        "benchmark",
        "V2 import",
        "GC",
    ] {
        assert!(quarantine
            .require_operation_allowed("incident-00", operation)
            .is_err());
    }
    let mut permissive = quarantine;
    permissive.policy.gc_allowed = true;
    assert!(permissive.validate().is_err());
}

#[test]
fn quarantine_relocation_is_same_filesystem_atomic_and_stops_before_a_mismatch() {
    let directory = tempfile::tempdir().unwrap();
    let index = filesystem_quarantine(directory.path());
    let receipts = relocate_v1_quarantine(&index).unwrap();
    assert_eq!(receipts.len(), 18);
    assert!(receipts.iter().all(|receipt| receipt.atomic_rename));
    assert!(index
        .trees
        .iter()
        .all(|tree| !std::path::Path::new(&tree.original_path).exists()
            && std::path::Path::new(&tree.quarantine_path).is_dir()));

    let mismatch_root = tempfile::tempdir().unwrap();
    let mismatch = filesystem_quarantine(mismatch_root.path());
    fs::write(
        std::path::Path::new(&mismatch.trees[17].original_path).join("cache-manifest.v1.json"),
        b"drift",
    )
    .unwrap();
    assert!(relocate_v1_quarantine(&mismatch).is_err());
    assert!(mismatch
        .trees
        .iter()
        .all(|tree| std::path::Path::new(&tree.original_path).is_dir()));
}
