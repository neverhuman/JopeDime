use super::*;

#[test]
fn npy_parser_reads_f32_and_shape() {
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    let mut header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }".to_vec();
    while !(bytes.len() + 2 + header.len() + 1).is_multiple_of(64) {
        header.push(b' ');
    }
    header.push(b'\n');
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&1.5_f32.to_le_bytes());
    bytes.extend_from_slice(&(-2.0_f32).to_le_bytes());
    let parsed = parse_npy(bytes.as_slice()).unwrap();
    assert_eq!(
        format!("{parsed:?}"),
        "NpyArray { shape: [2], values: [1.5, -2.0] }"
    );
}

#[test]
fn sealed_status_never_exposes_dataset_identities() {
    let temp = TempDir::new().unwrap();
    let paths = paths(&temp);
    let mut entries = Vec::new();
    for task in TASKS {
        for index in 0..sealed_target(task) {
            entries.push(InventoryEntry {
                task: task.to_owned(),
                dataset_id: format!("{index:016x}"),
                dataset_hash: format!("{index:016x}"),
                source: PathBuf::from("private"),
                source_family: format!("{task}-family-{index}"),
                train_sha256: "a".repeat(64),
                test_sha256: "b".repeat(64),
                meta_sha256: "c".repeat(64),
                mirror: None,
            });
        }
    }
    let redacted = freeze_sealed(&paths, &entries, 7).unwrap();
    let encoded = serde_json::to_string(&redacted).unwrap();
    assert!(!encoded.contains("dataset_id"));
    assert!(!encoded.contains("family-"));
    assert_eq!(
        fs::metadata(&paths.sealed_path)
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o600
    );
    let adaptive = adaptive_inventory(&paths, &entries).unwrap();
    assert!(adaptive.is_empty());
}

#[test]
fn family_split_is_deterministic_and_disjoint() {
    let family = "synthetic:abc";
    assert_eq!(
        jope_datasets::family_contracts::source_family_campaign_split(family),
        jope_datasets::family_contracts::source_family_campaign_split(family)
    );
    let split = jope_datasets::family_contracts::source_family_campaign_split(family);
    assert!(matches!(split, "train" | "validation"));
}
