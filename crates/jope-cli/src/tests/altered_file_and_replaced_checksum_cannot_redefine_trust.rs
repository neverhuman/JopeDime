use super::*;

#[test]
fn altered_file_and_replaced_checksum_cannot_redefine_trust() {
    let (_temporary, root) = corpus();
    let path = root.join("binary/linear.json");
    let mut data: Dataset = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    data.rows[0].target = 1 - data.rows[0].target;
    let bytes = serialize(&data).unwrap();
    fs::write(&path, &bytes).unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("digest mismatch"));
    rewrite_catalog(&root, |catalog| {
        catalog.datasets[0].sha256 = digest(&bytes);
        catalog.datasets[0].bytes = bytes.len() as u64;
    });
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("pinned deterministic generator"));
}
