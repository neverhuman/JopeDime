use super::*;

#[test]
fn missing_unmanifested_and_oversized_files_fail() {
    let (_temporary, root) = corpus();
    fs::remove_file(root.join("binary/linear.json")).unwrap();
    assert!(verify_inputs(&root).is_err());
    let (_temporary, root) = corpus();
    fs::write(root.join("regression/not_admitted.json"), b"{}").unwrap();
    assert!(verify_inputs(&root).is_err());
    let (_temporary, root) = corpus();
    fs::write(
        root.join("catalog.json"),
        vec![b' '; MAX_FILE_BYTES as usize + 1],
    )
    .unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("byte limit"));
}
