use super::*;

#[cfg(unix)]
#[test]
fn swapped_root_and_file_paths_cannot_change_bytes_already_opened() {
    use std::os::unix::fs::symlink;
    let (temporary, root) = corpus();
    let directory = Directory::open(&root).unwrap();
    let original = directory
        .read("binary/linear.json", MAX_FILE_BYTES)
        .unwrap();
    let detached = temporary.path().join("detached");
    fs::rename(&root, &detached).unwrap();
    let outside = temporary.path().join("outside");
    fs::create_dir(&outside).unwrap();
    symlink(&outside, &root).unwrap();
    assert_eq!(
        directory
            .read("binary/linear.json", MAX_FILE_BYTES)
            .unwrap(),
        original
    );
    fs::rename(
        detached.join("binary/linear.json"),
        detached.join("old.json"),
    )
    .unwrap();
    fs::write(detached.join("binary/linear.json"), b"tampered").unwrap();
    let parsed: Dataset = serde_json::from_slice(&original).unwrap();
    validate_dataset(&parsed).unwrap();
    assert_ne!(
        digest(
            &directory
                .read("binary/linear.json", MAX_FILE_BYTES)
                .unwrap()
        ),
        digest(&original)
    );
}
