use super::*;

#[test]
#[cfg(unix)]
fn individual_dataset_reads_reject_unsafe_objects_and_oversized_payloads() {
    use crate::kernel::read_dataset;
    use std::os::unix::fs::symlink;
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("input.json");
    fs::write(&path, serialize(&dataset(Task::Regression, 0)).unwrap()).unwrap();
    // Positive control: the same bytes through a regular file are admitted.
    assert_eq!(read_dataset(&path).unwrap(), dataset(Task::Regression, 0));
    let linked = temporary.path().join("linked.json");
    symlink(&path, &linked).unwrap();
    let error = format!("{:#}", read_dataset(&linked).unwrap_err());
    assert!(error.contains("without symlinks"), "{error}");
    let error = format!("{:#}", read_dataset(temporary.path()).unwrap_err());
    assert!(error.contains("read dataset"), "{error}");
    let directory = temporary.path().join("directory.json");
    fs::create_dir(&directory).unwrap();
    let error = format!("{:#}", read_dataset(&directory).unwrap_err());
    assert!(error.contains("regular file"), "{error}");
    fs::write(&path, vec![b' '; MAX_FILE_BYTES as usize + 1]).unwrap();
    let error = format!("{:#}", read_dataset(&path).unwrap_err());
    assert!(error.contains("fixture exceeds byte limit"), "{error}");
}
