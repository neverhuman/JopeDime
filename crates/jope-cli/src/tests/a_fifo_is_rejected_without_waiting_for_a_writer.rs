use super::*;

#[cfg(unix)]
#[test]
fn a_fifo_is_rejected_without_waiting_for_a_writer() {
    let (_temporary, root) = corpus();
    let path = root.join("binary/linear.json");
    fs::remove_file(&path).unwrap();
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        &path,
        rustix::fs::Mode::from_raw_mode(0o600),
    )
    .unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("regular file"));
}
