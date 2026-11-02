use super::*;

#[cfg(unix)]
#[test]
fn symlink_files_directories_and_roots_are_rejected() {
    use std::os::unix::fs::symlink;
    for relative in ["binary/linear.json", "binary", ""] {
        let (temporary, root) = corpus();
        let source = if relative.is_empty() {
            root.clone()
        } else {
            root.join(relative)
        };
        let detached = temporary.path().join("detached");
        fs::rename(&source, &detached).unwrap();
        symlink(&detached, &source).unwrap();
        assert!(verify_inputs(&root).is_err(), "symlink {relative}");
        if relative.is_empty() {
            assert!(verify_inputs(&root.join(".")).is_err());
            assert!(verify_inputs(&root.join("")).is_err());
        }
    }
}
