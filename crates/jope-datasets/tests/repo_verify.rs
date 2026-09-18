use jope_datasets::repo::verify_repo;
use std::path::Path;

#[test]
fn committed_repository_passes() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test");
    let rep = verify_repo(&repo, None).unwrap();
    assert!(rep.errors.is_empty(), "{:?}", rep.errors);
    assert_eq!(rep.datasets, 10);
    assert!(rep.total_bytes <= jope_datasets::repo::MAX_TOTAL_BYTES);
    assert_eq!(rep.collection.as_deref(), Some("test-10"));
}

#[test]
fn tampering_is_detected() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets");
    let tmp = tempfile::tempdir().unwrap();
    let copy = tmp.path().join("datasets");
    copy_dir(&repo, &copy);
    let slug = copy.join("test").join("pmlb-229-pwlinear");
    let mut csv = std::fs::read_to_string(slug.join("train.csv")).unwrap();
    csv.insert_str(0, "a1,a2,target\n");
    std::fs::write(slug.join("train.csv"), csv).unwrap();
    std::fs::write(slug.join("LICENSE"), "changed\n").unwrap();
    let rep = verify_repo(&copy.join("test"), None).unwrap();
    let text = rep.errors.join("\n");
    assert!(text.contains("sha256"), "{text}");
    assert!(text.contains("looks like a header"), "{text}");
    assert!(text.contains("license.text_sha256"), "{text}");
    // An eleventh dataset breaks the cap.
    copy_dir(&slug, &copy.join("test").join("eleventh"));
    let rep = verify_repo(&copy.join("test"), None).unwrap();
    assert!(
        rep.errors.iter().any(|e| e.contains("exceeds the cap 10")),
        "{:?}",
        rep.errors
    );
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}
