//! The production retention step, without clients, allocation, keys or numerical work.
use super::*;
use std::{
    collections::BTreeMap,
    fs::FileTimes,
    time::{Duration, UNIX_EPOCH},
};

#[derive(Debug, PartialEq, Eq)]
enum Entry {
    Directory,
    File(Vec<u8>),
    Symlink(PathBuf),
}
fn inventory(root: &Path) -> BTreeMap<PathBuf, Entry> {
    walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .map(|entry| {
            let entry = entry.unwrap();
            let kind = entry.file_type();
            let value = if kind.is_symlink() {
                Entry::Symlink(fs::read_link(entry.path()).unwrap())
            } else if kind.is_dir() {
                Entry::Directory
            } else {
                Entry::File(fs::read(entry.path()).unwrap())
            };
            (entry.path().strip_prefix(root).unwrap().to_owned(), value)
        })
        .collect()
}
fn fixture(read_only: bool) -> (tempfile::TempDir, Graph) {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/P6/native-proof-retention");
    fs::create_dir_all(&base).unwrap();
    let temporary = tempfile::tempdir_in(base).unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let directory = root.join("graph");
    let worker = root.join("worker");
    fs::create_dir(&directory).unwrap();
    fs::create_dir(&worker).unwrap();
    let tenant = TenantId::new("system:jope").unwrap();
    let project = ProjectId::new("proof-retention").unwrap();
    let graph = if read_only {
        Graph::read_only(&root, &directory, &worker, tenant, project)
    } else {
        Graph::open(&root, &directory, &worker, tenant, project)
    }
    .unwrap();
    (temporary, graph)
}

#[test]
fn readonly_missing_proof_does_not_create_proof_or_parent_directories() {
    for existing_parent in [false, true] {
        let (_temporary, graph) = fixture(true);
        let path = graph.directory.join("tasks/fixture/native-proof-7.json");
        if existing_parent {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
        }
        let before = inventory(&graph.root);
        assert!(
            graph
                .retain_native_proof(&path, br#"{"proof":"fixture"}"#)
                .is_err(),
            "a read-only verifier created previously missing evidence"
        );
        assert_eq!(inventory(&graph.root), before);
        assert!(!path.exists());
        assert!(!graph.directory.join("coordinator.lock").exists());
    }
}

#[test]
fn readonly_conflicting_or_unsafe_proof_keeps_every_original_and_creates_no_conflict() {
    use std::os::unix::fs::symlink;
    for case in 0..3 {
        let (_temporary, graph) = fixture(true);
        let path = graph.directory.join("tasks/fixture/native-proof-7.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let marker = graph.root.join("preserved-marker");
        fs::write(&marker, b"preserved fixture bytes").unwrap();
        match case {
            0 => fs::write(&path, b"original proof").unwrap(),
            1 => symlink(&marker, &path).unwrap(),
            _ => fs::create_dir(&path).unwrap(),
        }
        let before = inventory(&graph.root);
        assert!(graph
            .retain_native_proof(&path, b"different proof")
            .is_err());
        assert_eq!(
            inventory(&graph.root),
            before,
            "case {case} wrote retention/conflict artifacts"
        );
        assert_eq!(fs::read(&marker).unwrap(), b"preserved fixture bytes");
    }
}

#[test]
fn readonly_match_never_stages_a_file_and_mutable_retention_keeps_its_original_behavior() {
    let (_temporary, graph) = fixture(true);
    let path = graph.directory.join("tasks/fixture/native-proof-7.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"existing proof").unwrap();
    // Exact existing bytes previously staged/deleted a temporary anyway. A deliberate past
    // directory mtime makes that mutation observable without timing or permission assumptions.
    let parent = fs::File::open(path.parent().unwrap()).unwrap();
    parent
        .set_times(FileTimes::new().set_modified(UNIX_EPOCH + Duration::from_secs(1)))
        .unwrap();
    let modified = parent.metadata().unwrap().modified().unwrap();
    let before = inventory(&graph.root);
    graph.retain_native_proof(&path, b"existing proof").unwrap();
    assert_eq!(inventory(&graph.root), before);
    assert_eq!(
        parent.metadata().unwrap().modified().unwrap(),
        modified,
        "read-only matching replay staged a temporary or changed the directory"
    );

    let (_mutable_temporary, mutable) = fixture(false);
    let path = mutable.directory.join("new/nested/native-proof-7.json");
    mutable
        .retain_native_proof(&path, b"retained proof")
        .unwrap();
    mutable
        .retain_native_proof(&path, b"retained proof")
        .unwrap();
    assert!(mutable
        .retain_native_proof(&path, b"conflicting proof")
        .is_err());
    assert_eq!(fs::read(&path).unwrap(), b"retained proof");
    assert!(
        fs::read_dir(path.parent().unwrap()).unwrap().count() >= 2,
        "mutable replay must retain its existing immutable-conflict evidence"
    );
}
