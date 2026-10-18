use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn verify_i53_contract() {
    let root = workspace_root();
    let checklist_path =
        root.join("docs/plans/algorithmic-selfplay-20261017/LOCAL_CUSTODY_CHECKLIST.md");
    let bytes = fs::read(&checklist_path).expect("read LOCAL_CUSTODY_CHECKLIST.md");
    let text = std::str::from_utf8(&bytes).expect("utf-8 LOCAL_CUSTODY_CHECKLIST.md");
    let digest = sha256_hex(&bytes);
    assert_eq!(digest.len(), 64);

    for required in [
        "Symlink component/contained .venv link is HOLD",
        "Merged commit alone does not prove filesystem disposable",
        "Retired exp-next-02 remains unprobed; old paper tree remains held",
        "HOLD; do not delete a containing folder",
        "source merge does not close filesystem custody",
    ] {
        assert!(
            text.contains(required),
            "LOCAL_CUSTODY_CHECKLIST.md missing required I53 invariant: {required}"
        );
    }
}

pub fn verify_i62_contract() {
    let root = workspace_root();
    let archive_path = root.join("docs/plans/algorithmic-selfplay-20261017/ARCHIVE_CANDIDATES.md");
    let subsystem_path = root.join("docs/subsystems/algorithmic-selfplay.md");
    let archive_bytes = fs::read(&archive_path).expect("read ARCHIVE_CANDIDATES.md");
    let subsystem_bytes = fs::read(&subsystem_path).expect("read algorithmic-selfplay.md");
    let archive_text = std::str::from_utf8(&archive_bytes).expect("utf-8 ARCHIVE_CANDIDATES.md");
    let subsystem_text =
        std::str::from_utf8(&subsystem_bytes).expect("utf-8 algorithmic-selfplay.md");

    assert_eq!(sha256_hex(&archive_bytes).len(), 64);
    assert_eq!(sha256_hex(&subsystem_bytes).len(), 64);

    for required in [
        "Unknown consumers block archive",
        "UNKNOWN / HOLD",
        "Module/feature test inventories compared before/after proposed transition",
        "Paths and Rust modules",
        "Registered test targets",
        "Source hashing changes get explicit identity migration",
        "task-named evaluator identity transition",
    ] {
        assert!(
            archive_text.contains(required),
            "ARCHIVE_CANDIDATES.md missing required I62 invariant: {required}"
        );
    }

    assert!(
        subsystem_text.contains("Unknown consumers block archive"),
        "algorithmic-selfplay.md missing required I62 invariant"
    );
}

#[test]
fn i53_executable_contract_test() {
    verify_i53_contract();
}

#[test]
fn i62_executable_contract_test() {
    verify_i62_contract();
}

#[test]
fn i53_and_i62_contracts_execute() {
    verify_i53_contract();
    verify_i62_contract();
}
