use super::*;
#[test]
fn crash_after_intent_keeps_original_eligibility_identity() {
    let temp = tempfile::tempdir().unwrap();
    let original = br#"{"verified_unix_ms":111,"authorization_issued":false}"#;
    install_bytes(
        temp.path(),
        "00000001.intent.json",
        b"durable original intent includes exact proof",
    )
    .unwrap();
    assert_eq!(journal_length(temp.path()).unwrap(), 1);
    assert!(!temp.path().join("00000001.eligibility.json").exists());
    let first = install_bytes(temp.path(), "00000001.eligibility.json", original).unwrap();
    // Retry is a byte-for-byte reconstruction, never a fresh timestamp.
    let second = install_bytes(temp.path(), "00000001.eligibility.json", original).unwrap();
    assert_eq!(first, second);
    assert!(install_bytes(
        temp.path(),
        "00000001.eligibility.json",
        br#"{"verified_unix_ms":222}"#
    )
    .is_err());
    assert_eq!(read(&first.path).unwrap(), original);
}
#[test]
fn ack_without_policy_is_a_retained_incomplete_transaction_not_an_orphan() {
    let temp = tempfile::tempdir().unwrap();
    install_bytes(temp.path(), "00000001.intent.json", b"original intent").unwrap();
    install_bytes(temp.path(), "00000001.proof.json", b"original remote ACK").unwrap();
    assert_eq!(journal_length(temp.path()).unwrap(), 1);
    assert!(!temp.path().join("00000001.installed.json").exists());
    // A missing intent must not hide a later reservation or turn the ACK
    // into a new sequence. Every original byte remains available to restore.
    fs::remove_file(temp.path().join("00000001.intent.json")).unwrap();
    assert!(journal_length(temp.path()).is_err());
    assert_eq!(
        read(&temp.path().join("00000001.proof.json")).unwrap(),
        b"original remote ACK"
    );
    install_bytes(temp.path(), "00000001.intent.json", b"original intent").unwrap();
    install_bytes(
        temp.path(),
        "00000003.intent.json",
        b"unexpected later sequence",
    )
    .unwrap();
    assert!(journal_length(temp.path()).is_err());
}
#[test]
fn retained_failed_staging_does_not_collide_with_same_process_retry() {
    let temp = tempfile::tempdir().unwrap();
    let prior = temp.path().join(format!(
        ".pending-{}-{}",
        std::process::id(),
        hash(b"same bytes")
    ));
    fs::write(&prior, b"unfinished original bytes").unwrap();
    let installed = install_bytes(temp.path(), "00000001.intent.json", b"same bytes").unwrap();
    assert_eq!(read(&installed.path).unwrap(), b"same bytes");
    assert_eq!(fs::read(prior).unwrap(), b"unfinished original bytes");
    assert_eq!(journal_length(temp.path()).unwrap(), 1);
}
