//! W12-T05 at the W3 teacher/HWM receiving path. A teacher whose provenance closure names revoked
//! evidence is refused, and so is every descendant of a revoked parent. The revocations come from
//! the claims/revocation graph (`jope.core-ip-claims/v1`), imported into this host's append-only
//! revocation ledger. Unrelated revocations change nothing.
use super::provenance::{assert_positive_control, assert_refused, authority};
use super::*;
use crate::revocation::{import, testing, Revocations};

/// The retained bytes of the context's parent receipt and of the context receipt.
fn context_bytes(f: &Fixture) -> (Vec<u8>, Vec<u8>) {
    let context = f.evidence.context.as_ref().unwrap();
    let receipt: crate::encoder_receipt::EncoderReceipt =
        blob(&f.state, &context.receipt_sha256).unwrap();
    let parent = &receipt.inputs.parent_receipts[0];
    (
        f.state.read_blob(parent).unwrap().unwrap(),
        f.state.read_blob(&context.receipt_sha256).unwrap().unwrap(),
    )
}

/// A claims record pinning the parent receipt, the context receipt that descends from it, and an
/// unrelated node; `revoked` names the nodes revoked.
fn claims(f: &Fixture, repo: &Path, revoked: &[&str]) -> std::path::PathBuf {
    let (parent, receipt) = context_bytes(f);
    testing::claims(
        repo,
        &[
            ("E-parent-receipt", &parent, &[]),
            ("E-context-receipt", &receipt, &["E-parent-receipt"]),
            ("E-unrelated", b"unrelated evidence", &[]),
        ],
        revoked,
    )
}

fn revoke(f: &Fixture, repo: &Path, revoked: &[&str]) {
    let (parent, receipt) = context_bytes(f);
    testing::revoke(
        &f.state,
        repo,
        &[
            ("E-parent-receipt", &parent, &[]),
            ("E-context-receipt", &receipt, &["E-parent-receipt"]),
            ("E-unrelated", b"unrelated evidence", &[]),
        ],
        revoked,
    );
}

#[test]
fn a_revoked_parent_or_its_descendant_is_refused_at_teacher_admission() {
    // Revoking the context's parent receipt revokes the context receipt too (its descendant in
    // the claims graph); the teacher's closure names both.
    let f = fixture(Task::Regression, 5);
    let repo = tempfile::tempdir().unwrap();
    revoke(&f, repo.path(), &["E-parent-receipt"]);
    assert_eq!(Revocations::load(&f.state).unwrap().status().revoked, 2);
    assert_refused(&f, &f.evidence, "RevokedEvidence", "revoked parent");

    // A revocation of the context receipt alone refuses it as well.
    let f = fixture(Task::Regression, 5);
    let repo = tempfile::tempdir().unwrap();
    revoke(&f, repo.path(), &["E-context-receipt"]);
    assert_refused(
        &f,
        &f.evidence,
        "RevokedEvidence",
        "revoked context receipt",
    );

    // The consumer walks the context's parents itself: a revoked parent refuses its descendant
    // even when the claims graph does not name that descendant.
    let f = fixture(Task::Regression, 5);
    let repo = tempfile::tempdir().unwrap();
    let (parent, _) = context_bytes(&f);
    testing::revoke(
        &f.state,
        repo.path(),
        &[("E-parent-receipt", &parent, &[])],
        &["E-parent-receipt"],
    );
    assert_refused(&f, &f.evidence, "RevokedEvidence", "revoked parent alone");

    // A revoked fold output refuses the candidate: one fold's fitted state.
    let f = fixture(Task::Regression, 5);
    let fold: FoldPredictions = blob(&f.state, &f.evidence.fold_predictions_sha256[2]).unwrap();
    let fitted = f
        .state
        .read_blob(&fold.fitted_artifact_sha256)
        .unwrap()
        .unwrap();
    let repo = tempfile::tempdir().unwrap();
    testing::revoke(
        &f.state,
        repo.path(),
        &[("E-fold-2-fitted", &fitted, &[])],
        &["E-fold-2-fitted"],
    );
    assert_refused(&f, &f.evidence, "RevokedEvidence", "revoked fold output");

    // A revoked dataset input refuses every teacher that depends on it: the frozen targets.
    let f = fixture(Task::Regression, 5);
    let policy = load_freeze(&f.state, &f.evidence.freeze).unwrap();
    let targets = f.state.read_blob(&policy.targets_sha256).unwrap().unwrap();
    let repo = tempfile::tempdir().unwrap();
    testing::revoke(
        &f.state,
        repo.path(),
        &[("E-targets", &targets, &[])],
        &["E-targets"],
    );
    assert_refused(&f, &f.evidence, "RevokedEvidence", "revoked targets");

    // Positive control: an unrelated revocation changes nothing.
    let f = fixture(Task::Regression, 5);
    let repo = tempfile::tempdir().unwrap();
    revoke(&f, repo.path(), &["E-unrelated"]);
    assert_positive_control(&f);
}

#[test]
fn a_revocation_after_admission_withdraws_the_teacher_and_is_never_undone() {
    let f = fixture(Task::Regression, 5);
    qualify(&f.state, &f.record, &f.evidence).unwrap();
    let report = hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"kind": "w12-revocation"}),
    )
    .unwrap();
    assert_eq!(report.accepted, 1);
    assert_eq!(freeze_teaching(&f.state).unwrap().entries.len(), 1);
    let before = authority(&f.state);

    let repo = tempfile::tempdir().unwrap();
    revoke(&f, repo.path(), &["E-parent-receipt"]);
    // Every receiving gate that revalidates refuses it; nothing already durable is deleted.
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert!(
        manifest.exclusions[0].1.contains("RevokedEvidence"),
        "{:?}",
        manifest.exclusions
    );
    let refused = format!(
        "{:#}",
        revalidate_qualification(&f.state, &f.record).unwrap_err()
    );
    assert!(refused.contains("RevokedEvidence"), "{refused}");
    let after = authority(&f.state);
    assert_eq!(
        (&after.qualified, &after.pointers, &after.accepted),
        (&before.qualified, &before.pointers, &before.accepted)
    );

    // Importing the same document again appends nothing: the ledger is append-only.
    let path = claims(&f, repo.path(), &["E-parent-receipt"]);
    let again = import(&f.state, &path, Some(repo.path()), true).unwrap();
    assert!(again.new.is_empty() && again.batch.is_none());
    // A document that no longer revokes it does not un-revoke it here.
    let path = claims(&f, repo.path(), &[]);
    let edited = import(&f.state, &path, Some(repo.path()), true).unwrap();
    assert_eq!(
        edited.no_longer_in_document,
        ["E-context-receipt", "E-parent-receipt"]
    );
    assert!(freeze_teaching(&f.state).unwrap().entries.is_empty());
    // A document that re-pins a revoked node under other bytes is refused, writing nothing.
    let ledger = f.state.root().join(crate::revocation::LEDGER);
    let snapshot = |p: &Path| {
        walkdir::WalkDir::new(p)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|e| (e.path().to_path_buf(), fs::read(e.path()).unwrap()))
            .collect::<Vec<_>>()
    };
    let held = snapshot(&ledger);
    let repinned = testing::claims(
        repo.path(),
        &[("E-parent-receipt", b"other bytes", &[])],
        &["E-parent-receipt"],
    );
    let error = format!(
        "{:#}",
        import(&f.state, &repinned, Some(repo.path()), true).unwrap_err()
    );
    assert!(error.contains("re-pin"), "{error}");
    assert_eq!(snapshot(&ledger), held);
}
