//! W3 context provenance through the real teaching ingress: `qualify` -> `hwm::ingest`
//! -> `freeze_teaching`/`revalidate_replay`. Fixtures are synthetic; this proves the
//! receiving-gate contract, not a live producer or corpus qualification.
use super::*;

/// Authority-bearing state only. Rejected ledger events are submitted-record inventory.
#[derive(Debug, PartialEq)]
pub(super) struct Authority {
    pub(super) qualified: Vec<String>,
    pub(super) pointers: Vec<(String, String)>,
    pub(super) accepted: Vec<String>,
    pub(super) teaching: Vec<String>,
}

pub(super) fn authority(state: &State) -> Authority {
    let mut qualified = walkdir::WalkDir::new(state.root().join("eligibility/qualified"))
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| {
            entry
                .path()
                .strip_prefix(state.root())
                .unwrap()
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    qualified.sort();
    Authority {
        qualified,
        pointers: state
            .pointers()
            .unwrap()
            .into_iter()
            .map(|(key, record)| (key.rel(), record.provenance.record_sha256))
            .collect(),
        accepted: hwm::events(state)
            .unwrap()
            .into_iter()
            .filter(|event| event.outcome == "accepted")
            .map(|event| event.record.provenance.record_sha256)
            .collect(),
        teaching: freeze_teaching(state)
            .unwrap()
            .entries
            .into_iter()
            .map(|entry| entry.record_sha256)
            .collect(),
    }
}

fn submit(f: &Fixture, evidence: &OofEvidence) -> (Result<Qualification>, hwm::IngestReport) {
    submit_record(f, &f.record, evidence)
}

fn submit_record(
    f: &Fixture,
    record: &HwmRecord,
    evidence: &OofEvidence,
) -> (Result<Qualification>, hwm::IngestReport) {
    let qualified = qualify(&f.state, record, evidence);
    (qualified, ingest_record(f, record))
}

fn ingest(f: &Fixture) -> hwm::IngestReport {
    ingest_record(f, &f.record)
}

fn ingest_record(f: &Fixture, record: &HwmRecord) -> hwm::IngestReport {
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: record.clone(),
            blob: b"synthetic candidate".to_vec(),
            chain_from_source: false,
        }],
        json!({"kind":"w3-provenance"}),
    )
    .unwrap()
}

pub(super) fn rejected_decisions(state: &State) -> Vec<String> {
    Ledger::open(state.root(), "eligibility")
        .events()
        .unwrap()
        .into_iter()
        .filter(|event| event["outcome"] == "rejected")
        .map(|event| event["reason"].as_str().unwrap().to_owned())
        .collect()
}

fn with_context(f: &Fixture, context: Option<ContextEvidence>) -> OofEvidence {
    let mut evidence = f.evidence.clone();
    evidence.context = context;
    evidence
}

fn valid(f: &Fixture) -> ContextEvidence {
    f.evidence.context.clone().unwrap()
}

/// Authorized positive control: the same fixture then admits with valid provenance.
pub(super) fn assert_positive_control(f: &Fixture) {
    let (qualified, report) = submit(f, &f.evidence);
    let qualified = qualified.unwrap();
    assert_eq!((report.accepted, report.rejected), (1, 0));
    assert_eq!(
        qualified.context_receipt_sha256.as_deref(),
        Some(valid(f).receipt_sha256.as_str())
    );
    let after = authority(&f.state);
    assert_eq!(after.qualified.len(), 1);
    assert_eq!(after.pointers.len(), 1);
    assert_eq!(
        after.accepted,
        vec![f.record.provenance.record_sha256.clone()]
    );
    assert_eq!(
        after.teaching,
        vec![f.record.provenance.record_sha256.clone()]
    );
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![qualified]);
}

pub(super) fn assert_refused(f: &Fixture, evidence: &OofEvidence, expected: &str, label: &str) {
    assert_refused_record(f, &f.record, evidence, expected, label)
}

/// Refused through `qualify` and `hwm::ingest`, inventoried, and no authority changed.
pub(super) fn assert_refused_record(
    f: &Fixture,
    record: &HwmRecord,
    evidence: &OofEvidence,
    expected: &str,
    label: &str,
) {
    let before = authority(&f.state);
    let inventoried = rejected_decisions(&f.state).len();
    let (qualified, report) = submit_record(f, record, evidence);
    let error = format!("{:#}", qualified.expect_err(label));
    assert!(error.contains(expected), "{label}: {error}");
    assert_eq!(report.rejects.len(), 1, "{label}");
    assert_eq!(report.rejects[0].reason, "teaching-ineligible", "{label}");
    assert_eq!(before, authority(&f.state), "{label} changed authority");
    let decisions = rejected_decisions(&f.state);
    assert_eq!(decisions.len(), inventoried + 1, "{label} not inventoried");
    assert!(decisions.last().unwrap().contains(expected), "{label}");
}

#[path = "provenance/cases.rs"]
mod cases;

#[test]
fn missing_context_provenance_is_refused_at_real_teaching_ingress() {
    cases::missing_context_provenance_is_refused_at_real_teaching_ingress();
}

#[test]
fn wrong_scope_consumer_encoder_or_altered_context_is_refused_before_authority() {
    cases::wrong_scope_consumer_encoder_or_altered_context_is_refused_before_authority();
}

#[test]
fn freeze_without_preregistered_context_encoder_is_diagnostic_only() {
    cases::freeze_without_preregistered_context_encoder_is_diagnostic_only();
}

#[test]
fn retained_context_is_revalidated_by_every_receiving_gate() {
    cases::retained_context_is_revalidated_by_every_receiving_gate();
}

#[test]
fn legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded() {
    cases::legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded();
}

#[test]
fn inner_partition_scope_forbids_outer_hold_rows() {
    cases::inner_partition_scope_forbids_outer_hold_rows();
}

#[test]
fn requalification_is_refused_in_place_and_happens_only_through_a_new_qualification() {
    cases::requalification_is_refused_in_place_and_happens_only_through_a_new_qualification();
}
