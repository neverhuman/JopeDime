//! W3-T06: a crash between durable steps (qualification file -> decision event,
//! HWM batch -> pointer) leaves visibly incomplete, recoverable state; redelivery and
//! racing comparable teachers reproduce the winning state without a duplicate improvement.
//! Crashes are simulated by removing exactly the later step's durable effect.
use super::*;
use std::path::PathBuf;

struct LedgerMark {
    batches: BTreeSet<PathBuf>,
    head: Option<Vec<u8>>,
}

fn mark(state: &State, name: &str) -> LedgerMark {
    let ledger = Ledger::open(state.root(), name);
    LedgerMark {
        batches: ledger
            .batches()
            .unwrap()
            .into_iter()
            .map(|(_, path)| path)
            .collect(),
        head: fs::read(ledger.root().join("HEAD.json")).ok(),
    }
}

/// The process died before the batch appended after `mark` became durable.
fn crash_before_append(state: &State, name: &str, mark: LedgerMark) {
    let ledger = Ledger::open(state.root(), name);
    for (_, path) in ledger.batches().unwrap() {
        if !mark.batches.contains(&path) {
            fs::remove_file(path).unwrap();
        }
    }
    let head = ledger.root().join("HEAD.json");
    match mark.head {
        Some(bytes) => fs::write(head, bytes).unwrap(),
        None => fs::remove_file(head).unwrap(),
    }
    ledger.require_intact().unwrap();
}

fn pointer_bytes(state: &State, record: &HwmRecord) -> Option<Vec<u8>> {
    fs::read(state.pointer_path(&PointerKey::of(record)).unwrap()).ok()
}

/// The process died after the HWM batch was durable but before the pointer moved.
fn crash_before_pointer(state: &State, record: &HwmRecord, before: Option<Vec<u8>>) {
    let path = state.pointer_path(&PointerKey::of(record)).unwrap();
    match before {
        Some(bytes) => fs::write(path, bytes).unwrap(),
        None => fs::remove_file(path).unwrap(),
    }
}

fn deliver(f: &Fixture, record: &HwmRecord, tag: &str) -> hwm::IngestReport {
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: record.clone(),
            blob: candidate_blob(tag),
            chain_from_source: false,
        }],
        json!({"kind":"w3-t06"}),
    )
    .unwrap()
}

fn qualified_decisions(state: &State) -> usize {
    Ledger::open(state.root(), "eligibility")
        .events()
        .unwrap()
        .iter()
        .filter(|event| event["outcome"] == "qualified")
        .count()
}

fn transitions(state: &State) -> Vec<String> {
    hwm::events(state)
        .unwrap()
        .into_iter()
        .map(|event| serde_json::to_value(event.transition).unwrap())
        .map(|value| value.as_str().unwrap_or("none").to_owned())
        .collect()
}

fn current(state: &State, record: &HwmRecord) -> String {
    state
        .read_pointer(&PointerKey::of(record))
        .unwrap()
        .unwrap()
        .provenance
        .record_sha256
}

/// A weaker (four exact folds) and a stronger (five) teacher for one frozen comparison.
fn teachers() -> (Fixture, HwmRecord, Qualification, Qualification) {
    let f = fixture(Task::Regression, 4);
    let (strong, evidence) = f.candidate("strong", &exact_on(5));
    let weak = qualify(&f.state, &f.record, &f.evidence).unwrap();
    let stronger = qualify(&f.state, &strong, &evidence).unwrap();
    assert!(stronger.candidate_loss < weak.candidate_loss);
    (f, strong, weak, stronger)
}

#[test]
fn qualification_without_a_durable_decision_is_not_eligible_until_qualify_resumes() {
    let f = fixture(Task::Regression, 5);
    let before = mark(&f.state, "eligibility");
    qualify(&f.state, &f.record, &f.evidence).unwrap();
    crash_before_append(&f.state, "eligibility", before);
    assert!(qualification_path(&f.state, &f.record).exists());
    assert_eq!(qualified_decisions(&f.state), 0);
    let report = deliver(&f, &f.record, "");
    assert_eq!(
        report.rejects.len(),
        1,
        "a qualification with no durable decision entered the HWM"
    );
    assert_eq!(report.rejects[0].reason, "teaching-ineligible");
    assert!(f.state.pointers().unwrap().is_empty());
    assert!(freeze_teaching(&f.state).unwrap().entries.is_empty());
    assert!(revalidate_qualification(&f.state, &f.record).is_err());
    // Resume: redelivering the same submission completes the link exactly once.
    let resumed = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert_eq!(qualify(&f.state, &f.record, &f.evidence).unwrap(), resumed);
    assert_eq!(qualified_decisions(&f.state), 1);
    assert_eq!(deliver(&f, &f.record, "").accepted, 1);
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![resumed]);
    // The HWM decision names the exact stored qualification bytes it relied on.
    let stored = fs::read(qualification_path(&f.state, &f.record)).unwrap();
    assert_eq!(
        hwm::events(&f.state)
            .unwrap()
            .pop()
            .unwrap()
            .qualification_sha256,
        Some(sha256_hex(&stored))
    );
}

#[test]
fn crash_between_hwm_append_and_pointer_advance_never_lists_a_superseded_teacher() {
    let (f, strong, _, stronger) = teachers();
    assert_eq!(deliver(&f, &f.record, "").accepted, 1);
    let lagging = pointer_bytes(&f.state, &f.record);
    assert_eq!(deliver(&f, &strong, "strong").accepted, 1);
    crash_before_pointer(&f.state, &f.record, lagging);
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(
        manifest
            .entries
            .iter()
            .all(|entry| entry.record_sha256 != f.record.provenance.record_sha256),
        "the ledger superseded this teacher, yet the stale pointer presented it"
    );
    assert!(manifest.entries.is_empty() && manifest.excluded_diagnostic == 1);
    assert!(manifest.exclusions[0].1.contains("lags its HWM ledger tip"));
    // Redelivering the weaker teacher resolves against the ledger, not the stale pointer.
    let report = deliver(&f, &f.record, "");
    assert_eq!((report.accepted, report.rejected), (0, 1));
    assert_eq!(
        current(&f.state, &f.record),
        strong.provenance.record_sha256
    );
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![stronger]);
    let verified = hwm::verify(&f.state, &[]).unwrap();
    assert!(verified.ok(), "{:?}", verified.breaks);
    assert_eq!((verified.initialized, verified.improved), (1, 1));
}

#[test]
fn first_acceptance_lost_before_its_pointer_is_visible_not_silently_absent() {
    let f = fixture(Task::Regression, 5);
    let q = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert_eq!(deliver(&f, &f.record, "").accepted, 1);
    crash_before_pointer(&f.state, &f.record, None);
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert_eq!(
        manifest.excluded_diagnostic, 1,
        "an accepted teacher without its pointer vanished from the manifest"
    );
    assert!(manifest.exclusions[0].1.contains("has no pointer"));
    let report = deliver(&f, &f.record, "");
    assert_eq!(report.accepted, 0, "the lost acceptance was counted twice");
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![q]);
    let verified = hwm::verify(&f.state, &[]).unwrap();
    assert!(verified.ok(), "{:?}", verified.breaks);
    assert_eq!((verified.initialized, verified.improved), (1, 0));
}

#[test]
fn stronger_and_weaker_teachers_race_in_both_orders_to_the_same_state() {
    for strong_first in [false, true] {
        let (f, strong, _, stronger) = teachers();
        let order = if strong_first {
            [(&strong, "strong"), (&f.record, "")]
        } else {
            [(&f.record, ""), (&strong, "strong")]
        };
        for (record, tag) in order {
            deliver(&f, record, tag);
        }
        assert_eq!(
            current(&f.state, &f.record),
            strong.provenance.record_sha256
        );
        assert_eq!(
            freeze_teaching(&f.state).unwrap().entries,
            vec![stronger.clone()]
        );
        assert_eq!(
            transitions(&f.state),
            if strong_first {
                ["initialized", "not-improved"]
            } else {
                ["initialized", "improved"]
            }
        );
        assert!(hwm::verify(&f.state, &[]).unwrap().ok());
    }
    // Concurrent delivery serialises on the HWM lock and reaches the same winner.
    let (f, strong, _, stronger) = teachers();
    std::thread::scope(|scope| {
        scope.spawn(|| deliver(&f, &f.record, ""));
        scope.spawn(|| deliver(&f, &strong, "strong"));
    });
    assert_eq!(
        current(&f.state, &f.record),
        strong.provenance.record_sha256
    );
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![stronger]);
    let verified = hwm::verify(&f.state, &[]).unwrap();
    assert!(verified.ok() && verified.initialized == 1 && verified.improved <= 1);
}
