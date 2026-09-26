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

#[test]
fn missing_context_provenance_is_refused_at_real_teaching_ingress() {
    let f = fixture(Task::Regression, 5);
    assert_eq!(
        authority(&f.state),
        Authority {
            qualified: vec![],
            pointers: vec![],
            accepted: vec![],
            teaching: vec![],
        }
    );
    assert_refused(
        &f,
        &with_context(&f, None),
        "MissingEncoderProvenance",
        "missing receipt",
    );
    assert_positive_control(&f);
}

#[test]
fn wrong_scope_consumer_encoder_or_altered_context_is_refused_before_authority() {
    let f = fixture(Task::Regression, 5);
    let comparison = &f.plan.comparison;
    let scope = || context_scope(comparison, &f.rows);
    let encoder = || context_plan().encoder;
    let output = context_output(2048, 0.5);
    let parent = || {
        let receipt: crate::encoder_receipt::EncoderReceipt =
            blob(&f.state, &valid(&f).receipt_sha256).unwrap();
        receipt.inputs.parent_receipts
    };
    let produce = |scope: Scope, encoder: Encoder, parents: Vec<String>, output: &[u8]| {
        produce_context(&f.state, scope, encoder, parents, output)
    };
    let mut variants: Vec<(&str, &str, ContextEvidence)> = Vec::new();
    let mut other = comparison.clone();
    other.cut_id = sha256_hex(b"another cut");
    variants.push((
        "receipt replayed from another cut",
        "EncoderScopeMismatch",
        produce_context(
            &f.state,
            context_scope(&other, &f.rows),
            encoder(),
            parent(),
            &output,
        ),
    ));
    let mut other = comparison.clone();
    other.dataset_hash = sha256_hex(b"another dataset");
    variants.push((
        "receipt bound to another consumer dataset",
        "EncoderScopeMismatch",
        produce(context_scope(&other, &f.rows), encoder(), parent(), &output),
    ));
    let mut reordered = scope();
    reordered.fitting.reverse();
    variants.push((
        "reordered fitting rows",
        "EncoderScopeMismatch",
        produce(reordered, encoder(), parent(), &output),
    ));
    let mut partial = scope();
    partial.fitting.pop();
    variants.push((
        "fitting row omitted",
        "EncoderScopeMismatch",
        produce(partial, encoder(), parent(), &output),
    ));
    let mut declared = scope();
    declared.forbidden.push(Member {
        row: "undeclared-held-row".into(),
        group: "undeclared-held-group".into(),
    });
    variants.push((
        "different forbidden membership",
        "EncoderScopeMismatch",
        produce(declared, encoder(), parent(), &output),
    ));
    let mut family = scope();
    family.source_family = "another-family".into();
    variants.push((
        "wrong source family",
        "EncoderScopeMismatch",
        produce(family, encoder(), parent(), &output),
    ));
    variants.push((
        "unregistered encoder artifact",
        "EncoderScopeMismatch",
        produce(
            scope(),
            Encoder {
                artifact_sha256: sha256_hex(b"unregistered encoder"),
                ..encoder()
            },
            parent(),
            &output,
        ),
    ));
    variants.push((
        "unregistered encoder seed",
        "EncoderScopeMismatch",
        produce(
            scope(),
            Encoder {
                seed: 8,
                ..encoder()
            },
            parent(),
            &output,
        ),
    ));
    variants.push((
        "wrong context product",
        "wrong output ABI",
        produce(
            scope(),
            Encoder {
                product: Product::DatasetSignature512,
                ..encoder()
            },
            parent(),
            &context_output(512, 0.5),
        ),
    ));
    variants.push((
        "altered context bytes",
        "EncoderScopeMismatch",
        ContextEvidence {
            output_sha256: f.state.put_blob(&context_output(2048, 0.75)).unwrap(),
            ..valid(&f)
        },
    ));
    variants.push((
        "altered fitted context state",
        "EncoderScopeMismatch",
        ContextEvidence {
            fitted_state_sha256: f.state.put_blob(b"another fitted state").unwrap(),
            ..valid(&f)
        },
    ));
    let receipt: crate::encoder_receipt::EncoderReceipt =
        blob(&f.state, &valid(&f).receipt_sha256).unwrap();
    variants.push((
        "noncanonical receipt bytes",
        "EncoderScopeMismatch",
        ContextEvidence {
            receipt_sha256: f
                .state
                .put_blob(&serde_json::to_vec_pretty(&receipt).unwrap())
                .unwrap(),
            ..valid(&f)
        },
    ));
    variants.push((
        "unretained receipt",
        "missing",
        ContextEvidence {
            receipt_sha256: sha256_hex(b"never retained receipt"),
            ..valid(&f)
        },
    ));
    variants.push((
        "unretained output",
        "retained context artifact is missing",
        ContextEvidence {
            output_sha256: sha256_hex(b"never retained output"),
            ..valid(&f)
        },
    ));
    variants.push((
        "unresolved parent receipt",
        "unresolved context parent receipt",
        produce(
            scope(),
            encoder(),
            vec![sha256_hex(b"never retained parent")],
            &output,
        ),
    ));
    let mut outside = scope();
    outside.fitting.swap(0, 1);
    let foreign_parent = produce(
        outside,
        Encoder {
            product: Product::DatasetSignature512,
            ..encoder()
        },
        vec![],
        &context_output(512, 0.25),
    );
    variants.push((
        "parent fitted outside the admitted scope",
        "parent was fitted outside the admitted scope",
        produce(
            scope(),
            encoder(),
            vec![foreign_parent.receipt_sha256],
            &output,
        ),
    ));
    variants.push((
        "path-shaped receipt identity",
        "invalid artifact identity",
        ContextEvidence {
            receipt_sha256: "../../hwm/ledger/HEAD.json".into(),
            ..valid(&f)
        },
    ));
    for (label, expected, context) in variants {
        assert_refused(&f, &with_context(&f, Some(context)), expected, label);
    }
    assert_positive_control(&f);
}

#[test]
fn freeze_without_preregistered_context_encoder_is_diagnostic_only() {
    let f = fixture_with(Task::Regression, 5, None);
    assert!(f.evidence.context.is_some());
    assert_refused(
        &f,
        &f.evidence,
        "MissingEncoderProvenance: freeze preregistered no teaching context encoder",
        "legacy freeze",
    );
}

#[test]
fn retained_context_is_revalidated_by_every_receiving_gate() {
    let f = fixture(Task::Regression, 5);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    let output = f.state.blob_path(&valid(&f).output_sha256);
    let bytes = fs::read(&output).unwrap();
    fs::write(&output, context_output(2048, 0.75)).unwrap();
    let before = authority(&f.state);
    let report = ingest(&f);
    assert_eq!(report.rejects[0].reason, "teaching-ineligible");
    assert_eq!(before, authority(&f.state));
    let decisions = Decisions::load(&f.state).unwrap();
    assert!(require_qualified(&f.state, &f.record, &decisions).is_err());
    assert!(revalidate_qualification(&f.state, &f.record).is_err());
    fs::write(&output, &bytes).unwrap();
    assert_eq!(ingest(&f).accepted, 1);
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![qualified]);
    fs::remove_file(&output).unwrap();
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert!(manifest.exclusions[0]
        .1
        .contains("retained context artifact"));
}

#[test]
fn legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded() {
    let f = fixture(Task::Regression, 5);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert_eq!(ingest(&f).accepted, 1);
    // Simulate bytes persisted before provenance was mandatory: no context field at all.
    let mut legacy = qualified.clone();
    legacy.context_receipt_sha256 = None;
    legacy.diagnostics = None;
    let stored = StoredQualification {
        evidence: with_context(&f, None),
        qualification: legacy.clone(),
    };
    let path = qualification_path(&f.state, &f.record);
    let bytes = canonical_json(&stored).unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("context_receipt"));
    fs::write(&path, &bytes).unwrap();
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert_eq!(manifest.excluded_diagnostic, 1);
    assert!(manifest.exclusions[0]
        .1
        .contains("MissingEncoderProvenance"));
    let decisions = Decisions::load(&f.state).unwrap();
    assert!(require_qualified(&f.state, &f.record, &decisions).is_err());
    assert!(revalidate_qualification(&f.state, &f.record).is_err());
    assert!(revalidate_replay(&f.state, &[legacy]).is_err());
    // Still readable diagnostically: the pointer and accepted event are retained as history.
    assert_eq!(
        f.state
            .read_pointer(&PointerKey::of(&f.record))
            .unwrap()
            .unwrap()
            .provenance
            .record_sha256,
        f.record.provenance.record_sha256
    );
    // Presenting valid evidence later cannot silently upgrade the retained legacy decision.
    assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert!(freeze_teaching(&f.state).unwrap().entries.is_empty());
}

#[test]
fn inner_partition_scope_forbids_outer_hold_rows() {
    let f = fixture(Task::Regression, 5);
    let (plan, partition, freeze, _) = inner_fixture(&f);
    let policy = load_freeze(&f.state, &freeze).unwrap();
    let scope = context::scope(&f.state, &policy, &context_plan()).unwrap();
    assert_eq!(scope.outer_fold, Some(partition.outer_fold));
    assert_eq!(
        scope.fitting.iter().map(|m| &m.row).collect::<Vec<_>>(),
        partition.fitting_rows.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        scope.forbidden.iter().map(|m| &m.row).collect::<Vec<_>>(),
        partition.hold_rows.iter().collect::<Vec<_>>()
    );
    let mut leaking = scope.clone();
    leaking.fitting.push(scope.forbidden[0].clone());
    assert!(EncoderCapability::bind(
        leaking,
        context_plan().encoder,
        Inputs {
            features_sha256: sha256_hex(b"fixture projected features"),
            labels_sha256: None,
            parent_receipts: vec![],
        },
    )
    .is_err());
    let mut record = f.record.clone();
    record.identity = plan.comparison.clone();
    record.identity_sha256 = plan.comparison.sha256().unwrap();
    let mut evidence = f.evidence.clone();
    evidence.freeze = freeze;
    // The whole-development receipt is not a receipt for the inner comparison.
    assert!(context::verify(&f.state, &policy, &record, &evidence).is_err());
    evidence.context = Some(produce_context(
        &f.state,
        scope,
        context_plan().encoder,
        vec![],
        &context_output(2048, 0.5),
    ));
    assert_eq!(
        context::verify(&f.state, &policy, &record, &evidence).unwrap(),
        evidence.context.unwrap().receipt_sha256
    );
}
