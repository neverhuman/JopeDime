use super::super::{context_output, context_plan, context_scope, produce_context};
use super::*;

pub(crate) fn missing_context_provenance_is_refused_at_real_teaching_ingress() {
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

pub(crate) fn wrong_scope_consumer_encoder_or_altered_context_is_refused_before_authority() {
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

pub(crate) fn freeze_without_preregistered_context_encoder_is_diagnostic_only() {
    let f = fixture_with(Task::Regression, 5, None);
    assert!(f.evidence.context.is_some());
    assert_refused(
        &f,
        &f.evidence,
        "MissingEncoderProvenance: freeze preregistered no teaching context encoder",
        "legacy freeze",
    );
    // The refusal names the quarantine and the only way back (owner decision D4).
    let error = format!(
        "{:#}",
        qualify(&f.state, &f.record, &f.evidence).unwrap_err()
    );
    assert!(error.contains(REQUALIFY), "{error}");
}

pub(crate) fn retained_context_is_revalidated_by_every_receiving_gate() {
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

pub(crate) fn legacy_qualification_without_provenance_stays_quarantined_and_is_not_upgraded() {
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

pub(crate) fn inner_partition_scope_forbids_outer_hold_rows() {
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
    // The inner comparison's teaching window opened with the inner freeze; its context is handed
    // off by the preregistered producer inside it.
    let window = ExecutionWindow {
        experiment_sha256: f.window.experiment_sha256.clone(),
        started_unix_ms: policy.created_at_unix * 1000,
        deadline_unix_ms: policy.created_at_unix * 1000 + 60_000,
    };
    let teaching = context::Teaching {
        experiment: &f.experiment,
        window: &window,
    };
    // The whole-development receipt is not a receipt for the inner comparison.
    assert!(context::verify(
        &f.state,
        &policy,
        &record,
        &evidence,
        &Default::default(),
        &teaching
    )
    .is_err());
    evidence.context = Some(handed_off(
        &f.state,
        produce_context(
            &f.state,
            scope,
            context_plan().encoder,
            vec![],
            &context_output(2048, 0.5),
        ),
        &plan.comparison,
        &f.experiment,
        &window,
    ));
    assert_eq!(
        context::verify(
            &f.state,
            &policy,
            &record,
            &evidence,
            &Default::default(),
            &teaching
        )
        .unwrap()
        .0,
        evidence.context.unwrap().receipt_sha256
    );
}

/// The rule every pre-provenance refusal and `RequalificationRefused` end with.
pub(super) const REQUALIFY: &str = "new, fully provenanced qualification";

/// Owner decision D4, 2026-09-26: a pre-provenance qualification stays quarantined, and
/// requalification is a refusal with a clear message, not a procedure. The only route back is a
/// new, fully provenanced qualification of a new record, which then competes against the
/// quarantined tip's bar (the adopted HWM-bar decision, D1).
pub(crate) fn requalification_is_refused_in_place_and_happens_only_through_a_new_qualification() {
    let f = fixture(Task::Regression, 4);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert_eq!(ingest(&f).accepted, 1);
    // The same teacher's bytes as persisted before provenance was mandatory.
    let mut legacy = qualified;
    legacy.context_receipt_sha256 = None;
    legacy.context_handoff_sha256 = None;
    legacy.diagnostics = None;
    let path = qualification_path(&f.state, &f.record);
    let bytes = canonical_json(&StoredQualification {
        evidence: with_context(&f, None),
        qualification: legacy,
    })
    .unwrap();
    fs::write(&path, &bytes).unwrap();

    // Every gate names the quarantine and the only way back.
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    let exclusion = &manifest.exclusions[0].1;
    assert!(
        exclusion.contains("MissingEncoderProvenance") && exclusion.contains(REQUALIFY),
        "{exclusion}"
    );
    let error = format!(
        "{:#}",
        revalidate_qualification(&f.state, &f.record).unwrap_err()
    );
    assert!(error.contains(REQUALIFY), "{error}");

    // Requalifying the same record in place is refused with a clear message, inventoried, and
    // neither authority nor the retained bytes change.
    let before = authority(&f.state);
    let inventoried = rejected_decisions(&f.state).len();
    let error = format!(
        "{:#}",
        qualify(&f.state, &f.record, &f.evidence).unwrap_err()
    );
    assert!(
        error.contains("RequalificationRefused")
            && error.contains(&f.record.provenance.record_sha256)
            && error.contains(REQUALIFY),
        "{error}"
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(before, authority(&f.state));
    let decisions = rejected_decisions(&f.state);
    assert_eq!(decisions.len(), inventoried + 1);
    assert!(decisions.last().unwrap().contains("RequalificationRefused"));

    // Positive control: a new, fully provenanced qualification of a new record is admitted and,
    // strictly beating the quarantined tip's bar, becomes the teacher. The quarantined bytes stay.
    let (fresh, evidence) = f.candidate("requalified", &exact_on(5));
    let requalified = qualify(&f.state, &fresh, &evidence).unwrap();
    let report = hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: fresh.clone(),
            blob: candidate_blob("requalified"),
            chain_from_source: false,
        }],
        json!({"kind":"w3-requalification"}),
    )
    .unwrap();
    assert_eq!((report.accepted, report.rejected), (1, 0));
    assert_eq!(
        freeze_teaching(&f.state).unwrap().entries,
        vec![requalified]
    );
    assert_eq!(fs::read(&path).unwrap(), bytes);
}
