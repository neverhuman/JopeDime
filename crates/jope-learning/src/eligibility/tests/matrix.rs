//! W3-T07: every jope-learning teaching ingress crossed with every submission mutation class.
//! `qualify` receives the mutated submission directly. The stored routes are then attacked
//! with a forged but internally consistent success claim: the mutated evidence stored with
//! the valid measurement, its `qualified` decision, an accepted HWM event and a pointer.
//! Every route must recompute and refuse without trusting those flags. Synthetic fixtures
//! only; this is not a live admission. Revoked parents (W12-T05) are covered by
//! `eligibility::tests::revocation`.
use super::metrics::{rebind, stored};
use super::provenance::{authority, rejected_decisions};
use super::*;

const ROUTES: [&str; 5] = ["qualify", "revalidate", "ingest", "freeze", "replay"];

struct Mutation {
    class: &'static str,
    /// Expected substring of the live `qualify` refusal.
    expected: &'static str,
    apply: fn(&Fixture) -> (HwmRecord, OofEvidence),
}

fn context(f: &Fixture, context: ContextEvidence) -> (HwmRecord, OofEvidence) {
    let mut evidence = f.evidence.clone();
    evidence.context = Some(context);
    (f.record.clone(), evidence)
}

fn folds(f: &Fixture, mutate: fn(&mut Vec<String>)) -> (HwmRecord, OofEvidence) {
    let mut evidence = f.evidence.clone();
    mutate(&mut evidence.fold_predictions_sha256);
    (f.record.clone(), evidence)
}

fn record(f: &Fixture, mutate: fn(&mut HwmRecord)) -> (HwmRecord, OofEvidence) {
    let mut record = f.record.clone();
    mutate(&mut record);
    record.identity_sha256 = record.identity.sha256().unwrap();
    (record, f.evidence.clone())
}

const MUTATIONS: [Mutation; 16] = [
    Mutation {
        class: "missing-context",
        expected: "MissingEncoderProvenance",
        apply: |f| {
            let mut evidence = f.evidence.clone();
            evidence.context = None;
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "wrong-consumer-context",
        expected: "EncoderScopeMismatch",
        apply: |f| {
            let mut other = f.plan.comparison.clone();
            other.dataset_hash = sha256_hex(b"another consumer dataset");
            let receipt = produce_context(
                &f.state,
                context_scope(&other, &f.rows),
                context_plan().encoder,
                vec![],
                &context_output(2048, 0.5),
            );
            context(f, receipt)
        },
    },
    Mutation {
        class: "altered-context",
        expected: "EncoderScopeMismatch",
        apply: |f| {
            let altered = ContextEvidence {
                output_sha256: f.state.put_blob(&context_output(2048, 0.75)).unwrap(),
                ..f.evidence.context.clone().unwrap()
            };
            context(f, altered)
        },
    },
    Mutation {
        class: "unresolved-parent",
        expected: "unresolved context parent receipt",
        apply: |f| {
            let receipt = produce_context(
                &f.state,
                context_scope(&f.plan.comparison, &f.rows),
                context_plan().encoder,
                vec![sha256_hex(b"withdrawn parent receipt")],
                &context_output(2048, 0.5),
            );
            context(f, receipt)
        },
    },
    Mutation {
        class: "missing-fold",
        expected: "complete five-fold",
        apply: |f| folds(f, |folds| drop(folds.pop())),
    },
    Mutation {
        class: "repeated-fold",
        expected: "prediction provenance differs",
        apply: |f| folds(f, |folds| folds[1] = folds[0].clone()),
    },
    Mutation {
        class: "wrong-fold-order",
        expected: "prediction provenance differs",
        apply: |f| folds(f, |folds| folds.swap(0, 1)),
    },
    Mutation {
        class: "duplicated-row",
        expected: "OOF row identity or order changed",
        apply: |f| {
            let evidence = rebind(f, &f.evidence, 0, |p, _, _| {
                p.row_ids[1] = p.row_ids[0].clone()
            });
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "nonfinite-prediction",
        expected: "invalid type: null",
        apply: |f| {
            let prediction: FoldPredictions = stored(f, &f.evidence.fold_predictions_sha256[0]);
            let result: EvaluationResult = stored(f, &prediction.execution_receipt_sha256);
            let worker: WorkerOutcome = stored(f, &result.worker_receipt_sha256);
            // JSON has no NaN: a nonfinite prediction reaches the receiver only as `null`.
            let nonfinite = br#"{"kind":"values","data":[null,1.0]}"#;
            fs::write(&worker.outputs[0].path, nonfinite).unwrap();
            let nulls = f.state.put_blob(nonfinite).unwrap();
            let evidence = rebind(f, &f.evidence, 0, |p, r, w| {
                p.predictions_sha256 = nulls.clone();
                r.predictions_sha256 = Some(nulls.clone());
                w.outputs[0].sha256 = nulls.clone();
            });
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "wrong-fitted-subject",
        expected: "absent from the bound worker outputs",
        apply: |f| {
            let other: FoldPredictions = stored(f, &f.evidence.fold_predictions_sha256[0]);
            let fitted = other.fitted_artifact_sha256;
            let evidence = rebind(f, &f.evidence, 1, |p, r, _| {
                p.fitted_artifact_sha256 = fitted.clone();
                r.fitted_state_sha256 = Some(fitted.clone());
            });
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "forged-fold-score",
        expected: "execution score differs from stored predictions",
        apply: |f| {
            let evidence = rebind(f, &f.evidence, 3, |_, r, _| {
                r.primary = r.primary.map(|value| value + 0.1)
            });
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "forged-aggregate-score",
        expected: "claimed HWM score differs",
        apply: |f| record(f, |r| r.score.value -= 1e-9),
    },
    Mutation {
        class: "relabelled-units",
        expected: "score units",
        apply: |f| record(f, |r| r.score.units = "normalized_rmse".into()),
    },
    Mutation {
        class: "wrong-comparison-identity",
        expected: "EncoderScopeMismatch",
        apply: |f| record(f, |r| r.identity.vocabulary_sha256 = "f".repeat(64)),
    },
    Mutation {
        class: "substituted-freeze",
        expected: "independent optimization freeze changed",
        apply: |f| {
            let mut evidence = f.evidence.clone();
            evidence.freeze.freeze_sha256 = sha256_hex(b"another optimization freeze");
            (f.record.clone(), evidence)
        },
    },
    Mutation {
        class: "wrong-program",
        expected: "prediction provenance differs",
        apply: |f| {
            record(f, |r| {
                r.genome.dsl = "(pipe X (another-program))".into();
                r.genome.program_sha256 = sha256_hex(r.genome.dsl.as_bytes());
            })
        },
    },
];

/// Written apart from `hwm::decide`, `hwm::verify` and `freeze_teaching`. Replays the
/// original HWM events in ledger order, ignoring every recorded outcome, transition,
/// displaced link, improvement counter and pointer, and keeps per comparison the earliest
/// record with the lowest mean fold loss among those whose qualification recomputes.
fn independent_replay(state: &State) -> BTreeMap<String, String> {
    let mut best: BTreeMap<String, (f64, String)> = BTreeMap::new();
    for event in Ledger::open(state.root(), "hwm").events().unwrap() {
        let record: HwmRecord = serde_json::from_value(event["record"].clone()).unwrap();
        let Ok(q) = revalidate_qualification(state, &record) else {
            continue;
        };
        let comparison = q.comparison.sha256().unwrap();
        if best
            .get(&comparison)
            .is_none_or(|(loss, _)| q.candidate_loss < *loss)
        {
            best.insert(comparison, (q.candidate_loss, q.record_sha256));
        }
    }
    best.into_iter()
        .map(|(k, (_, record))| (k, record))
        .collect()
}

fn manifest(state: &State) -> BTreeMap<String, String> {
    freeze_teaching(state)
        .unwrap()
        .entries
        .into_iter()
        .map(|q| (q.comparison.sha256().unwrap(), q.record_sha256))
        .collect()
}

/// A forged, internally consistent success claim: stored evidence plus its decision event.
fn forge_qualification(
    f: &Fixture,
    record: &HwmRecord,
    evidence: &OofEvidence,
    claim: &Qualification,
) -> String {
    let bytes = canonical_json(&StoredQualification {
        evidence: evidence.clone(),
        qualification: claim.clone(),
    })
    .unwrap();
    let path = qualification_path(&f.state, record);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, &bytes).unwrap();
    let sha = sha256_hex(&bytes);
    Ledger::open(f.state.root(), "eligibility")
        .append(
            &[json!({"schema":"jope.teaching-decision/v1",
                "comparison_sha256":record.identity_sha256,
                "record_sha256":record.provenance.record_sha256,
                "evidence_sha256":canonical_sha256(evidence).unwrap(),
                "guard":GuardPolicy::current(), "outcome":"qualified",
                "qualification_sha256":sha, "reason":null})],
            json!({"source":"w3-t07-forged-claim"}),
        )
        .unwrap();
    sha
}

/// A forged accepted HWM event bound to that claim, and the pointer it names.
fn forge_acceptance(f: &Fixture, record: &HwmRecord, qualification_sha256: &str) {
    let key = PointerKey::of(record);
    let event = hwm::HwmEvent {
        schema: hwm::EVENT_SCHEMA.into(),
        seq: 0,
        outcome: "accepted".into(),
        reason: None,
        pointer: key.clone(),
        record: record.clone(),
        displaced_record_sha256: None,
        ts_unix: now_unix(),
        transition: Some(hwm::Transition::Initialized),
        qualification_sha256: Some(qualification_sha256.into()),
    };
    Ledger::open(f.state.root(), "hwm")
        .append(
            &[serde_json::to_value(&event).unwrap()],
            json!({"source":"w3-t07-forged-claim"}),
        )
        .unwrap();
    f.state.write_pointer(&key, record, None).unwrap();
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
        json!({"kind":"w3-t07"}),
    )
    .unwrap()
}

/// One mutation class through all five routes; returns each route's refusal.
fn attack(m: &Mutation) -> Vec<(&'static str, String)> {
    let f = fixture(Task::Regression, 5);
    // The success a forger would claim: the valid submission's measurement.
    let claim = measured(&f.state, &f.record, &f.evidence, false).unwrap();
    let (record, evidence) = (m.apply)(&f);
    let mut refusals = Vec::new();

    let before = authority(&f.state);
    let inventoried = rejected_decisions(&f.state).len();
    let error = format!(
        "{:#}",
        qualify(&f.state, &record, &evidence).expect_err(m.class)
    );
    assert!(error.contains(m.expected), "{}: {error}", m.class);
    assert_eq!(before, authority(&f.state), "{} qualify", m.class);
    assert_eq!(rejected_decisions(&f.state).len(), inventoried + 1);
    refusals.push(("qualify", error));

    let sha = forge_qualification(&f, &record, &evidence, &claim);
    let error = format!(
        "{:#}",
        revalidate_qualification(&f.state, &record).expect_err(m.class)
    );
    refusals.push(("revalidate", error));

    let before = authority(&f.state);
    let report = deliver(&f, &record, "");
    assert_eq!(
        (report.accepted, report.rejects[0].reason.as_str()),
        (0, "teaching-ineligible"),
        "{} ingest",
        m.class
    );
    assert_eq!(before, authority(&f.state), "{} ingest", m.class);
    refusals.push(("ingest", report.rejects[0].reason.clone()));

    forge_acceptance(&f, &record, &sha);
    assert!(f
        .state
        .read_pointer(&PointerKey::of(&record))
        .unwrap()
        .is_some());
    let frozen = freeze_teaching(&f.state).unwrap();
    assert!(
        frozen.entries.is_empty() && frozen.excluded_diagnostic == 1,
        "{} freeze",
        m.class
    );
    refusals.push(("freeze", frozen.exclusions[0].1.clone()));

    let error = format!(
        "{:#}",
        revalidate_replay(&f.state, std::slice::from_ref(&claim)).expect_err(m.class)
    );
    refusals.push(("replay", error));

    // The forged flags stand in both ledgers and on disk; neither replay trusts them.
    assert!(independent_replay(&f.state).is_empty(), "{}", m.class);
    assert!(manifest(&f.state).is_empty());
    assert_eq!(
        refusals.iter().map(|(route, _)| *route).collect::<Vec<_>>(),
        ROUTES
    );
    for (route, reason) in &refusals {
        assert!(!reason.is_empty());
        eprintln!("W3-T07 {} {route}: refused: {reason}", m.class);
    }
    refusals
}

macro_rules! cells {
    ($($test:ident: $index:expr,)*) => {
        $(
            #[test]
            fn $test() {
                assert_eq!(attack(&MUTATIONS[$index]).len(), ROUTES.len());
            }
        )*
        #[test]
        fn every_mutation_class_is_run_through_every_route() {
            let mut covered = vec![$($index),*];
            covered.sort();
            assert_eq!(covered, (0..MUTATIONS.len()).collect::<Vec<_>>());
            let classes: BTreeSet<_> = MUTATIONS.iter().map(|m| m.class).collect();
            assert_eq!(classes.len(), MUTATIONS.len());
            eprintln!(
                "W3-T07 matrix: {} classes x {} routes = {} refusals",
                MUTATIONS.len(),
                ROUTES.len(),
                MUTATIONS.len() * ROUTES.len()
            );
        }
    };
}

cells! {
    missing_context: 0,
    wrong_consumer_context: 1,
    altered_context: 2,
    unresolved_parent: 3,
    missing_fold: 4,
    repeated_fold: 5,
    wrong_fold_order: 6,
    duplicated_row: 7,
    nonfinite_prediction: 8,
    wrong_fitted_subject: 9,
    forged_fold_score: 10,
    forged_aggregate_score: 11,
    relabelled_units: 12,
    wrong_comparison_identity: 13,
    substituted_freeze: 14,
    wrong_program: 15,
}

#[test]
fn authorized_submission_passes_every_route_for_every_task() {
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let f = fixture(task, 5);
        let q = qualify(&f.state, &f.record, &f.evidence).unwrap();
        assert_eq!(revalidate_qualification(&f.state, &f.record).unwrap(), q);
        let report = deliver(&f, &f.record, "");
        assert_eq!((report.accepted, report.rejected), (1, 0));
        assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![q.clone()]);
        assert_eq!(
            revalidate_replay(&f.state, std::slice::from_ref(&q))
                .unwrap()
                .entries,
            vec![q.clone()]
        );
        let expected = BTreeMap::from([(q.comparison.sha256().unwrap(), q.record_sha256)]);
        assert_eq!(independent_replay(&f.state), expected);
        assert_eq!(manifest(&f.state), expected);
        eprintln!(
            "W3-T07 authorized {}: all five routes admitted",
            task.as_str()
        );
    }
}

#[test]
fn independent_replay_reproduces_the_winner_in_either_arrival_order() {
    for strong_first in [false, true] {
        let f = fixture(Task::Regression, 4);
        let (strong, strong_evidence) = f.candidate("strong", &exact_on(5));
        let (unqualified, _) = f.candidate("unqualified", &exact_on(5));
        qualify(&f.state, &f.record, &f.evidence).unwrap();
        let stronger = qualify(&f.state, &strong, &strong_evidence).unwrap();
        let mut order = vec![(&f.record, ""), (&strong, "strong")];
        if strong_first {
            order.reverse();
        }
        order.insert(1, (&unqualified, "unqualified"));
        for (record, tag) in order {
            deliver(&f, record, tag);
        }
        let expected = BTreeMap::from([(
            stronger.comparison.sha256().unwrap(),
            stronger.record_sha256.clone(),
        )]);
        assert_eq!(independent_replay(&f.state), expected);
        assert_eq!(manifest(&f.state), expected);
        let verified = hwm::verify(&f.state, &[]).unwrap();
        assert!(verified.ok(), "{:?}", verified.breaks);
        assert_eq!(verified.initialized, 1);
    }
}
