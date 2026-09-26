//! W3-T04: a fitted artifact is bound to its fold's training input. A
//! `jope.panel-fitted-rust/v2` artifact names the exact fitting input it was fitted on
//! (`prepared_data::fitting_input_sha256` of its training table and targets, which the fit worker
//! computes from its own input). On the train-side OOF route the receiver recomputes that digest
//! from the fold's authenticated training partition. A v1 artifact names none: it is unbound, as
//! every artifact before v2, and its qualification keeps its bytes. Synthetic fixtures; nothing is
//! fitted.
use super::metrics::{rebind, stored};
use super::provenance::{assert_positive_control, assert_refused};
use super::*;
use jope_pipeline::prepared_data::{fitting_input_sha256, raw_table};

/// The fixture's authenticated prepared development data, as the receiver reads it.
fn development(f: &Fixture) -> DevelopmentData {
    let policy = load_freeze(&f.state, &f.evidence.freeze).unwrap();
    let bytes = f
        .state
        .read_blob(&policy.dataset.split.development_sha256)
        .unwrap()
        .unwrap();
    parse_development(&bytes, &policy.dataset).unwrap()
}

/// The fitting input a fit worker receives for `fold`, digested as the worker digests it.
fn fold_input(data: &DevelopmentData, fold: u8) -> String {
    data.cut(fold)
        .unwrap()
        .train
        .fitting_input_sha256()
        .unwrap()
}

/// Re-store fold `fold`'s fitted artifact after `edit`, keeping every digest link coherent: the
/// worker's output file and receipt, the result and the fold's predictions record.
fn refit(
    f: &Fixture,
    evidence: &OofEvidence,
    fold: usize,
    edit: impl FnOnce(&mut serde_json::Value),
) -> OofEvidence {
    let prediction: FoldPredictions = stored(f, &evidence.fold_predictions_sha256[fold]);
    let mut fitted: serde_json::Value = stored(f, &prediction.fitted_artifact_sha256);
    edit(&mut fitted);
    let bytes = canonical_json(&fitted).unwrap();
    let sha = f.state.put_blob(&bytes).unwrap();
    rebind(f, evidence, fold, |p, r, w| {
        let output = w
            .outputs
            .iter_mut()
            .find(|artifact| artifact.sha256 == p.fitted_artifact_sha256)
            .unwrap();
        fs::write(&output.path, &bytes).unwrap();
        output.sha256 = sha.clone();
        p.fitted_artifact_sha256 = sha.clone();
        r.fitted_state_sha256 = Some(sha.clone());
    })
}

/// Fold `fold`'s artifact as v2, naming `input`.
fn name(f: &Fixture, evidence: &OofEvidence, fold: usize, input: &str) -> OofEvidence {
    refit(f, evidence, fold, |fitted| {
        fitted["schema"] = "jope.panel-fitted-rust/v2".into();
        fitted["fitting_input_sha256"] = input.into();
    })
}

/// Every fold's artifact as v2, each naming its own fold's fitting input.
fn bound(f: &Fixture) -> OofEvidence {
    let data = development(f);
    (0..5).fold(f.evidence.clone(), |evidence, fold| {
        name(f, &evidence, fold, &fold_input(&data, fold as u8))
    })
}

/// Qualify, deliver and freeze `evidence` for `f.record`; returns the qualification.
fn admit(f: &Fixture, evidence: &OofEvidence) -> Qualification {
    let q = qualify(&f.state, &f.record, evidence).unwrap();
    let report = hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: candidate_blob(""),
            chain_from_source: false,
        }],
        json!({"kind":"w3-t04-fitted"}),
    )
    .unwrap();
    assert_eq!((report.accepted, report.rejected), (1, 0));
    assert_eq!(freeze_teaching(&f.state).unwrap().entries, vec![q.clone()]);
    q
}

#[test]
fn v2_fitted_artifacts_bind_every_fold_to_its_own_training_input() {
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let f = fixture(task, 5);
        let q = admit(&f, &bound(&f));
        assert_eq!(q.fitted_inputs_bound, Some(true), "{}", task.as_str());
        assert_eq!(
            revalidate_qualification(&f.state, &f.record).unwrap(),
            q,
            "{}",
            task.as_str()
        );
    }
}

/// One all-row fit reused for a fold: its input includes the fold's held-out rows.
fn all_rows(data: &DevelopmentData) -> String {
    fitting_input_sha256(
        &raw_table(
            &data.dataset.features,
            &data
                .rows
                .iter()
                .map(|row| row.values.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        &data
            .rows
            .iter()
            .map(|row| row.target.parse::<f64>().unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

#[test]
fn a_fitted_artifact_naming_another_input_is_refused() {
    // Each case rewrites worker outputs on disk, so each gets its own fixture.
    type Case = (&'static str, fn(&Fixture, &DevelopmentData) -> OofEvidence);
    let cases: [Case; 4] = [
        ("a fit for another fold's training rows", |f, data| {
            name(f, &bound(f), 1, &fold_input(data, 0))
        }),
        ("one all-row fit reused for a fold", |f, data| {
            name(f, &bound(f), 2, &all_rows(data))
        }),
        ("v2 naming no input", |f, _| {
            refit(f, &bound(f), 3, |fitted| {
                fitted
                    .as_object_mut()
                    .unwrap()
                    .remove("fitting_input_sha256");
            })
        }),
        ("v1 naming an input", |f, data| {
            let input = fold_input(data, 3);
            refit(f, &f.evidence, 3, move |fitted| {
                fitted["fitting_input_sha256"] = input.into();
            })
        }),
    ];
    for (label, case) in cases {
        let f = fixture(Task::Regression, 5);
        let data = development(&f);
        assert_refused(&f, &case(&f, &data), "FittedInputMismatch", label);
        // Positive control on the same state: its properly bound submission admits.
        assert_eq!(admit(&f, &bound(&f)).fitted_inputs_bound, Some(true));
    }
    // And an unbound (v1) submission admits as before.
    assert_positive_control(&fixture(Task::Regression, 5));
}

/// v1 keeps its meaning: unbound, admitted as before, and its qualification bytes carry no new
/// field, so qualifications stored before v2 revalidate unchanged. A mix of v1 and v2 folds is
/// unbound too.
#[test]
fn v1_artifacts_stay_unbound_and_their_qualifications_keep_their_bytes() {
    let f = fixture(Task::Regression, 5);
    let q = admit(&f, &f.evidence);
    assert_eq!(q.fitted_inputs_bound, None);
    assert!(!String::from_utf8_lossy(&canonical_json(&q).unwrap()).contains("fitted_inputs"));

    let f = fixture(Task::Binary, 5);
    let data = development(&f);
    let mixed = (0..4).fold(f.evidence.clone(), |evidence, fold| {
        name(&f, &evidence, fold, &fold_input(&data, fold as u8))
    });
    assert_eq!(admit(&f, &mixed).fitted_inputs_bound, None);
}
