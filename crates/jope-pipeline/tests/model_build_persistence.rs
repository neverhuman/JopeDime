//! Persistence of the winner ensemble and fresh-process reload parity (SC-DEC-184M C06).
//!
//! Authored for `cargo check --tests`. Execution is deferred to JA-06 / C14.
mod model_build_support;

use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_score::Predictions;
use jope_pipeline::model_build::artifact::{
    decode, encode, identity as persisted_identity, verify_reload, PersistedEnsemble,
};
use jope_pipeline::model_build::native::NativeFitter;
use jope_pipeline::model_build::{run_automl, AutomlOutput, EvaluationBudget};
use model_build_support::*;
use std::collections::BTreeMap;
use std::path::PathBuf;

const CHILD_DIR: &str = "JOPE_MODEL_BUILD_FRESH_PROCESS_DIR";
const CHILD_OUTPUT: &str = "child-predictions.txt";
const PERSISTED: &str = "persisted-ensemble.json";

/// Runs the regression AutoML request and returns its output with the fixtures it used.
fn regression_run() -> (Union, AutomlOutput) {
    let u = regression_union();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let roster = roster_of(Task::Regression, &["rust.ridge"]);
    let mut budget = EvaluationBudget::new(100);
    let out = run_automl(
        &plan(Task::Regression, &u),
        &roster,
        &fitter,
        &id,
        &report_ids(&u),
        &u.report_labels,
        &mut budget,
    )
    .unwrap();
    (u, out)
}

/// Restores each member from its persisted state bytes and predicts the report rows.
fn restore_predictions(
    fitter: &NativeFitter,
    persisted: &PersistedEnsemble,
) -> BTreeMap<String, Vec<f64>> {
    persisted
        .member_states
        .iter()
        .map(|(id, bytes)| {
            let model = NativeFitter::restore_state(bytes).unwrap();
            (
                id.clone(),
                fitter.predict_rows(&model, &report_rows()).unwrap(),
            )
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn restorer(fitter: &NativeFitter) -> impl Fn(&str, &[u8]) -> anyhow::Result<Predictions> + '_ {
    move |_id, bytes| {
        let model = NativeFitter::restore_state(bytes)?;
        Ok(Predictions::Values(
            fitter.predict_rows(&model, &report_rows())?,
        ))
    }
}

#[test]
fn the_canonical_record_round_trips_with_a_stable_identity() {
    let (_, out) = regression_run();
    let persisted = out.persisted.expect("the bound head is admitted");
    let bytes = encode(&persisted).unwrap();
    let decoded = decode(&bytes).unwrap();
    assert_eq!(decoded, persisted);
    assert_eq!(encode(&decoded).unwrap(), bytes);
    assert_eq!(
        persisted_identity(&decoded).unwrap(),
        persisted_identity(&persisted).unwrap()
    );
}

#[test]
fn non_canonical_or_unknown_field_bytes_are_refused() {
    let (_, out) = regression_run();
    let bytes = encode(&out.persisted.unwrap()).unwrap();
    let pretty =
        serde_json::to_vec_pretty(&serde_json::from_slice::<serde_json::Value>(&bytes).unwrap())
            .unwrap();
    assert!(
        decode(&pretty).is_err(),
        "pretty-printed bytes are not canonical"
    );
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("extra".into(), serde_json::json!(1));
    assert!(decode(&serde_json::to_vec(&value).unwrap()).is_err());
}

#[test]
fn reload_refuses_a_member_state_that_does_not_reproduce_its_report_predictions() {
    let (u, out) = regression_run();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let report = out.report.clone().unwrap();
    let mut persisted = out.persisted.clone().unwrap();
    // A state fitted on a different selection subset is a valid model, but not this member's.
    let spec = roster_of(Task::Regression, &["rust.ridge"]).remove(0);
    let (_, other_state) = fitter
        .fit_state(&spec, &(0..10).collect::<Vec<_>>())
        .unwrap();
    persisted
        .member_states
        .insert("rust.ridge".into(), other_state);
    let mut budget = EvaluationBudget::new(100);
    let result = verify_reload(
        &persisted,
        &id,
        &admitted_for(&persisted, &u),
        &u.selection_labels,
        &report,
        &mut budget,
        &restorer(&fitter),
    );
    assert!(result.is_err());
}

#[test]
fn reload_refuses_a_record_missing_a_member_state() {
    let (u, out) = regression_run();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let mut persisted = out.persisted.clone().unwrap();
    persisted.member_states.clear();
    let mut budget = EvaluationBudget::new(100);
    let result = verify_reload(
        &persisted,
        &id,
        &admitted_for(&persisted, &u),
        &u.selection_labels,
        &out.report.clone().unwrap(),
        &mut budget,
        &restorer(&fitter),
    );
    assert!(result.is_err());
}

#[test]
fn truncated_state_bytes_are_a_typed_refusal() {
    let (_, out) = regression_run();
    let persisted = out.persisted.clone().unwrap();
    let state = persisted.member_states.values().next().unwrap();
    assert!(NativeFitter::restore_state(&state[..state.len() / 2]).is_err());
    assert!(NativeFitter::restore_state(b"not a model").is_err());
}

#[test]
fn in_process_reload_reproduces_the_report_predictions_bit_for_bit() {
    let (u, out) = regression_run();
    let id = identity(&u);
    let fitter = split_fitter(Task::Regression, &u, &id);
    let report = out.report.clone().unwrap();
    let persisted = out.persisted.clone().unwrap();
    let mut budget = EvaluationBudget::new(100);
    verify_reload(
        &persisted,
        &id,
        &admitted_for(&persisted, &u),
        &u.selection_labels,
        &report,
        &mut budget,
        &restorer(&fitter),
    )
    .unwrap();
}

/// Rebuilds the admitted members from the output's own persisted ensemble. The bake-off bundles
/// are not kept in `AutomlOutput`, so the reload check re-admits them from the same request.
fn admitted_for(
    persisted: &PersistedEnsemble,
    u: &Union,
) -> Vec<jope_pipeline::model_build::oof::AdmittedBundle> {
    let plan = plan(Task::Regression, u);
    let fitter_roster = roster_of(Task::Regression, &["rust.ridge"]);
    let mut budget = EvaluationBudget::new(100);
    let id = identity(u);
    let fitter = split_fitter(Task::Regression, u, &id);
    let outcome =
        jope_pipeline::model_build::run_bakeoff(&plan, &fitter_roster, &fitter, &mut budget)
            .unwrap();
    let expected = jope_pipeline::model_build::automl::expected_plan(&plan);
    let admitted: Vec<_> = outcome
        .bundles
        .into_iter()
        .map(|b| {
            let records = jope_pipeline::model_build::automl::fold_records(&b);
            jope_pipeline::model_build::oof::AdmittedBundle::admit(b, &expected, &records).unwrap()
        })
        .collect();
    assert_eq!(
        admitted
            .iter()
            .map(|m| m.bundle().candidate_id.clone())
            .collect::<Vec<_>>(),
        persisted.member_states.keys().cloned().collect::<Vec<_>>()
    );
    admitted
}

/// The child process entry point. It runs only when the parent names it; it rebuilds the fixtures
/// from the same generators, restores every member from the persisted bytes alone, and writes
/// the report predictions it obtained.
#[test]
#[ignore = "child process entry point for fresh_process_reload_matches_in_process_predictions_bitwise"]
fn fresh_process_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR) else {
        panic!("run through fresh_process_reload_matches_in_process_predictions_bitwise");
    };
    let dir = PathBuf::from(dir);
    let persisted = decode(&std::fs::read(dir.join(PERSISTED)).unwrap()).unwrap();
    let u = regression_union();
    let id = identity(&u);
    assert_eq!(
        persisted.ensemble.identity, id,
        "the fresh process rebuilt a different identity"
    );
    let fitter = split_fitter(Task::Regression, &u, &id);
    let mut text = String::new();
    for (member, predictions) in restore_predictions(&fitter, &persisted) {
        let row: Vec<String> = bits(&predictions).iter().map(|b| b.to_string()).collect();
        text.push_str(&format!("{member} {}\n", row.join(" ")));
    }
    std::fs::write(dir.join(CHILD_OUTPUT), text).unwrap();
}

#[test]
fn fresh_process_reload_matches_in_process_predictions_bitwise() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("model-build-fresh-process");
    std::fs::create_dir_all(&dir).unwrap();
    let (_, out) = regression_run();
    let persisted = out.persisted.clone().unwrap();
    std::fs::write(dir.join(PERSISTED), encode(&persisted).unwrap()).unwrap();
    let _ = std::fs::remove_file(dir.join(CHILD_OUTPUT));

    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "fresh_process_child",
            "--test-threads=1",
            "--nocapture",
        ])
        .env(CHILD_DIR, &dir)
        .status()
        .unwrap();
    assert!(status.success(), "fresh process failed: {status}");

    let text = std::fs::read_to_string(dir.join(CHILD_OUTPUT)).unwrap();
    let report = out.report.unwrap();
    let mut seen = 0;
    for line in text.lines() {
        let mut parts = line.split(' ');
        let member = parts.next().unwrap();
        let child: Vec<u64> = parts.map(|p| p.parse().unwrap()).collect();
        let Predictions::Values(expected) = &report.per_member[member] else {
            panic!("regression predictions are values");
        };
        assert_eq!(
            child,
            bits(expected),
            "member {member} differs across processes"
        );
        seen += 1;
    }
    assert_eq!(seen, persisted.member_states.len());
    assert!(seen > 0);
}
