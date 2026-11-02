use super::*;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn load_i01_json() -> Value {
    let root = repo_root();
    read_json(&root.join(FILE_MAP_I01_PATH)).expect("parse FILE_MAP.i01.json")
}

#[test]
fn reconcile_benchmark_and_product_profile_identities() {
    let root = repo_root();
    let summary = verify_selfplay_baseline(&root).expect("I01 baseline must verify");
    assert_eq!(summary.schema, "jope.unified-workplan.file-map/v2");
    assert_eq!(summary.task_id, "I01");
    assert_eq!(summary.workspace_crate_count, 26);
    assert_eq!(summary.neural_gated_test_count, 18);
    assert_eq!(summary.partially_neural_gated_test_count, 4);
    assert_eq!(summary.schema_file_count, 29);
    assert_eq!(summary.symbol_count, 4);
    assert_eq!(summary.held_gpu_route_count, 23);

    // Negative case 1a: conflating SearchProfile and SearchStrategy must fail.
    let mut conflated = load_i01_json();
    conflated
        .as_dict_mut()
        .unwrap()
        .get_mut("profile_and_model_reconciliation")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .get_mut("search_vocabulary_vs_strategy")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .insert("vocabulary_symbol", Value::from("SearchStrategy"));
    let err = verify_profile_identity_reconciliation(
        &root,
        conflated.get("profile_and_model_reconciliation").unwrap(),
    )
    .expect_err("conflating SearchProfile and SearchStrategy must be rejected");
    assert!(
        err.to_string().contains("SearchProfile"),
        "unexpected error: {err}"
    );

    // Negative case 1b: recording an unverified profile ID not in feat_math::kernel::profile::profiles() must fail.
    let mut bogus_profile = load_i01_json();
    bogus_profile
        .as_dict_mut()
        .unwrap()
        .get_mut("profile_and_model_reconciliation")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .get_mut("search_vocabulary_vs_strategy")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .get_mut("registered_profiles")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .insert("id", Value::from("legacy_v1"));
    let err = verify_profile_identity_reconciliation(
        &root,
        bogus_profile
            .get("profile_and_model_reconciliation")
            .unwrap(),
    )
    .expect_err("mismatching profile ID against profile.rs registry must fail");
    assert!(
        err.to_string().contains("legacy-wrappers-v1"),
        "unexpected error: {err}"
    );

    // Negative case 1c: conflating 401-token native PolicyLM with 282-token CLOVER vocabulary must fail.
    let mut wrong_vocab = load_i01_json();
    let lineages = wrong_vocab
        .as_dict_mut()
        .unwrap()
        .get_mut("profile_and_model_reconciliation")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .get_mut("model_lineages")
        .unwrap()
        .as_list_mut()
        .unwrap();
    lineages[0]
        .as_dict_mut()
        .unwrap()
        .insert("vocabulary_tokens", Value::from(282_i64));
    let err = verify_profile_identity_reconciliation(
        &root,
        wrong_vocab.get("profile_and_model_reconciliation").unwrap(),
    )
    .expect_err("conflating 401-token native PolicyLM and 282-token CLOVER vocab must fail");
    assert!(err.to_string().contains("401"), "unexpected error: {err}");

    // Negative case 1d: promoting unaccepted v2 warm-start as accepted serving must fail.
    let mut unaccepted_promoted = load_i01_json();
    let lineages = unaccepted_promoted
        .as_dict_mut()
        .unwrap()
        .get_mut("profile_and_model_reconciliation")
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .get_mut("model_lineages")
        .unwrap()
        .as_list_mut()
        .unwrap();
    lineages[2]
        .as_dict_mut()
        .unwrap()
        .insert("status", Value::from("CONFIRMED_ACCEPTED_SERVING_BASELINE"));
    let err = verify_profile_identity_reconciliation(
        &root,
        unaccepted_promoted
            .get("profile_and_model_reconciliation")
            .unwrap(),
    )
    .expect_err("promoting unaccepted warm-start checkpoint must fail");
    assert!(
        err.to_string()
            .contains("CONFIRMED_EXPERIMENTAL_NO_SUCCESSOR_PROTOCOL"),
        "unexpected error: {err}"
    );
}

#[test]
fn distinguish_current_ci_historical_runs_and_proposed_protocols() {
    let root = repo_root();
    let doc = load_i01_json();
    let findings = doc.get("source_findings").unwrap();
    let (confirmed, rejected, unknown) =
        verify_finding_classifications(&root, findings).expect("findings must verify");
    assert_eq!(confirmed, 6);
    assert_eq!(rejected, 2);
    assert_eq!(unknown, 2);

    // Negative case 2a: confirmed finding with missing/invalid 40-hex commit SHA must fail.
    let mut bad_sha = doc.clone();
    bad_sha
        .as_dict_mut()
        .unwrap()
        .get_mut("source_findings")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .insert("source_sha", Value::from("not-a-40-hex-sha"));
    let err = verify_finding_classifications(&root, bad_sha.get("source_findings").unwrap())
        .expect_err("confirmed finding without 40-hex SHA must fail");
    assert!(
        err.to_string().contains("40-hex source_sha"),
        "unexpected error: {err}"
    );

    // Negative case 2b: unknown proposed_protocol finding coerced to confirmed must fail.
    let mut coerced_unknown = doc.clone();
    let entries = coerced_unknown
        .as_dict_mut()
        .unwrap()
        .get_mut("source_findings")
        .unwrap()
        .as_list_mut()
        .unwrap();
    let unknown_entry = entries
        .iter_mut()
        .find(|e| e.get("classification").and_then(Value::as_str) == Some("unknown"))
        .expect("at least one unknown entry");
    unknown_entry
        .as_dict_mut()
        .unwrap()
        .insert("classification", Value::from("confirmed"));
    let err =
        verify_finding_classifications(&root, coerced_unknown.get("source_findings").unwrap())
            .expect_err("coercing unknown proposed_protocol finding to confirmed must fail");
    assert!(
        err.to_string()
            .contains("`proposed_protocol` finding cannot be classified as `confirmed`"),
        "unexpected error: {err}"
    );

    // Negative case 2c: unknown finding in current_ci category coerced to confirmed must also fail.
    let mut coerced_ci_unknown = doc.clone();
    let entries = coerced_ci_unknown
        .as_dict_mut()
        .unwrap()
        .get_mut("source_findings")
        .unwrap()
        .as_list_mut()
        .unwrap();
    let unknown_entry = entries
        .iter_mut()
        .find(|e| e.get("classification").and_then(Value::as_str) == Some("unknown"))
        .expect("at least one unknown entry")
        .as_dict_mut()
        .unwrap();
    unknown_entry.insert("category", Value::from("current_ci"));
    unknown_entry.insert("classification", Value::from("confirmed"));
    let err =
        verify_finding_classifications(&root, coerced_ci_unknown.get("source_findings").unwrap())
            .expect_err("coercing unknown finding in current_ci to confirmed must fail");
    assert!(
        err.to_string().contains("unknown"),
        "unexpected error: {err}"
    );

    // Negative case 2d: claiming live_qualification_active=true (boolean or string) must fail.
    let mut false_live_bool = doc.clone();
    false_live_bool
        .as_dict_mut()
        .unwrap()
        .get_mut("source_findings")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .insert("live_qualification_active", Value::Bool(true));
    let err =
        verify_finding_classifications(&root, false_live_bool.get("source_findings").unwrap())
            .expect_err("claiming boolean live_qualification_active=true must fail");
    assert!(
        err.to_string().contains("live_qualification_active"),
        "unexpected error: {err}"
    );

    let mut false_live_str = doc;
    false_live_str
        .as_dict_mut()
        .unwrap()
        .get_mut("source_findings")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .insert("live_qualification_active", Value::from("true"));
    let err = verify_finding_classifications(&root, false_live_str.get("source_findings").unwrap())
        .expect_err("claiming string live_qualification_active=\"true\" must fail");
    assert!(
        err.to_string().contains("live_qualification_active"),
        "unexpected error: {err}"
    );
}

#[test]
fn record_all_skipped_and_unavailable_runtime_combinations() {
    let root = repo_root();
    let doc = load_i01_json();
    let runtime = doc.get("skipped_and_unavailable_runtimes").unwrap();
    let count =
        verify_runtime_combinations(&root, runtime).expect("runtime combinations must verify");
    assert_eq!(count, 7);

    // Negative case 3a: marking a skipped/zero-test/GPU-gated runtime combination as passed/active must fail.
    let mut false_pass = doc.clone();
    false_pass
        .as_dict_mut()
        .unwrap()
        .get_mut("skipped_and_unavailable_runtimes")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .insert("status", Value::from("PASSED_DEFAULT_LANE"));
    let err = verify_runtime_combinations(
        &root,
        false_pass.get("skipped_and_unavailable_runtimes").unwrap(),
    )
    .expect_err("counting a 0-collected or skipped runtime combination as passed must fail");
    assert!(
        err.to_string().contains("unrecognized runtime status"),
        "unexpected error: {err}"
    );

    // Negative case 3b: under-reporting jope-parallel neural-gated test files must fail.
    let mut wrong_neural_count = doc.clone();
    wrong_neural_count
        .as_dict_mut()
        .unwrap()
        .get_mut("skipped_and_unavailable_runtimes")
        .unwrap()
        .as_list_mut()
        .unwrap()[0]
        .as_dict_mut()
        .unwrap()
        .get_mut("affected_targets")
        .unwrap()
        .as_list_mut()
        .unwrap()
        .pop();
    let err = verify_runtime_combinations(
        &root,
        wrong_neural_count
            .get("skipped_and_unavailable_runtimes")
            .unwrap(),
    )
    .expect_err("under-reporting neural-gated test files must fail");
    assert!(
        err.to_string().contains("neural-gated"),
        "unexpected error: {err}"
    );

    // Negative case 3c: omitting one of the 6 indirect require_qualified_gpu_experiment routes in RT03 must fail.
    let mut missing_gpu_route = doc;
    missing_gpu_route
        .as_dict_mut()
        .unwrap()
        .get_mut("skipped_and_unavailable_runtimes")
        .unwrap()
        .as_list_mut()
        .unwrap()[2]
        .as_dict_mut()
        .unwrap()
        .get_mut("affected_targets")
        .unwrap()
        .as_list_mut()
        .unwrap()
        .pop();
    let err = verify_runtime_combinations(
        &root,
        missing_gpu_route
            .get("skipped_and_unavailable_runtimes")
            .unwrap(),
    )
    .expect_err("omitting a held GPU route in RT03 must fail");
    assert!(
        err.to_string().contains("held GPU routes"),
        "unexpected error: {err}"
    );
}
