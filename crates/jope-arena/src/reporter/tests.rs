use super::*;

/// The reporter's `patterns_are_exact_or_one_trailing_prefix_wildcard`.
#[test]
fn patterns_are_exact_or_one_trailing_prefix_wildcard() {
    assert!(validate_pattern("jope-*").is_ok());
    assert!(validate_pattern("tabfm").is_ok());
    assert!(validate_pattern("*").is_err());
    assert!(validate_pattern("a*b").is_err());
    assert!(validate_pattern("").is_err());
    assert!(pattern_matches("jope-*", "jope-dime"));
    assert!(!pattern_matches("jope-*", "jopedime"));
    assert!(pattern_matches("tabfm", "tabfm"));
    assert!(!pattern_matches("tabfm", "tabfm_ensemble"));
    assert!(patterns_overlap("tabfm*", "tabfm_ensemble*"));
    assert!(!patterns_overlap("tabfm", "tabfm_ensemble*"));
}

/// The reporter's `hashes_and_urls_fail_closed`.
#[test]
fn hashes_and_urls_fail_closed() {
    assert!(valid_hash("0123456789abcdef"));
    assert!(!valid_hash("0123456789ABCDEF"));
    assert!(!valid_hash("0123456789abcde"));
    assert!(valid_https_csv_url(
        "https://data.example/regression/x/train.csv"
    ));
    assert!(!valid_https_csv_url("http://data.example/train.csv"));
    assert!(!valid_https_csv_url("https://data.example/train.csv?x=1"));
    assert!(valid_metric_value("rmse", 0.0));
    assert!(!valid_metric_value("roc_auc", 1.5));
    assert!(!valid_metric_value("rmse", f64::NAN));
}

#[test]
fn config_validation_rejects_overlaps_and_bad_weights() {
    let mut config = fixtures::test_config(Direction::Maximize);
    assert!(validate_config(&config).is_ok());
    config.families.push(FamilyConfig {
        id: "other".to_owned(),
        display_name: "Other".to_owned(),
        patterns: vec!["family-x*".to_owned()],
    });
    assert!(validate_config(&config)
        .unwrap_err()
        .to_string()
        .contains("overlapping patterns"));
    config.families.pop();
    config.tasks[0].weight = 0.5;
    assert!(validate_config(&config)
        .unwrap_err()
        .to_string()
        .contains("sum to 1.0"));
}

#[test]
fn results_keep_the_best_score_and_track_validity() {
    let dir = tempfile::tempdir().unwrap();
    fixtures::write_reporter_root(
        dir.path(),
        &[
            ("family-a", "0123456789abcdef", 0.7, false),
            ("family-a", "0123456789abcdef", 0.9, true),
            ("other", "0123456789abcdef", 0.8, false),
        ],
    );
    let config = load_config(dir.path()).unwrap();
    let datasets = load_datasets(dir.path(), &config).unwrap();
    assert_eq!(datasets["binary"].len(), 1);
    let loaded = load_results(dir.path(), &config, &datasets).unwrap();
    assert_eq!(loaded.result_jsons_total, 3);
    assert_eq!(
        loaded.catalog["binary"]["family-a"]["0123456789abcdef"],
        0.9
    );
    assert_eq!(loaded.catalog["binary"]["other"]["0123456789abcdef"], 0.8);
    assert_eq!(loaded.valid_catalog["binary"].len(), 1);
    let blacklists = load_analysis_blacklists(dir.path(), &config, &datasets).unwrap();
    assert!(blacklists["binary"].is_empty());
}
