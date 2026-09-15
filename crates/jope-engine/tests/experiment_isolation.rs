use jope_parallel::gp::verify_experiment_output;

#[test]
fn comparison_cannot_escape_experiment_or_follow_an_external_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let experiment = dir.path().join("experiment");
    std::fs::create_dir_all(&experiment).unwrap();
    std::fs::write(
        experiment.join("manifest.json"),
        r#"{"schema":"parallel.seed_experiment.v1"}"#,
    )
    .unwrap();
    assert!(verify_experiment_output(&experiment.join("new/output"), &experiment).is_ok());
    assert!(verify_experiment_output(&dir.path().join("champions"), &experiment).is_err());
    assert!(verify_experiment_output(&experiment.join("../champions"), &experiment).is_err());
    std::os::unix::fs::symlink(dir.path(), experiment.join("escape")).unwrap();
    assert!(verify_experiment_output(&experiment.join("escape/champions"), &experiment).is_err());
}
