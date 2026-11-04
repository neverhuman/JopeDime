use super::*;

struct FixtureRuntime {
    evaluations: Vec<(usize, Value)>,
    emitted: Option<Value>,
}

impl SelectionRuntime for FixtureRuntime {
    fn evaluate_regression_with_ledger(
        &mut self,
        rows: &[Value],
        _ledger: &Path,
        workers: usize,
    ) -> Result<Vec<(usize, Value)>> {
        assert_eq!(rows.len(), 3);
        assert_eq!(workers, 2);
        Ok(std::mem::take(&mut self.evaluations))
    }

    fn emit(&mut self, dataset: &str, stage: &str, event: &str, payload: Value) -> Result<()> {
        assert_eq!((dataset, stage, event), ("ds", "robust_cv", "done"));
        self.emitted = Some(payload);
        Ok(())
    }
}

#[test]
fn robust_rerank_preserves_input_ties_and_skips_non_finite_oof() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let candidates = vec!["a".into(), "b".into(), "c".into()];
    let runtime = FixtureRuntime {
        // Deliberately return results out of order, as the ledger iterator may.
        evaluations: vec![
            (2, json!({"cv": {"oof_r2": "bad", "fold_r2": [0.2]}})),
            (
                1,
                json!({"cv": {"oof_r2": 0.8, "fold_r2": [0.7, 0.9]}, "timings": {"total_ms": 4}}),
            ),
            (
                0,
                json!({"cv": {"oof_r2": 0.8, "fold_r2": [0.7, 0.9]}, "timings": {"total_ms": 4}}),
            ),
        ],
        emitted: None,
    };
    let mut study = StudySelection {
        config: SelectionConfig {
            root: temp.path().into(),
            source_root: temp.path().into(),
            out_dir: temp.path().into(),
            jain_bin: "jain".into(),
            study_tag: "test".into(),
            robust_folds: 8,
            robust_workers: 2,
        },
        robust_path: temp.path().join("robust.jsonl"),
        runtime,
    };

    let ranked = study.robust_rerank("ds", &json!({"train_csv": "train.csv"}), &candidates)?;
    assert_eq!(ranked.len(), 2);
    assert_eq!(ranked[0]["genome"], "a");
    assert_eq!(ranked[1]["genome"], "b");
    assert_eq!(ranked[0]["robust_rank"], 1);
    assert_eq!(study.runtime.emitted.as_ref().unwrap()["ranked"], 2);
    assert_eq!(fs::read_to_string(study.robust_path)?.lines().count(), 3);
    Ok(())
}

#[test]
fn robust_score_and_lcb_use_population_standard_deviation() {
    assert!((robust_score(0.8, &[0.7, 0.9]) - 0.775).abs() < 1e-12);
    assert!((lcb95(&[0.7, 0.9]).unwrap() - (0.8 - 1.96 * 0.1_f64 / 2.0_f64.sqrt())).abs() < 1e-12);
    assert_eq!(lcb95(&[]), None);
}

#[test]
fn regression_metrics_match_the_study_fixture() -> Result<()> {
    let metrics = regression_metrics(&[1.0, 2.0, 3.0], &[1.0, 2.0, 4.0])?;
    assert_eq!(metrics["n"], 3);
    assert!((metrics["rmse"].as_f64().unwrap() - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    assert!((metrics["r2"].as_f64().unwrap() - 0.5).abs() < 1e-12);
    assert!((metrics["mae"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-12);
    Ok(())
}

#[test]
fn incumbent_keeps_first_sorted_model_on_equal_rmse() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let reports = temp.path().join("ds").join("reports");
    fs::create_dir_all(&reports)?;
    fs::write(
        reports.join("a.json"),
        r#"{"metrics":{"rmse":1.0,"r2":0.2}}"#,
    )?;
    fs::write(
        reports.join("b.json"),
        r#"{"metrics":{"rmse":1.0,"r2":0.3}}"#,
    )?;
    fs::write(
        reports.join("test__ds__old.json"),
        r#"{"metrics":{"rmse":0.1}}"#,
    )?;
    let study = StudySelection {
        config: SelectionConfig {
            root: temp.path().into(),
            source_root: temp.path().into(),
            out_dir: temp.path().into(),
            jain_bin: "jain".into(),
            study_tag: "test".into(),
            robust_folds: 8,
            robust_workers: 8,
        },
        robust_path: temp.path().join("unused.jsonl"),
        runtime: FixtureRuntime {
            evaluations: Vec::new(),
            emitted: None,
        },
    };
    let incumbent = study.incumbent("ds");
    assert_eq!(incumbent["model"], "a");
    assert_eq!(incumbent["rmse"], 1.0);
    Ok(())
}
