use super::*;

fn member(name: &str, scores: Vec<f64>) -> Member {
    Member {
        canonical: name.to_string(),
        semantic: name.to_string(),
        complexity: 1,
        origin: "fixture".to_string(),
        scores,
        fitness_complete: false,
    }
}

fn strict_config() -> Config {
    Config {
        schema: SCHEMA_CONFIG_V2.to_string(),
        strategy: "fixture".to_string(),
        dataset_hash: "development-fixture".to_string(),
        master_seed: 101,
        dl_genes: vec![BASELINE_SEEDS[0].to_string()],
        dl_draw_bank: Vec::new(),
        global_prior: Vec::new(),
        neighbor_prior: Vec::new(),
        budget: Budget {
            population: 40,
            generations: 25,
            splits_per_gen: 0,
            elites: 3,
            cv_fitness: 2,
            dl_inject_per_gen: 0,
            halving: Vec::new(),
        },
        final_cv_folds: 7,
        final_top_k: 10,
        perturb_fitness: false,
        vocab: "expanded".to_string(),
        baseline_seeds: true,
        pooled_oof: true,
        winner_rule: "rcv_1se".to_string(),
        drop_failed_folds: false,
        se_band: 0.5,
    }
}

#[test]
fn v2_config_binds_strict_selection_semantics() {
    let config = strict_config();
    config.validate().unwrap();

    let mut incomplete = strict_config();
    incomplete.drop_failed_folds = true;
    assert!(incomplete.validate().is_err());

    let mut winner_curse = strict_config();
    winner_curse.winner_rule = "cv7".to_string();
    assert!(winner_curse.validate().is_err());
}

#[test]
fn v1_config_retains_replay_compatibility() {
    let mut config = strict_config();
    config.schema = SCHEMA_CONFIG_V1.to_string();
    config.budget.cv_fitness = 0;
    config.pooled_oof = false;
    config.drop_failed_folds = true;
    config.winner_rule = "cv7".to_string();
    config.perturb_fitness = true;
    config.validate().unwrap();
}

#[test]
fn strict_fold_plan_rejects_nine_row_cv7() {
    let y: Vec<f64> = (0..9).map(|value| value as f64).collect();
    let folds = kfold(&y, 7, 17);
    assert!(validate_strict_fold_plan(y.len(), &folds).is_err());
}

#[test]
fn strict_fold_plan_rejects_one_row_holds() {
    let y: Vec<f64> = (0..40).map(|value| value as f64).collect();
    let folds = kfold(&y, 39, 18);
    let error = validate_strict_fold_plan(y.len(), &folds).unwrap_err();
    assert!(error.contains("hold rows"), "{error}");
}

#[test]
fn strict_fold_plan_rejects_too_few_fit_rows() {
    let y: Vec<f64> = (0..60).map(|value| value as f64).collect();
    let folds = kfold(&y, 2, 19);
    let error = validate_strict_fold_plan(y.len(), &folds).unwrap_err();
    assert!(error.contains("fit rows"), "{error}");
}

#[test]
fn strict_fold_plan_accepts_complete_feasible_plan() {
    let y: Vec<f64> = (0..80).map(|value| value as f64).collect();
    let folds = kfold(&y, 2, 23);
    validate_strict_fold_plan(y.len(), &folds).unwrap();
}

#[test]
fn complete_pooled_score_restores_original_row_order() {
    let y = vec![0.0, 1.0, 2.0, 3.0];
    let folds = vec![vec![0, 2], vec![1, 3]];
    let first = vec![0.1, 2.1];
    let second = vec![0.9, 3.2];
    let views = vec![Some(first.as_slice()), Some(second.as_slice())];
    let actual = complete_pooled_oof_r2(&y, &folds, &views).unwrap();
    let expected = r2(&y, &[0.1, 0.9, 2.1, 3.2]).unwrap();
    assert!((actual - expected).abs() < 1e-12);
}

#[test]
fn complete_pooled_score_fails_closed_on_missing_fold() {
    let y = vec![0.0, 1.0, 2.0, 3.0];
    let folds = vec![vec![0, 2], vec![1, 3]];
    let first = vec![0.1, 2.1];
    let views = vec![Some(first.as_slice()), None];
    assert!(complete_pooled_oof_r2(&y, &folds, &views).is_none());
}

#[test]
fn complete_pooled_score_fails_closed_on_duplicate_row() {
    let y = vec![0.0, 1.0, 2.0, 3.0];
    let folds = vec![vec![0, 2], vec![0, 3]];
    let first = vec![0.1, 2.1];
    let second = vec![0.2, 3.2];
    let views = vec![Some(first.as_slice()), Some(second.as_slice())];
    assert!(complete_pooled_oof_r2(&y, &folds, &views).is_none());
}

#[test]
fn strict_generation_scores_replace_member_history() {
    let mut population = vec![
        member("old-elite", vec![0.1, 0.2, 0.3]),
        member("new-child", Vec::new()),
    ];
    let invalid = replace_generation_outcomes(&mut population, vec![Some(0.4), Some(0.5)]);
    assert_eq!(invalid, 0);
    assert_eq!(population[0].scores, vec![0.4]);
    assert_eq!(population[1].scores, vec![0.5]);
    assert!(population.iter().all(|member| member.fitness_complete));
}

#[test]
fn incomplete_fitness_cannot_win_a_clip_floor_tie() {
    let mut valid = member("valid", vec![-1.0]);
    valid.fitness_complete = true;
    let invalid = member("invalid-but-simpler", vec![-1.0]);
    let mut population = vec![invalid, valid];
    rank(&mut population);
    assert_eq!(population[0].semantic, "valid");
}

#[test]
fn strict_generation_aborts_on_one_missing_candidate_with_failure_telemetry() {
    let failures = EvalFailureCounts {
        timeout: 1,
        ..EvalFailureCounts::default()
    };
    let error = strict_generation_completion(21, 1, failures).unwrap_err();
    assert!(error.contains("generation 21"), "{error}");
    assert!(error.contains("timeout=1"), "{error}");
    assert!(error.contains("invalid_oof=0"), "{error}");
}

#[test]
fn strict_generation_accepts_complete_candidate_coverage() {
    strict_generation_completion(3, 0, EvalFailureCounts::default()).unwrap();
}

#[test]
fn forty_by_twenty_five_by_two_is_exactly_two_thousand_fits() {
    let schedule = vec![(40, 0); 25];
    assert_eq!(expected_search_evaluations(&schedule, 2), 2_000);
}
