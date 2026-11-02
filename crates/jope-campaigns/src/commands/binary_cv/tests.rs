use super::*;

pub(crate) mod tests {
    use super::*;
    use feat_math::Pcg32;
    use std::path::PathBuf;

    pub(crate) fn registry_root() -> PathBuf {
        std::env::temp_dir().join("jope-dime-binary-registry")
    }

    /// Deterministic fixture registry for the manual CLI smoke: interleaved
    /// train/test splits, both classes guaranteed in every file.
    pub(crate) fn export_registry_to(root: &Path) {
        let _ = fs::remove_dir_all(root);
        let (train_a, test_a) = fixture_balanced();
        write_csv(
            &root.join("b1a0f1c500000001").join("train.csv"),
            &train_a.x,
            &train_a.y,
        );
        write_csv(
            &root.join("b1a0f1c500000001").join("test.csv"),
            &test_a.x,
            &test_a.y,
        );
        let full_b = fixture_imbalanced();
        let mut train_x = Vec::new();
        let mut train_y = Vec::new();
        let mut test_x = Vec::new();
        let mut test_y = Vec::new();
        // Class-stratified deterministic split: every 5th row PER CLASS goes
        // to test, so both files always hold >= 2 rows of each class.
        let mut class_counts = [0usize; 2];
        for (row, target) in full_b.x.iter().zip(&full_b.y) {
            let class = usize::from(*target == 1.0);
            let index = class_counts[class];
            class_counts[class] += 1;
            if index % 5 == 4 {
                test_x.push(row.clone());
                test_y.push(*target);
            } else {
                train_x.push(row.clone());
                train_y.push(*target);
            }
        }
        assert!(
            test_y.iter().filter(|v| **v == 1.0).count() >= 2,
            "smoke test split lacks positives"
        );
        assert!(train_y.iter().filter(|v| **v == 1.0).count() >= 2);
        write_csv(
            &root.join("b1a0f1c500000002").join("train.csv"),
            &train_x,
            &train_y,
        );
        write_csv(
            &root.join("b1a0f1c500000002").join("test.csv"),
            &test_x,
            &test_y,
        );
    }

    fn write_csv(path: &Path, x: &[Vec<f64>], y: &[f64]) {
        let mut text = String::new();
        for (row, target) in x.iter().zip(y) {
            for value in row {
                text.push_str(&format!("{value:?},"));
            }
            text.push_str(&format!("{target:?}\n"));
        }
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }

    fn regression_data(x: Vec<Vec<f64>>, y: Vec<f64>) -> RegressionData {
        let features = x[0].len();
        RegressionData { x, y, features }
    }

    /// Balanced nonlinear fixture: 240 train + 100 test rows, 6 features.
    fn fixture_balanced() -> (RegressionData, RegressionData) {
        let mut rng = Pcg32::new(97, 11);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..340 {
            let row: Vec<f64> = (0..6).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let score =
                (2.0 * row[0]).sin() + row[1] * row[2] - 0.3 * row[3] + 0.2 * rng.gaussian();
            x.push(row);
            y.push(if score > 0.0 { 1.0 } else { 0.0 });
        }
        let test_x = x.split_off(240);
        let test_y = y.split_off(240);
        (regression_data(x, y), regression_data(test_x, test_y))
    }

    /// Imbalanced (~9:1) nonlinear fixture: 200 rows, 5 features.
    fn fixture_imbalanced() -> RegressionData {
        let mut rng = Pcg32::new(131, 7);
        let mut x = Vec::new();
        let mut scores = Vec::new();
        for _ in 0..200 {
            let row: Vec<f64> = (0..5).map(|_| rng.next_f64() * 4.0 - 2.0).collect();
            let score = row[0] * row[1] + (3.0 * row[2]).sin() + 0.15 * rng.gaussian();
            x.push(row);
            scores.push(score);
        }
        let mut sorted = scores.clone();
        sorted.sort_by(f64::total_cmp);
        let threshold = sorted[180];
        let y = scores
            .iter()
            .map(|s| if *s > threshold { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        regression_data(x, y)
    }

    fn fixture_degenerate() -> RegressionData {
        let mut rng = Pcg32::new(41, 3);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..12 {
            x.push((0..3).map(|_| rng.next_f64() * 2.0 - 1.0).collect());
            y.push(if i < 3 { 1.0 } else { 0.0 });
        }
        regression_data(x, y)
    }

    fn temp_root(label: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("jope-dime-binary-{}-{label}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        root
    }

    fn seed_bytes() -> &'static [u8] {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/seeds/generation-zero-binary.txt"
        ))
    }

    #[test]
    fn binary_generation_zero_is_pinned_canonical_unique_and_valid() {
        assert_eq!(sha256_bytes(seed_bytes()), SEED_BINARY_SHA256);
        let population =
            parse_binary_population(seed_bytes(), POPULATION, &binary_task(), &binary_schema(6))
                .unwrap();
        assert_eq!(population.len(), POPULATION);
    }

    #[test]
    fn probability_validation_rejects_out_of_range() {
        assert!(validate_probabilities(&[0.0, 0.5, 1.0], 3).is_ok());
        assert!(validate_probabilities(&[0.5], 2).is_err());
        assert!(validate_probabilities(&[1.2, 0.5], 2).is_err());
        assert!(validate_probabilities(&[-0.1, 0.5], 2).is_err());
        assert!(validate_probabilities(&[f64::NAN, 0.5], 2).is_err());
    }

    #[test]
    fn degenerate_folds_are_marked_undefined_and_pool_finite() {
        let train = fixture_degenerate();
        let genome = GenomeIr::parse("(pipe X (logit_ridge))").unwrap();
        let evaluated = (0..CV7_FOLDS).collect::<Vec<_>>();
        let result =
            exact_cv_on_folds_binary(&genome, &train, CV7_FOLDS, &evaluated, "degenerate-fixture")
                .unwrap();
        assert_eq!(result.fold_defined.len(), CV7_FOLDS);
        assert!(
            result.fold_defined.iter().any(|defined| !defined),
            "a 12-row/3-positive CV7 must contain single-class validation folds"
        );
        assert!(result.pooled_oof_auc.is_finite());
        assert!(result.robust_fitness.is_finite());
        // Undefined folds are NaN in-memory and excluded from the penalty.
        for (auc_value, defined) in result.fold_auc.iter().zip(&result.fold_defined) {
            assert_eq!(auc_value.is_finite(), *defined);
        }
    }

    #[test]
    fn campaign_is_deterministic_and_winner_beats_linear_baseline() {
        let (train, test) = fixture_balanced();
        let seed = 0x6a70_6269_6e5f_7631_u64 ^ 20260616;
        let root_a = temp_root("determinism-a");
        let root_b = temp_root("determinism-b");
        let value_a = select_dataset_binary(
            &train,
            "b1a0f1c500000001",
            "fixture-train-sha",
            seed,
            seed_bytes(),
            &root_a,
            2,
        )
        .unwrap();
        let value_b = select_dataset_binary(
            &train,
            "b1a0f1c500000001",
            "fixture-train-sha",
            seed,
            seed_bytes(),
            &root_b,
            2,
        )
        .unwrap();
        let freeze_a = fs::read(root_a.join("selection-freeze.v2-binary.json")).unwrap();
        let freeze_b = fs::read(root_b.join("selection-freeze.v2-binary.json")).unwrap();
        assert_eq!(freeze_a, freeze_b, "binary freezes must be byte-identical");
        let ledger_a = fs::read(root_a.join("slots.v2-binary.jsonl")).unwrap();
        let ledger_b = fs::read(root_b.join("slots.v2-binary.jsonl")).unwrap();
        assert_eq!(ledger_a, ledger_b, "binary ledgers must be byte-identical");
        assert_eq!(value_a, value_b);

        // AUC sanity on held-out test data: the evolved winner must clearly
        // beat a plain linear-probability baseline on this nonlinear fixture.
        let score = score_dataset_binary(
            &root_a,
            "b1a0f1c500000001",
            &train,
            "fixture-train-sha",
            &test,
            2,
        )
        .unwrap();
        let task = binary_task();
        let schema = binary_schema(train.features);
        let baseline_genome = GenomeIr::parse("(pipe X (logit_ridge))").unwrap();
        let baseline_model = fit_table(
            &baseline_genome,
            &task,
            &schema,
            &raw_table(train.features, &train.x).unwrap(),
            &train.y,
        )
        .unwrap();
        let baseline_predictions = baseline_model
            .predict_table(&raw_table(test.features, &test.x).unwrap())
            .unwrap();
        let baseline_auc = auc(&test.y, &baseline_predictions);
        assert!(score.auc > 0.8, "winner test AUC too weak: {}", score.auc);
        assert!(
            score.auc > baseline_auc + 0.05,
            "winner {} must clearly beat the linear baseline {}",
            score.auc,
            baseline_auc
        );
        assert!(score.frozen_f1_threshold > 0.0 && score.frozen_f1_threshold < 1.0);
        // Thresholded report metrics are present and sane.
        assert!(score.f1_at_frozen_threshold >= 0.0 && score.f1_at_frozen_threshold <= 1.0);
        let _ = fs::remove_dir_all(&root_a);
        let _ = fs::remove_dir_all(&root_b);
    }

    #[test]
    fn imbalanced_campaign_and_scoring_smoke() {
        let train = fixture_imbalanced();
        let seed = 0x6a70_6269_6e5f_7631_u64 ^ 20260616 ^ 2;
        let root = temp_root("imbalanced");
        select_dataset_binary(
            &train,
            "b1a0f1c500000002",
            "fixture-train-sha-2",
            seed,
            seed_bytes(),
            &root,
            1,
        )
        .unwrap();
        let freeze = validate_freeze_binary(
            &root,
            "b1a0f1c500000002",
            "fixture-train-sha-2",
            train.features,
            1,
        )
        .unwrap();
        assert!(freeze.cv7_pooled_oof_auc.is_finite());
        // Winner bound artifacts were published content-addressed.
        let artifact_dir = root.join("artifacts").join(&freeze.semantic_sha256);
        assert!(artifact_dir.exists());
        let entries = fs::read_dir(&artifact_dir).unwrap().count();
        assert_eq!(entries, 3, "python + rust + manifest artifacts expected");
        let _ = fs::remove_dir_all(&root);
    }
}
