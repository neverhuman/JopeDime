use super::*;

use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temp_dir(name: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "prime-jain-exact-cv-{name}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn toy_data(rows: usize) -> RegressionData {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for row in 0..rows {
        let a = row as f64 / rows as f64;
        let b = ((row * 17) % rows) as f64 / rows as f64;
        x.push(vec![a, b, a * b]);
        y.push(2.0 * a - 0.7 * b + 0.1 * a * b);
    }
    RegressionData { x, y, features: 3 }
}

fn fixture_population(size: usize) -> Vec<PopulationMember> {
    (0..size)
        .map(|index| {
            let canonical = if index == 0 {
                "(pipe X (ridge))".to_string()
            } else {
                format!(
                    "(pipe X (winsor_clip q={}) (ridge))",
                    0.01 + index as f64 * 0.001
                )
            };
            let canonical = GenomeIr::parse(&canonical).unwrap().canonical();
            PopulationMember {
                semantic_sha256: sha256_bytes(canonical.as_bytes()),
                canonical_genome: canonical,
                ancestry: Ancestry {
                    operation: "seed".to_string(),
                    parents: Vec::new(),
                    retry_count: 0,
                },
            }
        })
        .collect()
}

fn fixture_config(size: usize) -> CampaignConfig {
    CampaignConfig {
        dataset: "fixture".to_string(),
        train_sha256: "0".repeat(64),
        generations: 2,
        population: size,
        cv_folds: 5,
        confirm_folds: 7,
        elites: 2,
        parent_pool: 4,
        tournament: 3,
        final_cv7_candidates: 4,
        max_transforms: 3,
        seed: 42,
        rayon_threads: size,
        expected_train_rows: None,
    }
}

#[test]
fn robust_fitness_clips_and_penalizes_population_sd() {
    assert!((robust_fitness(-2.0, &[0.0, 1.0]) + 1.125).abs() < 1e-12);
    assert_eq!(robust_fitness(0.5, &[0.5; 5]), 0.5);
    assert_eq!(robust_fitness(0.5, &[f64::NAN]), 0.5);
}

#[test]
fn production_generation_zero_is_pinned_canonical_and_unique() {
    let bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/seeds/generation-zero.txt"
    ));
    assert_eq!(
        sha256_bytes(bytes),
        "5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105"
    );
    let population = parse_initial_population(bytes, POPULATION).unwrap();
    validate_unique_population(&population, POPULATION).unwrap();
}

#[test]
fn fold_manifests_are_stable_and_cover_each_row_once() {
    for folds in [5, 7] {
        let left = build_fold_manifest("fixture", &"0".repeat(64), 103, folds).unwrap();
        let right = build_fold_manifest("fixture", &"0".repeat(64), 103, folds).unwrap();
        assert_eq!(
            canonical_json(&left).unwrap(),
            canonical_json(&right).unwrap()
        );
        let mut validation = left
            .assignments
            .iter()
            .flat_map(|fold| fold.validation_rows.iter().copied())
            .collect::<Vec<_>>();
        validation.sort_unstable();
        assert_eq!(validation, (0..103).collect::<Vec<_>>());
    }
}

#[test]
fn stable_order_uses_fitness_complexity_then_semantic_hash() {
    let row = |fitness, complexity, semantic: &str| SlotLedger {
        schema: LEDGER_SCHEMA.to_string(),
        generation: 0,
        slot: 0,
        canonical_genome: "(pipe X (ridge))".to_string(),
        semantic_sha256: semantic.to_string(),
        ancestry: Ancestry {
            operation: "seed".into(),
            parents: vec![],
            retry_count: 0,
        },
        head_family: "ridge".into(),
        complexity,
        valid: true,
        rejection_reason: None,
        cache_disposition: "miss".into(),
        cache_source_generation: None,
        cache_source_slot: None,
        artifacts: None,
        fold_r2: Some(vec![fitness; 5]),
        pooled_oof_r2: Some(fitness),
        pooled_oof_rmse: Some(1.0 - fitness),
        robust_fitness: Some(fitness),
        fold_fit_seconds: vec![],
        fold_predict_seconds: vec![],
        cumulative_unique_evaluations: 0,
        cumulative_terminal_slots: 0,
        cumulative_throughput_slots_per_second: 0.0,
    };
    let mut values = vec![
        row(0.8, 3, "b"),
        row(0.9, 9, "z"),
        row(0.8, 2, "z"),
        row(0.8, 2, "a"),
    ];
    values.sort_by(rank_slots);
    assert_eq!(
        values
            .iter()
            .map(|v| v.semantic_sha256.as_str())
            .collect::<Vec<_>>(),
        vec!["z", "a", "z", "b"]
    );
}

#[test]
fn refill_is_unique_replayable_and_records_ancestry() {
    let data = toy_data(80);
    let population = fixture_population(8);
    let root = temp_dir("refill");
    let config = fixture_config(8);
    let live = LiveLogger::create(&root, 8, 8, 8).unwrap();
    let evaluations = evaluate_population(
        0,
        &population,
        &data,
        &config,
        &root,
        &HashMap::new(),
        &live,
    )
    .unwrap();
    let mut ranked = evaluations
        .into_iter()
        .enumerate()
        .map(|(slot, item)| {
            let result = item.cached.result.unwrap();
            SlotLedger {
                schema: LEDGER_SCHEMA.into(),
                generation: 0,
                slot,
                canonical_genome: population[slot].canonical_genome.clone(),
                semantic_sha256: population[slot].semantic_sha256.clone(),
                ancestry: population[slot].ancestry.clone(),
                head_family: item.cached.head_family,
                complexity: item.cached.complexity,
                valid: true,
                rejection_reason: None,
                cache_disposition: "miss".into(),
                cache_source_generation: None,
                cache_source_slot: None,
                artifacts: item.cached.artifacts,
                fold_r2: Some(result.fold_r2),
                pooled_oof_r2: Some(result.pooled_oof_r2),
                pooled_oof_rmse: Some(result.pooled_oof_rmse),
                robust_fitness: Some(result.robust_fitness),
                fold_fit_seconds: result.fit_seconds,
                fold_predict_seconds: result.predict_seconds,
                cumulative_unique_evaluations: 0,
                cumulative_terminal_slots: 0,
                cumulative_throughput_slots_per_second: 0.0,
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(rank_slots);
    let left = refill_population(&config, 1, &ranked).unwrap();
    let right = refill_population(&config, 1, &ranked).unwrap();
    assert_eq!(
        canonical_json(&left).unwrap(),
        canonical_json(&right).unwrap()
    );
    validate_unique_population(&left, 8).unwrap();
    assert_eq!(
        left.iter()
            .filter(|m| m.ancestry.operation == "elite")
            .count(),
        2
    );
    assert!(left.iter().skip(2).all(|m| !m.ancestry.parents.is_empty()));
    let _ = fs::remove_dir_all(root);
}

#[test]
fn serial_and_parallel_cv_scores_and_order_agree() {
    let data = toy_data(96);
    let genomes = fixture_population(8);
    let serial = genomes
        .iter()
        .map(|member| {
            exact_cv(
                &GenomeIr::parse(&member.canonical_genome).unwrap(),
                &data,
                5,
                "fixture",
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let parallel = pool.install(|| {
        genomes
            .par_iter()
            .map(|member| {
                exact_cv(
                    &GenomeIr::parse(&member.canonical_genome).unwrap(),
                    &data,
                    5,
                    "fixture",
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    });
    for (left, right) in serial.iter().zip(&parallel) {
        assert!((left.robust_fitness - right.robust_fitness).abs() <= 1e-12);
        assert!((left.pooled_oof_r2 - right.pooled_oof_r2).abs() <= 1e-12);
        assert!(left
            .fold_r2
            .iter()
            .zip(&right.fold_r2)
            .all(|(a, b)| (a - b).abs() <= 1e-12));
    }
    let mut serial_order = (0..serial.len()).collect::<Vec<_>>();
    serial_order.sort_by(|&a, &b| {
        parallel[b]
            .robust_fitness
            .total_cmp(&parallel[a].robust_fitness)
            .then_with(|| genomes[a].semantic_sha256.cmp(&genomes[b].semantic_sha256))
    });
    let mut parallel_order = (0..parallel.len()).collect::<Vec<_>>();
    parallel_order.sort_by(|&a, &b| {
        parallel[b]
            .robust_fitness
            .total_cmp(&parallel[a].robust_fitness)
            .then_with(|| genomes[a].semantic_sha256.cmp(&genomes[b].semantic_sha256))
    });
    assert_eq!(serial_order, parallel_order);
}

#[test]
fn two_generation_fixture_has_exact_accounting_cache_and_artifacts() {
    let root = temp_dir("campaign");
    let data = toy_data(96);
    let population = fixture_population(8);
    let config = fixture_config(8);
    let population_bytes = population
        .iter()
        .map(|m| format!("{}\n", m.canonical_genome))
        .collect::<String>();
    let population_sha = sha256_bytes(population_bytes.as_bytes());
    let selection = execute_campaign(&config, &data, population, &population_sha, &root).unwrap();
    assert_eq!(selection.generation, 1);
    assert_eq!(selection.cv7_finalists.len(), 4);
    assert_eq!(selection.quick_cv5_fold0.evaluated_fold_ids, vec![0]);
    assert_eq!(selection.quick_cv5_fold0.fold_r2.len(), 1);
    assert_eq!(selection.cv7.fold_r2.len(), 7);
    let ledger = fs::read_to_string(root.join("slots.v1.jsonl")).unwrap();
    let slots = ledger
        .lines()
        .map(|line| serde_json::from_str::<SlotLedger>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(slots.len(), 16);
    assert_eq!(slots.iter().filter(|slot| slot.generation == 0).count(), 8);
    assert_eq!(slots.iter().filter(|slot| slot.generation == 1).count(), 8);
    assert!(slots
        .iter()
        .all(|slot| slot.cache_disposition == "hit" || slot.cache_disposition == "miss"));
    assert!(slots.iter().filter(|slot| slot.valid).all(|slot| slot
        .fold_r2
        .as_ref()
        .unwrap()
        .len()
        == 1));
    assert!(slots.iter().filter(|slot| slot.valid).all(|slot| {
        let artifacts = slot.artifacts.as_ref().unwrap();
        let dir = root.join("artifacts").join(&slot.semantic_sha256);
        dir.join(format!("{}.py", artifacts.python_sha256))
            .is_file()
            && dir.join(format!("{}.rs", artifacts.rust_sha256)).is_file()
            && dir
                .join(format!("{}.manifest.json", artifacts.manifest_sha256))
                .is_file()
    }));
    for generation in 0..2 {
        assert!(root
            .join(format!("checkpoints/generation-{generation:03}.v1.json"))
            .is_file());
        assert!(root
            .join(format!("summaries/generation-{generation:03}.v1.json"))
            .is_file());
    }
    let (_, recovered_sha) = validate_selection_freeze(&root).unwrap();
    assert_eq!(
        recovered_sha,
        sha256_file(&root.join("selection-freeze.v1.json")).unwrap()
    );
    assert!(!root.join("test.csv").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn invalid_selection_fails_without_touching_test_path() {
    let root = temp_dir("invalid-selection");
    let error = validate_selection_freeze(&root).unwrap_err().to_string();
    assert!(error.contains("selection marker"));
    assert!(!root.join("test.csv").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn predictions_round_trip_and_metrics_are_independent() {
    let root = temp_dir("predictions");
    let path = root.join("predictions.csv");
    let predictions = vec![1.0, 2.0, 3.0];
    write_predictions_atomic(&path, &predictions).unwrap();
    let reopened = read_predictions(&path).unwrap();
    assert_eq!(reopened, predictions);
    let (rmse, r2, mae) = independent_metrics(&[1.0, 1.0, 4.0], &reopened).unwrap();
    assert!((rmse - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
    assert!((r2 - 2.0 / 3.0).abs() < 1e-12);
    assert!((mae - 2.0 / 3.0).abs() < 1e-12);
    let _ = fs::remove_dir_all(root);
}
