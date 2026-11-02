use super::*;
use super::*;
use crate::{
    emit::Language,
    run::{run, Policy, RunOptions},
    search::Budget,
};

fn dataset(slug: &str) -> String {
    format!("{}/../../datasets/test/{slug}", env!("CARGO_MANIFEST_DIR"))
}

fn a_run(root: &Path, name: &str) -> PathBuf {
    a_run_of(root, name, 6)
}

fn a_run_of(root: &Path, name: &str, evaluations: usize) -> PathBuf {
    let output = root.join(name);
    run(&RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(dataset("pmlb-229-pwlinear"), &output)
    })
    .unwrap();
    output.join("receipt.json")
}

fn options(receipt: &Path, registry: &Path) -> ScoreOptions {
    ScoreOptions {
        receipt: receipt.to_path_buf(),
        dataset: None,
        registry: registry.to_path_buf(),
        clock: Clock::Wall,
        objective: None,
    }
}

#[test]
fn a_test_score_reports_the_run_objective_or_a_requested_one() {
    let temp = tempfile::tempdir().unwrap();
    let registry = temp.path().join("registry");
    let objective = jope_runtime::fitness::Fitness::parse("pinball:quantile=0.9").unwrap();
    let ranked = temp.path().join("ranked");
    run(&RunOptions {
        budget: Budget {
            evaluations: 6,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        objective: Some(objective.clone()),
        ..RunOptions::new(dataset("pmlb-229-pwlinear"), &ranked)
    })
    .unwrap();
    let (score_ranked, _) = score(&options(&ranked.join("receipt.json"), &registry)).unwrap();
    let reported = score_ranked.objective.unwrap();
    assert_eq!(reported.name, "pinball");
    assert_eq!(reported.fitness_sha256, objective.identity().unwrap());
    assert!(reported.validation_loss.is_some());
    assert!(reported
        .test_loss
        .is_some_and(|l| l.is_finite() && l >= 0.0));
    assert!(reported.undefined.is_none());
    // An RMSE run's single look can also report the objective, with no development value.
    let plain = a_run_of(temp.path(), "plain", 8);
    let (score_plain, _) = score(&ScoreOptions {
        objective: Some(objective),
        ..options(&plain, &temp.path().join("another-registry"))
    })
    .unwrap_or_else(|e| panic!("{e:#}"));
    let reported = score_plain.objective.unwrap();
    assert!(reported.validation_loss.is_none());
    // An objective that is undefined on the test rows still spends the one look: the score
    // and its RMSE are recorded, and a second probe of the same genome is refused.
    let probe: jope_runtime::fitness::Fitness = serde_json::from_value(serde_json::json!({
            "metric": "custom", "name": "probe", "direction": "minimize",
            "expression": {"op": "ln", "arg": {"op": "sub", "left": {"op": "y"},
                "right": {"op": "constant", "value": 1e300}}},
            "aggregation": {"kind": "mean"}}))
    .unwrap();
    let probed = a_run_of(temp.path(), "probed", 9);
    let probe_registry = temp.path().join("probe-registry");
    let (scored, path) = score(&ScoreOptions {
        objective: Some(probe.clone()),
        ..options(&probed, &probe_registry)
    })
    .unwrap_or_else(|e| panic!("{e:#}"));
    let reported = scored.objective.as_ref().unwrap();
    assert!(reported.test_value.is_none() && reported.test_loss.is_none());
    assert!(reported.undefined.is_some());
    assert!(scored.test_rmse.is_finite());
    assert_eq!(read(&path).unwrap(), scored);
    assert!(score(&ScoreOptions {
        objective: Some(probe),
        ..options(&probed, &probe_registry)
    })
    .is_err());
}

#[test]
fn the_test_split_is_scored_once_per_dataset_and_genome() {
    let temp = tempfile::tempdir().unwrap();
    let receipt = a_run(temp.path(), "run");
    let before = std::fs::read(&receipt).unwrap();
    let registry = temp.path().join("registry");
    let (scored, path) = score(&options(&receipt, &registry)).unwrap();
    assert_eq!(scored.schema, TEST_SCORE_SCHEMA_ID);
    assert_eq!(scored.split, "test");
    assert!(!scored.feeds_selection);
    assert_eq!(scored.dataset.test_rows, 34);
    assert!(scored.test_rmse.is_finite() && scored.test_rmse >= 0.0);
    assert_eq!(scored.receipt_sha256, crate::sha256_hex(&before));
    assert_eq!(read(&path).unwrap(), scored);
    assert!(path.ends_with(format!("1cb97b1d7314d43d/{}.json", scored.program_sha256)));
    // The run's own receipt is untouched: the test score lives beside it, never in it.
    assert_eq!(std::fs::read(&receipt).unwrap(), before);

    // A repeat is refused, and so is any other run that selected the same genome.
    let again = format!("{:#}", score(&options(&receipt, &registry)).unwrap_err());
    assert!(again.contains("already scored"), "{again}");
    let twin = a_run(temp.path(), "twin");
    let error = format!("{:#}", score(&options(&twin, &registry)).unwrap_err());
    assert!(error.contains("already scored"), "{error}");
    assert_eq!(read(&path).unwrap(), scored);
}

/// The registry enforces once per (dataset, genome), not once per dataset: every other genome
/// of the same dataset can still be scored. Each score therefore names the looks at that
/// dataset's test split the registry already holds, so choosing among several test-scored
/// genomes is visible in the receipts rather than hidden.
#[test]
fn a_score_names_the_earlier_looks_at_its_datasets_test_split() {
    let temp = tempfile::tempdir().unwrap();
    let registry = temp.path().join("registry");
    let run_with = |name: &str, evaluations: usize| {
        let output = temp.path().join(name);
        run(&RunOptions {
            budget: Budget {
                evaluations,
                seconds: 600,
            },
            rng_seed: 7,
            languages: vec![Language::Rust],
            policy: Policy::Skip,
            ..RunOptions::new(dataset("pmlb-229-pwlinear"), &output)
        })
        .unwrap();
        output.join("receipt.json")
    };
    let bootstrap = run_with("bootstrap", 1);
    let searched = run_with("searched", 24);
    let (first, first_path) = score(&options(&bootstrap, &registry)).unwrap();
    let (second, second_path) = score(&options(&searched, &registry)).unwrap();
    assert_ne!(first.program_sha256, second.program_sha256);
    let looks = |path: &Path| -> serde_json::Value {
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).unwrap()).unwrap()
            ["earlier_looks"]
            .clone()
    };
    assert_eq!(looks(&first_path), serde_json::json!([]));
    assert_eq!(
        looks(&second_path),
        serde_json::json!([first.program_sha256]),
        "the second genome's test score hides the first look"
    );
    // Another dataset's looks are not this one's.
    let other = temp.path().join("other");
    run(&RunOptions {
        budget: Budget {
            evaluations: 1,
            seconds: 600,
        },
        languages: vec![Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(dataset("pmlb-695-chatfield-4"), &other)
    })
    .unwrap();
    let (_, other_path) = score(&options(&other.join("receipt.json"), &registry)).unwrap();
    assert_eq!(looks(&other_path), serde_json::json!([]));
}

type Scored = Result<(TestScore, PathBuf)>;

/// Scores `first` and `second` into their registries at once, deterministically interleaved:
/// `first` runs until it is admitted (it holds the dataset's lock and has read the earlier
/// looks), then `second` starts, and `first` goes on to fit and publish only once `second`
/// has found the lock held. Without the lock `second` would read the looks and publish while
/// `first` is still paused, however quick either fit is. Returns both results and the stages
/// `second` reported. Threads with their own lock files contend exactly as processes do:
/// `flock` locks belong to the open file, not to the process.
fn race(first: &ScoreOptions, second: &ScoreOptions) -> (Scored, Scored, Vec<Stage>) {
    use std::{sync::mpsc, time::Duration};
    let patience = Duration::from_secs(600);
    let (admitted_tx, admitted_rx) = mpsc::channel::<()>();
    let (stage_tx, stage_rx) = mpsc::channel::<Stage>();
    let seen = std::sync::Mutex::new(Vec::new());
    let seen = &seen;
    std::thread::scope(|scope| {
        let holder = scope.spawn(move || {
            score_observed(first, &|stage| {
                if stage == Stage::Admitted {
                    admitted_tx.send(()).unwrap();
                    // Every stage `second` reports is sent here, and its sender is dropped
                    // when it returns, so a `second` that never waits fails this at once.
                    match stage_rx.recv_timeout(patience) {
                        Ok(Stage::Waiting) => {}
                        other => {
                            panic!("the second scorer did not wait for the lock: {other:?}")
                        }
                    }
                }
            })
        });
        admitted_rx
            .recv_timeout(patience)
            .expect("the first scorer was not admitted");
        let waiter = scope.spawn(move || {
            let scored = score_observed(second, &|stage| {
                seen.lock().unwrap().push(stage);
                let _ = stage_tx.send(stage);
            });
            drop(stage_tx);
            scored
        });
        let first = holder.join().expect("the first scorer panicked");
        let second = waiter.join().expect("the second scorer panicked");
        (first, second, seen.lock().unwrap().clone())
    })
}

/// Review 5334625316 on PR #118: two genomes of one dataset scored into one registry at once
/// both recorded `earlier_looks: []`. The reviewer's pair: the slower fit starts first
/// (24 evaluations select `(pipe X (rank append=true) (gpr m=400 steps=60))`), the quicker
/// one (1 evaluation, `(pipe X (ridge))`) while it fits. The second now waits for the first
/// and names it.
#[test]
fn a_concurrent_score_waits_and_names_the_look_it_waited_for() {
    let temp = tempfile::tempdir().unwrap();
    let registry = temp.path().join("registry");
    let slow = a_run_of(temp.path(), "slow", 24);
    let quick = a_run_of(temp.path(), "quick", 1);
    let (first, second, stages) = race(&options(&slow, &registry), &options(&quick, &registry));
    let (first, first_path) = first.unwrap();
    let (second, second_path) = second.unwrap();
    assert_ne!(first.program_sha256, second.program_sha256);
    assert_eq!(stages, [Stage::Waiting, Stage::Admitted]);
    assert!(first.earlier_looks.is_empty(), "{:?}", first.earlier_looks);
    assert_eq!(
        second.earlier_looks,
        std::slice::from_ref(&first.program_sha256),
        "the second score hides the look it waited for"
    );
    assert_eq!(read(&first_path).unwrap(), first);
    assert_eq!(read(&second_path).unwrap(), second);
    // The lock file stays beside the scores and is not a look.
    let directory = first_path.parent().unwrap();
    assert_eq!(second_path.parent().unwrap(), directory);
    assert!(directory.join(LOCK_FILE).is_file());
    let mut both = vec![first.program_sha256.clone(), second.program_sha256.clone()];
    both.sort();
    assert_eq!(earlier_looks(directory).unwrap(), both);
}

/// The same genome scored twice at once (two runs that selected it): the second waits, then
/// is refused by the repeat check before it fits (it is never admitted), and the first
/// score is untouched.
#[test]
fn a_concurrent_repeat_is_refused_before_it_fits() {
    let temp = tempfile::tempdir().unwrap();
    let registry = temp.path().join("registry");
    let receipt = a_run(temp.path(), "run");
    let twin = a_run(temp.path(), "twin");
    let (first, second, stages) = race(&options(&receipt, &registry), &options(&twin, &registry));
    let (first, path) = first.unwrap();
    let error = format!("{:#}", second.unwrap_err());
    assert!(error.contains("already scored"), "{error}");
    assert_eq!(stages, [Stage::Waiting], "the repeat was admitted to a fit");
    assert!(first.earlier_looks.is_empty(), "{:?}", first.earlier_looks);
    assert_eq!(read(&path).unwrap(), first);
    assert_eq!(
        earlier_looks(path.parent().unwrap()).unwrap(),
        std::slice::from_ref(&first.program_sha256)
    );
}

#[test]
fn a_receipt_that_does_not_verify_or_another_dataset_scores_nothing() {
    let temp = tempfile::tempdir().unwrap();
    let receipt = a_run(temp.path(), "run");
    let registry = temp.path().join("registry");

    let mismatch = ScoreOptions {
        dataset: Some(dataset("pmlb-695-chatfield-4")),
        ..options(&receipt, &registry)
    };
    let error = format!("{:#}", score(&mismatch).unwrap_err());
    assert!(error.contains("the receipt measured"), "{error}");

    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    let rmse = value["search"]["best"]["validation_rmse"].as_f64().unwrap();
    value["search"]["best"]["validation_rmse"] = (rmse / 2.0).into();
    std::fs::write(&receipt, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    assert!(score(&options(&receipt, &registry)).is_err());
    assert!(!registry.exists(), "a refusal wrote to the registry");
}

#[test]
fn test_rows_are_the_sealed_split_parsed_like_the_training_rows() {
    let resolved = dataset::resolve(&dataset("pmlb-229-pwlinear")).unwrap();
    let train = dataset::training_rows(&resolved).unwrap();
    let test = dataset::test_rows(&resolved).unwrap();
    assert_eq!(train.len() as u64, resolved.row.split.train_rows);
    assert_eq!(test.len() as u64, resolved.row.split.test_rows);
    assert!(test
        .iter()
        .all(|row| row.len() == resolved.row.shape.columns as usize));
}
