//! Binary and multiclass through the whole benchmark, offline, on catalog-layout fixtures
//! written to a temporary directory: the repository commits no classification dataset yet, so
//! the fixture borrows `pmlb-229-pwlinear`'s manifest shape (provenance included, so it passes
//! the catalog schema) with the identity, task, shape and file entries replaced. It is a test
//! fixture, not a dataset: nothing here is a benchmark result.

use std::fs;
use std::path::{Path, PathBuf};

use jope_arena::leaderboard::Ladder;
use jope_bench::cuts::{build, FoldPlan};
use jope_bench::emit::{self, Stage};
use jope_bench::load::{load_dataset, sha256_hex};
use jope_bench::methods::{default_baselines, program, MethodKind};
use jope_bench::protocol::{self, Protocol};
use jope_bench::report::{Report, Stamp};
use jope_bench::task::Task;
use jope_bench::{DEFAULT_FOLDS, DEFAULT_SEED};
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/jope-bench sits two levels below the repository root")
        .to_path_buf()
}

/// Four features; the target follows three of them with a little noise.
fn synth(task: Task) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = feat_math::Pcg32::new(
        match task {
            Task::Binary => 101,
            Task::Multiclass => 202,
            Task::Regression => 303,
        },
        1,
    );
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..240 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        let c = rng.next_f64();
        let noise = rng.next_f64();
        x.push(vec![a, b, c, noise]);
        y.push(match task {
            Task::Binary => {
                let logit = 3.0 * (a - b) + 1.5 * (c - 0.5) + 0.8 * (noise - 0.5);
                if logit > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Task::Multiclass => {
                let scores = [a + 0.3 * (noise - 0.5), b, c];
                scores
                    .iter()
                    .enumerate()
                    .max_by(|l, r| l.1.total_cmp(r.1))
                    .map(|(i, _)| i as f64)
                    .unwrap()
            }
            Task::Regression => a - b + 0.1 * noise,
        });
    }
    (x, y)
}

fn csv(rows: &[Vec<f64>], targets: &[f64]) -> String {
    rows.iter()
        .zip(targets)
        .map(|(row, target)| {
            let mut cells: Vec<String> = row.iter().map(|v| format!("{v:?}")).collect();
            cells.push(format!("{target:?}"));
            cells.join(",") + "\n"
        })
        .collect()
}

/// A catalog-layout dataset directory for `task` under `dir`: `metadata.json`, `train.csv` (the
/// first 80% of the rows), `test.csv` (the rest; never read by the benchmark).
fn write_dataset(dir: &Path, task: Task, x: &[Vec<f64>], y: &[f64]) -> PathBuf {
    let features = x[0].len();
    let split = x.len() * 4 / 5;
    let train = csv(&x[..split], &y[..split]);
    let test = csv(&x[split..], &y[split..]);
    let train_sha = sha256_hex(train.as_bytes());
    let test_sha = sha256_hex(test.as_bytes());
    let canonical = sha256_hex(format!("fixture:{}:{train_sha}", task.as_str()).as_bytes());
    let hash = canonical[..16].to_owned();
    let mut meta: Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join("datasets/test/pmlb-229-pwlinear/metadata.json"))
            .unwrap(),
    )
    .unwrap();
    meta["id"] = json!(hash);
    meta["dataset_hash"] = json!(hash);
    meta["canonical_sha256"] = json!(canonical);
    meta["display_name"] = json!(format!("synthetic {} (test fixture)", task.as_str()));
    meta["task"] = json!(task.as_str());
    meta["shape"] = json!({
        "rows": x.len(),
        "features": features,
        "categorical_features": 0,
        "columns": features + 1,
    });
    meta["target"] = json!({"name": "target", "index": -1, "type": task.as_str()});
    meta["files"] = json!({
        "train": {"sha256": train_sha, "bytes": train.len(), "rows": split, "columns": features + 1, "has_header": false},
        "test": {"sha256": test_sha, "bytes": test.len(), "rows": x.len() - split, "columns": features + 1, "has_header": false},
    });
    meta["split"] = json!({
        "method": "first 80% of the generated rows (test fixture)",
        "seed": 0,
        "train_rows": split,
        "test_rows": x.len() - split,
    });
    meta["collections"] = json!(["bench-fixture"]);
    meta["aliases"] = json!([]);
    meta["ingest"]["relpath"] = json!(format!("fixture/{}", task.as_str()));
    let dataset_dir = dir.join(format!("fixture-{}", task.as_str()));
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::write(
        dataset_dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap() + "\n",
    )
    .unwrap();
    fs::write(dataset_dir.join("train.csv"), train).unwrap();
    fs::write(dataset_dir.join("test.csv"), test).unwrap();
    dataset_dir
}

#[test]
fn binary_and_multiclass_run_end_to_end_and_rank_by_maximize() {
    let temp = tempfile::tempdir().unwrap();
    for task in [Task::Binary, Task::Multiclass] {
        let (x, y) = synth(task);
        let dir = write_dataset(temp.path(), task, &x, &y);
        let dataset = load_dataset(&dir).unwrap();
        assert_eq!(dataset.bench_task().unwrap(), task);
        assert_eq!(dataset.x.len(), 192);

        let plan = FoldPlan::new(&dataset.row.dataset_hash, DEFAULT_SEED, DEFAULT_FOLDS);
        let cuts = build(
            &plan,
            &dataset.row.dataset_hash,
            &dataset.train_sha256,
            dataset.x.len(),
        )
        .unwrap();
        let mut methods = default_baselines(task).unwrap();
        // A regression head on a classification dataset is a rejection with the grammar's
        // reason, ranked last; not a crash.
        methods.push(program("wrong-task=(pipe X (ridge))").unwrap());
        let protocol = Protocol {
            screen_fold: 0,
            top: 3,
        };
        let results =
            protocol::run(&dataset.x, &dataset.y, &cuts, methods, &protocol, task).unwrap();
        let wrong = results
            .iter()
            .find(|r| r.method.id == "program.wrong-task")
            .unwrap();
        assert_eq!(wrong.screen.status, "rejected");
        assert!(wrong
            .screen
            .reason
            .as_deref()
            .unwrap_or("")
            .contains("incompatible"));
        assert_eq!(wrong.screen_rank, results.len());
        assert!(
            results
                .iter()
                .filter(|r| r.method.kind == MethodKind::Baseline)
                .all(|r| r.screen.is_ok()),
            "{results:#?}"
        );
        assert_eq!(results.iter().filter(|r| r.deep_rank.is_some()).count(), 3);
        let best = results.iter().find(|r| r.deep_rank == Some(1)).unwrap();
        let deep = best.deep.as_ref().unwrap();
        assert!(deep.oof_score.unwrap() > 0.7, "{deep:?}");
        assert_eq!(deep.oof_score, deep.oof.get(task.score_metric()).copied());
        assert_eq!(deep.fold_score.len(), DEFAULT_FOLDS);
        for metric in task.metrics() {
            assert!(
                deep.oof.contains_key(*metric),
                "{metric} missing from {deep:?}"
            );
        }

        let report = Report::new(
            &dataset,
            task,
            &plan,
            &cuts,
            &protocol,
            results,
            Stamp {
                workers: 2,
                git_sha: None,
                search: None,
            },
        );
        assert_eq!(report.task, task.as_str());
        assert_eq!(report.metric_direction, "maximize");
        let path = temp.path().join(report.file_name());
        report.write(&path).unwrap();
        let back = Report::read(&path).unwrap();
        assert_eq!(back, report);

        let loaded = vec![(path.clone(), back)];
        let track_root = temp.path().join(format!("track-{}", task.as_str()));
        for stage in [Stage::Screen, Stage::Deep] {
            let scores =
                emit::write_track(&track_root, "jope-same-cut-v1", stage, &loaded, temp.path())
                    .unwrap();
            let rows = jope_arena::scores::read_scores(&scores).unwrap();
            let (metric, _) = task.ladder_metric();
            assert!(rows.iter().all(|r| r.metric == metric));
            let ok_rows = rows.iter().filter(|r| r.status == "ok").count();
            match stage {
                Stage::Screen => assert_eq!(ok_rows, rows.len() - 1, "the wrong-task row failed"),
                Stage::Deep => assert_eq!(ok_rows, 3),
            }
            let track_id = format!("jope-same-cut-v1-{}", stage.suffix());
            let (board, _) = jope_arena::leaderboard::compute(
                &track_root,
                &track_id,
                None,
                Ladder::Published,
                None,
            )
            .unwrap();
            assert_eq!(board.datasets_scored, 1);
            assert_eq!(board.rankable_rows, ok_rows);
            // Maximize: the ladder's first row holds the largest value, and wins the dataset.
            let largest = rows
                .iter()
                .filter(|r| r.status == "ok")
                .map(|r| r.value_f64().unwrap())
                .fold(f64::NEG_INFINITY, f64::max);
            let top = rows
                .iter()
                .find(|r| r.method_id == board.rows[0].method_id)
                .unwrap();
            assert_eq!(top.value_f64(), Some(largest), "{:?}", board.rows);
            assert_eq!(board.rows[0].dataset_wins, 1);
            assert_eq!(board.rows[0].average_rank, "1.0");
        }
    }
}

#[test]
fn a_regression_fixture_still_loads_as_regression() {
    let temp = tempfile::tempdir().unwrap();
    let (x, y) = synth(Task::Regression);
    let dir = write_dataset(temp.path(), Task::Regression, &x, &y);
    // The catalog spells a numeric target "numeric", not "regression".
    let mut meta: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("metadata.json")).unwrap()).unwrap();
    meta["target"]["type"] = json!("numeric");
    fs::write(
        dir.join("metadata.json"),
        serde_json::to_string_pretty(&meta).unwrap() + "\n",
    )
    .unwrap();
    let dataset = load_dataset(&dir).unwrap();
    assert_eq!(dataset.bench_task().unwrap(), Task::Regression);
}
