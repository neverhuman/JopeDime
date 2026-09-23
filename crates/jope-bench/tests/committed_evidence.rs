//! The committed same-cut evidence (`assets/same-cut/v*`, see each `SOURCES.md`) re-derives
//! from itself: the tracks `jope-bench ladder` wrote come back byte for byte from the committed
//! reports, both ladders come back byte for byte from the committed tracks, and the committed
//! baselines of one dataset re-evaluate bit for bit on the same cuts. Anything that moves a
//! committed number without regenerating the evidence fails here. v1 is the heads plus the
//! searched program; v2 adds the four external engines.

use std::path::{Path, PathBuf};

use jope_arena::leaderboard::{self, Ladder, Snapshot};
use jope_bench::cuts::{build, FoldPlan};
use jope_bench::emit::{self, Stage};
use jope_bench::eval::{evaluate, evaluate_deep_programs, fixed_programs};
use jope_bench::load::load_dataset;
use jope_bench::methods::{MethodKind, SearchConfig, RESAMPLED_SEARCH_METHOD_ID};
use jope_bench::protocol::Protocol;
use jope_bench::report::{Report, SCHEMA};
use jope_bench::task::Task;
use jope_bench::{DEFAULT_FOLDS, DEFAULT_SEED, EVALUATOR};

const TRACK: &str = "jope-same-cut-v1";
const DATASETS: usize = 10;

/// One committed evidence directory and what it must contain.
struct Version {
    dir: &'static str,
    /// The engines it was recorded with, by adapter name; empty for v1.
    adapters: &'static [&'static str],
}

const VERSIONS: [Version; 2] = [
    Version {
        dir: "assets/same-cut/v1",
        adapters: &[],
    },
    Version {
        dir: "assets/same-cut/v2",
        adapters: &["catboost", "lightgbm", "rf", "xgboost"],
    },
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/jope-bench sits two levels below the repository root")
        .to_path_buf()
}

fn committed_reports(evidence: &Path) -> Vec<(PathBuf, Report)> {
    let dir = evidence.join("reports");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".bench-report.json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let report = Report::read(&p).unwrap();
            (p, report)
        })
        .collect()
}

#[test]
fn the_reports_are_one_consistent_run() {
    for version in &VERSIONS {
        let reports = committed_reports(&repo_root().join(version.dir));
        assert_eq!(reports.len(), DATASETS, "{}", version.dir);
        let first = &reports[0].1;
        for (path, r) in &reports {
            let what = format!(
                "{}/{}",
                version.dir,
                path.file_name().unwrap().to_string_lossy()
            );
            assert_eq!(r.schema, SCHEMA, "{what}");
            assert_eq!(r.evaluator, EVALUATOR, "{what}");
            assert_eq!(r.task, Task::Regression.as_str(), "{what}");
            assert_eq!(
                (r.score_metric.as_str(), r.ladder_metric.as_str()),
                ("r2", "rmse")
            );
            assert_eq!(r.metric_direction, "minimize");
            assert_eq!(
                r.plan,
                FoldPlan::new(&r.dataset_hash, DEFAULT_SEED, DEFAULT_FOLDS)
            );
            assert_eq!(
                r.protocol,
                Protocol {
                    screen_fold: 0,
                    top: 5
                }
            );
            assert_eq!(
                r.search,
                Some(SearchConfig {
                    pop: 32,
                    gens: 8,
                    seed: 0x1EA7_BEEF,
                    seconds: 0.0
                })
            );
            assert_eq!(r.git_sha, first.git_sha, "{what}: one run, one tree");
            assert_eq!(r.host, first.host, "{what}: one run, one host");
            assert_eq!(r.file_name(), path.file_name().unwrap().to_string_lossy());
            assert_eq!(r.cuts.len(), DEFAULT_FOLDS);
            let searched: Vec<_> = r
                .methods
                .iter()
                .filter(|m| m.method.kind == MethodKind::Search)
                .collect();
            assert_eq!(searched.len(), 1, "{what}");
            assert_eq!(searched[0].method.id, RESAMPLED_SEARCH_METHOD_ID);
            assert_eq!(searched[0].method.resample, r.search);
            let mut adapters: Vec<String> = r
                .methods
                .iter()
                .filter(|m| m.method.kind == MethodKind::Adapter)
                .map(|m| {
                    let spec = m
                        .method
                        .adapter
                        .as_ref()
                        .expect("adapter rows carry their spec");
                    assert_eq!(m.method.id, spec.id(), "{what}");
                    assert!(
                        spec.version.as_deref().is_some_and(|v| !v.is_empty()),
                        "{what}: {} records its engine version",
                        spec.name
                    );
                    assert!(m.method.dsl.is_empty(), "{what}: adapters run no program");
                    spec.name.clone()
                })
                .collect();
            adapters.sort();
            assert_eq!(adapters, version.adapters, "{what}");
            assert_eq!(
                r.methods.len(),
                Task::Regression.default_heads().len() + 1 + version.adapters.len(),
                "{what}: every default head, the search and every engine"
            );
            assert_eq!(
                r.methods.iter().filter(|m| m.deep.is_some()).count(),
                r.protocol.top,
                "{what}"
            );
            for m in &r.methods {
                assert_eq!(
                    m.screen.is_ok(),
                    m.screen_fitness.is_some(),
                    "{what}/{}",
                    m.method.id
                );
                if let Some(deep) = &m.deep {
                    assert_eq!(deep.status == "ok", deep.oof_score.is_some());
                    if deep.status == "ok" {
                        assert_eq!(deep.oof_score, deep.oof.get("r2").copied());
                    }
                }
            }
        }
        assert!(first.git_sha.as_deref().is_some_and(|s| s.len() == 40));
    }
}

#[test]
fn the_tracks_and_ladders_re_derive_byte_for_byte() {
    for version in &VERSIONS {
        let evidence = repo_root().join(version.dir);
        let reports = committed_reports(&evidence);
        let temp = tempfile::tempdir().unwrap();
        for stage in [Stage::Screen, Stage::Deep] {
            let track_id = format!("{TRACK}-{}", stage.suffix());
            let scores = emit::write_track(
                temp.path(),
                TRACK,
                stage,
                &reports,
                &evidence.join("reports"),
            )
            .unwrap();
            for rel in [
                format!("results/{track_id}/scores.csv"),
                format!("tracks/{track_id}/track.json"),
                format!("registry/datasets.{track_id}.json"),
                "registry/methods.json".to_owned(),
            ] {
                let fresh = std::fs::read(temp.path().join(&rel)).unwrap();
                let committed = std::fs::read(evidence.join("track").join(&rel)).unwrap();
                assert!(
                    fresh == committed,
                    "{}: {rel} differs from the committed track",
                    version.dir
                );
            }
            assert_eq!(
                scores,
                temp.path().join(format!("results/{track_id}/scores.csv"))
            );

            let (board, _) = leaderboard::compute(
                &evidence.join("track"),
                &track_id,
                None,
                Ladder::Published,
                None,
            )
            .unwrap();
            let csv = leaderboard::to_csv(&board.rows).unwrap();
            let committed_csv =
                std::fs::read_to_string(evidence.join(format!("leaderboard-{track_id}.csv")))
                    .unwrap();
            assert_eq!(csv, committed_csv, "{}: {track_id} ladder", version.dir);
            let snapshot = Snapshot::new(&board).unwrap();
            let committed: Snapshot = serde_json::from_str(
                &std::fs::read_to_string(evidence.join(format!("elo-{track_id}.json"))).unwrap(),
            )
            .unwrap();
            assert_eq!(snapshot, committed, "{}: {track_id} snapshot", version.dir);
            assert_eq!(board.datasets_scored, DATASETS);
            let ours = board
                .rows
                .iter()
                .find(|r| r.method_id == RESAMPLED_SEARCH_METHOD_ID)
                .expect("the search is on the ladder");
            assert!(ours.elo_rank >= 1);
            for name in version.adapters {
                assert!(
                    board
                        .rows
                        .iter()
                        .any(|r| r.method_id == format!("adapter.{name}")),
                    "{}: adapter.{name} is on the {track_id} ladder",
                    version.dir
                );
            }
        }
    }
}

#[test]
fn the_pwlinear_baselines_re_evaluate_bit_for_bit() {
    let dataset = load_dataset(&repo_root().join("datasets/test/pmlb-229-pwlinear")).unwrap();
    for version in &VERSIONS {
        let (_, report) = committed_reports(&repo_root().join(version.dir))
            .into_iter()
            .find(|(_, r)| r.dataset_hash == dataset.row.dataset_hash)
            .expect("pwlinear is in the evidence");
        assert_eq!(report.train_sha256, dataset.train_sha256);
        let cuts = build(
            &report.plan,
            &dataset.row.dataset_hash,
            &dataset.train_sha256,
            dataset.x.len(),
        )
        .unwrap();
        for (cut, info) in cuts.iter().zip(&report.cuts) {
            assert_eq!(cut.cut_id, info.cut_id);
            assert_eq!(
                (cut.train.len(), cut.hold.len()),
                (info.train_rows, info.hold_rows)
            );
        }
        let screen_cut = &cuts[report.protocol.screen_fold];
        let mut checked = 0;
        for m in report
            .methods
            .iter()
            .filter(|m| m.method.kind == MethodKind::Baseline)
        {
            let screen = evaluate(
                &m.method.dsl,
                &dataset.x,
                &dataset.y,
                screen_cut,
                Task::Regression,
            );
            assert_eq!(
                screen.status, m.screen.status,
                "{}/{}",
                version.dir, m.method.id
            );
            assert_eq!(
                screen.score, m.screen.score,
                "{}/{}",
                version.dir, m.method.id
            );
            assert_eq!(
                screen.metrics, m.screen.metrics,
                "{}/{}",
                version.dir, m.method.id
            );
            checked += 1;
            if let Some(deep) = &m.deep {
                let fresh = evaluate_deep_programs(
                    &fixed_programs(&m.method.dsl, &cuts),
                    &dataset.x,
                    &dataset.y,
                    &cuts,
                    Task::Regression,
                );
                assert_eq!(fresh.status, deep.status, "{}", m.method.id);
                assert_eq!(fresh.fold_score, deep.fold_score, "{}", m.method.id);
                assert_eq!(fresh.oof_score, deep.oof_score, "{}", m.method.id);
                assert_eq!(fresh.oof, deep.oof, "{}", m.method.id);
                assert_eq!(fresh.robust_fitness, deep.robust_fitness, "{}", m.method.id);
            }
        }
        assert_eq!(
            checked,
            Task::Regression.default_heads().len(),
            "{}",
            version.dir
        );
    }
}
