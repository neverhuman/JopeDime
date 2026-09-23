//! The committed `datasets/test/pmlb-229-pwlinear` through the whole benchmark, offline.

use std::path::{Path, PathBuf};

use jope_bench::cuts::{build, FoldPlan};
use jope_bench::emit::{self, Stage};
use jope_bench::eval::evaluate_deep;
use jope_bench::load::load_dataset;
use jope_bench::methods::{baselines, MethodKind};
use jope_bench::protocol::{self, Protocol};
use jope_bench::report::{Report, Stamp, SCHEMA};
use jope_bench::{DEFAULT_FOLDS, DEFAULT_SEED};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/jope-bench sits two levels below the repository root")
        .to_path_buf()
}

fn pwlinear() -> PathBuf {
    repo_root().join("datasets/test/pmlb-229-pwlinear")
}

#[test]
fn cuts_agree_with_feat_math_cv_and_the_run_is_deterministic() {
    let dataset = load_dataset(&pwlinear()).unwrap();
    assert_eq!(dataset.row.dataset_hash, "1cb97b1d7314d43d");
    assert_eq!(dataset.x.len(), 166);
    assert_eq!(dataset.x[0].len(), 10);
    assert_eq!(dataset.task(), "regression");

    let plan = FoldPlan::new(&dataset.row.dataset_hash, DEFAULT_SEED, DEFAULT_FOLDS);
    assert_eq!(plan.key, "1cb97b1d7314d43d:20260907");
    let cuts = build(
        &plan,
        &dataset.row.dataset_hash,
        &dataset.train_sha256,
        dataset.x.len(),
    )
    .unwrap();
    assert_eq!(cuts.len(), 5);

    // The deep ridge score pools out of fold in row order exactly as feat-math's own cv_r2 does
    // on the same key, so the two agree to the last bit: one fold authority, one number.
    let deep = evaluate_deep("(pipe X (ridge))", &dataset.x, &dataset.y, &cuts);
    assert_eq!(deep.status, "ok", "{deep:?}");
    let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let reference =
        feat_math::cv_r2(&genome, &dataset.x, &dataset.y, DEFAULT_FOLDS, &plan.key).unwrap();
    assert_eq!(deep.fold_r2, reference.fold_r2);
    assert_eq!(deep.oof_r2, Some(reference.oof_r2));

    let methods = baselines(&[
        "ridge".to_owned(),
        "elasticnet".to_owned(),
        "histgbm".to_owned(),
    ])
    .unwrap();
    let protocol = Protocol {
        screen_fold: 0,
        top: 2,
    };
    let first = protocol::run(&dataset.x, &dataset.y, &cuts, methods.clone(), &protocol).unwrap();
    let second = protocol::run(&dataset.x, &dataset.y, &cuts, methods, &protocol).unwrap();
    let strip = |results: &[protocol::MethodResult]| -> Vec<serde_json::Value> {
        results
            .iter()
            .map(|r| {
                let mut v = serde_json::to_value(r).unwrap();
                // timings are the only non-deterministic fields
                if let Some(o) = v.pointer_mut("/screen") {
                    o["fit_seconds"] = 0.into();
                    o["predict_seconds"] = 0.into();
                }
                if let Some(d) = v.pointer_mut("/deep") {
                    d["seconds"] = 0.into();
                    if let Some(folds) = d["folds"].as_array_mut() {
                        for f in folds {
                            f["outcome"]["fit_seconds"] = 0.into();
                            f["outcome"]["predict_seconds"] = 0.into();
                        }
                    }
                }
                v
            })
            .collect()
    };
    assert_eq!(strip(&first), strip(&second), "same inputs, same scores");
    assert!(first.iter().all(|r| r.screen.is_ok()));
    assert_eq!(first.iter().filter(|r| r.deep.is_some()).count(), 2);
    assert!(first.iter().all(|r| r.method.kind == MethodKind::Baseline));

    let temp = tempfile::tempdir().unwrap();
    let report = Report::new(
        &dataset,
        &plan,
        &cuts,
        &protocol,
        None,
        first,
        Stamp {
            workers: 2,
            git_sha: None,
        },
    );
    let path = temp.path().join(report.file_name());
    let sha = report.write(&path).unwrap();
    assert_eq!(sha.len(), 64);
    let back = Report::read(&path).unwrap();
    assert_eq!(back.schema, SCHEMA);
    assert_eq!(back, report);

    let loaded = vec![(path.clone(), back)];
    for stage in [Stage::Screen, Stage::Deep] {
        let scores =
            emit::write_track(temp.path(), "jope-same-cut-v1", stage, &loaded, temp.path())
                .unwrap();
        let rows = jope_arena::scores::read_scores(&scores).unwrap();
        assert_eq!(rows.len(), 3);
        assert!(rows
            .iter()
            .all(|r| r.protocol == "same-cut" && r.benchmark_valid == "True"));
        match stage {
            Stage::Screen => assert!(rows.iter().all(|r| r.status == "ok")),
            Stage::Deep => {
                assert_eq!(rows.iter().filter(|r| r.status == "ok").count(), 2);
                assert_eq!(rows.iter().filter(|r| r.status == "missing").count(), 1);
            }
        }
        let track_id = format!("jope-same-cut-v1-{}", stage.suffix());
        let (board, _) = jope_arena::leaderboard::compute(
            temp.path(),
            &track_id,
            None,
            jope_arena::leaderboard::Ladder::Published,
            None,
        )
        .unwrap();
        assert_eq!(board.datasets_scored, 1);
        assert!(board.rows.iter().all(|r| r.benchmark_valid == "True"));
    }
}
