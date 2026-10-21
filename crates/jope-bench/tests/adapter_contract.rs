//! The jope.bench-adapter/v1 process contract, end to end: adapters are real child processes.
//! The constant adapter is this package's own binary (`jope-bench mean-adapter`); failing, silent
//! and slow adapters are one-line POSIX shell commands. No test here needs Python.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use feat_math::Rows;
use jope_bench::adapter::{evaluate_adapter, evaluate_deep_adapter, parse_spec, AdapterSpec};
use jope_bench::cuts::{build, FoldPlan};
use jope_bench::eval::take;
use jope_bench::methods::{self, baselines, Method, MethodKind};
use jope_bench::protocol::{run, Protocol};
use jope_bench::task::Task;

const BINARY: &str = env!("CARGO_BIN_EXE_jope-bench");

fn mean_adapter(timeout: f64) -> AdapterSpec {
    parse_spec(&format!("mean={BINARY} mean-adapter"), timeout).unwrap()
}

/// A shell adapter: the contract's four arguments arrive as `$1`..`$4`.
fn shell(name: &str, script: &str, timeout_seconds: f64) -> AdapterSpec {
    AdapterSpec {
        name: name.to_owned(),
        command: vec!["sh".into(), "-c".into(), script.into(), "sh".into()],
        version: None,
        timeout_seconds,
    }
}

fn toy() -> (Rows, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut rng = feat_math::Pcg32::new(9, 2);
    for _ in 0..80 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        x.push(vec![a, b]);
        y.push(a + 0.5 * b);
    }
    (x, y)
}

#[test]
fn specs_parse_and_the_mean_adapter_answers_the_contract() {
    assert!(parse_spec("x", 1.0).is_err());
    assert!(parse_spec("bad name=cmd", 1.0).is_err());
    assert!(parse_spec("ok=", 1.0).is_err());
    assert!(parse_spec("ok=cmd", 0.0).is_err());
    let spec = parse_spec("rf=python3 adapters/x.py --trees 10", 5.0).unwrap();
    assert_eq!(spec.id(), "adapter.rf");
    assert_eq!(spec.command, ["python3", "adapters/x.py", "--trees", "10"]);

    let mut spec = mean_adapter(30.0);
    spec.probe().unwrap();
    assert_eq!(spec.version.as_deref(), Some("mean 1"));

    let (x, y) = toy();
    let cuts = build(&FoldPlan::new("m", 1, 4), "m", "s", x.len()).unwrap();
    let outcome = evaluate_adapter(&spec, &x, &y, &cuts[0], Task::Regression);
    assert!(outcome.is_ok(), "{outcome:?}");
    let (_, y_train) = take(&x, &y, &cuts[0].train);
    let mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
    assert!(outcome.predictions.iter().all(|p| (p - mean).abs() < 1e-12));
    assert!(
        outcome.score.unwrap() <= 0.0,
        "a constant predicts no variance"
    );
    assert!(outcome.fit_seconds > 0.0 && outcome.predict_seconds == 0.0);

    let deep = evaluate_deep_adapter(&spec, &x, &y, &cuts, Task::Regression);
    assert_eq!(deep.status, "ok", "{deep:?}");
    assert_eq!(deep.folds.len(), 4);
    assert!(deep.folds.iter().all(|f| f.dsl.is_empty()));

    // Binary: the positive rate; multiclass: the majority class.
    let yb: Vec<f64> = y
        .iter()
        .map(|v| if *v > 0.75 { 1.0 } else { 0.0 })
        .collect();
    let binary = evaluate_adapter(&spec, &x, &yb, &cuts[0], Task::Binary);
    assert!(binary.is_ok(), "{binary:?}");
    assert!(binary.predictions.iter().all(|p| (0.0..=1.0).contains(p)));
    let ym: Vec<f64> = y
        .iter()
        .map(|v| (v * 2.0).floor().clamp(0.0, 2.0))
        .collect();
    let multi = evaluate_adapter(&spec, &x, &ym, &cuts[0], Task::Multiclass);
    assert!(multi.is_ok(), "{multi:?}");
    assert!(multi.predictions.iter().all(|p| p.fract() == 0.0));
}

#[test]
fn adapter_failures_are_rejections_with_the_cause() {
    let (x, y) = toy();
    let cuts = build(&FoldPlan::new("f", 1, 3), "f", "s", x.len()).unwrap();
    assert!(
        parse_spec("huge=python3 x.py", 1e30).is_err(),
        "timeout must fit a Duration"
    );
    // Specs have no quoting, so a command with spaces inside one argument is built directly.
    let failing = shell("fail", "printf boom >&2; exit 3", 30.0);
    let outcome = evaluate_adapter(&failing, &x, &y, &cuts[0], Task::Regression);
    assert_eq!(outcome.status, "rejected");
    let reason = outcome.reason.as_deref().unwrap_or("");
    assert!(
        reason.contains("exit") && reason.contains("boom"),
        "{reason}"
    );

    let silent = shell("fail", ":", 30.0);
    let outcome = evaluate_adapter(&silent, &x, &y, &cuts[0], Task::Regression);
    assert!(outcome
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("no out.csv"));

    let slow = shell("fail", "sleep 5", 0.3);
    let started = Instant::now();
    let outcome = evaluate_adapter(&slow, &x, &y, &cuts[0], Task::Regression);
    assert!(started.elapsed() < Duration::from_secs(4));
    assert!(outcome
        .reason
        .as_deref()
        .unwrap_or("")
        .contains("timed out"));

    let mut unprobeable = silent;
    assert!(unprobeable.probe().is_err(), "prints nothing for --version");
}

fn mean_process(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(BINARY)
        .arg("mean-adapter")
        .args(args)
        .output()
        .unwrap();
    (
        output.status.code().unwrap(),
        String::from_utf8(output.stdout).unwrap(),
        String::from_utf8(output.stderr).unwrap(),
    )
}

fn predict_with(dir: &Path, train: &str, task: &str) -> (i32, String) {
    std::fs::write(dir.join("train.csv"), train).unwrap();
    std::fs::write(dir.join("predict.csv"), "0.5\n\n1.5\n0.25\n").unwrap();
    let out = dir.join("out.csv");
    let _ = std::fs::remove_file(&out);
    let paths = [dir.join("train.csv"), dir.join("predict.csv"), out.clone()];
    let paths: Vec<&str> = paths.iter().map(|p| p.to_str().unwrap()).collect();
    let (code, _, _) = mean_process(&[paths[0], paths[1], paths[2], task]);
    (code, std::fs::read_to_string(out).unwrap_or_default())
}

#[test]
fn mean_adapter_process_follows_the_retired_script() {
    assert_eq!(
        mean_process(&["--version"]),
        (0, "mean 1\n".into(), "".into())
    );
    let (code, _, usage) = mean_process(&[]);
    assert_eq!(code, 2);
    assert!(usage.contains("mean-adapter <train.csv> <predict.csv> <out.csv> <task>"));
    assert_eq!(mean_process(&["a", "b", "c"]).0, 2);

    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    // One prediction per non-blank held-out row; the target is the last field.
    assert_eq!(
        predict_with(dir, "1,2\n3,4\n\n5,6.5\n", "regression"),
        (0, "4.166666666666667\n".repeat(3))
    );
    assert_eq!(
        predict_with(dir, "0,1\n0,0\n0,1\n0,1\n", "binary"),
        (0, "0.75\n".repeat(3))
    );
    // Codes truncate toward zero; the earliest class wins a tie.
    assert_eq!(
        predict_with(dir, "0,2.9\n0,1\n0,1\n0,2\n", "multiclass"),
        (0, "2.0\n".repeat(3))
    );
    assert_eq!(predict_with(dir, "0,1\n", "ranking"), (2, String::new()));
    assert_eq!(predict_with(dir, "\n \n", "regression"), (1, String::new()));
    assert_eq!(predict_with(dir, "7\n", "regression"), (1, String::new()));
}

#[test]
fn screen_ranks_everyone_and_deepens_the_top() {
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut rng = feat_math::Pcg32::new(11, 5);
    for _ in 0..160 {
        let a = rng.next_f64();
        let b = rng.next_f64();
        x.push(vec![a, b, rng.next_f64()]);
        y.push(a * a + 0.5 * b);
    }
    let cuts = build(&FoldPlan::new("p", 1, 4), "p", "s", x.len()).unwrap();
    let mut methods = baselines(
        &[
            "ridge".to_owned(),
            "histgbm".to_owned(),
            "ksmooth".to_owned(),
        ],
        Task::Regression,
    )
    .unwrap();
    methods.push(Method {
        id: "program.broken".to_owned(),
        kind: MethodKind::Program,
        dsl: "(pipe X (no_such_head_ever))".to_owned(),
        produce_seconds: 0.0,
        resample: None,
        adapter: None,
    });
    // An adapter goes through the same screen and deep stages as a program.
    methods.push(methods::adapter(mean_adapter(30.0)));
    let protocol = Protocol {
        screen_fold: 0,
        top: 2,
    };
    let results = run(&x, &y, &cuts, methods.clone(), &protocol, Task::Regression).unwrap();
    assert_eq!(results.len(), 5);
    assert_eq!(
        results.iter().map(|r| r.screen_rank).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5]
    );
    assert_eq!(
        results[4].method.id, "program.broken",
        "rejected ranks last"
    );
    assert!(results[4].deep.is_none());
    let mean = results
        .iter()
        .find(|r| r.method.id == "adapter.mean")
        .expect("the adapter was screened");
    assert!(mean.screen.is_ok(), "{:?}", mean.screen);
    assert_eq!(
        mean.screen_rank, 4,
        "a constant prediction ranks below every fitted head"
    );
    assert_eq!(results.iter().filter(|r| r.deep.is_some()).count(), 2);
    assert!(results[0].deep_rank.is_some());
    let deep = results[0].deep.as_ref().unwrap();
    assert_eq!(deep.folds.len(), 4);
    assert!(results[0].screen.predictions.is_empty());
    for r in &results {
        assert_eq!(r.screen_cut_id, cuts[0].cut_id);
    }
    // Screen fitness is the raw held-out R², so ranks follow it strictly.
    let fits: Vec<f64> = results[..4]
        .iter()
        .map(|r| r.screen_fitness.unwrap())
        .collect();
    assert!(fits.windows(2).all(|w| w[0] >= w[1]), "{fits:?}");

    // Duplicate ids are refused.
    methods.push(methods[0].clone());
    let error = run(&x, &y, &cuts, methods, &protocol, Task::Regression).unwrap_err();
    assert!(error.to_string().contains("duplicate method id"), "{error}");
}

fn entrant_spec(name: &str) -> AdapterSpec {
    AdapterSpec {
        name: name.to_owned(),
        command: vec![BINARY.into(), "regressor-adapter".into(), name.into()],
        version: None,
        timeout_seconds: 120.0,
    }
}

#[test]
fn frozen_entrants_answer_the_contract_and_beat_the_constant() {
    let (x, y) = toy();
    let cuts = build(&FoldPlan::new("e", 1, 4), "e", "s", x.len()).unwrap();
    let mean = evaluate_adapter(&mean_adapter(30.0), &x, &y, &cuts[0], Task::Regression);
    for e in jope_bench::regressors::ENTRANTS {
        let mut spec = entrant_spec(e.name);
        spec.probe().unwrap();
        let version = spec.version.clone().unwrap();
        assert!(
            version.starts_with(e.name) && version.ends_with(e.source_sha256),
            "{version}"
        );
        let outcome = evaluate_adapter(&spec, &x, &y, &cuts[0], Task::Regression);
        assert!(outcome.is_ok(), "{outcome:?}");
        assert_eq!(outcome.predictions.len(), cuts[0].hold.len());
        // y = a + b/2 is learnable from 60 rows: the entrant must clear the constant floor.
        assert!(
            outcome.score.unwrap() > mean.score.unwrap() + 0.5,
            "{}: {:?} vs mean {:?}",
            e.name,
            outcome.score,
            mean.score
        );
        let deep = evaluate_deep_adapter(&spec, &x, &y, &cuts, Task::Regression);
        assert_eq!(deep.status, "ok", "{deep:?}");
        // Regressors refuse other tasks with the usage code, so they rank as rejections there.
        let yb: Vec<f64> = y
            .iter()
            .map(|v| if *v > 0.75 { 1.0 } else { 0.0 })
            .collect();
        let binary = evaluate_adapter(&spec, &x, &yb, &cuts[0], Task::Binary);
        assert_eq!(binary.status, "rejected");
        assert!(binary.reason.unwrap_or_default().contains("regression"));
    }
    let output = Command::new(BINARY)
        .args(["regressor-adapter", "no-such-entrant", "--version"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
}
