use feat_math::{evolve::FaultPolicy, GenomeIr};
use jope_pipeline::{
    kernel2::search::{self, Options, Refinement},
    prepare::Cut,
    search::{Budget, SearchOptions},
    search_strategy,
};
use std::{collections::BTreeMap, fs, process::Command};

#[test]
fn kernel2_and_shared_strategy_capture_complete_parseable_programs() {
    if std::env::var_os("JOPE_GP_KERNEL2_CAPTURE_CHILD").is_some() {
        let train_x = (0..16).map(|i| vec![i as f64 / 10.0]).collect::<Vec<_>>();
        let train_y = train_x.iter().map(|r| 3.0 * r[0] + 0.5).collect();
        let hold_x = vec![vec![0.2], vec![0.8]];
        let hold_y = hold_x.iter().map(|r| 3.0 * r[0] + 0.5).collect();
        let cut = Cut {
            index: 0,
            train_x,
            train_y,
            hold_x,
            hold_y,
            hold_indices: vec![0, 1],
        };
        let source = SearchOptions::new(
            Budget {
                evaluations: 4,
                seconds: 120,
            },
            7,
        );
        let options = Options::new(
            source,
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            Refinement {
                top_k: 1,
                trials_per_program: 1,
            },
        )
        .unwrap();
        assert!(!search::run_cut(&cut, &options)
            .unwrap()
            .observations
            .is_empty());
        assert!(
            !search_strategy::run_kernel2(&cut, options.evaluator(), 7, &"a".repeat(40), 2)
                .unwrap()
                .measurements
                .is_empty()
        );
        return;
    }
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/gp-source-tests");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "kernel2_and_shared_strategy_capture_complete_parseable_programs",
            "--nocapture",
        ])
        .env("JOPE_GP_KERNEL2_CAPTURE_CHILD", "1")
        .env("JOPE_GP_SOURCE_DIR", directory.path())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let mut counts = BTreeMap::<String, usize>::new();
    for entry in fs::read_dir(directory.path()).unwrap() {
        for line in fs::read_to_string(entry.unwrap().path()).unwrap().lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            let program = record["program"].as_str().unwrap();
            assert!(!program.is_empty());
            GenomeIr::parse(program).unwrap();
            *counts
                .entry(record["engine"].as_str().unwrap().into())
                .or_default() += 1;
        }
    }
    for engine in [
        "kernel2",
        "kernel2-evaluation",
        "kernel2-operator",
        "shared-strategy",
    ] {
        assert!(counts[engine] > 0, "missing {engine}");
    }
}
