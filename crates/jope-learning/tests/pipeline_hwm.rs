//! `jope-learning hwm ingest --source pipeline`: a verified pipeline receipt's champion becomes
//! a `jope.hwm/v1` record under a comparison identity naming the dataset, the cuts, the protocol
//! and the source commit that measured it, and the ledger's admission rules apply to it
//! unchanged: initialized, then improved; a tie and a not-better are rejected; a different cut
//! set, dataset or source commit is a different identity that never displaces this one; a
//! receipt naming no commit is refused; nothing it records is teaching-eligible.

use jope_learning::hwm::{self, Decision, RejectReason};
use jope_learning::sources::pipeline;
use jope_learning::state::State;
use jope_pipeline::run::{run, Policy, RunOptions};
use jope_pipeline::search::Budget;
use std::path::{Path, PathBuf};

fn dataset(slug: &str) -> String {
    format!("{}/../../datasets/test/{slug}", env!("CARGO_MANIFEST_DIR"))
}

fn run_into(
    root: &Path,
    name: &str,
    evaluations: usize,
    cuts: usize,
    slug: &str,
    incumbent: Option<&Path>,
) -> PathBuf {
    let output = root.join(name);
    run(&RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        cuts,
        languages: vec![jope_pipeline::emit::Language::Rust],
        policy: Policy::Skip,
        incumbent: incumbent.map(Path::to_path_buf),
        ..RunOptions::new(dataset(slug), &output)
    })
    .unwrap();
    output.join("receipt.json")
}

/// Each test's pipeline runs: a bootstrap-only run and two byte-distinct replays of it (same
/// score), a better run seeded from it, and runs on another cut set and another dataset.
struct Runs {
    _root: tempfile::TempDir,
    weak: PathBuf,
    weak_again: PathBuf,
    weak_third: PathBuf,
    strong: PathBuf,
    other_cuts: PathBuf,
    other_dataset: PathBuf,
}

fn runs() -> Runs {
    let root = tempfile::tempdir().unwrap();
    let slug = "pmlb-229-pwlinear";
    let weak = run_into(root.path(), "weak", 1, 1, slug, None);
    let weak_again = run_into(root.path(), "weak-again", 1, 1, slug, None);
    let weak_third = run_into(root.path(), "weak-third", 1, 1, slug, None);
    let strong = run_into(root.path(), "strong", 24, 1, slug, Some(&weak));
    let other_cuts = run_into(root.path(), "other-cuts", 1, 2, slug, None);
    let other_dataset = run_into(root.path(), "other-dataset", 1, 1, OTHER_DATASET, None);
    Runs {
        _root: root,
        weak,
        weak_again,
        weak_third,
        strong,
        other_cuts,
        other_dataset,
    }
}

const OTHER_DATASET: &str = "pmlb-695-chatfield-4";

fn ingest(state: &State, receipt: &Path) -> hwm::IngestReport {
    let (candidates, scan) = pipeline::scan(receipt, None).unwrap();
    hwm::ingest(
        state,
        None,
        candidates,
        pipeline::scan_source(receipt, &scan),
    )
    .unwrap()
}

fn transitions(report: &hwm::IngestReport) -> Vec<(String, u64)> {
    report
        .transitions
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect()
}

fn record(receipt: &Path) -> jope_learning::record::HwmRecord {
    pipeline::scan(receipt, None).unwrap().0.remove(0).record
}

#[test]
fn an_improvement_is_accepted_after_initialization_and_the_ledger_verifies() {
    let runs = runs();
    let temp = tempfile::tempdir().unwrap();
    let state = State::at(temp.path());
    assert!(record(&runs.strong).score.value < record(&runs.weak).score.value);
    assert_eq!(
        transitions(&ingest(&state, &runs.weak)),
        vec![("initialized".to_owned(), 1)]
    );
    assert_eq!(
        transitions(&ingest(&state, &runs.strong)),
        vec![("improved".to_owned(), 1)]
    );
    let report = hwm::verify(&state, &[]).unwrap();
    assert!(report.ok(), "{:?}", report.breaks);
    assert_eq!((report.initialized, report.improved), (1, 1));
    let pointers = state.pointers().unwrap();
    assert_eq!(pointers.len(), 1);
    let (weak, strong) = (record(&runs.weak), record(&runs.strong));
    let current = &pointers[0].1;
    // The ledger chained the improvement onto the record it displaced.
    assert_eq!(
        current.provenance.record_sha256,
        strong.provenance.record_sha256
    );
    assert_eq!(current.identity_sha256, strong.identity_sha256);
    assert_eq!(current.genome, strong.genome);
    assert_eq!(current.score, strong.score);
    assert_eq!(current.improvement_number, 1);
    assert_eq!(
        current.prev_event_sha256.as_deref(),
        Some(weak.provenance.record_sha256.as_str())
    );
    // The blob is the receipt, byte for byte.
    let sha = &current.provenance.record_sha256;
    assert_eq!(
        state.read_blob(sha).unwrap().unwrap(),
        std::fs::read(&runs.strong).unwrap()
    );
    // And the record is a valid jope.hwm/v1 under the committed schema.
    let schema: serde_json::Value = serde_json::from_str(jope_learning::HWM_SCHEMA_JSON).unwrap();
    let errors = jope_datasets::schema::validate(&schema, &serde_json::to_value(current).unwrap());
    assert!(errors.is_empty(), "{errors:?}");
}

#[test]
fn a_tie_and_a_not_better_champion_are_rejected() {
    let runs = runs();
    let temp = tempfile::tempdir().unwrap();
    let state = State::at(temp.path());
    ingest(&state, &runs.weak);
    // Same cuts, same fit, same score; different receipt bytes.
    assert_eq!(
        record(&runs.weak_again).score.value,
        record(&runs.weak).score.value
    );
    let tie = ingest(&state, &runs.weak_again);
    assert_eq!(transitions(&tie), vec![("tie".to_owned(), 1)]);
    assert_eq!(tie.rejects[0].reason, RejectReason::Tie.as_str());
    ingest(&state, &runs.strong);
    let worse = ingest(&state, &runs.weak_third);
    assert_eq!(transitions(&worse), vec![("not-improved".to_owned(), 1)]);
    assert_eq!(worse.rejects[0].reason, RejectReason::NotBetter.as_str());
    let report = hwm::verify(&state, &[]).unwrap();
    assert!(report.ok(), "{:?}", report.breaks);
    assert_eq!(
        (report.accepted, report.rejected, report.improved),
        (2, 2, 1)
    );
    // No authority moved on a rejection: the pointer still holds the improvement.
    let current = state.pointers().unwrap().remove(0).1;
    assert_eq!(
        current.provenance.record_sha256,
        record(&runs.strong).provenance.record_sha256
    );
    assert_eq!(current.improvement_number, 1);
}

#[test]
fn another_cut_set_or_dataset_is_another_identity_and_never_displaces() {
    let runs = runs();
    let weak = record(&runs.weak);
    for other in [&runs.other_cuts, &runs.other_dataset] {
        let other = record(other);
        assert_ne!(other.identity_sha256, weak.identity_sha256);
        assert_eq!(
            hwm::decide(Some(&weak), &other),
            Decision::Rejected(RejectReason::IdentityMismatch)
        );
    }
    let temp = tempfile::tempdir().unwrap();
    let state = State::at(temp.path());
    ingest(&state, &runs.weak);
    let separate = ingest(&state, &runs.other_cuts);
    assert_eq!(transitions(&separate), vec![("initialized".to_owned(), 1)]);
    let pointers = state.pointers().unwrap();
    assert_eq!(pointers.len(), 2);
    assert!(pointers.iter().any(|(_, r)| *r == weak));
    assert!(hwm::verify(&state, &[]).unwrap().ok());
}

/// A run at `commit` (as `--source-commit` records it).
fn run_at(
    root: &Path,
    name: &str,
    evaluations: usize,
    commit: &str,
    incumbent: Option<&Path>,
) -> PathBuf {
    let output = root.join(name);
    run(&RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![jope_pipeline::emit::Language::Rust],
        policy: Policy::Skip,
        incumbent: incumbent.map(Path::to_path_buf),
        source_commit: Some(commit.to_owned()),
        ..RunOptions::new(dataset("pmlb-229-pwlinear"), &output)
    })
    .unwrap();
    output.join("receipt.json")
}

/// The evaluator is the code that measured: the workspace version never moves, so the source
/// commit is what tells two evaluators apart. A lower score from other code is another identity,
/// never an improvement, and a receipt that names no commit cannot be compared at all.
#[test]
fn another_source_commit_is_another_identity_and_an_unbound_one_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let (a, b) = ("a".repeat(40), "b".repeat(40));
    let weak = run_at(root.path(), "weak", 1, &a, None);
    let weak_again = run_at(root.path(), "weak-again", 1, &a, None);
    let strong_elsewhere = run_at(root.path(), "strong-b", 24, &b, Some(&weak));
    let (weak_r, strong_r) = (record(&weak), record(&strong_elsewhere));
    assert!(strong_r.score.value < weak_r.score.value);
    assert_eq!(weak_r.identity_sha256, record(&weak_again).identity_sha256);
    assert_ne!(weak_r.identity_sha256, strong_r.identity_sha256);
    assert_eq!(
        hwm::decide(Some(&weak_r), &strong_r),
        Decision::Rejected(RejectReason::IdentityMismatch)
    );
    let temp = tempfile::tempdir().unwrap();
    let state = State::at(temp.path());
    ingest(&state, &weak);
    assert_eq!(
        transitions(&ingest(&state, &strong_elsewhere)),
        vec![("initialized".to_owned(), 1)],
        "a better score from other code displaced this code's record"
    );
    assert_eq!(state.pointers().unwrap().len(), 2);

    // The same run with no recorded commit.
    let copy = root.path().join("unbound");
    copy_tree(weak.parent().unwrap(), &copy);
    let receipt = copy.join("receipt.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
    value["git_sha"] = serde_json::Value::Null;
    std::fs::write(&receipt, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    jope_pipeline::evidence::verify_champion_run(&receipt).unwrap();
    let Err(error) = pipeline::scan(&receipt, None) else {
        panic!("a receipt that names no source commit was mapped");
    };
    let error = format!("{error:#}");
    assert!(error.contains("source commit"), "{error}");
}

#[test]
fn pipeline_records_are_never_teaching_eligible() {
    let runs = runs();
    let temp = tempfile::tempdir().unwrap();
    let state = State::at(temp.path());
    ingest(&state, &runs.weak);
    let r = record(&runs.weak);
    assert!(!r.replayable && !r.benchmark_valid && !r.sealed);
    assert_eq!(r.evaluator.sha256, jope_learning::record::UNBOUND);
    let context = r.identity.comparison.as_ref().unwrap();
    assert_ne!(context.stage, "train_oof");
    let manifest = jope_learning::eligibility::freeze_teaching(&state).unwrap();
    assert!(manifest.entries.is_empty());
    assert_eq!(manifest.excluded_diagnostic, 1);
    let out = tempfile::tempdir().unwrap();
    let frozen = hwm::freeze(&state, out.path(), true).unwrap();
    assert!(!frozen.teaching_eligible);
}

#[test]
fn a_receipt_its_evidence_does_not_reproduce_writes_nothing() {
    let runs = runs();
    let temp = tempfile::tempdir().unwrap();
    // A copy of the weak run with one observation edited.
    let copy = temp.path().join("copy");
    copy_tree(runs.weak.parent().unwrap(), &copy);
    let observations = copy.join("search/cut-0/observations.json");
    let mut rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&observations).unwrap()).unwrap();
    rows[0]["rmse"] = 0.0.into();
    std::fs::write(&observations, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    let Err(error) = pipeline::scan(&copy.join("receipt.json"), None) else {
        panic!("a receipt whose evidence was edited was mapped");
    };
    let error = format!("{error:#}");
    assert!(error.contains("hashes to"), "{error}");
    let state = State::at(&temp.path().join("state"));
    assert!(!state.root().exists());
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let target = to.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

#[test]
fn the_cli_ingests_a_pipeline_receipt() {
    let runs = runs();
    let temp = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_jope-learning"))
        .args(["--state"])
        .arg(temp.path())
        .args(["hwm", "ingest", "--source", "pipeline"])
        .arg(&runs.weak)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["transitions"]["initialized"], 1);
    assert_eq!(report["scan"]["dataset_hash"], "1cb97b1d7314d43d");
}
