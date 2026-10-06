//! The champion bank: only HWM-improved pipeline champions whose receipt and HWM event verify
//! are admitted, the bank is append-only and content-addressed, and `bank export` writes the
//! `jope.pipeline-seeds/v1` file `jope pipeline run --seed-genomes` consumes.

use jope_learning::bank;
use jope_learning::hwm;
use jope_learning::ledger::Ledger;
use jope_learning::sources::pipeline;
use jope_learning::state::State;
use jope_pipeline::run::{run, Policy, RunOptions};
use jope_pipeline::search::Budget;
use std::path::{Path, PathBuf};

fn dataset() -> String {
    format!(
        "{}/../../datasets/test/pmlb-229-pwlinear",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn options(output: &Path, evaluations: usize) -> RunOptions {
    RunOptions {
        budget: Budget {
            evaluations,
            seconds: 600,
        },
        rng_seed: 7,
        languages: vec![jope_pipeline::emit::Language::Rust],
        policy: Policy::Skip,
        ..RunOptions::new(dataset(), output)
    }
}

/// Each test's fixture: a bootstrap-only run, a byte-distinct replay of it (same score), a better
/// run seeded from it, and a state whose HWM ledger initialized on the first, rejected the
/// replay as a tie and improved on the third.
struct Runs {
    temp: tempfile::TempDir,
    weak: PathBuf,
    tie: PathBuf,
    strong: PathBuf,
    state: State,
}

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

fn ingested() -> Runs {
    let temp = tempfile::tempdir().unwrap();
    let weak = temp.path().join("weak");
    run(&options(&weak, 1)).unwrap();
    let tie = temp.path().join("tie");
    run(&options(&tie, 1)).unwrap();
    let strong = temp.path().join("strong");
    run(&RunOptions {
        incumbent: Some(weak.join("receipt.json")),
        ..options(&strong, 24)
    })
    .unwrap();
    let state = State::at(&temp.path().join("state"));
    let (weak, tie, strong) = (
        weak.join("receipt.json"),
        tie.join("receipt.json"),
        strong.join("receipt.json"),
    );
    assert_eq!(ingest(&state, &weak).transitions["initialized"], 1);
    assert_eq!(ingest(&state, &tie).transitions["tie"], 1);
    assert_eq!(ingest(&state, &strong).transitions["improved"], 1);
    Runs {
        temp,
        weak,
        tie,
        strong,
        state,
    }
}

fn bank_batches(state: &State) -> usize {
    Ledger::open(state.root(), bank::LEDGER)
        .batches()
        .unwrap()
        .len()
}

#[test]
fn only_an_hwm_improved_champion_is_banked() {
    let runs = ingested();
    let state = &runs.state;
    let initialized = format!("{:#}", bank::add(state, &runs.weak, None).unwrap_err());
    assert!(
        initialized.contains("`initialized`, not `improved`"),
        "{initialized}"
    );
    let tie = format!("{:#}", bank::add(state, &runs.tie, None).unwrap_err());
    assert!(tie.contains("rejected this receipt"), "{tie}");
    assert_eq!(bank_batches(state), 0, "a refusal wrote to the bank");

    let added = bank::add(state, &runs.strong, None).unwrap();
    assert!(!added.already_banked);
    assert_eq!(added.entry.hwm.transition, "improved");
    assert_eq!(
        added.entry.receipt_sha256,
        jope_pipeline::sha256_file(&runs.strong).unwrap()
    );
    let listed = bank::list(state).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].entry_sha256, added.entry_sha256);
}

#[test]
fn the_bank_is_append_only_and_content_addressed() {
    let runs = ingested();
    let state = &runs.state;
    let first = bank::add(state, &runs.strong, None).unwrap();
    let again = bank::add(state, &runs.strong, None).unwrap();
    assert!(again.already_banked);
    assert_eq!(again.entry_sha256, first.entry_sha256);
    assert_eq!(
        bank_batches(state),
        1,
        "a repeated add wrote a second batch"
    );
    // Editing a banked entry breaks the chain; the bank then refuses to list or export.
    let (_, path) = Ledger::open(state.root(), bank::LEDGER)
        .batches()
        .unwrap()
        .remove(0);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replacen("\"validation_rmse\":", "\"validation_rmse\":1", 1),
    )
    .unwrap();
    assert!(bank::list(state).is_err());
}

#[test]
fn an_unverifiable_receipt_or_hwm_ledger_is_refused() {
    let runs = ingested();
    let (temp, state) = (&runs.temp, &runs.state);
    // A receipt never ingested has no HWM decision.
    let fresh = temp.path().join("fresh");
    run(&options(&fresh, 2)).unwrap();
    let error = format!(
        "{:#}",
        bank::add(state, &fresh.join("receipt.json"), None).unwrap_err()
    );
    assert!(error.contains("holds no decision"), "{error}");
    // A receipt whose evidence was edited.
    let copy = temp.path().join("copy");
    copy_tree(runs.strong.parent().unwrap(), &copy);
    let observations = copy.join("search/cut-0/observations.json");
    let mut rows: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&observations).unwrap()).unwrap();
    rows[0]["seconds"] = 1e9.into();
    std::fs::write(&observations, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
    let error = format!(
        "{:#}",
        bank::add(state, &copy.join("receipt.json"), None).unwrap_err()
    );
    assert!(error.contains("does not verify"), "{error}");
    // An HWM ledger that no longer walks clean.
    let (_, batch) = Ledger::open(state.root(), "hwm")
        .batches()
        .unwrap()
        .pop()
        .unwrap();
    let mut bytes = std::fs::read(&batch).unwrap();
    let at = bytes.iter().position(|b| *b == b'{').unwrap() + 1;
    bytes.insert(at, b' ');
    std::fs::write(&batch, bytes).unwrap();
    let error = format!("{:#}", bank::add(state, &runs.strong, None).unwrap_err());
    assert!(
        error.contains("does not verify") || error.contains("break"),
        "{error}"
    );
    assert_eq!(bank_batches(state), 0);
}

#[test]
fn an_export_seeds_the_next_run() {
    let runs = ingested();
    let (temp, state) = (&runs.temp, &runs.state);
    bank::add(state, &runs.strong, None).unwrap();
    let seeds = temp.path().join("seeds.json");
    let exported = bank::export(state, "1cb97b1d7314d43d", None, None, 4, &seeds).unwrap();
    assert_eq!(exported.genomes, 1);
    // Never overwritten.
    assert!(bank::export(state, "1cb97b1d7314d43d", None, None, 4, &seeds).is_err());
    assert!(bank::export(
        state,
        "0000000000000000",
        None,
        None,
        4,
        &temp.path().join("none.json")
    )
    .is_err());

    let (file, sha) = jope_pipeline::seeds::read(&seeds).unwrap();
    let banked = bank::list(state).unwrap().remove(0);
    assert_eq!(file.genomes[0].genome, banked.entry.genome);
    assert_eq!(
        file.genomes[0].bank_entry_sha256.as_deref(),
        Some(banked.entry_sha256.as_str())
    );
    let next = temp.path().join("next");
    let receipt = run(&RunOptions {
        seed_genomes: Some(seeds.clone()),
        ..options(&next, 16)
    })
    .unwrap();
    let seeding = receipt.search.seeding.unwrap();
    assert_eq!(seeding.seeds_file_sha256.as_deref(), Some(sha.as_str()));
    if banked.entry.genome != jope_pipeline::search::BOOTSTRAP_GENOME {
        assert_eq!(seeding.seeds[0].genome, banked.entry.genome);
    }
}

/// A second comparison identity on the same dataset: another fold seed prepares other cuts. Its
/// bootstrap-only run initializes that identity's pointer and a better run improves it.
fn bank_a_second_identity(runs: &Runs) -> PathBuf {
    let (temp, state) = (&runs.temp, &runs.state);
    let other = |name: &str, evaluations: usize, incumbent: Option<PathBuf>| {
        let output = temp.path().join(name);
        run(&RunOptions {
            seed: 11,
            incumbent,
            ..options(&output, evaluations)
        })
        .unwrap();
        output.join("receipt.json")
    };
    let weak = other("weak-11", 1, None);
    let strong = other("strong-11", 24, Some(weak.clone()));
    assert_eq!(ingest(state, &weak).transitions["initialized"], 1);
    assert_eq!(ingest(state, &strong).transitions["improved"], 1);
    bank::add(state, &strong, None).unwrap();
    strong
}

/// `bank export` never ranks champions measured under different comparison identities (other
/// cuts, fold seed, protocol or source commit) against each other.
#[test]
fn an_export_never_ranks_across_comparison_identities() {
    let runs = ingested();
    let (temp, state) = (&runs.temp, &runs.state);
    let first = bank::add(state, &runs.strong, None).unwrap().entry;
    bank_a_second_identity(&runs);
    let entries: Vec<bank::BankEntry> = bank::list(state)
        .unwrap()
        .into_iter()
        .map(|e| e.entry)
        .collect();
    assert_eq!(entries.len(), 2);
    assert_ne!(entries[0].identity_sha256, entries[1].identity_sha256);

    let mixed = temp.path().join("mixed.json");
    let refused = bank::export(state, "1cb97b1d7314d43d", None, None, 4, &mixed);
    let error = format!(
        "{:#}",
        refused.expect_err("an export ranked across identities")
    );
    assert!(error.contains("comparison identities"), "{error}");
    for entry in &entries {
        assert!(error.contains(&entry.identity_sha256), "{error}");
    }
    assert!(!mixed.exists());
    let unknown = "0".repeat(64);
    assert!(bank::export(state, "1cb97b1d7314d43d", None, Some(&unknown), 4, &mixed).is_err());
    assert!(!mixed.exists());

    // Naming one identity exports that identity's champions only.
    for entry in &entries {
        let out = temp
            .path()
            .join(format!("{}.json", &entry.identity_sha256[..12]));
        let report = bank::export(
            state,
            "1cb97b1d7314d43d",
            None,
            Some(&entry.identity_sha256),
            4,
            &out,
        )
        .unwrap();
        assert_eq!(report.identity_sha256, entry.identity_sha256);
        let (file, _) = jope_pipeline::seeds::read(&out).unwrap();
        assert_eq!(file.genomes.len(), 1);
        assert_eq!(
            file.genomes[0].receipt_sha256.as_deref(),
            Some(entry.receipt_sha256.as_str())
        );
    }
    assert!(entries.contains(&first));
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
fn the_cli_adds_lists_and_exports() {
    let runs = ingested();
    let (temp, state) = (&runs.temp, &runs.state);
    let cli = |args: &[&str]| {
        std::process::Command::new(env!("CARGO_BIN_EXE_jope-learning"))
            .arg("--state")
            .arg(state.root())
            .args(args)
            .output()
            .unwrap()
    };
    let strong = runs.strong.display().to_string();
    let weak = runs.weak.display().to_string();
    let refused = cli(&["bank", "add", "--receipt", &weak]);
    assert!(!refused.status.success());
    let added = cli(&["bank", "add", "--receipt", &strong]);
    assert!(
        added.status.success(),
        "{}",
        String::from_utf8_lossy(&added.stderr)
    );
    let listed = cli(&["bank", "list", "--dataset", "1cb97b1d7314d43d"]);
    let listed: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(listed["entries"].as_array().unwrap().len(), 1);
    let out = temp.path().join("cli-seeds.json");
    let exported = cli(&[
        "bank",
        "export",
        "--dataset",
        "1cb97b1d7314d43d",
        "--out",
        &out.display().to_string(),
    ]);
    assert!(
        exported.status.success(),
        "{}",
        String::from_utf8_lossy(&exported.stderr)
    );
    jope_pipeline::seeds::read(&out).unwrap();
}
