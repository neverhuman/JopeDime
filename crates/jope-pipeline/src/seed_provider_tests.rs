use super::*;
use crate::{clock::Clock, dataset, emit::Language, evidence, search::Budget, seeds::SeedGenome};
use std::sync::Mutex;

fn committed_dataset() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../datasets/test/pmlb-229-pwlinear"
    )
    .to_owned()
}

fn copy_dataset_with_poisoned_holdout(
    dest_root: &std::path::Path,
    seed: u64,
    cut_index: usize,
) -> String {
    let src = std::path::PathBuf::from(committed_dataset());
    let resolved = dataset::resolve(&committed_dataset()).unwrap();
    let rows = dataset::training_rows(&resolved).unwrap();
    let prep_dir = dest_root.join("ref-prep");
    let prep = prepare::build(&rows, &resolved.row.dataset_hash, seed, 1, &prep_dir).unwrap();
    let hold_set: std::collections::BTreeSet<usize> =
        prep.cuts[cut_index].hold_indices.iter().copied().collect();

    let out = dest_root.join("poisoned-holdout");
    std::fs::create_dir_all(&out).unwrap();

    let train_text = std::fs::read_to_string(src.join("train.csv")).unwrap();
    let mut mutated_train = String::new();
    for (row_idx, line) in train_text.lines().enumerate() {
        if hold_set.contains(&row_idx) {
            let vals: Vec<f64> = line
                .split(',')
                .enumerate()
                .map(|(col_idx, s)| {
                    let v: f64 = s.parse().unwrap();
                    -1000.0 * v + 777.0 + (row_idx * 13 + col_idx) as f64
                })
                .collect();
            let joined = vals
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(",");
            mutated_train.push_str(&joined);
            mutated_train.push('\n');
        } else {
            mutated_train.push_str(line);
            mutated_train.push('\n');
        }
    }

    let test_text = std::fs::read_to_string(src.join("test.csv")).unwrap();
    let mut mutated_test = String::new();
    for (row_idx, line) in test_text.lines().enumerate() {
        let vals: Vec<f64> = line
            .split(',')
            .map(|s| -500.0 * s.parse::<f64>().unwrap() - 333.0 - row_idx as f64)
            .collect();
        let joined = vals
            .iter()
            .map(|v| format!("{v}"))
            .collect::<Vec<_>>()
            .join(",");
        mutated_test.push_str(&joined);
        mutated_test.push('\n');
    }

    std::fs::write(out.join("train.csv"), &mutated_train).unwrap();
    std::fs::write(out.join("test.csv"), &mutated_test).unwrap();

    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("metadata.json")).unwrap()).unwrap();
    meta["files"]["train"]["sha256"] = sha256_hex(mutated_train.as_bytes()).into();
    meta["files"]["train"]["bytes"] = mutated_train.len().into();
    meta["files"]["test"]["sha256"] = sha256_hex(mutated_test.as_bytes()).into();
    meta["files"]["test"]["bytes"] = mutated_test.len().into();
    std::fs::write(out.join("metadata.json"), meta.to_string()).unwrap();
    out.display().to_string()
}

struct RecordingFitProvider {
    captured_digests: Mutex<Vec<String>>,
    captured_proposals: Mutex<Vec<SeedGenome>>,
}

impl RecordingFitProvider {
    fn new() -> Self {
        Self {
            captured_digests: Mutex::new(Vec::new()),
            captured_proposals: Mutex::new(Vec::new()),
        }
    }
}

impl CutSeedProvider for RecordingFitProvider {
    fn propose_seeds(&self, fit_cuts: &[FitCutView<'_>], limit: usize) -> Result<Vec<SeedGenome>> {
        let digests: Vec<String> = fit_cuts.iter().map(FitCutView::fit_digest).collect();
        let mean_y: f64 =
            fit_cuts[0].train_y().iter().sum::<f64>() / (fit_cuts[0].train_rows().max(1) as f64);
        let first = if mean_y.is_sign_negative() {
            "(pipe X (robust_scale) (ridge))"
        } else {
            "(pipe X (standardize) (ridge))"
        };
        let candidates = [first, "(pipe X (rank) (ridge))"];
        let out: Vec<SeedGenome> = candidates
            .into_iter()
            .take(limit)
            .map(|g| SeedGenome {
                genome: g.to_owned(),
                program_sha256: sha256_hex(g.as_bytes()),
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect();
        *self.captured_digests.lock().unwrap() = digests;
        *self.captured_proposals.lock().unwrap() = out.clone();
        Ok(out)
    }
}

#[test]
fn run_with_none_provider_is_byte_identical_to_run() {
    let tmp = tempfile::tempdir().unwrap();
    let mk_opts = |dir: &std::path::Path| RunOptions {
        budget: Budget {
            evaluations: 12,
            seconds: 300,
        },
        rng_seed: 7,
        languages: vec![Language::Rust],
        policy: Policy::Auto,
        clock: Clock::Frozen,
        source_commit: Some("81dd03702a846d652f15a1f56480d8f074bddec4".into()),
        ..RunOptions::new(committed_dataset(), dir)
    };
    let out_a = tmp.path().join("run-plain");
    let out_b = tmp.path().join("run-none");
    let r_a = crate::run(&mk_opts(&out_a)).unwrap();
    let r_b = run_with_seed_provider(&mk_opts(&out_b), None).unwrap();
    assert_eq!(r_a, r_b);
    assert_eq!(
        std::fs::read(out_a.join("receipt.json")).unwrap(),
        std::fs::read(out_b.join("receipt.json")).unwrap()
    );
}

#[test]
fn provider_receives_only_fit_rows_and_is_invariant_to_holdout_poisoning() {
    let tmp = tempfile::tempdir().unwrap();
    let poisoned_ds = copy_dataset_with_poisoned_holdout(tmp.path(), crate::DEFAULT_SEED, 0);

    let prov_clean = RecordingFitProvider::new();
    let prov_poison = RecordingFitProvider::new();

    let out_clean = tmp.path().join("clean-run");
    let out_poison = tmp.path().join("poison-run");

    let base = |ds: String, out: &std::path::Path| RunOptions {
        budget: Budget {
            evaluations: 16,
            seconds: 300,
        },
        rng_seed: 11,
        cuts: 1,
        languages: vec![Language::Rust],
        policy: Policy::Require,
        clock: Clock::Frozen,
        source_commit: Some("81dd03702a846d652f15a1f56480d8f074bddec4".into()),
        ..RunOptions::new(ds, out)
    };

    let receipt_clean =
        run_with_seed_provider(&base(committed_dataset(), &out_clean), Some(&prov_clean)).unwrap();
    let receipt_poison =
        run_with_seed_provider(&base(poisoned_ds, &out_poison), Some(&prov_poison)).unwrap();

    // Verify full on-disk evidence re-derivation on the seeded run.
    evidence::verify_run(&out_clean.join("receipt.json")).unwrap();

    assert_eq!(
        *prov_clean.captured_digests.lock().unwrap(),
        *prov_poison.captured_digests.lock().unwrap()
    );
    assert_eq!(
        *prov_clean.captured_proposals.lock().unwrap(),
        *prov_poison.captured_proposals.lock().unwrap()
    );
    assert_eq!(
        std::fs::read(out_clean.join("prepared/provider-seeds.json")).unwrap(),
        std::fs::read(out_poison.join("prepared/provider-seeds.json")).unwrap()
    );
    let seeding_clean = receipt_clean.search.seeding.as_ref().unwrap();
    let seeding_poison = receipt_poison.search.seeding.as_ref().unwrap();
    assert_eq!(seeding_clean, seeding_poison);
    assert_eq!(seeding_clean.seeds.len(), 2);
    assert!(seeding_clean
        .seeds
        .iter()
        .all(|s| s.source == FROM_SEED_PROVIDER));
    assert!(receipt_clean
        .stages
        .iter()
        .any(|s| s.name == "propose" && s.status == "passed"));
}
