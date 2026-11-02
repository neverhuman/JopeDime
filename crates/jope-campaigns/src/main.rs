use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::commands::exact_cv::{self, ValidatedFreeze, ORIGINAL_GENERATIONS};
use crate::core::require_core_plan;
use crate::data::{read_binary_csv, read_regression_csv};
use jope_dime_regressor::paths;

mod aggressive;
mod args;
mod commands;
mod core;
mod data;
// frozen, hash-pinned module; bytes immutable until unfrozen
mod binary_panel;
#[allow(dead_code, clippy::too_many_arguments, clippy::let_and_return)]
mod campaign_evaluator;
mod cli;
mod followup;
mod genome_boost;
mod golden42_panel;
mod main_app;
mod ratchet;
mod regression_panel;
mod runtime;
mod score_corpus;
mod test_adaptive;

use binary_panel::*;
use cli::{CampaignCommand, Cli, KernelVersion};
use regression_panel::*;

jope_paths::once_str!(
    fn registry_str => paths::jope_registry()
);
const PANEL_SHA256: &str = "b86c8ea5f5ab1051e8aafb4368f8dabdcfd8c740c63eaa1d086ddb72fae0ba75";
const PANEL_ANALYSIS_SHA256: &str =
    "0117abb32d547b134706e210376f2e6f437b706b2a12edb6d9302d74d4749282";
const SEED_SHA256: &str = "5f097a0017ddc7e92ad5ea8dfadaf783b548e82287d92626dc9d2fa9429b2105";
const BASE_SEED: u64 = 20260616;
// --- binary-classification lane (v2-binary; fixture-backed until real datasets land) ---
const PANEL_BINARY_SHA256: &str =
    "be3e8ff9654e31bbb1015a1e783bb9ba58f86295706dbdbaf137509e0b69d575";
const PANEL_ANALYSIS_BINARY_SHA256: &str =
    "34f476c21e055d58033eb9278247d5ddfdf9c7c65e4660f4e7372f8345b26da6";
const MODEL_BINARY: &str = "jope-dime-binary";
const BINARY_REPORT_FILE: &str = "jope-dime-binary-report.v1.json";
const BINARY_PREDICTION_FILE: &str = "jope-dime-binary-predictions.csv";
const BINARY_RESULTS_FILE: &str = "jope-dime-binary-panel-results.v1.json";
/// Lane tag ("jpbin_v1" as bytes) XORed into the seed derivation so binary
/// campaigns never share offspring streams with the regression lane.
const BINARY_LANE_TAG: u64 = 0x6a70_6269_6e5f_7631;
const MODEL: &str = "jope-dime";
const REPORT_FILE: &str = "jope-dime-regressor.json";
const PREDICTION_FILE: &str = "jope-dime-regressor.csv";
const BASELINES: &[&str] = &[
    "TabFM ensemble",
    "TabPFNv3",
    "Fable5",
    "Grok4p5",
    "OpenAI GPT-5.6-sol",
    "Kimik3",
];
const DATASETS: &[&str] = &[
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "17a4542ee4fb9e44",
    "1a803b27c2400dc4",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "471f11cc2d38853a",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "7af85a04de0c7018",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "bd49e740a79bcb6c",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "cf5e74b4e9594944",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e1c1db5260d0894e",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[derive(Debug)]
struct ScoredDataset {
    dataset_hash: String,
    predictions: Vec<f64>,
    prediction_bytes: Vec<u8>,
    prediction_sha256: String,
    report_bytes: Vec<u8>,
    report_sha256: String,
    scores: BTreeMap<String, f64>,
    outcome: Value,
}

fn main() -> Result<()> {
    main_app::main()
}

fn command_starts_campaign(command: &Command) -> bool {
    matches!(
        command,
        CampaignCommand::SelectPanel
            | CampaignCommand::RunPanel
            | CampaignCommand::PrepareTrainOnlyStudy { .. }
            | CampaignCommand::RunTrainOnlyStudy { .. }
            | CampaignCommand::PrepareTrainOnlyRedteam { .. }
            | CampaignCommand::RunTrainOnlyRedteam { .. }
            | CampaignCommand::RunLosses3x
            | CampaignCommand::RunTimeseriesTestAdaptiveAll { .. }
            | CampaignCommand::RunTimeseriesTestAdaptiveGen30 { .. }
            | CampaignCommand::TimeseriesTestAdaptiveBatchWorker { .. }
            | CampaignCommand::RunTestAdaptive { .. }
            | CampaignCommand::RunTestAdaptivePanel { .. }
            | CampaignCommand::RunTestAdaptiveExtension { .. }
            | CampaignCommand::RunTestAdaptiveIslandExtension { .. }
            | CampaignCommand::RunGolden42Champions { .. }
            | CampaignCommand::RunTabarenaLiteIslands { .. }
            | CampaignCommand::TestAdaptiveWorker { .. }
            | CampaignCommand::TestAdaptiveIslandBatchWorker { .. }
            | CampaignCommand::RunGolden42Aggressive { .. }
            | CampaignCommand::RunGolden46 { .. }
            | CampaignCommand::Golden42AggressiveBatchWorker { .. }
    )
}

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("manifests")
}

fn run_root(panel_id: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("runs")
        .join(panel_id)
}

fn seed_bytes() -> &'static [u8] {
    include_bytes!("../seeds/generation-zero.txt")
}

fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> Result<String> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn dataset_seed(hash: &str) -> Result<u64> {
    Ok(BASE_SEED ^ u64::from_str_radix(hash, 16).context("dataset hash is not hexadecimal")?)
}

/// Binary-lane seed derivation: BASE_SEED ^ lane tag ^ dataset hash, so the
/// binary offspring streams never coincide with the regression lane's.

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn binary_elo_is_deterministic_and_rewards_strictly_greater_auc() {
        let mut dataset = BTreeMap::new();
        dataset.insert("A".to_string(), 0.9);
        dataset.insert("B".to_string(), 0.7);
        let ratings = binary_elo(&[dataset.clone()]);
        // One game: A expected 0.5, wins -> +8; B symmetric -> -8.
        assert_eq!(ratings["A"], 1508.0);
        assert_eq!(ratings["B"], 1492.0);
        // Exact tie moves nothing at equal ratings.
        let mut tie = BTreeMap::new();
        tie.insert("A".to_string(), 0.8);
        tie.insert("B".to_string(), 0.8);
        let tie_ratings = binary_elo(&[tie]);
        assert_eq!(tie_ratings["A"], 1500.0);
        assert_eq!(tie_ratings["B"], 1500.0);
        // Determinism across repeated evaluation.
        assert_eq!(binary_elo(&[dataset.clone()]), binary_elo(&[dataset]));
    }

    #[test]
    fn binary_static_inputs_are_pinned() {
        assert_eq!(
            sha256_bytes(binary_seed_bytes()),
            commands::binary_cv::SEED_BINARY_SHA256
        );
        let panel = binary_panel().unwrap();
        assert_eq!(panel, vec!["b1a0f1c500000001", "b1a0f1c500000002"]);
        let baselines = binary_baselines().unwrap();
        assert_eq!(baselines.len(), 2);
        assert!(baselines["b1a0f1c500000001"]["BaselineBoosted"] > 0.8);
    }

    #[test]
    fn strict_relation_has_no_tolerance() {
        assert_eq!(relation(0.9, 1.0), "win");
        assert_eq!(relation(1.0, 1.0), "tie");
        assert_eq!(relation(1.0 + f64::EPSILON, 1.0), "loss");
    }

    #[test]
    fn baseline_comparison_covers_all_six_models() {
        let baselines = BASELINES
            .iter()
            .enumerate()
            .map(|(index, model)| ((*model).to_string(), 1.0 + index as f64))
            .collect::<BTreeMap<_, _>>();
        let (comparisons, best, models, overall, tabfm) =
            baseline_comparisons(0.5, &baselines).unwrap();
        assert_eq!(comparisons.as_object().unwrap().len(), 6);
        assert_eq!(best, 1.0);
        assert_eq!(models, vec!["TabFM ensemble"]);
        assert_eq!(overall, "win");
        assert_eq!(tabfm, "win");
    }

    #[test]
    fn immutable_publication_refuses_overwrite() {
        let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "jope-dime-publication-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("report.json");
        publish_new(&path, b"first\n").unwrap();
        assert!(publish_new(&path, b"second\n").is_err());
        assert_eq!(fs::read(&path).unwrap(), b"first\n");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn timeseries_gen30_cli_parses_base_and_run_ids() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-gen30",
            "--base-run-id",
            "base",
            "--run-id",
            "next",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            CampaignCommand::RunTimeseriesTestAdaptiveGen30 {
                base_run_id,
                run_id,
                resume,
            } => {
                assert_eq!(base_run_id, "base");
                assert_eq!(run_id, "next");
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }

    #[test]
    fn timeseries_winall_cli_parses_source_run_id_and_generation_limit() {
        let cli = Cli::try_parse_from([
            "jope-dime-regressor",
            "run-timeseries-test-adaptive-winall",
            "--source-run-id",
            "gen30",
            "--run-id",
            "winall",
            "--max-generations",
            "200",
            "--lineage-salt",
            "salt-a",
            "--resume",
        ])
        .unwrap();
        match cli.command {
            CampaignCommand::RunTimeseriesTestAdaptiveWinall {
                source_run_id,
                run_id,
                max_generations,
                lineage_salt,
                resume,
            } => {
                assert_eq!(source_run_id, "gen30");
                assert_eq!(run_id, "winall");
                assert_eq!(max_generations, Some(200));
                assert_eq!(lineage_salt.as_deref(), Some("salt-a"));
                assert!(resume);
            }
            _ => panic!("wrong command variant"),
        }
    }
}
