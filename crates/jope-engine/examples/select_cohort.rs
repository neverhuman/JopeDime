//! Frozen, score-blind qualification family selection and context pointer subsets.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_parallel::{
    families::Registry,
    manifest::*,
    prepare::Prepared,
    service::{GpJob, GpJobs},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Freeze {
        #[arg(long)]
        registry: PathBuf,
        #[arg(long, required = true)]
        jobs: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    Finalize {
        #[arg(long)]
        selection: PathBuf,
        #[arg(long)]
        contexts: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Binding {
    path: PathBuf,
    sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Candidate {
    dataset: String,
    family: String,
    partition: String,
    source_key: String,
    metadata_sha256: String,
    source_identity_sha256: String,
    rows: usize,
    features: usize,
    eligible: bool,
    holds: Vec<String>,
    prepared_versions: Vec<Binding>,
}
#[derive(Serialize, Deserialize)]
struct Selection {
    schema: String,
    seed: u64,
    qualification_only: bool,
    independent_selection_eligible: bool,
    registry: Binding,
    job_manifests: Vec<Binding>,
    vocabulary: Binding,
    method: String,
    projection_limit: String,
    quotas: BTreeMap<String, usize>,
    candidates: Vec<Candidate>,
    selected: Vec<Candidate>,
    original_final_test_targets_opened: bool,
    model_scores_consulted: bool,
}

#[path = "select_cohort/finalize.rs"]
mod finalize;
#[path = "select_cohort/freeze.rs"]
mod freeze;
#[path = "select_cohort/selection.rs"]
mod selection;
use finalize::finalize;
use freeze::{binding, freeze};
use selection::{quotas, select};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Freeze {
            registry,
            jobs,
            output,
        } => freeze(&cli.root, &registry, &jobs, &output),
        Command::Finalize {
            selection,
            contexts,
            output,
        } => finalize(&cli.root, &selection, &contexts, &output),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn candidates() -> Vec<Candidate> {
        let mut result = Vec::new();
        for (partition, n) in [("meta_train", 55), ("meta_validation", 9), ("meta_test", 8)] {
            for i in 0..n {
                result.push(Candidate {
                    dataset: format!("{partition}-{i}"),
                    family: format!("{partition}-family-{i}"),
                    partition: partition.into(),
                    source_key: format!("source-{partition}-{i}"),
                    metadata_sha256: "0".repeat(64),
                    source_identity_sha256: "1".repeat(64),
                    rows: 64 << (i % 6),
                    features: 1 << (i % 7),
                    eligible: true,
                    holds: vec![],
                    prepared_versions: vec![],
                });
            }
        }
        result
    }
    #[test]
    fn stable_under_input_permutations_and_one_dataset_per_family() {
        let mut input = candidates();
        let mut alias = input[0].clone();
        alias.dataset = "alternate-target".into();
        input.push(alias);
        let a = select(&input).unwrap();
        input.reverse();
        input.rotate_left(13);
        assert_eq!(a, select(&input).unwrap());
        assert_eq!(a.len(), 64);
        assert_eq!(
            a.iter().map(|c| &c.family).collect::<BTreeSet<_>>().len(),
            64
        );
        for (partition, n) in quotas() {
            assert_eq!(a.iter().filter(|c| c.partition == partition).count(), n);
        }
    }
    #[test]
    fn holds_and_insufficient_partition_quotas_are_rejected() {
        let mut input = candidates();
        input[0].holds.push("unresolved_alias".into());
        assert!(select(&input).is_err());
        input[0].holds.clear();
        input.retain(|c| c.dataset != "meta_test-0");
        assert!(select(&input)
            .unwrap_err()
            .to_string()
            .contains("insufficient meta_test"));
    }
    #[test]
    fn family_partition_collisions_are_rejected() {
        let mut input = candidates();
        input[55].family = input[0].family.clone();
        assert!(select(&input)
            .unwrap_err()
            .to_string()
            .contains("crosses partitions"));
    }
}
