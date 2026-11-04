use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::commands::exact_cv::{self, ValidatedFreeze, FOLLOW_UP_GENERATIONS};
use crate::core::require_core_plan;
use crate::data::read_regression_csv;
use crate::{
    canonical_json, dataset_seed, elo_standings, hash_file, load_baselines, pairwise_results,
    score_dataset, sha256_bytes, validate_static_inputs, DATASETS, PREDICTION_FILE, REPORT_FILE,
};

const ORIGINAL_PANEL_ID: &str = "grok4p5-42";
const FOLLOW_UP_PANEL_ID: &str = "grok4p5-42-losses-g30";
const ORIGINAL_LEDGER_FILE: &str = "jope-dime-panel-results.v1.json";
const ORIGINAL_LEDGER_SHA256: &str =
    "90791906ae616dcd3d6f6e13e18e34d3efca977f4faa8e82d51a04f372c8da57";
const ORIGINAL_SELECTION_SLOTS: usize = 25_200;
const FOLLOW_UP_SELECTION_SLOTS: usize = 63_000;
const ACTIVE_PANEL_SELECTION_SLOTS: usize = 67_200;
const HISTORICAL_SELECTION_SLOTS: usize = 88_200;
const FOLLOW_UP_FINALISTS: usize = 350;

const LOSS_DATASETS: &[&str] = &[
    "019de5d76a718163",
    "047e62b52a3f4ede",
    "13b5df533e59a626",
    "1cb97b1d7314d43d",
    "21fbfe8b53dbcd60",
    "24713c7f7bfa7807",
    "30ef3eaa6d2971a9",
    "3305a3ad973fb227",
    "3baa7e37fc1f7861",
    "3f781a51af45387e",
    "43b4cc3ab0ebbd97",
    "4f68d277d770477f",
    "542e75bc0cfdd1bf",
    "55b144de19659467",
    "55f06d147bcc2a32",
    "5f91a00bd3b66742",
    "693890aae9dc860b",
    "79e600bc5fcd5da7",
    "860e5558f732f638",
    "ab0239b9ef0ba28e",
    "aebf5d861a8365b4",
    "b1f909fcfceeef4b",
    "b253fbfae8460dc1",
    "b5beff4a53fa12c0",
    "b78d323566930bd7",
    "c32fa09b48fb2f27",
    "c7d815714ca52bce",
    "c8f90fc9cc5364be",
    "d2daf53412ab18dc",
    "dee1fc8e9afd81c9",
    "e5a3a7b3668a1cc2",
    "ee56a1015e344c11",
    "f0e7eb3d790df37d",
    "f8536dfeef471c2f",
    "fe3cc716a05d036f",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TargetEntry {
    kind: String,
    dataset_hash: Option<String>,
    staged_file: String,
    archive_file: String,
    old_sha256: String,
    new_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct UntouchedEntry {
    kind: String,
    dataset_hash: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StageManifest {
    schema: String,
    follow_up_label: String,
    exploratory_reused_test: bool,
    original_panel_ledger_sha256: String,
    cohort: Vec<String>,
    targets: Vec<TargetEntry>,
    untouched_original_winners: Vec<UntouchedEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CompletionMarker {
    schema: String,
    status: String,
    manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ArchivePointer {
    schema: String,
    status: String,
    archive_manifest_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ArchiveManifest {
    schema: String,
    source_panel_ledger_sha256: String,
    entries: Vec<ArchiveEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ArchiveEntry {
    kind: String,
    dataset_hash: Option<String>,
    archive_file: String,
    sha256: String,
}

mod archive;
mod publication;
mod report;
mod source;
mod stage;

use archive::*;
use publication::*;
use report::*;
use source::*;
use stage::*;

pub(crate) fn validate(registry: &Path) -> Result<()> {
    let freezes = load_validated_follow_up(registry)?;
    println!(
        "validated {} generation-29 freezes, {FOLLOW_UP_SELECTION_SLOTS} evolutionary slots, {FOLLOW_UP_FINALISTS} complete CV7 finalist receipts, and 35 exact 600-slot replay prefixes without opening test data or baseline metrics",
        freezes.len()
    );
    Ok(())
}

pub(crate) fn summarize(registry: &Path) -> Result<()> {
    let stage = load_stage()?;
    audit_complete(registry, &stage)?;
    let ledger: Value = serde_json::from_slice(&fs::read(original_ledger_path())?)?;
    println!(
        "42 outcomes; exploratory follow-up {} new wins; strict overall {}W/{}L/{}T; TabFM {}W/{}L/{}T ({})",
        ledger["follow_up"]["new_wins_among_original_losses"],
        ledger["strict_overall"]["wins"],
        ledger["strict_overall"]["losses"],
        ledger["strict_overall"]["ties"],
        ledger["tabfm_head_to_head"]["wins"],
        ledger["tabfm_head_to_head"]["losses"],
        ledger["tabfm_head_to_head"]["ties"],
        ledger["tabfm_head_to_head"]["status"]
            .as_str()
            .unwrap_or("unknown"),
    );
    Ok(())
}

pub(crate) fn run(registry: &Path) -> Result<()> {
    validate_static_inputs()?;
    validate_source_hash_only()?;
    fs::create_dir_all(follow_up_root())?;
    let mut core_plan = None;
    let started = Instant::now();
    for (index, hash) in LOSS_DATASETS.iter().enumerate() {
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let artifact = follow_up_root().join(hash);
        if artifact.join("SELECTION_COMPLETE").exists() {
            exact_cv::validate_freeze(
                &artifact,
                hash,
                &train_sha256,
                train.y.len(),
                dataset_seed(hash)?,
                FOLLOW_UP_GENERATIONS,
            )
            .with_context(|| format!("existing follow-up freeze is invalid for {hash}"))?;
            exact_cv::validate_original_prefix(&original_root().join(hash), &artifact)?;
            continue;
        }
        recover_incomplete_dataset(&artifact, hash)?;
        let cores = match &core_plan {
            Some(cores) => cores,
            None => core_plan.insert(require_core_plan(60)?),
        };
        eprintln!(
            "[jope-dime] 30-generation follow-up {hash} ({}/{})",
            index + 1,
            LOSS_DATASETS.len()
        );
        exact_cv::select_dataset(
            &train,
            hash,
            &train_sha256,
            dataset_seed(hash)?,
            crate::seed_bytes(),
            &artifact,
            cores.clone(),
            FOLLOW_UP_GENERATIONS,
        )?;
        exact_cv::validate_freeze(
            &artifact,
            hash,
            &train_sha256,
            train.y.len(),
            dataset_seed(hash)?,
            FOLLOW_UP_GENERATIONS,
        )?;
        exact_cv::validate_original_prefix(&original_root().join(hash), &artifact)?;
    }
    eprintln!(
        "[jope-dime] follow-up selection pass complete in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    let freezes = load_validated_follow_up(registry)?;
    publish_follow_up(registry, &freezes)
}

#[cfg(test)]
mod tests;
