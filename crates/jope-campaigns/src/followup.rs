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

fn manifest_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("runs")
}

fn original_root() -> PathBuf {
    manifest_dir().join(ORIGINAL_PANEL_ID)
}

fn follow_up_root() -> PathBuf {
    manifest_dir().join(FOLLOW_UP_PANEL_ID)
}

fn replacement_root() -> PathBuf {
    follow_up_root().join("replacement")
}

fn original_ledger_path() -> PathBuf {
    original_root().join(ORIGINAL_LEDGER_FILE)
}

fn public_target(registry: &Path, entry: &TargetEntry) -> Result<PathBuf> {
    match (entry.kind.as_str(), entry.dataset_hash.as_deref()) {
        ("prediction", Some(hash)) if LOSS_DATASETS.contains(&hash) => Ok(registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE)),
        ("report", Some(hash)) if LOSS_DATASETS.contains(&hash) => {
            Ok(registry.join(hash).join("reports").join(REPORT_FILE))
        }
        ("panel_ledger", None) => Ok(original_ledger_path()),
        _ => bail!("replacement manifest contains a target outside the allowlist"),
    }
}

fn untouched_target(registry: &Path, entry: &UntouchedEntry) -> Result<PathBuf> {
    if LOSS_DATASETS.contains(&entry.dataset_hash.as_str())
        || !DATASETS.contains(&entry.dataset_hash.as_str())
    {
        bail!("untouched-winner manifest contains an invalid dataset");
    }
    match entry.kind.as_str() {
        "prediction" => Ok(registry
            .join(&entry.dataset_hash)
            .join("predictions")
            .join(PREDICTION_FILE)),
        "report" => Ok(registry
            .join(&entry.dataset_hash)
            .join("reports")
            .join(REPORT_FILE)),
        _ => bail!("untouched-winner manifest contains an invalid target kind"),
    }
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} is not a lowercase SHA-256 digest");
    }
    if value != value.to_ascii_lowercase() {
        bail!("{label} is not a lowercase SHA-256 digest");
    }
    Ok(())
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("artifact path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("replacement path has no parent")?;
    fs::create_dir_all(parent)?;
    let staged = parent.join(format!(
        ".jope-dime-replace-{}-{}",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    if staged.exists() {
        bail!(
            "stale replacement staging file exists: {}",
            staged.display()
        );
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(&staged)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&staged, permissions)?;
    fs::rename(&staged, path).with_context(|| format!("replace {}", path.display()))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn replacement_complete_path() -> PathBuf {
    replacement_root().join("OFFICIAL_REPLACEMENT_COMPLETE")
}

fn authorized_source_ledger_bytes() -> Result<Vec<u8>> {
    let current = fs::read(original_ledger_path()).context("read current panel ledger")?;
    if sha256_bytes(&current) == ORIGINAL_LEDGER_SHA256 {
        return Ok(current);
    }
    let pointer_path = replacement_root().join("ARCHIVE_COMPLETE");
    let pointer: ArchivePointer = serde_json::from_slice(
        &fs::read(&pointer_path)
            .context("current panel ledger drifted before a complete archive")?,
    )?;
    validate_archive_pointer(&pointer)?;
    let archive = replacement_root()
        .join("archives")
        .join(&pointer.archive_manifest_sha256);
    let bytes = fs::read(archive.join("payload/panel-ledger.json"))?;
    if sha256_bytes(&bytes) != ORIGINAL_LEDGER_SHA256 {
        bail!("archived source panel ledger differs from the pinned SHA-256");
    }
    let stage = load_stage()?;
    let ledger = stage
        .targets
        .iter()
        .find(|entry| entry.kind == "panel_ledger")
        .context("stage manifest has no panel ledger")?;
    if sha256_bytes(&current) != ledger.new_sha256 {
        bail!("current panel ledger is neither the archived old hash nor staged new hash");
    }
    Ok(bytes)
}

fn validate_source_hash_only() -> Result<()> {
    if sha256_bytes(&authorized_source_ledger_bytes()?) != ORIGINAL_LEDGER_SHA256 {
        bail!("source panel ledger SHA-256 drift");
    }
    Ok(())
}

fn validate_compiled_cohort(source_ledger: &[u8]) -> Result<Value> {
    let value: Value = serde_json::from_slice(source_ledger)?;
    if value["schema"] != "JopeDimePanelResults.v1"
        || value["dataset_count"].as_u64() != Some(42)
        || value["outcomes"].as_array().map(Vec::len) != Some(42)
    {
        bail!("pinned source ledger is not the original complete panel");
    }
    let losses = value["outcomes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["strict_overall_outcome"] == "loss")
        .map(|row| {
            row["dataset_hash"]
                .as_str()
                .context("source outcome has no dataset hash")
        })
        .collect::<Result<Vec<_>>>()?;
    if losses != LOSS_DATASETS {
        bail!("compiled 35-dataset cohort differs from the pinned source ledger");
    }
    Ok(value)
}

fn directory_digest(root: &Path) -> Result<String> {
    fn visit(root: &Path, current: &Path, rows: &mut Vec<String>) -> Result<()> {
        let mut entries = fs::read_dir(current)?.collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                visit(root, &path, rows)?;
            } else if path.is_file() {
                rows.push(format!(
                    "{} {}",
                    path.strip_prefix(root)?.display(),
                    hash_file(&path)?
                ));
            }
        }
        Ok(())
    }
    let mut rows = Vec::new();
    visit(root, root, &mut rows)?;
    Ok(sha256_bytes(format!("{}\n", rows.join("\n")).as_bytes()))
}

fn recover_incomplete_dataset(root: &Path, hash: &str) -> Result<()> {
    if !root.exists() || fs::read_dir(root)?.next().transpose()?.is_none() {
        return Ok(());
    }
    let digest = directory_digest(root)?;
    let recovery = follow_up_root().join("recovery");
    fs::create_dir_all(&recovery)?;
    let destination = recovery.join(format!("{hash}-{}", &digest[..16]));
    if destination.exists() {
        bail!(
            "incomplete recovery destination already exists: {}",
            destination.display()
        );
    }
    fs::rename(root, &destination)?;
    File::open(&recovery)?.sync_all()?;
    eprintln!(
        "[jope-dime] preserved incomplete {hash} attempt at {}",
        destination.display()
    );
    Ok(())
}

fn load_validated_follow_up(registry: &Path) -> Result<BTreeMap<String, ValidatedFreeze>> {
    validate_static_inputs()?;
    validate_source_hash_only()?;
    let mut freezes = BTreeMap::new();
    let mut semantics = BTreeSet::new();
    let mut slots = 0usize;
    let mut finalists = 0usize;
    for hash in LOSS_DATASETS {
        let train_path = registry.join(hash).join("train.csv");
        let train = read_regression_csv(&train_path)?;
        let train_sha256 = hash_file(&train_path)?;
        let artifact = follow_up_root().join(hash);
        let validated = exact_cv::validate_freeze(
            &artifact,
            hash,
            &train_sha256,
            train.y.len(),
            dataset_seed(hash)?,
            FOLLOW_UP_GENERATIONS,
        )
        .with_context(|| format!("30-generation freeze validation failed for {hash}"))?;
        exact_cv::validate_original_prefix(&original_root().join(hash), &artifact)
            .with_context(|| format!("10-generation replay prefix mismatch for {hash}"))?;
        if !semantics.insert(validated.semantic_sha256.clone()) {
            bail!("follow-up selected semantic genome is reused across datasets");
        }
        slots += validated.evolutionary_slots;
        finalists += validated.finalist_receipts;
        freezes.insert((*hash).to_string(), validated);
    }
    if freezes.len() != LOSS_DATASETS.len()
        || semantics.len() != LOSS_DATASETS.len()
        || slots != FOLLOW_UP_SELECTION_SLOTS
        || finalists != FOLLOW_UP_FINALISTS
    {
        bail!(
            "follow-up accounting failed: freezes={} distinct={} slots={slots} finalists={finalists}",
            freezes.len(),
            semantics.len()
        );
    }
    Ok(freezes)
}

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

fn source_outcomes(source: &Value) -> Result<BTreeMap<String, Value>> {
    source["outcomes"]
        .as_array()
        .context("source ledger outcomes are missing")?
        .iter()
        .map(|row| {
            let hash = row["dataset_hash"]
                .as_str()
                .context("source outcome has no dataset hash")?;
            Ok((hash.to_string(), row.clone()))
        })
        .collect()
}

fn original_report(registry: &Path, hash: &str) -> Result<Value> {
    let path = registry.join(hash).join("reports").join(REPORT_FILE);
    let report: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if report["schema"] != "JopeDimeDatasetReport.v1"
        || report["dataset_hash"].as_str() != Some(hash)
        || report["provenance"]["evolutionary_slots"].as_u64() != Some(600)
        || report["provenance"]["cv7_finalist_receipts"].as_u64() != Some(10)
        || report["comparisons"]["baselines"]
            .as_object()
            .map(|value| value.len())
            != Some(6)
    {
        bail!("original published report is invalid for {hash}");
    }
    Ok(report)
}

fn follow_up_comparison(old: &Value, new: &ValidatedFreeze, new_test_rmse: f64) -> Result<Value> {
    let old_genome = old["algorithm"]["canonical_genome"]
        .as_str()
        .context("old report genome is missing")?;
    let old_semantic = old["algorithm"]["semantic_sha256"]
        .as_str()
        .context("old report semantic SHA-256 is missing")?;
    let old_cv7_fitness = old["selection_proxy"]["full_cv7"]["robust_fitness"]
        .as_f64()
        .context("old report CV7 fitness is missing")?;
    let old_cv7_rmse = old["selection_proxy"]["full_cv7"]["pooled_oof_rmse"]
        .as_f64()
        .context("old report CV7 RMSE is missing")?;
    let old_test_rmse = old["metrics"]["rmse"]
        .as_f64()
        .context("old report test RMSE is missing")?;
    Ok(json!({
        "label": "exploratory, test-adaptive follow-up on the 35 original strict losses",
        "exploratory_reused_test": true,
        "generation_10": {
            "generation": 9,
            "canonical_genome": old_genome,
            "semantic_sha256": old_semantic,
            "cv7_robust_fitness": old_cv7_fitness,
            "cv7_pooled_oof_rmse": old_cv7_rmse,
            "test_rmse": old_test_rmse,
        },
        "generation_30": {
            "generation": new.generation,
            "canonical_genome": new.canonical_genome,
            "semantic_sha256": new.semantic_sha256,
            "cv7_robust_fitness": new.cv7_robust_fitness,
            "cv7_pooled_oof_rmse": new.cv7_pooled_oof_rmse,
            "test_rmse": new_test_rmse,
        },
        "deltas_30_minus_10": {
            "cv7_robust_fitness": new.cv7_robust_fitness - old_cv7_fitness,
            "cv7_pooled_oof_rmse": new.cv7_pooled_oof_rmse - old_cv7_rmse,
            "test_rmse": new_test_rmse - old_test_rmse,
        },
        "selected_genome_changed": old_semantic != new.semantic_sha256,
    }))
}

fn build_v2_report(
    old: &Value,
    new_freeze: &ValidatedFreeze,
    mut generated: Value,
) -> Result<Vec<u8>> {
    let new_test_rmse = generated["metrics"]["rmse"]
        .as_f64()
        .context("generated test RMSE is missing")?;
    generated["schema"] = json!("JopeDimeDatasetReport.v2");
    generated["follow_up"] = follow_up_comparison(old, new_freeze, new_test_rmse)?;
    generated["provenance"]["exploratory_reused_test"] = json!(true);
    generated["provenance"]["source_panel_ledger_sha256"] = json!(ORIGINAL_LEDGER_SHA256);
    generated["provenance"]["follow_up_panel_id"] = json!(FOLLOW_UP_PANEL_ID);
    generated["provenance"]["historical_evolutionary_slots_for_dataset"] = json!(2_400);
    canonical_json(&generated)
}

fn stage_manifest_path() -> PathBuf {
    replacement_root().join("staged/manifest.v1.json")
}

fn stage_marker_path() -> PathBuf {
    replacement_root().join("staged/STAGING_COMPLETE")
}

fn load_stage() -> Result<StageManifest> {
    let marker: CompletionMarker = serde_json::from_slice(&fs::read(stage_marker_path())?)?;
    if marker.schema != "JopeDimeFollowUpStagingComplete.v1" || marker.status != "COMPLETE" {
        bail!("follow-up staging marker is invalid");
    }
    validate_sha256("staging manifest SHA-256", &marker.manifest_sha256)?;
    let bytes = fs::read(stage_manifest_path())?;
    if sha256_bytes(&bytes) != marker.manifest_sha256 {
        bail!("follow-up staging manifest differs from its marker hash");
    }
    let manifest: StageManifest = serde_json::from_slice(&bytes)?;
    validate_stage_manifest(&manifest)?;
    for entry in &manifest.targets {
        let path = replacement_root().join("staged").join(&entry.staged_file);
        if hash_file(&path)? != entry.new_sha256 {
            bail!("staged target hash mismatch for {}", entry.staged_file);
        }
    }
    Ok(manifest)
}

fn validate_stage_manifest(manifest: &StageManifest) -> Result<()> {
    if manifest.schema != "JopeDimeFollowUpStaging.v1"
        || !manifest.exploratory_reused_test
        || manifest.original_panel_ledger_sha256 != ORIGINAL_LEDGER_SHA256
        || manifest.cohort != LOSS_DATASETS
        || manifest.targets.len() != 71
        || manifest.untouched_original_winners.len() != 14
    {
        bail!("follow-up staging manifest has invalid accounting or provenance");
    }
    let mut keys = BTreeSet::new();
    for entry in &manifest.targets {
        validate_sha256("old target SHA-256", &entry.old_sha256)?;
        validate_sha256("new target SHA-256", &entry.new_sha256)?;
        if !keys.insert((entry.kind.clone(), entry.dataset_hash.clone())) {
            bail!("follow-up staging manifest has duplicate targets");
        }
    }
    if manifest
        .targets
        .iter()
        .filter(|entry| entry.kind == "prediction")
        .count()
        != 35
        || manifest
            .targets
            .iter()
            .filter(|entry| entry.kind == "report")
            .count()
            != 35
        || manifest
            .targets
            .iter()
            .filter(|entry| entry.kind == "panel_ledger")
            .count()
            != 1
    {
        bail!("follow-up staging manifest target classes are invalid");
    }
    Ok(())
}

fn create_stage(
    registry: &Path,
    freezes: &BTreeMap<String, ValidatedFreeze>,
    source: &Value,
) -> Result<StageManifest> {
    if stage_marker_path().exists() {
        return load_stage();
    }
    if replacement_root().join("staged").exists() {
        bail!("incomplete staged publication requires operator recovery");
    }
    let source_rows = source_outcomes(source)?;
    let baselines = load_baselines()?;
    let incoming = replacement_root().join(format!("staged.incoming-{}", std::process::id()));
    if incoming.exists() {
        bail!(
            "stale incoming publication directory exists: {}",
            incoming.display()
        );
    }
    fs::create_dir_all(&incoming)?;

    let mut targets = Vec::new();
    let mut follow_up_rows = BTreeMap::<String, crate::ScoredDataset>::new();
    for hash in LOSS_DATASETS {
        let old = original_report(registry, hash)?;
        let old_report_path = registry.join(hash).join("reports").join(REPORT_FILE);
        let old_prediction_path = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let old_report_sha256 = hash_file(&old_report_path)?;
        let old_prediction_sha256 = hash_file(&old_prediction_path)?;
        if source_rows[*hash]["report_sha256"].as_str() != Some(old_report_sha256.as_str())
            || source_rows[*hash]["prediction_sha256"].as_str()
                != Some(old_prediction_sha256.as_str())
        {
            bail!("published v1 artifacts drifted from the pinned source ledger for {hash}");
        }
        let mut row = score_dataset(registry, hash, &freezes[*hash], &baselines[*hash])?;
        let generated: Value = serde_json::from_slice(&row.report_bytes)?;
        row.report_bytes = build_v2_report(&old, &freezes[*hash], generated)?;
        row.report_sha256 = sha256_bytes(&row.report_bytes);
        row.outcome["report_sha256"] = json!(row.report_sha256);
        row.outcome["follow_up"] = follow_up_comparison(
            &old,
            &freezes[*hash],
            row.outcome["test_metrics"]["rmse"]
                .as_f64()
                .context("new outcome test RMSE is missing")?,
        )?;
        row.outcome["original_strict_overall_outcome"] =
            source_rows[*hash]["strict_overall_outcome"].clone();
        row.outcome["new_win_among_original_losses"] =
            json!(row.outcome["strict_overall_outcome"] == "win");

        let prediction_file = format!("predictions/{hash}.csv");
        immutable_write(&incoming.join(&prediction_file), &row.prediction_bytes)?;
        targets.push(TargetEntry {
            kind: "prediction".to_string(),
            dataset_hash: Some((*hash).to_string()),
            staged_file: prediction_file,
            archive_file: format!("payload/predictions/{hash}.csv"),
            old_sha256: old_prediction_sha256,
            new_sha256: row.prediction_sha256.clone(),
        });
        let report_file = format!("reports/{hash}.json");
        immutable_write(&incoming.join(&report_file), &row.report_bytes)?;
        targets.push(TargetEntry {
            kind: "report".to_string(),
            dataset_hash: Some((*hash).to_string()),
            staged_file: report_file,
            archive_file: format!("payload/reports/{hash}.json"),
            old_sha256: old_report_sha256,
            new_sha256: row.report_sha256.clone(),
        });
        follow_up_rows.insert((*hash).to_string(), row);
    }

    let mut all_scored = Vec::with_capacity(DATASETS.len());
    let mut outcomes = Vec::with_capacity(DATASETS.len());
    for hash in DATASETS {
        if let Some(row) = follow_up_rows.remove(*hash) {
            outcomes.push(row.outcome.clone());
            all_scored.push(row);
        } else {
            let mut outcome = source_rows[*hash].clone();
            outcome["follow_up_status"] = json!("original_winner_untouched");
            let scores = outcome["rmse_by_algorithm"]
                .as_object()
                .context("source outcome RMSE map is missing")?
                .iter()
                .map(|(name, score)| {
                    Ok((
                        name.clone(),
                        score.as_f64().context("source RMSE is not finite")?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            outcomes.push(outcome.clone());
            all_scored.push(crate::ScoredDataset {
                dataset_hash: (*hash).to_string(),
                predictions: Vec::new(),
                prediction_bytes: Vec::new(),
                prediction_sha256: outcome["prediction_sha256"]
                    .as_str()
                    .context("source prediction hash is missing")?
                    .to_string(),
                report_bytes: Vec::new(),
                report_sha256: outcome["report_sha256"]
                    .as_str()
                    .context("source report hash is missing")?
                    .to_string(),
                scores,
                outcome,
            });
        }
    }
    if all_scored.len() != 42 || outcomes.len() != 42 {
        bail!("updated panel does not contain exactly 42 outcomes");
    }
    let wins = all_scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let losses = all_scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "loss")
        .count();
    let ties = 42 - wins - losses;
    let tabfm_wins = all_scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "win")
        .count();
    let tabfm_losses = all_scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "loss")
        .count();
    let tabfm_ties = 42 - tabfm_wins - tabfm_losses;
    let tabfm_status = match tabfm_wins.cmp(&tabfm_losses) {
        Ordering::Greater => "winning",
        Ordering::Equal => "tied",
        Ordering::Less => "losing",
    };
    let new_wins = all_scored
        .iter()
        .filter(|row| LOSS_DATASETS.contains(&row.dataset_hash.as_str()))
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let ledger = json!({
        "schema": "JopeDimePanelResults.v2",
        "panel_id": ORIGINAL_PANEL_ID,
        "follow_up_panel_id": FOLLOW_UP_PANEL_ID,
        "dataset_count": 42,
        "follow_up": {
            "label": "exploratory, test-adaptive follow-up on the 35 original strict losses",
            "exploratory_reused_test": true,
            "source_panel_ledger_sha256": ORIGINAL_LEDGER_SHA256,
            "cohort_size": 35,
            "new_wins_among_original_losses": new_wins,
            "all_35_follow_up_results_published_without_cherry_picking": true,
        },
        "panel_text_sha256": crate::PANEL_SHA256,
        "baseline_panel_sha256": crate::PANEL_ANALYSIS_SHA256,
        "selection_accounting": {
            "follow_up_evolutionary_slots": FOLLOW_UP_SELECTION_SLOTS,
            "updated_active_panel_evolutionary_slots": ACTIVE_PANEL_SELECTION_SLOTS,
            "historical_evolutionary_slots_across_both_experiments": HISTORICAL_SELECTION_SLOTS,
            "original_experiment_evolutionary_slots": ORIGINAL_SELECTION_SLOTS,
            "follow_up_generation_29_freezes": 35,
            "follow_up_full_cv7_finalist_receipts": FOLLOW_UP_FINALISTS,
            "untouched_original_winner_freezes": 7,
        },
        "strict_overall": {
            "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "tabfm_head_to_head": {
            "wins": tabfm_wins,
            "losses": tabfm_losses,
            "ties": tabfm_ties,
            "status": tabfm_status,
        },
        "pairwise_results": pairwise_results(&all_scored),
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 16.0,
            "method": "Each dataset is a simultaneous 7-algorithm round robin; lower RMSE wins and exact equality ties; panel order is pinned.",
            "standings": elo_standings(&all_scored),
        },
        "outcomes": outcomes,
    });
    let ledger_bytes = canonical_json(&ledger)?;
    immutable_write(&incoming.join("panel-ledger.json"), &ledger_bytes)?;
    targets.push(TargetEntry {
        kind: "panel_ledger".to_string(),
        dataset_hash: None,
        staged_file: "panel-ledger.json".to_string(),
        archive_file: "payload/panel-ledger.json".to_string(),
        old_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        new_sha256: sha256_bytes(&ledger_bytes),
    });

    let mut untouched = Vec::new();
    for hash in DATASETS
        .iter()
        .copied()
        .filter(|hash| !LOSS_DATASETS.contains(hash))
    {
        for (kind, path) in [
            (
                "prediction",
                registry
                    .join(hash)
                    .join("predictions")
                    .join(PREDICTION_FILE),
            ),
            (
                "report",
                registry.join(hash).join("reports").join(REPORT_FILE),
            ),
        ] {
            untouched.push(UntouchedEntry {
                kind: kind.to_string(),
                dataset_hash: hash.to_string(),
                sha256: hash_file(&path)?,
            });
        }
    }
    targets.sort_by(|left, right| {
        let order = |kind: &str| match kind {
            "prediction" => 0,
            "report" => 1,
            "panel_ledger" => 2,
            _ => 3,
        };
        order(&left.kind)
            .cmp(&order(&right.kind))
            .then_with(|| left.dataset_hash.cmp(&right.dataset_hash))
    });
    let manifest = StageManifest {
        schema: "JopeDimeFollowUpStaging.v1".to_string(),
        follow_up_label: "exploratory, test-adaptive follow-up".to_string(),
        exploratory_reused_test: true,
        original_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        cohort: LOSS_DATASETS
            .iter()
            .map(|hash| (*hash).to_string())
            .collect(),
        targets,
        untouched_original_winners: untouched,
    };
    validate_stage_manifest(&manifest)?;
    let manifest_bytes = canonical_json(&serde_json::to_value(&manifest)?)?;
    immutable_write(&incoming.join("manifest.v1.json"), &manifest_bytes)?;
    let marker = CompletionMarker {
        schema: "JopeDimeFollowUpStagingComplete.v1".to_string(),
        status: "COMPLETE".to_string(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
    };
    immutable_write(
        &incoming.join("STAGING_COMPLETE"),
        &canonical_json(&serde_json::to_value(marker)?)?,
    )?;
    fs::create_dir_all(replacement_root())?;
    fs::rename(&incoming, replacement_root().join("staged"))?;
    File::open(replacement_root())?.sync_all()?;
    load_stage()
}

fn archive_manifest(stage: &StageManifest) -> ArchiveManifest {
    ArchiveManifest {
        schema: "JopeDimeFollowUpArchive.v1".to_string(),
        source_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        entries: stage
            .targets
            .iter()
            .map(|entry| ArchiveEntry {
                kind: entry.kind.clone(),
                dataset_hash: entry.dataset_hash.clone(),
                archive_file: entry.archive_file.clone(),
                sha256: entry.old_sha256.clone(),
            })
            .collect(),
    }
}

fn validate_archive_pointer(pointer: &ArchivePointer) -> Result<()> {
    if pointer.schema != "JopeDimeFollowUpArchivePointer.v1" || pointer.status != "COMPLETE" {
        bail!("follow-up archive pointer is invalid");
    }
    validate_sha256("archive manifest SHA-256", &pointer.archive_manifest_sha256)
}

fn validate_archive(stage: &StageManifest, pointer: &ArchivePointer) -> Result<PathBuf> {
    validate_archive_pointer(pointer)?;
    let root = replacement_root()
        .join("archives")
        .join(&pointer.archive_manifest_sha256);
    let marker: CompletionMarker =
        serde_json::from_slice(&fs::read(root.join("ARCHIVE_COMPLETE"))?)?;
    if marker.schema != "JopeDimeFollowUpArchiveComplete.v1"
        || marker.status != "COMPLETE"
        || marker.manifest_sha256 != pointer.archive_manifest_sha256
    {
        bail!("content-addressed archive completion marker is invalid");
    }
    let manifest_bytes = fs::read(root.join("manifest.v1.json"))?;
    if sha256_bytes(&manifest_bytes) != pointer.archive_manifest_sha256 {
        bail!("content-addressed archive manifest hash mismatch");
    }
    let manifest: ArchiveManifest = serde_json::from_slice(&manifest_bytes)?;
    if canonical_json(&serde_json::to_value(&manifest)?)?
        != canonical_json(&serde_json::to_value(archive_manifest(stage))?)?
    {
        bail!("content-addressed archive does not match the staged replacement");
    }
    for entry in &manifest.entries {
        if hash_file(&root.join(&entry.archive_file))? != entry.sha256 {
            bail!("archived payload hash mismatch for {}", entry.archive_file);
        }
    }
    Ok(root)
}

fn ensure_archive(registry: &Path, stage: &StageManifest) -> Result<(ArchivePointer, PathBuf)> {
    let pointer_path = replacement_root().join("ARCHIVE_COMPLETE");
    if pointer_path.exists() {
        let pointer: ArchivePointer = serde_json::from_slice(&fs::read(&pointer_path)?)?;
        let root = validate_archive(stage, &pointer)?;
        return Ok((pointer, root));
    }
    for entry in &stage.targets {
        let target = public_target(registry, entry)?;
        if hash_file(&target)? != entry.old_sha256 {
            bail!("cannot archive {} after it changed", target.display());
        }
    }
    let manifest = archive_manifest(stage);
    let manifest_bytes = canonical_json(&serde_json::to_value(&manifest)?)?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let archive_parent = replacement_root().join("archives");
    fs::create_dir_all(&archive_parent)?;
    let final_root = archive_parent.join(&manifest_sha256);
    if !final_root.exists() {
        let incoming = archive_parent.join(format!(".incoming-{}", std::process::id()));
        if incoming.exists() {
            bail!(
                "stale incoming archive directory exists: {}",
                incoming.display()
            );
        }
        fs::create_dir_all(&incoming)?;
        for entry in &stage.targets {
            let bytes = fs::read(public_target(registry, entry)?)?;
            if sha256_bytes(&bytes) != entry.old_sha256 {
                bail!("archive source drifted during copy");
            }
            immutable_write(&incoming.join(&entry.archive_file), &bytes)?;
        }
        immutable_write(&incoming.join("manifest.v1.json"), &manifest_bytes)?;
        let marker = CompletionMarker {
            schema: "JopeDimeFollowUpArchiveComplete.v1".to_string(),
            status: "COMPLETE".to_string(),
            manifest_sha256: manifest_sha256.clone(),
        };
        // Marker-last: no archive member is created after this receipt.
        immutable_write(
            &incoming.join("ARCHIVE_COMPLETE"),
            &canonical_json(&serde_json::to_value(marker)?)?,
        )?;
        fs::rename(&incoming, &final_root)?;
        File::open(&archive_parent)?.sync_all()?;
    }
    let pointer = ArchivePointer {
        schema: "JopeDimeFollowUpArchivePointer.v1".to_string(),
        status: "COMPLETE".to_string(),
        archive_manifest_sha256: manifest_sha256,
    };
    validate_archive(stage, &pointer)?;
    immutable_write(
        &pointer_path,
        &canonical_json(&serde_json::to_value(&pointer)?)?,
    )?;
    Ok((pointer, final_root))
}

fn preflight_targets(registry: &Path, stage: &StageManifest) -> Result<()> {
    for entry in &stage.targets {
        let path = public_target(registry, entry)?;
        let current = hash_file(&path)?;
        validate_target_hash(&path, &current, &entry.old_sha256, &entry.new_sha256)?;
    }
    for entry in &stage.untouched_original_winners {
        let path = untouched_target(registry, entry)?;
        if hash_file(&path)? != entry.sha256 {
            bail!("untouched original winner drifted: {}", path.display());
        }
    }
    Ok(())
}

fn validate_target_hash(path: &Path, current: &str, old: &str, new: &str) -> Result<()> {
    if current != old && current != new {
        bail!(
            "{} has an unauthorized third hash {current}; expected old {old} or new {new}",
            path.display()
        );
    }
    Ok(())
}

fn apply_targets(registry: &Path, stage: &StageManifest) -> Result<()> {
    preflight_targets(registry, stage)?;
    for kind in ["prediction", "report", "panel_ledger"] {
        for entry in stage.targets.iter().filter(|entry| entry.kind == kind) {
            let target = public_target(registry, entry)?;
            let current = hash_file(&target)?;
            validate_target_hash(&target, &current, &entry.old_sha256, &entry.new_sha256)?;
            if current == entry.new_sha256 {
                continue;
            }
            let bytes = fs::read(replacement_root().join("staged").join(&entry.staged_file))?;
            if sha256_bytes(&bytes) != entry.new_sha256 {
                bail!("staged replacement changed before publication");
            }
            atomic_replace(&target, &bytes)?;
            if hash_file(&target)? != entry.new_sha256 {
                bail!(
                    "atomic replacement hash audit failed for {}",
                    target.display()
                );
            }
        }
    }
    preflight_targets(registry, stage)?;
    for entry in &stage.targets {
        let path = public_target(registry, entry)?;
        if hash_file(&path)? != entry.new_sha256 {
            bail!("not every staged target reached its new hash");
        }
    }
    Ok(())
}

fn rollback_targets(registry: &Path, stage: &StageManifest, archive: &Path) -> Result<()> {
    preflight_targets(registry, stage)?;
    for kind in ["prediction", "report", "panel_ledger"] {
        for entry in stage.targets.iter().filter(|entry| entry.kind == kind) {
            let target = public_target(registry, entry)?;
            let current = hash_file(&target)?;
            validate_target_hash(&target, &current, &entry.old_sha256, &entry.new_sha256)?;
            if current == entry.old_sha256 {
                continue;
            }
            let bytes = fs::read(archive.join(&entry.archive_file))?;
            if sha256_bytes(&bytes) != entry.old_sha256 {
                bail!("rollback archive payload hash mismatch");
            }
            atomic_replace(&target, &bytes)?;
        }
    }
    Ok(())
}

fn audit_complete(registry: &Path, stage: &StageManifest) -> Result<()> {
    preflight_targets(registry, stage)?;
    for entry in &stage.targets {
        let target = public_target(registry, entry)?;
        if hash_file(&target)? != entry.new_sha256 {
            bail!(
                "publication target is not at its staged new hash: {}",
                target.display()
            );
        }
        if entry.kind == "report" {
            let report: Value = serde_json::from_slice(&fs::read(&target)?)?;
            if report["schema"] != "JopeDimeDatasetReport.v2"
                || report["provenance"]["exploratory_reused_test"] != true
                || report["comparisons"]["baselines"]
                    .as_object()
                    .map(|value| value.len())
                    != Some(6)
                || report["timings"]["fit_seconds"].as_f64().is_none()
                || report["timings"]["predict_seconds"].as_f64().is_none()
                || report["prediction"]["sha256"].as_str().is_none()
                || report["selection_proxy"]["authority_label"]
                    != "CV7 training-only selection proxy; not final test authority"
            {
                bail!("published v2 report audit failed for {}", target.display());
            }
        }
    }
    let ledger: Value = serde_json::from_slice(&fs::read(original_ledger_path())?)?;
    if ledger["schema"] != "JopeDimePanelResults.v2"
        || ledger["dataset_count"].as_u64() != Some(42)
        || ledger["follow_up"]["exploratory_reused_test"] != true
        || ledger["selection_accounting"]["follow_up_evolutionary_slots"].as_u64()
            != Some(FOLLOW_UP_SELECTION_SLOTS as u64)
        || ledger["selection_accounting"]["updated_active_panel_evolutionary_slots"].as_u64()
            != Some(ACTIVE_PANEL_SELECTION_SLOTS as u64)
        || ledger["selection_accounting"]["historical_evolutionary_slots_across_both_experiments"]
            .as_u64()
            != Some(HISTORICAL_SELECTION_SLOTS as u64)
        || ledger["outcomes"].as_array().map(Vec::len) != Some(42)
        || ledger["pairwise_results"].as_array().map(Vec::len) != Some(21)
        || ledger["elo"]["standings"].as_array().map(Vec::len) != Some(7)
    {
        bail!("published v2 panel ledger audit failed");
    }
    Ok(())
}

fn publish_follow_up(registry: &Path, freezes: &BTreeMap<String, ValidatedFreeze>) -> Result<()> {
    if replacement_complete_path().exists() {
        let stage = load_stage()?;
        audit_complete(registry, &stage)?;
        println!("follow-up replacement is already complete and passed the full hash audit");
        return Ok(());
    }
    // Baseline/test data are first opened below, after all 35 freezes and all
    // replay prefixes have passed the training-only validator.
    let source_bytes = authorized_source_ledger_bytes()?;
    let source = validate_compiled_cohort(&source_bytes)?;
    let stage = create_stage(registry, freezes, &source)?;
    let (_pointer, archive) = ensure_archive(registry, &stage)?;
    validate_archive(
        &stage,
        &serde_json::from_slice(&fs::read(replacement_root().join("ARCHIVE_COMPLETE"))?)?,
    )?;
    apply_targets(registry, &stage)?;
    audit_complete(registry, &stage)?;
    let marker = json!({
        "schema": "JopeDimeOfficialReplacementComplete.v1",
        "status": "COMPLETE",
        "stage_manifest_sha256": hash_file(&stage_manifest_path())?,
        "archive_manifest_sha256": archive.file_name().and_then(|name| name.to_str()),
        "reports_replaced": 35,
        "predictions_replaced": 35,
        "untouched_original_winners": 7,
        "panel_ledgers_replaced": 1,
        "exploratory_reused_test": true,
    });
    immutable_write(&replacement_complete_path(), &canonical_json(&marker)?)?;
    println!(
        "published all 35 exploratory follow-up outcomes, one v2 panel ledger, and a complete rollback archive"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(name: &str) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "jope-dime-followup-{name}-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn generation_accounting_is_exact_for_10_and_30() {
        assert_eq!(
            exact_cv::ORIGINAL_GENERATIONS * 60 * 42,
            ORIGINAL_SELECTION_SLOTS
        );
        assert_eq!(FOLLOW_UP_GENERATIONS * 60 * 35, FOLLOW_UP_SELECTION_SLOTS);
        assert_eq!(
            7 * 600 + FOLLOW_UP_SELECTION_SLOTS,
            ACTIVE_PANEL_SELECTION_SLOTS
        );
        assert_eq!(
            ORIGINAL_SELECTION_SLOTS + FOLLOW_UP_SELECTION_SLOTS,
            HISTORICAL_SELECTION_SLOTS
        );
    }

    #[test]
    fn compiled_cohort_is_exact_and_pinned() {
        assert_eq!(LOSS_DATASETS.len(), 35);
        assert_eq!(
            LOSS_DATASETS.iter().copied().collect::<BTreeSet<_>>().len(),
            35
        );
        assert!(LOSS_DATASETS.iter().all(|hash| DATASETS.contains(hash)));
        assert_eq!(DATASETS.len() - LOSS_DATASETS.len(), 7);
        assert_eq!(ORIGINAL_LEDGER_SHA256.len(), 64);
    }

    #[test]
    fn cohort_derivation_rejects_any_outcome_drift() {
        let rows = DATASETS
            .iter()
            .map(|hash| {
                json!({
                    "dataset_hash": hash,
                    "strict_overall_outcome": if LOSS_DATASETS.contains(hash) { "loss" } else { "win" },
                })
            })
            .collect::<Vec<_>>();
        let source = json!({
            "schema": "JopeDimePanelResults.v1",
            "dataset_count": 42,
            "outcomes": rows,
        });
        let bytes = canonical_json(&source).unwrap();
        validate_compiled_cohort(&bytes).unwrap();
        let mut drifted = source;
        let first_loss = drifted["outcomes"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|row| row["strict_overall_outcome"] == "loss")
            .unwrap();
        first_loss["strict_overall_outcome"] = json!("win");
        assert!(validate_compiled_cohort(&canonical_json(&drifted).unwrap()).is_err());
    }

    #[test]
    fn overwrite_allowlist_rejects_a_third_hash() {
        let old = sha256_bytes(b"old");
        let new = sha256_bytes(b"new");
        let current = sha256_bytes(b"third");
        assert!(validate_target_hash(Path::new("target"), &old, &old, &new).is_ok());
        assert!(validate_target_hash(Path::new("target"), &new, &old, &new).is_ok());
        assert!(validate_target_hash(Path::new("target"), &current, &old, &new).is_err());
    }

    #[test]
    fn atomic_replacement_recovers_after_an_interruption() {
        let root = temp_dir("interrupted");
        let path = root.join("target");
        fs::write(&path, b"old").unwrap();
        atomic_replace(&path, b"new").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"new");
        // Replaying an already-completed target is a no-op at the state-machine level.
        let old = sha256_bytes(b"old");
        let new = sha256_bytes(b"new");
        let current = hash_file(&path).unwrap();
        assert!(current == old || current == new);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn archive_payload_supports_exact_rollback() {
        let root = temp_dir("rollback");
        let registry = root.join("registry");
        let archive = root.join("archive");
        let hash = LOSS_DATASETS[0];
        let target = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let archive_file = "payload/predictions/old.csv";
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::create_dir_all(archive.join("payload/predictions")).unwrap();
        fs::write(&target, b"new").unwrap();
        fs::write(archive.join(archive_file), b"old").unwrap();
        let stage = StageManifest {
            schema: "fixture".to_string(),
            follow_up_label: "fixture".to_string(),
            exploratory_reused_test: true,
            original_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
            cohort: vec![],
            targets: vec![TargetEntry {
                kind: "prediction".to_string(),
                dataset_hash: Some(hash.to_string()),
                staged_file: "unused".to_string(),
                archive_file: archive_file.to_string(),
                old_sha256: sha256_bytes(b"old"),
                new_sha256: sha256_bytes(b"new"),
            }],
            untouched_original_winners: vec![],
        };
        rollback_targets(&registry, &stage, &archive).unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"old");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_float_ordering_never_uses_a_tolerance() {
        let one = 1.0_f64;
        let next = f64::from_bits(one.to_bits() + 1);
        assert_eq!(one.total_cmp(&one), Ordering::Equal);
        assert_eq!(one.total_cmp(&next), Ordering::Less);
        assert_eq!(next.total_cmp(&one), Ordering::Greater);
    }
}
