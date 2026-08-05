//! Checkpoint-bound, fail-closed Rust micro-wave planner and builder.
//!
//! This binary deliberately has no scheduler submission surface. `plan` freezes
//! the 128-dataset cohort first; the PyTorch proposal generator may then consume
//! that cohort at the explicit ML boundary; `build` accepts only 64 proposals
//! per selected dataset that bind to the cohort's checkpoint.

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};
use feat_math::genome::GenomeIr;
#[cfg(test)]
use jope_dime_regressor::selfplay_data::validate_wave_composition;
use jope_dime_regressor::selfplay_data::{
    micro_wave_size, select_micro_cohort, validate_dl_slot_rung,
    validate_wave_composition_with_dl, CandidateOrigin, GpProposal, MicroAdmission, MicroDataset,
    MicroLane, WaveSlot, WaveSlotGroup, CHECKPOINT_PROPOSAL_SLOTS, COMMON_WINNER_SLOTS,
    LOCAL_ELITE_SLOTS, MICRO_TASK_SIZE, OFFSPRING_SLOTS, WAVE_SIZE,
};
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, canonical_json_bytes, checkpoint_epoch, evaluation_protocol_sha256,
    expected_evaluation_metric, now_ms, read_json, sha256_bytes, sha256_file,
};
use rusqlite::{params, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

const COHORT_SCHEMA: &str = "JopeMicroCohort.v2";
const WAVE_SCHEMA: &str = "JopeRealtimeSelfplayWave.v1";
const MANIFEST_SCHEMA: &str = "JopeFlywheelCorpusEvalShard.v1";
const MANIFEST_SCHEMA_V2: &str = "JopeFlywheelCorpusEvalShard.v2";
const TS_REDUCTION_V2: &str = "ts-reduction-v2";
const ROBUST_BANK_SCHEMA: &str = "JopeRobustGpSeedBank.v1";
const SEALED_SCHEMA: &str = "JopeSealedCohort.v1";
const SEALED_SCHEMAS: [&str; 2] = ["JopeSealedCohort.v1", "JopeSealedMilestoneCohort.v1"];
const SEALED_SELECTION_PROTOCOL: &str = "catalog-sha256-seeded-global-family-disjoint-v1";
const COHORT_PROPOSAL_SEED: u64 = 20_260_726;
const MAX_PROPOSAL_RESERVOIR_PER_CONTEXT: usize = 256;
const REQUIRED_SEALED_PER_TASK: usize = 256;
const MAX_MUTATION_ATTEMPTS: usize = 100_000;
const OPTIONAL_TASK_INDEPENDENT_DATASET_FLOOR: usize = 500;

#[derive(Parser, Debug)]
#[command(
    name = "jope-selfplay-wave",
    about = "Plan and build immutable checkpoint-fresh 128-dataset self-play micro-waves",
    long_about = "Rust-only orchestration for deterministic 64-regression/64-binary micro-waves. \
The plan/build split ensures cohort selection happens before PyTorch proposal generation. \
The binary never submits, holds, or cancels SCQ jobs."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Freeze the private family-disjoint 512-dataset milestone cohort.
    FreezeSealed(FreezeSealedArgs),
    /// Sanitize and deterministically backfill cohort-scoped checkpoint proposals.
    SanitizeProposals(SanitizeProposalsArgs),
    /// Select and freeze one 128-dataset cohort; does not read or generate proposals.
    Plan(PlanArgs),
    /// Build manifests from a frozen cohort and current-checkpoint proposal file.
    Build(BuildArgs),
}

#[derive(Args, Debug)]
struct FreezeSealedArgs {
    /// Immutable catalog pointer containing catalog path and SHA-256.
    #[arg(long)]
    catalog_pointer: PathBuf,
    /// Private immutable JopeSealedCohort.v1 output, created with mode 0600.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args, Debug)]
struct SanitizeProposalsArgs {
    /// Frozen JopeMicroCohort.v2 produced by `plan`.
    #[arg(long)]
    cohort: PathBuf,
    /// Current proposal pointer containing path, file hash, and checkpoint hash.
    #[arg(long)]
    proposals_pointer: PathBuf,
    /// Root receiving content-addressed JSONL and pointer outputs.
    #[arg(long)]
    output_root: PathBuf,
}

#[derive(Args, Debug)]
struct PlanArgs {
    /// Immutable catalog pointer containing catalog path and SHA-256.
    #[arg(long)]
    catalog_pointer: PathBuf,
    /// Pointer containing the active checkpoint_sha256 (and optionally checkpoint path).
    #[arg(long)]
    checkpoint_pointer: PathBuf,
    /// Read-only strict corpus used only for per-dataset observation counts.
    #[arg(long)]
    corpus: PathBuf,
    /// Frozen 512-dataset sealed cohort deny-list (256 per task).
    #[arg(long)]
    sealed_cohort: PathBuf,
    /// Root receiving pending cohorts and complete immutable waves.
    #[arg(long)]
    output_root: PathBuf,
    /// Explicit strict-wave-NNNNNN identity; otherwise the next identity is derived.
    #[arg(long)]
    wave_id: Option<String>,
    /// Permit the supply supervisor to freeze additional immutable cohorts
    /// behind an already launch-ready cohort. Normal manual planning remains
    /// single-flight unless this explicit lookahead flag is present.
    #[arg(long)]
    allow_pending_lookahead: bool,
    /// Plan the cohort and print it WITHOUT writing anything. Used to diff a
    /// policy change against the live catalog and corpus before deploying it.
    #[arg(long)]
    dry_run: bool,

    // ---- breadth-over-depth admission (WS-3) ---------------------------
    // The supply supervisor invokes `plan` with a fixed argument list
    // (jope_selfplay_supply.rs:864), so these must be defaults, not required
    // flags. Each is ALSO readable from the environment -- clap is not built
    // with its `env` feature here and adding it would perturb every other
    // binary in the workspace, so the fallback is read by hand in
    // `PlanArgs::admission` -- which lets the policy be retuned, or reverted
    // wholesale with JOPE_MICRO_ADMISSION_UNBOUNDED=1, through the unit's
    // Environment= with no rebuild and no code change.
    /// Hard per-dataset evaluation cap. At or above this, a dataset stops
    /// competing for cohort slots. Env: JOPE_MICRO_EVAL_CAP.
    #[arg(long)]
    eval_cap: Option<u64>,
    /// Minimum observations before a dataset can be judged converged.
    /// Env: JOPE_MICRO_RETIRE_MIN_OBS.
    #[arg(long)]
    retire_min_observations: Option<u64>,
    /// Retire after this many waves with no improvement to the all-time best.
    /// Env: JOPE_MICRO_RETIRE_STALE_WAVES.
    #[arg(long)]
    retire_stale_waves: Option<u32>,
    /// Retire when the head spread (best - p90) falls to or below this.
    /// Env: JOPE_MICRO_RETIRE_SPREAD_FLOOR.
    #[arg(long)]
    retire_spread_floor: Option<f64>,
    /// Parole rotation period in waves; bounds how long retirement can last.
    /// Env: JOPE_MICRO_PAROLE_PERIOD.
    #[arg(long)]
    parole_period: Option<u32>,
    /// Hard-uncertain slots per task reserved for paroled datasets.
    /// Env: JOPE_MICRO_PAROLE_SLOTS.
    #[arg(long)]
    parole_slots: Option<usize>,
    /// Observation-count band width for the least-observed spread tie-break.
    /// Env: JOPE_MICRO_LEAST_BAND.
    #[arg(long)]
    least_observed_band: Option<u64>,
    /// Revert to the pre-breadth admission policy (no cap, no retirement, no
    /// spread ordering). Env: JOPE_MICRO_ADMISSION_UNBOUNDED.
    #[arg(long)]
    admission_unbounded: bool,
}

/// Flag, else environment, else the compiled default. A malformed environment
/// value is IGNORED rather than fatal: a typo in a unit file must not be able
/// to stop the only path that produces strict waves.
fn tuned<T: std::str::FromStr>(flag: Option<T>, key: &str, fallback: T) -> T {
    flag.or_else(|| {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse().ok())
    })
    .unwrap_or(fallback)
}

impl PlanArgs {
    fn admission(&self) -> MicroAdmission {
        let unbounded = self.admission_unbounded
            || matches!(
                std::env::var("JOPE_MICRO_ADMISSION_UNBOUNDED")
                    .unwrap_or_default()
                    .trim(),
                "1" | "true" | "TRUE" | "yes"
            );
        if unbounded {
            return MicroAdmission::unbounded();
        }
        let base = MicroAdmission::default();
        MicroAdmission {
            eval_cap: tuned(self.eval_cap, "JOPE_MICRO_EVAL_CAP", base.eval_cap),
            retire_min_observations: tuned(
                self.retire_min_observations,
                "JOPE_MICRO_RETIRE_MIN_OBS",
                base.retire_min_observations,
            ),
            retire_stale_waves: tuned(
                self.retire_stale_waves,
                "JOPE_MICRO_RETIRE_STALE_WAVES",
                base.retire_stale_waves,
            ),
            retire_spread_floor: tuned(
                self.retire_spread_floor,
                "JOPE_MICRO_RETIRE_SPREAD_FLOOR",
                base.retire_spread_floor,
            ),
            parole_period: tuned(
                self.parole_period,
                "JOPE_MICRO_PAROLE_PERIOD",
                base.parole_period,
            ),
            parole_slots: tuned(
                self.parole_slots,
                "JOPE_MICRO_PAROLE_SLOTS",
                base.parole_slots,
            ),
            least_observed_band: tuned(
                self.least_observed_band,
                "JOPE_MICRO_LEAST_BAND",
                base.least_observed_band,
            ),
        }
    }
}

#[derive(Args, Debug)]
struct BuildArgs {
    /// Cohort produced by `plan`.
    #[arg(long)]
    cohort: PathBuf,
    /// Same immutable catalog pointer used by `plan`.
    #[arg(long)]
    catalog_pointer: PathBuf,
    /// Current proposal pointer containing path, file hash, and checkpoint hash.
    #[arg(long)]
    proposals_pointer: PathBuf,
    /// Read-only strict corpus providing ranked dataset-local elites.
    #[arg(long)]
    corpus: PathBuf,
    /// Rust-exported JopeRobustGpSeedBank.v1 with repeated cross-dataset winners.
    #[arg(long)]
    robust_seed_bank: PathBuf,
    /// Same frozen sealed cohort deny-list used by `plan`.
    #[arg(long)]
    sealed_cohort: PathBuf,
    /// Authoritative exactly-once pair registry used to remove blocked exact pairs.
    #[arg(long)]
    pair_registry: PathBuf,
    /// Root receiving the complete immutable wave directory.
    #[arg(long)]
    output_root: PathBuf,
    /// Maximum candidates in one manifest. Hard-capped at 100.
    #[arg(long, default_value_t = 100)]
    max_candidates_per_manifest: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct CatalogPointer {
    catalog: PathBuf,
    catalog_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CheckpointPointer {
    checkpoint_sha256: String,
    #[serde(default)]
    checkpoint: Option<PathBuf>,
}

#[derive(Clone, Debug, Deserialize)]
struct ProposalsPointer {
    #[serde(default)]
    schema: Option<String>,
    proposals: PathBuf,
    proposals_sha256: String,
    checkpoint_sha256: String,
    #[serde(default)]
    dl_slots_by_task: BTreeMap<String, usize>,
}

impl ProposalsPointer {
    fn dl_slots(&self, task: &str) -> Result<usize> {
        if let Some(schema) = self.schema.as_deref() {
            ensure!(
                matches!(
                    schema,
                    "JopeSelfplayProposalPointer.v1"
                        | "DlProposalSet.v2"
                        | "JopeSanitizedProposalPointer.v1"
                        | "JopeSanitizedProposalPointer.v2"
                        | "CohortProposalPointer.v1"
                ),
                "unsupported proposal pointer schema"
            );
        }
        if self.dl_slots_by_task.is_empty() {
            return Ok(CHECKPOINT_PROPOSAL_SLOTS);
        }
        ensure!(
            self.dl_slots_by_task.keys().all(|key| {
                matches!(
                    key.as_str(),
                    "regression" | "binary" | "timeseries" | "multiclass"
                )
            }),
            "proposal pointer has an unknown task allocation"
        );
        let slots = *self
            .dl_slots_by_task
            .get(task)
            .with_context(|| format!("proposal pointer lacks {task} DL allocation"))?;
        validate_dl_slot_rung(slots)?;
        Ok(slots)
    }

    /// DL allocation for exactly the tasks the cohort carries.
    ///
    /// This used to return a fixed two-key map while callers indexed it by
    /// `cohort_row.task`. On a cohort containing any multiclass or timeseries row that index
    /// is a missing key, and `BTreeMap`'s `Index` impl panics rather than returning an error
    /// — so the wave binary aborted outright instead of failing closed with a message.
    fn normalized_dl_slots(&self, tasks: &[String]) -> Result<BTreeMap<String, usize>> {
        tasks
            .iter()
            .map(|task| Ok((task.clone(), self.dl_slots(task)?)))
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Catalog {
    entries: Vec<CatalogEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CatalogEntry {
    task: String,
    dataset_hash: String,
    train_csv: PathBuf,
    test_csv: PathBuf,
    train_sha256: String,
    test_sha256: String,
    source_family: String,
    #[serde(default)]
    stable_anchor: bool,
    #[serde(default)]
    golden42: bool,
    #[serde(default)]
    current_catalog: Option<bool>,
    #[serde(default)]
    observation_count: Option<u64>,
    #[serde(default)]
    uncertainty: Option<f64>,
    #[serde(default)]
    hardness: Option<f64>,
    #[serde(default)]
    incumbent_regret: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct SealedCohort {
    schema: String,
    entries: Vec<SealedEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct SealedEntry {
    task: String,
    dataset_hash: String,
    #[serde(default)]
    source_family: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FrozenSealedCohort {
    schema: String,
    immutable: bool,
    catalog_sha256: String,
    selection_protocol: String,
    entries: Vec<FrozenSealedEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct FrozenSealedEntry {
    task: String,
    dataset_hash: String,
    source_family: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CohortPlan {
    schema: String,
    immutable: bool,
    wave: String,
    checkpoint_epoch: String,
    checkpoint_sha256: String,
    catalog_sha256: String,
    sealed_cohort_sha256: String,
    /// Tasks this cohort actually carries, in the order they were laned.
    ///
    /// The wave shape is `tasks.len() * MICRO_TASK_SIZE`, so a cohort grows from 128 to
    /// 192 to 256 as multiclass and then timeseries reach the per-task floor, instead of
    /// requiring a flag day where every task must qualify at once.
    ///
    /// The default is load-bearing: every cohort already on disk was written before this
    /// field existed and is a two-task cohort, so an absent `tasks` MUST deserialize to
    /// exactly `["regression", "binary"]` or replaying those waves would fail validation.
    #[serde(default = "legacy_cohort_tasks")]
    tasks: Vec<String>,
    entries: Vec<CohortEntry>,
}

/// The task set every cohort written before `CohortPlan::tasks` existed implicitly had.
fn legacy_cohort_tasks() -> Vec<String> {
    vec!["regression".to_owned(), "binary".to_owned()]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CohortEntry {
    task: String,
    dataset_hash: String,
    micro_lane: MicroLane,
}

#[derive(Clone, Debug, Deserialize)]
struct RobustSeedBank {
    schema: String,
    tasks: BTreeMap<String, Vec<RobustSeed>>,
}

#[derive(Clone, Debug, Deserialize)]
struct RobustSeed {
    task: String,
    semantic_sha256: String,
    canonical_genome: String,
    dataset_count: usize,
    mean_within_dataset_percentile: f64,
    worst_within_dataset_percentile: f64,
    strict_win_count: usize,
    #[serde(default)]
    dl_origin: bool,
}

#[derive(Clone, Debug)]
struct LocalElite {
    semantic_sha256: String,
    canonical_genome: String,
    score: f64,
    secondary_score: Option<f64>,
    receipt_sha256: String,
    candidate_source: String,
    dl_ancestry_kind: String,
}

#[derive(Clone, Debug, Serialize)]
struct Candidate {
    canonical_genome: String,
    semantic_sha256: String,
    origin: CandidateOrigin,
    source: String,
    slot_group: WaveSlotGroup,
    slot_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_checkpoint_sha256: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    parentage: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
struct SanitizedProposal {
    task: String,
    dataset_hash: String,
    canonical_genome: String,
    semantic_sha256: String,
    model_checkpoint_sha256: String,
    slot: usize,
    score: Option<f64>,
    proposal_score: Option<f64>,
    origin: &'static str,
    distinct_dataset_repeat_count: usize,
    source_task: String,
    source_dataset_hash: String,
    source_slot: usize,
    retargeted: bool,
    backfill_rank: Option<usize>,
}

#[derive(Clone, Debug, Default)]
struct RegistryPairView {
    unavailable: HashSet<String>,
}

#[derive(Clone, Debug, Serialize)]
struct Manifest {
    schema: &'static str,
    evaluation_lane: &'static str,
    strict_wave: String,
    generation: String,
    micro_lane: MicroLane,
    task: String,
    dataset_hash: String,
    checkpoint_epoch: String,
    checkpoint_sha256: String,
    catalog_sha256: String,
    protocol_sha256: String,
    generated_at_ms: u128,
    train_csv: PathBuf,
    test_csv: PathBuf,
    train_sha256: String,
    test_sha256: String,
    /// Pinned dataset meta.json, REQUIRED by the evaluator for multiclass
    /// (class-label decode authority) and timeseries (reduction contract).
    /// Skipped when absent so regression/binary manifests stay byte-identical
    /// to every wave ever built. Wave 552's 704 multiclass manifests were all
    /// rejected with "must pin meta_json" because these fields did not exist.
    #[serde(skip_serializing_if = "Option::is_none")]
    meta_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    horizon: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seasonality: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_is_final_column: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    known_future_covariates: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    past_only_covariates: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reduction: Option<Value>,
    label_authority: Value,
    candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestReceipt {
    relative_path: String,
    sha256: String,
    task: String,
    dataset_hash: String,
    candidate_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::FreezeSealed(args) => freeze_sealed(args),
        Command::SanitizeProposals(args) => sanitize_proposals(args),
        Command::Plan(args) => plan(args),
        Command::Build(args) => build(args),
    }?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} is not a lowercase SHA-256 value"
    );
    Ok(())
}

fn resolve_pointer_path(pointer: &Path, target: &Path) -> PathBuf {
    if target.is_absolute() {
        target.to_owned()
    } else {
        pointer
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(target)
    }
}

fn load_catalog(pointer_path: &Path) -> Result<(Catalog, String)> {
    let pointer: CatalogPointer = serde_json::from_value(read_json(pointer_path)?)
        .with_context(|| format!("parse catalog pointer {}", pointer_path.display()))?;
    validate_sha256(&pointer.catalog_sha256, "catalog SHA-256")?;
    let path = resolve_pointer_path(pointer_path, &pointer.catalog);
    ensure!(path.is_file(), "catalog is absent: {}", path.display());
    ensure!(
        sha256_file(&path)? == pointer.catalog_sha256,
        "catalog pointer content drift"
    );
    let catalog: Catalog = serde_json::from_value(read_json(&path)?)
        .with_context(|| format!("parse catalog {}", path.display()))?;
    validate_catalog_entries(&catalog.entries)?;
    Ok((catalog, pointer.catalog_sha256))
}

fn validate_catalog_entries(entries: &[CatalogEntry]) -> Result<()> {
    let mut identities = HashSet::new();
    ensure!(!entries.is_empty(), "catalog is empty");
    for entry in entries {
        ensure!(
            matches!(
                entry.task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            ),
            "catalog has invalid task"
        );
        ensure!(
            !entry.dataset_hash.is_empty() && !entry.source_family.is_empty(),
            "catalog identity or source family is absent"
        );
        ensure!(
            identities.insert((entry.task.as_str(), entry.dataset_hash.as_str())),
            "catalog repeats a task/dataset identity"
        );
        validate_sha256(&entry.train_sha256, "train SHA-256")?;
        validate_sha256(&entry.test_sha256, "test SHA-256")?;
        ensure!(
            [entry.uncertainty, entry.hardness, entry.incumbent_regret]
                .into_iter()
                .flatten()
                .all(f64::is_finite),
            "catalog has non-finite selection evidence"
        );
    }
    Ok(())
}

/// Read the time-series contract from the pinned dataset metadata. The
/// reduction policy is deliberately not read from metadata: it is a builder
/// decision and is stamped into every v2 shard so a later evaluator cannot
/// silently execute a different reduction.
fn timeseries_manifest_contract(
    train_csv: &Path,
) -> Result<(PathBuf, String, usize, usize, String, bool, Vec<String>, Vec<String>, Value)> {
    let meta_path = train_csv
        .parent()
        .context("timeseries train_csv has no parent directory")?
        .join("meta.json");
    let meta_bytes = fs::read(&meta_path)
        .with_context(|| format!("read timeseries meta.json {}", meta_path.display()))?;
    let meta_sha256 = sha256_bytes(&meta_bytes);
    let meta: Value = serde_json::from_slice(&meta_bytes)
        .with_context(|| format!("parse timeseries meta.json {}", meta_path.display()))?;
    ensure!(
        meta.get("task_type").and_then(Value::as_str) == Some("timeseries"),
        "timeseries meta.json must declare task_type=timeseries"
    );
    let required_usize = |field: &str| -> Result<usize> {
        meta.get(field)
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .with_context(|| format!("timeseries meta.json missing numeric {field}"))
    };
    let horizon = required_usize("horizon")?;
    let seasonality = required_usize("seasonality")?;
    let frequency = meta
        .get("frequency")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .context("timeseries meta.json missing frequency")?;
    let target_is_final_column = meta
        .get("target_is_final_column")
        .and_then(Value::as_bool)
        .context("timeseries meta.json missing target_is_final_column")?;
    ensure!(
        target_is_final_column,
        "timeseries meta.json target_is_final_column must be true"
    );
    let string_array = |field: &str| -> Result<Vec<String>> {
        meta.get(field)
            .and_then(Value::as_array)
            .with_context(|| format!("timeseries meta.json missing array {field}"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .with_context(|| format!("timeseries meta.json {field} contains non-string"))
            })
            .collect()
    };
    // Covariate roles come from `feature_columns`, the ONLY column list any
    // timeseries dataset actually declares: all 94 encoded datasets (52
    // synthetic + 42 real) carry `feature_columns` and none carries
    // `known_dynamic_columns` or `past_dynamic_columns_excluded`. Requiring
    // those names would fail every ts shard build and wedge supply in a retry
    // storm, exactly as wave 552 did for multiclass. The evaluator agrees they
    // are optional: corpus_eval.rs declares both `#[serde(default)]`, with
    // `known_dynamic_columns` merely an alias for `known_future_covariates`.
    // Every declared feature is observed for the whole forecast window, so all
    // of them are known-future and none is past-only; an empty past-only list
    // is skipped by serde and defaults back to empty at the evaluator.
    let known_future_covariates = string_array("feature_columns")?;
    let past_only_covariates: Vec<String> = Vec::new();
    Ok((
        meta_path,
        meta_sha256,
        horizon,
        seasonality,
        frequency,
        target_is_final_column,
        known_future_covariates,
        past_only_covariates,
        json!({"version": TS_REDUCTION_V2}),
    ))
}

fn sealed_rank(catalog_sha256: &str, entry: &CatalogEntry) -> [u8; 32] {
    Sha256::digest(
        format!(
            "{SEALED_SELECTION_PROTOCOL}|{catalog_sha256}|{}|{}|{}",
            entry.task, entry.source_family, entry.dataset_hash
        )
        .as_bytes(),
    )
    .into()
}

fn select_sealed_cohort(catalog: &Catalog, catalog_sha256: &str) -> Result<FrozenSealedCohort> {
    validate_sha256(catalog_sha256, "catalog SHA-256")?;
    let mut used_families = HashSet::with_capacity(REQUIRED_SEALED_PER_TASK * 2);
    let mut entries = Vec::with_capacity(REQUIRED_SEALED_PER_TASK * 2);

    // Binary has the smaller candidate pool in the current catalog, so reserve it
    // first. The SHA-bound ordering remains stable if catalog rows are reordered.
    // Multiclass and timeseries follow with their own, far smaller carve-outs
    // (`sealed_target`: 256/256/64/8) and are skipped entirely until their corpora can
    // supply that many *globally family-disjoint* datasets. Skipping rather than failing
    // is deliberate: a hard requirement here would block sealed-cohort generation, and
    // with it the whole loop, on a task that is still filling up.
    for task in ["binary", "regression", "multiclass", "timeseries"] {
        let target = jope_dime_regressor::selfplay_expansion::sealed_target(task);
        if target == 0 {
            continue;
        }
        let mandatory = matches!(task, "regression" | "binary");
        let mut candidates = catalog
            .entries
            .iter()
            .filter(|entry| entry.task == task)
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            sealed_rank(catalog_sha256, left)
                .cmp(&sealed_rank(catalog_sha256, right))
                .then_with(|| left.source_family.cmp(&right.source_family))
                .then_with(|| left.dataset_hash.cmp(&right.dataset_hash))
        });
        // Reserve against a scratch copy so a task that cannot reach its target does not
        // consume families the later tasks may need.
        let mut claimed = Vec::new();
        let mut claimed_families = Vec::new();
        for entry in candidates {
            if used_families.contains(entry.source_family.as_str())
                || claimed_families.contains(&entry.source_family.as_str())
            {
                continue;
            }
            claimed_families.push(entry.source_family.as_str());
            claimed.push(FrozenSealedEntry {
                task: entry.task.clone(),
                dataset_hash: entry.dataset_hash.clone(),
                source_family: entry.source_family.clone(),
            });
            if claimed.len() == target {
                break;
            }
        }
        if claimed.len() < target {
            ensure!(
                !mandatory,
                "catalog lacks {target} globally family-disjoint {task} datasets"
            );
            continue;
        }
        for family in claimed_families {
            used_families.insert(family);
        }
        entries.extend(claimed);
    }
    entries.sort_by(|left, right| {
        left.task
            .cmp(&right.task)
            .then_with(|| left.dataset_hash.cmp(&right.dataset_hash))
    });
    let cohort = FrozenSealedCohort {
        schema: SEALED_SCHEMA.to_owned(),
        immutable: true,
        catalog_sha256: catalog_sha256.to_owned(),
        selection_protocol: SEALED_SELECTION_PROTOCOL.to_owned(),
        entries,
    };
    validate_frozen_sealed(&cohort, catalog_sha256)?;
    Ok(cohort)
}

fn validate_frozen_sealed(cohort: &FrozenSealedCohort, catalog_sha256: &str) -> Result<()> {
    ensure!(cohort.schema == SEALED_SCHEMA, "sealed schema drift");
    ensure!(cohort.immutable, "sealed cohort is not immutable");
    ensure!(
        cohort.catalog_sha256 == catalog_sha256,
        "sealed catalog provenance drift"
    );
    ensure!(
        cohort.selection_protocol == SEALED_SELECTION_PROTOCOL,
        "sealed selection protocol drift"
    );
    let mut identities = HashSet::with_capacity(cohort.entries.len());
    let mut families = HashSet::with_capacity(cohort.entries.len());
    for entry in &cohort.entries {
        ensure!(
            matches!(
                entry.task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            ) && !entry.dataset_hash.is_empty()
                && !entry.source_family.is_empty(),
            "sealed cohort contains an invalid entry"
        );
        ensure!(
            identities.insert((entry.task.as_str(), entry.dataset_hash.as_str())),
            "sealed cohort repeats an identity"
        );
        ensure!(
            families.insert(entry.source_family.as_str()),
            "sealed cohort repeats a source family"
        );
    }
    // Each task present must hold exactly its carve-out. Regression and binary are
    // mandatory; multiclass and timeseries are optional, so an older 512-row sealed
    // cohort (256 + 256, written before either carve-out existed) still validates.
    let mut expected_total = 0usize;
    for task in jope_dime_regressor::selfplay_expansion::TASKS {
        let present = cohort
            .entries
            .iter()
            .filter(|entry| entry.task == task)
            .count();
        let target = jope_dime_regressor::selfplay_expansion::sealed_target(task);
        if present == 0 && !matches!(task, "regression" | "binary") {
            continue;
        }
        ensure!(
            present == target,
            "sealed cohort must contain exactly {target} {task} datasets, found {present}"
        );
        expected_total += target;
    }
    ensure!(
        cohort.entries.len() == expected_total,
        "sealed cohort size drift"
    );
    Ok(())
}

fn validate_existing_private_file(path: &Path, expected: &[u8]) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "sealed output exists but is not a regular file"
    );
    ensure!(
        metadata.permissions().mode() & 0o7777 == 0o600,
        "sealed output permission drift"
    );
    ensure!(
        fs::read(path)? == expected,
        "immutable sealed output content drift"
    );
    Ok(true)
}

fn publish_private_immutable(path: &Path, bytes: &[u8]) -> Result<bool> {
    if validate_existing_private_file(path, bytes)? {
        return Ok(false);
    }
    let parent = path.parent().context("sealed output has no parent")?;
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("sealed output has no filename")?;
    let temporary = parent.join(format!(".{name}.tmp-{}-{}", std::process::id(), now_ms()));
    let publish = (|| -> Result<bool> {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&temporary)?;
        output.write_all(bytes)?;
        output.sync_all()?;
        fs::set_permissions(&temporary, fs::Permissions::from_mode(0o600))?;
        let created = match fs::hard_link(&temporary, path) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                ensure!(
                    validate_existing_private_file(path, bytes)?,
                    "sealed output publication race"
                );
                false
            }
            Err(error) => return Err(error.into()),
        };
        File::open(parent)?.sync_all()?;
        Ok(created)
    })();
    let cleanup = fs::remove_file(&temporary);
    let created = publish?;
    cleanup?;
    ensure!(
        validate_existing_private_file(path, bytes)?,
        "sealed output validation failed after publication"
    );
    Ok(created)
}

fn freeze_sealed(args: FreezeSealedArgs) -> Result<Value> {
    let (catalog, catalog_sha256) = load_catalog(&args.catalog_pointer)?;
    let cohort = select_sealed_cohort(&catalog, &catalog_sha256)?;
    let mut bytes = canonical_json_bytes(&serde_json::to_value(&cohort)?)?;
    bytes.push(b'\n');
    let created = publish_private_immutable(&args.output, &bytes)?;
    Ok(json!({
        "schema": SEALED_SCHEMA,
        "immutable": true,
        "catalog_sha256": catalog_sha256,
        "output": args.output,
        "sealed_sha256": sha256_bytes(&bytes),
        // Report what was actually sealed, not a hard-coded two-task shape: which optional
        // carve-outs made it in is the signal that tells an operator whether multiclass and
        // timeseries have enough family-disjoint data yet.
        "datasets": cohort.entries.len(),
        "task_counts": jope_dime_regressor::selfplay_expansion::TASKS
            .iter()
            .map(|task| {
                (
                    (*task).to_owned(),
                    Value::from(
                        cohort
                            .entries
                            .iter()
                            .filter(|entry| entry.task == *task)
                            .count(),
                    ),
                )
            })
            .filter(|(_, count)| count.as_u64() != Some(0))
            .collect::<serde_json::Map<_, _>>(),
        "mode": "0600",
        "created": created,
        "identities_redacted": true,
    }))
}

fn load_checkpoint(pointer_path: &Path) -> Result<String> {
    let pointer: CheckpointPointer = serde_json::from_value(read_json(pointer_path)?)
        .with_context(|| format!("parse checkpoint pointer {}", pointer_path.display()))?;
    validate_sha256(&pointer.checkpoint_sha256, "checkpoint SHA-256")?;
    if let Some(path) = pointer.checkpoint {
        let path = resolve_pointer_path(pointer_path, &path);
        ensure!(path.is_file(), "checkpoint is absent: {}", path.display());
        ensure!(
            sha256_file(&path)? == pointer.checkpoint_sha256,
            "checkpoint pointer content drift"
        );
    }
    Ok(pointer.checkpoint_sha256)
}

fn load_sealed(path: &Path, catalog: &[CatalogEntry]) -> Result<(HashSet<String>, String)> {
    let digest =
        sha256_file(path).with_context(|| format!("hash sealed cohort {}", path.display()))?;
    let sealed: SealedCohort = serde_json::from_value(read_json(path)?)
        .with_context(|| format!("parse sealed cohort {}", path.display()))?;
    ensure!(
        SEALED_SCHEMAS.contains(&sealed.schema.as_str()),
        "unsupported sealed cohort schema"
    );
    let lookup = catalog
        .iter()
        .map(|row| {
            (
                (row.task.as_str(), row.dataset_hash.as_str()),
                row.source_family.as_str(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut identities = HashSet::new();
    let mut families = HashSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for row in sealed.entries {
        ensure!(
            matches!(
                row.task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            ),
            "sealed cohort has invalid task"
        );
        ensure!(
            identities.insert((row.task.clone(), row.dataset_hash.clone())),
            "sealed cohort repeats an identity"
        );
        let catalog_family = lookup
            .get(&(row.task.as_str(), row.dataset_hash.as_str()))
            .copied();
        let family = match (catalog_family, row.source_family.as_deref()) {
            (Some(catalog_family), Some(declared)) => {
                ensure!(
                    catalog_family == declared,
                    "sealed cohort source-family provenance drift"
                );
                declared
            }
            (Some(catalog_family), None) => catalog_family,
            (None, Some(declared)) if !declared.is_empty() => declared,
            (None, _) => {
                bail!("sealed identity absent from catalog and lacks source-family provenance")
            }
        };
        families.insert(family.to_owned());
        *counts.entry(row.task).or_default() += 1;
    }
    for task in ["regression", "binary"] {
        ensure!(
            counts.get(task).copied().unwrap_or(0) == REQUIRED_SEALED_PER_TASK,
            "sealed cohort must contain exactly 256 {task} datasets"
        );
    }
    Ok((families, digest))
}

/// Live per-dataset admission signals read from the strict corpus.
#[derive(Clone, Copy, Debug, Default)]
struct DatasetSignal {
    /// `best - p90` in the dataset's own metric.
    score_spread: f64,
    /// Distinct waves evaluated since the all-time best last improved.
    stale_waves: u32,
}

/// Head score spread per dataset: `best - p90`, in ONE index-only pass.
///
/// `observations_best_v1` is `(task,dataset_hash,score DESC,semantic_sha256)`,
/// so this walks the covering index in exactly the grouping order it needs and
/// never touches the 21 GB table. Measured 2026-07-28 on the live corpus:
/// 4.68 M rows, 18,649 datasets, 41 s in Python and well under that here.
fn dataset_spreads(connection: &Connection) -> Result<HashMap<(String, String), f64>> {
    let mut statement = connection.prepare(
        "SELECT task,dataset_hash,score FROM observations \
         ORDER BY task,dataset_hash,score DESC",
    )?;
    let mut rows = statement.query([])?;
    let mut result = HashMap::new();
    let mut key: Option<(String, String)> = None;
    let mut scores: Vec<f64> = Vec::new();
    // Both tasks score on bounded-ish metrics (r2, AUC), so a head spread above
    // 1.0 does not mean "very learnable" -- it means the p90 genome diverged.
    // Measured 2026-07-28 over all 18,649 datasets: 99.91% sit at or below 1.0
    // (regression p95 = 0.120, binary p95 = 0.167) and the tail runs to 2.99e28.
    // Left unclamped that tail would dominate every "widest spread first"
    // ordering -- the hard-uncertain lane and the parole reserve -- and hand
    // slots to numerically degenerate datasets. Winsorising at 1.0 touches 17
    // datasets and nothing else.
    const SPREAD_CEILING: f64 = 1.0;
    let flush = |key: &Option<(String, String)>,
                 scores: &mut Vec<f64>,
                 result: &mut HashMap<(String, String), f64>| {
        if let (Some(key), false) = (key.as_ref(), scores.is_empty()) {
            // scores arrive best-first; p90 is the 10th percentile from the top.
            let index = (scores.len() as f64 * 0.10) as usize;
            let spread = scores[0] - scores[index.min(scores.len() - 1)];
            result.insert(
                key.clone(),
                if spread.is_finite() {
                    spread.clamp(0.0, SPREAD_CEILING)
                } else {
                    0.0
                },
            );
        }
        scores.clear();
    };
    while let Some(row) = rows.next()? {
        let task: String = row.get(0)?;
        let dataset: String = row.get(1)?;
        let score: f64 = row.get(2)?;
        let current = (task, dataset);
        if key.as_ref() != Some(&current) {
            flush(&key, &mut scores, &mut result);
            key = Some(current);
        }
        if score.is_finite() {
            scores.push(score);
        }
    }
    flush(&key, &mut scores, &mut result);
    Ok(result)
}

/// Waves evaluated since each dataset's all-time best last improved, in ONE
/// index-only pass over `observations_wave_v1`
/// (`strict_wave,task,dataset_hash,score DESC`). Measured 2026-07-28: 32,917
/// (wave, dataset) groups over 18,285 datasets, 29 s in Python.
///
/// Wave chronology comes from the wave directories rather than the wave id,
/// because the coverage lane writes into the `strict-wave-9NNNNN` band and its
/// identities are NOT ordered against the strict lane's `strict-wave-0NNNNN`.
/// A wave with no directory sorts last (it is the newest thing there is).
fn dataset_staleness(
    connection: &Connection,
    wave_order: &HashMap<String, u64>,
) -> Result<HashMap<(String, String), u32>> {
    let mut statement = connection.prepare(
        "SELECT strict_wave,task,dataset_hash,MAX(score) FROM observations \
         GROUP BY strict_wave,task,dataset_hash",
    )?;
    let mut rows = statement.query([])?;
    let mut per: HashMap<(String, String), Vec<(u64, f64)>> = HashMap::new();
    while let Some(row) = rows.next()? {
        let Some(wave): Option<String> = row.get(0)? else {
            continue;
        };
        let task: String = row.get(1)?;
        let dataset: String = row.get(2)?;
        let best: f64 = row.get(3)?;
        if !best.is_finite() {
            continue;
        }
        let when = wave_order.get(&wave).copied().unwrap_or(u64::MAX);
        per.entry((task, dataset)).or_default().push((when, best));
    }
    let mut result = HashMap::with_capacity(per.len());
    for (key, mut history) in per {
        history.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.total_cmp(&right.1)));
        let mut best = f64::NEG_INFINITY;
        let mut best_index = 0_usize;
        for (index, (_, score)) in history.iter().enumerate() {
            if *score > best {
                best = *score;
                best_index = index;
            }
        }
        result.insert(key, (history.len() - 1 - best_index) as u32);
    }
    Ok(result)
}

/// Chronological rank of every wave that has a directory under `waves/` or
/// `coverage/`. Directory mtime is the only ordering that is correct across
/// both identity bands.
fn wave_chronology(output_root: &Path) -> HashMap<String, u64> {
    let mut stamped = Vec::new();
    let coverage = output_root
        .parent()
        .map(|root| root.join("coverage"))
        .unwrap_or_else(|| output_root.join("coverage"));
    for base in [output_root.to_path_buf(), coverage] {
        let Ok(entries) = fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.starts_with("strict-wave-") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_dir() {
                continue;
            }
            let when = meta
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|delta| delta.as_secs())
                .unwrap_or(0);
            stamped.push((when, name));
        }
    }
    stamped.sort();
    stamped
        .into_iter()
        .enumerate()
        .map(|(rank, (_, name))| (name, rank as u64))
        .collect()
}

/// Everything the admission policy needs, read once from the read-only corpus.
fn dataset_signals(
    corpus: &Path,
    output_root: &Path,
) -> Result<(
    HashMap<(String, String), u64>,
    HashMap<(String, String), DatasetSignal>,
)> {
    let connection = open_readonly(corpus)?;
    connection.busy_timeout(std::time::Duration::from_secs(180))?;
    let counts = {
        let mut statement = connection.prepare(
            "SELECT task,dataset_hash,COUNT(*) FROM observations GROUP BY task,dataset_hash",
        )?;
        let mut rows = statement.query([])?;
        let mut result = HashMap::new();
        while let Some(row) = rows.next()? {
            let task: String = row.get(0)?;
            let dataset: String = row.get(1)?;
            let count: i64 = row.get(2)?;
            ensure!(count >= 0, "negative corpus observation count");
            result.insert((task, dataset), count as u64);
        }
        result
    };
    let spreads = dataset_spreads(&connection)?;
    let stale = dataset_staleness(&connection, &wave_chronology(output_root))?;
    let mut signals = HashMap::with_capacity(counts.len());
    for key in counts.keys() {
        signals.insert(
            key.clone(),
            DatasetSignal {
                score_spread: spreads.get(key).copied().unwrap_or(0.0),
                stale_waves: stale.get(key).copied().unwrap_or(0),
            },
        );
    }
    Ok((counts, signals))
}

fn open_readonly(path: &Path) -> Result<Connection> {
    ensure!(path.is_file(), "corpus is absent: {}", path.display());
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("open corpus read-only {}", path.display()))
}

fn open_pair_registry_readonly(path: &Path) -> Result<Connection> {
    ensure!(
        path.is_file(),
        "pair registry is absent: {}",
        path.display()
    );
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .with_context(|| format!("open pair registry read-only {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    Ok(connection)
}

fn registry_pair_view(
    connection: &Connection,
    task: &str,
    dataset_hash: &str,
    protocol_sha256: &str,
) -> Result<RegistryPairView> {
    let mut statement = connection.prepare(
        "SELECT semantic_sha256,canonical_genome,state,receipt_sha256,job_id
         FROM pairs WHERE task=?1 AND dataset_hash=?2 AND protocol_sha256=?3",
    )?;
    let mut rows = statement.query(params![task, dataset_hash, protocol_sha256])?;
    let mut view = RegistryPairView::default();
    while let Some(row) = rows.next()? {
        let semantic_sha256: String = row.get(0)?;
        let canonical_genome: Option<String> = row.get(1)?;
        let state: String = row.get(2)?;
        let receipt_sha256: Option<String> = row.get(3)?;
        let job_id: Option<String> = row.get(4)?;
        validate_sha256(&semantic_sha256, "pair-registry semantic SHA-256")?;
        if let Some(genome) = canonical_genome {
            ensure!(
                semantic(&genome) == semantic_sha256,
                "pair registry semantic/canonical-genome drift"
            );
        }
        let reusable = state == "available" && receipt_sha256.is_none() && job_id.is_none();
        if !reusable {
            view.unavailable.insert(semantic_sha256);
        }
    }
    Ok(view)
}

fn next_wave_id(output_root: &Path) -> Result<String> {
    let mut maximum = 0_u64;
    if output_root.is_dir() {
        for entry in fs::read_dir(output_root)? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if let Some(index) = parse_wave_id(name) {
                maximum = maximum.max(index);
            }
        }
    }
    let pending = output_root.join("pending");
    if pending.is_dir() {
        for entry in fs::read_dir(pending)? {
            let path = entry?.path();
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let stem = stem.strip_suffix(".cohort").unwrap_or(stem);
            if let Some(index) = parse_wave_id(stem) {
                maximum = maximum.max(index);
            }
        }
    }
    ensure!(maximum < 999_999, "strict wave identity space exhausted");
    Ok(format!("strict-wave-{:06}", maximum + 1))
}

fn parse_wave_id(value: &str) -> Option<u64> {
    let digits = value.strip_prefix("strict-wave-")?;
    (digits.len() == 6 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn valid_proposal_reservoir_slot(slot: usize) -> bool {
    slot < MAX_PROPOSAL_RESERVOIR_PER_CONTEXT
}

fn unresolved_pending(output_root: &Path, requested: &str) -> Result<Option<PathBuf>> {
    let pending = output_root.join("pending");
    if !pending.is_dir() {
        return Ok(None);
    }
    for entry in fs::read_dir(pending)? {
        let path = entry?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value = read_json(&path)?;
        if value.get("schema").and_then(Value::as_str) != Some(COHORT_SCHEMA) {
            continue;
        }
        let wave = value
            .get("wave")
            .and_then(Value::as_str)
            .context("pending cohort lacks wave")?;
        if wave != requested && !output_root.join(wave).is_dir() {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn plan(args: PlanArgs) -> Result<Value> {
    let admission = args.admission();
    let (catalog, catalog_sha256) = load_catalog(&args.catalog_pointer)?;
    let checkpoint_sha256 = load_checkpoint(&args.checkpoint_pointer)?;
    let (sealed_families, sealed_sha256) = load_sealed(&args.sealed_cohort, &catalog.entries)?;
    let wave = args
        .wave_id
        .clone()
        .unwrap_or(next_wave_id(&args.output_root)?);
    ensure!(
        parse_wave_id(&wave).is_some(),
        "invalid strict wave identity"
    );
    if !args.dry_run {
        ensure!(
            !args.output_root.join(&wave).exists(),
            "immutable wave already exists"
        );
        if !args.allow_pending_lookahead {
            if let Some(path) = unresolved_pending(&args.output_root, &wave)? {
                bail!(
                    "another launch-ready cohort is unresolved: {}",
                    path.display()
                );
            }
        }
    }

    let (observed, signals) = dataset_signals(&args.corpus, &args.output_root)?;
    // Drop datasets the evaluator can provably never score. `corpus_eval` rejects
    // a feature matrix with no feature columns ("needs at least one feature and
    // one target", data.rs:130), so a single-column train.csv is a guaranteed
    // exit 1 for every candidate ever drawn against it -- 4 wasted evaluations
    // per wave slot, forever, because the dataset stays eligible and keeps being
    // reselected. On 2026-08-04 exactly 7 of 27,539 regression datasets were in
    // this state (their features live in sibling `.vs` files and only the target
    // was written to CSV); they produced 100 % of the 348 eval-job failures in a
    // two-hour window and nothing else did.
    //
    // This is deliberately filtered HERE rather than at catalog merge: merged
    // entries are positionally indexed into a strided context blob, so skipping
    // one at merge time would silently misalign every entry after it. Eligibility
    // is the first point where a dataset can be excluded without disturbing that
    // mapping or mutating an immutable catalog.
    let mut unevaluatable = Vec::new();
    let eligible = catalog
        .entries
        .iter()
        .filter(|row| !sealed_families.contains(&row.source_family))
        .filter(|row| {
            if dataset_has_feature_columns(&row.train_csv) {
                return true;
            }
            unevaluatable.push(format!("{}/{}", row.task, row.dataset_hash));
            false
        })
        .map(|row| MicroDataset {
            task: row.task.clone(),
            dataset_hash: row.dataset_hash.clone(),
            stable_anchor: row.stable_anchor,
            golden42: row.golden42,
            current_catalog: row.current_catalog,
            observation_count: row.observation_count,
            uncertainty: row.uncertainty,
            hardness: row.hardness,
            incumbent_regret: row.incumbent_regret,
            score_spread: signals
                .get(&(row.task.clone(), row.dataset_hash.clone()))
                .map(|signal| signal.score_spread),
            stale_waves: signals
                .get(&(row.task.clone(), row.dataset_hash.clone()))
                .map(|signal| signal.stale_waves),
        })
        .collect::<Vec<_>>();
    // Optional-task activation is an operator decision bounded by an
    // independent-dataset floor. The explicit environment opt-in alone must
    // not turn a 29-dataset multiclass sample into a shared-trunk training
    // distribution; until the floor is met, the task remains advisory.
    let requested_optional_tasks = std::env::var("JOPE_MICRO_COHORT_OPTIONAL_TASKS")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|task| !task.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let independent_dataset_counts = jope_dime_regressor::selfplay_expansion::TASKS
        .iter()
        .map(|task| {
            (
                (*task).to_owned(),
                eligible.iter().filter(|row| row.task == *task).count(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let active_optional_tasks = requested_optional_tasks
        .iter()
        .filter(|task| {
            independent_dataset_counts
                .get(*task)
                .copied()
                .unwrap_or(0)
                >= OPTIONAL_TASK_INDEPENDENT_DATASET_FLOOR
        })
        .cloned()
        .collect::<Vec<_>>();
    let selected =
        select_micro_cohort(&eligible, &observed, &wave, &admission, &active_optional_tasks)?;
    let admission_report = admission_summary(&eligible, &observed, &selected, &admission);
    if args.dry_run {
        return Ok(json!({
            "schema": COHORT_SCHEMA,
            "wave": wave,
            "dry_run": true,
            "independent_dataset_counts": independent_dataset_counts,
            "optional_task_floor": OPTIONAL_TASK_INDEPENDENT_DATASET_FLOOR,
            "requested_optional_tasks": requested_optional_tasks,
            "active_optional_tasks": active_optional_tasks,
            "admission": admission_report,
            "entries": selected
                .iter()
                .map(|row| json!({
                    "task": row.dataset.task,
                    "dataset_hash": row.dataset.dataset_hash,
                    "micro_lane": row.micro_lane,
                    "observation_count": observed
                        .get(&(row.dataset.task.clone(), row.dataset.dataset_hash.clone()))
                        .copied()
                        .unwrap_or(0),
                    "score_spread": row.dataset.score_spread,
                    "stale_waves": row.dataset.stale_waves,
                }))
                .collect::<Vec<_>>(),
        }));
    }
    // Declare exactly the tasks the selector actually laned, ordered by the canonical
    // `TASKS` array so the field is deterministic rather than dependent on selection order.
    // Deriving it from `selected` (instead of recomputing eligibility) guarantees the
    // declaration and the rows can never disagree, which is what `validate_cohort` checks.
    let cohort_tasks = jope_dime_regressor::selfplay_expansion::TASKS
        .iter()
        .filter(|task| selected.iter().any(|row| row.dataset.task == **task))
        .map(|task| (*task).to_owned())
        .collect::<Vec<_>>();
    let plan = CohortPlan {
        schema: COHORT_SCHEMA.to_owned(),
        immutable: true,
        wave: wave.clone(),
        checkpoint_epoch: checkpoint_epoch(&checkpoint_sha256)?,
        checkpoint_sha256,
        catalog_sha256,
        sealed_cohort_sha256: sealed_sha256,
        tasks: cohort_tasks,
        entries: selected
            .into_iter()
            .map(|row| CohortEntry {
                task: row.dataset.task,
                dataset_hash: row.dataset.dataset_hash,
                micro_lane: row.micro_lane,
            })
            .collect(),
    };
    validate_cohort(&plan)?;
    let path = args
        .output_root
        .join("pending")
        .join(format!("{wave}.cohort.json"));
    // The cohort this call is answering for: the adopted immutable file when one
    // already exists, otherwise the plan just written. Everything reported below
    // must describe THIS cohort, because that is what downstream stages resolve.
    let effective: CohortPlan;
    if path.exists() {
        // Adopt the immutable pending cohort; do NOT compare it against a fresh
        // derivation. `select_micro_cohort` ranks its `least_observed` lane by
        // live observation counts, and those grow as ingestion advances -- so
        // both the ORDER and the MEMBERSHIP of that lane change minutes after
        // the file is written. A byte-equality check against a re-derivation
        // can therefore never succeed again: the planner deadlocks permanently
        // and the queue starves behind a wave that can never be built.
        //
        // What must actually hold is the cohort's BINDING (built for this wave,
        // checkpoint, catalog and sealed receipt) and its STRUCTURE (validated
        // lanes over datasets drawn from the bound catalog). Both are checked
        // here; a violation is real drift and still fails.
        let current: CohortPlan = serde_json::from_value(read_json(&path)?)?;
        validate_cohort(&current)?;
        // NOTE: catalog_sha256 is deliberately NOT an equality check. The
        // expansion driver merges a new tranche and rewrites catalog/current.json
        // roughly hourly, so requiring the live catalog digest to still match the
        // one the cohort was planned against makes ANY cohort that outlives one
        // merge permanently unbuildable -- the queue then starves behind a wave
        // that can never be built. Growing the catalog must never invalidate work
        // already in flight. The cohort is a SUBSET of the catalog, so the
        // meaningful invariant is membership, enforced below: every entry must
        // still exist in the live catalog. A dataset that disappeared is real
        // drift and still fails.
        for (field, existing, expected) in [
            ("wave", &current.wave, &plan.wave),
            (
                "checkpoint",
                &current.checkpoint_sha256,
                &plan.checkpoint_sha256,
            ),
            (
                "sealed receipt",
                &current.sealed_cohort_sha256,
                &plan.sealed_cohort_sha256,
            ),
        ] {
            ensure!(
                existing == expected,
                "immutable pending cohort {field} drift: {existing} != {expected}"
            );
        }
        let catalogued = catalog
            .entries
            .iter()
            .map(|row| (row.task.as_str(), row.dataset_hash.as_str()))
            .collect::<HashSet<_>>();
        for entry in &current.entries {
            ensure!(
                catalogued.contains(&(entry.task.as_str(), entry.dataset_hash.as_str())),
                "immutable pending cohort dataset absent from the bound catalog: {}/{}",
                entry.task,
                entry.dataset_hash
            );
        }
        effective = current;
    } else {
        atomic_json(&path, &serde_json::to_value(&plan)?)?;
        effective = plan;
    }
    // Report the ADOPTED cohort's catalog binding, never the live one. The
    // adoption branch above deliberately tolerates a cohort that outlived a
    // catalog rotation (see the NOTE on membership-instead-of-equality), so
    // `plan.catalog_sha256` here is the digest of whatever `catalog/current.json`
    // happens to say right now -- which is precisely the binding the cohort on
    // disk is NOT scoped to. Supply persists this value into its state, the
    // proposer stamps it onto the pointer it emits, and `sanitize_proposals`
    // then compares that pointer against the frozen cohort's own
    // `catalog_sha256`. Leaking the live digest here therefore re-imposes the
    // exact equality this function disclaims, one stage later and as an
    // unrecoverable error: the cohort is immutable, so the two can never be made
    // to agree and the wave retries identically forever.
    Ok(json!({
        "schema": COHORT_SCHEMA,
        "wave": wave,
        "checkpoint_sha256": effective.checkpoint_sha256,
        "catalog_sha256": effective.catalog_sha256,
        "datasets": effective.entries.len(),
        "cohort": path,
        "cohort_sha256": sha256_file(&path)?,
        "admission": admission_report,
        // Reported, never silent: a dataset dropped from eligibility is a dataset
        // the flywheel will never observe again, and that must be visible.
        "unevaluatable_datasets_excluded": unevaluatable.len(),
        "unevaluatable_datasets_sample": unevaluatable.iter().take(16).collect::<Vec<_>>(),
        "proposal_generation_boundary": "run the PyTorch proposal generator for this cohort, then invoke build"
    }))
}

/// Does this training CSV carry at least one feature column alongside the target?
///
/// `corpus_eval` needs >= 2 columns; with one it bails before fitting anything, so
/// such a dataset can never yield an observation no matter which genome is drawn.
/// Reading the first line is enough to tell, and it is cheap: the header is one
/// short read per catalog row per wave plan, against files the page cache already
/// holds. A missing or unreadable file is treated as unevaluatable for the same
/// reason -- the evaluation would fail anyway, and failing to select is strictly
/// better than dispatching work that cannot succeed.
fn dataset_has_feature_columns(train_csv: &Path) -> bool {
    let Ok(file) = File::open(train_csv) else {
        return false;
    };
    let mut first = String::new();
    if BufReader::new(file).read_line(&mut first).is_err() {
        return false;
    }
    // Delimiter presence is the test, not a full parse: one column means no
    // separator on the first row.
    first.trim_end_matches(['\r', '\n']).contains(',')
}

/// Per-wave breadth accounting. This is the measurement surface for the
/// reallocation: how many cohort slots this wave spent on datasets that are at
/// or over the cap, and how many went to datasets with no observations at all.
fn admission_summary(
    eligible: &[MicroDataset],
    observed: &HashMap<(String, String), u64>,
    selected: &[jope_dime_regressor::selfplay_data::SelectedMicroDataset],
    admission: &MicroAdmission,
) -> Value {
    let count = |task: &str, dataset: &str| {
        observed
            .get(&(task.to_owned(), dataset.to_owned()))
            .copied()
            .unwrap_or(0)
    };
    let mut pool_total = 0_u64;
    let mut pool_capped = 0_u64;
    let mut pool_unseen = 0_u64;
    let mut pool_retired = 0_u64;
    let mut pool_retired_flat = 0_u64;
    let mut pool_retired_stale = 0_u64;
    for row in eligible {
        pool_total += 1;
        let observations = count(&row.task, &row.dataset_hash);
        if observations >= admission.eval_cap {
            pool_capped += 1;
            continue;
        }
        if observations == 0 {
            pool_unseen += 1;
            continue;
        }
        // Retirement only ever bites in the [retire_min_observations, eval_cap)
        // band -- everything above is already capped and everything below is
        // under-measured. Counting the two halves separately is the only way to
        // see the mechanism start to matter as the cap holds datasets in-band.
        if observations >= admission.retire_min_observations {
            let flat = row
                .score_spread
                .is_some_and(|spread| spread <= admission.retire_spread_floor);
            let stale = row.stale_waves.unwrap_or(0) >= admission.retire_stale_waves;
            if flat || stale {
                pool_retired += 1;
                pool_retired_flat += u64::from(flat);
                pool_retired_stale += u64::from(stale);
            }
        }
    }
    let mut lanes = BTreeMap::<String, (u64, u64, u64)>::new();
    for row in selected {
        let observations = count(&row.dataset.task, &row.dataset.dataset_hash);
        let key = format!(
            "{}/{}",
            row.dataset.task,
            serde_json::to_value(row.micro_lane)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned))
                .unwrap_or_else(|| "unknown".to_owned())
        );
        let slot = lanes.entry(key).or_default();
        slot.0 += 1;
        if observations >= admission.eval_cap {
            slot.1 += 1;
        }
        if observations == 0 {
            slot.2 += 1;
        }
    }
    let slots_capped: u64 = lanes.values().map(|slot| slot.1).sum();
    let slots_unseen: u64 = lanes.values().map(|slot| slot.2).sum();
    json!({
        "eval_cap": admission.eval_cap,
        "retire_min_observations": admission.retire_min_observations,
        "retire_stale_waves": admission.retire_stale_waves,
        "retire_spread_floor": admission.retire_spread_floor,
        "parole_period": admission.parole_period,
        "parole_slots": admission.parole_slots,
        "least_observed_band": admission.least_observed_band,
        "pool_datasets": pool_total,
        "pool_at_or_over_cap": pool_capped,
        "pool_never_observed": pool_unseen,
        "pool_retired": pool_retired,
        "pool_retired_flat_spread": pool_retired_flat,
        "pool_retired_stale_best": pool_retired_stale,
        "slots_at_or_over_cap": slots_capped,
        "slots_never_observed": slots_unseen,
        "lanes": lanes
            .into_iter()
            .map(|(key, slot)| (key, json!({
                "slots": slot.0,
                "at_or_over_cap": slot.1,
                "never_observed": slot.2,
            })))
            .collect::<serde_json::Map<_, _>>(),
    })
}

fn validate_cohort(plan: &CohortPlan) -> Result<()> {
    ensure!(plan.schema == COHORT_SCHEMA, "unsupported cohort schema");
    ensure!(plan.immutable, "cohort is not immutable");
    ensure!(parse_wave_id(&plan.wave).is_some(), "invalid cohort wave");
    validate_sha256(&plan.checkpoint_sha256, "cohort checkpoint SHA-256")?;
    ensure!(
        plan.checkpoint_epoch == checkpoint_epoch(&plan.checkpoint_sha256)?,
        "cohort checkpoint epoch drift"
    );
    validate_sha256(&plan.catalog_sha256, "cohort catalog SHA-256")?;
    validate_sha256(&plan.sealed_cohort_sha256, "cohort sealed receipt SHA-256")?;
    ensure!(!plan.tasks.is_empty(), "cohort declares no tasks");
    ensure!(
        plan.tasks
            .iter()
            .all(|task| jope_dime_regressor::selfplay_expansion::TASKS
                .contains(&task.as_str())),
        "cohort declares an unknown task"
    );
    ensure!(
        plan.tasks.iter().collect::<HashSet<_>>().len() == plan.tasks.len(),
        "cohort repeats a task"
    );
    // Regression and binary are the frozen backbone of every wave; a cohort that dropped
    // one of them would silently stop training the tasks that already work.
    for required in ["regression", "binary"] {
        ensure!(
            plan.tasks.iter().any(|task| task == required),
            "cohort omits the mandatory {required} lane"
        );
    }
    let expected_datasets = micro_wave_size(&plan.tasks);
    ensure!(
        plan.entries.len() == expected_datasets,
        "cohort must contain exactly {expected_datasets} datasets ({} tasks x {MICRO_TASK_SIZE})",
        plan.tasks.len()
    );
    ensure!(
        plan.entries
            .iter()
            .all(|row| plan.tasks.contains(&row.task)),
        "cohort carries a row for a task it does not declare"
    );
    let mut identities = HashSet::new();
    for task in plan.tasks.iter().map(String::as_str) {
        let task_rows = plan
            .entries
            .iter()
            .filter(|row| row.task == task)
            .collect::<Vec<_>>();
        ensure!(
            task_rows.len() == 64,
            "cohort must contain 64 {task} datasets"
        );
        for (lane, count) in [
            (MicroLane::StableAnchor, 16),
            (MicroLane::LeastObserved, 32),
            (MicroLane::HardUncertain, 16),
        ] {
            ensure!(
                task_rows
                    .iter()
                    .filter(|row| row.micro_lane == lane)
                    .count()
                    == count,
                "{task} cohort lane composition drift"
            );
        }
    }
    ensure!(
        plan.entries
            .iter()
            .all(|row| identities.insert((row.task.as_str(), row.dataset_hash.as_str()))),
        "cohort repeats a dataset identity"
    );
    Ok(())
}

fn sanitize_proposals(args: SanitizeProposalsArgs) -> Result<Value> {
    let plan: CohortPlan = serde_json::from_value(read_json(&args.cohort)?)
        .with_context(|| format!("parse cohort {}", args.cohort.display()))?;
    validate_cohort(&plan)?;
    let cohort_sha256 = sha256_file(&args.cohort)?;
    let source_pointer_value = read_json(&args.proposals_pointer)?;
    let pointer: ProposalsPointer = serde_json::from_value(source_pointer_value.clone())
        .with_context(|| {
            format!(
                "parse proposal pointer {}",
                args.proposals_pointer.display()
            )
        })?;
    validate_sha256(&pointer.proposals_sha256, "proposal file SHA-256")?;
    ensure!(
        pointer.checkpoint_sha256 == plan.checkpoint_sha256,
        "proposal pointer checkpoint differs from frozen cohort"
    );
    if source_pointer_value.get("schema").and_then(Value::as_str)
        == Some("CohortProposalPointer.v1")
    {
        ensure!(
            source_pointer_value
                .get("cohort_sha256")
                .and_then(Value::as_str)
                == Some(cohort_sha256.as_str())
                && source_pointer_value
                    .get("catalog_sha256")
                    .and_then(Value::as_str)
                    == Some(plan.catalog_sha256.as_str()),
            "cohort proposal pointer scope differs from the frozen cohort"
        );
    }
    let dl_slots_by_task = pointer.normalized_dl_slots(&plan.tasks)?;
    let source_path = resolve_pointer_path(&args.proposals_pointer, &pointer.proposals);
    ensure!(
        source_path.is_file(),
        "proposal file is absent: {}",
        source_path.display()
    );
    ensure!(
        sha256_file(&source_path)? == pointer.proposals_sha256,
        "proposal pointer content drift"
    );

    let cohort_keys = plan
        .entries
        .iter()
        .map(|entry| (entry.task.clone(), entry.dataset_hash.clone()))
        .collect::<HashSet<_>>();
    let mut local = HashMap::<(String, String), Vec<(usize, GpProposal)>>::new();
    let mut semantic_genomes = HashMap::<String, String>::new();
    for (ordinal, line) in BufReader::new(File::open(&source_path)?)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: GpProposal = serde_json::from_str(&line)
            .with_context(|| format!("parse proposal JSONL line {}", ordinal + 1))?;
        ensure!(
            matches!(
                row.task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            ) && !row.dataset_hash.is_empty()
                && valid_proposal_reservoir_slot(row.slot),
            "proposal file contains an invalid identity or slot at line {}",
            ordinal + 1
        );
        ensure!(
            row.model_checkpoint_sha256 == plan.checkpoint_sha256,
            "proposal file mixes checkpoint epochs at line {}",
            ordinal + 1
        );
        ensure!(
            semantic(&row.canonical_genome) == row.semantic_sha256,
            "proposal semantic identity drift at line {}",
            ordinal + 1
        );
        ensure!(
            row.score.into_iter().all(f64::is_finite)
                && row.proposal_score.into_iter().all(f64::is_finite),
            "proposal file has non-finite evidence at line {}",
            ordinal + 1
        );
        if let Some(existing) = semantic_genomes.get(&row.semantic_sha256) {
            ensure!(
                existing == &row.canonical_genome,
                "proposal semantic maps to conflicting genomes"
            );
        } else {
            semantic_genomes.insert(row.semantic_sha256.clone(), row.canonical_genome.clone());
        }
        let dataset_key = (row.task.clone(), row.dataset_hash.clone());
        if cohort_keys.contains(&dataset_key) && task_valid(&row.canonical_genome, &row.task) {
            local.entry(dataset_key).or_default().push((ordinal, row));
        }
    }

    let mut output_rows = Vec::with_capacity(plan.entries.len() * CHECKPOINT_PROPOSAL_SLOTS);
    let mut local_retained = BTreeMap::<String, usize>::new();
    let mut deficient_contexts = BTreeMap::<String, usize>::new();
    for cohort_entry in &plan.entries {
        let key = (cohort_entry.task.clone(), cohort_entry.dataset_hash.clone());
        let mut local_rows = local.remove(&key).unwrap_or_default();
        local_rows.sort_by_key(|(ordinal, row)| (row.slot, *ordinal));
        let mut used = HashSet::with_capacity(CHECKPOINT_PROPOSAL_SLOTS);
        for (_, row) in local_rows {
            if used.len() == CHECKPOINT_PROPOSAL_SLOTS {
                break;
            }
            if !used.insert(row.semantic_sha256.clone()) {
                continue;
            }
            let slot = used.len() - 1;
            output_rows.push(SanitizedProposal {
                task: row.task.clone(),
                dataset_hash: row.dataset_hash.clone(),
                canonical_genome: row.canonical_genome,
                semantic_sha256: row.semantic_sha256,
                model_checkpoint_sha256: row.model_checkpoint_sha256,
                slot,
                score: row.score,
                proposal_score: row.proposal_score,
                origin: "local_current_checkpoint",
                distinct_dataset_repeat_count: 1,
                source_task: row.task,
                source_dataset_hash: row.dataset_hash,
                source_slot: row.slot,
                retargeted: false,
                backfill_rank: None,
            });
            *local_retained.entry(cohort_entry.task.clone()).or_default() += 1;
        }
        if used.len() < CHECKPOINT_PROPOSAL_SLOTS {
            deficient_contexts.insert(
                format!("{}:{}", cohort_entry.task, cohort_entry.dataset_hash),
                used.len(),
            );
        }
    }

    let mut jsonl = Vec::new();
    for row in &output_rows {
        serde_json::to_writer(&mut jsonl, row)?;
        jsonl.push(b'\n');
    }
    let proposals_sha256 = sha256_bytes(&jsonl);
    let output_directory = args.output_root.join(&plan.wave);
    let proposals_path = output_directory.join(format!("{proposals_sha256}.jsonl"));
    publish_private_immutable(&proposals_path, &jsonl)?;
    let seed = source_pointer_value
        .get("seed")
        .or_else(|| source_pointer_value.get("proposal_seed"))
        .and_then(Value::as_u64)
        .unwrap_or(COHORT_PROPOSAL_SEED);
    let decoder_configuration_sha256 = sha256_bytes(&canonical_json_bytes(&json!({
        "generator": "test_exact_dl_generate.py",
        "grammar_masked_candidates_minimum": 96,
        "dl_slots_by_task": dl_slots_by_task,
        "selection_protocol": "reward-ranked-local-only-zero-cross-context-backfill-v2",
    }))?);
    let mut scope_counts = BTreeMap::<String, usize>::new();
    for entry in &plan.entries {
        *scope_counts.entry(entry.task.clone()).or_default() += 1;
    }
    let task_scopes = scope_counts
        .into_iter()
        .map(|(task, context_count)| {
            json!({
                "task": task,
                "accepted_checkpoint_sha256": plan.checkpoint_sha256,
                "context_count": context_count,
            })
        })
        .collect::<Vec<_>>();
    let proposal_job_id = source_pointer_value
        .get("proposal_job_id")
        .or_else(|| source_pointer_value.get("job_id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("recovery-source-{}", &pointer.proposals_sha256[..16]));
    let pointer_value = json!({
        "schema": "CohortProposalPointer.v1",
        "immutable": true,
        "wave": plan.wave,
        "task_scopes": task_scopes,
        "cohort": args.cohort,
        "cohort_sha256": cohort_sha256,
        "catalog_sha256": plan.catalog_sha256,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "decoder_configuration_sha256": decoder_configuration_sha256,
        "seed": seed,
        "proposal_job_id": proposal_job_id,
        "source_proposals_sha256": pointer.proposals_sha256,
        "proposals": proposals_path,
        "proposals_sha256": proposals_sha256,
        "rows": output_rows.len(),
        "datasets": plan.entries.len(),
        "proposals_per_dataset": CHECKPOINT_PROPOSAL_SLOTS,
        "dl_slots_by_task": dl_slots_by_task,
        "selection_protocol": "reward-ranked-local-only-zero-cross-context-backfill-v2",
        "cross_context_backfill_rows": 0,
        "summary": {
            "local_retained": local_retained,
            "deficient_contexts": deficient_contexts,
        },
    });
    let mut pointer_bytes = canonical_json_bytes(&pointer_value)?;
    pointer_bytes.push(b'\n');
    let pointer_sha256 = sha256_bytes(&pointer_bytes);
    let pointer_path = output_directory.join(format!("{pointer_sha256}.pointer.json"));
    publish_private_immutable(&pointer_path, &pointer_bytes)?;
    Ok(json!({
        "schema": "CohortProposalPointer.v1",
        "wave": plan.wave,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "cohort_sha256": cohort_sha256,
        "catalog_sha256": plan.catalog_sha256,
        "decoder_configuration_sha256": decoder_configuration_sha256,
        "seed": seed,
        "proposal_job_id": proposal_job_id,
        "task_scopes": task_scopes,
        "pointer": pointer_path,
        "pointer_sha256": pointer_sha256,
        "proposals": proposals_path,
        "proposals_sha256": proposals_sha256,
        "rows": output_rows.len(),
        "datasets": plan.entries.len(),
        "proposals_per_dataset": CHECKPOINT_PROPOSAL_SLOTS,
        "dl_slots_by_task": dl_slots_by_task,
        "cross_context_backfill_rows": 0,
        "local_retained": local_retained,
        "deficient_contexts": deficient_contexts,
    }))
}

fn load_proposals(
    pointer_path: &Path,
    plan: &CohortPlan,
) -> Result<HashMap<(String, String), Vec<GpProposal>>> {
    let pointer: ProposalsPointer = serde_json::from_value(read_json(pointer_path)?)
        .with_context(|| format!("parse proposal pointer {}", pointer_path.display()))?;
    validate_sha256(&pointer.proposals_sha256, "proposal file SHA-256")?;
    ensure!(
        pointer.checkpoint_sha256 == plan.checkpoint_sha256,
        "proposal pointer checkpoint differs from frozen cohort"
    );
    let path = resolve_pointer_path(pointer_path, &pointer.proposals);
    ensure!(
        path.is_file(),
        "proposal file is absent: {}",
        path.display()
    );
    ensure!(
        sha256_file(&path)? == pointer.proposals_sha256,
        "proposal pointer content drift"
    );
    let cohort = plan
        .entries
        .iter()
        .map(|row| (row.task.as_str(), row.dataset_hash.as_str()))
        .collect::<HashSet<_>>();
    let mut selected = HashMap::<(String, String), Vec<GpProposal>>::new();
    for (index, line) in BufReader::new(File::open(&path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: GpProposal = serde_json::from_str(&line)
            .with_context(|| format!("parse proposal JSONL line {}", index + 1))?;
        ensure!(
            row.model_checkpoint_sha256 == plan.checkpoint_sha256,
            "proposal file mixes checkpoint epochs at line {}",
            index + 1
        );
        ensure!(
            semantic(&row.canonical_genome) == row.semantic_sha256,
            "proposal semantic identity drift at line {}",
            index + 1
        );
        if cohort.contains(&(row.task.as_str(), row.dataset_hash.as_str())) {
            selected
                .entry((row.task.clone(), row.dataset_hash.clone()))
                .or_default()
                .push(row);
        }
    }
    for row in &plan.entries {
        let key = (row.task.clone(), row.dataset_hash.clone());
        let mut proposals = selected.get(&key).cloned().unwrap_or_default();
        proposals.sort_by_key(|proposal| proposal.slot);
        ensure!(
            proposals.len() <= CHECKPOINT_PROPOSAL_SLOTS,
            "{}:{} exceeds the direct-DL proposal budget",
            row.task,
            row.dataset_hash
        );
        let mut semantics = HashSet::new();
        for (slot, proposal) in proposals.iter().enumerate() {
            ensure!(
                proposal.task == row.task
                    && proposal.dataset_hash == row.dataset_hash
                    && proposal.model_checkpoint_sha256 == plan.checkpoint_sha256
                    && proposal.slot == slot,
                "{}:{} has nonlocal, mixed-checkpoint, or noncontiguous proposals",
                row.task,
                row.dataset_hash
            );
            ensure!(
                semantics.insert(proposal.semantic_sha256.as_str()),
                "{}:{} repeats a direct-DL semantic",
                row.task,
                row.dataset_hash
            );
        }
        ensure!(
            proposals
                .iter()
                .all(|proposal| task_valid(&proposal.canonical_genome, &row.task)),
            "{}:{} contains a task-invalid checkpoint proposal",
            row.task,
            row.dataset_hash
        );
        selected.insert(key, proposals);
    }
    Ok(selected)
}

fn load_robust(path: &Path) -> Result<(BTreeMap<String, Vec<RobustSeed>>, String)> {
    let digest = sha256_file(path)?;
    let mut bank: RobustSeedBank = serde_json::from_value(read_json(path)?)
        .with_context(|| format!("parse robust seed bank {}", path.display()))?;
    ensure!(
        bank.schema == ROBUST_BANK_SCHEMA,
        "unsupported robust seed-bank schema"
    );
    ensure!(
        bank.tasks.keys().all(|task| {
            matches!(
                task.as_str(),
                "regression" | "binary" | "timeseries" | "multiclass"
            )
        }) && bank.tasks.contains_key("regression")
            && bank.tasks.contains_key("binary"),
        "robust seed bank must contain regression/binary and only supported tasks"
    );
    for task in ["regression", "binary"] {
        let rows = bank.tasks.get_mut(task).expect("keys checked");
        let mut semantics = HashSet::new();
        for row in rows.iter() {
            ensure!(row.task == task, "robust seed task-map provenance drift");
            ensure!(
                row.dataset_count >= 2,
                "robust seed is not repeated across datasets"
            );
            ensure!(
                row.mean_within_dataset_percentile.is_finite()
                    && row.worst_within_dataset_percentile.is_finite(),
                "robust seed has non-finite rank evidence"
            );
            ensure!(
                semantic(&row.canonical_genome) == row.semantic_sha256,
                "robust seed semantic identity drift"
            );
            ensure!(
                semantics.insert(row.semantic_sha256.as_str()),
                "robust seed bank repeats a semantic"
            );
            ensure!(
                task_valid(&row.canonical_genome, task),
                "robust seed is task-invalid"
            );
        }
        rows.sort_by(|left, right| {
            right
                .mean_within_dataset_percentile
                .total_cmp(&left.mean_within_dataset_percentile)
                .then_with(|| {
                    right
                        .worst_within_dataset_percentile
                        .total_cmp(&left.worst_within_dataset_percentile)
                })
                .then_with(|| right.strict_win_count.cmp(&left.strict_win_count))
                .then_with(|| right.dataset_count.cmp(&left.dataset_count))
                .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
        });
        ensure!(
            rows.len() >= COMMON_WINNER_SLOTS,
            "{task} robust seed bank contains fewer than 96 repeated winners"
        );
    }
    Ok((bank.tasks, digest))
}

fn load_local_elites(corpus: &Path, task: &str, dataset: &str) -> Result<Vec<LocalElite>> {
    let connection = open_readonly(corpus)?;
    let mut statement = connection.prepare(
        "SELECT semantic_sha256,canonical_genome,score,secondary_score,\
                receipt_sha256,candidate_source,origin_class \
         FROM observations WHERE task=?1 AND dataset_hash=?2 \
         ORDER BY score DESC,\
           CASE WHEN ?1='binary' THEN COALESCE(secondary_score,1.0e300) ELSE 0 END ASC,\
           semantic_sha256 ASC",
    )?;
    let rows = statement.query_map(params![task, dataset], |row| {
        Ok(LocalElite {
            semantic_sha256: row.get(0)?,
            canonical_genome: row.get(1)?,
            score: row.get(2)?,
            secondary_score: row.get(3)?,
            receipt_sha256: row.get(4)?,
            candidate_source: row.get(5)?,
            dl_ancestry_kind: match row.get::<_, String>(6)?.as_str() {
                "gp_origin" => "none",
                "dl_origin" => "accepted_dl",
                "dl_descendant" => "mixed",
                _ => "mixed",
            }
            .to_owned(),
        })
    })?;
    let mut result = Vec::new();
    for row in rows {
        let row = row?;
        ensure!(
            row.score.is_finite() && row.secondary_score.into_iter().all(f64::is_finite),
            "local elite has non-finite score evidence"
        );
        ensure!(
            semantic(&row.canonical_genome) == row.semantic_sha256,
            "local elite semantic identity drift"
        );
        if task_valid(&row.canonical_genome, task) {
            result.push(row);
        }
    }
    Ok(result)
}

/// Deterministically re-head an offspring whose mutation left its task.
///
/// The mutation and crossover operators draw replacement heads from the FROZEN
/// evolve head list -- an RNG modulus that predates multiclass and timeseries
/// and must never change, or every historical regression/binary stream forks.
/// For those new tasks nearly every head-mutation therefore produces a child
/// whose terminal head fails `task_valid`, and 100,000 attempts could not fill
/// 125 offspring slots for the first multiclass wave. Grafting the LEFT
/// parent's terminal head back onto the child keeps the mutated transform
/// stack (the part the operator actually varied) while restoring a task-legal
/// head, with no randomness of its own. Callers gate this to tasks outside the
/// legacy pair so regression/binary offspring selection is byte-identical to
/// every wave ever built.
fn graft_task_head(child: &str, parent: &str) -> Option<String> {
    let GenomeIr::Pipeline(mut child_pipeline) = GenomeIr::parse(child).ok()? else {
        return None;
    };
    let GenomeIr::Pipeline(parent_pipeline) = GenomeIr::parse(parent).ok()? else {
        return None;
    };
    let parent_head = parent_pipeline.stages.last()?.clone();
    *child_pipeline.stages.last_mut()? = parent_head;
    Some(GenomeIr::Pipeline(child_pipeline).canonical())
}

fn task_valid(source: &str, task: &str) -> bool {
    let Ok(genome) = GenomeIr::parse(source) else {
        return false;
    };
    if genome.canonical() != source {
        return false;
    }
    let GenomeIr::Pipeline(pipeline) = genome else {
        return false;
    };
    let Some(head) = pipeline.stages.last().map(|stage| stage.op.as_str()) else {
        return false;
    };
    let is_model_head = |op: &str| {
        feat_math::heads::is_head_op(op)
            || matches!(
                op,
                "logit_ridge"
                    | "logistic"
                    | "tab_boost"
                    | "extra_trees"
                    | "gam_boost"
                    | "portfolio"
                    | "pls"
            )
    };
    if pipeline
        .stages
        .iter()
        .take(pipeline.stages.len().saturating_sub(1))
        .any(|stage| is_model_head(&stage.op))
    {
        return false;
    }
    match task {
        "binary" => matches!(
            head,
            "logit_ridge" | "logistic" | "tab_boost" | "extra_trees" | "gam_boost" | "portfolio"
        ),
        // Timeseries shares the regression allowlist because `corpus_eval` validates a
        // timeseries genome as exactly that: `TaskSpec::regression(Objective::Rmse)` against
        // the reduced schema (corpus_eval.rs:1682-1690). The v2 direct reduction has already
        // moved temporal structure into columns, so what remains is a regression fit. That
        // same branch bails on `logit_ridge`/`logistic` as "a binary classification head",
        // which is the exclusion the regression arm below already encodes -- so routing both
        // through one predicate keeps the wave and the evaluator from ever disagreeing.
        "regression" | "timeseries" => {
            head != "logit_ridge"
                && head != "logistic"
                && (feat_math::heads::is_head_op(head)
                    || matches!(head, "tab_boost" | "extra_trees" | "gam_boost" | "pls"))
        }
        // Only the four multiclass heads emit a K-class prediction. They are the same set
        // `corpus_eval` carries in V3_EXTRA_HEADS (:71-74) and the campaign pins as
        // MULTICLASS_HEADS (test_adaptive/multiclass60.rs:55-60); a binary or regression head
        // cannot satisfy `TaskSpec::multiclass(Objective::MacroF1)`.
        "multiclass" => matches!(
            head,
            "multiclass_ridge"
                | "multiclass_ovr_logistic"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
        ),
        _ => false,
    }
}

fn semantic(source: &str) -> String {
    sha256_bytes(source.as_bytes())
}

fn mutation_seed(wave: &str, task: &str, dataset: &str, attempt: usize) -> u64 {
    let digest = Sha256::digest(format!("{wave}|{task}|{dataset}|{attempt}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix is 8 bytes"))
}

fn registry_backfill_seed(
    wave: &str,
    task: &str,
    dataset: &str,
    logical: &Candidate,
    attempt: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!(
            "registry-backfill-v1|{wave}|{task}|{dataset}|{:?}|{}|{}|{attempt}",
            logical.slot_group, logical.slot_index, logical.semantic_sha256
        )
        .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix is 8 bytes"))
}

fn registry_filter_and_fill(
    wave: &str,
    task: &str,
    dataset: &str,
    _checkpoint: &str,
    logical: &[Candidate],
    registry: &RegistryPairView,
) -> Result<(Vec<Candidate>, usize)> {
    ensure!(
        logical.len() == WAVE_SIZE,
        "{task}:{dataset} logical wave composition is not 256"
    );
    let parents = logical
        .iter()
        .map(|candidate| {
            (
                candidate.canonical_genome.clone(),
                candidate.semantic_sha256.clone(),
            )
        })
        .collect::<Vec<_>>();
    let logical_semantics = logical
        .iter()
        .map(|candidate| candidate.semantic_sha256.as_str())
        .collect::<HashSet<_>>();
    let mut used = HashSet::with_capacity(WAVE_SIZE);
    let mut launchable = Vec::with_capacity(WAVE_SIZE);
    let mut filtered = 0;
    for candidate in logical {
        if !registry.unavailable.contains(&candidate.semantic_sha256)
            && used.insert(candidate.semantic_sha256.clone())
        {
            launchable.push(candidate.clone());
            continue;
        }
        filtered += 1;
        let mut replacement = None;
        for attempt in 0..MAX_MUTATION_ATTEMPTS {
            let seed = registry_backfill_seed(wave, task, dataset, candidate, attempt);
            let left = (seed as usize) % parents.len();
            let right = ((seed.rotate_left(23) as usize) + attempt) % parents.len();
            let child = if attempt % 2 == 0 {
                mutate_canonical_pipeline_v2(&parents[left].0, seed, 12, true)
            } else {
                crossover_canonical_pipelines(&parents[left].0, &parents[right].0, seed, 12)
            };
            let Ok(child) = child else {
                continue;
            };
            // Same frozen-mutator problem the offspring loop has: the evolve
            // head list predates multiclass/timeseries, so a mutated child
            // almost always carries an off-task head and 100_000 attempts can
            // retire without one legal candidate -- which wedged supply for
            // 3.5h on multiclass:06d990577f1565f5. Legacy tasks keep their
            // historical accept/reject stream byte-for-byte; the new tasks get
            // the same deterministic head graft used for offspring.
            let child = if task_valid(&child, task) {
                child
            } else if matches!(task, "multiclass" | "timeseries") {
                match graft_task_head(&child, &parents[left].0) {
                    Some(grafted) if task_valid(&grafted, task) => grafted,
                    _ => continue,
                }
            } else {
                continue;
            };
            let child_semantic = semantic(&child);
            if logical_semantics.contains(child_semantic.as_str())
                || registry.unavailable.contains(&child_semantic)
                || !used.insert(child_semantic.clone())
            {
                continue;
            }
            let mut parentage = vec![candidate.semantic_sha256.clone()];
            for semantic in candidate
                .parentage
                .iter()
                .chain([&parents[left].1, &parents[right].1])
            {
                if !parentage.contains(semantic) {
                    parentage.push(semantic.clone());
                }
            }
            replacement = Some(Candidate {
                canonical_genome: child,
                semantic_sha256: child_semantic,
                origin: CandidateOrigin::GpOffspring,
                source: "registry_aware_deterministic_backfill".to_owned(),
                slot_group: candidate.slot_group,
                slot_index: candidate.slot_index,
                model_checkpoint_sha256: None,
                parentage,
                evidence: Some(json!({
                    "replacement_reason": "exact_pair_authoritative_active_or_otherwise_unavailable",
                    "logical_source": candidate.source,
                    "logical_semantic_sha256": candidate.semantic_sha256,
                    "logical_evidence": candidate.evidence,
                    "attempt": attempt,
                    "seed": seed,
                })),
            });
            break;
        }
        launchable.push(replacement.with_context(|| {
            format!(
                "{task}:{dataset} could not deterministically backfill an unavailable exact pair"
            )
        })?);
    }
    // Registry replacement changes attribution, not merely the program bytes.
    // A blocked native proposal is replaced by GP and therefore reduces the
    // context's actual direct-DL allocation. Compact the surviving native slots
    // and move only those replaced DL slots into the offspring lane. Replaced
    // local/robust slots keep their budget group, but still retain GP origin and
    // never carry direct model attribution.
    let mut direct_index = 0;
    let mut offspring_index = 0;
    for candidate in &mut launchable {
        match candidate.slot_group {
            WaveSlotGroup::DlProposal if candidate.origin == CandidateOrigin::DlDirect => {
                candidate.slot_index = direct_index;
                direct_index += 1;
            }
            WaveSlotGroup::DlProposal => {
                candidate.slot_group = WaveSlotGroup::Offspring;
                candidate.slot_index = offspring_index;
                offspring_index += 1;
            }
            WaveSlotGroup::Offspring => {
                candidate.slot_index = offspring_index;
                offspring_index += 1;
            }
            WaveSlotGroup::LocalElite | WaveSlotGroup::CommonWinner => {}
        }
    }
    let slots = launchable
        .iter()
        .map(|candidate| WaveSlot {
            canonical_genome: candidate.canonical_genome.clone(),
            semantic_sha256: candidate.semantic_sha256.clone(),
            slot_group: candidate.slot_group,
            slot_index: candidate.slot_index,
        })
        .collect::<Vec<_>>();
    validate_wave_composition_with_dl(&slots, direct_index)?;
    ensure!(
        launchable
            .iter()
            .all(|candidate| !registry.unavailable.contains(&candidate.semantic_sha256)),
        "{task}:{dataset} registry filter leaked an unavailable exact pair"
    );
    Ok((launchable, filtered))
}

fn cache_filter_proposal_reservoir(
    proposals: &[GpProposal],
    registry: &RegistryPairView,
    requested_dl_slots: usize,
) -> (Vec<GpProposal>, usize) {
    let mut selected = Vec::with_capacity(requested_dl_slots);
    let mut resampled = 0;
    for proposal in proposals {
        if registry.unavailable.contains(&proposal.semantic_sha256) {
            resampled += 1;
            continue;
        }
        let mut proposal = proposal.clone();
        proposal.slot = selected.len();
        selected.push(proposal);
        if selected.len() == requested_dl_slots {
            break;
        }
    }
    (selected, resampled)
}

fn extend_candidate_evidence(candidate: &mut Candidate, fields: &[(&str, Value)]) -> Result<()> {
    let mut evidence = candidate
        .evidence
        .take()
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    let object = evidence
        .as_object_mut()
        .context("candidate evidence must be a JSON object")?;
    for (key, value) in fields {
        object.insert((*key).to_owned(), value.clone());
    }
    candidate.evidence = Some(evidence);
    Ok(())
}

fn annotate_matched_evaluation_lanes(
    candidates: &mut [Candidate],
    actual_d: usize,
    accepted_parent_checkpoint_sha256: &str,
    wave: &str,
    task: &str,
    context_identity: &str,
) -> Result<(usize, usize)> {
    let direct_semantics = candidates
        .iter()
        .filter(|candidate| candidate.origin == CandidateOrigin::DlDirect)
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    ensure!(
        direct_semantics.len() == actual_d,
        "native direct-DL accounting drift before matched-lane selection"
    );

    let mut direct_index = 0;
    for candidate in candidates
        .iter_mut()
        .filter(|candidate| candidate.origin == CandidateOrigin::DlDirect)
    {
        let pair_id = causal_pair_identity(
            accepted_parent_checkpoint_sha256,
            wave,
            task,
            context_identity,
            direct_index,
        );
        extend_candidate_evidence(
            candidate,
            &[
                ("evaluation_lane", json!("native_dl_direct")),
                ("matched_pair_index", json!(direct_index)),
                ("pair_id", json!(pair_id)),
                ("proposal_valid", json!(true)),
                ("dl_ancestry_kind", json!("accepted_dl")),
                ("valid_lineage", json!(true)),
            ],
        )?;
        direct_index += 1;
    }

    let matched_indexes = candidates
        .iter()
        .enumerate()
        .filter(|(_, candidate)| {
            candidate.origin != CandidateOrigin::DlDirect
                && candidate.model_checkpoint_sha256.is_none()
                && candidate
                    .parentage
                    .iter()
                    .all(|parent| !direct_semantics.contains(parent))
                && candidate
                    .evidence
                    .as_ref()
                    .and_then(|evidence| evidence["dl_ancestry_kind"].as_str())
                    == Some("none")
        })
        .take(actual_d)
        .map(|(index, _)| index)
        .collect::<HashSet<_>>();
    ensure!(
        matched_indexes.len() == actual_d,
        "context cannot fill its matched no-DL-ancestry GP lane: requested {actual_d}, found {}",
        matched_indexes.len()
    );

    let mut matched_index = 0;
    let mut exploration_count = 0;
    for (index, candidate) in candidates.iter_mut().enumerate() {
        if candidate.origin == CandidateOrigin::DlDirect {
            continue;
        }
        if matched_indexes.contains(&index) {
            let pair_id = causal_pair_identity(
                accepted_parent_checkpoint_sha256,
                wave,
                task,
                context_identity,
                matched_index,
            );
            extend_candidate_evidence(
                candidate,
                &[
                    ("evaluation_lane", json!("matched_gp_comparator")),
                    ("matched_pair_index", json!(matched_index)),
                    ("pair_id", json!(pair_id)),
                    ("zero_dl_ancestry", json!(true)),
                    ("proposal_valid", json!(true)),
                    ("dl_ancestry_kind", json!("none")),
                    ("valid_lineage", json!(true)),
                ],
            )?;
            matched_index += 1;
        } else {
            extend_candidate_evidence(candidate, &[("evaluation_lane", json!("gp_exploration"))])?;
            exploration_count += 1;
        }
    }
    ensure!(
        matched_index == actual_d && exploration_count + matched_index + actual_d == WAVE_SIZE,
        "matched/native/exploration lane accounting drift"
    );
    Ok((matched_index, exploration_count))
}

fn causal_pair_identity(
    accepted_parent_checkpoint_sha256: &str,
    wave: &str,
    task: &str,
    context_identity: &str,
    pair_index: usize,
) -> String {
    let wave_number = wave
        .strip_prefix("strict-wave-")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or_default();
    sha256_bytes(
        format!(
            "CausalPair.v1\0{accepted_parent_checkpoint_sha256}\0{wave_number}\0{task}\0{context_identity}\0{pair_index}"
        )
        .as_bytes(),
    )
}

#[cfg(test)]
fn compose_dataset(
    wave: &str,
    task: &str,
    dataset: &str,
    checkpoint: &str,
    proposals: &[GpProposal],
    local: &[LocalElite],
    robust: &[RobustSeed],
) -> Result<Vec<Candidate>> {
    compose_dataset_with_dl(
        wave,
        task,
        dataset,
        checkpoint,
        proposals,
        local,
        robust,
        CHECKPOINT_PROPOSAL_SLOTS,
    )
}

fn compose_dataset_with_dl(
    wave: &str,
    task: &str,
    dataset: &str,
    checkpoint: &str,
    proposals: &[GpProposal],
    local: &[LocalElite],
    robust: &[RobustSeed],
    requested_dl_slots: usize,
) -> Result<Vec<Candidate>> {
    validate_dl_slot_rung(requested_dl_slots)?;
    ensure!(
        proposals.len() <= CHECKPOINT_PROPOSAL_SLOTS,
        "checkpoint proposal count exceeds the decoder budget"
    );
    let direct_dl_slots = proposals.len().min(requested_dl_slots);
    let mut used = HashSet::with_capacity(WAVE_SIZE);
    let mut selected = Vec::with_capacity(WAVE_SIZE);

    for proposal in proposals.iter().take(direct_dl_slots) {
        ensure!(
            used.insert(proposal.semantic_sha256.clone()),
            "checkpoint proposals repeat semantics"
        );
        selected.push(Candidate {
            canonical_genome: proposal.canonical_genome.clone(),
            semantic_sha256: proposal.semantic_sha256.clone(),
            origin: CandidateOrigin::DlDirect,
            source: "current_checkpoint_dl_proposal".to_owned(),
            slot_group: WaveSlotGroup::DlProposal,
            slot_index: proposal.slot,
            model_checkpoint_sha256: Some(checkpoint.to_owned()),
            parentage: Vec::new(),
            evidence: proposal
                .proposal_score
                .or(proposal.score)
                .map(|score| json!({"proposal_score": score})),
        });
    }

    let mut local_count = 0;
    for row in local {
        if !used.insert(row.semantic_sha256.clone()) {
            continue;
        }
        selected.push(Candidate {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            origin: CandidateOrigin::LocalElite,
            source: "dataset_local_elite".to_owned(),
            slot_group: WaveSlotGroup::LocalElite,
            slot_index: local_count,
            model_checkpoint_sha256: None,
            parentage: Vec::new(),
            evidence: Some(json!({
                "score": row.score,
                "secondary_score": row.secondary_score,
                "receipt_sha256": row.receipt_sha256,
                "candidate_source": row.candidate_source,
                "dl_ancestry_kind": row.dl_ancestry_kind,
            })),
        });
        local_count += 1;
        if local_count == LOCAL_ELITE_SLOTS {
            break;
        }
    }
    for row in robust {
        if local_count == LOCAL_ELITE_SLOTS {
            break;
        }
        if !used.insert(row.semantic_sha256.clone()) {
            continue;
        }
        selected.push(Candidate {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            origin: CandidateOrigin::RobustWinner,
            source: "cold_start_repeated_robust_gp_surrogate".to_owned(),
            slot_group: WaveSlotGroup::LocalElite,
            slot_index: local_count,
            model_checkpoint_sha256: None,
            parentage: Vec::new(),
            evidence: Some(json!({
                "surrogate_for": "dataset_local_elite",
                "cold_start": true,
                "distinct_dataset_count": row.dataset_count,
                "mean_within_dataset_percentile": row.mean_within_dataset_percentile,
                "worst_within_dataset_percentile": row.worst_within_dataset_percentile,
                "strict_win_count": row.strict_win_count,
                "dl_ancestry_kind": if row.dl_origin { "accepted_dl" } else { "none" },
            })),
        });
        local_count += 1;
    }
    ensure!(
        local_count == LOCAL_ELITE_SLOTS,
        "{task}:{dataset} cannot fill 32 local-elite slots from local evidence plus robust cold-start surrogates"
    );

    let mut robust_count = 0;
    for row in robust {
        if !used.insert(row.semantic_sha256.clone()) {
            continue;
        }
        selected.push(Candidate {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            origin: CandidateOrigin::RobustWinner,
            source: "cross_dataset_repeated_robust_gp_winner".to_owned(),
            slot_group: WaveSlotGroup::CommonWinner,
            slot_index: robust_count,
            model_checkpoint_sha256: None,
            parentage: Vec::new(),
            evidence: Some(json!({
                "distinct_dataset_count": row.dataset_count,
                "mean_within_dataset_percentile": row.mean_within_dataset_percentile,
                "worst_within_dataset_percentile": row.worst_within_dataset_percentile,
                "strict_win_count": row.strict_win_count,
                "dl_ancestry_kind": if row.dl_origin { "accepted_dl" } else { "none" },
            })),
        });
        robust_count += 1;
        if robust_count == COMMON_WINNER_SLOTS {
            break;
        }
    }
    ensure!(
        robust_count == COMMON_WINNER_SLOTS,
        "{task}:{dataset} has fewer than 96 unique repeated robust winners after deduplication"
    );

    let parents = selected
        .iter()
        .filter(|row| row.origin != CandidateOrigin::DlDirect)
        .map(|row| {
            (
                row.canonical_genome.clone(),
                row.semantic_sha256.clone(),
                row.evidence
                    .as_ref()
                    .and_then(|evidence| evidence["dl_ancestry_kind"].as_str())
                    .unwrap_or("mixed")
                    .to_owned(),
            )
        })
        .collect::<Vec<_>>();
    let offspring_slots = CHECKPOINT_PROPOSAL_SLOTS + OFFSPRING_SLOTS - direct_dl_slots;
    let mut offspring_count = 0;
    for attempt in 0..MAX_MUTATION_ATTEMPTS {
        let seed = mutation_seed(wave, task, dataset, attempt);
        let left = (seed as usize) % parents.len();
        let right = ((seed.rotate_left(23) as usize) + attempt) % parents.len();
        let child = if attempt % 2 == 0 {
            mutate_canonical_pipeline_v2(&parents[left].0, seed, 12, true)
        } else {
            crossover_canonical_pipelines(&parents[left].0, &parents[right].0, seed, 12)
        };
        let Ok(child) = child else {
            continue;
        };
        // Legacy tasks keep the historical accept/reject stream exactly; the
        // new tasks (no history to fork) get a deterministic head graft when
        // the frozen mutator swapped their head off-task. See graft_task_head.
        let child = if task_valid(&child, task) {
            child
        } else if matches!(task, "multiclass" | "timeseries") {
            match graft_task_head(&child, &parents[left].0) {
                Some(grafted) if task_valid(&grafted, task) => grafted,
                _ => continue,
            }
        } else {
            continue;
        };
        let child_semantic = semantic(&child);
        if !used.insert(child_semantic.clone()) {
            continue;
        }
        selected.push(Candidate {
            canonical_genome: child,
            semantic_sha256: child_semantic,
            origin: CandidateOrigin::GpOffspring,
            source: "deterministic_rust_mutation_crossover".to_owned(),
            slot_group: WaveSlotGroup::Offspring,
            slot_index: offspring_count,
            model_checkpoint_sha256: None,
            parentage: vec![parents[left].1.clone(), parents[right].1.clone()],
            evidence: Some(json!({
                "attempt": attempt,
                "seed": seed,
                "dl_ancestry_kind": if parents[left].2 == "none" && parents[right].2 == "none" {
                    "none"
                } else if parents[left].2 == parents[right].2 {
                    parents[left].2.as_str()
                } else {
                    "mixed"
                },
            })),
        });
        offspring_count += 1;
        if offspring_count == offspring_slots {
            break;
        }
    }
    ensure!(
        offspring_count == offspring_slots,
        "{task}:{dataset} could not produce {offspring_slots} unique task-valid Rust offspring"
    );

    // Stable manifest order is the specified 32/64/96/64 composition.
    selected.sort_by_key(|row| {
        let group = match row.slot_group {
            WaveSlotGroup::LocalElite => 0,
            WaveSlotGroup::DlProposal => 1,
            WaveSlotGroup::CommonWinner => 2,
            WaveSlotGroup::Offspring => 3,
        };
        (group, row.slot_index)
    });
    let slots = selected
        .iter()
        .map(|row| WaveSlot {
            canonical_genome: row.canonical_genome.clone(),
            semantic_sha256: row.semantic_sha256.clone(),
            slot_group: row.slot_group,
            slot_index: row.slot_index,
        })
        .collect::<Vec<_>>();
    validate_wave_composition_with_dl(&slots, direct_dl_slots)?;
    Ok(selected)
}

fn validate_selected_files(entry: &CatalogEntry) -> Result<()> {
    ensure!(
        entry.train_csv.is_file() && entry.test_csv.is_file(),
        "{}:{} train/test data are absent",
        entry.task,
        entry.dataset_hash
    );
    ensure!(
        sha256_file(&entry.train_csv)? == entry.train_sha256,
        "{}:{} train data hash drift",
        entry.task,
        entry.dataset_hash
    );
    ensure!(
        sha256_file(&entry.test_csv)? == entry.test_sha256,
        "{}:{} test data hash drift",
        entry.task,
        entry.dataset_hash
    );
    Ok(())
}

fn build(args: BuildArgs) -> Result<Value> {
    ensure!(
        (1..=100).contains(&args.max_candidates_per_manifest),
        "manifest candidate cap must be in 1..=100"
    );
    let plan: CohortPlan = serde_json::from_value(read_json(&args.cohort)?)
        .with_context(|| format!("parse cohort {}", args.cohort.display()))?;
    validate_cohort(&plan)?;
    // Datasets in THIS cohort. `validate_cohort` has already proved it equals
    // `plan.entries.len()`, so downstream pair-count arithmetic tracks a 192- or 256-dataset
    // wave instead of assuming the two-task 128.
    let cohort_datasets = micro_wave_size(&plan.tasks);
    let (catalog, catalog_sha256) = load_catalog(&args.catalog_pointer)?;
    // Catalog GROWTH must not invalidate a frozen cohort. The expansion driver
    // merges a tranche and rewrites catalog/current.json roughly hourly, so a
    // digest-equality check here means any cohort that outlives one merge can
    // never be built -- the queue then starves behind a wave that is stuck
    // forever. (Observed 2026-07-28: supply looped on this while the cluster sat
    // empty.) The cohort is a SUBSET of the catalog, so the real invariant is
    // membership: every dataset the cohort names must still be present. A
    // dataset that vanished is genuine drift and still fails loudly below.
    let catalogued = catalog
        .entries
        .iter()
        .map(|row| (row.task.as_str(), row.dataset_hash.as_str()))
        .collect::<HashSet<_>>();
    let missing = plan
        .entries
        .iter()
        .filter(|entry| !catalogued.contains(&(entry.task.as_str(), entry.dataset_hash.as_str())))
        .count();
    ensure!(
        missing == 0,
        "frozen cohort names {missing} dataset(s) absent from the live catalog \
         (cohort {}, live {catalog_sha256})",
        plan.catalog_sha256
    );
    let (sealed_families, sealed_sha256) = load_sealed(&args.sealed_cohort, &catalog.entries)?;
    ensure!(
        sealed_sha256 == plan.sealed_cohort_sha256,
        "sealed cohort differs from frozen cohort"
    );
    let proposals = load_proposals(&args.proposals_pointer, &plan)?;
    let proposal_pointer: ProposalsPointer =
        serde_json::from_value(read_json(&args.proposals_pointer)?)?;
    let requested_dl_slots_by_task = proposal_pointer.normalized_dl_slots(&plan.tasks)?;
    let (robust_by_task, robust_sha256) = load_robust(&args.robust_seed_bank)?;
    let pair_registry = open_pair_registry_readonly(&args.pair_registry)?;
    let final_directory = args.output_root.join(&plan.wave);
    ensure!(
        !final_directory.exists(),
        "immutable wave already exists: {}",
        final_directory.display()
    );

    let catalog_by_identity = catalog
        .entries
        .iter()
        .map(|row| ((row.task.clone(), row.dataset_hash.clone()), row))
        .collect::<HashMap<_, _>>();
    for row in &plan.entries {
        let entry = catalog_by_identity
            .get(&(row.task.clone(), row.dataset_hash.clone()))
            .with_context(|| {
                format!(
                    "cohort identity absent from catalog {}:{}",
                    row.task, row.dataset_hash
                )
            })?;
        ensure!(
            !sealed_families.contains(&entry.source_family),
            "cohort exposes a sealed source family"
        );
        validate_selected_files(entry)?;
    }

    let temporary = args
        .output_root
        .join(format!(".{}.tmp-{}", plan.wave, std::process::id()));
    ensure!(
        !temporary.exists(),
        "temporary wave path already exists: {}",
        temporary.display()
    );
    fs::create_dir_all(temporary.join("manifests"))?;
    let generated_at_ms = now_ms();
    let mut receipts = Vec::new();
    let mut registry_filtered_pairs = 0_usize;
    let mut proposal_cache_resampled = 0_usize;
    let mut proposal_reservoir_considered = 0_usize;
    let mut native_direct_count = 0_usize;
    let mut matched_gp_count = 0_usize;
    let mut gp_exploration_count = 0_usize;
    let mut thin_contexts = BTreeMap::<String, usize>::new();
    let mut observed_dl_slots_by_task = BTreeMap::<String, BTreeSet<usize>>::new();
    let mut cohort_order = plan.entries.clone();
    cohort_order.sort_by(|left, right| {
        left.task
            .cmp(&right.task)
            .then_with(|| left.dataset_hash.cmp(&right.dataset_hash))
    });

    for (dataset_ordinal, cohort_row) in cohort_order.iter().enumerate() {
        let key = (cohort_row.task.clone(), cohort_row.dataset_hash.clone());
        let entry = catalog_by_identity[&key];
        let local = load_local_elites(&args.corpus, &cohort_row.task, &cohort_row.dataset_hash)?;
        // Ask the one function that knows all four metrics rather than assuming anything
        // that is not regression is binary. The old ternary stamped "auc" onto timeseries
        // and multiclass rows, and because the protocol sha256 is derived from (task,
        // metric) the shard then failed closed against the evaluator's `negative_mase` /
        // `macro_f1` -- a silent-looking wave that could never produce an observation.
        let metric = expected_evaluation_metric(&cohort_row.task)?;
        let protocol_sha256 = evaluation_protocol_sha256(
            &cohort_row.task,
            metric,
            "test_exact",
            &entry.train_sha256,
            &entry.test_sha256,
        )?;
        let registry = registry_pair_view(
            &pair_registry,
            &cohort_row.task,
            &cohort_row.dataset_hash,
            &protocol_sha256,
        )?;
        let requested_dl_slots = requested_dl_slots_by_task[&cohort_row.task];
        let (available_proposals, resampled) =
            cache_filter_proposal_reservoir(&proposals[&key], &registry, requested_dl_slots);
        proposal_cache_resampled += resampled;
        proposal_reservoir_considered += resampled + available_proposals.len();
        if available_proposals.len() < requested_dl_slots {
            thin_contexts.insert(
                format!("{}:{}", cohort_row.task, cohort_row.dataset_hash),
                available_proposals.len(),
            );
        }
        // Fail closed with a message, never panic: a task can be catalogued,
        // cohort-eligible, and proposal-backed while its robust seed bank does
        // not exist yet (a brand-new task has zero observations, so it cannot
        // have repeated cross-dataset winners). Composition needs 32 local-
        // surrogate + 96 common-winner rows from this bank, so the task cannot
        // build waves until a bank is bootstrapped -- say so.
        let robust_rows = robust_by_task.get(&cohort_row.task).with_context(|| {
            format!(
                "{}: robust seed bank has no entry for this task; a cold task needs \
                 a bootstrapped bank (>= {} unique repeated winners) before its \
                 wave lane can open",
                cohort_row.task,
                COMMON_WINNER_SLOTS + LOCAL_ELITE_SLOTS
            )
        })?;
        let logical_candidates = compose_dataset_with_dl(
            &plan.wave,
            &cohort_row.task,
            &cohort_row.dataset_hash,
            &plan.checkpoint_sha256,
            &available_proposals,
            &local,
            robust_rows,
            requested_dl_slots,
        )?;
        let (mut candidates, filtered) = registry_filter_and_fill(
            &plan.wave,
            &cohort_row.task,
            &cohort_row.dataset_hash,
            &plan.checkpoint_sha256,
            &logical_candidates,
            &registry,
        )?;
        registry_filtered_pairs += filtered;
        let actual_dl_slots = candidates
            .iter()
            .filter(|candidate| candidate.origin == CandidateOrigin::DlDirect)
            .count();
        ensure!(
            actual_dl_slots == available_proposals.len(),
            "{}:{} direct-DL proposal changed after cache resampling",
            cohort_row.task,
            cohort_row.dataset_hash
        );
        let (matched, exploration) = annotate_matched_evaluation_lanes(
            &mut candidates,
            actual_dl_slots,
            &plan.checkpoint_sha256,
            &plan.wave,
            &cohort_row.task,
            &cohort_row.dataset_hash,
        )?;
        native_direct_count += actual_dl_slots;
        matched_gp_count += matched;
        gp_exploration_count += exploration;
        observed_dl_slots_by_task
            .entry(cohort_row.task.clone())
            .or_default()
            .insert(actual_dl_slots);
        let frozen_seed_sha256 = sha256_bytes(&canonical_json_bytes(&json!({
            "wave": plan.wave,
            "task": cohort_row.task,
            "dataset_hash": cohort_row.dataset_hash,
            "semantics": candidates.iter().map(|row| row.semantic_sha256.as_str()).collect::<Vec<_>>(),
        }))?);
        let authority = json!({
            "score_tier": "test_exact",
            "metric": metric,
            "test_adaptive": true,
            "benchmark_valid": false,
            "selection_bias": "test-exact-reused",
            "generation": plan.wave,
            "frozen_seed_sha256": frozen_seed_sha256,
            "model_checkpoint_sha256": plan.checkpoint_sha256,
            "hall_of_fame_sha256": robust_sha256,
        });
        // Multiclass and timeseries shards pin the dataset's meta.json. A
        // timeseries shard additionally carries the complete v2 contract and
        // an explicit builder-owned reduction policy; no policy is inferred
        // from metadata or evaluator defaults.
        let (
            dataset_meta_json,
            dataset_meta_sha256,
            horizon,
            seasonality,
            frequency,
            target_is_final_column,
            known_future_covariates,
            past_only_covariates,
            reduction,
        ) = if cohort_row.task == "timeseries" {
            let contract = timeseries_manifest_contract(&entry.train_csv).with_context(|| {
                format!(
                    "{}:{} timeseries shard contract",
                    cohort_row.task, cohort_row.dataset_hash
                )
            })?;
            (
                Some(contract.0.to_string_lossy().into_owned()),
                Some(contract.1),
                Some(contract.2),
                Some(contract.3),
                Some(contract.4),
                Some(contract.5),
                contract.6,
                contract.7,
                Some(contract.8),
            )
        } else if cohort_row.task == "multiclass" {
            let meta_path = entry
                .train_csv
                .parent()
                .with_context(|| {
                    format!("{} train_csv has no parent directory", cohort_row.task)
                })?
                .join("meta.json");
            let sha = sha256_file(&meta_path).with_context(|| {
                format!(
                    "{}:{} shard meta.json is absent beside the catalogued train.csv",
                    cohort_row.task, cohort_row.dataset_hash
                )
            })?;
            (
                Some(meta_path.to_string_lossy().into_owned()),
                Some(sha),
                None,
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                None,
            )
        } else {
            (
                None,
                None,
                None,
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                None,
            )
        };
        for (shard, chunk) in candidates
            .chunks(args.max_candidates_per_manifest)
            .enumerate()
        {
            let manifest = Manifest {
                schema: if cohort_row.task == "timeseries" {
                    MANIFEST_SCHEMA_V2
                } else {
                    MANIFEST_SCHEMA
                },
                evaluation_lane: "adaptive_selfplay",
                strict_wave: plan.wave.clone(),
                generation: plan.wave.clone(),
                micro_lane: cohort_row.micro_lane,
                task: cohort_row.task.clone(),
                dataset_hash: cohort_row.dataset_hash.clone(),
                checkpoint_epoch: plan.checkpoint_epoch.clone(),
                checkpoint_sha256: plan.checkpoint_sha256.clone(),
                catalog_sha256: plan.catalog_sha256.clone(),
                protocol_sha256: protocol_sha256.clone(),
                generated_at_ms,
                train_csv: entry.train_csv.clone(),
                test_csv: entry.test_csv.clone(),
                train_sha256: entry.train_sha256.clone(),
                test_sha256: entry.test_sha256.clone(),
                meta_json: dataset_meta_json.clone(),
                meta_sha256: dataset_meta_sha256.clone(),
                horizon,
                seasonality,
                frequency: frequency.clone(),
                target_is_final_column,
                known_future_covariates: known_future_covariates.clone(),
                past_only_covariates: past_only_covariates.clone(),
                reduction: reduction.clone(),
                label_authority: authority.clone(),
                candidates: chunk.to_vec(),
            };
            let name = format!(
                "{dataset_ordinal:06}-{}-{}-{shard:02}.json",
                &cohort_row.task[..1],
                cohort_row.dataset_hash
            );
            let relative = PathBuf::from("manifests").join(&name);
            let path = temporary.join(&relative);
            atomic_json(&path, &serde_json::to_value(&manifest)?)?;
            receipts.push(ManifestReceipt {
                relative_path: relative.to_string_lossy().into_owned(),
                sha256: sha256_file(&path)?,
                task: cohort_row.task.clone(),
                dataset_hash: cohort_row.dataset_hash.clone(),
                candidate_count: chunk.len(),
            });
        }
    }

    ensure!(
        receipts
            .iter()
            .map(|row| row.candidate_count)
            .sum::<usize>()
            == cohort_datasets * WAVE_SIZE,
        "wave candidate count drift"
    );
    ensure!(
        receipts
            .iter()
            .all(|row| row.candidate_count <= args.max_candidates_per_manifest),
        "manifest shard cap drift"
    );
    let cohort_copy = temporary.join("cohort.json");
    atomic_json(&cohort_copy, &serde_json::to_value(&plan)?)?;
    let recovery_accounting_only = plan.wave == "strict-wave-000189";
    let matched_evidence = !recovery_accounting_only
        && native_direct_count > 0
        && native_direct_count == matched_gp_count;
    let summary = json!({
        "schema": WAVE_SCHEMA,
        "immutable": true,
        "wave_kind": "deterministic_128_dataset_micro_wave",
        "wave": plan.wave,
        "checkpoint_epoch": plan.checkpoint_epoch,
        "checkpoint_sha256": plan.checkpoint_sha256,
        "catalog_sha256": plan.catalog_sha256,
        "proposal_file_sha256": proposal_pointer.proposals_sha256,
        "robust_seed_bank_sha256": robust_sha256,
        "generated_at_ms": generated_at_ms,
        "datasets": cohort_datasets,
        // One 64-lane per DECLARED task. The queue's valid_micro_waves proves
        // datasets == 64 * task_datasets.len() with every entry exactly 64, so
        // a hard-coded two-task map here made every wider wave invisible to
        // the queue -- built, published, and silently never admitted.
        "task_datasets": plan
            .tasks
            .iter()
            .map(|task| (task.clone(), Value::from(MICRO_TASK_SIZE)))
            .collect::<serde_json::Map<_, _>>(),
        "cohort_sha256": sha256_file(&cohort_copy)?,
        "candidates_per_dataset": WAVE_SIZE,
        "slot_composition": {
            "local_elite": LOCAL_ELITE_SLOTS,
            "common_winner": COMMON_WINNER_SLOTS,
            "requested_dl_direct_by_task": requested_dl_slots_by_task,
            "observed_dl_direct_by_task": observed_dl_slots_by_task,
            "gp_offspring_formula": "128-actual_dl_direct",
        },
        "candidate_origin_schema": "CandidateOrigin.v2",
        "cross_context_backfill_rows": 0,
        "wave_summary": {
            "schema": "WaveSummary.v1",
            "nominal_d_by_task": requested_dl_slots_by_task,
            "actual_d_by_task": observed_dl_slots_by_task,
            "native_direct_count": native_direct_count,
            "matched_gp_count": matched_gp_count,
            "gp_exploration_count": gp_exploration_count,
            "thin_contexts": thin_contexts,
            "cache_dispositions": {
                "novel": cohort_datasets * WAVE_SIZE - registry_filtered_pairs,
                "registry_replaced": registry_filtered_pairs,
                "proposal_resampled": proposal_cache_resampled,
                "proposal_reservoir_considered": proposal_reservoir_considered,
            },
            "zero_cross_context_backfill": true,
            "evidence_eligibility": {
                "reward_and_stability_heads": true,
                "matched_dl_efficiency": matched_evidence,
                "promotion": matched_evidence,
                "recovery_accounting_only": recovery_accounting_only,
                "legacy_unmatched_composition": false,
            },
        },
        "manifest_count": receipts.len(),
        "max_candidates_per_manifest": args.max_candidates_per_manifest,
        "registry_filtered_pairs": registry_filtered_pairs,
        "registry_filter": "absent-or-available-without-receipt-or-job-v1",
        "manifests": receipts,
        "scheduler_submission": false,
    });
    atomic_json(&temporary.join("summary.json"), &summary)?;
    fs::create_dir_all(&args.output_root)?;
    fs::rename(&temporary, &final_directory)
        .with_context(|| format!("atomically publish wave {}", final_directory.display()))?;
    Ok(json!({
        "schema": WAVE_SCHEMA,
        "wave": plan.wave,
        "wave_directory": final_directory,
        "summary_sha256": sha256_file(&final_directory.join("summary.json"))?,
        "datasets": cohort_datasets,
        "candidate_pairs": cohort_datasets * WAVE_SIZE,
        "registry_filtered_pairs": registry_filtered_pairs,
        "manifests": receipts.len(),
        "submitted_to_scq": false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_dime_regressor::selfplay_data::{
        HARD_UNCERTAIN_SLOTS, LEAST_OBSERVED_SLOTS, MICRO_WAVE_SIZE, STABLE_ANCHOR_SLOTS,
    };
    use tempfile::tempdir;

    #[test]
    fn timeseries_builder_pins_complete_contract_and_builder_policy() {
        let directory = tempdir().unwrap();
        let dataset = directory.path().join("dataset");
        fs::create_dir_all(&dataset).unwrap();
        fs::write(dataset.join("train.csv"), "1.0,2.0\n").unwrap();
        fs::write(
            dataset.join("meta.json"),
            serde_json::to_vec(&json!({
                "task_type": "timeseries",
                "horizon": 3,
                "seasonality": 7,
                "frequency": "D",
                "target_is_final_column": true,
                "feature_columns": ["promo"],
                "reduction": {"version": "ts-reduction-v1"}
            }))
            .unwrap(),
        )
        .unwrap();

        let contract = timeseries_manifest_contract(&dataset.join("train.csv")).unwrap();
        assert_eq!(contract.2, 3);
        assert_eq!(contract.3, 7);
        assert_eq!(contract.4, "D");
        assert_eq!(contract.5, true);
        assert_eq!(contract.6, vec!["promo"]);
        assert!(contract.7.is_empty());
        assert_eq!(contract.8, json!({"version": TS_REDUCTION_V2}));
        assert_eq!(contract.1, sha256_file(&contract.0).unwrap());
    }

    #[test]
    fn timeseries_builder_rejects_incomplete_meta_contract() {
        let directory = tempdir().unwrap();
        let dataset = directory.path().join("dataset");
        fs::create_dir_all(&dataset).unwrap();
        fs::write(dataset.join("train.csv"), "1.0,2.0\n").unwrap();
        fs::write(
            dataset.join("meta.json"),
            br#"{"task_type":"timeseries","horizon":3,"seasonality":7,"frequency":"D","target_is_final_column":true}"#,
        )
        .unwrap();
        assert!(timeseries_manifest_contract(&dataset.join("train.csv")).is_err());
    }

    /// Build a well-formed cohort over `tasks`, 16/32/16 per task.
    fn cohort_over(tasks: &[&str]) -> CohortPlan {
        let checkpoint = "a".repeat(64);
        let mut entries = Vec::new();
        for (task_index, task) in tasks.iter().enumerate() {
            for (lane, count) in [
                (MicroLane::StableAnchor, STABLE_ANCHOR_SLOTS),
                (MicroLane::LeastObserved, LEAST_OBSERVED_SLOTS),
                (MicroLane::HardUncertain, HARD_UNCERTAIN_SLOTS),
            ] {
                for _ in 0..count {
                    entries.push(CohortEntry {
                        task: (*task).to_owned(),
                        dataset_hash: format!("{:016x}", entries.len() + task_index * 1_000),
                        micro_lane: lane,
                    });
                }
            }
        }
        CohortPlan {
            schema: COHORT_SCHEMA.to_owned(),
            immutable: true,
            wave: "strict-wave-000002".to_owned(),
            checkpoint_epoch: checkpoint_epoch(&checkpoint).expect("epoch"),
            checkpoint_sha256: checkpoint,
            catalog_sha256: "b".repeat(64),
            sealed_cohort_sha256: "c".repeat(64),
            tasks: tasks.iter().map(|task| (*task).to_owned()).collect(),
            entries,
        }
    }

    /// Every cohort already on disk predates `CohortPlan::tasks`. If an absent field did not
    /// deserialize to exactly the two-task set, replaying those waves would fail validation
    /// on a size mismatch -- so this is the property that makes the change non-breaking.
    #[test]
    fn a_legacy_cohort_without_a_tasks_field_replays_as_two_tasks() {
        let mut value = serde_json::to_value(cohort_over(&["regression", "binary"])).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .remove("tasks")
            .expect("fixture carries tasks");
        assert!(
            value.get("tasks").is_none(),
            "fixture must omit the field to exercise the default"
        );

        let plan: CohortPlan = serde_json::from_value(value).expect("legacy cohort parses");
        assert_eq!(plan.tasks, vec!["regression", "binary"]);
        assert_eq!(plan.entries.len(), MICRO_WAVE_SIZE);
        validate_cohort(&plan).expect("a legacy two-task cohort still validates");
    }

    /// Multiclass joining widens the wave to 192 without touching per-task structure.
    #[test]
    fn a_three_task_cohort_validates_at_one_lane_per_task() {
        let plan = cohort_over(&["regression", "binary", "multiclass"]);
        assert_eq!(plan.entries.len(), 3 * MICRO_TASK_SIZE);
        validate_cohort(&plan).expect("a three-task cohort validates");

        let four = cohort_over(&["regression", "binary", "multiclass", "timeseries"]);
        assert_eq!(four.entries.len(), 4 * MICRO_TASK_SIZE);
        validate_cohort(&four).expect("a four-task cohort validates");
    }

    /// The declaration and the rows must agree, in both directions.
    #[test]
    fn a_cohort_whose_rows_contradict_its_declaration_is_rejected() {
        let mut undersized = cohort_over(&["regression", "binary", "multiclass"]);
        undersized.tasks = vec!["regression".to_owned(), "binary".to_owned()];
        assert!(
            validate_cohort(&undersized).is_err(),
            "192 rows against a two-task declaration must fail"
        );

        let mut dropped = cohort_over(&["regression", "binary"]);
        dropped.tasks = vec!["regression".to_owned()];
        assert!(
            validate_cohort(&dropped).is_err(),
            "dropping a mandatory lane must fail"
        );

        let mut unknown = cohort_over(&["regression", "binary"]);
        unknown.tasks.push("clustering".to_owned());
        assert!(
            validate_cohort(&unknown).is_err(),
            "an unknown task must fail"
        );
    }

    #[test]
    fn adaptive_reservoir_accepts_declared_rungs_but_nothing_larger() {
        assert!(valid_proposal_reservoir_slot(0));
        assert!(valid_proposal_reservoir_slot(63));
        assert!(valid_proposal_reservoir_slot(127));
        assert!(valid_proposal_reservoir_slot(255));
        assert!(!valid_proposal_reservoir_slot(256));
    }

    fn catalog_entry(task: &str, index: usize, family: String) -> CatalogEntry {
        CatalogEntry {
            task: task.to_owned(),
            dataset_hash: format!("{index:016x}"),
            train_csv: PathBuf::from("/not-read/train.csv"),
            test_csv: PathBuf::from("/not-read/test.csv"),
            train_sha256: "a".repeat(64),
            test_sha256: "b".repeat(64),
            source_family: family,
            stable_anchor: false,
            golden42: false,
            current_catalog: None,
            observation_count: None,
            uncertainty: None,
            hardness: None,
            incumbent_regret: None,
        }
    }

    fn ample_catalog() -> Catalog {
        let mut entries = Vec::new();
        for index in 0..300 {
            entries.push(catalog_entry(
                "binary",
                index,
                format!("shared-family-{index:04}"),
            ));
            entries.push(catalog_entry(
                "regression",
                20_000 + index,
                format!("shared-family-{index:04}"),
            ));
            entries.push(catalog_entry(
                "regression",
                10_000 + index,
                format!("regression-family-{index:04}"),
            ));
        }
        Catalog { entries }
    }

    fn sanitizer_fixture(directory: &Path) -> (PathBuf, PathBuf, PathBuf, CohortPlan, String) {
        let checkpoint = "d".repeat(64);
        let mut cohort_entries = Vec::new();
        for task in ["regression", "binary"] {
            for index in 0..64 {
                let micro_lane = match index {
                    0..=15 => MicroLane::StableAnchor,
                    16..=47 => MicroLane::LeastObserved,
                    _ => MicroLane::HardUncertain,
                };
                cohort_entries.push(CohortEntry {
                    task: task.to_owned(),
                    dataset_hash: format!("{task}-cohort-{index:02}"),
                    micro_lane,
                });
            }
        }
        let plan = CohortPlan {
            schema: COHORT_SCHEMA.to_owned(),
            immutable: true,
            wave: "strict-wave-000001".to_owned(),
            checkpoint_epoch: checkpoint_epoch(&checkpoint).unwrap(),
            checkpoint_sha256: checkpoint.clone(),
            catalog_sha256: "a".repeat(64),
            sealed_cohort_sha256: "b".repeat(64),
            // Evaluated before `entries` moves the vec (struct fields evaluate in order).
            tasks: jope_dime_regressor::selfplay_expansion::TASKS
                .iter()
                .filter(|task| cohort_entries.iter().any(|row| row.task == **task))
                .map(|task| (*task).to_owned())
                .collect(),
            entries: cohort_entries,
        };
        let cohort_path = directory.join("cohort.json");
        atomic_json(&cohort_path, &serde_json::to_value(&plan).unwrap()).unwrap();

        let source_path = directory.join("source.jsonl");
        let mut source = Vec::new();
        for (dataset_index, entry) in plan.entries.iter().enumerate() {
            for slot in 0..4 {
                let q = 0.001 + (dataset_index * 4 + slot) as f64 / 10_000.0;
                let head = if entry.task == "binary" {
                    "logistic"
                } else {
                    "ridge"
                };
                let genome = canonical(&format!("(pipe X (winsor_clip q={q}) ({head}))"));
                let row = GpProposal {
                    task: entry.task.clone(),
                    dataset_hash: entry.dataset_hash.clone(),
                    semantic_sha256: semantic(&genome),
                    canonical_genome: genome,
                    model_checkpoint_sha256: checkpoint.clone(),
                    slot,
                    score: None,
                    proposal_score: None,
                };
                serde_json::to_writer(&mut source, &row).unwrap();
                source.push(b'\n');
            }
        }
        fs::write(&source_path, source).unwrap();
        let pointer_path = directory.join("current.json");
        atomic_json(
            &pointer_path,
            &json!({
                "proposals": source_path,
                "proposals_sha256": sha256_file(&source_path).unwrap(),
                "checkpoint_sha256": checkpoint,
                "dl_slots_by_task": {"regression": 8, "binary": 8},
            }),
        )
        .unwrap();
        (
            cohort_path,
            pointer_path,
            directory.join("by-cohort"),
            plan,
            checkpoint,
        )
    }

    fn canonical(source: &str) -> String {
        GenomeIr::parse(source).unwrap().canonical()
    }

    fn local_rows(count: usize) -> Vec<LocalElite> {
        (0..count)
            .map(|index| {
                let genome = canonical(&format!(
                    "(pipe X (winsor_clip q={}) (ridge))",
                    0.001 + index as f64 / 10000.0
                ));
                LocalElite {
                    semantic_sha256: semantic(&genome),
                    canonical_genome: genome,
                    score: 1.0 - index as f64 / 1000.0,
                    secondary_score: None,
                    receipt_sha256: format!("{index:064x}"),
                    candidate_source: "fixture".to_owned(),
                    dl_ancestry_kind: "none".to_owned(),
                }
            })
            .collect()
    }

    fn proposals(checkpoint: &str) -> Vec<GpProposal> {
        (0..64)
            .map(|slot| {
                let genome = canonical(&format!(
                    "(pipe X (quantile_bins bins={}) (ridge))",
                    slot + 2
                ));
                GpProposal {
                    task: "regression".to_owned(),
                    dataset_hash: "dataset".to_owned(),
                    semantic_sha256: semantic(&genome),
                    canonical_genome: genome,
                    model_checkpoint_sha256: checkpoint.to_owned(),
                    slot,
                    score: None,
                    proposal_score: Some(64.0 - slot as f64),
                }
            })
            .collect()
    }

    fn robust_rows(count: usize) -> Vec<RobustSeed> {
        (0..count)
            .map(|index| {
                let genome = canonical(&format!(
                    "(pipe X (cheb degree={} append=true) (ridge))",
                    index + 2
                ));
                RobustSeed {
                    task: "regression".to_owned(),
                    semantic_sha256: semantic(&genome),
                    canonical_genome: genome,
                    dataset_count: index + 2,
                    mean_within_dataset_percentile: 1.0 - index as f64 / 1000.0,
                    worst_within_dataset_percentile: 0.5,
                    strict_win_count: index,
                    dl_origin: false,
                }
            })
            .collect()
    }

    #[test]
    fn exact_composition_is_deterministic_and_unique() {
        let checkpoint = "a".repeat(64);
        let left = compose_dataset(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &proposals(&checkpoint),
            &local_rows(40),
            &robust_rows(110),
        )
        .unwrap();
        let right = compose_dataset(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &proposals(&checkpoint),
            &local_rows(40),
            &robust_rows(110),
        )
        .unwrap();
        assert_eq!(left.len(), WAVE_SIZE);
        assert_eq!(
            left.iter()
                .map(|row| row.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len(),
            WAVE_SIZE
        );
        assert_eq!(
            left.iter()
                .map(|row| (&row.semantic_sha256, row.slot_group, row.slot_index))
                .collect::<Vec<_>>(),
            right
                .iter()
                .map(|row| (&row.semantic_sha256, row.slot_group, row.slot_index))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn adaptive_composition_and_origin_attribution_are_exact() {
        let checkpoint = "a".repeat(64);
        let all_proposals = proposals(&checkpoint);
        for actual in 0..=CHECKPOINT_PROPOSAL_SLOTS {
            let selected = compose_dataset_with_dl(
                "strict-wave-000001",
                "regression",
                "dataset",
                &checkpoint,
                &all_proposals[..actual],
                &local_rows(40),
                &robust_rows(150),
                CHECKPOINT_PROPOSAL_SLOTS,
            )
            .unwrap();
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.origin == CandidateOrigin::DlDirect)
                    .count(),
                actual
            );
            assert_eq!(
                selected
                    .iter()
                    .filter(|row| row.origin == CandidateOrigin::GpOffspring)
                    .count(),
                128 - actual
            );
            assert!(selected.iter().all(|row| {
                row.model_checkpoint_sha256.is_some() == (row.origin == CandidateOrigin::DlDirect)
            }));
            let direct = selected
                .iter()
                .filter(|row| row.origin == CandidateOrigin::DlDirect)
                .map(|row| row.semantic_sha256.as_str())
                .collect::<HashSet<_>>();
            assert!(selected
                .iter()
                .filter(|row| row.origin == CandidateOrigin::GpOffspring)
                .flat_map(|row| row.parentage.iter())
                .all(|parent| !direct.contains(parent.as_str())));
        }
    }

    #[test]
    fn cache_filter_resamples_the_native_reservoir_before_composition() {
        let checkpoint = "a".repeat(64);
        let proposals = proposals(&checkpoint);
        let unavailable = proposals[..56]
            .iter()
            .map(|proposal| proposal.semantic_sha256.clone())
            .collect();
        let (selected, resampled) =
            cache_filter_proposal_reservoir(&proposals, &RegistryPairView { unavailable }, 8);
        assert_eq!(resampled, 56);
        assert_eq!(selected.len(), 8);
        assert_eq!(
            selected
                .iter()
                .map(|proposal| proposal.slot)
                .collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        assert_eq!(
            selected
                .iter()
                .map(|proposal| proposal.semantic_sha256.as_str())
                .collect::<Vec<_>>(),
            proposals[56..64]
                .iter()
                .map(|proposal| proposal.semantic_sha256.as_str())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn matched_gp_lane_tracks_every_adaptive_actual_d_without_dl_ancestry() {
        let checkpoint = "a".repeat(64);
        let all_proposals = proposals(&checkpoint);
        for actual in 0..=CHECKPOINT_PROPOSAL_SLOTS {
            let mut selected = compose_dataset_with_dl(
                "strict-wave-000001",
                "regression",
                "dataset",
                &checkpoint,
                &all_proposals[..actual],
                &local_rows(40),
                &robust_rows(150),
                CHECKPOINT_PROPOSAL_SLOTS,
            )
            .unwrap();
            let (matched, exploration) = annotate_matched_evaluation_lanes(
                &mut selected,
                actual,
                &checkpoint,
                "strict-wave-000191",
                "regression",
                "dataset",
            )
            .unwrap();
            assert_eq!(matched, actual);
            assert_eq!(exploration, WAVE_SIZE - 2 * actual);
            assert_eq!(
                selected
                    .iter()
                    .filter(|candidate| {
                        candidate
                            .evidence
                            .as_ref()
                            .and_then(|evidence| evidence["evaluation_lane"].as_str())
                            == Some("matched_gp_comparator")
                    })
                    .count(),
                actual
            );
            assert!(selected.iter().all(|candidate| {
                candidate
                    .evidence
                    .as_ref()
                    .and_then(|evidence| evidence["evaluation_lane"].as_str())
                    != Some("matched_gp_comparator")
                    || (candidate.origin != CandidateOrigin::DlDirect
                        && candidate.model_checkpoint_sha256.is_none()
                        && candidate.parentage.iter().all(|parent| {
                            !selected
                                .iter()
                                .filter(|row| row.origin == CandidateOrigin::DlDirect)
                                .any(|direct| &direct.semantic_sha256 == parent)
                        })
                        && candidate
                            .evidence
                            .as_ref()
                            .and_then(|evidence| evidence["zero_dl_ancestry"].as_bool())
                            == Some(true))
            }));
        }
    }

    #[test]
    fn duplicate_categories_fail_closed_when_exact_slots_are_impossible() {
        let checkpoint = "a".repeat(64);
        let proposals = proposals(&checkpoint);
        let mut locals = local_rows(31);
        locals.push(LocalElite {
            semantic_sha256: proposals[0].semantic_sha256.clone(),
            canonical_genome: proposals[0].canonical_genome.clone(),
            score: 2.0,
            secondary_score: None,
            receipt_sha256: "b".repeat(64),
            candidate_source: "duplicate".to_owned(),
            dl_ancestry_kind: "none".to_owned(),
        });
        assert!(compose_dataset(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &proposals,
            &locals,
            &robust_rows(96),
        )
        .is_err());
    }

    #[test]
    fn cold_start_datasets_fill_local_slots_from_distinct_robust_surrogates() {
        let checkpoint = "a".repeat(64);
        let selected = compose_dataset(
            "strict-wave-000001",
            "regression",
            "cold-start-dataset",
            &checkpoint,
            &proposals(&checkpoint),
            &[],
            &robust_rows(140),
        )
        .unwrap();
        assert_eq!(
            selected
                .iter()
                .filter(|row| {
                    row.slot_group == WaveSlotGroup::LocalElite
                        && row.source == "cold_start_repeated_robust_gp_surrogate"
                })
                .count(),
            LOCAL_ELITE_SLOTS
        );
        assert_eq!(
            selected
                .iter()
                .filter(|row| row.slot_group == WaveSlotGroup::CommonWinner)
                .count(),
            COMMON_WINNER_SLOTS
        );
        assert_eq!(
            selected
                .iter()
                .map(|row| row.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len(),
            WAVE_SIZE
        );
    }

    #[test]
    fn registry_filter_deterministically_replaces_blocked_pairs_and_preserves_slots() {
        let checkpoint = "a".repeat(64);
        let logical = compose_dataset(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &proposals(&checkpoint),
            &local_rows(40),
            &robust_rows(150),
        )
        .unwrap();
        let blocked_indexes = [0, 10, 31, 32, 50, 95, 96, 130, 190, 192, 220, 255];
        let unavailable = blocked_indexes
            .iter()
            .map(|index| logical[*index].semantic_sha256.clone())
            .collect::<HashSet<_>>();
        let registry = RegistryPairView {
            unavailable: unavailable.clone(),
        };
        let (left, left_filtered) = registry_filter_and_fill(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &logical,
            &registry,
        )
        .unwrap();
        let (right, right_filtered) = registry_filter_and_fill(
            "strict-wave-000001",
            "regression",
            "dataset",
            &checkpoint,
            &logical,
            &registry,
        )
        .unwrap();

        assert_eq!(left_filtered, unavailable.len());
        assert_eq!(right_filtered, unavailable.len());
        assert_eq!(left.len(), WAVE_SIZE);
        assert_eq!(
            left.iter()
                .map(|candidate| candidate.semantic_sha256.as_str())
                .collect::<Vec<_>>(),
            right
                .iter()
                .map(|candidate| candidate.semantic_sha256.as_str())
                .collect::<Vec<_>>()
        );
        assert!(left
            .iter()
            .all(|candidate| !unavailable.contains(&candidate.semantic_sha256)));
        assert_eq!(
            left.iter()
                .map(|candidate| candidate.semantic_sha256.as_str())
                .collect::<HashSet<_>>()
                .len(),
            WAVE_SIZE
        );
        for (logical, launchable) in logical.iter().zip(&left) {
            if unavailable.contains(&logical.semantic_sha256) {
                assert_eq!(launchable.source, "registry_aware_deterministic_backfill");
                assert_eq!(launchable.origin, CandidateOrigin::GpOffspring);
                assert!(launchable.model_checkpoint_sha256.is_none());
                assert!(launchable.parentage.contains(&logical.semantic_sha256));
                assert_eq!(
                    launchable
                        .evidence
                        .as_ref()
                        .and_then(|value| value["logical_semantic_sha256"].as_str()),
                    Some(logical.semantic_sha256.as_str())
                );
                if logical.slot_group == WaveSlotGroup::DlProposal {
                    assert_eq!(launchable.slot_group, WaveSlotGroup::Offspring);
                } else {
                    assert_eq!(launchable.slot_group, logical.slot_group);
                    if logical.slot_group != WaveSlotGroup::Offspring {
                        assert_eq!(launchable.slot_index, logical.slot_index);
                    }
                }
            } else if !matches!(
                logical.slot_group,
                WaveSlotGroup::DlProposal | WaveSlotGroup::Offspring
            ) {
                assert_eq!(launchable.slot_group, logical.slot_group);
                assert_eq!(launchable.slot_index, logical.slot_index);
            }
        }
    }

    #[test]
    fn registry_filter_compacts_actual_direct_slots_for_every_count() {
        let checkpoint = "a".repeat(64);
        for retained in 0..=CHECKPOINT_PROPOSAL_SLOTS {
            let logical = compose_dataset(
                "strict-wave-000189",
                "regression",
                "dataset",
                &checkpoint,
                &proposals(&checkpoint),
                &local_rows(40),
                &robust_rows(150),
            )
            .unwrap();
            let unavailable = logical
                .iter()
                .filter(|candidate| {
                    candidate.origin == CandidateOrigin::DlDirect
                        && candidate.slot_index >= retained
                })
                .map(|candidate| candidate.semantic_sha256.clone())
                .collect::<HashSet<_>>();
            let (launchable, filtered) = registry_filter_and_fill(
                "strict-wave-000189",
                "regression",
                "dataset",
                &checkpoint,
                &logical,
                &RegistryPairView { unavailable },
            )
            .unwrap();
            assert_eq!(filtered, CHECKPOINT_PROPOSAL_SLOTS - retained);
            assert_eq!(
                launchable
                    .iter()
                    .filter(|candidate| candidate.origin == CandidateOrigin::DlDirect)
                    .count(),
                retained
            );
            assert_eq!(
                launchable
                    .iter()
                    .filter(|candidate| candidate.slot_group == WaveSlotGroup::DlProposal)
                    .map(|candidate| candidate.slot_index)
                    .collect::<Vec<_>>(),
                (0..retained).collect::<Vec<_>>()
            );
            assert!(launchable
                .iter()
                .filter(|candidate| candidate.origin == CandidateOrigin::GpOffspring)
                .all(|candidate| candidate.model_checkpoint_sha256.is_none()));
        }
    }

    #[test]
    fn registry_view_allows_only_absent_or_clean_available_pairs() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE pairs(
                   task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
                   canonical_genome TEXT,state TEXT,receipt_sha256 TEXT,job_id TEXT
                 );",
            )
            .unwrap();
        let genomes = [
            canonical("(pipe X (ridge))"),
            canonical("(pipe X (ridge_gcv))"),
            canonical("(pipe X (elasticnet))"),
        ];
        for (index, (state, receipt, job)) in [
            ("authoritative", Some("r"), None),
            ("reserved", None, Some("j")),
            ("available", None, None),
        ]
        .into_iter()
        .enumerate()
        {
            connection
                .execute(
                    "INSERT INTO pairs VALUES('regression','dataset',?1,?2,?3,?4,?5,?6)",
                    params![
                        semantic(&genomes[index]),
                        "b".repeat(64),
                        genomes[index],
                        state,
                        receipt,
                        job
                    ],
                )
                .unwrap();
        }
        let view =
            registry_pair_view(&connection, "regression", "dataset", &"b".repeat(64)).unwrap();
        assert!(view.unavailable.contains(&semantic(&genomes[0])));
        assert!(view.unavailable.contains(&semantic(&genomes[1])));
        assert!(!view.unavailable.contains(&semantic(&genomes[2])));
        assert_eq!(view.unavailable.len(), 2);
    }

    #[test]
    /// A head-swapped multiclass offspring gets its parent's head back and is
    /// then task-valid; a graft that still cannot become valid returns None
    /// through the caller's filter. Regression/binary never take this path.
    #[test]
    fn grafting_restores_a_task_legal_head_deterministically() {
        let parent =
            "(pipe X (standardize) (multiclass_ridge class_weight=balanced l2=0.02))";
        assert!(task_valid(parent, "multiclass"));
        // The frozen mutator swapped the head to a regression one.
        let child = "(pipe X (winsor_clip q=0.02) (standardize) (ridge))";
        assert!(!task_valid(child, "multiclass"));
        let grafted = graft_task_head(child, parent).expect("graft succeeds");
        assert!(task_valid(&grafted, "multiclass"), "grafted: {grafted}");
        assert!(grafted.contains("multiclass_ridge"));
        assert!(grafted.contains("winsor_clip"), "transform stack preserved");
        // Deterministic: same inputs, same output.
        assert_eq!(graft_task_head(child, parent).unwrap(), grafted);
    }

    /// The ts contract must be satisfiable by the metadata that actually
    /// exists. Every encoded timeseries dataset declares `feature_columns` and
    /// NONE declares `known_dynamic_columns` / `past_dynamic_columns_excluded`,
    /// so requiring those names failed all 94 datasets and would have wedged
    /// supply the way wave 552 did for multiclass.
    #[test]
    fn timeseries_contract_reads_the_metadata_datasets_actually_have() {
        let directory = tempfile::tempdir().expect("tempdir");
        let train_csv = directory.path().join("train.csv");
        fs::write(&train_csv, b"1,2\n").expect("write train");
        // Exactly the key set of a real encodings/timeseries-expanded meta.json.
        fs::write(
            directory.path().join("meta.json"),
            serde_json::to_vec(&json!({
                "task_type": "timeseries",
                "horizon": 28,
                "seasonality": 7,
                "frequency": "D",
                "target_is_final_column": true,
                "reduction_version": "ts-reduction-v2",
                "feature_columns": ["lag_1", "lag_7"],
            }))
            .expect("meta bytes"),
        )
        .expect("write meta");

        let contract = timeseries_manifest_contract(&train_csv).expect("contract resolves");
        assert_eq!(contract.2, 28, "horizon");
        assert_eq!(contract.3, 7, "seasonality");
        assert_eq!(contract.4, "D", "frequency");
        assert!(contract.5, "target_is_final_column");
        assert_eq!(contract.6, vec!["lag_1", "lag_7"], "known-future covariates");
        assert!(contract.7.is_empty(), "no past-only covariates are declared");
        // The reduction is the BUILDER's policy, never read from metadata.
        assert_eq!(contract.8, json!({"version": TS_REDUCTION_V2}));
    }

    #[test]
    fn task_validation_rejects_cross_task_heads_and_noncanonical_input() {
        assert!(task_valid("(pipe X (ridge))", "regression"));
        assert!(!task_valid("(pipe X (ridge))", "binary"));
        assert!(task_valid("(pipe X (logistic))", "binary"));
        assert!(!task_valid("(pipe X (logistic))", "regression"));
        assert!(!task_valid(" (pipe X (ridge))", "regression"));
    }

    #[test]
    fn wave_ids_are_strict() {
        assert_eq!(parse_wave_id("strict-wave-000042"), Some(42));
        assert_eq!(parse_wave_id("strict-wave-42"), None);
        assert_eq!(parse_wave_id("other-000042"), None);
    }

    #[test]
    fn sealed_selection_is_deterministic_exact_and_globally_family_disjoint() {
        let catalog_sha256 = "c".repeat(64);
        let catalog = ample_catalog();
        let left = select_sealed_cohort(&catalog, &catalog_sha256).unwrap();
        let mut reversed = ample_catalog();
        reversed.entries.reverse();
        let right = select_sealed_cohort(&reversed, &catalog_sha256).unwrap();

        assert_eq!(
            canonical_json_bytes(&serde_json::to_value(&left).unwrap()).unwrap(),
            canonical_json_bytes(&serde_json::to_value(&right).unwrap()).unwrap()
        );
        assert_eq!(left.entries.len(), 512);
        assert_eq!(
            left.entries
                .iter()
                .filter(|entry| entry.task == "binary")
                .count(),
            256
        );
        assert_eq!(
            left.entries
                .iter()
                .filter(|entry| entry.task == "regression")
                .count(),
            256
        );
        assert_eq!(
            left.entries
                .iter()
                .map(|entry| entry.source_family.as_str())
                .collect::<HashSet<_>>()
                .len(),
            512
        );
    }

    #[test]
    fn sealed_selection_fails_closed_without_enough_families() {
        let mut catalog = ample_catalog();
        let mut binary_seen = 0;
        catalog.entries.retain(|entry| {
            if entry.task != "binary" {
                return true;
            }
            binary_seen += 1;
            binary_seen <= 255
        });
        assert!(select_sealed_cohort(&catalog, &"c".repeat(64)).is_err());
    }

    #[test]
    fn private_publish_is_mode_0600_idempotent_and_immutable() {
        let directory = tempdir().unwrap();
        let output = directory.path().join("sealed.json");
        let bytes = b"{\"schema\":\"JopeSealedCohort.v1\"}\n";

        assert!(publish_private_immutable(&output, bytes).unwrap());
        assert!(!publish_private_immutable(&output, bytes).unwrap());
        assert_eq!(
            fs::metadata(&output).unwrap().permissions().mode() & 0o7777,
            0o600
        );
        fs::write(&output, b"drift\n").unwrap();
        assert!(publish_private_immutable(&output, bytes).is_err());
    }

    #[test]
    fn freeze_summary_redacts_dataset_identities_and_families() {
        let directory = tempdir().unwrap();
        let catalog_path = directory.path().join("catalog.json");
        let pointer_path = directory.path().join("current.json");
        let output = directory.path().join("private").join("sealed.json");
        let catalog = ample_catalog();
        atomic_json(
            &catalog_path,
            &serde_json::to_value(&catalog).expect("serialize catalog"),
        )
        .unwrap();
        let catalog_sha256 = sha256_file(&catalog_path).unwrap();
        atomic_json(
            &pointer_path,
            &json!({
                "catalog": catalog_path,
                "catalog_sha256": catalog_sha256,
            }),
        )
        .unwrap();

        let result = freeze_sealed(FreezeSealedArgs {
            catalog_pointer: pointer_path,
            output,
        })
        .unwrap();
        let printed = serde_json::to_string(&result).unwrap();
        for entry in catalog.entries {
            assert!(!printed.contains(&entry.dataset_hash));
            assert!(!printed.contains(&entry.source_family));
        }
    }

    #[test]
    fn proposal_sanitizer_is_deterministic_local_only_and_zero_backfill() {
        let directory = tempdir().unwrap();
        let (cohort, source_pointer, output_root, plan, checkpoint) =
            sanitizer_fixture(directory.path());
        let first = sanitize_proposals(SanitizeProposalsArgs {
            cohort: cohort.clone(),
            proposals_pointer: source_pointer.clone(),
            output_root: output_root.clone(),
        })
        .unwrap();
        let second = sanitize_proposals(SanitizeProposalsArgs {
            cohort,
            proposals_pointer: source_pointer,
            output_root,
        })
        .unwrap();
        assert_eq!(first["pointer_sha256"], second["pointer_sha256"]);
        assert_eq!(first["proposals_sha256"], second["proposals_sha256"]);

        let pointer_path = PathBuf::from(first["pointer"].as_str().unwrap());
        assert_eq!(
            sha256_file(&pointer_path).unwrap(),
            first["pointer_sha256"].as_str().unwrap()
        );
        let pointer = read_json(&pointer_path).unwrap();
        assert_eq!(pointer["schema"], "CohortProposalPointer.v1");
        assert_eq!(pointer["cohort_sha256"], first["cohort_sha256"]);
        assert_eq!(pointer["catalog_sha256"], plan.catalog_sha256);
        assert_eq!(pointer["seed"], COHORT_PROPOSAL_SEED);
        assert_eq!(pointer["task_scopes"].as_array().unwrap().len(), 2);
        assert_eq!(pointer["cross_context_backfill_rows"], 0);
        let proposals_path = PathBuf::from(pointer["proposals"].as_str().unwrap());
        assert_eq!(
            sha256_file(&proposals_path).unwrap(),
            pointer["proposals_sha256"].as_str().unwrap()
        );

        let mut by_dataset = HashMap::<(String, String), Vec<GpProposal>>::new();
        let mut rows = 0;
        for line in BufReader::new(File::open(proposals_path).unwrap()).lines() {
            let line = line.unwrap();
            let value: Value = serde_json::from_str(&line).unwrap();
            let row: GpProposal = serde_json::from_value(value.clone()).unwrap();
            assert!(task_valid(&row.canonical_genome, &row.task));
            assert_eq!(row.model_checkpoint_sha256, checkpoint);
            assert_eq!(
                value["origin"],
                Value::String("local_current_checkpoint".to_owned())
            );
            assert_eq!(value["distinct_dataset_repeat_count"], 1);
            assert!(value["backfill_rank"].is_null());
            assert!(value["source_task"].is_string());
            assert!(value["source_dataset_hash"].is_string());
            assert!(value["source_slot"].is_number());
            assert_eq!(value["source_task"], value["task"]);
            assert_eq!(value["source_dataset_hash"], value["dataset_hash"]);
            assert_eq!(value["retargeted"], false);
            by_dataset
                .entry((row.task.clone(), row.dataset_hash.clone()))
                .or_default()
                .push(row);
            rows += 1;
        }
        assert_eq!(rows, plan.entries.len() * 4);
        for entry in plan.entries {
            let mut proposals = by_dataset
                .remove(&(entry.task.clone(), entry.dataset_hash.clone()))
                .unwrap();
            proposals.sort_by_key(|proposal| proposal.slot);
            assert_eq!(proposals.len(), 4);
            assert_eq!(
                proposals
                    .iter()
                    .map(|proposal| proposal.slot)
                    .collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
        }
        assert!(by_dataset.is_empty());
    }

    #[test]
    fn proposal_sanitizer_rejects_source_hash_drift() {
        let directory = tempdir().unwrap();
        let (cohort, pointer, output_root, _, _) = sanitizer_fixture(directory.path());
        let mut value = read_json(&pointer).unwrap();
        value["proposals_sha256"] = Value::String("e".repeat(64));
        atomic_json(&pointer, &value).unwrap();
        assert!(sanitize_proposals(SanitizeProposalsArgs {
            cohort,
            proposals_pointer: pointer,
            output_root,
        })
        .is_err());
    }

    /// Pins the eligibility guard against the exact shape that broke production on
    /// 2026-08-04: 7 regression datasets whose features live in sibling `.vs` files
    /// and whose train.csv holds only the target column. Every candidate drawn
    /// against them exits 1 in `corpus_eval`, so they burn four evaluations per
    /// wave slot and stay eligible to do it again next wave.
    #[test]
    fn a_dataset_with_no_feature_columns_is_never_eligible() {
        let directory = tempdir().unwrap();

        // The real failing shape: one bare numeric column, no header, no comma.
        let target_only = directory.path().join("target-only.csv");
        fs::write(&target_only, "0.0123044169\n0.695948541\n0.31\n").unwrap();
        assert!(
            !dataset_has_feature_columns(&target_only),
            "a single-column train.csv can never be evaluated"
        );

        // A normal supervised table is eligible.
        let usable = directory.path().join("usable.csv");
        fs::write(&usable, "f0,f1,target\n0.1,0.2,1.0\n0.3,0.4,0.0\n").unwrap();
        assert!(dataset_has_feature_columns(&usable));

        // Header-less multi-column data is still usable.
        let headerless = directory.path().join("headerless.csv");
        fs::write(&headerless, "0.1,0.2,1.0\n0.3,0.4,0.0\n").unwrap();
        assert!(dataset_has_feature_columns(&headerless));

        // An absent file is unevaluatable rather than a panic: selecting it would
        // dispatch work that cannot succeed.
        assert!(!dataset_has_feature_columns(&directory.path().join("gone.csv")));

        // A trailing newline must not be mistaken for a delimiter.
        let empty = directory.path().join("empty.csv");
        fs::write(&empty, "\n").unwrap();
        assert!(!dataset_has_feature_columns(&empty));
    }

    /// The wedge this pins: a proposal pointer scoped to the LIVE catalog while
    /// the frozen cohort is bound to an older one is unrecoverable. The cohort is
    /// immutable, so the two digests can never be reconciled and the wave retries
    /// identically forever -- which is exactly how the queue drained to zero on
    /// 2026-08-04 (23 identical failures on wave 474).
    ///
    /// The pointer's catalog scope is not chosen by the proposer. It is copied
    /// from the digest the `cohort` subcommand reports, so the only way to keep
    /// this invariant satisfiable is for that subcommand to report the ADOPTED
    /// cohort's binding rather than whatever `catalog/current.json` says at the
    /// moment it runs. This test fails if that reporting ever regresses.
    #[test]
    fn a_pointer_scoped_to_a_rotated_catalog_cannot_satisfy_a_frozen_cohort() {
        let directory = tempdir().unwrap();
        let (cohort, pointer, output_root, plan, _) = sanitizer_fixture(directory.path());

        // Same pointer, declared under the schema that carries a catalog scope,
        // bound to the catalog the cohort was actually frozen against.
        let mut matched = read_json(&pointer).unwrap();
        matched["schema"] = Value::String("CohortProposalPointer.v1".to_owned());
        matched["cohort_sha256"] = Value::String(sha256_file(&cohort).unwrap());
        matched["catalog_sha256"] = Value::String(plan.catalog_sha256.clone());
        atomic_json(&pointer, &matched).unwrap();
        assert!(
            sanitize_proposals(SanitizeProposalsArgs {
                cohort: cohort.clone(),
                proposals_pointer: pointer.clone(),
                output_root: output_root.clone(),
            })
            .is_ok(),
            "a pointer bound to the cohort's own catalog must build"
        );

        // The same pointer after a catalog rotation leaked into its scope.
        let mut rotated = matched;
        rotated["catalog_sha256"] = Value::String("c".repeat(64));
        assert_ne!(rotated["catalog_sha256"], Value::String(plan.catalog_sha256));
        atomic_json(&pointer, &rotated).unwrap();
        let error = sanitize_proposals(SanitizeProposalsArgs {
            cohort,
            proposals_pointer: pointer,
            output_root,
        })
        .expect_err("a live-catalog-scoped pointer must not satisfy a frozen cohort");
        assert!(
            error
                .to_string()
                .contains("cohort proposal pointer scope differs from the frozen cohort"),
            "unexpected failure: {error}"
        );
    }
}
