//! Equal-budget comparison of two search profiles (kernel2 WP-0.4).
//!
//! A profile is a named configuration of the shipping in-process search,
//! `jope_pipeline::search::run_cuts_with`. The harness runs profile A and profile B separately on
//! the same datasets, the same cuts, the same seeds and the same evaluation budget, and writes
//! one `jope.bench-ablation/v1` document:
//!
//! * `body`: the inputs, one record per (profile, dataset, seed) with the metrics M1..M11 (M10
//!   excepted), an aggregate per profile, the promotion gates G-A..G-E evaluated as data, and a
//!   verdict. The same inputs give byte-identical body bytes on one host and toolchain;
//!   `body_sha256` is the sha256 of the compact bytes the harness serializes.
//! * `timing` (M10, wall seconds per evaluation, and the gate clause that reads it) and
//!   `provenance` (paths, host and time). Neither is part of the digest.
//!
//! Replay rules. Every search runs with the largest seconds bound the search accepts
//! ([`SEARCH_SECONDS`]), one worker and one island, so the evaluation budget ends every phase and
//! the decisions replay from the seed alone. A run in which the seconds bound ended any phase is
//! refused ([`check_replayable`]) rather than reported. Every (profile, dataset, seed) search
//! writes into its own directory, and profile B is executed, never copied from profile A, so an
//! A=A comparison runs the search twice.
//!
//! Data rules. Rows come from each dataset's `train.csv` only, through [`crate::load`]; the
//! sealed `test.csv` is never opened. Cuts are the pipeline's own ([`prepare::build`], keyed on
//! `<dataset_hash>:<fold seed>`), so M1 is a held-out score on a fold of the training rows.
//!
//! Profiles: `legacy`, the shipping search unchanged, and the kernel2 route
//! (`jope_pipeline::kernel2::search::run_cut`) with the shipped `kernel2-baseline` and
//! `kernel2-experimental` profiles, an explicit refinement allowance shared by every kernel2
//! search, one thread and one island. A comparison that involves a kernel2 profile also carries
//! the paired comparison of M1 ([`Paired`], [`PAIRED_DEFINITIONS`]) in the body and each slot's
//! wall seconds beside it in `timing`; a legacy-only document is unchanged. Later profiles are
//! added as new [`Profile`] variants. `docs/internal/kernel2/ablation-protocol.md` states the
//! metric, gate and paired definitions.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::ParamValue;
use feat_math::evolve::FaultPolicy;
use feat_math::kernel::profile::Limit;
use feat_math::math_expr::{Expr, MathExpr};
use feat_math::transpile::EmitOptions;
use feat_math::GenomeIr;
use jope_pipeline::kernel2;
use jope_pipeline::kernel2::search::{PolicyRecord, Refinement};
use jope_pipeline::prepare::{self, Cut};
use jope_pipeline::search::{self, Budget, CutSearch, Observation, SearchOptions};
use jope_pipeline::select;
use serde::Serialize;

use crate::load::{discover, load_dataset, sha256_hex};
use crate::task::Task;

/// The document schema.
pub const SCHEMA: &str = "jope.bench-ablation/v1";
/// The version of the metric and gate definitions the body carries.
pub const DEFINITIONS: &str = "kernel2-ablation-metrics.v1";
/// Seconds bound of every search: the largest the search accepts, so the evaluation budget, not
/// the clock, ends every phase of a small search.
pub const SEARCH_SECONDS: u64 = search::MAX_SECONDS;
/// M2 is the median over this many of the best programs.
pub const TOP_MEDIAN: usize = 10;
/// M7 describes this many of the best programs: the search's own carried set size.
pub const TOP_K: usize = search::ELITES;
/// M3 counts evaluations until the running best closes this fraction of its total improvement.
pub const M3_FRACTION: f64 = 0.95;
/// The verdict when every profile-B record equals its profile-A record.
pub const NO_DIFFERENCE: &str = "no difference";
/// The verdict otherwise.
pub const DIFFERENCE: &str = "difference";

/// Gate and clause outcomes.
pub const PASS: &str = "pass";
pub const FAIL: &str = "fail";
pub const SAME: &str = "no-difference";
pub const NOT_EVALUATED: &str = "not-evaluated";

/// The version of the paired-comparison definitions ([`Paired`]).
pub const PAIRED_DEFINITIONS: &str = "kernel2-ablation-paired.v1";
/// The paired comparison's rule, as the body states it.
pub const PAIRED_RULE: &str = "runs pair by dataset and seed; B wins a pair when its M1 RMSE is \
                               lower than A's and loses when it is higher; a tie is bit-equal \
                               M1 RMSE; a difference is B minus A";
/// The fault policy of every kernel2 search: the route's default, as `jope pipeline run` uses it
/// without `--fault-policy`.
const KERNEL2_FAULT_POLICY: FaultPolicy = FaultPolicy::RejectHard;

/// Observation origins of candidates derived from measured parents
/// (`jope_pipeline::search::ORIGINS`).
const DERIVED_ORIGINS: [&str; 2] = ["mutate", "crossover"];
/// Stop reasons that leave a phase replayable from its seed.
const REPLAYABLE_STOPS: [&str; 4] = [
    search::STOPPED_BY_EVALUATIONS,
    search::STOPPED_BY_EXHAUSTED,
    search::STOPPED_BY_POOL,
    search::NOT_REQUESTED,
];

// ------------------------------------------------------------------------------------------------
// Profiles and inputs
// ------------------------------------------------------------------------------------------------

/// A named search configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// The shipping search unchanged: `run_cuts_with` with `SearchOptions::new`, which draws
    /// candidates with the v1 canonical-pipeline wrappers of `feat_math::evolve`.
    Legacy,
    /// The kernel2 route with the shipped `kernel2-baseline` profile, as
    /// `jope pipeline run --search-profile kernel2-baseline` runs it: `kernel2::search::run_cut`
    /// on the comparison's one cut, with [`Config::refinement`] and the route's default fault
    /// policy.
    Kernel2Baseline,
    /// The same route with the shipped `kernel2-experimental` profile.
    Kernel2Experimental,
}

impl Profile {
    /// Every profile, in declaration order.
    pub const ALL: [Profile; 3] = [
        Profile::Legacy,
        Profile::Kernel2Baseline,
        Profile::Kernel2Experimental,
    ];

    /// The profile's id.
    pub fn id(self) -> &'static str {
        match self {
            Profile::Legacy => "legacy",
            Profile::Kernel2Baseline => "kernel2-baseline",
            Profile::Kernel2Experimental => "kernel2-experimental",
        }
    }

    /// Whether the profile runs the kernel2 route.
    pub fn is_kernel2(self) -> bool {
        match self {
            Profile::Legacy => false,
            Profile::Kernel2Baseline | Profile::Kernel2Experimental => true,
        }
    }

    /// The kernel2 route's options for one search, or `None` for the legacy profile. Building
    /// them makes every refusal the route makes: an unknown profile, a refinement allowance above
    /// the carried set or the cut's budget, and anything but one thread and one island.
    fn kernel2_options(
        self,
        budget: Budget,
        rng_seed: u64,
        refinement: Option<Refinement>,
    ) -> Result<Option<kernel2::search::Options>> {
        if !self.is_kernel2() {
            return Ok(None);
        }
        let refinement = refinement.with_context(|| {
            format!(
                "{} requires an explicit --refine-top-k (0..={}) and --refine-trials",
                self.id(),
                search::ELITES
            )
        })?;
        kernel2::search::Options::new(
            SearchOptions::new(budget, rng_seed),
            self.id(),
            KERNEL2_FAULT_POLICY,
            refinement,
        )
        .map(Some)
    }

    /// The profile named `text`.
    pub fn parse(text: &str) -> Result<Profile> {
        Self::ALL
            .into_iter()
            .find(|profile| profile.id() == text)
            .with_context(|| {
                format!(
                    "unknown search profile {text:?}; known: {}",
                    Self::ALL.map(Profile::id).join(", ")
                )
            })
    }

    fn record(self, budget: Budget, refinement: Option<Refinement>) -> Result<ProfileRecord> {
        let Some(options) = self.kernel2_options(budget, 0, refinement)? else {
            return Ok(ProfileRecord {
                id: self.id(),
                search: search::SEARCH_ENGINE,
                candidate_source: "feat_math::evolve canonical-pipeline wrappers, v1",
                candidates_per_round: search::POPULATION,
                carried_per_round: search::ELITES,
                max_transforms: search::MAX_TRANSFORMS,
                threads: 1,
                islands: 1,
                kernel2: None,
            });
        };
        let profile = options.evaluator().profile();
        let fault_policy = match options.evaluator().fault_policy() {
            FaultPolicy::RejectHard => PolicyRecord::RejectHard,
            FaultPolicy::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            } => PolicyRecord::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            },
            other => bail!("the kernel2 route refuses fault policy {other:?}"),
        };
        Ok(ProfileRecord {
            id: self.id(),
            search: kernel2::receipt::ENGINE,
            candidate_source: "feat_math::kernel::search profile-aware pipeline wrappers",
            candidates_per_round: search::POPULATION,
            carried_per_round: search::ELITES,
            max_transforms: match profile.max_transforms.limit {
                Limit::Fixed(limit) => limit,
                Limit::Caller => search::MAX_TRANSFORMS,
            },
            threads: options.search.threads,
            islands: options.search.islands,
            kernel2: Some(Kernel2Record {
                revision: profile.revision,
                profile_sha256: profile.sha256(),
                fault_policy,
                refinement: options.refinement,
                selection: kernel2::receipt::SELECTION,
            }),
        })
    }
}

/// What to compare.
#[derive(Clone, Debug)]
pub struct Config {
    pub profile_a: Profile,
    pub profile_b: Profile,
    /// Dataset directories, or directories of them ([`crate::load::discover`]).
    pub datasets: Vec<PathBuf>,
    /// Search seeds (`--rng-seed` of the search); the fold seed stays fixed.
    pub seeds: Vec<u64>,
    /// Evaluation budget of every cut's search.
    pub evaluations: usize,
    /// Cuts per dataset.
    pub cuts: usize,
    /// Fold seed of the cut partition.
    pub fold_seed: u64,
    /// Searches run concurrently. Decisions do not depend on it.
    pub jobs: usize,
    /// The kernel2 route's refinement allowance (`--refine-top-k`, `--refine-trials`), the same
    /// for every kernel2 search of the comparison. Required when either profile is a kernel2
    /// profile and refused otherwise, as `jope pipeline run` requires and refuses it.
    pub refinement: Option<Refinement>,
}

impl Config {
    fn check(&self) -> Result<()> {
        ensure!(!self.datasets.is_empty(), "no dataset was given");
        ensure!(!self.seeds.is_empty(), "no seed was given");
        let distinct: BTreeSet<u64> = self.seeds.iter().copied().collect();
        ensure!(distinct.len() == self.seeds.len(), "a seed is listed twice");
        Budget {
            evaluations: self.evaluations,
            seconds: SEARCH_SECONDS,
        }
        .check()?;
        prepare::check_cuts(self.cuts)?;
        ensure!(self.jobs >= 1, "--jobs must be at least 1");
        self.check_kernel2()
    }

    /// Whether either profile runs the kernel2 route.
    fn has_kernel2(&self) -> bool {
        self.profile_a.is_kernel2() || self.profile_b.is_kernel2()
    }

    /// The refusals `jope pipeline run` makes for a kernel2 profile, made before anything runs:
    /// refinement options without a kernel2 profile, a kernel2 profile without them, more than
    /// one cut, and every refusal of the route's own options.
    fn check_kernel2(&self) -> Result<()> {
        if !self.has_kernel2() {
            ensure!(self.refinement.is_none(), REFINEMENT_REFUSAL);
            return Ok(());
        }
        ensure!(
            self.cuts == 1,
            "kernel2 profiles currently require --cuts 1, as jope pipeline run does"
        );
        let budget = Budget {
            evaluations: self.evaluations,
            seconds: SEARCH_SECONDS,
        };
        for profile in [self.profile_a, self.profile_b] {
            profile.kernel2_options(budget, 0, self.refinement)?;
        }
        Ok(())
    }
}

/// The refusal of refinement options when neither profile runs the kernel2 route.
const REFINEMENT_REFUSAL: &str =
    "--refine-top-k and --refine-trials apply only to kernel2 profiles; the legacy profile refuses them";

/// One dataset of the comparison.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DatasetRecord {
    pub id: String,
    pub dataset_hash: String,
    /// Verified against the catalog row before any row was parsed.
    pub train_sha256: String,
    pub train_rows: usize,
    pub features: usize,
    /// sha256 of the cut partition (`prepared.json`) every search of this dataset uses.
    pub prepared_sha256: String,
}

struct Dataset {
    record: DatasetRecord,
    path: PathBuf,
    cuts: Vec<Cut>,
}

/// Load and partition every dataset, then write each partition under `scratch`. Every refusal,
/// including a kernel2 search this comparison would repeat ([`check_unused`]), is made before
/// anything is written: `prepare::build` is [`prepare::partition`] followed by
/// [`prepare::Partition::write`], and the two halves run here with the refusals between them.
fn load_datasets(config: &Config, scratch: &Path) -> Result<Vec<Dataset>> {
    let mut dirs = Vec::new();
    for path in &config.datasets {
        dirs.extend(discover(path)?);
    }
    let mut seen = BTreeSet::new();
    let mut planned = Vec::new();
    for dir in dirs {
        let dataset = load_dataset(&dir)?;
        let task = dataset.bench_task()?;
        ensure!(
            task == Task::Regression,
            "{} ({}): the search is regression-only and this dataset is {}",
            dataset.row.id,
            dir.display(),
            task.as_str()
        );
        let hash = dataset.row.dataset_hash.clone();
        ensure!(
            seen.insert(hash.clone()),
            "{} ({}): dataset {hash} is listed twice",
            dataset.row.id,
            dir.display()
        );
        // Rows are `features..., target`, as the pipeline's own loader returns them.
        let rows: Vec<Vec<f64>> = dataset
            .x
            .iter()
            .zip(&dataset.y)
            .map(|(x, y)| {
                let mut row = x.clone();
                row.push(*y);
                row
            })
            .collect();
        let partition = prepare::partition(&rows, &hash, config.fold_seed, config.cuts)
            .with_context(|| format!("{}: prepare cuts", dataset.row.id))?;
        let record = DatasetRecord {
            id: dataset.row.id.clone(),
            dataset_hash: hash,
            train_sha256: dataset.train_sha256.clone(),
            train_rows: dataset.x.len(),
            features: dataset.x.first().map_or(0, Vec::len),
            // Filled in from the bytes the partition writes, once every refusal is made.
            prepared_sha256: String::new(),
        };
        planned.push((record, dir, partition));
    }
    for (slot, profile) in [("A", config.profile_a), ("B", config.profile_b)] {
        if !profile.is_kernel2() {
            continue;
        }
        for (record, _, _) in &planned {
            for &seed in &config.seeds {
                check_unused(&search_output(scratch, slot, &record.dataset_hash, seed))?;
            }
        }
    }
    planned
        .into_iter()
        .map(|(mut record, path, partition)| {
            let prepared = partition
                .write(&scratch.join("prepared").join(&record.dataset_hash))
                .with_context(|| format!("{}: prepare cuts", record.id))?;
            record.prepared_sha256 = prepared.sha256;
            Ok(Dataset {
                record,
                path,
                cuts: prepared.cuts,
            })
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// The document
// ------------------------------------------------------------------------------------------------

/// A `jope.bench-ablation/v1` document.
#[derive(Clone, Debug, Serialize)]
pub struct Document {
    pub schema: &'static str,
    /// sha256 of the compact body bytes the harness serialized ([`Report::body_bytes`]).
    pub body_sha256: String,
    pub body: Body,
    /// Wall-clock measurements: M10 and the gate clause that reads it. Not in the digest.
    pub timing: Timing,
    /// Where and when the document was produced. Not in the digest.
    pub provenance: Provenance,
}

/// Everything that must replay byte for byte.
#[derive(Clone, Debug, Serialize)]
pub struct Body {
    pub protocol: ProtocolRecord,
    pub datasets: Vec<DatasetRecord>,
    pub runs: Vec<RunRecord>,
    pub aggregates: Vec<SlotAggregate>,
    pub gates: Gates,
    /// [`NO_DIFFERENCE`] when every profile-B record equals its profile-A record, otherwise
    /// [`DIFFERENCE`].
    pub verdict: &'static str,
    /// The paired comparison of M1 ([`PAIRED_DEFINITIONS`]). Present when either profile is a
    /// kernel2 profile; absent, and not serialized, on a legacy-only comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paired: Option<Paired>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProtocolRecord {
    pub definitions: &'static str,
    pub profile_a: ProfileRecord,
    pub profile_b: ProfileRecord,
    pub seeds: Vec<u64>,
    pub seed_role: &'static str,
    pub budget_evaluations: usize,
    pub budget_seconds: u64,
    pub budget_policy: &'static str,
    pub cuts: usize,
    pub folds: usize,
    pub fold_seed: u64,
    pub held_out: &'static str,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProfileRecord {
    pub id: &'static str,
    pub search: &'static str,
    pub candidate_source: &'static str,
    pub candidates_per_round: usize,
    pub carried_per_round: usize,
    pub max_transforms: usize,
    pub threads: usize,
    pub islands: usize,
    /// The kernel2 route's settings; absent, and not serialized, on the legacy profile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel2: Option<Kernel2Record>,
}

/// What a kernel2 profile's record adds.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Kernel2Record {
    /// The shipped search profile's revision and the sha256 of its canonical JSON.
    pub revision: u32,
    pub profile_sha256: String,
    /// Every kernel2 search's fault policy: the route's default.
    pub fault_policy: PolicyRecord,
    /// The comparison's refinement allowance ([`Config::refinement`]).
    pub refinement: Refinement,
    /// The route's own selection rule; M1 is the program it would publish.
    pub selection: &'static str,
}

/// One (profile, dataset, seed) search.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunRecord {
    /// `A` or `B`.
    pub slot: &'static str,
    pub profile: &'static str,
    pub dataset: String,
    pub seed: u64,
    pub metrics: Metrics,
    pub cuts: Vec<CutRecord>,
}

/// The search's own accounting for one cut, without its wall-clock fields.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CutRecord {
    pub cut: usize,
    pub generations: u64,
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_measurements: u64,
    pub exploration_stopped_by: String,
    pub comparison_stopped_by: String,
    /// The kernel2 route's own accounting; absent, and not serialized, on a legacy cut.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel2: Option<Kernel2Cut>,
}

/// A kernel2 cut's accounting beyond the shared fields. The route has one phase per cut, so the
/// shared record states its stop reason as `exploration_stopped_by` and `comparison_stopped_by`
/// is `not_requested`; `completed_evaluations` is `measurements + objective_trials`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Kernel2Cut {
    /// Programs measured, refinement children included.
    pub measurements: usize,
    /// Refinement objective trials charged to the budget, and how many of them were rejected.
    pub objective_trials: usize,
    pub rejected_trials: usize,
    /// Wrapper arm draws, including unchanged edits and duplicate retries.
    pub arm_counts: BTreeMap<&'static str, usize>,
}

/// The per-run metrics. M10 is wall time and lives in [`Timing`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Metrics {
    /// Best held-out score: the program the pipeline's selection rule picks.
    #[serde(rename = "M1")]
    pub m1: Best,
    /// Median over the [`TOP_MEDIAN`] best programs of the selection ranking.
    #[serde(rename = "M2")]
    pub m2: TopMedian,
    /// Evaluations until the running best closes [`M3_FRACTION`] of its improvement.
    #[serde(rename = "M3")]
    pub m3: Speed,
    /// Fault rate: `null` until fault telemetry exists.
    #[serde(rename = "M4")]
    pub m4: Option<f64>,
    /// Hard-reject rate: `null` until fault telemetry exists.
    #[serde(rename = "M5")]
    pub m5: Option<f64>,
    /// Invalid-candidate rate from the search's own observation statuses.
    #[serde(rename = "M6")]
    pub m6: Invalid,
    /// Shape of the [`TOP_K`] best programs.
    #[serde(rename = "M7")]
    pub m7: Shape,
    /// Emitted artifact size and token cost of the best program.
    #[serde(rename = "M8")]
    pub m8: Cost,
    /// New-token usage share: `null` until profiles add tokens.
    #[serde(rename = "M9")]
    pub m9: Option<f64>,
    /// Valid unique derived candidates per evaluation.
    #[serde(rename = "M11")]
    pub m11: Yield,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Best {
    pub rmse: f64,
    pub r2: f64,
    pub cuts_measured: usize,
    pub program_sha256: String,
    pub program: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TopMedian {
    pub programs: usize,
    pub rmse: f64,
    pub r2: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Speed {
    /// Per cut, 1-based, counting every evaluation of the cut's exploration.
    pub per_cut: Vec<u64>,
    pub mean: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Invalid {
    pub rate: f64,
    pub invalid: u64,
    pub attempted: u64,
    /// Failures by the search's status, every failure status listed.
    pub by_status: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Shape {
    pub programs: usize,
    pub mean_stages: f64,
    /// `map` and `derive` stages among those programs.
    pub expression_stages: usize,
    pub mean_expression_nodes: Option<f64>,
    pub mean_expression_depth: Option<f64>,
    pub max_expression_nodes: Option<usize>,
    pub max_expression_depth: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Cost {
    pub python_bytes: Option<usize>,
    pub rust_bytes: Option<usize>,
    /// `feat_math::tokens::try_estimate_tokens`.
    pub python_tokens: Option<usize>,
    pub rust_tokens: Option<usize>,
    /// Why the emitters or the estimator refused the program, when they did.
    pub unsupported: Option<String>,
}

impl Cost {
    fn tokens(&self) -> Option<f64> {
        Some((self.python_tokens? + self.rust_tokens?) as f64)
    }

    fn bytes(&self) -> Option<f64> {
        Some((self.python_bytes? + self.rust_bytes?) as f64)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Yield {
    pub per_evaluation: f64,
    pub valid_derived: u64,
    pub evaluations: u64,
}

/// A profile's metrics averaged over seeds per dataset, then over datasets.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SlotAggregate {
    pub slot: &'static str,
    pub profile: &'static str,
    pub datasets: Vec<DatasetAggregate>,
    pub overall: Summary,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DatasetAggregate {
    pub dataset: String,
    pub seeds: usize,
    /// Sample standard deviation of M1 RMSE over seeds (`null` with one seed).
    pub m1_rmse_seed_sd: Option<f64>,
    pub mean: Summary,
}

/// Means of the per-run metrics.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Summary {
    pub m1_rmse: f64,
    pub m1_r2: f64,
    pub m2_rmse: f64,
    pub m3: f64,
    pub m6_rate: f64,
    pub m7_stages: f64,
    pub m7_expression_nodes: Option<f64>,
    pub m8_tokens: Option<f64>,
    pub m8_bytes: Option<f64>,
    pub m11: f64,
}

/// The promotion gates (workplan §17.3) as data.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Gates {
    #[serde(rename = "G-A")]
    pub g_a: Gate,
    #[serde(rename = "G-B")]
    pub g_b: Gate,
    #[serde(rename = "G-C")]
    pub g_c: Gate,
    #[serde(rename = "G-D")]
    pub g_d: Gate,
    #[serde(rename = "G-E")]
    pub g_e: Gate,
}

impl Gates {
    /// Every gate, in order.
    pub fn all(&self) -> [(&'static str, &Gate); 5] {
        [
            ("G-A", &self.g_a),
            ("G-B", &self.g_b),
            ("G-C", &self.g_c),
            ("G-D", &self.g_d),
            ("G-E", &self.g_e),
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Gate {
    /// [`PASS`], [`FAIL`], [`SAME`] or [`NOT_EVALUATED`].
    pub status: &'static str,
    /// False when a clause could not be evaluated.
    pub complete: bool,
    pub rule: &'static str,
    pub clauses: Vec<Clause>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub per_dataset: Vec<DatasetDelta>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Clause {
    pub metric: &'static str,
    pub rule: &'static str,
    pub status: &'static str,
    pub a: Option<f64>,
    pub b: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<&'static str>,
}

/// G-A's per-dataset comparison of M1 RMSE (lower is better).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DatasetDelta {
    pub dataset: String,
    /// Mean B RMSE minus mean A RMSE: positive when B is worse.
    pub delta_rmse: f64,
    /// Pooled seed standard deviation of A and B (`null` with one seed).
    pub sigma_seed: Option<f64>,
    pub worse: bool,
    pub better: bool,
}

/// The paired comparison of M1 ([`PAIRED_DEFINITIONS`], [`PAIRED_RULE`]), overall and per
/// dataset. A comparison is not accuracy evidence without these counts beside both medians.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Paired {
    pub definitions: &'static str,
    pub rule: &'static str,
    pub overall: PairedCounts,
    /// In the order of `body.datasets`.
    pub per_dataset: Vec<DatasetPaired>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DatasetPaired {
    pub dataset: String,
    #[serde(flatten)]
    pub counts: PairedCounts,
}

/// Paired counts and statistics over a set of (dataset, seed) pairs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairedCounts {
    pub pairs: usize,
    /// Pairs whose B M1 RMSE is lower than A's.
    pub wins: usize,
    /// Pairs whose two M1 RMSE values are bit-equal.
    pub ties: usize,
    /// Pairs whose B M1 RMSE is higher than A's.
    pub losses: usize,
    /// Median M1 RMSE of the A runs and of the B runs.
    pub median_a: f64,
    pub median_b: f64,
    /// Median and mean of the per-pair difference, B minus A: negative when B is better.
    pub median_difference: f64,
    pub mean_difference: f64,
    /// The pair with the smallest difference (B gains most; the first such pair), and the mean
    /// difference of the other pairs (`null` with one pair).
    pub largest_gain: PairDifference,
    pub mean_difference_without_largest_gain: Option<f64>,
    /// Two-sided exact sign test of wins against losses, ties left out ([`sign_test`]).
    pub sign_test_p: f64,
    /// Pairs whose A and B M1 programs are the same (`program_sha256`).
    pub same_winner: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairDifference {
    pub dataset: String,
    pub seed: u64,
    /// B minus A.
    pub difference: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Timing {
    pub wall_seconds: f64,
    pub runs: Vec<RunTiming>,
    /// Mean M10 per profile slot.
    #[serde(rename = "M10")]
    pub m10: BTreeMap<&'static str, f64>,
    /// G-D's wall-time clause: M10 increase at most 10%.
    #[serde(rename = "G-D-M10")]
    pub gate_d_m10: Clause,
    /// Each slot's search seconds beside `body.paired`, present exactly when it is: an equal
    /// evaluation budget is not equal compute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paired: Option<PairedTiming>,
}

/// Wall seconds of each slot's searches (`RunTiming::search_seconds`).
#[derive(Clone, Debug, Serialize)]
pub struct PairedTiming {
    pub total_seconds_a: f64,
    pub total_seconds_b: f64,
    pub median_run_seconds_a: f64,
    pub median_run_seconds_b: f64,
    /// `total_seconds_b / total_seconds_a` (`null` when A's total is zero).
    pub ratio_b_to_a: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunTiming {
    pub slot: &'static str,
    pub dataset: String,
    pub seed: u64,
    pub search_seconds: f64,
    pub evaluations: u64,
    /// Wall seconds per evaluation.
    #[serde(rename = "M10")]
    pub m10: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Provenance {
    pub tool: String,
    pub jobs: usize,
    pub dataset_paths: Vec<String>,
    pub generated_at_unix: u64,
}

/// A finished comparison: the document and the exact body bytes its digest covers.
#[derive(Clone, Debug)]
pub struct Report {
    pub document: Document,
    pub body_bytes: Vec<u8>,
}

// ------------------------------------------------------------------------------------------------
// Running
// ------------------------------------------------------------------------------------------------

struct Spec {
    slot: &'static str,
    profile: Profile,
    dataset: usize,
    seed: u64,
}

/// Run the comparison. Every search writes its evidence under `scratch`, which must not already
/// hold a search for the same (slot, dataset, seed).
pub fn run(config: &Config, scratch: &Path) -> Result<Report> {
    config.check()?;
    let started = Instant::now();
    let datasets = load_datasets(config, scratch)?;
    let mut specs = Vec::new();
    for (slot, profile) in [("A", config.profile_a), ("B", config.profile_b)] {
        for dataset in 0..datasets.len() {
            for &seed in &config.seeds {
                specs.push(Spec {
                    slot,
                    profile,
                    dataset,
                    seed,
                });
            }
        }
    }
    let budget = Budget {
        evaluations: config.evaluations,
        seconds: SEARCH_SECONDS,
    };
    let outcomes = parallel_map(&specs, config.jobs, |spec| {
        run_one(
            spec,
            &datasets[spec.dataset],
            budget,
            config.refinement,
            scratch,
        )
    });
    let mut runs = Vec::with_capacity(specs.len());
    let mut run_timing = Vec::with_capacity(specs.len());
    for outcome in outcomes {
        let (record, timing) = outcome?;
        runs.push(record);
        run_timing.push(timing);
    }

    let aggregates = vec![
        aggregate("A", config.profile_a, &datasets, &runs)?,
        aggregate("B", config.profile_b, &datasets, &runs)?,
    ];
    let gates = gates(
        config.profile_a,
        config.profile_b,
        &aggregates[0],
        &aggregates[1],
    );
    let verdict = verdict(&runs);
    let dataset_hashes: Vec<String> = datasets
        .iter()
        .map(|d| d.record.dataset_hash.clone())
        .collect();
    let paired = config
        .has_kernel2()
        .then(|| paired(&dataset_hashes, &runs))
        .transpose()?;
    let body = Body {
        protocol: ProtocolRecord {
            definitions: DEFINITIONS,
            profile_a: config.profile_a.record(budget, config.refinement)?,
            profile_b: config.profile_b.record(budget, config.refinement)?,
            seeds: config.seeds.clone(),
            seed_role: "the search's rng_seed; the fold seed is fixed, so every seed searches the same cuts",
            budget_evaluations: config.evaluations,
            budget_seconds: SEARCH_SECONDS,
            budget_policy: search::BUDGET_POLICY,
            cuts: config.cuts,
            folds: prepare::FOLDS,
            fold_seed: config.fold_seed,
            held_out: "the cut's held-out fold of train.csv; test.csv is never opened",
        },
        datasets: datasets.iter().map(|d| d.record.clone()).collect(),
        runs,
        aggregates,
        gates,
        verdict,
        paired,
    };
    let body_bytes = serde_json::to_vec(&body)?;
    let m10 = BTreeMap::from([
        (
            "A",
            mean_of(run_timing.iter().filter(|t| t.slot == "A").map(|t| t.m10)),
        ),
        (
            "B",
            mean_of(run_timing.iter().filter(|t| t.slot == "B").map(|t| t.m10)),
        ),
    ]);
    let gate_d_m10 = ratio_clause(
        "M10",
        "wall seconds per evaluation increase at most 10%",
        m10["A"],
        m10["B"],
        1.10,
    );
    let paired_timing = body.paired.is_some().then(|| paired_timing(&run_timing));
    let document = Document {
        schema: SCHEMA,
        body_sha256: sha256_hex(&body_bytes),
        body,
        timing: Timing {
            wall_seconds: started.elapsed().as_secs_f64(),
            runs: run_timing,
            m10,
            gate_d_m10,
            paired: paired_timing,
        },
        provenance: Provenance {
            tool: concat!("jope-bench ", env!("CARGO_PKG_VERSION")).to_owned(),
            jobs: config.jobs,
            dataset_paths: datasets
                .iter()
                .map(|d| d.path.display().to_string())
                .collect(),
            generated_at_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_secs()),
        },
    };
    Ok(Report {
        document,
        body_bytes,
    })
}

/// One search reduced to what the document records.
struct Searched {
    metrics: Metrics,
    cuts: Vec<CutRecord>,
    search_seconds: f64,
    evaluations: u64,
}

fn run_one(
    spec: &Spec,
    dataset: &Dataset,
    budget: Budget,
    refinement: Option<Refinement>,
    scratch: &Path,
) -> Result<(RunRecord, RunTiming)> {
    let hash = &dataset.record.dataset_hash;
    let output = search_output(scratch, spec.slot, hash, spec.seed);
    let label = || {
        format!(
            "profile {} ({}) on {} seed {}",
            spec.profile.id(),
            spec.slot,
            dataset.record.id,
            spec.seed
        )
    };
    let searched = if spec.profile.is_kernel2() {
        kernel2_search(
            spec.profile,
            &dataset.cuts,
            budget,
            spec.seed,
            refinement,
            &output,
        )
        .with_context(label)?
    } else {
        let searches = search::run_cuts_with(
            &dataset.cuts,
            &SearchOptions::new(budget, spec.seed),
            &output,
        )
        .with_context(label)?;
        check_replayable(&searches).with_context(label)?;
        legacy_searched(&searches)?
    };
    let Searched {
        metrics,
        cuts,
        search_seconds,
        evaluations,
    } = searched;
    Ok((
        RunRecord {
            slot: spec.slot,
            profile: spec.profile.id(),
            dataset: hash.clone(),
            seed: spec.seed,
            metrics,
            cuts,
        },
        RunTiming {
            slot: spec.slot,
            dataset: hash.clone(),
            seed: spec.seed,
            search_seconds,
            evaluations,
            m10: search_seconds / evaluations.max(1) as f64,
        },
    ))
}

/// Where the search of one (slot, dataset, seed) writes its evidence.
fn search_output(scratch: &Path, slot: &str, dataset_hash: &str, seed: u64) -> PathBuf {
    scratch
        .join("search")
        .join(slot)
        .join(dataset_hash)
        .join(format!("seed-{seed}"))
}

/// The refusal of a kernel2 search output that already exists.
const SCRATCH_IN_USE: &str = "already holds a kernel2 search; choose a new --scratch";

/// Refuse a kernel2 search's output directory when anything is already there, a symbolic link
/// included. The route publishes `search/cut-0` with plain writes, so a second search into the
/// same directory would replace the first one's evidence; the legacy search refuses its own
/// published `search.json` the same way, and `jope pipeline run` refuses a used `--output`.
fn check_unused(output: &Path) -> Result<()> {
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", output.display())),
        Ok(_) => bail!("{}: {SCRATCH_IN_USE}", output.display()),
    }
}

fn legacy_searched(searches: &[CutSearch]) -> Result<Searched> {
    let metrics = metrics(searches)?;
    let cuts = searches
        .iter()
        .map(|s| CutRecord {
            cut: s.cut,
            generations: s.generations,
            completed_evaluations: s.completed_evaluations,
            successful_measurements: s.successful_measurements,
            failed_measurements: s.failed_or_unavailable_measurements,
            exploration_stopped_by: s.exploration.stopped_by.clone(),
            comparison_stopped_by: s.comparison.stopped_by.clone(),
            kernel2: None,
        })
        .collect();
    Ok(Searched {
        metrics,
        cuts,
        search_seconds: searches.iter().map(|s| s.seconds).sum(),
        evaluations: searches.iter().map(|s| s.completed_evaluations).sum(),
    })
}

/// One kernel2 search on the comparison's single cut, as `jope pipeline run` runs it: refused
/// when `output` is already in use or the clock ended it, its evidence published under `output`
/// (`search/cut-0`), and reduced.
fn kernel2_search(
    profile: Profile,
    cuts: &[Cut],
    budget: Budget,
    rng_seed: u64,
    refinement: Option<Refinement>,
    output: &Path,
) -> Result<Searched> {
    check_unused(output)?;
    let options = profile
        .kernel2_options(budget, rng_seed, refinement)?
        .with_context(|| format!("{} is not a kernel2 profile", profile.id()))?;
    let [cut] = cuts else {
        bail!(
            "kernel2 profiles search exactly one cut, got {}",
            cuts.len()
        );
    };
    let searched = kernel2::search::run_cut(cut, &options)?;
    check_kernel2_replayable(&searched)?;
    let unseeded = jope_pipeline::receipt::Seeding {
        offered: 0,
        seeds: Vec::new(),
        parent_receipt_sha256: None,
        seeds_file_sha256: None,
    };
    kernel2::publication::publish(cut, &searched, unseeded, output)?;
    let failed = searched
        .candidate_measurements
        .checked_sub(searched.successful_measurements)
        .context("more successful measurements than measurements")?;
    Ok(Searched {
        metrics: kernel2_metrics(&options, &searched)?,
        cuts: vec![CutRecord {
            cut: searched.cut,
            generations: searched.generations as u64,
            completed_evaluations: searched.completed_evaluations as u64,
            successful_measurements: searched.successful_measurements as u64,
            failed_measurements: failed as u64,
            exploration_stopped_by: searched.stopped_by.to_owned(),
            comparison_stopped_by: search::NOT_REQUESTED.to_owned(),
            kernel2: Some(Kernel2Cut {
                measurements: searched.candidate_measurements,
                objective_trials: searched.objective_trials,
                rejected_trials: searched.rejected_trials,
                arm_counts: searched.arm_counts.clone(),
            }),
        }],
        search_seconds: searched.seconds,
        evaluations: searched.completed_evaluations as u64,
    })
}

/// Refuse a search whose decisions depended on the clock: a phase the seconds bound ended, or a
/// cut whose active time overran its bound.
pub fn check_replayable(searches: &[CutSearch]) -> Result<()> {
    for s in searches {
        check_phases(
            s.cut,
            &[
                ("exploration", s.exploration.stopped_by.as_str()),
                ("comparison", s.comparison.stopped_by.as_str()),
            ],
            s.seconds,
            s.budget.seconds,
        )?;
    }
    Ok(())
}

/// [`check_replayable`] for a kernel2 search, whose cut has one phase.
pub fn check_kernel2_replayable(searched: &kernel2::search::CutSearch) -> Result<()> {
    check_phases(
        searched.cut,
        &[("search", searched.stopped_by)],
        searched.seconds,
        searched.budget_seconds,
    )
}

fn check_phases(cut: usize, phases: &[(&str, &str)], seconds: f64, bound: u64) -> Result<()> {
    for (phase, stop) in phases {
        ensure!(
            REPLAYABLE_STOPS.contains(stop),
            "cut {cut}: the {phase} phase stopped by {stop:?}; only the evaluation budget or an \
             exhausted pool keeps a search replayable, so lower the evaluation budget"
        );
    }
    ensure!(
        seconds <= bound as f64,
        "cut {cut}: {seconds:.1} active seconds overran the {bound} second bound"
    );
    Ok(())
}

/// Run `work` over `items` on up to `jobs` threads; results keep the items' order.
fn parallel_map<T: Sync, R: Send>(
    items: &[T],
    jobs: usize,
    work: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    let jobs = jobs.clamp(1, items.len().max(1));
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(index) else {
                    break;
                };
                let result = work(item);
                *slots[index].lock().expect("result slot") = Some(result);
            });
        }
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .expect("result slot")
                .expect("every item ran")
        })
        .collect()
}

// ------------------------------------------------------------------------------------------------
// Metrics
// ------------------------------------------------------------------------------------------------

fn finite(what: &str, value: f64) -> Result<f64> {
    ensure!(value.is_finite(), "{what} is not finite: {value}");
    Ok(value)
}

/// The median of `values` (the mean of the middle two for an even count).
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    Some(if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    })
}

/// Sample standard deviation (`n - 1`), or `None` for fewer than two values.
pub fn sample_sd(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let squares: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    Some((squares / (values.len() - 1) as f64).sqrt())
}

fn mean_of(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), v| (sum + v, count + 1));
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn mean_of_options(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values: Option<Vec<f64>> = values.collect();
    let values = values?;
    (!values.is_empty()).then(|| mean_of(values.into_iter()))
}

/// M3: the 1-based number of evaluations after which the running best held-out RMSE has closed
/// `fraction` of the gap between the first measured candidate and the final best. A search that
/// never improves on its first measured candidate reports that candidate's position. `rmse` holds
/// one entry per evaluation, in order, `None` for a candidate that did not measure.
pub fn evaluations_to_fraction(rmse: &[Option<f64>], fraction: f64) -> Option<u64> {
    let first = rmse.iter().position(Option::is_some)?;
    let start = rmse[first]?;
    let final_best = rmse.iter().flatten().copied().fold(f64::INFINITY, f64::min);
    let gap = start - final_best;
    if gap <= 0.0 {
        return Some(first as u64 + 1);
    }
    let mut best = f64::INFINITY;
    for (index, value) in rmse.iter().enumerate() {
        if let Some(value) = value {
            if *value < best {
                best = *value;
            }
        }
        if start - best >= fraction * gap {
            return Some(index as u64 + 1);
        }
    }
    Some(rmse.len() as u64)
}

/// A program's stage count and the node count and depth of each of its expressions.
fn program_shape(program: &str) -> Result<(usize, Vec<(usize, usize)>)> {
    let genome = GenomeIr::parse(program).with_context(|| format!("parse {program}"))?;
    Ok(match genome {
        GenomeIr::Pipeline(dsl) => {
            let mut expressions = Vec::new();
            for stage in &dsl.stages {
                if !matches!(stage.op.as_str(), "map" | "derive") {
                    continue;
                }
                expressions.push(match stage.params.get("expr") {
                    Some(ParamValue::Str(source)) => expression_size(
                        MathExpr::parse(source)
                            .with_context(|| format!("expression {source:?}"))?
                            .ast(),
                    ),
                    // A constant-only expression prints unquoted and parses back as a number.
                    Some(ParamValue::Int(_) | ParamValue::Float(_)) => (1, 1),
                    other => bail!("{program}: expression parameter {other:?}"),
                });
            }
            (dsl.stages.len(), expressions)
        }
        GenomeIr::Symbolic { expr, .. } => (1, vec![expression_size(expr.ast())]),
        GenomeIr::Portfolio(portfolio) => (
            portfolio.members.iter().map(|m| m.stages.len()).sum(),
            Vec::new(),
        ),
    })
}

/// `(nodes, depth)` of an expression, counted as the expression parser counts them.
pub fn expression_size(expr: &Expr) -> (usize, usize) {
    match expr {
        Expr::Const(_) | Expr::Var(_) => (1, 1),
        Expr::Unary(_, child) => {
            let (nodes, depth) = expression_size(child);
            (nodes + 1, depth + 1)
        }
        Expr::Binary(_, left, right) => {
            let (left_nodes, left_depth) = expression_size(left);
            let (right_nodes, right_depth) = expression_size(right);
            (
                left_nodes + right_nodes + 1,
                left_depth.max(right_depth) + 1,
            )
        }
        Expr::Call(_, args) => {
            let sizes: Vec<(usize, usize)> = args.iter().map(expression_size).collect();
            (
                sizes.iter().map(|(nodes, _)| nodes).sum::<usize>() + 1,
                sizes.iter().map(|(_, depth)| *depth).max().unwrap_or(0) + 1,
            )
        }
    }
}

fn cost(program: &str) -> Cost {
    let unsupported = |reason: String| Cost {
        python_bytes: None,
        rust_bytes: None,
        python_tokens: None,
        rust_tokens: None,
        unsupported: Some(reason),
    };
    let genome = match GenomeIr::parse(program) {
        Ok(genome) => genome,
        Err(error) => return unsupported(format!("{error:#}")),
    };
    let options = EmitOptions::default();
    let emitted = match feat_math::transpile::emit(&genome, &options) {
        Ok(emitted) => emitted,
        Err(error) => return unsupported(format!("{error:#}")),
    };
    match feat_math::tokens::try_estimate_tokens(&genome, &options) {
        Ok(tokens) => Cost {
            python_bytes: Some(emitted.python.len()),
            rust_bytes: Some(emitted.rust.len()),
            python_tokens: Some(tokens.py),
            rust_tokens: Some(tokens.rs),
            unsupported: None,
        },
        Err(missing) => unsupported(missing.join("; ")),
    }
}

/// One program of a route's selection ranking, best first.
struct Ranked<'a> {
    rmse: f64,
    r2: f64,
    cuts_measured: usize,
    program_sha256: &'a str,
    genome: &'a str,
}

/// Every metric of one run from its cuts' searches.
pub fn metrics(searches: &[CutSearch]) -> Result<Metrics> {
    let per_cut: Vec<(usize, &[Observation])> = searches
        .iter()
        .map(|s| (s.cut, s.observations.as_slice()))
        .collect();
    let selected = select::ranking(&per_cut)?;
    let ranking: Vec<Ranked<'_>> = selected
        .iter()
        .map(|b| Ranked {
            rmse: b.validation_rmse,
            r2: b.validation_r2,
            cuts_measured: b.cuts_measured,
            program_sha256: &b.program_sha256,
            genome: &b.genome,
        })
        .collect();
    // The search's own candidates, in evaluation order; comparison re-measurements excluded.
    let exploration: Vec<Vec<&Observation>> = searches
        .iter()
        .map(|s| {
            s.observations
                .iter()
                .filter(|o| o.origin != search::REMEASURE)
                .collect()
        })
        .collect();
    ranked_metrics(&ranking, &exploration)
}

/// Every metric of one kernel2 run. The ranking is the route's own (the evaluator's
/// fault-aware validation score, programs once each), so M1 is the program the route
/// publishes, measured on its one cut. The route has no comparison phase, so every measurement,
/// refinement children included, counts as exploration.
pub fn kernel2_metrics(
    options: &kernel2::search::Options,
    searched: &kernel2::search::CutSearch,
) -> Result<Metrics> {
    let mut seen = BTreeSet::new();
    let mut ranking = Vec::new();
    for measured in options.evaluator().rank(&searched.observations) {
        let o = &measured.observation;
        if !seen.insert(o.genome.as_str()) {
            continue;
        }
        ranking.push(Ranked {
            rmse: o
                .rmse
                .context("a ranked kernel2 measurement has no held-out RMSE")?,
            r2: o
                .hold_r2
                .context("a ranked kernel2 measurement has no held-out R2")?,
            cuts_measured: 1,
            program_sha256: &o.program_sha256,
            genome: &o.genome,
        });
    }
    ensure!(
        ranking.first().map(|b| b.genome) == Some(searched.best.observation.genome.as_str()),
        "the kernel2 ranking does not start with the program the search published"
    );
    let exploration = vec![searched
        .observations
        .iter()
        .map(|o| &o.observation)
        .collect()];
    ranked_metrics(&ranking, &exploration)
}

/// Every metric from a route's ranking and each cut's exploration observations, in evaluation
/// order.
fn ranked_metrics(ranking: &[Ranked<'_>], exploration: &[Vec<&Observation>]) -> Result<Metrics> {
    let best = ranking
        .first()
        .context("no program was measured on every cut")?;

    let top: Vec<&Ranked<'_>> = ranking.iter().take(TOP_MEDIAN).collect();
    let top_rmse: Vec<f64> = top.iter().map(|b| b.rmse).collect();
    let top_r2: Vec<f64> = top.iter().map(|b| b.r2).collect();

    let mut per_cut_speed = Vec::with_capacity(exploration.len());
    for observations in exploration {
        let rmse: Vec<Option<f64>> = observations
            .iter()
            .map(|o| if o.measured() { o.rmse } else { None })
            .collect();
        per_cut_speed.push(
            evaluations_to_fraction(&rmse, M3_FRACTION)
                .context("a cut measured no candidate in exploration")?,
        );
    }
    let attempted = exploration.iter().map(Vec::len).sum::<usize>() as u64;
    ensure!(attempted > 0, "the search attempted no candidate");
    let mut by_status: BTreeMap<String, u64> = search::STATUSES
        .iter()
        .filter(|status| **status != search::MEASURED)
        .map(|status| ((*status).to_owned(), 0))
        .collect();
    let mut valid_derived = 0u64;
    for observation in exploration.iter().flatten() {
        if observation.measured() {
            if DERIVED_ORIGINS.contains(&observation.origin) {
                valid_derived += 1;
            }
        } else {
            *by_status.entry(observation.status.to_owned()).or_default() += 1;
        }
    }
    let invalid: u64 = by_status.values().sum();

    let mut stages = Vec::new();
    let mut expressions = Vec::new();
    for program in ranking.iter().take(TOP_K) {
        let (count, sizes) = program_shape(program.genome)?;
        stages.push(count as f64);
        expressions.extend(sizes);
    }
    let nodes: Vec<f64> = expressions.iter().map(|(n, _)| *n as f64).collect();
    let depths: Vec<f64> = expressions.iter().map(|(_, d)| *d as f64).collect();

    Ok(Metrics {
        m1: Best {
            rmse: finite("M1 rmse", best.rmse)?,
            r2: finite("M1 r2", best.r2)?,
            cuts_measured: best.cuts_measured,
            program_sha256: best.program_sha256.to_owned(),
            program: best.genome.to_owned(),
        },
        m2: TopMedian {
            programs: top.len(),
            rmse: finite("M2 rmse", median(&top_rmse).context("empty ranking")?)?,
            r2: finite("M2 r2", median(&top_r2).context("empty ranking")?)?,
        },
        m3: Speed {
            mean: mean_of(per_cut_speed.iter().map(|v| *v as f64)),
            per_cut: per_cut_speed,
        },
        m4: None,
        m5: None,
        m6: Invalid {
            rate: invalid as f64 / attempted as f64,
            invalid,
            attempted,
            by_status,
        },
        m7: Shape {
            programs: stages.len(),
            mean_stages: mean_of(stages.iter().copied()),
            expression_stages: expressions.len(),
            mean_expression_nodes: (!nodes.is_empty()).then(|| mean_of(nodes.iter().copied())),
            mean_expression_depth: (!depths.is_empty()).then(|| mean_of(depths.iter().copied())),
            max_expression_nodes: expressions.iter().map(|(n, _)| *n).max(),
            max_expression_depth: expressions.iter().map(|(_, d)| *d).max(),
        },
        m8: cost(best.genome),
        m9: None,
        m11: Yield {
            per_evaluation: valid_derived as f64 / attempted as f64,
            valid_derived,
            evaluations: attempted,
        },
    })
}

// ------------------------------------------------------------------------------------------------
// Aggregates, gates and the verdict
// ------------------------------------------------------------------------------------------------

fn summary<'a>(runs: impl Iterator<Item = &'a RunRecord> + Clone) -> Summary {
    Summary {
        m1_rmse: mean_of(runs.clone().map(|r| r.metrics.m1.rmse)),
        m1_r2: mean_of(runs.clone().map(|r| r.metrics.m1.r2)),
        m2_rmse: mean_of(runs.clone().map(|r| r.metrics.m2.rmse)),
        m3: mean_of(runs.clone().map(|r| r.metrics.m3.mean)),
        m6_rate: mean_of(runs.clone().map(|r| r.metrics.m6.rate)),
        m7_stages: mean_of(runs.clone().map(|r| r.metrics.m7.mean_stages)),
        m7_expression_nodes: mean_of_options(
            runs.clone()
                .filter(|r| r.metrics.m7.expression_stages > 0)
                .map(|r| r.metrics.m7.mean_expression_nodes),
        ),
        m8_tokens: mean_of_options(runs.clone().map(|r| r.metrics.m8.tokens())),
        m8_bytes: mean_of_options(runs.clone().map(|r| r.metrics.m8.bytes())),
        m11: mean_of(runs.map(|r| r.metrics.m11.per_evaluation)),
    }
}

fn aggregate(
    slot: &'static str,
    profile: Profile,
    datasets: &[Dataset],
    runs: &[RunRecord],
) -> Result<SlotAggregate> {
    let mut per_dataset = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let hash = &dataset.record.dataset_hash;
        let mine = runs.iter().filter(|r| r.slot == slot && &r.dataset == hash);
        let rmse: Vec<f64> = mine.clone().map(|r| r.metrics.m1.rmse).collect();
        ensure!(!rmse.is_empty(), "slot {slot} has no run on {hash}");
        per_dataset.push(DatasetAggregate {
            dataset: hash.clone(),
            seeds: rmse.len(),
            m1_rmse_seed_sd: sample_sd(&rmse),
            mean: summary(mine),
        });
    }
    let means = per_dataset.iter().map(|d| &d.mean);
    let overall = Summary {
        m1_rmse: mean_of(means.clone().map(|m| m.m1_rmse)),
        m1_r2: mean_of(means.clone().map(|m| m.m1_r2)),
        m2_rmse: mean_of(means.clone().map(|m| m.m2_rmse)),
        m3: mean_of(means.clone().map(|m| m.m3)),
        m6_rate: mean_of(means.clone().map(|m| m.m6_rate)),
        m7_stages: mean_of(means.clone().map(|m| m.m7_stages)),
        m7_expression_nodes: mean_of_options(
            means
                .clone()
                .filter(|m| m.m7_expression_nodes.is_some())
                .map(|m| m.m7_expression_nodes),
        ),
        m8_tokens: mean_of_options(means.clone().map(|m| m.m8_tokens)),
        m8_bytes: mean_of_options(means.clone().map(|m| m.m8_bytes)),
        m11: mean_of(means.map(|m| m.m11)),
    };
    Ok(SlotAggregate {
        slot,
        profile: profile.id(),
        datasets: per_dataset,
        overall,
    })
}

/// A clause comparing B with A where lower is better: pass when `b <= a * limit`.
fn ratio_clause(metric: &'static str, rule: &'static str, a: f64, b: f64, limit: f64) -> Clause {
    let status = if a == b {
        SAME
    } else if b <= a * limit {
        PASS
    } else {
        FAIL
    };
    Clause {
        metric,
        rule,
        status,
        a: Some(a),
        b: Some(b),
        note: None,
    }
}

/// A clause over a value both profiles may lack for the same reason, such as the M7 expression
/// size when no best program of either profile has a `map` or `derive` stage: equal absence is no
/// difference.
fn optional_ratio_clause(
    metric: &'static str,
    rule: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    limit: f64,
) -> Clause {
    match (a, b) {
        (Some(a), Some(b)) => ratio_clause(metric, rule, a, b, limit),
        (None, None) => Clause {
            metric,
            rule,
            status: SAME,
            a: None,
            b: None,
            note: Some("neither profile has a value"),
        },
        _ => Clause {
            metric,
            rule,
            status: NOT_EVALUATED,
            a,
            b,
            note: Some("only one profile has a value"),
        },
    }
}

/// A clause over a value every run should supply, such as M8: an aggregate is `None` when some
/// run lacks it, so the clause is evaluated only when both profiles have a value. A value missing
/// from both profiles was never compared, so it is not evaluated, never no difference.
fn every_run_ratio_clause(
    metric: &'static str,
    rule: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    limit: f64,
) -> Clause {
    match (a, b) {
        (Some(a), Some(b)) => ratio_clause(metric, rule, a, b, limit),
        _ => Clause {
            metric,
            rule,
            status: NOT_EVALUATED,
            a,
            b,
            note: Some(
                "a run's best program has no value: the emitters or the token estimator refused \
                 it (see that run's M8 unsupported)",
            ),
        },
    }
}

fn not_evaluated(metric: &'static str, rule: &'static str, note: &'static str) -> Clause {
    Clause {
        metric,
        rule,
        status: NOT_EVALUATED,
        a: None,
        b: None,
        note: Some(note),
    }
}

/// A gate whose clauses must all hold: it fails on any failing clause, reports no difference
/// when every evaluated clause does, and is not evaluated when no clause could be.
fn all_of(rule: &'static str, clauses: Vec<Clause>) -> Gate {
    let evaluated: Vec<&Clause> = clauses
        .iter()
        .filter(|c| c.status != NOT_EVALUATED)
        .collect();
    let status = if evaluated.is_empty() {
        NOT_EVALUATED
    } else if evaluated.iter().any(|c| c.status == FAIL) {
        FAIL
    } else if evaluated.iter().all(|c| c.status == SAME) {
        SAME
    } else {
        PASS
    };
    Gate {
        status,
        complete: evaluated.len() == clauses.len(),
        rule,
        clauses,
        per_dataset: Vec::new(),
    }
}

/// G-A: not worse than A by more than 0.25 sigma on any dataset, and better by at least one
/// sigma on two datasets or M3 at least 10% lower.
fn quality_gate(a: &SlotAggregate, b: &SlotAggregate) -> Gate {
    const RULE: &str = "mean M1 RMSE of B is not worse than A by more than 0.25 sigma_seed on any \
                        dataset, and better by at least 1 sigma_seed on at least 2 datasets or \
                        mean M3 at least 10% lower";
    let mut per_dataset = Vec::new();
    for (da, db) in a.datasets.iter().zip(&b.datasets) {
        let delta = db.mean.m1_rmse - da.mean.m1_rmse;
        let sigma = match (da.m1_rmse_seed_sd, db.m1_rmse_seed_sd) {
            (Some(sa), Some(sb)) => Some(((sa * sa + sb * sb) / 2.0).sqrt()),
            _ => None,
        };
        let (worse, better) = match sigma {
            Some(sigma) => (
                delta > 0.0 && delta > 0.25 * sigma,
                delta < 0.0 && -delta >= sigma,
            ),
            None => (false, false),
        };
        per_dataset.push(DatasetDelta {
            dataset: da.dataset.clone(),
            delta_rmse: delta,
            sigma_seed: sigma,
            worse,
            better,
        });
    }
    let identical = per_dataset.iter().all(|d| d.delta_rmse == 0.0) && a.overall.m3 == b.overall.m3;
    let sigma_known = per_dataset.iter().all(|d| d.sigma_seed.is_some());
    let non_inferior = !per_dataset.iter().any(|d| d.worse);
    let better = per_dataset.iter().filter(|d| d.better).count();
    let speed = b.overall.m3 <= 0.9 * a.overall.m3;
    let clause = |metric, rule, holds: bool| Clause {
        metric,
        rule,
        status: if identical {
            SAME
        } else if !sigma_known {
            NOT_EVALUATED
        } else if holds {
            PASS
        } else {
            FAIL
        },
        a: None,
        b: None,
        note: (!sigma_known && !identical).then_some("sigma_seed needs at least two seeds"),
    };
    let clauses = vec![
        clause(
            "M1",
            "not worse by more than 0.25 sigma_seed on any dataset",
            non_inferior,
        ),
        clause(
            "M1",
            "better by at least 1 sigma_seed on at least 2 datasets",
            better >= 2,
        ),
        Clause {
            a: Some(a.overall.m3),
            b: Some(b.overall.m3),
            ..clause("M3", "mean M3 at least 10% lower", speed)
        },
    ];
    let status = if identical {
        SAME
    } else if !sigma_known {
        NOT_EVALUATED
    } else if non_inferior && (better >= 2 || speed) {
        PASS
    } else {
        FAIL
    };
    Gate {
        status,
        complete: identical || sigma_known,
        rule: RULE,
        clauses,
        per_dataset,
    }
}

/// The gates of workplan §17.3 over the two profiles' aggregates.
fn gates(profile_a: Profile, profile_b: Profile, a: &SlotAggregate, b: &SlotAggregate) -> Gates {
    let (oa, ob) = (&a.overall, &b.overall);
    let parity = Gate {
        status: if profile_a == profile_b {
            SAME
        } else {
            NOT_EVALUATED
        },
        complete: profile_a == profile_b,
        rule: "the parity lanes are bit-green with the profile's new elements in the corpus",
        clauses: vec![Clause {
            metric: "parity",
            rule: "crates/feat-math/tests/parity_bits.rs is green",
            status: if profile_a == profile_b {
                SAME
            } else {
                NOT_EVALUATED
            },
            a: None,
            b: None,
            note: Some(
                "evaluated by the parity test, not by the search; the same profile adds no \
                 element",
            ),
        }],
        per_dataset: Vec::new(),
    };
    Gates {
        g_a: quality_gate(a, b),
        g_b: parity,
        g_c: all_of(
            "M4(B) - M4(A) at most 0.02; M5 and M6 increase at most 0.01",
            vec![
                not_evaluated("M4", "increase at most 0.02", "no fault telemetry yet"),
                not_evaluated("M5", "increase at most 0.01", "no fault telemetry yet"),
                Clause {
                    metric: "M6",
                    rule: "increase at most 0.01",
                    status: if oa.m6_rate == ob.m6_rate {
                        SAME
                    } else if ob.m6_rate - oa.m6_rate <= 0.01 {
                        PASS
                    } else {
                        FAIL
                    },
                    a: Some(oa.m6_rate),
                    b: Some(ob.m6_rate),
                    note: None,
                },
            ],
        ),
        g_d: all_of(
            "M7 non-increasing on average; M8 increase at most 5%; M10 (timing) at most 10%",
            vec![
                ratio_clause(
                    "M7",
                    "mean stages non-increasing",
                    oa.m7_stages,
                    ob.m7_stages,
                    1.0,
                ),
                optional_ratio_clause(
                    "M7",
                    "mean expression nodes non-increasing",
                    oa.m7_expression_nodes,
                    ob.m7_expression_nodes,
                    1.0,
                ),
                every_run_ratio_clause(
                    "M8",
                    "token cost increase at most 5%",
                    oa.m8_tokens,
                    ob.m8_tokens,
                    1.05,
                ),
                every_run_ratio_clause(
                    "M8",
                    "artifact bytes increase at most 5%",
                    oa.m8_bytes,
                    ob.m8_bytes,
                    1.05,
                ),
            ],
        ),
        g_e: all_of(
            "M9 > 0: the new elements are used by the best programs",
            vec![not_evaluated(
                "M9",
                "greater than 0",
                "no profile adds elements yet",
            )],
        ),
    }
}

/// [`NO_DIFFERENCE`] when every B run's metrics and cut accounting equal the A run's for the
/// same dataset and seed.
fn verdict(runs: &[RunRecord]) -> &'static str {
    let key = |r: &RunRecord| (r.dataset.clone(), r.seed);
    let a: BTreeMap<_, _> = runs
        .iter()
        .filter(|r| r.slot == "A")
        .map(|r| (key(r), r))
        .collect();
    let b: BTreeMap<_, _> = runs
        .iter()
        .filter(|r| r.slot == "B")
        .map(|r| (key(r), r))
        .collect();
    let same = a.len() == b.len()
        && a.iter().all(|(k, ra)| {
            b.get(k)
                .is_some_and(|rb| ra.metrics == rb.metrics && ra.cuts == rb.cuts)
        });
    if same {
        NO_DIFFERENCE
    } else {
        DIFFERENCE
    }
}

// ------------------------------------------------------------------------------------------------
// The paired comparison
// ------------------------------------------------------------------------------------------------

/// One (dataset, seed) pair: A's and B's M1.
struct Pair<'a> {
    dataset: &'a str,
    seed: u64,
    a: &'a Best,
    b: &'a Best,
}

/// The paired comparison of M1 over `runs` ([`PAIRED_RULE`]): overall, and per dataset in the
/// order of `datasets` (dataset hashes). The pairing is one to one on (dataset, seed): a dataset
/// listed twice, a slot other than A or B, a run on a dataset not listed, two runs of one slot
/// with the same dataset and seed, and a run without its partner in the other slot are refused.
pub fn paired(datasets: &[String], runs: &[RunRecord]) -> Result<Paired> {
    let mut listed = BTreeSet::new();
    for dataset in datasets {
        ensure!(
            listed.insert(dataset.as_str()),
            "dataset {dataset} is listed twice"
        );
    }
    let mut a_keys = BTreeSet::new();
    let mut b_runs = BTreeMap::new();
    for run in runs {
        let (dataset, seed, slot) = (run.dataset.as_str(), run.seed, run.slot);
        ensure!(
            listed.contains(dataset),
            "dataset {dataset} seed {seed}: the {slot} run is on a dataset the comparison does \
             not list"
        );
        let first = match slot {
            "A" => a_keys.insert((dataset, seed)),
            "B" => b_runs.insert((dataset, seed), run).is_none(),
            other => bail!("dataset {dataset} seed {seed}: run slot {other:?} is neither A nor B"),
        };
        ensure!(
            first,
            "dataset {dataset} seed {seed}: two {slot} runs; runs pair one to one"
        );
    }
    let mut all = Vec::new();
    let mut per_dataset = Vec::with_capacity(datasets.len());
    for dataset in datasets {
        let mut pairs = Vec::new();
        for a in runs
            .iter()
            .filter(|r| r.slot == "A" && &r.dataset == dataset)
        {
            let b = b_runs
                .remove(&(a.dataset.as_str(), a.seed))
                .with_context(|| {
                    format!(
                        "dataset {dataset} seed {}: an A run needs exactly one B run",
                        a.seed
                    )
                })?;
            pairs.push(Pair {
                dataset,
                seed: a.seed,
                a: &a.metrics.m1,
                b: &b.metrics.m1,
            });
        }
        per_dataset.push(DatasetPaired {
            dataset: dataset.clone(),
            counts: paired_counts(&pairs).with_context(|| format!("dataset {dataset}"))?,
        });
        all.extend(pairs);
    }
    if let Some((dataset, seed)) = b_runs.keys().next() {
        bail!("dataset {dataset} seed {seed}: a B run needs exactly one A run");
    }
    Ok(Paired {
        definitions: PAIRED_DEFINITIONS,
        rule: PAIRED_RULE,
        overall: paired_counts(&all)?,
        per_dataset,
    })
}

fn paired_counts(pairs: &[Pair<'_>]) -> Result<PairedCounts> {
    ensure!(!pairs.is_empty(), "no pair to compare");
    let (mut wins, mut ties, mut losses) = (0, 0, 0);
    for pair in pairs {
        let (a, b) = (pair.a.rmse, pair.b.rmse);
        if a.to_bits() == b.to_bits() {
            ties += 1;
        } else if b < a {
            wins += 1;
        } else if b > a {
            losses += 1;
        } else {
            bail!(
                "dataset {} seed {}: M1 RMSE {a} and {b} do not compare",
                pair.dataset,
                pair.seed
            );
        }
    }
    let a: Vec<f64> = pairs.iter().map(|p| p.a.rmse).collect();
    let b: Vec<f64> = pairs.iter().map(|p| p.b.rmse).collect();
    let differences: Vec<f64> = pairs.iter().map(|p| p.b.rmse - p.a.rmse).collect();
    // The first pair with the smallest difference.
    let mut largest = 0;
    for (index, difference) in differences.iter().enumerate() {
        if *difference < differences[largest] {
            largest = index;
        }
    }
    let others = differences
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != largest)
        .map(|(_, difference)| *difference);
    Ok(PairedCounts {
        pairs: pairs.len(),
        wins,
        ties,
        losses,
        median_a: median(&a).context("no pair")?,
        median_b: median(&b).context("no pair")?,
        median_difference: median(&differences).context("no pair")?,
        mean_difference: mean_of(differences.iter().copied()),
        largest_gain: PairDifference {
            dataset: pairs[largest].dataset.to_owned(),
            seed: pairs[largest].seed,
            difference: differences[largest],
        },
        mean_difference_without_largest_gain: (pairs.len() > 1).then(|| mean_of(others)),
        sign_test_p: sign_test(wins as u64, losses as u64),
        same_winner: pairs
            .iter()
            .filter(|p| p.a.program_sha256 == p.b.program_sha256)
            .count(),
    })
}

/// The two-sided exact sign test of `wins` against `losses` (ties already left out): the
/// probability, for a fair coin over `wins + losses` tosses, of a split at least as uneven as the
/// one observed, capped at 1. It is 1 when there is no untied pair. Only multiplications and
/// divisions are used, with the halvings of `2^-n` interleaved so no step overflows, so the value
/// replays on any host.
pub fn sign_test(wins: u64, losses: u64) -> f64 {
    let n = wins + losses;
    let tail = wins.min(losses);
    // P(X = tail) = C(n, tail) / 2^n.
    let mut point = 1.0_f64;
    let mut halvings = n;
    for i in 1..=tail {
        point *= (n - tail + i) as f64 / i as f64;
        while point >= 1.0 && halvings > 0 {
            point *= 0.5;
            halvings -= 1;
        }
    }
    for _ in 0..halvings {
        point *= 0.5;
    }
    // P(X <= tail) / P(X = tail) = sum over k <= tail of C(n, k) / C(n, tail).
    let mut ratio = 1.0_f64;
    let mut sum = 1.0_f64;
    for k in (1..=tail).rev() {
        ratio *= k as f64 / (n - k + 1) as f64;
        sum += ratio;
    }
    (2.0 * point * sum).min(1.0)
}

/// Each slot's search seconds, for `timing.paired`.
fn paired_timing(runs: &[RunTiming]) -> PairedTiming {
    let seconds = |slot: &str| -> Vec<f64> {
        runs.iter()
            .filter(|t| t.slot == slot)
            .map(|t| t.search_seconds)
            .collect()
    };
    let (a, b) = (seconds("A"), seconds("B"));
    let (total_a, total_b) = (a.iter().sum::<f64>(), b.iter().sum::<f64>());
    PairedTiming {
        total_seconds_a: total_a,
        total_seconds_b: total_b,
        median_run_seconds_a: median(&a).unwrap_or(0.0),
        median_run_seconds_b: median(&b).unwrap_or(0.0),
        ratio_b_to_a: (total_a > 0.0).then(|| total_b / total_a),
    }
}

/// Write the document as pretty JSON plus a newline; returns the file's sha256.
pub fn write_document(path: &Path, document: &Document) -> Result<String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
    }
    let mut text = serde_json::to_string_pretty(document)?;
    text.push('\n');
    std::fs::write(path, &text).with_context(|| format!("write {}", path.display()))?;
    Ok(sha256_hex(text.as_bytes()))
}

// ------------------------------------------------------------------------------------------------
// Command line
// ------------------------------------------------------------------------------------------------

/// `jope-bench ablation`: compare two search profiles at an equal evaluation budget.
#[derive(clap::Args, Clone, Debug)]
pub struct AblationArgs {
    /// Profile A, the reference.
    #[arg(long, default_value = "legacy")]
    pub profile_a: String,
    /// Profile B, the candidate.
    #[arg(long, default_value = "legacy")]
    pub profile_b: String,
    /// Refinement parents per round for kernel2-baseline and kernel2-experimental (0..=4), the
    /// same for both profiles. Required when either profile is one of them; refused otherwise.
    #[arg(long)]
    pub refine_top_k: Option<usize>,
    /// Objective trials per refined parent, inside each cut's evaluation budget. Required and
    /// refused together with --refine-top-k.
    #[arg(long)]
    pub refine_trials: Option<usize>,
    /// A dataset directory (metadata.json + train.csv) or a directory of them (repeatable).
    #[arg(long = "datasets", required = true)]
    pub datasets: Vec<PathBuf>,
    /// Search seeds, comma-separated.
    #[arg(long, value_delimiter = ',', default_value = "1,2,3,4,5")]
    pub seeds: Vec<u64>,
    /// Evaluation budget of every cut's search (1..=1000).
    #[arg(long, default_value_t = 64)]
    pub evaluations: usize,
    /// Cuts per dataset (1..=5).
    #[arg(long, default_value_t = 1)]
    pub cuts: usize,
    /// Fold seed of the cut partition (the pipeline's default).
    #[arg(long, default_value_t = jope_pipeline::DEFAULT_SEED)]
    pub fold_seed: u64,
    /// Searches run concurrently; the results do not depend on it.
    #[arg(long, default_value_t = 1)]
    pub jobs: usize,
    /// Where to write the jope.bench-ablation/v1 document.
    #[arg(long)]
    pub out: PathBuf,
    /// Keep every search's evidence under this directory (default: a temporary directory under
    /// TMPDIR, removed afterwards; with neither this nor TMPDIR the run is refused rather than use
    /// the system temporary directory). It must not already hold a search of the same run.
    #[arg(long)]
    pub scratch: Option<PathBuf>,
}

/// Where a run without `--scratch` makes its temporary scratch directory: `TMPDIR`, when it is
/// set and not empty. There is no fallback to the system temporary directory; without `TMPDIR`
/// the run is refused and asks for `--scratch`.
fn default_scratch_root(tmpdir: Option<std::ffi::OsString>) -> Result<PathBuf> {
    match tmpdir {
        Some(dir) if !dir.is_empty() => Ok(PathBuf::from(dir)),
        _ => bail!(
            "no scratch directory: pass --scratch <dir> or set TMPDIR; the harness does not fall \
             back to the system temporary directory"
        ),
    }
}

/// The comparison `args` asks for. A refinement option given alone is refused here; every other
/// refusal of the options is [`Config`]'s, which [`execute`] makes before it creates a scratch
/// directory and [`run`] makes again.
fn config(args: &AblationArgs) -> Result<Config> {
    let profile_a = Profile::parse(&args.profile_a)?;
    let profile_b = Profile::parse(&args.profile_b)?;
    let kernel2 = profile_a.is_kernel2() || profile_b.is_kernel2();
    let refinement = match (args.refine_top_k, args.refine_trials) {
        (None, None) => None,
        (Some(top_k), Some(trials_per_program)) => Some(Refinement {
            top_k,
            trials_per_program,
        }),
        (None, Some(_)) if kernel2 => bail!(
            "a kernel2 profile requires an explicit --refine-top-k (0..={})",
            search::ELITES
        ),
        (Some(_), None) if kernel2 => {
            bail!("a kernel2 profile requires an explicit --refine-trials")
        }
        _ => bail!(REFINEMENT_REFUSAL),
    };
    Ok(Config {
        profile_a,
        profile_b,
        datasets: args.datasets.clone(),
        seeds: args.seeds.clone(),
        evaluations: args.evaluations,
        cuts: args.cuts,
        fold_seed: args.fold_seed,
        jobs: args.jobs,
        refinement,
    })
}

/// One line of the command's paired summary.
fn paired_line(label: &str, c: &PairedCounts) -> String {
    format!(
        "  {label}: W/T/L {}/{}/{} over {} pairs, median M1 RMSE A {} B {}, sign test p {}, \
         same M1 program {}/{}",
        c.wins,
        c.ties,
        c.losses,
        c.pairs,
        c.median_a,
        c.median_b,
        c.sign_test_p,
        c.same_winner,
        c.pairs
    )
}

/// Run `jope-bench ablation`. Every refusal of the options is made before the scratch directory
/// is created.
pub fn execute(args: &AblationArgs) -> Result<()> {
    let config = config(args)?;
    config.check()?;
    let temporary;
    let scratch = match &args.scratch {
        Some(dir) => {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            dir.clone()
        }
        None => {
            let root = default_scratch_root(std::env::var_os("TMPDIR"))?;
            std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
            temporary = tempfile::Builder::new()
                .prefix("jope-bench-ablation-")
                .tempdir_in(&root)
                .with_context(|| format!("create a scratch directory under {}", root.display()))?;
            temporary.path().to_path_buf()
        }
    };
    let report = run(&config, &scratch)?;
    let file_sha256 = write_document(&args.out, &report.document)?;
    let body = &report.document.body;
    eprintln!(
        "ablation {} (A) vs {} (B): {} datasets x {} seeds, {} evaluations per cut: {}",
        config.profile_a.id(),
        config.profile_b.id(),
        body.datasets.len(),
        config.seeds.len(),
        config.evaluations,
        body.verdict
    );
    for (name, gate) in body.gates.all() {
        eprintln!("  {name}: {}", gate.status);
    }
    if let Some(paired) = &body.paired {
        eprintln!("{}", paired_line("paired (B against A)", &paired.overall));
        for (dataset, record) in paired.per_dataset.iter().zip(&body.datasets) {
            eprintln!("{}", paired_line(&record.id, &dataset.counts));
        }
    }
    if let Some(seconds) = &report.document.timing.paired {
        eprintln!(
            "  search seconds: A {:.3} B {:.3} (B/A {})",
            seconds.total_seconds_a,
            seconds.total_seconds_b,
            seconds
                .ratio_b_to_a
                .map_or_else(|| "undefined".to_owned(), |r| format!("{r:.3}"))
        );
    }
    eprintln!(
        "wrote {} (body_sha256 {}, file sha256 {})",
        args.out.display(),
        report.document.body_sha256,
        file_sha256
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d5(slug: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../datasets/test")
            .join(slug)
    }

    /// A scratch directory under `TMPDIR`, or under the workspace's `target/tmp` when `TMPDIR` is
    /// not set (`CARGO_TARGET_TMPDIR` is not defined for unit tests); never under the system
    /// temporary directory.
    fn scratch() -> tempfile::TempDir {
        let root = default_scratch_root(std::env::var_os("TMPDIR"))
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp"));
        std::fs::create_dir_all(&root).unwrap();
        tempfile::Builder::new()
            .prefix("jope-bench-ablation-test-")
            .tempdir_in(root)
            .unwrap()
    }

    /// Evaluations of the PR-tier comparison: the twelve first-round candidates and four
    /// derived ones.
    const SMALL_EVALUATIONS: usize = 16;

    /// Two narrow D5 datasets (3 and 1 features) and two seeds whose candidates fit in seconds
    /// in a debug build, so the A=A test stays in the PR tier.
    fn small_config() -> Config {
        Config {
            profile_a: Profile::Legacy,
            profile_b: Profile::Legacy,
            datasets: vec![
                d5("rdatasets-agridat-bancroft-peanut-uniformity"),
                d5("pmlb-first-principles-supernovae-zg"),
            ],
            seeds: vec![5, 6],
            evaluations: SMALL_EVALUATIONS,
            cuts: 1,
            fold_seed: jope_pipeline::DEFAULT_SEED,
            jobs: 4,
            refinement: None,
        }
    }

    /// A=A twice on a tiny budget, with different job counts: byte-identical bodies, and every
    /// gate says there is no difference or that it has nothing to evaluate.
    #[test]
    fn ablation_a_equals_a_replays_byte_for_byte_and_reports_no_difference() {
        let config = small_config();
        let (first_dir, second_dir) = (scratch(), scratch());
        let first = run(&config, first_dir.path()).unwrap();
        let second = run(
            &Config {
                jobs: 2,
                ..config.clone()
            },
            second_dir.path(),
        )
        .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&first.body_bytes),
            String::from_utf8_lossy(&second.body_bytes)
        );
        assert_eq!(first.document.body_sha256, second.document.body_sha256);
        assert_eq!(
            first.document.body_sha256,
            sha256_hex(&serde_json::to_vec(&first.document.body).unwrap())
        );

        let body = &first.document.body;
        assert_eq!(body.verdict, NO_DIFFERENCE);
        for (name, gate) in body.gates.all() {
            assert!(
                gate.status == SAME || gate.status == NOT_EVALUATED,
                "{name}: {}",
                gate.status
            );
        }
        assert_eq!(body.gates.g_a.status, SAME);
        assert_eq!(body.datasets.len(), 2);
        assert_eq!(body.runs.len(), 2 * 2 * 2);
        assert_eq!(body.aggregates[0].overall, body.aggregates[1].overall);
        for record in &body.runs {
            let m = &record.metrics;
            assert!(m.m1.rmse.is_finite() && m.m1.rmse >= 0.0, "{record:?}");
            assert!(m.m2.programs >= 1 && m.m2.programs <= TOP_MEDIAN);
            assert!(m
                .m3
                .per_cut
                .iter()
                .all(|v| (1..=SMALL_EVALUATIONS as u64).contains(v)));
            assert!(m.m4.is_none() && m.m5.is_none() && m.m9.is_none());
            assert_eq!(m.m6.attempted, SMALL_EVALUATIONS as u64);
            assert!(m.m7.programs >= 1 && m.m7.mean_stages >= 1.0);
            assert!(m.m8.unsupported.is_none(), "{:?}", m.m8);
            assert!(m.m8.python_tokens.unwrap() > 0 && m.m8.rust_bytes.unwrap() > 0);
            for cut in &record.cuts {
                assert!(
                    cut.exploration_stopped_by == search::STOPPED_BY_EVALUATIONS
                        || cut.exploration_stopped_by == search::STOPPED_BY_EXHAUSTED,
                    "{cut:?}"
                );
            }
        }
        // Sixteen evaluations reach the first derived candidates.
        assert!(body.runs.iter().any(|r| r.metrics.m11.valid_derived > 0));
        // Wall time lives outside the body.
        let text = String::from_utf8(first.body_bytes).unwrap();
        assert!(!text.contains("seconds_per") && !text.contains("\"M10\""));
        assert_eq!(first.document.timing.runs.len(), body.runs.len());
        assert!(first.document.timing.m10["A"] > 0.0);
    }

    #[test]
    fn ablation_metric_definitions() {
        // M3: 95% of the improvement from the first measured candidate to the final best.
        let curve = [
            None,
            Some(10.0),
            Some(12.0),
            Some(6.0),
            Some(5.2),
            Some(5.0),
        ];
        // gap 5.0; 95% is 4.75 -> the running best must reach 5.25: the fifth evaluation.
        assert_eq!(evaluations_to_fraction(&curve, 0.95), Some(5));
        assert_eq!(evaluations_to_fraction(&curve, 0.5), Some(4));
        assert_eq!(
            evaluations_to_fraction(&[Some(3.0), Some(4.0)], 0.95),
            Some(1)
        );
        assert_eq!(evaluations_to_fraction(&[None, None], 0.95), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        assert_eq!(sample_sd(&[1.0]), None);
        assert_eq!(sample_sd(&[1.0, 3.0]), Some(2f64.sqrt()));
        let expr = MathExpr::parse("sin(c0) + 2 * c1").unwrap();
        assert_eq!(expression_size(expr.ast()), (6, 3));
        let (stages, expressions) = program_shape(
            "(pipe X (derive expr=\"sin(c0) + 2 * c1\" append=true) (standardize) (ridge))",
        )
        .unwrap();
        assert_eq!((stages, expressions), (3, vec![(6, 3)]));
        assert_eq!(program_shape("(pipe X (ridge))").unwrap(), (1, vec![]));
        for origin in DERIVED_ORIGINS {
            assert!(search::ORIGINS.contains(&origin), "{origin}");
        }
    }

    fn summary_with(m1_rmse: f64, m3: f64) -> Summary {
        Summary {
            m1_rmse,
            m1_r2: 0.5,
            m2_rmse: m1_rmse,
            m3,
            m6_rate: 0.0,
            m7_stages: 2.0,
            m7_expression_nodes: None,
            m8_tokens: Some(1000.0),
            m8_bytes: Some(9000.0),
            m11: 0.3,
        }
    }

    fn slot(name: &'static str, rmse: [f64; 2], sd: Option<f64>, m3: f64) -> SlotAggregate {
        SlotAggregate {
            slot: name,
            profile: "legacy",
            datasets: rmse
                .iter()
                .enumerate()
                .map(|(i, r)| DatasetAggregate {
                    dataset: format!("d{i}"),
                    seeds: 5,
                    m1_rmse_seed_sd: sd,
                    mean: summary_with(*r, m3),
                })
                .collect(),
            overall: summary_with((rmse[0] + rmse[1]) / 2.0, m3),
        }
    }

    #[test]
    fn ablation_gates_report_a_difference() {
        let a = slot("A", [1.0, 2.0], Some(0.1), 10.0);
        // B is worse on d1 by more than 0.25 sigma: G-A fails.
        let worse = slot("B", [1.0, 2.1], Some(0.1), 10.0);
        let gate = quality_gate(&a, &worse);
        assert_eq!(gate.status, FAIL);
        assert!(gate.per_dataset[1].worse);
        // B is better by at least one sigma on both datasets: G-A passes.
        let better = slot("B", [0.8, 1.8], Some(0.1), 10.0);
        assert_eq!(quality_gate(&a, &better).status, PASS);
        // Equal quality but 20% faster: G-A passes through M3.
        let faster = slot("B", [1.0, 2.0], Some(0.1), 8.0);
        assert_eq!(quality_gate(&a, &faster).status, PASS);
        // One seed leaves sigma undefined.
        let one_seed = slot("B", [0.9, 1.9], None, 10.0);
        assert_eq!(
            quality_gate(&slot("A", [1.0, 2.0], None, 10.0), &one_seed).status,
            NOT_EVALUATED
        );
        // Identical aggregates: no difference.
        assert_eq!(quality_gate(&a, &a.clone()).status, SAME);
        // G-D: a token cost 10% higher fails.
        let mut costly = a.clone();
        costly.overall.m8_tokens = Some(1100.0);
        let g = gates(Profile::Legacy, Profile::Legacy, &a, &costly);
        assert_eq!(g.g_d.status, FAIL);
        assert_eq!(g.g_c.status, SAME);
        assert!(!g.g_c.complete);
        assert_eq!(g.g_e.status, NOT_EVALUATED);
        // G-D: an M8 value some run lacks was never compared, so its clauses are not evaluated,
        // even when both profiles lack it; M7 expression size missing from both is no difference.
        let clauses = |g: &Gates, metric: &str| -> Vec<&'static str> {
            g.g_d
                .clauses
                .iter()
                .filter(|c| c.metric == metric)
                .map(|c| c.status)
                .collect()
        };
        let mut refused = a.clone();
        refused.overall.m8_tokens = None;
        refused.overall.m8_bytes = None;
        let neither = gates(Profile::Legacy, Profile::Legacy, &refused, &refused.clone());
        assert_eq!(clauses(&neither, "M8"), [NOT_EVALUATED; 2]);
        assert_eq!(clauses(&neither, "M7"), [SAME; 2]);
        assert_eq!(neither.g_d.status, SAME);
        assert!(!neither.g_d.complete);
        let one = gates(Profile::Legacy, Profile::Legacy, &a, &refused);
        assert_eq!(clauses(&one, "M8"), [NOT_EVALUATED; 2]);
        assert!(!one.g_d.complete);
        let both = gates(Profile::Legacy, Profile::Legacy, &a, &a.clone());
        assert_eq!(clauses(&both, "M8"), [SAME; 2]);
        assert!(both.g_d.complete);
    }

    #[test]
    fn ablation_refuses_a_phase_the_seconds_bound_ended() {
        let phase = |stopped_by: &str| jope_pipeline::receipt::Phase {
            evaluation_limit: 20,
            seconds_limit: 600.0,
            completed_evaluations: 3,
            successful_measurements: 3,
            failed_measurements: 0,
            seconds: 600.5,
            stopped_by: stopped_by.to_owned(),
        };
        let cut = |exploration: &str, seconds: f64| CutSearch {
            cut: 0,
            output: PathBuf::new(),
            train_rows: 10,
            hold_rows: 3,
            generations: 1,
            completed_evaluations: 3,
            successful_measurements: 3,
            failed_or_unavailable_measurements: 0,
            stopped_by: search::STOPPED_BY_EVALUATIONS,
            seconds,
            exploration: phase(exploration),
            comparison: jope_pipeline::receipt::Phase::not_requested(),
            best_validation_rmse: None,
            best_validation_r2: None,
            best_program: None,
            migrations: Vec::new(),
            observations: Vec::new(),
            search_sha256: String::new(),
            observations_sha256: String::new(),
            rng_seed: 1,
            budget: Budget {
                evaluations: 20,
                seconds: SEARCH_SECONDS,
            },
        };
        check_replayable(&[cut(search::STOPPED_BY_EVALUATIONS, 1.0)]).unwrap();
        let error = check_replayable(&[cut(search::STOPPED_BY_SECONDS, 1.0)]).unwrap_err();
        assert!(error.to_string().contains("replayable"), "{error:#}");
        let error = check_replayable(&[cut(search::STOPPED_BY_EVALUATIONS, 601.0)]).unwrap_err();
        assert!(error.to_string().contains("overran"), "{error:#}");
    }

    #[test]
    fn ablation_profiles_and_inputs_are_checked() {
        assert_eq!(Profile::parse("legacy").unwrap(), Profile::Legacy);
        let error = Profile::parse("other").unwrap_err().to_string();
        assert!(error.contains("known: legacy"), "{error}");
        // Without --scratch the scratch root is TMPDIR, and never the system default.
        assert_eq!(
            default_scratch_root(Some("/data/scratch".into())).unwrap(),
            PathBuf::from("/data/scratch")
        );
        for unset in [None, Some(std::ffi::OsString::new())] {
            let error = default_scratch_root(unset).unwrap_err().to_string();
            assert!(
                error.contains("--scratch") && error.contains("TMPDIR"),
                "{error}"
            );
        }
        let dir = scratch();
        for broken in [
            Config {
                seeds: vec![],
                ..small_config()
            },
            Config {
                seeds: vec![1, 1],
                ..small_config()
            },
            Config {
                evaluations: search::MAX_EVALUATIONS + 1,
                ..small_config()
            },
            Config {
                cuts: prepare::FOLDS + 1,
                ..small_config()
            },
            Config {
                datasets: vec![d5("pmlb-229-pwlinear"), d5("pmlb-229-pwlinear")],
                ..small_config()
            },
        ] {
            assert!(run(&broken, dir.path()).is_err(), "{broken:?}");
        }
    }

    /// The ten D5 regression datasets, five seeds, A=A twice. Minutes of CPU: run it on purpose
    /// with `JOPE_ABLATION_HEAVY=1` and `--run-ignored only`.
    #[test]
    #[ignore = "heavy: the ten D5 regression datasets x five seeds; set JOPE_ABLATION_HEAVY=1"]
    fn ablation_heavy_a_equals_a_on_every_d5_regression_dataset() {
        assert!(
            std::env::var_os("JOPE_ABLATION_HEAVY").is_some_and(|v| v == "1"),
            "the heavy ablation runs only with JOPE_ABLATION_HEAVY=1"
        );
        let config = Config {
            datasets: vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test")],
            seeds: vec![1, 2, 3, 4, 5],
            evaluations: 64,
            jobs: std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8),
            ..small_config()
        };
        let (first_dir, second_dir) = (scratch(), scratch());
        let first = run(&config, first_dir.path()).unwrap();
        let second = run(&config, second_dir.path()).unwrap();
        assert_eq!(first.body_bytes, second.body_bytes);
        assert_eq!(first.document.body.datasets.len(), 10);
        assert_eq!(first.document.body.verdict, NO_DIFFERENCE);
    }

    /// Evaluations of the PR-tier kernel2 comparison, and its refinement allowance: one refined
    /// parent per round with two objective trials.
    const KERNEL2_EVALUATIONS: usize = 16;
    const SMALL_REFINEMENT: Refinement = Refinement {
        top_k: 1,
        trials_per_program: 2,
    };

    /// The PR-tier kernel2 comparison: the legacy comparison's two datasets with kernel2-baseline
    /// on both sides and seed 6 only. Seed 5 draws boosted-tree heads of 1000 rounds there, which
    /// take 13 to 20 seconds a search in a debug build; seed 6's take about 3 and still spend
    /// refinement trials.
    fn kernel2_config() -> Config {
        Config {
            profile_a: Profile::Kernel2Baseline,
            profile_b: Profile::Kernel2Baseline,
            seeds: vec![6],
            evaluations: KERNEL2_EVALUATIONS,
            refinement: Some(SMALL_REFINEMENT),
            ..small_config()
        }
    }

    /// kernel2-baseline against itself, twice with different job counts: byte-identical bodies
    /// (wall time is in `timing`, never in the body), no difference, the route's own winner as
    /// M1, consistent budget accounting, and a paired comparison of nothing but ties.
    #[test]
    fn ablation_kernel2_a_equals_a_replays_byte_for_byte() {
        let config = kernel2_config();
        let (first_dir, second_dir) = (scratch(), scratch());
        let first = run(&config, first_dir.path()).unwrap();
        let second = run(
            &Config {
                jobs: 2,
                ..config.clone()
            },
            second_dir.path(),
        )
        .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&first.body_bytes),
            String::from_utf8_lossy(&second.body_bytes)
        );
        assert_eq!(first.document.body_sha256, second.document.body_sha256);

        let body = &first.document.body;
        assert_eq!(body.verdict, NO_DIFFERENCE);
        assert_eq!(body.gates.g_a.status, SAME);
        assert_eq!(body.gates.g_b.status, SAME);
        // The profile record names the shipped profile, its digest, the policy and the allowance.
        let shipped = feat_math::kernel::profile::shipped_kernel2_profiles()
            .into_iter()
            .find(|p| p.id == "kernel2-baseline")
            .unwrap();
        let record = &body.protocol.profile_a;
        assert_eq!(record.id, "kernel2-baseline");
        assert_eq!(record.search, kernel2::receipt::ENGINE);
        assert_eq!((record.threads, record.islands), (1, 1));
        let settings = record.kernel2.as_ref().unwrap();
        assert_eq!(settings.profile_sha256, shipped.sha256());
        assert_eq!(settings.revision, shipped.revision);
        assert_eq!(settings.fault_policy, PolicyRecord::RejectHard);
        assert_eq!(settings.refinement, SMALL_REFINEMENT);
        assert_eq!(settings.selection, kernel2::receipt::SELECTION);
        assert_eq!(body.protocol.profile_a, body.protocol.profile_b);

        assert_eq!(body.runs.len(), 2 * 2);
        for searched in &body.runs {
            let m = &searched.metrics;
            assert!(m.m1.rmse.is_finite() && m.m1.rmse >= 0.0, "{searched:?}");
            assert_eq!(m.m1.cuts_measured, 1);
            assert_eq!(m.m1.program_sha256, sha256_hex(m.m1.program.as_bytes()));
            let [cut] = searched.cuts.as_slice() else {
                panic!("one cut: {searched:?}");
            };
            let accounting = cut.kernel2.as_ref().unwrap();
            assert_eq!(
                cut.completed_evaluations as usize,
                accounting.measurements + accounting.objective_trials
            );
            assert!(cut.completed_evaluations as usize <= KERNEL2_EVALUATIONS);
            assert_eq!(m.m6.attempted as usize, accounting.measurements);
            assert!(REPLAYABLE_STOPS.contains(&cut.exploration_stopped_by.as_str()));
            assert_eq!(cut.comparison_stopped_by, search::NOT_REQUESTED);
        }
        // The allowance is spent: some run charged refinement trials.
        assert!(body
            .runs
            .iter()
            .any(|r| r.cuts[0].kernel2.as_ref().unwrap().objective_trials > 0));

        let paired = body.paired.as_ref().unwrap();
        assert_eq!(paired.definitions, PAIRED_DEFINITIONS);
        let all = &paired.overall;
        assert_eq!((all.pairs, all.wins, all.ties, all.losses), (2, 0, 2, 0));
        assert_eq!((all.sign_test_p, all.same_winner), (1.0, 2));
        assert_eq!(all.median_a, all.median_b);
        assert_eq!((all.median_difference, all.mean_difference), (0.0, 0.0));
        assert_eq!(paired.per_dataset.len(), 2);
        for (dataset, record) in paired.per_dataset.iter().zip(&body.datasets) {
            assert_eq!(dataset.dataset, record.dataset_hash);
            assert_eq!((dataset.counts.pairs, dataset.counts.ties), (1, 1));
        }

        let text = String::from_utf8(first.body_bytes.clone()).unwrap();
        assert!(!text.contains("seconds_per") && !text.contains("\"active_seconds\""));
        assert!(!text.contains("\"M10\""));
        let seconds = first.document.timing.paired.as_ref().unwrap();
        assert!(seconds.total_seconds_a > 0.0 && seconds.total_seconds_b > 0.0);
        assert_eq!(first.document.timing.runs.len(), body.runs.len());
        // Each search's evidence is published under the scratch directory.
        let evidence = first_dir
            .path()
            .join("search/A")
            .join(&body.datasets[0].dataset_hash)
            .join("seed-6/search/cut-0/search.json");
        assert!(evidence.is_file(), "{}", evidence.display());
    }

    fn sorted_keys(value: &serde_json::Value) -> Vec<String> {
        let mut keys: Vec<String> = value.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        keys
    }

    /// The kernel2 arms leave the legacy document as it was: the legacy profile record and a
    /// legacy cut record serialize to the bytes they had before, and a legacy-only run's body,
    /// run records and timing carry exactly the keys they had before.
    #[test]
    fn ablation_legacy_document_is_unchanged_by_the_kernel2_arms() {
        let budget = Budget {
            evaluations: SMALL_EVALUATIONS,
            seconds: SEARCH_SECONDS,
        };
        assert_eq!(
            serde_json::to_string(&Profile::Legacy.record(budget, None).unwrap()).unwrap(),
            "{\"id\":\"legacy\",\"search\":\"feat-math canonical-pipeline GP (in-process)\",\
             \"candidate_source\":\"feat_math::evolve canonical-pipeline wrappers, v1\",\
             \"candidates_per_round\":12,\"carried_per_round\":4,\"max_transforms\":4,\
             \"threads\":1,\"islands\":1}"
        );
        let cut = CutRecord {
            cut: 0,
            generations: 2,
            completed_evaluations: 16,
            successful_measurements: 15,
            failed_measurements: 1,
            exploration_stopped_by: search::STOPPED_BY_EVALUATIONS.to_owned(),
            comparison_stopped_by: search::NOT_REQUESTED.to_owned(),
            kernel2: None,
        };
        assert_eq!(
            serde_json::to_string(&cut).unwrap(),
            "{\"cut\":0,\"generations\":2,\"completed_evaluations\":16,\
             \"successful_measurements\":15,\"failed_measurements\":1,\
             \"exploration_stopped_by\":\"evaluations\",\
             \"comparison_stopped_by\":\"not_requested\"}"
        );

        let config = Config {
            datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
            seeds: vec![5],
            ..small_config()
        };
        let dir = scratch();
        let report = run(&config, dir.path()).unwrap();
        let body: serde_json::Value = serde_json::from_slice(&report.body_bytes).unwrap();
        assert_eq!(
            sorted_keys(&body),
            [
                "aggregates",
                "datasets",
                "gates",
                "protocol",
                "runs",
                "verdict"
            ]
        );
        assert_eq!(
            sorted_keys(&body["protocol"]["profile_a"]),
            [
                "candidate_source",
                "candidates_per_round",
                "carried_per_round",
                "id",
                "islands",
                "max_transforms",
                "search",
                "threads"
            ]
        );
        assert_eq!(
            sorted_keys(&body["runs"][0]),
            ["cuts", "dataset", "metrics", "profile", "seed", "slot"]
        );
        assert_eq!(
            sorted_keys(&body["runs"][0]["cuts"][0]),
            [
                "comparison_stopped_by",
                "completed_evaluations",
                "cut",
                "exploration_stopped_by",
                "failed_measurements",
                "generations",
                "successful_measurements"
            ]
        );
        let text = String::from_utf8(report.body_bytes.clone()).unwrap();
        assert!(!text.contains("\"paired\"") && !text.contains("\"kernel2\""));
        let timing = serde_json::to_value(&report.document.timing).unwrap();
        assert_eq!(
            sorted_keys(&timing),
            ["G-D-M10", "M10", "runs", "wall_seconds"]
        );
    }

    /// The kernel2 arms refuse what `jope pipeline run` refuses for a kernel2 profile, before any
    /// search starts, and the command line refuses a refinement option given alone.
    #[test]
    fn ablation_kernel2_refuses_what_the_pipeline_refuses() {
        assert_eq!(
            Profile::parse("kernel2-baseline").unwrap(),
            Profile::Kernel2Baseline
        );
        assert_eq!(
            Profile::parse("kernel2-experimental").unwrap(),
            Profile::Kernel2Experimental
        );
        let error = Profile::parse("kernel2").unwrap_err().to_string();
        assert!(
            error.contains("known: legacy, kernel2-baseline, kernel2-experimental"),
            "{error}"
        );
        for profile in Profile::ALL {
            assert_eq!(profile.is_kernel2(), profile.id().starts_with("kernel2-"));
        }

        // Accepted: a kernel2 profile on either side or both, with the allowance.
        kernel2_config().check().unwrap();
        Config {
            profile_a: Profile::Legacy,
            ..kernel2_config()
        }
        .check()
        .unwrap();
        Config {
            profile_b: Profile::Kernel2Experimental,
            ..kernel2_config()
        }
        .check()
        .unwrap();

        let refused = |config: Config, needle: &str| {
            let error = format!("{:#}", config.check().unwrap_err());
            assert!(error.contains(needle), "{needle:?} not in {error:?}");
        };
        // A kernel2 profile without the allowance.
        refused(
            Config {
                refinement: None,
                ..kernel2_config()
            },
            "--refine-top-k",
        );
        refused(
            Config {
                profile_a: Profile::Legacy,
                refinement: None,
                ..kernel2_config()
            },
            "--refine-trials",
        );
        // The allowance without a kernel2 profile.
        refused(
            Config {
                refinement: Some(SMALL_REFINEMENT),
                ..small_config()
            },
            "legacy profile refuses",
        );
        // More than one cut.
        refused(
            Config {
                cuts: 2,
                ..kernel2_config()
            },
            "--cuts 1",
        );
        // More refined parents than the carried set; more trials than the cut's budget.
        for refinement in [
            Refinement {
                top_k: search::ELITES + 1,
                trials_per_program: 2,
            },
            Refinement {
                top_k: 1,
                trials_per_program: KERNEL2_EVALUATIONS + 1,
            },
        ] {
            refused(
                Config {
                    refinement: Some(refinement),
                    ..kernel2_config()
                },
                "refinement allowance",
            );
        }
        // run refuses before preparing a cut or starting a search.
        let dir = scratch();
        assert!(run(
            &Config {
                cuts: 2,
                ..kernel2_config()
            },
            dir.path()
        )
        .is_err());
        assert!(!dir.path().join("prepared").exists() && !dir.path().join("search").exists());

        // The command line: the two options go together, and only with a kernel2 profile.
        let args = |a: &str, b: &str, top_k: Option<usize>, trials: Option<usize>| AblationArgs {
            profile_a: a.to_owned(),
            profile_b: b.to_owned(),
            refine_top_k: top_k,
            refine_trials: trials,
            datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
            seeds: vec![5],
            evaluations: KERNEL2_EVALUATIONS,
            cuts: 1,
            fold_seed: jope_pipeline::DEFAULT_SEED,
            jobs: 1,
            out: PathBuf::from("unused.json"),
            scratch: None,
        };
        let parsed = config(&args("legacy", "kernel2-baseline", Some(1), Some(2))).unwrap();
        assert_eq!(
            (parsed.profile_a, parsed.profile_b, parsed.refinement),
            (
                Profile::Legacy,
                Profile::Kernel2Baseline,
                Some(SMALL_REFINEMENT)
            )
        );
        assert_eq!(
            config(&args("legacy", "legacy", None, None))
                .unwrap()
                .refinement,
            None
        );
        for (profile, top_k, trials, needle) in [
            ("kernel2-baseline", None, Some(2), "--refine-top-k"),
            ("kernel2-experimental", Some(1), None, "--refine-trials"),
            ("legacy", Some(1), None, "legacy profile refuses"),
            ("legacy", None, Some(2), "legacy profile refuses"),
        ] {
            let error = config(&args(profile, profile, top_k, trials))
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{needle:?} not in {error:?}");
        }
        let both = config(&args("legacy", "legacy", Some(1), Some(2))).unwrap();
        assert!(format!("{:#}", both.check().unwrap_err()).contains("legacy profile refuses"));
    }

    /// A run record carrying what the paired comparison reads: slot, dataset, seed and M1.
    fn paired_run(
        slot: &'static str,
        dataset: &str,
        seed: u64,
        rmse: f64,
        program: &str,
    ) -> RunRecord {
        RunRecord {
            slot,
            profile: "kernel2-baseline",
            dataset: dataset.to_owned(),
            seed,
            metrics: Metrics {
                m1: Best {
                    rmse,
                    r2: 0.5,
                    cuts_measured: 1,
                    program_sha256: sha256_hex(program.as_bytes()),
                    program: program.to_owned(),
                },
                m2: TopMedian {
                    programs: 1,
                    rmse,
                    r2: 0.5,
                },
                m3: Speed {
                    per_cut: vec![1],
                    mean: 1.0,
                },
                m4: None,
                m5: None,
                m6: Invalid {
                    rate: 0.0,
                    invalid: 0,
                    attempted: 1,
                    by_status: BTreeMap::new(),
                },
                m7: Shape {
                    programs: 1,
                    mean_stages: 1.0,
                    expression_stages: 0,
                    mean_expression_nodes: None,
                    mean_expression_depth: None,
                    max_expression_nodes: None,
                    max_expression_depth: None,
                },
                m8: Cost {
                    python_bytes: None,
                    rust_bytes: None,
                    python_tokens: None,
                    rust_tokens: None,
                    unsupported: None,
                },
                m9: None,
                m11: Yield {
                    per_evaluation: 0.0,
                    valid_derived: 0,
                    evaluations: 1,
                },
            },
            cuts: Vec::new(),
        }
    }

    /// The paired comparison on a synthetic input whose every value is exact in binary: wins,
    /// ties and losses overall and per dataset, both medians, the differences, the largest gain
    /// and the mean without it, the sign test, the same-winner count, an all-tie input, the
    /// bit-equality rule for a tie, and the refusal of unpaired runs.
    #[test]
    fn ablation_paired_summary_arithmetic() {
        let runs: Vec<RunRecord> = [
            // (dataset, seed, A RMSE, B RMSE, A program, B program)
            ("d0", 1, 1.0, 0.75, "p", "q"), // win, -0.25
            ("d0", 2, 1.0, 1.0, "p", "p"),  // tie, same program
            ("d0", 3, 1.0, 1.5, "p", "q"),  // loss, +0.5
            ("d1", 1, 2.0, 1.5, "p", "q"),  // win, -0.5
            ("d1", 2, 2.0, 1.75, "p", "q"), // win, -0.25
            ("d1", 3, 2.0, 2.0, "p", "p"),  // tie, same program
        ]
        .iter()
        .flat_map(|&(dataset, seed, a, b, program_a, program_b)| {
            [
                paired_run("A", dataset, seed, a, program_a),
                paired_run("B", dataset, seed, b, program_b),
            ]
        })
        .collect();
        let datasets = vec!["d0".to_owned(), "d1".to_owned()];
        let summary = paired(&datasets, &runs).unwrap();
        assert_eq!(
            (summary.definitions, summary.rule),
            (PAIRED_DEFINITIONS, PAIRED_RULE)
        );
        let all = &summary.overall;
        assert_eq!((all.pairs, all.wins, all.ties, all.losses), (6, 3, 2, 1));
        assert_eq!((all.median_a, all.median_b), (1.5, 1.5));
        assert_eq!(all.median_difference, -0.125);
        assert_eq!(all.mean_difference, -0.5 / 6.0);
        assert_eq!(
            all.largest_gain,
            PairDifference {
                dataset: "d1".to_owned(),
                seed: 1,
                difference: -0.5
            }
        );
        assert_eq!(all.mean_difference_without_largest_gain, Some(0.0));
        // 2 * (C(4,0) + C(4,1)) / 2^4.
        assert_eq!(all.sign_test_p, 0.625);
        assert_eq!(all.same_winner, 2);

        let [d0, d1] = summary.per_dataset.as_slice() else {
            panic!("two datasets: {summary:?}");
        };
        assert_eq!((d0.dataset.as_str(), d1.dataset.as_str()), ("d0", "d1"));
        let c = &d0.counts;
        assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 1, 1, 1));
        assert_eq!(
            (c.median_a, c.median_b, c.median_difference),
            (1.0, 1.0, 0.0)
        );
        assert_eq!(c.mean_difference, 0.25 / 3.0);
        assert_eq!((c.largest_gain.seed, c.largest_gain.difference), (1, -0.25));
        assert_eq!(c.mean_difference_without_largest_gain, Some(0.25));
        assert_eq!((c.sign_test_p, c.same_winner), (1.0, 1));
        let c = &d1.counts;
        assert_eq!((c.pairs, c.wins, c.ties, c.losses), (3, 2, 1, 0));
        assert_eq!(
            (c.median_a, c.median_b, c.median_difference),
            (2.0, 1.75, -0.25)
        );
        assert_eq!(c.mean_difference, -0.25);
        assert_eq!(c.mean_difference_without_largest_gain, Some(-0.125));
        // 2 * C(2,0) / 2^2.
        assert_eq!((c.sign_test_p, c.same_winner), (0.5, 1));

        // Nothing but ties: no untied pair, so p is 1, and every difference is zero.
        let ties: Vec<RunRecord> = [1, 2, 3]
            .iter()
            .flat_map(|&seed| {
                [
                    paired_run("A", "d0", seed, 0.5, "p"),
                    paired_run("B", "d0", seed, 0.5, "p"),
                ]
            })
            .collect();
        let tied = paired(&datasets[..1], &ties).unwrap().overall;
        assert_eq!(
            (tied.wins, tied.ties, tied.losses, tied.same_winner),
            (0, 3, 0, 3)
        );
        assert_eq!(tied.sign_test_p, 1.0);
        assert_eq!((tied.median_difference, tied.mean_difference), (0.0, 0.0));
        assert_eq!(tied.largest_gain.seed, 1);
        assert_eq!(tied.mean_difference_without_largest_gain, Some(0.0));

        // A tie is bit-equality: the next RMSE above A's is a loss.
        let near = [
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, f64::from_bits(1.0_f64.to_bits() + 1), "q"),
        ];
        let near = paired(&datasets[..1], &near).unwrap().overall;
        assert_eq!((near.wins, near.ties, near.losses), (0, 0, 1));
        assert_eq!(near.mean_difference_without_largest_gain, None);

        // Every A run needs exactly one B run, and every B run an A run.
        assert!(paired(&datasets, &runs[..runs.len() - 1]).is_err());
        for extra in [
            paired_run("B", "d0", 9, 1.0, "p"),
            paired_run("B", "d0", 1, 1.0, "p"),
        ] {
            let mut runs = runs.clone();
            runs.push(extra);
            assert!(paired(&datasets, &runs).is_err());
        }
    }

    /// The sign test against the exact binomial tail computed in integers, and finite far past
    /// the 1023 pairs where `2^n` leaves the range of an f64.
    #[test]
    fn ablation_sign_test_matches_the_exact_binomial_tail() {
        assert_eq!(sign_test(0, 0), 1.0);
        assert_eq!(sign_test(5, 0), 0.0625);
        assert_eq!(sign_test(0, 6), 0.031_25);
        assert_eq!(sign_test(3, 3), 1.0);
        for n in 1u32..=60 {
            // C(n, k) for every k, exactly.
            let mut binomial = vec![1u128];
            for k in 1..=u128::from(n) {
                let next = binomial[k as usize - 1] * (u128::from(n) - k + 1) / k;
                binomial.push(next);
            }
            for wins in 0..=n {
                let losses = n - wins;
                let tail = wins.min(losses) as usize;
                let lower: u128 = binomial[..=tail].iter().sum();
                let exact = (2.0 * lower as f64 / 2f64.powi(n as i32)).min(1.0);
                let p = sign_test(u64::from(wins), u64::from(losses));
                assert!(
                    (p - exact).abs() <= 1e-12 * exact,
                    "n {n} wins {wins}: {p} against {exact}"
                );
                assert_eq!(p, sign_test(u64::from(losses), u64::from(wins)));
            }
        }
        assert_eq!(sign_test(1000, 1000), 1.0);
        let p = sign_test(1100, 900);
        assert!(p > 1e-6 && p < 1e-4, "{p}");
        assert!(sign_test(2000, 0).is_finite());
    }

    /// Every text the kernel2 arms add, the option names and help, the profile ids, the new
    /// document keys and fixed values, and the refusal, passes the shipped-text lint.
    #[test]
    fn ablation_kernel2_text_passes_the_shipped_text_lint() {
        use clap::Args as _;
        let command = AblationArgs::augment_args(clap::Command::new("ablation"));
        let mut texts: Vec<String> = Vec::new();
        for id in ["refine_top_k", "refine_trials"] {
            let arg = command
                .get_arguments()
                .find(|a| a.get_id().as_str() == id)
                .unwrap();
            texts.push(arg.get_long().unwrap().to_owned());
            for help in [arg.get_help(), arg.get_long_help()].into_iter().flatten() {
                texts.push(help.to_string());
            }
        }
        texts.extend(
            [
                Profile::Kernel2Baseline.id(),
                Profile::Kernel2Experimental.id(),
                PAIRED_DEFINITIONS,
                PAIRED_RULE,
                REFINEMENT_REFUSAL,
                SCRATCH_IN_USE,
            ]
            .map(str::to_owned),
        );
        fn keys(value: &serde_json::Value, out: &mut Vec<String>) {
            match value {
                serde_json::Value::Object(map) => {
                    for (key, value) in map {
                        out.push(key.clone());
                        keys(value, out);
                    }
                }
                serde_json::Value::Array(items) => items.iter().for_each(|v| keys(v, out)),
                _ => {}
            }
        }
        let budget = Budget {
            evaluations: KERNEL2_EVALUATIONS,
            seconds: SEARCH_SECONDS,
        };
        let record = Profile::Kernel2Baseline
            .record(budget, Some(SMALL_REFINEMENT))
            .unwrap();
        // The record's `search` and `selection` values are the route's existing engine and
        // selection ids (`kernel2::receipt::{ENGINE, SELECTION}`), which this harness reuses
        // and does not lint; the legacy record's keys predate the lint.
        texts.push(record.candidate_source.to_owned());
        keys(&serde_json::to_value(&record).unwrap(), &mut texts);
        let summary = paired(
            &["d0".to_owned()],
            &[
                paired_run("A", "d0", 1, 1.0, "p"),
                paired_run("B", "d0", 1, 0.5, "q"),
            ],
        )
        .unwrap();
        keys(&serde_json::to_value(&summary).unwrap(), &mut texts);
        let cut = Kernel2Cut {
            measurements: 1,
            objective_trials: 0,
            rejected_trials: 0,
            arm_counts: BTreeMap::new(),
        };
        keys(&serde_json::to_value(&cut).unwrap(), &mut texts);
        keys(
            &serde_json::to_value(paired_timing(&[])).unwrap(),
            &mut texts,
        );
        texts.extend(["kernel2", "paired"].map(str::to_owned));
        let legacy_keys = [
            "id",
            "search",
            "candidate_source",
            "candidates_per_round",
            "carried_per_round",
            "max_transforms",
            "threads",
            "islands",
        ];
        for text in texts.iter().filter(|t| !legacy_keys.contains(&t.as_str())) {
            let hits = feat_math::transpile::lint_shipped_text(text);
            assert!(hits.is_empty(), "{text:?}: {hits:?}");
        }
    }

    /// An entry of a directory tree: its kind, its modification time, and a file's bytes or a
    /// link's target.
    type Entry = (&'static str, std::time::SystemTime, Vec<u8>);

    /// Every entry under `root`, symbolic links not followed.
    fn tree(root: &Path) -> BTreeMap<PathBuf, Entry> {
        let mut entries = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                let metadata = std::fs::symlink_metadata(&path).unwrap();
                let kind = metadata.file_type();
                let (name, content) = if kind.is_symlink() {
                    let target = std::fs::read_link(&path).unwrap();
                    ("link", target.into_os_string().into_encoded_bytes())
                } else if kind.is_dir() {
                    pending.push(path.clone());
                    ("dir", Vec::new())
                } else {
                    ("file", std::fs::read(&path).unwrap())
                };
                let relative = path.strip_prefix(root).unwrap().to_path_buf();
                entries.insert(relative, (name, metadata.modified().unwrap(), content));
            }
        }
        entries
    }

    /// The entries whose kind, modification time or content differs between two [`tree`]s, or
    /// that only one of them has.
    fn changed_entries(
        before: &BTreeMap<PathBuf, Entry>,
        after: &BTreeMap<PathBuf, Entry>,
    ) -> Vec<PathBuf> {
        before
            .keys()
            .chain(after.keys().filter(|path| !before.contains_key(*path)))
            .filter(|path| before.get(*path) != after.get(*path))
            .cloned()
            .collect()
    }

    /// A comparison into a scratch directory that already holds its kernel2 searches is refused
    /// before anything is prepared or searched (review finding PR0-R1): every entry the first
    /// run left keeps its bytes and modification time and nothing is added, also when the fold
    /// seed, the jobs or the other slot's profile differ. A planned kernel2 output that is a
    /// dangling symbolic link is refused the same way, with nothing written.
    #[test]
    fn ablation_kernel2_refuses_a_scratch_that_holds_its_searches() {
        let config = Config {
            datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
            evaluations: 1,
            refinement: Some(Refinement {
                top_k: 0,
                trials_per_program: 0,
            }),
            jobs: 1,
            ..kernel2_config()
        };
        let dir = scratch();
        let first = run(&config, dir.path()).unwrap();
        let hash = first.document.body.datasets[0].dataset_hash.clone();
        let before = tree(dir.path());
        for slot in ["A", "B"] {
            let published = search_output(Path::new(""), slot, &hash, 6).join("search/cut-0");
            for name in ["search.json", "observations.json", "refinements.json"] {
                let path = published.join(name);
                assert!(
                    before.get(&path).is_some_and(|entry| entry.0 == "file"),
                    "{}",
                    path.display()
                );
            }
        }
        assert!(before.contains_key(&Path::new("prepared").join(&hash).join("prepared.json")));

        for again in [
            config.clone(),
            Config {
                fold_seed: config.fold_seed + 1,
                ..config.clone()
            },
            Config {
                jobs: 2,
                ..config.clone()
            },
            Config {
                profile_a: Profile::Legacy,
                ..config.clone()
            },
            Config {
                profile_b: Profile::Kernel2Experimental,
                ..config.clone()
            },
        ] {
            let error = format!("{:#}", run(&again, dir.path()).unwrap_err());
            assert!(error.contains(SCRATCH_IN_USE), "{again:?}: {error}");
            let changed = changed_entries(&before, &tree(dir.path()));
            assert!(
                changed.is_empty(),
                "{again:?}: a refused rerun changed {changed:?}"
            );
        }

        #[cfg(unix)]
        {
            let dir = scratch();
            let output = search_output(dir.path(), "B", &hash, 6);
            std::fs::create_dir_all(output.parent().unwrap()).unwrap();
            std::os::unix::fs::symlink("missing", &output).unwrap();
            let before = tree(dir.path());
            let error = format!("{:#}", run(&config, dir.path()).unwrap_err());
            assert!(error.contains(SCRATCH_IN_USE), "{error}");
            let changed = changed_entries(&before, &tree(dir.path()));
            assert!(changed.is_empty(), "a refused run changed {changed:?}");
            assert!(std::fs::symlink_metadata(dir.path().join("prepared")).is_err());
        }
    }

    /// A paired refusal: its message, which must pass the shipped-text lint.
    fn paired_refusal(datasets: &[&str], runs: &[RunRecord]) -> String {
        let datasets: Vec<String> = datasets.iter().map(|d| (*d).to_owned()).collect();
        let error = match paired(&datasets, runs) {
            Ok(summary) => panic!("refusal expected, received {summary:?}"),
            Err(error) => format!("{error:#}"),
        };
        let hits = feat_math::transpile::lint_shipped_text(&error);
        assert!(hits.is_empty(), "{error:?}: {hits:?}");
        error
    }

    /// Review finding PR0-R2, `reject_duplicate_a_and_unpaired_b`: a repeated A run whose extra
    /// pair is balanced by an unpaired B run is refused, not counted as two wins; so is the
    /// mirror case on the B side.
    #[test]
    fn ablation_paired_rejects_duplicate_a_and_unpaired_b() {
        let error = paired_refusal(
            &["d0"],
            &[
                paired_run("A", "d0", 1, 1.0, "p"),
                paired_run("A", "d0", 1, 1.0, "p"),
                paired_run("B", "d0", 1, 0.5, "q"),
                paired_run("B", "d0", 2, 99.0, "r"),
            ],
        );
        assert!(error.contains("two A runs"), "{error}");
        let error = paired_refusal(
            &["d0"],
            &[
                paired_run("A", "d0", 1, 1.0, "p"),
                paired_run("A", "d0", 2, 1.0, "p"),
                paired_run("B", "d0", 1, 0.5, "q"),
                paired_run("B", "d0", 1, 0.5, "q"),
            ],
        );
        assert!(error.contains("two B runs"), "{error}");
    }

    /// Review finding PR0-R2, `reject_unlisted_unpaired_a`: an A run on a dataset the
    /// comparison does not list is refused rather than left out; so is a B run there, paired or
    /// not, and a B run on a listed dataset without its A run.
    #[test]
    fn ablation_paired_rejects_unlisted_unpaired_a() {
        let pair = [
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
        ];
        for extra in [
            vec![paired_run("A", "outside", 9, 2.0, "p")],
            vec![paired_run("B", "outside", 9, 2.0, "p")],
            vec![
                paired_run("A", "outside", 9, 2.0, "p"),
                paired_run("B", "outside", 9, 2.0, "p"),
            ],
        ] {
            let runs: Vec<RunRecord> = pair.iter().cloned().chain(extra).collect();
            let error = paired_refusal(&["d0"], &runs);
            assert!(error.contains("does not list"), "{error}");
        }
        let mut runs = pair.to_vec();
        runs.push(paired_run("B", "d0", 2, 0.5, "q"));
        let error = paired_refusal(&["d0"], &runs);
        assert!(error.contains("a B run needs exactly one A run"), "{error}");
    }

    /// The pairing refuses a run whose slot is neither A nor B, and a dataset listed twice.
    #[test]
    fn ablation_paired_rejects_a_repeated_dataset_or_an_unknown_slot() {
        let pair = [
            paired_run("A", "d0", 1, 1.0, "p"),
            paired_run("B", "d0", 1, 0.5, "q"),
        ];
        let mut runs = pair.to_vec();
        runs.push(paired_run("C", "d0", 1, 0.5, "q"));
        let error = paired_refusal(&["d0"], &runs);
        assert!(error.contains("neither A nor B"), "{error}");
        let error = paired_refusal(&["d0", "d0"], &pair);
        assert!(error.contains("listed twice"), "{error}");
    }

    /// `execute` makes every refusal of the options before it creates a scratch directory
    /// (review finding PR0-R3): for each refused combination neither the `--scratch` path, nor
    /// any of its parents under the test directory, nor `--out` exists afterwards.
    #[test]
    fn ablation_execute_refuses_before_creating_scratch() {
        let root = scratch();
        let base = AblationArgs {
            profile_a: Profile::Kernel2Baseline.id().to_owned(),
            profile_b: Profile::Kernel2Baseline.id().to_owned(),
            refine_top_k: Some(1),
            refine_trials: Some(1),
            datasets: vec![d5("pmlb-first-principles-supernovae-zg")],
            seeds: vec![6],
            evaluations: 1,
            cuts: 1,
            fold_seed: jope_pipeline::DEFAULT_SEED,
            jobs: 1,
            out: PathBuf::new(),
            scratch: None,
        };
        // The base passes every check, so each case that follows is refused for its one change.
        config(&base).unwrap().check().unwrap();
        let legacy = AblationArgs {
            profile_a: Profile::Legacy.id().to_owned(),
            profile_b: Profile::Legacy.id().to_owned(),
            refine_top_k: None,
            refine_trials: None,
            ..base.clone()
        };
        config(&legacy).unwrap().check().unwrap();
        let cases = [
            // A kernel2 profile without the allowance.
            (
                AblationArgs {
                    refine_top_k: None,
                    refine_trials: None,
                    ..base.clone()
                },
                "--refine-top-k",
            ),
            (
                AblationArgs {
                    profile_a: Profile::Legacy.id().to_owned(),
                    refine_top_k: None,
                    refine_trials: None,
                    ..base.clone()
                },
                "--refine-trials",
            ),
            // The allowance with legacy only.
            (
                AblationArgs {
                    refine_top_k: Some(0),
                    refine_trials: Some(0),
                    ..legacy.clone()
                },
                "legacy profile refuses",
            ),
            // More than one cut.
            (
                AblationArgs {
                    cuts: 2,
                    ..base.clone()
                },
                "--cuts 1",
            ),
            // More refined parents than the carried set.
            (
                AblationArgs {
                    refine_top_k: Some(search::ELITES + 1),
                    refine_trials: Some(0),
                    ..base.clone()
                },
                "refinement allowance",
            ),
            // More trials than the cut's budget.
            (
                AblationArgs {
                    refine_trials: Some(2),
                    ..base.clone()
                },
                "refinement allowance",
            ),
            // A refinement option alone, refused by `config`.
            (
                AblationArgs {
                    refine_trials: None,
                    ..base.clone()
                },
                "--refine-trials",
            ),
            // The checks every profile makes.
            (
                AblationArgs {
                    seeds: vec![6, 6],
                    ..legacy.clone()
                },
                "a seed is listed twice",
            ),
            (
                AblationArgs {
                    evaluations: search::MAX_EVALUATIONS + 1,
                    ..legacy.clone()
                },
                "budget evaluations must be",
            ),
            (
                AblationArgs {
                    cuts: prepare::FOLDS + 1,
                    ..legacy.clone()
                },
                "--cuts must be",
            ),
            (
                AblationArgs {
                    jobs: 0,
                    ..legacy.clone()
                },
                "--jobs must be at least 1",
            ),
        ];
        for (index, (args, needle)) in cases.into_iter().enumerate() {
            let case = root.path().join(format!("case-{index}"));
            let args = AblationArgs {
                scratch: Some(case.join("scratch/nested")),
                out: case.join("out/ablation.json"),
                ..args
            };
            let error = format!("{:#}", execute(&args).unwrap_err());
            assert!(
                error.contains(needle),
                "case {index}: {needle:?} not in {error:?}"
            );
            assert!(
                std::fs::symlink_metadata(&case).is_err(),
                "case {index} ({error}) created {}",
                case.display()
            );
        }
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}
