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
//! ([`crate::ablation::SEARCH_SECONDS`]), one worker and one island, so the evaluation budget ends
//! every phase and the decisions replay from the seed alone. A run in which the seconds bound ended
//! any phase is refused ([`crate::ablation::check_replayable`]) rather than reported. Every (profile,
//! dataset, seed) search
//! writes into its own directory, and profile B is executed, never copied from profile A, so an
//! A=A comparison runs the search twice.
//!
//! Data rules. Rows come from each dataset's `train.csv` only, through [`crate::load`]; the
//! sealed `test.csv` is never opened. Cuts are the pipeline's own
//! (`jope_pipeline::prepare::build`, keyed on `<dataset_hash>:<fold seed>`), so M1 is a held-out
//! score on a fold of the training rows.
//!
//! Profiles: `legacy`, the shipping search unchanged, and the kernel2 route
//! (`jope_pipeline::kernel2::search::run_cut`) with the shipped `kernel2-baseline` and
//! `kernel2-experimental` profiles, an explicit refinement allowance shared by every kernel2
//! search, one thread and one island. A comparison that involves a kernel2 profile also carries
//! the paired comparison of M1 ([`crate::ablation::Paired`],
//! [`crate::ablation::PAIRED_DEFINITIONS`]) in the body and each slot's wall seconds beside it in
//! `timing`; a legacy-only document is unchanged. Later profiles are added as new
//! [`crate::ablation::Profile`] variants. `docs/internal/kernel2/ablation-protocol.md` states the
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

/// Load and partition every dataset, then write each partition under `scratch`. Called while the
/// comparison holds its [`Reservation`] of `scratch`. Every refusal is made before anything is
/// written: a kernel2 search this comparison would repeat ([`check_unused`]), a scratch directory
/// a comparison with a kernel2 profile used ([`check_not_kernel2`], for every comparison) and, for
/// a comparison with a kernel2 profile, a scratch directory that holds anything but its own lock
/// file ([`check_scratch_unused`]). Such a comparison then marks the scratch ([`mark_kernel2`])
/// before its first other write. `prepare::build` is [`prepare::partition`] followed by
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
    check_not_kernel2(scratch)?;
    if config.has_kernel2() {
        check_scratch_unused(scratch)?;
        mark_kernel2(scratch)?;
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

/// Run the comparison. Every search writes its evidence under `scratch`, which is created when
/// absent. The comparison holds an exclusive `Reservation` of `scratch` from before its first
/// check of the scratch to its end, so a second comparison on the same scratch is refused while
/// it runs. `scratch` must not already hold a search for the same (slot, dataset, seed), nor a
/// kernel2 comparison's marker (`check_not_kernel2`); with a kernel2 profile it must be absent
/// or an empty directory that is not a symbolic link (`check_scratch_kind`,
/// `check_scratch_unused`), which the comparison then marks for good (`mark_kernel2`).
mod execution;
pub(super) use execution::*;
pub use execution::{check_kernel2_replayable, check_replayable, run};
mod metrics;
pub(super) use metrics::*;
pub use metrics::{
    evaluations_to_fraction, expression_size, kernel2_metrics, median, metrics, sample_sd,
};
mod gates;
pub(super) use gates::*;
mod paired;
pub(super) use paired::*;
pub use paired::{paired, sign_test, MAX_SIGN_TEST_PAIRS};
mod document;
pub use document::write_document;
pub(super) use document::*;
mod cli;
pub use cli::execute;
pub use cli::AblationArgs;
pub(super) use cli::*;

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

    #[path = "tests_1.rs"]
    mod tests_1;
    pub(super) use tests_1::*;
    #[path = "tests_2.rs"]
    mod tests_2;
    pub(super) use tests_2::*;
    #[path = "tests_3.rs"]
    mod tests_3;
    pub(super) use tests_3::*;
    #[path = "tests_4.rs"]
    mod tests_4;
    pub(super) use tests_4::*;
}
