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
//! Today the only profile is `legacy`, the shipping search unchanged. Later profiles are added
//! as new [`Profile`] variants. `docs/internal/kernel2/ablation-protocol.md` states the metric
//! and gate definitions.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::ParamValue;
use feat_math::math_expr::{Expr, MathExpr};
use feat_math::transpile::EmitOptions;
use feat_math::GenomeIr;
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
}

impl Profile {
    /// Every profile, in declaration order.
    pub const ALL: [Profile; 1] = [Profile::Legacy];

    /// The profile's id.
    pub fn id(self) -> &'static str {
        match self {
            Profile::Legacy => "legacy",
        }
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

    fn record(self) -> ProfileRecord {
        match self {
            Profile::Legacy => ProfileRecord {
                id: self.id(),
                search: search::SEARCH_ENGINE,
                candidate_source: "feat_math::evolve canonical-pipeline wrappers, v1",
                candidates_per_round: search::POPULATION,
                carried_per_round: search::ELITES,
                max_transforms: search::MAX_TRANSFORMS,
                threads: 1,
                islands: 1,
            },
        }
    }

    fn search(
        self,
        cuts: &[Cut],
        budget: Budget,
        rng_seed: u64,
        output: &Path,
    ) -> Result<Vec<CutSearch>> {
        match self {
            Profile::Legacy => {
                search::run_cuts_with(cuts, &SearchOptions::new(budget, rng_seed), output)
            }
        }
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
        Ok(())
    }
}

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

fn load_datasets(config: &Config, scratch: &Path) -> Result<Vec<Dataset>> {
    let mut dirs = Vec::new();
    for path in &config.datasets {
        dirs.extend(discover(path)?);
    }
    let mut seen = BTreeSet::new();
    let mut datasets = Vec::new();
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
        let prepared = prepare::build(
            &rows,
            &hash,
            config.fold_seed,
            config.cuts,
            &scratch.join("prepared").join(&hash),
        )
        .with_context(|| format!("{}: prepare cuts", dataset.row.id))?;
        datasets.push(Dataset {
            record: DatasetRecord {
                id: dataset.row.id.clone(),
                dataset_hash: hash,
                train_sha256: dataset.train_sha256.clone(),
                train_rows: dataset.x.len(),
                features: dataset.x.first().map_or(0, Vec::len),
                prepared_sha256: prepared.sha256,
            },
            path: dir,
            cuts: prepared.cuts,
        });
    }
    Ok(datasets)
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
        run_one(spec, &datasets[spec.dataset], budget, scratch)
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
    let body = Body {
        protocol: ProtocolRecord {
            definitions: DEFINITIONS,
            profile_a: config.profile_a.record(),
            profile_b: config.profile_b.record(),
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
    let document = Document {
        schema: SCHEMA,
        body_sha256: sha256_hex(&body_bytes),
        body,
        timing: Timing {
            wall_seconds: started.elapsed().as_secs_f64(),
            runs: run_timing,
            m10,
            gate_d_m10,
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

fn run_one(
    spec: &Spec,
    dataset: &Dataset,
    budget: Budget,
    scratch: &Path,
) -> Result<(RunRecord, RunTiming)> {
    let hash = &dataset.record.dataset_hash;
    let output = scratch
        .join("search")
        .join(spec.slot)
        .join(hash)
        .join(format!("seed-{}", spec.seed));
    let searches = spec
        .profile
        .search(&dataset.cuts, budget, spec.seed, &output)
        .with_context(|| {
            format!(
                "profile {} ({}) on {} seed {}",
                spec.profile.id(),
                spec.slot,
                dataset.record.id,
                spec.seed
            )
        })?;
    check_replayable(&searches).with_context(|| {
        format!(
            "profile {} ({}) on {} seed {}",
            spec.profile.id(),
            spec.slot,
            dataset.record.id,
            spec.seed
        )
    })?;
    let metrics = metrics(&searches)?;
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
        })
        .collect();
    let search_seconds: f64 = searches.iter().map(|s| s.seconds).sum();
    let evaluations: u64 = searches.iter().map(|s| s.completed_evaluations).sum();
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

/// Refuse a search whose decisions depended on the clock: a phase the seconds bound ended, or a
/// cut whose active time overran its bound.
pub fn check_replayable(searches: &[CutSearch]) -> Result<()> {
    for s in searches {
        for (phase, stop) in [
            ("exploration", s.exploration.stopped_by.as_str()),
            ("comparison", s.comparison.stopped_by.as_str()),
        ] {
            ensure!(
                REPLAYABLE_STOPS.contains(&stop),
                "cut {}: the {phase} phase stopped by {stop:?}; only the evaluation budget or an \
                 exhausted pool keeps a search replayable, so lower the evaluation budget",
                s.cut
            );
        }
        ensure!(
            s.seconds <= s.budget.seconds as f64,
            "cut {}: {:.1} active seconds overran the {} second bound",
            s.cut,
            s.seconds,
            s.budget.seconds
        );
    }
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

/// Every metric of one run from its cuts' searches.
pub fn metrics(searches: &[CutSearch]) -> Result<Metrics> {
    let per_cut: Vec<(usize, &[Observation])> = searches
        .iter()
        .map(|s| (s.cut, s.observations.as_slice()))
        .collect();
    let ranking = select::ranking(&per_cut)?;
    let best = ranking
        .first()
        .context("no program was measured on every cut")?;

    let top: Vec<&select::Best> = ranking.iter().take(TOP_MEDIAN).collect();
    let top_rmse: Vec<f64> = top.iter().map(|b| b.validation_rmse).collect();
    let top_r2: Vec<f64> = top.iter().map(|b| b.validation_r2).collect();

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
    let mut per_cut_speed = Vec::with_capacity(exploration.len());
    for observations in &exploration {
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
        let (count, sizes) = program_shape(&program.genome)?;
        stages.push(count as f64);
        expressions.extend(sizes);
    }
    let nodes: Vec<f64> = expressions.iter().map(|(n, _)| *n as f64).collect();
    let depths: Vec<f64> = expressions.iter().map(|(_, d)| *d as f64).collect();

    Ok(Metrics {
        m1: Best {
            rmse: finite("M1 rmse", best.validation_rmse)?,
            r2: finite("M1 r2", best.validation_r2)?,
            cuts_measured: best.cuts_measured,
            program_sha256: best.program_sha256.clone(),
            program: best.genome.clone(),
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
        m8: cost(&best.genome),
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
                optional_ratio_clause(
                    "M8",
                    "token cost increase at most 5%",
                    oa.m8_tokens,
                    ob.m8_tokens,
                    1.05,
                ),
                optional_ratio_clause(
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
    /// TMPDIR, removed afterwards). It must not already hold a search of the same run.
    #[arg(long)]
    pub scratch: Option<PathBuf>,
}

/// Run `jope-bench ablation`.
pub fn execute(args: &AblationArgs) -> Result<()> {
    let config = Config {
        profile_a: Profile::parse(&args.profile_a)?,
        profile_b: Profile::parse(&args.profile_b)?,
        datasets: args.datasets.clone(),
        seeds: args.seeds.clone(),
        evaluations: args.evaluations,
        cuts: args.cuts,
        fold_seed: args.fold_seed,
        jobs: args.jobs,
    };
    let temporary;
    let scratch = match &args.scratch {
        Some(dir) => {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
            dir.clone()
        }
        None => {
            temporary = tempfile::Builder::new()
                .prefix("jope-bench-ablation-")
                .tempdir()
                .context("create a scratch directory")?;
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

    fn scratch() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("jope-bench-ablation-test-")
            .tempdir()
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
}
