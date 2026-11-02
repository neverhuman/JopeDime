use super::*;

/// The regression objective a search ranked programs by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Objective {
    /// The objective's name (`pinball`, a custom name, `weighted_…`, `composite`).
    pub name: String,
    /// `minimize` or `maximize`; selection always minimizes the loss (maximized values negated).
    pub direction: String,
    /// sha256 of `["jope.regression-fitness/v1", specification]`.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub fitness_sha256: String,
    /// The complete specification (`schemas/jope.regression-fitness.v1.schema.json`).
    pub specification: serde_json::Value,
}

impl Objective {
    pub fn of(fitness: &jope_runtime::fitness::Fitness) -> anyhow::Result<Self> {
        Ok(Objective {
            name: fitness.name(),
            direction: match fitness.direction() {
                jope_runtime::fitness::Direction::Minimize => "minimize",
                jope_runtime::fitness::Direction::Maximize => "maximize",
            }
            .to_owned(),
            fitness_sha256: fitness.identity()?,
            specification: serde_json::to_value(fitness)?,
        })
    }
    /// The specification, re-validated against its recorded identity.
    pub fn fitness(&self) -> anyhow::Result<jope_runtime::fitness::Fitness> {
        let fitness: jope_runtime::fitness::Fitness =
            serde_json::from_value(self.specification.clone())?;
        anyhow::ensure!(
            fitness.identity()? == self.fitness_sha256
                && fitness.name() == self.name
                && Self::of(&fitness)?.direction == self.direction,
            "the recorded objective does not match its identity, name or direction"
        );
        Ok(fitness)
    }
}

/// One genome generation 0 was seeded with.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Seed {
    /// Canonical pipeline DSL, exactly as it entered generation 0 with origin `incumbent`.
    pub genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// `incumbent` (the parent receipt's selected genome) or `seeds_file`.
    pub source: String,
}

/// One champion carried from one island into another on one cut.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Migration {
    pub cut: u64,
    /// The generation after which the migration happened.
    pub after_generation: u64,
    pub from_island: u64,
    pub to_island: u64,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// The migrant's held-out RMSE on this cut, as its home island measured it: a migrant is
    /// never re-fitted.
    pub rmse: f64,
    /// Whether it entered the receiving island's elites (only a champion better than one of them
    /// does).
    pub admitted: bool,
}

/// v3: the island model the search ran.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Islands {
    /// Independent random streams over the same cuts; 1 is the plain search.
    pub count: u64,
    /// Generations between migrations of each island's top-1 into the others.
    pub migration_interval: u64,
    /// Every migration on every cut, in cut order and then the order it happened.
    pub migrations: Vec<Migration>,
}

/// v3: how generation 0 was seeded. Present, with an empty `seeds`, on an unseeded run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Seeding {
    /// Genomes offered by `--incumbent` and `--seed-genomes` before canonicalisation,
    /// deduplication (including against the bootstrap) and the cap at the elite count.
    pub offered: u64,
    /// Admitted seeds, in the order they entered generation 0 after the bootstrap. Every one is
    /// re-measured on this run's cuts; no score is carried over from where it came from.
    pub seeds: Vec<Seed>,
    /// sha256 of the `--incumbent` receipt's bytes, which verified with its on-disk evidence.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub parent_receipt_sha256: Option<String>,
    /// sha256 of the `--seed-genomes` file's bytes.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub seeds_file_sha256: Option<String>,
}

/// Per-phase admission limits and actual work. One non-preemptible fit can
/// exceed seconds_limit; no later fit is admitted in that phase.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Phase {
    pub evaluation_limit: u64,
    pub seconds_limit: f64,
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_measurements: u64,
    pub seconds: f64,
    pub stopped_by: String,
}

impl Phase {
    pub fn not_requested() -> Self {
        Self {
            evaluation_limit: 0,
            seconds_limit: 0.0,
            completed_evaluations: 0,
            successful_measurements: 0,
            failed_measurements: 0,
            seconds: 0.0,
            stopped_by: crate::search::NOT_REQUESTED.to_owned(),
        }
    }
}

/// Required for v2; absent for historical v1 receipts, which made no total-budget guarantee.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Accounting {
    pub exploration: Phase,
    pub comparison: Phase,
    pub active_seconds: f64,
    pub deadline_overrun: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Coverage {
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    pub measured_cuts: Vec<u64>,
    pub failed_cuts: Vec<u64>,
    pub missing_cuts: Vec<u64>,
}

/// A stage either ran, was deliberately skipped with a reason, or failed.
pub const STATUSES: [&str; 3] = ["passed", "skipped", "failed"];

/// One pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Stage {
    pub name: String,
    /// `passed`, `skipped` or `failed`.
    pub status: String,
    /// Why, for anything that is not a plain pass. A skip always carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub seconds: f64,
}

/// One file of the dataset, as hashed at resolution time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetFile {
    pub name: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Dataset {
    pub id: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{16}$"))]
    pub dataset_hash: String,
    /// Where the dataset was read from, as given on the command line.
    pub path: String,
    pub task: String,
    pub rows: u64,
    pub columns: u64,
    pub training_rows: u64,
    pub files: Vec<DatasetFile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Prepared {
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub seed: u64,
    pub rows: u64,
    /// The partition each cut holds one fold out of.
    pub folds: u64,
    /// The cut indices that were built, in build order.
    pub cuts: Vec<u64>,
}

/// One cut's search, as its `search.json` published it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CutSearch {
    pub cut: u64,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub search_sha256: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
    /// Rows the fits on this cut saw, and rows they were scored on. Disjoint.
    pub train_rows: u64,
    pub hold_rows: u64,
    pub generations: u64,
    /// Fits attempted. One fit is one evaluation, whatever its outcome, so this
    /// is exactly `successful + failed`.
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_or_unavailable_measurements: u64,
    /// Single cut: exploration's `evaluations`, `seconds`, or `search_space_exhausted`.
    /// Multiple cuts: comparison's outcome, with `seconds` before `evaluations`
    /// when both are reached, else `comparison_pool_exhausted`. Exploration's
    /// original reason remains in accounting.exploration.stopped_by.
    pub stopped_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub accounting: Option<Accounting>,
    /// The best held-out RMSE measured on this cut, in the dataset's target units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_rmse: Option<f64>,
    /// That same candidate's held-out R^2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_r2: Option<f64>,
    /// With `search.objective`: that same candidate's held-out objective loss, by which it was
    /// the best; `best_validation_rmse` is then its RMSE, not necessarily the lowest RMSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_objective: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_program: Option<String>,
}

/// The selected program and its measured evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Best {
    pub genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// Mean held-out RMSE over the cuts this program was measured on, in the
    /// dataset's own target units. There is no normalisation to undo: the search
    /// fits the committed rows as they are.
    pub validation_rmse: f64,
    /// With `search.objective`: the mean held-out objective loss (minimized) the program was
    /// selected by.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_objective: Option<f64>,
    /// Mean held-out R^2 over the same cuts — the fits never saw these rows.
    pub validation_r2: f64,
    /// Mean in-sample R^2, for the gap against `validation_r2`.
    pub train_r2: f64,
    /// Fit-and-score seconds this program cost across those cuts.
    pub seconds: f64,
    pub cuts_measured: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Search {
    /// The search implementation that produced these measurements.
    pub engine: String,
    pub rng_seed: u64,
    /// The `--budget` bounds, per cut.
    pub evaluations: u64,
    pub seconds: u64,
    /// Candidates alive per generation, and the transform depth they may reach.
    pub population: u64,
    pub max_transforms: u64,
    /// The genome generation 0 always contains, so a measured champion is never
    /// worse than it.
    pub bootstrap_genome: String,
    pub cuts: Vec<CutSearch>,
    /// v2: require every requested cut, then minimum mean RMSE, then canonical
    /// genome. v1 used coverage count before RMSE, including different partial sets.
    pub selection: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub budget_policy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub compared_cuts: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub coverage: Option<Vec<Coverage>>,
    /// The genome handed to the emitters, always present.
    pub selected_genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub selected_program_sha256: String,
    /// Absent when no cut produced a successful measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<Best>,
    /// v3: the seeds generation 0 started from and where they came from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub seeding: Option<Seeding>,
    /// v3: island count, migration interval and every migration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(required)]
    pub islands: Option<Islands>,
    /// The regression objective (`--fitness`) every candidate was ranked by on its held-out
    /// rows. Absent: ranked by held-out RMSE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<Objective>,
}

/// One emitted source file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Artifact {
    pub language: String,
    /// Relative to the run's `emit/` directory. `cpp17` and `cpp20` share one
    /// path: there is a single C++ emitter and the standard is a compile flag.
    pub path: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
}

/// A requested language the selected genome is not covered for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Exclusion {
    pub language: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Receipt {
    pub schema: String,
    pub ok: bool,
    pub tool_version: String,
    /// The checkout this ran from, when it could be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    pub created_at: String,
    /// The `--policy` the run was given: `auto`, `skip` or `require`.
    pub policy: String,
    pub dataset: Dataset,
    pub prepared: Prepared,
    pub stages: Vec<Stage>,
    pub search: Search,
    pub artifacts: Vec<Artifact>,
    pub emit_exclusions: Vec<Exclusion>,
    /// sha256 over every cut's `observations.json` digest, concatenated as hex
    /// in cut order: one value binding the run to all of its evidence.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
}
