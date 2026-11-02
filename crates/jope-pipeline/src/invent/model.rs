use super::*;

#[derive(Clone, Debug)]
pub struct InventOptions {
    /// Header CSV: every column numeric and finite, one of them the target.
    pub train: PathBuf,
    pub target: String,
    /// A new or empty directory the search publishes its per-cut evidence under.
    pub work_dir: PathBuf,
    pub budget: search::Budget,
    pub seed: u64,
    pub rng_seed: u64,
    pub cuts: usize,
    pub top_k: usize,
    pub threads: usize,
    pub islands: usize,
    pub migration_interval: usize,
    pub cpu_ceiling_millicores: Option<u64>,
    pub source_commit: Option<String>,
    pub cancel: Option<Arc<AtomicBool>>,
    /// A finish request ([`crate::parallel::finish_flag`]): the search stops starting new work,
    /// completes, and the candidates measured so far are ranked and returned as usual, with
    /// [`Candidates::finished_early`] set.
    pub finish: Option<Arc<AtomicBool>>,
    pub clock: Clock,
    /// Write `jope.invent.progress.v1` records to this file (created or truncated) while the
    /// search runs ([`crate::progress`]); `None` writes none. It may not alias `train` or lie
    /// in `work_dir` (`check_events`).
    pub events: Option<PathBuf>,
    /// Rank programs by this regression objective on held-out rows instead of RMSE.
    pub objective: Option<jope_runtime::fitness::Fitness>,
}

impl InventOptions {
    pub fn new(
        train: impl Into<PathBuf>,
        target: impl Into<String>,
        work_dir: impl Into<PathBuf>,
    ) -> Self {
        InventOptions {
            train: train.into(),
            target: target.into(),
            work_dir: work_dir.into(),
            budget: search::Budget::default(),
            seed: crate::DEFAULT_SEED,
            rng_seed: crate::DEFAULT_SEED,
            cuts: 3,
            top_k: 8,
            threads: 1,
            islands: 1,
            migration_interval: search::DEFAULT_MIGRATION_INTERVAL,
            cpu_ceiling_millicores: None,
            source_commit: None,
            cancel: None,
            finish: None,
            clock: Clock::Wall,
            events: None,
            objective: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidates {
    pub schema: &'static str,
    pub ok: bool,
    pub tool_version: &'static str,
    /// `--source-commit` when given; `git` is never spawned.
    pub git_sha: Option<String>,
    pub dataset_sha256: String,
    pub rows: usize,
    pub target: String,
    pub feature_names: Vec<String>,
    pub cuts: usize,
    pub seed: u64,
    pub rng_seed: u64,
    pub budget: BudgetRecord,
    pub vocabulary: VocabularyRecord,
    pub searches: Vec<CutRecord>,
    /// Programs measured on every cut, before the jain-math restriction.
    pub ranked: usize,
    /// Of those, programs with a stage jain-math cannot execute.
    pub excluded_not_executable: usize,
    pub elapsed_seconds: f64,
    /// A finish request ended the search before its budget did (`--stop-file` / SIGUSR1). The
    /// candidates are still ranked on every cut; only fewer programs were tried. Omitted when
    /// false, so a run nobody stopped prints the same document as before.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub finished_early: bool,
    /// The regression objective candidates were ranked by; absent means held-out RMSE.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<crate::receipt::Objective>,
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BudgetRecord {
    pub evaluations: usize,
    pub seconds: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct VocabularyRecord {
    pub engine: String,
    pub jain_math_rev: String,
    pub executable_ops: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CutRecord {
    pub cut: usize,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub stopped_by: &'static str,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Candidate {
    pub rank: usize,
    /// Canonical pipeline DSL, e.g. `(pipe X (standardize) (ridge))`.
    pub genome: String,
    pub program_sha256: String,
    /// The search operator that first produced the program (`bootstrap`, `mutation`, ...).
    pub origin: String,
    pub mean_hold_rmse: f64,
    /// With an objective: its mean held-out minimization loss, the value candidates are
    /// ranked by.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_hold_objective: Option<f64>,
    pub mean_hold_r2: f64,
    pub train_r2: f64,
    pub cuts_measured: usize,
    pub seconds: f64,
}

/// The op names jain-math executes, and the revision they were read from.
pub struct Executable {
    pub names: BTreeSet<String>,
    pub engine: String,
    pub rev: String,
}

impl Executable {
    pub fn load() -> Result<Self> {
        let doc: serde_json::Value =
            serde_json::from_str(JAIN_EXECUTABLE_OPS).context("parse the jain-math op snapshot")?;
        let names: BTreeSet<String> = doc["executable"]
            .as_array()
            .context("op snapshot: executable is not a list")?
            .iter()
            .filter_map(|op| op["name"].as_str().map(str::to_owned))
            .collect();
        ensure!(!names.is_empty(), "op snapshot lists no executable op");
        Ok(Executable {
            names,
            engine: doc["engine"].as_str().unwrap_or_default().to_owned(),
            rev: doc["provenance"]["rev"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        })
    }

    /// A pipeline genome whose every stage op jain-math executes.
    pub fn admits(&self, genome: &str) -> bool {
        feat_math::dsl::parse_dsl(genome).is_ok_and(|expr| {
            expr.stages
                .iter()
                .all(|stage| self.names.contains(&stage.op))
        })
    }
}
