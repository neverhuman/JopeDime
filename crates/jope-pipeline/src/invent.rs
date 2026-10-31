//! `jope pipeline invent`: the pipeline's GP search as a candidate generator for another engine.
//!
//! jain-core's Phase-6 invention (the `Jope` engine) hands this command a header CSV of the
//! champion feature frame and reads back ranked genomes. It re-fits, cross-validates, reports and
//! emits every genome itself, so what crosses the boundary is the canonical DSL string and the
//! measurements that ranked it here, not predictions.
//!
//! The search is [`crate::run()`]'s, minus everything a caller with its own data does not need: no
//! catalog row, no receipt, no emitted sources. Rows are partitioned by [`prepare::partition`]
//! (keyed on the CSV's sha256), searched by [`search::run_cuts_with`] and ranked by
//! [`select::ranking`], unchanged.
//!
//! A ranked genome is offered only when every stage op is executable in the jain-math revision
//! the caller links ([`JAIN_EXECUTABLE_OPS`], the same snapshot `jope-engine` restricts its
//! vocabulary with). The search grammar is wider (its heads include `ridge_gcv`), so programs
//! outside that set are counted and left out rather than offered for the caller to reject.

use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{clock::Clock, prepare, search, select, sha256_hex};

/// The stdout document's schema.
pub const SCHEMA: &str = "jope.candidates.v1";

/// The jain-math registry snapshot that decides which genomes are offered.
pub const JAIN_EXECUTABLE_OPS: &str =
    include_str!("../../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json");

/// Exit status when the search measured nothing the caller can execute.
pub const NO_CANDIDATES_EXIT: u8 = 3;

/// Most genomes one call may return.
pub const MAX_TOP_K: usize = 64;

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

/// The header CSV as `features..., target` rows, plus the feature names in column order.
pub fn load_csv(path: &Path, target: &str) -> Result<(Vec<String>, Vec<Vec<f64>>, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = sha256_hex(&bytes);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(bytes.as_slice());
    let header: Vec<String> = reader
        .headers()
        .context("read the CSV header")?
        .iter()
        .map(str::to_owned)
        .collect();
    let target_index = header
        .iter()
        .position(|name| name == target)
        .with_context(|| format!("--target {target:?} is not a column of {}", path.display()))?;
    ensure!(
        header.len() >= 2,
        "{}: {} column(s); features and a target are required",
        path.display(),
        header.len()
    );
    let mut rows = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("CSV data row {}", line + 1))?;
        ensure!(
            record.len() == header.len(),
            "CSV data row {}: {} field(s), the header has {}",
            line + 1,
            record.len(),
            header.len()
        );
        let mut features = Vec::with_capacity(header.len());
        let mut value_of_target = 0.0;
        for (column, field) in record.iter().enumerate() {
            let value: f64 = field.trim().parse().with_context(|| {
                format!(
                    "CSV data row {}, column {:?}: {field:?} is not a number",
                    line + 1,
                    header[column]
                )
            })?;
            ensure!(
                value.is_finite(),
                "CSV data row {}, column {:?}: {value} is not finite",
                line + 1,
                header[column]
            );
            if column == target_index {
                value_of_target = value;
            } else {
                features.push(value);
            }
        }
        features.push(value_of_target);
        rows.push(features);
    }
    let features = header
        .into_iter()
        .enumerate()
        .filter(|&(column, _)| column != target_index)
        .map(|(_, name)| name)
        .collect();
    Ok((features, rows, digest))
}

/// Every refusal that needs no data and writes nothing; returns the worker count.
pub fn preflight(options: &InventOptions) -> Result<usize> {
    prepare::check_cuts(options.cuts)?;
    options.budget.check()?;
    ensure!(
        (1..=MAX_TOP_K).contains(&options.top_k),
        "--top-k must be 1..={MAX_TOP_K}, got {}",
        options.top_k
    );
    ensure!(
        options.islands <= 1 || options.migration_interval >= 1,
        "--migration-interval must be at least 1 with more than one island"
    );
    if let Some(commit) = &options.source_commit {
        ensure!(
            crate::run::is_commit(commit),
            "--source-commit {commit:?} is not a full commit id (40 or 64 lowercase hex digits)"
        );
    }
    crate::parallel::admit(
        options.threads,
        options.islands,
        options.cpu_ceiling_millicores,
    )
}

/// `--work-dir` must be new or an empty directory: nothing under it is overwritten, so a second
/// run cannot mix its evidence with an earlier one's (`prepared/` would otherwise be rewritten in
/// place). Checked before anything is read or written.
fn ensure_fresh_work_dir(work_dir: &Path) -> Result<()> {
    match std::fs::symlink_metadata(work_dir) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", work_dir.display())),
        Ok(meta) => {
            ensure!(
                meta.is_dir(),
                "{}: --work-dir exists and is not a directory; choose a new or empty directory",
                work_dir.display()
            );
            let mut entries = std::fs::read_dir(work_dir)
                .with_context(|| format!("read {}", work_dir.display()))?;
            ensure!(
                entries.next().is_none(),
                "{}: --work-dir is not empty; choose a new or empty directory",
                work_dir.display()
            );
            Ok(())
        }
    }
}

/// `path` with its existing ancestors resolved, so that two spellings of one file (`a/../b`,
/// a symbolic link) compare equal whether or not the file exists yet.
fn resolve_lexically(path: &Path) -> Result<PathBuf> {
    use std::path::Component;
    let absolute = std::path::absolute(path)?;
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                match std::fs::symlink_metadata(&resolved) {
                    Ok(_) => {
                        resolved = resolved
                            .canonicalize()
                            .with_context(|| format!("resolve {}", resolved.display()))?
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(e).with_context(|| format!("inspect {}", resolved.display()))
                    }
                }
            }
        }
    }
    Ok(resolved)
}

#[cfg(unix)]
fn same_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

/// `--events` must name a file that is neither the input nor inside the work directory, and not
/// an existing directory ("reject input/output aliases"). Checked before anything is written.
fn check_events(events: &Path, train: &Path, work_dir: &Path) -> Result<()> {
    ensure!(
        !events.is_dir(),
        "{}: --events is a directory; name a file",
        events.display()
    );
    match std::fs::symlink_metadata(events) {
        Ok(meta) => {
            ensure!(
                meta.is_file(),
                "{}: --events must be a regular file, not a symlink or special target",
                events.display()
            );
            #[cfg(unix)]
            ensure!(
                !same_file(&meta, &std::fs::metadata(train)?),
                "{}: --events aliases the --train input",
                events.display()
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e).with_context(|| format!("inspect --events {}", events.display())),
    }
    let target = resolve_lexically(events)?;
    ensure!(
        target != resolve_lexically(train)?,
        "{}: --events names the --train input",
        events.display()
    );
    ensure!(
        !target.starts_with(resolve_lexically(work_dir)?),
        "{}: --events is the --work-dir or inside it; write it elsewhere",
        events.display()
    );
    Ok(())
}

/// Open without truncation, then validate the actual descriptor before changing any bytes.
#[cfg(target_os = "linux")]
fn open_events(events: &Path, train: &Path, work_dir: &Path) -> Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    check_events(events, train, work_dir)?;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(nix::libc::O_NONBLOCK | nix::libc::O_NOFOLLOW)
        .open(events)
        .with_context(|| format!("open --events {}", events.display()))?;
    let actual = file.metadata()?;
    ensure!(actual.is_file(), "--events opened a nonregular file");
    ensure!(
        !same_file(&actual, &std::fs::metadata(train)?),
        "--events aliases the --train input"
    );
    let named = std::fs::symlink_metadata(events)?;
    ensure!(
        named.is_file() && same_file(&actual, &named),
        "--events target changed while opening"
    );
    check_events(events, train, work_dir)?;
    file.set_len(0)
        .context("truncate validated --events file")?;
    Ok(file)
}

#[cfg(not(target_os = "linux"))]
fn open_events(_events: &Path, _train: &Path, _work_dir: &Path) -> Result<std::fs::File> {
    anyhow::bail!("safe --events file opening is currently supported on Linux only")
}

/// Search the CSV and return the ranked genomes jain-math can execute.
pub fn invent(options: &InventOptions) -> Result<Candidates> {
    crate::narration::from_env(
        |tap| invent_observed(options, tap),
        || {
            options
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::SeqCst))
        },
    )
}

/// Explicit observer entrypoint. Offers are distinct from a consuming engine's final selection.
pub fn invent_with_narration(
    options: &InventOptions,
    narration: Option<jope_narrator::Tap>,
) -> Result<Candidates> {
    crate::narration::with_tap(
        narration,
        |tap| invent_observed(options, tap),
        || {
            options
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::SeqCst))
        },
    )
}

fn invent_observed(
    options: &InventOptions,
    narration: Option<&jope_narrator::Tap>,
) -> Result<Candidates> {
    preflight(options)?;
    if let Some(objective) = &options.objective {
        crate::run::check_objective(objective)?;
    }
    ensure_fresh_work_dir(&options.work_dir)?;
    if let Some(events) = &options.events {
        check_events(events, &options.train, &options.work_dir)?;
    }
    let executable = Executable::load()?;
    let started = options.clock.start();

    let (feature_names, rows, dataset_sha256) = load_csv(&options.train, &options.target)?;
    // Refusals (too few rows in a cut, a constant held-out target) come before anything is written.
    let partition = prepare::partition(&rows, &dataset_sha256, options.seed, options.cuts)?;

    let event_file = options
        .events
        .as_ref()
        .map(|path| open_events(path, &options.train, &options.work_dir))
        .transpose()?;

    std::fs::create_dir_all(&options.work_dir)
        .with_context(|| format!("create {}", options.work_dir.display()))?;
    // The search rejects traversal and symlink components beneath this root.
    let root = options
        .work_dir
        .canonicalize()
        .with_context(|| format!("resolve {}", options.work_dir.display()))?;
    let preparation = partition.write(&root.join("prepared"))?;

    let search_options = search::SearchOptions {
        threads: options.threads,
        islands: options.islands,
        migration_interval: options.migration_interval,
        clock: options.clock,
        cancel: options.cancel.clone(),
        finish: options.finish.clone(),
        progress: event_file.map(|file| {
            Arc::new(crate::progress::Progress::file(
                file,
                preparation.cuts.len(),
                options.budget,
            ))
        }),
        objective: options.objective.clone(),
        narration: narration.cloned(),
        ..search::SearchOptions::new(options.budget, options.rng_seed)
    };
    let searched =
        search::run_cuts_observed(&preparation.cuts, &search_options, &root.join("search"))?;
    if options
        .cancel
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::SeqCst))
    {
        return Err(crate::parallel::Cancelled.into());
    }

    let per_cut: Vec<(usize, &[search::Observation])> = searched
        .iter()
        .map(|c| (c.cut, c.observations.as_slice()))
        .collect();
    let ranking = select::ranking(&per_cut)?;
    // The operator that produced each program, in search order. The comparison phase re-measures
    // programs other cuts produced, so its label stands only when nothing else measured them.
    let mut origin: BTreeMap<&str, &'static str> = BTreeMap::new();
    for search in &searched {
        for observation in &search.observations {
            let entry = origin
                .entry(observation.genome.as_str())
                .or_insert(observation.origin);
            if *entry == search::REMEASURE {
                *entry = observation.origin;
            }
        }
    }
    let ranked = ranking.len();
    let admitted: Vec<&select::Best> = ranking
        .iter()
        .filter(|best| executable.admits(&best.genome))
        .collect();
    let excluded_not_executable = ranked - admitted.len();
    let candidates = admitted
        .into_iter()
        .take(options.top_k)
        .enumerate()
        .map(|(index, best)| Candidate {
            rank: index + 1,
            genome: best.genome.clone(),
            program_sha256: best.program_sha256.clone(),
            origin: origin
                .get(best.genome.as_str())
                .copied()
                .unwrap_or("unknown")
                .to_owned(),
            mean_hold_rmse: best.validation_rmse,
            mean_hold_objective: best.validation_objective,
            mean_hold_r2: best.validation_r2,
            train_r2: best.train_r2,
            cuts_measured: best.cuts_measured,
            seconds: best.seconds,
        })
        .collect::<Vec<_>>();

    if let Some(tap) = narration {
        tap.emit(jope_narrator::LineageFact::CandidatesOffered {
            program_sha256s: candidates
                .iter()
                .map(|candidate| candidate.program_sha256.clone())
                .collect(),
            rule: format!(
                "rank_then_executable_filter_then_top_k:{}",
                if options.objective.is_some() {
                    select::MIN_MEAN_VALIDATION_OBJECTIVE
                } else {
                    select::MIN_MEAN_VALIDATION_RMSE
                }
            ),
        });
    }
    Ok(Candidates {
        schema: SCHEMA,
        ok: !candidates.is_empty(),
        tool_version: env!("CARGO_PKG_VERSION"),
        git_sha: options.source_commit.clone(),
        dataset_sha256,
        rows: rows.len(),
        target: options.target.clone(),
        feature_names,
        cuts: options.cuts,
        seed: options.seed,
        rng_seed: options.rng_seed,
        budget: BudgetRecord {
            evaluations: options.budget.evaluations,
            seconds: options.budget.seconds,
        },
        vocabulary: VocabularyRecord {
            engine: executable.engine,
            jain_math_rev: executable.rev,
            executable_ops: executable.names.len(),
        },
        searches: searched
            .iter()
            .map(|c| CutRecord {
                cut: c.cut,
                train_rows: c.train_rows,
                hold_rows: c.hold_rows,
                completed_evaluations: c.completed_evaluations,
                successful_measurements: c.successful_measurements,
                stopped_by: c.stopped_by,
                seconds: c.seconds,
            })
            .collect(),
        ranked,
        excluded_not_executable,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        finished_early: options
            .finish
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst)),
        objective: options
            .objective
            .as_ref()
            .map(crate::receipt::Objective::of)
            .transpose()?,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_csv(dir: &Path, rows: usize) -> PathBuf {
        let path = dir.join("train.csv");
        let mut text = String::from("a,y,b\n");
        for i in 0..rows {
            let a = i as f64 * 0.37 % 5.0;
            let b = (i as f64 * 1.3).sin();
            let y = 2.0 * a - 3.0 * b + 0.01 * (i % 7) as f64;
            text.push_str(&format!("{a},{y},{b}\n"));
        }
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn csv_moves_the_named_target_last_and_keeps_feature_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(dir.path(), 4);
        let (features, rows, digest) = load_csv(&path, "y").unwrap();
        assert_eq!(features, ["a", "b"]);
        assert_eq!(rows.len(), 4);
        let (a, b) = (rows[1][0], rows[1][1]);
        assert!((rows[1][2] - (2.0 * a - 3.0 * b + 0.01)).abs() < 1e-12);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn csv_refuses_missing_target_text_and_non_finite_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_csv(dir.path(), 4);
        let error = load_csv(&path, "z").unwrap_err().to_string();
        assert!(error.contains("--target \"z\""), "{error}");
        for (body, needle) in [
            ("a,y\n1,NA\n", "not a number"),
            ("a,y\n1,inf\n", "not finite"),
            ("a,y\n1\n", "field"),
        ] {
            let bad = dir.path().join("bad.csv");
            std::fs::write(&bad, body).unwrap();
            let error = format!("{:#}", load_csv(&bad, "y").unwrap_err());
            assert!(error.contains(needle), "{body:?}: {error}");
        }
    }

    #[test]
    fn work_dir_must_be_new_or_empty_and_is_checked_before_anything_is_written() {
        let dir = tempfile::tempdir().unwrap();
        ensure_fresh_work_dir(&dir.path().join("new")).unwrap();
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        ensure_fresh_work_dir(&empty).unwrap();

        let used = dir.path().join("used");
        std::fs::create_dir_all(used.join("prepared")).unwrap();
        std::fs::write(used.join("prepared").join("prepared.json"), "{}").unwrap();
        let error = ensure_fresh_work_dir(&used).unwrap_err().to_string();
        assert!(error.contains("not empty"), "{error}");

        let file = dir.path().join("file");
        std::fs::write(&file, "x").unwrap();
        let error = ensure_fresh_work_dir(&file).unwrap_err().to_string();
        assert!(error.contains("not a directory"), "{error}");

        // `invent` refuses the used directory and leaves what was there untouched.
        let train = write_csv(dir.path(), 60);
        let options = InventOptions::new(train, "y", &used);
        let error = format!("{:#}", invent(&options).unwrap_err());
        assert!(error.contains("not empty"), "{error}");
        assert_eq!(
            std::fs::read_to_string(used.join("prepared").join("prepared.json")).unwrap(),
            "{}"
        );
        assert_eq!(std::fs::read_dir(&used).unwrap().count(), 1);
    }

    #[test]
    fn only_jain_executable_pipelines_are_admitted() {
        let executable = Executable::load().unwrap();
        assert_eq!(executable.engine, "featmath");
        assert_eq!(executable.rev.len(), 40);
        assert!(executable.admits("(pipe X (standardize) (ridge))"));
        // A search head jain-math 9fa594e does not implement.
        assert!(!executable.admits("(pipe X (standardize) (ridge_gcv))"));
        // Quarantined in jain-math.
        assert!(!executable.admits("(pipe X (rff_orthogonal) (ridge))"));
        assert!(!executable.admits("not a genome"));
    }

    #[test]
    fn events_refuses_unresolved_parent_work_alias() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 4);
        let work = dir.path().join("missing/../work");
        let events = dir.path().join("work/prepared/cuts.ndjson");
        let result = check_events(&events, &train, &work);
        assert!(
            result.is_err(),
            "unresolved-parent --work-dir alias was accepted"
        );
        assert!(!dir.path().join("work").exists());
        assert!(!dir.path().join("missing").exists());
    }

    #[test]
    fn events_refuses_input_and_output_aliases() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 4);
        let work = dir.path().join("work");
        let check = |events: &Path| check_events(events, &train, &work);
        // The input, by another spelling too.
        let msg = check(&train).unwrap_err().to_string();
        assert!(msg.contains("--train"), "{msg}");
        assert!(check(&dir.path().join("work/../train.csv")).is_err());
        // The work directory itself (not yet created) and anything under it.
        assert!(check(&work).unwrap_err().to_string().contains("--work-dir"));
        assert!(check(&work.join("progress.ndjson")).is_err());
        // An existing directory.
        let msg = check(dir.path()).unwrap_err().to_string();
        assert!(msg.contains("directory"), "{msg}");
        // A new file beside them is fine.
        check(&dir.path().join("progress.ndjson")).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn events_refuses_hard_link_input_and_nonregular_targets() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 4);
        let before = std::fs::read(&train).unwrap();
        let work = dir.path().join("work");
        let alias = dir.path().join("input-alias.ndjson");
        std::fs::hard_link(&train, &alias).unwrap();
        let refused = check_events(&alias, &train, &work);
        assert!(refused.is_err(), "hard-linked --train was accepted");
        assert_eq!(std::fs::read(&train).unwrap(), before);
        assert!(!work.exists());
        // A Unix socket is an existing nonregular filesystem target. Never open it: the
        // preflight itself must refuse it, just as it must refuse a FIFO/device.
        let socket = dir.path().join("events.socket");
        let _listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        assert!(check_events(&socket, &train, &work).is_err());
        assert_eq!(std::fs::read(&train).unwrap(), before);
        assert!(!work.exists());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn events_open_preserves_input_and_truncates_only_valid_regular_output() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 4);
        let before = std::fs::read(&train).unwrap();
        let work = dir.path().join("work");
        let alias = dir.path().join("alias");
        std::fs::hard_link(&train, &alias).unwrap();
        assert!(open_events(&alias, &train, &work).is_err());
        assert_eq!(std::fs::read(&train).unwrap(), before);
        let output = dir.path().join("events.ndjson");
        std::fs::write(&output, b"old journal").unwrap();
        let file = open_events(&output, &train, &work).unwrap();
        assert_eq!(file.metadata().unwrap().len(), 0);
        assert_eq!(std::fs::read(&train).unwrap(), before);
        assert!(!work.exists());
    }

    #[test]
    fn preflight_refuses_bad_counts_before_reading_anything() {
        let mut options = InventOptions::new("/nonexistent.csv", "y", "/nonexistent");
        options.top_k = 0;
        assert!(preflight(&options)
            .unwrap_err()
            .to_string()
            .contains("--top-k"));
        options.top_k = 8;
        options.cuts = prepare::FOLDS + 1;
        assert!(preflight(&options).is_err());
        options.cuts = 3;
        options.source_commit = Some("abc".into());
        assert!(preflight(&options)
            .unwrap_err()
            .to_string()
            .contains("--source-commit"));
    }

    #[test]
    fn invent_ranks_executable_genomes_and_is_deterministic_under_an_evaluation_budget() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 120);
        let run = |work: &str| {
            let mut options = InventOptions::new(&train, "y", dir.path().join(work));
            options.budget = search::Budget {
                evaluations: 24,
                seconds: 600,
            };
            options.cuts = 2;
            options.top_k = 4;
            invent(&options).unwrap()
        };
        let first = run("one");
        assert!(first.ok, "{first:?}");
        assert_eq!(first.schema, SCHEMA);
        assert_eq!(first.rows, 120);
        assert_eq!(first.feature_names, ["a", "b"]);
        assert!(!first.candidates.is_empty() && first.candidates.len() <= 4);
        let executable = Executable::load().unwrap();
        for (index, candidate) in first.candidates.iter().enumerate() {
            assert_eq!(candidate.rank, index + 1);
            assert_ne!(candidate.origin, search::REMEASURE, "{}", candidate.genome);
            assert!(executable.admits(&candidate.genome), "{}", candidate.genome);
            assert_eq!(candidate.cuts_measured, 2);
            assert!(candidate.mean_hold_rmse.is_finite());
        }
        assert!(first
            .candidates
            .windows(2)
            .all(|w| w[0].mean_hold_rmse <= w[1].mean_hold_rmse));
        assert!(first.ranked - first.excluded_not_executable >= first.candidates.len());
        let second = run("two");
        let genomes = |c: &Candidates| {
            c.candidates
                .iter()
                .map(|c| c.genome.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(genomes(&first), genomes(&second));
        // A used work directory is refused.
        let mut again = InventOptions::new(&train, "y", dir.path().join("one"));
        again.budget = search::Budget {
            evaluations: 24,
            seconds: 600,
        };
        assert!(invent(&again)
            .unwrap_err()
            .to_string()
            .contains("--work-dir"));
    }

    #[test]
    fn a_finish_request_ranks_what_was_measured_and_says_it_finished_early() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 120);
        let budget = search::Budget {
            evaluations: 24,
            seconds: 600,
        };
        // Asked to finish before the first generation: only generation 0's exempt head is fitted
        // on each cut, the comparison still measures it on every cut, and it is ranked as usual.
        let mut options = InventOptions::new(&train, "y", dir.path().join("finished"));
        options.budget = budget;
        options.cuts = 2;
        options.top_k = 4;
        options.finish = Some(Arc::new(AtomicBool::new(true)));
        let finished = invent(&options).unwrap();
        assert!(finished.ok, "{finished:?}");
        assert!(finished.finished_early);
        assert!(!finished.candidates.is_empty());
        assert!(finished.candidates.iter().all(|c| c.cuts_measured == 2));
        // The cut record's `stopped_by` is the comparison's outcome, as on any run; what shows
        // the finish is how little each cut tried.
        for cut in &finished.searches {
            assert!(
                cut.completed_evaluations < budget.evaluations as u64,
                "{cut:?}"
            );
        }
        let document = serde_json::to_value(&finished).unwrap();
        assert_eq!(document["finished_early"], true);

        // Nobody asked: the flag stays down and the document has no such key.
        let mut options = InventOptions::new(&train, "y", dir.path().join("full"));
        options.budget = budget;
        options.cuts = 2;
        options.top_k = 4;
        options.finish = Some(Arc::new(AtomicBool::new(false)));
        let full = invent(&options).unwrap();
        assert!(!full.finished_early);
        assert!(serde_json::to_value(&full)
            .unwrap()
            .get("finished_early")
            .is_none());
        let tried = |c: &Candidates| {
            c.searches
                .iter()
                .map(|s| s.completed_evaluations)
                .sum::<u64>()
        };
        assert!(tried(&finished) < tried(&full));
    }

    #[test]
    fn invent_ranks_candidates_by_an_objective_and_records_it() {
        let dir = tempfile::tempdir().unwrap();
        let train = write_csv(dir.path(), 120);
        let objective = jope_runtime::fitness::Fitness::parse(
            r#"{"metric":"asymmetric_absolute","under_cost":4,"over_cost":1}"#,
        )
        .unwrap();
        let mut options = InventOptions::new(&train, "y", dir.path().join("objective"));
        options.budget = search::Budget {
            evaluations: 24,
            seconds: 600,
        };
        options.cuts = 2;
        options.top_k = 4;
        options.objective = Some(objective.clone());
        let ranked = invent(&options).unwrap();
        assert_eq!(
            ranked.objective.as_ref().unwrap().fitness_sha256,
            objective.identity().unwrap()
        );
        assert!(ranked
            .candidates
            .iter()
            .all(|c| c.mean_hold_objective.is_some_and(f64::is_finite)));
        assert!(ranked
            .candidates
            .windows(2)
            .all(|w| w[0].mean_hold_objective <= w[1].mean_hold_objective));
        let mut refused = InventOptions::new(&train, "y", dir.path().join("refused"));
        refused.objective = Some(jope_runtime::fitness::Fitness::parse("sharpe").unwrap());
        assert!(invent(&refused).is_err());
        assert!(!dir.path().join("refused").exists());
    }
}
