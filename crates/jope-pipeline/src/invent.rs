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

mod model;
pub use model::{
    BudgetRecord, Candidate, Candidates, CutRecord, Executable, InventOptions, VocabularyRecord,
};
#[cfg(test)]
mod tests;

/// The stdout document's schema.
pub const SCHEMA: &str = "jope.candidates.v1";

/// The jain-math registry snapshot that decides which genomes are offered.
pub const JAIN_EXECUTABLE_OPS: &str =
    include_str!("../../jope-engine/assets/grammar/jain-math-executable-9fa594e2.json");

/// Exit status when the search measured nothing the caller can execute.
pub const NO_CANDIDATES_EXIT: u8 = 3;

/// Most genomes one call may return.
pub const MAX_TOP_K: usize = 64;

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
