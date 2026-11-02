use super::*;

pub fn run(config: &Config, scratch: &Path) -> Result<Report> {
    config.check()?;
    let started = Instant::now();
    if config.has_kernel2() {
        // Before the lock file is created, so nothing is written through a symbolic link or into
        // a path that is not a directory; checked again under the reservation.
        check_scratch_kind(scratch)?;
    }
    let _reservation = Reservation::take(scratch)?;
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
pub(super) struct Searched {
    metrics: Metrics,
    cuts: Vec<CutRecord>,
    search_seconds: f64,
    evaluations: u64,
}

pub(super) fn run_one(
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
pub(super) fn search_output(scratch: &Path, slot: &str, dataset_hash: &str, seed: u64) -> PathBuf {
    scratch
        .join("search")
        .join(slot)
        .join(dataset_hash)
        .join(format!("seed-{seed}"))
}

/// The refusal of a kernel2 search output that already exists.
pub(super) const SCRATCH_IN_USE: &str = "already holds a kernel2 search; choose a new --scratch";

/// Refuse a kernel2 search's output directory when anything is already there, a symbolic link
/// included. The route publishes `search/cut-0` with plain writes, so a second search into the
/// same directory would replace the first one's evidence; the legacy search refuses its own
/// published `search.json` the same way, and `jope pipeline run` refuses a used `--output`.
pub(super) fn check_unused(output: &Path) -> Result<()> {
    match std::fs::symlink_metadata(output) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {}", output.display())),
        Ok(_) => bail!("{}: {SCRATCH_IN_USE}", output.display()),
    }
}

/// The refusal of a scratch directory that a comparison with a kernel2 profile cannot use
/// ([`check_scratch_kind`], [`check_scratch_unused`]).
pub(super) const SCRATCH_NOT_UNUSED: &str =
    "already holds ablation output; a kernel2 comparison needs an absent or empty --scratch";

/// Refuse, for a comparison with a kernel2 profile, a scratch path that exists but is not a
/// directory, a symbolic link included, also when the path ends in `/` (which would make
/// `symlink_metadata` follow the link). An absent path is admitted: [`Reservation::take`]
/// creates it. Anything else, an inspection error included, is refused.
pub(super) fn check_scratch_kind(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_NOT_UNUSED}", scratch.display());
    let path: PathBuf = scratch.components().collect();
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(refused),
        Ok(metadata) => {
            ensure!(metadata.file_type().is_dir(), refused());
            Ok(())
        }
    }
}

/// Refuse, for a comparison with a kernel2 profile and under its [`Reservation`], a scratch
/// directory that is not unused: it must be a directory, not a symbolic link
/// ([`check_scratch_kind`]), whose only entry is this comparison's own [`LOCK_FILE`]. Anything
/// else, an inspection error included, is refused. What this enforces: every dataset's
/// `prepared/<dataset hash>/prepared.json` is shared by its searches, so a comparison with a
/// kernel2 arm writes its evidence only into a scratch that held nothing but its own lock file,
/// and it then marks the scratch ([`mark_kernel2`]). A later comparison through this harness
/// refuses the marked scratch as its scratch ([`check_not_kernel2`]) and as the place of its
/// `--out` document ([`check_out`]). Writers outside the harness are not covered.
pub(super) fn check_scratch_unused(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_NOT_UNUSED}", scratch.display());
    check_scratch_kind(scratch)?;
    for entry in std::fs::read_dir(scratch).with_context(refused)? {
        let name = entry.with_context(refused)?.file_name();
        ensure!(
            name.as_os_str() == std::ffi::OsStr::new(LOCK_FILE),
            refused()
        );
    }
    Ok(())
}

/// The file a comparison holds in its scratch directory while it runs ([`Reservation`]).
pub(super) const LOCK_FILE: &str = ".ablation-lock";
/// The permanent marker of a scratch directory a comparison with a kernel2 profile used
/// ([`mark_kernel2`]).
pub(super) const KERNEL2_MARKER: &str = ".ablation-kernel2";
/// The refusal of a scratch directory whose [`LOCK_FILE`] exists.
pub(super) const SCRATCH_RESERVED: &str = "another comparison holds this --scratch";
/// The refusal, by every comparison, of a scratch directory that holds [`KERNEL2_MARKER`].
pub(super) const SCRATCH_HOLDS_KERNEL2: &str =
    "holds a kernel2 comparison's evidence; use a new --scratch";

/// An exclusive reservation of a scratch directory for one comparison: its [`LOCK_FILE`],
/// created atomically (`create_new`) before the comparison checks or writes anything in the
/// scratch, and removed, with nothing else, when the reservation is dropped, also while
/// unwinding. A second comparison on the same scratch, in this process or another, is refused
/// while the first holds it. The file records the holder's process id and start time for a
/// person; it holds no evidence. A crash can leave it behind; it is then removed by hand, once no
/// comparison is running.
pub(super) struct Reservation {
    lock: PathBuf,
}

impl Reservation {
    /// Create `scratch` when it is absent (an empty directory changes no evidence), then its
    /// [`LOCK_FILE`], refusing a lock file that already exists.
    pub(super) fn take(scratch: &Path) -> Result<Reservation> {
        use std::io::Write as _;
        std::fs::create_dir_all(scratch)
            .with_context(|| format!("create {}", scratch.display()))?;
        let lock = scratch.join(LOCK_FILE);
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&lock)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => bail!(
                "{}: {SCRATCH_RESERVED} ({}); if no comparison is running, remove the lock file",
                scratch.display(),
                lock.display()
            ),
            Err(error) => return Err(error).with_context(|| format!("create {}", lock.display())),
        };
        let reservation = Reservation { lock };
        let started = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        writeln!(file, "pid {} started_unix {started}", std::process::id())
            .with_context(|| format!("write {}", reservation.lock.display()))?;
        Ok(reservation)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // Only the lock file: everything the comparison wrote stays. A failure leaves a lock
        // that the next comparison reports by name.
        let _ = std::fs::remove_file(&self.lock);
    }
}

/// Refuse, for every comparison and under its [`Reservation`], a scratch directory that holds
/// [`KERNEL2_MARKER`], so no later comparison through this harness, legacy-only included, uses a
/// kernel2 comparison's scratch as its own scratch. An inspection error is refused too. The
/// `--out` document is admitted separately ([`check_out`]).
pub(super) fn check_not_kernel2(scratch: &Path) -> Result<()> {
    let refused = || format!("{}: {SCRATCH_HOLDS_KERNEL2}", scratch.display());
    match std::fs::symlink_metadata(scratch.join(KERNEL2_MARKER)) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(refused),
        Ok(_) => bail!(refused()),
    }
}

/// Mark a scratch directory, under the [`Reservation`] and before any other write, as one a
/// comparison with a kernel2 profile used: [`KERNEL2_MARKER`], created atomically and never
/// removed. Every later comparison refuses the scratch ([`check_not_kernel2`]).
pub(super) fn mark_kernel2(scratch: &Path) -> Result<()> {
    use std::io::Write as _;
    let marker = scratch.join(KERNEL2_MARKER);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&marker)
        .with_context(|| format!("create {}", marker.display()))?;
    file.write_all(b"this --scratch holds a kernel2 comparison's evidence; use a new --scratch\n")
        .with_context(|| format!("write {}", marker.display()))
}

/// The refusal of a `--out` destination ([`check_out`]); the message names the reason.
pub(super) const OUT_REFUSED: &str = "--out refused";

/// `path` with its symbolic links and `..` resolved, also when it does not exist yet: the nearest
/// existing ancestor (inspected without following a final symbolic link, so a dangling link
/// counts as existing and fails to resolve) is canonicalized, and the components that do not
/// exist yet are appended. A `..` among those is refused, since what it names depends on
/// directories that do not exist yet, and so is an ancestor that cannot be inspected. The error
/// is the reason, for the caller's message.
pub(super) fn resolve_destination(path: &Path) -> std::result::Result<PathBuf, String> {
    let path: PathBuf = path.components().collect();
    let mut existing = Path::new("");
    for ancestor in path.ancestors() {
        if ancestor.as_os_str().is_empty() {
            break;
        }
        match std::fs::symlink_metadata(ancestor) {
            Ok(_) => {
                existing = ancestor;
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "{} cannot be inspected: {error}",
                    ancestor.display()
                ))
            }
        }
    }
    let rest = path
        .strip_prefix(existing)
        .map_err(|_| format!("{} cannot be resolved", path.display()))?;
    let base = if existing.as_os_str().is_empty() {
        Path::new(".")
    } else {
        existing
    };
    let mut resolved = std::fs::canonicalize(base)
        .map_err(|error| format!("{} cannot be resolved: {error}", base.display()))?;
    for component in rest.components() {
        match component {
            std::path::Component::Normal(name) => resolved.push(name),
            std::path::Component::CurDir => {}
            _ => {
                return Err(format!(
                    "{} has a `..` after a directory that does not exist yet",
                    path.display()
                ))
            }
        }
    }
    Ok(resolved)
}

/// Decide whether the document may be written to `out`, for every comparison: [`execute`] asks
/// before the comparison creates its scratch or computes anything, and again immediately before
/// it creates the temporary file of the document ([`replace_document`]). Refused, with
/// [`OUT_REFUSED`] and the reason:
/// - `out`'s file name is [`LOCK_FILE`] or [`KERNEL2_MARKER`]: those names are reserved, since a
///   document written there would itself hold or mark its directory (a legacy-only `out` in its
///   own scratch, or a directory a later scratch or `out` lies under);
/// - `out` is a symbolic link, or exists and is not a regular file, or cannot be inspected;
/// - `out`, resolved by [`resolve_destination`] (symbolic links and `..` included), lies in a
///   directory that holds [`KERNEL2_MARKER`] or [`LOCK_FILE`], checked from the directory that
///   would hold the document up to the filesystem root, so a marked or reserved scratch anywhere
///   above it is caught;
/// - the comparison has a kernel2 profile and `out` lies inside its own scratch, resolved the
///   same way, which the scratch's marker does not cover before the comparison creates it.
///
/// A legacy-only comparison whose `out` is elsewhere, its own scratch included, is admitted, and
/// gets the same document bytes as before (see [`replace_document`] for what else differs). A
/// hard link at `out` is admitted: [`replace_document`] renames a new file over it, so the inode
/// it shares is not written. The check and the rename are separate steps; a change to `out`'s
/// directory between them is not prevented.
pub(super) fn check_out(out: &Path, scratch: Option<&Path>, kernel2: bool) -> Result<()> {
    let refused = |reason: &str| {
        format!(
            "{}: {OUT_REFUSED} ({reason}); choose another --out",
            out.display()
        )
    };
    let literal: PathBuf = out.components().collect();
    if let Some(name) = literal
        .file_name()
        .filter(|name| *name == LOCK_FILE || *name == KERNEL2_MARKER)
    {
        bail!(refused(&format!(
            "its file name {} is reserved for a comparison's lock file or marker",
            name.to_string_lossy()
        )))
    }
    match std::fs::symlink_metadata(&literal) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| refused("it cannot be inspected")),
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(refused("it is a symbolic link"))
        }
        Ok(metadata) if !metadata.file_type().is_file() => {
            bail!(refused("it exists and is not a regular file"))
        }
        Ok(_) => {}
    }
    let destination =
        resolve_destination(out).map_err(|reason| anyhow::anyhow!(refused(&reason)))?;
    let holder = destination.parent().unwrap_or(&destination);
    for dir in holder.ancestors() {
        for (name, what) in [
            (KERNEL2_MARKER, "a kernel2 comparison's evidence"),
            (LOCK_FILE, "a comparison in progress"),
        ] {
            let entry = dir.join(name);
            match std::fs::symlink_metadata(&entry) {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(error).with_context(|| {
                        refused(&format!("{} cannot be inspected", entry.display()))
                    })
                }
                Ok(_) => bail!(refused(&format!("{} holds {name}, {what}", dir.display()))),
            }
        }
    }
    if let (true, Some(scratch)) = (kernel2, scratch) {
        let own = resolve_destination(scratch).map_err(|reason| {
            anyhow::anyhow!(refused(&format!(
                "this comparison's kernel2 --scratch cannot be resolved: {reason}"
            )))
        })?;
        ensure!(
            !destination.starts_with(&own),
            refused(&format!(
                "it is inside this comparison's kernel2 --scratch {}",
                own.display()
            ))
        );
    }
    Ok(())
}

pub(super) fn legacy_searched(searches: &[CutSearch]) -> Result<Searched> {
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
pub(super) fn kernel2_search(
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

pub(super) fn check_phases(
    cut: usize,
    phases: &[(&str, &str)],
    seconds: f64,
    bound: u64,
) -> Result<()> {
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
pub(super) fn parallel_map<T: Sync, R: Send>(
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
