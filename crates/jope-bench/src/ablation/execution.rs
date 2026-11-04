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
pub(crate) struct Searched {
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
pub(crate) fn search_output(scratch: &Path, slot: &str, dataset_hash: &str, seed: u64) -> PathBuf {
    scratch
        .join("search")
        .join(slot)
        .join(dataset_hash)
        .join(format!("seed-{seed}"))
}

mod scratch;
pub(crate) use scratch::*;

pub(crate) fn legacy_searched(searches: &[CutSearch]) -> Result<Searched> {
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
pub(crate) fn kernel2_search(
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

pub(crate) fn check_phases(
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
pub(crate) fn parallel_map<T: Sync, R: Send>(
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
