use super::*;

/// `defer_report` is honoured only by the single-dataset origin proof.
pub fn run_with(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
    defer_report: bool,
) -> Result<Value> {
    #[cfg(feature = "asm")]
    if defer_report && plan.scope == super::super::origin_proof::SCOPE {
        return super::super::origin_proof::run_deferred(
            plan, manifest, datasets, journal, pool, cancel,
        );
    }
    ensure!(
        !defer_report,
        "deferred reporting requires a single-dataset origin proof plan"
    );
    run(plan, manifest, datasets, journal, pool, cancel)
}

/// One pool owns both stages. Re-entering this function authenticates original
/// inputs and receipts and uses the pool's existing absolute deadline.
pub fn run(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
) -> Result<Value> {
    #[cfg(feature = "asm")]
    if plan.scope == super::super::origin_proof::SCOPE {
        return super::super::origin_proof::run(plan, manifest, datasets, journal, pool, cancel);
    }
    let controller_start = std::time::Instant::now();
    let controller_cpu_start = cpu_seconds()?;
    plan.validate()?;
    pool.reserve()?;
    // A run includes controller validation and preparation in its original budget.
    // The separate plan/preparation commands remain clock-free.
    pool.admit()?;
    for runtime in plan.runtimes.values().filter(|r| r.unavailable.is_none()) {
        runtime.program.verify()?;
        for input in runtime.inputs.iter().chain(&runtime.package_files) {
            input.verify()?;
        }
    }
    let (approved, hash) = super::super::cli::read_panel(manifest)?;
    ensure!(
        hash == plan.experiment.panel_sha256 && digest(&approved)? == digest(&plan.panel)?,
        "plan changed approved panel contents"
    );
    immutable(&journal.join("campaign.json"), plan)?;
    let data: Vec<_> = plan
        .panel
        .datasets
        .iter()
        .map(|d| super::super::data::load_development(datasets, d))
        .collect::<Result<_>>()?;
    let mut fits = Vec::new();
    for dataset in &data {
        let cut = dataset.cut(0)?;
        for method in &plan.methods[dataset.dataset.task.as_str()] {
            fits.push(prepare_fit(
                plan,
                dataset,
                &cut,
                method,
                EvaluationStage::Screen,
                journal,
            )?);
        }
    }
    immutable(
        &journal.join("screen-jobs.json"),
        &fits.iter().map(|f| &f.job).collect::<Vec<_>>(),
    )?;
    let outcomes = pool.run(fits.iter().map(|f| f.job.clone()).collect(), cancel)?;
    // Planned unavailable engines are immutable roster entries, not fit attempts.
    // Other never-started outcomes remain a resumable inventory.
    for outcome in &outcomes {
        let fit = fits
            .iter()
            .find(|fit| {
                fit.request
                    .identity()
                    .is_ok_and(|id| id == outcome.request_sha256)
            })
            .context("unknown dependency outcome")?;
        if plan.runtimes[fit.method.runtime()].unavailable.is_some() {
            immutable(
                &journal
                    .join("planned-dependencies")
                    .join(format!("{}.json", outcome.job_sha256)),
                outcome,
            )?;
        }
    }
    let has_pending = outcomes.iter().any(|outcome| {
        !journal
            .join("finished")
            .join(format!("{}.json", outcome.job_sha256))
            .exists()
            && !journal
                .join("planned-dependencies")
                .join(format!("{}.json", outcome.job_sha256))
                .exists()
    });
    // A missing supervisor is a dependency inventory, not a terminal scientific
    // attempt. Do not freeze empty winners or poison a future admitted resume.
    if has_pending
        || pool.window().is_none()
        || outcomes.iter().any(|outcome| {
            outcome
                .reason
                .as_ref()
                .is_some_and(|reason| reason.contains("execution admission unavailable"))
        })
    {
        return Ok(json!({"schema":"jope.panel-admission/v2", "ok":false,
            "experiment_sha256":plan.experiment.identity()?,"window":pool.window(),
            "outcomes":outcomes,"fits_started_by_this_call":null,
            "datasets":plan.panel.datasets.iter().map(|d|json!({"dataset":d.id,"task":d.task,"asm":"unknown_no_admitted_arm_outcome"})).collect::<Vec<_>>(),
            "accounting_note":"Partial inventory can include completed and interrupted attempts; no new-fit count is inferred from gate release.",
            "reason":"bounded execution unavailable; no selection or scientific result was frozen"}));
    }
    #[allow(unused_mut)]
    let mut screen = score_batch(&fits, outcomes, &data, journal)?;
    #[allow(unused_mut)]
    let mut roster: BTreeMap<String, Vec<Method>> = data
        .iter()
        .map(|d| {
            (
                d.dataset.id.clone(),
                plan.methods[d.dataset.task.as_str()].clone(),
            )
        })
        .collect();
    #[allow(unused_mut)]
    let mut asm_inventory = Value::Null;
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        let attached =
            super::super::asm_campaign::attach(plan, &data, journal, pool, cancel, &screen)?;
        asm_inventory = attached.inventory;
        if !attached.ready {
            return Ok(
                json!({"schema":"jope.asm-partial/v1","datasets":asm_inventory,"reason":"paired arms contain resumable inventory","window":pool.window(),"screen":screen}),
            );
        }
        for (dataset, methods) in attached.methods {
            roster
                .get_mut(&dataset)
                .context("unknown generated dataset")?
                .extend(methods);
        }
        screen.extend(attached.screen);
    }
    let mut selections = BTreeMap::new();
    let mut confirms = Vec::new();
    let mut reused = Vec::new();
    for dataset in &data {
        let dataset_screen: Vec<_> = screen
            .iter()
            .filter(|o| o.request.dataset_sha256 == dataset.dataset.source.sha256)
            .cloned()
            .collect();
        let expected: Vec<_> = dataset_screen
            .iter()
            .map(|o| (o.method.clone(), o.request.clone()))
            .collect();
        let asm_method = roster[&dataset.dataset.id]
            .iter()
            .find(|m| m.id() == "asm.neural")
            .map(Method::id);
        let selection = protocol::freeze_screen(&expected, &dataset_screen, asm_method)?;
        immutable(
            &journal
                .join("frozen")
                .join(format!("{}.json", dataset.dataset.id)),
            &selection,
        )?;
        for recipe in &selection.recipes {
            let method = roster[&dataset.dataset.id]
                .iter()
                .find(|m| m.id() == recipe.method)
                .context("frozen recipe absent")?;
            for fold in 0..5 {
                let cut = dataset.cut(fold)?;
                let fit = prepare_fit(
                    plan,
                    dataset,
                    &cut,
                    method,
                    EvaluationStage::Confirm,
                    journal,
                )?;
                if fold == 0 {
                    let screen_fit = dataset_screen
                        .iter()
                        .find(|o| o.method == recipe.method)
                        .context("screen absent")?;
                    let reuse = reuse_fold_zero(screen_fit, fit.request)?;
                    immutable(
                        &journal
                            .join("observations")
                            .join(format!("{}.json", reuse.request.identity()?)),
                        &reuse,
                    )?;
                    reused.push(reuse);
                } else {
                    confirms.push(fit);
                }
            }
        }
        selections.insert(
            dataset.dataset.id.clone(),
            (selection, expected, dataset_screen),
        );
    }
    immutable(
        &journal.join("confirm-jobs.json"),
        &confirms.iter().map(|f| &f.job).collect::<Vec<_>>(),
    )?;
    let outcomes = pool.run(confirms.iter().map(|f| f.job.clone()).collect(), cancel)?;
    if outcomes.iter().any(|outcome| {
        !journal
            .join("finished")
            .join(format!("{}.json", outcome.job_sha256))
            .exists()
    }) {
        return Ok(json!({"schema":"jope.panel-partial/v2","ok":false,
            "reason":"confirmation has resumable non-attempts","outcomes":outcomes,
            "window":pool.window(),"experiment_sha256":plan.experiment.identity()?}));
    }
    let mut confirmation = score_batch(&confirms, outcomes, &data, journal)?;
    confirmation.extend(reused);
    let mut datasets_report = Vec::new();
    for dataset in &data {
        let (frozen, expected_screen, screen_data) = &selections[&dataset.dataset.id];
        let observations: Vec<_> = confirmation
            .iter()
            .filter(|o| o.request.dataset_sha256 == dataset.dataset.source.sha256)
            .cloned()
            .collect();
        let expected: Vec<_> = observations
            .iter()
            .map(|o| (o.method.clone(), o.request.clone()))
            .collect();
        let confirmed = protocol::confirm(
            frozen,
            expected_screen,
            screen_data,
            roster[&dataset.dataset.id]
                .iter()
                .find(|m| m.id() == "asm.neural")
                .map(Method::id),
            &expected,
            &observations,
        )?;
        #[cfg(feature = "asm")]
        let fitted_inference =
            super::super::fitted_export::report(journal, &confirmed, &observations)?;
        #[cfg(not(feature = "asm"))]
        let fitted_inference = json!({"status":"unavailable_build_requires_asm_feature"});
        let asm_status = match asm_inventory.get(&dataset.dataset.id) {
            Some(inventory) => inventory.clone(),
            // Preserve this emitted qualification status when no comparator
            // inventory exists for the dataset.
            None => json!("not_run_comparator_qualification_only"),
        };
        datasets_report.push(
            json!({"dataset":dataset.dataset.id,"task":dataset.dataset.task,
            "screen":screen_data,"frozen":frozen,"confirmation":observations,"recipes":confirmed,
            "fitted_inference":fitted_inference,
            "selection_exposure":super::super::selection_exposure::report(dataset,frozen,&roster[&dataset.dataset.id])?,
            "asm":asm_status}),
        );
    }
    let all: Vec<_> = screen.iter().chain(&confirmation).collect();
    for runtime in plan.runtimes.values().filter(|r| r.unavailable.is_none()) {
        for input in runtime.inputs.iter().chain(&runtime.package_files) {
            input.verify()?;
        }
    }
    let mut accounting = cost(&all);
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        accounting = super::super::asm_campaign::accounting(plan, journal)?;
    }
    accounting["controller_cpu_seconds_this_invocation"] =
        json!(cpu_seconds()? - controller_cpu_start);
    accounting["controller_wall_seconds_this_invocation"] =
        json!(controller_start.elapsed().as_secs_f64());
    let report = json!({"schema":"jope.panel-report/v2", "scope":plan.scope,
        "plan_sha256":plan.identity()?, "experiment_sha256":plan.experiment.identity()?,
        "window":pool.window(), "datasets":datasets_report, "cost":accounting,
        "sealed_inputs_opened":false,"optimization_visible_labels":"development",
        "confirmation_role":"adaptive_development","five_fold_summary_is_unbiased_generalization_estimate":false,
        "complete_asm_demonstration":false});
    let report_bytes = serde_json::to_vec(&report)?;
    let report_hash = jope_pipeline::sha256_hex(&report_bytes);
    immutable(
        &journal.join("reports").join(format!("{report_hash}.json")),
        &report,
    )?;
    Ok(report)
}
