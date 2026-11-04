use super::*;

/// Unique physical receipts count once even if fold-zero evidence is reused.
pub fn cost(observations: &[&Observation]) -> Value {
    let mut receipts = BTreeSet::new();
    let mut predictors = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    let mut cpu = 0.;
    let mut wall = 0.;
    let mut unknown_cpu = 0usize;
    let mut unknown_wall = 0usize;
    let mut completed = 0usize;
    for observation in observations {
        let result = &observation.result;
        if result.reused_from_request_sha256.is_some()
            || !receipts.insert(&result.worker_receipt_sha256)
        {
            continue;
        }
        if let Some(value) = result.consumption.cpu_seconds {
            cpu += value;
        } else {
            unknown_cpu += 1;
        }
        if let Some(value) = result.consumption.wall_seconds {
            wall += value;
        } else {
            unknown_wall += 1;
        }
        if result.status == EvaluationStatus::Completed {
            completed += 1;
            predictors.insert(&result.fitted_state_sha256);
            recipes.insert(&observation.request.recipe_sha256);
        }
    }
    json!({"worker_command_receipts":receipts.len(),"completed_fits":completed,
        "unique_fitted_predictors":predictors.len(),"unique_reusable_recipes":recipes.len(),
        "known_worker_cpu_seconds":cpu,"unknown_cpu_attempts":unknown_cpu,
        "known_sum_worker_wall_seconds":wall,"unknown_wall_attempts":unknown_wall,
        "cpu_seconds_per_unique_predictor_known":(!predictors.is_empty()).then(||cpu/predictors.len() as f64),
        "cpu_seconds_per_unique_recipe_known":(!recipes.is_empty()).then(||cpu/recipes.len() as f64),
        "controller_cpu_seconds":null,"gpu_seconds":null,
        "predictor_identity_policy":"complete retained fitted-artifact SHA256; behavioral equivalence is not inferred",
        "recipe_identity_policy":"canonical native DSL SHA256 or pinned external method configuration SHA256",
        "accounting_note":"Worker totals exclude controller work; unknown costs remain unknown. No cloud price is assumed."})
}

pub(crate) fn cpu_seconds() -> Result<f64> {
    let time = nix::time::clock_gettime(nix::time::ClockId::CLOCK_PROCESS_CPUTIME_ID)?;
    Ok(time.tv_sec() as f64 + time.tv_nsec() as f64 / 1e9)
}

/// Read-only recovery/reporting. Completed worker outputs are authenticated and
/// rescored with original parent-side labels. Missing receipts stay missing.
/// No process is launched, input generated, selection changed or clock started.
pub fn report(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
) -> Result<Value> {
    #[cfg(feature = "asm")]
    if plan.scope == super::super::origin_proof::SCOPE {
        return super::super::origin_proof::report(plan, manifest, datasets, journal);
    }
    plan.validate()?;
    let (approved, hash) = super::super::cli::read_panel(manifest)?;
    ensure!(
        hash == plan.experiment.panel_sha256 && digest(&approved)? == digest(&plan.panel)?,
        "report panel differs from approved input"
    );
    let retained: CampaignPlan = serde_json::from_slice(&fs::read(journal.join("campaign.json"))?)?;
    ensure!(
        retained.identity()? == plan.identity()?,
        "journal belongs to another plan"
    );
    let data = plan
        .panel
        .datasets
        .iter()
        .map(|d| super::super::data::load_development(datasets, d))
        .collect::<Result<Vec<_>>>()?;
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
    #[allow(unused_mut)]
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        let recovered = super::super::asm_campaign::recover(plan, &data, journal)?;
        asm_inventory = recovered.inventory;
        aliases = recovered.aliases;
        for (dataset, methods) in recovered.methods {
            roster
                .get_mut(&dataset)
                .context("unknown generated dataset")?
                .extend(methods);
        }
    }
    let mut jobs = BTreeMap::new();
    for stage in [
        "screen-jobs.json",
        "confirm-jobs.json",
        "asm-screen-jobs.json",
    ] {
        let path = journal.join(stage);
        if !path.try_exists()? {
            continue;
        }
        for job in serde_json::from_slice::<Vec<WorkerJob>>(&fs::read(path)?)? {
            ensure!(
                jobs.insert(job.request.identity()?, job).is_none(),
                "duplicate retained job"
            );
        }
    }
    let mut observations = Vec::new();
    let mut report = Vec::new();
    for dataset in &data {
        let mut screen = Vec::new();
        let mut confirmations = Vec::new();
        let mut inventory = Vec::new();
        for method in &roster[&dataset.dataset.id] {
            for (stage, folds) in [
                (EvaluationStage::Screen, 0..1),
                (EvaluationStage::Confirm, 0..5),
            ] {
                for fold in folds {
                    let cut = dataset.cut(fold)?;
                    let request = request(dataset, &cut, method, &plan.experiment, stage)?;
                    let request_id = request.identity()?;
                    let Some(job) = jobs.get(&request_id) else {
                        if stage == EvaluationStage::Screen {
                            inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                                "status":"not_started","reason":"no scheduled job"}));
                        }
                        continue;
                    };
                    ensure!(
                        job.request.identity()? == request_id
                            && job.directory == journal.join("jobs").join(&request_id),
                        "stale retained job"
                    );
                    let input_path = journal.join("inputs").join(format!("{request_id}.json"));
                    let input = job
                        .command
                        .inputs
                        .iter()
                        .find(|a| a.path == input_path)
                        .context("retained job lacks raw fit input")?;
                    input.verify()?;
                    let outcome_path = if plan.runtimes[method.runtime()].unavailable.is_some() {
                        journal
                            .join("planned-dependencies")
                            .join(format!("{}.json", job.identity()?))
                    } else {
                        journal
                            .join("finished")
                            .join(format!("{}.json", job.identity()?))
                    };
                    if !outcome_path.try_exists()? {
                        inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                            "status":"unknown","reason":"scheduled job has no terminal receipt"}));
                        continue;
                    }
                    let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(outcome_path)?)?;
                    let execution_method = if stage == EvaluationStage::Screen {
                        aliases
                            .get(&format!("{}/{}", dataset.dataset.id, method.id()))
                            .and_then(|id| {
                                roster[&dataset.dataset.id].iter().find(|m| m.id() == id)
                            })
                            .unwrap_or(method)
                    } else {
                        method
                    };
                    let fit = prepare_fit_impl(
                        plan,
                        dataset,
                        &cut,
                        execution_method,
                        stage,
                        journal,
                        false,
                    )?;
                    ensure!(
                        fit.job.identity()? == job.identity()?,
                        "retained command changed the pinned runtime or input closure"
                    );
                    let mut observation = score_fit(&fit, &outcome, &cut.hold.labels)?;
                    observation.method = method.id().to_owned();
                    inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                        "status":observation.result.status,"reason":observation.result.reason}));
                    if stage == EvaluationStage::Screen {
                        screen.push(observation.clone());
                    } else {
                        confirmations.push(observation.clone());
                    }
                    observations.push(observation);
                }
            }
        }
        let expected = roster[&dataset.dataset.id]
            .iter()
            .map(|method| {
                Ok((
                    method.id().to_owned(),
                    request(
                        dataset,
                        &dataset.cut(0)?,
                        method,
                        &plan.experiment,
                        EvaluationStage::Screen,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let frozen_path = journal
            .join("frozen")
            .join(format!("{}.json", dataset.dataset.id));
        let mut qualified = Value::Null;
        let mut selection_exposure = Value::Null;
        if frozen_path.try_exists()? {
            let frozen: protocol::FrozenSelection =
                serde_json::from_slice(&fs::read(frozen_path)?)?;
            ensure!(
                frozen
                    == protocol::freeze_screen(
                        &expected,
                        &screen,
                        roster[&dataset.dataset.id]
                            .iter()
                            .find(|m| m.id() == "asm.neural")
                            .map(Method::id)
                    )?,
                "stale frozen selection"
            );
            selection_exposure = super::super::selection_exposure::report(
                dataset,
                &frozen,
                &roster[&dataset.dataset.id],
            )?;
            let mut expected_confirm = Vec::new();
            for recipe in &frozen.recipes {
                let method = roster[&dataset.dataset.id]
                    .iter()
                    .find(|m| m.id() == recipe.method)
                    .context("unknown frozen method")?;
                for fold in 0..5 {
                    let req = request(
                        dataset,
                        &dataset.cut(fold)?,
                        method,
                        &plan.experiment,
                        EvaluationStage::Confirm,
                    )?;
                    if fold == 0 {
                        let reuse = reuse_fold_zero(
                            screen
                                .iter()
                                .find(|o| o.method == recipe.method)
                                .context("screen result absent")?,
                            req.clone(),
                        )?;
                        confirmations.push(reuse.clone());
                        observations.push(reuse);
                    }
                    expected_confirm.push((method.id().to_owned(), req));
                }
            }
            if confirmations.len() == expected_confirm.len() {
                qualified = serde_json::to_value(protocol::confirm(
                    &frozen,
                    &expected,
                    &screen,
                    roster[&dataset.dataset.id]
                        .iter()
                        .find(|m| m.id() == "asm.neural")
                        .map(Method::id),
                    &expected_confirm,
                    &confirmations,
                )?)?;
            }
        }
        #[cfg(feature = "asm")]
        let fitted_inference = if qualified.is_null() {
            json!({"winner":null,"status":"unattempted_confirmation_incomplete","ranking_changed":false})
        } else {
            super::super::fitted_export::report(
                journal,
                &serde_json::from_value::<Vec<protocol::ConfirmedRecipe>>(qualified.clone())?,
                &confirmations,
            )?
        };
        #[cfg(not(feature = "asm"))]
        let fitted_inference = json!({"status":"unavailable_build_requires_asm_feature"});
        report.push(
            json!({"dataset":dataset.dataset.id,"task":dataset.dataset.task,
            "inventory":inventory,"screen":screen,"confirmation":confirmations,"recipes":qualified,
            "fitted_inference":fitted_inference,
            "selection_exposure":selection_exposure,
            "asm":asm_inventory.get(&dataset.dataset.id)}),
        );
    }
    let window: Option<jope_pipeline::experiment::ExecutionWindow> =
        if journal.join("window.json").try_exists()? {
            let window: jope_pipeline::experiment::ExecutionWindow =
                serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
            window.validate(&plan.experiment)?;
            Some(window)
        } else {
            None
        };
    #[allow(unused_mut)]
    let mut accounting = cost(&observations.iter().collect::<Vec<_>>());
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        accounting = super::super::asm_campaign::accounting(plan, journal)?;
    }
    Ok(
        json!({"schema":"jope.panel-recovery-report/v2","scope":plan.scope,
        "plan_sha256":plan.identity()?,"experiment_sha256":plan.experiment.identity()?,
        "window":window,"datasets":report,"cost":accounting,
        "sealed_inputs_opened":false,"fits_started":0,"clock_started":false,
        "optimization_visible_labels":"development","confirmation_role":"adaptive_development",
        "five_fold_summary_is_unbiased_generalization_estimate":false,"test_headline_eligible":false,
        "complete_asm_demonstration":false}),
    )
}
