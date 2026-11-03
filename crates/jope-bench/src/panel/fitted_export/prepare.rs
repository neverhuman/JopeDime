use super::*;

/// Write each prepared public bundle once (0600, create-new); an existing file must match.
pub(super) fn materialize(public: &Path, entries: &[Entry]) -> Result<()> {
    for entry in entries {
        if let Entry::Prepared {
            backend, bundle, ..
        } = entry
        {
            bundle.validate()?;
            let public = public.join(name(*backend));
            fs::create_dir_all(&public)?;
            for (name, bytes) in &bundle.files {
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                let path = public.join(name);
                if path.try_exists()? {
                    ensure!(
                        fs::read(&path)? == bytes.as_bytes(),
                        "existing public model differs"
                    );
                } else {
                    let mut file = fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .mode(0o600)
                        .open(path)?;
                    file.write_all(bytes.as_bytes())?;
                    file.sync_all()?;
                }
            }
            campaign::immutable(&public.join("manifest.json"), &bundle.manifest)?;
        }
    }
    Ok(())
}
pub(super) fn winner(recipes: &[ConfirmedRecipe]) -> Result<Option<&ConfirmedRecipe>> {
    for recipe in recipes {
        ensure!(
            recipe
                .mean_loss
                .is_none_or(|loss| loss.is_finite() && recipe.completed_folds == 5),
            "invalid five-fold loss"
        );
    }
    Ok(recipes
        .iter()
        .filter(|r| r.mean_loss.is_some())
        .min_by(|a, b| {
            a.mean_loss
                .unwrap()
                .total_cmp(&b.mean_loss.unwrap())
                .then(a.method.cmp(&b.method))
        }))
}
pub(super) fn prepare_entries(
    view: &FittedInferenceView<'_>,
    rows: &[Vec<Option<String>>],
) -> Result<Vec<Entry>> {
    ORDER
        .into_iter()
        .map(
            |backend| match inference::export_inference(view, backend)? {
                ExportOutcome::Unsupported { reason, .. } => Ok(Entry::Unsupported {
                    backend,
                    reason: serde_json::to_value(reason)?,
                }),
                ExportOutcome::Emitted { bundle } => {
                    let qualification =
                        Box::new(inference::qualification_plan_for_view(view, &bundle, rows)?);
                    Ok(Entry::Prepared {
                        backend,
                        bundle,
                        qualification,
                    })
                }
            },
        )
        .collect()
}
pub(in crate::panel) fn prepare(args: ExportArgs) -> Result<Value> {
    let preparation_start = std::time::Instant::now();
    let preparation_cpu_start = campaign::cpu_seconds()?;
    ensure!(args.fold < 5, "fold must be0..4");
    let campaign_plan = artifact(&args.campaign.plan)?;
    let campaign: CampaignPlan = serde_json::from_slice(&fs::read(&campaign_plan.path)?)?;
    ensure!(
        campaign.identity()? == args.campaign.plan_sha256,
        "campaign identity changed"
    );
    let journal = fs::canonicalize(&args.campaign.journal)?;
    let report = campaign::report(
        &campaign,
        &args.campaign.manifest,
        &args.campaign.datasets,
        &journal,
    )?;
    let dataset_report = report["datasets"]
        .as_array()
        .context("dataset report absent")?
        .iter()
        .find(|r| r["dataset"] == args.dataset)
        .context("dataset absent")?;
    let recipes: Vec<ConfirmedRecipe> = serde_json::from_value(dataset_report["recipes"].clone())
        .context("five-fold confirmation unavailable")?;
    let winner =
        winner(&recipes)?.context("no complete five-fold winner; no replacement fitted")?;
    let observations: Vec<Observation> =
        serde_json::from_value(dataset_report["confirmation"].clone())?;
    let observation = observations
        .into_iter()
        .find(|o| o.method == winner.method && o.request.fold == args.fold)
        .context("winner fitted fold absent")?;
    ensure!(
        observation.result.status == EvaluationStatus::Completed,
        "winner fit incomplete"
    );
    let existing = location(&journal, &observation)?.join("prepared.json");
    if existing.try_exists()? {
        let previous: PreparedExport = serde_json::from_slice(&fs::read(&existing)?)?;
        ensure!(
            serde_json::to_value(&previous.observation)? == serde_json::to_value(&observation)?,
            "retained export changed selected predictor"
        );
        previous.verify()?;
        return Ok(
            json!({"ok":true,"prepared":existing,"prepared_sha256":previous.identity()?,"reused":true,"fits":0,"compiler_commands":0,"ranking_changed":false,"preparation_cpu_seconds_this_invocation":campaign::cpu_seconds()? - preparation_cpu_start,"preparation_wall_seconds_this_invocation":preparation_start.elapsed().as_secs_f64()}),
        );
    }
    let original = observation
        .result
        .reused_from_request_sha256
        .as_ref()
        .cloned()
        .unwrap_or(observation.request.identity()?);
    let mut jobs = BTreeMap::new();
    for filename in [
        "screen-jobs.json",
        "asm-screen-jobs.json",
        "confirm-jobs.json",
    ] {
        let path = journal.join(filename);
        if !path.is_file() {
            continue;
        }
        for job in serde_json::from_slice::<Vec<WorkerJob>>(&fs::read(path)?)? {
            if job.request.identity()? == original {
                jobs.insert(job.identity()?, job);
            }
        }
    }
    ensure!(
        jobs.len() == 1,
        "missing or ambiguous original fitted worker"
    );
    let (_, original_job) = jobs.into_iter().next().unwrap();
    let original_outcome = artifact(
        &journal
            .join("finished")
            .join(format!("{}.json", original_job.identity()?)),
    )?;
    let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(&original_outcome.path)?)?;
    outcome.verify_outputs()?;
    let output: Value =
        serde_json::from_slice(&fs::read(original_job.directory.join("output.json"))?)?;
    let spec = campaign
        .panel
        .datasets
        .iter()
        .find(|d| d.id == args.dataset)
        .context("dataset specification absent")?;
    let data = super::super::data::load_development(&args.campaign.datasets, spec)?;
    let cut = data.cut(args.fold)?;
    ensure!(
        cut.hold.rows_sha256 == observation.request.scoring_rows_sha256,
        "export prediction rows changed"
    );
    let rows: Vec<_> = cut.hold.rows.iter().map(|r| r.values.clone()).collect();
    let fitted_sha = observation
        .result
        .fitted_state_sha256
        .clone()
        .context("fitted hash absent")?;
    let entries = if output["schema"] == "jope.panel-rust-worker-output/v1" {
        let output: super::super::worker::RustWorkerOutput = serde_json::from_value(output)?;
        ensure!(
            output.artifact_sha256 == fitted_sha
                && super::super::methods::digest(&output.artifact.bytes()?) == fitted_sha,
            "native fitted artifact changed"
        );
        prepare_entries(&FittedInferenceView::registered(&output.artifact)?, &rows)?
    } else if output["schema"] == "jope.panel-synthesized-worker-output/v1" {
        let output: super::super::synthesized::SynthesizedWorkerOutput =
            serde_json::from_value(output)?;
        ensure!(
            output.artifact_sha256 == fitted_sha
                && super::super::methods::digest(&output.artifact.bytes()?) == fitted_sha,
            "generated fitted artifact changed"
        );
        prepare_entries(&FittedInferenceView::synthesized(&output.artifact)?, &rows)?
    } else {
        ensure!(
            output["contract"] == super::super::methods::ADAPTER_CONTRACT
                && output["fitted"]["sha256"] == fitted_sha
                && output["fitted"]["reload_verified"] == false,
            "external native fitted receipt changed"
        );
        ORDER.into_iter().map(|backend|Entry::Unsupported{backend,reason:json!({"kind":"external_library_state","detail":"external fitted pickle is retained; reload is not verified, so standalone lowering is unavailable"})}).collect()
    };
    let mut prepared = PreparedExport {
        schema: "jope.fitted-winner-export/v1".into(),
        campaign_plan,
        source_sha40: campaign.experiment.source_sha.clone(),
        program: campaign.runtimes["rust"].program.clone(),
        dataset: args.dataset,
        observation,
        original_job,
        original_outcome,
        original_outputs: outcome.outputs,
        entries,
        preparation_wall_seconds: 0.,
        preparation_cpu_seconds: 0.,
    };
    let directory = location(&journal, &prepared.observation)?;
    fs::create_dir_all(&directory)?;
    materialize(&directory.join("public"), &prepared.entries)?;
    prepared.verify()?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    prepared.preparation_wall_seconds = preparation_start.elapsed().as_secs_f64();
    prepared.preparation_cpu_seconds = campaign::cpu_seconds()? - preparation_cpu_start;
    campaign::immutable(&directory.join("prepared.json"), &prepared)?;
    Ok(
        json!({"ok":true,"prepared":directory.join("prepared.json"),"prepared_sha256":prepared.identity()?,"dataset":prepared.dataset,"method":winner.method,"fold":args.fold,"native_fitted_sha256":fitted_sha,"public_bundle_root":directory.join("public"),"fits":0,"compiler_commands":0,"ranking_changed":false,"backends":prepared.entries.iter().map(|e|json!({"backend":e.backend(),"status":if matches!(e,Entry::Unsupported{..}){"unsupported"}else{"unattempted"}})).collect::<Vec<_>>()}),
    )
}
