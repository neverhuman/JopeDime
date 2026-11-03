use super::*;

pub(crate) fn handle(root: &Path, command: EngineCmd) -> Result<Option<EngineCmd>> {
    match command {
        #[cfg(feature = "neural")]
        EngineCmd::GpuWorker { pool, rank } => jope_parallel::gpu_worker::serve(root, &pool, rank)?,
        EngineCmd::Fit {
            train,
            genome,
            output,
        } => jope_parallel::gp::fit(&train, &genome, &output)?,
        EngineCmd::FitView {
            prepared,
            cut,
            genome,
            output,
        } => jope_parallel::gp::fit_view_in_root(root, &prepared, cut, &genome, &output)?,
        EngineCmd::FitMemory {
            fd,
            training_sha256,
            preprocessing_sha256,
            genome,
            output,
        } => jope_parallel::gp::fit_memory(
            fd,
            &training_sha256,
            &preprocessing_sha256,
            &genome,
            &output,
        )?,
        EngineCmd::Predict {
            model,
            features,
            output,
        } => jope_parallel::gp::predict(&model, &features, &output)?,
        EngineCmd::Score {
            model,
            test,
            receipt,
            output,
        } => jope_parallel::gp::score(&model, &test, &receipt, &output)?,
        EngineCmd::OuterRelease { command } => {
            println!("{}", jope_parallel::gp::outer_release_cli::run(command)?)
        }
        EngineCmd::Bootstrap {
            prepared,
            cut,
            output,
        } => jope_parallel::gp::search_seeded_with_rng_in_root(
            root,
            &prepared,
            cut,
            &output,
            jope_parallel::gp::SearchOptions {
                evaluations: 80,
                seconds: 600,
                qualification: false,
                seeds: None,
                vocabulary: None,
            },
            jope_parallel::manifest::SEED,
        )?,
        EngineCmd::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            vocabulary,
            rng_seed,
        } => jope_parallel::gp::search_seeded_with_rng_in_root(
            root,
            &prepared,
            cut,
            &output,
            jope_parallel::gp::SearchOptions {
                evaluations,
                seconds,
                qualification,
                seeds: seeds.as_deref(),
                vocabulary: vocabulary.as_deref(),
            },
            rng_seed,
        )?,
        EngineCmd::GpService {
            wait_for_work,
            manifest,
            workers,
            threads,
        } => {
            jope_parallel::service::run_with_wait(root, &manifest, workers, threads, wait_for_work)?
        }
        EngineCmd::EvaluateWorker {
            train,
            validation,
            genome,
            output,
        } => jope_parallel::gp::evaluate(&train, &validation, &genome, &output)?,
        EngineCmd::Snapshot {
            source,
            label,
            include,
            output,
        } => {
            let m = snapshot::create(&source, &label, &include, &output)?;
            println!("snapshotted {} source files", m.files.len());
        }
        EngineCmd::FreezeFamilies {
            audit,
            prepared,
            review,
            decisions,
            parent,
            output,
        } => {
            let registry = jope_parallel::families::freeze(
                &audit,
                &prepared,
                &review,
                &decisions,
                parent.as_deref(),
                &output,
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&registry.quota_report())?
            );
        }
        EngineCmd::Audit {
            source,
            decisions,
            output,
        } => {
            let r = audit::run(&source, &decisions, &output)?;
            println!(
                "{}",
                serde_json::json!({"datasets":r.datasets.len(),"artifacts":r.artifact_counts,"complete":r.complete,"missing_roots":r.missing_roots})
            );
        }
        EngineCmd::Prepare {
            audit,
            dataset,
            output,
            archive_legacy,
            qualification,
            backing_manifest,
            family_registry,
            entity_review,
            authoritative_inventory_root,
        } => {
            let p = prepare::run_with_admission(
                &audit,
                &dataset,
                &output,
                archive_legacy,
                qualification,
                &backing_manifest,
                family_registry.as_deref(),
                entity_review.as_deref(),
                authoritative_inventory_root.as_deref(),
            )?;
            println!(
                "{}",
                serde_json::json!({"dataset":p.dataset,"rows":p.rows_total,"cuts":p.cuts.len(),"clean_evaluation":p.clean_evaluation,"family_partition_frozen":p.family_partition_frozen,"manifest":output.join("prepared.json")})
            );
        }
        EngineCmd::Stage {
            manifest,
            destination,
        } => transport::stage(&manifest, &destination)?,
        EngineCmd::Transfer {
            manifest,
            stage,
            host,
            identity,
            destination,
        } => transport::transfer(&manifest, &stage, &host, &identity, &destination)?,
        EngineCmd::Verify {
            manifest,
            destination,
        } => println!(
            "verified {} files",
            transport::verify(&manifest, &destination)?
        ),
        other => return Ok(Some(other)),
    }
    Ok(None)
}
