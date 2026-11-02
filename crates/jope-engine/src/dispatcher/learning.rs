use super::*;

pub(crate) fn handle(root: &Path, command: EngineCmd) -> Result<Option<EngineCmd>> {
    match command {
        #[cfg(feature = "neural")]
        EngineCmd::Train {
            corpus,
            output,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize,
            fail_rank,
            replay_policy,
        } => jope_parallel::controller::train_with_policy(
            &root,
            &corpus,
            &output,
            jope_parallel::controller::TrainingOptions {
                world,
                micro: microbatch,
                steps,
                checkpoint_steps,
                resume,
                fail_rank,
                initialize: initialize.as_deref(),
            },
            replay_policy.as_deref(),
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::TrainRank {
            corpus,
            output,
            launch,
            rank,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize,
            replay_policy,
            continuation_mode,
        } => jope_parallel::training::rank_with_policy(
            &root,
            &corpus,
            &output,
            &launch,
            rank,
            world,
            microbatch,
            steps,
            checkpoint_steps,
            resume,
            initialize.as_deref(),
            replay_policy.as_deref(),
            continuation_mode
                .as_deref()
                .map(jope_parallel::task_contract::ContinuationMode::parse)
                .transpose()?,
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::QualifyBatchRank {
            corpus,
            output,
            launch,
            rank,
            microbatch,
            global_batch,
            steps,
            initialize,
        } => jope_parallel::training::qualify_batch_rank(
            &root,
            &corpus,
            &output,
            &launch,
            rank,
            microbatch,
            global_batch,
            steps,
            &initialize,
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::ProposeRanked {
            manifest,
            rank,
            world,
        } => jope_parallel::ranked_proposals::run(&root, &manifest, rank, world)?,
        #[cfg(feature = "neural")]
        EngineCmd::AssembleRanked { manifest } => {
            jope_parallel::ranked_proposals::assemble(&root, &manifest)?
        }
        #[cfg(feature = "neural")]
        EngineCmd::Propose {
            generation_manifest,
            corpus,
            checkpoint,
            output,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::controller::propose_with_generation(
            &root,
            &corpus,
            &checkpoint,
            &output,
            jope_parallel::controller::ProposalOptions {
                cuts,
                cut_start,
                shuffled,
            },
            generation_manifest.as_deref(),
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::ProposeRank {
            corpus,
            checkpoint,
            output,
            rank,
            world,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::proposals::rank(
            &root,
            &corpus,
            &checkpoint,
            &output,
            rank,
            world,
            cuts,
            cut_start,
            shuffled,
        )?,
        #[cfg(feature = "neural")]
        EngineCmd::MonitorCheckpoint {
            corpus,
            checkpoint,
            panel,
            output,
        } => jope_parallel::monitoring::evaluate(&root, &corpus, &checkpoint, &panel, &output)?,
        EngineCmd::ComparisonBaseline {
            jobs,
            vocabulary,
            output,
        } => {
            jope_parallel::tournament::baseline(
                &root,
                &jope_parallel::manifest::read_json(&jobs)?,
                &jope_parallel::manifest::read_json(&vocabulary)?,
                &output,
            )?;
        }
        EngineCmd::BuildCorpus {
            contexts,
            vocabulary,
            mut observations,
            observation_manifest,
            output,
            qualification,
            scan_cache,
            progress,
            teaching_store,
            label_policy,
            admission_manifest,
        } => {
            if let Some(path) = observation_manifest {
                observations.extend(jope_parallel::manifest::read_json::<Vec<PathBuf>>(&path)?);
            }
            let heartbeat = progress.as_deref().map(|p| jope_parallel::telemetry::StageHeartbeat::start(p,
                serde_json::json!({"schema":"parallel.corpus_progress.v1","phase":"packing","output":output,"complete":false}))).transpose()?;
            if let Some(store) = teaching_store {
                if let Some(admission) = admission_manifest {
                    anyhow::ensure!(
                        label_policy == "strict_midrank_v4",
                        "admission manifest requires strict policy"
                    );
                    jope_parallel::teaching_store::publish_admitted(
                        &root,
                        jope_parallel::teaching_store::PublishRequest {
                            contexts_path: &contexts,
                            vocabulary: &vocabulary,
                            sources: &observations,
                            output: &output,
                            store: &store,
                            qualification,
                        },
                        &admission,
                        |operation, done, total| {
                            heartbeat
                                .as_ref()
                                .map_or(Ok(()), |h| h.advance(operation, done, total))
                        },
                    )?;
                } else {
                    jope_parallel::teaching_store::publish(
                        &root,
                        jope_parallel::teaching_store::PublishRequest {
                            contexts_path: &contexts,
                            vocabulary: &vocabulary,
                            sources: &observations,
                            output: &output,
                            store: &store,
                            qualification,
                        },
                        &label_policy,
                        |operation, done, total| {
                            heartbeat
                                .as_ref()
                                .map_or(Ok(()), |h| h.advance(operation, done, total))
                        },
                    )?;
                }
            } else {
                anyhow::ensure!(
                    label_policy == "reference_v1",
                    "label-policy changes require a separate teaching store"
                );
                jope_parallel::corpus_publication::build(
                    &root,
                    jope_parallel::corpus_publication::PublicationSources {
                        contexts: &contexts,
                        vocab: &vocabulary,
                        observations: &observations,
                    },
                    &output,
                    qualification,
                    scan_cache.as_deref(),
                    heartbeat.as_ref(),
                )?;
            }
            if let Some(h) = heartbeat {
                h.advance("verified corpus published", 1, Some(1))?;
                h.finish()?;
                let path = progress.as_deref().expect("heartbeat has a progress path");
                let mut receipt: serde_json::Value = jope_parallel::manifest::read_json(path)?;
                receipt["complete"] = true.into();
                receipt["phase"] = "completed".into();
                receipt["timestamp"] = jope_parallel::telemetry::now().into();
                jope_parallel::manifest::write_json(path, &receipt)?;
            }
        }
        EngineCmd::ExportTeachingCorpus { corpus, output } => {
            jope_parallel::teaching_store::export(&root, &corpus, &output)?
        }
        EngineCmd::ProjectCorpus {
            source_root,
            corpus,
            output,
        } => {
            let receipt = jope_parallel::corpus_projection::project(
                &root,
                source_root.as_deref().unwrap_or(root.as_path()),
                &corpus,
                &output,
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        EngineCmd::Doctor => println!("{}", serde_json::to_string_pretty(&operations::doctor())?),
        other => return Ok(Some(other)),
    }
    Ok(None)
}
