use anyhow::Result;
use crate::command::{Cli, Cmd};
use jope_parallel::{audit, operations, prepare, snapshot, transport};
use std::path::PathBuf;

fn write_query_result(path: &std::path::Path, result: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let bytes = serde_json::to_vec_pretty(result)?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    output.write_all(&bytes)?;
    output.sync_all()?;
    std::fs::File::open(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(std::path::Path::new(".")),
    )?
    .sync_all()?;
    Ok(())
}

pub(crate) fn run(cli: crate::command::Cli) -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Cmd::VerifyFitCompletion {
            authority,
            witness,
            completion_sha256,
            authority_instance,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            let completion = registry.verify_committed_completion(&completion_sha256)?;
            anyhow::ensure!(
                completion.authority_instance == authority_instance,
                "unapproved fit authority instance"
            );
            println!("{}", serde_json::to_string(&completion)?);
        }
        Cmd::RegistryFit { fd, output } => jope_parallel::fit_boundary::execute(fd, &output)?,
        Cmd::RegistryInitializeNative {
            authority,
            witness,
            scopes,
        } => {
            let scopes: Vec<jope_parallel::eval_registry::ScopePolicy> =
                jope_parallel::manifest::read_json(&scopes)?;
            jope_parallel::eval_registry::Authority::initialize_native(
                &authority, &witness, scopes,
            )?;
        }
        Cmd::RegistryWitnessInitialize {
            witness_root,
            policy,
        } => {
            anyhow::ensure!(
                witness_root.is_absolute() && policy.is_absolute(),
                "witness initialization requires absolute custody paths"
            );
            let policy: jope_parallel::eval_registry::witness::WitnessPolicy =
                jope_parallel::manifest::read_json(&policy)?;
            jope_parallel::eval_registry::witness::WitnessStore::initialize(
                &witness_root,
                &policy,
            )?;
        }
        Cmd::RegistryInspectRuntime => {
            anyhow::ensure!(
                !cfg!(feature = "neural"),
                "fit authority runtime inspection requires the CPU-only executable"
            );
            let contract = jope_parallel::eval_registry::NumericalContract {
                protocol: "featmath_fixed_seed0_f64_single_thread_v1".into(),
                precision: "f64".into(),
                omp_threads: 1,
                rayon_threads: 1,
                blas_threads: 1,
                mkl_threads: 1,
            };
            let identity = jope_parallel::eval_registry::ProcessIdentity::current()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "schema": "parallel.fit_runtime_observation.v1",
                    "process": identity,
                    "executable_sha256": jope_parallel::manifest::file_hash(
                        std::path::Path::new("/proc/self/exe")
                    )?,
                    "authority_source_sha256": jope_parallel::eval_registry::witness::authority_source_sha256(),
                    "witness_source_sha256": jope_parallel::eval_registry::witness::source_sha256(),
                    "runtime_sha256": jope_parallel::eval_registry::current_runtime_digest(&contract)?,
                    "numerical_contract_sha256": jope_parallel::eval_registry::numerical_contract_sha256(&contract)?,
                    "contract": contract,
                    "execution_authority": false,
                    "numerical_execution": false
                }))?
            );
        }
        Cmd::RegistryAuthority {
            authority,
            witness,
            socket,
            workers,
            event_witness_config,
            event_witness_config_sha256,
            funding_config,
            native_fit_policy,
        } => jope_parallel::fit_boundary::serve_with_native_policy(
            authority,
            witness,
            socket,
            workers,
            &event_witness_config,
            &event_witness_config_sha256,
            jope_parallel::fit_boundary::FitPolicies {
                funding_config: funding_config.as_deref(),
                native_fit_policy: native_fit_policy.as_deref(),
            },
        )?,
        Cmd::FundingRequest { config } => {
            use std::io::Read;
            let mut bytes = Vec::new();
            std::io::stdin()
                .take(8 * 1024 * 1024 + 1)
                .read_to_end(&mut bytes)?;
            anyhow::ensure!(
                bytes.len() <= 8 * 1024 * 1024,
                "funding stdin exceeds bound"
            );
            let request: jope_parallel::eval_registry::service::FundingRequest =
                serde_json::from_slice(&bytes)?;
            let reply = jope_parallel::eval_registry::service::funding_request(&config, &request)?;
            println!("{}", serde_json::to_string(&reply)?);
        }
        Cmd::QueryEvaluate {
            authority,
            witness,
            socket,
            request,
            output,
        } => {
            anyhow::ensure!(!output.exists(), "query output reference already exists");
            let inputs = jope_parallel::manifest::read_json(&request)?;
            let result = jope_parallel::eval_registry::service::query_evaluate(
                &socket, &authority, &witness, &inputs,
            )?;
            write_query_result(&output, &result)?;
            println!("{}", serde_json::to_string(&result)?);
        }
        Cmd::QueryRetrieve {
            authority,
            witness,
            socket,
            request,
            output,
            deadline_monotonic_ns,
        } => {
            anyhow::ensure!(!output.exists(), "query output reference already exists");
            let inputs = jope_parallel::manifest::read_json(&request)?;
            let result = jope_parallel::eval_registry::service::query_retrieve(
                &socket,
                &authority,
                &witness,
                &inputs,
                deadline_monotonic_ns,
            )?;
            write_query_result(&output, &result)?;
            println!("{}", serde_json::to_string(&result)?);
        }
        Cmd::RegistryWitnessInspect {
            root,
            policy,
            policy_sha256,
        } => {
            use std::{io::Read, os::unix::fs::OpenOptionsExt};
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&policy)?;
            anyhow::ensure!(
                file.metadata()?.is_file() && file.metadata()?.len() <= 4 * 1024 * 1024,
                "bounded regular witness policy required"
            );
            let mut bytes = Vec::new();
            file.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
            anyhow::ensure!(
                bytes.len() <= 4 * 1024 * 1024
                    && jope_parallel::manifest::hash(&bytes) == policy_sha256,
                "original witness policy changed"
            );
            let policy = serde_json::from_slice(&bytes)?;
            let result =
                jope_parallel::eval_registry::witness::WitnessStore::verify_public_history(
                    &root, &policy,
                )?;
            println!("{}", serde_json::to_string(&result)?);
        }
        Cmd::RegistryWitnessRequest { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio(&root, &policy, &key)?
        }
        Cmd::RegistryWitnessAddNativeGrant { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_add_native_grant(
                &root, &policy, &key,
            )?
        }
        Cmd::RegistryWitnessEpochTransition { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_epoch_transition(
                &root, &policy, &key,
            )?
        }
        Cmd::RegistryWitnessGetAck { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_get_ack(&root, &policy, &key)?
        }
        Cmd::HistoryStage {
            authority,
            witness,
            inventory,
        } => {
            let inventory = jope_parallel::manifest::read_json(&inventory)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_stage(&inventory)?);
        }
        Cmd::HistoryImportBatch {
            authority,
            witness,
            batch,
        } => {
            let batch = jope_parallel::manifest::read_json(&batch)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_import_batch(&batch)?);
        }
        Cmd::HistoryCommitCoverage {
            authority,
            witness,
            stage_sha256,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_commit_coverage(&stage_sha256)?);
        }
        Cmd::HistoryPrepareLookupIndex { authority, witness } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.prepare_history_lookup_index()?);
        }
        Cmd::NativeScopeInspect {
            authority,
            witness,
            root,
            overlay,
            overlay_sha256,
            run,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!(
                "{}",
                registry.native_scope_inspect(
                    &root,
                    &jope_parallel::native_graph::Artifact {
                        path: overlay,
                        sha256: overlay_sha256
                    },
                    &run
                )?
            );
        }
        Cmd::NativeScopeRegister {
            authority,
            witness,
            registration,
            registration_sha256,
        } => {
            use std::io::Read;
            use std::os::unix::fs::OpenOptionsExt;
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&registration)?;
            anyhow::ensure!(
                file.metadata()?.is_file() && file.metadata()?.len() <= 1024 * 1024,
                "bounded regular scope registration required"
            );
            let mut bytes = Vec::new();
            file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
            anyhow::ensure!(
                bytes.len() <= 1024 * 1024
                    && jope_parallel::manifest::hash(&bytes) == registration_sha256,
                "scope registration changed"
            );
            let signed = serde_json::from_slice(&bytes)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.register_native_scope(&signed)?);
        }
        Cmd::HistoryCommitAggregateCoverage {
            authority,
            witness,
            aggregate,
            aggregate_sha256,
        } => {
            use std::{io::Read, os::unix::fs::OpenOptionsExt};
            let file = std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
                .open(&aggregate)?;
            anyhow::ensure!(
                file.metadata()?.is_file() && file.metadata()?.len() <= 4 * 1024 * 1024,
                "bounded regular aggregate required"
            );
            let mut bytes = Vec::new();
            file.take(4 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
            anyhow::ensure!(
                bytes.len() <= 4 * 1024 * 1024
                    && jope_parallel::manifest::hash(&bytes) == aggregate_sha256,
                "aggregate input changed"
            );
            let aggregate = serde_json::from_slice(&bytes)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!(
                "{}",
                registry.history_commit_aggregate_coverage(&aggregate)?
            );
        }
        Cmd::RecoveryInspect { authority, witness } => {
            let report =
                jope_parallel::eval_registry::Authority::recovery_inspect(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::RecoveryReplay { authority, witness } => {
            let report =
                jope_parallel::eval_registry::Authority::recovery_replay(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Cmd::RecoveryReconcile { authority, witness } => {
            println!(
                "{}",
                jope_parallel::eval_registry::Authority::recovery_reconcile(&authority, &witness)?
            );
        }
        Cmd::RecoveryCommitEpoch {
            authority,
            witness,
            report_sha256,
        } => {
            println!(
                "{}",
                jope_parallel::eval_registry::Authority::recovery_commit_epoch(
                    &authority,
                    &witness,
                    &report_sha256
                )?
            );
        }
        Cmd::NamespaceActivate {
            authority,
            witness,
            scope,
            capability,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.namespace_activate(&scope, &capability)?);
        }
        Cmd::NamespaceDeactivate {
            authority,
            witness,
            scope,
        } => {
            jope_parallel::eval_registry::Authority::open(&authority, &witness)?
                .namespace_deactivate(&scope)?;
        }
        Cmd::RegistryServiceFit => jope_parallel::fit_boundary::execute_service()?,
        Cmd::RegistryServiceQuery => jope_parallel::eval_registry::service::execute_query_worker()?,
        Cmd::CompareSeeds {
            manifest,
            workers,
            prefix,
            seconds,
            downstream,
        } => {
            use std::os::unix::process::CommandExt;
            let mut command = std::process::Command::new("python3");
            command
                .arg(cli.root.join("tools/seed_experiment.py"))
                .arg("--root")
                .arg(&cli.root)
                .arg("compare")
                .arg("--manifest")
                .arg(manifest)
                .arg("--workers")
                .arg(workers.to_string())
                .arg("--prefix")
                .arg(prefix.to_string())
                .arg("--seconds")
                .arg(seconds.to_string());
            if downstream {
                command.arg("--downstream");
            }
            return Err(command.exec().into());
        }
        #[cfg(feature = "neural")]
        Cmd::ApplyBinaryContexts {
            jobs,
            hyperion,
            foundation,
            target,
            assembly,
            output,
            aliases,
        } => jope_parallel::contexts::apply_binary_qualification(
            &cli.root,
            &jobs,
            &hyperion,
            &foundation,
            &target,
            &assembly,
            &output,
            &aliases,
        )?,
        Cmd::Enqueue {
            jobs,
            contexts,
            campaign_contexts,
        } => {
            println!(
                "{} new GP jobs admitted with verified saved contexts",
                jope_parallel::queue::enqueue_bound(
                    &cli.root,
                    &jobs,
                    &contexts,
                    campaign_contexts.as_deref()
                )?
            );
        }
        Cmd::CampaignCoverage {
            campaign,
            union,
            output,
            gp_root,
        } => {
            let roots: Vec<_> = gp_root.iter().map(|p| cli.root.join(p)).collect();
            let refs: Vec<_> = roots.iter().map(|p| p.as_path()).collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&jope_parallel::queue::write_coverage(
                    &cli.root, &campaign, &union, &refs, &output
                )?)?
            );
        }
        #[cfg(feature = "neural")]
        Cmd::ApplyContexts {
            jobs,
            hyperion,
            foundation,
            target,
            assembly,
            output,
            qualification,
        } => jope_parallel::contexts::apply(
            &cli.root,
            &jobs,
            &hyperion,
            &foundation,
            &target,
            &assembly,
            &output,
            qualification,
        )?,
        #[cfg(feature = "neural")]
        Cmd::UnionContexts { indexes, output } => {
            jope_parallel::contexts::union(&cli.root, &indexes, &output)?
        }
        #[cfg(feature = "neural")]
        Cmd::AssembleContexts {
            jobs,
            hyperion,
            foundation,
            target,
            output,
            registry,
            qualification,
        } => {
            if let Some(registry) = registry {
                jope_parallel::contexts::assemble_registered(
                    &cli.root,
                    &jobs,
                    &hyperion,
                    &foundation,
                    &target,
                    &registry,
                    &output,
                    qualification,
                )?;
            } else {
                jope_parallel::contexts::assemble(
                    &cli.root,
                    &jobs,
                    &hyperion,
                    &foundation,
                    &target,
                    &output,
                    qualification,
                )?;
            }
        }

        #[cfg(feature = "neural")]
        Cmd::FoundationService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(
            &cli.root,
            &jobs,
            &spec,
            &output,
            batch,
            "foundation",
        )?,
        #[cfg(feature = "neural")]
        Cmd::FoundationRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::foundation_rank(
            &cli.root, &jobs, &spec, &output, rank, world, batch,
        )?,

        #[cfg(feature = "neural")]
        Cmd::TargetService {
            jobs,
            spec,
            output,
            batch,
        } => jope_parallel::components::service(&cli.root, &jobs, &spec, &output, batch, "target")?,
        #[cfg(feature = "neural")]
        Cmd::TargetRank {
            jobs,
            spec,
            output,
            rank,
            world,
            batch,
        } => jope_parallel::components::target_rank(
            &cli.root, &jobs, &spec, &output, rank, world, batch,
        )?,

        Cmd::HyperionService { jobs, spec, output } => {
            jope_parallel::hyperion::service(&cli.root, &jobs, &spec, &output)?
        }
        Cmd::HyperionRank {
            jobs,
            spec,
            output,
            rank,
            world,
        } => jope_parallel::hyperion::rank(&cli.root, &jobs, &spec, &output, rank, world)?,
        Cmd::ExpandCuts {
            prepared,
            output,
            rounds,
        } => {
            prepare::expand_cuts(&prepared, &output, rounds)?;
        }
        Cmd::IndexCuts {
            prepared,
            backing_manifest,
            metadata,
            output,
        } => prepare::index_existing(&prepared, &backing_manifest, &metadata, &output)?,
        Cmd::EvaluateView {
            prepared,
            cut,
            genome,
            output,
        } => jope_parallel::gp::evaluate_view(&prepared, cut, &genome, &output)?,
        Cmd::EvaluateMemory {
            fd,
            genome,
            output,
            prepared,
            cut,
        } => {
            // W5-T05: a guarded evaluator's process group dies with its search controller.
            jope_parallel::gp::guard_evaluator_group()?;
            jope_parallel::gp::evaluate_prepared_memory(
                fd,
                &genome,
                &output,
                prepared.as_deref().zip(cut),
            )?
        }
        #[cfg(feature = "neural")]
        Cmd::Train {
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
            &cli.root,
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
        Cmd::TrainRank {
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
            &cli.root,
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
        Cmd::QualifyBatchRank {
            corpus,
            output,
            launch,
            rank,
            microbatch,
            global_batch,
            steps,
            initialize,
        } => jope_parallel::training::qualify_batch_rank(
            &cli.root,
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
        Cmd::ProposeRanked {
            manifest,
            rank,
            world,
        } => jope_parallel::ranked_proposals::run(&cli.root, &manifest, rank, world)?,
        #[cfg(feature = "neural")]
        Cmd::AssembleRanked { manifest } => {
            jope_parallel::ranked_proposals::assemble(&cli.root, &manifest)?
        }
        #[cfg(feature = "neural")]
        Cmd::Propose {
            generation_manifest,
            corpus,
            checkpoint,
            output,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::controller::propose_with_generation(
            &cli.root,
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
        Cmd::ProposeRank {
            corpus,
            checkpoint,
            output,
            rank,
            world,
            cuts,
            cut_start,
            shuffled,
        } => jope_parallel::proposals::rank(
            &cli.root,
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
        Cmd::MonitorCheckpoint {
            corpus,
            checkpoint,
            panel,
            output,
        } => jope_parallel::monitoring::evaluate(&cli.root, &corpus, &checkpoint, &panel, &output)?,
        Cmd::ComparisonBaseline {
            jobs,
            vocabulary,
            output,
        } => {
            jope_parallel::tournament::baseline(
                &cli.root,
                &jope_parallel::manifest::read_json(&jobs)?,
                &jope_parallel::manifest::read_json(&vocabulary)?,
                &output,
            )?;
        }
        Cmd::BuildCorpus {
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
                        &cli.root,
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
                        &cli.root,
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
                    &cli.root,
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
        Cmd::ExportTeachingCorpus { corpus, output } => {
            jope_parallel::teaching_store::export(&cli.root, &corpus, &output)?
        }
        Cmd::ProjectCorpus {
            source_root,
            corpus,
            output,
        } => {
            let receipt = jope_parallel::corpus_projection::project(
                &cli.root,
                source_root.as_deref().unwrap_or(cli.root.as_path()),
                &corpus,
                &output,
            )?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
        Cmd::Doctor => println!("{}", serde_json::to_string_pretty(&operations::doctor())?),
        #[cfg(feature = "neural")]
        Cmd::GpuWorker { pool, rank } => jope_parallel::gpu_worker::serve(&cli.root, &pool, rank)?,
        Cmd::Fit {
            train,
            genome,
            output,
        } => jope_parallel::gp::fit(&train, &genome, &output)?,
        Cmd::FitView {
            prepared,
            cut,
            genome,
            output,
        } => jope_parallel::gp::fit_view_in_root(&cli.root, &prepared, cut, &genome, &output)?,
        Cmd::FitMemory {
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
        Cmd::Predict {
            model,
            features,
            output,
        } => jope_parallel::gp::predict(&model, &features, &output)?,
        Cmd::Score {
            model,
            test,
            receipt,
            output,
        } => jope_parallel::gp::score(&model, &test, &receipt, &output)?,
        Cmd::OuterRelease { command } => {
            println!("{}", jope_parallel::gp::outer_release_cli::run(command)?)
        }
        Cmd::Bootstrap {
            prepared,
            cut,
            output,
        } => jope_parallel::gp::search_seeded_with_rng_in_root(
            &cli.root,
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
        Cmd::Search {
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
            &cli.root,
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
        Cmd::GpService {
            wait_for_work,
            manifest,
            workers,
            threads,
        } => jope_parallel::service::run_with_wait(
            &cli.root,
            &manifest,
            workers,
            threads,
            wait_for_work,
        )?,
        Cmd::EvaluateWorker {
            train,
            validation,
            genome,
            output,
        } => jope_parallel::gp::evaluate(&train, &validation, &genome, &output)?,
        Cmd::Snapshot {
            source,
            label,
            include,
            output,
        } => {
            let m = snapshot::create(&source, &label, &include, &output)?;
            println!("snapshotted {} source files", m.files.len());
        }
        Cmd::FreezeFamilies {
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
        Cmd::Audit {
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
        Cmd::Prepare {
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
        Cmd::Stage {
            manifest,
            destination,
        } => transport::stage(&manifest, &destination)?,
        Cmd::Transfer {
            manifest,
            stage,
            host,
            identity,
            destination,
        } => transport::transfer(&manifest, &stage, &host, &identity, &destination)?,
        Cmd::Verify {
            manifest,
            destination,
        } => println!(
            "verified {} files",
            transport::verify(&manifest, &destination)?
        ),
        #[cfg(feature = "neural")]
        Cmd::PolicyParity { directory } => jope_parallel::neural::parity(&directory)?,
        #[cfg(feature = "neural")]
        Cmd::ExportPolicy {
            checkpoint,
            output,
            receipt,
            expect_state_sha256,
            parent,
        } => println!(
            "{}",
            jope_parallel::policy_export::export(&jope_parallel::policy_export::ExportRequest {
                checkpoint,
                output,
                receipt,
                expect_state_sha256,
                parent,
            })?
        ),
        #[cfg(feature = "neural")]
        Cmd::LossOptimizerParity { directory } => {
            jope_parallel::neural::loss_optimizer_parity(&directory)?
        }
        #[cfg(feature = "neural")]
        Cmd::Encode {
            prepared,
            cut,
            spec,
            output,
            gpu,
        } => {
            let e =
                jope_parallel::encoding::encode(&cli.root, &prepared, cut, &spec, &output, gpu)?;
            println!("{}", serde_json::to_string(&e)?);
        }
        #[cfg(feature = "neural")]
        Cmd::DistributedProbe {
            world,
            output,
            timeout,
            fail_rank,
        } => jope_parallel::controller::probe(world, &output, timeout, fail_rank)?,
        #[cfg(feature = "neural")]
        Cmd::TrainingProbe {
            output,
            vocabulary,
            microbatch,
            global_batch,
            steps,
            timeout,
        } => jope_parallel::controller::training_probe(
            &output,
            &vocabulary,
            microbatch,
            global_batch,
            steps,
            timeout,
        )?,
        #[cfg(feature = "neural")]
        Cmd::TrainingProbeRank {
            directory,
            vocabulary,
            rank,
            world,
            microbatch,
            global_batch,
            steps,
        } => jope_parallel::neural::training_probe(
            rank,
            world,
            &directory,
            &vocabulary,
            microbatch,
            global_batch,
            steps,
        )?,
        #[cfg(feature = "neural")]
        Cmd::NcclProbe {
            directory,
            rank,
            world,
        } => jope_parallel::neural::nccl_probe(rank, world, &directory)?,
        Cmd::Status => println!(
            "{}",
            serde_json::to_string_pretty(&operations::status(&cli.root)?)?
        ),
        Cmd::Tui {
            experiment_manifest,
        } => {
            jope_parallel::dashboard::run_with_manifest(&cli.root, experiment_manifest.as_deref())?
        }
        Cmd::LearningHistory => println!(
            "{}",
            serde_json::to_string_pretty(
                &jope_parallel::learning_history::Cache::default()
                    .refresh(&cli.root)
                    .report()
            )?
        ),
        Cmd::Monitor => jope_parallel::telemetry::monitor(&cli.root)?,
        Cmd::PilotGate => operations::pilot_gate(&cli.root)?,
        #[cfg(feature = "neural")]
        Cmd::Campaign { spec } => jope_parallel::campaign::run(&cli.root, &spec)?,
        #[cfg(feature = "neural")]
        Cmd::Cycle {
            corpus,
            output,
            steps,
            cuts,
            cut_start,
            contexts,
            initialize,
            observations,
            qualification,
        } => jope_parallel::pipeline::run(
            &cli.root,
            &corpus,
            &output,
            steps,
            cuts,
            cut_start,
            contexts.as_deref(),
            initialize.as_deref(),
            &observations,
            qualification,
        )?,
    }
    Ok(())
}
