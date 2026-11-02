use super::*;

pub(super) fn handle(root: &Path, command: EngineCmd) -> Result<Option<EngineCmd>> {
    match command {
        EngineCmd::VerifyFitCompletion {
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
        EngineCmd::RegistryFit { fd, output } => jope_parallel::fit_boundary::execute(fd, &output)?,
        EngineCmd::RegistryInitializeNative {
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
        EngineCmd::RegistryWitnessInitialize {
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
        EngineCmd::RegistryInspectRuntime => {
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
        EngineCmd::RegistryAuthority {
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
        EngineCmd::FundingRequest { config } => {
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
        EngineCmd::QueryEvaluate {
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
        EngineCmd::QueryRetrieve {
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
        EngineCmd::RegistryWitnessInspect {
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
        EngineCmd::RegistryWitnessRequest { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio(&root, &policy, &key)?
        }
        EngineCmd::RegistryWitnessAddNativeGrant { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_add_native_grant(
                &root, &policy, &key,
            )?
        }
        EngineCmd::RegistryWitnessEpochTransition { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_epoch_transition(
                &root, &policy, &key,
            )?
        }
        EngineCmd::RegistryWitnessGetAck { root, policy, key } => {
            jope_parallel::eval_registry::witness::serve_stdio_get_ack(&root, &policy, &key)?
        }
        EngineCmd::HistoryStage {
            authority,
            witness,
            inventory,
        } => {
            let inventory = jope_parallel::manifest::read_json(&inventory)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_stage(&inventory)?);
        }
        EngineCmd::HistoryImportBatch {
            authority,
            witness,
            batch,
        } => {
            let batch = jope_parallel::manifest::read_json(&batch)?;
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_import_batch(&batch)?);
        }
        EngineCmd::HistoryCommitCoverage {
            authority,
            witness,
            stage_sha256,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.history_commit_coverage(&stage_sha256)?);
        }
        EngineCmd::HistoryPrepareLookupIndex { authority, witness } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.prepare_history_lookup_index()?);
        }
        EngineCmd::NativeScopeInspect {
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
        EngineCmd::NativeScopeRegister {
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
        EngineCmd::HistoryCommitAggregateCoverage {
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
        EngineCmd::RecoveryInspect { authority, witness } => {
            let report =
                jope_parallel::eval_registry::Authority::recovery_inspect(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        EngineCmd::RecoveryReplay { authority, witness } => {
            let report =
                jope_parallel::eval_registry::Authority::recovery_replay(&authority, &witness)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        EngineCmd::RecoveryReconcile { authority, witness } => {
            println!(
                "{}",
                jope_parallel::eval_registry::Authority::recovery_reconcile(&authority, &witness)?
            );
        }
        EngineCmd::RecoveryCommitEpoch {
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
        EngineCmd::NamespaceActivate {
            authority,
            witness,
            scope,
            capability,
        } => {
            let mut registry = jope_parallel::eval_registry::Authority::open(&authority, &witness)?;
            println!("{}", registry.namespace_activate(&scope, &capability)?);
        }
        EngineCmd::NamespaceDeactivate {
            authority,
            witness,
            scope,
        } => {
            jope_parallel::eval_registry::Authority::open(&authority, &witness)?
                .namespace_deactivate(&scope)?;
        }
        EngineCmd::RegistryServiceFit => jope_parallel::fit_boundary::execute_service()?,
        EngineCmd::RegistryServiceQuery => {
            jope_parallel::eval_registry::service::execute_query_worker()?
        }
        EngineCmd::CompareSeeds {
            manifest,
            workers,
            prefix,
            seconds,
            downstream,
        } => {
            use std::os::unix::process::CommandExt;
            let mut command = std::process::Command::new("python3");
            command
                .arg(root.join("tools/seed_experiment.py"))
                .arg("--root")
                .arg(&root)
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
        other => return Ok(Some(other)),
    }
    Ok(None)
}
