use super::*;

pub(crate) fn execute(
    allocation: &Allocation,
    path: &Path,
    sha: &str,
    stop: Arc<AtomicBool>,
) -> Result<Value> {
    admission_guards(&allocation.policy)?;
    verify_loaded_runtime(&allocation.policy)?;
    let current = current_job(
        &allocation.policy,
        allocation.attempt.job_id,
        &allocation.daemon_identity,
    )?;
    ensure!(
        current.active_attempt == Some(allocation.attempt)
            && matches!(current.state, JobState::Starting | JobState::Running),
        "superseded native allocation"
    );
    let (task, input) = load_task(&allocation.policy, path, sha)?;
    if allocation.bootstrap.kind == TaskKind::ResidentGpu {
        within_allocation(&task, &allocation.bootstrap)?;
    }
    let key = hash(&serde_json::to_vec(&(
        &task.run,
        task.generation,
        &task.task_id,
    ))?);
    let directory = allocation.policy.state_root.join("tasks").join(key);
    fs::create_dir_all(&directory)?;
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join("lock"))?;
    ensure!(lock.try_lock().is_ok(), "task is already executing");
    let intent = json!({"manifest_sha256":sha,"task":task,"input_manifest_sha256":task.input_manifest_sha256,"operation":input.operation,"output_receipt":input.output_receipt});
    let intent_path = directory.join("intent.json");
    if intent_path.exists() {
        ensure!(
            read_json::<Value>(&intent_path)? == intent,
            "conflicting duplicate task identity"
        );
    } else {
        create_immutable(&intent_path, &serde_json::to_vec(&intent)?)?;
    }
    let output = resolve(&allocation.policy.root, &input.output_receipt)?;
    let complete = directory.join("complete.json");
    if complete.exists() {
        let result: Value = read_json(&complete)?;
        ensure!(
            result["output_sha256"] == file_hash(&output)? && result["manifest_sha256"] == sha,
            "durable result changed"
        );
        return Ok(result);
    }
    let overlay = match &input.operation {
        Operation::Proposal {
            preparation_overlay,
            ..
        } => preparation_overlay.clone(),
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            let g: crate::native_genome::GenomeTask = crate::native_genome::load(
                &input.root,
                &crate::native_graph::Artifact {
                    path: manifest.clone(),
                    sha256: sha256.clone(),
                },
            )?;
            g.preparation_overlay
        }
        _ => None,
    };
    if let Some(a) = overlay {
        crate::preparation_overlay::load(&input.root, &a, &task.run)?
            .require_live(resident::now_ms()?, task.deadline.as_unix_millis())?;
    }
    let active = directory.join("active.json");
    let recovering = active.exists();
    if recovering {
        let previous: Value = read_json(&active)?;
        ensure!(
            previous["job_id"] == allocation.attempt.job_id.to_string()
                && previous["fence"].as_u64().context("prior fence absent")?
                    < allocation.attempt.fence.get(),
            "duplicate or stale task attempt"
        );
        ensure!(
            matches!(input.operation, Operation::Train { .. }),
            "unfinished nontraining attempt requires explicit reconciliation"
        );
    }
    write_json(
        &active,
        &json!({"job_id":allocation.attempt.job_id.to_string(),"fence":allocation.attempt.fence.get(),"attempt":allocation.attempt.number.get()}),
    )?;
    let _deadline = DeadlineWatch::start(
        Arc::clone(&stop),
        task.deadline.as_unix_millis(),
        task.checkpoint_allowance_ms.unwrap_or(1000),
    );
    let verified = VerifiedTask {
        #[cfg(feature = "neural")]
        attempt: allocation.attempt,
        #[cfg(feature = "neural")]
        manifest: task.clone(),
        #[cfg(feature = "neural")]
        root: allocation.policy.root.clone(),
        operation: input.operation.clone(),
        stop,
        #[cfg(feature = "neural")]
        recovering,
    };
    let _scratch = ScratchGuard::start(allocation, &task, &directory)?;
    let outcome = match &input.operation {
        Operation::DatasetPublication {
            manifest, sha256, ..
        }
        | Operation::PackAndPublish {
            manifest, sha256, ..
        } => run_dataset_operation(allocation, &task, &input, manifest, sha256, &output),
        _ => run_operation(&allocation.policy, &verified, &output),
    };
    if let Err(error) = outcome {
        write_json(
            &directory.join("interrupted.json"),
            &json!({"manifest_sha256":sha,"error":error.to_string(),"checkpoint_requested":verified.stop.load(Ordering::Acquire)}),
        )?;
        return Err(error);
    }
    ensure!(
        output.is_file(),
        "numerical operation produced no output receipt"
    );
    fs::File::open(&output)?.sync_all()?;
    let result = json!({"schema":"jope.smartcluster_completion.v1","task_id":task.task_id,"receipt_id":task.expected_receipt_id,"manifest_sha256":sha,"native_attempt":allocation.attempt,"dataset_logical_attempt":input.operation.dataset_logical_attempt(),"output_path":input.output_receipt,"output_sha256":file_hash(&output)?});
    create_immutable(&complete, &serde_json::to_vec(&result)?)?;
    Ok(result)
}

fn run_dataset_operation(
    allocation: &Allocation,
    task: &TaskManifest,
    input: &InputManifest,
    manifest: &Path,
    sha256: &str,
    output: &Path,
) -> Result<()> {
    let policy = &allocation.policy;
    let executable = policy
        .datasets_executable
        .as_ref()
        .context("fixed dataset executable unbound")?;
    let executable_sha = policy
        .datasets_sha256
        .as_ref()
        .context("fixed dataset digest unbound")?;
    ensure!(
        executable.is_absolute() && fs::canonicalize(executable)? == *executable,
        "physical absolute dataset executable required"
    );
    let mut executable_file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(executable)?;
    let meta = executable_file.metadata()?;
    ensure!(
        meta.is_file() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
        "root-owned immutable dataset executable required"
    );
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = executable_file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    ensure!(
        hex::encode(digest.finalize()) == *executable_sha,
        "dataset executable changed"
    );
    let manifest_path = resolve(&policy.root, manifest)?;
    let bytes = read_bounded(&manifest_path, Some(sha256))?;
    let request: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        request["schema"] == "jain.datasets_service_manifest.v1"
            && request["root"] == serde_json::to_value(&policy.root)?
            && request["task_id"] == task.task_id
            && request["attempt"]
                == input
                    .operation
                    .dataset_attempt(u64::from(allocation.attempt.number.get()))?
            && request["expected_receipt_id"] == task.expected_receipt_id
            && request["output_receipt"] == serde_json::to_value(&input.output_receipt)?,
        "dataset manifest differs from original native task/attempt/output"
    );
    let kind = request["operation"]["kind"]
        .as_str()
        .context("dataset operation absent")?;
    ensure!(
        match input.operation {
            Operation::PackAndPublish { .. } =>
                matches!(kind, "pack_and_publish" | "pack_continuation_and_publish"),
            Operation::DatasetPublication { .. } => matches!(
                kind,
                "winner_publish" | "corpus_publish" | "publish_pending"
            ),
            _ => false,
        },
        "dataset operation is not allowed by this service"
    );
    validate_dataset_packer(&policy.approved_dataset_packers, &request)?;
    if kind == "winner_publish" {
        crate::champions::recover_outbox(&policy.root)?;
    }
    // Fixed executable and fixed argument grammar. The original native cgroup
    // contains this child and drains it before any subsequent attempt.
    let status =
        std::process::Command::new(format!("/proc/self/fd/{}", executable_file.as_raw_fd()))
            .arg("service")
            .arg("--manifest")
            .arg(&manifest_path)
            .arg("--manifest-sha256")
            .arg(sha256)
            .stdin(std::process::Stdio::null())
            .status()?;
    ensure!(
        status.success(),
        "dataset authority refused operation: {status}"
    );
    let output_file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(output)?;
    ensure!(
        output_file.metadata()?.is_file() && output_file.metadata()?.len() <= 256 * 1024 * 1024,
        "dataset receipt exceeds 256 MiB bound or is not regular"
    );
    let mut output_bytes = Vec::new();
    output_file
        .take(256 * 1024 * 1024 + 1)
        .read_to_end(&mut output_bytes)?;
    ensure!(
        output_bytes.len() <= 256 * 1024 * 1024,
        "dataset receipt grew beyond bound"
    );
    let receipt: Value = serde_json::from_slice(&output_bytes)?;
    ensure!(
        receipt["schema"] == "jain.datasets_service_receipt.v1"
            && receipt["passed"] == true
            && receipt["task_id"] == task.task_id
            && receipt["attempt"] == request["attempt"]
            && receipt["receipt_id"] == task.expected_receipt_id
            && receipt["manifest_sha256"] == sha256
            && receipt["tool_sha256"] == *executable_sha,
        "dataset receipt identity or result differs"
    );
    let files = receipt["output_files"]
        .as_array()
        .context("dataset output closure missing")?;
    for file in files {
        let path = Path::new(
            file["path"]
                .as_str()
                .context("dataset output path absent")?,
        );
        ensure!(
            file_hash(&resolve(&policy.root, path)?)? == file["sha256"],
            "dataset output artifact changed"
        );
    }
    if matches!(kind, "pack_and_publish" | "pack_continuation_and_publish") {
        ensure!(
            !files.is_empty() && receipt["local_corpus_committed"] == true,
            "changed corpus has no durable local commit"
        );
    }
    Ok(())
}

pub(in crate::smartcluster_worker::policy) fn validate_dataset_packer(
    approved: &BTreeMap<PathBuf, String>,
    request: &Value,
) -> Result<()> {
    let kind = request["operation"]["kind"]
        .as_str()
        .context("dataset operation absent")?;
    if matches!(kind, "pack_and_publish" | "pack_continuation_and_publish") {
        let packer = PathBuf::from(
            request["operation"]["packer"]["path"]
                .as_str()
                .context("fixed dataset CPU packer path absent")?,
        );
        let digest = request["operation"]["packer"]["sha256"]
            .as_str()
            .context("fixed dataset CPU packer digest absent")?;
        ensure!(
            approved.get(&packer).is_some_and(|sha| sha == digest),
            "dataset CPU packer is not independently allowlisted by worker policy"
        );
    }
    Ok(())
}

fn bind_native_fit_policy(policy: &WorkerPolicy) -> Result<()> {
    for (name, path) in [
        ("JOPE_CPU_EVALUATOR", &policy.cpu_evaluator),
        ("JOPE_FIT_AUTHORITY", &policy.fit_authority),
        ("JOPE_FIT_SERVICE_SOCKET", &policy.fit_service_socket),
        ("JOPE_FIT_WITNESS", &policy.fit_witness),
    ] {
        let path = path.as_ref().context("native fixed fit binding absent")?;
        ensure!(path.is_absolute(), "native fixed fit binding is relative");
        std::env::set_var(name, path);
    }
    let scope = policy
        .fit_scope
        .as_ref()
        .context("native fixed fit scope absent")?;
    ensure!(!scope.trim().is_empty(), "native fit scope empty");
    std::env::set_var("JOPE_FIT_SCOPE", scope);
    Ok(())
}

fn run_operation(policy: &WorkerPolicy, task: &VerifiedTask, receipt: &Path) -> Result<()> {
    ensure!(
        !task.stop.load(Ordering::Acquire),
        "native control already lost"
    );
    match task.operation() {
        Operation::DirectEvaluation { manifest, sha256 }
        | Operation::SeededContinuation { manifest, sha256 } => {
            bind_native_fit_policy(policy)?;
            crate::native_genome::execute(
                &policy.root,
                &crate::native_graph::Artifact {
                    path: manifest.clone(),
                    sha256: sha256.clone(),
                },
                if matches!(task.operation(), Operation::DirectEvaluation { .. }) {
                    crate::native_genome::Stage::Direct
                } else {
                    crate::native_genome::Stage::Continuation
                },
                receipt,
            )
        }
        Operation::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            vocabulary,
            rng_seed,
        } => {
            ensure!(*evaluations > 0 && *seconds > 0, "empty search budget");
            // Search may only use the authority selected by the root-owned
            // policy. Never inherit an unrelated service or grant from the
            // daemon's environment when a policy binding is missing.
            let evaluator = policy
                .cpu_evaluator
                .as_ref()
                .context("native CPU evaluator unbound")?;
            let authority = policy
                .fit_authority
                .as_ref()
                .context("native fit authority unbound")?;
            let socket = policy
                .fit_service_socket
                .as_ref()
                .context("native fit socket unbound")?;
            let witness = policy
                .fit_witness
                .as_ref()
                .context("native fit witness unbound")?;
            let scope = policy
                .fit_scope
                .as_ref()
                .context("native fit scope unbound")?;
            ensure!(!scope.trim().is_empty(), "native fit scope empty");
            for path in [evaluator, authority, socket, witness] {
                ensure!(path.is_absolute(), "native fit binding must be absolute");
            }
            std::env::set_var("JOPE_CPU_EVALUATOR", evaluator);
            std::env::set_var("JOPE_FIT_AUTHORITY", authority);
            std::env::set_var("JOPE_FIT_SERVICE_SOCKET", socket);
            std::env::set_var("JOPE_FIT_WITNESS", witness);
            std::env::set_var("JOPE_FIT_SCOPE", scope);
            let seeds = seeds
                .as_ref()
                .map(|path| resolve(&policy.root, path))
                .transpose()?;
            let vocabulary = vocabulary
                .as_ref()
                .map(|path| resolve(&policy.root, path))
                .transpose()?;
            crate::gp::search_seeded_with_rng_in_root(
                &policy.root,
                &resolve(&policy.root, prepared)?,
                *cut,
                &resolve(&policy.root, output)?,
                crate::gp::SearchOptions {
                    evaluations: *evaluations,
                    seconds: *seconds,
                    qualification: *qualification,
                    seeds: seeds.as_deref(),
                    vocabulary: vocabulary.as_deref(),
                },
                *rng_seed,
            )?;
            write_json(
                receipt,
                &json!({"schema":"jope.native_gp_completion.v1","output":output,"evaluations_budget":evaluations,"seed":rng_seed,"artifacts":hash_directory(&resolve(&policy.root,output)?)?}),
            )
        }
        Operation::Pack {
            contexts,
            vocabulary,
            observations,
            output,
            qualification,
        } => {
            let observations = observations
                .iter()
                .map(|path| resolve(&policy.root, path))
                .collect::<Result<Vec<_>>>()?;
            crate::corpus_publication::build(
                &policy.root,
                crate::corpus_publication::PublicationSources {
                    contexts: &resolve(&policy.root, contexts)?,
                    vocab: &resolve(&policy.root, vocabulary)?,
                    observations: &observations,
                },
                &resolve(&policy.root, output)?,
                *qualification,
                None,
                None,
            )?;
            write_json(
                receipt,
                &json!({"schema":"jope.native_pack_completion.v1","artifacts":hash_directory(&resolve(&policy.root,output)?)?}),
            )
        }
        Operation::Verify => write_json(
            receipt,
            &json!({"schema":"jope.native_verification.v1","verified":true}),
        ),
        #[cfg(feature = "neural")]
        Operation::Train { .. } => crate::training::smartcluster_rank(task, receipt),
        #[cfg(feature = "neural")]
        Operation::Proposal { .. } => crate::proposals::smartcluster_rank(task, receipt),
        _ => anyhow::bail!("operation requires qualified neural runtime or is resident bootstrap"),
    }
}
pub(crate) fn hash_directory(path: &Path) -> Result<BTreeMap<PathBuf, String>> {
    let mut entries = BTreeMap::new();
    for entry in walkdir::WalkDir::new(path).follow_links(false) {
        let entry = entry?;
        ensure!(!entry.file_type().is_symlink(), "output symlink refused");
        if entry.file_type().is_file() {
            entries.insert(
                entry.path().strip_prefix(path)?.to_owned(),
                file_hash(entry.path())?,
            );
        }
    }
    ensure!(!entries.is_empty(), "numerical artifact directory is empty");
    Ok(entries)
}
