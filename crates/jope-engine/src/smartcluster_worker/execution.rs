use super::*;

fn read_frame(stream: &mut UnixStream) -> Result<WorkerControlFrame> {
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let length = u32::from_be_bytes(header) as usize;
    ensure!(length <= FRAME_LIMIT, "control frame too large");
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes)?;
    let (frame, remaining): (WorkerControlFrame, &[u8]) = postcard::take_from_bytes(&bytes)?;
    ensure!(remaining.is_empty(), "trailing control bytes");
    Ok(frame)
}
fn report(
    stream: &mut UnixStream,
    attempt: FencedAttempt,
    sequence: &mut u64,
    report: WorkerReport,
) -> Result<()> {
    report.validate()?;
    *sequence = sequence
        .checked_add(1)
        .context("report sequence overflow")?;
    let bytes = postcard::to_stdvec(&WorkerReportFrame {
        version: ProtocolVersion::CURRENT,
        attempt,
        sequence: *sequence,
        report,
    })?;
    ensure!(bytes.len() <= FRAME_LIMIT, "report too large");
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

fn within_allocation(child: &TaskManifest, parent: &TaskManifest) -> Result<()> {
    ensure!(
        child.kind == TaskKind::ResidentGpu
            && child.checkpoint_allowance_ms.unwrap_or(1000)
                <= parent.checkpoint_allowance_ms.unwrap_or(1000)
            && child.run == parent.run
            && child.deadline <= parent.deadline
            && child.resources.cpu.maximum_millicores <= parent.resources.cpu.maximum_millicores
            && child.resources.ram.hard_bytes <= parent.resources.ram.hard_bytes
            && child.resources.scratch_bytes <= parent.resources.scratch_bytes
            && child.resources.gpu == parent.resources.gpu,
        "resident subtask exceeds native allocation"
    );
    Ok(())
}

/// Persist the coordinator's dispatch intent before `Client::send_control` delivery.
pub fn freeze_dispatch_intent(path: &Path, command: &SubtaskCommand) -> Result<()> {
    let bytes = serde_json::to_vec(command)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "conflicting subtask dispatch intent"
        );
        return Ok(());
    }
    create_immutable(path, &bytes)
}
/// Delivery acknowledgement is not completion; verify the durable receipt separately.
pub async fn dispatch_subtask(
    client: &Client,
    job_id: jain_smartcluster::JobId,
    intent: &Path,
    command: &SubtaskCommand,
) -> Result<jain_smartcluster::OperationAck> {
    let payload = serde_json::to_vec(command)?;
    ensure!(payload.len() <= 64 * 1024, "subtask command too large");
    freeze_dispatch_intent(intent, command)?;
    Ok(client
        .send_control(jain_smartcluster::SendControlRequest {
            job_id,
            control: ControlMessage::Application {
                name: CONTROL_NAME.to_owned(),
                payload,
            },
        })
        .await?)
}
pub(crate) fn write_immutable_record(path: &Path, value: &Value) -> Result<()> {
    create_immutable(path, &serde_json::to_vec(value)?)
}
fn create_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    ensure!(
        bytes.len() <= FRAME_LIMIT,
        "durable control record too large"
    );
    let parent = path.parent().context("record parent absent")?;
    fs::create_dir_all(parent)?;
    // Keep interrupted staging files: final names become visible only after a
    // complete fsynced payload, using an atomic no-replace rename publication.
    let temporary = tempfile::Builder::new()
        .prefix(".immutable-pending-")
        .tempfile_in(parent)?;
    let (mut file, temporary_path) = temporary.keep()?;
    file.write_all(bytes)?;
    file.sync_all()?;
    match crate::native_persistence::publish_staged(&temporary_path, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            ensure!(fs::read(path)? == bytes, "conflicting immutable record");
        }
        Err(error) => return Err(error.into()),
    }
    fs::File::open(parent)?.sync_all()?;
    crate::native_persistence::discard_staging(&temporary_path)?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn execute(
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

fn validate_dataset_packer(approved: &BTreeMap<PathBuf, String>, request: &Value) -> Result<()> {
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

/// Cooperative scratch accounting supplements the daemon's reservation. A quota
/// is not claimed: immutable numerical outputs remain on the guarded data root.
struct ScratchGuard {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl ScratchGuard {
    fn start(allocation: &Allocation, task: &TaskManifest, state: &Path) -> Result<Self> {
        let done = Arc::new(AtomicBool::new(false));
        if task.resources.scratch_bytes == 0 {
            return Ok(Self { done, thread: None });
        }
        let inherited_root = PathBuf::from(
            std::env::var_os("SCQ_SCRATCH_ROOT").context("native scratch root absent")?,
        );
        let inherited_budget: u64 = std::env::var("SCQ_SCRATCH_RESERVATION_BYTES")?.parse()?;
        ensure!(
            inherited_root == allocation.policy.scratch_root
                && inherited_budget == allocation.bootstrap.resources.scratch_bytes
                && task.resources.scratch_bytes <= inherited_budget,
            "native scratch reservation changed"
        );
        let allocation_root = inherited_root.join(allocation.attempt.job_id.to_string());
        let path = allocation_root.join(hash(&serde_json::to_vec(&(
            &task.run,
            task.generation,
            &task.task_id,
        ))?));
        fs::create_dir_all(&path)?;
        let within = |usage: ScratchUsage| {
            usage.task <= task.resources.scratch_bytes
                && usage.allocation <= inherited_budget
                && usage.root <= SCRATCH_ROOT_LIMIT
        };
        ensure!(
            within(scratch_usage(&inherited_root, &allocation_root, &path)?),
            "retained scratch exceeds task reservation"
        );
        std::env::set_var("TMPDIR", &path);
        let observed = Arc::clone(&done);
        let budget = task.resources.scratch_bytes;
        let state = state.to_owned();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                // One walk of the shared root per tick yields all three totals (the task and
                // allocation directories are inside it); walking each tree separately stat'ed
                // the task's files three times every 100 ms.
                let usage = scratch_usage(&inherited_root, &allocation_root, &path);
                if !usage.is_ok_and(|usage| {
                    usage.task <= budget
                        && usage.allocation <= inherited_budget
                        && usage.root <= SCRATCH_ROOT_LIMIT
                }) {
                    let _ = write_json(
                        &state.join("scratch-violation.json"),
                        &json!({"budget":budget,"path":path,"action":"terminate_and_preserve_attempt"}),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: the over-budget task stops now, after a best-effort record.
                    unsafe {
                        libc::_exit(78);
                    }
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        });
        Ok(Self {
            done,
            thread: Some(thread),
        })
    }
}
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
/// Byte cap on the whole inherited scratch root.
const SCRATCH_ROOT_LIMIT: u64 = 32 * 1024 * 1024 * 1024;

/// Regular-file bytes under the scratch root, its allocation directory and the task directory.
#[derive(Clone, Copy, Debug, PartialEq)]
struct ScratchUsage {
    root: u64,
    allocation: u64,
    task: u64,
}

/// Sums [`ScratchUsage`] in one walk of `root` (`allocation` and `task` are inside it). A
/// symlink anywhere is refused. A file or directory that disappears between the directory read
/// and its `lstat` is a temporary file the task removed (TMPDIR points here), not a violation, so
/// it is skipped. The root, allocation and task directories themselves must exist (each
/// separate walk used to fail on a missing one): the root as a directory, possibly behind a link
/// as before (walkdir follows a root link), the other two as physical directories. Every other
/// error still fails closed. Entries renamed during a walk are counted on the next tick.
fn scratch_usage(root: &Path, allocation: &Path, task: &Path) -> Result<ScratchUsage> {
    let root_metadata =
        fs::metadata(root).with_context(|| format!("scratch root {}", root.display()))?;
    ensure!(root_metadata.is_dir(), "scratch root is not a directory");
    for directory in [allocation, task] {
        let metadata = fs::symlink_metadata(directory)
            .with_context(|| format!("scratch directory {}", directory.display()))?;
        ensure!(
            metadata.is_dir(),
            "scratch path is not a physical directory: {}",
            directory.display()
        );
    }
    let vanished = |error: &std::io::Error| error.kind() == std::io::ErrorKind::NotFound;
    let mut usage = ScratchUsage {
        root: 0,
        allocation: 0,
        task: 0,
    };
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) if error.depth() > 0 && error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        ensure!(!entry.file_type().is_symlink(), "scratch symlink rejected");
        if !entry.file_type().is_file() {
            continue;
        }
        let bytes = match entry.metadata() {
            Ok(metadata) => metadata.len(),
            Err(error) if error.io_error().is_some_and(vanished) => continue,
            Err(error) => return Err(error.into()),
        };
        let add = |total: u64| total.checked_add(bytes).context("scratch size overflow");
        usage.root = add(usage.root)?;
        if entry.path().starts_with(allocation) {
            usage.allocation = add(usage.allocation)?;
            if entry.path().starts_with(task) {
                usage.task = add(usage.task)?;
            }
        }
    }
    Ok(usage)
}

struct DeadlineWatch {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}
impl DeadlineWatch {
    fn start(stop: Arc<AtomicBool>, deadline: u64, allowance: u64) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&done);
        let remaining =
            Duration::from_millis(deadline.saturating_sub(resident::now_ms().unwrap_or(u64::MAX)));
        let started = std::time::Instant::now();
        let thread = std::thread::spawn(move || {
            while !observed.load(Ordering::Acquire) {
                let now = resident::now_ms().unwrap_or(u64::MAX);
                if now >= deadline || started.elapsed() >= remaining {
                    stop.store(true, Ordering::Release);
                }
                if now >= deadline.saturating_add(allowance)
                    || started.elapsed()
                        >= remaining.saturating_add(Duration::from_millis(allowance))
                {
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: past deadline plus allowance the process is ended at once.
                    unsafe {
                        libc::_exit(76);
                    }
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }
}
impl Drop for DeadlineWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlStopReason {
    EndOfFile,
    FrameReadError,
    Checkpoint,
    MalformedNamedControl,
    UnknownControl,
    QueueFull,
    QueueDisconnected,
}

#[derive(Clone, Copy)]
struct ControlStopDeadline {
    instant: std::time::Instant,
    monotonic_ns: Option<u64>,
}

fn stop_control_with(
    stop: &AtomicBool,
    allowance: Duration,
    watchdog: impl FnOnce(std::time::Instant),
    persist: impl FnOnce(ControlStopDeadline) -> Result<()>,
) -> Result<()> {
    let now = std::time::Instant::now();
    let instant = now.checked_add(allowance).unwrap_or(now);
    stop.store(true, Ordering::Release);
    // Arm independently of all diagnostic filesystem I/O.
    watchdog(instant);
    let monotonic_ns = crate::host_os::monotonic_ns().ok().and_then(|n| {
        u64::try_from(
            instant
                .saturating_duration_since(std::time::Instant::now())
                .as_nanos(),
        )
        .ok()
        .and_then(|v| n.checked_add(v))
    });
    persist(ControlStopDeadline {
        instant,
        monotonic_ns,
    })
}

fn record_control_stop(
    state_root: &Path,
    attempt: FencedAttempt,
    sequence: u64,
    reason: ControlStopReason,
    deadline: ControlStopDeadline,
) -> Result<()> {
    let value = json!({"schema":"jope.smartcluster_control_stop.v1","attempt":attempt,
        "sequence":sequence,"reason":reason,"observed_at_unix_ms":resident::now_ms()?,
        "checkpoint_deadline_monotonic_ns":deadline.monotonic_ns,
        "checkpoint_remaining_ms":deadline.instant.saturating_duration_since(std::time::Instant::now()).as_millis()});
    ensure!(
        serde_json::to_vec(&value)?.len() <= 4096,
        "control stop receipt exceeds bound"
    );
    write_immutable_record(
        &state_root.join(format!(
            "control-stop-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &value,
    )
}

pub fn serve(path: &Path, sha: &str) -> Result<()> {
    let policy = load_policy()?;
    let mut allocation = authenticate(policy, path, sha)?;
    let (_, input) = load_task(&allocation.policy, path, sha)?;
    let stop = Arc::new(AtomicBool::new(false));
    let mut reader = allocation.stream.try_clone()?;
    let observed = Arc::clone(&stop);
    let attempt = allocation.attempt;
    let control_state_root = allocation.policy.state_root.clone();
    let allowance = allocation.bootstrap.checkpoint_allowance_ms.unwrap_or(1000);
    let (send, receive) = mpsc::sync_channel::<SubtaskCommand>(1);
    std::thread::spawn(move || {
        let mut previous = 0;
        let reason = loop {
            let frame = match read_frame(&mut reader) {
                Ok(frame) => frame,
                Err(error) => {
                    break if error
                        .downcast_ref::<std::io::Error>()
                        .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
                    {
                        ControlStopReason::EndOfFile
                    } else {
                        ControlStopReason::FrameReadError
                    }
                }
            };
            if frame.attempt != attempt
                || frame.sequence <= previous
                || frame.version != ProtocolVersion::CURRENT
            {
                continue;
            }
            previous = frame.sequence;
            match frame.control {
                ControlMessage::Checkpoint { .. } => {
                    break ControlStopReason::Checkpoint;
                }
                ControlMessage::Application { name, payload } if name == CONTROL_NAME => {
                    match serde_json::from_slice(&payload) {
                        Ok(command) => match send.try_send(command) {
                            Ok(()) => {}
                            Err(mpsc::TrySendError::Full(_)) => break ControlStopReason::QueueFull,
                            Err(mpsc::TrySendError::Disconnected(_)) => {
                                break ControlStopReason::QueueDisconnected
                            }
                        },
                        Err(_) => {
                            break ControlStopReason::MalformedNamedControl;
                        }
                    }
                }
                _ => {
                    break ControlStopReason::UnknownControl;
                }
            }
        };
        if stop_control_with(
            &observed,
            Duration::from_millis(allowance),
            |deadline| {
                std::thread::spawn(move || {
                    std::thread::sleep(
                        deadline.saturating_duration_since(std::time::Instant::now()),
                    );
                    // SAFETY: _exit has no preconditions. Skipping destructors and flushes is
                    // intended: a worker still running past its checkpoint allowance is killed.
                    unsafe {
                        libc::_exit(76);
                    }
                });
            },
            |deadline| {
                record_control_stop(&control_state_root, attempt, previous, reason, deadline)
            },
        )
        .is_err()
        {
            eprintln!(
                "control stop diagnostic could not be persisted; numerical admission stopped"
            );
        }
    });
    let mut sequence = 0;
    report(
        &mut allocation.stream,
        attempt,
        &mut sequence,
        WorkerReport::Ready,
    )?;
    if !matches!(input.operation, Operation::Resident) {
        let result = execute(&allocation, path, sha, Arc::clone(&stop))?;
        return report(
            &mut allocation.stream,
            attempt,
            &mut sequence,
            result_progress(&result)?,
        );
    }
    while !stop.load(Ordering::Acquire) {
        let command = match receive.recv_timeout(Duration::from_millis(100)) {
            Ok(command) => command,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(_) => break,
        };
        match execute(
            &allocation,
            &command.manifest,
            &command.manifest_sha256,
            Arc::clone(&stop),
        ) {
            Ok(result) => report(
                &mut allocation.stream,
                attempt,
                &mut sequence,
                result_progress(&result)?,
            )?,
            Err(error) => {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                report(
                    &mut allocation.stream,
                    attempt,
                    &mut sequence,
                    WorkerReport::Failure {
                        class: jain_smartcluster::FailureClass::Application,
                        message: format!("{error:#}").chars().take(4000).collect(),
                    },
                )?;
                return Err(error);
            }
        }
    }
    write_json(
        &allocation.policy.state_root.join(format!(
            "recovery-{}-{}.json",
            attempt.job_id, attempt.fence
        )),
        &json!({"schema":"jope.smartcluster_recovery.v1","attempt":attempt,"recovery_state_fsynced":true,"numerical_completion_requires_verified_task_receipt":true}),
    )?;
    Ok(())
}
fn result_progress(result: &Value) -> Result<WorkerReport> {
    Ok(WorkerReport::Progress(StructuredProgress {
        phase: "durable_subtask_result".into(),
        completed: 1,
        total: Some(1),
        message: None,
        fields: BTreeMap::from([("receipt".into(), serde_json::to_string(result)?)]),
    }))
}
