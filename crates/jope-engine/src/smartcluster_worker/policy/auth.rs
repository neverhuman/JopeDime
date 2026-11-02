use super::*;

pub(super) fn read_bounded(path: &Path, expected: Option<&str>) -> Result<Vec<u8>> {
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    ensure!(file.metadata()?.is_file(), "expected regular manifest");
    let mut bytes = Vec::new();
    file.take(FRAME_LIMIT as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() <= FRAME_LIMIT, "manifest exceeds 1 MiB");
    if let Some(expected) = expected {
        ensure!(hash(&bytes) == expected, "immutable manifest changed");
    }
    Ok(bytes)
}

pub(super) fn load_policy() -> Result<WorkerPolicy> {
    let path = Path::new(POLICY_PATH);
    for ancestor in path.ancestors() {
        let meta = fs::symlink_metadata(ancestor)?;
        ensure!(
            !meta.file_type().is_symlink() && meta.uid() == 0 && meta.mode() & 0o022 == 0,
            "worker policy path is not root-controlled"
        );
    }
    let policy: WorkerPolicy = serde_json::from_slice(&read_bounded(path, None)?)?;
    for path in [
        &policy.root,
        &policy.state_root,
        &policy.daemon_socket,
        &policy.cgroup_root,
        &policy.scratch_root,
    ] {
        ensure!(path.is_absolute(), "worker policy paths must be absolute");
    }
    ensure!(
        policy.root.canonicalize()? == policy.root,
        "worker root must be canonical"
    );
    ensure!(
        file_hash(Path::new("/proc/self/exe"))? == policy.worker_sha256,
        "loaded worker binary is not qualified"
    );
    verify_loaded_runtime(&policy)?;
    Ok(policy)
}

pub fn runtime_observation() -> Result<Value> {
    crate::runtime_custody::loaded_runtime_descriptor(&crate::eval_registry::ProcessIdentity::read(
        std::process::id(),
    )?)
}
pub(super) fn verify_loaded_runtime(policy: &WorkerPolicy) -> Result<()> {
    let observed = runtime_observation()?;
    let digest = observed["runtime_sha256"]
        .as_str()
        .context("loaded runtime digest absent")?;
    ensure!(
        policy.allowed_runtime_sha256.contains(digest),
        "loaded runtime is not in qualified policy"
    );
    Ok(())
}

pub(super) struct Allocation {
    pub(super) policy: WorkerPolicy,
    pub(super) attempt: FencedAttempt,
    pub(super) stream: UnixStream,
    pub(super) bootstrap: TaskManifest,
    pub(super) daemon_identity: crate::eval_registry::ProcessIdentity,
}

pub(super) fn current_job(
    policy: &WorkerPolicy,
    id: jain_smartcluster::JobId,
    expected: &crate::eval_registry::ProcessIdentity,
) -> Result<JobRecord> {
    match crate::eval_registry::native_scq::authenticated_exchange(
        &policy.daemon_socket,
        policy.daemon_uid,
        &policy.daemon_sha256,
        Some(expected),
        jain_smartcluster::Request::Get(jain_smartcluster::GetRequest { job_id: id }),
    )? {
        jain_smartcluster::Response::Job(job) => {
            let job = job.context("native allocation absent")?;
            let attempt = job.active_attempt.context("native active attempt absent")?;
            let leader = crate::eval_registry::native_scq::active_process(
                &policy.daemon_socket,
                policy.daemon_uid,
                &policy.daemon_sha256,
                Some(expected),
                attempt,
            )?;
            ensure!(
                leader == crate::eval_registry::ProcessIdentity::current()?,
                "worker is not its original scheduler leader"
            );
            Ok(job)
        }
        _ => anyhow::bail!("native allocation response differs"),
    }
}

pub(super) fn retain_control_descriptor(descriptor: i32) -> Result<UnixStream> {
    // SAFETY: F_DUPFD_CLOEXEC takes an integer minimum and touches no memory; an invalid
    // `descriptor` only yields -1 (EBADF), checked below.
    let duplicate = unsafe { libc::fcntl(descriptor, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(duplicate >= 0, "cannot retain control descriptor");
    ensure!(
        // SAFETY: F_SETFD takes an integer flag and touches no memory; an invalid
        // `descriptor` only yields -1, which is checked.
        unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) } == 0,
        "cannot protect inherited control descriptor"
    );
    // SAFETY: F_DUPFD_CLOEXEC just returned this fresh, nonnegative duplicate that nothing else
    // owns; UnixStream becomes its sole owner and closes it once. `descriptor` stays separate.
    Ok(unsafe { UnixStream::from_raw_fd(duplicate) })
}

pub(super) fn authenticate(policy: WorkerPolicy, path: &Path, sha: &str) -> Result<Allocation> {
    let descriptor: i32 = std::env::var("SCQ_CONTROL_FD")
        .context("inherited native control socket required")?
        .parse()?;
    ensure!(descriptor >= 3, "invalid control descriptor");
    let stream = retain_control_descriptor(descriptor)?;
    std::env::remove_var("SCQ_CONTROL_FD");
    let peer = resident::peer_credentials(&stream)?;
    let birth = resident::process_identity(peer.pid as u32)?;
    ensure!(
        peer.uid == policy.daemon_uid
            && file_hash(Path::new(&format!("/proc/{}/exe", peer.pid)))? == policy.daemon_sha256,
        "unqualified daemon control peer"
    );
    let daemon_identity = crate::eval_registry::ProcessIdentity::read(peer.pid as u32)?;
    let id = std::env::var("SCQ_JOB_ID")?.parse()?;
    let job = current_job(&policy, id, &daemon_identity)?;
    let attempt = job
        .active_attempt
        .context("allocation has no active fenced attempt")?;
    ensure!(
        attempt.number.to_string() == std::env::var("SCQ_ATTEMPT_NUMBER")?
            && attempt.fence.to_string() == std::env::var("SCQ_ATTEMPT_FENCE")?,
        "native attempt fence changed"
    );
    ensure!(
        matches!(job.state, JobState::Starting | JobState::Running),
        "native allocation is not live"
    );
    let expected = policy.cgroup_root.join(format!(
        "scq-{}-attempt-{}-fence-{}",
        attempt.job_id, attempt.number, attempt.fence
    ));
    let own = fs::read_to_string("/proc/self/cgroup")?;
    let actual = own
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("unified cgroup membership absent")?;
    ensure!(
        Path::new("/sys/fs/cgroup").join(actual.trim_start_matches('/')) == expected,
        "worker is outside its native fenced cgroup"
    );
    let spec = jain_smartcluster::jope::job_spec(
        path,
        sha,
        job.spec.tenant.clone(),
        job.spec.project.clone(),
        Timestamp::from(resident::now_ms()?),
    )?;
    ensure!(job.spec == spec, "manifest does not match native JobSpec");
    ensure!(
        resident::process_identity(peer.pid as u32)? == birth,
        "daemon peer changed during authentication"
    );
    let bootstrap = serde_json::from_slice(&read_bounded(path, Some(sha))?)?;
    Ok(Allocation {
        policy,
        attempt,
        stream,
        bootstrap,
        daemon_identity,
    })
}

pub(super) fn load_task(
    policy: &WorkerPolicy,
    path: &Path,
    sha: &str,
) -> Result<(TaskManifest, InputManifest)> {
    verify_task_inputs(&policy.root, path, sha)
}

pub(crate) fn verify_task_inputs(
    root: &Path,
    path: &Path,
    sha: &str,
) -> Result<(TaskManifest, InputManifest)> {
    let task: TaskManifest = serde_json::from_slice(&read_bounded(path, Some(sha))?)?;
    task.validate_at(Timestamp::from(resident::now_ms()?))?;
    let input_path = path.with_file_name(format!(
        "{}.input.json",
        path.file_stem()
            .context("manifest stem absent")?
            .to_string_lossy()
    ));
    let mut input: InputManifest = serde_json::from_slice(&read_bounded(
        &input_path,
        Some(&task.input_manifest_sha256),
    )?)?;
    ensure!(input.root == root, "input root/schema changed");
    input
        .validate_version()
        .context("input root/schema changed")?;
    let dependencies: BTreeMap<String, String> = if let Some(closure) = &input.closure {
        ensure!(
            input.input_artifacts.is_empty(),
            "closure and inline artifacts cannot be mixed"
        );
        BTreeMap::from([(
            closure.path.to_string_lossy().into_owned(),
            closure.sha256.clone(),
        )])
    } else {
        input
            .input_artifacts
            .iter()
            .map(|(path, sha)| (path.to_string_lossy().into_owned(), sha.clone()))
            .collect()
    };
    ensure!(
        dependencies == task.dependencies,
        "input closure differs from task dependencies"
    );
    if let Some(closure) = &input.closure {
        input.input_artifacts = read_closure(root, closure)?;
    }
    verify_training_closure(root, &input)?;
    let _verified_input_scope = crate::native_input_cache::enter(root, &input.input_artifacts)?;
    for (path, sha) in &input.input_artifacts {
        ensure!(
            file_hash(&resolve(root, path)?)? == *sha,
            "dependency changed: {}",
            path.display()
        );
    }
    relative(&input.output_receipt)?;
    validate_operation(&task, &input)?;
    match &input.operation {
        Operation::Train {
            corpus, initialize, ..
        }
        | Operation::Proposal {
            corpus,
            checkpoint: initialize,
            ..
        } => {
            ensure!(
                task.corpus_sha256.as_deref()
                    == Some(file_hash(&resolve(root, corpus)?.join("index.json"))?.as_str()),
                "task corpus identity differs from numerical input"
            );
            ensure!(
                task.parent_checkpoint_sha256.as_deref()
                    == Some(file_hash(&resolve(root, initialize)?.join("state.json"))?.as_str()),
                "task parent identity differs from numerical input"
            );
        }
        _ => {}
    }
    verify_required_inputs(&input)?;
    Ok((task, input))
}
