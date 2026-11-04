use super::*;

pub(super) struct OriginalChild {
    pub(super) child: Child,
    pub(super) process: ProcessIdentity,
    pub(super) reaped: bool,
}
impl Drop for OriginalChild {
    fn drop(&mut self) {
        if !self.reaped
            && ProcessIdentity::read(self.child.id()).ok().as_ref() == Some(&self.process)
        {
            // SAFETY: the process identity was revalidated immediately above;
            // the negative pid addresses only its owned process group.
            unsafe {
                libc::kill(-(self.child.id() as i32), libc::SIGKILL);
            }
            let _ = self.child.wait();
        }
    }
}
pub(crate) fn supervise(
    state: &Arc<State>,
    fit: &VerifiedFit,
    inputs: &ServiceInputs,
    input: &File,
    claim: &Claim,
    request_deadline: u64,
    mut native: Option<&mut super::super::native_scq::NativeAdmission>,
) -> Result<ServiceResult> {
    remaining_request(request_deadline)?;
    {
        let _dispatch = state.dispatch_gate()?;
    }
    let ticket = claim.claim_token.clone();
    // Each child writes only in its own ticket directory.
    let outputs = state.config.authority_root.join("worker-output");
    let custody = outputs.join(&ticket);
    fs::create_dir(&custody).context("original worker artifact already exists")?;
    File::open(&outputs)?.sync_all()?;
    let output = custody.join("output.json");
    let fd = input.as_raw_fd();
    let parent = std::process::id();
    let confinement = sandbox::Confinement::fit(&state.config, &custody)?;
    let attestation = confinement.attestation();
    let mut command = Command::new(&state.config.worker_executable);
    confinement.configure(&mut command);
    command
        .env_remove("SCQ_CONTROL_FD")
        .args(&state.config.worker_arguments)
        .env("JOPE_SERVICE_INPUT_FD", fd.to_string())
        .env("JOPE_SERVICE_TICKET", &ticket)
        .env("JOPE_SERVICE_SOCKET", &state.config.socket)
        .env("JOPE_SERVICE_OUTPUT", &output)
        .env(
            "OMP_NUM_THREADS",
            state.config.contract.omp_threads.to_string(),
        )
        .env(
            "RAYON_NUM_THREADS",
            state.config.contract.rayon_threads.to_string(),
        )
        .env(
            "OPENBLAS_NUM_THREADS",
            state.config.contract.blas_threads.to_string(),
        )
        .env(
            "MKL_NUM_THREADS",
            state.config.contract.mkl_threads.to_string(),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    // SAFETY: the pre-exec closure captures only the verified parent pid, a
    // live descriptor, and prepared confinement state; it performs child-side
    // syscalls before exec and returns errors through the documented hook.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                || libc::getppid() as u32 != parent
                || libc::setpgid(0, 0) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            confinement.enter(fd)
        });
    }
    let child = command.spawn()?;
    let process = ProcessIdentity::read(child.id())?;
    let mut original = OriginalChild {
        child,
        process: process.clone(),
        reaped: false,
    };
    if let Some(proof) = native.as_deref_mut() {
        // SAFETY: `process` is the authenticated original child bound to this
        // launch; kill takes only its pid and a scalar signal number.
        ensure!(
            unsafe { libc::kill(process.pid as i32, libc::SIGSTOP) } == 0,
            "cannot stop original native child"
        );
        let until = Instant::now()
            + remaining_request(request_deadline)?.min(state.config.handshake_timeout);
        loop {
            ensure!(
                ProcessIdentity::read(process.pid)? == process,
                "native child birth changed before placement"
            );
            let status = fs::read_to_string(format!("/proc/{}/status", process.pid))?;
            if status.lines().any(|line| {
                line.starts_with("State:") && line.split_whitespace().nth(1) == Some("T")
            }) {
                break;
            }
            ensure!(Instant::now() < until, "native child stop gate timeout");
            std::thread::sleep(Duration::from_millis(2));
        }
        proof.attach_child(&process)?;
        // SAFETY: this resumes the same authenticated child stopped above.
        ensure!(
            unsafe { libc::kill(process.pid as i32, libc::SIGCONT) } == 0,
            "cannot release placed original child"
        );
    }
    state.worker_spawns.fetch_add(1, Ordering::SeqCst);
    let (sender, receiver) = sync_channel(1);
    lock(&state.pending)?.insert(
        ticket.clone(),
        Pending {
            process: process.clone(),
            send: sender,
        },
    );
    let launch = WorkerLaunch {
        service: state.identity.clone(),
        contract: state.config.contract.clone(),
        fit_seed: state.config.fit_seed,
    };
    {
        let mut gate = original
            .child
            .stdin
            .take()
            .context("worker launch gate missing")?;
        gate.write_all(&serde_json::to_vec(&launch)?)?;
    }
    let readiness = remaining_request(request_deadline)?.min(state.config.handshake_timeout);
    let received = receiver.recv_timeout(readiness);
    lock(&state.pending)?.remove(&ticket);
    let (mut worker, reported_key) =
        received.context("worker readiness timed out; original claim remains unavailable")?;
    ensure!(
        reported_key == fit.key(),
        "worker descriptor/runtime differs from service"
    );
    attestation.verify(process.pid)?;
    // Verify the actual original executor, not a caller-supplied digest. This
    // expensive work occurs outside the authority mutex and only on new fits.
    let actual = VerifiedFit::for_executor_process(
        process.pid,
        &inputs.program,
        crate::eval_registry::FitInputs {
            x: inputs.x.clone(),
            y: inputs.y.clone(),
            preprocessing: inputs.preprocessing.clone(),
            feature_view: inputs.feature_view.clone(),
        },
        state.config.fit_seed,
        &state.config.contract,
    )?;
    ensure!(
        actual.key() == fit.key(),
        "loaded original worker runtime differs"
    );
    #[cfg(test)]
    if let Some(pause) = &state.config.before_authorize {
        pause.reached.store(true, Ordering::SeqCst);
        let end = Instant::now() + Duration::from_secs(60);
        while !pause.release.load(Ordering::SeqCst) {
            ensure!(Instant::now() < end, "synthetic readiness pause expired");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    let (start, custody) = {
        let _dispatch = state.dispatch_gate()?;
        let remaining = remaining_request(request_deadline)?.min(state.config.fit_deadline);
        let mut authority = lock(&state.authority)?;
        let start = if let Some(proof) = native.as_deref() {
            authority.authorize_native(&actual, claim, &process, remaining, proof)?
        } else {
            authority.authorize_with_deadline(&actual, claim, &process, remaining)?
        };
        remaining_request(request_deadline)?;
        state.lease.verify()?;
        check_socket(&state.identity, &state.config.socket)?;
        let custody = if let Some(proof) = native.as_deref() {
            authority.consume_native_owned_child_start(
                &actual,
                &start,
                &original.child,
                &worker,
                proof,
            )?
        } else {
            authority.consume_owned_child_start(&actual, &start, &original.child, &worker)?
        };
        (start, custody)
    };
    let (device, inode) = custody.identity()?;
    let permission = Permission {
        native_placement: native.as_deref().and_then(|p| p.placement.clone()),
        start: start.clone(),
        custody_device: device,
        custody_inode: inode,
        custody_path: state.config.authority_root.join("custody").join(fit.key()),
        authority_instance: state.identity.authority_instance.clone(),
    };
    #[cfg(not(test))]
    let send_allowed = true;
    #[cfg(test)]
    let send_allowed = !state.config.drop_start_reply;
    let sent = send_allowed
        && (|| {
            send_json(
                &mut worker,
                &json!({"ok":true,"permission":permission}),
                Duration::from_secs(5),
            )?;
            send_fd(&worker, custody.fd(), Duration::from_secs(5))
        })()
        .is_ok();
    drop(worker);
    if !sent {
        // SAFETY: `original` still owns the verified child and its process group.
        unsafe {
            libc::kill(-(original.child.id() as i32), libc::SIGKILL);
        }
    }
    let observed = ObservedCompletion::wait_owned_group(
        &mut original.child,
        &start,
        state.config.fit_deadline,
    )?;
    original.reaped = true;
    let bytes = if fs::symlink_metadata(&output).is_ok_and(|m| {
        m.is_file() && !m.file_type().is_symlink() && m.len() <= state.config.max_descriptor_bytes
    }) {
        fs::read(&output)?
    } else {
        Vec::new()
    };
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let (outcome, model) = match value["status"].as_str() {
        Some("success")
            if serde_json::from_value::<feat_math::Model>(value["model"].clone()).is_ok() =>
        {
            (
                ExecutionOutcome::Success,
                Some(serde_json::to_vec(&value["model"])?),
            )
        }
        Some("deterministic_failure") => (ExecutionOutcome::DeterministicFailure, None),
        _ => (ExecutionOutcome::InfrastructureFailure, None),
    };
    let (start, sha) = prepare_observed_completion(
        &state.config.authority_root,
        &state.identity.authority_instance,
        &actual,
        observed,
        outcome,
        crate::eval_registry::CompletionArtifacts {
            output: &bytes,
            model: model.as_deref(),
            predictions: None,
            metrics: None,
        },
    )?;
    {
        lock(&state.authority)?.commit_original_completion(&actual, &start, &sha)?;
    }
    drop(custody);
    state.committed.fetch_add(1, Ordering::SeqCst);
    result_from_original(state, &sha, false)
}

pub(crate) fn result_from_original(
    state: &State,
    completion: &str,
    reused: bool,
) -> Result<ServiceResult> {
    let receipt = { lock(&state.authority)?.verify_committed_record(completion)? };
    let completion_path = state.config.authority_root.join("objects").join(completion);
    let model_path = receipt
        .model_sha256
        .as_ref()
        .map(|sha| state.config.authority_root.join("objects").join(sha));
    let result = ServiceResult {
        fit_key: receipt.fit_key.clone(),
        authority_instance: receipt.authority_instance.clone(),
        completion_sha256: completion.into(),
        completion_path,
        model_sha256: receipt.model_sha256.clone(),
        model_path,
        receipt,
        reused,
    };
    read_content(
        &state
            .config
            .authority_root
            .join("objects")
            .join(&result.receipt.output_sha256),
        &result.receipt.output_sha256,
    )?;
    for digest in [
        &result.receipt.predictions_sha256,
        &result.receipt.metrics_sha256,
    ]
    .into_iter()
    .flatten()
    {
        read_content(
            &state.config.authority_root.join("objects").join(digest),
            digest,
        )?;
    }
    if result.receipt.outcome == ExecutionOutcome::Success {
        result.model_bytes()?;
    }
    Ok(result)
}

/// Returned only after the service's durable start transition and custody FD.
pub struct WorkerPermit {
    permission: Permission,
    custody: OwnedFd,
}
impl WorkerPermit {
    #[cfg(test)]
    pub(super) fn custody_path(&self) -> &std::path::Path {
        &self.permission.custody_path
    }

    pub fn verify(&self) -> Result<()> {
        if let Some(p) = &self.permission.native_placement {
            super::super::native_scq::verify_placement(
                p,
                &self.permission.start.executor,
                p.target_attempt,
                &p.task_manifest_sha256,
                &p.descriptor_sha256,
                p.candidate_index,
            )?;
        }
        ensure!(
            super::super::authority::monotonic_ns()? < self.permission.start.deadline_monotonic_ns,
            "original worker start deadline expired"
        );
        ensure!(
            ProcessIdentity::current()? == self.permission.start.executor,
            "worker permit inherited by another process"
        );
        // A dup shares the custody open file description, so it reads the same inode and
        // flock(2) on it re-asserts the lock that `self.custody` holds; closing the dup
        // afterwards does not release it.
        let custody = File::from(self.custody.try_clone()?);
        let m = custody.metadata()?;
        ensure!(
            (m.dev(), m.ino())
                == (
                    self.permission.custody_device,
                    self.permission.custody_inode
                ),
            "worker custody descriptor replaced"
        );
        let path = fs::symlink_metadata(&self.permission.custody_path)?;
        ensure!(
            path.is_file()
                && !path.file_type().is_symlink()
                && (path.dev(), path.ino()) == (m.dev(), m.ino()),
            "worker custody pathname replaced"
        );
        ensure!(custody.try_lock().is_ok(), "worker exclusive custody lost");
        Ok(())
    }
}
pub fn await_start_from_environment() -> Result<(VerifiedFit, WorkerPermit, PathBuf)> {
    let fd: i32 = std::env::var("JOPE_SERVICE_INPUT_FD")?.parse()?;
    ensure!(fd >= 3, "worker sealed descriptor missing");
    let ticket = std::env::var("JOPE_SERVICE_TICKET")?;
    let socket = PathBuf::from(
        std::env::var_os("JOPE_SERVICE_SOCKET").context("worker service socket absent")?,
    );
    let output = PathBuf::from(
        std::env::var_os("JOPE_SERVICE_OUTPUT").context("worker output custody absent")?,
    );
    let mut launch = Vec::new();
    std::io::stdin()
        .take(MAX_FRAME as u64 + 1)
        .read_to_end(&mut launch)?;
    ensure!(launch.len() <= MAX_FRAME, "worker launch exceeds protocol");
    let launch: WorkerLaunch = serde_json::from_slice(&launch)?;
    ensure!(
        launch.service.process.pid == crate::host_os::parent_pid(),
        "worker original parent changed"
    );
    // SAFETY: fd is the inherited live launch descriptor; F_DUPFD_CLOEXEC takes
    // an integer minimum and returns a new descriptor or -1.
    let duplicate = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    ensure!(duplicate >= 3, "worker descriptor duplication failed");
    // SAFETY: duplicate is the fresh successful fcntl result and has no owner.
    let file = File::from(unsafe { OwnedFd::from_raw_fd(duplicate) });
    let inputs: ServiceInputs =
        serde_json::from_slice(&read_sealed(&file, 2 * 1024 * 1024 * 1024)?)?;
    let fit = VerifiedFit::for_current_executor(
        &inputs.program,
        inputs.x,
        inputs.y,
        inputs.preprocessing,
        inputs.feature_view,
        launch.fit_seed,
        &launch.contract,
    )?;
    ensure!(
        verified_executable(&launch.service.process)? == fit.identity().executable_sha256,
        "worker service executable changed"
    );
    check_socket(&launch.service, &socket)?;
    let mut stream = connect_bounded(&socket, Duration::from_secs(5))?;
    ensure!(
        authenticated_peer(&stream)? == launch.service.process,
        "worker service peer changed"
    );
    // Confine this child before it requests start permission and fits.
    sandbox::lockdown()?;
    send_json(
        &mut stream,
        &Request::Consume {
            ticket,
            fit_key: fit.key().into(),
        },
        Duration::from_secs(5),
    )?;
    let response: Value = receive_json(&mut stream, Duration::from_secs(30))?;
    ensure!(
        response["ok"] == true,
        "worker start rejected: {}",
        response["error"]
    );
    let permission: Permission = serde_json::from_value(response["permission"].clone())?;
    ensure!(
        permission.authority_instance == launch.service.authority_instance
            && permission.start.fit_key == fit.key()
            && permission.custody_path
                == launch
                    .service
                    .authority_root
                    .join("custody")
                    .join(fit.key()),
        "worker permission authority/fit mismatch"
    );
    let custody = receive_fd(&stream, Duration::from_secs(5))?;
    let permit = WorkerPermit {
        permission,
        custody,
    };
    permit.verify()?;
    Ok((fit, permit, output))
}
