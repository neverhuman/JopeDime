fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(role) = args.register_role {
        let spec = args.spec.as_deref().context("--spec is required")?;
        let job_id = args.job_id.as_deref().context("--job-id is required")?;
        let request = register_request(&args.root, role.into(), spec, job_id)?;
        println!("{}", serde_json::to_string(&request)?);
        return Ok(());
    }
    loop {
        let cycle_started = Instant::now();
        if let Err(error) = cycle(&args) {
            let failure = json!({
                "schema": "GpuLaneAdmissionError.v1",
                "generated_at_ms": now_ms(),
                "error": format!("{error:#}"),
            });
            atomic_json(
                &args.root.join("ops/gpu-lane-admission/error.v1.json"),
                &failure,
            )?;
            eprintln!("{}", serde_json::to_string(&failure)?);
            if args.once {
                return Err(error);
            }
        }
        if args.once {
            return Ok(());
        }
        let interval = Duration::from_secs(args.interval_seconds);
        if let Some(remaining) = interval.checked_sub(cycle_started.elapsed()) {
            thread::sleep(remaining);
        }
    }
}

fn request_path(root: &Path, role: GpuLaneRole) -> PathBuf {
    root.join("ops/gpu-lane-admission/requests")
        .join(format!("{}.json", role.as_str()))
}

fn state_path(root: &Path, role: GpuLaneRole) -> PathBuf {
    root.join("ops/gpu-lane-admission/state")
        .join(format!("{}.v1.json", role.as_str()))
}

fn sentinel_path(root: &Path) -> PathBuf {
    root.join("ops/gpu-lane.pause.json")
}

fn register_request(
    root: &Path,
    role: GpuLaneRole,
    spec: &Path,
    job_id: &str,
) -> Result<GpuLaneAdmissionV1> {
    ensure!(spec.is_file(), "GPU immutable spec is absent");
    let digest = Sha256Digest::parse(sha256_file(spec)?)?;
    let request =
        GpuLaneAdmissionV1::new(role, spec.to_owned(), digest, job_id.to_owned(), now_ms())?;
    let path = request_path(root, role);
    if path.is_file() {
        let existing: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&path)?)?;
        existing.validate()?;
        ensure!(
            existing.immutable_job_identity == request.immutable_job_identity
                && existing.immutable_spec_sha256 == request.immutable_spec_sha256,
            "a different immutable GPU request already owns this role"
        );
        return Ok(existing);
    }
    atomic_json(&path, &serde_json::to_value(&request)?)?;
    append_event(root, "request_registered", &request)?;
    Ok(request)
}

fn cycle(args: &Args) -> Result<()> {
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;
    let nodes = node_capacity(&scq_json(&args.scq, &["nodes"])?)?;
    let mut states = load_active_states(&args.root)?;
    if states.is_empty() {
        if sentinel_path(&args.root).is_file() {
            recover_orphan_sentinel(args, &jobs)?;
        }
        publish_status(args, &states, &nodes, false)?;
        return Ok(());
    }
    for state in states.values_mut() {
        state.recover();
        if jobs.iter().any(|job| {
            job.id == state.immutable_job_identity
                && !matches!(job.state.as_str(), "failed" | "finished" | "cancelled")
        }) {
            state.stage = GpuAdmissionStage::Submitted;
            state.recovery_disposition = RecoveryDisposition::ConfirmSubmissionThenRelease;
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
    }
    // Old deployments may have crashed inside the former evaluator-hold
    // transaction. Recover those exact IDs first; new admissions never create
    // a pause sentinel or hold evaluator work.
    if sentinel_path(&args.root).is_file() {
        recover_orphan_sentinel(args, &jobs)?;
    }
    for state in states.values_mut() {
        if state.stage == GpuAdmissionStage::Submitted {
            if state.held_evaluator_ids != state.released_evaluator_ids {
                state.released_evaluator_ids = archived_exact_release(&args.root, state)?
                    .with_context(|| {
                        format!(
                            "submitted {} admission has no exact archived release receipt",
                            state.role.as_str()
                        )
                    })?;
            }
            state.stage = GpuAdmissionStage::Complete;
            state.recovery_disposition = RecoveryDisposition::NothingToRecover;
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
            archive_request(&args.root, state)?;
        } else if state.stage == GpuAdmissionStage::DependencyFailed {
            archive_request(&args.root, state)?;
        }
    }

    let mut roles = Vec::new();
    for state in states.values_mut() {
        if matches!(
            state.stage,
            GpuAdmissionStage::Submitted
                | GpuAdmissionStage::Complete
                | GpuAdmissionStage::DependencyFailed
        ) {
            continue;
        }
        let capacity = nodes
            .get(&state.target_node)
            .with_context(|| format!("target node {} is absent", state.target_node))?;
        if capacity_available(state, capacity) {
            roles.push(state.role);
        } else {
            state.stage = GpuAdmissionStage::WaitingForCapacity;
            state.recovery_disposition = RecoveryDisposition::ResumeCapacityWait;
            state.last_capacity_backoff = Some(format!(
                "{}: up={} free_gpus={} free_core_budget={} companion_reserve={} requires 1 GPU + 4 reserved cores",
                capacity.node,
                capacity.up,
                capacity.gpus_free,
                capacity.free_core_budget,
                capacity.gpu_companion_core_reserve,
            ));
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
    }
    if roles.is_empty() {
        return publish_status(args, &states, &nodes, false);
    }

    let evidence = load_arbitration_evidence(&args.root)?;
    let selected_roles = select_gpu_roles(&states, &roles, evidence);
    if selected_roles.is_empty() {
        for role in roles {
            let state = states.get_mut(&role).expect("role came from active states");
            state.stage = GpuAdmissionStage::WaitingForCapacity;
            state.recovery_disposition = RecoveryDisposition::ResumeCapacityWait;
            state.last_capacity_backoff = Some(if evidence.runway_low() {
                "scientific runway is low; awaiting a cohort-proposer request".to_owned()
            } else {
                "no GPU role is scientifically eligible".to_owned()
            });
            state.updated_at_ms = now_ms();
            save_state(&args.root, state)?;
        }
        return publish_status(args, &states, &nodes, false);
    }

    // Roles are pinned to independent GPUs/nodes. Low runway changes order so
    // the proposer is checked first, but it never suppresses a ready trainer or
    // encoder on another available resource.
    for selected_role in selected_roles {
        let state = states
            .get_mut(&selected_role)
            .expect("selected GPU role came from active states");
        match validate_and_submit(args, state, &jobs) {
            Ok(true) => {
                state.stage = GpuAdmissionStage::Complete;
                state.recovery_disposition = RecoveryDisposition::NothingToRecover;
                state.last_capacity_backoff = None;
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
                append_event(&args.root, "gpu_spec_submitted", state)?;
                archive_request(&args.root, state)?;
            }
            Ok(false) => {
                state.stage = GpuAdmissionStage::WaitingForCapacity;
                state.recovery_disposition = RecoveryDisposition::ResumeCapacityWait;
                state.last_capacity_backoff =
                    Some("reserved companion capacity changed during verification".to_owned());
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
            }
            Err(error) => {
                state.stage = GpuAdmissionStage::DependencyFailed;
                state.dependency_failure = Some(format!("{error:#}"));
                state.recovery_disposition = RecoveryDisposition::NothingToRecover;
                state.updated_at_ms = now_ms();
                save_state(&args.root, state)?;
                append_event(&args.root, "dependency_failed", state)?;
                archive_request(&args.root, state)?;
            }
        }
    }
    publish_status(args, &states, &nodes, false)
}
