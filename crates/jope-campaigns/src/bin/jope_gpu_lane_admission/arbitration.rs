fn capacity_available(state: &GpuLaneAdmissionV1, capacity: &NodeCapacity) -> bool {
    capacity.up
        && capacity.gpus_free >= state.resource_reservation.gpus as u64
        && capacity.gpu_companion_core_reserve
            >= state.resource_reservation.companion_cpu_cores as u64
        && capacity.free_core_budget >= state.resource_reservation.companion_cpu_cores as i64
}

fn load_arbitration_evidence(root: &Path) -> Result<GpuArbitrationEvidence> {
    let path = root.join("ops/micro-wave-supply-v2.json");
    let value =
        read_json(&path).with_context(|| format!("read GPU arbitration {}", path.display()))?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeMicroWaveSupplyStatus.v2"),
        "GPU arbitration supply status is off schema"
    );
    let generated_at_ms = value
        .get("generated_at_ms")
        .and_then(Value::as_u64)
        .map(u128::from)
        .context("GPU arbitration supply status lacks a timestamp")?;
    ensure!(
        now_ms().saturating_sub(generated_at_ms) <= ARBITRATION_STATUS_MAX_AGE_MS,
        "GPU arbitration supply status is stale"
    );
    ensure!(
        value.get("performance_fresh").and_then(Value::as_bool) == Some(true),
        "GPU arbitration supply performance is stale"
    );
    let evidence = GpuArbitrationEvidence {
        candidate_evaluation_work: value
            .get("candidate_evaluation_work")
            .and_then(Value::as_u64)
            .context("GPU arbitration lacks candidate work")?,
        candidate_evaluation_work_target: value
            .get("candidate_evaluation_work_target")
            .and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .context("GPU arbitration lacks a candidate-work target")?,
        protected_complete_waves: value
            .get("complete_reserve_waves")
            .and_then(Value::as_u64)
            .context("GPU arbitration lacks protected-wave evidence")?,
    };
    Ok(evidence)
}

fn select_gpu_roles(
    states: &BTreeMap<GpuLaneRole, GpuLaneAdmissionV1>,
    capacity_ready: &[GpuLaneRole],
    evidence: GpuArbitrationEvidence,
) -> Vec<GpuLaneRole> {
    let pending = |role| {
        states.get(&role).is_some_and(|state| {
            !matches!(
                state.stage,
                GpuAdmissionStage::Submitted
                    | GpuAdmissionStage::Complete
                    | GpuAdmissionStage::DependencyFailed
            )
        })
    };
    let ready = |role| capacity_ready.contains(&role);
    let order = if evidence.runway_low() {
        [
            GpuLaneRole::CohortProposer,
            GpuLaneRole::AlternatingTrainer,
            GpuLaneRole::BreadthEncoder,
        ]
    } else {
        [
            GpuLaneRole::AlternatingTrainer,
            GpuLaneRole::CohortProposer,
            GpuLaneRole::BreadthEncoder,
        ]
    };
    order
        .into_iter()
        .filter(|role| pending(*role) && ready(*role))
        .collect()
}

fn load_active_states(root: &Path) -> Result<BTreeMap<GpuLaneRole, GpuLaneAdmissionV1>> {
    let mut output = BTreeMap::new();
    for role in [
        GpuLaneRole::CohortProposer,
        GpuLaneRole::AlternatingTrainer,
        GpuLaneRole::BreadthEncoder,
    ] {
        let request = request_path(root, role);
        let state = state_path(root, role);
        let selected = if request.is_file() {
            request
        } else if state.is_file() {
            state.clone()
        } else {
            continue;
        };
        let mut value: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&selected)?)?;
        value.validate()?;
        if matches!(
            value.stage,
            GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
        ) && !request_path(root, role).is_file()
        {
            continue;
        }
        if state.is_file() {
            let persisted: GpuLaneAdmissionV1 = serde_json::from_value(read_json(&state)?)?;
            persisted.validate()?;
            if persisted.immutable_job_identity == value.immutable_job_identity {
                value = persisted;
            } else if !matches!(
                persisted.stage,
                GpuAdmissionStage::Complete | GpuAdmissionStage::DependencyFailed
            ) {
                bail!("active GPU role state conflicts with a new request");
            }
        }
        output.insert(role, value);
    }
    Ok(output)
}

fn recover_inflight(
    args: &Args,
    sentinel: &mut PauseSentinel,
    jobs: &[SchedulerJob],
) -> Result<()> {
    if let Some(id) = sentinel.hold_inflight_id.take() {
        if jobs.iter().any(|job| job.id == id && job.state == "held")
            && !sentinel.held_evaluator_ids.contains(&id)
        {
            sentinel.held_evaluator_ids.push(id);
            sentinel.held_evaluator_ids.sort();
        }
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    if let Some(id) = sentinel.release_inflight_id.take() {
        if !jobs.iter().any(|job| job.id == id && job.state == "held")
            && !sentinel.released_evaluator_ids.contains(&id)
        {
            sentinel.released_evaluator_ids.push(id);
            sentinel.released_evaluator_ids.sort();
        }
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    Ok(())
}

fn release_exact_evaluators(args: &Args, sentinel: &mut PauseSentinel) -> Result<()> {
    for id in sentinel.unreleased() {
        sentinel.release_inflight_id = Some(id.clone());
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
        let output = Command::new(&args.scq).args(["release", &id]).output()?;
        if !output.status.success() {
            let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;
            if jobs.iter().any(|job| job.id == id && job.state == "held") {
                bail!(
                    "failed to release exact evaluator {id}: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
        }
        sentinel.released_evaluator_ids.push(id);
        sentinel.released_evaluator_ids.sort();
        sentinel.release_inflight_id = None;
        sentinel.updated_at_ms = now_ms();
        save_sentinel(&args.root, sentinel)?;
    }
    sentinel.validate()
}
