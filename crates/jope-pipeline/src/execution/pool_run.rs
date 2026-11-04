use super::*;

impl ExecutionPool {
    pub(super) fn run_inner(
        &mut self,
        jobs: Vec<WorkerJob>,
        cancellation: Arc<AtomicBool>,
    ) -> Result<Vec<WorkerOutcome>> {
        let mut ids = BTreeSet::new();
        let mut requests = BTreeSet::new();
        let mut queue = FairQueue::default();
        let mut outcomes = Vec::new();
        let mut preflight = BatchPreflight::default();
        for job in jobs {
            let id = job.identity()?;
            ensure!(ids.insert(id.clone()), "duplicate job in batch");
            ensure!(
                job.request.experiment_sha256 == self.experiment.identity()?,
                "request belongs to another experiment"
            );
            let request_id = job.request.identity()?;
            let ownership = self
                .directory
                .join("requests")
                .join(format!("{request_id}.json"));
            if ownership.exists() {
                ensure!(
                    read_json::<String>(&ownership)? == id,
                    "request identity already belongs to another arm/command; derive distinct request recipe/seed binding"
                );
            }
            ensure!(
                requests.insert(request_id.clone()),
                "request repeated with a different command"
            );
            ensure!(
                job.directory == self.directory.join("jobs").join(request_id),
                "job directory must equal journal/jobs/<request identity>"
            );
            if let Some(prior) = self.cached(&job)? {
                outcomes.push(prior);
            } else if cancellation.load(Ordering::Acquire) && !self.started_path(&id).exists() {
                outcomes.push(WorkerOutcome::unstarted(
                    &job,
                    EvaluationStatus::Cancelled,
                    "cancelled before admission".into(),
                )?);
            } else if !self.started_path(&id).exists() {
                if let Err(error) = preflight.verify(&job) {
                    let outcome = WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Unavailable,
                        format!("pre-admission input/dependency unavailable: {error:#}"),
                    )?;
                    // Dependency inventory is not a scientific attempt.
                    outcomes.push(outcome);
                } else {
                    queue.push(job.request.dataset_sha256.clone(), job);
                }
            } else {
                queue.push(job.request.dataset_sha256.clone(), job);
            }
        }
        if queue.is_empty() {
            return Ok(outcomes);
        }
        let admitted = now_ms()?;
        let candidate_window = self.window.clone().unwrap_or(ExecutionWindow {
            experiment_sha256: self.experiment.identity()?,
            started_unix_ms: admitted,
            deadline_unix_ms: admitted
                .checked_add(self.experiment.resources.total_seconds * 1000)
                .context("deadline overflow")?,
        });
        let may_admit = candidate_window.may_admit(&self.experiment, admitted)?;
        let execution_deadline =
            candidate_window.deadline_unix_ms - self.experiment.resources.reporting_seconds * 1000;
        if !may_admit {
            while let Some(job) = queue.pop() {
                let id = job.identity()?;
                let outcome = if self.started_path(&id).exists() {
                    self.interrupted(&job, false)?
                } else {
                    WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original execution window exhausted".into(),
                    )?
                };
                immutable(&self.finished_path(&id), &outcome)?;
                outcomes.push(outcome);
            }
            return Ok(outcomes);
        }
        let admission = self.acquire_host().and_then(|()| {
            self.supervisor
                .verify(&self.experiment, candidate_window.deadline_unix_ms)
        });
        if let Err(error) = admission {
            while let Some(job) = queue.pop() {
                let id = job.identity()?;
                let mut outcome = WorkerOutcome::unstarted(
                    &job,
                    EvaluationStatus::Unavailable,
                    format!("execution admission unavailable: {error:#}"),
                )?;
                if self.started_path(&id).exists() {
                    outcome = self.interrupted(&job, false)?;
                    outcome.command_status = EvaluationStatus::Unavailable;
                    outcome.reason = Some(format!(
                        "original attempt unknown; current execution admission unavailable: {error:#}"
                    ));
                }
                outcomes.push(outcome);
            }
            // Admission failures are an inventory, not cached fit results or a started clock.
            atomic_json(
                &self.directory.join("admission-unavailable.json"),
                &outcomes,
            )?;
            return Ok(outcomes);
        }
        if self.window.is_none() {
            immutable(&self.directory.join("window.json"), &candidate_window)?;
            self.window = Some(candidate_window);
        }
        let memory = fit_memory_allocation(&self.experiment)?;
        let cpus = parse_cpus(&control_text(
            &self.supervisor.cgroup_root,
            "cpuset.cpus.effective",
        )?)?;
        let mut active = ActiveFits(Vec::new());
        let mut admission_blocked = self.reconcile_cleanup()?;
        let mut pending = FairQueue::default();
        while let Some(job) = queue.pop() {
            let id = job.identity()?;
            if self.started_path(&id).exists() {
                let outcome = self.interrupted(&job, true)?;
                if outcome
                    .reason
                    .as_ref()
                    .is_some_and(|r| r.contains("recovery_not_drained"))
                {
                    admission_blocked = Some("prior attempt could not be safely drained".into());
                }
                if !outcome
                    .reason
                    .as_ref()
                    .is_some_and(|r| r.contains("recovery_not_drained"))
                {
                    immutable(&self.finished_path(&id), &outcome)?;
                }
                outcomes.push(outcome);
            } else {
                pending.push(job.request.dataset_sha256.clone(), job);
            }
        }
        queue = pending;
        while !queue.is_empty() || !active.is_empty() {
            for index in (0..active.len()).rev() {
                if active[index].is_done() {
                    let running = active.swap_remove(index);
                    let started = running.started.clone();
                    let group = started.cgroup.clone();
                    let outcome = running.finish()?;
                    // Scientific terminal evidence is durable even if cleanup
                    // is delayed. The separate cleanup owner continues to block
                    // admission until the group has actually drained.
                    if self.started_path(&outcome.job_sha256).exists()
                        || outcome.command_status == EvaluationStatus::BudgetExhausted
                    {
                        immutable(&self.finished_path(&outcome.job_sha256), &outcome)?;
                    }
                    // Setup now belongs to the watcher. Preserve the same
                    // bounded cleanup used by synchronous launch failures before
                    // releasing its reserved CPU or admitting replacement work.
                    if !self.started_path(&outcome.job_sha256).exists() {
                        let recovery = self.recover_group(&started, true)?;
                        if !recovery.is_empty() {
                            admission_blocked = Some(format!("launch cleanup pending{recovery}"));
                        }
                    }
                    if !group.exists()
                        || named_counter(&group, "cgroup.events", "populated").ok() == Some(0)
                    {
                        self.clear_cleanup(&outcome.job_sha256)?;
                        // Kernel counters are now retained in the finished receipt.
                        let _ = fs::remove_dir(&group);
                    } else {
                        admission_blocked = Some(
                            "previous fit cgroup did not drain; no more fits admitted".to_owned(),
                        );
                    }
                    outcomes.push(outcome);
                }
            }
            while active.len() < self.experiment.resources.workers as usize {
                let Some(job) = queue.pop() else {
                    break;
                };
                let id = job.identity()?;
                let terminal = if let Some(reason) = &admission_blocked {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Unavailable,
                        reason.clone(),
                    )?)
                } else if self.started_path(&id).exists() {
                    Some(self.interrupted(&job, true)?)
                } else if cancellation.load(Ordering::Acquire) {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::Cancelled,
                        "cancelled before admission".into(),
                    )?)
                } else if now_ms()? >= execution_deadline {
                    Some(WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original execution window exhausted".into(),
                    )?)
                } else {
                    None
                };
                if let Some(outcome) = terminal {
                    if self.started_path(&id).exists()
                        || outcome.command_status == EvaluationStatus::BudgetExhausted
                    {
                        immutable(&self.finished_path(&id), &outcome)?;
                    }
                    outcomes.push(outcome);
                    continue;
                }
                let arm_deadline = job
                    .arm
                    .as_ref()
                    .map(|arm| self.arm_deadline(arm))
                    .transpose()?
                    .unwrap_or(execution_deadline);
                let deadline = execution_deadline.min(arm_deadline);
                if deadline <= now_ms()? {
                    let outcome = WorkerOutcome::unstarted(
                        &job,
                        EvaluationStatus::BudgetExhausted,
                        "original search arm window exhausted".into(),
                    )?;
                    immutable(&self.finished_path(&id), &outcome)?;
                    outcomes.push(outcome);
                    continue;
                }
                let used: BTreeSet<_> = active.iter().map(|a| a.cpu).collect();
                let cpu = *cpus
                    .iter()
                    .find(|cpu| !used.contains(*cpu))
                    .context("no free isolated CPU")?;
                let started = Started {
                    job,
                    job_sha256: id.clone(),
                    started_unix_ms: now_ms()?,
                    fit_deadline_unix_ms: deadline,
                    cgroup: self.supervisor.cgroup_root.join(format!("fit-{id}")),
                };
                immutable(
                    &self
                        .directory
                        .join("requests")
                        .join(format!("{}.json", started.job.request.identity()?)),
                    &id,
                )?;
                // Cleanup ownership is durable before a cgroup or process can exist.
                // It remains independent of scientific attempt/finished state.
                immutable(&self.cleanup_path(&id), &started)?;
                match Running::start(
                    started.clone(),
                    cpu,
                    memory.per_fit_bytes,
                    self.experiment.resources.per_fit_seconds,
                    self.started_path(&id),
                    self.directory.join("logs"),
                    cancellation.clone(),
                ) {
                    Ok(running) => active.push(running),
                    Err(error) => {
                        let recovery = self.recover_group(&started, true)?;
                        if recovery.is_empty() {
                            self.clear_cleanup(&id)?;
                        } else {
                            admission_blocked = Some(format!("launch cleanup pending{recovery}"));
                        }
                        let outcome = pre_gate_failure(
                            &started,
                            now_ms()?,
                            format!("fit launch unavailable: {error:#}"),
                        )?;
                        if outcome.command_status == EvaluationStatus::BudgetExhausted {
                            immutable(&self.finished_path(&id), &outcome)?;
                        }
                        outcomes.push(outcome);
                    }
                }
            }
            if !active.is_empty() {
                thread::sleep(Duration::from_millis(POLL_MS));
            }
        }
        Ok(outcomes)
    }
}
