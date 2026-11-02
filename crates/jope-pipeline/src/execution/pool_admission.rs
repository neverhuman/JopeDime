use super::*;

impl ExecutionPool {
    /// Restore the original policy and complete resource history under the
    /// existing execution journal lock. This never opens a new execution window.
    pub fn resource_ledger_v1(
        &mut self,
        policy: crate::resource_ledger::Policy,
    ) -> Result<crate::resource_ledger::ResourceLedger> {
        let ledger = crate::resource_ledger::ResourceLedger::new(policy.clone())?;
        immutable(&self.directory.join("resource-policy-v1.json"), &policy)?;
        let path = self.directory.join("resource-events-v1.json");
        if path.exists() {
            crate::resource_ledger::ResourceLedger::restore(
                policy,
                &read_json::<Vec<crate::resource_ledger::Event>>(&path)?,
            )
        } else {
            Ok(ledger)
        }
    }

    /// Persist upstream stage transitions in the same bounded experiment's
    /// journal. Validation replays the actual history before appending the event.
    pub fn persist_resource_event_v1(
        &mut self,
        event: &crate::resource_ledger::Event,
    ) -> Result<()> {
        let policy = read_json(&self.directory.join("resource-policy-v1.json"))?;
        let path = self.directory.join("resource-events-v1.json");
        let mut events: Vec<crate::resource_ledger::Event> = if path.exists() {
            read_json(&path)?
        } else {
            Vec::new()
        };
        events.push(event.clone());
        crate::resource_ledger::ResourceLedger::restore(policy, &events)?;
        atomic_json(&path, &events)
    }

    /// Opt-in accounting boundary. The caller admits the original window before
    /// encoding/compilation and supplies separately accounted controller stages.
    /// This method reserves a bounded batch before input hashing or child release,
    /// using this pool's existing journal lock, supervisor and descendant cleanup.
    /// No new scheduler or host qualification is introduced.
    pub fn run_accounted_v1(
        &mut self,
        jobs: Vec<WorkerJob>,
        cancellation: &AtomicBool,
        ledger: &mut crate::resource_ledger::ResourceLedger,
        requests: BTreeMap<String, crate::resource_ledger::ReservationRequest>,
    ) -> Result<Vec<WorkerOutcome>> {
        use crate::resource_ledger::{Disposition, Event, Unit};
        ensure!(
            !jobs.is_empty() && requests.len() == jobs.len(),
            "accounted batch requires exact nonempty request census"
        );
        let window = self
            .window
            .as_ref()
            .context("admit original window before resource accounting")?;
        let policy = ledger.policy();
        ensure!(
            policy.experiment_sha256 == self.experiment.identity()?
                && policy.opened_unix_us
                    == window
                        .started_unix_ms
                        .checked_mul(1000)
                        .context("window overflow")?
                && policy.deadline_unix_us
                    == window
                        .deadline_unix_ms
                        .checked_mul(1000)
                        .context("window overflow")?
                && policy.terminal_headroom_us
                    >= self.experiment.resources.reporting_seconds * 1_000_000,
            "resource policy changed original window/reporting headroom"
        );
        let memory = fit_memory_allocation(&self.experiment)?;
        ensure!(
            policy.cap.get(Unit::PeakMemoryBytes) <= memory.total_bytes
                && policy.terminal_reserve.get(Unit::PeakMemoryBytes)
                    >= memory.controller_reserve_bytes,
            "resource policy must preserve fixed controller/fit memory policy"
        );
        immutable(&self.directory.join("resource-policy-v1.json"), policy)?;
        let journal = self.directory.join("resource-events-v1.json");
        let mut events: Vec<Event> = if journal.exists() {
            read_json(&journal)?
        } else {
            Vec::new()
        };
        ensure!(
            crate::resource_ledger::ResourceLedger::restore(policy.clone(), &events)? == *ledger,
            "resource ledger does not match durable original history"
        );
        let mut persist = |event: &Event| -> Result<()> {
            events.push(event.clone());
            if let Err(error) = atomic_json(&journal, &events) {
                events.pop();
                return Err(error);
            }
            Ok(())
        };
        let now = now_ms()?
            .checked_mul(1000)
            .context("resource clock overflow")?;
        let mut joins = BTreeMap::new();
        for job in &jobs {
            let id = job.identity()?;
            let request = requests.get(&id).context("missing job reservation")?;
            ensure!(
                request.ids.request_id == job.request.identity()?
                    && request.requested.get(Unit::PeakMemoryBytes) == memory.per_fit_bytes
                    && request.stage == crate::resource_ledger::Stage::Fit
                    && request.requested.get(Unit::CpuCoreMicros)
                        == request.requested.get(Unit::AttemptWallMicros)
                    && request.requested.get(Unit::AttemptWallMicros)
                        == self.experiment.resources.per_fit_seconds * 1_000_000
                    && !request.requested.0.contains_key(&Unit::GpuDeviceMicros),
                "reservation does not match fixed one-CPU fit boundary"
            );
            let reservation = ledger.reserve(request.clone(), now, &mut persist)?;
            ensure!(
                joins.insert(id, reservation).is_none(),
                "duplicate accounted job"
            );
        }
        // A controller error leaves durable unresolved reservations, fully held.
        // Recovery must reconcile the existing started/finished child journals;
        // it cannot infer zero or open a fresh original window.
        let outcomes = self.run(jobs, cancellation)?;
        for outcome in &outcomes {
            let reservation = &joins[&outcome.job_sha256];
            if ledger.cost(&reservation.request.ids.attempt_id).is_some() {
                continue;
            }
            let ended = now_ms()?
                .checked_mul(1000)
                .context("resource clock overflow")?;
            let cost = outcome.resource_cost_v1(reservation, ended)?;
            if cost.disposition != Disposition::ProvenNotStarted
                && ledger.started_at(&cost.ids.attempt_id).is_none()
            {
                // The original child journal is the durable pre-release start
                // authority. This is its after-completion physical-cost join.
                ledger.start(
                    &cost.ids.attempt_id,
                    reservation.reserved_unix_us,
                    &mut persist,
                )?;
            }
            ledger.finish(cost, &mut persist)?;
        }
        Ok(outcomes)
    }

    pub fn open(experiment: &Experiment, directory: &Path, supervisor: Supervisor) -> Result<Self> {
        experiment.validate()?;
        ensure!(
            directory.is_absolute() && !directory.starts_with("/tmp"),
            "journal requires absolute durable path outside /tmp"
        );
        fs::create_dir_all(directory)?;
        let directory = fs::canonicalize(directory)?;
        ensure!(
            !directory.starts_with("/tmp"),
            "journal resolves under /tmp"
        );
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(directory.join("execution.lock"))?;
        let journal_lock = Flock::lock(lock, FlockArg::LockExclusiveNonblock)
            .map_err(|(_, error)| anyhow::anyhow!("journal already owned: {error}"))?;
        immutable(&directory.join("experiment.json"), experiment)?;
        for child in [
            "started", "finished", "arms", "jobs", "requests", "logs", "cleanup",
        ] {
            fs::create_dir_all(directory.join(child))?;
        }
        let window = if directory.join("window.json").exists() {
            let window: ExecutionWindow = read_json(&directory.join("window.json"))?;
            window.validate(experiment)?;
            Some(window)
        } else {
            None
        };
        Ok(Self {
            experiment: experiment.clone(),
            directory,
            supervisor,
            _journal_lock: journal_lock,
            host_lock: None,
            window,
        })
    }
    pub fn window(&self) -> Option<&ExecutionWindow> {
        self.window.as_ref()
    }

    /// Acquire and retain exclusive host allocation and verify the delegated
    /// supervisor before package hashing, data preparation or training preflight.
    /// This does not persist/start/reset either experiment or arm clocks. Holding
    /// the reservation forbids overlapping exclusive full-CPU CI. Drop the pool
    /// to release it; later admit/begin_arm performs the checks again.
    pub fn reserve(&mut self) -> Result<()> {
        let now = now_ms()?;
        let window = self.window.clone().unwrap_or(ExecutionWindow {
            experiment_sha256: self.experiment.identity()?,
            started_unix_ms: now,
            deadline_unix_ms: now
                .checked_add(self.experiment.resources.total_seconds * 1000)
                .context("deadline overflow")?,
        });
        ensure!(
            window.may_admit(&self.experiment, now)?,
            "original execution window exhausted"
        );
        self.acquire_host()?;
        self.supervisor
            .verify(&self.experiment, window.deadline_unix_ms)
    }

    /// Starts the single experiment window after the caller's dependency
    /// inventory. Call before any proposal work, not after expensive encoding.
    pub fn admit(&mut self) -> Result<&ExecutionWindow> {
        self.reserve()?;
        if self.window.is_none() {
            let now = now_ms()?;
            let window = ExecutionWindow {
                experiment_sha256: self.experiment.identity()?,
                started_unix_ms: now,
                deadline_unix_ms: now
                    .checked_add(self.experiment.resources.total_seconds * 1000)
                    .context("deadline overflow")?,
            };
            immutable(&self.directory.join("window.json"), &window)?;
            self.window = Some(window);
        }
        self.window.as_ref().context("execution window absent")
    }
    /// Returns the persisted absolute cutoff. Encoding/proposals and fits all
    /// share this arm clock. A second call never grants a fresh allowance.
    pub fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        ensure!(
            crate::experiment::sha256(&arm.id_sha256) && arm.seconds > 0 && arm.seconds <= 120,
            "invalid arm allowance"
        );
        let reporting_ms = self.experiment.resources.reporting_seconds * 1000;
        let execution_deadline = self.admit()?.deadline_unix_ms - reporting_ms;
        Ok(self.arm_deadline(arm)?.min(execution_deadline))
    }

    /// Dataset round-robin admission; each slot is a process, never an in-process fit.
    /// An interrupted started job is terminal Unknown accounting, not an implicit retry.
    pub fn run(
        &mut self,
        jobs: Vec<WorkerJob>,
        cancellation: &AtomicBool,
    ) -> Result<Vec<WorkerOutcome>> {
        // Reservation-only consumers (category training) share the supervisor
        // and original window, while every actual fit job keeps this strict cap.
        ensure!(
            self.experiment.resources.per_fit_seconds <= 30,
            "fit allowance exceeds30 seconds"
        );
        // A changed derived policy cannot silently resume the same journal.
        // Reservation-only train never reaches this fit-allocation boundary.
        let memory = fit_memory_allocation(&self.experiment)?;
        immutable(&self.directory.join("fit-memory-policy.json"), &memory)?;
        let order = jobs
            .iter()
            .enumerate()
            .map(|(i, j)| Ok((j.identity()?, i)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let active_cancel = Arc::new(AtomicBool::new(cancellation.load(Ordering::Acquire)));
        let finished = AtomicBool::new(false);
        struct Done<'a>(&'a AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        thread::scope(|scope| {
            let _done = Done(&finished);
            let mirrored = active_cancel.clone();
            let completion = &finished;
            scope.spawn(move || {
                while !completion.load(Ordering::Acquire) {
                    if cancellation.load(Ordering::Acquire) {
                        mirrored.store(true, Ordering::Release);
                    }
                    thread::sleep(Duration::from_millis(POLL_MS));
                }
            });
            let mut results = self.run_inner(jobs, active_cancel)?;
            results.sort_by_key(|o| order.get(&o.job_sha256).copied().unwrap_or(usize::MAX));
            Ok(results)
        })
    }
}
