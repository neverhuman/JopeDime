use super::*;

impl Discovery {
    /// Commit initial run state and its public retry receipt in one transaction.
    /// Workers may be dispatched only after this returns successfully.
    pub(crate) fn start_command(
        store: Arc<Store>,
        plan: Plan,
        journal: Arc<EventLog>,
        identity: crate::idempotency::RequestIdentity,
        input_roots: Vec<BlobRef>,
        warnings: Vec<jope_api::Warning>,
    ) -> Result<Self> {
        let start = Some(StartCommand {
            journal,
            identity,
            input_roots,
            warnings,
            resume: false,
        });
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, start)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, start)
        }
    }
    /// Resume an owned paused run and atomically retain its public retry receipt.
    pub(crate) fn continue_command(
        store: Arc<Store>,
        plan: Plan,
        journal: Arc<EventLog>,
        identity: crate::idempotency::RequestIdentity,
    ) -> Result<Self> {
        let command = Some(StartCommand {
            journal,
            identity,
            input_roots: vec![],
            warnings: vec![],
            resume: true,
        });
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, command)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, command)
        }
    }
    /// The caller exclusively owns this run and supplies a fresh worker pool on
    /// reopen. Store::open verifies committed blobs and marks interrupted runs.
    pub fn open(store: Arc<Store>, plan: Plan) -> Result<Self> {
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, None)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, None)
        }
    }

    /// Requires the opaque, reviewed capture context; no production resolver is registered.
    #[cfg(feature = "research")]
    pub fn open_research(
        store: Arc<Store>,
        plan: Plan,
        context: Arc<CaptureContext>,
    ) -> Result<Self> {
        context.validate(&plan)?;
        Self::open_inner(store, plan, Some(context), None)
    }

    pub(super) fn open_inner(
        store: Arc<Store>,
        plan: Plan,
        #[cfg(feature = "research")] capture_context: Option<Arc<CaptureContext>>,
        start: Option<StartCommand>,
    ) -> Result<Self> {
        plan.validate()?;
        let expected_version = CHECKPOINT_VERSION;
        #[cfg(feature = "research")]
        let expected_version = if capture_context.is_some() {
            capture::CHECKPOINT_VERSION
        } else {
            expected_version
        };
        let log = start
            .as_ref()
            .map(|s| Arc::clone(&s.journal))
            .unwrap_or_else(|| Arc::new(EventLog::new(Arc::clone(&store))));
        let old = log.snapshot(&plan.run_id)?;
        if let Some(command) = &start {
            ensure!(
                if command.resume {
                    old.as_ref()
                        .is_some_and(|snapshot| snapshot.state == RunState::Paused)
                } else {
                    old.is_none()
                },
                "command requires a new run or an explicitly continued paused run"
            );
        }
        let (mut checkpoint, mut snapshot) = if let Some(snapshot) = &old {
            let record = store
                .get(&RecordKey::Run(plan.run_id.clone()))?
                .context("run missing")?;
            let saved: Checkpoint = serde_json::from_value(record.data["checkpoint"].clone())?;
            ensure!(
                saved.version == expected_version && saved.plan == plan,
                "resume plan identity mismatch"
            );
            #[cfg(feature = "research")]
            match (&saved.capture, &capture_context) {
                (Some(state), Some(context)) => state.check_binding(&store, context)?,
                (None, None) => {}
                _ => bail!("capture mode cannot change on reopen"),
            }
            if let Some(c) = &saved.champion {
                store.verify_blob(&c.model)?;
            }
            let recovery = record
                .run
                .as_ref()
                .context("run recovery metadata missing")?;
            ensure!(
                saved.completed.is_some()
                    || saved.active_since_ms.is_some()
                    || (snapshot.state == RunState::Paused && !recovery.interrupted),
                "interrupted wall budget lacks an active-span anchor; resume refused"
            );
            (saved, snapshot.clone())
        } else {
            let checkpoint = Checkpoint {
                version: expected_version,
                input_roots: start
                    .as_ref()
                    .map(|s| s.input_roots.clone())
                    .unwrap_or_default(),
                #[cfg(feature = "research")]
                capture: capture_context
                    .as_deref()
                    .map(|context| CaptureState::new(&store, context))
                    .transpose()?,
                plan: plan.clone(),
                elites: ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize),
                programs: BTreeMap::new(),
                memo: BTreeSet::new(),
                champion: None,
                pending: vec![],
                discarded: vec![],
                next_round: 0,
                admitted: 0,
                stalled: 0,
                epoch: 0,
                nondeterminism: vec![],
                elapsed_ms: 0,
                active_since_ms: None,
                completed: None,
            };
            let snapshot = RunSnapshot {
                run_id: plan.run_id.clone(),
                revision: DecimalU64(1),
                state: RunState::Running,
                phase: RunPhase::Seeding,
                pending: None,
                effort: plan.effort,
                budget_used: BudgetUsed {
                    evaluations: DecimalU64(0),
                    wall_seconds: "0".into(),
                },
                budget_total: BudgetTotal {
                    evaluations: DecimalU64(plan.evaluations),
                    wall_seconds: DecimalU64(plan.wall_seconds),
                },
                best: None,
                event_watermark: DecimalU64(0),
                warnings: start
                    .as_ref()
                    .map(|s| s.warnings.clone())
                    .unwrap_or_default(),
                completion_reason: None,
                evaluation_receipt: None,
            };
            (checkpoint, snapshot)
        };
        #[cfg(feature = "research")]
        if let Some(state) = checkpoint.capture.as_mut() {
            // A reserved request with no durable observation may have executed.
            if state.unknown() {
                state.tainted = true;
            }
        }
        #[cfg(feature = "research")]
        if let Some(capture) = &checkpoint.capture {
            snapshot.budget_total.evaluations = DecimalU64(capture.config.max_attempts);
        }
        let now = wall_clock_ms()?;
        checkpoint.elapsed_ms = checkpoint.elapsed_at(now)?;
        if checkpoint.completed.is_none()
            && checkpoint.elapsed_ms >= plan.wall_seconds.saturating_mul(1000)
        {
            checkpoint.completed = Some(CompletionReason::BudgetTime);
            snapshot.phase = RunPhase::Finalizing;
        }
        if checkpoint.completed.is_some() {
            discard_pending(&mut checkpoint, None, &[], &[], &mut vec![])?;
        }
        checkpoint.active_since_ms = checkpoint.completed.is_none().then_some(now);
        snapshot.budget_used.wall_seconds = format!("{:.3}", checkpoint.elapsed_ms as f64 / 1000.0);
        checkpoint.epoch = checkpoint
            .epoch
            .checked_add(1)
            .context("attempt epochs exhausted")?;
        if old.is_some() {
            snapshot.revision.0 = snapshot
                .revision
                .0
                .checked_add(1)
                .context("revision exhausted")?;
        }
        snapshot.state = RunState::Running;
        let blobs = checkpoint_blobs(&checkpoint)?;
        let initial_response = serde_json::to_value(&snapshot)?;
        // Initialization/recovery is a control transaction, never a synthetic search event.
        let snapshot = log
            .commit(
                RunUpdate {
                    expected_revision: old.map(|s| s.revision.0),
                    snapshot,
                    checkpoint: serde_json::to_value(&checkpoint)?,
                    blobs,
                },
                vec![],
                |tx| {
                    if let Some(start) = &start {
                        tx.record_new_command(
                            &start.identity,
                            crate::idempotency::CommandOutcome::json(
                                if start.resume { 200 } else { 201 },
                                initial_response,
                            )?,
                        )?;
                    }
                    Ok(())
                },
            )?
            .snapshot;
        Ok(Self {
            #[cfg(feature = "research")]
            capture_context,
            store,
            log,
            snapshot,
            checkpoint,
            attempt: 0,
            poisoned: false,
        })
    }

    pub fn champion(&self) -> Option<&Champion> {
        self.checkpoint.champion.as_ref()
    }
    pub fn completed(&self) -> Option<CompletionReason> {
        self.checkpoint.completed
    }
    /// True when an attempted durable commit may have succeeded despite an
    /// error return. Callers must retain an unknown outcome in this case.
    pub fn outcome_unknown(&self) -> bool {
        self.poisoned
    }
    pub fn admitted(&self) -> u64 {
        self.checkpoint.admitted
    }
    pub fn budget_bound_by(&self) -> Option<&'static str> {
        match self.completed() {
            Some(CompletionReason::BudgetTime) => Some("time"),
            Some(CompletionReason::BudgetEvaluations) => Some("evaluations"),
            _ => None,
        }
    }
    pub fn snapshot(&self) -> &RunSnapshot {
        &self.snapshot
    }

    pub fn nondeterminism(&self) -> Nondeterminism {
        Nondeterminism {
            count: DecimalU64(self.checkpoint.nondeterminism.len() as u64),
            experiment_ids: self.checkpoint.nondeterminism.clone(),
        }
    }

    /// Canonical event projection under owner ruling rulings_t1900 R2 (§5.6).
    /// Operational envelopes and receipts retain run identity and elapsed usage.
    pub fn projection(&self) -> Result<Value> {
        let mut after = 0;
        let mut values = vec![];
        while after < self.snapshot.event_watermark.0 {
            let page = self
                .log
                .page(&self.checkpoint.plan.run_id, after, 256, &[])?
                .context("run missing")?;
            after = page.next_after.0;
            for event in page.events {
                let mut value = serde_json::to_value(event)?;
                let envelope = value.as_object_mut().context("event is not an object")?;
                envelope.remove("run_id");
                envelope.remove("time");
                if let Some(warnings) = value
                    .get_mut("body")
                    .and_then(|body| body.get_mut("warnings"))
                    .and_then(Value::as_array_mut)
                {
                    warnings.retain(|warning| warning.get("host") != Some(&Value::Bool(true)));
                }
                if let Some(budget) = value
                    .get_mut("body")
                    .and_then(|body| body.get_mut("budget_used"))
                    .and_then(Value::as_object_mut)
                {
                    budget.remove("wall_seconds");
                }
                if value
                    .get("body")
                    .and_then(|b| b.get("host"))
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    values.push(value);
                }
            }
        }
        Ok(Value::Array(values))
    }

    /// Pause only at a committed boundary. Pending siblings remain in the
    /// checkpoint; no uncommitted result is installed or publicly visible.
    pub fn pause(&mut self) -> Result<()> {
        let mut next = self.checkpoint.clone();
        next.elapsed_ms = next.elapsed_at(wall_clock_ms()?)?;
        next.active_since_ms = None;
        self.commit_control(next, RunState::Paused)
    }
}
