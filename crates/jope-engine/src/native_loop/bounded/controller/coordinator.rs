use super::*;

enum Retained {
    Absent,
    Valid(Box<VerifiedResult>),
    Invalid(String),
}
/// Proof that an admission was journaled before any participant was told of it.
pub struct PersistedAdmission {
    admission: GroupAdmission,
    admission_sha256: String,
}
impl PersistedAdmission {
    pub fn admission_sha256(&self) -> &str {
        &self.admission_sha256
    }
}

/// The single writer of one planned run attempt.
pub struct Coordinator<S: Supervisor> {
    dir: PathBuf,
    contract: BoundedRunContract,
    plan_sha256: String,
    records: Vec<(JournalRecord, String)>,
    supervisor: S,
    needs_reconcile: bool,
    _lock: fs::File,
}
impl<S: Supervisor> Coordinator<S> {
    /// Opens the run bound to the reviewed plan digest. A non-terminal journal that already
    /// dispatched work is a coordinator restart: nothing further may be dispatched until
    /// [`Coordinator::reconcile`] has settled every owned rank.
    pub fn open(dir: &Path, plan_sha256: &str, supervisor: S, now: u64) -> Result<Self> {
        existing_dir(dir)?;
        let (contract, actual) = load_plan(dir)?;
        ensure!(
            actual == plan_sha256,
            "coordinator binds only the reviewed plan artifact"
        );
        let lock = lock_file(dir, false)?;
        ensure!(try_lock(&lock), "another coordinator owns this run");
        let records = load_journal(dir)?;
        let s = replay(&contract, plan_sha256, &records)?;
        ensure!(!records.is_empty(), "planned run has no journal");
        let mut c = Self {
            dir: dir.into(),
            contract,
            plan_sha256: plan_sha256.into(),
            records,
            supervisor,
            needs_reconcile: false,
            _lock: lock,
        };
        if s.terminal.is_none() && !s.intents.is_empty() {
            let me = crate::eval_registry::ProcessIdentity::current()?;
            c.append(
                BoundedControllerEvent::CoordinatorRestarted {
                    boot_id: me.boot_id,
                    pid: me.pid,
                },
                now,
            )?;
            c.needs_reconcile = true;
        }
        Ok(c)
    }
    pub fn contract(&self) -> &BoundedRunContract {
        &self.contract
    }
    pub fn plan_sha256(&self) -> &str {
        &self.plan_sha256
    }
    pub fn needs_reconcile(&self) -> bool {
        self.needs_reconcile
    }
    pub fn supervisor(&mut self) -> &mut S {
        &mut self.supervisor
    }
    pub fn journal(&self) -> Vec<JournalRecord> {
        self.records.iter().map(|(r, _)| r.clone()).collect()
    }
    fn append(&mut self, event: BoundedControllerEvent, now: u64) -> Result<()> {
        append(&self.dir, &mut self.records, &self.plan_sha256, event, now)?;
        replay(&self.contract, &self.plan_sha256, &self.records)?;
        Ok(())
    }
    fn state(&self) -> Result<Replay> {
        replay(&self.contract, &self.plan_sha256, &self.records)
    }
    fn window_closed(&self, now: u64) -> Option<DrainCause> {
        journal_window_closed(&self.contract, &self.records, now)
    }
    pub fn status(&self) -> Result<StatusReport> {
        build_status(
            &self.dir,
            &self.contract,
            &self.plan_sha256,
            &self.records,
            &self.state()?,
            true,
        )
    }
    /// Journal a preflight of an untrusted census record. A refusal is journaled too
    /// (for audit) and returned; it is not terminal while the window remains.
    pub fn preflight(&mut self, census_bytes: &[u8], now: u64) -> Result<PreflightReceipt> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && !self.needs_reconcile,
            "preflight is only permitted before admission"
        );
        self.contract.validate_at(now).map_err(|e| anyhow!("{e}"))?;
        let census = HostCensus::from_bytes(census_bytes)?;
        let census_sha256 = hash(census_bytes);
        match census::preflight(
            &self.contract,
            &self.plan_sha256,
            &census,
            &census_sha256,
            now,
        ) {
            Ok(receipt) => {
                self.append(
                    BoundedControllerEvent::PreflightAdmitted {
                        receipt: receipt.clone(),
                    },
                    now,
                )?;
                Ok(receipt)
            }
            Err(refusal) => {
                self.append(
                    BoundedControllerEvent::PreflightRefused {
                        census_sha256,
                        reasons: refusal.reasons.clone(),
                    },
                    now,
                )?;
                Err(refusal.into())
            }
        }
    }
    /// Open a rendezvous; requires a preflight receipt still fresh under the frozen
    /// telemetry policy.
    pub fn open_rendezvous(
        &self,
        coordinator: &SigningKey,
        coordinator_boot_id: &str,
        nonce: &[u8; 16],
        now: u64,
    ) -> Result<Rendezvous> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && !self.needs_reconcile,
            "rendezvous is only opened before admission"
        );
        let receipt = s
            .preflight
            .context("rendezvous requires an admitted preflight")?;
        ensure!(
            now.saturating_sub(receipt.checked_unix_ms) <= self.contract.telemetry.max_age_ms,
            "preflight receipt is stale; preflight a fresh census"
        );
        Rendezvous::open(
            &self.contract,
            &self.plan_sha256,
            coordinator,
            coordinator_boot_id,
            nonce,
            now,
        )
    }
    /// Persist the full-group admission *before* any participant learns of it. Requires
    /// the persisted dispatch and launch identity (ownership) of every rank.
    pub fn record_admission(
        &mut self,
        admission: GroupAdmission,
        now: u64,
    ) -> Result<PersistedAdmission> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none()
                && s.admission.is_none()
                && s.drain.is_none()
                && !self.needs_reconcile,
            "admission is recorded once, while the run is live"
        );
        ensure!(
            s.launched.len() == self.contract.world_size(),
            "ownership of every rank must be persisted before admission"
        );
        if let Some(cause) = self.window_closed(now) {
            bail!("no admission is journaled after the group window closed ({cause:?})");
        }
        rendezvous::verify_group_admission(&self.contract, &self.plan_sha256, &admission)?;
        let admission_sha256 = rendezvous::admission_sha256(&admission)?;
        self.append(
            BoundedControllerEvent::GroupAdmitted {
                admission_sha256: admission_sha256.clone(),
                admission: admission.clone(),
            },
            now,
        )?;
        Ok(PersistedAdmission {
            admission,
            admission_sha256,
        })
    }
    /// Acknowledge an admission to participants; only a persisted admission can be sent.
    /// Every participant must receive it; a failed delivery drains the whole group.
    pub fn deliver_admission(
        &mut self,
        persisted: &PersistedAdmission,
        connected: &mut [Connected],
        now: u64,
    ) -> Result<()> {
        self.deliver_admission_with_clock(persisted, connected, || now)
    }
    /// Refresh the original execution window around each potentially blocking delivery.
    pub fn deliver_admission_with_clock(
        &mut self,
        persisted: &PersistedAdmission,
        connected: &mut [Connected],
        clock: impl Fn() -> u64,
    ) -> Result<()> {
        let now = clock();
        let s = self.state()?;
        ensure!(
            s.admission.as_ref().map(|(_, h)| h.as_str())
                == Some(persisted.admission_sha256.as_str())
                && !s.delivered
                && s.drain.is_none(),
            "admission is not the journaled, undelivered admission"
        );
        // Delivery is when ranks may start work: after the window closed, drain instead.
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause.clone(), now)?;
            bail!("admission not delivered: the group window closed ({cause:?}); group drained");
        }
        let ranks: BTreeSet<u32> = connected.iter().map(|c| c.global_rank).collect();
        if !ranks
            .iter()
            .copied()
            .eq(0..self.contract.world_size() as u32)
            || ranks.len() != connected.len()
        {
            let missing = (0..self.contract.world_size() as u32)
                .find(|r| !ranks.contains(r))
                .unwrap_or(0);
            self.drain(
                DrainCause::ControlLost {
                    global_rank: missing,
                },
                now,
            )?;
            bail!("admission delivery requires one control stream per planned rank");
        }
        let bytes = serde_json::to_vec(&persisted.admission)?;
        let limit = self.contract.rendezvous.max_message_bytes;
        for c in connected.iter_mut() {
            let now = clock();
            if let Some(cause) = self.window_closed(now) {
                self.drain(cause.clone(), now)?;
                bail!("admission delivery window closed ({cause:?}); group drained");
            }
            // Each delivery ends by the plan's I/O timeout and never after the group window
            // closes, however slowly the rank reads.
            let allowance = self
                .contract
                .rendezvous
                .heartbeat_timeout_ms
                .min(group_window_end(&self.contract).saturating_sub(now))
                .max(1);
            let deadline = Instant::now() + Duration::from_millis(allowance);
            if let Err(e) = rendezvous::write_frame_by(&mut *c.stream, &bytes, limit, deadline) {
                let rank = c.global_rank;
                self.drain(DrainCause::ControlLost { global_rank: rank }, clock())?;
                return Err(e.context(format!(
                    "admission delivery to rank {rank} failed; group drained"
                )));
            }
        }
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause.clone(), now)?;
            bail!("admission delivery window closed ({cause:?}); group drained");
        }
        self.append(
            BoundedControllerEvent::AdmissionDelivered {
                admission_sha256: persisted.admission_sha256.clone(),
            },
            now,
        )
    }
    /// A refused rendezvous ends the attempt: launched rank work is stopped before it
    /// could start any optimizer attempt, and nothing is committed.
    pub fn record_refusal(&mut self, refusal: &GroupRefusal, now: u64) -> Result<()> {
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.admission.is_none() && s.drain.is_none(),
            "refusal after admission"
        );
        self.append(
            BoundedControllerEvent::GroupRefused {
                refusal: refusal.clone(),
            },
            now,
        )?;
        for rank in 0..self.contract.world_size() as u32 {
            let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                (Some(h), _) => Some(h.clone()),
                (None, Some(id)) => self.supervisor.find(id)?,
                (None, None) => None,
            };
            let stopped = match handle {
                Some(h) => {
                    let last = self.supervisor.terminate(&h)?;
                    ensure!(last != Liveness::Running, "rank {rank} survived refusal");
                    format!("; unadmitted rank work stopped ({last:?})")
                }
                None => "; no rank work was started".into(),
            };
            self.append(
                BoundedControllerEvent::BoundedControllerDisposition {
                    global_rank: rank,
                    disposition: BoundedControllerDisposition::Refused,
                    detail: format!("{}{stopped}", serde_json::to_string(&refusal.cause)?),
                },
                now,
            )?;
        }
        self.append(
            BoundedControllerEvent::Terminal {
                outcome: BoundedControllerOutcome::Refused,
                committed_updates_total: self.contract.counters.committed_updates_at_start,
            },
            now,
        )
    }
    /// Dispatch is permitted only after a fresh admitted preflight, before admission, while
    /// no cancellation, drain or restart is pending, and inside the group window: the
    /// original deadline and every participant's grant, not only this rank's. A group that
    /// can no longer run as a whole starts nothing more.
    fn require_dispatchable(&mut self, rank: u32, now: u64) -> Result<Replay> {
        ensure!(
            !self.needs_reconcile,
            "coordinator restarted: reconcile before dispatch"
        );
        ensure!(
            !self.cancel_pending(now)?,
            "cancellation requested; nothing is dispatched"
        );
        let s = self.state()?;
        ensure!(
            s.terminal.is_none() && s.drain.is_none() && !s.refused && s.admission.is_none(),
            "run is admitted, refused, draining or terminal; no further dispatch"
        );
        let receipt = s
            .preflight
            .as_ref()
            .context("dispatch requires an admitted preflight")?;
        ensure!(
            now.saturating_sub(receipt.checked_unix_ms) <= self.contract.telemetry.max_age_ms,
            "preflight receipt is stale; preflight a fresh census before dispatch"
        );
        self.contract
            .participant(rank)
            .context("rank outside the plan")?;
        if let Some(cause) = self.window_closed(now) {
            bail!("rank {rank} is not dispatched: the group window closed ({cause:?})");
        }
        Ok(s)
    }
    /// Persist the dispatch intent (idempotent) before any launch side effect.
    pub fn record_dispatch_intent(&mut self, rank: u32, now: u64) -> Result<String> {
        let s = self.require_dispatchable(rank, now)?;
        if let Some(id) = s.intents.get(&rank) {
            return Ok(id.clone());
        }
        let id = rendezvous::dispatch_id(&self.plan_sha256, self.contract.attempt, rank)?;
        self.append(
            BoundedControllerEvent::DispatchIntent {
                global_rank: rank,
                dispatch_id: id.clone(),
            },
            now,
        )?;
        Ok(id)
    }
    /// Launch a rank whose intent is persisted. Never relaunches: journaled or already
    /// supervised work under the same dispatch id is recorded, not restarted.
    pub fn launch(&mut self, rank: u32, now: u64) -> Result<SupervisorHandle> {
        self.launch_with_clock(rank, || now)
    }
    pub fn launch_with_clock(
        &mut self,
        rank: u32,
        clock: impl Fn() -> u64,
    ) -> Result<SupervisorHandle> {
        let s = self.require_dispatchable(rank, clock())?;
        let id = s
            .intents
            .get(&rank)
            .cloned()
            .context("dispatch intent must be journaled before launch")?;
        if let Some(h) = s.launched.get(&rank) {
            return Ok(h.clone());
        }
        let intent = self
            .records
            .iter()
            .find_map(|(r, _)| match &r.event {
                BoundedControllerEvent::DispatchIntent {
                    global_rank,
                    dispatch_id,
                } if *global_rank == rank && *dispatch_id == id => Some(r.recorded_unix_ms),
                _ => None,
            })
            .context("dispatch intent timestamp absent")?;
        let found = self.supervisor.find(&id)?;
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            bail!("execution window closed while finding rank {rank}");
        }
        self.require_dispatchable(rank, now)?;
        let handle = match found {
            Some(h) => h,
            None => self.supervisor.launch(&RankLaunch {
                contract: &self.contract,
                plan_sha256: &self.plan_sha256,
                global_rank: rank,
                dispatch_id: &id,
                dispatch_intent_unix_ms: intent,
            })?,
        };
        let now = clock();
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            bail!("execution window closed while launching rank {rank}");
        }
        self.append(
            BoundedControllerEvent::Launched {
                global_rank: rank,
                dispatch_id: id,
                handle: handle.clone(),
            },
            now,
        )?;
        Ok(handle)
    }
    pub fn dispatch_all(&mut self, now: u64) -> Result<()> {
        self.dispatch_all_with_clock(|| now)
    }
    pub fn dispatch_all_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<()> {
        for rank in 0..self.contract.world_size() as u32 {
            self.record_dispatch_intent(rank, clock())?;
            self.launch_with_clock(rank, &clock)?;
        }
        Ok(())
    }
    fn retained(&mut self, handle: &SupervisorHandle) -> Result<Retained> {
        let s = self.state()?;
        // A result can only bind an admission; before one exists nothing is a result.
        let Some((_, adm)) = s.admission.as_ref() else {
            return Ok(Retained::Absent);
        };
        Ok(match self.supervisor.retained_result(handle)? {
            None => Retained::Absent,
            Some((envelope, bytes)) => match rendezvous::verify_result(
                &self.contract,
                &self.plan_sha256,
                adm,
                &envelope,
                &bytes,
            ) {
                Ok(v) => Retained::Valid(Box::new(v)),
                Err(e) => Retained::Invalid(format!("{e:#}")),
            },
        })
    }
    fn acknowledge(
        &mut self,
        v: &VerifiedResult,
        rank: u32,
        clock: &impl Fn() -> u64,
    ) -> Result<bool> {
        let now = clock();
        let s = self.state()?;
        ensure!(
            v.body().global_rank == rank && s.intents.contains_key(&rank),
            "result does not belong to dispatched rank {rank}"
        );
        if let Some(previous) = s.acked.get(&rank) {
            ensure!(
                *previous == v.body().result_sha256,
                "conflicting second result for rank {rank}"
            );
            return Ok(true);
        }
        // Adopting a retained result is authority-bearing: the same gate as poll and
        // recovery, whichever path reaches it.
        if let Some(cause) = self.window_closed(now) {
            self.drain(cause, now)?;
            return Ok(false);
        }
        let p = self.contract.participant(rank).context("rank")?.clone();
        publish_immutable(
            &self.dir.join(RESULTS),
            &p.result_destination.path,
            v.bytes(),
        )?;
        publish_immutable(
            &self.dir.join(ENVELOPES),
            Path::new(&format!("rank-{rank}.json")),
            v.envelope(),
        )?;
        let now = clock();
        let cause = if self.cancel_pending(now)? {
            Some(DrainCause::Cancelled)
        } else {
            self.window_closed(now)
        };
        if let Some(cause) = cause {
            self.drain(cause, now)?;
            return Ok(false);
        }
        self.append(
            BoundedControllerEvent::ResultAcknowledged {
                global_rank: rank,
                delivery_id: v.body().delivery_id,
                result_sha256: v.body().result_sha256.clone(),
                envelope_sha256: hash(v.envelope()),
                updates: v.body().updates,
            },
            now,
        )?;
        Ok(true)
    }
    /// Journal the single group commit (inside the group window only), or finish a commit
    /// that is already journaled (at any time, exactly once, never drained).
    fn commit(&mut self, clock: &impl Fn() -> u64) -> Result<()> {
        let s = self.state()?;
        let now = clock();
        if s.committed.is_none() {
            ensure!(
                s.drain.is_none() && s.acked.len() == self.contract.world_size(),
                "group commit requires every authenticated result"
            );
            if let Some(cause) = self.window_closed(now) {
                self.drain(cause, now)?;
                return Ok(());
            }
            self.append(
                BoundedControllerEvent::GroupCommitted {
                    committed_updates_total: self.contract.counters.committed_updates_at_start
                        + self.contract.counters.planned_updates,
                    results: s.acked.clone().into_iter().collect(),
                },
                now,
            )?;
        }
        let s = self.state()?;
        let total = s.committed.context("commit")?;
        for rank in 0..self.contract.world_size() as u32 {
            if !s.dispositions.contains_key(&rank) {
                self.append(
                    BoundedControllerEvent::BoundedControllerDisposition {
                        global_rank: rank,
                        disposition: BoundedControllerDisposition::Committed,
                        detail: "authenticated result committed with the whole group".into(),
                    },
                    now,
                )?;
            }
        }
        if self.state()?.terminal.is_none() {
            self.append(
                BoundedControllerEvent::Terminal {
                    outcome: BoundedControllerOutcome::Committed,
                    committed_updates_total: total,
                },
                now,
            )?;
        }
        Ok(())
    }
    /// Drain the whole synchronous group. Idempotent and resumable; the first recorded
    /// cause governs. Every rank ends with exactly one disposition and no supervised work
    /// survives.
    pub fn drain(&mut self, cause: DrainCause, now: u64) -> Result<StatusReport> {
        let s = self.state()?;
        if s.terminal.is_some() {
            return self.status();
        }
        ensure!(s.committed.is_none(), "a committed group is never drained");
        ensure!(
            !s.intents.is_empty(),
            "nothing was dispatched; there is no group to drain"
        );
        if s.drain.is_none() {
            self.append(BoundedControllerEvent::DrainStarted { cause }, now)?;
        }
        let cause = self.state()?.drain.context("drain cause")?;
        let failing = match &cause {
            DrainCause::RankFailed { global_rank, .. } => Some(*global_rank),
            _ => None,
        };
        for rank in 0..self.contract.world_size() as u32 {
            let s = self.state()?;
            if s.dispositions.contains_key(&rank) {
                continue;
            }
            let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                (Some(h), _) => Some(h.clone()),
                (None, Some(id)) => self.supervisor.find(id)?,
                (None, None) => None,
            };
            let (disposition, detail) = match handle {
                None => (
                    BoundedControllerDisposition::NeverStarted,
                    if s.intents.contains_key(&rank) {
                        "dispatch intent persisted; supervisor holds no started work".to_string()
                    } else {
                        "never dispatched".to_string()
                    },
                ),
                Some(h) => {
                    let last = self.supervisor.terminate(&h)?;
                    ensure!(last != Liveness::Running, "rank {rank} survived the drain");
                    let started_unrecorded = !s.launched.contains_key(&rank);
                    if failing == Some(rank) {
                        (
                            BoundedControllerDisposition::Failed,
                            format!("{cause:?}; final {last:?}"),
                        )
                    } else if s.acked.contains_key(&rank) {
                        (
                            BoundedControllerDisposition::CompletedUncommitted,
                            "authenticated result acknowledged; group drained before commit".into(),
                        )
                    } else if let Retained::Valid(v) = self.retained(&h)? {
                        (
                            BoundedControllerDisposition::CompletedUnacknowledged,
                            format!(
                                "authenticated result {} retained, never acknowledged; not committed",
                                v.body().result_sha256
                            ),
                        )
                    } else {
                        (
                            BoundedControllerDisposition::Drained,
                            format!(
                                "stopped by whole-group drain; final {last:?}{}",
                                if started_unrecorded {
                                    "; launch was never journaled"
                                } else {
                                    ""
                                }
                            ),
                        )
                    }
                }
            };
            self.append(
                BoundedControllerEvent::BoundedControllerDisposition {
                    global_rank: rank,
                    disposition,
                    detail,
                },
                now,
            )?;
        }
        self.append(
            BoundedControllerEvent::Terminal {
                outcome: if cause == DrainCause::Cancelled {
                    BoundedControllerOutcome::Cancelled
                } else {
                    BoundedControllerOutcome::Drained
                },
                committed_updates_total: self.contract.counters.committed_updates_at_start,
            },
            now,
        )?;
        self.status()
    }
    /// Report a control channel failure observed by the transport (heartbeat timeout).
    pub fn control_lost(&mut self, rank: u32, now: u64) -> Result<StatusReport> {
        self.drain(DrainCause::ControlLost { global_rank: rank }, now)
    }
    fn cancel_pending(&mut self, now: u64) -> Result<bool> {
        let s = self.state()?;
        if s.cancel {
            return Ok(true);
        }
        let path = self.dir.join(CANCEL);
        if fs::symlink_metadata(&path).is_err() {
            return Ok(false);
        }
        let bytes = read_bounded(&path, MAX_RECORD_BYTES)?;
        let request: CancelRequest = serde_json::from_slice(&bytes)?;
        ensure!(
            request.plan_sha256 == self.plan_sha256,
            "cancellation bound to another plan"
        );
        self.append(
            BoundedControllerEvent::CancelRequested {
                request_sha256: hash(&bytes),
            },
            now,
        )?;
        Ok(true)
    }
    /// One supervision step: cancellation, deadline, grant expiry, rank exits, result
    /// acknowledgement and — once every rank's result is authenticated — the single commit.
    pub fn poll(&mut self, now: u64) -> Result<StatusReport> {
        self.poll_with_clock(|| now)
    }
    /// Live callers refresh time after blocking supervisor operations. The fixed-time
    /// wrapper remains available for deterministic journal replay fixtures.
    pub fn poll_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<StatusReport> {
        let now = clock();
        ensure!(
            !self.needs_reconcile,
            "coordinator restarted: reconcile before polling"
        );
        let s = self.state()?;
        if s.terminal.is_some() {
            return self.status();
        }
        if s.intents.is_empty() {
            if self.cancel_pending(now)? {
                finish_cancel_before_dispatch(
                    &self.dir,
                    &self.contract,
                    &self.plan_sha256,
                    &mut self.records,
                    now,
                )?;
            }
            return self.status();
        }
        if let Some(cause) = s.drain.clone() {
            return self.drain(cause, now);
        }
        if self.cancel_pending(now)? {
            return self.drain(DrainCause::Cancelled, now);
        }
        if let Some(cause) = self.window_closed(now) {
            return self.drain(cause, now);
        }
        for rank in 0..self.contract.world_size() as u32 {
            let s = self.state()?;
            if s.acked.contains_key(&rank) {
                continue;
            }
            let Some(handle) = s.launched.get(&rank).cloned() else {
                continue;
            };
            let observed = self.supervisor.probe(&handle)?;
            let now = clock();
            if let Some(cause) = self.window_closed(now) {
                return self.drain(cause, now);
            }
            match observed {
                Liveness::Running => {}
                Liveness::Exited { code: Some(0), .. } if !s.delivered => {
                    return self.drain(
                        DrainCause::RankFailed {
                            global_rank: rank,
                            detail: "exited before the full-group admission was delivered".into(),
                        },
                        now,
                    )
                }
                Liveness::Exited { code: Some(0), .. } => {
                    let retained = self.retained(&handle)?;
                    let now = clock();
                    if self.cancel_pending(now)? {
                        return self.drain(DrainCause::Cancelled, now);
                    }
                    if let Some(cause) = self.window_closed(now) {
                        return self.drain(cause, now);
                    }
                    match retained {
                        Retained::Valid(v) => {
                            if !self.acknowledge(&v, rank, &clock)? {
                                return self.status();
                            }
                        }
                        Retained::Absent => {
                            return self.drain(
                                DrainCause::RankFailed {
                                    global_rank: rank,
                                    detail: "exited without a retained result".into(),
                                },
                                now,
                            )
                        }
                        Retained::Invalid(detail) => {
                            return self.drain(
                                DrainCause::RankFailed {
                                    global_rank: rank,
                                    detail: format!("result refused: {detail}"),
                                },
                                now,
                            )
                        }
                    }
                }
                Liveness::Exited {
                    code: Some(EXIT_LOST_CONTROL),
                    ..
                } => return self.drain(DrainCause::ControlLost { global_rank: rank }, now),
                other => {
                    return self.drain(
                        DrainCause::RankFailed {
                            global_rank: rank,
                            detail: format!("{other:?}"),
                        },
                        now,
                    )
                }
            }
        }
        let now = clock();
        if self.cancel_pending(now)? {
            return self.drain(DrainCause::Cancelled, now);
        }
        if let Some(cause) = self.window_closed(now) {
            return self.drain(cause, now);
        }
        let s = self.state()?;
        if s.delivered && s.acked.len() == self.contract.world_size() {
            self.commit(&clock)?;
        }
        self.status()
    }
    /// Settle every owned rank after a coordinator restart.
    ///
    /// * A commit already journaled before the crash is finished (dispositions and the
    ///   terminal record), exactly once and at any time; it is never drained.
    /// * A drain already journaled is resumed with its original cause.
    /// * Otherwise a *new* recovery commit is possible only through the same gates as
    ///   [`Coordinator::poll`]: no pending cancellation, and the group window (original
    ///   deadline and every participant grant) still open. If either fails, the group is
    ///   drained with that cause before any retained result is adopted.
    /// * Inside the window, complete, authenticated, unacknowledged results for the
    ///   *whole* group are acknowledged and committed once; anything less drains, and each
    ///   rank is marked never-started, completed-unacknowledged or drained.
    ///
    /// A restart therefore never widens the window `poll` enforces (review of #111).
    pub fn reconcile(&mut self, now: u64) -> Result<StatusReport> {
        self.reconcile_with_clock(|| now)
    }
    pub fn reconcile_with_clock(&mut self, clock: impl Fn() -> u64) -> Result<StatusReport> {
        let now = clock();
        let s = self.state()?;
        if s.terminal.is_none() && s.intents.is_empty() {
            if self.cancel_pending(now)? {
                finish_cancel_before_dispatch(
                    &self.dir,
                    &self.contract,
                    &self.plan_sha256,
                    &mut self.records,
                    now,
                )?;
            }
        } else if s.terminal.is_none() {
            if s.committed.is_some() {
                self.commit(&clock)?;
            } else if let Some(cause) = s.drain.clone() {
                self.drain(cause, now)?;
            } else if self.cancel_pending(now)? {
                self.drain(DrainCause::Cancelled, now)?;
            } else if let Some(cause) = self.window_closed(now) {
                // Poll's gate, applied before any retained result is adopted.
                self.drain(cause, now)?;
            } else {
                // Unacknowledged completion is committed only when the *whole* group
                // completed with authenticated results; anything less drains.
                let mut complete = s.delivered;
                let mut found = Vec::new();
                for rank in 0..self.contract.world_size() as u32 {
                    if s.acked.contains_key(&rank) {
                        continue;
                    }
                    let handle = match (s.launched.get(&rank), s.intents.get(&rank)) {
                        (Some(h), _) => Some(h.clone()),
                        (None, Some(id)) => self.supervisor.find(id)?,
                        (None, None) => None,
                    };
                    let Some(h) = handle else {
                        complete = false;
                        continue;
                    };
                    match self.supervisor.probe(&h)? {
                        Liveness::Exited { code: Some(0), .. } => match self.retained(&h)? {
                            Retained::Valid(v) => found.push((rank, v)),
                            _ => complete = false,
                        },
                        _ => complete = false,
                    }
                }
                let now = clock();
                if self.cancel_pending(now)? {
                    self.drain(DrainCause::Cancelled, now)?;
                } else if let Some(cause) = self.window_closed(now) {
                    self.drain(cause, now)?;
                } else if complete {
                    for (rank, v) in &found {
                        if !self.acknowledge(v, *rank, &clock)? {
                            self.needs_reconcile = false;
                            return self.status();
                        }
                    }
                    let now = clock();
                    if self.cancel_pending(now)? {
                        self.drain(DrainCause::Cancelled, now)?;
                    } else if let Some(cause) = self.window_closed(now) {
                        self.drain(cause, now)?;
                    } else {
                        self.commit(&clock)?;
                    }
                } else {
                    self.drain(DrainCause::CoordinatorRestart, now)?;
                }
            }
        }
        self.needs_reconcile = false;
        self.status()
    }
}
