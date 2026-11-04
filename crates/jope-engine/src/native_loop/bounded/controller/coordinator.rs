use super::*;

#[path = "coordinator/dispatch.rs"]
mod dispatch;
#[path = "coordinator/polling.rs"]
mod polling;
#[path = "coordinator/recovery.rs"]
mod recovery;
#[path = "coordinator/results.rs"]
mod results;

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
}
