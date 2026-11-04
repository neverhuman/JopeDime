use super::*;

impl<S: Supervisor> Coordinator<S> {
    pub(super) fn retained(&mut self, handle: &SupervisorHandle) -> Result<Retained> {
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
    pub(super) fn acknowledge(
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
    pub(super) fn commit(&mut self, clock: &impl Fn() -> u64) -> Result<()> {
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
}
