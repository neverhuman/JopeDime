use super::*;

impl<S: Supervisor> Coordinator<S> {
    pub(super) fn cancel_pending(&mut self, now: u64) -> Result<bool> {
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
}
