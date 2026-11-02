use super::*;

impl<S: Supervisor> Coordinator<S> {
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
