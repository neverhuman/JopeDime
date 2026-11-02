use super::*;

impl<S: Supervisor> Coordinator<S> {
    /// Dispatch is permitted only after a fresh admitted preflight, before admission, while
    /// no cancellation, drain or restart is pending, and inside the group window: the
    /// original deadline and every participant's grant, not only this rank's. A group that
    /// can no longer run as a whole starts nothing more.
    pub(super) fn require_dispatchable(&mut self, rank: u32, now: u64) -> Result<Replay> {
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
}
