use super::*;
use crate::progress::cadence::Clock;
use crate::progress::inventory::encode;

impl<'r> Driver<'r> {
    fn planned(&self, work_id: &str) -> Result<()> {
        ensure!(
            self.manifest
                .value()
                .units
                .iter()
                .any(|u| u.work_id == work_id),
            "work {work_id} is not planned in run {}; nothing was written",
            self.manifest.value().run_id
        );
        Ok(())
    }

    fn event(&mut self, event: Decoded<Event>, blobs: Vec<Vec<u8>>) -> Result<Step> {
        let mut step = Step::default();
        self.append(
            &Mapped {
                blobs,
                objects: vec![event.encoded()?],
            },
            &mut step,
        )?;
        Ok(self.finish(step))
    }

    /// Attempt 1 of a planned unit was scheduled. It opens new work, so a paused outbox refuses
    /// it (and writes nothing).
    pub fn schedule(&mut self, work_id: &str) -> Result<Step> {
        self.planned(work_id)?;
        let e = Decoded::new(Event::new(&self.manifest, work_id, 1, EventBody::Scheduled))?;
        self.event(e, Vec::new())
    }

    /// Attempt `attempt` of a planned unit was scheduled as a retry of the recorded failure of
    /// the attempt before it.
    pub fn retry(&mut self, work_id: &str, attempt: u32) -> Result<Step> {
        self.planned(work_id)?;
        ensure!(attempt >= 2, "a retry opens attempt 2 or later");
        let failure = self
            .find_event(work_id, attempt - 1, |b| {
                matches!(b, EventBody::Failed { .. })
            })?
            .with_context(|| {
                format!(
                    "attempt {} of {work_id} has no recorded failure to retry",
                    attempt - 1
                )
            })?;
        let e = Decoded::new(Event::new(
            &self.manifest,
            work_id,
            attempt,
            EventBody::Retry {
                failed_event: failure,
            },
        ))?;
        self.event(e, Vec::new())
    }

    /// The digest of the recorded event of `(work_id, attempt)` whose body `matches`.
    fn find_event(
        &self,
        work_id: &str,
        attempt: u32,
        matches: impl Fn(&EventBody) -> bool,
    ) -> Result<Option<String>> {
        for o in run_objects(&self.outbox, &self.manifest.value().run_id)? {
            if o.kind != Kind::Event {
                continue;
            }
            let e = decode_as::<Event>(&o.bytes, &o.digest)?;
            let v = e.value();
            if v.work_id == work_id && v.attempt == attempt && matches(&v.body) {
                return Ok(Some(o.digest));
            }
        }
        Ok(None)
    }

    /// A checkpoint recovery reaped the ranks of `(work_id, attempt)`: its retryable failure is
    /// recorded. The receipt must be complete; its sha256 is in the failure detail.
    pub fn recovery(&mut self, work_id: &str, attempt: u32, receipt: &Path) -> Result<Step> {
        let (_, event) = engine::recovery(&self.manifest, work_id, attempt, receipt)?;
        self.event(event, Vec::new())
    }

    /// A no-replace engine publication is the verified result of `(work_id, attempt)`.
    pub fn result(
        &mut self,
        work_id: &str,
        attempt: u32,
        engine_root: &Path,
        artifact: &Path,
        sha256: &str,
    ) -> Result<Step> {
        let (bytes, event) = engine::result(
            &self.manifest,
            work_id,
            attempt,
            engine_root,
            artifact,
            sha256,
            self.config.limits.max_bytes,
        )?;
        self.event(event, vec![bytes])
    }

    /// Link the completed `(work_id, attempt)` to the decision this host's HWM ledger recorded
    /// at `ledger_path` (`yyyy/mm/<batch>.jsonl`, relative to the ledger) event `seq`. The link
    /// is built from the ledger itself and checked against it again before anything is
    /// appended. An initialization or improvement flushes immediately.
    pub fn decision(
        &mut self,
        work_id: &str,
        attempt: u32,
        ledger_path: &str,
        seq: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.planned(work_id)?;
        let completion = self
            .find_event(work_id, attempt, |b| {
                matches!(b, EventBody::Completed { .. })
            })?
            .with_context(|| {
                format!("attempt {attempt} of {work_id} has no recorded completion")
            })?;
        let state = State::at(self.outbox.state().root());
        let ledger = Ledger::open(state.root(), "hwm");
        let path = ledger
            .batches()?
            .into_iter()
            .find(|(rel, _)| rel == ledger_path)
            .map(|(_, path)| path)
            .with_context(|| format!("{ledger_path} is not a batch of this host's HWM ledger"))?;
        let (events, seal, file_sha, _) = Ledger::read_batch(&path)?;
        let index = usize::try_from(seq)?;
        let event = parse_event(
            index,
            events
                .get(index)
                .with_context(|| format!("{ledger_path} has no event {seq}"))?,
        )?;
        let transition = event
            .transition
            .with_context(|| format!("{ledger_path}#{seq} records no transition"))?;
        let record_sha = &event.record.provenance.record_sha256;
        let blob = state
            .read_blob(record_sha)?
            .with_context(|| format!("record blob {record_sha} is missing from this host"))?;
        let decision = HwmDecision {
            schema: Kind::HwmDecision.schema().into(),
            run_id: self.manifest.value().run_id.clone(),
            work_id: work_id.into(),
            attempt,
            completion,
            record: BlobRef::of(&blob),
            identity_sha256: event.record.identity_sha256.clone(),
            ledger_batch: seal.batch,
            ledger_path: ledger_path.into(),
            ledger_batch_sha256: file_sha,
            seq,
            transition,
            qualification_sha256: event.qualification_sha256.clone(),
        };
        decision.verify_against(&ledger)?;
        let mut step = Step::default();
        self.append(
            &Mapped {
                blobs: vec![blob],
                objects: vec![encode(&decision)?],
            },
            &mut step,
        )?;
        if matches!(transition, Transition::Initialized | Transition::Improved) {
            return self.urgent(Urgent::HwmChange, clock, step);
        }
        Ok(self.finish(step))
    }
}
