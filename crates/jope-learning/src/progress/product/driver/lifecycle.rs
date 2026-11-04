use super::*;
use crate::progress::cadence::Clock;

impl<'r> Driver<'r> {
    /// A published engine corpus: its payload and manifest are appended, and the corpus freeze
    /// flushes immediately.
    pub fn corpus(&mut self, dir: &Path, clock: &dyn Clock) -> Result<Step> {
        let mapped = self.map_corpus(dir)?;
        let mut step = Step::default();
        self.append(&mapped.mapped, &mut step)?;
        self.urgent(Urgent::CorpusFreeze, clock, step)
    }

    fn map_corpus(&self, dir: &Path) -> Result<EngineCorpus> {
        engine::corpus(
            &self.manifest.value().run_id,
            dir,
            None,
            self.config.limits.max_bytes,
        )
    }

    /// The engine run `run` (relative to `engine_root`), training on the corpus at `corpus_dir`,
    /// has published a checkpoint. Its new optimizer steps are counted by the checkpoint cadence;
    /// when a checkpoint is due, the published checkpoint becomes the next consecutive W8 update
    /// and flushes immediately. Otherwise the steps wait for the next due checkpoint.
    /// `start_step` is the optimizer step the run started from (0 for a fresh run).
    pub fn checkpoint(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.observe(engine_root, run, corpus_dir, start_step, clock, false)
    }

    /// Controlled shutdown: if any counted optimizer step is not yet in a W8 checkpoint, the
    /// published engine checkpoint becomes one; then everything flushes.
    pub fn shutdown(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
    ) -> Result<Step> {
        self.observe(engine_root, run, corpus_dir, start_step, clock, true)
    }

    fn observe(
        &mut self,
        engine_root: &Path,
        run: &Path,
        corpus_dir: &Path,
        start_step: u64,
        clock: &dyn Clock,
        shutdown: bool,
    ) -> Result<Step> {
        let now = clock.now_ms();
        let mut step = Step::default();
        let published = engine::published(engine_root, run)?;
        let counted = self.observed_step.unwrap_or(start_step);
        // Counted on a copy: the new steps are counted only once this observation succeeds, so
        // a checkpoint that fails to map is counted, and due, again when it is fixed.
        let mut counter = self.checkpoints.clone();
        let mut due = None;
        let mut observed = self.observed_step;
        if let Some(p) = &published {
            if p.step > counted {
                due = counter.committed_updates(p.step - counted, now);
                observed = Some(p.step);
            }
        }
        if shutdown {
            due = due.or_else(|| counter.shutdown());
        }
        let mapped_through = self.previous.as_ref().map_or(start_step, |p| p.step);
        let decided = match (published, due) {
            (Some(p), Some(due)) if p.step > mapped_through => {
                let corpus = self.map_corpus(corpus_dir)?;
                let mapped = engine::checkpoint(
                    engine_root,
                    run,
                    &corpus,
                    self.previous.as_ref(),
                    start_step,
                    self.config.limits.max_bytes,
                )?;
                // The corpus is a dependency of the exposure: appended (or found) first.
                self.append(&corpus.mapped, &mut step)?;
                self.append(&mapped.mapped, &mut step)?;
                let decided = CheckpointStep::Mapped {
                    step: mapped.step,
                    update_index: mapped.checkpoint.value().update_index,
                    consumed_rows: mapped.exposure.value().consumed_rows,
                    due,
                };
                self.previous = Some(Previous {
                    checkpoint: mapped.checkpoint,
                    step: mapped.step,
                });
                self.checkpoint_pending = true;
                self.cadence.request(Urgent::CheckpointCommit);
                decided
            }
            (Some(p), None) if p.step > mapped_through => CheckpointStep::Deferred {
                step: p.step,
                uncheckpointed: counter.uncheckpointed(),
            },
            _ => CheckpointStep::Nothing,
        };
        self.checkpoints = counter;
        self.observed_step = observed;
        let mapped = matches!(decided, CheckpointStep::Mapped { .. });
        step.checkpoint = Some(decided);
        if shutdown {
            self.cadence.request(Urgent::Shutdown);
        }
        if mapped || shutdown {
            let flushed = self.tick(clock)?;
            step.due = flushed.due;
            step.publication = flushed.publication;
            step.tick = flushed.tick;
            step.flushed = flushed.flushed;
        }
        Ok(self.finish(step))
    }
}
