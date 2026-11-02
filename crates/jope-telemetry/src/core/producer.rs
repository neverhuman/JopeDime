use super::*;

impl ProducerHandle {
    pub fn source_id(&self) -> u16 {
        self.source_id
    }

    /// Wall-clock seconds since the recorder started, on the clock behind `monotonic_ns`.
    pub fn run_elapsed_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn set_metric_layout(&mut self, columns: Vec<(u16, u16)>) -> Result<()> {
        if columns.len() > 4096 {
            bail!("metric layout exceeds the 4,096-column registry cap");
        }
        for &(metric_id, _) in &columns {
            if crate::train::metric(metric_id).is_none() {
                bail!("unknown metric id {metric_id:#06x}");
            }
        }
        self.metric_layout = Arc::new(columns);
        Ok(())
    }

    #[inline]
    pub fn emit_scalar(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> bool {
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let event = self.event(metric_id, group_id, step, value, sampled);
        let pushed = self.try_push(event);
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        pushed
    }

    #[inline]
    pub fn emit_unavailable(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        reason_id: u16,
        sampled: bool,
    ) -> bool {
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let event = Event::unavailable(
            self.run_id,
            self.next_sequence(),
            self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            self.source_id,
            metric_id,
            group_id,
            step,
            reason_id,
            sampled,
        );
        let pushed = self.try_push(event);
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        pushed
    }

    /// Submit row-major float64 values and exact u64 steps from one Python batch.
    /// Missing values use the `StepBatch` sentinel and are not turned into fake zero samples.
    pub fn submit_values(&mut self, steps: &[u64], rows: usize, values: &[f64]) -> Result<u64> {
        let metric_layout = self.metric_layout.clone();
        let width = metric_layout.len();
        let expected = rows
            .checked_mul(width)
            .ok_or_else(|| anyhow!("metric batch dimensions overflow"))?;
        if rows > 32 || steps.len() < rows || values.len() < expected {
            bail!("metric batch dimensions exceed or do not match the registered layout");
        }
        self.stats.in_flight.fetch_add(1, Ordering::AcqRel);
        let mut dropped = 0;
        for (row, &step) in steps[..rows].iter().enumerate() {
            for (column, &(metric_id, group_id)) in metric_layout.iter().enumerate() {
                let value = values[row * width + column];
                if value.to_bits() == MISSING_VALUE_BITS {
                    continue;
                }
                let event = self.event(metric_id, group_id, step, value, false);
                if !self.try_push(event) {
                    dropped += 1;
                }
            }
        }
        self.stats.in_flight.fetch_sub(1, Ordering::Release);
        Ok(dropped)
    }

    #[inline]
    fn event(
        &mut self,
        metric_id: u16,
        group_id: u16,
        step: u64,
        value: f64,
        sampled: bool,
    ) -> Event {
        let sequence = self.next_sequence();
        Event::scalar(
            self.run_id,
            sequence,
            self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            self.source_id,
            metric_id,
            group_id,
            step,
            value,
            sampled,
        )
    }

    #[inline]
    fn next_sequence(&mut self) -> u64 {
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        sequence
    }

    #[inline]
    fn try_push(&mut self, event: Event) -> bool {
        if self.stopping.load(Ordering::Relaxed) {
            self.stats.late_drops.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        match self.producer.push(event) {
            Ok(()) => true,
            Err(_) => {
                self.stats.drops.fetch_add(1, Ordering::Relaxed);
                false
            }
        }
    }
}
