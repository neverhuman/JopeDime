use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FitPurpose {
    PriorArgmaxControl,
    MeasuredLeaf,
    BestOfNControl,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BudgetSnapshot {
    pub fits_charged: u64,
    pub cpu_ns_charged: u64,
    pub failed_fits_charged: u64,
    pub visits: u64,
    pub by_purpose: BTreeMap<FitPurpose, u64>,
    pub over_budget: bool,
}

/// Counts the prior-argmax control, every measured leaf, best-of-N evaluations and failed fits.
/// Visits are a separate counter and never substitute for measured fits.
#[derive(Clone, Debug)]
pub struct MeasuredFitBudget {
    max_fits: u64,
    max_cpu_ns: u64,
    snapshot: BudgetSnapshot,
}

impl MeasuredFitBudget {
    pub fn new(max_fits: u64, max_cpu_ns: u64) -> Self {
        Self {
            max_fits,
            max_cpu_ns,
            snapshot: BudgetSnapshot::default(),
        }
    }

    pub fn can_start_fit(&self, estimated_cpu_ns: u64) -> bool {
        estimated_cpu_ns > 0
            && self.snapshot.fits_charged < self.max_fits
            && self
                .snapshot
                .cpu_ns_charged
                .checked_add(estimated_cpu_ns)
                .is_some_and(|total| total <= self.max_cpu_ns)
    }

    /// Conservative slot count using a caller-supplied per-fit CPU estimate. Actual CPU is
    /// charged on completion; any underestimate remains visible as an over-budget result.
    pub fn available_fit_slots(&self, estimated_cpu_ns: u64) -> usize {
        if estimated_cpu_ns == 0 {
            return 0;
        }
        let fit_slots = self.max_fits.saturating_sub(self.snapshot.fits_charged);
        let cpu_slots =
            self.max_cpu_ns.saturating_sub(self.snapshot.cpu_ns_charged) / estimated_cpu_ns;
        fit_slots.min(cpu_slots).min(usize::MAX as u64) as usize
    }

    /// Charges a fit after completion. Failed attempts and actual CPU overrun remain recorded.
    pub fn charge_fit(
        &mut self,
        purpose: FitPurpose,
        cpu_ns: u64,
        succeeded: bool,
    ) -> Result<(), RootSearchError> {
        let fits_charged = self
            .snapshot
            .fits_charged
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        let cpu_ns_charged = self
            .snapshot
            .cpu_ns_charged
            .checked_add(cpu_ns)
            .ok_or(RootSearchError::CounterOverflow)?;
        let failed_fits_charged = if succeeded {
            self.snapshot.failed_fits_charged
        } else {
            self.snapshot
                .failed_fits_charged
                .checked_add(1)
                .ok_or(RootSearchError::CounterOverflow)?
        };
        let purpose_count = self
            .snapshot
            .by_purpose
            .get(&purpose)
            .copied()
            .unwrap_or_default()
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        self.snapshot.fits_charged = fits_charged;
        self.snapshot.cpu_ns_charged = cpu_ns_charged;
        self.snapshot.failed_fits_charged = failed_fits_charged;
        self.snapshot.by_purpose.insert(purpose, purpose_count);
        self.snapshot.over_budget =
            fits_charged > self.max_fits || cpu_ns_charged > self.max_cpu_ns;
        Ok(())
    }

    pub fn record_visit(&mut self) -> Result<(), RootSearchError> {
        self.snapshot.visits = self
            .snapshot
            .visits
            .checked_add(1)
            .ok_or(RootSearchError::CounterOverflow)?;
        Ok(())
    }

    pub fn snapshot(&self) -> &BudgetSnapshot {
        &self.snapshot
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TeacherTargetError {
    UnsupportedSamplingLaw,
    IncompleteSearch,
    MissingMeasuredTarget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeacherTarget {
    pub profile: &'static str,
    pub law: SamplingLaw,
    pub fidelity: Fidelity,
    pub measured_action_ids: Vec<String>,
}
