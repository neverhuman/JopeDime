use super::*;

pub(super) fn validate_dataset_set(label: &str, datasets: &[String]) -> Result<()> {
    for dataset in datasets {
        validate_sha256(label, dataset)?;
    }
    require_unique(label, datasets.iter().map(String::as_str))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SchedulerLaneV4 {
    Challenge,
    Uniform,
    StaleUnderexposed,
    GlobalCvConfirmation,
}

impl SchedulerLaneV4 {
    pub(super) fn all() -> [Self; 4] {
        [
            Self::Challenge,
            Self::Uniform,
            Self::StaleUnderexposed,
            Self::GlobalCvConfirmation,
        ]
    }

    pub(super) fn share_basis_points(self) -> i128 {
        match self {
            Self::Challenge => 7_000,
            Self::Uniform => 1_500,
            Self::StaleUnderexposed => 1_000,
            Self::GlobalCvConfirmation => 500,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SchedulerStateV4 {
    pub(crate) schema: String,
    pub(crate) total_measured_cpu_micros: u64,
    pub(crate) measured_cpu_micros: BTreeMap<SchedulerLaneV4, u64>,
    pub(crate) lane_debt_basis_micros: BTreeMap<SchedulerLaneV4, i128>,
    pub(crate) draws: u64,
}

impl Default for SchedulerStateV4 {
    fn default() -> Self {
        Self {
            schema: "LearningSchedulerState.v4".into(),
            total_measured_cpu_micros: 0,
            measured_cpu_micros: SchedulerLaneV4::all()
                .into_iter()
                .map(|lane| (lane, 0))
                .collect(),
            lane_debt_basis_micros: SchedulerLaneV4::all()
                .into_iter()
                .map(|lane| (lane, 0))
                .collect(),
            draws: 0,
        }
    }
}

impl SchedulerStateV4 {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema != "LearningSchedulerState.v4"
            || self.measured_cpu_micros.len() != 4
            || self.lane_debt_basis_micros.len() != 4
            || SchedulerLaneV4::all().iter().any(|lane| {
                !self.measured_cpu_micros.contains_key(lane)
                    || !self.lane_debt_basis_micros.contains_key(lane)
            })
            || self.measured_cpu_micros.values().sum::<u64>() != self.total_measured_cpu_micros
        {
            bail!("scheduler v4 state is incomplete or does not reconcile measured CPU");
        }
        for lane in SchedulerLaneV4::all() {
            let expected = lane.share_basis_points() * self.total_measured_cpu_micros as i128
                - 10_000 * self.measured_cpu_micros[&lane] as i128;
            if self.lane_debt_basis_micros[&lane] != expected {
                bail!("scheduler lane debt does not reconstruct from measured CPU");
            }
        }
        Ok(())
    }

    /// Highest positive debt wins. Deterministic enum order breaks ties.
    #[cfg(test)]
    pub(crate) fn next_lane(&self) -> Result<SchedulerLaneV4> {
        self.validate()?;
        Ok(SchedulerLaneV4::all()
            .into_iter()
            .max_by(|left, right| {
                self.lane_debt_basis_micros[left]
                    .cmp(&self.lane_debt_basis_micros[right])
                    .then_with(|| right.cmp(left))
            })
            .expect("four scheduler lanes"))
    }

    #[cfg(test)]
    pub(crate) fn record_measured_cpu(
        &mut self,
        lane: SchedulerLaneV4,
        cpu_micros: u64,
    ) -> Result<()> {
        if cpu_micros == 0 {
            bail!("scheduler measured-CPU charge must be positive");
        }
        self.total_measured_cpu_micros = self
            .total_measured_cpu_micros
            .checked_add(cpu_micros)
            .context("scheduler total CPU overflow")?;
        *self
            .measured_cpu_micros
            .get_mut(&lane)
            .context("scheduler lane missing")? = self.measured_cpu_micros[&lane]
            .checked_add(cpu_micros)
            .context("scheduler lane CPU overflow")?;
        self.draws = self
            .draws
            .checked_add(1)
            .context("scheduler draw overflow")?;
        for current in SchedulerLaneV4::all() {
            self.lane_debt_basis_micros.insert(
                current,
                current.share_basis_points() * self.total_measured_cpu_micros as i128
                    - 10_000 * self.measured_cpu_micros[&current] as i128,
            );
        }
        self.validate()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SchedulerDrawV4 {
    pub(crate) draw_index: u64,
    pub(crate) lane: SchedulerLaneV4,
    pub(crate) dataset_sha256: String,
    pub(crate) state_snapshot_sha256: String,
    pub(crate) draw_seed: u64,
}

impl SchedulerDrawV4 {
    pub(super) fn validate(&self) -> Result<()> {
        validate_hashes([
            ("scheduler dataset", self.dataset_sha256.as_str()),
            (
                "scheduler state snapshot",
                self.state_snapshot_sha256.as_str(),
            ),
        ])
    }
}
