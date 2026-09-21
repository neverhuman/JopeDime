//! Sweep plan (vendored `SweepConfig`, reshaped for the composite multi-scale sweep).

/// Orchestration knobs for the composite sweep. The model geometry itself (dim/heads/layers/
/// max_patch/pad_u8/quant_scale/ws_list) comes from the v7 manifest via
/// [`crate::hyperion::HyperionSweepSettings`].
#[derive(Clone, Debug)]
pub struct SweepPlan {
    /// Number of sweeps K (sweep 0 is always the identity permutation; later sweeps shuffle).
    pub sweeps: u32,
    /// Patch scales ("areas"), e.g. `[3, 5, 7, 9]`; each must be `2..=max_patch`.
    pub scales: Vec<usize>,
    /// Max patches per model batch.
    pub batch: usize,
    /// Base seed; sweep k shuffles with `seed ^ mix64(k)`.
    pub seed: u64,
    /// Aim for roughly this many streamed chunks per sweep (progress granularity).
    pub target_chunks_per_sweep: usize,
    /// Test hook: explicit (row, col) permutations per sweep (validated), overriding the RNG.
    pub fixed_perms: Option<Vec<(Vec<usize>, Vec<usize>)>>,
}

impl Default for SweepPlan {
    fn default() -> Self {
        Self {
            sweeps: 30,
            scales: vec![4, 7, 11, 15],
            batch: 256,
            seed: 0,
            target_chunks_per_sweep: 8,
            fixed_perms: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SweepPlan;

    #[test]
    fn default_sweep_plan_uses_thirty_sweeps() {
        assert_eq!(SweepPlan::default().sweeps, 30);
    }
}
