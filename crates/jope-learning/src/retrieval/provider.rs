//! Post-partition [`CutSeedProvider`] adapter backed by [`super::materialize`] and [`SeedSetV2`].

use anyhow::{ensure, Result};
use jope_pipeline::{
    seed_provider::{CutSeedProvider, FitCutView},
    seeds::{admit_v2, SeedGenome, SeedSourceV2},
};
use std::collections::BTreeMap;

use super::{materialize, Candidate, RetrievalMode, RetrievalRequest};

/// Compute a deterministic four-coordinate FIT-only descriptor from prepared `fit_cuts`.
///
/// Reads only `train_x` and `train_y` of `fit_cuts` so held-out fold rows cannot influence
/// retrieval distance ordering.
pub fn fit_cut_summary_context(fit_cuts: &[FitCutView<'_>]) -> [f64; 4] {
    if fit_cuts.is_empty() {
        return [0.0; 4];
    }
    let inv_cuts = 1.0 / (fit_cuts.len() as f64);
    let mut log_rows = 0.0;
    let mut log_cols = 0.0;
    let mut y_std_acc = 0.0;
    let mut abs_corr_acc = 0.0;
    for cut in fit_cuts {
        let n = cut.train_rows();
        let p = cut.feature_count();
        log_rows += ((n as f64) + 1.0).ln() * inv_cuts;
        log_cols += ((p as f64) + 1.0).ln() * inv_cuts;
        if n < 2 || p == 0 {
            continue;
        }
        let inv_n = 1.0 / (n as f64);
        let mean_y = cut.train_y().iter().sum::<f64>() * inv_n;
        let var_y = cut
            .train_y()
            .iter()
            .map(|y| (y - mean_y).powi(2))
            .sum::<f64>()
            * inv_n;
        let std_y = var_y.sqrt();
        y_std_acc += (1.0 + std_y).ln() * inv_cuts;
        if std_y <= 1e-12 {
            continue;
        }
        let mut corr_sum = 0.0;
        let mut active_cols = 0usize;
        for col in 0..p {
            let mean_x = cut.train_x().iter().map(|r| r[col]).sum::<f64>() * inv_n;
            let mut var_x = 0.0;
            let mut cov = 0.0;
            for (row, &y) in cut.train_x().iter().zip(cut.train_y()) {
                let dx = row[col] - mean_x;
                var_x += dx * dx;
                cov += dx * (y - mean_y);
            }
            var_x *= inv_n;
            cov *= inv_n;
            let denom = var_x.sqrt() * std_y;
            if denom > 1e-12 {
                corr_sum += (cov / denom).abs().clamp(0.0, 1.0);
                active_cols += 1;
            }
        }
        if active_cols > 0 {
            abs_corr_acc += (corr_sum / (active_cols as f64)) * inv_cuts;
        }
    }
    [log_rows, log_cols, y_std_acc, abs_corr_acc]
}

/// Family-disjoint [`CutSeedProvider`] backed by [`super::materialize`] and `SeedSetV2`.
#[derive(Clone, Debug)]
pub struct RetrievalSeedProvider {
    /// Stable request identifier prefix recorded in `SeedSetV2` draw IDs.
    pub request_id: String,
    /// Caller's parent lineage families excluded from transfer eligibility.
    pub own_parent_families: Vec<String>,
    /// Caller's dataset families excluded from transfer eligibility.
    pub own_dataset_families: Vec<String>,
    /// Caller's own canonical program SHA-256 digests excluded from transfer eligibility.
    pub own_program_sha256: Vec<String>,
    /// SHA-256 digest of the caller-verified frozen role manifest.
    pub role_manifest_sha256: String,
    /// SHA-256 digest of the immutable candidate snapshot.
    pub snapshot_sha256: String,
    /// Frozen support roster of eligible candidate identifiers.
    pub eligible_support: Vec<String>,
    /// Immutable candidate bank snapshot (`context.len() == 4` matching [`fit_cut_summary_context`]).
    pub candidates: Vec<Candidate>,
    /// Retrieval evaluation mode (`Cold`, `Warm`, `ShuffledContext`, or `DonorDatasetShuffle`).
    pub mode: RetrievalMode,
}

impl CutSeedProvider for RetrievalSeedProvider {
    fn propose_seeds(&self, fit_cuts: &[FitCutView<'_>], limit: usize) -> Result<Vec<SeedGenome>> {
        ensure!(!fit_cuts.is_empty(), "fit_cuts must be non-empty");
        let context = fit_cut_summary_context(fit_cuts).to_vec();
        let full_cap = self.candidates.len().max(1);
        let request = RetrievalRequest {
            request_id: self.request_id.clone(),
            own_parent_families: self.own_parent_families.clone(),
            own_dataset_families: self.own_dataset_families.clone(),
            own_program_sha256: self.own_program_sha256.clone(),
            role_manifest_sha256: self.role_manifest_sha256.clone(),
            context,
            eligible_support: self.eligible_support.clone(),
            cap: full_cap,
            requested_by_source: BTreeMap::from([(
                SeedSourceV2::OtherLineageRetrieval,
                limit.max(1),
            )]),
            snapshot_sha256: self.snapshot_sha256.clone(),
            cold_cost_ms: 0,
            warm_cost_ms: 0,
            cache_warmup_ms: 0,
        };
        let (eligible, _) = super::filter_eligible(&request, &self.candidates)?;
        if limit == 0 || eligible.is_empty() {
            return Ok(Vec::new());
        }
        let materialized = materialize(&request, &self.candidates, self.mode)?;
        let mut seed_set = materialized.seeds;
        seed_set.cap = limit;
        seed_set.requested_by_source =
            BTreeMap::from([(SeedSourceV2::OtherLineageRetrieval, limit)]);
        let admitted = admit_v2(&seed_set)?;
        Ok(admitted
            .seeds
            .into_iter()
            .map(|s| SeedGenome {
                genome: s.genome,
                program_sha256: s.program_sha256,
                bank_entry_sha256: None,
                receipt_sha256: None,
            })
            .collect())
    }
}
