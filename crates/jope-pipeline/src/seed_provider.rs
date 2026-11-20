//! Post-partition generation-0 seed proposal (`CutSeedProvider`), invoked strictly after
//! `prepare` constructs FIT-only fold views so held-out fold rows never enter seed selection.

use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::{
    prepare, receipt,
    run::{Policy, RunOptions},
    search, seeds, sha256_hex, Receipt,
};

/// Receipt seed-source label for genomes proposed via [`CutSeedProvider`].
pub const FROM_SEED_PROVIDER: &str = "seed_provider";

/// Read-only view of one prepared cut's FIT-only rows (`train_x`, `train_y`, `dataset_hash`).
/// Held-out rows (`hold_x`, `hold_y`, `hold_indices`) are excluded so a [`CutSeedProvider`]
/// never inspects evaluation folds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitCutView<'a> {
    cut_index: usize,
    dataset_hash: &'a str,
    train_x: &'a [Vec<f64>],
    train_y: &'a [f64],
}

impl<'a> FitCutView<'a> {
    /// Construct a FIT-only view from one prepared [`prepare::Cut`].
    pub fn from_cut(cut: &'a prepare::Cut, dataset_hash: &'a str) -> Self {
        Self {
            cut_index: cut.index,
            dataset_hash,
            train_x: &cut.train_x,
            train_y: &cut.train_y,
        }
    }

    /// Project all prepared cuts of `preparation` into FIT-only views.
    pub fn from_preparation(prep: &'a prepare::Preparation, dataset_hash: &'a str) -> Vec<Self> {
        prep.cuts
            .iter()
            .map(|c| Self::from_cut(c, dataset_hash))
            .collect()
    }

    /// Zero-based cut index in `0..cuts`.
    pub fn cut_index(&self) -> usize {
        self.cut_index
    }
    /// Canonical 16-hex dataset hash for this run.
    pub fn dataset_hash(&self) -> &'a str {
        self.dataset_hash
    }
    /// Training feature rows for this cut (`hold_x` excluded).
    pub fn train_x(&self) -> &'a [Vec<f64>] {
        self.train_x
    }
    /// Training target values for this cut (`hold_y` excluded).
    pub fn train_y(&self) -> &'a [f64] {
        self.train_y
    }
    /// Number of training rows in this cut.
    pub fn train_rows(&self) -> usize {
        self.train_x.len()
    }
    /// Feature column count of this cut's training matrix.
    pub fn feature_count(&self) -> usize {
        self.train_x.first().map_or(0, Vec::len)
    }

    /// Deterministic SHA-256 commitment over `(cut_index, dataset_hash, train_x, train_y)`.
    pub fn fit_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"jope.fit-cut-view/v1\0");
        h.update((self.cut_index as u64).to_le_bytes());
        h.update(self.dataset_hash.as_bytes());
        h.update([0]);
        h.update((self.train_x.len() as u64).to_le_bytes());
        for row in self.train_x {
            h.update((row.len() as u64).to_le_bytes());
            for &v in row {
                h.update((if v == 0.0 { 0.0_f64 } else { v }).to_bits().to_le_bytes());
            }
        }
        for &y in self.train_y {
            h.update((if y == 0.0 { 0.0_f64 } else { y }).to_bits().to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }
}

/// Supplies generation-0 candidate seed genomes from FIT-only fold views after `prepare`.
pub trait CutSeedProvider: Send + Sync {
    /// Propose up to `limit` verified pipeline seed genomes from `fit_cuts`.
    fn propose_seeds(
        &self,
        fit_cuts: &[FitCutView<'_>],
        limit: usize,
    ) -> Result<Vec<seeds::SeedGenome>>;
}

/// Run the pipeline with an optional post-partition [`CutSeedProvider`].
/// Passing `None` executes the exact unprovidered [`crate::run::run`] path.
pub fn run_with_seed_provider(
    options: &RunOptions,
    provider: Option<&dyn CutSeedProvider>,
) -> Result<Receipt> {
    crate::narration::from_env(
        |tap| crate::run::run_observed_with_provider(options, tap, provider),
        || {
            options
                .cancel
                .as_ref()
                .is_some_and(|flag| flag.load(std::sync::atomic::Ordering::SeqCst))
        },
    )
}

pub(crate) enum ProposeOutcome {
    Skipped(String),
    Passed(Option<String>),
}

fn is_hex64(v: &str) -> bool {
    v.len() == 64 && v.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_provider(
    options: &RunOptions,
    dataset_hash: &str,
    preparation: &prepare::Preparation,
    root: &Path,
    base_offered: &[(String, &'static str)],
    base_genomes: Vec<String>,
    base_seeding: receipt::Seeding,
    provider: &dyn CutSeedProvider,
) -> Result<(Vec<String>, receipt::Seeding, ProposeOutcome)> {
    if options.policy == Policy::Skip {
        return Ok((
            base_genomes,
            base_seeding,
            ProposeOutcome::Skipped("--policy skip requested".to_owned()),
        ));
    }
    if base_seeding.seeds_file_sha256.is_some() || options.seed_genomes.is_some() {
        anyhow::bail!(
            "--seed-genomes cannot be combined with CutSeedProvider; seeds_file_sha256 is already bound"
        );
    }
    ensure!(
        (1..=seeds::MAX_SEED_LIMIT).contains(&options.seed_limit),
        "the seed limit must be 1..={}, got {}",
        seeds::MAX_SEED_LIMIT,
        options.seed_limit
    );
    let budget_cap = search::exploration_limit(options.budget, options.cuts).saturating_sub(1);
    let effective_limit = options.seed_limit.min(budget_cap);
    let fit_views = FitCutView::from_preparation(preparation, dataset_hash);
    let proposed = provider.propose_seeds(&fit_views, effective_limit)?;
    for (idx, seed) in proposed.iter().enumerate() {
        ensure!(
            seed.program_sha256 == sha256_hex(seed.genome.as_bytes()),
            "proposed seed {idx}: program_sha256 does not match genome"
        );
        for d in seed.bank_entry_sha256.iter().chain(&seed.receipt_sha256) {
            ensure!(is_hex64(d), "proposed seed {idx}: digest is not 64 hex");
        }
        seeds::canonical_pipeline(&seed.genome)?;
    }
    let provider_sha = (!proposed.is_empty())
        .then(|| {
            let file = seeds::SeedsFile::new(Some(dataset_hash.to_owned()), proposed.clone());
            seeds::write(&root.join("prepared").join("provider-seeds.json"), &file)
        })
        .transpose()?;
    let mut combined = base_offered.to_vec();
    combined.extend(
        proposed
            .iter()
            .map(|g| (g.genome.clone(), FROM_SEED_PROVIDER)),
    );
    if effective_limit == 0 {
        if options.policy == Policy::Require {
            anyhow::bail!("--policy require: budget permits 0 generation-0 seeds");
        }
        let mut seeding = base_seeding;
        seeding.offered = combined.len() as u64;
        return Ok((
            Vec::new(),
            seeding,
            ProposeOutcome::Passed(Some("budget permits 0 generation-0 seeds".into())),
        ));
    }
    let admitted = seeds::admit_up_to(&combined, effective_limit)?;
    if options.policy == Policy::Require && admitted.seeds.is_empty() {
        anyhow::bail!("--policy require: CutSeedProvider admitted 0 generation-0 seeds");
    }
    let seeding = receipt::Seeding {
        offered: admitted.offered as u64,
        seeds: admitted
            .seeds
            .iter()
            .map(|s| receipt::Seed {
                genome: s.genome.clone(),
                program_sha256: s.program_sha256.clone(),
                source: s.source.to_owned(),
            })
            .collect(),
        parent_receipt_sha256: base_seeding.parent_receipt_sha256,
        seeds_file_sha256: provider_sha,
    };
    let summary = format!(
        "seed_provider proposed {} genome(s); {} seed(s) admitted to generation 0",
        proposed.len(),
        seeding.seeds.len()
    );
    Ok((
        admitted.seeds.into_iter().map(|s| s.genome).collect(),
        seeding,
        ProposeOutcome::Passed(Some(summary)),
    ))
}

#[cfg(test)]
#[path = "seed_provider_tests.rs"]
mod tests;
