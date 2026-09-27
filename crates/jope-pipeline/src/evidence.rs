//! Re-derive a pipeline receipt from the evidence its run directory holds.
//!
//! [`crate::receipt::read`] proves a receipt is well formed and internally consistent. That is
//! not enough for anything that *acts* on a receipt — seeding a later run from its champion, a
//! lineage, the learning store's HWM ingest and champion bank — because a well-formed receipt can
//! still name a champion its run never measured. [`verify_run`] closes that gap: every digest the
//! receipt records must match the bytes on disk, and the selection (genome, mean held-out RMSE,
//! R^2, coverage) must re-derive from the published observations exactly, bit for bit.

use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};

use crate::{
    receipt::{self, Receipt},
    search::{Observation, ObservationRecord, INCUMBENT, REMEASURE},
    select, sha256_hex,
};

/// A receipt whose run directory re-derives it.
#[derive(Clone, Debug)]
pub struct VerifiedRun {
    pub receipt: Receipt,
    /// sha256 of the receipt file's bytes: the digest a child run records as its parent.
    pub receipt_sha256: String,
    /// The run directory (the receipt's parent).
    pub root: PathBuf,
    /// Each searched cut's observations, in receipt cut order, as `(cut index, observations)`.
    pub cuts: Vec<(usize, Vec<Observation>)>,
}

impl VerifiedRun {
    /// Every program measured on all of this run's cuts, best first under the selection rule.
    pub fn ranking(&self) -> Result<Vec<select::Best>> {
        let cuts: Vec<(usize, &[Observation])> = self
            .cuts
            .iter()
            .map(|(cut, observations)| (*cut, observations.as_slice()))
            .collect();
        select::ranking(&cuts)
    }
}

fn read_digest(path: &Path, expected: &str, what: &str) -> Result<Vec<u8>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("{what}: read {} (the run's evidence)", path.display()))?;
    let actual = sha256_hex(&bytes);
    ensure!(
        actual == expected,
        "{what}: {} hashes to {actual}, the receipt records {expected}",
        path.display()
    );
    Ok(bytes)
}

/// Validate `receipt_path`, then re-derive it from its run directory. Refuses v1 receipts: their
/// selection rule compared different cut sets and cannot be re-derived under today's rule.
pub fn verify_run(receipt_path: &Path) -> Result<VerifiedRun> {
    let bytes =
        std::fs::read(receipt_path).with_context(|| format!("read {}", receipt_path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", receipt_path.display()))?;
    let receipt = receipt::validate_value(&value)
        .map_err(|errors| anyhow::anyhow!("{}: {}", receipt_path.display(), errors.join("; ")))?;
    ensure!(
        receipt.schema != receipt::LEGACY_SCHEMA_ID,
        "{}: a {} receipt cannot be re-derived: its selection rule compared programs over \
         different cut sets",
        receipt_path.display(),
        receipt.schema
    );
    let root = receipt_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let context = || {
        format!(
            "{}: evidence does not re-derive the receipt",
            receipt_path.display()
        )
    };

    (|| -> Result<VerifiedRun> {
        read_digest(
            &root.join("prepared").join("prepared.json"),
            &receipt.prepared.sha256,
            "prepared",
        )?;
        let mut digests = String::new();
        let mut cuts = Vec::new();
        let mut published_migrations: Vec<receipt::Migration> = Vec::new();
        for cut in &receipt.search.cuts {
            let dir = root.join("search").join(format!("cut-{}", cut.cut));
            let summary = read_digest(&dir.join("search.json"), &cut.search_sha256, "search")?;
            let summary: serde_json::Value = serde_json::from_slice(&summary)?;
            ensure!(
                summary["observations_sha256"] == cut.observations_sha256.as_str(),
                "cut {}: search.json names other observations",
                cut.cut
            );
            if let Some(migrations) = summary.get("migrations") {
                published_migrations.extend(
                    serde_json::from_value::<Vec<receipt::Migration>>(migrations.clone())
                        .with_context(|| format!("cut {}: search.json migrations", cut.cut))?,
                );
            }
            let published = read_digest(
                &dir.join("observations.json"),
                &cut.observations_sha256,
                "observations",
            )?;
            digests.push_str(&cut.observations_sha256);
            let records: Vec<ObservationRecord> = serde_json::from_slice(&published)
                .with_context(|| format!("cut {}: parse observations.json", cut.cut))?;
            let observations = records
                .into_iter()
                .map(ObservationRecord::into_observation)
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("cut {}", cut.cut))?;
            let successful = observations.iter().filter(|o| o.measured()).count() as u64;
            let comparison = observations
                .iter()
                .filter(|o| o.origin == REMEASURE)
                .count() as u64;
            ensure!(
                observations.len() as u64 == cut.completed_evaluations
                    && successful == cut.successful_measurements
                    && observations.iter().all(|o| o.cut == cut.cut as usize),
                "cut {}: the observations do not account for the recorded outcomes",
                cut.cut
            );
            if let Some(accounting) = &cut.accounting {
                ensure!(
                    accounting.comparison.completed_evaluations == comparison,
                    "cut {}: the comparison phase does not match the remeasured observations",
                    cut.cut
                );
            }
            let best = observations.iter().filter(|o| o.measured()).min_by(|a, b| {
                a.rmse
                    .unwrap()
                    .total_cmp(&b.rmse.unwrap())
                    .then(a.genome.cmp(&b.genome))
            });
            ensure!(
                best.map(|o| (o.genome.clone(), o.rmse, o.hold_r2))
                    == cut.best_program.clone().map(|g| (
                        g,
                        cut.best_validation_rmse,
                        cut.best_validation_r2
                    )),
                "cut {}: the recorded best is not the best observation",
                cut.cut
            );
            cuts.push((cut.cut as usize, observations));
        }
        ensure!(
            sha256_hex(digests.as_bytes()) == receipt.observations_sha256,
            "the run-level observations digest does not bind the per-cut digests"
        );

        let per_cut: Vec<(usize, &[Observation])> =
            cuts.iter().map(|(c, o)| (*c, o.as_slice())).collect();
        let selection = select::select_from(&per_cut)?;
        let recorded = receipt
            .search
            .best
            .as_ref()
            .context("the receipt records no measured champion")?;
        let derived = selection.best.as_ref().context("no derived champion")?;
        ensure!(
            selection.genome == receipt.search.selected_genome
                && selection.program_sha256 == receipt.search.selected_program_sha256
                && derived.genome == recorded.genome
                && derived.validation_rmse.to_bits() == recorded.validation_rmse.to_bits()
                && derived.validation_r2.to_bits() == recorded.validation_r2.to_bits()
                && derived.train_r2.to_bits() == recorded.train_r2.to_bits()
                && derived.seconds.to_bits() == recorded.seconds.to_bits()
                && derived.cuts_measured as u64 == recorded.cuts_measured,
            "the recorded champion {} (mean held-out RMSE {}) is not what the observations select: \
             {} (mean held-out RMSE {})",
            recorded.genome,
            recorded.validation_rmse,
            derived.genome,
            derived.validation_rmse
        );
        if let Some(coverage) = &receipt.search.coverage {
            ensure!(
                *coverage == selection.coverage,
                "the recorded program coverage is not what the observations show"
            );
        }
        match &receipt.search.islands {
            Some(islands) => {
                for (cut, observations) in &cuts {
                    ensure!(
                        observations.iter().all(|o| match o.island {
                            Some(island) => islands.count > 1 && (island as u64) < islands.count,
                            None => islands.count == 1 || o.origin == REMEASURE,
                        }),
                        "cut {cut}: observations name islands this run did not have"
                    );
                }
                ensure!(
                    published_migrations == islands.migrations,
                    "the recorded migrations are not the ones the cuts' searches published"
                );
                for migration in &islands.migrations {
                    let measured = cuts.iter().any(|(cut, observations)| {
                        *cut as u64 == migration.cut
                            && observations.iter().any(|o| {
                                o.program_sha256 == migration.program_sha256
                                    && o.rmse.map(f64::to_bits) == Some(migration.rmse.to_bits())
                            })
                    });
                    ensure!(
                        measured,
                        "migrant {} was never measured with that score on cut {}",
                        migration.program_sha256,
                        migration.cut
                    );
                }
            }
            None => ensure!(
                published_migrations.is_empty(),
                "the cuts published migrations a receipt without islands does not record"
            ),
        }
        if let Some(seeding) = &receipt.search.seeding {
            for seed in &seeding.seeds {
                for (cut, observations) in &cuts {
                    ensure!(
                        observations.iter().any(|o| o.genome == seed.genome
                            && o.origin == INCUMBENT
                            && o.generation == 0),
                        "seed {} was not re-measured in generation 0 of cut {cut}",
                        seed.program_sha256
                    );
                }
            }
        }
        for artifact in &receipt.artifacts {
            let path = root.join("emit").join(&artifact.path);
            let bytes = read_digest(&path, &artifact.sha256, "artifact")?;
            ensure!(
                bytes.len() as u64 == artifact.bytes,
                "artifact {}: {} bytes, the receipt records {}",
                artifact.path,
                bytes.len(),
                artifact.bytes
            );
        }
        Ok(VerifiedRun {
            receipt: receipt.clone(),
            receipt_sha256: sha256_hex(&bytes),
            root: root.clone(),
            cuts,
        })
    })()
    .with_context(context)
}

/// [`verify_run`], refusing unless the run's receipt is `ok` and names a measured champion.
pub fn verify_champion_run(receipt_path: &Path) -> Result<VerifiedRun> {
    let run = verify_run(receipt_path)?;
    if !run.receipt.ok {
        bail!(
            "{}: the run did not pass (ok is false); its champion cannot be carried forward",
            receipt_path.display()
        );
    }
    Ok(run)
}
