//! `jope.pipeline-seeds/v1`: the genomes a search is asked to start from.
//!
//! A seeds file is the one input format [`crate::run()`] accepts for seeding besides a verified
//! parent receipt (`--incumbent`). The learning store's champion bank exports it
//! (`jope-learning bank export`), and `jope pipeline evolve` writes one per round from the previous
//! round's champion and top-K. This crate only reads it: it never depends on the learning store.
//!
//! A seed is a *starting point*, never a score. Whatever the file or the parent receipt says a
//! genome measured, it is fitted again on this run's own cuts and competes on that measurement
//! alone. [`admit`] canonicalises, deduplicates (the bootstrap genome is always in generation 0
//! already) and caps the offered genomes at [`crate::search::ELITES`], in offered order.

use anyhow::{bail, ensure, Context, Result};
use feat_math::genome::GenomeIr;
use serde::{Deserialize, Serialize};
use std::{io::Write, path::Path};

use crate::{search, sha256_hex};

pub const SEEDS_SCHEMA_ID: &str = "jope.pipeline-seeds/v1";
pub const SEEDS_V2_SCHEMA_ID: &str = "jope.pipeline-seeds/v2";

/// Where an admitted seed came from.
pub const FROM_SEEDS_FILE: &str = "seeds_file";
pub const FROM_INCUMBENT: &str = "incumbent";

/// One offered genome with the digests that tie it to where it was banked or selected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedGenome {
    pub genome: String,
    /// sha256 of `genome` exactly as written here.
    pub program_sha256: String,
    /// The champion-bank entry it was exported from, when it came from the bank.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank_entry_sha256: Option<String>,
    /// The pipeline receipt that measured it, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedsFile {
    pub schema: String,
    /// The dataset the genomes were selected on, when known. Informational only: seeds are
    /// re-measured on the consuming run's cuts, so crossing datasets is allowed and honest.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_hash: Option<String>,
    pub genomes: Vec<SeedGenome>,
}

fn is_hex64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl SeedsFile {
    pub fn new(dataset_hash: Option<String>, genomes: Vec<SeedGenome>) -> Self {
        SeedsFile {
            schema: SEEDS_SCHEMA_ID.to_owned(),
            dataset_hash,
            genomes,
        }
    }

    /// Structural checks: schema, at least one genome, every digest well formed and every
    /// `program_sha256` the digest of its genome.
    pub fn check(&self) -> Result<()> {
        ensure!(
            self.schema == SEEDS_SCHEMA_ID,
            "seeds schema is {:?}, expected {SEEDS_SCHEMA_ID:?}",
            self.schema
        );
        ensure!(!self.genomes.is_empty(), "the seeds file names no genome");
        for (index, seed) in self.genomes.iter().enumerate() {
            ensure!(
                seed.program_sha256 == sha256_hex(seed.genome.as_bytes()),
                "seed {index}: program_sha256 {} is not the digest of its genome",
                seed.program_sha256
            );
            for (name, digest) in [
                ("bank_entry_sha256", &seed.bank_entry_sha256),
                ("receipt_sha256", &seed.receipt_sha256),
            ] {
                if let Some(digest) = digest {
                    ensure!(is_hex64(digest), "seed {index}: {name} is not 64 hex");
                }
            }
        }
        Ok(())
    }
}

/// Read and check a seeds file; returns it with the sha256 of its bytes.
pub fn read(path: &Path) -> Result<(SeedsFile, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let file: SeedsFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {} as {SEEDS_SCHEMA_ID}", path.display()))?;
    file.check()
        .with_context(|| format!("{}: invalid seeds file", path.display()))?;
    Ok((file, sha256_hex(&bytes)))
}

/// Write a seeds file that does not exist yet (fsynced); returns the sha256 of its bytes.
pub fn write(path: &Path, file: &SeedsFile) -> Result<String> {
    file.check()?;
    let mut bytes = serde_json::to_vec_pretty(file)?;
    bytes.push(b'\n');
    let mut handle = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {} (it must not exist yet)", path.display()))?;
    handle.write_all(&bytes)?;
    handle.sync_all()?;
    Ok(sha256_hex(&bytes))
}

/// The canonical pipeline form of `genome`. Seeds must be pipelines: the search's mutation and
/// crossover operators only act on the pipeline grammar.
pub fn canonical_pipeline(genome: &str) -> Result<String> {
    match GenomeIr::parse(genome) {
        Ok(parsed @ GenomeIr::Pipeline(_)) => Ok(parsed.canonical()),
        Ok(_) => bail!(
            "seed {genome:?} is not a (pipe ...) program; only pipelines can seed this search"
        ),
        Err(error) => bail!("seed {genome:?} does not parse: {error}"),
    }
}

/// A genome admitted into generation 0.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seed {
    /// Canonical pipeline DSL.
    pub genome: String,
    pub program_sha256: String,
    /// [`FROM_INCUMBENT`] or [`FROM_SEEDS_FILE`].
    pub source: &'static str,
}

/// What [`admit`] kept, and how many genomes were offered before it canonicalised,
/// deduplicated and capped them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Admitted {
    pub seeds: Vec<Seed>,
    pub offered: usize,
}

/// Canonicalise, deduplicate and cap `offered` (in order) at [`search::ELITES`].
///
/// A genome equal to the bootstrap is dropped: generation 0 always measures it already. A
/// genome that does not parse as a pipeline fails the whole admission rather than being
/// silently skipped.
pub fn admit(offered: &[(String, &'static str)]) -> Result<Admitted> {
    let mut seeds: Vec<Seed> = Vec::new();
    for (genome, source) in offered {
        let canonical = canonical_pipeline(genome)?;
        if canonical == search::BOOTSTRAP_GENOME || seeds.iter().any(|s| s.genome == canonical) {
            continue;
        }
        if seeds.len() == search::ELITES {
            break;
        }
        seeds.push(Seed {
            program_sha256: sha256_hex(canonical.as_bytes()),
            genome: canonical,
            source,
        });
    }
    Ok(Admitted {
        seeds,
        offered: offered.len(),
    })
}

/// Provenance category for an immutable v2 seed draw. v1 remains the legacy interchange.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedSourceV2 {
    OwnIncumbent,
    OtherLineageRetrieval,
    NeuralProposal,
}

/// Complete, already flattened ancestry supplied by the archive reader. Missing ancestry is
/// representable for audit, but learning retrieval must refuse to call it transfer.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedAncestryV2 {
    pub parent_families: Vec<String>,
    pub dataset_families: Vec<String>,
    pub canonical_program_sha256: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedDrawV2 {
    pub draw_id: String,
    pub genome: String,
    pub program_sha256: String,
    pub source: SeedSourceV2,
    #[serde(default)]
    pub ancestry: Option<SeedAncestryV2>,
}

/// Versioned input with explicit cap and requested source mixture. It is data only: callers
/// materialize it before pipeline execution; the pipeline never loads an archive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedSetV2 {
    pub schema: String,
    pub source_snapshot_sha256: String,
    pub cap: usize,
    pub requested_by_source: std::collections::BTreeMap<SeedSourceV2, usize>,
    pub draws: Vec<SeedDrawV2>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdmittedV2 {
    pub seeds: Vec<Seed>,
    pub provenance: Vec<AdmittedSeedV2>,
    pub offered_by_source: std::collections::BTreeMap<SeedSourceV2, usize>,
    pub realized_by_source: std::collections::BTreeMap<SeedSourceV2, usize>,
    pub rejected_duplicates: usize,
    pub rejected_bootstrap: usize,
    pub surplus: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedSeedV2 {
    pub draw_id: String,
    pub source: SeedSourceV2,
    pub ancestry: Option<SeedAncestryV2>,
    pub program_sha256: String,
}

impl SeedSetV2 {
    pub fn check(&self) -> Result<()> {
        ensure!(self.schema == SEEDS_V2_SCHEMA_ID, "unknown v2 seeds schema");
        ensure!(
            is_hex64(&self.source_snapshot_sha256),
            "invalid source snapshot digest"
        );
        ensure!(self.cap > 0, "v2 seed cap must be explicit and positive");
        ensure!(
            !self.requested_by_source.is_empty()
                && self.requested_by_source.values().any(|count| *count > 0),
            "v2 requested source mixture must be explicit and nonzero"
        );
        ensure!(!self.draws.is_empty(), "v2 seed set has no draws");
        let mut ids = std::collections::BTreeSet::new();
        for (i, draw) in self.draws.iter().enumerate() {
            ensure!(
                !draw.draw_id.is_empty() && ids.insert(&draw.draw_id),
                "duplicate or empty draw id at {i}"
            );
            ensure!(
                draw.program_sha256 == sha256_hex(draw.genome.as_bytes()),
                "draw {i} program digest mismatch"
            );
            if let Some(a) = &draw.ancestry {
                for digest in &a.canonical_program_sha256 {
                    ensure!(is_hex64(digest), "draw {i} ancestry program digest invalid");
                }
            }
            if draw.source == SeedSourceV2::OtherLineageRetrieval {
                let ancestry = draw
                    .ancestry
                    .as_ref()
                    .context("retrieval draw ancestry is unknown")?;
                ensure!(
                    !ancestry.parent_families.is_empty()
                        && !ancestry.dataset_families.is_empty()
                        && ancestry
                            .canonical_program_sha256
                            .contains(&draw.program_sha256),
                    "retrieval draw {i} ancestry is incomplete"
                );
            }
        }
        Ok(())
    }
}

/// Canonicalize/deduplicate with the explicitly versioned cap, retaining raw effort counts.
pub fn admit_v2(set: &SeedSetV2) -> Result<AdmittedV2> {
    set.check()?;
    let mut result = AdmittedV2::default();
    let mut seen = std::collections::BTreeSet::new();
    for draw in &set.draws {
        *result.offered_by_source.entry(draw.source).or_default() += 1;
        let genome = canonical_pipeline(&draw.genome)?;
        if genome == search::BOOTSTRAP_GENOME {
            result.rejected_bootstrap += 1;
            continue;
        }
        if !seen.insert(genome.clone()) {
            result.rejected_duplicates += 1;
            continue;
        }
        if result.seeds.len() == set.cap {
            result.surplus += 1;
            continue;
        }
        let canonical_sha256 = sha256_hex(genome.as_bytes());
        result.seeds.push(Seed {
            program_sha256: canonical_sha256.clone(),
            genome,
            source: match draw.source {
                SeedSourceV2::OwnIncumbent => FROM_INCUMBENT,
                SeedSourceV2::OtherLineageRetrieval | SeedSourceV2::NeuralProposal => {
                    FROM_SEEDS_FILE
                }
            },
        });
        result.provenance.push(AdmittedSeedV2 {
            draw_id: draw.draw_id.clone(),
            source: draw.source,
            ancestry: draw.ancestry.clone(),
            program_sha256: canonical_sha256,
        });
        *result.realized_by_source.entry(draw.source).or_default() += 1;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(genome: &str) -> SeedGenome {
        SeedGenome {
            genome: genome.to_owned(),
            program_sha256: sha256_hex(genome.as_bytes()),
            bank_entry_sha256: None,
            receipt_sha256: None,
        }
    }

    #[test]
    fn admission_canonicalises_deduplicates_drops_the_bootstrap_and_caps() {
        let a = canonical_pipeline("(pipe X (standardize) (ridge))").unwrap();
        let offered: Vec<(String, &'static str)> = vec![
            (search::BOOTSTRAP_GENOME.to_owned(), FROM_INCUMBENT),
            (format!("  {a}  "), FROM_INCUMBENT),
            (a.clone(), FROM_SEEDS_FILE),
            (
                "(pipe X (robust_scale) (ridge))".to_owned(),
                FROM_SEEDS_FILE,
            ),
            ("(pipe X (rank) (ridge))".to_owned(), FROM_SEEDS_FILE),
            (
                "(pipe X (standardize) (ridge_gcv))".to_owned(),
                FROM_SEEDS_FILE,
            ),
            ("(pipe X (blom) (ridge))".to_owned(), FROM_SEEDS_FILE),
        ];
        let admitted = admit(&offered).unwrap();
        assert_eq!(admitted.offered, 7);
        assert_eq!(admitted.seeds.len(), search::ELITES);
        assert_eq!(admitted.seeds[0].genome, a);
        assert_eq!(admitted.seeds[0].source, FROM_INCUMBENT);
        assert!(admitted
            .seeds
            .iter()
            .all(|s| s.genome != search::BOOTSTRAP_GENOME
                && s.program_sha256 == sha256_hex(s.genome.as_bytes())));
        let distinct: std::collections::BTreeSet<_> =
            admitted.seeds.iter().map(|s| &s.genome).collect();
        assert_eq!(distinct.len(), admitted.seeds.len());
    }

    #[test]
    fn a_seed_that_is_not_a_pipeline_is_refused_not_skipped() {
        for bad in ["(pipe X (ridge", "(sym affine \"x0\")", "hello"] {
            let error = admit(&[(bad.to_owned(), FROM_SEEDS_FILE)]).unwrap_err();
            assert!(error.to_string().contains("seed"), "{error}");
        }
    }

    #[test]
    fn a_seeds_file_round_trips_and_a_tampered_one_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("seeds.json");
        let file = SeedsFile::new(None, vec![entry("(pipe X (standardize) (ridge))")]);
        let sha = write(&path, &file).unwrap();
        let (back, again) = read(&path).unwrap();
        assert_eq!(back, file);
        assert_eq!(sha, again);
        // A second write never replaces the first.
        assert!(write(&path, &file).is_err());

        let mut tampered = serde_json::to_value(&file).unwrap();
        tampered["genomes"][0]["genome"] = "(pipe X (rank) (ridge))".into();
        let bad = temp.path().join("tampered.json");
        std::fs::write(&bad, tampered.to_string()).unwrap();
        let error = format!("{:#}", read(&bad).unwrap_err());
        assert!(error.contains("not the digest of its genome"), "{error}");

        let mut wrong = serde_json::to_value(&file).unwrap();
        wrong["schema"] = "jope.pipeline-seeds/v0".into();
        std::fs::write(&bad, wrong.to_string()).unwrap();
        assert!(read(&bad).is_err());

        let empty = SeedsFile::new(None, vec![]);
        assert!(write(&temp.path().join("empty.json"), &empty).is_err());
    }
}
