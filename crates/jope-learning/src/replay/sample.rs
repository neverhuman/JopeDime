//! Deterministic, bounded replay snapshots.
//!
//! This is a retention design only. It does not select training examples, account for
//! repeated training exposure, or delete source records. Inclusion probabilities describe
//! only the fixed bottom-k design over the declared frame.

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Domain separation for the stable per-record ranking.
pub const BOTTOM_K_HASH_DOMAIN: &str = "jopedime.replay.bottom-k/v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// `ReplayDesign` is a record for the replay design contract in this module: Deterministic, bounded replay snapshots.
pub struct ReplayDesign {
    /// Uniformly sampled, caller-provided seed, committed before inspecting this frame. It
    /// must remain unchanged across restart and merge. Reported probabilities are design-based
    /// over this seed distribution; a hand-picked or data-dependent seed invalidates them.
    pub seed: Vec<u8>,
    /// Maximum number of retained records, including protected lineage records.
    pub capacity: usize,
    /// Maximum sum of retained record payload bytes.
    pub max_bytes: u64,
}

impl ReplayDesign {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.seed.is_empty(), "replay seed must not be empty");
        ensure!(self.capacity > 0, "replay capacity must be positive");
        ensure!(self.max_bytes > 0, "replay byte cap must be positive");
        Ok(())
    }
}

/// Distinguishes retention membership from training-time selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ReplayRole {
    /// The `General` case of `ReplayRole`.
    General,
    /// The `RareFailure` case of `ReplayRole`.
    RareFailure,
    /// The `AcceptedParent` case of `ReplayRole`.
    AcceptedParent,
    /// The `AcceptedExposure` case of `ReplayRole`.
    AcceptedExposure,
}

/// One immutable replay identity. `payload_bytes` is the exact serialized payload size used
/// by the byte budget; payload bodies remain in their existing custody store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRecord {
    /// Stable identifier for this value.
    pub id: String,
    /// Data or authority role assigned to this value.
    pub role: ReplayRole,
    /// Byte count for the associated payload.
    pub payload_bytes: u64,
    /// Direct accepted-parent/source lineage required to reconstruct this record.
    #[serde(default)]
    pub parents: BTreeSet<String>,
}

impl ReplayRecord {
    fn validate(&self) -> Result<()> {
        ensure!(!self.id.is_empty(), "replay record id must not be empty");
        ensure!(
            self.payload_bytes > 0,
            "replay record payload must be non-empty"
        );
        ensure!(
            !self.parents.contains(&self.id),
            "record {} cannot be its own parent",
            self.id
        );
        Ok(())
    }

    fn protected(&self) -> bool {
        self.role != ReplayRole::General
    }
}

/// Inclusion probability is deliberately separate from training selection and repeat
/// exposure. Unknown means no design-based probability claim is justified.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum InclusionProbability {
    /// Field value recorded by this public API type.
    /// The `Known` case of `InclusionProbability`.
    Known {
        /// Numerator recorded for this outcome.
        numerator: u64,
        /// Denominator recorded for this outcome.
        denominator: u64,
    },
    /// The `Unknown` case of `InclusionProbability`.
    Unknown,
}

impl InclusionProbability {
    /// Performs the as f64 operation.
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::Known {
                numerator,
                denominator,
            } if denominator > 0 => Some(numerator as f64 / denominator as f64),
            _ => None,
        }
    }
}

/// A bounded snapshot from one disjoint, caller-identified frame partition.
///
/// `partition_id` must identify a stable, disjoint source partition. Merging overlapping
/// partitions is rejected by duplicate partition ID; callers must not relabel overlapping
/// frames as disjoint. Protected records and their transitive parents are always retained.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BottomKSnapshot {
    /// Design recorded for this bottom ksnapshot.
    pub design: ReplayDesign,
    /// Identifier for the associated partition.
    pub partition_id: String,
    /// Number of population represented here.
    pub population_count: u64,
    /// The per-partition bottom-k candidates. This remains bounded by capacity.
    pub records: Vec<ReplayRecord>,
    /// Exact selected identities, ordered by stable rank and then ID.
    pub selected_ids: Vec<String>,
    /// Byte count for the associated selected.
    pub selected_bytes: u64,
    /// False when protected records or their ancestors exceed the fixed candidate capacity.
    pub complete: bool,
}

impl BottomKSnapshot {
    /// Performs the build operation and returns an error if it cannot be completed.
    pub fn build(
        design: ReplayDesign,
        partition_id: impl Into<String>,
        population: impl IntoIterator<Item = ReplayRecord>,
    ) -> Result<Self> {
        design.validate()?;
        let partition_id = partition_id.into();
        ensure!(!partition_id.is_empty(), "partition id must not be empty");
        let mut by_id = BTreeMap::new();
        for record in population {
            record.validate()?;
            match by_id.insert(record.id.clone(), record.clone()) {
                Some(previous) if previous != record => {
                    bail!("conflicting duplicate replay identity {}", record.id)
                }
                _ => {}
            }
        }
        let population_count = by_id.len() as u64;
        let protected = ancestry_closure(&by_id)?;
        ensure!(
            protected.len() <= design.capacity,
            "protected lineage closure exceeds replay capacity"
        );

        let mut ranked: Vec<(String, [u8; 32])> = by_id
            .keys()
            .filter(|id| !protected.contains(*id))
            .map(|id| (id.clone(), rank(&design, id)))
            .collect();
        ranked.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        let available = design.capacity - protected.len();
        let mut retained_ids = protected;
        retained_ids.extend(ranked.into_iter().take(available).map(|(id, _)| id));
        let mut records: Vec<_> = retained_ids
            .iter()
            .filter_map(|id| by_id.get(id).cloned())
            .collect();
        records.sort_by(|a, b| {
            rank(&design, &a.id)
                .cmp(&rank(&design, &b.id))
                .then(a.id.cmp(&b.id))
        });
        let selected_bytes = records.iter().try_fold(0u64, |sum, record| {
            sum.checked_add(record.payload_bytes)
                .ok_or_else(|| anyhow::anyhow!("selected byte count overflow"))
        })?;
        ensure!(
            selected_bytes <= design.max_bytes,
            "protected/bottom-k snapshot needs {selected_bytes} bytes, cap is {}",
            design.max_bytes
        );
        let selected_ids = records.iter().map(|r| r.id.clone()).collect();
        Ok(Self {
            design,
            partition_id,
            population_count,
            records,
            selected_ids,
            selected_bytes,
            complete: true,
        })
    }

    /// Merge independently built snapshots from disjoint partitions under the same design.
    /// The union of each partition's bottom-k candidates contains the global bottom-k.
    pub fn merge(&self, other: &Self) -> Result<Self> {
        ensure!(
            self.design == other.design,
            "cannot merge different replay designs"
        );
        ensure!(
            self.partition_id != other.partition_id,
            "duplicate replay partition {}",
            self.partition_id
        );
        ensure!(
            self.complete && other.complete,
            "cannot merge incomplete snapshots"
        );
        let mut records = BTreeMap::new();
        for record in self.records.iter().chain(&other.records) {
            if records.insert(record.id.clone(), record.clone()).is_some() {
                bail!(
                    "overlapping replay identity {} across declared disjoint partitions",
                    record.id
                );
            }
        }
        let mut merged = Self::build(
            self.design.clone(),
            format!("{}+{}", self.partition_id, other.partition_id),
            records.into_values(),
        )?;
        merged.population_count = self
            .population_count
            .checked_add(other.population_count)
            .ok_or_else(|| anyhow::anyhow!("population count overflow"))?;
        Ok(merged)
    }

    /// Fixed-frame first-order inclusion probability for an unprotected item. This is known
    /// only when the whole frame fits the design without the byte cap forcing a refusal.
    pub fn inclusion_probability(&self, id: &str) -> InclusionProbability {
        if self.population_count == 0
            || !self.complete
            || !self.selected_ids.iter().any(|selected| selected == id)
        {
            return InclusionProbability::Unknown;
        }
        let records: BTreeMap<_, _> = self
            .records
            .iter()
            .map(|record| (record.id.clone(), record.clone()))
            .collect();
        let Ok(protected) = ancestry_closure(&records) else {
            return InclusionProbability::Unknown;
        };
        if protected.contains(id) {
            return InclusionProbability::Known {
                numerator: 1,
                denominator: 1,
            };
        }
        let protected_count = protected.len() as u64;
        let nonprotected = self.population_count.saturating_sub(protected_count);
        let slots = self.design.capacity.saturating_sub(protected.len()) as u64;
        if nonprotected == 0 || nonprotected <= slots {
            return InclusionProbability::Known {
                numerator: 1,
                denominator: 1,
            };
        }
        InclusionProbability::Known {
            numerator: slots,
            denominator: nonprotected,
        }
    }

    /// Horvitz-Thompson totals require a positive, known probability for every unit.
    pub fn ht_total(&self, contributions: &BTreeMap<String, f64>) -> Result<f64> {
        ensure!(
            contributions
                .keys()
                .all(|id| self.selected_ids.contains(id)),
            "HT contributions must correspond to selected identities"
        );
        let mut total = 0.0;
        for (id, value) in contributions {
            ensure!(value.is_finite(), "non-finite contribution for {id}");
            let p = self.inclusion_probability(id).as_f64().ok_or_else(|| {
                anyhow::anyhow!("unknown inclusion probability for {id}; HT total refused")
            })?;
            ensure!(p > 0.0, "non-positive inclusion probability for {id}");
            total += value / p;
        }
        ensure!(total.is_finite(), "HT total is non-finite");
        Ok(total)
    }

    /// A ratio estimator has its own contract and is not declared unbiased by HT support.
    pub fn ratio_mean(
        &self,
        numerator: &BTreeMap<String, f64>,
        denominator: &BTreeMap<String, f64>,
    ) -> Result<f64> {
        ensure!(
            numerator.keys().eq(denominator.keys()),
            "ratio numerator and denominator identities differ"
        );
        let num = self.ht_total(numerator)?;
        let den = self.ht_total(denominator)?;
        ensure!(den > 0.0, "ratio estimator denominator must be positive");
        Ok(num / den)
    }
}

fn ancestry_closure(records: &BTreeMap<String, ReplayRecord>) -> Result<BTreeSet<String>> {
    let mut closure = BTreeSet::new();
    let mut pending: Vec<String> = records
        .values()
        .filter(|record| record.protected())
        .map(|record| record.id.clone())
        .collect();
    while let Some(id) = pending.pop() {
        if !closure.insert(id.clone()) {
            continue;
        }
        let record = records
            .get(&id)
            .ok_or_else(|| anyhow::anyhow!("required replay ancestry {id} is missing"))?;
        for parent in &record.parents {
            ensure!(
                records.contains_key(parent),
                "required replay ancestry {parent} for {id} is missing"
            );
            pending.push(parent.clone());
        }
    }
    Ok(closure)
}

fn rank(design: &ReplayDesign, id: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(BOTTOM_K_HASH_DOMAIN.as_bytes());
    hash.update((design.seed.len() as u64).to_be_bytes());
    hash.update(&design.seed);
    hash.update((id.len() as u64).to_be_bytes());
    hash.update(id.as_bytes());
    hash.finalize().into()
}
