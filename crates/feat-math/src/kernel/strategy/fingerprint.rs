//! Fingerprints for identity and advisory finite-probe similarity.
//!
//! These types deliberately have separate domains. Matching probe fingerprints
//! means only that two output vectors matched after the declared rounding on
//! this exact ordered probe set. It is never a semantic equality proof and must
//! never be used to copy a measured result or champion receipt.

use feat_math::GenomeIr;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const STRATEGY_ID: &str = "advisory-fingerprint.v1";

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ByteIdentity([u8; 32]);

impl ByteIdentity {
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CanonicalIdentity([u8; 32]);

impl CanonicalIdentity {
    pub fn of_genome(genome: &GenomeIr) -> Self {
        Self(domain_hash(
            b"jopedime.canonical-genome.v1\0",
            genome.canonical().as_bytes(),
        ))
    }

    /// Parse, canonicalize, then fingerprint. Source bytes remain separately
    /// available through [`ByteIdentity`].
    pub fn of_source(source: &str) -> Result<Self, FingerprintError> {
        let genome = GenomeIr::parse(source).map_err(|_| FingerprintError::InvalidGenome)?;
        Ok(Self::of_genome(&genome))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeRole {
    Train,
    Validation,
    OuterEvaluation,
    Sealed,
    Unknown,
}

#[derive(Clone, Debug)]
pub struct ProbeRow {
    pub row_id: u64,
    pub values: Vec<f64>,
    /// Role assigned to this row by the caller's frozen partition manifest.
    pub role: ProbeRole,
    /// All known source/partition ancestors. Every ancestor must remain TRAIN.
    pub ancestry: Vec<ProbeRole>,
}

#[derive(Clone, Debug)]
pub struct TrainProbeSet {
    rows: Vec<ProbeRow>,
    identity: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FingerprintError {
    EmptySourceIdentity,
    EmptyProbeSet,
    NonTrainRole { row_id: u64, role: ProbeRole },
    NonTrainAncestor { row_id: u64, ancestor: ProbeRole },
    DuplicateRowId(u64),
    InvalidRow { row_id: u64 },
    MisalignedOutputs { expected: usize, actual: usize },
    InvalidRounding(u8),
    UnrepresentableOutput,
    InvalidGenome,
}

impl std::fmt::Display for FingerprintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FingerprintError {}

/// Construct an ordered probe set from rows whose declared role and every
/// recorded ancestor are TRAIN. The caller must bind `source_identity` and role
/// declarations to its accepted partition/lineage authority before use.
impl TrainProbeSet {
    pub fn new(
        source_identity: impl Into<String>,
        rows: Vec<ProbeRow>,
    ) -> Result<Self, FingerprintError> {
        let source_identity = source_identity.into();
        if source_identity.is_empty() {
            return Err(FingerprintError::EmptySourceIdentity);
        }
        if rows.is_empty() {
            return Err(FingerprintError::EmptyProbeSet);
        }
        let mut seen = BTreeSet::new();
        let width = rows[0].values.len();
        for row in &rows {
            if row.role != ProbeRole::Train {
                return Err(FingerprintError::NonTrainRole {
                    row_id: row.row_id,
                    role: row.role,
                });
            }
            if let Some(ancestor) = row
                .ancestry
                .iter()
                .copied()
                .find(|r| *r != ProbeRole::Train)
            {
                return Err(FingerprintError::NonTrainAncestor {
                    row_id: row.row_id,
                    ancestor,
                });
            }
            if !seen.insert(row.row_id) {
                return Err(FingerprintError::DuplicateRowId(row.row_id));
            }
            if row.values.is_empty()
                || row.values.len() != width
                || row.values.iter().any(|v| !v.is_finite())
            {
                return Err(FingerprintError::InvalidRow { row_id: row.row_id });
            }
        }

        let identity = probe_set_hash(&source_identity, &rows);
        Ok(Self { rows, identity })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn fingerprint_outputs(
        &self,
        outputs: &[f64],
        decimal_places: u8,
    ) -> Result<AdvisoryFingerprint, FingerprintError> {
        if outputs.len() != self.rows.len() {
            return Err(FingerprintError::MisalignedOutputs {
                expected: self.rows.len(),
                actual: outputs.len(),
            });
        }
        if decimal_places > 12 {
            return Err(FingerprintError::InvalidRounding(decimal_places));
        }
        let scale = 10_f64.powi(decimal_places as i32);
        let mut hasher = Sha256::new();
        hasher.update(b"jopedime.advisory-probe-fingerprint.v1\0");
        hasher.update(self.identity);
        hasher.update([decimal_places]);
        for output in outputs {
            if !output.is_finite() {
                return Err(FingerprintError::UnrepresentableOutput);
            }
            let rounded = (output * scale).round();
            if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
                return Err(FingerprintError::UnrepresentableOutput);
            }
            // Normalize both signed zero encodings to the same rounded value.
            let quantized = if rounded == 0.0 {
                0_i64
            } else {
                rounded as i64
            };
            hasher.update(quantized.to_le_bytes());
        }
        Ok(AdvisoryFingerprint {
            probe_identity: self.identity,
            decimal_places,
            digest: hasher.finalize().into(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct AdvisoryFingerprint {
    probe_identity: [u8; 32],
    decimal_places: u8,
    digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeRelation {
    SameOnFiniteProbes,
    DifferentOnFiniteProbes,
    IncomparableProbeSets,
}

impl AdvisoryFingerprint {
    pub fn compare(&self, other: &Self) -> ProbeRelation {
        if self.probe_identity != other.probe_identity
            || self.decimal_places != other.decimal_places
        {
            ProbeRelation::IncomparableProbeSets
        } else if self.digest == other.digest {
            ProbeRelation::SameOnFiniteProbes
        } else {
            ProbeRelation::DifferentOnFiniteProbes
        }
    }

    pub fn digest(&self) -> &[u8; 32] {
        &self.digest
    }
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher.finalize().into()
}

fn probe_set_hash(source_identity: &str, rows: &[ProbeRow]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"jopedime.train-probe-set.v1\0");
    hasher.update((source_identity.len() as u64).to_le_bytes());
    hasher.update(source_identity.as_bytes());
    hasher.update((rows.len() as u64).to_le_bytes());
    for row in rows {
        hasher.update(row.row_id.to_le_bytes());
        hasher.update((row.values.len() as u64).to_le_bytes());
        for value in &row.values {
            hasher.update(value.to_bits().to_le_bytes());
        }
    }
    hasher.finalize().into()
}
