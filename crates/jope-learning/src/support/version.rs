use std::collections::{BTreeMap, BTreeSet};

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use super::{tokenizer::validate_program, Vocabulary};
use crate::canon::{canonical_sha256, is_hex64};

/// Constant identifying the v1 trie SHA-256.
pub const V1_TRIE_SHA256: &str = "116154767f2df56fb065c9b6bc3ce07df38c3672843c0eb4b944d84e8e8473bd";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// `Program` is a public program record.
pub struct Program {
    /// Dsl recorded for this program.
    pub dsl: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Collection of tokens values carried by this program.
    pub tokens: Vec<u16>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `Catalog` is a public catalog record.
pub struct Catalog {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated vocabulary.
    pub vocabulary_sha256: String,
    /// Upper bound on seq.
    pub max_seq: usize,
    /// Collection of programs values carried by this catalog.
    pub programs: Vec<Program>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// `Role` is a public role enum.
pub enum Role {
    /// The `Train` case of `Role`.
    Train,
    /// The `Validation` case of `Role`.
    Validation,
    /// The `Heldout` case of `Role`.
    Heldout,
    /// The `Sealed` case of `Role`.
    Sealed,
    /// The `Unknown` case of `Role`.
    Unknown,
}

/// Immutable evidence stays separate from action availability and numeric targets.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    /// Identifier for the associated context.
    pub context_id: String,
    /// Program recorded for this candidate.
    pub program: Program,
    /// SHA-256 digest identifying the associated source receipt.
    pub source_receipt_sha256: String,
    /// SHA-256 digest identifying the associated parity receipt.
    pub parity_receipt_sha256: Option<String>,
    /// Optional inner loss value; `None` means it is not available.
    pub inner_loss: Option<f64>,
    /// Optional incumbent inner loss value; `None` means it is not available.
    pub incumbent_inner_loss: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `FreezeRequest` is a public freeze request record.
pub struct FreezeRequest {
    /// Catalog recorded for this freeze request.
    pub catalog: Catalog,
    /// SHA-256 digest identifying the associated expected catalog.
    pub expected_catalog_sha256: String,
    /// Require v1 catalog recorded for this freeze request.
    pub require_v1_catalog: bool,
    /// Mapping of role map values carried by this freeze request.
    pub role_map: BTreeMap<String, Role>,
    /// SHA-256 digest identifying the associated expected role map.
    pub expected_role_map_sha256: String,
    /// Collection of candidates values carried by this freeze request.
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// `FrozenSupport` is a public frozen support record.
pub struct FrozenSupport {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Qualified recorded for this frozen support.
    pub qualified: bool,
    /// Qualified use recorded for this frozen support.
    pub qualified_use: String,
    /// Version recorded for this frozen support.
    pub version: u32,
    /// SHA-256 digest identifying the associated parent support.
    pub parent_support_sha256: String,
    /// SHA-256 digest identifying the associated catalog trie.
    pub catalog_trie_sha256: String,
    /// Vocabulary recorded for this frozen support.
    pub vocabulary: Vocabulary,
    /// Mapping of role map values carried by this frozen support.
    pub role_map: BTreeMap<String, Role>,
    /// SHA-256 digest identifying the associated role map.
    pub role_map_sha256: String,
    /// SHA-256 digest identifying the associated catalog program.
    pub catalog_program_sha256: Vec<String>,
    /// Collection of programs values carried by this frozen support.
    pub programs: Vec<Program>,
    /// Collection of provenance values carried by this frozen support.
    pub provenance: Vec<Candidate>,
    /// Availability grants context outcome eligibility recorded for this frozen support.
    pub availability_grants_context_outcome_eligibility: bool,
    /// SHA-256 digest identifying the associated support.
    pub support_sha256: String,
}

impl FrozenSupport {
    /// Whether the immutable support already contains this canonical program identity.
    pub fn contains_program(&self, program: &Program) -> bool {
        self.programs
            .iter()
            .any(|existing| existing.program_sha256 == program.program_sha256)
    }

    /// Performs the freeze operation and returns an error if it cannot be completed.
    pub fn freeze(request: FreezeRequest) -> Result<Self> {
        let hash = canonical_sha256(&request.catalog)?;
        ensure!(
            hash == request.expected_catalog_sha256 && is_hex64(&hash),
            "catalog identity mismatch"
        );
        ensure!(
            request.catalog.schema == "jope.flywheel-full-catalog-trie/v1",
            "catalog schema mismatch"
        );
        if request.require_v1_catalog {
            ensure!(
                hash == V1_TRIE_SHA256 && request.catalog.programs.len() == 6640,
                "exact deployed 6640 catalog required"
            );
        }
        ensure!(
            canonical_sha256(&request.role_map)? == request.expected_role_map_sha256,
            "role map identity mismatch"
        );
        let vocabulary = Vocabulary {
            sha256: request.catalog.vocabulary_sha256,
            max_seq: request.catalog.max_seq,
        };
        vocabulary.validate()?;
        let mut programs = BTreeMap::new();
        let catalog_program_sha256 = request
            .catalog
            .programs
            .iter()
            .map(|p| p.program_sha256.clone())
            .collect();
        for program in request.catalog.programs {
            validate_program(&vocabulary, &program)?;
            ensure!(
                programs
                    .insert(program.program_sha256.clone(), program)
                    .is_none(),
                "duplicate catalog program"
            );
        }
        ensure!(!programs.is_empty(), "nonempty catalog required");
        let mut support = Self {
            schema: "jope.rlfw-successor-support/v2".into(),
            qualified: false,
            qualified_use: "BLOCKED_JEPSON_G12".into(),
            version: 1,
            parent_support_sha256: hash.clone(),
            catalog_trie_sha256: hash,
            vocabulary,
            role_map: request.role_map,
            role_map_sha256: request.expected_role_map_sha256,
            catalog_program_sha256,
            programs: programs.into_values().collect(),
            provenance: Vec::new(),
            availability_grants_context_outcome_eligibility: false,
            support_sha256: String::new(),
        };
        support.add_candidates(request.candidates)?;
        support.seal()?;
        support.verify()?;
        Ok(support)
    }

    /// Extending creates a new immutable version; old bytes and identity are preserved.
    pub fn extend(&self, candidates: Vec<Candidate>) -> Result<Self> {
        self.verify()?;
        let mut next = self.clone();
        next.parent_support_sha256 = self.support_sha256.clone();
        next.version = self
            .version
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("support version overflow"))?;
        next.add_candidates(candidates)?;
        next.seal()?;
        next.verify()?;
        Ok(next)
    }

    fn validate_candidate(&self, candidate: &Candidate) -> Result<()> {
        ensure!(
            is_hex64(&candidate.context_id) && is_hex64(&candidate.source_receipt_sha256),
            "bound context/source receipt required"
        );
        ensure!(
            self.role_map.get(&candidate.context_id) == Some(&Role::Train),
            "support extension requires frozen TRAIN role"
        );
        if let Some(pin) = &candidate.parity_receipt_sha256 {
            ensure!(is_hex64(pin), "parity receipt SHA required");
        }
        for value in [candidate.inner_loss, candidate.incumbent_inner_loss]
            .into_iter()
            .flatten()
        {
            ensure!(
                value.is_finite() && value >= 0.,
                "finite nonnegative measured loss required"
            );
        }
        validate_program(&self.vocabulary, &candidate.program)
    }

    fn add_candidates(&mut self, candidates: Vec<Candidate>) -> Result<()> {
        let mut programs: BTreeMap<_, _> = self
            .programs
            .iter()
            .map(|p| (p.program_sha256.clone(), p.clone()))
            .collect();
        let mut provenance: BTreeMap<_, _> = self
            .provenance
            .iter()
            .map(|p| Ok((canonical_sha256(p)?, p.clone())))
            .collect::<Result<_>>()?;
        for candidate in candidates {
            self.validate_candidate(&candidate)?;
            if let Some(previous) = programs.get(&candidate.program.program_sha256) {
                ensure!(previous == &candidate.program, "program identity collision");
            }
            programs.insert(
                candidate.program.program_sha256.clone(),
                candidate.program.clone(),
            );
            provenance.insert(canonical_sha256(&candidate)?, candidate);
        }
        ensure!(programs.len() <= 4_000_000, "bounded support required");
        self.programs = programs.into_values().collect();
        self.provenance = provenance.into_values().collect();
        Ok(())
    }

    fn content_sha256(&self) -> Result<String> {
        let mut value = serde_json::to_value(self)?;
        value
            .as_object_mut()
            .expect("support object")
            .remove("support_sha256");
        canonical_sha256(&value)
    }

    fn seal(&mut self) -> Result<()> {
        self.support_sha256 = self.content_sha256()?;
        Ok(())
    }

    /// Verifies the bound evidence and returns an error when verification fails.
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.rlfw-successor-support/v2"
                && !self.qualified
                && self.qualified_use == "BLOCKED_JEPSON_G12"
                && !self.availability_grants_context_outcome_eligibility,
            "G12-blocked nonqualifying support required"
        );
        ensure!(
            self.version > 0
                && is_hex64(&self.parent_support_sha256)
                && is_hex64(&self.catalog_trie_sha256),
            "version/parent identity required"
        );
        self.vocabulary.validate()?;
        ensure!(
            self.role_map.keys().all(|k| is_hex64(k))
                && canonical_sha256(&self.role_map)? == self.role_map_sha256,
            "role map binding drift"
        );
        ensure!(
            !self.programs.is_empty() && self.programs.len() <= 4_000_000,
            "bounded nonempty support required"
        );
        let mut previous = None;
        let mut ids = BTreeSet::new();
        for program in &self.programs {
            validate_program(&self.vocabulary, program)?;
            ensure!(
                previous.is_none_or(|p: &str| p < program.program_sha256.as_str()),
                "programs must be unique and sorted"
            );
            previous = Some(&program.program_sha256);
            ids.insert(program.program_sha256.as_str());
        }
        let catalog: BTreeSet<_> = self
            .catalog_program_sha256
            .iter()
            .map(String::as_str)
            .collect();
        ensure!(
            !catalog.is_empty()
                && catalog.len() == self.catalog_program_sha256.len()
                && catalog.is_subset(&ids),
            "catalog containment drift"
        );
        let by_sha: BTreeMap<_, _> = self
            .programs
            .iter()
            .map(|p| (&p.program_sha256, p))
            .collect();
        let original = Catalog {
            schema: "jope.flywheel-full-catalog-trie/v1".into(),
            vocabulary_sha256: self.vocabulary.sha256.clone(),
            max_seq: self.vocabulary.max_seq,
            programs: self
                .catalog_program_sha256
                .iter()
                .map(|p| by_sha[p].clone())
                .collect(),
        };
        ensure!(
            canonical_sha256(&original)? == self.catalog_trie_sha256,
            "original catalog binding drift"
        );
        let mut covered = BTreeSet::new();
        let mut provenance_hashes = BTreeSet::new();
        for row in &self.provenance {
            self.validate_candidate(row)?;
            ensure!(
                ids.contains(row.program.program_sha256.as_str()),
                "provenance outside support"
            );
            ensure!(
                provenance_hashes.insert(canonical_sha256(row)?),
                "duplicate provenance"
            );
            covered.insert(row.program.program_sha256.as_str());
        }
        ensure!(
            ids.difference(&catalog).all(|id| covered.contains(id)),
            "noncatalog program lacks TRAIN provenance"
        );
        ensure!(
            self.content_sha256()? == self.support_sha256,
            "frozen support hash mismatch"
        );
        Ok(())
    }

    /// Performs the from bytes operation and returns an error if it cannot be completed.
    pub fn from_bytes(bytes: &[u8], expected_support_sha256: &str) -> Result<Self> {
        let value: Self = serde_json::from_slice(bytes)?;
        ensure!(
            value.support_sha256 == expected_support_sha256,
            "caller-pinned support identity required"
        );
        value.verify()?;
        Ok(value)
    }
}
