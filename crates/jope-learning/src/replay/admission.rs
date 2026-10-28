//! State adapter for the shared, controller-adopted replay admission verifier.
//! Public wire types and admission rules are defined once in jope-records.
use crate::revocation::Revocations;
use crate::state::State;
use anyhow::Result;
use std::path::Path;

mod component_authority;
pub use component_authority::component_authority_snapshot;

pub use jope_records::replay_admission::{
    AdmissionPolicy, ArtifactStore, ComponentAuthoritySnapshot, ExactValue, ExposureClosure,
    ExposureReceipt, FitAdmission, Label, LabelEligibility, Observation, PolicyPin, Purpose,
    QualityLabel, ReplayAdmission, RevocationSnapshot, RuntimeLabel, TimeoutCause,
    ADMISSION_SCHEMA, ADOPTION_PATH, ADOPTION_SCHEMA, COMPONENT_AUTHORITY_PATH,
    COMPONENT_AUTHORITY_SCHEMA, EXPOSURE_SCHEMA, POLICY_SCHEMA,
};

/// Preserve the existing State-bound learning API without duplicating admission rules.
pub type AdmissionVerifier<'a> = jope_records::replay_admission::AdmissionVerifier<'a, State>;

impl RevocationSnapshot for Revocations {
    fn refuse(&self, what: &str, digests: &[&str]) -> Result<()> {
        Revocations::refuse(self, what, digests.iter().copied())
    }
}

impl ArtifactStore for State {
    type StoreId = Path;
    type Revocations = Revocations;

    fn store_id(&self) -> &Path {
        self.root()
    }
    fn adopted_policy_bytes(&self) -> Result<Vec<u8>> {
        Ok(std::fs::read(self.root().join(ADOPTION_PATH))?)
    }
    fn read_blob(&self, digest: &str) -> Result<Option<Vec<u8>>> {
        State::read_blob(self, digest)
    }
    fn revocations(&self) -> Result<Revocations> {
        Revocations::load(self)
    }
}
