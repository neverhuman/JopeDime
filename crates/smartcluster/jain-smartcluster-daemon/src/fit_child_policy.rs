//! Optional root-installed policy; absence closes all child placement admission.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedFitChildPolicy {
    pub tenant: String,
    pub project: String,
    pub authority_executable_sha256: String,
    pub evaluator_executable_sha256: BTreeSet<String>,
}
impl TrustedFitChildPolicy {
    pub fn validate(&self) -> Result<(), String> {
        jain_smartcluster_core::TenantId::new(self.tenant.clone()).map_err(|e| e.to_string())?;
        jain_smartcluster_core::ProjectId::new(self.project.clone()).map_err(|e| e.to_string())?;
        if self.evaluator_executable_sha256.is_empty()
            || self.evaluator_executable_sha256.len() > 16
        {
            return Err("fit child evaluator allowlist must contain1..16 qualified hashes".into());
        }
        for sha in std::iter::once(&self.authority_executable_sha256)
            .chain(self.evaluator_executable_sha256.iter())
        {
            if sha.len() != 64
                || !sha
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("fit child policy requires exact lowercase SHA256 identities".into());
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_policy_rejects_empty_and_non_digest_allowlists() {
        let mut p = TrustedFitChildPolicy {
            tenant: "jope".into(),
            project: "qualification".into(),
            authority_executable_sha256: "a".repeat(64),
            evaluator_executable_sha256: BTreeSet::new(),
        };
        assert!(p.validate().is_err());
        p.evaluator_executable_sha256.insert("b".repeat(64));
        assert!(p.validate().is_ok());
        p.authority_executable_sha256 = "A".repeat(64);
        assert!(p.validate().is_err());
    }
}
