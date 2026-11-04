use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct VocabularyMigration {
    pub(super) schema: String,
    pub(super) old_profile_id: String,
    pub(super) old_profile_revision: u32,
    pub(super) new_profile_id: String,
    pub(super) new_profile_revision: u32,
    pub(super) motif_id: String,
    pub(super) macro_name: String,
    pub(super) expansion_sha256: String,
    /// Old policies continue to use the expanded canonical text; only the new version admits
    /// the macro token.
    pub old_policy_expands_macro: bool,
}

impl VocabularyMigration {
    pub fn new(
        candidate: &MotifCandidate,
        old_profile_id: impl Into<String>,
        old_profile_revision: u32,
        new_profile_id: impl Into<String>,
        new_profile_revision: u32,
    ) -> Result<Self> {
        let old_profile_id = old_profile_id.into();
        let new_profile_id = new_profile_id.into();
        if old_profile_id.is_empty()
            || new_profile_id.is_empty()
            || old_profile_revision == 0
            || new_profile_revision == 0
            || (old_profile_id == new_profile_id && new_profile_revision <= old_profile_revision)
        {
            return Err(AdmissionError("vocabulary_version_invalid"));
        }
        Ok(Self {
            schema: VOCABULARY_SCHEMA.to_owned(),
            old_profile_id,
            old_profile_revision,
            new_profile_id,
            new_profile_revision,
            motif_id: candidate.motif_id.clone(),
            macro_name: candidate.motif_name.clone(),
            expansion_sha256: sha256(candidate.expansion.as_bytes()),
            old_policy_expands_macro: true,
        })
    }

    pub fn enables_macro(&self, profile_id: &str, revision: u32) -> bool {
        self.schema == VOCABULARY_SCHEMA
            && profile_id == self.new_profile_id
            && revision == self.new_profile_revision
    }

    pub fn old_profile(&self) -> (&str, u32) {
        (&self.old_profile_id, self.old_profile_revision)
    }

    pub fn new_profile(&self) -> (&str, u32) {
        (&self.new_profile_id, self.new_profile_revision)
    }

    pub fn old_policy_expands_macro(&self) -> bool {
        self.old_policy_expands_macro
    }
}
