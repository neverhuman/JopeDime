use super::*;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MotifAdmission {
    pub schema: String,
    pub table_sha256: String,
    pub motif_id: String,
    pub expansion_witnesses: Vec<ExpansionWitness>,
    pub contrast: ValidatedContrast,
    pub vocabulary: VocabularyMigration,
    pub validation_sha256: String,
}

impl MotifAdmission {
    /// Validate and admit a candidate using a fresh source read and the package's real motif
    /// validator. A report supplied by a caller is never sufficient by itself.
    pub fn qualify(
        table: &MotifTable,
        set: &QualifiedSet,
        id: &str,
        contrast: &PreregisteredContrast,
        boundary: &EvidenceBoundary,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        let candidate = MotifCandidate::from_table(table, id)?;
        if vocabulary.motif_id != candidate.motif_id
            || vocabulary.expansion_sha256 != sha256(candidate.expansion.as_bytes())
        {
            return Err(AdmissionError("vocabulary_version_invalid"));
        }
        let motif = table
            .body
            .motifs
            .get(candidate.index as usize)
            .ok_or(AdmissionError("candidate_not_in_table"))?;
        let limits = ExpansionLimits {
            max_expansion_stages: usize::try_from(table.body.parameters.max_stages)
                .map_err(|_| AdmissionError("expansion_limits_invalid"))?,
            max_nested_macros: 0,
        };
        let mut witnesses = Vec::with_capacity(motif.provenance.len());
        for occurrence in &motif.provenance {
            let host = table
                .body
                .hosts
                .get(&occurrence.recipe_sha256)
                .ok_or(AdmissionError("expansion_occurrence_missing"))?;
            witnesses.push(ExpansionWitness::verify_host(
                &candidate,
                host,
                usize::try_from(occurrence.stage_offset)
                    .map_err(|_| AdmissionError("expansion_offset_invalid"))?,
                limits,
            )?);
        }
        let mut validation = validate::validate(table, &ValidateOptions::default())
            .map_err(|_| AdmissionError("validation_failed"))?;
        if validate::verify_source(&mut validation, table, set).is_some() {
            return Err(AdmissionError("source_not_verified"));
        }
        let contrast = contrast.validate(&candidate, &vocabulary, boundary)?;
        Self::finish(&candidate, witnesses, &validation, contrast, vocabulary)
    }

    /// Shared final predicate; the live entry point above obtains these inputs itself. The
    /// test-only route exercises this gate without claiming an admitted source or campaign.
    pub(super) fn finish(
        candidate: &MotifCandidate,
        witnesses: Vec<ExpansionWitness>,
        validation: &Validation,
        contrast: ValidatedContrast,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        let result = validation
            .motifs
            .iter()
            .find(|result| result.index == candidate.index)
            .ok_or(AdmissionError("validation_not_exercised"))?;
        if validation.schema != VALIDATION_SCHEMA
            || validation.table_sha256 != candidate.table_sha256
            || !validation.passed()
            || !validation.source_verified
            || validation.source_digest_equal != Some(true)
            || !result.passed()
            || witnesses.is_empty()
            || candidate.support_families < 2
            || witnesses.iter().any(|witness| {
                witness.table_sha256 != candidate.table_sha256
                    || witness.motif_id != candidate.motif_id
                    || witness.expanded_sha256 != witness.recipe_sha256
                    || witness.expanded_stages == 0
                    || witness.nested_macros != 1
            })
            || contrast.motif_id != candidate.motif_id
            || contrast.new_profile_id != vocabulary.new_profile_id
            || contrast.new_profile_revision != vocabulary.new_profile_revision
            || vocabulary.schema != VOCABULARY_SCHEMA
            || vocabulary.motif_id != candidate.motif_id
            || vocabulary.expansion_sha256 != sha256(candidate.expansion.as_bytes())
            || !vocabulary.old_policy_expands_macro
        {
            return Err(AdmissionError("admission_evidence_incomplete"));
        }
        let validation_sha256 = sha256(
            &serde_json::to_vec(validation).map_err(|_| AdmissionError("validation_invalid"))?,
        );
        Ok(Self {
            schema: ADMISSION_SCHEMA.to_owned(),
            table_sha256: candidate.table_sha256.clone(),
            motif_id: candidate.motif_id.clone(),
            expansion_witnesses: witnesses,
            contrast,
            vocabulary,
            validation_sha256,
        })
    }

    #[cfg(test)]
    pub(crate) fn qualify_after_verified_source(
        candidate: &MotifCandidate,
        witnesses: Vec<ExpansionWitness>,
        validation: &Validation,
        contrast: ValidatedContrast,
        vocabulary: VocabularyMigration,
    ) -> Result<Self> {
        Self::finish(candidate, witnesses, validation, contrast, vocabulary)
    }
}
