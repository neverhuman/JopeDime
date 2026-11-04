use super::*;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MotifCandidate {
    pub(super) table_sha256: String,
    pub(super) motif_id: String,
    pub(super) motif_name: String,
    pub(super) expansion: String,
    pub(super) index: u32,
    pub(super) support_families: u32,
}

impl MotifCandidate {
    /// Get a candidate only from an internally consistent, still-quarantined table.
    pub fn from_table(table: &MotifTable, id: &str) -> Result<Self> {
        table
            .verify()
            .map_err(|_| AdmissionError("candidate_table_invalid"))?;
        let (index, motif) = table
            .body
            .motifs
            .iter()
            .enumerate()
            .find(|(_, motif)| motif.id == id)
            .ok_or(AdmissionError("candidate_not_in_table"))?;
        if motif.lifecycle != Lifecycle::Quarantined
            || motif.promotion != Promotion::required_not_run()
        {
            return Err(AdmissionError("candidate_not_quarantined"));
        }
        Ok(Self {
            table_sha256: table.table_sha256.clone(),
            motif_id: motif.id.clone(),
            motif_name: motif.name.clone(),
            expansion: motif.expansion.clone(),
            index: u32::try_from(index).map_err(|_| AdmissionError("candidate_index_invalid"))?,
            support_families: motif.support.families.unwrap_or(0),
        })
    }

    // Used by this module's focused integration target to exercise the pure gate without
    // manufacturing an authenticated source table. Production callers use `from_table`.
    #[cfg(test)]
    pub(crate) fn draft_for_gate(
        table_sha256: String,
        motif_id: String,
        motif_name: String,
        expansion: String,
        index: u32,
    ) -> Result<Self> {
        if !is_hex64(&table_sha256)
            || motif_id != motif_id_fn(&expansion)
            || motif_name != motif_name_fn(&motif_id)
            || expansion.contains(NAME_PREFIX)
        {
            return Err(AdmissionError("candidate_identity_invalid"));
        }
        Ok(Self {
            table_sha256,
            motif_id,
            motif_name,
            expansion,
            index,
            support_families: 2,
        })
    }

    pub fn table_sha256(&self) -> &str {
        &self.table_sha256
    }

    pub fn motif_id(&self) -> &str {
        &self.motif_id
    }

    pub fn motif_name(&self) -> &str {
        &self.motif_name
    }

    pub fn expansion(&self) -> &str {
        &self.expansion
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpansionLimits {
    /// Maximum expanded transform stages; the table format itself caps this at six.
    pub max_expansion_stages: usize,
    /// Maximum nested motif references inside an expansion. V1 requires zero.
    pub max_nested_macros: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpansionWitness {
    pub(super) table_sha256: String,
    pub(super) motif_id: String,
    pub(super) recipe_sha256: String,
    pub(super) stage_offset: u32,
    pub(super) rewritten_sha256: String,
    pub(super) expanded_sha256: String,
    pub(super) expanded_stages: u32,
    pub(super) nested_macros: u32,
}

impl ExpansionWitness {
    /// Reconstruct the macro rewrite and expansion independently, then demand byte-exact host
    /// equality. Finite-probe similarity is deliberately not an input to this proof.
    pub fn verify_host(
        candidate: &MotifCandidate,
        host: &str,
        stage_offset: usize,
        limits: ExpansionLimits,
    ) -> Result<Self> {
        if limits.max_expansion_stages == 0
            || limits.max_expansion_stages > 6
            || limits.max_nested_macros != 0
            || !is_hex64(&candidate.table_sha256)
            || candidate.motif_id != motif_id_fn(&candidate.expansion)
            || candidate.motif_name != motif_name_fn(&candidate.motif_id)
        {
            return Err(AdmissionError("expansion_limits_invalid"));
        }
        let expansion = fragment::expansion_stages(&candidate.expansion)
            .map_err(|_| AdmissionError("expansion_not_canonical"))?;
        if expansion.is_empty() || expansion.len() > limits.max_expansion_stages {
            return Err(AdmissionError("expansion_stage_limit"));
        }
        if candidate.expansion.contains(NAME_PREFIX) {
            return Err(AdmissionError("nested_macro_refused"));
        }
        let host_ir = GenomeIr::parse(host).map_err(|_| AdmissionError("host_not_canonical"))?;
        if host_ir.canonical() != host || host.contains(NAME_PREFIX) {
            return Err(AdmissionError("host_not_canonical"));
        }
        let parsed = parse_dsl(host).map_err(|_| AdmissionError("host_not_canonical"))?;
        let host_texts = stage_texts(&parsed.stages);
        let end = stage_offset
            .checked_add(expansion.len())
            .ok_or(AdmissionError("expansion_offset_invalid"))?;
        if host_texts.get(stage_offset..end) != Some(stage_texts(&expansion).as_slice()) {
            return Err(AdmissionError("expansion_occurrence_missing"));
        }

        let mut rewritten_stages = parsed.stages.clone();
        rewritten_stages.splice(
            stage_offset..end,
            [Stage::new(candidate.motif_name.clone())],
        );
        let rewritten = DslExpr {
            stages: rewritten_stages,
        }
        .to_string();
        let rewritten_parsed =
            parse_dsl(&rewritten).map_err(|_| AdmissionError("rewrite_invalid"))?;
        if rewritten_parsed
            .stages
            .iter()
            .filter(|stage| stage.op == candidate.motif_name)
            .count()
            != 1
        {
            return Err(AdmissionError("rewrite_invalid"));
        }
        let mut expanded_stages = Vec::new();
        let mut nested_macros = 0_u32;
        for stage in rewritten_parsed.stages {
            if stage.op == candidate.motif_name {
                nested_macros = nested_macros.saturating_add(1);
                expanded_stages.extend(expansion.iter().cloned());
            } else if stage.op.starts_with(NAME_PREFIX) {
                return Err(AdmissionError("nested_macro_refused"));
            } else {
                expanded_stages.push(stage);
            }
        }
        if nested_macros != 1 {
            return Err(AdmissionError("rewrite_invalid"));
        }
        let expanded = DslExpr {
            stages: expanded_stages,
        }
        .to_string();
        let expanded_ir =
            GenomeIr::parse(&expanded).map_err(|_| AdmissionError("expansion_not_canonical"))?;
        if expanded_ir.canonical() != host || expanded.contains(NAME_PREFIX) {
            return Err(AdmissionError("expansion_not_exact"));
        }
        let recipe_sha256 = sha256(host.as_bytes());
        Ok(Self {
            table_sha256: candidate.table_sha256.clone(),
            motif_id: candidate.motif_id.clone(),
            recipe_sha256,
            stage_offset: u32::try_from(stage_offset)
                .map_err(|_| AdmissionError("expansion_offset_invalid"))?,
            rewritten_sha256: sha256(rewritten.as_bytes()),
            expanded_sha256: sha256(expanded.as_bytes()),
            expanded_stages: u32::try_from(expansion.len())
                .map_err(|_| AdmissionError("expansion_stage_limit"))?,
            nested_macros,
        })
    }

    pub fn expanded_stages(&self) -> u32 {
        self.expanded_stages
    }

    pub fn nested_macros(&self) -> u32 {
        self.nested_macros
    }
}

/// Similarity on a finite set of probes. This diagnostic cannot be converted to an admission
/// witness and never establishes semantic equivalence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProbeSimilarity {
    pub probe_count: u32,
    pub max_absolute_error: f64,
    pub within_tolerance: bool,
}

impl ProbeSimilarity {
    pub fn compare(left: &[f64], right: &[f64], tolerance: f64) -> Result<Self> {
        if left.is_empty()
            || left.len() != right.len()
            || !tolerance.is_finite()
            || tolerance < 0.0
            || left.iter().chain(right).any(|value| !value.is_finite())
        {
            return Err(AdmissionError("probe_input_invalid"));
        }
        let max_absolute_error = left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if !max_absolute_error.is_finite() {
            return Err(AdmissionError("probe_input_invalid"));
        }
        Ok(Self {
            probe_count: u32::try_from(left.len())
                .map_err(|_| AdmissionError("probe_input_invalid"))?,
            max_absolute_error,
            within_tolerance: max_absolute_error <= tolerance,
        })
    }
}
