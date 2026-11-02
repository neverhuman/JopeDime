use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PolicySnapshot` is a record for the policy snapshot contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct PolicySnapshot {
    /// Model class recorded for this policy snapshot.
    pub model_class: String,
    /// Context width recorded for this policy snapshot.
    pub context_width: usize,
    /// SHA-256 digest identifying the associated parameter.
    pub parameter_sha256: String,
    /// SHA-256 digest of the checkpoint bound to this evidence.
    pub checkpoint_sha256: String,
    /// SHA-256 digest identifying the associated code.
    pub code_sha256: String,
    /// SHA-256 digest identifying the associated tokenizer.
    pub tokenizer_sha256: String,
    /// SHA-256 digest identifying the associated support.
    pub support_sha256: String,
    /// SHA-256 digest identifying the associated normalizer.
    pub normalizer_sha256: String,
}

impl PolicySnapshot {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_class == "PolicyLM" && self.context_width == 2051,
            "GB-06 requires current PolicyLM/Kumo-2051, not the legacy policy"
        );
        ensure!(
            [
                &self.parameter_sha256,
                &self.checkpoint_sha256,
                &self.code_sha256,
                &self.tokenizer_sha256,
                &self.support_sha256,
                &self.normalizer_sha256,
            ]
            .into_iter()
            .all(|h| is_hex64(h)),
            "incomplete policy snapshot pins"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PolicyProposal` is a record for the policy proposal contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct PolicyProposal {
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Draw seed recorded for this policy proposal.
    pub draw_seed: u64,
    /// Likelihood of the original draw under the declared behavior sampler.
    /// It is retained as provenance, never assigned to evolved descendants.
    pub behavior_logprob: f64,
}

/// Constant identifying the sampling trie temperature v1.
pub const SAMPLING_TRIE_TEMPERATURE_V1: &str = "trie_children_temperature_v1";
/// Constant identifying the sampling surrogate funnel topb v1.
pub const SAMPLING_SURROGATE_FUNNEL_TOPB_V1: &str = "surrogate_funnel_topb_v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `SurrogateFunnelProvenance` is a record for the surrogate funnel provenance contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SurrogateFunnelProvenance {
    /// Selector name recorded for this surrogate funnel provenance.
    pub selector_name: String,
    /// Selector rule recorded for this surrogate funnel provenance.
    pub selector_rule: String,
    /// SHA-256 digest identifying the associated candidate catalog.
    pub candidate_catalog_sha256: String,
    /// Allocation recorded for this surrogate funnel provenance.
    pub allocation: crate::pools::selectors::FunnelAllocation,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `PolicySeeds` is a record for the policy seeds contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct PolicySeeds {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Snapshot recorded for this policy seeds.
    pub snapshot: PolicySnapshot,
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// SHA-256 digest identifying the associated encoder input.
    pub encoder_input_sha256: String,
    /// Sampling protocol recorded for this policy seeds.
    pub sampling_protocol: String,
    /// SHA-256 digest identifying the associated batch protocol.
    pub batch_protocol_sha256: String,
    /// Temperature recorded for this policy seeds.
    pub temperature: f64,
    /// Original draws in order; duplicates remain here for honest draw accounting.
    pub proposals: Vec<PolicyProposal>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional surrogate funnel value; `None` means it is not available.
    pub surrogate_funnel: Option<SurrogateFunnelProvenance>,
}

/// Performs the canonical genome operation and returns an error if it cannot be completed.
pub fn canonical_genome(raw: &str) -> Result<String> {
    // The production pipeline's parser/canonicalizer. One item avoids its top-K cap
    // skipping validation of later inputs.
    let admitted = seeds::admit(&[(raw.to_owned(), seeds::FROM_SEEDS_FILE)])?;
    Ok(admitted
        .seeds
        .first()
        .map(|s| s.genome.clone())
        .unwrap_or_else(|| search::BOOTSTRAP_GENOME.to_owned()))
}

impl PolicySeeds {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self, context: &DiscoveryContext) -> Result<()> {
        ensure!(
            self.schema == "clover.policy-seeds/v1",
            "unknown policy seed schema"
        );
        self.snapshot.validate()?;
        ensure!(
            context.role == "train",
            "policy seeds require a training context"
        );
        ensure!(
            self.context_sha256 == context.context_sha256
                && self.encoder_input_sha256 == context.encoder_input_sha256,
            "policy context identity mismatch"
        );
        match self.sampling_protocol.as_str() {
            SAMPLING_TRIE_TEMPERATURE_V1 => {
                ensure!(
                    self.surrogate_funnel.is_none(),
                    "trie_children_temperature_v1 must not carry surrogate_funnel provenance"
                );
                ensure!(
                    is_hex64(&self.batch_protocol_sha256),
                    "decoder batch protocol pin missing"
                );
                ensure!(
                    self.temperature.is_finite() && (0.1..=2.0).contains(&self.temperature),
                    "invalid temperature"
                );
                ensure!(
                    !self.proposals.is_empty() && self.proposals.len() <= 1024,
                    "bounded policy draws required"
                );
                for p in &self.proposals {
                    ensure!(p.draw_seed < 1 << 63, "decoder seed outside CPU RNG domain");
                    ensure!(
                        p.behavior_logprob.is_finite() && p.behavior_logprob <= 0.,
                        "invalid behavior likelihood"
                    );
                    ensure!(
                        p.program_sha256 == jope_pipeline::sha256_hex(p.genome.as_bytes())
                            && canonical_genome(&p.genome)? == p.genome,
                        "policy program must be canonical and hash-bound"
                    );
                }
            }
            SAMPLING_SURROGATE_FUNNEL_TOPB_V1 => {
                let prov = self
                    .surrogate_funnel
                    .as_ref()
                    .context("surrogate_funnel_topb_v1 requires surrogate_funnel provenance")?;
                ensure!(
                    !prov.selector_name.trim().is_empty() && !prov.selector_rule.trim().is_empty(),
                    "surrogate_funnel provenance requires non-empty selector_name and selector_rule"
                );
                ensure!(
                    is_hex64(&prov.candidate_catalog_sha256),
                    "surrogate_funnel candidate_catalog_sha256 must be lowercase 64-hex"
                );
                ensure!(
                    prov.allocation.context_id == self.context_sha256,
                    "surrogate_funnel allocation context_id mismatch"
                );
                ensure!(
                    self.batch_protocol_sha256 == canonical_sha256(prov)?,
                    "surrogate_funnel batch_protocol_sha256 must equal canonical_sha256(surrogate_funnel)"
                );
                ensure!(
                    self.temperature == 0.0,
                    "deterministic surrogate_funnel_topb_v1 requires temperature == 0.0"
                );
                ensure!(
                    !self.proposals.is_empty() && self.proposals.len() <= 1024,
                    "bounded policy draws required"
                );
                ensure!(
                    prov.allocation.selected_program_sha256.len() == self.proposals.len()
                        && prov.allocation.selected_scores.len() == self.proposals.len()
                        && prov
                            .allocation
                            .selected_scores
                            .iter()
                            .all(|s| s.is_finite()),
                    "surrogate_funnel allocation lengths/scores must match proposals"
                );
                for (idx, p) in self.proposals.iter().enumerate() {
                    ensure!(
                        p.draw_seed == 0 && p.behavior_logprob == 0.0,
                        "deterministic surrogate_funnel_topb_v1 requires draw_seed == 0 and behavior_logprob == 0.0"
                    );
                    ensure!(
                        p.program_sha256 == prov.allocation.selected_program_sha256[idx],
                        "proposal {idx} program_sha256 does not match surrogate_funnel allocation"
                    );
                    ensure!(
                        p.program_sha256 == jope_pipeline::sha256_hex(p.genome.as_bytes())
                            && canonical_genome(&p.genome)? == p.genome,
                        "policy program must be canonical and hash-bound"
                    );
                }
            }
            _ => bail!("undeclared behavior sampler"),
        }
        Ok(())
    }

    /// Exact input accepted by `jope pipeline run --seeds`. Every admitted seed is
    /// fitted again on this run's own cuts. No proposal/q-head score is transferred.
    pub fn pipeline_seeds(&self, context: &DiscoveryContext) -> Result<seeds::SeedsFile> {
        self.validate(context)?;
        let offered = self
            .proposals
            .iter()
            .map(|p| (p.genome.clone(), seeds::FROM_SEEDS_FILE))
            .collect::<Vec<_>>();
        let admitted = seeds::admit(&offered)?;
        ensure!(
            !admitted.seeds.is_empty(),
            "policy supplied only bootstrap duplicates"
        );
        Ok(seeds::SeedsFile::new(
            Some(context.dataset_sha256.clone()),
            admitted
                .seeds
                .into_iter()
                .map(|s| seeds::SeedGenome {
                    genome: s.genome,
                    program_sha256: s.program_sha256,
                    bank_entry_sha256: None,
                    receipt_sha256: None,
                })
                .collect(),
        ))
    }

    /// The older engine consumes exactly 80 distinct programs. Caller supplies the
    /// independent/retrieval slots. Parent weights stay null so the intervention is
    /// population seeding alone; original behavior likelihoods remain in this batch.
    pub fn engine_population(
        &self,
        context: &DiscoveryContext,
        prepared_sha256: &str,
        training_csv_sha256: &str,
        independent: &[String],
    ) -> Result<Value> {
        self.validate(context)?;
        ensure!(
            is_hex64(prepared_sha256) && is_hex64(training_csv_sha256),
            "engine input pins missing"
        );
        let mut seen = BTreeSet::new();
        let mut candidates = Vec::new();
        let batch_sha = canonical_sha256(self)?;
        for (genome, origin, record) in self
            .proposals
            .iter()
            .map(|p| (&p.genome, "neural_policy", Some(batch_sha.as_str())))
            .chain(independent.iter().map(|g| (g, "independent", None)))
        {
            let canonical = canonical_genome(genome)?;
            ensure!(canonical == *genome, "seed adaptation must be explicit");
            if seen.insert(canonical.clone()) && candidates.len() < 80 {
                candidates.push(json!({"genome":canonical,"origin":origin,
                    "policy_logprob":null,"proposal_record_sha256":record}));
            }
        }
        ensure!(
            candidates.len() == 80 && candidates.iter().any(|c| c["origin"] == "independent"),
            "80 distinct seeds including independent discovery required"
        );
        Ok(
            json!({"schema":"parallel.seed_population.v1","prepared_sha256":prepared_sha256,
            "cut":context.cut,"training_sha256":training_csv_sha256,
            "context_sha256":context.context_sha256,"checkpoint_sha256":self.snapshot.checkpoint_sha256,
            "candidates":candidates}),
        )
    }
}
