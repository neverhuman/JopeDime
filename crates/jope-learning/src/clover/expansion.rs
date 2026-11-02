//! GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
//!
//! Search measurements here are adaptive inner-cut measurements, never qualified rewards.
use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{bail, ensure, Context, Result};
use jope_pipeline::{prepare::Cut, search, seeds};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeSet, path::Path};

pub mod config;

mod selfplay_ratchet;
pub use selfplay_ratchet::{
    run_selfplay_ratchet, RoundLeafLogProbFn, SelfPlayRatchetRequest, SelfPlayRoundReceipt,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `DiscoveryContext` is a record for the discovery context contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct DiscoveryContext {
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: String,
    /// Parent family recorded for this discovery context.
    pub parent_family: String,
    /// Must be issued by the frozen role manifest, never inferred from a dataset name.
    pub role: String,
    /// SHA-256 digest identifying the associated dataset.
    pub dataset_sha256: String,
    /// Cut recorded for this discovery context.
    pub cut: usize,
    /// SHA-256 digest identifying the associated training.
    pub training_sha256: String,
    /// SHA-256 digest identifying the associated inner holdout.
    pub inner_holdout_sha256: String,
    /// SHA-256 digest identifying the associated preprocessing.
    pub preprocessing_sha256: String,
    /// SHA-256 digest identifying the associated encoder input.
    pub encoder_input_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `TrainingRoster` is a record for the training roster contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct TrainingRoster {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated role manifest.
    pub role_manifest_sha256: String,
    /// Collection of contexts values carried by this training roster.
    pub contexts: Vec<DiscoveryContext>,
}

impl TrainingRoster {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "clover.discovery-roster/v1",
            "unknown roster schema"
        );
        ensure!(
            is_hex64(&self.role_manifest_sha256),
            "role manifest pin missing"
        );
        ensure!(!self.contexts.is_empty(), "empty discovery roster");
        let mut seen = BTreeSet::new();
        for c in &self.contexts {
            ensure!(
                c.role == "train",
                "discovery forbids validation or sealed families"
            );
            ensure!(!c.parent_family.is_empty(), "parent family missing");
            ensure!(
                seen.insert(&c.context_sha256),
                "duplicate discovery context"
            );
            ensure!(
                [
                    &c.context_sha256,
                    &c.dataset_sha256,
                    &c.training_sha256,
                    &c.inner_holdout_sha256,
                    &c.preprocessing_sha256,
                    &c.encoder_input_sha256,
                ]
                .into_iter()
                .all(|h| is_hex64(h)),
                "incomplete discovery input pins"
            );
            ensure!(
                c.training_sha256 != c.inner_holdout_sha256,
                "fit/holdout identity collision"
            );
        }
        Ok(())
    }

    /// Computes the canonical SHA-256 digest of this value.
    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }

    /// Verifies the bound evidence and returns an error when verification fails.
    pub fn verify_roles(&self, role_manifest: &[u8]) -> Result<()> {
        self.validate()?;
        ensure!(
            jope_pipeline::sha256_hex(role_manifest) == self.role_manifest_sha256,
            "frozen family role manifest changed"
        );
        let roles: Value = serde_json::from_slice(role_manifest)?;
        ensure!(
            roles["schema"] == "clover.discovery-family-roles/v1",
            "unknown family role schema"
        );
        for context in &self.contexts {
            ensure!(
                roles["families"][&context.parent_family] == "train",
                "family is not authorized for discovery"
            );
        }
        Ok(())
    }

    fn bound_context(&self, context: &str, cut: &Cut) -> Result<&DiscoveryContext> {
        self.validate()?;
        let c = self
            .contexts
            .iter()
            .find(|c| c.context_sha256 == context)
            .context("context outside frozen training roster")?;
        ensure!(c.cut == cut.index, "discovery cut mismatch");
        ensure!(
            c.training_sha256 == rows_digest(&cut.train_x, &cut.train_y)?,
            "training rows changed"
        );
        ensure!(
            c.inner_holdout_sha256 == rows_digest(&cut.hold_x, &cut.hold_y)?,
            "inner holdout rows changed"
        );
        Ok(c)
    }
}

/// Ordered, transformed rows with labels intact. The preprocessing pin lives in the roster.
pub fn rows_digest(x: &[Vec<f64>], y: &[f64]) -> Result<String> {
    ensure!(!x.is_empty() && x.len() == y.len(), "row/label alignment");
    let width = x[0].len();
    ensure!(
        width > 0
            && x.iter()
                .all(|row| row.len() == width && row.iter().all(|v| v.is_finite()))
            && y.iter().all(|v| v.is_finite()),
        "finite rectangular rows required"
    );
    canonical_sha256(&(x, y))
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `RetrievalSeed` is a record for the retrieval seed contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct RetrievalSeed {
    /// Canonical program or genome representation.
    pub genome: String,
    /// SHA-256 digest identifying the associated source context.
    pub source_context_sha256: String,
    /// SHA-256 digest identifying the associated measurement receipt.
    pub measurement_receipt_sha256: String,
}

/// Retrieval ancestry must also be in the frozen TRAIN roster. Receipts are
/// metadata pins; this interface does not qualify their measurements.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSeeds {
    /// Collection of seeds values carried by this retrieval seeds.
    pub seeds: Vec<RetrievalSeed>,
}

impl RetrievalSeeds {
    fn pipeline_seeds(&self, roster: &TrainingRoster, dataset: &str) -> Result<seeds::SeedsFile> {
        roster.validate()?;
        ensure!(
            !self.seeds.is_empty() && self.seeds.len() <= 1024,
            "bounded retrieval seeds required"
        );
        let mut genomes = Vec::new();
        for seed in &self.seeds {
            ensure!(
                roster
                    .contexts
                    .iter()
                    .any(|c| c.context_sha256 == seed.source_context_sha256),
                "retrieval ancestry outside training roster"
            );
            ensure!(
                is_hex64(&seed.measurement_receipt_sha256),
                "retrieval receipt pin missing"
            );
            let genome = canonical_genome(&seed.genome)?;
            genomes.push(seeds::SeedGenome {
                program_sha256: jope_pipeline::sha256_hex(genome.as_bytes()),
                genome,
                bank_entry_sha256: None,
                receipt_sha256: Some(seed.measurement_receipt_sha256.clone()),
            });
        }
        Ok(seeds::SeedsFile::new(Some(dataset.to_owned()), genomes))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `ScratchBudget` is a record for the scratch budget contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct ScratchBudget {
    /// Evaluations per arm recorded for this scratch budget.
    pub evaluations_per_arm: usize,
    /// Seconds per arm recorded for this scratch budget.
    pub seconds_per_arm: u64,
    /// Rng seed recorded for this scratch budget.
    pub rng_seed: u64,
}

#[derive(Clone, Copy)]
/// `ScratchRequest` is a record for the scratch request contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct ScratchRequest<'a> {
    /// Roster recorded for this scratch request.
    pub roster: &'a TrainingRoster,
    /// Role manifest recorded for this scratch request.
    pub role_manifest: &'a [u8],
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: &'a str,
    /// Cut recorded for this scratch request.
    pub cut: &'a Cut,
    /// Policy recorded for this scratch request.
    pub policy: &'a PolicySeeds,
    /// Retrieval recorded for this scratch request.
    pub retrieval: &'a RetrievalSeeds,
    /// Budget recorded for this scratch request.
    pub budget: &'a ScratchBudget,
}

/// Execute ordinary, policy-seeded and retrieval-seeded GP with the same actual
/// pipeline operators, rows, RNG seed, single worker and attempt/time caps.
/// Output must be fresh. This is an offline scratch API, not an adoption service.
pub fn compare_scratch(request: ScratchRequest<'_>, output: &Path) -> Result<Value> {
    let ScratchRequest {
        roster,
        role_manifest,
        context_sha256,
        cut,
        policy,
        retrieval,
        budget,
    } = request;
    roster.verify_roles(role_manifest)?;
    let context = roster.bound_context(context_sha256, cut)?;
    let offered = policy.pipeline_seeds(context)?;
    let retrieval_input = retrieval.pipeline_seeds(roster, &context.dataset_sha256)?;
    retrieval_input.check()?;
    let limits = search::Budget {
        evaluations: budget.evaluations_per_arm,
        seconds: budget.seconds_per_arm,
    };
    limits.check()?;
    ensure!(!output.exists(), "scratch comparison output must be fresh");
    let policy_inputs = seeds::admit(
        &offered
            .genomes
            .iter()
            .map(|g| (g.genome.clone(), seeds::FROM_SEEDS_FILE))
            .collect::<Vec<_>>(),
    )?;
    let retrieval_inputs = seeds::admit(
        &retrieval_input
            .genomes
            .iter()
            .map(|g| (g.genome.clone(), seeds::FROM_SEEDS_FILE))
            .collect::<Vec<_>>(),
    )?;
    ensure!(
        !retrieval_inputs.seeds.is_empty(),
        "retrieval has no admitted seeds"
    );
    let mut arms = Vec::new();
    for (name, initial) in [
        ("ordinary", Vec::new()),
        (
            "policy_seeded",
            policy_inputs
                .seeds
                .iter()
                .map(|s| s.genome.clone())
                .collect(),
        ),
        (
            "retrieval_seeded",
            retrieval_inputs
                .seeds
                .iter()
                .map(|s| s.genome.clone())
                .collect(),
        ),
    ] {
        let mut options = search::SearchOptions::new(limits, budget.rng_seed);
        options.seeds = initial;
        let results =
            search::run_cuts_with(std::slice::from_ref(cut), &options, &output.join(name))?;
        let result = &results[0];
        ensure!(
            result.successful_measurements > 0,
            "scratch arm has no measured candidates"
        );
        let mut operators = std::collections::BTreeMap::<&str, usize>::new();
        for observation in &result.observations {
            *operators.entry(observation.origin).or_default() += 1;
        }
        let consumed = options.seeds.iter().map(|g| json!({"genome":g,
            "consumed":result.observations.iter().any(|o| o.genome == *g && o.origin == search::INCUMBENT)})).collect::<Vec<_>>();
        arms.push(
            json!({"arm":name,"search":result,"observations":result.observations,
            "search_sha256":result.search_sha256,"observations_sha256":result.observations_sha256,
            "operator_counts":operators,"seed_consumption":consumed}),
        );
    }
    let equal_attempts = arms.iter().all(|a| {
        a["search"]["completed_evaluations"] == json!(budget.evaluations_per_arm)
            && a["search"]["stopped_by"] == search::STOPPED_BY_EVALUATIONS
    });
    let within_wall = arms.iter().all(|a| {
        a["search"]["seconds"]
            .as_f64()
            .is_some_and(|s| s <= budget.seconds_per_arm as f64)
    });
    Ok(
        json!({"schema":"clover.gp-scratch-comparison/v1","qualification":"scratch_only",
        "roster_sha256":roster.digest()?,"context_sha256":context_sha256,"budget":budget,
        "policy_batch_sha256":canonical_sha256(policy)?,"policy_snapshot":policy.snapshot,
        "pipeline_seed_input":offered,"retrieval_input_sha256":canonical_sha256(retrieval)?,
        "arms":arms,"equal_attempts":equal_attempts,"within_wall_cap":within_wall,
        "cpu_seconds":null,"equal_resource_comparison":false,
        "stop_policy":"no_extension_on_timeout_failure_or_missing_measurement",
        "budget_caveat":"in-process fits are nonpreemptive; bootstrap/seeds are seconds-exempt; external resource receipt required",
        "qualified_rewards":0,"h1_established":false,"policy_optimizer_updates":0}),
    )
}

/// Distinct canonical strings are only a syntactic novelty test. Equal prediction
/// bits on the same legal training probe collapse behavior; absence stays unknown.
pub fn novelty(
    genome: &str,
    catalog: &[String],
    predictions: Option<(&str, &[f64])>,
    catalog_behavior_sha256: &BTreeSet<String>,
) -> Result<Value> {
    let canonical = canonical_genome(genome)?;
    let catalog = catalog
        .iter()
        .map(|g| canonical_genome(g))
        .collect::<Result<BTreeSet<_>>>()?;
    let fingerprint = predictions
        .map(|(probe, values)| {
            ensure!(
                is_hex64(probe) && !values.is_empty() && values.iter().all(|v| v.is_finite()),
                "legal finite probe required"
            );
            let bits = values
                .iter()
                .map(|v| if *v == 0. { 0 } else { v.to_bits() })
                .collect::<Vec<_>>();
            canonical_sha256(&(probe, bits))
        })
        .transpose()?;
    Ok(
        json!({"program_sha256":jope_pipeline::sha256_hex(canonical.as_bytes()),
        "catalog_sha256":canonical_sha256(&catalog)?,"catalog_programs":catalog.len(),
        "catalog_behavior_sha256":canonical_sha256(catalog_behavior_sha256)?,
        "syntactically_novel":!catalog.contains(&canonical),"behavior_sha256":fingerprint,
        "novel_on_training_probe":fingerprint.as_ref().map(|h| !catalog_behavior_sha256.contains(h)),
        "qualification":"unqualified","semantic_equivalence":"not_established"}),
    )
}

#[derive(Clone, Copy)]
/// `SurrogateSeedRequest` is a record for the surrogate seed request contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SurrogateSeedRequest<'a> {
    /// Context recorded for this surrogate seed request.
    pub context: &'a DiscoveryContext,
    /// Snapshot recorded for this surrogate seed request.
    pub snapshot: &'a PolicySnapshot,
    /// Candidate genomes recorded for this surrogate seed request.
    pub candidate_genomes: &'a [String],
    /// Selector recorded for this surrogate seed request.
    pub selector: &'a dyn crate::pools::selectors::Selector,
    /// Proposal pool k recorded for this surrogate seed request.
    pub proposal_pool_k: usize,
    /// Eval budget b recorded for this surrogate seed request.
    pub eval_budget_b: usize,
}

/// Pre-screens a candidate genome pool (`proposal_pool_k -> eval_budget_b`) for a target
/// `DiscoveryContext` using a fitted `Selector` (such as `ContextualSurrogateQHat`) and packages
/// the selected genomes into a validated [`PolicySeeds`] batch under
/// [`SAMPLING_SURROGATE_FUNNEL_TOPB_V1`] with explicit [`SurrogateFunnelProvenance`].
pub fn select_seeds_with_surrogate_funnel(
    request: SurrogateSeedRequest<'_>,
) -> Result<(PolicySeeds, crate::pools::selectors::FunnelAllocation)> {
    let SurrogateSeedRequest {
        context,
        snapshot,
        candidate_genomes,
        selector,
        proposal_pool_k,
        eval_budget_b,
    } = request;
    ensure!(
        !candidate_genomes.is_empty(),
        "candidate_genomes must not be empty"
    );
    let mut sha_to_genome = std::collections::BTreeMap::new();
    let mut ordered_shas = Vec::with_capacity(candidate_genomes.len());
    for raw in candidate_genomes {
        let canonical = canonical_genome(raw)?;
        ensure!(
            canonical != search::BOOTSTRAP_GENOME,
            "candidate_genomes must not include bootstrap genome"
        );
        let sha = jope_pipeline::sha256_hex(canonical.as_bytes());
        sha_to_genome.entry(sha.clone()).or_insert(canonical);
        ordered_shas.push(sha);
    }
    let candidate_catalog_sha256 = canonical_sha256(&ordered_shas)?;
    let prog_refs: Vec<&str> = ordered_shas.iter().map(String::as_str).collect();
    let view = crate::pools::pool::PoolView {
        context_id: &context.context_sha256,
        parent_id: &context.parent_family,
        incumbent_q: None,
        programs: prog_refs,
        tokens: vec![None; ordered_shas.len()],
    };
    let funnel = crate::pools::selectors::pre_screen_funnel(
        &view,
        selector,
        proposal_pool_k,
        eval_budget_b,
    )?;
    let proposals: Vec<PolicyProposal> = funnel
        .selected_program_sha256
        .iter()
        .map(|sha| {
            let genome = sha_to_genome
                .get(sha)
                .expect("selected program sha came from candidate_genomes")
                .clone();
            PolicyProposal {
                genome,
                program_sha256: sha.clone(),
                draw_seed: 0,
                behavior_logprob: 0.0,
            }
        })
        .collect();
    let provenance = SurrogateFunnelProvenance {
        selector_name: selector.name().to_owned(),
        selector_rule: selector.rule(),
        candidate_catalog_sha256,
        allocation: funnel.clone(),
    };
    let batch_protocol_sha256 = canonical_sha256(&provenance)?;
    let seeds = PolicySeeds {
        schema: "clover.policy-seeds/v1".into(),
        snapshot: snapshot.clone(),
        context_sha256: context.context_sha256.clone(),
        encoder_input_sha256: context.encoder_input_sha256.clone(),
        sampling_protocol: SAMPLING_SURROGATE_FUNNEL_TOPB_V1.into(),
        batch_protocol_sha256,
        temperature: 0.0,
        proposals,
        surrogate_funnel: Some(provenance),
    };
    seeds.validate(context)?;
    Ok((seeds, funnel))
}

/// Configuration for closed-loop measured-tilt (`L_PI` / AWR) policy updates across self-play rounds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredTiltPolicyConfig {
    /// Positive temperature `tau > 0` scaling inner-cut advantages `(q - b_c) / tau`.
    pub temperature_tau: f64,
    /// Positive symmetric clipping bound `A_max > 0` on scaled advantages.
    pub advantage_clip: f64,
    /// Reference probability mass `q_new in (0, 1)` reserved for newly discovered archive leaves when `|S_new| > 0`.
    pub ref_new_support_mass: f64,
    /// Maximum policy probability mass `p_new_cap in [q_new, 1)` allocated to newly discovered archive leaves when `|S_new| > 0`.
    pub policy_new_support_cap: f64,
    /// Per-round step size `eta in (0, 1]` damping advantage logits (`u_i = eta * clip((q_i - b_c)/tau, -A_max, A_max)`).
    pub step_size_eta: f64,
}

impl MeasuredTiltPolicyConfig {
    /// Validates this value against its invariants and returns an error when validation fails.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature_tau.is_finite() && self.temperature_tau > 0.0,
            "temperature_tau must be finite and > 0"
        );
        ensure!(
            self.advantage_clip.is_finite() && self.advantage_clip > 0.0,
            "advantage_clip must be finite and > 0"
        );
        ensure!(
            self.ref_new_support_mass.is_finite()
                && self.ref_new_support_mass > 0.0
                && self.ref_new_support_mass < 1.0,
            "ref_new_support_mass must be in (0, 1)"
        );
        ensure!(
            self.policy_new_support_cap.is_finite()
                && self.policy_new_support_cap >= self.ref_new_support_mass
                && self.policy_new_support_cap < 1.0,
            "policy_new_support_cap must be in [ref_new_support_mass, 1)"
        );
        ensure!(
            self.step_size_eta.is_finite() && self.step_size_eta > 0.0 && self.step_size_eta <= 1.0,
            "step_size_eta must be in (0, 1]"
        );
        Ok(())
    }
}

/// Evaluates the closed-loop measured-tilt (`L_PI` / AWR) policy and reference leaf
/// log-probabilities over `S_base U S_new` together with the weighted archive cross-entropy
/// objective value (`crate::support::weighted_archive_ce`) against the undamped advantage-tilted
/// target distribution `w_i propto pi_ref(i) * exp(clip((q_i - b_c)/tau, -A_max, A_max))`.
pub fn measured_tilt_leaf_log_probs(
    base_leaf_q: &[f64],
    archive_leaf_q: &[f64],
    bootstrap_q: f64,
    cfg: &MeasuredTiltPolicyConfig,
) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    cfg.validate()?;
    ensure!(!base_leaf_q.is_empty(), "base_leaf_q must not be empty");
    ensure!(
        bootstrap_q.is_finite()
            && base_leaf_q.iter().all(|q| q.is_finite())
            && archive_leaf_q.iter().all(|q| q.is_finite()),
        "finite base, archive, and bootstrap q values required"
    );
    let log_softmax = |logits: &[f64]| -> Vec<f64> {
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + logits.iter().map(|&x| (x - m).exp()).sum::<f64>().ln();
        logits.iter().map(|&x| x - lse).collect()
    };
    let clipped_base_adv: Vec<f64> = base_leaf_q
        .iter()
        .map(|&q| {
            ((q - bootstrap_q) / cfg.temperature_tau).clamp(-cfg.advantage_clip, cfg.advantage_clip)
        })
        .collect();
    let base_logits: Vec<f64> = clipped_base_adv
        .iter()
        .map(|&a| cfg.step_size_eta * a)
        .collect();
    let pol_base_cond_lp = log_softmax(&base_logits);
    let ref_base_cond_lp = -(base_leaf_q.len() as f64).ln();

    let (pol_lp, ref_lp, target_adv) = if archive_leaf_q.is_empty() {
        (
            pol_base_cond_lp,
            vec![ref_base_cond_lp; base_leaf_q.len()],
            clipped_base_adv,
        )
    } else {
        let clipped_arch_adv: Vec<f64> = archive_leaf_q
            .iter()
            .map(|&q| {
                ((q - bootstrap_q) / cfg.temperature_tau)
                    .clamp(-cfg.advantage_clip, cfg.advantage_clip)
            })
            .collect();
        let arch_logits: Vec<f64> = clipped_arch_adv
            .iter()
            .map(|&a| cfg.step_size_eta * a)
            .collect();
        let pol_arch_cond_lp = log_softmax(&arch_logits);
        let ref_arch_cond_lp = -(archive_leaf_q.len() as f64).ln();
        let mean_pos_adv = archive_leaf_q
            .iter()
            .map(|&q| ((q - bootstrap_q) / cfg.temperature_tau).max(0.0))
            .sum::<f64>()
            / (archive_leaf_q.len() as f64);
        let p_new = (cfg.ref_new_support_mass
            + (cfg.policy_new_support_cap - cfg.ref_new_support_mass)
                * (cfg.step_size_eta * mean_pos_adv).tanh())
        .clamp(cfg.ref_new_support_mass, cfg.policy_new_support_cap);
        let p_base = 1.0 - p_new;
        let q_new = cfg.ref_new_support_mass;
        let q_base = 1.0 - q_new;

        let mut pol = Vec::with_capacity(base_leaf_q.len() + archive_leaf_q.len());
        let mut rlp = Vec::with_capacity(base_leaf_q.len() + archive_leaf_q.len());
        for &cl in &pol_base_cond_lp {
            pol.push(p_base.ln() + cl);
            rlp.push(q_base.ln() + ref_base_cond_lp);
        }
        for &cl in &pol_arch_cond_lp {
            pol.push(p_new.ln() + cl);
            rlp.push(q_new.ln() + ref_arch_cond_lp);
        }
        let mut all_adv = clipped_base_adv;
        all_adv.extend(clipped_arch_adv);
        (pol, rlp, all_adv)
    };

    // Compute normalized AWR target weights w_i propto pi_ref(i) * exp(clipped_adv_i)
    let target_logits: Vec<f64> = ref_lp
        .iter()
        .zip(&target_adv)
        .map(|(&rlp, &adv)| rlp + adv)
        .collect();
    let target_lp = log_softmax(&target_logits);
    let raw_w: Vec<f64> = target_lp.into_iter().map(f64::exp).collect();
    let w_sum: f64 = raw_w.iter().sum();
    let weights: Vec<f64> = raw_w.into_iter().map(|w| w / w_sum).collect();
    let ce_loss = crate::support::weighted_archive_ce(&pol_lp, &weights)?;
    Ok((pol_lp, ref_lp, ce_loss))
}

#[cfg(test)]
mod tests;
