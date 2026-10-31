//! GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
//!
//! Search measurements here are adaptive inner-cut measurements, never qualified rewards.
use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{bail, ensure, Context, Result};
use jope_pipeline::{prepare::Cut, search, seeds};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

pub mod config;

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

/// Per-round policy and reference log-probability evaluator over the actual runtime leaf set:
/// `(round_index, ordered_base_genomes, ordered_archive_genomes) -> (policy_leaf_log_probs, reference_leaf_log_probs)`
/// where each returned vector must have length `ordered_base_genomes.len() + ordered_archive_genomes.len()`
/// in the exact order `ordered_base_genomes` followed by `ordered_archive_genomes`.
pub type RoundLeafLogProbFn<'a> = &'a dyn Fn(usize, &[String], &[String]) -> (Vec<f64>, Vec<f64>);

#[derive(Clone, Copy)]
/// `SelfPlayRatchetRequest` is a record for the self play ratchet request contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SelfPlayRatchetRequest<'a> {
    /// Roster recorded for this self play ratchet request.
    pub roster: &'a TrainingRoster,
    /// Role manifest recorded for this self play ratchet request.
    pub role_manifest: &'a [u8],
    /// SHA-256 digest identifying the associated context.
    pub context_sha256: &'a str,
    /// Cut recorded for this self play ratchet request.
    pub cut: &'a Cut,
    /// Initial policy recorded for this self play ratchet request.
    pub initial_policy: &'a PolicySeeds,
    /// Initial retrieval recorded for this self play ratchet request.
    pub initial_retrieval: &'a RetrievalSeeds,
    /// Round budgets recorded for this self play ratchet request.
    pub round_budgets: &'a [ScratchBudget],
    /// Seed cap b recorded for this self play ratchet request.
    pub seed_cap_b: usize,
    /// Upper bound on conditional base KL.
    pub max_conditional_base_kl: f64,
    /// Upper bound on new support mass.
    pub max_new_support_mass: f64,
    /// Optional round leaf log prob fn value; `None` means it is not available.
    pub round_leaf_log_prob_fn: Option<RoundLeafLogProbFn<'a>>,
    /// Optional measured tilt policy value; `None` means it is not available.
    pub measured_tilt_policy: Option<MeasuredTiltPolicyConfig>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `SelfPlayRoundReceipt` is a record for the self play round receipt contract in this module: GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
pub struct SelfPlayRoundReceipt {
    /// Round recorded for this self play round receipt.
    pub round: usize,
    /// Evaluations per arm recorded for this self play round receipt.
    pub evaluations_per_arm: usize,
    /// Rng seed recorded for this self play round receipt.
    pub rng_seed: u64,
    /// Bootstrap anchor source recorded for this self play round receipt.
    pub bootstrap_anchor_source: String,
    /// Active support leaves recorded for this self play round receipt.
    pub active_support_leaves: usize,
    /// Collection of base leaf genomes values carried by this self play round receipt.
    pub base_leaf_genomes: Vec<String>,
    /// Collection of archive leaf genomes values carried by this self play round receipt.
    pub archive_leaf_genomes: Vec<String>,
    /// Newly admitted winners recorded for this self play round receipt.
    pub newly_admitted_winners: usize,
    /// Collection of seeded genomes values carried by this self play round receipt.
    pub seeded_genomes: Vec<String>,
    /// Collection of seed consumption values carried by this self play round receipt.
    pub seed_consumption: Vec<Value>,
    /// Ordinary best rmse recorded for this self play round receipt.
    pub ordinary_best_rmse: f64,
    /// Ordinary best q recorded for this self play round receipt.
    pub ordinary_best_q: f64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    /// Ordinary champion genome recorded for this self play round receipt.
    pub ordinary_champion_genome: String,
    /// Ordinary evals to best recorded for this self play round receipt.
    pub ordinary_evals_to_best: usize,
    /// SHA-256 digest identifying the associated ordinary search.
    pub ordinary_search_sha256: String,
    /// SHA-256 digest identifying the associated ordinary observations.
    pub ordinary_observations_sha256: String,
    /// Ratchet best rmse recorded for this self play round receipt.
    pub ratchet_best_rmse: f64,
    /// Ratchet best q recorded for this self play round receipt.
    pub ratchet_best_q: f64,
    /// Ratchet champion genome recorded for this self play round receipt.
    pub ratchet_champion_genome: String,
    /// SHA-256 digest identifying the associated ratchet search.
    pub ratchet_search_sha256: String,
    /// SHA-256 digest identifying the associated ratchet observations.
    pub ratchet_observations_sha256: String,
    /// Surplus q vs ordinary recorded for this self play round receipt.
    pub surplus_q_vs_ordinary: f64,
    /// Optional evals to match ordinary final value; `None` means it is not available.
    pub evals_to_match_ordinary_final: Option<usize>,
    /// Optional speedup vs ordinary value; `None` means it is not available.
    pub speedup_vs_ordinary: Option<f64>,
    /// Equal attempts in round recorded for this self play round receipt.
    pub equal_attempts_in_round: bool,
    /// Within wall cap recorded for this self play round receipt.
    pub within_wall_cap: bool,
    /// Cumulative prior rounds recorded for this self play round receipt.
    pub cumulative_prior_rounds: usize,
    /// Equal resource across rounds recorded for this self play round receipt.
    pub equal_resource_across_rounds: bool,
    /// Kl source recorded for this self play round receipt.
    pub kl_source: String,
    /// Optional KL malformed reason value; `None` means it is not available.
    pub kl_malformed_reason: Option<String>,
    /// Optional policy archive CE loss value; `None` means it is not available.
    pub policy_archive_ce_loss: Option<f64>,
    /// Optional decomposed KL value; `None` means it is not available.
    pub decomposed_kl: Option<crate::support::DecomposedSupportKl>,
    /// Optional seed update frozen by KL guard value; `None` means it is not available.
    pub seed_update_frozen_by_kl_guard: Option<bool>,
    /// Next round seed selector recorded for this self play round receipt.
    pub next_round_seed_selector: String,
}

/// Execute a multi-round (`Round 0 -> Round 1 -> ...`) algorithmic self-play regression ratchet.
///
/// In Round 0, `selfplay_ratchet` is seeded from interleaved `initial_policy` and `initial_retrieval`
/// genomes admitted through `seeds::admit` (capped to `seed_cap_b <= search::ELITES`). After each
/// round `t`, newly discovered non-base genomes whose measured inner-cut RMSE strictly improves
/// over the Round-0 bootstrap baseline are admitted into the round archive `A_{t+1}`.
///
/// Policy KL guard evaluation supports two mutually exclusive runtime modes:
/// 1. `measured_tilt_policy: Some(cfg)`: Closed-form non-parametric measured-tilt (`L_PI` / AWR)
///    leaf policy over `S_base U S_new` derived from runtime measured inner-cut qualities
///    `base_best_q` and `archive_best_rmse`, recording `policy_archive_ce_loss`
///    (`weighted_archive_ce`), `decomposed_support_kl`, `measured_tilt_rounds_admissible`, and
///    selecting next-round seeds by descending tilted leaf log-probability when admissible
///    (`policy_optimizer_updates` remains `0` because no neural checkpoint parameters change).
/// 2. `round_leaf_log_prob_fn: Some(eval_fn)`: Custom caller callback invoked with
///    `(t, &base_leaf_genomes, &archive_leaf_genomes)`.
///
/// When both are `None`, no policy KL guard is evaluated
/// (`kl_source: "unmeasured_no_policy_log_probs"`, `decomposed_kl: None`,
/// `kl_guard_evaluated: false`, `all_rounds_kl_admissible: null`).
pub fn run_selfplay_ratchet(request: SelfPlayRatchetRequest<'_>, output: &Path) -> Result<Value> {
    let SelfPlayRatchetRequest {
        roster,
        role_manifest,
        context_sha256,
        cut,
        initial_policy,
        initial_retrieval,
        round_budgets,
        seed_cap_b,
        max_conditional_base_kl,
        max_new_support_mass,
        round_leaf_log_prob_fn,
        measured_tilt_policy,
    } = request;
    ensure!(
        round_budgets.len() >= 2,
        "self-play ratchet requires at least 2 rounds"
    );
    ensure!(
        (1..=search::ELITES).contains(&seed_cap_b),
        "seed_cap_b must be in 1..={}",
        search::ELITES
    );
    ensure!(
        max_conditional_base_kl.is_finite() && max_conditional_base_kl >= 0.0,
        "max_conditional_base_kl must be finite and >= 0"
    );
    ensure!(
        max_new_support_mass.is_finite() && (0.0..=1.0).contains(&max_new_support_mass),
        "max_new_support_mass must be in [0, 1]"
    );
    ensure!(
        !(round_leaf_log_prob_fn.is_some() && measured_tilt_policy.is_some()),
        "cannot specify both round_leaf_log_prob_fn and measured_tilt_policy"
    );
    if let Some(ref tilt_cfg) = measured_tilt_policy {
        tilt_cfg.validate()?;
    }
    roster.verify_roles(role_manifest)?;
    let context = roster.bound_context(context_sha256, cut)?;
    let offered = initial_policy.pipeline_seeds(context)?;
    let retrieval_input = initial_retrieval.pipeline_seeds(roster, &context.dataset_sha256)?;
    retrieval_input.check()?;
    ensure!(!output.exists(), "self-play ratchet output must be fresh");

    // Admit policy and retrieval seeds separately first so policy cannot starve retrieval.
    let admitted_policy = seeds::admit(
        &offered
            .genomes
            .iter()
            .map(|g| (g.genome.clone(), seeds::FROM_SEEDS_FILE))
            .collect::<Vec<_>>(),
    )?;
    let admitted_retrieval = seeds::admit(
        &retrieval_input
            .genomes
            .iter()
            .map(|g| (g.genome.clone(), seeds::FROM_SEEDS_FILE))
            .collect::<Vec<_>>(),
    )?;
    let mut interleaved_pairs = Vec::new();
    let max_len = admitted_policy
        .seeds
        .len()
        .max(admitted_retrieval.seeds.len());
    for i in 0..max_len {
        if let Some(p) = admitted_policy.seeds.get(i) {
            interleaved_pairs.push((p.genome.clone(), seeds::FROM_SEEDS_FILE));
        }
        if let Some(r) = admitted_retrieval.seeds.get(i) {
            interleaved_pairs.push((r.genome.clone(), seeds::FROM_SEEDS_FILE));
        }
    }
    let admitted_initial = seeds::admit(&interleaved_pairs)?;
    ensure!(
        !admitted_initial.seeds.is_empty(),
        "initial policy+retrieval seeds must not be empty"
    );

    let bootstrap_canonical = canonical_genome(search::BOOTSTRAP_GENOME)?;
    let mut base_catalog_ordered: Vec<String> = Vec::new();
    let mut base_catalog: BTreeSet<String> = BTreeSet::new();
    base_catalog.insert(bootstrap_canonical.clone());
    base_catalog_ordered.push(bootstrap_canonical);
    for s in &admitted_initial.seeds {
        if base_catalog.insert(s.genome.clone()) {
            base_catalog_ordered.push(s.genome.clone());
        }
    }
    let base_leaf_count = base_catalog_ordered.len();

    let mut base_best_q: std::collections::BTreeMap<String, f64> =
        std::collections::BTreeMap::new();

    // Archive maps non-base canonical genome (strictly beating bootstrap_rmse) -> best inner-cut RMSE.
    let mut archive_best_rmse: std::collections::BTreeMap<String, f64> =
        std::collections::BTreeMap::new();
    let mut bootstrap_rmse_anchor: Option<(f64, &'static str)> = None;

    let mut current_seeds: Vec<String> = admitted_initial
        .seeds
        .iter()
        .take(seed_cap_b)
        .map(|s| s.genome.clone())
        .collect();

    let mut round_receipts: Vec<SelfPlayRoundReceipt> = Vec::with_capacity(round_budgets.len());
    let mut measured_tilt_rounds_admissible = 0usize;

    for (round_idx, budget) in round_budgets.iter().enumerate() {
        let limits = search::Budget {
            evaluations: budget.evaluations_per_arm,
            seconds: budget.seconds_per_arm,
        };
        limits.check()?;
        let round_dir = output.join(format!("round_{round_idx}"));

        // 1. Run unseeded ordinary GP baseline for this round.
        let ordinary_opts = search::SearchOptions::new(limits, budget.rng_seed);
        let ordinary_res = search::run_cuts_with(
            std::slice::from_ref(cut),
            &ordinary_opts,
            &round_dir.join("ordinary"),
        )?;
        let ordinary = &ordinary_res[0];
        ensure!(
            ordinary.successful_measurements > 0,
            "ordinary arm produced no measured candidates in round {round_idx}"
        );
        let mut ordinary_best_rmse = f64::INFINITY;
        let mut ordinary_champion_genome = String::new();
        let mut ordinary_evals_to_best = 1usize;
        for (idx, obs) in ordinary.observations.iter().enumerate() {
            if obs.measured() {
                if let Some(r) = obs.rmse {
                    if r.is_finite()
                        && (ordinary_champion_genome.is_empty() || r + 1e-12 < ordinary_best_rmse)
                    {
                        ordinary_best_rmse = r;
                        ordinary_champion_genome = obs.genome.clone();
                        ordinary_evals_to_best = idx + 1;
                    }
                }
            }
        }
        ensure!(
            !ordinary_champion_genome.is_empty(),
            "ordinary arm produced no finite measured champion genome in round {round_idx}"
        );
        let ordinary_best_q = 1.0 / (1.0 + ordinary_best_rmse);

        // 2. Run self-play ratchet GP arm seeded with `current_seeds` (already admitted via seeds::admit).
        let mut ratchet_opts = search::SearchOptions::new(limits, budget.rng_seed);
        ratchet_opts.seeds = current_seeds.clone();
        let ratchet_res = search::run_cuts_with(
            std::slice::from_ref(cut),
            &ratchet_opts,
            &round_dir.join("selfplay_ratchet"),
        )?;
        let ratchet = &ratchet_res[0];
        ensure!(
            ratchet.successful_measurements > 0,
            "selfplay_ratchet arm produced no measured candidates in round {round_idx}"
        );

        if bootstrap_rmse_anchor.is_none() {
            let measured_boot = ratchet
                .observations
                .iter()
                .chain(ordinary.observations.iter())
                .find(|o| o.measured() && o.origin == search::BOOTSTRAP && o.rmse.is_some())
                .and_then(|o| o.rmse);
            bootstrap_rmse_anchor = Some(match measured_boot {
                Some(r) => (r, "measured_bootstrap"),
                None => (ordinary_best_rmse, "ordinary_best_fallback"),
            });
        }
        let (bootstrap_rmse, bootstrap_anchor_source) = bootstrap_rmse_anchor.unwrap();
        let bootstrap_q = 1.0 / (1.0 + bootstrap_rmse);

        let mut ratchet_best_rmse = f64::INFINITY;
        let mut ratchet_champion_genome = String::new();
        let mut evals_to_match_ordinary_final: Option<usize> = None;
        let mut newly_admitted = 0usize;

        for (eval_idx, obs) in ratchet.observations.iter().enumerate() {
            if !obs.measured() {
                continue;
            }
            if let Some(r) = obs.rmse {
                let q = 1.0 / (1.0 + r);
                if r < ratchet_best_rmse {
                    ratchet_best_rmse = r;
                    ratchet_champion_genome = obs.genome.clone();
                }
                if evals_to_match_ordinary_final.is_none() && r <= ordinary_best_rmse + 1e-12 {
                    evals_to_match_ordinary_final = Some(eval_idx + 1);
                }
                if base_catalog.contains(&obs.genome) {
                    base_best_q
                        .entry(obs.genome.clone())
                        .and_modify(|prev| {
                            if q > *prev {
                                *prev = q;
                            }
                        })
                        .or_insert(q);
                } else if r + 1e-12 < bootstrap_rmse {
                    if !archive_best_rmse.contains_key(&obs.genome) {
                        newly_admitted += 1;
                    }
                    archive_best_rmse
                        .entry(obs.genome.clone())
                        .and_modify(|prev| {
                            if r < *prev {
                                *prev = r;
                            }
                        })
                        .or_insert(r);
                }
            }
        }
        if round_idx == 0 {
            for g in &base_catalog_ordered {
                let q = base_best_q.get(g).copied().unwrap_or(bootstrap_q);
                base_best_q.entry(g.clone()).or_insert(q);
            }
        }

        let ratchet_best_q = 1.0 / (1.0 + ratchet_best_rmse);
        let surplus_q_vs_ordinary = ratchet_best_q - ordinary_best_q;
        let speedup_vs_ordinary =
            evals_to_match_ordinary_final.map(|k| ordinary_evals_to_best as f64 / k as f64);

        let equal_attempts_in_round = ordinary.completed_evaluations
            == budget.evaluations_per_arm as u64
            && ratchet.completed_evaluations == budget.evaluations_per_arm as u64
            && ordinary.stopped_by == search::STOPPED_BY_EVALUATIONS
            && ratchet.stopped_by == search::STOPPED_BY_EVALUATIONS;
        let within_wall_cap = ordinary.seconds <= budget.seconds_per_arm as f64
            && ratchet.seconds <= budget.seconds_per_arm as f64;
        let seed_consumption = current_seeds
            .iter()
            .map(|g| {
                json!({
                    "genome": g,
                    "consumed": ratchet.observations.iter().any(|o| o.genome == *g && o.origin == search::INCUMBENT),
                })
            })
            .collect::<Vec<_>>();

        // 3. Evaluate decomposed support KL if a policy evaluator or measured-tilt config was provided.
        let archive_leaf_genomes: Vec<String> = archive_best_rmse.keys().cloned().collect();
        let n_new = archive_leaf_genomes.len();
        let active_support_leaves = base_leaf_count + n_new;
        let mut is_base = Vec::with_capacity(active_support_leaves);
        is_base.extend(std::iter::repeat_n(true, base_leaf_count));
        is_base.extend(std::iter::repeat_n(false, n_new));

        let mut tilted_leaf_log_probs: Option<Vec<f64>> = None;
        let (
            kl_source,
            kl_malformed_reason,
            policy_archive_ce_loss,
            decomposed_kl,
            seed_update_frozen_by_kl_guard,
        ) = if let Some(eval_fn) = round_leaf_log_prob_fn {
            let (pol_lp, ref_lp) = eval_fn(round_idx, &base_catalog_ordered, &archive_leaf_genomes);
            if pol_lp.len() == active_support_leaves && ref_lp.len() == active_support_leaves {
                match crate::support::decomposed_support_kl(
                    &pol_lp,
                    &ref_lp,
                    &is_base,
                    max_conditional_base_kl,
                    max_new_support_mass,
                ) {
                    Ok(report) => {
                        let frozen = !report.admissible;
                        (
                            "caller_evaluated_leaf_log_probs".to_string(),
                            None,
                            None,
                            Some(report),
                            Some(frozen),
                        )
                    }
                    Err(err) => (
                        "caller_evaluated_leaf_log_probs_malformed".to_string(),
                        Some(format!("{err:#}")),
                        None,
                        None,
                        Some(true),
                    ),
                }
            } else {
                (
                    "caller_evaluated_leaf_log_probs_malformed".to_string(),
                    Some(format!(
                        "expected {} leaf log-probs, got policy={} ref={}",
                        active_support_leaves,
                        pol_lp.len(),
                        ref_lp.len()
                    )),
                    None,
                    None,
                    Some(true),
                )
            }
        } else if let Some(ref tilt_cfg) = measured_tilt_policy {
            let base_q_vec: Vec<f64> = base_catalog_ordered
                .iter()
                .map(|g| base_best_q.get(g).copied().unwrap_or(bootstrap_q))
                .collect();
            let arch_q_vec: Vec<f64> = archive_leaf_genomes
                .iter()
                .map(|g| 1.0 / (1.0 + archive_best_rmse[g]))
                .collect();
            match measured_tilt_leaf_log_probs(&base_q_vec, &arch_q_vec, bootstrap_q, tilt_cfg)
                .and_then(|(pol_lp, ref_lp, ce_loss)| {
                    let report = crate::support::decomposed_support_kl(
                        &pol_lp,
                        &ref_lp,
                        &is_base,
                        max_conditional_base_kl,
                        max_new_support_mass,
                    )?;
                    Ok((pol_lp, ce_loss, report))
                }) {
                Ok((pol_lp, ce_loss, report)) => {
                    let frozen = !report.admissible;
                    if report.admissible {
                        measured_tilt_rounds_admissible += 1;
                        tilted_leaf_log_probs = Some(pol_lp);
                    }
                    (
                        "measured_tilt_leaf_policy".to_string(),
                        None,
                        Some(ce_loss),
                        Some(report),
                        Some(frozen),
                    )
                }
                Err(err) => (
                    "measured_tilt_leaf_policy_malformed".to_string(),
                    Some(format!("{err:#}")),
                    None,
                    None,
                    Some(true),
                ),
            }
        } else {
            (
                "unmeasured_no_policy_log_probs".to_string(),
                None,
                None,
                None,
                None,
            )
        };

        // 4. Select top `seed_cap_b` archive + base winners through `seeds::admit` to seed Round `t+1`,
        // unless a policy KL guard tripped and froze seed updates.
        let next_round_seed_selector = if seed_update_frozen_by_kl_guard == Some(true) {
            "frozen_by_kl_guard".to_string()
        } else if let Some(ref pol_lp) = tilted_leaf_log_probs {
            // Rank eligible leaves in S_base U S_new by descending tilted leaf log-probability,
            // breaking ties by ascending measured RMSE and ascending genome.
            let current_seed_set: BTreeSet<&str> =
                current_seeds.iter().map(String::as_str).collect();
            let mut tilted_candidates: Vec<(String, f64, f64)> = Vec::new();
            for (idx, g) in base_catalog_ordered.iter().enumerate() {
                if g != search::BOOTSTRAP_GENOME && current_seed_set.contains(g.as_str()) {
                    let q = base_best_q.get(g).copied().unwrap_or(bootstrap_q);
                    let r = (1.0 / q) - 1.0;
                    tilted_candidates.push((g.clone(), pol_lp[idx], r));
                }
            }
            for (offset, g) in archive_leaf_genomes.iter().enumerate() {
                if g != search::BOOTSTRAP_GENOME {
                    let r = archive_best_rmse[g];
                    tilted_candidates.push((g.clone(), pol_lp[base_leaf_count + offset], r));
                }
            }
            tilted_candidates.sort_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| a.2.total_cmp(&b.2))
                    .then_with(|| a.0.cmp(&b.0))
            });
            let candidate_pairs: Vec<(String, &'static str)> = tilted_candidates
                .into_iter()
                .map(|(g, _, _)| (g, seeds::FROM_SEEDS_FILE))
                .collect();
            if !candidate_pairs.is_empty() {
                let admitted_next = seeds::admit(&candidate_pairs)?;
                let next_seeds: Vec<String> = admitted_next
                    .seeds
                    .into_iter()
                    .take(seed_cap_b)
                    .map(|s| s.genome)
                    .collect();
                if !next_seeds.is_empty() {
                    current_seeds = next_seeds;
                }
            }
            "measured_tilt_leaf_policy".to_string()
        } else {
            let mut ranked_candidates: Vec<(String, f64)> = archive_best_rmse
                .iter()
                .map(|(g, &r)| (g.clone(), r))
                .collect();
            for g in &current_seeds {
                if let Some(&q) = base_best_q.get(g) {
                    let r = (1.0 / q) - 1.0;
                    ranked_candidates.push((g.clone(), r));
                }
            }
            ranked_candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
            let candidate_pairs: Vec<(String, &'static str)> = ranked_candidates
                .into_iter()
                .filter(|(g, _)| g != search::BOOTSTRAP_GENOME)
                .map(|(g, _)| (g, seeds::FROM_SEEDS_FILE))
                .collect();
            if !candidate_pairs.is_empty() {
                let admitted_next = seeds::admit(&candidate_pairs)?;
                let next_seeds: Vec<String> = admitted_next
                    .seeds
                    .into_iter()
                    .take(seed_cap_b)
                    .map(|s| s.genome)
                    .collect();
                if !next_seeds.is_empty() {
                    current_seeds = next_seeds;
                }
            }
            "archive_best_rmse_greedy".to_string()
        };

        round_receipts.push(SelfPlayRoundReceipt {
            round: round_idx,
            evaluations_per_arm: budget.evaluations_per_arm,
            rng_seed: budget.rng_seed,
            bootstrap_anchor_source: bootstrap_anchor_source.to_string(),
            active_support_leaves,
            base_leaf_genomes: base_catalog_ordered.clone(),
            archive_leaf_genomes,
            newly_admitted_winners: newly_admitted,
            seeded_genomes: ratchet_opts.seeds,
            seed_consumption,
            ordinary_best_rmse,
            ordinary_best_q,
            ordinary_champion_genome,
            ordinary_evals_to_best,
            ordinary_search_sha256: ordinary.search_sha256.clone(),
            ordinary_observations_sha256: ordinary.observations_sha256.clone(),
            ratchet_best_rmse,
            ratchet_best_q,
            ratchet_champion_genome,
            ratchet_search_sha256: ratchet.search_sha256.clone(),
            ratchet_observations_sha256: ratchet.observations_sha256.clone(),
            surplus_q_vs_ordinary,
            evals_to_match_ordinary_final,
            speedup_vs_ordinary,
            equal_attempts_in_round,
            within_wall_cap,
            cumulative_prior_rounds: round_idx,
            equal_resource_across_rounds: round_idx == 0,
            kl_source,
            kl_malformed_reason,
            policy_archive_ce_loss,
            decomposed_kl,
            seed_update_frozen_by_kl_guard,
            next_round_seed_selector,
        });
    }

    let monotonic_non_decreasing_best_q = round_receipts
        .windows(2)
        .all(|w| w[1].ratchet_best_q + 1e-12 >= w[0].ratchet_best_q);
    let first_q = round_receipts
        .first()
        .map(|r| r.ratchet_best_q)
        .unwrap_or(0.0);
    let last_round = round_receipts.last().unwrap();
    let strictly_improved_over_round0 = last_round.ratchet_best_q > first_q + 1e-12;
    let beat_ordinary_in_final_round =
        last_round.ratchet_best_q > last_round.ordinary_best_q + 1e-12;
    let matched_or_beat_ordinary_in_final_round =
        last_round.ratchet_best_q + 1e-12 >= last_round.ordinary_best_q;
    let kl_guard_evaluated = round_leaf_log_prob_fn.is_some() || measured_tilt_policy.is_some();
    let all_rounds_kl_admissible: Option<bool> = kl_guard_evaluated.then(|| {
        round_receipts
            .iter()
            .all(|r| r.decomposed_kl.as_ref().is_some_and(|d| d.admissible))
    });
    let any_round_frozen_by_kl_guard: Option<bool> = kl_guard_evaluated.then(|| {
        round_receipts
            .iter()
            .any(|r| r.seed_update_frozen_by_kl_guard == Some(true))
    });
    let total_admitted_new_winners: usize = round_receipts
        .iter()
        .map(|r| r.newly_admitted_winners)
        .sum();

    Ok(json!({
        "schema": "clover.selfplay-ratchet/v1",
        "qualification": "scratch_only",
        "roster_sha256": roster.digest()?,
        "context_sha256": context_sha256,
        "seed_cap_b": seed_cap_b,
        "max_conditional_base_kl": max_conditional_base_kl,
        "max_new_support_mass": max_new_support_mass,
        "measured_tilt_policy": measured_tilt_policy,
        "rounds": round_receipts,
        "monotonic_non_decreasing_best_q": monotonic_non_decreasing_best_q,
        "strictly_improved_over_round0": strictly_improved_over_round0,
        "beat_ordinary_in_final_round": beat_ordinary_in_final_round,
        "matched_or_beat_ordinary_in_final_round": matched_or_beat_ordinary_in_final_round,
        "kl_guard_evaluated": kl_guard_evaluated,
        "all_rounds_kl_admissible": all_rounds_kl_admissible,
        "any_round_frozen_by_kl_guard": any_round_frozen_by_kl_guard,
        "total_admitted_new_winners": total_admitted_new_winners,
        "measured_tilt_rounds_admissible": measured_tilt_rounds_admissible,
        "equal_resource_comparison": false,
        "stop_policy": "no_extension_on_timeout_failure_or_missing_measurement",
        "budget_caveat": "in-process fits are nonpreemptive; bootstrap/seeds are seconds-exempt; round t>0 reuses prior rounds' inner-holdout archive champions",
        "qualified_rewards": 0,
        "h1_established": false,
        "policy_optimizer_updates": 0,
    }))
}

#[cfg(test)]
mod tests;
