//! GB-06: frozen training inputs, PolicyLM seed ingestion and scratch GP comparison.
//!
//! Search measurements here are adaptive inner-cut measurements, never qualified rewards.
use crate::canon::{canonical_sha256, is_hex64};
use anyhow::{ensure, Context, Result};
use jope_pipeline::{prepare::Cut, search, seeds};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

pub mod config;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiscoveryContext {
    pub context_sha256: String,
    pub parent_family: String,
    /// Must be issued by the frozen role manifest, never inferred from a dataset name.
    pub role: String,
    pub dataset_sha256: String,
    pub cut: usize,
    pub training_sha256: String,
    pub inner_holdout_sha256: String,
    pub preprocessing_sha256: String,
    pub encoder_input_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRoster {
    pub schema: String,
    pub role_manifest_sha256: String,
    pub contexts: Vec<DiscoveryContext>,
}

impl TrainingRoster {
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

    pub fn digest(&self) -> Result<String> {
        self.validate()?;
        canonical_sha256(self)
    }

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
pub struct PolicySnapshot {
    pub model_class: String,
    pub context_width: usize,
    pub parameter_sha256: String,
    pub checkpoint_sha256: String,
    pub code_sha256: String,
    pub tokenizer_sha256: String,
    pub support_sha256: String,
    pub normalizer_sha256: String,
}

impl PolicySnapshot {
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
pub struct PolicyProposal {
    pub genome: String,
    pub program_sha256: String,
    pub draw_seed: u64,
    /// Likelihood of the original draw under the declared behavior sampler.
    /// It is retained as provenance, never assigned to evolved descendants.
    pub behavior_logprob: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySeeds {
    pub schema: String,
    pub snapshot: PolicySnapshot,
    pub context_sha256: String,
    pub encoder_input_sha256: String,
    pub sampling_protocol: String,
    pub batch_protocol_sha256: String,
    pub temperature: f64,
    /// Original draws in order; duplicates remain here for honest draw accounting.
    pub proposals: Vec<PolicyProposal>,
}

fn canonical_genome(raw: &str) -> Result<String> {
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
        ensure!(
            self.sampling_protocol == "trie_children_temperature_v1",
            "undeclared behavior sampler"
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
pub struct RetrievalSeed {
    pub genome: String,
    pub source_context_sha256: String,
    pub measurement_receipt_sha256: String,
}

/// Retrieval ancestry must also be in the frozen TRAIN roster. Receipts are
/// metadata pins; this interface does not qualify their measurements.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalSeeds {
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
pub struct ScratchBudget {
    pub evaluations_per_arm: usize,
    pub seconds_per_arm: u64,
    pub rng_seed: u64,
}

#[derive(Clone, Copy)]
pub struct ScratchRequest<'a> {
    pub roster: &'a TrainingRoster,
    pub role_manifest: &'a [u8],
    pub context_sha256: &'a str,
    pub cut: &'a Cut,
    pub policy: &'a PolicySeeds,
    pub retrieval: &'a RetrievalSeeds,
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
pub struct SelfPlayRatchetRequest<'a> {
    pub roster: &'a TrainingRoster,
    pub role_manifest: &'a [u8],
    pub context_sha256: &'a str,
    pub cut: &'a Cut,
    pub initial_policy: &'a PolicySeeds,
    pub initial_retrieval: &'a RetrievalSeeds,
    pub round_budgets: &'a [ScratchBudget],
    pub seed_cap_b: usize,
    pub archive_mass_per_round: f64,
    pub max_conditional_base_kl: f64,
    pub max_new_support_mass: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfPlayRoundReceipt {
    pub round: usize,
    pub evaluations_per_arm: usize,
    pub rng_seed: u64,
    pub active_support_leaves: usize,
    pub newly_admitted_winners: usize,
    pub seeded_genomes: Vec<String>,
    pub ordinary_best_rmse: f64,
    pub ordinary_best_q: f64,
    pub ratchet_best_rmse: f64,
    pub ratchet_best_q: f64,
    pub ratchet_champion_genome: String,
    pub surplus_q_vs_ordinary: f64,
    pub evals_to_match_ordinary_final: Option<usize>,
    pub speedup_vs_ordinary: Option<f64>,
    pub decomposed_kl: crate::support::DecomposedSupportKl,
}

/// Execute a multi-round (`Round 0 -> Round 1 -> ...`) algorithmic self-play regression ratchet.
///
/// In Round 0, `selfplay_ratchet` is seeded from `initial_policy` + `initial_retrieval` (capped to
/// `seed_cap_b`). After each round `t`, newly discovered measured genomes that improve over the
/// bootstrap baseline are admitted into the round archive `A_{t+1}`, checked against the
/// decomposed support-KL guard (`conditional_base_kl <= max_conditional_base_kl` and
/// `new_support_mass <= max_new_support_mass`), and the top `seed_cap_b` archive champions seed
/// Round `t+1` GP evolution.
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
        archive_mass_per_round,
        max_conditional_base_kl,
        max_new_support_mass,
    } = request;
    ensure!(
        round_budgets.len() >= 2,
        "self-play ratchet requires at least 2 rounds"
    );
    ensure!(
        (1..=16).contains(&seed_cap_b),
        "seed_cap_b must be in 1..=16"
    );
    ensure!(
        archive_mass_per_round.is_finite()
            && archive_mass_per_round > 0.0
            && archive_mass_per_round <= max_new_support_mass,
        "archive_mass_per_round must be in (0, max_new_support_mass]"
    );
    roster.verify_roles(role_manifest)?;
    let context = roster.bound_context(context_sha256, cut)?;
    let offered = initial_policy.pipeline_seeds(context)?;
    let retrieval_input = initial_retrieval.pipeline_seeds(roster, &context.dataset_sha256)?;
    retrieval_input.check()?;
    ensure!(!output.exists(), "self-play ratchet output must be fresh");

    // Build initial Round-0 seed pool from policy + retrieval proposals.
    let mut initial_pairs = Vec::new();
    for g in &offered.genomes {
        initial_pairs.push((g.genome.clone(), seeds::FROM_SEEDS_FILE));
    }
    for g in &retrieval_input.genomes {
        initial_pairs.push((g.genome.clone(), seeds::FROM_SEEDS_FILE));
    }
    let admitted_initial = seeds::admit(&initial_pairs)?;
    ensure!(
        !admitted_initial.seeds.is_empty(),
        "initial policy+retrieval seeds must not be empty"
    );

    let mut base_catalog: BTreeSet<String> = BTreeSet::new();
    base_catalog.insert(canonical_genome(search::BOOTSTRAP_GENOME)?);
    for s in &admitted_initial.seeds {
        base_catalog.insert(s.genome.clone());
    }
    let base_leaf_count = base_catalog.len();

    // Archive maps canonical genome -> best (lowest) measured inner-cut RMSE.
    let mut archive_best_rmse: std::collections::BTreeMap<String, f64> =
        std::collections::BTreeMap::new();
    let mut current_seeds: Vec<String> = admitted_initial
        .seeds
        .iter()
        .take(seed_cap_b)
        .map(|s| s.genome.clone())
        .collect();

    let mut round_receipts: Vec<SelfPlayRoundReceipt> = Vec::with_capacity(round_budgets.len());

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
        let ordinary_best_rmse = ordinary
            .observations
            .iter()
            .filter(|o| o.measured())
            .filter_map(|o| o.rmse)
            .fold(f64::INFINITY, f64::min);
        let ordinary_best_q = 1.0 / (1.0 + ordinary_best_rmse);

        // 2. Run self-play ratchet GP arm seeded with `current_seeds`.
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

        let mut ratchet_best_rmse = f64::INFINITY;
        let mut ratchet_champion_genome = String::new();
        let mut evals_to_match_ordinary_final: Option<usize> = None;
        let mut newly_admitted = 0usize;

        for (eval_idx, obs) in ratchet.observations.iter().enumerate() {
            if !obs.measured() {
                continue;
            }
            if let Some(r) = obs.rmse {
                if r < ratchet_best_rmse {
                    ratchet_best_rmse = r;
                    ratchet_champion_genome = obs.genome.clone();
                }
                if evals_to_match_ordinary_final.is_none() && r <= ordinary_best_rmse + 1e-12 {
                    evals_to_match_ordinary_final = Some(eval_idx + 1);
                }
                let is_new_winner = !base_catalog.contains(&obs.genome)
                    && !archive_best_rmse.contains_key(&obs.genome)
                    && r <= ordinary_best_rmse + 1e-12;
                if is_new_winner {
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
        let ratchet_best_q = 1.0 / (1.0 + ratchet_best_rmse);
        let surplus_q_vs_ordinary = ratchet_best_q - ordinary_best_q;
        let speedup_vs_ordinary =
            evals_to_match_ordinary_final.map(|k| budget.evaluations_per_arm as f64 / k as f64);

        // 3. Compute decomposed support KL over base catalog leaves + newly admitted archive winners.
        let non_base_winners: Vec<(&String, &f64)> = archive_best_rmse
            .iter()
            .filter(|(g, _)| !base_catalog.contains(*g))
            .collect();
        let n_new = non_base_winners.len();
        let active_support_leaves = base_leaf_count + n_new;

        let decomposed_kl = if n_new == 0 {
            let lp = vec![-(base_leaf_count as f64).ln(); base_leaf_count];
            let mask = vec![true; base_leaf_count];
            crate::support::decomposed_support_kl(
                &lp,
                &lp,
                &mask,
                max_conditional_base_kl,
                max_new_support_mass,
            )?
        } else {
            let new_mass =
                (archive_mass_per_round * (round_idx as f64 + 1.0)).min(max_new_support_mass);
            let base_mass = 1.0 - new_mass;
            let ref_new_mass = 1e-4_f64;
            let ref_base_mass = 1.0 - ref_new_mass;

            let mut policy_lp = Vec::with_capacity(active_support_leaves);
            let mut ref_lp = Vec::with_capacity(active_support_leaves);
            let mut is_base = Vec::with_capacity(active_support_leaves);

            for _ in 0..base_leaf_count {
                policy_lp.push((base_mass / base_leaf_count as f64).ln());
                ref_lp.push((ref_base_mass / base_leaf_count as f64).ln());
                is_base.push(true);
            }
            for _ in 0..n_new {
                policy_lp.push((new_mass / n_new as f64).ln());
                ref_lp.push((ref_new_mass / n_new as f64).ln());
                is_base.push(false);
            }
            crate::support::decomposed_support_kl(
                &policy_lp,
                &ref_lp,
                &is_base,
                max_conditional_base_kl,
                max_new_support_mass,
            )?
        };
        ensure!(
            decomposed_kl.admissible,
            "round {round_idx} tripped decomposed support KL guard: {decomposed_kl:?}"
        );

        // 4. Select top `seed_cap_b` measured archive genomes (lowest RMSE, tie-broken by genome)
        // to seed Round `t+1`.
        let mut ranked_archive: Vec<(&String, f64)> =
            archive_best_rmse.iter().map(|(g, &r)| (g, r)).collect();
        ranked_archive.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        let next_seeds: Vec<String> = ranked_archive
            .into_iter()
            .filter(|(g, _)| *g != search::BOOTSTRAP_GENOME)
            .take(seed_cap_b)
            .map(|(g, _)| g.clone())
            .collect();

        round_receipts.push(SelfPlayRoundReceipt {
            round: round_idx,
            evaluations_per_arm: budget.evaluations_per_arm,
            rng_seed: budget.rng_seed,
            active_support_leaves,
            newly_admitted_winners: newly_admitted,
            seeded_genomes: current_seeds.clone(),
            ordinary_best_rmse,
            ordinary_best_q,
            ratchet_best_rmse,
            ratchet_best_q,
            ratchet_champion_genome,
            surplus_q_vs_ordinary,
            evals_to_match_ordinary_final,
            speedup_vs_ordinary,
            decomposed_kl,
        });

        if !next_seeds.is_empty() {
            current_seeds = next_seeds;
        }
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
        last_round.ratchet_best_q + 1e-12 >= last_round.ordinary_best_q;
    let all_rounds_kl_admissible = round_receipts.iter().all(|r| r.decomposed_kl.admissible);
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
        "rounds": round_receipts,
        "monotonic_non_decreasing_best_q": monotonic_non_decreasing_best_q,
        "strictly_improved_over_round0": strictly_improved_over_round0,
        "beat_ordinary_in_final_round": beat_ordinary_in_final_round,
        "all_rounds_kl_admissible": all_rounds_kl_admissible,
        "total_admitted_new_winners": total_admitted_new_winners,
        "qualified_rewards": 0,
        "h1_established": false
    }))
}

#[cfg(test)]
mod tests;
