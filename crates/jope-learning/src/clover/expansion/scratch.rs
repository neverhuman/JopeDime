use super::*;

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
    pub(super) fn pipeline_seeds(
        &self,
        roster: &TrainingRoster,
        dataset: &str,
    ) -> Result<seeds::SeedsFile> {
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
