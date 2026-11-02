use super::{
    canonical_genome, measured_tilt_leaf_log_probs, rows_digest, MeasuredTiltPolicyConfig,
    PolicySeeds, RetrievalSeeds, ScratchBudget, TrainingRoster,
};
use anyhow::{ensure, Result};
use jope_pipeline::{prepare::Cut, search, seeds};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

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
