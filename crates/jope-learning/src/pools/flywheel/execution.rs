use super::*;

/// Performs the run distilled flywheel operation and returns an error if it cannot be completed.
pub fn run_distilled_flywheel(
    pools: &[Pool],
    manifest: &Manifest,
    context_features: &Features,
    program_features: Option<&Features>,
    external: &[&dyn Selector],
    cfg: &FlywheelConfig,
    parquet_dir: Option<&Path>,
) -> Result<FlywheelReceipt> {
    run_distilled_flywheel_with_s3_prior(
        pools,
        manifest,
        context_features,
        program_features,
        external,
        None,
        cfg,
        parquet_dir,
    )
}

#[allow(clippy::too_many_arguments)]
/// Performs the run distilled flywheel with S3 prior operation and returns an error if it cannot be completed.
pub fn run_distilled_flywheel_with_s3_prior(
    pools: &[Pool],
    manifest: &Manifest,
    context_features: &Features,
    program_features: Option<&Features>,
    external: &[&dyn Selector],
    s3_prior_rows: Option<&[distilled::DistilledRow]>,
    cfg: &FlywheelConfig,
    parquet_dir: Option<&Path>,
) -> Result<FlywheelReceipt> {
    verify(pools, manifest)?;
    context_features
        .verify_context_descriptor()
        .context("flywheel nested/v3 context provenance unavailable")?;
    ensure!(!pools.is_empty(), "no frozen pools");
    ensure!(cfg.rounds >= 2, "flywheel requires at least 2 rounds");
    ensure!(
        cfg.initial_budget_per_context >= 2,
        "initial_budget_per_context must be >= 2"
    );
    ensure!(
        cfg.step_budget_per_context >= 1 && cfg.proposal_pool_k >= cfg.step_budget_per_context,
        "requires 1 <= step_budget_per_context <= proposal_pool_k"
    );
    ensure!(
        !cfg.eval_budgets.is_empty() && cfg.eval_budgets.iter().all(|&b| b >= 1),
        "eval_budgets must be non-empty and positive"
    );
    ensure!(
        cfg.eval_budgets.contains(&cfg.eval.budget),
        "eval_budgets must include primary eval.budget ({})",
        cfg.eval.budget
    );
    ensure!(
        cfg.score_name != "l6",
        "shadow flywheel distillation may not use score_name \"l6\""
    );
    cfg.tilt.validate()?;
    ensure!(
        cfg.max_conditional_base_kl.is_finite() && cfg.max_conditional_base_kl > 0.0,
        "max_conditional_base_kl must be finite and > 0"
    );
    ensure!(
        cfg.max_new_support_mass.is_finite()
            && cfg.max_new_support_mass > 0.0
            && cfg.max_new_support_mass < 1.0,
        "max_new_support_mass must be in (0, 1)"
    );

    let scores_set = known_scores();
    let direction = score_direction(&cfg.score_name)?;
    let derived_pf = match program_features {
        Some(_) => None,
        None => token_bag_features(pools),
    };
    let pf = program_features
        .or(derived_pf.as_ref())
        .context("program features or pool tokens required for flywheel")?;

    if let Some(dir) = parquet_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create parquet directory {}", dir.display()))?;
    }

    let manifest_sha256 = manifest.sha256()?;
    let (fold_s3_priors, s3_prior_rows_by_fold) = match s3_prior_rows {
        Some(rows) => {
            let (p, c) = build_fold_s3_priors(pools, &cfg.eval, rows)?;
            (Some(p), c)
        }
        None => (None, vec![0usize; cfg.eval.folds]),
    };

    let round_results = rounds::run_rounds(
        pools,
        context_features,
        pf,
        &scores_set,
        direction,
        cfg,
        parquet_dir,
        fold_s3_priors.as_deref(),
    )?;
    let rounds::RoundResults {
        round_summaries,
        round_metrics_by_budget,
        passive_round_metrics_by_budget,
        final_active_discovery_hwm,
        final_passive_discovery_hwm,
        all_rounds_kl_admissible,
        final_train_sub_pools,
        round_cv_trace,
    } = round_results;

    let monotonic_train_hwm_gain = round_summaries
        .windows(2)
        .all(|w| w[1].mean_train_hwm_gain + 1e-12 >= w[0].mean_train_hwm_gain);
    let primary_b = cfg.eval.budget;
    let monotonic_heldout_pairwise_accuracy = round_summaries.windows(2).all(|w| {
        match (
            w[0].heldout_by_budget[&primary_b].mean_pairwise_accuracy,
            w[1].heldout_by_budget[&primary_b].mean_pairwise_accuracy,
        ) {
            (Some(a), Some(b)) => b + 1e-12 >= a,
            _ => false,
        }
    });

    let last_r = cfg.rounds - 1;
    let mut round_lifts = Vec::new();
    let mut active_vs_passive_lifts = Vec::new();
    for &b in &cfg.eval_budgets {
        let m0 = &round_metrics_by_budget[0][&b];
        let m_last = &round_metrics_by_budget[last_r][&b];
        let gain_lift = paired_metrics(pools, m_last, m0, |m| Some(m.realized_gain), &cfg.eval)?
            .context("paired realized_gain_lift requires at least 2 contexts")?;
        let top1_lift = paired_metrics(pools, m_last, m0, |m| Some(m.top1_gain), &cfg.eval)?
            .context("paired top1_gain_lift requires at least 2 contexts")?;
        let pw_lift = paired_metrics(pools, m_last, m0, |m| m.pairwise_accuracy, &cfg.eval)?;
        round_lifts.push(RoundLiftComparison {
            budget: b,
            from_round: 0,
            to_round: last_r,
            realized_gain_lift: gain_lift,
            top1_gain_lift: top1_lift,
            pairwise_accuracy_lift: pw_lift,
        });

        let pm_last = &passive_round_metrics_by_budget[last_r][&b];
        let ap_gain_lift =
            paired_metrics(pools, m_last, pm_last, |m| Some(m.realized_gain), &cfg.eval)?.context(
                "paired active_vs_passive realized_gain_lift requires at least 2 contexts",
            )?;
        let ap_top1_lift =
            paired_metrics(pools, m_last, pm_last, |m| Some(m.top1_gain), &cfg.eval)?
                .context("paired active_vs_passive top1_gain_lift requires at least 2 contexts")?;
        let ap_pw_lift =
            paired_metrics(pools, m_last, pm_last, |m| m.pairwise_accuracy, &cfg.eval)?;
        active_vs_passive_lifts.push(RoundLiftComparison {
            budget: b,
            from_round: last_r,
            to_round: last_r,
            realized_gain_lift: ap_gain_lift,
            top1_gain_lift: ap_top1_lift,
            pairwise_accuracy_lift: ap_pw_lift,
        });
    }

    let active_vs_passive_discovery_lift = paired_scalars(
        pools,
        &final_active_discovery_hwm,
        &final_passive_discovery_hwm,
        &cfg.eval,
    )?;

    let final_vs_round0_primary_verdict = round_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .map(|l| l.realized_gain_lift.verdict)
        .unwrap_or(IntervalVerdict::Inconclusive);
    let final_vs_round0_pairwise_verdict = round_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .and_then(|l| l.pairwise_accuracy_lift.as_ref())
        .map(|c| c.verdict)
        .unwrap_or(IntervalVerdict::Inconclusive);

    let active_beats_passive_discovery_verdict = active_vs_passive_discovery_lift.verdict;
    let active_beats_passive_primary_verdict = active_vs_passive_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .map(|l| l.realized_gain_lift.verdict)
        .unwrap_or(IntervalVerdict::Inconclusive);
    let active_beats_passive_pairwise_verdict = active_vs_passive_lifts
        .iter()
        .find(|l| l.budget == primary_b)
        .and_then(|l| l.pairwise_accuracy_lift.as_ref())
        .map(|c| c.verdict)
        .unwrap_or(IntervalVerdict::Inconclusive);

    // Evaluate Round R-1's outer-fold distilled training sub-pools against the full held-out pools
    // at the primary budget so `final_round_receipt` matches Round R-1's held-out evaluation bit-for-bit.
    let eval_inputs = Inputs {
        context_features: Some(context_features),
        program_features,
        external: external.to_vec(),
        unavailable: Vec::new(),
    };
    let mut final_round_receipt = evaluate_with_fold_train_pools_and_priors(
        pools,
        Some(&final_train_sub_pools),
        fold_s3_priors.as_deref(),
        manifest,
        &eval_inputs,
        &cfg.eval,
    )?;
    if !round_cv_trace.is_empty() {
        if let Some(sel_rec) = final_round_receipt
            .selectors
            .iter_mut()
            .find(|s| s.name == "contextual_surrogate_qhat")
        {
            sel_rec.rule.push_str(&format!(
                "; flywheel nested/v3 fits={}; see typed surrogate_cv",
                round_cv_trace.len()
            ));
        }
    }
    let surrogate_beats_strongest_baseline = final_round_receipt
        .comparisons
        .iter()
        .find(|c| {
            c.candidate == "contextual_surrogate_qhat"
                && c.baseline == final_round_receipt.strongest_baseline
        })
        .map(|c| {
            c.realized_gain.verdict == IntervalVerdict::Greater
                && c.pairwise_accuracy.as_ref().map(|p| p.verdict) == Some(IntervalVerdict::Greater)
        })
        .unwrap_or(false);

    let active_discovery_established = all_rounds_kl_admissible
        && active_beats_passive_discovery_verdict == IntervalVerdict::Greater;
    let data_accumulation_lift_established = monotonic_train_hwm_gain
        && all_rounds_kl_admissible
        && final_vs_round0_primary_verdict == IntervalVerdict::Greater
        && final_vs_round0_pairwise_verdict == IntervalVerdict::Greater;
    let flywheel_established = active_discovery_established
        && data_accumulation_lift_established
        && surrogate_beats_strongest_baseline
        && active_beats_passive_primary_verdict == IntervalVerdict::Greater
        && active_beats_passive_pairwise_verdict == IntervalVerdict::Greater;

    Ok(FlywheelReceipt {
        schema: FLYWHEEL_RECEIPT_SCHEMA,
        shadow_only: true,
        qualified_claim: false,
        l6_claim: false,
        manifest_sha256,
        s3_prior_rows_by_fold,
        config: cfg.clone(),
        rounds: round_summaries,
        surrogate_cv: round_cv_trace,
        monotonic_train_hwm_gain,
        monotonic_heldout_pairwise_accuracy,
        all_rounds_kl_admissible,
        round_lifts,
        active_vs_passive_lifts,
        active_vs_passive_discovery_lift,
        final_round_receipt,
        surrogate_beats_strongest_baseline,
        final_vs_round0_primary_verdict,
        final_vs_round0_pairwise_verdict,
        active_beats_passive_discovery_verdict,
        active_beats_passive_primary_verdict,
        active_beats_passive_pairwise_verdict,
        active_discovery_established,
        data_accumulation_lift_established,
        flywheel_established,
    })
}
