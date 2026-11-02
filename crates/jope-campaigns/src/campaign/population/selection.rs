use super::*;

pub(super) fn offspring_seed(base: u64, generation: usize, slot: usize, retry: usize) -> u64 {
    base ^ (generation as u64).wrapping_mul(0xd6e8_feb8_6659_fd93)
        ^ (slot as u64).wrapping_mul(0xa076_1d64_78bd_642f)
        ^ (retry as u64).wrapping_mul(0xe703_7ed1_a0b4_28db)
}

pub(super) fn rank_slots(left: &SlotLedger, right: &SlotLedger) -> Ordering {
    right
        .robust_fitness
        .unwrap()
        .total_cmp(&left.robust_fitness.unwrap())
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(super) fn rank_finalists(left: &&FinalistCv7, right: &&FinalistCv7) -> Ordering {
    let left_score = left.cv7.as_ref().unwrap().robust_fitness;
    let right_score = right.cv7.as_ref().unwrap().robust_fitness;
    right_score
        .total_cmp(&left_score)
        .then_with(|| left.complexity.cmp(&right.complexity))
        .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
}

pub(super) fn selection_evidence_from_slot(
    slot: &SlotLedger,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> Result<SelectionEvidence> {
    if !slot.rankable() || slot.fold_r2.as_ref().unwrap().len() != evaluated_fold_ids.len() {
        bail!("quick-score finalist lacks complete held-out evidence");
    }
    Ok(SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: slot.fold_r2.clone().unwrap(),
        pooled_oof_r2: slot.pooled_oof_r2.unwrap(),
        pooled_oof_rmse: slot.pooled_oof_rmse.unwrap(),
        robust_fitness: slot.robust_fitness.unwrap(),
        fit_seconds: slot.fold_fit_seconds.clone(),
        predict_seconds: slot.fold_predict_seconds.clone(),
    })
}

pub(super) fn selection_evidence_from_result(
    result: ExactCvResult,
    fold_plan_folds: usize,
    evaluated_fold_ids: Vec<usize>,
    fold_manifest_sha256: &str,
) -> SelectionEvidence {
    SelectionEvidence {
        fold_plan_folds,
        evaluated_fold_ids,
        fold_manifest_sha256: fold_manifest_sha256.to_string(),
        fold_r2: result.fold_r2,
        pooled_oof_r2: result.pooled_oof_r2,
        pooled_oof_rmse: result.pooled_oof_rmse,
        robust_fitness: result.robust_fitness,
        fit_seconds: result.fit_seconds,
        predict_seconds: result.predict_seconds,
    }
}

pub(super) fn generation_summary<'a>(
    generation: usize,
    slots: &[SlotLedger],
    generation_zero_best: f64,
    incumbent_history: &'a [Incumbent],
    next_population: &'a [PopulationMember],
    terminal_slots: usize,
    unique_evaluations: usize,
    ledger_sha256: &'a str,
) -> Result<GenerationSummary<'a>> {
    let mut fitness = slots
        .iter()
        .filter_map(|slot| slot.rankable().then_some(slot.robust_fitness.unwrap()))
        .collect::<Vec<_>>();
    fitness.sort_by(f64::total_cmp);
    let best = fitness
        .last()
        .copied()
        .context("generation has no valid fitness")?;
    let incumbent = slots
        .iter()
        .filter(|slot| slot.rankable())
        .min_by(|left, right| rank_slots(left, right))
        .context("generation has no ranked incumbent")?;
    let incumbent_pooled_oof_rmse = incumbent.pooled_oof_rmse.unwrap();
    let median = quantile(&fitness, 0.5);
    let mut fitness_quantiles = BTreeMap::new();
    for (name, q) in [
        ("p00", 0.0),
        ("p25", 0.25),
        ("p50", 0.5),
        ("p75", 0.75),
        ("p100", 1.0),
    ] {
        fitness_quantiles.insert(name.to_string(), quantile(&fitness, q));
    }
    let mut head_mix = BTreeMap::new();
    for slot in slots {
        *head_mix.entry(slot.head_family.clone()).or_insert(0) += 1;
    }
    let diversity = slots
        .iter()
        .map(|slot| slot.semantic_sha256.as_str())
        .collect::<HashSet<_>>()
        .len() as f64
        / slots.len() as f64;
    Ok(GenerationSummary {
        schema: SUMMARY_SCHEMA,
        generation,
        population: slots.len(),
        valid: fitness.len(),
        invalid: slots.len() - fitness.len(),
        cache_hits: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "hit")
            .count(),
        cache_misses: slots
            .iter()
            .filter(|slot| slot.cache_disposition == "miss")
            .count(),
        fitness_quantiles,
        best_fitness: best,
        incumbent_pooled_oof_rmse,
        cv_rmse_signed_margin_vs_tabfm: TABFM_RMSE - incumbent_pooled_oof_rmse,
        cv_proxy_track_status: if incumbent_pooled_oof_rmse < TABFM_RMSE {
            "CV_PROXY_AHEAD"
        } else {
            "CV_PROXY_BEHIND"
        }
        .to_string(),
        median_fitness: median,
        improvement_from_generation_zero: best - generation_zero_best,
        canonical_diversity: diversity,
        head_mix,
        incumbent_history,
        next_population,
        cumulative_terminal_slots: terminal_slots,
        cumulative_unique_evaluations: unique_evaluations,
        cumulative_ledger_sha256: ledger_sha256,
    })
}
