use feat_math::kernel::strategy::gumbel_root::{
    sample_legal, ActionPrior, CompletedQ, Fidelity, FitPurpose, GumbelRoot, HalvingTerminal,
    MeasuredFitBudget, MeasuredQ, RootSearchError, SamplingLaw, TeacherTargetError,
};

fn actions(n: usize) -> Vec<ActionPrior> {
    (0..n)
        .map(|i| ActionPrior {
            action_id: format!("action-{i}"),
            prior_weight: (i + 1) as f64,
        })
        .collect()
}

fn fidelity(rank: u16, id: &str) -> Fidelity {
    Fidelity {
        rank,
        id: id.to_owned(),
    }
}

fn measured(action_id: &str, at: &Fidelity, loss: f64, fit_ordinal: u64) -> MeasuredQ {
    MeasuredQ {
        action_id: action_id.to_owned(),
        fidelity: at.clone(),
        loss,
        fit_ordinal,
    }
}

#[test]
fn gumbel_top_k_is_deterministic_unique_and_reports_truncated_legal_mass() {
    let legal = actions(9);
    let first = sample_legal(&legal, 3, 17).unwrap();
    let replay = sample_legal(&legal, 3, 17).unwrap();
    assert_eq!(first, replay);
    assert_eq!(first.law, SamplingLaw::GumbelTopKWithoutReplacementV1);
    assert_eq!(first.requested, 3);
    assert_eq!(first.legal_count, 9);
    assert_eq!(first.legal_actions, legal);
    assert_eq!(first.selected.len(), 3);
    let unique: std::collections::BTreeSet<_> = first
        .selected
        .iter()
        .map(|action| action.action_id.as_str())
        .collect();
    assert_eq!(unique.len(), 3);
}

#[test]
fn incomplete_halving_keeps_unscheduled_actions_and_never_names_a_winner() {
    let sampled = sample_legal(&actions(7), 3, 5).unwrap();
    let low = fidelity(1, "cheap");
    let high = fidelity(2, "final");
    let mut root = GumbelRoot::new(sampled, vec![low.clone(), high]).unwrap();
    let plan = root.plan_round(2).unwrap().unwrap();
    assert_eq!(plan.action_ids.len(), 2);
    assert_eq!(plan.unscheduled_action_ids.len(), 1);
    assert!(!plan.is_complete());
    let values = plan
        .action_ids
        .iter()
        .enumerate()
        .map(|(i, id)| measured(id, &low, i as f64, i as u64))
        .collect();
    root.record_round(plan, values).unwrap();
    assert_eq!(
        root.terminal(),
        &HalvingTerminal::Incomplete {
            round: 0,
            reason: "fit budget did not schedule every action"
        }
    );
}

#[test]
fn measured_fidelity_controls_halving_and_completed_q_cannot_win() {
    let sampled = sample_legal(&actions(6), 3, 23).unwrap();
    let sample_ids: Vec<_> = sampled
        .selected
        .iter()
        .map(|a| a.action_id.clone())
        .collect();
    let low = fidelity(1, "coarse");
    let high = fidelity(2, "full");
    let mut root = GumbelRoot::new(sampled, vec![low.clone(), high.clone()]).unwrap();
    root.record_completed_q(CompletedQ {
        action_id: sample_ids[2].clone(),
        target_fidelity: high.clone(),
        estimate: -1_000.0,
        completion_model: "draft-surrogate-v1".into(),
    })
    .unwrap();

    let first = root.plan_round(3).unwrap().unwrap();
    root.record_round(
        first.clone(),
        first
            .action_ids
            .iter()
            .enumerate()
            .map(|(i, id)| measured(id, &low, [3.0, 1.0, 2.0][i], i as u64))
            .collect(),
    )
    .unwrap();

    let second = root.plan_round(2).unwrap().unwrap();
    assert_eq!(second.fidelity, high);
    assert_eq!(second.action_ids.len(), 2);
    let final_values: Vec<_> = second
        .action_ids
        .iter()
        .enumerate()
        .map(|(i, id)| measured(id, &second.fidelity, [2.0, 1.0][i], 10 + i as u64))
        .collect();
    let expected_winner = final_values[1].action_id.clone();
    root.record_round(second, final_values).unwrap();
    assert_eq!(
        root.terminal(),
        &HalvingTerminal::Complete {
            winner: expected_winner,
            fidelity: fidelity(2, "full")
        }
    );
    assert_eq!(root.completed_q().len(), 1);
    assert_eq!(root.measured_q().len(), 5);
}

#[test]
fn measured_fit_budget_charges_prior_argmax_best_of_n_and_failures_separately_from_visits() {
    let mut fit_matched_control = MeasuredFitBudget::new(4, 100);
    for _ in 0..4 {
        fit_matched_control
            .charge_fit(FitPurpose::BestOfNControl, 10, true)
            .unwrap();
    }
    let mut fit_matched_search = MeasuredFitBudget::new(4, 100);
    fit_matched_search
        .charge_fit(FitPurpose::PriorArgmaxControl, 10, true)
        .unwrap();
    for _ in 0..3 {
        fit_matched_search
            .charge_fit(FitPurpose::MeasuredLeaf, 7, true)
            .unwrap();
    }
    assert_eq!(
        fit_matched_control.snapshot().fits_charged,
        fit_matched_search.snapshot().fits_charged
    );
    assert_ne!(
        fit_matched_control.snapshot().cpu_ns_charged,
        fit_matched_search.snapshot().cpu_ns_charged
    );

    let mut cpu_matched_best_of_n = MeasuredFitBudget::new(10, 20);
    for _ in 0..4 {
        cpu_matched_best_of_n
            .charge_fit(FitPurpose::BestOfNControl, 5, true)
            .unwrap();
    }
    let mut cpu_matched_prior = MeasuredFitBudget::new(10, 20);
    cpu_matched_prior
        .charge_fit(FitPurpose::PriorArgmaxControl, 10, true)
        .unwrap();
    cpu_matched_prior
        .charge_fit(FitPurpose::MeasuredLeaf, 10, true)
        .unwrap();
    assert_eq!(
        cpu_matched_best_of_n.snapshot().cpu_ns_charged,
        cpu_matched_prior.snapshot().cpu_ns_charged
    );
    assert_ne!(
        cpu_matched_best_of_n.snapshot().fits_charged,
        cpu_matched_prior.snapshot().fits_charged
    );

    let mut ledger = MeasuredFitBudget::new(4, 40);
    ledger
        .charge_fit(FitPurpose::PriorArgmaxControl, 10, true)
        .unwrap();
    for i in 0..3 {
        ledger
            .charge_fit(FitPurpose::BestOfNControl, 10, i != 1)
            .unwrap();
    }
    for _ in 0..11 {
        ledger.record_visit().unwrap();
    }
    assert_eq!(ledger.snapshot().fits_charged, 4);
    assert_eq!(ledger.snapshot().cpu_ns_charged, 40);
    assert_eq!(ledger.snapshot().failed_fits_charged, 1);
    assert_eq!(ledger.snapshot().visits, 11);
    assert_eq!(
        ledger.snapshot().by_purpose[&FitPurpose::PriorArgmaxControl],
        1
    );
    assert_eq!(ledger.snapshot().by_purpose[&FitPurpose::BestOfNControl], 3);
}

#[test]
fn teacher_targets_reject_unsupported_sampling_laws() {
    let sample = sample_legal(&actions(4), 2, 41).unwrap();
    let low = fidelity(1, "cheap");
    let mut root = GumbelRoot::new(sample, vec![low.clone()]).unwrap();
    let plan = root.plan_round(2).unwrap().unwrap();
    let values = plan
        .action_ids
        .iter()
        .enumerate()
        .map(|(i, id)| measured(id, &low, i as f64, i as u64))
        .collect();
    root.record_round(plan, values).unwrap();
    assert_eq!(
        root.teacher_target(SamplingLaw::WeightedWithReplacement, &low),
        Err(TeacherTargetError::UnsupportedSamplingLaw)
    );
    assert_eq!(
        root.teacher_target(SamplingLaw::Unknown, &low),
        Err(TeacherTargetError::UnsupportedSamplingLaw)
    );
    assert!(root
        .teacher_target(SamplingLaw::GumbelTopKWithoutReplacementV1, &low)
        .is_ok());
}

#[test]
fn root_rejects_unsupported_sampling_trace_and_invalid_measurement_can_be_retried() {
    let mut unsupported = sample_legal(&actions(3), 2, 8).unwrap();
    unsupported.law = SamplingLaw::WeightedWithReplacement;
    assert_eq!(
        GumbelRoot::new(unsupported, vec![fidelity(1, "full")]).unwrap_err(),
        RootSearchError::UnsupportedSamplingLaw
    );

    let sample = sample_legal(&actions(3), 2, 9).unwrap();
    let level = fidelity(1, "full");
    let mut root = GumbelRoot::new(sample, vec![level.clone()]).unwrap();
    let plan = root.plan_round(2).unwrap().unwrap();
    let invalid = plan
        .action_ids
        .iter()
        .map(|id| measured(id, &level, 1.0, 0))
        .collect();
    assert_eq!(
        root.record_round(plan.clone(), invalid),
        Err(RootSearchError::DuplicateFitOrdinal(0))
    );
    let corrected = plan
        .action_ids
        .iter()
        .enumerate()
        .map(|(i, id)| measured(id, &level, i as f64, i as u64))
        .collect();
    assert_eq!(root.record_round(plan, corrected), Ok(()));
    assert!(matches!(root.terminal(), HalvingTerminal::Complete { .. }));
}

#[test]
fn budget_charges_failed_work_even_when_actual_cpu_crosses_ceiling() {
    let mut budget = MeasuredFitBudget::new(1, 5);
    assert!(budget.can_start_fit(5));
    budget
        .charge_fit(FitPurpose::MeasuredLeaf, 8, false)
        .unwrap();
    assert!(!budget.can_start_fit(1));
    assert_eq!(budget.snapshot().fits_charged, 1);
    assert_eq!(budget.snapshot().cpu_ns_charged, 8);
    assert_eq!(budget.snapshot().failed_fits_charged, 1);
    assert!(budget.snapshot().over_budget);
}

#[test]
fn fit_slot_planning_respects_remaining_fit_and_estimated_cpu_limits() {
    let mut budget = MeasuredFitBudget::new(5, 21);
    budget
        .charge_fit(FitPurpose::PriorArgmaxControl, 7, true)
        .unwrap();
    assert_eq!(budget.available_fit_slots(7), 2);
    assert!(budget.can_start_fit(7));
    assert!(!budget.can_start_fit(15));
}
