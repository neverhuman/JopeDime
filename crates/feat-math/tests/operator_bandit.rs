#[path = "../src/kernel/strategy/bandit.rs"]
mod bandit;

use bandit::{
    AttemptDisposition, BanditConfig, BanditError, BehaviorLaw, FeedbackStatus, OperatorBandit,
};

fn config() -> BanditConfig {
    BanditConfig {
        context_id: "ctx-v1:small-table".into(),
        arm_order: vec!["mutate".into(), "crossover".into()],
        epsilon: 0.2,
        learning_rate: 0.5,
        reward_min: -1.0,
        reward_max: 1.0,
    }
}

fn arms(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn select(bandit: &mut OperatorBandit, id: &str, legal: &[&str], draw: f64) -> bandit::Selection {
    bandit
        .select(id, "ctx-v1:small-table", &arms(legal), draw)
        .unwrap()
        .unwrap()
}

#[test]
fn known_law_records_actual_categorical_propensity_and_exact_log_probability() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    let selection = select(&mut bandit, "a0", &["crossover", "mutate"], 0.61);

    assert_eq!(selection.legal_arms, arms(&["mutate", "crossover"]));
    assert_eq!(selection.chosen_arm, "crossover");
    assert_eq!(selection.distribution[0].probability, 0.5);
    assert_eq!(selection.distribution[1].probability, 0.5);
    assert_eq!(selection.propensity, 0.5);
    assert_eq!(selection.log_propensity, 0.5_f64.ln());
    assert!(matches!(selection.law, BehaviorLaw::KnownExp3 { .. }));
    assert_eq!(
        OperatorBandit::propensity_for_pg_ope(&selection.law, Some(selection.propensity)),
        Ok(0.5)
    );
}

#[test]
fn exact_tiny_exp3_distribution_updates_from_eligible_gain_only() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    let chosen = select(&mut bandit, "a0", &["mutate", "crossover"], 0.1);
    bandit
        .record_feedback("a0", FeedbackStatus::EligibleMeasured, Some(0.8))
        .unwrap();

    // eta=0.5, gain=0.8, K=2, p=0.5: eta * gain / p / K = 0.4.
    let exp = 0.4_f64.exp();
    let mutate_probability = 0.8 * (exp / (exp + 1.0)) + 0.1;
    let next = select(&mut bandit, "a1", &["mutate", "crossover"], 0.0);
    assert_eq!(chosen.chosen_arm, "mutate");
    assert!((next.distribution[0].probability - mutate_probability).abs() < 1e-15);
    assert!((next.distribution.iter().map(|p| p.probability).sum::<f64>() - 1.0).abs() < 1e-15);
}

#[test]
fn zero_gain_ties_and_single_legal_arm_have_defined_probabilities() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    let first = select(&mut bandit, "a0", &["crossover", "mutate"], 0.0);
    bandit
        .record_feedback("a0", FeedbackStatus::EligibleMeasured, Some(0.0))
        .unwrap();
    let tied = select(&mut bandit, "a1", &["mutate", "crossover"], 0.0);
    assert_eq!(first.chosen_arm, "mutate");
    assert_eq!(tied.chosen_arm, "mutate");
    assert_eq!(tied.distribution, first.distribution);

    let singleton = select(&mut bandit, "a2", &["crossover"], 0.999);
    assert_eq!(singleton.chosen_arm, "crossover");
    assert_eq!(singleton.propensity, 1.0);
}

#[test]
fn empty_legal_set_is_retained_as_unavailable_without_invented_probability() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    assert!(bandit
        .select("empty", "ctx-v1:small-table", &[], 0.25)
        .unwrap()
        .is_none());
    let attempt = &bandit.attempts()[0];
    assert_eq!(attempt.disposition, AttemptDisposition::Unavailable);
    assert!(attempt.selection.is_none());
    assert!(attempt.feedback.is_empty());
}

#[test]
fn delayed_failure_and_ineligible_feedback_are_logged_but_do_not_update() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    let initial = select(&mut bandit, "delayed", &["mutate", "crossover"], 0.0);
    bandit
        .record_feedback("delayed", FeedbackStatus::Delayed, None)
        .unwrap();
    bandit
        .record_feedback("delayed", FeedbackStatus::EligibleMeasured, Some(0.5))
        .unwrap();
    let delayed_record = &bandit.attempts()[0];
    assert_eq!(delayed_record.feedback.len(), 2);
    assert!(!delayed_record.feedback[0].updated);
    assert!(delayed_record.feedback[1].updated);
    let after_delayed = select(&mut bandit, "failed", &["mutate", "crossover"], 0.0);
    bandit
        .record_feedback("failed", FeedbackStatus::Failed, None)
        .unwrap();
    let after_failure = select(&mut bandit, "ineligible", &["mutate", "crossover"], 0.0);
    bandit
        .record_feedback("ineligible", FeedbackStatus::Ineligible, None)
        .unwrap();
    let after_ineligible = select(&mut bandit, "next", &["mutate", "crossover"], 0.0);

    assert_eq!(
        initial.distribution,
        vec![
            bandit::ActionProbability {
                arm: "mutate".into(),
                probability: 0.5
            },
            bandit::ActionProbability {
                arm: "crossover".into(),
                probability: 0.5
            },
        ]
    );
    assert_eq!(after_failure.distribution, after_ineligible.distribution);
    assert_eq!(after_ineligible.distribution, after_delayed.distribution);
}

#[test]
fn rejected_actions_are_retained_and_rejection_does_not_create_a_reward() {
    let mut bandit = OperatorBandit::new(config()).unwrap();
    select(&mut bandit, "rejected", &["mutate", "crossover"], 0.0);
    bandit
        .mark_rejected("rejected", "operator produced illegal child")
        .unwrap();
    let attempt = &bandit.attempts()[0];
    assert_eq!(attempt.disposition, AttemptDisposition::Rejected);
    assert_eq!(
        attempt.rejection_reason.as_deref(),
        Some("operator produced illegal child")
    );
    assert!(attempt.feedback.is_empty());
}

#[test]
fn unknown_thompson_law_cannot_supply_unbiased_pg_or_ope_propensity() {
    let unknown = BehaviorLaw::Unknown {
        reason: "sampled posterior score is not action probability".into(),
    };
    assert_eq!(
        OperatorBandit::propensity_for_pg_ope(&unknown, Some(0.9)),
        Err(BanditError::UnknownLaw)
    );
    assert_eq!(
        OperatorBandit::propensity_for_pg_ope(
            &BehaviorLaw::KnownExp3 {
                config_sha256: "x".into()
            },
            None
        ),
        Err(BanditError::PropensityUnavailable)
    );
}

#[test]
fn extreme_weight_underflow_keeps_a_positive_uniform_floor() {
    let mut cfg = config();
    cfg.learning_rate = 8.0;
    let mut bandit = OperatorBandit::new(cfg).unwrap();
    for i in 0..120 {
        let id = format!("boost-{i}");
        let chosen = select(&mut bandit, &id, &["mutate", "crossover"], 0.0);
        assert_eq!(chosen.chosen_arm, "mutate");
        bandit
            .record_feedback(&id, FeedbackStatus::EligibleMeasured, Some(1.0))
            .unwrap();
    }
    let selection = select(
        &mut bandit,
        "underflow",
        &["mutate", "crossover"],
        0.999_999,
    );
    assert!(selection.distribution.iter().all(|p| p.probability > 0.0));
    assert!(
        (selection
            .distribution
            .iter()
            .map(|p| p.probability)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-15
    );
}
