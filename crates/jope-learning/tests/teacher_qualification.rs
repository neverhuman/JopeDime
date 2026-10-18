#[path = "../src/qualify/teacher.rs"]
mod teacher;

use teacher::{
    decide, ChargedCost, DevelopmentContext, TeacherQualification, TeacherTarget,
    TEACHER_CONTRAST_SCHEMA,
};

fn hash(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn target(row: &str, support: &str) -> TeacherTarget {
    TeacherTarget {
        row_id: row.to_owned(),
        action_id: "action-a".to_owned(),
        action_probability: 0.7,
        support_sha256: support.to_owned(),
        measured: true,
        source_id: "teacher-eval-v1".to_owned(),
        source_sha256: hash('a'),
        evaluation_id: "dev-eval-01".to_owned(),
        evaluation_sha256: hash('b'),
        supervised_label_eligible: true,
    }
}

fn context(index: usize, prior: f64, teacher: f64) -> DevelopmentContext {
    DevelopmentContext {
        context_id: format!("context-{index}"),
        independence_group_id: format!("family-{index}"),
        raw_prior_expected_utility: prior,
        teacher_expected_utility: teacher,
        selected_root_won: true,
        selected_candidate_id: "expert-2".to_owned(),
        included_candidate_ids: vec!["expert-1".to_owned(), "expert-2".to_owned()],
        targets: vec![target(&format!("row-{index}"), &hash('c'))],
    }
}

fn valid() -> TeacherQualification {
    TeacherQualification {
        schema: TEACHER_CONTRAST_SCHEMA.to_owned(),
        contrast_id: "contrast-01".to_owned(),
        raw_prior_id: "raw-prior".to_owned(),
        best_of_n_teacher_id: "expert-2".to_owned(),
        candidate_ids: vec!["expert-1".to_owned(), "expert-2".to_owned()],
        selected_candidate_id: "expert-2".to_owned(),
        selection_law: "best_of_n_selected_in_candidate_set/v1".to_owned(),
        expected_utility_metric: "mean_expected_return/v1".to_owned(),
        minimum_mean_gain: 0.1,
        minimum_independent_contexts: 3,
        raw_prior_cost: ChargedCost {
            cpu_micros: 10_000,
            wall_micros: 20_000,
        },
        teacher_cost: ChargedCost {
            cpu_micros: 10_000,
            wall_micros: 20_000,
        },
        prior_support_sha256: hash('c'),
        teacher_support_sha256: hash('c'),
        contexts: vec![
            context(1, 1.0, 1.5),
            context(2, 1.0, 1.4),
            context(3, 1.0, 1.6),
        ],
    }
}

#[test]
fn supported_measured_teacher_with_frozen_equal_cost_contrast_is_admitted() {
    let result = decide(&valid());
    assert!(result.qualified, "{:?}", result.reasons);
    assert!(result.policy_cross_entropy_authorized);
    assert_eq!(result.independent_contexts, 3);
    assert_eq!(result.eligible_policy_target_rows.len(), 3);
    assert_eq!(result.eligible_supervised_label_rows.len(), 3);
    assert_eq!(result.contrast_sha256.len(), 64);
}

#[test]
fn weak_teacher_cannot_create_policy_ce_authority_but_keeps_supervised_rows() {
    let mut input = valid();
    for context in &mut input.contexts {
        context.teacher_expected_utility = context.raw_prior_expected_utility - 0.01;
    }
    let result = decide(&input);
    assert!(!result.qualified);
    assert!(!result.policy_cross_entropy_authorized);
    assert!(result.eligible_policy_target_rows.is_empty());
    assert_eq!(result.eligible_supervised_label_rows.len(), 3);
    assert_eq!(result.rejected_policy_target_rows.len(), 3);
}

#[test]
fn selected_root_wins_do_not_substitute_for_expected_policy_improvement() {
    let mut input = valid();
    for context in &mut input.contexts {
        context.teacher_expected_utility = context.raw_prior_expected_utility;
        context.selected_root_won = true;
    }
    let result = decide(&input);
    assert!(!result.qualified);
    assert!(!result.policy_cross_entropy_authorized);
    assert!(result
        .reasons
        .iter()
        .any(|reason| reason.contains("expected policy utility")));
}

#[test]
fn missing_cost_or_changed_support_invalidates_the_frozen_contrast() {
    let mut missing_cost = valid();
    missing_cost.teacher_cost.wall_micros = 0;
    let result = decide(&missing_cost);
    assert!(!result.qualified);
    assert!(result
        .reasons
        .iter()
        .any(|reason| reason.contains("charged resources")));

    let mut changed_support = valid();
    changed_support.contexts[1].targets[0].support_sha256 = hash('d');
    let result = decide(&changed_support);
    assert!(!result.qualified);
    assert!(result
        .reasons
        .iter()
        .any(|reason| reason.contains("changed measured support")));
    assert_eq!(result.eligible_supervised_label_rows.len(), 3);
}

#[test]
fn unsupported_or_unmeasured_targets_are_not_policy_targets() {
    let mut input = valid();
    input.contexts[0].targets[0].measured = false;
    input.contexts[1].targets[0].evaluation_sha256 = "unknown".to_owned();
    let result = decide(&input);
    assert!(!result.qualified);
    assert!(!result.policy_cross_entropy_authorized);
    assert_eq!(result.eligible_supervised_label_rows.len(), 3);
    assert_eq!(result.rejected_policy_target_rows.len(), 3);
}
