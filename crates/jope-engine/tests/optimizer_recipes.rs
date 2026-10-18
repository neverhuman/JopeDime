#[path = "../src/training/loss_admission.rs"]
mod loss_admission;
#[path = "../src/training/optimizer_recipe.rs"]
mod optimizer_recipe;

use loss_admission::{
    awr_weight, pkpo_weights, rloo_weights, AwrAdmission, AwrInput, HeadGradient, KlPolicy,
    PerHeadEligibility, StickyKlGuard, UpdateAccounting,
};
use optimizer_recipe::{
    empty_state_sha256, ContinuationDecision, ObjectiveId, OptimizerRecipe, OptimizerStateIdentity,
    ACCEPTED_CONTROL_ID,
};

const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

#[test]
fn rloo_singleton_zero_ties_and_nonfinite_are_explicit() {
    assert_eq!(rloo_weights(&[7.0], false).unwrap(), vec![7.0]);
    assert_eq!(rloo_weights(&[2.0, 2.0, 2.0], false).unwrap(), vec![0.0; 3]);
    let weights = rloo_weights(&[1.0, 3.0, 5.0], false).unwrap();
    assert_eq!(weights, vec![-1.0, 0.0, 1.0]);
    assert!(rloo_weights(&[1.0, 2.0], true).is_err());
    assert!(rloo_weights(&[f64::NAN], false).is_err());
    assert!(rloo_weights(&[], false).is_err());
}

fn pkpo_reference(u: &[f64], h: f64, b: usize) -> Vec<f64> {
    // Independent subset marginal: remove each draw and compare it with
    // max(h, all remaining utilities), then average over all C(G,B) subsets.
    fn subsets(n: usize, b: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == b {
            out.push(cur.clone());
            return;
        }
        for i in start..n {
            cur.push(i);
            subsets(n, b, i + 1, cur, out);
            cur.pop();
        }
    }
    let clipped: Vec<f64> = u.iter().map(|x| h.max(*x)).collect();
    let mut sets = Vec::new();
    subsets(u.len(), b, 0, &mut Vec::new(), &mut sets);
    let mut out = vec![0.0; u.len()];
    for set in &sets {
        for i in set {
            let baseline = set
                .iter()
                .copied()
                .filter(|j| j != i)
                .map(|j| clipped[j])
                .fold(h, f64::max);
            out[*i] += (clipped[*i] - baseline).max(0.0) / sets.len() as f64;
        }
    }
    out
}

#[test]
fn pkpo_matches_independent_subset_fixture_including_ties() {
    let utilities = [0.1, 1.2, 1.2, -0.5, 0.3, 0.3, 0.8, 0.0];
    for b in [1, 2, 4, 8] {
        let actual = pkpo_weights(&utilities, 0.2, b).unwrap();
        let expected = pkpo_reference(&utilities, 0.2, b);
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-12, "B={b}: {a} != {e}");
        }
    }
    assert_eq!(
        pkpo_weights(&[0.0, 0.0, 0.0], 0.2, 2).unwrap(),
        vec![0.0; 3]
    );
    assert!(pkpo_weights(&[1.0], 0.0, 0).is_err());
    assert!(pkpo_weights(&[f64::INFINITY], 0.0, 1).is_err());
}

#[test]
fn awr_requires_role_exact_behavior_support_and_positive_scale() {
    let base = AwrInput {
        advantage: 2.0,
        pooled_scale: 0.0,
        beta: 1.0,
        max_weight: 10.0,
        exact_eos_behavior_logprob: true,
        support_match: true,
        train_role: true,
    };
    assert_eq!(
        awr_weight(&base),
        AwrAdmission::Ineligible {
            reason: "pooled_scale_not_positive_finite"
        }
    );
    let valid = AwrInput {
        pooled_scale: 2.0,
        ..base.clone()
    };
    assert_eq!(
        awr_weight(&valid),
        AwrAdmission::Eligible {
            normalized_advantage: 1.0,
            weight: std::f64::consts::E
        }
    );
    assert_eq!(
        awr_weight(&AwrInput {
            train_role: false,
            ..valid.clone()
        }),
        AwrAdmission::Ineligible {
            reason: "not_train_role"
        }
    );
    assert_eq!(
        awr_weight(&AwrInput {
            exact_eos_behavior_logprob: false,
            ..valid.clone()
        }),
        AwrAdmission::Ineligible {
            reason: "missing_exact_eos_behavior_logprob"
        }
    );
    assert_eq!(
        awr_weight(&AwrInput {
            support_match: false,
            ..valid
        }),
        AwrAdmission::Ineligible {
            reason: "behavior_support_mismatch"
        }
    );
    let eligible = PerHeadEligibility {
        head: "reward".into(),
        train_role: true,
        exact_eos_behavior_logprob: true,
        support_match: true,
        complete_sibling_group: true,
        positive_finite_scale: true,
        reason: None,
    };
    assert!(eligible.require().is_ok());
}

#[test]
fn sticky_kl_distinguishes_alarm_from_hold_and_binds_identity() {
    let policy = KlPolicy {
        alarm_mean: 0.01,
        hold_mean: 0.02,
        hold_max: 0.05,
    };
    let mut guard = StickyKlGuard::new(A.into(), B.into()).unwrap();
    guard.observe(&policy, 1, 10, 0.01, 0.02).unwrap();
    assert_eq!(guard.alarm_count, 1);
    assert!(!guard.tripped);
    guard.observe(&policy, 2, 20, 0.02, 0.03).unwrap();
    assert!(guard.tripped);
    assert_eq!(guard.first_trip_epoch, Some(2));
    guard.observe(&policy, 3, 30, 0.001, 0.001).unwrap();
    assert!(guard.tripped);
    assert_eq!(guard.first_trip_epoch, Some(2));
    assert!(guard.check_identity(C, B).is_err());
    assert!(guard.observe(&policy, 3, 31, 0.001, 0.001).is_err());
    assert!(guard.observe(&policy, 4, 40, -0.001, 0.0).is_err());
}

fn state(moments: &str, step: u64, schedule: &str, ema: Option<&str>) -> OptimizerStateIdentity {
    OptimizerStateIdentity {
        moments_sha256: moments.into(),
        step,
        schedule_state_sha256: schedule.into(),
        ema_state_sha256: ema.map(str::to_owned),
    }
}

#[test]
fn accepted_control_and_continuations_are_recipe_and_state_bound() {
    let objective = ObjectiveId {
        name: "lineage_b".into(),
        version: 1,
        manifest_sha256: A.into(),
    };
    let control = OptimizerRecipe::accepted_control(objective, B.into()).unwrap();
    assert_eq!(control.recipe_id, ACCEPTED_CONTROL_ID);
    assert_eq!(control.optimizer.learning_rate, 1e-4);
    assert_eq!(control.optimizer.weight_decay, 0.01);
    assert_eq!(
        control.digest_sha256().unwrap(),
        control.clone().digest_sha256().unwrap()
    );
    let alt = control
        .clone()
        .alternative("cosine_schedule_ablation_v1".into(), C.into())
        .unwrap();
    assert_ne!(
        control.digest_sha256().unwrap(),
        alt.digest_sha256().unwrap()
    );
    let parent_state = state(A, 0, B, None);
    control
        .continuation_from(
            &control,
            &parent_state,
            &parent_state,
            ContinuationDecision::ExactResume,
        )
        .unwrap();
    assert!(alt
        .continuation_from(
            &control,
            &parent_state,
            &parent_state,
            ContinuationDecision::ExactResume
        )
        .is_err());
    control
        .continuation_from(
            &control,
            &parent_state,
            &parent_state,
            ContinuationDecision::CarryMoments,
        )
        .unwrap();
    let reset = state(&empty_state_sha256(), 0, B, None);
    control
        .continuation_from(
            &control,
            &reset,
            &parent_state,
            ContinuationDecision::ResetMomentsWarmStart,
        )
        .unwrap();
    assert!(control
        .continuation_from(
            &control,
            &parent_state,
            &parent_state,
            ContinuationDecision::ResetMomentsWarmStart
        )
        .is_err());
}

#[test]
fn update_accounting_separates_counts_gradients_and_parameter_change() {
    let good = UpdateAccounting {
        computed_updates: 5,
        applied_updates: 4,
        committed_updates: 3,
        active_heads: vec![HeadGradient {
            head: "reward".into(),
            active_parameter_count: 2,
            gradient_l2: 0.25,
        }],
        parameters_before_sha256: A.into(),
        parameters_after_sha256: B.into(),
        optimizer_state_sha256: C.into(),
    };
    assert!(good.validate().is_ok());
    let decay_only = UpdateAccounting {
        active_heads: vec![HeadGradient {
            head: "reward".into(),
            active_parameter_count: 0,
            gradient_l2: 0.0,
        }],
        parameters_after_sha256: A.into(),
        ..good.clone()
    };
    assert!(decay_only.validate().is_err());
    assert!(UpdateAccounting {
        applied_updates: 6,
        ..good
    }
    .validate()
    .is_err());
}
