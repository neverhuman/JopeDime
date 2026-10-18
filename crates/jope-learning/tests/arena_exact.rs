use jope_learning::arena_exact::*;

struct Accept;
impl LawVerifier for Accept {
    fn verify(&self, _: &DiscreteLaw) -> anyhow::Result<()> {
        Ok(())
    }
}
fn law() -> DiscreteLaw {
    serde_json::from_str(include_str!("fixtures/arena-exact-v1/tiny.json")).unwrap()
}
fn ids(l: &DiscreteLaw) -> Vec<String> {
    l.atoms.iter().map(|a| a.action_sha256.clone()).collect()
}

#[test]
fn exact_tiny_law_matches_ordered_draw_enumeration() {
    let l = law();
    let r = exact_best_of_b(&l, &ids(&l), 2, None, &Accept).unwrap();
    // Directly enumerate all 16 ordered pairs; failure is not a score.
    let mut bins = [0u32; 3];
    for a in 0..4 {
        for b in 0..4 {
            let scores = [None, Some(0.0), Some(1.0), Some(1.0)];
            match [scores[a], scores[b]]
                .into_iter()
                .flatten()
                .max_by(f64::total_cmp)
            {
                Some(0.0) => bins[0] += 1,
                Some(1.0) => bins[1] += 1,
                None => bins[2] += 1,
                _ => (),
            }
        }
    }
    assert!(
        (r.atoms.iter().find(|x| x.score == 0.0).unwrap().probability - bins[0] as f64 / 16.0)
            .abs()
            < 1e-12
    );
    assert!(
        (r.atoms.iter().find(|x| x.score == 1.0).unwrap().probability - bins[1] as f64 / 16.0)
            .abs()
            < 1e-12
    );
    assert!((r.failure_mass.powi(2) - bins[2] as f64 / 16.0).abs() < 1e-12);
}
#[test]
fn fixed_seed_independent_monte_carlo_within_predeclared_four_se() {
    const N: usize = 200_000;
    let mut state = 0x30_2026_1018_u64;
    let mut successes = 0usize;
    for _ in 0..N {
        let mut best: Option<u8> = None;
        for _ in 0..2 {
            // Independent xorshift sampler over four equally likely table entries.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let outcome = match state & 3 {
                0 => None,
                1 => Some(0),
                _ => Some(1),
            };
            best = match (best, outcome) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (x, None) => x,
                (None, x) => x,
            };
        }
        if best == Some(1) {
            successes += 1;
        }
    }
    let estimate = successes as f64 / N as f64;
    let exact = 0.75;
    let se = (exact * (1.0 - exact) / N as f64).sqrt();
    assert!(
        (estimate - exact).abs() <= 4.0 * se,
        "estimate={estimate}, SE={se}"
    );
}
#[test]
fn b_zero_requires_incumbent_for_available_result() {
    let l = law();
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 0, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unavailable
    );
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 0, Some(0.25), &Accept)
            .unwrap()
            .status,
        ExactStatus::Exact
    );
}
#[test]
fn failures_remain_explicit_mass() {
    let l = law();
    let r = exact_best_of_b(&l, &ids(&l), 1, None, &Accept).unwrap();
    assert_eq!(r.failure_mass, 0.25);
}
#[test]
fn all_failures_have_typed_outcome() {
    let mut l = law();
    for atom in &mut l.atoms {
        atom.score = None;
    }
    let r = exact_best_of_b(&l, &ids(&l), 3, None, &Accept).unwrap();
    assert_eq!(r.status, ExactStatus::AllDrawsFailed);
    assert_eq!(r.all_draws_fail_probability, 1.0);
}
#[test]
fn incumbent_collects_failures_and_non_improving_outcomes() {
    let l = law();
    let r = exact_best_of_b(&l, &ids(&l), 2, Some(0.5), &Accept).unwrap();
    let total: f64 = r.atoms.iter().map(|a| a.probability).sum();
    assert!((total - 1.0).abs() < 1e-12);
    assert!(
        (r.atoms.iter().find(|a| a.score == 0.5).unwrap().probability - 0.5f64.powi(2)).abs()
            < 1e-12
    );
}
#[test]
fn partial_support_is_not_full_exact() {
    let l = law();
    let r = exact_best_of_b(&l, &ids(&l)[1..], 1, None, &Accept).unwrap();
    assert_eq!(r.status, ExactStatus::Partial);
    assert_eq!(r.missing_mass, 0.25);
}
#[test]
fn duplicate_action_mass_is_aggregated() {
    let mut l = law();
    l.atoms[2].action_sha256 = l.atoms[1].action_sha256.clone();
    l.atoms[2].score = Some(0.0);
    let r = exact_best_of_b(&l, &ids(&l), 1, None, &Accept).unwrap();
    assert_eq!(r.atoms.len(), 1);
    assert_eq!(r.atoms[0].probability, 0.75);
}
#[test]
fn conflicting_duplicate_outcome_refused() {
    let mut l = law();
    l.atoms[2].action_sha256 = l.atoms[1].action_sha256.clone();
    assert!(exact_best_of_b(&l, &ids(&l), 1, None, &Accept).is_err());
}
#[test]
fn unsupported_sampling_is_typed() {
    let mut l = law();
    l.sampling = SamplingMode::Adaptive;
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 3, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unsupported
    );
}
#[test]
fn without_replacement_is_not_iid_formula() {
    let mut l = law();
    l.sampling = SamplingMode::WithoutReplacement;
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 2, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unsupported
    );
}
#[test]
fn correlated_sampling_is_refused() {
    let mut l = law();
    l.sampling = SamplingMode::Correlated;
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 2, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unsupported
    );
}
#[test]
fn deduplicated_sampling_is_refused() {
    let mut l = law();
    l.sampling = SamplingMode::Deduplicated;
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 2, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unsupported
    );
}
#[test]
fn selected_handoff_is_refused() {
    let mut l = law();
    l.sampling = SamplingMode::SelectedHandoff;
    assert_eq!(
        exact_best_of_b(&l, &ids(&l), 2, None, &Accept)
            .unwrap()
            .status,
        ExactStatus::Unsupported
    );
}
#[test]
fn rejects_nonfinite_score() {
    let mut l = law();
    l.atoms[1].score = Some(f64::NAN);
    assert!(exact_best_of_b(&l, &ids(&l), 1, None, &Accept).is_err());
}
#[test]
fn large_b_stays_finite_with_small_mass() {
    let mut l = law();
    l.atoms[1].mass = 1;
    l.atoms[2].mass = 2;
    l.atoms[0].mass = 1;
    let r = exact_best_of_b(&l, &ids(&l), 1_000_000, None, &Accept).unwrap();
    assert!(r
        .atoms
        .iter()
        .all(|x| x.probability.is_finite() && x.probability >= 0.0));
    assert!(r.underflowed_atoms > 0);
}
#[test]
fn auc_declares_clock_failures_and_diagnostic_status() {
    let a = DiscoveryAuc::compute(
        CostClock::Log1p,
        "cost unit",
        10.0,
        0.0,
        &[(2.0, Some(1.0)), (5.0, None)],
        1,
        1,
    )
    .unwrap();
    assert_eq!(a.adoption, "diagnostic_only_d26_unresolved");
    assert_eq!(a.failures, 1);
    assert!(a.interpolation.contains("step"));
}
#[test]
fn auc_rejects_unknown_or_invalid_horizon() {
    assert!(DiscoveryAuc::compute(CostClock::Linear, "units", f64::NAN, 0.0, &[], 0, 0).is_err());
}
