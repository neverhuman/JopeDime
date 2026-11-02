use super::support::*;
use super::*;

#[test]
fn numeric_mutation_preserves_seed_and_source_clamps() {
    let mut parsed = parse_canonical_pipeline(
        "(pipe X (winsor_clip q=0.5 seed=701) (histgbm depth=9 lr=0.9 rounds=999))",
    )
    .unwrap();
    let mut rng = round_rng(11, 9);
    for stage in &mut parsed.stages {
        mutate_parameters(stage, &mut rng);
    }
    assert_eq!(parsed.stages[0].get_u64("seed", 0), 701);
    assert!((0.01..=0.25).contains(&parsed.stages[0].get_f64("q", 0.0)));
    assert!((2..=6).contains(&parsed.stages[1].get_u64("depth", 0)));
    assert!((0.015..=0.30).contains(&parsed.stages[1].get_f64("lr", 0.0)));
    assert!((40..=280).contains(&parsed.stages[1].get_u64("rounds", 0)));
}

#[test]
fn odd_length_crossover_uses_the_source_floor_half_and_other_head() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let parents = vec![
        Program::from_dsl("(pipe X (standardize) (cheb degree=3) (winsor_clip q=0.05) (ridge))"),
        Program::from_dsl(
            "(pipe X (robust_scale) (cheb degree=4) (quantile_bins bins=16) (row_stats) (histgbm))",
        ),
    ];
    let lm = [0.0; 48];
    let batch = proposer
        .propose(&request(&parents, &lm, true), &NoBank, &BTreeSet::new())
        .unwrap();
    let cross = batch
        .proposals
        .iter()
        .find(|p| p.family == ProposalFamily::Crossover)
        .unwrap();
    let parsed = parse_canonical_pipeline(&cross.program.dsl).unwrap();
    assert_eq!(
        parsed
            .stages
            .iter()
            .map(|s| s.op.as_str())
            .collect::<Vec<_>>(),
        ["standardize", "quantile_bins", "row_stats", "histgbm"]
    );
    assert_eq!(
        cross.parents,
        parents
            .iter()
            .map(|p| p.program_sha256.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn empty_capability_intersection_cannot_spin_forever_in_synthesis() {
    let mut proposer = Proposer::new(&[Language::Rust]).unwrap();
    let mut registry = Registry::current();
    for entry in &mut registry.entries {
        for supported in entry.supported.values_mut() {
            *supported = entry.kind == CapabilityKind::Head && entry.name == "ridge_gcv";
        }
    }
    proposer.filter = registry.allowed_ops(&[Language::Rust]).unwrap();
    let lm = [0.0; 48];
    let batch = proposer
        .propose(&request(&[], &lm, false), &NoBank, &BTreeSet::new())
        .unwrap();
    assert!(batch.synthesis_exhausted);
    assert!(batch.proposals.is_empty());
    assert_eq!(batch.attempts[&ProposalFamily::Synthesis], 64 * 8);
    assert_eq!(batch.invalid_or_filtered, 64 * 8);
}

#[test]
fn invalid_parent_hash_or_landmarks_fail_before_bank_reads() {
    let proposer = Proposer::new(&[Language::Rust]).unwrap();
    let bank = Bank::new();
    let mut parents = parents();
    let lm = [0.0; 48];
    parents[0].program_sha256 = "0".repeat(64);
    assert!(proposer
        .propose(&request(&parents, &lm, false), &bank, &BTreeSet::new())
        .is_err());
    let mut lm = lm;
    lm[47] = f64::INFINITY;
    assert!(proposer
        .propose(&request(&[], &lm, false), &bank, &BTreeSet::new())
        .is_err());
    assert!(bank.called.borrow().is_empty());
}
