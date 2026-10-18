use feat_math::kernel::strategy::{
    migration::{admit, MigrationProposal},
    modern_gp::{
        complete, exhausted, CandidateConfig, CandidatePlan, SearchOutcome, VerifiedIncumbent,
    },
    race::{ResourceLadder, Rung, RungLimit, RungResult},
    selection::{account_mixture, multiplicities, MixtureSource, ScoredCandidate},
};

fn candidate(id: &str, sequence: u64, score: f64) -> ScoredCandidate {
    ScoredCandidate {
        proposal_sequence: sequence,
        genome_id: id.into(),
        score,
    }
}

#[test]
fn cheap_rung_reversal_cannot_name_final_champion() {
    let results = [
        RungResult {
            candidate: candidate("cheap-winner", 0, 0.1),
            rung: Rung::Cheap,
            source_head: "head-a".into(),
            eligible: true,
        },
        RungResult {
            candidate: candidate("final-winner", 1, 0.8),
            rung: Rung::Medium,
            source_head: "head-a".into(),
            eligible: true,
        },
        RungResult {
            candidate: candidate("cheap-winner", 0, 2.0),
            rung: Rung::Final,
            source_head: "head-old".into(),
            eligible: true,
        },
        RungResult {
            candidate: candidate("final-winner", 1, 1.0),
            rung: Rung::Final,
            source_head: "head-a".into(),
            eligible: true,
        },
    ];
    assert_eq!(
        complete(&results, "head-a"),
        SearchOutcome::Champion(
            feat_math::kernel::strategy::race::ChampionDecision::Verified {
                genome_id: "final-winner".into(),
                score: 1.0,
                proposal_sequence: 1,
            }
        )
    );
}

#[test]
fn exhaustion_returns_verified_incumbent_or_explicit_unavailable() {
    let incumbent = VerifiedIncumbent {
        genome_id: "incumbent".into(),
        score: 3.0,
        verification_id: "receipt-1".into(),
    };
    assert_eq!(
        exhausted(Some(incumbent.clone()), 42),
        SearchOutcome::Incumbent(incumbent)
    );
    assert_eq!(
        exhausted(None, 42),
        SearchOutcome::Unavailable {
            reason: "budget exhausted without verified incumbent",
            effort_charged: 42,
        }
    );
}

#[test]
fn absent_source_mass_and_deduplicated_realized_mixture_are_recorded() {
    let sources = [
        MixtureSource {
            source_id: "seed".into(),
            requested_mass: 2,
            admitted_genomes: vec!["a".into(), "a".into()],
        },
        MixtureSource {
            source_id: "archive".into(),
            requested_mass: 1,
            admitted_genomes: vec!["a".into(), "b".into()],
        },
    ];
    let mass = account_mixture(&sources);
    assert_eq!(
        (
            mass[0].requested_mass,
            mass[0].admitted_mass,
            mass[0].realized_mass
        ),
        (2, 2, 1)
    );
    assert_eq!(mass[1].admitted_mass, 2);
    assert_eq!(mass[1].realized_mass, 1);
    assert_eq!(mass[0].behavior_probability, None);
    assert_eq!(mass[1].behavior_probability, None);
    assert_eq!(multiplicities(&sources).get("a"), Some(&3));
    let absent = account_mixture(&[MixtureSource {
        source_id: "empty".into(),
        requested_mass: 3,
        admitted_genomes: vec![],
    }]);
    assert_eq!(absent[0].behavior_probability, None);
}

#[test]
fn candidate_profiles_remain_distinct_and_bounded() {
    assert!(CandidateConfig::island128().validate());
    assert_eq!(
        CandidateConfig::island128().identity(),
        "modern-gp-island128.v1"
    );
    assert_eq!(
        CandidateConfig::steady_state(16).unwrap().identity(),
        "modern-gp-steady-state.v1"
    );
    assert!(CandidateConfig::steady_state(15).is_none());
    assert!(CandidateConfig::steady_state(65).is_none());
    let ladder = ResourceLadder {
        cheap: RungLimit {
            evaluations: 8,
            effort: 8,
            memory_bytes: 1024,
        },
        medium: RungLimit {
            evaluations: 16,
            effort: 16,
            memory_bytes: 2048,
        },
        deep: RungLimit {
            evaluations: 32,
            effort: 32,
            memory_bytes: 4096,
        },
        final_rung: RungLimit {
            evaluations: 64,
            effort: 64,
            memory_bytes: 8192,
        },
    };
    assert!(CandidatePlan {
        config: CandidateConfig::island128(),
        ladder
    }
    .validate());
}

#[test]
fn migration_admission_uses_sequence_order_and_resource_caps() {
    let proposal = |sequence, id: &str| MigrationProposal {
        source_island: 0,
        destination_island: 1,
        proposal_sequence: sequence,
        genome_id: id.into(),
        effort: 2,
        bytes: 8,
    };
    let admitted = admit(vec![proposal(2, "later"), proposal(1, "first")], 2, 8);
    assert_eq!(admitted.admitted[0].genome_id, "first");
    assert_eq!(admitted.rejected[0].0.genome_id, "later");
    assert_eq!(admitted.rejected[0].1, "effort cap");
}
