use super::*;

use crate::selfplay_shadow::{Parentage, SelectionSignals};

fn digest(character: char) -> Sha256Digest {
    Sha256Digest::parse(character.to_string().repeat(64)).unwrap()
}

fn context() -> DatasetContext {
    DatasetContext {
        task: Task::Binary,
        dataset_id: "dataset-1".to_owned(),
        dataset_content_sha256: digest('a'),
        source_family: "family-a".to_owned(),
        context_sha256: digest('b'),
    }
}

fn candidate(origin: Origin, family: &str, index: usize) -> PolicyCandidate {
    let program = format!("{origin:?}-{family}-{index}");
    PolicyCandidate {
        task: Task::Binary,
        dataset_id: "dataset-1".to_owned(),
        dataset_content_sha256: digest('a'),
        context_sha256: digest('b'),
        semantic_sha256: Sha256Digest::of_bytes(program.as_bytes()),
        canonical_program: program,
        source_family: family.to_owned(),
        origin,
        parentage: Parentage::default(),
        producer_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('c')),
        behavior_checkpoint_sha256: (origin == Origin::DlDirect).then(|| digest('c')),
        decoder_log_probability: Some(-1.0),
        proposal_rank: index as u32,
        seed: index as u64,
        signals: SelectionSignals {
            conditional_reward: 1.0 / (index + 1) as f64,
            stability: 0.8,
            incumbent_lift_probability: 0.7,
            diversity: 0.5,
            raw_uncertainty: index as f64,
        },
    }
}

fn pool(d: usize) -> ContextProposalPool {
    ContextProposalPool {
        context: context(),
        dl_direct: (0..d)
            .map(|index| candidate(Origin::DlDirect, "family-a", index))
            .collect(),
        gp_comparator: (0..d)
            .map(|index| candidate(Origin::GpComparator, "family-a", index + 100))
            .collect(),
        gp_exploration: (0..EVALUATIONS_PER_CONTEXT)
            .map(|index| candidate(Origin::GpExploration, "explore", index + 1_000))
            .collect(),
    }
}

#[test]
fn matched_lane_composition_is_exact_at_every_d() {
    for d in DIRECT_BUDGETS {
        let plan = build_context_lane(
            pool(d),
            d,
            &digest('c'),
            &digest('d'),
            &digest('e'),
            &EvaluationIndex::default(),
        )
        .unwrap();
        assert_eq!(plan.actual_d, d);
        assert_eq!(plan.actions.len(), EVALUATIONS_PER_CONTEXT);
        assert_eq!(
            plan.actions
                .iter()
                .filter(|action| action.candidate.origin == Origin::DlDirect)
                .count(),
            d
        );
        assert_eq!(
            plan.actions
                .iter()
                .filter(|action| action.candidate.origin == Origin::GpComparator)
                .count(),
            d
        );
    }
}

#[test]
fn wave_summary_preserves_thin_matched_lanes_and_recovery_eligibility() {
    let mut thin = pool(8);
    thin.dl_direct.truncate(3);
    thin.gp_comparator.truncate(5);
    let lane = build_context_lane(
        thin,
        8,
        &digest('c'),
        &digest('d'),
        &digest('e'),
        &EvaluationIndex::default(),
    )
    .unwrap();
    let wave = SelfPlayWaveV2 {
        schema: SHADOW_WAVE_SCHEMA.into(),
        wave_id: "wave-thin".into(),
        task: Task::Binary,
        parent_checkpoint_sha256: digest('c'),
        evaluator_sha256: digest('d'),
        protocol_sha256: digest('e'),
        contexts: vec![lane],
    };
    let summary = summarize_wave(&wave, &EvaluationIndex::default(), false, false).unwrap();
    assert_eq!(summary.nominal_d, 8);
    assert_eq!(summary.actual_d_total, 3);
    assert_eq!(summary.native_direct_count, 3);
    assert_eq!(summary.matched_gp_count, 3);
    assert!(summary.zero_cross_context_backfill);
    assert!(summary.evidence_eligibility.matched_dl_efficiency);

    let recovery = summarize_wave(&wave, &EvaluationIndex::default(), false, true).unwrap();
    assert!(recovery.evidence_eligibility.reward_and_stability);
    assert!(!recovery.evidence_eligibility.matched_dl_efficiency);
    assert!(!recovery.evidence_eligibility.promotion);
}

#[test]
fn cache_hits_are_filtered_without_consuming_slots() {
    let mut index = EvaluationIndex::default();
    let cached = &pool(8).dl_direct[0];
    let key = EvaluationKeyV2::new(
        Task::Binary,
        digest('a'),
        cached.semantic_sha256.clone(),
        digest('d'),
        digest('e'),
    );
    index.completed.insert(key, digest('f'));
    let plan =
        build_context_lane(pool(8), 8, &digest('c'), &digest('d'), &digest('e'), &index).unwrap();
    assert_eq!(plan.actual_d, 7);
    assert_eq!(plan.actions.len(), EVALUATIONS_PER_CONTEXT);
    assert_eq!(plan.cache_receipts.len(), 1);
    assert_eq!(
        plan.cache_receipts[0].disposition,
        CacheDisposition::ExactCacheHit
    );
}

#[test]
fn cross_protocol_is_only_allowed_in_requalification() {
    let candidate = candidate(Origin::Requalification, "family-a", 9);
    let old = EvaluationKeyV2::new(
        Task::Binary,
        digest('a'),
        candidate.semantic_sha256.clone(),
        digest('d'),
        digest('e'),
    );
    let mut index = EvaluationIndex::default();
    index.completed.insert(old, digest('f'));
    let new = EvaluationKeyV2::new(
        Task::Binary,
        digest('a'),
        candidate.semantic_sha256.clone(),
        digest('d'),
        digest('1'),
    );
    assert_eq!(
        index.classify(&new, EvaluationLane::NormalSelfPlay),
        CacheDecision::CrossProtocolBlocked
    );
    assert_eq!(
        index.classify(&new, EvaluationLane::ProtocolRequalification),
        CacheDecision::Requalification
    );
    assert_eq!(
        build_requalification_lane(
            &context(),
            vec![candidate],
            &digest('c'),
            &digest('d'),
            &digest('1'),
            &index,
            1,
        )
        .unwrap()
        .len(),
        1
    );
}

#[test]
fn thin_contexts_do_not_backfill_from_other_contexts() {
    let mut thin = pool(8);
    thin.gp_exploration.truncate(3);
    let plan = build_context_lane(
        thin,
        8,
        &digest('c'),
        &digest('d'),
        &digest('e'),
        &EvaluationIndex::default(),
    )
    .unwrap();
    assert!(plan.unused_evaluations > 0);
    assert!(plan.zero_backfill);
    assert!(plan
        .actions
        .iter()
        .all(|action| action.candidate.dataset_id == "dataset-1"));
}

#[test]
fn direct_budget_ratchets_only_on_completed_evidence() {
    assert_eq!(
        next_direct_budget(
            8,
            EfficiencyEvidence {
                complete_waves: 3,
                direct_evaluations: 1_024,
                lower_90_efficiency_ratio: 1.01,
                upper_90_efficiency_ratio: 1.2,
                validity: 0.99,
                failure_guard_triggered: false,
            },
        )
        .unwrap(),
        16
    );
    assert_eq!(
        next_direct_budget(
            16,
            EfficiencyEvidence {
                complete_waves: 3,
                direct_evaluations: 1_024,
                lower_90_efficiency_ratio: 0.8,
                upper_90_efficiency_ratio: 0.99,
                validity: 1.0,
                failure_guard_triggered: false,
            },
        )
        .unwrap(),
        8
    );
}

#[test]
fn reservoir_is_breadth_first() {
    let rows = vec![
        CorpusRowRef {
            row_id: "a1".into(),
            dataset_id: "a".into(),
            observed_at_ms: 2,
        },
        CorpusRowRef {
            row_id: "a2".into(),
            dataset_id: "a".into(),
            observed_at_ms: 1,
        },
        CorpusRowRef {
            row_id: "b1".into(),
            dataset_id: "b".into(),
            observed_at_ms: 1,
        },
    ];
    let ordered = breadth_first_reservoir(&rows);
    assert_eq!(&ordered[..2], ["a1", "b1"]);
    assert_eq!(ordered[2], "a2");
}

#[test]
fn encoding_chunks_are_content_addressed_and_small() {
    let rows = (0..600)
        .map(|index| EncodingDataset {
            dataset_id: format!("d-{index:04}"),
            content_sha256: Sha256Digest::of_bytes(format!("{index}").as_bytes()),
            encoded: false,
        })
        .collect::<Vec<_>>();
    let chunks = plan_encoding_chunks(&rows, &digest('e'));
    assert_eq!(chunks.len(), 3);
    assert!(chunks.iter().all(|chunk| chunk.datasets.len() <= 256));
    assert!(chunks
        .iter()
        .all(|chunk| chunk.preemptible && chunk.reserved_cpu_cores == 4));
    assert!(chunks.iter().all(|chunk| chunk.validate().is_ok()));
}

#[test]
fn frontier_rotates_before_any_dataset_repeats() {
    let mut datasets = (0..180)
        .map(|index| BreadthDataset {
            task: Task::Binary,
            dataset_id: format!("dataset-{index:03}"),
            dataset_content_sha256: Sha256Digest::of_bytes(
                format!("dataset-{index:03}").as_bytes(),
            ),
            source_family: format!("family-{}", index % 20),
            encoded: true,
            sealed: false,
            eligible: true,
            observation_count: 0,
            frontier_visits: 0,
            last_hard_wave: None,
            hard_score: index as f64,
            uncertainty: (index % 7) as f64,
            stable_anchor: true,
        })
        .collect::<Vec<_>>();
    let first = plan_breadth_cohort(Task::Binary, 1, &datasets).unwrap();
    let first_frontier = first
        .members
        .iter()
        .filter(|row| row.role == CohortRole::Frontier)
        .map(|row| row.dataset_id.clone())
        .collect::<BTreeSet<_>>();
    for row in &mut datasets {
        if first_frontier.contains(&row.dataset_id) {
            row.frontier_visits += 1;
        }
    }
    let second = plan_breadth_cohort(Task::Binary, 2, &datasets).unwrap();
    let second_frontier = second
        .members
        .iter()
        .filter(|row| row.role == CohortRole::Frontier)
        .map(|row| row.dataset_id.clone())
        .collect::<BTreeSet<_>>();
    assert!(first_frontier.is_disjoint(&second_frontier));
    assert_eq!(first.frontier_target, 48);
    assert_eq!(second.frontier_target, 48);
}

#[test]
fn hard_contexts_observe_three_complete_wave_cooldown() {
    let datasets = (0..180)
        .map(|index| BreadthDataset {
            task: Task::Regression,
            dataset_id: format!("dataset-{index:03}"),
            dataset_content_sha256: Sha256Digest::of_bytes(
                format!("regression-{index:03}").as_bytes(),
            ),
            source_family: format!("family-{}", index % 20),
            encoded: true,
            sealed: false,
            eligible: true,
            observation_count: 0,
            frontier_visits: 0,
            last_hard_wave: Some(10),
            hard_score: index as f64,
            uncertainty: 1.0,
            stable_anchor: true,
        })
        .collect::<Vec<_>>();
    assert!(plan_breadth_cohort(Task::Regression, 13, &datasets).is_err());
    assert!(plan_breadth_cohort(Task::Regression, 14, &datasets).is_ok());
}
