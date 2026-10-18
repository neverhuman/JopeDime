use jope_learning::canon::sha256_hex;
use jope_learning::retention::policy::{Policy, DEFAULT_POLICY_JSON};
use jope_learning::retention::value::{
    Candidate, ConsumerClosure, LearningSignals, PhysicalObject, Recommendation, ReplacementProof,
    RetentionPlanV2, RetentionPolicyV2, ValueWeights,
};

const NOW: u64 = 1_800_000_000;

fn policy() -> RetentionPolicyV2 {
    let v1 = Policy::parse(DEFAULT_POLICY_JSON.as_bytes()).unwrap();
    RetentionPolicyV2::from_v1(
        &v1,
        ValueWeights {
            influence: 1.0,
            novelty: 0.5,
            uncertainty: 0.25,
            reproduction_cost: 0.1,
            storage_cost: 2.0,
        },
    )
    .unwrap()
}

fn object(key: &str, version: &str, digest: &str, bytes: u64) -> PhysicalObject {
    PhysicalObject {
        bucket: "research-bucket".into(),
        key: key.into(),
        version_id: version.into(),
        sha256: digest.into(),
        bytes,
    }
}

fn candidate(source: PhysicalObject) -> Candidate {
    Candidate {
        sources: vec![source],
        hard_t0: false,
        active_pin: false,
        active_promise: false,
        score_row: false,
        inventory_at_unix: Some(NOW - 60),
        consumer_closure: Some(ConsumerClosure {
            scanned_at_unix: NOW - 30,
            observed_root_generation: "roots-42".into(),
            current_root_generation: "roots-42".into(),
            active_consumers: Vec::new(),
            unresolved: Vec::new(),
        }),
        replacement: None,
        signals: Some(LearningSignals {
            influence: 0.03,
            novelty: 0.5,
            uncertainty: 0.2,
            reproduction_cost_seconds: 120.0,
            reproduction_probability: 0.25,
            storage_usd_per_month: 0.02,
            surprise: 0.4,
        }),
        context_rank: Some(100),
        lineage_rank: Some(100),
        reservoir_selected: Some(false),
        ranking_current: true,
    }
}

fn keeper(source: PhysicalObject) -> Candidate {
    let mut item = candidate(source);
    item.context_rank = Some(1);
    item
}

fn item_for_source<'a>(
    plan: &'a RetentionPlanV2,
    key: &str,
) -> &'a jope_learning::retention::value::PlanItem {
    plan.items
        .iter()
        .find(|item| item.source.iter().any(|source| source.key == key))
        .unwrap()
}

fn plan(candidates: Vec<Candidate>) -> RetentionPlanV2 {
    RetentionPlanV2::build(NOW, 86_400, &"a".repeat(64), policy(), candidates).unwrap()
}

#[test]
fn stale_september_inventory_cannot_propose_a_present_change() {
    let mut item = candidate(object("runs/old.log", "v7", &"b".repeat(64), 500));
    item.inventory_at_unix = Some(NOW - 35 * 86_400);
    item.replacement = Some(ReplacementProof::ExactDuplicate {
        replacement: object("runs/primary.log", "v4", &"b".repeat(64), 500),
        content_hash_verified: true,
        restore_tested: true,
    });
    let original = item.clone();
    let primary = keeper(object("runs/primary.log", "v4", &"b".repeat(64), 500));
    let result = plan(vec![item, primary]);
    let row = item_for_source(&result, "runs/old.log");
    assert_eq!(row.recommendation, Recommendation::Hold);
    assert!(row
        .unresolved
        .iter()
        .any(|gap| gap.contains("inventory is stale")));
    assert_eq!(row.hypothetical_bytes_after_review, 500);
    assert_eq!(row.source[0], original.sources[0]);
}

#[test]
fn exact_duplicate_requires_current_consumer_and_recovery_proofs() {
    let digest = "c".repeat(64);
    let mut complete = candidate(object("raw/run-a", "source-v1", &digest, 900));
    complete.replacement = Some(ReplacementProof::ExactDuplicate {
        replacement: object("raw/keeper", "keeper-v9", &digest, 900),
        content_hash_verified: true,
        restore_tested: true,
    });
    let primary = keeper(object("raw/keeper", "keeper-v9", &digest, 900));
    let complete_plan = plan(vec![complete, primary.clone()]);
    assert_eq!(
        item_for_source(&complete_plan, "raw/run-a").recommendation,
        Recommendation::ReviewDuplicate
    );
    assert_eq!(complete_plan.current_bytes, 1800);
    assert_eq!(complete_plan.hypothetical_bytes_after_review, 900);
    assert_eq!(complete_plan.hypothetical_bytes_avoided, 900);

    let mut incomplete = candidate(object("raw/run-b", "source-v1", &digest, 900));
    incomplete
        .consumer_closure
        .as_mut()
        .unwrap()
        .current_root_generation = "roots-43".into();
    incomplete.replacement = Some(ReplacementProof::ExactDuplicate {
        replacement: object("raw/keeper", "keeper-v9", &digest, 900),
        content_hash_verified: true,
        restore_tested: true,
    });
    let primary = keeper(object("raw/keeper", "keeper-v9", &digest, 900));
    let held = plan(vec![incomplete, primary]);
    let row = item_for_source(&held, "raw/run-b");
    assert_eq!(row.recommendation, Recommendation::Hold);
    assert!(row.unresolved.iter().any(|gap| gap.contains("generation")));
}

#[test]
fn verified_pack_plan_accounts_for_added_pack_bytes_without_applying() {
    let first = object("small/a", "a-v1", &"1".repeat(64), 400);
    let second = object("small/b", "b-v1", &"2".repeat(64), 300);
    let pack = object("packs/run/1", "pack-v1", &"3".repeat(64), 500);
    let mut item = candidate(first.clone());
    item.sources.push(second.clone());
    item.replacement = Some(ReplacementProof::Pack {
        replacement: pack,
        members: vec![first, second],
        manifest_verified: true,
        restore_tested: true,
    });
    let original = item.clone();
    let result = plan(vec![item]);
    assert_eq!(result.items[0].recommendation, Recommendation::ReviewPack);
    assert_eq!(result.current_bytes, 700);
    assert_eq!(result.hypothetical_bytes_after_review, 500);
    assert_eq!(result.hypothetical_bytes_avoided, 200);
    assert_eq!(result.items[0].source, original.sources);
}

#[test]
fn t0_pins_score_rows_and_keeper_reservoir_are_never_candidates() {
    let digest = "d".repeat(64);
    let mut t0 = candidate(object("sealed/evidence", "v1", &digest, 100));
    t0.hard_t0 = true;
    t0.replacement = Some(ReplacementProof::ExactDuplicate {
        replacement: object("primary", "v1", &digest, 100),
        content_hash_verified: true,
        restore_tested: true,
    });
    let mut keeper = candidate(object("programs/keeper", "v2", &"e".repeat(64), 200));
    keeper.context_rank = Some(1);
    let mut reservoir = candidate(object("programs/reservoir", "v3", &"f".repeat(64), 300));
    reservoir.reservoir_selected = Some(true);
    let mut score = candidate(object("scores/row", "v4", &"0".repeat(64), 10));
    score.score_row = true;
    let result = plan(vec![t0, keeper, reservoir, score]);
    assert_eq!(
        item_for_source(&result, "sealed/evidence").recommendation,
        Recommendation::KeepT0
    );
    assert!(result.items.iter().all(|item| matches!(
        item.recommendation,
        Recommendation::Keep | Recommendation::KeepT0 | Recommendation::KeepScoreRows
    )));
    assert_eq!(
        item_for_source(&result, "scores/row").recommendation,
        Recommendation::KeepScoreRows
    );
    assert_eq!(result.hypothetical_bytes_avoided, 0);
}

#[test]
fn supersession_requires_qualified_margin_grace_and_score_row_retention() {
    let source = object("exe/old", "v1", &"1".repeat(64), 800);
    let replacement = object("exe/champion", "v3", &"2".repeat(64), 700);
    let mut valid = candidate(source.clone());
    valid.replacement = Some(ReplacementProof::Supersession {
        replacement: replacement.clone(),
        qualified_champion: true,
        relative_margin: policy().margin_rel + 0.01,
        days_since_dominance: policy().grace_days,
        score_rows_retained: true,
        recovery_closure_verified: true,
        restore_tested: true,
    });
    let champion_keeper = keeper(replacement.clone());
    let valid_plan = plan(vec![valid, champion_keeper.clone()]);
    assert_eq!(
        item_for_source(&valid_plan, "exe/old").recommendation,
        Recommendation::ReviewSupersession
    );

    let mut unqualified = candidate(source);
    unqualified.replacement = Some(ReplacementProof::Supersession {
        replacement,
        qualified_champion: false,
        relative_margin: 0.9,
        days_since_dominance: 365,
        score_rows_retained: true,
        recovery_closure_verified: true,
        restore_tested: true,
    });
    let unqualified_plan = plan(vec![unqualified, champion_keeper]);
    assert_eq!(
        item_for_source(&unqualified_plan, "exe/old").recommendation,
        Recommendation::Hold
    );
}

#[test]
fn value_signals_rank_only_and_v2_policy_rejects_invalid_knobs() {
    let high = candidate(object("x/high", "v1", &"1".repeat(64), 100));
    let mut low = candidate(object("x/low", "v1", &"2".repeat(64), 100));
    low.signals.as_mut().unwrap().influence = 0.0;
    low.signals.as_mut().unwrap().novelty = 0.0;
    low.signals.as_mut().unwrap().uncertainty = 0.0;
    low.signals.as_mut().unwrap().reproduction_cost_seconds = 0.0;
    let high_value = plan(vec![high]).items[0].values[0].marginal_value.unwrap();
    let low_value = plan(vec![low]).items[0].values[0].marginal_value.unwrap();
    assert!(high_value > low_value);

    let mut invalid = policy();
    invalid.reservoir_rate = 1.5;
    assert!(invalid.validate().is_err());
    invalid = policy();
    invalid.weights.influence = f64::NAN;
    assert!(invalid.validate().is_err());
}

#[test]
fn plan_is_deterministic_and_source_is_non_destructive() {
    let item = candidate(object("runs/a", "v1", &"a".repeat(64), 50));
    let original = item.clone();
    let first = plan(vec![item.clone()]);
    let second = plan(vec![item.clone()]);
    assert_eq!(first.plan_sha256, second.plan_sha256);
    assert_eq!(item.sources, original.sources);
    assert_eq!(first.schema, "jope.retention-plan/v2");
    assert_eq!(first.items[0].recommendation, Recommendation::Hold);
}

#[test]
fn new_classifier_source_contains_no_storage_mutation_primitive() {
    let source = include_str!("../src/retention/value.rs");
    for forbidden in [
        "object_store",
        "jope_datasets",
        "remove_file",
        "remove_dir",
        "process::Command",
        "DeleteObject",
        "delete_object",
        "PutObject",
        "put_object",
        "PutBucketLifecycle",
        "CopyObject",
    ] {
        assert!(!source.contains(forbidden), "value.rs contains {forbidden}");
    }
    assert_eq!(sha256_hex(source.as_bytes()).len(), 64);
}
