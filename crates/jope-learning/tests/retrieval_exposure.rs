use jope_learning::retrieval::{self, Candidate, RetrievalMode, RetrievalRequest};
use jope_pipeline::{
    seeds::{self, SeedSourceV2},
    sha256_hex,
};
use serde::Deserialize;
use std::collections::BTreeMap;

const FIXTURE: &str = include_str!("fixtures/retrieval-exposure-v1/lineage.json");
#[derive(Deserialize)]
struct Histories {
    own_parent_families: Vec<String>,
    own_dataset_families: Vec<String>,
    own_program_sha256: Vec<String>,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    id: String,
    parent_families: Vec<String>,
    dataset_families: Vec<String>,
    program: String,
    role: String,
    context: Vec<f64>,
}

fn input() -> (RetrievalRequest, Vec<Candidate>) {
    let fixture: Histories = serde_json::from_str(FIXTURE).unwrap();
    let own_hash = sha256_hex(fixture.own_program_sha256[0].as_bytes());
    let request = RetrievalRequest {
        request_id: "request-1".into(),
        own_parent_families: fixture.own_parent_families,
        own_dataset_families: fixture.own_dataset_families,
        own_program_sha256: vec![own_hash],
        role_manifest_sha256: "2".repeat(64),
        context: vec![0.0, 10.0],
        eligible_support: fixture.cases.iter().map(|x| x.id.clone()).collect(),
        cap: 3,
        requested_by_source: BTreeMap::from([(SeedSourceV2::OtherLineageRetrieval, 3)]),
        snapshot_sha256: "1".repeat(64),
        cold_cost_ms: 40,
        warm_cost_ms: 5,
        cache_warmup_ms: 20,
    };
    let candidates = fixture
        .cases
        .into_iter()
        .map(|c| {
            let genome = match c.program.as_str() {
                "own-program" => fixture.own_program_sha256[0].as_str(),
                "program-e" => "(pipe X (robust_scale) (ridge))",
                "program-f" => "(pipe X (rank) (ridge))",
                "program-a" => "(pipe X (blom) (ridge))",
                "program-b" => "(pipe X (quantile) (ridge))",
                "program-c" => "(pipe X (minmax) (ridge))",
                _ => "(pipe X (standardize) (ridge_gcv))",
            }
            .to_owned();
            Candidate {
                candidate_id: c.id,
                program_sha256: sha256_hex(genome.as_bytes()),
                genome,
                parent_families: c.parent_families,
                dataset_families: c.dataset_families,
                context: c.context,
                role: c.role,
                role_manifest_sha256: "2".repeat(64),
            }
        })
        .collect();
    (request, candidates)
}

#[test]
fn own_hwm_parent_dataset_aliases_and_indirect_ancestry_never_become_transfer() {
    let (request, candidates) = input();
    let result = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    assert_eq!(
        result.selected_candidate_ids,
        ["eligible-near", "eligible-far"]
    );
    for excluded in [
        "own-parent-descendant",
        "own-dataset-alias",
        "same-program-renamed-bank-key",
        "unknown-ancestry",
        "validation-role",
    ] {
        assert!(!result
            .selected_candidate_ids
            .iter()
            .any(|id| id == excluded));
    }
}

#[test]
fn cold_warm_and_shuffled_controls_share_the_frozen_eligible_support() {
    let (request, candidates) = input();
    let cold = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    let warm = retrieval::materialize(&request, &candidates, RetrievalMode::Warm).unwrap();
    let shuffled =
        retrieval::materialize(&request, &candidates, RetrievalMode::ShuffledContext).unwrap();
    assert_eq!(cold.eligible_support_sha256, warm.eligible_support_sha256);
    assert_eq!(
        cold.eligible_support_sha256,
        shuffled.eligible_support_sha256
    );
    assert_eq!(cold.seeds.draws.len(), 2);
    assert_eq!(
        cold.seeds.draws[0].source,
        SeedSourceV2::OtherLineageRetrieval
    );
    assert_eq!(
        cold.selected_candidate_ids,
        ["eligible-near", "eligible-far"]
    );
    assert_eq!(
        shuffled.selected_candidate_ids,
        ["eligible-far", "eligible-near"]
    );
    assert_eq!(cold.candidate_counts.offered, 7);
    assert_eq!(cold.candidate_counts.own_lineage, 3);
    assert_eq!(cold.candidate_counts.missing_ancestry, 1);
    assert_eq!(cold.candidate_counts.non_train_role, 1);
    assert_eq!(cold.candidate_counts.eligible, 2);
}

#[test]
fn warm_cache_cost_and_warmup_are_disclosed_in_total_resource_clock() {
    let (request, candidates) = input();
    let warm = retrieval::materialize(&request, &candidates, RetrievalMode::Warm).unwrap();
    let cold = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    assert_eq!(warm.cache_warmup_ms, 20);
    assert_eq!(warm.resource_cost_ms, 25);
    assert_eq!(cold.resource_cost_ms, 40);
}

#[test]
fn retrieval_draws_materialize_through_the_pipeline_v2_seed_import() {
    let (request, candidates) = input();
    let retrieved = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    let admitted = seeds::admit_v2(&retrieved.seeds).unwrap();
    assert_eq!(
        admitted.offered_by_source[&SeedSourceV2::OtherLineageRetrieval],
        2
    );
    assert_eq!(
        admitted.realized_by_source[&SeedSourceV2::OtherLineageRetrieval],
        2
    );
    assert_eq!(admitted.seeds.len(), 2);
}

#[test]
fn missing_snapshot_support_or_ancestry_fails_closed() {
    let (mut request, candidates) = input();
    request.snapshot_sha256.clear();
    assert!(retrieval::materialize(&request, &candidates, RetrievalMode::Cold).is_err());
    let (mut request, candidates) = input();
    request.eligible_support.clear();
    assert!(retrieval::materialize(&request, &candidates, RetrievalMode::Cold).is_err());
    let (request, mut candidates) = input();
    candidates[5].role_manifest_sha256 = "3".repeat(64);
    assert!(retrieval::materialize(&request, &candidates, RetrievalMode::Cold).is_err());
}

#[test]
fn ancestry_and_digest_provenance_are_retained_in_immutable_seed_draws() {
    let (request, candidates) = input();
    let result = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    let draw = &result.seeds.draws[0];
    assert_eq!(
        draw.ancestry.as_ref().unwrap().parent_families,
        ["parent-2"]
    );
    assert_eq!(draw.program_sha256, sha256_hex(draw.genome.as_bytes()));
    assert_eq!(result.seeds.source_snapshot_sha256, request.snapshot_sha256);
}

#[test]
fn selected_support_membership_and_tie_order_are_deterministic() {
    let (mut request, mut candidates) = input();
    request.cap = 2;
    candidates[6].context = vec![0.0, 10.0];
    let first = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    let second = retrieval::materialize(&request, &candidates, RetrievalMode::Cold).unwrap();
    assert_eq!(first.selected_candidate_ids, second.selected_candidate_ids);
    assert_eq!(
        first.selected_candidate_ids,
        ["eligible-far", "eligible-near"]
    );
}
