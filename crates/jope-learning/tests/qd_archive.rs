use jope_learning::lake::archive::{
    Archive, ArchiveConfig, ArchiveScope, Candidate, CandidateRole, Decision, Fidelity,
    RetrievalPolicy, RetrievalRequest,
};
use std::cell::Cell;
use std::collections::BTreeSet;

fn config() -> ArchiveConfig {
    ArchiveConfig {
        descriptor_version: "movement-energy-v1".into(),
        descriptor_identity_sha256: "a".repeat(64),
        scope: ArchiveScope::PerContext,
        max_cells: 8,
        max_bytes: 32_000,
        max_events: 16,
    }
}

fn candidate(id: u32, quality: f64) -> Candidate {
    Candidate {
        evaluator_identity_sha256: "e".repeat(64),
        fidelity: Fidelity::Final,
        role: CandidateRole::Specialist,
        context_id: format!("context-{id}"),
        family_id: format!("family-{id}"),
        lineage_id: format!("lineage-{id}"),
        exposure_ancestry: BTreeSet::new(),
        observation_sha256: format!("{id:064x}"),
        quality,
        descriptor_bins: vec![id, 0],
    }
}

#[test]
fn descriptor_version_changes_archive_identity() {
    let first = Archive::new(config()).unwrap();
    let mut second_config = config();
    second_config.descriptor_version = "movement-energy-v2".into();
    second_config.descriptor_identity_sha256 = "b".repeat(64);
    let second = Archive::new(second_config).unwrap();
    assert_ne!(first.identity_sha256(), second.identity_sha256());
}

#[test]
fn low_fidelity_elite_remains_proposal_and_is_never_a_champion_result() {
    let mut archive = Archive::new(config()).unwrap();
    let mut low = candidate(1, 100.0);
    low.fidelity = Fidelity::Low;
    archive.insert(low).unwrap();
    let proposals = archive
        .retrieve_specialists(&request(), &AllowPolicy::default(), 1)
        .unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(proposals[0].candidate.fidelity, Fidelity::Low);
    // The API intentionally exposes proposal retrieval only; qualification belongs elsewhere.
}

#[derive(Default)]
struct AllowPolicy {
    calls: Cell<usize>,
}

impl RetrievalPolicy for AllowPolicy {
    fn allows(&self, _request: &RetrievalRequest, _candidate: &Candidate) -> bool {
        self.calls.set(self.calls.get() + 1);
        true
    }
}

fn request() -> RetrievalRequest {
    RetrievalRequest {
        context_id: "request-context".into(),
        family_id: "request-family".into(),
        lineage_id: "request-lineage".into(),
        forbidden_exposure_ids: BTreeSet::from(["sealed-family".into()]),
    }
}

#[test]
fn exposure_filters_apply_before_policy_and_ranking() {
    let mut archive = Archive::new(config()).unwrap();
    let mut own_context = candidate(1, 100.0);
    own_context.context_id = "request-context".into();
    let mut own_family = candidate(2, 99.0);
    own_family.family_id = "request-family".into();
    let mut own_lineage = candidate(3, 98.0);
    own_lineage.lineage_id = "request-lineage".into();
    let mut sealed = candidate(4, 97.0);
    sealed.family_id = "sealed-family".into();
    let mut sealed_ancestry = candidate(6, 96.0);
    sealed_ancestry
        .exposure_ancestry
        .insert("sealed-family".into());
    let safe = candidate(5, 1.0);
    for item in [
        own_context,
        own_family,
        own_lineage,
        sealed,
        sealed_ancestry,
        safe,
    ] {
        archive.insert(item).unwrap();
    }
    let policy = AllowPolicy::default();
    let proposals = archive
        .retrieve_specialists(&request(), &policy, 8)
        .unwrap();
    assert_eq!(proposals.len(), 1);
    assert_eq!(
        proposals[0].candidate.observation_sha256,
        format!("{:064x}", 5)
    );
    assert_eq!(
        policy.calls.get(),
        1,
        "exposure-ineligible rows must not reach role policy"
    );
}

#[test]
fn evaluator_fidelity_and_role_are_part_of_cell_identity() {
    let mut archive = Archive::new(config()).unwrap();
    let low = candidate(1, 1.0);
    let mut other_fidelity = candidate(2, 100.0);
    other_fidelity.descriptor_bins = low.descriptor_bins.clone();
    other_fidelity.fidelity = Fidelity::Low;
    other_fidelity.context_id = low.context_id.clone();
    other_fidelity.family_id = low.family_id.clone();
    other_fidelity.lineage_id = low.lineage_id.clone();
    other_fidelity.evaluator_identity_sha256 = low.evaluator_identity_sha256.clone();
    other_fidelity.role = low.role;
    let mut other_role = other_fidelity.clone();
    other_role.observation_sha256 = format!("{:064x}", 3);
    other_role.role = CandidateRole::General;
    archive.insert(low).unwrap();
    archive.insert(other_fidelity).unwrap();
    archive.insert(other_role).unwrap();
    assert_eq!(archive.cells().len(), 3);
}

#[test]
fn replacement_ties_are_deterministic_and_logged() {
    let mut archive = Archive::new(config()).unwrap();
    let mut later_digest = candidate(9, 2.0);
    later_digest.descriptor_bins = vec![1, 1];
    let mut earlier_digest = candidate(8, 2.0);
    earlier_digest.descriptor_bins = vec![1, 1];
    earlier_digest.context_id = later_digest.context_id.clone();
    earlier_digest.family_id = later_digest.family_id.clone();
    earlier_digest.lineage_id = later_digest.lineage_id.clone();
    let inserted = archive.insert(later_digest).unwrap();
    assert_eq!(inserted.decision, Decision::Inserted);
    let replaced = archive.insert(earlier_digest.clone()).unwrap();
    assert_eq!(
        replaced.decision,
        Decision::Replaced {
            old_observation_sha256: format!("{:064x}", 9)
        }
    );
    let only = archive.cells().values().next().unwrap();
    assert_eq!(
        only.candidate.observation_sha256,
        earlier_digest.observation_sha256
    );
    assert!(archive
        .events()
        .iter()
        .any(|event| matches!(event.decision, Decision::Replaced { .. })));
}

#[test]
fn lower_quality_and_cell_capacity_are_retained_as_rejections() {
    let mut bounded = config();
    bounded.max_cells = 1;
    let mut archive = Archive::new(bounded).unwrap();
    let mut elite = candidate(1, 10.0);
    elite.descriptor_bins = vec![0, 0];
    archive.insert(elite.clone()).unwrap();
    let mut lower = elite.clone();
    lower.observation_sha256 = format!("{:064x}", 2);
    lower.quality = 9.0;
    assert_eq!(
        archive.insert(lower).unwrap().decision,
        Decision::RejectedLowerQuality
    );
    let mut new_cell = candidate(3, 11.0);
    new_cell.descriptor_bins = vec![1, 0];
    assert_eq!(
        archive.insert(new_cell).unwrap().decision,
        Decision::RejectedCapacity
    );
    assert!(archive
        .events()
        .iter()
        .any(|e| e.decision == Decision::RejectedCapacity));
    assert!(archive.cells().len() <= archive.config.max_cells);
}

#[test]
fn archive_enforces_encoded_byte_cap() {
    let mut bounded = config();
    bounded.max_bytes = 1024;
    let mut archive = Archive::new(bounded).unwrap();
    let mut item = candidate(1, 1.0);
    item.context_id = "long-context-".repeat(200);
    let event = archive.insert(item).unwrap();
    assert_eq!(event.decision, Decision::RejectedByteLimit);
    assert!(archive.cells().is_empty());
    assert!(archive.encoded_bytes().unwrap() <= archive.config.max_bytes);
}

#[test]
fn malformed_candidate_identity_and_descriptor_shape_are_refused() {
    let mut archive = Archive::new(config()).unwrap();
    let mut bad_digest = candidate(1, 1.0);
    bad_digest.observation_sha256 = "not-a-digest".into();
    assert!(archive.insert(bad_digest).is_err());
    let mut bad_bins = candidate(2, 1.0);
    bad_bins.descriptor_bins = vec![0];
    assert!(archive.insert(bad_bins).is_err());
    let mut nonfinite = candidate(3, f64::NAN);
    nonfinite.descriptor_bins = vec![0, 0];
    assert!(archive.insert(nonfinite).is_err());
}

#[test]
fn archive_bytes_and_event_history_remain_bounded() {
    let mut bounded = config();
    bounded.max_events = 2;
    let mut archive = Archive::new(bounded).unwrap();
    for id in 1..=4 {
        let mut item = candidate(id, f64::from(id));
        item.descriptor_bins = vec![id, 0];
        archive.insert(item).unwrap();
    }
    assert_eq!(archive.events().len(), 2);
    assert!(archive.cells().len() <= archive.config.max_cells);
    assert!(archive.encoded_bytes().unwrap() <= archive.config.max_bytes);
}

#[test]
fn persisted_archive_round_trips_and_rejects_identity_tampering() {
    let mut archive = Archive::new(config()).unwrap();
    archive.insert(candidate(1, 1.0)).unwrap();
    let bytes = archive.to_bytes().unwrap();
    let restored = Archive::from_bytes(&bytes).unwrap();
    assert_eq!(restored, archive);

    let mut forged: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    forged["config"]["descriptor_version"] = "forged-v2".into();
    assert!(Archive::from_bytes(&serde_json::to_vec(&forged).unwrap()).is_err());
}

#[test]
fn per_family_scope_groups_contexts_but_per_context_does_not() {
    let mut scoped = config();
    scoped.scope = ArchiveScope::PerFamily;
    let mut archive = Archive::new(scoped).unwrap();
    let first = candidate(1, 1.0);
    let mut second = candidate(2, 2.0);
    second.family_id = first.family_id.clone();
    second.descriptor_bins = first.descriptor_bins.clone();
    archive.insert(first).unwrap();
    archive.insert(second).unwrap();
    assert_eq!(
        archive.cells().len(),
        1,
        "per-family scope groups candidates within a family"
    );

    let mut context_scoped = config();
    context_scoped.scope = ArchiveScope::PerContext;
    let mut archive = Archive::new(context_scoped).unwrap();
    let mut first = candidate(1, 1.0);
    first.family_id = "shared-family".into();
    first.descriptor_bins = vec![1, 1];
    let mut second = candidate(2, 2.0);
    second.family_id = first.family_id.clone();
    second.descriptor_bins = first.descriptor_bins.clone();
    archive.insert(first).unwrap();
    archive.insert(second).unwrap();
    assert_eq!(archive.cells().len(), 2);
}
