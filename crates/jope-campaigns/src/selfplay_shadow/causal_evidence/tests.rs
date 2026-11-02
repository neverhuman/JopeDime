use super::*;

fn digest(byte: char) -> Sha256Digest {
    Sha256Digest::parse(byte.to_string().repeat(64)).unwrap()
}

fn receipt(lane: CausalLane, pair_index: usize) -> CausalEvaluationReceiptV2 {
    let parent = digest('a');
    CausalEvaluationReceiptV2 {
        schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
        receipt_sha256: Sha256Digest::of_bytes(format!("receipt-{lane:?}-{pair_index}").as_bytes()),
        manifest_sha256: digest('d'),
        candidate_sha256: Sha256Digest::of_bytes(
            format!("candidate-{lane:?}-{pair_index}").as_bytes(),
        ),
        task: Task::Binary,
        context_identity: "context".to_owned(),
        accepted_parent_checkpoint_sha256: parent.clone(),
        wave: 191,
        lane,
        pair_index,
        pair_id: causal_pair_id(&parent, 191, Task::Binary, "context", pair_index),
        score: Some(1.0),
        success: true,
        proposal_valid: true,
        runtime_ms: 10,
        ancestry: if lane == CausalLane::NativeDlDirect {
            DlAncestryKind::AcceptedDl
        } else {
            DlAncestryKind::None
        },
        cross_context_backfill: false,
        completed_causal_wave: true,
        valid_lineage: true,
        canary: false,
    }
}

#[test]
fn exact_pair_matching_equalizes_lane_counts_and_repairs_partials() {
    let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
    let requested = BTreeMap::from([(Task::Binary, 16)]);
    let ledger = build_causal_evidence_ledger_v2(
        &parents,
        &requested,
        vec![
            receipt(CausalLane::NativeDlDirect, 0),
            receipt(CausalLane::MatchedGpComparator, 0),
            receipt(CausalLane::NativeDlDirect, 1),
        ],
        1,
    )
    .unwrap();
    let binary = &ledger.tasks[&Task::Binary];
    assert_eq!(binary.complete_pairs, 1);
    assert_eq!(binary.completed_native_direct, binary.completed_matched_gp);
    assert_eq!(ledger.pair_repairs.len(), 1);
    assert_eq!(
        ledger.pair_repairs[0].missing_lane,
        CausalLane::MatchedGpComparator
    );
    assert!(!binary.production_gate_open);
}

#[test]
fn requested_slot_validity_counts_missing_slots() {
    let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
    let requested = BTreeMap::from([(Task::Binary, 8)]);
    let ledger = build_causal_evidence_ledger_v2(
        &parents,
        &requested,
        vec![receipt(CausalLane::NativeDlDirect, 0)],
        1,
    )
    .unwrap();
    assert_eq!(ledger.tasks[&Task::Binary].proposal_validity, 0.125);
}

#[test]
fn optional_task_receipts_are_accounted_without_panicking() {
    let parent = digest('a');
    let mut direct = receipt(CausalLane::NativeDlDirect, 0);
    direct.task = Task::Multiclass;
    direct.pair_id = causal_pair_id(&parent, 191, Task::Multiclass, "context", 0);
    let mut comparator = receipt(CausalLane::MatchedGpComparator, 0);
    comparator.task = Task::Multiclass;
    comparator.pair_id = direct.pair_id.clone();
    let ledger = build_causal_evidence_ledger_v2(
        &BTreeMap::from([(Task::Multiclass, parent)]),
        &BTreeMap::from([(Task::Multiclass, 8)]),
        vec![direct, comparator],
        1,
    )
    .unwrap();
    assert_eq!(ledger.tasks[&Task::Multiclass].complete_pairs, 1);
    assert!(!ledger.tasks[&Task::Multiclass].production_gate_open);
}

#[test]
fn successful_exact_repair_supersedes_a_failed_receipt() {
    let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
    let requested = BTreeMap::from([(Task::Binary, 8)]);
    let direct = receipt(CausalLane::NativeDlDirect, 0);
    let mut failed = receipt(CausalLane::MatchedGpComparator, 0);
    failed.success = false;
    failed.score = None;
    let mut repaired = failed.clone();
    repaired.receipt_sha256 = Sha256Digest::of_bytes(b"repaired-receipt");
    repaired.success = true;
    repaired.score = Some(1.0);
    let ledger =
        build_causal_evidence_ledger_v2(&parents, &requested, vec![direct, failed, repaired], 1)
            .unwrap();
    assert_eq!(ledger.tasks[&Task::Binary].complete_pairs, 1);
    assert!(ledger.pair_repairs.is_empty());
}

#[test]
fn ambiguous_lane_candidates_are_permanently_excluded() {
    let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
    let requested = BTreeMap::from([(Task::Binary, 8)]);
    let direct = receipt(CausalLane::NativeDlDirect, 0);
    let comparator = receipt(CausalLane::MatchedGpComparator, 0);
    let mut ambiguous = comparator.clone();
    ambiguous.receipt_sha256 = Sha256Digest::of_bytes(b"ambiguous-receipt");
    ambiguous.candidate_sha256 = Sha256Digest::of_bytes(b"different-candidate");
    let ledger = build_causal_evidence_ledger_v2(
        &parents,
        &requested,
        vec![direct, comparator, ambiguous],
        1,
    )
    .unwrap();
    assert_eq!(ledger.tasks[&Task::Binary].complete_pairs, 0);
    assert_eq!(ledger.pair_repairs.len(), 1);
    assert!(ledger.pair_repairs[0].permanent_failure);
    assert_eq!(
        ledger.pair_repairs[0].missing_lane,
        CausalLane::MatchedGpComparator
    );
}

#[test]
fn invalid_proposal_is_not_retried_as_an_evaluator_failure() {
    let parents = BTreeMap::from([(Task::Binary, digest('a'))]);
    let requested = BTreeMap::from([(Task::Binary, 8)]);
    let mut direct = receipt(CausalLane::NativeDlDirect, 0);
    direct.proposal_valid = false;
    let comparator = receipt(CausalLane::MatchedGpComparator, 0);
    let ledger =
        build_causal_evidence_ledger_v2(&parents, &requested, vec![direct, comparator], 1).unwrap();
    assert_eq!(ledger.pair_repairs.len(), 1);
    assert!(ledger.pair_repairs[0].permanent_failure);
}

#[test]
fn transitive_ancestry_becomes_mixed() {
    let provenance = CandidateProvenanceV2::inherited(
        digest('a'),
        vec![digest('b'), digest('c')],
        [DlAncestryKind::AcceptedDl, DlAncestryKind::None],
    );
    assert_eq!(provenance.ancestry, DlAncestryKind::Mixed);
}

#[test]
fn production_canary_panels_are_disjoint() {
    let frontier = (0..36).map(|index| format!("f{index}")).collect::<Vec<_>>();
    let hard = (0..6).map(|index| format!("h{index}")).collect::<Vec<_>>();
    let anchor = (0..6).map(|index| format!("a{index}")).collect::<Vec<_>>();
    let panels = (0..3)
        .map(|run| {
            deterministic_canary_contexts(&digest('a'), run, &frontier, &hard, &anchor)
                .unwrap()
                .into_iter()
                .map(|row| row.context_identity)
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    assert!(panels[0].is_disjoint(&panels[1]));
    assert!(panels[0].is_disjoint(&panels[2]));
    assert!(panels[1].is_disjoint(&panels[2]));
}
